from typing import Callable, Literal

import torch 
import transformers 
from transformers import PreTrainedTokenizerBase 


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, torch.Tensor]: 
    token_ids = [] 
    prompt_lens = [] 
    token_lens = []
    for prompt, output in zip(prompt_strs, output_strs):
        promt_id = tokenizer.encode(prompt) 
        output_id = tokenizer.encode(output) 
        token_ids.append(promt_id + output_id) 
        prompt_lens.append(len(promt_id)) 
        token_lens.append(len(promt_id + output_id))

    max_len = max(token_lens)
    token_ids_pad = tokenizer.pad([{"input_ids": sample} for sample in token_ids], padding=True, return_tensors='pt') 
    input_ids = token_ids_pad['input_ids'][:, :-1] # 注意这里要用input_ids作第一层key
    labels = token_ids_pad['input_ids'][:, 1:]

    prompt_mask = torch.arange(0, max_len)[None, :] >= torch.Tensor(prompt_lens)[:, None] 
    pad_mask = torch.arange(0, max_len)[None, :] < torch.Tensor(token_lens)[:, None]
    response_mask = (prompt_mask & pad_mask)[:, 1:] 

    return {"input_ids": input_ids, 'labels': labels, 'response_mask': response_mask} 

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """ 
    model_outputs = model(input_ids).logits  # (b, s, d) 
    model_outputs_norm = model_outputs - torch.max(model_outputs, dim=-1, keepdim=True)[0] 
    outputs_log_probs = model_outputs_norm -torch.log(torch.sum(torch.exp(model_outputs_norm), dim=-1, keepdim=True)) 
    log_probs = torch.gather(outputs_log_probs, dim = -1, index=labels.unsqueeze(-1)).squeeze(-1)  # 注意dim参数 以及index tensor和inputs维度数及非收集维度对齐
    if return_token_entropy:
        outputs_probs = torch.exp(outputs_log_probs) 
        token_entropy = - torch.sum(outputs_log_probs * outputs_probs, dim=-1)
        return {"log_probs": log_probs, "token_entropy": token_entropy}
    return {"log_probs": log_probs}



def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]], #注意Callable的意思
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    raw_rewards = [] 
    format_rewards = []
    for respionse, gt in zip(rollout_responses, repeated_ground_truths): 
        rewards_dict = reward_fn(respionse, gt)
        raw_rewards.append(rewards_dict['reward'])
        format_rewards.append(rewards_dict['format_reward'])
    raw_rewards = torch.Tensor(raw_rewards) 
    format_rewads = torch.Tensor(format_rewards)
    mean_rewards = {'mean_total_rewards': raw_rewards.mean().data, 'mean_format_rewards':format_rewads.mean().data} 
    return raw_rewards,  mean_rewards

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    ) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """ 
    raw_rewards = raw_rewards.view(-1, group_size) 
    if baseline == 'mean':
        rewards_with_baseline = raw_rewards - raw_rewards.mean(dim=-1, keepdim=True) 
    else :
        rewards_with_baseline = raw_rewards 
    if advantage_normalizer == 'std':
        rewards_with_norm = rewards_with_baseline / (torch.std(raw_rewards, dim=-1, keepdim=True) + advantage_eps )
    elif advantage_normalizer == 'mean':
        rewards_with_norm = rewards_with_baseline / (torch.mean(raw_rewards, dim=-1, keepdim=True) + advantage_eps )
    else:
        rewards_with_norm = rewards_with_baseline 

    advantages = rewards_with_norm.flatten() 
    metadata = {'mean': advantages.mean().data, 'std': advantages.std().data, 'max': advantages.max().data, 'min': advantages.min().data} 
    return advantages, metadata
    



    raise NotImplementedError

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """ 

    raise NotImplementedError
