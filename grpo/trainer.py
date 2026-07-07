import gc

import torch
import torch.nn.functional as F

from grpo.advantage import compute_group_advantages
from grpo.loss import grpo_loss
from grpo.rollout import generate_rollouts
from rewards.countdown_verifier import countdown_reward
from rewards.gsm8k_verifier import gsm8k_reward

_TASK_REWARDS = {
    "countdown": lambda text, meta: countdown_reward(text, meta["numbers"], meta["target"]),
    "gsm8k": lambda text, meta: gsm8k_reward(text, meta["ground_truth"]),
}


def _sequence_logprobs(model, prompt_ids, completion_ids, completion_mask):
    """Per-token log p(completion_t | prompt, completion_<t>) under `model`,
    shape (B, T_completion) aligned with completion_mask.
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = (input_ids != model.config.pad_token_id).long() if model.config.pad_token_id is not None else torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # logits[:, t] predicts token at t+1; we need predictions for the
    # completion region, i.e. logits at positions [P-1 .. P-1+T_c-1]
    prompt_len = prompt_ids.shape[1]
    completion_logits = logits[:, prompt_len - 1: prompt_len - 1 + completion_ids.shape[1], :]
    logprobs = F.log_softmax(completion_logits.float(), dim=-1)
    token_logprobs = torch.gather(logprobs, 2, completion_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs * completion_mask  # zero out padding contributions


class GRPOTrainer:
    def __init__(self, policy, ref, tokenizer, optimizer, task: str, group_size: int,
                 max_new_tokens: int, temperature: float, clip_eps: float = 0.2,
                 kl_coef: float = 0.04, max_grad_norm: float = 1.0):
        if task not in _TASK_REWARDS:
            raise ValueError(f"unknown task {task!r}, expected one of {list(_TASK_REWARDS)}")
        self.policy = policy
        self.ref = ref
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.task = task
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm

    def train_step(self, batch_dataset) -> dict:
        prompts = list(batch_dataset["prompt"])
        rollouts = generate_rollouts(
            self.policy, self.tokenizer, prompts,
            group_size=self.group_size, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        rewards = []
        for row_i, text in enumerate(rollouts.completion_texts):
            example = batch_dataset[rollouts.prompt_index[row_i]]
            rewards.append(_TASK_REWARDS[self.task](text, example))
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.policy.device)

        advantages = compute_group_advantages(rewards_t, group_size=self.group_size)

        with torch.no_grad():
            old_logprobs = _sequence_logprobs(
                self.policy, rollouts.prompt_ids, rollouts.completion_ids, rollouts.completion_mask
            )
            ref_logprobs = _sequence_logprobs(
                self.ref, rollouts.prompt_ids, rollouts.completion_ids, rollouts.completion_mask
            )

        self.policy.train()
        policy_logprobs = _sequence_logprobs(
            self.policy, rollouts.prompt_ids, rollouts.completion_ids, rollouts.completion_mask
        )

        loss, loss_metrics = grpo_loss(
            policy_logprobs, ref_logprobs, old_logprobs, advantages,
            rollouts.completion_mask, clip_eps=self.clip_eps, kl_coef=self.kl_coef,
        )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        response_lengths = rollouts.completion_mask.sum(dim=1).float()

        metrics = {
            "loss": loss.item(),
            "mean_reward": rewards_t.mean().item(),
            "reward_std": rewards_t.std().item(),
            "kl": loss_metrics["kl"].item(),
            "clip_frac": loss_metrics["clip_frac"].item(),
            "grad_norm": grad_norm.item(),
            "response_length": response_lengths.mean().item(),
        }

        # Explicitly drop references to large intermediate tensors (rollout
        # batch, logprob tensors, loss) so refcounting frees them immediately,
        # then ask MPS's caching allocator to release cached-but-unused
        # memory back to the OS (it doesn't always do this proactively).
        del rollouts, old_logprobs, ref_logprobs, policy_logprobs, loss
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return metrics
