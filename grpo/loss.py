import torch


def grpo_loss(policy_logprobs: torch.Tensor, ref_logprobs: torch.Tensor,
              old_logprobs: torch.Tensor, advantages: torch.Tensor,
              completion_mask: torch.Tensor, clip_eps: float = 0.2,
              kl_coef: float = 0.04) -> tuple[torch.Tensor, dict]:
    """GRPO objective: PPO-style clipped policy gradient, group-relative
    advantages (broadcast per-sequence over tokens), plus a KL-to-frozen-
    reference penalty (no separate value/critic network — this IS the
    "no critic" simplification vs. PPO).

    Shapes: policy_logprobs/ref_logprobs/old_logprobs/completion_mask are
    (B, T) per-token log-probs (or mask) for the completion tokens only
    (prompt tokens must already be excluded/masked by the caller).
    advantages is (B,) — one scalar per sampled completion, broadcast
    across that completion's T tokens.

    KL uses the Schulman k3 estimator (unbiased, low-variance, always >= 0):
        KL_t = exp(ref - policy) - (ref - policy) - 1
    """
    advantages = advantages.unsqueeze(1)  # (B, 1) -> broadcasts over T

    log_ratio = policy_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surrogate = torch.min(unclipped, clipped)

    kl = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1

    per_token_loss = -(surrogate - kl_coef * kl)

    mask = completion_mask.float()
    bool_mask = completion_mask.bool()
    denom = mask.sum().clamp(min=1.0)
    zeros = torch.zeros_like(per_token_loss)
    # torch.where, not `* mask`, to avoid inf*0=nan when masked positions hold
    # non-finite ratios (e.g. from unclamped policy logprobs).
    loss = torch.where(bool_mask, per_token_loss, zeros).sum() / denom

    with torch.no_grad():
        # Same torch.where rationale as above, applied to the logging metrics.
        clip_frac = torch.where(bool_mask, (unclipped != clipped).float(), torch.zeros_like(mask)).sum() / denom
        mean_kl = torch.where(bool_mask, kl, torch.zeros_like(kl)).sum() / denom
        mean_ratio = torch.where(bool_mask, ratio, torch.zeros_like(ratio)).sum() / denom

    metrics = {
        "kl": mean_kl.detach(),
        "clip_frac": clip_frac.detach(),
        "mean_ratio": mean_ratio.detach(),
    }
    return loss, metrics
