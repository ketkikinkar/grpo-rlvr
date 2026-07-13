import torch


def compute_group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """Group-relative advantage estimation, the core of GRPO: replace a
    learned critic/value function with a per-prompt group baseline.

    `rewards` is a flat (N,) tensor where every contiguous block of
    `group_size` entries are the G completions sampled for one prompt
    (this matches grpo.rollout.generate_rollouts' row ordering).

    A_i = (r_i - mean(group)) / (std(group) + eps)

    When every completion in a group scores identically (std == 0, e.g. all
    correct or all wrong), there is no learning signal to extract from that
    group - return exactly zero advantage rather than dividing by eps
    (which would otherwise amplify float noise into a huge fake gradient).
    """
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length {rewards.numel()} is not divisible by group_size {group_size}"
        )

    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)

    zero_variance = std < 1e-8
    normalized = (grouped - mean) / (std + eps)
    normalized = torch.where(zero_variance.expand_as(normalized), torch.zeros_like(normalized), normalized)

    return normalized.view(-1)
