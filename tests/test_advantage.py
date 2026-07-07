import torch

from grpo.advantage import compute_group_advantages


def test_basic_normalization_within_group():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])  # one group, size 4
    adv = compute_group_advantages(rewards, group_size=4)
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-5)
    assert adv[0] > 0 and adv[1] < 0


def test_two_groups_normalized_independently():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])  # group A all high, group B all low
    adv = compute_group_advantages(rewards, group_size=2)
    # each group has zero variance -> should not blow up (handled by eps)
    assert torch.all(torch.isfinite(adv))


def test_zero_variance_group_gives_zero_advantage_not_nan():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])  # one group, all identical rewards
    adv = compute_group_advantages(rewards, group_size=4)
    assert torch.all(adv == 0.0)


def test_shape_preserved():
    rewards = torch.rand(16)
    adv = compute_group_advantages(rewards, group_size=8)
    assert adv.shape == rewards.shape


def test_group_size_must_divide_length():
    import pytest
    with pytest.raises(ValueError):
        compute_group_advantages(torch.rand(10), group_size=3)
