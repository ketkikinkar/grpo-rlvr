import pytest

from data.countdown import build_countdown_dataset
from eval.evaluate import aggregate_accuracy, evaluate_checkpoint


def test_aggregate_accuracy_counts_fraction_correct():
    # Rewards use the additive tiered scheme (0.0/0.5/1.0/1.5/2.0); only 2.0
    # (perfect format + perfect correctness) counts as "correct" here.
    rewards = [2.0, 0.0, 2.0, 2.0]
    assert aggregate_accuracy(rewards) == 0.75


def test_aggregate_accuracy_empty_raises():
    with pytest.raises(ValueError):
        aggregate_accuracy([])


def test_aggregate_accuracy_format_only_reward_not_counted_correct():
    # A reward of exactly 1.0 means format_tier + correctness summed to 1.0,
    # which under the additive tiered scheme (rewards/countdown_verifier.py,
    # rewards/gsm8k_verifier.py) is achievable purely via format compliance
    # (format_tier=1.0, correctness=0.0) with a WRONG answer -- this is
    # common for our trained model early on. It must NOT count as correct.
    rewards = [1.0]
    assert aggregate_accuracy(rewards) == 0.0


def test_aggregate_accuracy_only_max_reward_counts_as_correct():
    # Only the max value 2.0 (perfect format tier AND correct answer) counts.
    rewards = [1.0, 1.5, 2.0]
    assert aggregate_accuracy(rewards) == pytest.approx(1.0 / 3.0)


def test_evaluate_checkpoint_runs_end_to_end_on_tiny_dataset():
    # Real (small) model, tiny held-out set, greedy decode -- exercises the
    # full generate -> reward -> aggregate pipeline without a long runtime.
    dataset = build_countdown_dataset(n_examples=2, seed=123)
    result = evaluate_checkpoint(
        model_path="Qwen/Qwen2.5-0.5B",
        dataset=dataset,
        task="countdown",
        batch_size=2,
        max_new_tokens=32,
        temperature=0.0,
    )
    assert result["n"] == 2
    assert len(result["rewards"]) == 2
    assert 0.0 <= result["accuracy"] <= 1.0
    assert all(r in (0.0, 0.5, 1.0, 1.5, 2.0) for r in result["rewards"])
