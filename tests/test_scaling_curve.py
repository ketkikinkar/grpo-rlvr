from eval.scaling_curve import majority_vote_at_k, pass_at_k


def test_pass_at_k_true_if_any_correct():
    # Additive tiered reward scheme (see rewards/countdown_verifier.py,
    # rewards/gsm8k_verifier.py): only the max value 2.0 (perfect format AND
    # correct answer) counts as a "pass".
    rewards_for_one_prompt = [0.0, 0.0, 2.0, 0.0]
    assert pass_at_k(rewards_for_one_prompt, k=4) == 1.0


def test_pass_at_k_false_if_none_correct():
    rewards_for_one_prompt = [0.0, 0.0, 0.0, 0.0]
    assert pass_at_k(rewards_for_one_prompt, k=4) == 0.0


def test_pass_at_k_format_only_reward_does_not_count_as_pass():
    # A reward of exactly 1.0 can be achieved purely via format compliance
    # (format_tier=1.0, correctness=0.0) with a WRONG answer. Only 2.0
    # (perfect format tier AND correct answer) is a genuine pass.
    rewards_for_one_prompt = [1.0, 1.0, 1.5, 1.0]
    assert pass_at_k(rewards_for_one_prompt, k=4) == 0.0


def test_pass_at_k_respects_k_slice():
    rewards_for_one_prompt = [0.0, 0.0, 0.0, 2.0]
    assert pass_at_k(rewards_for_one_prompt, k=2) == 0.0
    assert pass_at_k(rewards_for_one_prompt, k=4) == 1.0


def test_majority_vote_at_k_uses_most_common_answer():
    # answers, not raw rewards: correctness determined by matching the
    # plurality answer string, independent of ground truth
    answers = ["24", "23", "24", "24"]
    assert majority_vote_at_k(answers, ground_truth="24") == 1.0


def test_majority_vote_at_k_wrong_when_plurality_is_wrong():
    answers = ["23", "23", "24"]
    assert majority_vote_at_k(answers, ground_truth="24") == 0.0


def test_majority_vote_at_k_cannot_detect_correct_countdown_expression():
    # A correct Countdown expression like "4*5+3" evaluates to 23 but is
    # never textually identical to the bare target string "23" -- this
    # demonstrates majority_vote_at_k's blind spot for expression-based
    # tasks (see eval/scaling_curve.py's caveat comment).
    # All three answers are identical correct expressions, but should return 0.0
    # because they don't match the bare target string.
    answers = ["4*5+3", "4*5+3", "4*5+3"]
    assert majority_vote_at_k(answers, ground_truth="23") == 0.0
