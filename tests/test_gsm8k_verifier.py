from rewards.gsm8k_verifier import gsm8k_reward, extract_gsm8k_ground_truth

def test_correct_numeric_answer():
    # format tier 1.0 (digits only) + correctness 1.0 = 2.0
    completion = "<think>She has 3 apples, buys 2 more, 3+2=5</think><answer>5</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 2.0

def test_wrong_numeric_answer_gets_format_credit_only():
    # well-formed, digits-only answer, but wrong value:
    # format tier 1.0 + correctness 0.0 = 1.0 (not 0.0 anymore).
    completion = "<think>...</think><answer>4</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 1.0

def test_answer_with_commas_and_units_gets_half_credit_plus_correctness():
    # "1,024 dollars" contains a comma and letters, outside the allowed
    # arithmetic character set -> format tier 0.5; parsed value is still
    # correct -> correctness 1.0 -> total 1.5.
    completion = "<think>...</think><answer>1,024 dollars</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=1024.0) == 1.5

def test_non_numeric_answer_gets_half_credit_only_not_crash():
    # "not sure" has disallowed letters -> format tier 0.5; no number to
    # parse -> correctness can't be assessed -> total stays at 0.5.
    completion = "<think>...</think><answer>not sure</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 0.5

def test_missing_format_gets_zero():
    completion = "the answer is 5"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 0.0

def test_disallowed_chars_in_answer_gets_half_credit_only():
    completion = "<think>...</think><answer>abc</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 0.5

def test_extract_ground_truth_from_gsm8k_style_solution():
    solution = "Natalia sold 48 clips in April... #### 72"
    assert extract_gsm8k_ground_truth(solution) == 72.0
