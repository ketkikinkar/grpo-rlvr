from rewards.gsm8k_verifier import gsm8k_reward, extract_gsm8k_ground_truth

def test_correct_numeric_answer():
    completion = "<think>She has 3 apples, buys 2 more, 3+2=5</think><answer>5</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 1.0

def test_wrong_numeric_answer():
    completion = "<think>...</think><answer>4</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 0.0

def test_answer_with_commas_and_units_still_parses():
    completion = "<think>...</think><answer>1,024 dollars</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=1024.0) == 1.0

def test_non_numeric_answer_gets_zero_not_crash():
    completion = "<think>...</think><answer>not sure</answer>"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 0.0

def test_missing_format_gets_zero():
    completion = "the answer is 5"
    assert gsm8k_reward(completion, ground_truth_answer=5.0) == 0.0

def test_extract_ground_truth_from_gsm8k_style_solution():
    solution = "Natalia sold 48 clips in April... #### 72"
    assert extract_gsm8k_ground_truth(solution) == 72.0
