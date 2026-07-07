from rewards.countdown_verifier import countdown_reward, extract_answer_expr

def test_correct_expression_hits_target():
    completion = "</think><answer>4*5+3</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=23) == 1.0

def test_wrong_target_gets_zero_correctness():
    completion = "</think><answer>4*5+3</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=24) == 0.0

def test_reused_number_gets_zero():
    # uses 4 twice, only one 4 is available
    completion = "</think><answer>4*4+5</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=21) == 0.0

def test_unused_number_gets_zero():
    # doesn't use the 3
    completion = "</think><answer>4*5</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=20) == 0.0

def test_malformed_expression_gets_zero_not_crash():
    completion = "</think><answer>4 * / 5</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=20) == 0.0

def test_missing_format_gets_zero():
    completion = "4*5+3=23"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=23) == 0.0

def test_division_and_floats_supported():
    completion = "</think><answer>(9-5)/2</answer>"
    assert countdown_reward(completion, numbers=[9, 5, 2], target=2) == 1.0

def test_extract_answer_expr_pulls_inner_text():
    completion = "</think><answer> 4 * 5 + 3 </answer>"
    assert extract_answer_expr(completion) == "4 * 5 + 3"

def test_extract_answer_expr_returns_none_when_absent():
    assert extract_answer_expr("no tags here") is None
