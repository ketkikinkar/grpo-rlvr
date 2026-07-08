from rewards.countdown_verifier import countdown_reward, extract_answer_expr

def test_correct_expression_hits_target():
    # format tier 1.0 (all-allowed chars) + correctness 1.0 = 2.0
    completion = "</think><answer>4*5+3</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=23) == 2.0

def test_wrong_target_gets_format_credit_only():
    # well-formed, all-allowed-char answer, but wrong result:
    # format tier 1.0 + correctness 0.0 = 1.0 (not 0.0 anymore).
    completion = "</think><answer>4*5+3</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=24) == 1.0

def test_reused_number_gets_format_credit_only():
    # uses 4 twice, only one 4 is available; still well-formed with
    # all-allowed chars, so format tier 1.0 is credited.
    completion = "</think><answer>4*4+5</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=21) == 1.0

def test_unused_number_gets_format_credit_only():
    # doesn't use the 3; format tier 1.0 still credited.
    completion = "</think><answer>4*5</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=20) == 1.0

def test_malformed_expression_gets_format_credit_only_not_crash():
    # "4 * / 5" contains only digits/operators/whitespace (all allowed),
    # so format tier is still 1.0 even though the expression can't evaluate.
    completion = "</think><answer>4 * / 5</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=20) == 1.0

def test_missing_format_gets_zero():
    completion = "4*5+3=23"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=23) == 0.0

def test_disallowed_chars_in_answer_gets_half_credit_only():
    # structure well-formed but "banana" has disallowed characters ->
    # format tier 0.5; numbers used don't match / can't safely eval ->
    # correctness 0.0 -> total 0.5.
    completion = "</think><answer>4*5+banana</answer>"
    assert countdown_reward(completion, numbers=[4, 5, 3], target=23) == 0.5

def test_division_and_floats_supported():
    completion = "</think><answer>(9-5)/2</answer>"
    assert countdown_reward(completion, numbers=[9, 5, 2], target=2) == 2.0

def test_extract_answer_expr_pulls_inner_text():
    completion = "</think><answer> 4 * 5 + 3 </answer>"
    assert extract_answer_expr(completion) == "4 * 5 + 3"

def test_extract_answer_expr_returns_none_when_absent():
    assert extract_answer_expr("no tags here") is None

def test_unary_negative_number_supported():
    completion = "</think><answer>-3+8</answer>"
    assert countdown_reward(completion, numbers=[3, 8], target=5) == 2.0
