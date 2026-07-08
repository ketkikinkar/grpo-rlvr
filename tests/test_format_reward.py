from rewards.format_reward import format_reward

def test_well_formed_gets_full_credit():
    text = "</think><answer>4*5+3</answer>"
    assert format_reward(text) == 1.0

def test_missing_answer_tag_gets_zero():
    text = "</think>4*5+3"
    assert format_reward(text) == 0.0

def test_missing_think_tag_gets_zero():
    text = "<answer>4*5+3</answer>"
    assert format_reward(text) == 0.0

def test_out_of_order_tags_gets_zero():
    text = "<answer>4*5+3</answer></think>"
    assert format_reward(text) == 0.0

def test_extra_text_after_answer_gets_zero():
    text = "</think><answer>4*5+3</answer> hope that's right!"
    assert format_reward(text) == 0.0

def test_whitespace_between_think_and_answer_is_tolerated():
    text = "reasoning here</think>\n<answer>44</answer>"
    assert format_reward(text) == 1.0

def test_disallowed_chars_in_answer_gets_half_credit():
    # structure is well-formed, but "banana" contains letters outside the
    # allowed arithmetic character set -> partial credit, not full or zero.
    text = "</think><answer>4*5+banana</answer>"
    assert format_reward(text) == 0.5

def test_answer_with_units_gets_half_credit():
    text = "</think>\n<answer>1,024 dollars</answer>"
    assert format_reward(text) == 0.5

def test_allowed_chars_with_parens_and_decimal_gets_full_credit():
    text = "</think><answer>(4.5 + 5) / 2</answer>"
    assert format_reward(text) == 1.0
