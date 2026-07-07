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
