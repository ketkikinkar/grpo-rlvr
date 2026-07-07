import re

_FORMAT_RE = re.compile(r"^\s*(.*?)</think><answer>(.*?)</answer>\s*$", re.DOTALL)


def format_reward(completion: str) -> float:
    """1.0 if completion is exactly `<think>...</think><answer>...</answer>`
    with nothing after the closing answer tag, else 0.0.

    `completion` is expected to be the text generated AFTER the prompt's
    trailing "<think>" (see data/prompts.py), so it should not itself
    contain a leading "<think>" tag — only the closing "</think>" onward.
    """
    match = _FORMAT_RE.match(completion)
    if match is None:
        return 0.0
    think_body, answer_body = match.groups()
    if "<answer>" in think_body or "</answer>" in think_body:
        return 0.0
    if not answer_body.strip():
        return 0.0
    return 1.0
