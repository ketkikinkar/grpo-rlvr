import re

_FORMAT_RE = re.compile(r"^\s*(.*?)</think>\s*<answer>(.*?)</answer>\s*$", re.DOTALL)

# Allowed characters for the content inside <answer>...</answer>: digits,
# the four arithmetic operators, parentheses, decimal point, and whitespace.
# GSM8K answers (plain numbers) are a strict subset of this set.
_ALLOWED_ANSWER_CHARS_RE = re.compile(r"^[0-9+\-*/().\s]*$")


def format_reward(completion: str) -> float:
    """3-tier format reward, nano-aha-moment style:

    - 0.0 if completion does not match `<think>...</think>\\n<answer>...</answer>`
      structure at all (or the answer body is empty / tags are nested wrong).
    - 0.5 if the structure matches but the content inside <answer> contains
      characters outside the allowed arithmetic set (digits, + - * / ( ) .
      and whitespace).
    - 1.0 if the structure matches AND the <answer> content is composed
      entirely of allowed characters.

    `completion` is expected to be the text generated AFTER the prompt's
    trailing "<think>" (see data/prompts.py), so it should not itself
    contain a leading "<think>" tag - only the closing "</think>" onward.
    """
    match = _FORMAT_RE.match(completion)
    if match is None:
        return 0.0
    think_body, answer_body = match.groups()
    if "<answer>" in think_body or "</answer>" in think_body:
        return 0.0
    if not answer_body.strip():
        return 0.0
    if _ALLOWED_ANSWER_CHARS_RE.match(answer_body) is None:
        return 0.5
    return 1.0
