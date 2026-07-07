import re

from rewards.format_reward import format_reward

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_GT_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def _parse_number(text: str) -> float | None:
    match = _NUMBER_RE.search(text.replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_gsm8k_ground_truth(solution_text: str) -> float | None:
    match = _GT_RE.search(solution_text.replace(",", ""))
    if match is None:
        return None
    return float(match.group(1))


def gsm8k_reward(completion: str, ground_truth_answer: float, eps: float = 1e-4) -> float:
    """1.0 iff completion is well-formed AND the numeric value inside
    <answer> matches ground_truth_answer within eps. Never raises on
    malformed model output.
    """
    if format_reward(completion) == 0.0:
        return 0.0

    match = _ANSWER_RE.search(completion)
    if match is None:
        return 0.0

    predicted = _parse_number(match.group(1))
    if predicted is None:
        return 0.0

    return 1.0 if abs(predicted - ground_truth_answer) < eps else 0.0
