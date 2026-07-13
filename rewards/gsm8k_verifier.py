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
    """Additive, tiered GSM8K reward (nano-aha-moment style):
    `format_reward(completion) + correctness`, where correctness is binary
    1.0/0.0 for whether the numeric value inside <answer> matches
    ground_truth_answer within eps. This avoids the degenerate strict-binary
    reward where every sample in a GRPO group can score identically,
    collapsing the group-relative advantage to exactly zero.

    Returns 0.0 only when the completion fails the format structure check
    entirely. Otherwise returns the format tier (0.5 or 1.0) plus 1.0 if the
    answer is correct, else plus 0.0 - total in {0.0, 0.5, 1.0, 1.5, 2.0}.
    Never raises on malformed model output.
    """
    fmt = format_reward(completion)
    if fmt == 0.0:
        return 0.0

    match = _ANSWER_RE.search(completion)
    if match is None:
        return fmt

    predicted = _parse_number(match.group(1))
    if predicted is None:
        return fmt

    correctness = 1.0 if abs(predicted - ground_truth_answer) < eps else 0.0
    return fmt + correctness
