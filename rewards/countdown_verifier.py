import ast
import re
from collections import Counter

from rewards.format_reward import format_reward

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd,
    ast.Constant, ast.Load,
) + _ALLOWED_BINOPS


def extract_answer_expr(completion: str) -> str | None:
    match = _ANSWER_RE.search(completion)
    if match is None:
        return None
    return match.group(1).strip()


def _safe_eval(expr: str) -> float | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        return eval(compile(tree, "<countdown>", "eval"), {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return None


def _extract_used_numbers(expr: str) -> list[float] | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    numbers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                return None
            numbers.append(node.value)
    return numbers


def countdown_reward(completion: str, numbers: list[int], target: int, eps: float = 1e-4) -> float:
    """Additive, tiered Countdown reward (nano-aha-moment style):
    `format_reward(completion) + correctness`, where correctness is a binary
    1.0/0.0 for whether the <answer> expression uses each of `numbers`
    exactly once and evaluates to `target`. This avoids the degenerate
    strict-binary reward where every sample in a GRPO group can score
    identically, collapsing the group-relative advantage to exactly zero.

    Returns 0.0 only when the completion fails the format structure check
    entirely (no usable answer region). Otherwise returns the format tier
    (0.5 or 1.0) plus 1.0 if the expression is correct, else plus 0.0 -
    total in {0.0, 0.5, 1.0, 1.5, 2.0}. Never raises.
    """
    fmt = format_reward(completion)
    if fmt == 0.0:
        return 0.0

    expr = extract_answer_expr(completion)
    if not expr:
        return fmt

    used = _extract_used_numbers(expr)
    if used is None or Counter(used) != Counter(numbers):
        return fmt

    result = _safe_eval(expr)
    if result is None:
        return fmt

    correctness = 1.0 if abs(result - target) < eps else 0.0
    return fmt + correctness
