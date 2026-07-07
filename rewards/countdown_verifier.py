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
    """Rule-based Countdown reward: 1.0 iff the completion is well-formed AND
    the <answer> expression uses each of `numbers` exactly once and evaluates
    to `target`. Returns 0.0 for any format violation, parse failure, wrong
    multiset of numbers used, or wrong result — never raises.
    """
    if format_reward(completion) == 0.0:
        return 0.0

    expr = extract_answer_expr(completion)
    if not expr:
        return 0.0

    used = _extract_used_numbers(expr)
    if used is None:
        return 0.0
    if Counter(used) != Counter(numbers):
        return 0.0

    result = _safe_eval(expr)
    if result is None:
        return 0.0

    return 1.0 if abs(result - target) < eps else 0.0
