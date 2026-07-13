import random

from datasets import Dataset

from data.prompts import build_prompt

_INTEGRAL_TOL = 1e-6


def _walk_expr(pool: list[int], rng: random.Random, ops: list[str]) -> tuple[float, str]:
    """Fold a random sequence of `ops` over `pool` left-to-right, building both
    the accumulated numeric result and a fully-parenthesized expression string
    that evaluates to exactly that result (parens pin down left-to-right
    evaluation order regardless of +-*/ precedence).
    """
    acc = float(pool[0])
    expr = str(pool[0])
    for n in pool[1:]:
        op = rng.choice(ops)
        if op == "+":
            acc = acc + n
        elif op == "-":
            acc = acc - n
        elif op == "*":
            acc = acc * n
        elif op == "/":
            acc = acc / n
        expr = f"({expr}{op}{n})"
    return acc, expr


def _random_solvable_puzzle(
    rng: random.Random, num_count: int = 4, max_num: int = 100, max_retries: int = 50
) -> tuple[list[int], int, str]:
    """Generate a Countdown puzzle by sampling `num_count` numbers, then
    computing a reachable target via a random sequence of +,-,*,/ over them -
    guarantees the puzzle is solvable, unlike pure random target sampling.

    Division can produce a non-integral accumulator (e.g. 22/3), in which case
    there is no exact expression over these numbers hitting the rounded
    target. We retry with a fresh draw/op sequence until the accumulator is
    (near-)integral, capped at `max_retries`; if we still haven't found one,
    fall back to +,-,* only, which always yields an integral result.
    """
    ops_full = ["+", "-", "*", "/"]
    for _ in range(max_retries):
        numbers = [rng.randint(1, max_num) for _ in range(num_count)]
        pool = numbers.copy()
        rng.shuffle(pool)
        acc, expr = _walk_expr(pool, rng, ops_full)
        if abs(acc - round(acc)) < _INTEGRAL_TOL:
            return numbers, int(round(acc)), expr

    numbers = [rng.randint(1, max_num) for _ in range(num_count)]
    pool = numbers.copy()
    rng.shuffle(pool)
    acc, expr = _walk_expr(pool, rng, ["+", "-", "*"])
    return numbers, int(round(acc)), expr


def build_countdown_dataset(n_examples: int, seed: int = 0, num_count: int = 4) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_examples):
        numbers, target, _expr = _random_solvable_puzzle(rng, num_count=num_count)
        question = (
            f"Using the numbers {numbers}, each exactly once, and the operations "
            f"+, -, *, /, write an arithmetic expression that evaluates to {target}. "
            f"Put only the expression (no equals sign) inside the answer tags."
        )
        rows.append({
            "prompt": build_prompt(question),
            "numbers": numbers,
            "target": target,
        })
    return Dataset.from_list(rows)
