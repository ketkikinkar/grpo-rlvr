import itertools
import random

from datasets import Dataset

from data.prompts import build_prompt


def _random_solvable_puzzle(rng: random.Random, num_count: int = 4, max_num: int = 100) -> tuple[list[int], int]:
    """Generate a Countdown puzzle by sampling `num_count` numbers, then
    computing a reachable target via a random sequence of +,-,*,/ over them —
    guarantees the puzzle is solvable, unlike pure random target sampling.
    """
    numbers = [rng.randint(1, max_num) for _ in range(num_count)]
    pool = numbers.copy()
    rng.shuffle(pool)
    acc = float(pool[0])
    for n in pool[1:]:
        op = rng.choice(["+", "-", "*", "/"])
        if op == "+":
            acc = acc + n
        elif op == "-":
            acc = acc - n
        elif op == "*":
            acc = acc * n
        elif op == "/" and n != 0:
            acc = acc / n
    target = int(round(acc))
    return numbers, target


def build_countdown_dataset(n_examples: int, seed: int = 0, num_count: int = 4) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_examples):
        numbers, target = _random_solvable_puzzle(rng, num_count=num_count)
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
