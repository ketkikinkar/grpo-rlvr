import random

from data.countdown import _random_solvable_puzzle, build_countdown_dataset
from rewards.countdown_verifier import countdown_reward


def test_random_solvable_puzzle_is_actually_solvable():
    # Exercise the real verifier with the exact expression the generator
    # walked — the strongest regression test against silently-unsolvable
    # puzzles (e.g. from division leaving a non-integral target).
    rng = random.Random(123)
    for _ in range(200):
        numbers, target, expr = _random_solvable_puzzle(rng)
        completion = f"</think><answer>{expr}</answer>"
        # additive tiered reward: format tier 1.0 (all-allowed chars) +
        # correctness 1.0 (puzzle is solvable by construction) = 2.0.
        assert countdown_reward(completion, numbers=numbers, target=target) == 2.0


def test_build_countdown_dataset_rows_are_solvable():
    # build_countdown_dataset discards the constructed expression, so
    # brute-force search over the (numbers, target) pair to confirm at
    # least one arrangement of ops/parenthesizations hits the target.
    dataset = build_countdown_dataset(n_examples=50, seed=7, num_count=4)
    for row in dataset:
        assert _is_solvable(row["numbers"], row["target"])


def _is_solvable(numbers: list[int], target: int, eps: float = 1e-4) -> bool:
    from itertools import permutations, product

    ops = ["+", "-", "*", "/"]
    for perm in permutations(numbers):
        for op_seq in product(ops, repeat=len(numbers) - 1):
            acc = float(perm[0])
            ok = True
            for op, n in zip(op_seq, perm[1:]):
                if op == "+":
                    acc = acc + n
                elif op == "-":
                    acc = acc - n
                elif op == "*":
                    acc = acc * n
                elif op == "/":
                    if n == 0:
                        ok = False
                        break
                    acc = acc / n
            if ok and abs(acc - target) < eps:
                return True
    return False
