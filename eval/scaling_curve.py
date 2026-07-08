from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewards.countdown_verifier import extract_answer_expr
from rewards.gsm8k_verifier import _ANSWER_RE as GSM8K_ANSWER_RE


def pass_at_k(rewards_for_one_prompt: list[float], k: int) -> float:
    """1.0 if ANY of the first k sampled completions for a prompt is correct.

    rewards/countdown_verifier.py::countdown_reward and
    rewards/gsm8k_verifier.py::gsm8k_reward return an additive tiered score
    in {0.0, 0.5, 1.0, 1.5, 2.0} = format_tier + binary_correctness, not the
    old binary {0.0, 1.0}. A reward of 1.0 can be achieved purely through
    perfect format compliance with a WRONG answer (format_tier=1.0,
    correctness=0.0), so only the max value 2.0 (perfect format AND correct
    answer) counts as a "pass" -- using `>= 1.0` would incorrectly credit
    format-only completions (consistent with eval/evaluate.py::aggregate_accuracy).
    """
    return 1.0 if any(r >= 2.0 for r in rewards_for_one_prompt[:k]) else 0.0


def majority_vote_at_k(answers: list[str], ground_truth: str) -> float:
    """1.0 if the plurality answer string among the first k completions
    equals ground_truth (ties broken by Counter's stable first-seen order).

    NOTE: For Countdown, this metric has a fundamental limitation. _extract_answer_string
    returns the raw, un-evaluated arithmetic expression string (e.g. "68 + (2-50)*54"),
    while ground_truth is the bare target number as a string (e.g. "3354"). A
    genuinely correct solution EVALUATES to the target but is almost never textually
    identical to it. So majority_vote_at_k for Countdown can only register a "pass"
    via a degenerate non-attempt (the model outputting just the bare number with no
    arithmetic) -- it is NOT a meaningful solve-rate signal for this task.
    Use pass_at_k (which checks reward >= 2.0, i.e. genuine numeric correctness) as
    the real Countdown correctness metric instead.
    This limitation does NOT apply to GSM8K, where ground_truth genuinely is the
    literal final numeric answer, so string-matching is meaningful there.
    """
    counts = Counter(a.strip() for a in answers)
    plurality, _ = counts.most_common(1)[0]
    return 1.0 if plurality == str(ground_truth).strip() else 0.0


def _extract_answer_string(text: str, task: str) -> str:
    if task == "countdown":
        # Returns the raw, un-evaluated arithmetic expression string (e.g. "68 + (2-50)*54")
        # This is compared against ground_truth (bare target number string, e.g. "3354") in
        # majority_vote_at_k, which is a fragile metric for this task -- see majority_vote_at_k
        # docstring for details.
        return extract_answer_expr(text) or ""
    match = GSM8K_ANSWER_RE.search(text)
    return match.group(1).strip() if match else ""


def build_scaling_curve(model_path: str, dataset, task: str, ks: list[int],
                          temperature: float = 0.8, max_new_tokens: int = 256) -> dict:
    """For each prompt, sample max(ks) completions once, then compute
    pass@k and majority-vote@k for every k in `ks` by slicing that same
    sample -- avoids re-generating per k.

    For Countdown: ground_truth is set to the bare target number (e.g. "3354"),
    which is compared against extracted answer strings that are raw arithmetic
    expressions (e.g. "68 + (2-50)*54"). This mismatch makes majority_vote_at_k
    an unreliable metric for Countdown -- prefer pass_at_k instead. See
    majority_vote_at_k docstring for details.
    """
    from rewards.countdown_verifier import countdown_reward
    from rewards.gsm8k_verifier import gsm8k_reward
    reward_fn = {
        "countdown": lambda text, meta: countdown_reward(text, meta["numbers"], meta["target"]),
        "gsm8k": lambda text, meta: gsm8k_reward(text, meta["ground_truth"]),
    }[task]

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.eval()

    max_k = max(ks)
    curve = {k: {"pass_at_k": [], "majority_at_k": []} for k in ks}

    for example in dataset:
        enc = tokenizer(example["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens,
                num_return_sequences=max_k, pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rewards = [reward_fn(t, example) for t in texts]
        answers = [_extract_answer_string(t, task) for t in texts]
        ground_truth = str(example["target"]) if task == "countdown" else str(example["ground_truth"])

        for k in ks:
            curve[k]["pass_at_k"].append(pass_at_k(rewards, k))
            curve[k]["majority_at_k"].append(majority_vote_at_k(answers[:k], ground_truth))

    return {
        k: {
            "pass_at_k": sum(v["pass_at_k"]) / len(v["pass_at_k"]),
            "majority_at_k": sum(v["majority_at_k"]) / len(v["majority_at_k"]),
        }
        for k, v in curve.items()
    }
