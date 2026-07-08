import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewards.countdown_verifier import countdown_reward
from rewards.gsm8k_verifier import gsm8k_reward

_TASK_REWARDS = {
    "countdown": lambda text, meta: countdown_reward(text, meta["numbers"], meta["target"]),
    "gsm8k": lambda text, meta: gsm8k_reward(text, meta["ground_truth"]),
}


def aggregate_accuracy(rewards: list[float]) -> float:
    if not rewards:
        raise ValueError("cannot aggregate accuracy over an empty rewards list")
    # rewards/countdown_verifier.py::countdown_reward and
    # rewards/gsm8k_verifier.py::gsm8k_reward return an additive tiered score
    # in {0.0, 0.5, 1.0, 1.5, 2.0} = format_tier + binary_correctness, not the
    # old binary {0.0, 1.0}. A reward of 1.0 can be achieved purely through
    # perfect format compliance with a WRONG answer (format_tier=1.0,
    # correctness=0.0), so only the max value 2.0 (perfect format AND correct
    # answer) counts as "correct" -- using `>= 1.0` would incorrectly credit
    # format-only completions.
    return sum(1.0 for r in rewards if r >= 2.0) / len(rewards)


def evaluate_checkpoint(model_path: str, dataset, task: str, batch_size: int = 8,
                         max_new_tokens: int = 256, temperature: float = 0.0) -> dict:
    """Greedy (temperature=0 -> do_sample=False) single-completion accuracy
    over `dataset`. Returns per-example rewards plus the aggregate accuracy,
    so the same rewards list can feed eval/scaling_curve.py without re-running
    generation.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # torch_dtype=torch.float32 explicitly for both base and trained
    # checkpoints so the comparison isn't confounded by mixed precision --
    # trained checkpoints are saved in fp16 (see Task 7's memory-optimization
    # change), but the base model would otherwise load in whatever dtype its
    # config specifies.
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.eval()

    rewards = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        # list(...) guards against `datasets` returning a Column object (not
        # a plain list) for `dataset[col]` on newer versions, which the
        # tokenizer rejects.
        enc = tokenizer(list(batch["prompt"]), return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=temperature > 0.0, temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, text in enumerate(texts):
            rewards.append(_TASK_REWARDS[task](text, batch[i]))

    return {"accuracy": aggregate_accuracy(rewards), "rewards": rewards, "n": len(rewards)}
