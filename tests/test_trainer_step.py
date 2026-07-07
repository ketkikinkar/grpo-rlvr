import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.countdown import build_countdown_dataset
from grpo.trainer import GRPOTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B"


def test_single_train_step_reduces_loss_is_finite_and_updates_weights():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Policy is trainable (backward + optimizer.step()) -> fp32 for numerical
    # stability; MPS backward under fp16 is untested/unvalidated (see Task 4).
    policy = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    policy.gradient_checkpointing_enable()
    # Ref is frozen/inference-only -> fp16 is fine and saves memory.
    ref = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    before = copy.deepcopy(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6)

    trainer = GRPOTrainer(
        policy=policy, ref=ref, tokenizer=tokenizer, optimizer=optimizer,
        task="countdown", group_size=4, max_new_tokens=64, temperature=0.8,
        clip_eps=0.2, kl_coef=0.04,
    )

    dataset = build_countdown_dataset(n_examples=2, seed=0)
    metrics = trainer.train_step(dataset)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert "mean_reward" in metrics and "reward_std" in metrics
    assert "kl" in metrics and "response_length" in metrics

    after = policy.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "policy weights did not change after a train_step"
