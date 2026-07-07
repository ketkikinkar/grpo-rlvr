import argparse
import json
import os
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.countdown import build_countdown_dataset
from data.gsm8k import build_gsm8k_dataset
from grpo.trainer import GRPOTrainer

_TASK_BUILDERS = {
    "countdown": lambda cfg: build_countdown_dataset(n_examples=cfg["num_train_steps"] * cfg["prompts_per_step"], seed=cfg["seed"]),
    "gsm8k": lambda cfg: build_gsm8k_dataset(split="train"),
}


def _init_logging(cfg, args, run_dir):
    """Try W&B in offline mode (no network/auth required); fall back to a
    local JSONL metrics file if wandb is unavailable or fails to init, so a
    logging failure never blocks the actual training loop.
    """
    try:
        import wandb
        wandb.init(project=cfg["wandb_project"], name=args.run_id, config=cfg, mode="offline")

        def log_fn(metrics):
            wandb.log(metrics)

        def finish_fn():
            wandb.finish()

        return log_fn, finish_fn
    except Exception as e:
        print(f"[train] wandb init failed ({e!r}); falling back to stdout + JSONL logging")
        os.makedirs(run_dir, exist_ok=True)
        jsonl_path = f"{run_dir}/metrics.jsonl"
        jsonl_file = open(jsonl_path, "a")

        def log_fn(metrics):
            print(f"[train] step {metrics.get('step')}: {metrics}")
            jsonl_file.write(json.dumps(metrics) + "\n")
            jsonl_file.flush()

        def finish_fn():
            jsonl_file.close()

        return log_fn, finish_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    run_dir = f"runs/{args.run_id}"
    os.makedirs(run_dir, exist_ok=True)

    log_metrics, finish_logging = _init_logging(cfg, args, run_dir)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy is trainable (backward + optimizer.step()) -> fp32 for numerical
    # stability on MPS (see Task 7). Ref is frozen/inference-only -> fp16 is
    # fine and saves memory. The config's `bf16` flag is retained for future
    # cloud-GPU use but intentionally not wired to policy dtype here.
    policy = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.float32).to(device)
    ref = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.float16).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if cfg["gradient_checkpointing"]:
        policy.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"])

    trainer = GRPOTrainer(
        policy=policy, ref=ref, tokenizer=tokenizer, optimizer=optimizer,
        task=cfg["task"], group_size=cfg["group_size"],
        max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"],
        clip_eps=cfg["clip_eps"], kl_coef=cfg["kl_coef"], max_grad_norm=cfg["max_grad_norm"],
    )

    dataset = _TASK_BUILDERS[cfg["task"]](cfg)
    start_time = time.time()

    metrics = {}
    for step in range(cfg["num_train_steps"]):
        batch_start = (step * cfg["prompts_per_step"]) % len(dataset)
        idx = list(range(batch_start, min(batch_start + cfg["prompts_per_step"], len(dataset))))
        batch = dataset.select(idx)

        step_start = time.time()
        metrics = trainer.train_step(batch)
        metrics["step"] = step
        metrics["step_seconds"] = time.time() - step_start
        log_metrics(metrics)

        if step % cfg["save_every_n_steps"] == 0:
            ckpt_dir = f"{run_dir}/checkpoints/step_{step}"
            policy.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    # Always save a final checkpoint so downstream inspection has the
    # fully-trained weights available, even if num_train_steps isn't a
    # multiple of save_every_n_steps.
    final_step = cfg["num_train_steps"] - 1
    final_ckpt_dir = f"{run_dir}/checkpoints/step_{final_step}"
    if not os.path.isdir(final_ckpt_dir):
        policy.save_pretrained(final_ckpt_dir)
        tokenizer.save_pretrained(final_ckpt_dir)

    elapsed_hours = (time.time() - start_time) / 3600
    manifest = {
        "run_id": args.run_id,
        "config": cfg,
        "final_metrics": metrics,
        "gpu_hours": elapsed_hours,
    }
    with open(f"{run_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    finish_logging()


if __name__ == "__main__":
    main()
