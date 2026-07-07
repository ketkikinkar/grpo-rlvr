from datasets import load_dataset

from data.prompts import build_prompt
from rewards.gsm8k_verifier import extract_gsm8k_ground_truth


def build_gsm8k_dataset(split: str = "train"):
    raw = load_dataset("openai/gsm8k", "main", split=split)

    def _map(example):
        gt = extract_gsm8k_ground_truth(example["answer"])
        return {"prompt": build_prompt(example["question"]), "ground_truth": gt}

    mapped = raw.map(_map, remove_columns=raw.column_names)
    return mapped.filter(lambda ex: ex["ground_truth"] is not None)
