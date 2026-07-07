from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    prompt_ids: torch.Tensor        # (B, P) left-padded prompt token ids, B = num_prompts * group_size
    completion_ids: torch.Tensor    # (B, C) right-padded completion token ids
    completion_mask: torch.Tensor   # (B, C) 1 for real generated tokens, 0 for padding
    completion_texts: list[str]     # decoded completion text, per row
    prompt_index: list[int]         # which original prompt (0-indexed) each row came from


def generate_rollouts(model, tokenizer, prompts: list[str], group_size: int,
                       max_new_tokens: int, temperature: float) -> RolloutBatch:
    """Sample `group_size` completions per prompt from `model` at `temperature`.

    Uses HF `generate` with `num_return_sequences=group_size`, which is the
    default rollout backend per the project scope; swap in vLLM behind the
    same function signature later (Task 4b) without touching callers.
    """
    model.eval()
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
        )

    completion_ids = out[:, prompt_len:]
    completion_mask = (completion_ids != tokenizer.pad_token_id).long()
    # first pad token after the model's own EOS should still count as "generated"
    # up to and including EOS; mask everything after via cumulative product
    eos_positions = (completion_ids == tokenizer.eos_token_id).long()
    seen_eos = eos_positions.cumsum(dim=1)
    completion_mask = torch.where(seen_eos <= 1, completion_mask, torch.zeros_like(completion_mask))

    completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    prompt_index = [i // group_size for i in range(len(prompts) * group_size)]

    prompt_ids = enc["input_ids"].repeat_interleave(group_size, dim=0)

    return RolloutBatch(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_texts=completion_texts,
        prompt_index=prompt_index,
    )
