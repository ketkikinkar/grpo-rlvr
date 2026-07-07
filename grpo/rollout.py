from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    prompt_ids: torch.Tensor        # (B, P) left-padded prompt token ids, B = num_prompts * group_size
    completion_ids: torch.Tensor    # (B, C) right-padded completion token ids
    completion_mask: torch.Tensor   # (B, C) 1 for real generated tokens, 0 for padding
    completion_texts: list[str]     # decoded completion text, per row
    prompt_index: list[int]         # which original prompt (0-indexed) each row came from


def _build_completion_mask(completion_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """1 for every generated completion token up to and including the first
    EOS token in each row; 0 for everything after (true padding).

    Pure/model-free so it can be unit tested in isolation. Handles the common
    case `pad_token_id == eos_token_id` correctly: a naive
    `(completion_ids != pad_token_id)` mask would zero out the EOS token
    itself in that case, and `torch.where(seen_eos <= 1, mask, 0)` can only
    ever turn an existing 1 into a 0 (never the reverse), so the EOS token
    would incorrectly stay excluded. Instead we derive the mask purely from
    "has an EOS occurred strictly before this position", which is 0 (i.e.
    included) at the EOS position itself and at every position before it, and
    1 (i.e. excluded, mask=0) at every position after it. Rows that never
    emit EOS within max_new_tokens keep `seen_eos_before` at 0 everywhere, so
    the whole row stays included.
    """
    eos_positions = (completion_ids == eos_token_id).long()
    seen_eos_before = eos_positions.cumsum(dim=1) - eos_positions
    return (seen_eos_before == 0).long()


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
    completion_mask = _build_completion_mask(
        completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id
    )

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
