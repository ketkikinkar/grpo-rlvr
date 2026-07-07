import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo.rollout import generate_rollouts

MODEL_ID = "Qwen/Qwen2.5-0.5B"


def test_generate_rollouts_shape_and_group_repetition():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)

    prompts = ["Say hello.", "Count to three."]
    group_size = 4
    batch = generate_rollouts(model, tokenizer, prompts, group_size=group_size,
                               max_new_tokens=16, temperature=0.8)

    assert batch.completion_ids.shape[0] == len(prompts) * group_size
    assert len(batch.completion_texts) == len(prompts) * group_size
    # rows 0..3 come from prompt 0, rows 4..7 from prompt 1
    assert batch.prompt_index[0] == 0 and batch.prompt_index[group_size] == 1
    # completion_mask must be 1 only over generated (non-pad) tokens
    assert batch.completion_mask.shape == batch.completion_ids.shape
    assert batch.completion_mask.dtype == torch.long
