# GRPO / RLVR Reasoning Trainer

From-scratch implementation of DeepSeek-R1-Zero-style RL post-training:
GRPO (group-relative advantages, no critic) + rule-based verifiable rewards (RLVR),
trained on Countdown and GSM8K starting from Qwen2.5 base checkpoints.

## Status
- [ ] M1: Reference repo reproduced
  - M1 reference-repo repro pending GPU access.
- [ ] M2: Verifiers implemented + unit-tested
- [ ] M3: Own GRPO loop trains stably
- [ ] M4: Trained checkpoint beats base
- [ ] M5: Test-time scaling curve + cost report

## Reference
See `runs/reference_transcripts/baseline_aha_examples.md` for a reference-repo
transcript captured before building the from-scratch loop.
