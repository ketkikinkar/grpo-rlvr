import torch

from grpo.loss import grpo_loss

def _toy_batch(B=2, T=3):
    policy_logprobs = torch.zeros(B, T, requires_grad=True)
    old_logprobs = torch.zeros(B, T)
    ref_logprobs = torch.zeros(B, T)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.ones(B, T, dtype=torch.long)
    return policy_logprobs, old_logprobs, ref_logprobs, advantages, mask

def test_loss_is_finite_and_scalar():
    policy_lp, old_lp, ref_lp, adv, mask = _toy_batch()
    loss, metrics = grpo_loss(policy_lp, ref_lp, old_lp, adv, mask)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

def test_zero_kl_when_policy_equals_reference():
    policy_lp, old_lp, ref_lp, adv, mask = _toy_batch()
    _, metrics = grpo_loss(policy_lp, ref_lp, old_lp, adv, mask)
    assert torch.isclose(metrics["kl"], torch.tensor(0.0), atol=1e-5)

def test_masked_tokens_do_not_affect_loss():
    policy_lp, old_lp, ref_lp, adv, mask = _toy_batch(B=2, T=4)
    mask[:, 2:] = 0  # last two tokens are padding
    loss_masked, _ = grpo_loss(policy_lp, ref_lp, old_lp, adv, mask)

    policy_lp2 = policy_lp.clone().detach().requires_grad_(True)
    policy_lp2.data[:, 2:] = 999.0  # garbage in masked positions
    loss_full, _ = grpo_loss(policy_lp2, ref_lp, old_lp, adv, mask)
    assert torch.isclose(loss_masked, loss_full, atol=1e-4)

def test_clipping_bounds_the_ratio_contribution():
    B, T = 1, 1
    policy_lp = torch.tensor([[2.0]], requires_grad=True)  # ratio = e^2 vs old
    old_lp = torch.tensor([[0.0]])
    ref_lp = torch.tensor([[0.0]])
    adv = torch.tensor([1.0])
    mask = torch.ones(B, T, dtype=torch.long)
    loss, metrics = grpo_loss(policy_lp, ref_lp, old_lp, adv, mask, clip_eps=0.2)
    # with a huge positive ratio and positive advantage, clipping caps the
    # surrogate at (1+eps)*A rather than letting ratio*A explode
    assert metrics["clip_frac"] > 0.0

def test_gradients_flow_to_policy_logprobs():
    policy_lp, old_lp, ref_lp, adv, mask = _toy_batch()
    loss, _ = grpo_loss(policy_lp, ref_lp, old_lp, adv, mask)
    loss.backward()
    assert policy_lp.grad is not None
    assert torch.any(policy_lp.grad != 0)
