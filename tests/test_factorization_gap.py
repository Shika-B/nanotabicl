import pytest
import torch

from nanotabicl.train import factorization_gap


def test_coherent_joint_and_target_order():
    # Unequal class counts catch incorrect conditional-axis alignment.
    joint = torch.tensor([[0.05, 0.15, 0.10], [0.20, 0.10, 0.40]], dtype=torch.float64)
    p_a, p_b = joint.sum(-1), joint.sum(-2)
    p_b_given_a = joint / p_a[:, None]
    p_a_given_b = joint.T / p_b[:, None]
    assert factorization_gap(p_a, p_b, p_b_given_a, p_a_given_b).item() == pytest.approx(0)
    inconsistent = torch.full_like(p_b_given_a, 1 / 3)
    forward = factorization_gap(p_a, p_b, inconsistent, p_a_given_b)
    reverse = factorization_gap(p_b, p_a, p_a_given_b, inconsistent)
    assert forward > 0
    torch.testing.assert_close(forward, reverse)


def test_known_gap_and_mean_reduction():
    # First example: disjoint diagonal/off-diagonal joints (gap 1).
    # Second example: identical diagonal joints (gap 0).
    marginals = torch.full((1, 2, 2), 0.5)
    conditional_ab = torch.eye(2).expand(1, 2, 2, 2)
    conditional_ba = torch.stack([1 - torch.eye(2), torch.eye(2)]).unsqueeze(0)
    gap = factorization_gap(marginals, marginals, conditional_ab, conditional_ba)
    assert gap.item() == pytest.approx(0.5)


def test_gradients_through_all_predictions():
    generator = torch.Generator().manual_seed(42)
    logits = [torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
              for shape in [(2, 2), (2, 3), (2, 2, 3), (2, 3, 2)]]

    def gap_from_logits(*inputs):
        return factorization_gap(*(x.softmax(-1) for x in inputs))

    assert torch.autograd.gradcheck(gap_from_logits, tuple(logits))
    gap_from_logits(*logits).backward()
    for x in logits:
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_rejects_broadcasted_conditionals():
    with pytest.raises(ValueError, match="conditionals"):
        factorization_gap(torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, 2))
