import pytest
import torch

from nanotabicl.train import conditional_logits, factorization_gap, loss_fn


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


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.8, -0.1]))
        self.calls = []

    def forward(self, x, y):
        self.calls.append((x.detach().clone(), y.detach().clone()))
        return x[:, y.shape[1]:].sum(-1, keepdim=True) * self.weight


def test_conditioning_preserves_context_and_enumerates_queries():
    model = RecordingModel()
    x = torch.arange(5).float().reshape(1, 5, 1)
    target = torch.tensor([[0., 2., 0., 2., 0.]])
    condition = torch.tensor([[1., 3., 1., 3., 1.]])
    pred = conditional_logits(model, x, target, condition, n_train=3, n_classes=4)
    assert pred.shape == (1, 2, 4, 4)
    for value, (features, labels) in enumerate(model.calls):
        torch.testing.assert_close(features[..., :-1], x)
        torch.testing.assert_close(features[:, :3, -1], condition[:, :3])
        assert (features[:, 3:, -1] == value).all()
        torch.testing.assert_close(labels, target[:, :3])


def test_four_view_ce_and_additive_penalty():
    model = RecordingModel()
    x = torch.tensor([[[0.], [1.], [2.], [3.]]])
    # Deliberately noncontiguous class supports: A={0,2}, B={1,3}.
    y = torch.tensor([[[0., 1.], [2., 3.], [0., 3.], [2., 1.]]])
    baseline, metrics = loss_fn(model, x, y, 2)
    baseline_calls = model.calls.copy()
    model.calls.clear()
    penalized, penalized_metrics = loss_fn(model, x, y, 2, lambda_fg=0.3)
    assert len(model.calls) == len(baseline_calls) == 10
    for before, after in zip(baseline_calls, model.calls):
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right)
    torch.testing.assert_close(penalized, baseline + 0.3 * metrics["factorization_gap"])
    for name in metrics:
        torch.testing.assert_close(metrics[name], penalized_metrics[name])

    # Independent reference: four direct forwards using the observed other label.
    a, b = y.unbind(-1)
    reference = []
    for features, target, support in [
        (x, a, [0, 2]), (x, b, [1, 3]),
        (torch.cat([x, b[..., None]], -1), a, [0, 2]),
        (torch.cat([x, a[..., None]], -1), b, [1, 3]),
    ]:
        logits = model(features, target[:, :2])[..., support]
        labels = (target[:, 2:] == support[1]).long()
        reference.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten()))
    torch.testing.assert_close(baseline, torch.stack(reference).mean())
    base_grad = torch.autograd.grad(baseline, model.weight)[0]
    penalized.backward()
    assert torch.isfinite(model.weight.grad).all()
    assert not torch.allclose(base_grad, model.weight.grad)
