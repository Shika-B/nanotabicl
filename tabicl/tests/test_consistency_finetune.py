import json
import copy
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

from tabicl._model.tabicl import TabICL
from tabicl.train.finetune_consistency import (
    GraphSCM, PriorConfig, learning_rate, loss_fn, main, sample_table, seed_all,
    backward_table, atomic_save, argument_parser,
)


def test_graph_single_and_joint_targets():
    for classes, shape in [(2, (24,)), ([2, 3], (24, 2))]:
        seed_all(7)
        x, y = GraphSCM(seq_len=24, num_features=2, max_features=2,
                        num_classes=classes, config=PriorConfig())()
        assert x.shape == (24, 2) and y.shape == shape
        assert y.min() >= 0 and y.max() < (classes if isinstance(classes, int) else max(classes))


class Predictions(torch.nn.Module):
    """Prescribed coherent joint [[.5,.1],[.2,.2]], with trainable logits."""
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([
            [.6, .4], [.7, .3], [5/6, 1/6], [.5, .5], [5/7, 2/7], [1/3, 2/3],
        ]).log())
        self.calls = []

    def forward(self, x, y):
        idx = len(self.calls)
        self.calls.append((x.detach().clone(), y.detach().clone()))
        return self.logits[idx].expand(x.shape[0], x.shape[1] - y.shape[1], -1)


def test_four_view_loss_gap_gradients_and_hidden_labels():
    table = {"x": torch.zeros(1, 4, 2), "y": torch.tensor([[[0, 1], [1, 0], [0, 1], [1, 0]]]),
             "classes": [2, 2]}
    model = Predictions()
    loss, metrics = loss_fn(model, table, 2, .5)
    assert metrics["factorization_gap"].item() == pytest.approx(0, abs=1e-7)
    loss.backward()
    assert torch.isfinite(model.logits.grad).all()
    assert len(model.calls) == 6
    for i, (x, y) in enumerate(model.calls):
        assert y.shape == (1, 2)
        if i >= 2:
            assert x.shape == (1, 4, 3)
            assert torch.all(x[:, 2:, -1] == (i % 2))
            condition = table["y"][:, :2, 0 if i < 4 else 1]
            torch.testing.assert_close(x[:, :2, -1], condition.float())
    # Perturb just the marginal: FG must affect gradients, with the same forwards.
    grads = []
    for penalty in [0., .5]:
        m = Predictions()
        with torch.no_grad():
            m.logits[0, 0] += .7
        loss, stats = loss_fn(m, table, 2, penalty)
        assert stats["factorization_gap"] > 0
        assert loss.item() == pytest.approx((stats["ce"] + penalty * stats["factorization_gap"]).item())
        loss.backward()
        grads.append(m.logits.grad)
        assert len(m.calls) == 6
    assert not torch.allclose(*grads)


def test_schedule():
    args = SimpleNamespace(warmup_steps=2, steps=10, lr=1e-5, end_lr=1e-6)
    assert learning_rate(1, args) == pytest.approx(5e-6)
    assert learning_rate(2, args) == pytest.approx(1e-5)
    assert learning_rate(10, args) == pytest.approx(1e-6)


@pytest.mark.parametrize("interrupt_data", [False, True])
def test_paired_runner_real_model(tmp_path, capfd, monkeypatch, interrupt_data):
    config = dict(max_classes=3, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
                  col_feature_group=False, col_ssmax=False, row_num_blocks=1, row_nhead=2,
                  row_num_cls=1, icl_num_blocks=1, icl_nhead=2, icl_ssmax=False)
    seed_all(5)
    original = TabICL(**config)
    ckpt = tmp_path / "initial.ckpt"
    torch.save({"config": config, "state_dict": original.state_dict()}, ckpt)
    output = tmp_path / "experiment"
    argv = ["--steps", "2", "--checkpoint", str(ckpt), "--output", str(output), "--devices", "cpu", "cpu",
          "--log-dir", str(tmp_path / "logs"),
          "--context-rows", "8", "--query-rows", "4", "--min-features", "2", "--max-features", "2",
          "--max-classes", "2", "--tables-per-step", "1", "--validation-tables", "2",
          "--validate-every", "1", "--save-every", "1"]
    if interrupt_data:
        from tabicl.train import finetune_consistency as runner
        def stop_after_table(*args, **kwargs):
            result = sample_table(*args, **kwargs)
            signal.raise_signal(signal.SIGTERM)
            return result
        monkeypatch.setattr(runner, "sample_table", stop_after_table)
        assert main(argv) == 130
        monkeypatch.setattr(runner, "sample_table", sample_table)
        assert main(["--resume", "--output", str(output)]) == 0
    else:
        assert main(argv) == 0
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""
    log_path = Path(json.loads((output / "experiment.json").read_text())["log_path"])
    records = [json.loads(line[line.index("{"):]) for line in log_path.read_text().splitlines() if 'INFO {' in line]
    logs = [[r for r in records if r.get("lambda_fg") == p and r["kind"] == "train"] for p in [0., .5]]
    assert logs[0][0]["ce"] == logs[1][0]["ce"]
    assert logs[0][0]["factorization_gap"] == logs[1][0]["factorization_gap"]
    assert len({r["pid"] for r in records if r["kind"] == "arm_start"}) == 2
    assert len(list((tmp_path / "logs").glob("*.log"))) == 1
    saved = [torch.load(output / f"lambda_{p}" / "latest.ckpt", weights_only=True) for p in ["0", "0.5"]]
    assert all(c["step"] == 2 for c in saved)
    assert all(t.dtype == torch.float32 for c in saved for t in c["state_dict"].values() if t.is_floating_point())
    assert any(not torch.equal(saved[0]["state_dict"][k], saved[1]["state_dict"][k]) for k in original.state_dict())
    # Native checkpoint loading still works without the experiment runner.
    from tabicl import TabICLClassifier
    clf = TabICLClassifier(model_path=str(output / "lambda_0.5" / "latest.ckpt"), device="cpu")
    clf._load_model()
    torch.testing.assert_close(clf.model_.state_dict(), saved[1]["state_dict"])
    table = torch.load(output / "data" / "step_000001.pt", weights_only=True)[0]
    params = SimpleNamespace(min_features=2, max_features=2, max_classes=2, context_rows=8, query_rows=4)
    replay = sample_table(params, table["seed"])
    torch.testing.assert_close(replay["x"], table["x"])
    torch.testing.assert_close(replay["y"], table["y"])
    # Resume one arm from step 1 while the other is already done. Step 2 must
    # reproduce uninterrupted weights AND optimizer state, without restarting LR.
    directory = output / "lambda_0"
    first = torch.load(directory / "step_000001.ckpt", weights_only=True)
    atomic_save(first, directory / "latest.ckpt")
    assert main(["--resume", "--output", str(output)]) == 0
    resumed = torch.load(directory / "latest.ckpt", weights_only=True)
    torch.testing.assert_close(resumed["state_dict"], saved[0]["state_dict"], rtol=0, atol=0)
    torch.testing.assert_close(resumed["optimizer_state"], saved[0]["optimizer_state"], rtol=0, atol=0)
    assert len(list((tmp_path / "logs").glob("*.log"))) == 1
    assert main(["--resume", "--output", str(output), "--lr", "0.1"]) == 1
    assert "Resume cannot change" in log_path.read_text()
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""


def test_replayed_backward_matches_full_graph():
    torch.set_num_threads(1)
    config = dict(max_classes=3, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
                  col_feature_group=False, col_ssmax=False, row_num_blocks=1, row_nhead=2,
                  row_num_cls=1, icl_num_blocks=1, icl_nhead=2, icl_ssmax=False, recompute=True)
    seed_all(4)
    model = TabICL(**config).train()
    other = copy.deepcopy(model)
    table = {"x": torch.randn(1, 6, 2), "y": torch.tensor([[[0, 0], [1, 1], [0, 2], [1, 0], [0, 1], [1, 2]]]),
             "classes": [2, 3], "n_context": 3}
    loss, expected = loss_fn(model, table, 3, .5)
    (loss / 4).backward()
    actual = backward_table(other, table, .5, 4)
    torch.testing.assert_close(actual, expected)
    for p, q in zip(model.parameters(), other.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=2e-5)


def test_broader_defaults():
    args = argument_parser().parse_args([])
    assert args.context_rows == [128, 512, 2048, 8192]
    assert args.max_features == 100 and args.max_classes == 10 and args.tables_per_step == 16
    assert args.devices == ["cuda:0"]
