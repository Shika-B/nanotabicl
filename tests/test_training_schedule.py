import importlib

import numpy as np
import pytest
import torch

from nanotabicl import load_config
from nanotabicl.train import build_model, loss_fn, lr_schedule, train, validation_tables


TINY = ["model.embed_dim=8", "model.col_nhead=2", "model.row_nhead=2", "model.icl_nhead=2",
        "model.col_num_blocks=1", "model.row_num_blocks=1", "model.icl_num_blocks=1", "model.n_cls_rows=4",
        "data.min_seq_len=16", "data.max_seq_len=16", "data.max_features=2", "data.max_classes=2",
        "data.micro_batch_size=1", "data.num_workers=0", "data.filter_unpredictable=false"]


def test_fixed_validation_rng_and_reload(tmp_path):
    cfg = load_config([], TINY + [f"out_dir={tmp_path}"])
    torch.manual_seed(42)
    np.random.seed(42)
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    tables = validation_tables(cfg)
    assert len(tables) == 128
    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    reloaded = validation_tables(cfg)
    for (x, y, n), (xx, yy, nn) in zip(tables, reloaded):
        torch.testing.assert_close(x, xx)
        torch.testing.assert_close(y, yy)
        assert n == nn
    cfg.validation.seed += 1
    with pytest.raises(ValueError, match="configuration changed"):
        validation_tables(cfg)


def test_warmup_and_cosine_endpoints():
    cfg = load_config().optim
    cfg.warmup_steps = 10
    assert lr_schedule(0, cfg) == pytest.approx(cfg.lr / 10)
    assert lr_schedule(9, cfg) == cfg.lr
    cfg.max_steps = 110
    assert lr_schedule(59, cfg) == pytest.approx((cfg.lr + cfg.min_lr) / 2)
    assert lr_schedule(109, cfg) == cfg.min_lr
    assert all(lr_schedule(i, cfg) >= lr_schedule(i + 1, cfg) for i in range(9, 109))


def test_validation_does_not_stop_training_or_select_checkpoint(tmp_path, monkeypatch):
    module = importlib.import_module("nanotabicl.train")
    monkeypatch.setattr(module, "validate", lambda *args: {"loss": 1.0})
    cfg = load_config([], TINY + [f"out_dir={tmp_path}", "optim.max_steps=4", "optim.accum_steps=1",
                                  "optim.lr=0.001", "optim.min_lr=0.0005",
                                  "optim.warmup_steps=0", "validation.every=1",
                                  "validation.n_tables=2"])
    train(cfg)
    latest = torch.load(tmp_path / "latest.pt")
    assert latest["step"] == 4 and "convergence" not in latest
    assert latest["optimizer"]["param_groups"][0]["lr"] == 0.0005
    assert not (tmp_path / "best.pt").exists()
    cfg.optim.max_steps = 5
    model = train(cfg)
    assert torch.load(tmp_path / "latest.pt")["step"] == 5
    for name, value in torch.load(tmp_path / "latest.pt")["model"].items():
        torch.testing.assert_close(model.state_dict()[name], value)


def test_direct_views_match_enumerated_loss_and_gradients():
    torch.manual_seed(42)
    model = build_model(load_config([], TINY))
    x = torch.randn(1, 8, 2)
    y = torch.tensor([[[0., 0.], [1., 1.], [0., 1.], [1., 0.]] * 2])
    direct, _ = loss_fn(model, x, y, 4, compute_gap=False)
    direct.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    enumerated, _ = loss_fn(model, x, y, 4)
    enumerated.backward()
    torch.testing.assert_close(direct, enumerated)
    for p, grad in zip(model.parameters(), gradients):
        torch.testing.assert_close(p.grad, grad, atol=1e-6, rtol=1e-4)
