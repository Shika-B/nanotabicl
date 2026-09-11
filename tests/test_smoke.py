import numpy as np
import pytest
import torch

from nanotabicl import load_config
from nanotabicl.data import PriorDataset
from nanotabicl.eval import evaluate
from nanotabicl.interface import NanoTabICLClassifier, NanoTabICLRegressor
from nanotabicl.train import build_model, train

TINY = ["model.embed_dim=16", "model.col_num_blocks=1", "model.row_num_blocks=1", "model.icl_num_blocks=2",
        "model.col_nhead=2", "model.row_nhead=2", "model.icl_nhead=2", "model.n_cls_rows=8",
        "data.micro_batch_size=2", "data.min_seq_len=32", "data.max_seq_len=64", "data.max_features=6", "data.num_workers=0",
        "data.filter_unpredictable=false", "optim.max_steps=2", "optim.accum_steps=2", "save_every=1", "log_every=1",
        "validation.n_tables=2", "validation.every=1", "optim.warmup_steps=0"]


@pytest.mark.parametrize("task,n_targets,lambda_fg", [
    ("classification", 1, 0), ("regression", 1, 0),
    ("classification", 2, 0), ("classification", 2, 0.3),
])
def test_train_resume_eval(tmp_path, task, n_targets, lambda_fg):
    cfg = load_config([], TINY + [f"data.task={task}", f"out_dir={tmp_path}",
                                 f"data.n_targets={n_targets}", f"optim.lambda_fg={lambda_fg}"])
    train(cfg)
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists() and (tmp_path / "validation.pt").exists()
    cfg.optim.max_steps = 3
    model = train(cfg)  # resumes from latest.pt and trains one more step
    checkpoint = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert checkpoint["step"] == 3
    assert all(t.dtype == torch.float32 for t in checkpoint["model"].values() if t.is_floating_point())
    assert all(state["momentum_buffer"].dtype == torch.float32
               for state in checkpoint["optimizer"]["state"].values())
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 3
    scores = evaluate(model.eval(), task, max_rows=100, include_pairs=False)
    assert len(scores) > 0 and all(np.isfinite(s) for s in scores.values())


def test_config_overrides():
    cfg = load_config(["configs/small.yaml", "configs/stage2.yaml"], ["optim.lr=1e-3", "model.ln_bias=false"])
    assert cfg.optim.lr == 1e-3 and cfg.model.ln_bias is False  # overrides win
    assert cfg.data.max_seq_len == 10240 and cfg.data.micro_batch_size == 1  # from stage2.yaml
    assert cfg.model.embed_dim == 64 and cfg.data.max_features == 20  # from small.yaml
    with pytest.raises(KeyError):
        load_config([], ["optim.typo=1"])


def test_prior_batch():
    overrides = ["data.micro_batch_size=3", "data.min_seq_len=50", "data.max_seq_len=60", "data.max_features=7"]
    x, y, n_train = PriorDataset(load_config([], overrides).data).sample_batch()
    assert x.shape[:2] == y.shape and x.shape[0] == 3 and 50 <= x.shape[1] <= 60 and 1 <= x.shape[2] <= 7
    assert torch.isfinite(x).all() and torch.isfinite(y).all() and 1 <= n_train < x.shape[1]
    for yi in y:  # train and test rows contain the same classes
        assert torch.equal(yi[:n_train].unique(), yi[n_train:].unique()) and yi.unique().numel() >= 2


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_sklearn_interface(task):
    cfg = load_config([], TINY + [f"data.task={task}"])
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 5)), rng.integers(0, 3, size=40) + 5
    x[0, 0] = np.nan  # imputed with the training mean
    x[:, 4] = 1.0  # constant feature is dropped
    if task == "classification":
        clf = NanoTabICLClassifier(model=build_model(cfg), device="cpu", n_estimators=3).fit(x[:30], y[:30])
        proba = clf.predict_proba(x[30:])
        assert proba.shape == (10, 3) and np.allclose(proba.sum(axis=1), 1) and set(clf.predict(x[30:])) <= {5, 6, 7}
        assert np.allclose(proba, clf.predict_proba(x[30:]))  # seeded ensemble is deterministic
    else:
        reg = NanoTabICLRegressor(model=build_model(cfg), device="cpu").fit(x[:30], y[:30].astype(float))
        assert reg.predict(x[30:]).shape == (10,) and reg.predict_quantiles(x[30:]).shape == (10, 3)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("filtered", [False, True])
def test_two_target_batch(task, filtered):
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = load_config(["configs/small.yaml"], [
        "data.n_targets=2", "data.max_classes=4", "data.micro_batch_size=2",
        "data.min_seq_len=64", "data.max_seq_len=64", "data.max_features=6",
        f"data.task={task}", f"data.filter_unpredictable={filtered}",
    ])
    x, y, n_train = PriorDataset(cfg.data).sample_batch()
    assert y.shape == (*x.shape[:2], 2)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    if task == "classification":
        assert (y == y.long()).all() and y.min() >= 0 and y.max() < 4
        for target in y.transpose(1, 2).flatten(0, 1):
            assert target[:n_train].unique().numel() >= 2
            assert torch.equal(target[:n_train].unique(), target[n_train:].unique())
    else:
        assert torch.allclose(y[:, :n_train].mean(dim=1), torch.zeros(2, 2), atol=1e-5)


def test_class_split_checks_each_target():
    from nanotabicl.data import fix_class_split

    # Flattening targets would conceal that the second target is constant.
    y = torch.tensor([[0, 1], [1, 1], [0, 1], [1, 1]])
    assert fix_class_split(y, n_train=2) is None


def test_load_legacy_precision_checkpoint(tmp_path):
    from dataclasses import asdict
    from nanotabicl.interface import load_model

    cfg = load_config([], TINY)
    config = asdict(cfg)
    config["optim"]["amp_dtype"] = "float32"  # retired field in existing checkpoints
    path = tmp_path / "legacy.pt"
    torch.save({"config": config, "model": build_model(cfg).state_dict()}, path)
    model, loaded_cfg = load_model(str(path), device="cpu")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert not hasattr(loaded_cfg.optim, "amp_dtype")
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cudnn.allow_tf32
