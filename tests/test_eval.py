import io
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from nanotabicl import load_config
from nanotabicl.eval import encode_features, evaluate_pair, load_pair_dataset, probability_scores
from nanotabicl.runtime import build_model


def test_train_only_encoding():
    train, test = encode_features(np.array([["red", "1"], ["blue", "2"]]),
                                  np.array([["green", "3"]]))
    assert train.dtype == test.dtype == np.float32
    np.testing.assert_array_equal(test, [[-1, 3]])


def test_probability_scores():
    scores = probability_scores(np.array([[0.75, 0.25], [0.5, 0.5]]), np.array([0, 1]))
    assert scores["accuracy"] == 0.5
    assert scores["log_loss"] == pytest.approx(-np.log([0.75, 0.5]).mean())
    assert scores["brier"] == pytest.approx((0.125 + 0.5) / 2)


def test_cached_loaders_exclude_targets(tmp_path):
    for dataset_id, name in [(19, "car"), (76, "nursery")]:
        with ZipFile(tmp_path / f"{dataset_id}.zip", "w") as archive:
            archive.writestr(f"{name}.data", "feature,condition,target\n")
        x, a, b = load_pair_dataset(name, str(tmp_path))
        assert x.tolist() == [["feature"]] and a.tolist() == ["target"] and b.tolist() == ["condition"]
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        for subject in ("mat", "por"):
            archive.writestr(f"student-{subject}.csv", 'school;age;G1;G2;G3\nGP;16;20;9;10\n')
    with ZipFile(tmp_path / "320.zip", "w") as archive:
        archive.writestr("student.zip", payload.getvalue())
    for name in ("student_mat", "student_por"):
        x, a, b = load_pair_dataset(name, str(tmp_path))
        assert x.tolist() == [["GP", "16"]] and a.tolist() == [0] and b.tolist() == [1]


def test_pair_evaluation_repeatable_and_joint_scores():
    torch.manual_seed(0)
    cfg = load_config([], ["model.embed_dim=8", "model.col_nhead=2", "model.row_nhead=2",
                           "model.icl_nhead=2", "model.col_num_blocks=1", "model.row_num_blocks=1",
                           "model.icl_num_blocks=1", "model.n_cls_rows=4"])
    model = build_model(cfg).eval()
    rng = np.random.default_rng(1)
    x, a, b = rng.normal(size=(40, 2)), np.arange(40) % 2, np.arange(40) % 3
    first = evaluate_pair(model, x, a, b, n_train=16, n_test=8)
    assert first == evaluate_pair(model, x, a, b, n_train=16, n_test=8)
    assert all(np.isfinite(value) for value in first.values())
    assert 0 <= first["factorization_gap"] <= 1
    assert first["joint_ab/log_loss"] == pytest.approx(first["a/log_loss"] + first["b_given_a/log_loss"], abs=1e-6)
    assert first["joint_ba/log_loss"] == pytest.approx(first["b/log_loss"] + first["a_given_b/log_loss"], abs=1e-6)
