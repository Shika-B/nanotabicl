import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("full_eval", ROOT / "eval_finetuned_tabarena.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_nested_contexts_identical_queries_and_class_alignment():
    frame = pd.DataFrame({"row_id": np.arange(12, dtype=float)})
    # Class 1 exists only in the query: output columns must be aligned globally.
    labels = pd.Series([0, 2] * 4 + [0, 1, 2, 1])
    calls = []

    class Task:
        task_id = 7
        target_name = "target"
        name = "fake"

        def get_dataset(self):
            return self

        def get_data(self, **kwargs):
            return frame, labels, [False], None

        def get_train_test_split_indices(self, **kwargs):
            return np.arange(8), np.arange(8, 12)

    class Classifier:
        def fit(self, x, y):
            self.classes_ = np.array([2, 0])
            self.call = {"context": x[:, 0].copy(), "queries": []}
            calls.append(self.call)

        def predict_proba(self, x):
            self.call["queries"].extend(x[:, 0].tolist())
            return np.tile([.75, .25], (len(x), 1))

    settings = {"seed": 0, "context_size": 2, "context_sizes": [2, 5, 20],
                "max_test_rows": 4, "query_batch_size": 2}
    reference = {"n_classes": 3, "folds": [{"fold": 0, "n_context": 2, "n_test": 4}]}
    result = evaluation.evaluate_task(Task(), reference, settings, [Classifier, Classifier])
    assert len(calls) == 6
    for i in range(0, 6, 2):
        np.testing.assert_array_equal(calls[i]["context"], calls[i+1]["context"])
    np.testing.assert_array_equal(calls[0]["context"], calls[2]["context"][:2])
    np.testing.assert_array_equal(calls[2]["context"], calls[4]["context"][:5])
    assert all(call["queries"] == calls[0]["queries"] for call in calls)
    fold = result["contexts"]["20"][0]
    assert fold["n_context"] == 8
    expected_ll = -np.log([.25, 1e-12, .75, 1e-12]).mean()
    assert fold["scores"]["control"]["log_loss"] == pytest.approx(expected_ll)
    assert fold["scores"]["control"] == fold["scores"]["penalized"]


def payload():
    def record(a, b):
        return {"n_context": 128, "n_test": 100, "scores": {
            "control": dict(zip(evaluation.METRICS, a)),
            "penalized": dict(zip(evaluation.METRICS, b))}}
    return {"settings": {"context_sizes": [128]}, "models": [
        {"path": "a.ckpt", "step": 100, "lambda_fg": 0},
        {"path": "b.ckpt", "step": 100, "lambda_fg": .5}],
        "tasks": {
            "1": {"name": "secret<dataset>", "status": "ok", "contexts": {"128": [
                record([.4, 2., 1.], [.5, 1., .5]), record([.6, 2., 1.], [.7, 1., .5])]}},
            "2": {"name": "second", "status": "ok", "contexts": {"128": [
                record([.8, 1., .5], [.8, 1., .5])]}},
            "3": {"name": "failed", "status": "failed", "reason": "test error"}}}


def test_equal_dataset_weights_gains_ranks_and_optional_details():
    data = payload()
    tables = evaluation.comparison_tables(data)
    accuracy = tables[0][2][0]
    assert accuracy[1:] == ["65.0000", "70.0000", "+5.0000 pp", "+7.69%"]
    outcomes = tables[1][2][0]
    assert outcomes[1] == "1/1/0"
    assert outcomes[-2:] == ["1.750", "1.250"]
    text = evaluation.render_report(data)
    assert "secret<dataset>" not in text
    assert "secret<dataset>" in evaluation.render_report(data, per_dataset=True)
    assert "test error" in evaluation.render_report(data, per_dataset=True)
    assert "<table>" in evaluation.render_report(data, html=True)
    assert "secret&lt;dataset&gt;" in evaluation.render_report(data, per_dataset=True, html=True)


def test_training_context_defaults_and_explicit_override():
    metadata = [{"context_rows": [128, 512, 2048]}] * 2
    assert evaluation.resolve_contexts(metadata, None) == [128, 512, 2048]
    assert evaluation.resolve_contexts([{}, {}], [64, 32, 64]) == [32, 64]
    with pytest.raises(ValueError, match="Missing or different"):
        evaluation.resolve_contexts([metadata[0], {}], None)
    with pytest.raises(ValueError, match="positive"):
        evaluation.resolve_contexts(metadata, [-1])


def test_summary_cli_ascii_and_html_without_evaluation(tmp_path, capsys):
    path = tmp_path / "scores.json"
    evaluation.save_json(path, payload())
    assert evaluation.main(["--summary", str(path)]) == 0
    assert "| Mean gain" in capsys.readouterr().out
    html = tmp_path / "report.html"
    assert evaluation.main(["--summary", str(path), "--html", str(html), "--per-dataset"]) == 0
    assert "secret&lt;dataset&gt;" in html.read_text()
    assert "| Mean gain" not in capsys.readouterr().out
