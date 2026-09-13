from types import SimpleNamespace
from enum import Enum

import pytest

from nanotabicl.eval_openml import evaluate_task, report_html
from nanotabicl import eval_tabarena


def test_tabarena_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(eval_tabarena, "run_benchmark", lambda *args, **kwargs: calls.append((args, kwargs)))
    eval_tabarena.main(["--device", "cpu"])
    args, settings = calls[0]
    assert args == (["--device", "cpu"],)
    assert settings["suite_id"] == 457
    assert settings["max_dataset_rows"] == 10000 and settings["max_features"] == 100


class TaskType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


@pytest.mark.parametrize("task_type", [2, TaskType.REGRESSION])
def test_regression_skipped_before_download(task_type):
    result = evaluate_task(SimpleNamespace(task_type_id=task_type), [], SimpleNamespace())
    assert result["status"] == "skipped" and "classification" in result["reason"]


@pytest.mark.parametrize("task_type", [1, TaskType.CLASSIFICATION])
def test_size_filters(task_type):
    dataset = SimpleNamespace(name="large", qualities={"NumberOfInstances": 20000})
    task = SimpleNamespace(task_type_id=task_type, target_name="y", get_dataset=lambda: dataset)
    args = SimpleNamespace(max_dataset_rows=10000, max_features=100)
    result = evaluate_task(task, [], args)
    assert result["status"] == "skipped" and "rows" in result["reason"]
    dataset.qualities = {}
    dataset.get_data = lambda **kwargs: (SimpleNamespace(shape=(10, 101)), None, None, None)
    args.max_dataset_rows = 0
    assert "features" in evaluate_task(task, [], args)["reason"]


def test_report_identifies_suite():
    html = report_html({"benchmark": "TabArena", "settings": {}, "tasks": {}})
    assert "TabArena · log_loss" in html and "OpenML-CC18" not in html


def test_relative_tables():
    payload = {"settings": {}, "model_keys": ["0", "0.3", "2.0"], "tasks": {
        "1": {"name": "example", "status": "ok", "scores": {
            "0": {"accuracy": 0.5, "log_loss": 2},
            "0.3": {"accuracy": 0.6, "log_loss": 1},
            "2.0": {"accuracy": 0.4, "log_loss": 3}}},
        "2": {"name": "zero", "status": "ok", "scores": {
            v: {"accuracy": 0, "log_loss": 0} for v in ["0", "0.3", "2.0"]}}}}
    html = eval_tabarena.report_html(payload)
    assert html.count("<table>") == 2
    assert "+20.00%" in html and "+50.00%" in html
    assert "-20.00%" in html and "-50.00%" in html
    assert "Mean improvement (1 datasets)" in html and "<td>N/A</td>" in html
    assert "λ = 2.0" in html
