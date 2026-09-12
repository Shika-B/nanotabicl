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
