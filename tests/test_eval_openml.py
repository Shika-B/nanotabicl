import numpy as np
import pytest

from nanotabicl.eval_openml import LAMBDAS, prepare_features, report_html


def test_rank_and_relative_improvement():
    tasks = {}
    for name, losses in [("one", [1, 0.5, 0.5, 2]), ("two", [2, 1, 2, 3])]:
        tasks[name] = {"name": name, "status": "ok", "scores": {
            regime: {"log_loss": value} for regime, value in zip(LAMBDAS, losses)}}
    tasks["bad"] = {"status": "failed", "reason": "<network error>"}
    html = report_html({"tasks": tasks, "settings": {}})
    assert "<th>λ = 0.3</th>" in html and "2 tasks evaluated; 1 skipped or failed" in html
    # Tied ranks: lambda 0.03 ranks 1.5 then 1, for a mean rank of 1.25.
    assert "<td>1.2500</td>" in html
    assert "<td>50.0000</td>" in html
    assert "&lt;network error&gt;" in html


def test_features_context_only_and_missing():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"category": ["a", "b", "new"], "numeric": [1., np.nan, 3.],
                          "empty_context": [np.nan, np.nan, 4.]})
    train, test = prepare_features(frame, [True, False, False], np.array([0, 1]), np.array([2]))
    assert train.shape == (2, 2) and test.shape == (1, 2)
    assert test[0, 0] == -1 and test[0, 1] == 3
    assert np.isnan(train[1, 1])


def test_no_successful_tasks_still_reports_failures():
    html = report_html({"tasks": {"1": {"status": "skipped", "reason": "too many classes"}}, "settings": {}})
    assert "0 tasks evaluated" in html and "too many classes" in html
