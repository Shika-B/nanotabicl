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


def test_five_model_report():
    scores = {regime: {"log_loss": 1.0} for regime in (*LAMBDAS, "TabICLv2")}
    scores["TabICLv2"]["log_loss"] = 0.5
    payload = {"settings": {}, "model_keys": [*LAMBDAS, "TabICLv2"],
               "tasks": {"1": {"name": "data", "status": "ok", "scores": scores}}}
    html = report_html(payload)
    assert "<th>TabICLv2</th>" in html and '<td class="best">0.5000</td>' in html
    assert "<td>50.0000</td>" in html


@pytest.mark.parametrize("regimes", [LAMBDAS, ("0", "0.03", "0.1", "0.3", "0.5", "1.0", "2.0")])
def test_standalone_replays_nano_evaluation(monkeypatch, regimes):
    from types import SimpleNamespace
    from nanotabicl import eval_openml as evaluation

    class Labels(np.ndarray):
        def isna(self):
            return np.zeros(len(self), dtype=bool)

    y = np.array([0, 1] * 6).view(Labels)
    x = np.arange(24).reshape(12, 2).astype(np.float32)
    dataset = SimpleNamespace(name="test", qualities={}, get_data=lambda **kwargs: (x, y, [], []))
    task = SimpleNamespace(task_type_id=1, task_id=3, target_name="y", get_dataset=lambda: dataset,
                           get_split_dimensions=lambda: (1, 1, 1),
                           get_train_test_split_indices=lambda **kwargs: (np.arange(8), np.arange(8, 12)))
    fits, queries = [], []

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, labels):
            fits.append((features.copy(), labels.copy()))
            self.classes_ = np.unique(labels)
            return self

        def predict_proba(self, features):
            queries.append(features.copy())
            return np.full((len(features), len(self.classes_)), 1 / len(self.classes_))

    monkeypatch.setattr(evaluation, "NanoTabICLClassifier", Estimator)
    monkeypatch.setattr(evaluation, "prepare_features", lambda frame, cats, train, test: (frame[train], frame[test]))
    models = [SimpleNamespace(out_mlp=[SimpleNamespace(out_features=2)])] * len(regimes)
    args = SimpleNamespace(folds=None, seed=0, context_size=8, max_test_rows=0,
                           device="cpu", n_estimators=1, query_batch_size=2, lambdas=regimes)
    result = evaluation.evaluate_task(task, models, args)
    assert set(result["scores"]) == set(regimes)
    assert len(fits) == len(regimes) and len(queries) == 2 * len(regimes)
    for features, labels in fits:
        np.testing.assert_array_equal(features, fits[0][0])
        np.testing.assert_array_equal(labels, fits[0][1])
    for i in range(0, 2 * len(regimes), 2):
        np.testing.assert_array_equal(queries[i], queries[0])
        np.testing.assert_array_equal(queries[i + 1], queries[1])

    import eval_tabicl_cpu as standalone
    monkeypatch.setattr(standalone, "prepare_features", evaluation.prepare_features)
    replay = standalone.evaluate_task(task, result, vars(args), Estimator())
    assert replay["scores"] == result["scores"]["0"]
    np.testing.assert_array_equal(fits[-1][0], fits[0][0])
    np.testing.assert_array_equal(fits[-1][1], fits[0][1])
    np.testing.assert_array_equal(queries[-2], queries[0])
    np.testing.assert_array_equal(queries[-1], queries[1])
