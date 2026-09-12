import pytest

from nanotabicl.eval import comparison_table, main


def test_paired_differences_and_order():
    keys = ["car/context_64/" + suffix for suffix in
            ("a/log_loss", "b/log_loss", "joint_ab/log_loss", "joint_ba/log_loss", "factorization_gap")]
    payload = {"results": {
        "control": {"0": dict.fromkeys(keys, 1.0), "1": dict.fromkeys(keys, 3.0)},
        "penalized": {"0": dict.fromkeys(keys, 0.5), "1": dict.fromkeys(keys, 2.0)},
    }}
    text = comparison_table(payload)
    assert "-0.7500 +/- 0.3536" in text
    assert "Control:   control" in text and "car" in text
    assert "+0.7500 +/- 0.3536" in comparison_table(payload, "penalized", "control")
    del payload["results"]["penalized"]["1"]
    with pytest.raises(ValueError, match="split seeds"):
        comparison_table(payload)


def test_single_seed_and_missing_metrics():
    payload = {"results": {"control": {"0": {"iris/log_loss": 1.0}},
                           "penalized": {"0": {"iris/log_loss": 0.8}}}}
    assert "-0.2000" in comparison_table(payload)
    payload["results"]["penalized"]["0"] = {}
    with pytest.raises(ValueError, match="Metric keys"):
        comparison_table(payload)


def test_single_checkpoint_and_regression():
    assert "0.8000" in comparison_table({"results": {"model": {"0": {"iris/log_loss": 0.8}}}})
    text = comparison_table({"results": {"model": {"0": {"diabetes": 0.4}}}})
    assert "R2" in text and "0.4000" in text


def test_eval_cli_prints_only_summary(tmp_path, monkeypatch, capsys):
    import json
    import nanotabicl.eval as evaluation
    from types import SimpleNamespace

    monkeypatch.setattr(evaluation, "load_model", lambda path: (path, SimpleNamespace(data=SimpleNamespace(task="classification"))))
    monkeypatch.setattr(evaluation, "evaluate", lambda model, *args, **kwargs: {"iris/log_loss": 1.0 if model == "control" else 0.8})
    path = tmp_path / "results.json"
    main(["control", "penalized", "--output", str(path)])
    output = capsys.readouterr().out
    assert "-0.2000" in output and "iris/log_loss:" not in output
    assert json.loads(path.read_text())["results"]["control"]["0"]["iris/log_loss"] == 1.0
    main(["--summary", str(path)])
    assert capsys.readouterr().out == output
