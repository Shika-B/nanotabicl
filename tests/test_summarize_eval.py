import pytest

from nanotabicl.summarize_eval import comparison_table


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
