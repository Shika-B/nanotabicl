"""Standalone TabICLv2 CPU evaluation; copy this file and a comparison JSON.

Install tabicl==2.2.0 and openml in a separate Python >=3.10 environment.
Run: python eval_tabicl_cpu.py openml_comparison.json --output tabicl_openml.json
Also accepts tabarena_comparison.json. No nanotabicl imports or checkpoints.
Replays successful tasks and saved fold settings from the reference evaluation.
"""
import argparse
from html import escape
import json
from pathlib import Path

import numpy as np

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def prepare_features(frame, categorical, context, query):
    """Match eval_openml's context-only encoding and missing-column removal."""
    import pandas as pd

    left, right = [], []
    for column, is_category in zip(frame.columns, categorical):
        a, b = frame[column].iloc[context], frame[column].iloc[query]
        if is_category:
            mapping = {value: i for i, value in enumerate(a.dropna().unique())}
            a, b = a.astype(object).map(mapping), b.astype(object).map(mapping)
            a, b = a.fillna(-1), b.fillna(-1)
        else:
            a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        left.append(a.to_numpy(dtype=np.float32))
        right.append(b.to_numpy(dtype=np.float32))
    train, test = np.column_stack(left), np.column_stack(right)
    train[~np.isfinite(train)], test[~np.isfinite(test)] = np.nan, np.nan
    keep = ~np.isnan(train).all(axis=0)
    if not keep.any():
        raise ValueError("All features are missing in context")
    return train[:, keep], test[:, keep]


def evaluate_task(task, reference, settings, clf):
    dataset = task.get_dataset()
    frame, labels, categorical, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    if labels.isna().any():
        raise ValueError("Missing target labels")
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    if len(classes) != reference["n_classes"]:
        raise ValueError("Class count differs from reference")
    records = []
    for saved in reference["folds"]:
        fold = saved["fold"]
        train_ids, test_ids = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], task.task_id, fold]))
        context = rng.permutation(train_ids)[:settings["context_size"]]
        query = rng.permutation(test_ids)
        if settings["max_test_rows"]:
            query = query[:settings["max_test_rows"]]
        if len(context) != saved["n_context"] or len(query) != saved["n_test"]:
            raise ValueError("Sample counts differ from reference")
        x_train, x_test = prepare_features(frame, categorical, context, query)
        clf.fit(x_train, y[context])
        proba = np.zeros((len(query), len(classes)), dtype=np.float32)
        batch = settings["query_batch_size"]
        for start in range(0, len(query), batch):
            proba[start:start + batch, clf.classes_.astype(int)] = clf.predict_proba(x_test[start:start + batch])
        truth = y[query]
        scores = {
            "accuracy": float(np.mean(proba.argmax(-1) == truth)),
            "log_loss": float(-np.log(np.maximum(proba[np.arange(len(truth)), truth], 1e-12)).mean()),
            "brier": float(((proba - np.eye(len(classes))[truth]) ** 2).sum(-1).mean()),
        }
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("Nonfinite scores")
        records.append({"fold": fold, "n_context": len(context), "n_test": len(query),
                        "scores": scores})
    return {"name": dataset.name, "status": "ok", "n_classes": len(classes), "folds": records,
            "scores": {metric: float(np.mean([r["scores"][metric] for r in records]))
                       for metric in ("log_loss", "accuracy", "brier")}}


def report_html(payload):
    rows = []
    for task_id, row in payload["tasks"].items():
        name = escape(row.get("name", task_id))
        if row["status"] == "ok":
            cells = "".join(f"<td>{row['scores'][m]:.6f}</td>" for m in ("log_loss", "accuracy", "brier"))
        else:
            cells = f'<td colspan="3">{escape(row["reason"])}</td>'
        rows.append(f"<tr><td>{escape(task_id)}</td><td>{name}</td>{cells}</tr>")
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>TabICLv2 CPU</title>
<style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}
th,td{border:1px solid #ddd;padding:10px;text-align:right}</style>
<h1>TabICLv2 CPU evaluation</h1><p>Each score averages the same folds as the reference.
Log loss is in nats; accuracy is a fraction; Brier is summed over classes.</p>
<table><tr><th>Task</th><th>Dataset</th><th>Log loss ↓</th><th>Accuracy ↑</th><th>Brier ↓</th></tr>
''' + "".join(rows) + "</table></html>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="Existing OpenML or TabArena comparison JSON")
    parser.add_argument("--output", default="tabicl_cpu.json", help="HTML saved alongside this JSON")
    parser.add_argument("--model-path", help="Optional local official v2 checkpoint")
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text())
    settings = reference["settings"]
    for key in ("seed", "context_size", "max_test_rows", "query_batch_size", "n_estimators"):
        if key not in settings:
            parser.error(f"Reference is missing evaluation setting: {key}")
    import openml
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(checkpoint_version=CHECKPOINT, model_path=args.model_path,
                          device="cpu", n_estimators=settings["n_estimators"],
                          random_state=settings["seed"], softmax_temperature=1.0,
                          use_amp=False, use_fa3=False, verbose=False)
    payload = {"reference": str(args.reference), "settings": settings,
               "model": {"name": "TabICLv2", "package_version": "2.2.0", "checkpoint": CHECKPOINT,
                         "device": "cpu", "temperature": 1.0, "use_amp": False}, "tasks": {}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = [(k, v) for k, v in reference["tasks"].items() if v["status"] == "ok"]
    for index, (task_id, saved) in enumerate(selected, 1):
        try:
            result = evaluate_task(openml.tasks.get_task(int(task_id)), saved, settings, clf)
        except Exception as error:
            result = {"name": saved.get("name", task_id), "status": "failed",
                      "reason": f"{type(error).__name__}: {error}"}
        payload["tasks"][task_id] = result
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        output.with_suffix(".html").write_text(report_html(payload), encoding="utf-8")
        print(f"[{index}/{len(selected)}] Task {task_id}: {result['status']} {result.get('reason', '')}", flush=True)
    if not selected:
        parser.error("Reference contains no successful tasks")
    print(f"Table: {output.with_suffix('.html').resolve()}")


if __name__ == "__main__":
    main()
