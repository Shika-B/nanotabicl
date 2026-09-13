"""Evaluate penalty regimes on OpenML-CC18 with small contexts."""
import argparse
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

from .eval import aligned_probabilities, probability_scores
from .interface import NanoTabICLClassifier, load_model

LAMBDAS = ("0", "0.03", "0.1", "0.3")


def prepare_features(frame, categorical, context, query):
    """Fit category vocabularies on context only; preserve numerical missing values."""
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
    # A column completely missing in context carries no learnable information.
    keep = ~np.isnan(train).all(axis=0)
    if not keep.any():
        raise ValueError("All features are missing in context")
    return train[:, keep], test[:, keep]


def evaluate_task(task, models, args):
    task_type = getattr(task, "task_type_id", 1)
    if getattr(task_type, "value", task_type) != 1:  # OpenML uses an Enum in newer versions
        return {"status": "skipped", "reason": "Not a classification task"}
    dataset = task.get_dataset()
    max_rows = getattr(args, "max_dataset_rows", 0)
    max_features = getattr(args, "max_features", 0)
    # Metadata lets us avoid downloading oversized tables where available.
    qualities = getattr(dataset, "qualities", {}) or {}
    if max_rows and qualities.get("NumberOfInstances", 0) > max_rows:
        return {"name": dataset.name, "status": "skipped", "reason": f"Dataset exceeds {max_rows} rows"}
    frame, labels, categorical, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    if max_rows and len(frame) > max_rows:
        return {"name": dataset.name, "status": "skipped", "reason": f"{len(frame)} rows exceeds limit {max_rows}"}
    if max_features and frame.shape[1] > max_features:
        return {"name": dataset.name, "status": "skipped", "reason": f"{frame.shape[1]} features exceeds limit {max_features}"}
    if labels.isna().any():
        raise ValueError("Missing target labels")
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    capacity = min(model.out_mlp[-1].out_features for model in models)
    if len(classes) > capacity:
        return {"name": dataset.name, "status": "skipped", "reason": f"{len(classes)} classes exceeds capacity {capacity}"}
    repeats, n_folds, samples = task.get_split_dimensions()
    folds = list(range(n_folds)) if args.folds is None else args.folds
    if any(fold < 0 or fold >= n_folds for fold in folds):
        raise ValueError(f"Requested fold outside 0..{n_folds - 1}")
    records = []
    for fold in folds:
        train_ids, test_ids = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, task.task_id, fold]))
        context = rng.permutation(train_ids)[:args.context_size]
        query = rng.permutation(test_ids)
        if args.max_test_rows:
            query = query[:args.max_test_rows]
        if len(context) < 2 or len(query) == 0:
            raise ValueError("Insufficient context or query rows")
        x_train, x_test = prepare_features(frame, categorical, context, query)
        scores = {}
        for regime, model in zip(getattr(args, "lambdas", LAMBDAS), models):
            clf = NanoTabICLClassifier(model=model, device=args.device, n_estimators=args.n_estimators,
                                      random_state=args.seed).fit(x_train, y[context])
            proba = np.concatenate([aligned_probabilities(clf, x_test[start:start + args.query_batch_size], len(classes))
                                    for start in range(0, len(query), args.query_batch_size)])
            scores[regime] = probability_scores(proba, y[query])
            if not all(np.isfinite(value) for value in scores[regime].values()):
                raise ValueError(f"Nonfinite scores for lambda={regime}")
        records.append({"fold": fold, "n_context": len(context), "n_test": len(query),
                        "unseen_label_rate": float(np.mean(~np.isin(y[query], y[context]))), "scores": scores})
    return {"name": dataset.name, "status": "ok", "n_classes": len(classes), "folds": records,
            "scores": {regime: {metric: float(np.mean([r["scores"][regime][metric] for r in records]))
                                for metric in ("log_loss", "accuracy", "brier")} for regime in records[0]["scores"]}}


def report_html(payload, metric="log_loss"):
    """Dataset-weighted summaries on the common successfully evaluated task set."""
    successful = [(key, row) for key, row in payload["tasks"].items() if row["status"] == "ok"]
    regimes = payload.get("model_keys", list(LAMBDAS))
    if any(set(row["scores"]) != set(regimes) for _, row in successful):
        raise ValueError("All successful tasks must contain the same models for comparable summaries")
    higher = metric == "accuracy"
    body, summaries = [], []
    if successful:
        values = np.array([[row["scores"][regime][metric] for regime in regimes] for _, row in successful])
        ranks = rankdata(-values if higher else values, axis=1, method="average")
        for (task_id, row), scores, rank in zip(successful, values, ranks):
            cells = "".join(f'<td class="{"best" if r == rank.min() else ""}">{score:.4f}</td>'
                            for score, r in zip(scores, rank))
            body.append(f'<tr><th>{escape(row["name"])} <small>task {escape(task_id)}</small></th>{cells}</tr>')
        change = (values - values[:, :1]) * (1 if higher else -1)
        valid = values[:, 0] > 0
        relative = 100 * change[valid] / values[valid, :1]
        stats = [("Average score", values.mean(axis=0)), ("Average rank (lower is better)", ranks.mean(axis=0)),
                 ("Average improvement vs λ=0 (score units)", change.mean(axis=0)),
                 ("Average relative improvement (%)", relative.mean(axis=0) if valid.any() else [np.nan] * len(regimes)),
                 ("Median relative improvement (%)", np.median(relative, axis=0) if valid.any() else [np.nan] * len(regimes)),
                 ("Datasets strictly improved vs λ=0 (%)", 100 * (change > 0).mean(axis=0))]
        for label, scores in stats:
            summaries.append(f"<tr><th>{label}</th>" + "".join(
                f"<td>{score:.4f}</td>" if np.isfinite(score) else "<td>N/A</td>" for score in scores) + "</tr>")
    excluded = [f'<li>{escape(row.get("name", key))} (task {escape(key)}): {escape(row["status"])} — '
                f'{escape(row["reason"])}</li>' for key, row in payload["tasks"].items() if row["status"] != "ok"]
    settings = escape(json.dumps(payload["settings"], indent=2))
    benchmark = escape(payload.get("benchmark", "OpenML-CC18"))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenML model comparison</title>
<style>body{{font:15px system-ui;background:#f4f6fa;color:#182230;margin:32px}}main{{max-width:1200px;margin:auto}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{border:1px solid #ddd;padding:12px;text-align:right}}
th:first-child{{text-align:left}}thead,tfoot{{background:#e9eef5}}td.best{{background:#dcfce7;font-weight:bold}}
small{{font-weight:normal;color:#667085}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><main>
<h1>{benchmark} · {escape(metric)}</h1><p>{len(successful)} tasks evaluated; {len(excluded)} skipped or failed.
{'Higher' if higher else 'Lower'} scores are better. Green marks the best model in each row (including ties).</p>
<p>Official repeat-0 folds, subsampled training contexts: this is a small-context adaptation, not the full benchmark protocol.
Cells average folds. Summary rows weight each task equally; tied ranks are averaged.
Positive improvement means better. Relative summaries exclude zero-baseline tasks. Split variation is not training-seed uncertainty.</p>
<table><thead><tr><th>Dataset</th>{''.join(f'<th>{escape(v if v == "TabICLv2" else "λ = " + v)}</th>' for v in regimes)}</tr></thead>
<tbody>{''.join(body)}</tbody><tfoot>{''.join(summaries)}</tfoot></table>
<h2>Skipped / failed tasks</h2><ul>{''.join(excluded)}</ul>
<details><summary>Protocol and checkpoints</summary><pre>{settings}</pre></details></main></html>"""


def main(argv=None, *, benchmark="OpenML-CC18", suite_id=99, output_prefix="openml", max_dataset_rows=0, max_features=0,
         report_builder=report_html):
    parser = argparse.ArgumentParser(description=f"Evaluate penalty regimes on {benchmark} with a small-context protocol.")
    parser.add_argument("--lambdas", nargs="+", default=list(LAMBDAS),
                        help="Penalty values in checkpoint order; first must be 0 (baseline)")
    parser.add_argument("--checkpoints", nargs="+",
                        help="Paths in --lambdas order; default: runs/cosine3000_fg{lambda}_seed0/latest.pt")
    parser.add_argument("--suite", type=int, default=suite_id, help="OpenML suite ID")
    parser.add_argument("--max-dataset-rows", type=int, default=max_dataset_rows, help="Exclude larger datasets; 0 disables the limit")
    parser.add_argument("--max-features", type=int, default=max_features, help="Exclude wider datasets; 0 disables the limit")
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--folds", type=int, nargs="+", help="Default: every official fold of repeat 0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=1024, help="Per fold; 0 uses the entire test fold")
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=f"runs/{output_prefix}_comparison.json")
    parser.add_argument("--html", default=f"runs/{output_prefix}_comparison.html")
    parser.add_argument("--metric", choices=["log_loss", "accuracy", "brier"], default="log_loss")
    parser.add_argument("--summary", help="Rebuild the HTML from existing JSON without evaluating")
    args = parser.parse_args(argv)
    if args.summary:
        payload = json.loads(Path(args.summary).read_text())
    else:
        try:
            values = [float(v) for v in args.lambdas]
        except ValueError:
            parser.error("--lambdas must be numeric")
        if values[0] != 0 or any(not np.isfinite(v) or v < 0 for v in values) or len(set(values)) != len(values):
            parser.error("--lambdas must be distinct, finite, nonnegative, and start with 0")
        if args.checkpoints is None:
            args.checkpoints = [f"runs/cosine3000_fg{v}_seed0/latest.pt" for v in args.lambdas]
        if len(args.checkpoints) != len(args.lambdas):
            parser.error("Provide one checkpoint per lambda")
        if min(args.context_size, args.query_batch_size, args.n_estimators) < 1 or args.context_size < 2 or args.max_test_rows < 0:
            parser.error("Require context-size >= 2, positive batch/ensemble sizes, and max-test-rows >= 0")
        if args.max_dataset_rows < 0 or args.max_features < 0:
            parser.error("Dataset size limits must be nonnegative")
        try:
            import openml
        except ImportError:
            parser.error("Install the optional dependency: python -m pip install openml")
        models = []
        for regime, path in zip(args.lambdas, args.checkpoints):
            model, cfg = load_model(path, args.device)
            if cfg.data.task != "classification" or not np.isclose(cfg.optim.lambda_fg, float(regime)):
                parser.error(f"{path} is not a classification checkpoint with lambda_fg={regime}")
            models.append(model)

        suite = openml.study.get_suite(args.suite)
        payload = {"benchmark": benchmark, "settings": vars(args), "suite_tasks": list(suite.tasks), "tasks": {},
                   "model_keys": args.lambdas}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        for task_id in suite.tasks:
            try:
                task = openml.tasks.get_task(task_id)
                result = evaluate_task(task, models, args)
            except Exception as error:
                result = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
            payload["tasks"][str(task_id)] = result
            output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            reason = f" — {result['reason']}" if "reason" in result else ""
            print(f"Task {task_id}: {result['status']}{reason}", file=sys.stderr, flush=True)
    path = Path(args.html)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_builder(payload, args.metric), encoding="utf-8")
    print(f"Table saved to {path.resolve()}")


if __name__ == "__main__":
    main()
