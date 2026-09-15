"""Compare two fine-tuned TabICLv2 checkpoints on the saved TabArena protocol.

Run from the full TabICL environment: python eval_finetuned_tabarena.py
Contexts default to checkpoint training context_rows; --context-sizes overrides.
Dataset selection, folds, query sampling and preprocessing match the reference.
"""
import argparse
from contextlib import redirect_stdout
import gc
from html import escape
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from eval_tabicl_cpu import CHECKPOINT, prepare_features

ROOT = Path(__file__).resolve().parent
ARMS = ("control", "penalized")
METRICS = ("accuracy", "log_loss", "brier")


def evaluate_task(task, reference, settings, classifiers):
    """Use identical queries and nested training prefixes for both models."""
    dataset = task.get_dataset()
    frame, labels, categorical, _ = dataset.get_data(
        target=task.target_name, dataset_format="dataframe")
    if labels.isna().any():
        raise ValueError("Missing target labels")
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    if len(classes) != reference["n_classes"]:
        raise ValueError("Class count differs from reference")
    contexts = {str(size): [] for size in settings["context_sizes"]}
    for saved in reference["folds"]:
        fold = saved["fold"]
        train_ids, test_ids = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        rng = np.random.default_rng(np.random.SeedSequence([settings["seed"], task.task_id, fold]))
        train = rng.permutation(train_ids)
        query = rng.permutation(test_ids)
        if settings["max_test_rows"]:
            query = query[:settings["max_test_rows"]]
        if min(len(train), settings["context_size"]) != saved["n_context"] or len(query) != saved["n_test"]:
            raise ValueError("Sample counts differ from reference")
        truth = y[query]
        for size in settings["context_sizes"]:
            context = train[:size]
            x_train, x_test = prepare_features(frame, categorical, context, query)
            scores = {}
            for arm, make_classifier in zip(ARMS, classifiers):
                # Release each estimator before constructing the next: one GPU model at a time.
                clf = make_classifier()
                try:
                    clf.fit(x_train.copy(), y[context].copy())
                    proba = np.zeros((len(query), len(classes)), dtype=np.float32)
                    batch = settings["query_batch_size"]
                    for start in range(0, len(query), batch):
                        proba[start:start + batch, clf.classes_.astype(int)] = clf.predict_proba(
                            x_test[start:start + batch].copy())
                    scores[arm] = {
                        "accuracy": float(np.mean(proba.argmax(-1) == truth)),
                        "log_loss": float(-np.log(np.maximum(proba[np.arange(len(truth)), truth], 1e-12)).mean()),
                        "brier": float(((proba - np.eye(len(classes))[truth]) ** 2).sum(-1).mean()),
                    }
                    if not all(np.isfinite(v) for v in scores[arm].values()):
                        raise ValueError("Nonfinite scores")
                finally:
                    del clf
                    gc.collect()
            contexts[str(size)].append({"fold": fold, "n_context": len(context),
                                       "n_test": len(query), "scores": scores})
    return {"name": dataset.name, "status": "ok", "contexts": contexts}


def dataset_scores(row, sizes):
    """Equal fold weights within contexts, then equal context weights."""
    return {arm: {metric: float(np.mean([
        np.mean([fold["scores"][arm][metric] for fold in row["contexts"][str(size)]])
        for size in sizes])) for metric in METRICS} for arm in ARMS}


def comparison_tables(payload, per_dataset=False):
    tasks = [row for row in payload["tasks"].values() if row["status"] == "ok"]
    if not tasks:
        return []
    sizes = payload["settings"]["context_sizes"]
    groups = [(f"Context cap {size}", [size]) for size in sizes]
    if len(sizes) > 1:
        groups.append(("Overall (equal context weights)", sizes))
    tables = []
    for title, group in groups:
        paired = [dataset_scores(row, group) for row in tasks]
        rows = []
        outcomes = []
        for metric in METRICS:
            a = np.array([r["control"][metric] for r in paired])
            b = np.array([r["penalized"][metric] for r in paired])
            gain = (b - a) if metric == "accuracy" else (a - b)
            scale = 100 if metric == "accuracy" else 1
            valid = a > 0
            relative = 100 * gain[valid] / a[valid]
            ties = np.isclose(a, b, atol=1e-8, rtol=0)
            wins, losses = int(((gain > 0) & ~ties).sum()), int(((gain < 0) & ~ties).sum())
            rows.append(["Accuracy (%)" if metric == "accuracy" else metric,
                         f"{a.mean() * scale:.4f}", f"{b.mean() * scale:.4f}",
                         f"{gain.mean() * scale:+.4f}" + (" pp" if metric == "accuracy" else ""),
                         f"{gain.mean() / a.mean() * 100:+.2f}%" if a.mean() > 0 else "n/a"])
            outcomes.append([metric, f"{wins}/{int(ties.sum())}/{losses}",
                             f"{relative.mean():+.2f}%" if len(relative) else "n/a",
                             f"{np.median(relative):+.2f}%" if len(relative) else "n/a",
                             str(int(valid.sum())),
                             f"{1 + (wins + .5 * ties.sum()) / len(a):.3f}",
                             f"{1 + (losses + .5 * ties.sum()) / len(a):.3f}"])
        a = np.array([r["control"]["accuracy"] for r in paired])
        b = np.array([r["penalized"]["accuracy"] for r in paired])
        error_reduction = 100 * (b.mean() - a.mean()) / (1 - a.mean()) if a.mean() < 1 else None
        actual = [f["n_context"] for row in tasks for size in group
                  for f in row["contexts"][str(size)]]
        title += f" | {len(tasks)} datasets | actual context {min(actual)}..{max(actual)}"
        tables.append((title, ["Metric", "Control", "Penalized", "Mean gain", "Gain / baseline mean"], rows))
        tables.append(("Paired dataset comparisons", ["Metric", "W/T/L", "Mean relative gain",
                       "Median relative gain", "Valid relative N", "Control rank", "Penalized rank"], outcomes))
        tables.append(("Classification error reduction", ["Statistic", "Value"], [
            ["Reduction in mean error rate", f"{error_reduction:+.2f}%" if error_reduction is not None else "n/a"]]))
        if per_dataset:
            rows = []
            for task, scores in zip(tasks, paired):
                a, b = scores["control"], scores["penalized"]
                rows.append([task["name"], f"{100*a['accuracy']:.3f}", f"{100*b['accuracy']:.3f}",
                             f"{100*(b['accuracy']-a['accuracy']):+.3f}",
                             f"{a['log_loss']:.5f}", f"{b['log_loss']:.5f}",
                             f"{a['log_loss']-b['log_loss']:+.5f}"])
            tables.append(("Per-dataset results", ["Dataset", "Control acc %", "Penalized acc %",
                           "Gain pp", "Control LL", "Penalized LL", "LL gain"], rows))
    return tables


def render_report(payload, per_dataset=False, html=False):
    ok = sum(row["status"] == "ok" for row in payload["tasks"].values())
    notes = ["Fine-tuned TabICLv2: control vs penalized",
             *[f"{arm}: {m['path']} (step {m.get('step')}, lambda={m.get('lambda_fg')})"
               for arm, m in zip(ARMS, payload["models"])],
             f"Completed paired datasets: {ok}/{len(payload['tasks'])}.",
             "Positive gain favors penalized. W/T/L = penalized wins/ties/losses. Lower rank is better.",
             "Folds, datasets and contexts have equal weights at each averaging level; contexts are capped by available rows.",
             "Relative gains omit zero baselines. These results do not estimate training-seed uncertainty."]
    tables = comparison_tables(payload, per_dataset)
    if per_dataset:
        failed = [[key, r.get("name", key), r["reason"]] for key, r in payload["tasks"].items()
                  if r["status"] != "ok"]
        if failed:
            tables.append(("Failed tasks (excluded from all summaries)", ["Task", "Dataset", "Reason"], failed))
    if html:
        parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Fine-tuned TabArena</title>',
                 '<style>body{font:14px system-ui;margin:32px}table{border-collapse:collapse;margin-bottom:24px}'
                 'th,td{border:1px solid #ccc;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}</style>',
                 *[f"<p>{escape(n)}</p>" for n in notes]]
        for title, headers, rows in tables:
            parts.append(f"<h2>{escape(title)}</h2><table><thead><tr>" +
                         "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead><tbody>")
            parts.extend("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
            parts.append("</tbody></table>")
        return "\n".join(parts) + "</html>"
    parts = notes[:]
    for title, headers, rows in tables:
        widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
        border = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        line = lambda row: "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
        parts.extend(["", title, border, line(headers), border, *map(line, rows), border])
    return "\n".join(parts)


def resolve_contexts(metadata, override):
    contexts = [m.get("context_rows") for m in metadata]
    if override:
        result = override
    elif contexts[0] and contexts[0] == contexts[1]:
        result = contexts[0]
    else:
        raise ValueError("Missing or different training contexts: supply --context-sizes explicitly")
    if not result or any(size <= 0 for size in result):
        raise ValueError("Context sizes must be positive")
    return sorted(set(result))


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", type=Path, help="Control checkpoint then penalized checkpoint")
    parser.add_argument("--reference", type=Path, default=ROOT / "tabarena_comparison.json")
    parser.add_argument("--context-sizes", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/finetuned_tabarena.json")
    parser.add_argument("--summary", type=Path, help="Render existing result JSON without evaluating")
    parser.add_argument("--per-dataset", action="store_true")
    parser.add_argument("--html", type=Path, metavar="PATH", help="Write HTML instead of printing ASCII tables")
    parser.add_argument("--verbose", action="store_true", help="Task progress on stderr")
    parser.add_argument("--allow-step-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if args.summary:
        payload = json.loads(args.summary.read_text())
    else:
        paths = args.checkpoints or [ROOT / f"runs/fg_full2000/lambda_{penalty}/latest.ckpt" for penalty in ("0", "0.5")]
        if len(paths) != 2:
            parser.error("Provide exactly two checkpoints: control then penalized")
        for path in paths:
            if not path.is_file():
                parser.error(f"Checkpoint not found: {path}")
        reference = json.loads(args.reference.read_text())
        import torch
        import openml
        from tabicl import TabICLClassifier

        with tempfile.TemporaryDirectory(prefix="tabarena-checkpoints-") as temporary:
            snapshots, metadata = [], []
            for i, path in enumerate(paths):
                snapshot = Path(temporary) / f"model_{i}.ckpt"
                # The trainer atomically replaces latest.ckpt, so an open file remains stable.
                with path.open("rb") as source, snapshot.open("wb") as target:
                    shutil.copyfileobj(source, target)
                saved = torch.load(snapshot, map_location="cpu", weights_only=True)
                experiment = saved.get("experiment", {})
                metadata.append({"path": str(path.resolve()), "step": saved.get("step"),
                                 "lambda_fg": experiment.get("lambda_fg"),
                                 "context_rows": experiment.get("context_rows")})
                torch.save({"config": saved["config"], "state_dict": saved["state_dict"]}, snapshot)
                del saved
                snapshots.append(snapshot)
            if metadata[0]["step"] != metadata[1]["step"] and not args.allow_step_mismatch:
                parser.error("Checkpoint steps differ; choose matching step_*.ckpt files or --allow-step-mismatch")
            try:
                sizes = resolve_contexts(metadata, args.context_sizes)
            except ValueError as error:
                parser.error(str(error))
            settings = {**reference["settings"], "context_sizes": sizes, "device": args.device}
            # Keep inference settings identical to the existing full-model reference evaluator.
            def factory(path):
                return lambda: TabICLClassifier(model_path=str(path), allow_auto_download=False,
                    checkpoint_version=CHECKPOINT, device=args.device, n_estimators=settings["n_estimators"],
                    random_state=settings["seed"], softmax_temperature=1., use_amp=False, use_fa3=False, verbose=False)
            classifiers = [factory(path) for path in snapshots]
            payload = {"reference": str(args.reference.resolve()), "settings": settings,
                       "models": metadata, "tasks": {}}
            selected = {key: row for key, row in reference["tasks"].items() if row["status"] == "ok"}
            if not selected:
                parser.error("Reference has no successful tasks")
            for index, (task_id, row) in enumerate(selected.items(), 1):
                try:
                    with redirect_stdout(sys.stderr):
                        result = evaluate_task(openml.tasks.get_task(int(task_id)), row, settings, classifiers)
                except Exception as error:
                    result = {"name": row.get("name", task_id), "status": "failed",
                              "reason": f"{type(error).__name__}: {error}"}
                payload["tasks"][task_id] = result
                save_json(args.output, payload)
                if args.verbose:
                    print(f"[{index}/{len(selected)}] Task {task_id}: {result['status']}"
                          + (f" ({result['reason']})" if result["status"] == "failed" else ""), file=sys.stderr)
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_report(payload, args.per_dataset, html=True))
        print(f"HTML report: {args.html}")
    else:
        print(render_report(payload, args.per_dataset))
    if not any(row["status"] == "ok" for row in payload["tasks"].values()):
        print("No paired results. Use --summary <result.json> --per-dataset to inspect failures.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
