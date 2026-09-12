"""Prediction quality and two-target coherence benchmarks; regression uses the original R^2 evaluation."""
import argparse
import csv
import io
import json
from html import escape
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from .interface import NanoTabICLClassifier, NanoTabICLRegressor, load_model

REAL_DATASETS = {
    "classification": [
        ("iris", lambda: datasets.load_iris(return_X_y=True)),
        ("wine", lambda: datasets.load_wine(return_X_y=True)),
        ("breast_cancer", lambda: datasets.load_breast_cancer(return_X_y=True)),
        ("digits", lambda: datasets.load_digits(return_X_y=True)),
    ],
    "regression": [
        ("diabetes", lambda: datasets.load_diabetes(return_X_y=True)),
        ("friedman1", lambda: datasets.make_friedman1(n_samples=600, noise=1.0, random_state=0)),
        ("linear", lambda: datasets.make_regression(n_samples=600, n_features=20, noise=10.0, random_state=0)),
    ],
}


PAIR_DATASETS = ("car", "nursery", "student_mat", "student_por")


def load_pair_dataset(name: str, cache_dir: str = "runs/eval_data"):
    """Load UCI (X, A, B) without target columns in X.

    Car: acceptability / safety; Nursery: recommendation / health.
    Student: G2 >= 10 / G3 >= 10, excluding G1, G2 and G3 from features.
    Math and Portuguese are evaluated separately, never pooled (students overlap).
    UCI sources: datasets 19, 76 and 320, respectively (CC BY 4.0).
    Archives are downloaded once, cached, and read without extracting files.
    """
    if name not in PAIR_DATASETS:
        raise ValueError(f"Unknown paired dataset: {name}")
    dataset_id, slug = {"car": (19, "car+evaluation"), "nursery": (76, "nursery")}.get(
        name, (320, "student+performance"))
    path = Path(cache_dir) / f"{dataset_id}.zip"
    if not path.exists():
        url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/{slug}.zip"
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        with ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt UCI archive downloaded from {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    with ZipFile(path) as archive:
        if name.startswith("student"):
            with ZipFile(io.BytesIO(archive.read("student.zip"))) as student:
                rows = list(csv.reader(io.StringIO(student.read(name.replace("_", "-") + ".csv").decode()),
                                       delimiter=";"))
            header, values = rows[0], np.asarray(rows[1:])
            keep = [i for i, col in enumerate(header) if col not in ("G1", "G2", "G3")]
            return (values[:, keep], (values[:, header.index("G2")].astype(int) >= 10).astype(int),
                    (values[:, header.index("G3")].astype(int) >= 10).astype(int))
        rows = [row for row in csv.reader(io.StringIO(archive.read(f"{name}.data").decode())) if row]
    values = np.asarray(rows)
    return values[:, :-2], values[:, -1], values[:, -2]


def encode_features(train, test):
    """Keep numerical columns; encode categories from context only, with unknown=-1."""
    train_columns, test_columns = [], []
    for left, right in zip(train.T, test.T):
        try:
            left, right = left.astype(np.float32), right.astype(np.float32)
        except ValueError:
            mapping = {value: i for i, value in enumerate(np.unique(left))}
            left = np.asarray([mapping[value] for value in left], dtype=np.float32)
            right = np.asarray([mapping.get(value, -1) for value in right], dtype=np.float32)
        train_columns.append(left)
        test_columns.append(right)
    return np.column_stack(train_columns), np.column_stack(test_columns)


def probability_scores(proba, y):
    """Accuracy, mean negative log probability, and multiclass (summed) Brier score.

    Clip true-label probabilities at 1e-12 only for log loss; zero probabilities
    for classes absent from context remain zero for all other calculations.
    """
    return {"accuracy": float(np.mean(proba.argmax(-1) == y)),
            "log_loss": float(-np.log(np.maximum(proba[np.arange(len(y)), y], 1e-12)).mean()),
            "brier": float(((proba - np.eye(proba.shape[-1])[y]) ** 2).sum(-1).mean())}


def aligned_probabilities(estimator, x, n_classes):
    proba = np.zeros((len(x), n_classes), dtype=np.float32)
    proba[:, estimator.classes_.astype(int)] = estimator.predict_proba(x)
    return proba


def evaluate_pair(model, x, a, b, n_train=64, n_test=256, seed=0, n_estimators=1):
    """Evaluate four views and both joint orders on one shared, seeded random split.

    No class-coverage rejection or rare-class removal: this preserves the natural
    sampling distribution. Vocabulary encoding does not estimate probabilities;
    missing context classes get probability zero and their test rate is reported.
    Conditionals use observed context values and enumerate hypothetical query
    values. Their observed-label slices are used for conditional quality metrics.
    """
    x, a, b = np.asarray(x), np.asarray(a), np.asarray(b)
    if not (len(x) == len(a) == len(b)) or n_train < 2 or n_test < 1 or n_train >= len(x):
        raise ValueError("Need aligned arrays, at least two context rows and at least one query row")
    classes_a, a = np.unique(a, return_inverse=True)
    classes_b, b = np.unique(b, return_inverse=True)
    ka, kb = len(classes_a), len(classes_b)
    if max(ka, kb) > model.out_mlp[-1].out_features:
        raise ValueError("Paired dataset has more classes than the model supports")
    order = np.random.default_rng(seed).permutation(len(x))
    context, query = order[:n_train], order[n_train:n_train + n_test]
    x_train, x_test = encode_features(x[context], x[query])
    a_train, b_train, a_test, b_test = a[context], b[context], a[query], b[query]
    device = str(next(model.parameters()).device)

    def fit(features, labels):
        return NanoTabICLClassifier(model=model, device=device, n_estimators=n_estimators,
                                   random_state=seed).fit(features, labels)

    pa = aligned_probabilities(fit(x_train, a_train), x_test, ka)
    pb = aligned_probabilities(fit(x_train, b_train), x_test, kb)
    b_given_a = fit(np.column_stack([x_train, a_train]), b_train)
    a_given_b = fit(np.column_stack([x_train, b_train]), a_train)
    pba = np.stack([aligned_probabilities(b_given_a, np.column_stack([x_test, np.full(len(query), value)]), kb)
                    for value in range(ka)], axis=1)
    pab = np.stack([aligned_probabilities(a_given_b, np.column_stack([x_test, np.full(len(query), value)]), ka)
                    for value in range(kb)], axis=1)
    rows = np.arange(len(query))
    scores = {}
    for name, proba, labels in [("a", pa, a_test), ("b", pb, b_test),
                                ("a_given_b", pab[rows, b_test], a_test),
                                ("b_given_a", pba[rows, a_test], b_test)]:
        scores.update({f"{name}/{metric}": value for metric, value in probability_scores(proba, labels).items()})
    joint_ab = pa[..., None] * pba
    joint_ba = (pb[..., None] * pab).transpose(0, 2, 1)
    scores["factorization_gap"] = float(0.5 * np.abs(joint_ab - joint_ba).sum(axis=(1, 2)).mean())
    for name, joint in [("joint_ab", joint_ab), ("joint_ba", joint_ba)]:
        scores.update({f"{name}/{metric}": value for metric, value in
                       probability_scores(joint.reshape(len(query), -1), a_test * kb + b_test).items()})
    scores.update(n_train=len(context), n_test=len(query),
                  unseen_a_rate=float(np.mean(~np.isin(a_test, a_train))),
                  unseen_b_rate=float(np.mean(~np.isin(b_test, b_train))))
    return scores


def evaluate(model, task: str, max_rows: int = 1024, seed: int = 0, *, include_pairs: bool = True,
             context_sizes=(64,), cache_dir="runs/eval_data", n_estimators=1) -> dict[str, float]:
    """Keep original 50/50 benchmarks; add probability metrics and paired tasks for classification.

    Paired benchmarks use context_sizes and at most 256 query rows, capped so
    context plus queries does not exceed max_rows. Regression's datasets, splits,
    estimator and R^2 outputs are unchanged. include_pairs=False avoids downloads.
    """
    regression = task == "regression"
    estimator = (NanoTabICLRegressor if regression else NanoTabICLClassifier)(model=model)
    scores = {}
    for name, load in REAL_DATASETS[task]:
        x, y = load()
        n = min(len(x), max_rows)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=n // 2, test_size=n - n // 2, random_state=seed)
        if not regression and len(np.unique(y)) > model.out_mlp[-1].out_features:
            raise ValueError(f"{name} exceeds model class capacity; use evaluate_pair for smaller-class benchmarks")
        pred = estimator.fit(x_train, y_train).predict(x_test)
        scores[name] = r2_score(y_test, pred) if regression else accuracy_score(y_test, pred)
        if not regression:
            classes = np.unique(y)
            proba = np.zeros((len(y_test), len(classes)), dtype=np.float32)
            proba[:, np.searchsorted(classes, estimator.classes_)] = estimator.predict_proba(x_test)
            for metric, value in probability_scores(proba, np.searchsorted(classes, y_test)).items():
                scores[f"{name}/{metric}"] = value
    if not regression and include_pairs:
        for name in PAIR_DATASETS:
            x, a, b = load_pair_dataset(name, cache_dir)
            for n_train in context_sizes:
                result = evaluate_pair(model, x, a, b, n_train=n_train, n_test=min(256, max_rows - n_train),
                                       seed=seed, n_estimators=n_estimators)
                scores.update({f"{name}/context_{n_train}/{key}": value for key, value in result.items()})
    return scores


def comparison_table(payload, control=None, penalized=None, html=False):
    """Compact mean +/- sample SD of paired split-seed differences (or single-model scores)."""
    results = payload["results"]
    if control is None and penalized is None:
        if not 1 <= len(results) <= 2:
            raise ValueError("Specify --control and --penalized to select two checkpoints")
        control = next(iter(results))
        penalized = list(results)[1] if len(results) == 2 else None
    if control not in results or (penalized is not None and (penalized not in results or penalized == control)):
        raise ValueError("Select valid, distinct checkpoint keys")
    baseline = results[control]
    treatment = results[penalized] if penalized is not None else baseline
    if not baseline or baseline.keys() != treatment.keys():
        raise ValueError("Both checkpoints must have the same nonempty set of split seeds")
    seeds = sorted(baseline)
    keys = set(baseline[seeds[0]])
    for seed in seeds:
        if set(baseline[seed]) != keys or set(treatment[seed]) != keys:
            raise ValueError("Metric keys must match across checkpoints and seeds")

    def entry(metric_keys):
        values = [np.mean([treatment[seed][key] - (baseline[seed][key] if penalized else 0)
                           for key in metric_keys]) for seed in seeds]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values for {metric_keys}")
        mean = f"{np.mean(values):+.4f}" if penalized else f"{np.mean(values):.4f}"
        return f"{mean} +/- {np.std(values, ddof=1):.4f}" if len(values) > 1 else mean

    def improvement(metric_keys):
        base = np.mean([baseline[seed][key] for seed in seeds for key in metric_keys])
        treated = np.mean([treatment[seed][key] for seed in seeds for key in metric_keys])
        if not np.isfinite([base, treated]).all():
            raise ValueError(f"Nonfinite values for {metric_keys}")
        return f"{100 * (base - treated) / base:+.2f}%" if base > 0 else "N/A"

    rows = []
    prefixes = [key.removesuffix("/factorization_gap") for key in keys if key.endswith("/factorization_gap")]
    for prefix in sorted(prefixes, key=lambda p: (p.split("/")[0], int(p.rsplit("_", 1)[1]))):
        dataset, context = prefix.split("/context_")
        percentage = [improvement([f"{prefix}/joint_ab/log_loss", f"{prefix}/joint_ba/log_loss"])] if penalized else []
        rows.append([dataset, context, *percentage, entry([f"{prefix}/a/log_loss"]), entry([f"{prefix}/b/log_loss"]),
                     entry([f"{prefix}/joint_ab/log_loss", f"{prefix}/joint_ba/log_loss"]),
                     entry([f"{prefix}/factorization_gap"])])
    for key in sorted(keys):
        if key.count("/") == 1 and key.endswith("/log_loss"):
            percentage = [improvement([key])] if penalized else []
            rows.append([key.split("/")[0], "50/50", *percentage, entry([key]), "-", "-", "-"])
    regression = not rows and keys and all("/" not in key for key in keys)
    if regression:
        rows = [[key, entry([key])] for key in sorted(keys)]
        header = ["Dataset", "Delta R2" if penalized else "R2"]
    else:
        prefix = "Delta " if penalized else ""
        header = ["Dataset", "Context", *(["Loss improvement %"] if penalized else []),
                  prefix + "LL A", prefix + "LL B", prefix + "joint LL", prefix + "gap"]
    if not rows:
        raise ValueError("No supported evaluation metrics found")
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(row):
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))

    introduction = ([f"Control:   {control}", f"Penalized: {penalized}",
                     "Delta = penalized - control; " + ("positive" if regression else "negative") + " is better."]
                    if penalized else [f"Checkpoint: {control}"])
    if penalized and not regression:
        introduction.append("Loss improvement % = 100 × (mean control loss − mean penalized loss) / mean control loss; "
                            "positive is better. Uses joint loss for paired tasks, ordinary loss otherwise. "
                            "This is a ratio of seed means; N/A means a zero baseline.")
    if html:
        cells = []
        for row in rows:
            rendered = []
            for i, value in enumerate(row):
                style = ""
                if penalized and i >= (1 if regression else 2) and value not in ("-", "N/A"):
                    number = float(value.split()[0].rstrip("%"))
                    if number != 0:
                        better = number > 0 if regression or i == 2 else number < 0
                        style = ' class="better"' if better else ' class="worse"'
                rendered.append(f"<td{style}>{escape(value).replace(' +/- ', ' ± ')}</td>")
            cells.append("<tr>" + "".join(rendered) + "</tr>")
        return ("<section>" + "".join(f"<p>{escape(text)}</p>" for text in introduction)
                + f"<p>{len(seeds)} split seeds · Mean ± sample standard deviation, not a confidence interval.</p>"
                + ("" if regression else "<p>Joint log loss averages both orders. Ordinary datasets use the A column.</p>")
                + "<div class='scroll'><table><thead><tr>"
                + "".join(f"<th>{escape(text.replace('Delta ', 'Δ ').replace('LL', 'log loss'))}</th>" for text in header)
                + "</tr></thead><tbody>" + "".join(cells) + "</tbody></table></div></section>")
    return "\n".join([*introduction, f"{len(seeds)} split seed(s); mean +/- sample SD (not a confidence interval).",
                      *([] if regression else ["Joint LL averages both orders; ordinary benchmarks use LL A."]),
                      "", line(header), "-+-".join("-" * width for width in widths), *map(line, rows)])


def write_report(payloads, path, control=None, penalized=None):
    """Write standalone browser-readable tables; no external scripts or styles required."""
    sections = [comparison_table(payload, control, penalized, html=True) for payload in payloads]
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model evaluation comparison</title><style>
body{font:15px system-ui,sans-serif;color:#182230;background:#f4f6fa;margin:32px}
main{max-width:1500px;margin:auto}h1{font-size:28px}section{background:white;padding:24px;margin:24px 0;border-radius:12px}
p{overflow-wrap:anywhere;color:#475467}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:12px 16px;border:1px solid #d0d5dd;text-align:right;white-space:nowrap}
th{background:#e9eef5}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f9fafb}
td.better{background:#dcfce7;color:#166534}td.worse{background:#fee2e2;color:#991b1b}.scroll{overflow-x:auto}
</style><main><h1>Model evaluation comparison</h1>
<p>Green: lower loss/gap or higher R². Red: the opposite. Colors show direction, not statistical significance.</p>
""" + "\n".join(sections) + "</main></html>"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    print(f"Table saved to {path.resolve()} — open this HTML file in a browser.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*")
    parser.add_argument("--summary", nargs="+", help="Summarize existing JSON files without reevaluating")
    parser.add_argument("--html", help="HTML table output path")
    parser.add_argument("--control", help="Control checkpoint key; default: first checkpoint")
    parser.add_argument("--penalized", help="Penalized checkpoint key; default: second checkpoint")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[64])
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--n-estimators", type=int, default=1, help="Ensemble size for paired benchmarks")
    parser.add_argument("--cache-dir", default="runs/eval_data")
    parser.add_argument("--skip-pairs", action="store_true")
    parser.add_argument("--output", help="Optional JSON results file")
    args = parser.parse_args(argv)
    if args.summary:
        if args.checkpoints:
            parser.error("Use either checkpoints or --summary")
        try:
            payloads = [json.loads(Path(path).read_text()) for path in args.summary]
            destination = args.html or (str(Path(args.summary[0]).with_suffix(".html")) if len(args.summary) == 1
                                        else str(Path(args.summary[0]).parent / "comparison_tables.html"))
            write_report(payloads, destination, args.control, args.penalized)
        except (ValueError, KeyError) as error:
            parser.error(str(error))
        return
    if not args.checkpoints:
        parser.error("Provide checkpoints or --summary JSON_FILE")
    if (args.control is None) != (args.penalized is None):
        parser.error("Supply both --control and --penalized")
    if len(args.checkpoints) > 2 and args.control is None:
        parser.error("Select --control and --penalized when evaluating more than two checkpoints")
    if args.control is not None and (args.control not in args.checkpoints or args.penalized not in args.checkpoints
                                     or args.control == args.penalized):
        parser.error("Select two distinct checkpoints present in the arguments")
    results = {}
    for checkpoint in args.checkpoints:
        model, cfg = load_model(checkpoint)
        results[checkpoint] = {}
        for seed in args.seeds:
            scores = evaluate(model, cfg.data.task, args.max_rows, seed, include_pairs=not args.skip_pairs,
                              context_sizes=args.context_sizes, cache_dir=args.cache_dir,
                              n_estimators=args.n_estimators)
            results[checkpoint][str(seed)] = scores
    payload = {"settings": vars(args), "results": results}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        destination = args.html or (str(Path(args.output).with_suffix(".html")) if args.output else "runs/evaluation.html")
        write_report([payload], destination, args.control, args.penalized)
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
