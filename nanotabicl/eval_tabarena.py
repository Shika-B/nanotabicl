"""Compare nano models on small TabArena classification datasets.

Uses the authors' OpenML suite 457 (https://arxiv.org/abs/2506.16791).
Defaults: <=100,000 dataset rows, <=20 input features, and model-supported
class counts. These are our resource limits, not TabArena's official 'small'
subset. Uses all repeat-0 folds, 128 context rows, and <=1,024 queries per fold.
This is a small-context adaptation, not the official TabArena leaderboard protocol.
Preprocessing, paired sampling, metrics and reporting are shared with eval_openml.
Evaluates training seeds 0, 1, 3 for penalties 0 and 0.5, with evaluation seed 0.
Writes only aggregate HTML to seeded_tabarena.html; per-seed JSON retains raw data.
Full TabICLv2 scores are read from tabarena_tabicl_comparison.html.
"""
from html import escape
from html.parser import HTMLParser
import argparse
import json
from pathlib import Path

import numpy as np

from .eval_openml import LAMBDAS, main as run_benchmark


def report_html(payload, metric=None):
    """Both relative-improvement tables, against λ=0; metric is unused."""
    regimes = payload.get("model_keys", list(LAMBDAS))
    baseline = next((v for v in regimes if v in ("0", "0.0")), None)
    if baseline is None:
        raise ValueError("Report requires a lambda=0 baseline")
    rows = [(k, r) for k, r in payload["tasks"].items() if r["status"] == "ok"]
    if any(set(r["scores"]) != set(regimes) for _, r in rows):
        raise ValueError("All successful tasks must contain the same models")

    def cells(values):
        return "".join('<td>N/A</td>' if not np.isfinite(v) else
                       f'<td class="{"positive" if v > 0 else "negative" if v < 0 else "neutral"}">{v:+.2f}%</td>'
                       for v in values)

    tables = []
    for key, title, sign in [("accuracy", "Accuracy improvement (%)", 1),
                             ("log_loss", "Log-loss improvement (%)", -1)]:
        body, valid = [], []
        for task_id, row in rows:
            base = row["scores"][baseline][key]
            values = np.array([row["scores"][v][key] for v in regimes])
            change = sign * 100 * (values - base) / base if base > 0 else np.full(len(regimes), np.nan)
            if np.isfinite(change).all():
                valid.append(change)
            body.append(f'<tr><th>{escape(row["name"])} <small>task {escape(str(task_id))}</small></th>{cells(change)}</tr>')
        footer = ""
        if valid:
            footer = (f'<tr><th>Mean improvement ({len(valid)} datasets)</th>{cells(np.mean(valid, axis=0))}</tr>'
                      f'<tr><th>Median improvement</th>{cells(np.median(valid, axis=0))}</tr>')
        headers = "".join(f'<th>λ = {escape(v)}</th>' for v in regimes)
        tables.append(f'<h2>{title}</h2><div class="scroll"><table><thead><tr><th>Dataset</th>{headers}</tr></thead>'
                      f'<tbody>{"".join(body)}</tbody><tfoot>{footer}</tfoot></table></div>')
    excluded = [f'<li>{escape(r.get("name", k))}: {escape(r.get("reason", r["status"]))}</li>'
                for k, r in payload["tasks"].items() if r["status"] != "ok"]
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>TabArena improvements</title>
<style>body{{font:15px system-ui;background:#f4f6fa;color:#182230;margin:32px}}main{{max-width:1400px;margin:auto}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:white}}
th,td{{padding:12px;border:1px solid #ddd;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}
thead,tfoot{{background:#e9eef5}}.positive{{color:#157347}}.negative{{color:#b42318}}
small{{font-weight:normal;color:#667085}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><main>
<h1>{escape(payload.get("benchmark", "TabArena"))} · improvements versus λ=0</h1>
<p>{len(rows)} datasets evaluated; {len(excluded)} skipped or failed. Positive means better; negative means worse.</p>
<p>Accuracy: 100 × (model − baseline) / baseline. Log loss: 100 × (baseline − model) / baseline.
These are relative percentages, not accuracy percentage points. Log-score improvement is reported as reduction in log loss (negative log likelihood).</p>
<p>Each dataset compares fold-averaged scores. Summary rows average dataset improvements with equal weight;
zero-baseline datasets show N/A and are excluded from that metric's summaries. Raw scores remain in JSON.</p>
{"".join(tables)}<h2>Skipped / failed datasets</h2><ul>{"".join(excluded)}</ul>
<details><summary>Protocol and checkpoints</summary><pre>{escape(json.dumps(payload["settings"], indent=2))}</pre></details>
</main></html>'''


def read_tabicl(path):
    """Read the standalone CPU report without importing the tabicl package."""
    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self.row, self.cell = [], [], None

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.row = []
            elif tag == "td":
                self.cell = ""

        def handle_data(self, data):
            if self.cell is not None:
                self.cell += data

        def handle_endtag(self, tag):
            if tag == "td":
                self.row.append(self.cell)
                self.cell = None
            elif tag == "tr" and len(self.row) == 5:
                self.rows.append(self.row)

    parser = Parser()
    parser.feed(Path(path).read_text())
    result = {r[0]: {"name": r[1], "log_loss": float(r[2]), "accuracy": float(r[3])} for r in parser.rows}
    if not result:
        raise ValueError("No TabICLv2 scores found in the standalone HTML")
    return result


def seeded_report(payloads, tabicl):
    """Paired seed statistics; overall statistics average datasets within seeds first."""
    if len(payloads) < 2:
        raise ValueError("At least two training seeds are required for sample SD")
    regimes = payloads[0]["model_keys"]
    if regimes[0] != "0":
        raise ValueError("First model must be lambda=0")
    protocol = ("suite", "seed", "context_size", "max_test_rows", "query_batch_size", "n_estimators",
                "folds", "max_dataset_rows", "max_features")
    for p in payloads[1:]:
        if p["model_keys"] != regimes or any(p["settings"].get(k) != payloads[0]["settings"].get(k) for k in protocol):
            raise ValueError("Seed evaluations must use identical models and evaluation settings")
    candidates = set().union(*(p["tasks"] for p in payloads))
    common = sorted(k for k in candidates if k in tabicl and
                    all(p["tasks"].get(k, {}).get("status") == "ok" for p in payloads))
    if not common:
        raise ValueError("No common successful datasets across all seeds and TabICLv2")
    for k in common:
        first = payloads[0]["tasks"][k]
        signature = lambda r: [(f["fold"], f["n_context"], f["n_test"]) for f in r["folds"]]
        if first["name"] != tabicl[k]["name"] or any(signature(p["tasks"][k]) != signature(first) for p in payloads):
            raise ValueError(f"Dataset or fold mismatch for task {k}")

    def stats(v, percent=False):
        if not np.isfinite(v).all():
            return "N/A"
        return f'{np.mean(v):+.2f} ± {np.std(v, ddof=1):.2f}%' if percent else f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}'

    tables = []
    for metric, sign in [("accuracy", 1), ("log_loss", -1)]:
        # Axes: dataset, training seed, penalty regime.
        values = np.array([[[p["tasks"][k]["scores"][r][metric] for r in regimes] for p in payloads] for k in common])
        full = np.array([tabicl[k][metric] for k in common])
        def row(label, scores, reference):
            base = scores[:, 0]
            cells = []
            for i in range(len(regimes)):
                delta = np.divide(sign * 100 * (scores[:, i] - base), base,
                                  out=np.full_like(base, np.nan), where=base > 0)
                cells.extend([f'<td>{stats(scores[:, i])}</td>', f'<td>{stats(delta, True)}</td>'])
            delta = np.divide(sign * 100 * (reference - base), base,
                              out=np.full_like(base, np.nan), where=base > 0)
            return f'<tr><th>{escape(label)}</th>{"".join(cells)}<td>{reference:.4f} (SD N/A)</td><td>{stats(delta, True)}</td></tr>'
        body = [row(tabicl[k]["name"], values[i], full[i]) for i, k in enumerate(common)]
        footer = row("Overall (equal dataset weights)", values.mean(axis=0), full.mean())
        headers = "".join(f'<th>λ={escape(r)} score</th><th>Improvement vs λ=0</th>' for r in regimes)
        tables.append(f'<h2>{"Accuracy" if metric == "accuracy" else "Log loss"}</h2><div><table><thead><tr><th>Dataset</th>{headers}'
                      f'<th>Full TabICLv2 score</th><th>Improvement vs λ=0</th></tr></thead><tbody>{"".join(body)}</tbody><tfoot>{footer}</tfoot></table></div>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Seeded TabArena</title>
<style>body{font:15px system-ui;margin:32px;background:#f4f6fa}div{overflow-x:auto}table{border-collapse:collapse;background:white}
th,td{padding:12px;border:1px solid #ddd;text-align:right;white-space:nowrap}th:first-child{text-align:left}thead,tfoot{background:#e9eef5}</style>
<h1>TabArena · training-seed comparison</h1>''' + f'''<p>{len(common)} common datasets; {len(payloads)} training seeds.
Mean ± sample SD across training seeds, not confidence intervals. Accuracy is a fraction; log loss is in nats.
Positive relative improvement is better: accuracy (model−baseline)/baseline; log loss (baseline−model)/baseline, multiplied by 100.
Improvements are paired within training seed. Overall scores average datasets within each seed before computing mean and SD.</p>
<p>TabICLv2 is one fixed pretrained model: its score has no training-seed SD available. Its improvement SD reflects variation in the nano baseline only.
Saved TabICLv2 HTML scores are rounded to six decimals and contain no sampling metadata for independent protocol verification.</p>
<p>Excluded task IDs (missing, skipped or failed in any source): {escape(", ".join(sorted(candidates - set(common))) or "none")}</p>
''' + "".join(tables) + f'<details><summary>Evaluation settings</summary><pre>{escape(json.dumps(payloads[0]["settings"], indent=2))}</pre></details></html>'


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare training seeds; additional evaluation flags are passed to the benchmark runner.")
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[0, 1, 3])
    parser.add_argument("--lambdas", nargs="+", default=["0", "0.5"])
    parser.add_argument("--checkpoint-template", default="runs/cosine3000_fg{lambda}_seed{seed}/latest.pt")
    parser.add_argument("--tabicl-html", default="tabarena_tabicl_comparison.html")
    parser.add_argument("--html", default="seeded_tabarena.html")
    parser.add_argument("--output-dir", default="runs/seeded_tabarena")
    parser.add_argument("--summary", nargs="+", help="Aggregate existing per-training-seed JSON files without inference")
    args, extra = parser.parse_known_args(argv)
    if len(set(args.training_seeds)) != len(args.training_seeds) or len(args.training_seeds) < 2:
        parser.error("Provide at least two distinct training seeds")
    if any(v.split("=")[0] in ("--checkpoints", "--output") for v in extra):
        parser.error("Use --checkpoint-template and --output-dir for seeded evaluation")
    tabicl = read_tabicl(args.tabicl_html)
    if args.summary:
        if extra:
            parser.error("Evaluation options cannot change saved results in --summary mode")
        payloads = [json.loads(Path(path).read_text()) for path in args.summary]
        seeds = [p["settings"].get("training_seed") for p in payloads]
        if all(s is not None for s in seeds) and len(set(seeds)) != len(seeds):
            parser.error("Summary files must represent distinct training seeds")
    else:
        payloads = []
        for seed in args.training_seeds:
            paths = [args.checkpoint_template.format(**{"lambda": r, "seed": seed}) for r in args.lambdas]
            if any(not Path(path).is_file() for path in paths):
                parser.error(f"Missing checkpoint for training seed {seed}: {paths}")
        for seed in args.training_seeds:
            paths = [args.checkpoint_template.format(**{"lambda": r, "seed": seed}) for r in args.lambdas]
            output = Path(args.output_dir) / f"seed{seed}.json"
            payload = run_benchmark([*extra, "--lambdas", *args.lambdas, "--checkpoints", *paths, "--output", str(output)],
                                    benchmark="TabArena-v0.1 classification (size-filtered)", suite_id=457,
                                    output_prefix="tabarena", max_dataset_rows=100000, max_features=20, report_builder=None)
            payload["settings"]["training_seed"] = seed
            output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            payloads.append(payload)
    path = Path(args.html)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(seeded_report(payloads, tabicl), encoding="utf-8")
    print(f"Table saved to {path.resolve()}")


if __name__ == "__main__":
    main()
