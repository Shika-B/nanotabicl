"""Compare nano models on small TabArena classification datasets.

Uses the authors' OpenML suite 457 (https://arxiv.org/abs/2506.16791).
Defaults: <=10,000 dataset rows, <=100 input features, and model-supported
class counts. These are our resource limits, not TabArena's official 'small'
subset. Uses all repeat-0 folds, 128 context rows, and <=1,024 queries per fold.
This is a small-context adaptation, not the official TabArena leaderboard protocol.
Preprocessing, paired sampling, metrics and reporting are shared with eval_openml.
"""
from html import escape
import json

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


def main(argv=None):
    run_benchmark(argv, benchmark="TabArena-v0.1 classification (size-filtered)", suite_id=457,
                  output_prefix="tabarena", max_dataset_rows=10000, max_features=100,
                  report_builder=report_html)


if __name__ == "__main__":
    main()
