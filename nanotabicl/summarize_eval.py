"""Summarize paired checkpoint differences from an existing evaluation JSON."""
import argparse
import json
from pathlib import Path

import numpy as np


def comparison_table(payload, control=None, penalized=None):
    """Report mean +/- sample SD of per-seed differences, not pooled row scores."""
    results = payload["results"]
    if control is None and penalized is None:
        if len(results) != 2:
            raise ValueError("Specify --control and --penalized when the JSON does not contain exactly two checkpoints")
        control, penalized = results
    if control not in results or penalized not in results or control == penalized:
        raise ValueError("Select two distinct checkpoint keys present in the JSON")
    baseline, treatment = results[control], results[penalized]
    if not baseline or baseline.keys() != treatment.keys():
        raise ValueError("Both checkpoints must have the same nonempty set of split seeds")
    seeds = sorted(baseline)
    keys = set(baseline[seeds[0]])
    for seed in seeds:
        if set(baseline[seed]) != keys or set(treatment[seed]) != keys:
            raise ValueError("Metric keys must match across checkpoints and seeds")

    def delta(metric_keys):
        values = [np.mean([treatment[seed][key] - baseline[seed][key] for key in metric_keys]) for seed in seeds]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values for {metric_keys}")
        mean = f"{np.mean(values):+.4f}"
        return f"{mean} +/- {np.std(values, ddof=1):.4f}" if len(values) > 1 else mean

    rows = []
    prefixes = [key.removesuffix("/factorization_gap") for key in keys if key.endswith("/factorization_gap")]
    for prefix in sorted(prefixes, key=lambda p: (p.split("/")[0], int(p.rsplit("_", 1)[1]))):
        dataset, context = prefix.split("/context_")
        rows.append([dataset, context, delta([f"{prefix}/a/log_loss"]), delta([f"{prefix}/b/log_loss"]),
                     delta([f"{prefix}/joint_ab/log_loss", f"{prefix}/joint_ba/log_loss"]),
                     delta([f"{prefix}/factorization_gap"])])
    # Retain a compact view of the ordinary classification benchmarks too.
    for key in sorted(keys):
        if key.count("/") == 1 and key.endswith("/log_loss"):
            rows.append([key.split("/")[0], "50/50", delta([key]), "-", "-", "-"])
    if not rows:
        raise ValueError("No classification comparison metrics found in this JSON")
    header = ["Dataset", "Context", "Delta LL A", "Delta LL B", "Delta joint LL", "Delta gap"]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(row):
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))

    return "\n".join([
        f"Control:   {control}", f"Penalized: {penalized}",
        f"Delta = penalized - control; negative is better. {len(seeds)} paired split seed(s).",
        "Entries: mean +/- sample SD across split seeds (not a confidence interval).",
        "Joint LL averages both factorization orders; ordinary benchmarks use the LL A column.",
        "", line(header), "-+-".join("-" * width for width in widths), *map(line, rows),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file")
    parser.add_argument("--control", help="Checkpoint key; default: first checkpoint in the JSON")
    parser.add_argument("--penalized", help="Checkpoint key; default: second checkpoint in the JSON")
    args = parser.parse_args()
    try:
        print(comparison_table(json.loads(Path(args.json_file).read_text()), args.control, args.penalized))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
