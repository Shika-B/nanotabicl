"""Compare nano models on small TabArena classification datasets.

Uses the authors' OpenML suite 457 (https://arxiv.org/abs/2506.16791).
Defaults: <=10,000 dataset rows, <=100 input features, and model-supported
class counts. These are our resource limits, not TabArena's official 'small'
subset. Uses all repeat-0 folds, 128 context rows, and <=1,024 queries per fold.
This is a small-context adaptation, not the official TabArena leaderboard protocol.
Preprocessing, paired sampling, metrics and reporting are shared with eval_openml.
"""
from .eval_openml import main as run_benchmark


def main(argv=None):
    run_benchmark(argv, benchmark="TabArena-v0.1 classification (size-filtered)", suite_id=457,
                  output_prefix="tabarena", max_dataset_rows=10000, max_features=100)


if __name__ == "__main__":
    main()
