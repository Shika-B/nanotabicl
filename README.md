# NanoTabICL: a minimal TabICLv2 implementation

| [Full TabICLv2 code](https://github.com/soda-inria/tabicl) | [TabICLv2 Paper](https://arxiv.org/abs/2602.11139) |
|----------------------------------------------------|------------------------|

This repository provides a short implementation of the [TabICLv2](https://arxiv.org/abs/2602.11139) architecture (<170 LOC)
and a slightly simplified implementation of the TabICLv2 prior (<330 LOC).
For using our pre-trained TabICLv2 model,
please visit the [main repository](https://github.com/soda-inria/tabicl). This repository also contains a compact
pre-training loop for experiments and reproduction attempts.

Compared to [nanoTabPFN](https://github.com/automl/nanoTabPFN),

- the model is TabICLv2, not TabPFN,
- we implement the full model (including RoPE, QASSMax, etc.) with speed optimizations
  (but without the inference wrappers + memory optimizations from the main repository),
- we implement regression as well,
- we provide a (slightly simplified but still well-performing) prior for dataset generation,
- we provide a small sklearn-compatible inference interface,
- we provide a minimal pre-training loop, but not the distributed and memory-optimized training system from the
  full repository.

Note that this repo uses LayerNorm with bias, which is used by the classification checkpoint of TabICLv2,
while the regression checkpoint of TabICLv2 uses LayerNorm without bias.

## Model usage

```python
from nanotabicl import NanoTabICLv2
import torch

batch_size, n_train, n_test, n_cols = 2, 32, 16, 10
model = NanoTabICLv2(max_classes=10, out_dim=10)  # original model size
X_train_and_test = torch.randn(batch_size, n_train+n_test, n_cols)
y_train = torch.randint(10, size=(batch_size, n_train)).float()
y_test_pred_logits = model(X_train_and_test, y_train)

# if you want a smaller model + regression with 999 quantiles instead
# warning: for regression, you need to standardize y yourself (and backtransform the output)
model = NanoTabICLv2(max_classes=0, out_dim=999, embed_dim=96,
                 col_num_blocks=2, row_num_blocks=2, icl_num_blocks=4,
                 col_nhead=4, row_nhead=4, icl_nhead=4)
y_train = torch.randn(batch_size, n_train)
y_test_pred_quantiles = model(X_train_and_test, y_train)
```

Note that `X_train_and_test` is standardized inside the model (based on train only).
We do not include other preprocessing options from TabICLv2
since they are normally not part of the architecture.

For a checkpoint, the sklearn-compatible wrappers apply the prior's preprocessing and average predictions over
random feature/class orders:

```python
from nanotabicl import NanoTabICLClassifier, NanoTabICLRegressor

clf = NanoTabICLClassifier(model="runs/small_cosine_3000/latest.pt", n_estimators=8).fit(X_train, y_train)
proba = clf.predict_proba(X_test)

reg = NanoTabICLRegressor(model="runs/regression/latest.pt").fit(X_train, y_train)
quantiles = reg.predict_quantiles(X_test, alphas=(0.1, 0.5, 0.9))
pred = reg.predict(X_test)  # median prediction
```

## Training and evaluation


> **Warning:** the complete three-stage recipe is far too costly for a casual experiment. Stage 1 is 500,000
> optimizer steps, stage 2 adds 40,000 steps with sequences up to 10,240 rows, and stage 3 adds 10,000 steps with
> sequences up to 60,000 rows. It requires substantial GPU time, CPU data-generation capacity, storage, and memory. These configuration files are only included to replicate TabICLv2 actual setup.
> Use `configs/small.yaml` or override `optim.max_steps`, sequence lengths, and model size for quick experiments.


Install the package in editable mode (the optional dependencies add pytest, matplotlib, and Weights & Biases):

```bash
python -m pip install -e ".[dev]"
# if you get an error about Go not being installed to build wandb, try instead
python -m pip install -e ".[dev]" "wandb==0.21.4" 
```

The small configuration trains two categorical targets with the mean cross-entropy of
A|X, B|X, A|B,X, and B|A,X, plus `optim.lambda_fg` times their factorization gap.
The penalty defaults to zero. Conditional views append the other target as a feature:
observed values in context rows and query rows for conditional cross-entropy. With
a nonzero penalty, hypothetical query classes are enumerated for the gap. The
zero-penalty baseline uses four direct forwards with the same supervision. Class probabilities are normalized over
each target's classes present in the context.

Use separate output directories and the same seed to compare objectives from the same initialization:

```bash
python -m nanotabicl.train configs/small.yaml optim.lambda_fg=0 seed=0 out_dir=runs/fg_control
python -m nanotabicl.train configs/small.yaml optim.lambda_fg=0.3 seed=0 out_dir=runs/fg_penalized
```

Logs include per-view cross-entropy and accuracy, mean `ce`, and total `loss`; penalized
training also logs `factorization_gap`. Enumeration uses `2 + 2 * data.max_classes`
forward passes per micro-batch, while zero-penalty training uses four.
For cheaper experiments, set `data.max_classes=4` in both runs; those models can only
evaluate tasks with up to four classes. Single-target training remains available with
`data.n_targets=1 optim.lambda_fg=0`.

To run and evaluate the default small configuration:

```bash
python -m nanotabicl.train configs/small.yaml
python -m nanotabicl.eval runs/small_cosine_3000/latest.pt
```

Training now uses 128 fixed validation tables (seed 1729), generated from the same
prior as training and persisted in `validation.pt`. Generation preserves the training
random streams. Every 100 steps, validation measures the equal-weight mean loss across
tables, using held-out rows and averaging the four CEs for two-target classification.
The penalty is excluded from validation loss. Per-view results go to
`validation.jsonl`.

The small run lasts exactly 3,000 optimizer steps: 100 steps of linear warmup to
LR 0.003, followed by cosine decay to 0.00001 on the final update. Validation is
for logging only: it never changes the LR, stops training, or selects a checkpoint.
`optim.warmup_frac` is a legacy config field and is no longer used.

`latest.pt` contains the last training state for resuming; training returns that
final model. No `best.pt` is created. Start in a fresh directory for this recipe
and compare final checkpoints using identical schedules and budgets.
Historical results from the original single-target small configuration (not the new two-target setup):
| Dataset | Score |
|---|---:|
| iris | 0.920 |
| wine | 0.966 |
| breast_cancer | 0.933 |
| digits | 0.904 |

### Classification and coherence evaluation

`eval_openml` and `eval_tabarena` evaluate the four nano models without the official
`tabicl` package. To evaluate pretrained TabICLv2 separately on CPU, copy
`eval_tabicl_cpu.py` and your comparison JSON into a separate environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tabicl==2.2.0 openml
python eval_tabicl_cpu.py tabarena_comparison.json --output tabicl_tabarena.json
```

The standalone script replays successful tasks with the saved sampling settings,
and writes an HTML table alongside its JSON. It uses float32 and temperature 1.0;
TabICLv2's internal preprocessing remains its official pipeline.

For a size-filtered TabArena classification comparison, using the same four checkpoints:

```bash
python -m nanotabicl.eval_tabarena --device cuda:0
```

This uses the authors' [TabArena-v0.1 OpenML suite 457](https://arxiv.org/abs/2506.16791),
excluding regression tasks and datasets above 10,000 rows, 100 input features, or
the models' class capacity. These are configurable resource limits, not TabArena's
official small subset: use `--max-dataset-rows 50000 --max-features 200` to expand it.
Sampling and scoring match `eval_openml`: 128 context rows, all repeat-0 folds,
up to 1,024 test rows per fold. Thus these results are not official leaderboard scores.
Open `runs/tabarena_comparison.html` for the four-column model table and aggregate
ranks/improvements. JSON includes fold scores and explicit skip/failure reasons.
The existing OpenML dependency is sufficient; no TabArena package is needed.

```bash
python -m nanotabicl.eval_tabarena --summary runs/tabarena_comparison.json --metric accuracy --html runs/tabarena_accuracy.html
```

For a broader four-model comparison on [OpenML-CC18](https://docs.openml.org/benchmark/):

```bash
python -m pip install openml
python -m nanotabicl.eval_openml --device cuda:0
```

This uses the four `runs/cosine3000_fg{0,0.03,0.1,0.3}_seed0/latest.pt` checkpoints.
Override with `--checkpoints PATH0 PATH003 PATH01 PATH03`. Open
`runs/openml_comparison.html`: each dataset has four log-loss columns, followed by
average score/rank, average and median relative improvement, and percentage of tasks
improved versus lambda zero. Detailed fold metrics are saved incrementally in JSON.

The protocol uses every task in suite 99, all repeat-0 official folds, 128 context
rows and up to 1,024 test rows per fold, with identical row samples for all models.
This is a small-context adaptation of CC18. `--max-test-rows 0` uses complete test
folds; `--folds 0 1 2` offers a shorter pilot. Classes absent from context are assigned
zero probability and their test frequency is recorded. Over-capacity tasks and failed
tasks are listed in the report and excluded for all models; features are not capped.
Summary rows weight tasks equally, not individual test examples. Tied ranks are averaged.

To display accuracy or Brier score without rerunning inference:

```bash
python -m nanotabicl.eval_openml --summary runs/openml_comparison.json --metric accuracy --html runs/openml_accuracy.html
```

Compare checkpoints with identical context sizes and split seeds:

```bash
python -m nanotabicl.eval runs/fg_control/latest.pt runs/fg_penalized/latest.pt \
  --context-sizes 32 64 128 --seeds 0 1 2 --output runs/fg_comparison.json
```

Evaluation creates an HTML table beside the JSON and prints its path. Open the HTML
file in a browser for formatted tables with green improvements and red regressions.
Detailed metrics remain in the JSON; use `--html PATH` to choose the report location.
Summarize an existing results file without rerunning evaluation:

```bash
python -m nanotabicl.eval --summary runs/fg_comparison.json
```

Combine several existing comparisons into one HTML report without reevaluation:

```bash
python -m nanotabicl.eval --summary runs/comparison_fg0.03.json runs/comparison_fg0.1.json runs/comparison_fg0.3.json --html runs/comparison.html
```

The first checkpoint is the control and the second is penalized; override with
`--control CHECKPOINT_KEY --penalized CHECKPOINT_KEY` if needed. The table reports
penalized minus control, averaged over paired split seeds, with their sample standard
deviation. Negative differences mean improvement. Split-seed variability does not
measure variability across independently trained models.

The original Iris, Wine, Breast Cancer and Digits tasks report accuracy, log loss,
and multiclass Brier score. The additional benchmarks are:

| Dataset | A | B | Features excluded |
|---|---|---|---|
| [Car Evaluation (Bohanec)](https://archive.ics.uci.edu/dataset/19/car+evaluation) | Acceptability | Safety | Both targets |
| [Nursery (Rajkovic)](https://archive.ics.uci.edu/dataset/76/nursery) | Recommendation | Health | Both targets |
| [Student Performance (Cortez)](https://archive.ics.uci.edu/dataset/320/student+performance) | G2 >= 10 | G3 >= 10 | G1, G2, G3 |

UCI datasets are CC BY 4.0; archives download once into `runs/eval_data`.
Math and Portuguese students are evaluated separately because the courses overlap.
Each paired benchmark reports accuracy, log loss and Brier score for A, B,
A given B, B given A, and each of the two joint factorizations, plus their mean
total-variation factorization gap. Brier scores sum over classes, then average
over examples; log loss uses natural logarithms and a probability floor of 1e-12.

Splits are uniform and fixed by seed, with up to 256 query rows. Category encodings
and feature preprocessing are fitted on context only. Rare labels are not dropped
or forced into context; classes absent from context get zero predicted probability,
and `unseen_a_rate` / `unseen_b_rate` report their query frequency. In particular,
Nursery has only two examples of the `recommend` class. Interpret gap alongside
prediction scores and these coverage diagnostics.

Paired tasks default to one ensemble member (`--n-estimators`); the original
benchmarks retain their eight-member estimator. Compare identical settings across
checkpoints and retain per-seed results rather than treating query rows as independent
training runs. The full suite requires capacity for ten classes; Nursery alone needs
five. `--skip-pairs` runs the original benchmarks without downloading UCI data.
Regression evaluation keeps its original datasets and R² calculation.

The reference stage-1 configurations are:

```bash
python -m nanotabicl.train configs/classification.yaml
python -m nanotabicl.train configs/regression.yaml
```

Training writes `latest.pt`, the resolved `config.yaml`, and `metrics.jsonl` to `out_dir`. If `latest.pt` already
exists, the run resumes automatically. To start the next curriculum stage from an earlier checkpoint, use a new output directory and `init_from`:

```bash
python -m nanotabicl.train configs/classification.yaml configs/stage2.yaml \
    init_from=runs/classification/latest.pt \
    out_dir=runs/classification_stage2

python -m nanotabicl.train configs/classification.yaml configs/stage2.yaml configs/stage3.yaml \
    init_from=runs/classification_stage2/latest.pt \
    out_dir=runs/classification_stage3
```

The same commands work with `configs/regression.yaml` in place of `configs/classification.yaml`.

Run the test suite with:

```bash
python -m pytest
```

## Nanoprior

In `nanotabicl/prior.py`, we provide a concise and slightly simplified
implementation of the TabICLv2 prior,
which nevertheless performs similarly well in our
(smaller-scale) experiments.
Changes compared to the TabICLv2 prior are:

- No correlated sampling of scalar variables / hyperparameters.
- Removed graph filtering (should be captured by dataset filtering anyway).
- No graph pruning of irrelevant nodes
(can be a bit slower, but yields the same output).
- Categorical converters with cardinality k always use k dimensions,
never fewer.
- Kumaraswamy warping now affects the extracted dataset column,
not the propagated value.
- Fixed the constant in the random EM function.
- The random activation used for random activation matrices
follows the general random activation now
by only using two of the four possible random power activation types.
- Use corrected categorical size limit of 100 instead of 10.
- Dataset filtering uses the same categorical sizes
in every attempt until a non-filtered dataset is generated.
- fill NaN/inf with zero instead of discarding the whole dataset.
- (Use torch.sign() in random activation,
which has a different behavior at 0 than `2*(x>=0).float()-1`.)

Preprocessing code (outlier handling, standard scaling)
is also not included as it could be done inside the model.

When running the prior module directly, it generates a plot of some random datasets.

## Updates

- 2026/08/17: Replace class-embedding with one-hot + linear, which has a bias compared to the nn.Embedding used before, and matches the full model implementation.
- 2026/06/10: Add nanoprior.
Bugfix based on [#2](https://github.com/soda-inria/nanotabicl/issues/2): Subtract mean when standardizing input data.
- 2026/03/25: Add faster + cached RoPE implementation based on the TabICLv2 version (warning: this permutes the neurons, so it's not compatible with older nanotabicl checkpoints).
- 2026/09/09: Add pretraining code and some evaluations for the trained checkpoints.
