# Paired consistency fine-tuning

From the cloned `tabicl` directory, install this checkout in the environment used
for full TabICLv2 (PyTorch >=2.2):

```bash
python -m pip install -e .
python -m tabicl.train.finetune_consistency --steps 1000 --devices cuda:0 --output runs/fg_broad
```

`1000` is an example budget, not a selected optimum. `--steps` is required for a new run.
After shared data preparation, both arms run **in parallel**. One device argument
runs both processes on that device; two arguments assign one device per arm.
Sharing one GPU requires memory for both full models and may be slower or exhaust VRAM.
Use a new output directory per experiment. Normal runs print nothing to stdout;
metrics, data progress, warnings and errors go to **one log file**, described below.
The official `tabicl-classifier-v2-20260212.ckpt` downloads automatically; use
`--checkpoint /path/to/checkpoint.ckpt` to load an existing copy.

## Experimental choices

- Start both arms from the same pretrained weights; update **all parameters**.
  Full fine-tuning remains the choice: LoRA is not established as better for this
  objective and imposes a rank constraint when parameter efficiency is not needed.
- GraphSCM samples two categorical target nodes from **one graph**. The shared
  graph can induce dependence but does not guarantee that every pair is dependent.
  Default 2–10 classes per target; each context must cover the sampled label support.
- Uniformly sample context sizes **128, 512, 2048, 8192** per table, with **128 query
  rows**, **2–100 input features**, and **16 tables** per optimizer update, accumulated
  one table at a time. The conditioning view adds one feature. Small contexts remain
  in the mixture to retain coverage of the 128-row evaluation protocol. Override
  the buckets with e.g. `--context-rows 128 512 2048`.
- These settings broaden training coverage. They do not guarantee gains on every
  dataset, and 8192-row tables still require substantial GPU memory in float32.
- Both arms minimize the average CE of `A|X`, `B|X`, `A|X,B`, `B|X,A`.
  Append the observed other target to **context** input columns; enumerate each
  possible other-target value in query rows. Use actual query labels only in CE.
- Both arms run the same enumeration and compute the gap, so forward workload and
  supervised views match. The loss is `CE + lambda_fg * TV(joint_AB, joint_BA)`,
  with lambda 0 versus 0.5. Class axes use their actual supported classes.
- Full float32, no AMP/FA3/TF32. Gradient checkpointing is enabled to save memory.
  A two-pass backward first computes gradients with respect to logits, then replays
  each view separately to update parameter gradients. This is the same CE+FG
  gradient (tested against a full computation graph), without retaining 22 view
  graphs simultaneously. Both arms use this path. Dropout must be zero.
- Fresh AdamW, LR 1e-5 -> 1e-6, 5% linear warmup then cosine, weight decay 0.01,
  gradient clipping 1.0. These are configurable pilot fine-tuning settings, not
  the original Muon pretraining recipe. No model-dependent stopping or scheduling.
- Save all prior tables once, then replay them for both arms. This also isolates
  data randomness from model computation. Graph-level predictability filtering is
  enabled; the original single-target dataset-level predictability filter is not.
  Graph feature groups are sorted to make seeded sampling stable across processes.
- Save 128 validation tables with seed 1729. Log validation CE and gap before
  fine-tuning, every 100 steps and at the end. Validation is logging only. Training
  seed defaults to 0; validation and training use separate seed namespaces.
  Validation also reports results separately by context bucket.

The experiment matches the four-view construction used in nanoTabICL. Marginals
and conditionals see different context columns; this is the same operational gap,
not a guarantee of Bayesian coherence under a shared information set.

## Outputs

`experiment.json` records settings, the source checkpoint SHA256 and the exact log
path. For the nested checkout, the default log folder is the **outer project's
`logs/`**, independent of working directory. `--log-dir` overrides it. Names include
the output folder, seed, step budget, penalties and a path hash, for example
`fg_broad_seed0_steps1000_lambda0-vs0.5_<hash>.log`. All processes use a queue and a
single writer; timestamps and arm names identify records. There are no per-arm logs.

`initial.ckpt`, `data/`, `validation_data/` and `validation.pt` preserve the initial
weights and shared tables. The broader data bank can occupy several GB.
Each of `lambda_0/` and `lambda_0.5/` contains `latest.ckpt` and step snapshots
**every 50 updates**, retaining the **last three** snapshots. Configure these with
`--save-every` and `--keep-checkpoints`. Saves use atomic replacement. Snapshots
are hard links to avoid duplicating disk blocks; treat checkpoint files as immutable.
Step counts are optimizer updates, each using `--tables-per-step` tables.

## Resume

```bash
python -m tabicl.train.finetune_consistency --resume --output runs/fg_broad
```

Settings are restored from the manifest; no need to repeat them. Each arm restores
its own optimizer, weights and update count, and continues the original cosine
schedule on the saved data. Completed arms are skipped. The original checkpoint
download is not needed again. To resume on one GPU, add `--devices cuda:2`; to
use two GPUs, add `--devices cuda:2 cuda:3`.
The existing single log is appended, with explicit resume events. Updates replayed
after a crash may appear twice in the log; use the most recent entry for that arm/step.

SIGINT/SIGTERM requests a graceful stop: finish the current optimizer update and
save both arms. A hard kill resumes from the latest periodic atomic checkpoint.
Data generation also resumes from saved tables. If one arm fails, its traceback is
logged and the other arm stops after saving its completed update. Do not launch
two commands against the same output folder simultaneously.
Changing N, LR, seeds or data settings on resume is rejected, preserving the paired
comparison. Earlier runner versions' experiments are not automatically migrated.

```python
from tabicl import TabICLClassifier

clf = TabICLClassifier(model_path="runs/fg_broad/lambda_0.5/latest.ckpt",
                      device="cuda:0", n_estimators=1, random_state=0,
                      softmax_temperature=1.0, use_amp=False, use_fa3=False)
```

For comparison, use the same evaluation protocol and include the untouched source
checkpoint. Choose N using pilot validation data, then fix it for both arms and
confirm across training seeds. Do not choose N separately per arm on test results.
Resuming is explicit; a new run never silently overwrites an existing experiment.
