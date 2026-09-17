"""Run three TabICLv2 checkpoints through TabArena-v0.1's official runner.

The default Lite scope is the first outer split of every classification dataset.
Training rows, test rows, preprocessing, inner validation and bagging are TabArena's.
This is a default-configuration comparison, with no hyperparameter search.

Install TabArena's benchmark extra in a Python 3.11-3.13 environment first. See
https://github.com/autogluon/tabarena#installation.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs/fg_full2000"
LABELS = ("original", "control", "penalized")


def checkpoint_info(path, label, expected_step):
    import torch

    if not path.is_file():
        raise ValueError(f"Missing {label} checkpoint: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in saved or "state_dict" not in saved:
        raise ValueError(f"{label} checkpoint is not a TabICL classification checkpoint: {path}")
    step = saved.get("step")
    penalty = saved.get("experiment", {}).get("lambda_fg")
    del saved
    if label != "original" and step != expected_step:
        raise ValueError(f"{label} checkpoint is at step {step}; expected {expected_step}: {path}")
    expected_penalty = {"control": 0., "penalized": .5}.get(label)
    if expected_penalty is not None and penalty != expected_penalty:
        raise ValueError(f"{label} checkpoint has lambda_fg={penalty}; expected {expected_penalty}")
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "step": step, "lambda_fg": penalty}


def build_experiments(checkpoints, *, num_gpus=1):
    """Three default-only configs, with common random seeds and official bagging."""
    from official_tabarena_models import MODEL_CLASSES
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.utils.config_utils import ConfigGenerator

    models = []
    for cls, checkpoint in zip(MODEL_CLASSES, checkpoints):
        generator = ConfigGenerator(model_cls=cls,
                                    manual_configs=[{"model_path": checkpoint["path"],
                                                     "allow_auto_download": False}],
                                    search_space={})
        models.append((generator, 0))
    # Static seeds remove a checkpoint-dependent source of randomness.
    bundle = TabArenaV0pt1ExperimentBundle(models=models, default_seed_config="static",
                                            sequential_local_fold_fitting=True)
    experiments = bundle.build_experiments(num_gpus=num_gpus)
    if len(experiments) != 3 or len({e.name for e in experiments}) != 3:
        raise RuntimeError("Expected exactly three distinct TabArena experiments")
    return experiments


def run_official(experiments, *, output, full=False):
    from tabarena.contexts import TabArenaContext

    context = TabArenaContext()
    scope = None if full else "lite"
    context.build_and_run_jobs(
        experiments,
        expname=str(output / "experiments"),
        subset=scope,
        build_kwargs={"problem_types": ["binary", "multiclass"]},
        new_result_prefix="[FG] ",
        debug_mode=True,  # sequential, in-process execution on one GPU
    )
    leaderboard = context.compare(output_dir=output / "comparison",
                                  subset=["classification"] if full else ["lite", "classification"],
                                  new_methods_only=True, plot=False)
    if len(leaderboard) != 3:
        raise RuntimeError(f"Expected three scored methods; got {len(leaderboard)}. "
                           f"Inspect {output / 'experiments'} for failed jobs.")
    return leaderboard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_RUN / "initial.ckpt")
    parser.add_argument("--control", type=Path, default=DEFAULT_RUN / "lambda_0/step_002000.ckpt")
    parser.add_argument("--penalized", type=Path, default=DEFAULT_RUN / "lambda_0.5/step_002000.ckpt")
    parser.add_argument("--step", type=int, default=2000, help="Expected fine-tuning step in both checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/tabarena_official_fg2000")
    parser.add_argument("--full", action="store_true", help="Run every split; default is first split per dataset")
    args = parser.parse_args(argv)
    try:
        checkpoints = [checkpoint_info(path, label, args.step) for path, label in
                       zip((args.original, args.control, args.penalized), LABELS)]
    except (ValueError, ImportError) as error:
        parser.error(str(error))
    try:
        experiments = build_experiments(checkpoints)
    except ImportError as error:
        parser.error(f"Install TabArena's benchmark extra and the local TabICL clone: {error}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"benchmark": "TabArena-v0.1", "scope": "full" if args.full else "lite",
                "task_type": "classification", "configs": "default-only",
                "validation_protocol": "official", "seed_assignment": "static",
                "checkpoints": dict(zip(LABELS, checkpoints)),
                "experiments": [experiment.name for experiment in experiments]}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error(f"Output contains a different experiment: {args.output}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    leaderboard = run_official(experiments, output=args.output, full=args.full)
    print(leaderboard.to_string(index=False))
    print(f"\nTabArena results: {args.output / 'comparison'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
