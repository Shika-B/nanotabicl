"""Paired full-model fine-tuning on a replayed two-target graph prior.

python -m tabicl.train.finetune_consistency --steps 1000 --devices cuda:0 cuda:1

Both arms use exactly the same checkpoint, tables, four-view supervision,
enumerated forward calls, optimizer, and schedule. Only lambda_fg differs.
Runs in parallel, logs to one project-root logs/ file, and supports --resume.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import logging
from logging.handlers import QueueHandler, QueueListener
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from tabicl._model.tabicl import TabICL
from tabicl._model.attention import set_flash_attn3_enabled
from tabicl.prior._graph_scm import GraphSCM
from tabicl.prior.graph_lib._config import PriorConfig

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
FORMAT_VERSION = 2


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_table(args, seed):
    """One graph -> X (1,T,F), y (1,T,2), class counts (K_A,K_B).

    Reject degenerate targets and splits with query classes absent from context.
    Remap each target to consecutive class IDs. Both arms use the saved result.
    """
    seed_all(seed)
    prior_config = PriorConfig(filter_unpredictable_graphs=True)
    contexts = args.context_rows if isinstance(args.context_rows, list) else [args.context_rows]
    n_context = int(np.random.choice(contexts))
    for _ in range(100):
        features = int(np.random.randint(args.min_features, args.max_features + 1))
        sizes = [int(np.random.randint(2, args.max_classes + 1)) for _ in range(2)]
        x, y = GraphSCM(seq_len=n_context + args.query_rows, num_features=features,
                        max_features=features, num_classes=sizes, config=prior_config)()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (y < 0).any():
            continue
        labels, counts = [], []
        for col in y.unbind(-1):
            classes, encoded = torch.unique(col, sorted=True, return_inverse=True)
            if len(classes) < 2 or len(torch.unique(encoded[:n_context])) != len(classes):
                break
            labels.append(encoded)
            counts.append(len(classes))
        if len(labels) == 2:
            return {"x": x.float().unsqueeze(0), "y": torch.stack(labels, -1).unsqueeze(0),
                    "classes": counts, "seed": seed, "n_context": n_context}
    raise RuntimeError(f"Could not sample a valid two-target table after 100 attempts (seed={seed})")


def conditional_logits(model, x, target, condition, n_context, condition_classes, target_classes):
    """Enumerate P(target | condition, X), keeping target query labels hidden.

    x: (B,T,F); target, condition: (B,T); n_context=C.
    For each conditioning class k, append a column with observed context labels
    and k in every query row. Forward input: (B,T,F+1), target context: (B,C).
    Output: (B,T-C,K_condition,K_target). Softmax is over the final axis.
    """
    outputs = []
    for value in range(condition_classes):
        column = torch.cat([condition[:, :n_context], torch.full_like(condition[:, n_context:], value)], 1)
        outputs.append(model(torch.cat([x, column.to(x.dtype).unsqueeze(-1)], -1),
                             target[:, :n_context].float())[..., :target_classes])
    return torch.stack(outputs, -2)


def view_inputs(table, n_context):
    """Yield the same 2+K_A+K_B inputs in each forward/backward pass."""
    x, y = table["x"], table["y"]
    a, b = y.unbind(-1)
    ka, kb = table["classes"]
    yield x, a[:, :n_context].float(), ka
    yield x, b[:, :n_context].float(), kb
    for target, condition, kc, kt in [(b, a, ka, kb), (a, b, kb, ka)]:
        for value in range(kc):
            column = torch.cat([condition[:, :n_context], torch.full_like(condition[:, n_context:], value)], 1)
            yield torch.cat([x, column.to(x.dtype).unsqueeze(-1)], -1), target[:, :n_context].float(), kt


def loss_from_views(views, table, n_context, lambda_fg):
    """Mean of four observed-label CEs + lambda_fg * mean joint TV gap.

    Both arms enumerate all conditioning classes, including when lambda_fg=0.
    Conditional CE gathers only the observed query conditioning label; all
    class pairs contribute to the gap, with gradients through both joint laws.
    """
    y = table["y"]
    a, b = y.unbind(-1)
    ka, kb = table["classes"]
    la, lb = views[:2]
    lba = torch.stack(views[2:2 + ka], -2)
    lab = torch.stack(views[2 + ka:], -2)
    at, bt = a[:, n_context:], b[:, n_context:]
    observed_ba = lba.gather(-2, at[..., None, None].expand(-1, -1, 1, kb)).squeeze(-2)
    observed_ab = lab.gather(-2, bt[..., None, None].expand(-1, -1, 1, ka)).squeeze(-2)
    metrics = {}
    ces = []
    for name, logits, labels in [("a", la, at), ("b", lb, bt),
                                 ("a_given_b", observed_ab, at), ("b_given_a", observed_ba, bt)]:
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        ces.append(ce)
        metrics[f"ce_{name}"] = ce.detach()
    ce = torch.stack(ces).mean()
    joint_ab = la.softmax(-1).unsqueeze(-1) * lba.softmax(-1)
    joint_ba = lb.softmax(-1).unsqueeze(-2) * lab.softmax(-1).transpose(-1, -2)
    gap = (joint_ab - joint_ba).abs().sum((-1, -2)).mean() / 2
    loss = ce + lambda_fg * gap
    metrics.update(ce=ce.detach(), factorization_gap=gap.detach(), loss=loss.detach())
    return loss, metrics


def loss_fn(model, table, n_context, lambda_fg):
    views = [model(x, y)[..., :k] for x, y, k in view_inputs(table, n_context)]
    return loss_from_views(views, table, n_context, lambda_fg)


def backward_table(model, table, penalty, divisor):
    """Exact two-pass VJP, retaining only one view's activations at a time.

    Obtain logits without a parameter graph; differentiate CE+FG with respect to
    those logits, then replay each view and backpropagate its logit gradient.
    Weights remain unchanged between passes and dropout must be zero. This keeps
    full gradients through both factorizations without retaining up to 22 graphs.
    """
    n_context = table["n_context"]
    with torch.no_grad():
        views = [model(x, y)[..., :k] for x, y, k in view_inputs(table, n_context)]
    views = [v.detach().requires_grad_() for v in views]
    loss, metrics = loss_from_views(views, table, n_context, penalty)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    gradients = torch.autograd.grad(loss / divisor, views)
    for (x, y, k), grad in zip(view_inputs(table, n_context), gradients):
        model(x, y)[..., :k].backward(grad)
    return metrics


def learning_rate(step, args):
    """One-based update index: linear warmup, then cosine ending at end_lr."""
    if args.warmup_steps and step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    progress = (step - args.warmup_steps - 1) / max(1, args.steps - args.warmup_steps - 1)
    return args.end_lr + (args.lr - args.end_lr) * (1 + math.cos(math.pi * progress)) / 2


def move_table(table, device):
    return {**table, "x": table["x"].to(device), "y": table["y"].to(device)}


@torch.no_grad()
def validate(model, tables, args, stop=None):
    # Training path returns raw logits; no inference ensemble/temperature/cache.
    # The released checkpoint has dropout=0, so this is deterministic.
    totals, grouped, counts = {}, {}, {}
    for table in tables:
        if stop is not None and stop.is_set():
            return None
        _, metrics = loss_fn(model, move_table(table, args.device), table["n_context"], 0.0)
        context = str(table["n_context"])
        counts[context] = counts.get(context, 0) + 1
        group = grouped.setdefault(context, {})
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) / len(tables)
            group[key] = group.get(key, 0.0) + float(value)
    totals["by_context"] = {c: {"tables": counts[c], **{k: v / counts[c] for k, v in group.items()}}
                            for c, group in grouped.items()}
    return totals


def atomic_save(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def emit(kind, **values):
    logging.getLogger("consistency").info(json.dumps({"kind": kind, **values}, allow_nan=False))


def configure_torch(args):
    torch.set_num_threads(args.cpu_threads)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    set_flash_attn3_enabled(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def save_checkpoint(directory, model, config, optimizer, step, args, penalty):
    """Atomic native classifier checkpoint plus optimizer and fixed-schedule state."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    latest = directory / "latest.ckpt"
    atomic_save({"config": config, "state_dict": state, "optimizer_state": optimizer.state_dict(),
                 "step": step, "format_version": FORMAT_VERSION,
                 "experiment": {**vars(args), "lambda_fg": penalty}}, latest)
    # Hard-link snapshots: no second serialization or duplicate disk blocks.
    snapshot = directory / f"step_{step:06d}.ckpt"
    temporary = snapshot.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    os.link(latest, temporary)
    temporary.replace(snapshot)
    for old in sorted(directory.glob("step_*.ckpt"))[:-args.keep_checkpoints]:
        old.unlink()
    emit("checkpoint", lambda_fg=penalty, step=step, path=str(latest))


def run_arm(args, penalty, stop):
    configure_torch(args)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    output = Path(args.output)
    manifest = json.loads((output / "experiment.json").read_text())
    config = manifest["model_config"]
    directory = output / f"lambda_{penalty:g}"
    directory.mkdir(exist_ok=True)
    latest = directory / "latest.ckpt"
    seed_all(args.seed)
    saved = torch.load(latest if latest.exists() else output / "initial.ckpt", map_location="cpu", weights_only=True)
    step = saved.get("step", 0)
    if latest.exists() and (saved.get("format_version") != FORMAT_VERSION or
                            saved["experiment"]["lambda_fg"] != penalty):
        raise ValueError("Incompatible arm checkpoint")
    model = TabICL(**config).float()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if "optimizer_state" in saved:
        optimizer.load_state_dict(saved["optimizer_state"])
    del saved
    emit("arm_start", lambda_fg=penalty, device=args.device, resume_step=step, pid=os.getpid())
    if step >= args.steps:
        emit("arm_already_complete", lambda_fg=penalty, step=step)
        return
    validation = torch.load(output / "validation.pt", weights_only=True)
    if not latest.exists():
        save_checkpoint(directory, model, config, optimizer, 0, args, penalty)
    metrics = validate(model, validation, args, stop)
    if metrics is not None:
        emit("validation", lambda_fg=penalty, step=step, **metrics)
    for update in range(step + 1, args.steps + 1):
        if stop.is_set():
            break
        started = time.monotonic()
        # Step-keyed RNG: replay after interruption uses the same stochastic stream.
        seed_all(args.seed + update)
        lr = learning_rate(update, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        tables = torch.load(output / "data" / f"step_{update:06d}.pt", weights_only=True)
        for table in tables:
            metrics = backward_table(model, move_table(table, args.device), penalty, len(tables))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) / len(tables)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        step = update
        emit("train", lambda_fg=penalty, step=step, lr=lr, grad_norm=float(norm),
             seconds=time.monotonic() - started,
             context_min=min(t["n_context"] for t in tables), context_max=max(t["n_context"] for t in tables),
             features_max=max(t["x"].shape[-1] for t in tables), **totals)
        if step % args.save_every == 0 or step == args.steps or stop.is_set():
            save_checkpoint(directory, model, config, optimizer, step, args, penalty)
        if not stop.is_set() and (step % args.validate_every == 0 or step == args.steps):
            metrics = validate(model, validation, args, stop)
            if metrics is not None:
                emit("validation", lambda_fg=penalty, step=step, **metrics)
    # Graceful SIGINT/SIGTERM finishes the current update before saving. A hard
    # kill resumes from the last atomic checkpoint; no partial gradients are saved.
    if stop.is_set():
        save_checkpoint(directory, model, config, optimizer, step, args, penalty)
    emit("arm_stopped" if stop.is_set() else "arm_complete", lambda_fg=penalty, step=step)


class LogStream:
    """Capture Python prints, warnings and tracebacks into the shared logging queue."""
    def __init__(self, logger, level):
        self.logger, self.level, self.pending = logger, level, ""

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line)
        return len(text)

    def flush(self):
        if self.pending.strip():
            self.logger.log(self.level, self.pending)
        self.pending = ""

    def isatty(self):
        return False


def worker(args, penalty, queue, stop):
    logger = logging.getLogger()
    logger.handlers = [QueueHandler(queue)]
    logger.setLevel(logging.INFO)
    stdout, stderr = LogStream(logger, logging.INFO), LogStream(logger, logging.WARNING)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            run_arm(args, penalty, stop)
        except BaseException:
            logger.exception("Arm failed: lambda=%s", penalty)
            stop.set()
            raise SystemExit(1)
        finally:
            stdout.flush()
            stderr.flush()


def prepare_data(args, stop):
    output = Path(args.output)
    (output / "data").mkdir(exist_ok=True)
    def table_seed(namespace, seed, index):
        return int(np.random.SeedSequence([namespace, seed, index]).generate_state(1)[0])
    # Each validation table is atomic, so initial data generation can resume too.
    validation_dir = output / "validation_data"
    validation_dir.mkdir(exist_ok=True)
    validation = []
    for i in range(args.validation_tables):
        if stop.is_set():
            return
        path = validation_dir / f"{i:04d}.pt"
        if not path.exists():
            atomic_save(sample_table(args, table_seed(1, args.validation_seed, i)), path)
        validation.append(torch.load(path, weights_only=True))
        emit("validation_data", tables=i + 1, total=args.validation_tables)
    atomic_save(validation, output / "validation.pt")
    for step in range(1, args.steps + 1):
        if stop.is_set():
            return
        path = output / "data" / f"step_{step:06d}.pt"
        if not path.exists():
            tables = [sample_table(args, table_seed(0, args.seed, (step - 1) * args.tables_per_step + i))
                      for i in range(args.tables_per_step)]
            atomic_save(tables, path)
        emit("shared_data", step=step, total=args.steps)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, help="Required for a new run; fixed for both arms and on resume")
    parser.add_argument("--checkpoint", help="Local pretrained classifier; default downloads pinned official v2")
    parser.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"], help="Devices for lambda=0 and 0.5")
    parser.add_argument("--output", default="runs/consistency_finetune")
    parser.add_argument("--log-dir", default=str(PROJECT_ROOT / "logs"))
    parser.add_argument("--resume", action="store_true", help="Restore saved settings and each arm's last complete update")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=1729)
    parser.add_argument("--context-rows", type=int, nargs="+", default=[128, 512, 2048, 8192], help="Uniformly sampled context buckets")
    parser.add_argument("--query-rows", type=int, default=128)
    parser.add_argument("--min-features", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--tables-per-step", type=int, default=16)
    parser.add_argument("--validation-tables", type=int, default=128)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=3, help="Retain latest N step snapshots per arm")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--end-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, help="Default: 5%% of total steps")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--no-recompute", action="store_true")
    return parser


def validate_args(args):
    if args.steps is None:
        raise ValueError("--steps is required for a new run")
    if args.warmup_steps is None:
        args.warmup_steps = args.steps // 20
    counts = [args.steps, *args.context_rows, args.query_rows, args.min_features, args.tables_per_step,
              args.validation_tables, args.validate_every, args.save_every, args.keep_checkpoints, args.cpu_threads]
    if min(counts) < 1 or not 0 <= args.warmup_steps < args.steps:
        raise ValueError("Counts must be positive and 0 <= warmup_steps < steps")
    if args.max_features < args.min_features or not 2 <= args.max_classes <= min(args.context_rows):
        raise ValueError("Invalid feature range or class count")
    if min(args.seed, args.validation_seed) < 0 or max(args.seed, args.validation_seed) + args.steps >= 2**32:
        raise ValueError("Seeds including step offsets must fit uint32")
    if not all(math.isfinite(v) for v in [args.lr, args.end_lr, args.weight_decay, args.clip_grad]) or not (
            0 < args.end_lr <= args.lr and args.weight_decay >= 0 and args.clip_grad > 0):
        raise ValueError("Require 0 < end_lr <= lr, nonnegative weight decay and positive clip_grad")
    devices = [torch.device(d) for d in args.devices]
    if any(d.type not in ("cpu", "cuda") for d in devices):
        raise ValueError("Use two CUDA devices (or cpu cpu for tests)")
    cuda_indices = [d.index if d.index is not None else 0 for d in devices if d.type == "cuda"]
    if len(set(cuda_indices)) != len(cuda_indices):
        raise ValueError("Parallel CUDA arms require two distinct GPUs")
    if cuda_indices and (not torch.cuda.is_available() or max(cuda_indices) >= torch.cuda.device_count()):
        raise ValueError("Requested CUDA devices are unavailable")


def initialize(args, saved, log_path):
    output = Path(args.output)
    if saved is not None:
        allowed = {"resume", "devices", "cpu_threads", "log_dir", "output", "checkpoint"}
        differences = [k for k, v in saved["settings"].items() if k not in allowed and getattr(args, k) != v]
        if differences:
            raise ValueError(f"Resume cannot change experiment settings: {differences}")
        if not (output / "initial.ckpt").exists():
            raise ValueError("Missing saved initial checkpoint")
        if args.checkpoint != saved["settings"]["checkpoint"]:
            digest = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
            if digest != saved["checkpoint_sha256"]:
                raise ValueError("Resume checkpoint differs from the original")
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use --resume or a new --output")
    output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is None:
        from huggingface_hub import hf_hub_download
        args.checkpoint = hf_hub_download(repo_id="jingang/TabICL", filename=CHECKPOINT)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = {**initial["config"], "recompute": not args.no_recompute}
    if config.get("max_classes", 10) < args.max_classes or config.get("dropout", 0) != 0:
        raise ValueError("Require a zero-dropout classifier with sufficient class capacity")
    atomic_save({"config": config, "state_dict": initial["state_dict"]}, output / "initial.ckpt")
    manifest = {"format_version": FORMAT_VERSION, "settings": vars(args), "log_path": str(log_path),
                "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "model_config": config, "penalties": [0.0, 0.5], "dtype": "float32",
                "prior": "graph_scm, joint two-target sampling; graph-level overlap filter"}
    temporary = output / "experiment.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "experiment.json")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(argv)
    args.output = str(Path(args.output).resolve())
    output = Path(args.output)
    saved = None
    resume_error = None
    if args.resume:
        try:
            saved = json.loads((output / "experiment.json").read_text())
            if saved.get("format_version") != FORMAT_VERSION:
                raise ValueError("This run uses an older, non-resumable format")
            provided = {a.split("=")[0] for a in argv if a.startswith("--")}
            for key, value in saved["settings"].items():
                if key not in ("resume", "output") and "--" + key.replace("_", "-") not in provided:
                    setattr(args, key, value)
        except Exception as error:
            resume_error = error
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:8]
    log_path = Path(saved["log_path"]) if saved and "log_path" in saved else Path(args.log_dir) / (
        f"{output.name}_seed{args.seed}_steps{args.steps}_lambda0-vs0.5_{suffix}.log")
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    queue, stop = ctx.Queue(), ctx.Event()
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(processName)s %(levelname)s %(message)s"))
    listener = QueueListener(queue, handler)
    logger = logging.getLogger()
    previous_handlers, previous_level = logger.handlers[:], logger.level
    logger.handlers, logger.level = [QueueHandler(queue)], logging.INFO
    listener.start()
    previous_signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in previous_signals:
        signal.signal(s, lambda *_: stop.set())
    stdout, stderr = LogStream(logger, logging.INFO), LogStream(logger, logging.WARNING)
    children = []
    status = 0
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            if resume_error is not None:
                raise resume_error
            validate_args(args)
            configure_torch(args)
            initialize(args, saved, log_path)
            emit("run_start", resume=args.resume, settings=vars(args), log_path=str(log_path))
            prepare_data(args, stop)
            if not stop.is_set():
                for penalty, device in zip((0.0, 0.5), args.devices):
                    arm_args = argparse.Namespace(**vars(args), device=device)
                    process = ctx.Process(target=worker, args=(arm_args, penalty, queue, stop), name=f"lambda_{penalty:g}")
                    process.start()
                    children.append(process)
                while any(p.is_alive() for p in children):
                    for p in children:
                        p.join(timeout=0.2)
                        if p.exitcode not in (None, 0):
                            stop.set()
                if any(p.exitcode != 0 for p in children):
                    raise RuntimeError("A training arm failed; inspect this log and resume after fixing the cause")
            status = 130 if stop.is_set() else 0
            emit("run_stopped" if stop.is_set() else "run_complete")
        except BaseException:
            logger.exception("Experiment failed")
            stop.set()
            status = 1
        finally:
            for p in children:
                p.join()
            stdout.flush()
            stderr.flush()
    for s, old in previous_signals.items():
        signal.signal(s, old)
    listener.stop()
    handler.close()
    queue.close()
    queue.join_thread()
    logger.handlers, logger.level = previous_handlers, previous_level
    return status


if __name__ == "__main__":
    raise SystemExit(main())
