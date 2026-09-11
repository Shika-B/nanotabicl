"""Float32 training with fixed validation tables, plateau LR reductions and early stopping."""
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .config import Config, OptimConfig, load_config
from .data import PriorDataset, iter_batches
from .model import NanoTabICLv2
from .muon import Muon
from .runtime import build_model, resolve_device


def cross_entropy_loss(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    # pred: (n_batch, n_test, max_classes), y: (n_batch, n_test) class labels stored as floats
    loss = F.cross_entropy(pred.flatten(0, 1), y.flatten().long())
    return loss, {"accuracy": (pred.argmax(dim=-1) == y).float().mean()}

def factorization_gap(p_a: torch.Tensor, p_b: torch.Tensor,
                      p_b_given_a: torch.Tensor, p_a_given_b: torch.Tensor) -> torch.Tensor:
    """Mean total variation between the A->B and B->A joint predictions.

    Inputs are probabilities, normalized over their last axis (not logits):
      p_a: (..., K_A), p_b: (..., K_B)
      p_b_given_a: (..., K_A, K_B), p_a_given_b: (..., K_B, K_A).
    Leading dimensions must match, typically (batch, test_rows). Conditionals
    must include every conditioning class, not just the observed target value.
    Mask unused output classes before softmax when preparing probabilities.

    Returns a scalar: mean over examples of 0.5 * sum_{a,b} |p(a)p(b|a)-p(b)p(a|b)|.
    Gradients flow through all four inputs.
    """
    if p_a.ndim < 1 or p_b.ndim < 1 or p_a.shape[:-1] != p_b.shape[:-1]:
        raise ValueError("Marginals must have matching leading dimensions and a class axis.")
    joint_shape = (*p_a.shape, p_b.shape[-1])
    reverse_shape = (*p_b.shape, p_a.shape[-1])
    if p_b_given_a.shape != joint_shape or p_a_given_b.shape != reverse_shape:
        raise ValueError("Expected conditionals shaped (..., K_A, K_B) and (..., K_B, K_A).")
    joint_ab = p_a.unsqueeze(-1) * p_b_given_a
    joint_ba = (p_b.unsqueeze(-1) * p_a_given_b).transpose(-1, -2)
    return 0.5 * (joint_ab - joint_ba).abs().sum(dim=(-2, -1)).mean()


def pinball_loss(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    # pred: (n_batch, n_test, n_quantiles) at levels linspace(0, 1, n_quantiles + 2)[1:-1], y: (n_batch, n_test)
    alphas = torch.linspace(0, 1, pred.shape[-1] + 2, device=pred.device)[1:-1]
    errors = y[..., None] - pred
    loss = torch.maximum(alphas * errors, (alphas - 1) * errors).mean()
    return loss, {"median_mse": (pred.detach()[..., pred.shape[-1] // 2] - y).square().mean()}


def conditional_logits(model: NanoTabICLv2, x: torch.Tensor, target: torch.Tensor,
                       condition: torch.Tensor, n_train: int, n_classes: int) -> torch.Tensor:
    """Predict the target for every possible value of the conditioning target.

    For P(B | A, X), pass target=B and condition=A. Let M be the batch size,
    N the total rows, T the context rows, and F the original feature count.

    Args:
        model: Predictor returning logits shaped (M, N - T, K_B).
        x: Features shaped (M, N, F), with context rows first.
        target: B labels shaped (M, N); only target[:, :T] enters the model.
        condition: A labels shaped (M, N); only condition[:, :T] is used.
        n_train: Number T of context rows.
        n_classes: Number K_A of conditioning class values to enumerate.

    For each a in range(K_A), build a column shaped (M, N) containing observed
    A labels in context rows and the constant a in every query row. Append it
    to x to obtain (M, N, F + 1), then predict using context B labels (M, T).
    Each forward returns (M, N - T, K_B); stack these along the A-value axis.
    The predicted target is never appended as a feature.

    Returns:
        Raw logits shaped (M, N - T, K_A, K_B). Entry [m, i, a, b] is the
        logit for B=b at query row i of table m, assuming A=a. softmax(-1)
        normalizes over B classes. The current caller uses the configured
        max_classes capacity for both axes (inferred from model output) and
        masks unsupported output classes in loss_fn before softmax.
    """
    predictions = []
    for value in range(n_classes):
        column = torch.cat([condition[:, :n_train], torch.full_like(condition[:, n_train:], value)], dim=1)
        predictions.append(model(torch.cat([x, column.unsqueeze(-1)], dim=-1), target[:, :n_train]))
    return torch.stack(predictions, dim=-2)


def loss_fn(model: NanoTabICLv2, x: torch.Tensor, y: torch.Tensor, n_train: int,
            task: str = "classification", lambda_fg: float = 0.0,
            compute_gap: bool = True) -> tuple[torch.Tensor, dict]:
    """Single-target loss, or mean four-view CE + lambda_fg * factorization gap.

    compute_gap=False skips enumeration when lambda_fg=0, using four direct
    observed-label views with identical supervision. A nonzero penalty always
    requires enumeration.
    Class support comes exclusively from context rows, including noncontiguous
    labels. Conditional CE selects the observed other label from the enumeration.
    """
    if not math.isfinite(lambda_fg) or lambda_fg < 0:
        raise ValueError("lambda_fg must be finite and nonnegative")
    if y.ndim == 2:
        if lambda_fg != 0:
            raise ValueError("lambda_fg requires two classification targets")
        pred = model(x, y[:, :n_train])
        return (pinball_loss if task == "regression" else cross_entropy_loss)(pred, y[:, n_train:])
    if task != "classification" or y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError("Multi-target training requires exactly two classification targets")

    a, b = y.unbind(-1)
    logits_a, logits_b = model(x, a[:, :n_train]), model(x, b[:, :n_train])
    n_classes = logits_a.shape[-1]
    support_a = F.one_hot(a[:, :n_train].long(), n_classes).bool().any(dim=1)
    support_b = F.one_hot(b[:, :n_train].long(), n_classes).bool().any(dim=1)
    logits_a = logits_a.masked_fill(~support_a[:, None, :], -torch.inf)
    logits_b = logits_b.masked_fill(~support_b[:, None, :], -torch.inf)
    a_test, b_test = a[:, n_train:], b[:, n_train:]
    if compute_gap or lambda_fg != 0:
        logits_b_given_a = conditional_logits(model, x, b, a, n_train, n_classes)
        logits_a_given_b = conditional_logits(model, x, a, b, n_train, n_classes)
        logits_b_given_a = logits_b_given_a.masked_fill(~support_b[:, None, None, :], -torch.inf)
        logits_a_given_b = logits_a_given_b.masked_fill(~support_a[:, None, None, :], -torch.inf)
        observed_b_given_a = logits_b_given_a.gather(
            -2, a_test.long()[..., None, None].expand(-1, -1, 1, n_classes)).squeeze(-2)
        observed_a_given_b = logits_a_given_b.gather(
            -2, b_test.long()[..., None, None].expand(-1, -1, 1, n_classes)).squeeze(-2)
    else:
        observed_b_given_a = model(torch.cat([x, a[..., None]], -1), b[:, :n_train])
        observed_a_given_b = model(torch.cat([x, b[..., None]], -1), a[:, :n_train])
        observed_b_given_a = observed_b_given_a.masked_fill(~support_b[:, None, :], -torch.inf)
        observed_a_given_b = observed_a_given_b.masked_fill(~support_a[:, None, :], -torch.inf)
    views = [("a", logits_a, a_test), ("b", logits_b, b_test),
             ("a_given_b", observed_a_given_b, a_test), ("b_given_a", observed_b_given_a, b_test)]
    losses, metrics = [], {}
    for name, logits, labels in views:
        ce, extra = cross_entropy_loss(logits, labels)
        losses.append(ce)
        metrics[f"ce_{name}"] = ce.detach()
        metrics[f"accuracy_{name}"] = extra["accuracy"]
    ce = torch.stack(losses).mean()
    metrics.update(ce=ce.detach())
    gap = ce.new_zeros(())
    if compute_gap or lambda_fg != 0:
        gap = factorization_gap(logits_a.softmax(-1), logits_b.softmax(-1),
                                logits_b_given_a.softmax(-1), logits_a_given_b.softmax(-1))
        metrics["factorization_gap"] = gap.detach()
    return ce + lambda_fg * gap, metrics


def lr_schedule(step: int, cfg: OptimConfig, plateau_lr: float | None = None) -> float:
    """Fixed-length warmup, then hold the validation-controlled learning rate."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr if plateau_lr is None else plateau_lr


def validation_tables(cfg: Config):
    """Persist a fixed independently seeded validation corpus without advancing training RNGs."""
    path = os.path.join(cfg.out_dir, "validation.pt")
    data_cfg = replace(cfg.data, micro_batch_size=1, num_workers=0)
    signature = {"data": asdict(data_cfg), "seed": cfg.validation.seed, "n_tables": cfg.validation.n_tables}
    if os.path.exists(path):
        saved = torch.load(path, map_location="cpu")
        if saved["signature"] != signature:
            raise ValueError("Validation configuration changed; use a new out_dir to keep comparisons consistent")
        return saved["tables"]
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(cfg.validation.seed)
            np.random.seed(cfg.validation.seed)
            prior = PriorDataset(data_cfg)
            tables = [prior.sample_batch() for _ in range(cfg.validation.n_tables)]
    finally:
        np.random.set_state(numpy_state)
    torch.save({"signature": signature, "tables": tables}, path)
    return tables


@torch.no_grad()
def validate(model, tables, task, device):
    """Equal-weight mean over tables, each scored on its held-out rows; penalty excluded."""
    was_training = model.training
    model.eval()
    totals = {}
    try:
        for x, y, n_train in tables:
            loss, extra = loss_fn(model, x.to(device), y.to(device), n_train, task, compute_gap=False)
            for name, value in {"loss": loss, **extra}.items():
                totals[name] = totals.get(name, 0.0) + value.item() / len(tables)
    finally:
        model.train(was_training)
    if not all(math.isfinite(value) for value in totals.values()):
        raise RuntimeError("Nonfinite validation metrics")
    return totals


def update_convergence(state, value, cfg):
    """Reduce LR after patience checks; stop after another plateau at min_lr."""
    improved = value < state["best"]
    state["best"] = min(value, state["best"])
    if value < state["reference"] - cfg.validation.min_delta:
        state["reference"], state["bad_checks"] = value, 0
    else:
        state["bad_checks"] += 1
    if state["bad_checks"] >= cfg.validation.patience:
        if state["lr"] <= cfg.optim.min_lr:
            state["stopped"] = True
        else:
            state["lr"] = max(cfg.optim.min_lr, state["lr"] * cfg.optim.lr_factor)
            state["bad_checks"] = 0
    return improved


def train(cfg: Config) -> NanoTabICLv2:
    if cfg.data.n_targets not in (1, 2) or (cfg.data.n_targets == 2 and cfg.data.task != "classification"):
        raise ValueError("Training supports one target, or two classification targets")
    if not math.isfinite(cfg.optim.lambda_fg) or cfg.optim.lambda_fg < 0:
        raise ValueError("lambda_fg must be finite and nonnegative")
    if cfg.optim.lambda_fg != 0 and cfg.data.n_targets != 2:
        raise ValueError("lambda_fg requires two classification targets")
    if min(cfg.validation.n_tables, cfg.validation.every, cfg.validation.patience, cfg.optim.max_steps,
           cfg.optim.accum_steps, cfg.save_every, cfg.log_every) < 1:
        raise ValueError("Table counts, patience, step limits and intervals must be positive")
    if not (0 < cfg.optim.min_lr <= cfg.optim.lr and 0 < cfg.optim.lr_factor < 1):
        raise ValueError("Require 0 < min_lr <= lr and 0 < lr_factor < 1")
    if cfg.optim.warmup_steps < 0 or not (0 <= cfg.validation.min_delta < float("inf")):
        raise ValueError("warmup_steps and finite min_delta must be nonnegative")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), os.path.join(cfg.out_dir, "config.yaml"))

    model = build_model(cfg).to(device)
    optimizer = Muon(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
                     momentum=cfg.optim.momentum, matched_adamw_rms=cfg.optim.matched_adamw_rms)
    step = 0
    convergence = {"best": float("inf"), "reference": float("inf"), "bad_checks": 0,
                   "lr": cfg.optim.lr, "stopped": False}

    ckpt_path = os.path.join(cfg.out_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        convergence.update(ckpt.get("convergence", {}))
        print(f"Resuming from {ckpt_path} at step {step}")
    elif cfg.init_from:  # start a new run (e.g. the next curriculum stage) from the weights of another checkpoint
        model.load_state_dict(torch.load(cfg.init_from, map_location=device)["model"])
        print(f"Initialized model weights from {cfg.init_from}")

    print(f"Preparing {cfg.validation.n_tables} fixed validation tables (seed={cfg.validation.seed})", flush=True)
    tables = validation_tables(cfg)
    loader = iter_batches(cfg.data, seed=cfg.seed + step)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, device={device}, "
          f"batch size={cfg.optim.accum_steps * cfg.data.micro_batch_size}", flush=True)

    if cfg.wandb_project:
        import wandb
        wandb.init(project=cfg.wandb_project, config=asdict(cfg), dir=cfg.out_dir, resume="allow")

    model.train()
    best_path = os.path.join(cfg.out_dir, "best.pt")
    while step < cfg.optim.max_steps and not convergence["stopped"]:
        t_start, t_data, metrics = time.time(), 0.0, {}
        lr = optimizer.param_groups[0]["lr"] = lr_schedule(step, cfg.optim, convergence["lr"])
        for _ in range(cfg.optim.accum_steps):
            t0 = time.time()
            x, y, n_train = next(loader)
            t_data += time.time() - t0
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss, extra = loss_fn(model, x, y, n_train, cfg.data.task, cfg.optim.lambda_fg, compute_gap=False)
            (loss / cfg.optim.accum_steps).backward()
            for name, value in {"loss": loss.detach(), **extra}.items():  # stays on device, no sync per micro-batch
                metrics[name] = metrics.get(name, 0.0) + value / cfg.optim.accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip or float("inf"))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        checked = step % cfg.validation.every == 0 or step == cfg.optim.max_steps
        improved = False
        if checked:
            validation = validate(model, tables, cfg.data.task, device)
            if step >= cfg.optim.warmup_steps:
                improved = update_convergence(convergence, validation["loss"], cfg)
            else:  # keep the best warmup checkpoint without consuming plateau patience
                improved = validation["loss"] < convergence["best"]
                convergence["best"] = min(convergence["best"], validation["loss"])
            record = {"step": step, "lr": lr, "next_lr": convergence["lr"],
                      "best_loss": convergence["best"], "bad_checks": convergence["bad_checks"],
                      "stopped": convergence["stopped"], **validation}
            with open(os.path.join(cfg.out_dir, "validation.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"validation step={step} loss={validation['loss']:.6g} best={convergence['best']:.6g} "
                  f"bad_checks={convergence['bad_checks']} next_lr={convergence['lr']:.4g}", flush=True)
            if cfg.wandb_project:
                wandb.log({"step": step, **{f"val/{k}": v for k, v in record.items() if k != "step"}})

        if step % cfg.log_every == 0 or step == cfg.optim.max_steps or convergence["stopped"]:
            metrics = {name: value.item() for name, value in metrics.items()}
            metrics.update(step=step, lr=lr, grad_norm=grad_norm.item(),
                           step_time=time.time() - t_start, data_wait=t_data)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()), flush=True)
            with open(os.path.join(cfg.out_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(metrics) + "\n")
            if cfg.wandb_project:
                wandb.log(metrics, step=step)
        if checked or step % cfg.save_every == 0:
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                          "config": asdict(cfg), "convergence": convergence}
            if os.path.exists(ckpt_path):
                os.replace(ckpt_path, os.path.join(cfg.out_dir, "previous.pt"))
            torch.save(checkpoint, ckpt_path)
            if improved:
                torch.save(checkpoint, best_path)
    reason = "validation plateau at minimum learning rate" if convergence["stopped"] else "max_steps safety budget"
    print(f"Stopped: {reason}; best validation loss={convergence['best']:.6g}", flush=True)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    return model


if __name__ == "__main__":  # usage: python -m nanotabicl.train [config.yaml ...] [key.subkey=value ...]
    yaml_paths = [arg for arg in sys.argv[1:] if "=" not in arg]
    train(load_config(yaml_paths, [arg for arg in sys.argv[1:] if "=" in arg]))
