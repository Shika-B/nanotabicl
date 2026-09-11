"""Shared model construction and device selection for training and inference."""
from dataclasses import asdict

import torch

from .config import Config
from .model import NanoTabICLv2


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: Config) -> NanoTabICLv2:
    torch.set_float32_matmul_precision("highest")  # full float32 CUDA matmuls
    torch.backends.cudnn.allow_tf32 = False
    regression = cfg.data.task == "regression"
    return NanoTabICLv2(max_classes=0 if regression else cfg.data.max_classes,
                        out_dim=cfg.data.n_quantiles if regression else cfg.data.max_classes, **asdict(cfg.model)).float()
