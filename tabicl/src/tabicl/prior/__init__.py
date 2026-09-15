"""Synthetic prior data generation for TabICL pre-training.

Only :class:`PriorDataset` is public. All other symbols in this subpackage
are internal pre-training utilities and may change without notice. A CLI
entry point is provided via ``python -m tabicl.prior``.
"""

__all__ = ["PriorDataset"]


def __getattr__(name):
    # The graph-only fine-tuning runner does not need the legacy XGBoost prior.
    if name == "PriorDataset":
        from ._dataset import PriorDataset
        return PriorDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
