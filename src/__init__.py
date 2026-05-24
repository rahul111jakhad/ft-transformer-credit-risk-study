"""Shared utilities for the credit-risk modeling experiments.

This package keeps preprocessing, evaluation, interpretation, tracking, and
FT-Transformer infrastructure in one place so that the baseline and
FT-Transformer notebooks for both datasets produce results from exactly the
same code path.

Note: `ft_transformer_utils` is not imported by default because it requires
torch + rtdl_revisiting_models. Import it explicitly when needed:

    from src.ft_transformer_utils import build_ft_transformer, train_ft_transformer
"""

from . import datasets, evaluation, interpretation, preprocessing, tracking

__all__ = ["datasets", "evaluation", "interpretation", "preprocessing", "tracking"]
