"""Shared evaluation metrics for binary credit-risk classifiers.

All models (LR / RF / XGBoost / FT-Transformer) report the same metric suite
through `evaluate_model_metrics` and `report_performance`, so test-set
numbers are directly comparable across notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model_metrics(y_true, y_proba, label="Overall") -> pd.DataFrame:
    """Compute discrimination + top-decile metrics for a binary classifier.

    Metrics: AUC, Gini, KS statistic, AUCPR, Precision @ top-10%,
    Recall @ top-10%, plus Total / Total Bad / Bad Rate for context.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    total = len(y_true)
    total_bad = int(y_true.sum())
    bad_rate = total_bad / total

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = np.nan
    gini = 2 * auc - 1
    aucpr = average_precision_score(y_true, y_proba)

    proba_bad = y_proba[y_true == 1]
    proba_good = y_proba[y_true == 0]
    if len(proba_bad) > 0 and len(proba_good) > 0:
        ks_stat, _ = ks_2samp(proba_bad, proba_good)
    else:
        ks_stat = np.nan

    # Top-decile cutoff: 90th percentile of scores -> riskiest 10%
    cutoff = np.percentile(y_proba, 90)
    y_pred_top10 = (y_proba >= cutoff).astype(int)
    prec_top10 = precision_score(y_true, y_pred_top10)
    rec_top10 = recall_score(y_true, y_pred_top10)

    return pd.DataFrame(
        {
            "Segment": label,
            "Total": total,
            "Total Bad": total_bad,
            "Bad Rate": f"{bad_rate:.4f}",
            "AUC": auc,
            "Gini": gini,
            "KS Statistic": ks_stat,
            "AUCPR": aucpr,
            "Precision @10%": prec_top10,
            "Recall @10% (Capture Rate)": rec_top10,
        },
        index=["Value"],
    )


def report_performance(model, X_train, y_train, X_valid, y_valid, X_test, y_test) -> pd.DataFrame:
    """Score a fitted classifier on all three splits and concat results.

    Expects `model.predict_proba(X)[:, 1]` to return positive-class scores
    (sklearn / xgboost convention). For PyTorch models, wrap the model in a
    thin adapter that exposes `predict_proba` first.
    """
    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_valid)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    return pd.concat([
        evaluate_model_metrics(y_train, train_pred, label="Train"),
        evaluate_model_metrics(y_valid, val_pred, label="Validation"),
        evaluate_model_metrics(y_test, test_pred, label="Test"),
    ])


def report_performance_from_probas(probas_dict, targets_dict) -> pd.DataFrame:
    """Like `report_performance` but takes pre-computed probas and targets.

    Useful for models that don't expose `predict_proba` (e.g. raw PyTorch loops).
    Both dicts must have keys 'train', 'valid', 'test'.
    """
    return pd.concat([
        evaluate_model_metrics(targets_dict["train"], probas_dict["train"], label="Train"),
        evaluate_model_metrics(targets_dict["valid"], probas_dict["valid"], label="Validation"),
        evaluate_model_metrics(targets_dict["test"], probas_dict["test"], label="Test"),
    ])
