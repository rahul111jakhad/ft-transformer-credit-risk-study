"""SHAP-based interpretation helpers for tree models.

These helpers are deliberately kept in their own module so that the
`shap` dependency is only required for notebooks that actually compute
feature attributions (the baselines), not for the FT-Transformer notebook
which uses attention maps instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_shap_importance_xgb(model, X_data):
    """SHAP global importance for an XGBoost classifier."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    feature_names = (
        X_data.columns
        if isinstance(X_data, pd.DataFrame)
        else [f"Feature {i}" for i in range(shap_values.shape[1])]
    )
    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP_Value_Contribution": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("SHAP_Value_Contribution", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df, shap_values


def get_shap_importance_rf(model, X_data):
    """SHAP global importance for a sklearn Random Forest classifier.

    Handles both the 'list of arrays' and the newer '3D ndarray' SHAP outputs,
    selecting the positive class.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_data)

    if isinstance(raw, list):
        shap_values = raw[-1]
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        shap_values = raw[:, :, 1]
    else:
        shap_values = raw

    importance_df = (
        pd.DataFrame(
            {
                "Feature": X_data.columns,
                "SHAP_Value_Contribution": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("SHAP_Value_Contribution", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df, shap_values


def plot_shap_summary(shap_values, X_data, max_features=15, save_prefix=None):
    """Plot SHAP bar and beeswarm summaries."""
    import matplotlib.pyplot as plt
    import shap

    n = min(max_features, X_data.shape[1])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, plot_type="bar", max_display=n, show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_bar.png", dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_data, max_display=n, show=False)
    plt.title("SHAP Feature Impact and Direction")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.show()
