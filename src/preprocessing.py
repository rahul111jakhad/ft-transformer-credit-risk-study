"""Leakage-free preprocessing pipeline shared across all notebooks.

The pipeline:
    1. Stratified 64/16/20 split.
    2. Drop columns whose train missing-rate exceeds `missing_threshold`.
    3. Drop numeric features highly correlated with another, keeping the one
       more correlated with the target.
    4. Median-impute numerics, label-encode categoricals (training-fit, then
       extended to unseen categories in valid/test), standard-scale numerics.

For FT-Transformer use, set `keep_split=True` to receive numerical and
categorical arrays separately along with the categorical cardinalities. The
default (`keep_split=False`) concatenates everything for sklearn / XGBoost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def seed_everything(seed: int = 42) -> None:
    """Seed Python and NumPy. Torch is seeded separately where used."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class PreprocessedData:
    """Container for outputs of `preprocess_data_pipeline`.

    For tree / linear models, call `.as_concatenated()` to get DataFrame-style
    splits where categorical and numerical features sit in one frame.

    For FT-Transformer, use `.x_num_*` and `.x_cat_*` arrays directly along
    with `cat_cardinalities`.
    """

    # Numerical arrays
    x_num_train: pd.DataFrame
    x_num_valid: pd.DataFrame
    x_num_test: pd.DataFrame

    # Categorical arrays (label-encoded, integer-valued)
    x_cat_train: pd.DataFrame
    x_cat_valid: pd.DataFrame
    x_cat_test: pd.DataFrame

    # Targets
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series

    # Metadata
    num_cols: list = field(default_factory=list)
    cat_cols: list = field(default_factory=list)
    cat_cardinalities: list = field(default_factory=list)
    encoders: dict = field(default_factory=dict)
    median_imputers: dict = field(default_factory=dict)
    scaler: StandardScaler | None = None

    def as_concatenated(self):
        """Return train / valid / test as concatenated (num + cat) DataFrames."""
        X_train = pd.concat([self.x_num_train, self.x_cat_train], axis=1)
        X_valid = pd.concat([self.x_num_valid, self.x_cat_valid], axis=1)
        X_test = pd.concat([self.x_num_test, self.x_cat_test], axis=1)
        return X_train, X_valid, X_test


def apply_missing_value_filter(X_train, X_valid, X_test, threshold=0.50, verbose=True):
    """Drop columns whose missing rate in X_train exceeds `threshold`."""
    missing_pct = X_train.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

    if verbose:
        print(f"  Threshold: {threshold:.0%} | Dropped {len(cols_to_drop)} columns")

    X_train = X_train.drop(columns=cols_to_drop, errors="ignore")
    X_valid = X_valid.drop(columns=cols_to_drop, errors="ignore")
    X_test = X_test.drop(columns=cols_to_drop, errors="ignore")

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    return X_train, X_valid, X_test, num_cols, cat_cols


def apply_correlation_drop(X_train, y_train, X_valid, X_test, threshold=0.9, verbose=True):
    """Drop highly correlated numeric features, keeping the one more correlated with the target."""
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    X_train_num = X_train[num_cols]
    corr_matrix = X_train_num.corr().abs()
    target_corr = X_train_num.apply(lambda s: s.corr(y_train)).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:
                if row in to_drop or col in to_drop:
                    continue
                weaker = row if target_corr.get(row, 0) < target_corr.get(col, 0) else col
                to_drop.add(weaker)

    if verbose:
        print(f"  Numeric features before: {len(num_cols)} | Dropped: {len(to_drop)}")

    X_train = X_train.drop(columns=list(to_drop), errors="ignore")
    X_valid = X_valid.drop(columns=list(to_drop), errors="ignore")
    X_test = X_test.drop(columns=list(to_drop), errors="ignore")

    num_cols = [c for c in num_cols if c not in to_drop]
    return X_train, X_valid, X_test, num_cols, cat_cols


def preprocess_data_pipeline(
    df: pd.DataFrame,
    target: str,
    corr_threshold: float = 0.9,
    missing_threshold: float = 0.50,
    random_state: int = 42,
    verbose: bool = True,
) -> PreprocessedData:
    """Full preprocessing pipeline: split -> filter -> encode/scale.

    Args:
        df: Input DataFrame with the target column included.
        target: Name of the binary target column.
        corr_threshold: Correlation above which numeric features are dropped.
        missing_threshold: Training missing-rate above which columns are dropped.
        random_state: For the stratified split.
        verbose: Print progress messages.

    Returns:
        PreprocessedData with separate numerical and categorical splits, plus
        metadata (cat_cardinalities, encoders, imputers, scaler).
    """

    def log(msg):
        if verbose:
            print(msg)

    log("Step 1: Stratified split (64/16/20)")
    X = df.drop(columns=[target])
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=random_state
    )
    log(f"  Train={len(X_train):,} | Valid={len(X_valid):,} | Test={len(X_test):,}")

    log("Step 2: Missing-value column filter")
    X_train, X_valid, X_test, num_cols, cat_cols = apply_missing_value_filter(
        X_train, X_valid, X_test, threshold=missing_threshold, verbose=verbose
    )

    log("Step 3: Correlation drop (numeric)")
    X_train, X_valid, X_test, num_cols, cat_cols = apply_correlation_drop(
        X_train, y_train, X_valid, X_test, threshold=corr_threshold, verbose=verbose
    )

    log("Step 4: Median imputation (numeric)")
    median_imputers = {}
    for col in num_cols:
        med = X_train[col].median()
        median_imputers[col] = med
        X_train[col] = X_train[col].fillna(med)
        X_valid[col] = X_valid[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    log("Step 5: Label encoding (categorical)")
    encoders = {}
    cat_cardinalities = []
    for col in cat_cols:
        for split in (X_train, X_valid, X_test):
            split[col] = split[col].fillna("Missing")

        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

        # Extend the encoder with any unseen categories from valid/test so the
        # transform on those splits never raises.
        seen = list(le.classes_)
        for split in (X_valid, X_test):
            for cat in split[col].astype(str).unique():
                if cat not in seen:
                    seen.append(cat)
        le.classes_ = np.array(seen)

        X_valid[col] = le.transform(X_valid[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

        encoders[col] = le
        # Use the encoder's full vocabulary, NOT X_train.nunique() — the
        # encoder is extended with unseen valid/test categories above, so its
        # vocabulary size is what FT-Transformer's embedding layer must match.
        # Sizing embeddings to the train-only count would cause out-of-bounds
        # indexing on unseen valid/test categories at inference time.
        cat_cardinalities.append(len(le.classes_))

    log("Step 6: Standard scaling (numeric)")
    scaler = StandardScaler()
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_valid[num_cols] = scaler.transform(X_valid[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    log(
        f"\nFinal feature counts: numeric={len(num_cols)} | "
        f"categorical={len(cat_cols)} | total={X_train.shape[1]}"
    )
    log(f"Categorical cardinalities: {cat_cardinalities}")

    return PreprocessedData(
        x_num_train=X_train[num_cols].reset_index(drop=True),
        x_num_valid=X_valid[num_cols].reset_index(drop=True),
        x_num_test=X_test[num_cols].reset_index(drop=True),
        x_cat_train=X_train[cat_cols].reset_index(drop=True),
        x_cat_valid=X_valid[cat_cols].reset_index(drop=True),
        x_cat_test=X_test[cat_cols].reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_cardinalities=cat_cardinalities,
        encoders=encoders,
        median_imputers=median_imputers,
        scaler=scaler,
    )
