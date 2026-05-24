"""Experiment tracking and persistence.

Every notebook in this repo writes its outputs through the helpers in this
module so that artifacts are uniformly named, timings are recorded, and the
test-set summary table grows monotonically across runs (append-with-dedupe).

Typical usage:

    from src.tracking import (
        ExperimentLogger, time_block, save_run_artifacts,
        capture_environment, measure_inference_time,
    )

    logger = ExperimentLogger(
        dataset="lending_club",
        artifacts_dir=ARTIFACTS_DIR,
        results_dir="../results",
    )
    capture_environment(logger)

    with time_block("xgb_tuned_train", logger):
        model.fit(X_train, y_train)

    save_run_artifacts(
        logger,
        model_name="xgb_tuned",
        perf_df=perf_xgb_tuned,
        best_params=study_xgb.best_params,
        study=study_xgb,
        test_predictions=model.predict_proba(X_test)[:, 1],
        y_test=y_test,
    )
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Logger
# =============================================================================

@dataclass
class ExperimentLogger:
    """Holds dataset name and output directories. Lightweight by design.

    Attributes:
        dataset: e.g. 'lending_club' or 'home_credit'. Used in row keys so
            the same model name in two datasets doesn't collide.
        artifacts_dir: per-dataset directory under artifacts/. Holds raw
            per-run outputs (large, gitignored).
        results_dir: shared results/ directory at the repo root. Holds
            paper-ready summary tables (committed to git).
    """

    dataset: str
    artifacts_dir: str
    results_dir: str

    def __post_init__(self):
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    # Convenience path helpers
    def artifact_path(self, *parts) -> str:
        return os.path.join(self.artifacts_dir, *parts)

    def result_path(self, *parts) -> str:
        return os.path.join(self.results_dir, *parts)


# =============================================================================
# Timing
# =============================================================================

@contextmanager
def time_block(block_name: str, logger: ExperimentLogger, model_name: str = ""):
    """Context manager that times a code block and appends to timings.csv.

    Args:
        block_name: short identifier (e.g. 'train', 'tuning_loop', 'inference_10k').
        logger: ExperimentLogger.
        model_name: optional model identifier ('lr_default', 'xgb_tuned', ...).
            If omitted, the block_name alone is used as the key.

    The timings CSV (`artifacts/<dataset>/timings.csv`) is append-only and
    has columns: timestamp, model_name, block, seconds.
    """
    start = time.time()
    print(f"  [timer] starting: {model_name or block_name}/{block_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        _append_timing(logger, model_name, block_name, elapsed)
        print(f"  [timer] {model_name or block_name}/{block_name}: {elapsed:.2f}s ({elapsed/60:.2f} min)")


def _append_timing(logger: ExperimentLogger, model_name: str, block: str, seconds: float):
    """Append one row to the per-dataset timings.csv."""
    path = logger.artifact_path("timings.csv")
    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "dataset": logger.dataset,
        "model_name": model_name,
        "block": block,
        "seconds": round(seconds, 4),
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", index=False, header=False)
    else:
        df_row.to_csv(path, index=False)


# =============================================================================
# Environment capture
# =============================================================================

def capture_environment(logger: ExperimentLogger, seed: int | None = None) -> dict:
    """Snapshot the runtime environment to `<artifacts_dir>/environment.json`.

    Captures Python / library versions, CUDA availability, GPU name. Safe to
    call even if torch / xgboost aren't installed (the entry is omitted).
    """
    env = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "seed": seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    # Library versions — soft imports so missing libs don't crash this
    for lib_name, import_name in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit_learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("optuna", "optuna"),
        ("torch", "torch"),
        ("rtdl_revisiting_models", "rtdl_revisiting_models"),
        ("shap", "shap"),
    ]:
        try:
            mod = __import__(import_name)
            env[f"{lib_name}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # GPU info via torch if available, else nvidia-smi parse
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            env["cuda_version"] = torch.version.cuda
    except ImportError:
        env["cuda_available"] = False

    path = logger.artifact_path("environment.json")
    with open(path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"  [env] captured -> {path}")
    return env


# =============================================================================
# Per-model artifact saving
# =============================================================================

def save_run_artifacts(
    logger: ExperimentLogger,
    model_name: str,
    perf_df: pd.DataFrame,
    *,
    best_params: dict | None = None,
    study=None,
    test_predictions: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Persist everything we'll later need to compare this model to others.

    Writes into <artifacts_dir>/:
        {model_name}_perf.csv             # the full train/valid/test metrics
        {model_name}_best_params.json     # tuned hyperparams (if provided)
        {model_name}_study.csv            # Optuna trial history (if provided)
        {model_name}_predictions_test.npy # test-set scores (if provided)
        y_test.npy                        # written once per dataset

    Also appends/updates the test-set row in
    <results_dir>/summary_test_metrics_{dataset}.csv.

    Args:
        model_name: snake_case identifier, e.g. 'xgb_tuned', 'lr_default'.
        perf_df: output of `report_performance`, must contain a 'Test' segment.
        best_params: dict of tuned hyperparameters.
        study: Optuna study; its trial dataframe is exported.
        test_predictions: 1-D array of test-set positive-class scores.
        y_test: matching test-set labels (saved once per dataset).
        extra_metadata: arbitrary additional fields for the summary row.

    Returns the summary row that was written.
    """
    paths_written = []

    # 1. Performance CSV
    perf_path = logger.artifact_path(f"{model_name}_perf.csv")
    perf_df.to_csv(perf_path, index=False)
    paths_written.append(perf_path)

    # 2. Best params
    if best_params is not None:
        params_path = logger.artifact_path(f"{model_name}_best_params.json")
        # Convert any numpy scalars in the dict (Optuna sometimes returns these)
        clean_params = {k: _to_json_safe(v) for k, v in best_params.items()}
        with open(params_path, "w") as f:
            json.dump(clean_params, f, indent=2)
        paths_written.append(params_path)

    # 3. Optuna study trial history
    if study is not None:
        try:
            study_df = study.trials_dataframe()
            study_path = logger.artifact_path(f"{model_name}_study.csv")
            study_df.to_csv(study_path, index=False)
            paths_written.append(study_path)
        except Exception as e:
            print(f"  [warn] could not export Optuna study: {e}")

    # 4. Predictions
    if test_predictions is not None:
        pred_path = logger.artifact_path(f"{model_name}_predictions_test.npy")
        np.save(pred_path, np.asarray(test_predictions))
        paths_written.append(pred_path)

    # 5. y_test (once per dataset)
    if y_test is not None:
        y_path = logger.artifact_path("y_test.npy")
        if not os.path.exists(y_path):
            np.save(y_path, np.asarray(y_test))
            paths_written.append(y_path)

    # 6. Append/update the per-dataset summary
    summary_row = append_to_summary(
        logger, model_name=model_name, perf_df=perf_df, extra_metadata=extra_metadata
    )

    print(f"  [save] {model_name}: wrote {len(paths_written)} artifacts")
    for p in paths_written:
        print(f"         - {p}")
    return summary_row


def append_to_summary(
    logger: ExperimentLogger,
    model_name: str,
    perf_df: pd.DataFrame,
    extra_metadata: dict | None = None,
) -> dict:
    """Append/update a row in results/summary_test_metrics_<dataset>.csv.

    Key for dedupe: model_name. Re-running the same model_name overwrites
    its row; new model_names are appended.
    """
    test_row = perf_df[perf_df["Segment"] == "Test"]
    if len(test_row) == 0:
        raise ValueError(f"perf_df has no 'Test' segment row")
    test_row = test_row.iloc[0]

    row = {
        "dataset": logger.dataset,
        "model_name": model_name,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "AUC": _to_json_safe(test_row["AUC"]),
        "Gini": _to_json_safe(test_row["Gini"]),
        "KS_Statistic": _to_json_safe(test_row["KS Statistic"]),
        "AUCPR": _to_json_safe(test_row["AUCPR"]),
        "Precision_top10": _to_json_safe(test_row["Precision @10%"]),
        "Recall_top10": _to_json_safe(test_row["Recall @10% (Capture Rate)"]),
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            row[k] = _to_json_safe(v)

    summary_path = logger.result_path(f"summary_test_metrics_{logger.dataset}.csv")

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        # Drop any existing row with this model_name, then append
        existing = existing[existing["model_name"] != model_name]
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(summary_path, index=False)
    return row


# =============================================================================
# Inference timing
# =============================================================================

def measure_inference_time(
    model,
    X,
    n_repeats: int = 5,
    logger: ExperimentLogger | None = None,
    model_name: str = "",
    sample_size: int = 10_000,
) -> dict:
    """Measure median inference latency for `sample_size` samples.

    Runs `model.predict_proba(X[:sample_size])` n_repeats times after one
    warmup pass, reports median and std. Logs to timings.csv as block
    'inference_{sample_size}'.

    Args:
        model: any model with `predict_proba` (sklearn convention).
        X: feature matrix; must have at least `sample_size` rows.
        n_repeats: number of measurement repeats.
        logger: if provided, the median is written to timings.csv.
        model_name: identifier used in the log.
        sample_size: number of rows to predict on.

    Returns a dict with keys: median_seconds, std_seconds, n_samples, n_repeats.
    """
    if len(X) < sample_size:
        sample_size = len(X)
        print(f"  [inference] X has only {len(X)} rows; using all of them")

    X_sample = X[:sample_size] if hasattr(X, "__getitem__") else X.iloc[:sample_size]

    # Warmup
    _ = model.predict_proba(X_sample)

    timings = []
    for _ in range(n_repeats):
        start = time.time()
        _ = model.predict_proba(X_sample)
        timings.append(time.time() - start)

    result = {
        "median_seconds": float(np.median(timings)),
        "std_seconds": float(np.std(timings)),
        "n_samples": int(sample_size),
        "n_repeats": int(n_repeats),
    }

    if logger is not None:
        _append_timing(logger, model_name, f"inference_{sample_size}", result["median_seconds"])

    print(
        f"  [inference] {model_name}: median={result['median_seconds']*1000:.2f}ms "
        f"std={result['std_seconds']*1000:.2f}ms (over {n_repeats} runs, {sample_size} samples)"
    )
    return result


# =============================================================================
# Helpers
# =============================================================================

def _to_json_safe(v):
    """Coerce numpy scalars/arrays to JSON-friendly Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def load_summary(results_dir: str, dataset: str) -> pd.DataFrame:
    """Reload the summary CSV for inspection."""
    path = os.path.join(results_dir, f"summary_test_metrics_{dataset}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_timings(artifacts_dir: str) -> pd.DataFrame:
    """Reload the timings CSV for inspection."""
    path = os.path.join(artifacts_dir, "timings.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
