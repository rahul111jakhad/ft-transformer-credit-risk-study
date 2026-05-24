"""Dataset-specific loaders and cleaners.

Each function takes a raw DataFrame (or path) and returns a cleaned DataFrame
ready for `preprocessing.preprocess_data_pipeline`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# Lending Club
# =============================================================================

# Application-time features only (no post-origination repayment / hardship /
# settlement fields). Includes the target column `loan_status`.
LENDING_CLUB_COLS = [
    "id", "loan_amnt", "term", "int_rate", "emp_title", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "zip_code", "addr_state", "dti", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "initial_list_status",
    "collections_12_mths_ex_med", "application_type", "annual_inc_joint",
    "dti_joint", "verification_status_joint", "open_act_il", "all_util",
    "total_rev_hi_lim", "total_cu_tl", "bc_open_to_buy", "bc_util",
    "mort_acc", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    "revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
    "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog", "disbursement_method",
    "loan_status",
]

LENDING_CLUB_COMPLETED_STATUSES = [
    "Fully Paid",
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status: Charged Off",
]

LENDING_CLUB_TARGET_MAP = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Late (16-30 days)": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status: Charged Off": 1,
}


def load_lending_club(path: str) -> pd.DataFrame:
    """Read raw Lending Club CSV, keeping only application-time columns."""
    df = pd.read_csv(path, usecols=LENDING_CLUB_COLS)
    print(f"Raw shape: {df.shape}")
    return df


def preprocess_lending_club(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Lending-Club-specific cleaning and target definition.

    Returns a DataFrame with a `target_binary` column and no `loan_status`.
    """
    df = df[df["loan_status"].isin(LENDING_CLUB_COMPLETED_STATUSES)].reset_index(drop=True)
    df["target_binary"] = df["loan_status"].map(LENDING_CLUB_TARGET_MAP).astype(int)
    df = df.drop(columns=["loan_status"])

    # Strip percent signs / normalize string-encoded numerics. Object-dtype
    # detection has to be flexible — pandas may report 'object' or a string
    # subtype depending on how the column was loaded.
    if not pd.api.types.is_numeric_dtype(df["int_rate"]):
        df["int_rate"] = df["int_rate"].astype(str).str.rstrip("%").astype(float) / 100
    if not pd.api.types.is_numeric_dtype(df["revol_util"]):
        df["revol_util"] = df["revol_util"].astype(str).str.rstrip("%").astype(float)
    if not pd.api.types.is_numeric_dtype(df["emp_length"]):
        df["emp_length"] = (
            df["emp_length"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
        )

    # Drop identifiers / high-cardinality free text we don't use
    df = df.drop(columns=["id", "emp_title"])

    # Treat emp_length as categorical (year buckets), not continuous
    df["emp_length"] = df["emp_length"].astype("object")

    # Collapse rare ZIPs (< 500 occurrences) into 'Others'
    zip_counts = df["zip_code"].value_counts()
    rare_zips = zip_counts[zip_counts < 500].index
    df["zip_code"] = df["zip_code"].where(~df["zip_code"].isin(rare_zips), "Others")

    print(f"Shape after cleaning: {df.shape}")
    print(f"Target rate: {df['target_binary'].mean():.4f}")
    return df


# =============================================================================
# Home Credit
# =============================================================================


def preprocess_home_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Home-Credit-specific data integrity fixes.

    * Drop rows where CODE_GENDER == 'XNA'.
    * Replace sentinel 365243 with NaN in all DAYS_* columns.
    * Flip DAYS_BIRTH from negative to positive age-in-days.
    * Coerce FLAG* columns to clean binary 0/1.
    """
    df = df.copy()

    df = df[df["CODE_GENDER"] != "XNA"]

    days_cols = [c for c in df.columns if c.startswith("DAYS_")]
    for col in days_cols:
        df[col] = df[col].replace(365243, np.nan)

    if "DAYS_BIRTH" in df.columns:
        df["DAYS_BIRTH"] = -df["DAYS_BIRTH"]

    flag_cols = [c for c in df.columns if c.startswith("FLAG")]
    for col in flag_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    return df
