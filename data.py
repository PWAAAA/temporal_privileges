"""
data.py — Load, clean, and engineer features for the KronoDroid dataset.

Provides `load_dataset()` which returns:
  - df: experiment-ready DataFrame (whitelisted columns, risk scores applied)
  - permission_flag_cols: list of static permission flag column names
  - temporal_cols: list of temporal feature column names
"""
import numpy as np
import pandas as pd

from config import (CSV_FILE, LIFECYCLE_FILE, LABEL_COL, DATE_COL,
                    MIN_YEAR, DROP_COLS, LIFECYCLE_YEAR_COLS,
                    TEMPORAL_COLS, TTL_SENTINEL,
                    HIGH_DRIFT_FLAGS, PER_FLAG_RISK_COLS)
from features import ttl_to_risk, lifecycle_features, per_flag_risk_columns


def load_dataset(verbose=True):
    """Load and return (df, permission_flag_cols, temporal_cols)."""
    if verbose:
        print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)
    df["year"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.year
    df = df.drop(columns=DROP_COLS + [DATE_COL], errors="ignore")
    df = df[df["year"] >= MIN_YEAR].dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    if verbose:
        print("App counts per year:\n", df["year"].value_counts().sort_index(), "\n")

    # ── Load lifecycle table ───────────────────────────────────────────────
    lifecycle_df = pd.read_csv(LIFECYCLE_FILE)
    for col in LIFECYCLE_YEAR_COLS:
        if col in lifecycle_df.columns:
            lifecycle_df[col] = pd.to_numeric(lifecycle_df[col], errors="coerce")

    lifecycle_perm_cols = [c for c in df.columns if c in lifecycle_df["permission"].values]
    if verbose:
        print(f"Permissions matched to lifecycle table: {len(lifecycle_perm_cols)}/{len(lifecycle_df)}")

    # ── Compute lifecycle features ─────────────────────────────────────────
    if verbose:
        print("Computing lifecycle features (may take ~1 min)...")
    lifecycle_feats = df.apply(
        lambda row: lifecycle_features(row, lifecycle_df, lifecycle_perm_cols), axis=1
    )
    df = pd.concat([df, lifecycle_feats], axis=1).copy()

    # ── TTL → risk scores ──────────────────────────────────────────────────
    raw_ttl_cols = [c for c in df.columns if c.startswith("perm_") and "ttl" in c]
    df[raw_ttl_cols] = df[raw_ttl_cols].fillna(TTL_SENTINEL)
    df["perm_worst_perm_age_at_restrict"] = df["perm_worst_perm_age_at_restrict"].fillna(0)

    ttl_to_risk_col_rename = {
        col: col.replace("perm_ttl_", "perm_risk_").replace("perm_worst_ttl_", "perm_risk_worst_")
        for col in raw_ttl_cols
    }
    for ttl_col, risk_col in ttl_to_risk_col_rename.items():
        df[risk_col] = ttl_to_risk(df[ttl_col])
    df = df.drop(columns=list(ttl_to_risk_col_rename))

    # ── Interaction features ───────────────────────────────────────────────
    df["perm_age_x_near_restrict"]       = df["perm_worst_age"] * df["perm_near_restrict_ratio"]
    df["perm_worst_age_x_risk_restrict"] = df["perm_worst_perm_age_at_restrict"] * df["perm_risk_worst_restrict"]

    # ── Per-flag temporal risk encoding ────────────────────────────────────
    drift_flags_in_data = [f for f in HIGH_DRIFT_FLAGS if f in df.columns]
    if drift_flags_in_data:
        pfr_df = per_flag_risk_columns(df, lifecycle_df, drift_flags_in_data)
        df = pd.concat([df, pfr_df], axis=1)
        if verbose:
            print(f"\nPer-flag risk columns computed: {len(pfr_df.columns)} "
                  f"(from {len(drift_flags_in_data)} high-drift flags)")

    # ── Whitelist columns ──────────────────────────────────────────────────
    permission_flag_cols = [c for c in df.columns if c.isupper()]
    pfr_cols             = [c for c in df.columns if c.startswith("pfr_")]
    temporal_cols        = [c for c in TEMPORAL_COLS if c in df.columns] + pfr_cols
    df = df[[c for c in permission_flag_cols + temporal_cols + [LABEL_COL, "year"] if c in df.columns]]

    if verbose:
        print(f"\nColumns after whitelist: {len(df.columns)} "
              f"({len(permission_flag_cols)} permission flags, {len(temporal_cols)} temporal)\n")
        print("Lifecycle feature coverage (% non-zero rows):")
        for col in temporal_cols:
            print(f"  {col:<45} {(df[col] != 0).mean():.2%}")

    return df, permission_flag_cols, temporal_cols
