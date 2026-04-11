"""
run_feature_importance.py — Feature importance analysis for section 5.4.

Runs three experiments on training data (2009-2012):
  1. RF impurity-based importance — full lifecycle feature set
  2. RF permutation importance — full lifecycle feature set
  3. LR absolute coefficient magnitudes — full lifecycle feature set
  4. RF impurity-based importance — PFR + restriction aggregates (best config)
  5. LR coefficients for age-only model (to show collapse behavior)

All trained on TRAIN_YEARS, reported for context only (not test evaluation).
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from config import TRAIN_YEARS, LABEL_COL
from data import load_dataset
from run_paper_ablation import (
    AGE_COLS, RESTRICTION_COLS, RESTRICTION_PFR_COLS,
    DEPRECATION_COLS, DEPRECATION_PFR_COLS, NO_EVENT_PFR_COLS, SHARED_COLS,
)

TOP_N = 25


def top_rf_impurity(model, feature_names, n=TOP_N):
    imp = pd.Series(model.feature_importances_, index=feature_names)
    return imp.nlargest(n).reset_index().rename(columns={"index": "feature", 0: "importance"})


def top_lr_coef(model, feature_names, n=TOP_N):
    coef = np.abs(model.coef_).ravel()
    imp = pd.Series(coef, index=feature_names)
    return imp.nlargest(n).reset_index().rename(columns={"index": "feature", 0: "abs_coef"})


def print_table(df, value_col, title):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"  {'Feature':<45} {value_col:>10}")
    print(f"  {'-'*57}")
    for _, row in df.iterrows():
        print(f"  {row['feature']:<45} {row[value_col]:>10.5f}")


def main():
    df, perm_cols, temporal_cols = load_dataset(verbose=False)

    def avail(cols):
        return [c for c in cols if c in df.columns]

    # Feature sets
    all_pfr      = avail(RESTRICTION_PFR_COLS + DEPRECATION_PFR_COLS + NO_EVENT_PFR_COLS)
    restrict_agg = avail(RESTRICTION_COLS)
    all_temporal = avail(
        AGE_COLS + RESTRICTION_COLS + RESTRICTION_PFR_COLS
        + DEPRECATION_COLS + DEPRECATION_PFR_COLS
        + NO_EVENT_PFR_COLS + SHARED_COLS
    )
    pfr_restrict_cols = perm_cols + all_pfr + restrict_agg  # best config

    train_mask = df["year"].isin(TRAIN_YEARS)
    X_full = df.loc[train_mask, perm_cols + all_temporal]
    y_train = df.loc[train_mask, LABEL_COL]
    X_pfr_restrict = df.loc[train_mask, pfr_restrict_cols]
    X_age_only = df.loc[train_mask, perm_cols + avail(AGE_COLS)]

    print(f"\nTraining years: {TRAIN_YEARS}")
    print(f"Training samples: {len(y_train)}")
    print(f"Full lifecycle features: {X_full.shape[1]}")
    print(f"PFR + Restrict Agg features: {X_pfr_restrict.shape[1]}")

    # ── 1. RF impurity importance — full lifecycle ─────────────────────────
    print("\nFitting RF (full lifecycle)...")
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42,
                                     n_jobs=-1, class_weight="balanced",
                                     max_features=0.5)
    rf_full.fit(X_full, y_train)
    imp_full = top_rf_impurity(rf_full, X_full.columns)
    print_table(imp_full, "importance", "RF Impurity Importance — Full Lifecycle (Top 25)")

    # ── 2. RF permutation importance — full lifecycle ──────────────────────
    print("\nRunning RF permutation importance (full lifecycle, n_repeats=10)...")
    pi = permutation_importance(rf_full, X_full, y_train,
                                n_repeats=10, random_state=42,
                                scoring="f1_macro", n_jobs=-1)
    pi_df = pd.DataFrame({
        "feature": X_full.columns,
        "importance": pi.importances_mean,
        "std": pi.importances_std,
    }).sort_values("importance", ascending=False).head(TOP_N).reset_index(drop=True)
    print(f"\n{'='*60}")
    print("RF Permutation Importance — Full Lifecycle (Top 25)")
    print(f"{'='*60}")
    print(f"  {'Feature':<45} {'mean':>8} {'std':>8}")
    print(f"  {'-'*63}")
    for _, row in pi_df.iterrows():
        print(f"  {row['feature']:<45} {row['importance']:>8.5f} {row['std']:>8.5f}")

    # ── 3. LR coefficients — full lifecycle ───────────────────────────────
    print("\nFitting LR (full lifecycle)...")
    lr_full = LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight="balanced", solver="saga")
    lr_full.fit(X_full, y_train)
    coef_full = top_lr_coef(lr_full, X_full.columns)
    print_table(coef_full, "abs_coef", "LR Absolute Coefficients — Full Lifecycle (Top 25)")

    # ── 4. RF impurity importance — PFR + restriction agg (best config) ───
    print("\nFitting RF (PFR + Restrict Agg)...")
    rf_pfr = RandomForestClassifier(n_estimators=100, random_state=42,
                                     n_jobs=-1, class_weight="balanced",
                                     max_features=0.5)
    rf_pfr.fit(X_pfr_restrict, y_train)
    imp_pfr = top_rf_impurity(rf_pfr, X_pfr_restrict.columns)
    print_table(imp_pfr, "importance", "RF Impurity Importance — PFR + Restrict Agg (Top 25)")

    # ── 5. LR coefficients — age-only (to show collapse behavior) ─────────
    print("\nFitting LR (static + age features only)...")
    lr_age = LogisticRegression(max_iter=2000, random_state=42,
                                 class_weight="balanced", solver="saga")
    lr_age.fit(X_age_only, y_train)
    coef_age = top_lr_coef(lr_age, X_age_only.columns)
    print_table(coef_age, "abs_coef", "LR Absolute Coefficients — Static + Age Features (Top 25)")

    # ── Feature group breakdown in RF importance ───────────────────────────
    print(f"\n{'='*60}")
    print("RF Impurity — Share of importance by feature group (Full Lifecycle)")
    print(f"{'='*60}")
    all_imp = pd.Series(rf_full.feature_importances_, index=X_full.columns)
    groups = {
        "Static flags":       perm_cols,
        "Age aggregates":     avail(AGE_COLS),
        "Restriction agg":    restrict_agg,
        "Deprecation agg":    avail(DEPRECATION_COLS),
        "Restriction PFR":    avail(RESTRICTION_PFR_COLS),
        "Deprecation PFR":    avail(DEPRECATION_PFR_COLS),
        "No-event PFR":       avail(NO_EVENT_PFR_COLS),
        "Shared cols":        avail(SHARED_COLS),
    }
    for group_name, cols in groups.items():
        present = [c for c in cols if c in all_imp.index]
        share = all_imp[present].sum() if present else 0.0
        print(f"  {group_name:<25}: {share:.4f}  ({len(present)} features)")

    # Save results
    results = {
        "rf_full_impurity_top25": imp_full.to_dict(orient="records"),
        "rf_pfr_restrict_impurity_top25": imp_pfr.to_dict(orient="records"),
        "lr_full_coef_top25": coef_full.to_dict(orient="records"),
        "lr_age_coef_top25": coef_age.to_dict(orient="records"),
        "rf_permutation_top25": pi_df[["feature", "importance", "std"]].to_dict(orient="records"),
    }
    with open("feature_importance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to feature_importance_results.json")


if __name__ == "__main__":
    main()
