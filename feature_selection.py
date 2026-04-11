"""
feature_selection.py — Permutation importance pruning and feature set construction.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from config import TRAIN_YEARS, LABEL_COL


def build_feature_sets(df, permission_flag_cols, temporal_cols, prune=True):
    """Return FEATURE_SETS dict and the pruned temporal column list.

    If *prune* is True, run permutation-importance pruning on training data
    and drop zero-importance temporal features.
    """
    useful_temporal = temporal_cols

    if prune and temporal_cols:
        print("\n--- Permutation importance analysis (training years only) ---")
        train_mask = df["year"].isin(TRAIN_YEARS)
        X_pi = df.loc[train_mask, temporal_cols]
        y_pi = df.loc[train_mask, LABEL_COL]

        rf_pi = RandomForestClassifier(n_estimators=100, random_state=42,
                                       n_jobs=-1, class_weight="balanced",
                                       max_features=0.5)
        rf_pi.fit(X_pi, y_pi)

        pi_result = permutation_importance(rf_pi, X_pi, y_pi,
                                           n_repeats=10, random_state=42,
                                           scoring="f1_macro", n_jobs=-1)

        pi_summary = pd.DataFrame({
            "feature":    temporal_cols,
            "importance": pi_result.importances_mean,
            "std":        pi_result.importances_std,
        }).sort_values("importance", ascending=False)

        print(pi_summary.to_string(index=False))

        # Keep only features whose importance is statistically significant
        # (mean importance exceeds 2x its standard deviation)
        significant = pi_summary["importance"] > 2 * pi_summary["std"]
        useful_temporal = pi_summary.loc[significant, "feature"].tolist()
        dropped = [c for c in temporal_cols if c not in useful_temporal]
        print(f"\nKept {len(useful_temporal)} temporal features, dropped {len(dropped)}:")
        for col in dropped:
            status = pi_summary.loc[pi_summary["feature"] == col, "importance"].values[0]
            print(f"  DROPPED: {col}  (importance={status:.6f})")

    # All pfr_ columns — let the RF decide which carry signal
    pfr_cols = [c for c in temporal_cols if c.startswith("pfr_")]
    non_pfr_temporal = [c for c in useful_temporal if not c.startswith("pfr_")]

    feature_sets = {
        "static_flags":  (permission_flag_cols,                       "[A] Static flags only"),
        "static_pfr":   (permission_flag_cols + pfr_cols,            "[B] Static + all PFR"),
        "best_validated":(permission_flag_cols + pfr_cols,            "[C] Best validated (static + PFR)"),
    }

    print("\nFeature sets:")
    for feature_cols, label in feature_sets.values():
        print(f"  {label:<28}: {len(feature_cols)} features")

    return feature_sets, useful_temporal
