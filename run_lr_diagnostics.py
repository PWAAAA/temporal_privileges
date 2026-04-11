"""
run_lr_diagnostics.py — Understand WHY certain temporal features work for LR.

Three analyses:
  1. COEFFICIENT ANALYSIS
     Fit LR on each feature set, print top coefficients by abs magnitude.
     For temporal features: do they get large weights? do they get zeroed by L1?

  2. TEMPORAL DRIFT
     For each temporal feature group, compute per-year feature means
     (train years 2009-2012 pooled vs each test year).
     High drift = feature changes meaning → LR coefficients trained on 2009-2012
     will misfire on 2015/2016.

  3. CLASS SEPARABILITY (Cohen's d)
     Within each year, how well does each temporal feature separate
     malware from benign? d = (mean_mal - mean_ben) / pooled_std.
     High d across years = feature is consistently informative.
     Inconsistent d across years = feature captures temporal noise.

Output: lr_diagnostics_output.txt
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import load_dataset
from config import TRAIN_YEARS, TEST_YEARS, AUT_EXCLUDE_YEARS, LABEL_COL
from run_paper_ablation import (
    AGE_COLS, RESTRICTION_COLS, RESTRICTION_PFR_COLS,
    DEPRECATION_COLS, DEPRECATION_PFR_COLS, NO_EVENT_PFR_COLS, SHARED_COLS,
)
from run_lr_feature_eng import (
    RF_RESTRICT_TOP, RF_PFR_TOP, RF_SHARED, AGE_RELATIVE_ONLY,
    ENG_DEPRECATION_COLS, engineer_deprecation_features,
)


def cohen_d(a, b):
    """Cohen's d: (mean_a - mean_b) / pooled_std."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled


def fit_lr(X_train, y_train, X_test, y_test, solver="saga", l1_ratio=1, C=1.0):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = LogisticRegression(
        C=C, solver=solver, l1_ratio=l1_ratio,
        max_iter=1000, random_state=42, class_weight="balanced",
    )
    model.fit(X_tr, y_train)
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
def analysis_1_coefficients(df, perm_cols, feature_sets, model_cfg, top_n=15):
    """Fit LR on k=1 (2013) and print top coefficients per feature set."""
    print(f"\n{'='*90}")
    print("ANALYSIS 1: LR COEFFICIENT MAGNITUDES  (k=1, 2013 test year)")
    print(f"{'='*90}")
    print("Shows which features get large weights and which L1 zeroes out.\n")

    train_mask = df["year"].isin(TRAIN_YEARS)
    test_mask  = df["year"] == 2013
    y_train = df.loc[train_mask, LABEL_COL].values
    y_test  = df.loc[test_mask,  LABEL_COL].values

    for feat_key, (feat_cols, label) in feature_sets.items():
        X_train = df.loc[train_mask, feat_cols].values
        X_test  = df.loc[test_mask,  feat_cols].values

        model, _ = fit_lr(X_train, y_train, X_test, y_test, **model_cfg)
        coefs = model.coef_[0]

        # Sort by absolute value
        idx_sorted = np.argsort(np.abs(coefs))[::-1]

        # Partition into static vs temporal
        temporal_set = set(
            AGE_COLS + RESTRICTION_COLS + RESTRICTION_PFR_COLS
            + DEPRECATION_COLS + DEPRECATION_PFR_COLS
            + NO_EVENT_PFR_COLS + SHARED_COLS
            + ENG_DEPRECATION_COLS
        )

        n_temporal = sum(1 for c in feat_cols if c in temporal_set)
        n_zero = np.sum(np.abs(coefs) < 1e-6)
        max_coef = np.max(np.abs(coefs))
        temporal_coefs = [coefs[i] for i, c in enumerate(feat_cols) if c in temporal_set]
        static_coefs   = [coefs[i] for i, c in enumerate(feat_cols) if c not in temporal_set]

        print(f"  {label}  ({len(feat_cols)} features, {n_temporal} temporal)")
        print(f"    Zeroed by L1: {n_zero}/{len(feat_cols)}  |  "
              f"Max |coef|: {max_coef:.4f}  |  "
              f"Temporal mean|coef|: {np.mean(np.abs(temporal_coefs)):.4f}  "
              f"Static mean|coef|: {np.mean(np.abs(static_coefs)):.4f}")

        # Top-n by abs coef, label with [T] temporal or [S] static
        print(f"    Top {top_n} by |coef|:")
        for rank, i in enumerate(idx_sorted[:top_n], 1):
            c = feat_cols[i]
            tag = "[T]" if c in temporal_set else "[S]"
            print(f"      {rank:>2}. {tag} {c:<45} coef={coefs[i]:+.4f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
def analysis_2_temporal_drift(df, temporal_feature_groups):
    """Compute per-year mean of each temporal feature group and measure drift."""
    print(f"\n{'='*90}")
    print("ANALYSIS 2: TEMPORAL DRIFT  (train pooled mean -> per test-year mean)")
    print(f"{'='*90}")
    print("High drift = feature distribution shifts after training → LR misfires.\n")

    train_mask = df["year"].isin(TRAIN_YEARS)

    for group_name, cols in temporal_feature_groups.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue

        print(f"  [{group_name}]")
        print(f"  {'Feature':<45} {'Train':>7}  " +
              "  ".join(f"k={y-2012}({y})" for y in TEST_YEARS))
        print("  " + "-"*90)

        for col in available:
            train_mean = df.loc[train_mask, col].mean()
            test_means = [df.loc[df["year"] == y, col].mean() for y in TEST_YEARS]
            drifts     = [f"{v:>7.4f}" for v in test_means]
            # Flag large drift: if any test year deviates >50% from train mean
            max_drift = max(abs(v - train_mean) for v in test_means) / (abs(train_mean) + 1e-9)
            flag = "  *** HIGH DRIFT" if max_drift > 0.5 else ""
            print(f"  {col:<45} {train_mean:>7.4f}  {'  '.join(drifts)}{flag}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
def analysis_3_separability(df, temporal_feature_groups):
    """Compute Cohen's d (malware vs benign) per feature per year."""
    print(f"\n{'='*90}")
    print("ANALYSIS 3: CLASS SEPARABILITY (Cohen's d, malware vs benign per year)")
    print(f"{'='*90}")
    print("High d = feature separates classes. Stable d across years = reliable signal.\n")
    print("  d > 0.8 = large, 0.5-0.8 = medium, 0.2-0.5 = small, < 0.2 = negligible\n")

    all_years = TRAIN_YEARS + TEST_YEARS

    for group_name, cols in temporal_feature_groups.items():
        available = [c for c in cols if c in df.columns]
        if not available:
            continue

        print(f"  [{group_name}]")
        header = f"  {'Feature':<45}" + "".join(f"  {y}" for y in all_years) + "   mean_d  std_d"
        print(header)
        print("  " + "-"*len(header))

        for col in available:
            ds = []
            for y in all_years:
                yr_mask = df["year"] == y
                mal  = df.loc[yr_mask & (df[LABEL_COL] == 1), col].dropna().values
                ben  = df.loc[yr_mask & (df[LABEL_COL] == 0), col].dropna().values
                d = cohen_d(mal, ben)
                ds.append(d)

            d_arr = [d for d in ds if not np.isnan(d)]
            mean_d = np.mean(np.abs(d_arr)) if d_arr else np.nan
            std_d  = np.std(np.abs(d_arr)) if d_arr else np.nan

            d_str = "".join(f"  {d:+5.2f}" if not np.isnan(d) else "    nan" for d in ds)
            flag = ""
            if mean_d > 0.5:
                flag = "  *** STRONG"
            elif mean_d > 0.2:
                flag = "  * medium"
            print(f"  {col:<45}{d_str}   {mean_d:5.2f}  {std_d:5.2f}{flag}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
def analysis_4_combo_vs_parts(df, perm_cols, model_cfg):
    """
    Fit LR on PFR-only, Restrict-only, and Best Combo across all test years.
    Print coefficients for the temporal features specifically.
    Goal: understand how coefficient estimates change when features are combined.
    """
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*90}")
    print("ANALYSIS 4: COEFFICIENT STABILITY — WHY PFR+RESTRICT COMBO BEATS PARTS")
    print(f"{'='*90}")
    print("Comparing temporal feature coefficients in isolation vs in full combo.\n")

    def avail(cols):
        return [c for c in cols if c in df.columns]

    rf_pfr      = avail(RF_PFR_TOP)
    rf_restrict = avail(RF_RESTRICT_TOP)
    rf_shared   = avail(RF_SHARED)
    dep_base    = avail(DEPRECATION_COLS + DEPRECATION_PFR_COLS)
    eng_dep     = avail(ENG_DEPRECATION_COLS)

    configs = {
        "pfr_only":   perm_cols + rf_pfr,
        "restrict_only": perm_cols + rf_restrict,
        "best_combo": perm_cols + rf_pfr + rf_restrict + rf_shared + dep_base + eng_dep,
    }

    all_temporal = set(rf_pfr + rf_restrict + rf_shared + dep_base + eng_dep)

    train_mask = df["year"].isin(TRAIN_YEARS)

    for test_year in [2013, 2014, 2015]:
        test_mask = df["year"] == test_year
        k = test_year - max(TRAIN_YEARS)
        print(f"  --- k={k} ({test_year}) ---")

        y_train = df.loc[train_mask, LABEL_COL].values
        y_test  = df.loc[test_mask,  LABEL_COL].values

        for cfg_name, feat_cols in configs.items():
            X_train = df.loc[train_mask, feat_cols].values
            X_test  = df.loc[test_mask,  feat_cols].values

            model, scaler = fit_lr(X_train, y_train, X_test, y_test, **model_cfg)
            preds = model.predict(scaler.transform(X_test))
            f1 = f1_score(y_test, preds, average="macro")

            coefs = model.coef_[0]
            temporal_entries = [(feat_cols[i], coefs[i])
                                for i in range(len(feat_cols))
                                if feat_cols[i] in all_temporal]
            temporal_entries.sort(key=lambda x: abs(x[1]), reverse=True)

            n_zero = sum(1 for _, c in temporal_entries if abs(c) < 1e-6)
            print(f"\n    {cfg_name}  F1={f1:.4f}  "
                  f"temporal_features={len(temporal_entries)}  zeroed={n_zero}")
            for feat, coef in temporal_entries[:10]:
                print(f"      {feat:<45} {coef:+.4f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    df_raw, perm_cols, temporal_cols = load_dataset(verbose=False)
    df = engineer_deprecation_features(df_raw)

    def avail(cols):
        return [c for c in cols if c in df.columns]

    # Feature sets from feature_eng (for analysis 1)
    dep_base = avail(DEPRECATION_COLS + DEPRECATION_PFR_COLS)
    rf_pfr   = avail(RF_PFR_TOP)
    rf_rest  = avail(RF_RESTRICT_TOP)
    rf_shared = avail(RF_SHARED)
    eng_dep  = avail(ENG_DEPRECATION_COLS)
    age_rel  = avail(AGE_RELATIVE_ONLY)

    feature_sets = {
        "static_only":    (perm_cols,                                         "Static Only"),
        "pfr_rf":         (perm_cols + rf_pfr,                                "PFR (RF-selected)"),
        "restrict_rf":    (perm_cols + rf_rest,                               "Restrict (RF-selected)"),
        "best_combo":     (perm_cols + rf_pfr + rf_rest + rf_shared
                           + dep_base + eng_dep,                              "Best Combo"),
        "age_relative":   (perm_cols + age_rel,                               "Age (relative only)"),
    }

    model_cfg = dict(solver="saga", l1_ratio=1, C=1.0)  # scaled_l1

    # Temporal feature groups for drift and separability analysis
    temporal_groups = {
        "RF-selected PFR (4 flags)":      RF_PFR_TOP,
        "RF-selected Restriction (3)":    RF_RESTRICT_TOP,
        "RF-selected Shared (3)":         RF_SHARED,
        "Deprecation (base)":             DEPRECATION_COLS,
        "Engineered Deprecation (3)":     ENG_DEPRECATION_COLS,
        "Age features (cause collapse)":  AGE_COLS,
        "Age relative-only":              AGE_RELATIVE_ONLY,
    }

    analysis_1_coefficients(df, perm_cols, feature_sets, model_cfg, top_n=15)
    analysis_2_temporal_drift(df, temporal_groups)
    analysis_3_separability(df, temporal_groups)
    analysis_4_combo_vs_parts(df, perm_cols, model_cfg)


class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message.encode("cp1252", errors="replace").decode("cp1252"))
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


if __name__ == "__main__":
    import datetime
    out_txt = os.path.join(SCRIPT_DIR, "lr_diagnostics_output.txt")
    tee = Tee(out_txt)
    sys.stdout = tee
    print(f"run_lr_diagnostics.py  —  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    sys.stdout = tee.terminal
    tee.close()
    print(f"\nSaved: {out_txt}")
