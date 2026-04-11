"""
run_lr_feature_eng.py — Feature engineering experiments to improve LR on temporal features.

Motivation (from feature_importance_results.json):
  - RF permutation importance shows only 1 restriction agg feature matters:
      perm_will_restrict_ratio (the rest of 20 cols are noise → collinear noise
      destabilizes LR coefficients)
  - PFR can be pruned to 4 flags: pfr_READ_PHONE_STATE, pfr_SEND_SMS,
      pfr_READ_LOGS, pfr_READ_SMS (only ones passing importance > 2*std)
  - Deprecation features have near-zero RF permutation importance — marginal
      AUT gains are fragile; engineered interaction terms may help
  - Age causes LR collapse; relative-only age features may be stable

Experiments:
  restrict_rf      — static + only RF-important restriction features (3 cols)
  pfr_rf           — static + only RF-important PFR flags (4 flags)
  pfr_rf_restrict  — static + RF PFR + RF restriction
  pfr_rf_shared    — static + RF PFR + RF restriction + shared lifecycle cols
  deprecation_base — static + original deprecation cols (reference)
  deprecation_eng  — static + deprecation + 3 engineered features
  best_combo       — static + RF PFR + RF restriction + engineered deprecation + shared
  age_relative     — static + relative-only age features (no absolute age cols)

Model: scaled_l1 (C=1.0, saga) — best for temporal features from prior run.
       Also runs scaled_l2 for comparison.
AUT = mean macro-F1 over k=1,2,3 (k=4/2016 excluded). Baseline: static LR AUT=0.8522.
"""
import sys
import os
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             accuracy_score, precision_score, recall_score)

from data import load_dataset
from config import TRAIN_YEARS, TEST_YEARS, AUT_EXCLUDE_YEARS, LABEL_COL
from run_paper_ablation import (
    AGE_COLS, RESTRICTION_COLS, RESTRICTION_PFR_COLS,
    DEPRECATION_COLS, DEPRECATION_PFR_COLS, NO_EVENT_PFR_COLS, SHARED_COLS,
)

# ── RF-selected feature subsets (from permutation importance > 2*std) ──────────
# Only restriction agg col with meaningful RF permutation importance
RF_RESTRICT_TOP = [
    "perm_will_restrict_ratio",
    "perm_count_x_will_restrict_ratio",
    "perm_will_restrict_count",
]

# PFR flags passing importance > 2*std in RF permutation importance
RF_PFR_TOP = [
    "pfr_READ_PHONE_STATE",
    "pfr_SEND_SMS",
    "pfr_READ_LOGS",
    "pfr_READ_SMS",
]

# Shared cols with RF permutation signal
RF_SHARED = [
    "perm_count",
    "perm_ever_lifecycle_ratio",
    "perm_ever_lifecycle_count",
]

# Relative-only age features (no absolute age in years — avoids temporal drift)
AGE_RELATIVE_ONLY = [
    "perm_new_perm_ratio",
    "perm_age_relative_max",
]

MODELS = {
    "l1": dict(solver="saga", l1_ratio=1, C=1.0),
    "l2": dict(solver="lbfgs", l1_ratio=0, C=1.0),
}


def run_one(X_train, y_train, X_test, y_test, solver="saga", l1_ratio=1, C=1.0):
    import warnings
    warnings.filterwarnings("ignore")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(
        C=C, solver=solver, l1_ratio=l1_ratio,
        max_iter=1000, random_state=42, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "f1_macro":  f1_score(y_test, preds, average="macro"),
        "accuracy":  accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="binary", zero_division=0),
        "recall":    recall_score(y_test, preds, average="binary"),
        "bal_acc":   balanced_accuracy_score(y_test, preds),
    }


def engineer_deprecation_features(df):
    """Add engineered deprecation interaction cols to df (in-place safe copy)."""
    df = df.copy()

    ratio_col = "perm_deprecated_ratio"
    risk_mean_col = "perm_risk_deprecate_mean"
    announced_col = "perm_announced_deprecate_ratio"

    if ratio_col in df.columns:
        df["dep_has_any"] = (df[ratio_col] > 0).astype(float)
    else:
        df["dep_has_any"] = 0.0

    if ratio_col in df.columns and risk_mean_col in df.columns:
        df["dep_risk_score"] = df[ratio_col] * df[risk_mean_col]
    else:
        df["dep_risk_score"] = 0.0

    if announced_col in df.columns and ratio_col in df.columns:
        df["dep_pending_pressure"] = (df[announced_col] - df[ratio_col]).clip(lower=0)
    else:
        df["dep_pending_pressure"] = 0.0

    return df


ENG_DEPRECATION_COLS = ["dep_has_any", "dep_risk_score", "dep_pending_pressure"]


def main():
    df_raw, perm_cols, temporal_cols = load_dataset(verbose=False)
    df = engineer_deprecation_features(df_raw)

    def avail(cols):
        return [c for c in cols if c in df.columns]

    # Resolve column lists
    deprecate_cols   = avail(DEPRECATION_COLS + DEPRECATION_PFR_COLS)
    rf_restrict      = avail(RF_RESTRICT_TOP)
    rf_pfr           = avail(RF_PFR_TOP)
    rf_shared        = avail(RF_SHARED)
    age_relative     = avail(AGE_RELATIVE_ONLY)
    eng_dep_cols     = avail(ENG_DEPRECATION_COLS)

    feature_sets = {
        # ── Baselines ────────────────────────────────────────────────────────
        "static_only":       (perm_cols,                                  "Static Only (baseline)"),
        "deprecation_base":  (perm_cols + deprecate_cols,                 "Deprecation (base)"),

        # ── RF-pruned restriction ─────────────────────────────────────────
        "restrict_rf":       (perm_cols + rf_restrict,                    "Restrict (RF-selected, 3)"),

        # ── RF-pruned PFR ─────────────────────────────────────────────────
        "pfr_rf":            (perm_cols + rf_pfr,                         "PFR (RF-selected, 4)"),

        # ── RF-pruned PFR + restriction ───────────────────────────────────
        "pfr_rf_restrict":   (perm_cols + rf_pfr + rf_restrict,           "PFR + Restrict (RF-sel)"),

        # ── RF-pruned PFR + restriction + shared lifecycle ────────────────
        "pfr_rf_shared":     (perm_cols + rf_pfr + rf_restrict + rf_shared, "PFR + Restrict + Shared"),

        # ── Engineered deprecation ────────────────────────────────────────
        "deprecation_eng":   (perm_cols + deprecate_cols + eng_dep_cols,  "Deprecation (engineered)"),

        # ── Best combo ────────────────────────────────────────────────────
        "best_combo":        (perm_cols + rf_pfr + rf_restrict + rf_shared
                              + deprecate_cols + eng_dep_cols,            "Best Combo"),

        # ── Age: relative-only ────────────────────────────────────────────
        "age_relative":      (perm_cols + age_relative,                   "Age (relative only)"),
    }

    print(f"\nFeature set sizes:")
    for key, (cols, label) in feature_sets.items():
        print(f"  {label:<35}: {len(cols)} features")
    print()

    all_results = {m: {k: {} for k in feature_sets} for m in MODELS}

    for model_name, model_cfg in MODELS.items():
        reg = "L1" if model_cfg["l1_ratio"] == 1 else "L2"
        print(f"\n{'=' * 70}")
        print(f"MODEL: scaled_{model_name}  ({reg}, C={model_cfg['C']})")
        print(f"{'=' * 70}")

        for test_year in TEST_YEARS:
            train_mask = df["year"].isin(TRAIN_YEARS)
            test_mask  = df["year"] == test_year
            k    = test_year - max(TRAIN_YEARS)
            excl = test_year in AUT_EXCLUDE_YEARS
            note = " [EXCL]" if excl else ""
            print(f"\n  --- k={k} ({test_year}){note} ---", flush=True)

            y_train = df.loc[train_mask, LABEL_COL]
            y_test  = df.loc[test_mask,  LABEL_COL]

            metrics_list = Parallel(n_jobs=-1)(
                delayed(run_one)(
                    df.loc[train_mask, feat_cols].values, y_train.values,
                    df.loc[test_mask,  feat_cols].values, y_test.values,
                    **model_cfg,
                )
                for feat_key, (feat_cols, label) in feature_sets.items()
            )

            for (feat_key, (_, label)), m in zip(feature_sets.items(), metrics_list):
                all_results[model_name][feat_key][test_year] = {
                    **m, "k": k, "excluded": excl
                }
                print(f"  {label:<35} "
                      f"F1={m['f1_macro']:.4f}  "
                      f"Rec={m['recall']:.4f}  Acc={m['accuracy']:.4f}",
                      flush=True)

    # ── AUT summary ────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("AUT SUMMARY  (macro F1, mean k=1,2,3 — k=4 excl)   BASELINE static LR = 0.8522")
    print(f"{'=' * 90}")
    print(f"  {'Feature Set':<35} {'L1 AUT':>8}  {'L2 AUT':>8}  {'Best':>8}")
    print("  " + "-" * 62)

    BASELINE = 0.8522
    for feat_key, (_, label) in feature_sets.items():
        row_vals = {}
        for model_name in MODELS:
            year_data = all_results[model_name][feat_key]
            aut = np.mean([v["f1_macro"] for v in year_data.values() if not v["excluded"]])
            row_vals[model_name] = aut
        best = max(row_vals.values())
        flag = "  *** BEATS BASELINE" if best > BASELINE else ""
        print(f"  {label:<35} {row_vals['l1']:>8.4f}  {row_vals['l2']:>8.4f}  {best:>8.4f}{flag}")

    # ── Per-year breakdown for best config ────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("PER-YEAR BREAKDOWN (best model per feature set)")
    print(f"{'=' * 90}")
    print(f"  {'Feature Set':<35} {'Best Model':<10} {'AUT':>7}  k=1    k=2    k=3    k=4")
    print("  " + "-" * 80)

    for feat_key, (_, label) in feature_sets.items():
        best_model, best_aut = None, -1
        for model_name in MODELS:
            year_data = all_results[model_name][feat_key]
            aut = np.mean([v["f1_macro"] for v in year_data.values() if not v["excluded"]])
            if aut > best_aut:
                best_aut, best_model = aut, model_name

        year_data = all_results[best_model][feat_key]
        per_year  = "  ".join(
            f"{year_data[y]['f1_macro']:.3f}" for y in sorted(TEST_YEARS)
        )
        print(f"  {label:<35} {best_model:<10} {best_aut:>7.4f}  {per_year}")

    # ── Collapse check ────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("COLLAPSE CHECK: recall > 0.99 on age-containing configs")
    print(f"{'=' * 90}")
    for model_name in MODELS:
        for feat_key in ["age_relative"]:
            label = feature_sets[feat_key][1]
            recalls = [all_results[model_name][feat_key][y]["recall"]
                       for y in sorted(TEST_YEARS)]
            flags   = ["*COLLAPSE*" if r > 0.99 else "ok" for r in recalls]
            rec_str = "  ".join(f"{r:.3f}({f})" for r, f in zip(recalls, flags))
            print(f"  {model_name} / {label:<35} {rec_str}")

    return all_results


class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


if __name__ == "__main__":
    out_txt  = os.path.join(SCRIPT_DIR, "lr_feature_eng_output.txt")
    out_json = os.path.join(SCRIPT_DIR, "lr_feature_eng_results.json")

    tee = Tee(out_txt)
    sys.stdout = tee

    print(f"run_lr_feature_eng.py  —  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {out_txt}")
    print(f"JSON:   {out_json}")

    all_results = main()

    sys.stdout = tee.terminal
    tee.close()

    output = {
        "meta": {
            "run_at": datetime.datetime.now().isoformat(),
            "models": MODELS,
            "rf_selected_restriction": RF_RESTRICT_TOP,
            "rf_selected_pfr": RF_PFR_TOP,
        },
        "results": {
            model_name: {
                feat_key: {str(year): metrics for year, metrics in year_data.items()}
                for feat_key, year_data in feat_data.items()
            }
            for model_name, feat_data in all_results.items()
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_json}")
