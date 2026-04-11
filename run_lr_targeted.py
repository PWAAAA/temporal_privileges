"""
run_lr_targeted.py -- Targeted follow-up: selective year-normalization.

Finding from run_lr_drift_fix.py:
  - PFR features: gap GROWING (signal amplifies beyond training distribution)
    -> year-norm removes scale explosion, helps at k=3
  - Restriction features: gap STABLE (perm_will_restrict_ratio is structural)
    -> year-norm destroys a stable signal, actively hurts
  - Applying yn to ALL temporal features at once is net-negative
  - PFR-only + yn + linear gets k=3=0.855 (=static) but loses k=1/k=2

Hypothesis:
  Apply yn ONLY to PFR features, keep restriction+shared features raw.
  This removes PFR scale explosion while preserving restriction's stable signal.
  Add recency weighting to help calibration on recent distributions.
  Target: beat Best Combo AUT (0.8795 with last2) AND static k=3 (0.855).

Benchmarks:
  BASELINE_AUT  = 0.8510  (scaled static LR)
  COMBO_AUT     = 0.8795  (Best Combo + last2, current best)
  STATIC_K3     = 0.8552
  COMBO_K3      = 0.8415  (Best Combo + last2)
  PFR_YN_K3     = 0.8546  (PFR-only + yn + linear, near static)

Performance note:
  Pre-compute all yn columns once upfront, then work with numpy arrays directly.
  Avoids repeated df.copy() on a fragmented DataFrame (which caused hangs).
"""
import sys
import os
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score

from data import load_dataset
from config import TRAIN_YEARS, TEST_YEARS, AUT_EXCLUDE_YEARS, LABEL_COL
from run_paper_ablation import DEPRECATION_COLS, DEPRECATION_PFR_COLS
from run_lr_feature_eng import (
    RF_RESTRICT_TOP, RF_PFR_TOP, RF_SHARED, ENG_DEPRECATION_COLS,
    engineer_deprecation_features,
)

BENCHMARKS = {
    "BASELINE_AUT": 0.8510,
    "COMBO_AUT":    0.8795,
    "STATIC_K3":    0.8552,
    "COMBO_K3":     0.8415,
    "PFR_YN_K3":    0.8546,
}


# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    return {
        "f1_macro":  f1_score(y_true, y_pred, average="macro"),
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="binary"),
        "bal_acc":   balanced_accuracy_score(y_true, y_pred),
    }


def aut(year_results):
    return np.mean([v["f1_macro"] for v in year_results.values() if not v["excluded"]])


def precompute_yn(df, cols, all_years):
    """
    Compute year-normalized versions of cols once.
    Returns a dict: col -> np.array of yn values (same index as df).
    Uses only marginal distribution per year (no labels).
    """
    yn_arrays = {}
    for col in cols:
        if col not in df.columns:
            continue
        arr = df[col].values.astype(float).copy()
        for y in all_years:
            mask = (df["year"] == y).values
            mu  = arr[mask].mean()
            sig = arr[mask].std()
            if sig > 1e-9:
                arr[mask] = (arr[mask] - mu) / sig
            else:
                arr[mask] = 0.0
        yn_arrays[col] = arr
    return yn_arrays


def recency_weights(years_arr, scheme):
    min_y, max_y = years_arr.min(), years_arr.max()
    if scheme == "linear":
        w = (years_arr - min_y + 1).astype(float)
    elif scheme == "exp":
        w = np.exp(0.5 * (years_arr - min_y))
    elif scheme == "last2":
        w = np.where(years_arr >= max_y - 1, 2.0, 1.0)
    else:
        w = np.ones(len(years_arr))
    return w / w.mean()


def run_experiment(df, col_data, feat_cols, yn_set, weight_scheme, model_cfg):
    """
    col_data: dict mapping col_name -> np.array (raw values, pre-extracted from df)
    yn_set:   set of col names to use yn version instead of raw
    feat_cols: list of column names defining the feature matrix
    weight_scheme: None, "linear", "exp", "last2"
    """
    train_mask = df["year"].isin(TRAIN_YEARS).values
    years_train = df.loc[train_mask, "year"].values

    results = {}
    for test_year in TEST_YEARS:
        test_mask = (df["year"] == test_year).values
        excl = test_year in AUT_EXCLUDE_YEARS

        # Build feature matrix from pre-extracted arrays
        avail = [c for c in feat_cols if c in col_data or (c + "_raw") in col_data]
        cols_to_use = []
        for c in feat_cols:
            if c not in col_data:
                continue
            cols_to_use.append(c)

        if not cols_to_use:
            continue

        X = np.column_stack([col_data[c] for c in cols_to_use])
        X_train = X[train_mask]
        X_test  = X[test_mask]
        y_train = df.loc[train_mask, LABEL_COL].values
        y_test  = df.loc[test_mask,  LABEL_COL].values

        sw = recency_weights(years_train, weight_scheme) if weight_scheme else None

        warnings.filterwarnings("ignore")
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = LogisticRegression(
            C=model_cfg["C"], solver=model_cfg["solver"],
            l1_ratio=model_cfg["l1_ratio"],
            max_iter=2000, random_state=42, class_weight="balanced",
        )
        model.fit(Xtr, y_train, sample_weight=sw)
        preds = model.predict(Xte)
        m = metrics(y_test, preds)
        m["k"] = test_year - max(TRAIN_YEARS)
        m["excluded"] = excl
        results[test_year] = m

    return results


# ---------------------------------------------------------------------------
def print_results_table(experiments, all_results):
    COMBO_AUT    = BENCHMARKS["COMBO_AUT"]
    BASELINE_AUT = BENCHMARKS["BASELINE_AUT"]

    print(f"\n{'='*108}")
    print("EXPERIMENT GRID: per-year F1 and AUT")
    print(f"{'='*108}")
    print(f"  {'Config':<55} {'AUT':>7}  k=1    k=2    k=3    k=4(excl)")
    print("  " + "-"*93)
    for label, *_ in experiments:
        res = all_results[label]
        a = aut(res)
        per_yr = "  ".join(f"{res[y]['f1_macro']:.3f}" for y in sorted(TEST_YEARS))
        flag = ""
        if a > COMBO_AUT:
            flag = "  *** BEATS COMBO"
        elif a > BASELINE_AUT:
            flag = "  * beats static"
        print(f"  {label:<55} {a:>7.4f}  {per_yr}{flag}")


def print_k3_table(experiments, all_results):
    STATIC_K3 = BENCHMARKS["STATIC_K3"]
    COMBO_K3  = BENCHMARKS["COMBO_K3"]

    print(f"\n{'='*108}")
    print("K=3 (2015) FOCUSED: does selective yn fix the regression?")
    print(f"  Targets: k=3 > {STATIC_K3:.4f} (static) and AUT > {BENCHMARKS['COMBO_AUT']:.4f} (best combo)")
    print(f"{'='*108}")
    print(f"  {'Config':<55} {'k=1':>7}  {'k=2':>7}  {'k=3':>7}  {'delta vs combo_k3':>18}")
    print("  " + "-"*93)
    for label, *_ in experiments:
        res = all_results[label]
        k1 = res[2013]["f1_macro"]
        k2 = res[2014]["f1_macro"]
        k3 = res[2015]["f1_macro"]
        delta = k3 - COMBO_K3
        flag = "  *** FIXES K3" if k3 > STATIC_K3 else (
               "  * partial"    if k3 > COMBO_K3  else "")
        print(f"  {label:<55} {k1:>7.4f}  {k2:>7.4f}  {k3:>7.4f}  {delta:>+18.4f}{flag}")


# ---------------------------------------------------------------------------
def main():
    print("Loading data...", flush=True)
    df_raw, perm_cols, _ = load_dataset(verbose=False)
    df = engineer_deprecation_features(df_raw)
    df = df.reset_index(drop=True)
    print(f"  {len(df)} rows, {len(df.columns)} cols", flush=True)

    def avail(cols):
        return [c for c in cols if c in df.columns]

    dep_base  = avail(DEPRECATION_COLS + DEPRECATION_PFR_COLS)
    eng_dep   = avail(ENG_DEPRECATION_COLS)
    rf_pfr    = avail(RF_PFR_TOP)
    rf_rest   = avail(RF_RESTRICT_TOP)
    rf_shared = avail(RF_SHARED)
    all_years = sorted(TRAIN_YEARS + TEST_YEARS)

    best_combo_cols = perm_cols + rf_pfr + rf_rest + rf_shared + dep_base + eng_dep

    # Pre-compute raw arrays for every column we'll ever need (once, fast)
    print("Pre-extracting column arrays...", flush=True)
    all_needed = list(set(best_combo_cols + rf_pfr + rf_rest + rf_shared))
    raw = {c: df[c].values.astype(float) for c in all_needed if c in df.columns}

    # Pre-compute yn arrays for all candidate columns (once)
    print("Pre-computing year-normalized arrays...", flush=True)
    yn_candidates = rf_pfr + rf_rest + rf_shared
    yn_arrays = precompute_yn(df, yn_candidates, all_years)
    # yn_arrays[col] is the year-normalized version of col

    def col_data_for(feat_cols, yn_set):
        """Build col_data dict: use yn array if col in yn_set, else raw."""
        cd = {}
        for c in feat_cols:
            if c not in raw:
                continue
            if c in yn_set and c in yn_arrays:
                cd[c] = yn_arrays[c]
            else:
                cd[c] = raw[c]
        return cd

    model_cfg = dict(solver="saga", l1_ratio=1, C=1.0)

    # ── Define experiments ──────────────────────────────────────────────────
    # (label, feat_cols, yn_set, weight_scheme)
    yn_pfr         = set(rf_pfr)
    yn_pfr_shared  = set(rf_pfr + rf_shared)
    yn_all_temporal = set(rf_pfr + rf_rest + rf_shared)

    experiments = [
        # Reference points
        ("Static (reference)",
         perm_cols, set(), None),

        ("Best Combo (no fix)",
         best_combo_cols, set(), None),

        ("Best Combo + last2 [current best]",
         best_combo_cols, set(), "last2"),

        # Core hypothesis: yn only on PFR, restriction stays raw
        ("Combo: PFR-yn, Rest-raw (no weight)",
         best_combo_cols, yn_pfr, None),

        ("Combo: PFR-yn, Rest-raw + linear",
         best_combo_cols, yn_pfr, "linear"),

        ("Combo: PFR-yn, Rest-raw + exp",
         best_combo_cols, yn_pfr, "exp"),

        ("Combo: PFR-yn, Rest-raw + last2",
         best_combo_cols, yn_pfr, "last2"),

        # Extend yn to shared features (also GROWING gap)
        ("Combo: PFR+Shared-yn, Rest-raw (no weight)",
         best_combo_cols, yn_pfr_shared, None),

        ("Combo: PFR+Shared-yn, Rest-raw + linear",
         best_combo_cols, yn_pfr_shared, "linear"),

        ("Combo: PFR+Shared-yn, Rest-raw + last2",
         best_combo_cols, yn_pfr_shared, "last2"),

        # Minimal: PFR-yn + Restrict-raw, no dep features
        ("PFR-yn + Restrict-raw (no dep, no weight)",
         perm_cols + rf_pfr + rf_rest, yn_pfr, None),

        ("PFR-yn + Restrict-raw + linear",
         perm_cols + rf_pfr + rf_rest, yn_pfr, "linear"),

        ("PFR-yn + Restrict-raw + last2",
         perm_cols + rf_pfr + rf_rest, yn_pfr, "last2"),

        # Sanity check: yn everything (prior result, should match drift_fix)
        ("Best Combo + full-yn + linear [prior sanity]",
         best_combo_cols, yn_all_temporal, "linear"),
    ]

    print(f"\nRunning {len(experiments)} experiments...\n", flush=True)
    all_results = {}
    for i, (label, feat_cols, yn_set, ws) in enumerate(experiments, 1):
        cd = col_data_for(feat_cols, yn_set)
        res = run_experiment(df, cd, list(feat_cols), yn_set, ws, model_cfg)
        all_results[label] = res
        a = aut(res)
        print(f"  [{i:02d}/{len(experiments)}] {label:<55} AUT={a:.4f}", flush=True)

    print_results_table(experiments, all_results)
    print_k3_table(experiments, all_results)

    return all_results


# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        safe = message.encode("cp1252", errors="replace").decode("cp1252")
        self.terminal.write(safe)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


if __name__ == "__main__":
    out_txt  = os.path.join(SCRIPT_DIR, "lr_targeted_output.txt")
    out_json = os.path.join(SCRIPT_DIR, "lr_targeted_results.json")

    tee = Tee(out_txt)
    sys.stdout = tee
    print(f"run_lr_targeted.py  --  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("Benchmarks:")
    for k, v in BENCHMARKS.items():
        print(f"  {k} = {v}")
    print()

    all_results = main()

    sys.stdout = tee.terminal
    tee.close()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({label: {str(y): v for y, v in res.items()}
                   for label, res in all_results.items()}, f, indent=2)

    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_json}")
