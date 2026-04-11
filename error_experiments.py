"""
error_experiments.py — Error analysis for section 5.5.

Three experiments:
  1. Improvement concentration — are gains from lifecycle features concentrated
     among apps that request permissions later restricted by Android?

  2. Malware family breakdown — which families benefit most from lifecycle
     features vs. static-only? Which families remain hard to detect?

  3. False negative analysis — do false negatives cluster around newly
     introduced or legacy permissions?

All models train on TRAIN_YEARS (2009-2012), evaluated on TEST_YEARS.
Best config = PFR + Restrict Agg (RF). Baseline = Static Only (RF).
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import TRAIN_YEARS, TEST_YEARS, LABEL_COL, CSV_FILE, HIGH_DRIFT_FLAGS
from data import load_dataset
from run_paper_ablation import (
    RESTRICTION_COLS, RESTRICTION_PFR_COLS, DEPRECATION_PFR_COLS,
    NO_EVENT_PFR_COLS,
)

RF_SEED = 42


def fit_rf(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=RF_SEED,
                                 n_jobs=-1, class_weight="balanced",
                                 max_features=0.5)
    clf.fit(X_train, y_train)
    return clf


def section(title):
    print(f"\n{'=' * 70}")
    print(title)
    print('=' * 70)


# ── Experiment 1: Improvement concentration by permission profile ──────────

def exp1_improvement_by_restriction_profile(df, perm_cols, pfr_restrict_cols):
    """
    Split test apps into two groups:
      - 'Has restricted perms': requests at least one permission that will be
        or has been restricted (perm_has_any_restricted OR perm_has_any_near_restrict)
      - 'No restricted perms': requests none

    Compare F1 improvement (best config vs static only) within each group.
    """
    section("EXPERIMENT 1: Improvement concentration by restriction profile")

    train_mask = df["year"].isin(TRAIN_YEARS)
    X_train_static = df.loc[train_mask, perm_cols]
    X_train_best   = df.loc[train_mask, pfr_restrict_cols]
    y_train        = df.loc[train_mask, LABEL_COL]

    clf_static = fit_rf(X_train_static, y_train)
    clf_best   = fit_rf(X_train_best, y_train)

    results = []
    for test_year in TEST_YEARS:
        test_mask = df["year"] == test_year
        df_test   = df[test_mask].copy()

        preds_static = clf_static.predict(df_test[perm_cols])
        preds_best   = clf_best.predict(df_test[pfr_restrict_cols])
        y_test       = df_test[LABEL_COL].values

        # Group: has any restricted or near-restrict permission
        has_restrict = (
            (df_test.get("perm_has_any_restricted", pd.Series(0, index=df_test.index)) == 1) |
            (df_test.get("perm_has_any_near_restrict", pd.Series(0, index=df_test.index)) == 1)
        ).values

        for group_name, mask in [("Has restricted perms", has_restrict),
                                  ("No restricted perms", ~has_restrict)]:
            if mask.sum() < 10:
                continue
            f1_s = f1_score(y_test[mask], preds_static[mask], average="macro", zero_division=0)
            f1_b = f1_score(y_test[mask], preds_best[mask],   average="macro", zero_division=0)
            n_mal = y_test[mask].sum()
            results.append({
                "year": test_year,
                "group": group_name,
                "n": int(mask.sum()),
                "n_malware": int(n_mal),
                "f1_static": f1_s,
                "f1_best": f1_b,
                "delta": f1_b - f1_s,
            })
            print(f"  {test_year} | {group_name:<25} | n={mask.sum():>5} mal={n_mal:>4} "
                  f"| static={f1_s:.3f}  best={f1_b:.3f}  D={f1_b-f1_s:+.3f}")

    return results


# ── Experiment 2: Per-family F1 breakdown ─────────────────────────────────

def exp2_family_breakdown(df, perm_cols, pfr_restrict_cols, mal_families):
    """
    For each malware family with >= 50 test samples, compute:
      - F1 under static-only RF
      - F1 under best-config RF
      - Delta

    mal_families: Series indexed like df, containing family labels (NaN for benign).
    """
    section("EXPERIMENT 2: Per-family F1 breakdown (test years combined)")

    train_mask = df["year"].isin(TRAIN_YEARS)
    X_train_static = df.loc[train_mask, perm_cols]
    X_train_best   = df.loc[train_mask, pfr_restrict_cols]
    y_train        = df.loc[train_mask, LABEL_COL]

    clf_static = fit_rf(X_train_static, y_train)
    clf_best   = fit_rf(X_train_best, y_train)

    test_mask    = df["year"].isin(TEST_YEARS)
    df_test      = df[test_mask].copy()
    fam_test     = mal_families[test_mask]
    y_test       = df_test[LABEL_COL].values
    preds_static = clf_static.predict(df_test[perm_cols])
    preds_best   = clf_best.predict(df_test[pfr_restrict_cols])

    family_results = []
    top_families = fam_test.value_counts()
    top_families = top_families[top_families >= 50].index.tolist()

    print(f"\n  {'Family':<35} {'N':>5} {'Static F1':>10} {'Best F1':>9} {'Delta':>7}")
    print(f"  {'-'*70}")

    for fam in top_families:
        fam_mask = (fam_test == fam).values
        if fam_mask.sum() == 0:
            continue
        # For family-level: treat family vs rest as binary
        # But we want to know: of this family's samples, how many are correctly classified?
        # Use recall (detection rate) for malware families
        y_fam    = y_test[fam_mask]
        ps_fam   = preds_static[fam_mask]
        pb_fam   = preds_best[fam_mask]

        # All family members are malware, so recall = detection rate
        rec_s = (ps_fam == y_fam).mean()
        rec_b = (pb_fam == y_fam).mean()

        family_results.append({
            "family": fam,
            "n": int(fam_mask.sum()),
            "detection_static": rec_s,
            "detection_best": rec_b,
            "delta": rec_b - rec_s,
        })
        print(f"  {fam:<35} {fam_mask.sum():>5} {rec_s:>10.3f} {rec_b:>9.3f} {rec_b-rec_s:>+7.3f}")

    family_results.sort(key=lambda x: x["delta"])
    print(f"\n  Most improved families:")
    for r in sorted(family_results, key=lambda x: -x["delta"])[:5]:
        print(f"    {r['family']:<35} D={r['delta']:+.3f}")
    print(f"\n  Least improved / degraded families:")
    for r in family_results[:5]:
        print(f"    {r['family']:<35} D={r['delta']:+.3f}")

    return family_results


# ── Experiment 3: False negative permission profile analysis ───────────────

def exp3_false_negative_analysis(df, perm_cols, pfr_restrict_cols):
    """
    For the best-config RF on combined test years:
      - Identify false negatives (malware predicted as benign)
      - Compare their permission profiles to true positives:
        * Average PFR values (are FNs lower risk by PFR?)
        * perm_age_mean, perm_count
        * Rate of requesting newly introduced vs legacy permissions
    """
    section("EXPERIMENT 3: False negative permission profile analysis")

    train_mask = df["year"].isin(TRAIN_YEARS)
    clf_best   = fit_rf(df.loc[train_mask, pfr_restrict_cols],
                        df.loc[train_mask, LABEL_COL])

    pfr_cols = [c for c in pfr_restrict_cols if c.startswith("pfr_")]
    restrict_agg_cols = [c for c in pfr_restrict_cols if c in RESTRICTION_COLS]

    results_by_year = []
    for test_year in TEST_YEARS:
        test_mask = df["year"] == test_year
        df_test   = df[test_mask].copy()
        y_test    = df_test[LABEL_COL].values
        preds     = clf_best.predict(df_test[pfr_restrict_cols])

        malware_mask = y_test == 1
        fn_mask      = (y_test == 1) & (preds == 0)   # false negatives
        tp_mask      = (y_test == 1) & (preds == 1)   # true positives

        n_fn = fn_mask.sum()
        n_tp = tp_mask.sum()
        fn_rate = n_fn / malware_mask.sum() if malware_mask.sum() > 0 else 0

        print(f"\n  Year {test_year}: {malware_mask.sum()} malware | "
              f"{n_tp} TP | {n_fn} FN ({fn_rate:.1%} miss rate)")

        if n_fn < 5:
            print("    Too few FNs to analyze.")
            continue

        df_fn = df_test[fn_mask]
        df_tp = df_test[tp_mask]

        # PFR profile comparison
        print(f"\n  {'Feature':<40} {'TP mean':>9} {'FN mean':>9} {'D':>8}")
        print(f"  {'-'*68}")

        profile_rows = []
        for col in pfr_cols + ["perm_age_mean", "perm_count",
                                "perm_restricted_ratio", "perm_will_restrict_ratio"]:
            if col not in df_test.columns:
                continue
            tp_mean = df_tp[col].mean()
            fn_mean = df_fn[col].mean()
            delta   = fn_mean - tp_mean
            profile_rows.append((col, tp_mean, fn_mean, delta))

        # Sort by absolute delta to show most discriminating features
        profile_rows.sort(key=lambda x: abs(x[3]), reverse=True)
        for col, tp_m, fn_m, d in profile_rows[:15]:
            print(f"  {col:<40} {tp_m:>9.4f} {fn_m:>9.4f} {d:>+8.4f}")

        # Permission age profile: are FNs using newer or older permissions?
        if "perm_age_mean" in df_test.columns:
            print(f"\n  Age profile:")
            print(f"    TP perm_age_mean: {df_tp['perm_age_mean'].mean():.2f} years")
            print(f"    FN perm_age_mean: {df_fn['perm_age_mean'].mean():.2f} years")

        if "perm_new_perm_ratio" in df_test.columns:
            print(f"    TP perm_new_perm_ratio: {df_tp['perm_new_perm_ratio'].mean():.3f}")
            print(f"    FN perm_new_perm_ratio: {df_fn['perm_new_perm_ratio'].mean():.3f}")

        results_by_year.append({
            "year": test_year,
            "n_malware": int(malware_mask.sum()),
            "n_fn": int(n_fn),
            "n_tp": int(n_tp),
            "fn_rate": fn_rate,
        })

    return results_by_year


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    df, perm_cols, temporal_cols = load_dataset(verbose=False)

    def avail(cols):
        return [c for c in cols if c in df.columns]

    all_pfr = avail(RESTRICTION_PFR_COLS + DEPRECATION_PFR_COLS + NO_EVENT_PFR_COLS)
    pfr_restrict_cols = perm_cols + all_pfr + avail(RESTRICTION_COLS)

    print(f"Training years: {TRAIN_YEARS}")
    print(f"Test years:     {TEST_YEARS}")
    print(f"Static features:     {len(perm_cols)}")
    print(f"Best config features: {len(pfr_restrict_cols)}")

    # Load family labels from raw CSV (MalFamily is dropped in load_dataset)
    raw_full = pd.read_csv(CSV_FILE, usecols=["MalFamily", "FirstModDate", "Malware"])
    raw_full["year"] = pd.to_datetime(raw_full["FirstModDate"], errors="coerce").dt.year
    raw_full = raw_full[raw_full["year"] >= 2008].dropna(subset=["year"])
    raw_full["year"] = raw_full["year"].astype(int)
    mal_families = raw_full["MalFamily"].reset_index(drop=True)
    # Align to df index
    mal_families = mal_families.reindex(df.index)

    r1 = exp1_improvement_by_restriction_profile(df, perm_cols, pfr_restrict_cols)
    r2 = exp2_family_breakdown(df, perm_cols, pfr_restrict_cols, mal_families)
    r3 = exp3_false_negative_analysis(df, perm_cols, pfr_restrict_cols)

    out = {
        "improvement_by_restriction_profile": r1,
        "family_breakdown": r2,
        "false_negative_profile": r3,
    }
    with open("error_analysis_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to error_analysis_results.json")


if __name__ == "__main__":
    main()
