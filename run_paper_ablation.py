"""
run_paper_ablation.py — Run the 5-row ablation from the paper's Table 2.

Feature groups:
  1. Static Only:          145 permission flags
  2. + Age Features:       static + age-based temporal cols
  3. + Restriction Features: static + restriction aggregate + restriction PFR cols
  4. + Deprecation Features: static + deprecation aggregate + deprecation PFR cols
  5. Full Lifecycle Set:   static + all temporal + all PFR

PFR columns are assigned to restriction/deprecation based on which lifecycle
event they encode (restrict_year takes priority over deprecate_year in the
PFR encoding logic). 4 PFR flags with no lifecycle event (always=1.0) go
into the full set only.

RF models are run with multiple seeds and averaged.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)
from config import (TRAIN_YEARS, TEST_YEARS, AUT_EXCLUDE_YEARS, LABEL_COL,
                    HIGH_DRIFT_FLAGS)
from data import load_dataset

RF_SEEDS = [42, 123, 256, 789, 1337]
METRIC_KEYS = ["f1_weighted", "f1_macro", "bal_acc", "accuracy", "precision", "recall"]

# ── Feature group definitions ─────────────────────────────────────────────

AGE_COLS = [
    "perm_age_mean", "perm_age_max", "perm_age_min", "perm_age_std",
    "perm_worst_age", "perm_new_perm_ratio", "perm_age_relative_max",
]

RESTRICTION_COLS = [
    "perm_restricted_ratio", "perm_near_restrict_ratio",
    "perm_risk_restrict_mean", "perm_risk_restrict_min",
    "perm_announced_restrict_ratio",
    "perm_risk_announced_restrict_mean", "perm_risk_announced_restrict_min",
    "perm_restricted_count", "perm_near_restrict_count",
    "perm_has_any_restricted", "perm_has_any_near_restrict",
    "perm_risk_worst_restrict", "perm_worst_perm_age_at_restrict",
    "perm_age_x_near_restrict", "perm_worst_age_x_risk_restrict",
    "perm_restrict_density", "perm_unrestricted_old_ratio",
    "perm_will_restrict_count", "perm_will_restrict_ratio",
    "perm_count_x_will_restrict_ratio",
]

# PFR flags that use restrict_year as their event (26 flags)
RESTRICTION_PFR_FLAGS = [
    "READ_CONTACTS", "ACCESS_COARSE_LOCATION", "SYSTEM_ALERT_WINDOW",
    "CALL_PHONE", "RECEIVE_SMS", "WRITE_SETTINGS", "READ_SMS",
    "ACCESS_FINE_LOCATION", "GET_ACCOUNTS", "READ_LOGS", "READ_CALL_LOG",
    "WRITE_CONTACTS", "BROADCAST_SMS", "CAMERA", "PROCESS_OUTGOING_CALLS",
    "BROADCAST_WAP_PUSH", "READ_PHONE_STATE", "READ_EXTERNAL_STORAGE",
    "RECORD_AUDIO", "WRITE_CALL_LOG", "CHANGE_CONFIGURATION",
    "WRITE_EXTERNAL_STORAGE", "SEND_SMS", "RECEIVE_MMS",
    "RECEIVE_WAP_PUSH", "READ_CALENDAR",
]
RESTRICTION_PFR_COLS = [f"pfr_{p}" for p in RESTRICTION_PFR_FLAGS]

DEPRECATION_COLS = [
    "perm_deprecated_ratio", "perm_deprecated_count",
    "perm_risk_deprecate_mean", "perm_risk_deprecate_min",
    "perm_risk_deprecate_max", "perm_risk_deprecate_std",
    "perm_announced_deprecate_ratio",
    "perm_risk_announced_deprecate_mean", "perm_risk_announced_deprecate_min",
    "perm_count_x_risk_deprecate",
]

# PFR flags that use deprecate_year as their event (4 flags)
DEPRECATION_PFR_FLAGS = [
    "GET_TASKS", "BROADCAST_STICKY", "RESTART_PACKAGES", "USE_FINGERPRINT",
]
DEPRECATION_PFR_COLS = [f"pfr_{p}" for p in DEPRECATION_PFR_FLAGS]

# PFR flags with no lifecycle event (always 1.0) — full set only
NO_EVENT_PFR_FLAGS = [
    "ACCESS_WIFI_STATE", "CHANGE_NETWORK_STATE", "CHANGE_WIFI_STATE", "INTERNET",
]
NO_EVENT_PFR_COLS = [f"pfr_{p}" for p in NO_EVENT_PFR_FLAGS]

# Shared / cross-cutting cols that go into full set only
SHARED_COLS = [
    "perm_count",
    "perm_ever_lifecycle_count", "perm_ever_lifecycle_ratio",
]


def run_single(X_train, y_train, X_test, y_test, model_type, seed=42):
    """Train and evaluate a single model, return metrics dict."""
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=seed,
                                        n_jobs=-1, class_weight="balanced",
                                        max_features=0.5)
    elif model_type == "lr":
        model = LogisticRegression(max_iter=2000, random_state=42,
                                    class_weight="balanced", solver="saga")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "bal_acc": balanced_accuracy_score(y_test, preds),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="binary"),
        "recall": recall_score(y_test, preds, average="binary"),
    }


def run_averaged(X_train, y_train, X_test, y_test, model_type):
    """For RF: run multiple seeds and return mean + std. For LR: single run."""
    if model_type == "lr":
        metrics = run_single(X_train, y_train, X_test, y_test, "lr")
        return {k: metrics[k] for k in METRIC_KEYS}, {k: 0.0 for k in METRIC_KEYS}

    all_runs = [run_single(X_train, y_train, X_test, y_test, "rf", seed=s)
                for s in RF_SEEDS]
    means = {k: np.mean([r[k] for r in all_runs]) for k in METRIC_KEYS}
    stds = {k: np.std([r[k] for r in all_runs]) for k in METRIC_KEYS}
    return means, stds


def main():
    df, perm_cols, temporal_cols = load_dataset(verbose=False)

    # Filter to cols actually present in df
    def available(cols):
        return [c for c in cols if c in df.columns]

    age_cols = available(AGE_COLS)
    restrict_cols = available(RESTRICTION_COLS + RESTRICTION_PFR_COLS)
    deprecate_cols = available(DEPRECATION_COLS + DEPRECATION_PFR_COLS)
    shared_cols = available(SHARED_COLS)
    no_event_pfr = available(NO_EVENT_PFR_COLS)
    all_temporal = available(
        AGE_COLS + RESTRICTION_COLS + RESTRICTION_PFR_COLS
        + DEPRECATION_COLS + DEPRECATION_PFR_COLS
        + NO_EVENT_PFR_COLS + SHARED_COLS
    )

    all_pfr = available(RESTRICTION_PFR_COLS + DEPRECATION_PFR_COLS + NO_EVENT_PFR_COLS)
    no_age_temporal = [c for c in all_temporal if c not in age_cols]

    feature_sets = {
        "static_only":    (perm_cols,                          "Static Only"),
        "age":            (perm_cols + age_cols,               "+ Age Features"),
        "restriction":    (perm_cols + restrict_cols,          "+ Restriction Features"),
        "deprecation":    (perm_cols + deprecate_cols,         "+ Deprecation Features"),
        "full_lifecycle": (perm_cols + all_temporal,           "Full Lifecycle Set"),
        "full_no_age":    (perm_cols + no_age_temporal,        "Full Lifecycle (No Age)"),
        "pfr_only":       (perm_cols + all_pfr,               "Static + PFR Only"),
        "pfr_age":        (perm_cols + all_pfr + age_cols,    "PFR + Age Agg"),
        "pfr_restrict":   (perm_cols + all_pfr + available(RESTRICTION_COLS), "PFR + Restrict Agg"),
    }

    print(f"\nFeature set sizes:")
    for key, (cols, label) in feature_sets.items():
        print(f"  {label:<25}: {len(cols)} features")

    print(f"\nRF seeds: {RF_SEEDS} ({len(RF_SEEDS)} runs per configuration)")

    # ── Run forward-year evaluation ───────────────────────────────────────
    results = {}
    for model_type in ["lr", "rf"]:
        results[model_type] = {}
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_type.upper()}" + (" (averaged over {} seeds)".format(len(RF_SEEDS)) if model_type == "rf" else ""))
        print(f"{'=' * 70}")

        for test_year in TEST_YEARS:
            train_mask = df["year"].isin(TRAIN_YEARS)
            test_mask = df["year"] == test_year
            k = test_year - max(TRAIN_YEARS)
            exclude = test_year in AUT_EXCLUDE_YEARS
            note = " [EXCL]" if exclude else ""

            print(f"\n--- Test: {test_year} (k={k}){note} ---")

            y_train = df.loc[train_mask, LABEL_COL]
            y_test = df.loc[test_mask, LABEL_COL]

            for key, (feat_cols, label) in feature_sets.items():
                means, stds = run_averaged(
                    df.loc[train_mask, feat_cols], y_train,
                    df.loc[test_mask, feat_cols], y_test,
                    model_type=model_type,
                )

                if key not in results[model_type]:
                    results[model_type][key] = {}

                entry = {
                    "k": k,
                    "excluded": exclude,
                }
                for m in METRIC_KEYS:
                    entry[m] = means[m]
                if model_type == "rf":
                    for m in METRIC_KEYS:
                        entry[f"{m}_std"] = stds[m]

                results[model_type][key][test_year] = entry

                std_note = f" ±{stds['f1_macro']:.4f}" if model_type == "rf" else ""
                print(f"  {label:<25} F1={means['f1_macro']:.4f}{std_note}  "
                      f"Acc={means['accuracy']:.4f}  Rec={means['recall']:.4f}")

    # ── Print summary tables ──────────────────────────────────────────────
    for model_type in ["lr", "rf"]:
        print(f"\n{'=' * 80}")
        print(f"TABLE 2 — {model_type.upper()}")
        print(f"{'=' * 80}")

        header = f"  {'Feature Set':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUT':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for key, (_, label) in feature_sets.items():
            year_results = results[model_type][key]
            # AUT = mean of non-excluded years
            aut_scores = [v["f1_macro"] for y, v in year_results.items()
                          if not v["excluded"]]
            aut = np.mean(aut_scores)

            # Average metrics across all test years for the table
            all_acc = np.mean([v["accuracy"] for v in year_results.values()])
            all_prec = np.mean([v["precision"] for v in year_results.values()])
            all_rec = np.mean([v["recall"] for v in year_results.values()])
            all_f1 = np.mean([v["f1_macro"] for v in year_results.values()])

            # Per-year breakdown
            per_year = "  ".join(
                f"k={v['k']}:{v['f1_macro']:.3f}" for y, v in sorted(year_results.items())
            )

            print(f"  {label:<25} {all_acc:>6.3f} {all_prec:>6.3f} {all_rec:>6.3f} {all_f1:>6.3f} {aut:>7.4f}  | {per_year}")

    # ── Save JSON for dashboard ───────────────────────────────────────────
    output = {
        "meta": {
            "train_years": TRAIN_YEARS,
            "test_years": TEST_YEARS,
            "aut_exclude": AUT_EXCLUDE_YEARS,
            "rf_seeds": RF_SEEDS,
            "rf_n_runs": len(RF_SEEDS),
        },
        "results": {},
    }

    for model_type in ["lr", "rf"]:
        for key, (_, label) in feature_sets.items():
            out_key = f"{key}_{model_type}"
            output["results"][out_key] = {
                "label": f"{label} ({model_type.upper()})",
                "model": model_type,
                "feature_set": key,
                "years": {},
            }
            for test_year, metrics in sorted(results[model_type][key].items()):
                output["results"][out_key]["years"][str(test_year)] = metrics

    with open("paper_ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to paper_ablation_results.json")


if __name__ == "__main__":
    main()
