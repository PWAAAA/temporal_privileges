"""
longRF_v8.py — Android permission temporal drift study
-------------------------------------------------------
3-way ablation RF experiment: does adding temporal permission lifecycle
features improve concept-drift resilience in Android malware detection?

Feature sets
  [A] Static permission flags only
  [B] Temporal lifecycle features only
  [C] Combined (A + B, no raw syscall counts)

Evaluation: forward-year temporal holdout, AUT over reliable test years.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CSV_FILE       = f"{BASE_DIR}/kronodroid-2021-emu-v1.csv"
LIFECYCLE_FILE = f"{BASE_DIR}/permission_lifecycle_v3.csv"

LABEL_COL = "Malware"
DATE_COL  = "FirstModDate"
MIN_YEAR  = 2008

DROP_COLS = ["Package", "sha256", "MalFamily", "Detection_Ratio",
             "Scanners", "LastModDate"]

TRAIN_YEARS       = list(range(2008, 2015))
TEST_YEARS        = [2015, 2016, 2017, 2018]
AUT_EXCLUDE_YEARS = [2018]   # k=4: only 30 benign samples — macro F1 is noise

TTL_SENTINEL = 99            # placeholder meaning "no restriction data"

LIFECYCLE_YEAR_COLS = ["intro_year", "restrict_year",
                       "deprecate_year", "announced_restriction_year"]

# ── TTL → Risk transformation ───────────────────────────────────────────────
# Remaps TTL to a monotonic [0, 1.5] risk score:
#   sentinel (99) → 0.0    no restriction data
#   large positive → low   restriction far away
#   small positive → higher restriction imminent
#   0             → 1.0    restriction happening now
#   negative      → >1.0   restriction already passed (still suspicious)
# Asymmetric by design: past-restricted permissions score higher than
# equivalent-distance future ones — malware clusters around restriction events.
def ttl_to_risk(ttl_series, sentinel=TTL_SENTINEL):
    risk        = pd.Series(0.0, index=ttl_series.index, dtype=float)
    is_sentinel = ttl_series == sentinel
    is_past     = (ttl_series <= 0) & ~is_sentinel
    is_future   = (ttl_series >  0) & ~is_sentinel
    risk[is_past]   = 1.0 / (ttl_series[is_past].abs() + 1) + 0.5
    risk[is_future] = 1.0 / (ttl_series[is_future]      + 1)
    return risk

# ── Lifecycle feature engineering ───────────────────────────────────────────
_ZERO_FEATS = {
    "perm_age_mean": 0.0, "perm_age_max": 0.0, "perm_age_min": 0.0,
    "perm_restricted_ratio": 0.0,      "perm_deprecated_ratio": 0.0,
    "perm_near_restrict_ratio": 0.0,
    "perm_ttl_restrict_mean": np.nan,  "perm_ttl_restrict_min": np.nan,
    "perm_ttl_deprecate_mean": np.nan, "perm_ttl_deprecate_min": np.nan,
    "perm_announced_restrict_ratio": 0.0,
    "perm_ttl_announced_restrict_mean": np.nan,
    "perm_ttl_announced_restrict_min": np.nan,
    "perm_restricted_count": 0, "perm_near_restrict_count": 0,
    "perm_deprecated_count": 0,
    "perm_has_any_restricted": 0, "perm_has_any_near_restrict": 0,
    "perm_worst_age": 0.0,
    "perm_worst_ttl_restrict": np.nan,
    "perm_worst_perm_age_at_restrict": np.nan,
}

def lifecycle_features(row, lifecycle_df, lifecycle_perm_cols):
    app_year   = row["year"]
    used_perms = [p for p in lifecycle_perm_cols if row[p] == 1]
    if not used_perms:
        return pd.Series(_ZERO_FEATS)

    app_perms_lc = lifecycle_df[lifecycle_df["permission"].isin(used_perms)].copy()
    perm_ages    = (app_year - app_perms_lc["intro_year"]).clip(lower=0)

    is_past_restricted = (
        app_perms_lc["restrict_year"].notna() &
        (app_perms_lc["restrict_year"] <= app_year)
    )

    # Announcement is valid only when both dates are known, correctly ordered,
    # announcement has occurred, and restriction is still in the future.
    is_valid_announced_restrict = (
        app_perms_lc["announced_restriction_year"].notna() &
        app_perms_lc["restrict_year"].notna() &
        (app_perms_lc["announced_restriction_year"] < app_perms_lc["restrict_year"]) &
        (app_perms_lc["announced_restriction_year"] <= app_year) &
        (app_perms_lc["restrict_year"]              >  app_year)
    )

    has_known_restrict_event = is_past_restricted | is_valid_announced_restrict
    years_to_restriction     = app_perms_lc.loc[has_known_restrict_event, "restrict_year"] - app_year
    is_near_restriction      = is_valid_announced_restrict & (app_perms_lc["restrict_year"] - app_year).between(0, 2)

    is_deprecated       = app_perms_lc["deprecate_year"].notna() & (app_perms_lc["deprecate_year"] <= app_year)
    years_to_deprecation = app_perms_lc.loc[app_perms_lc["deprecate_year"].notna(), "deprecate_year"] - app_year

    # Scoped to is_valid_announced_restrict only so ratio and TTL describe the same population.
    years_to_announced_restrict = app_perms_lc.loc[is_valid_announced_restrict, "announced_restriction_year"] - app_year

    # Riskiest = permission closest to / most past its restriction event.
    # riskiest_perm_age is the age of that permission, not the oldest permission.
    min_years_to_restrict       = years_to_restriction.min() if len(years_to_restriction) > 0 else np.nan
    riskiest_perm_age           = np.nan
    riskiest_perm_age_at_restrict = np.nan
    if len(years_to_restriction) > 0:
        riskiest_perm_idx = years_to_restriction.idxmin()
        if riskiest_perm_idx in app_perms_lc.index and pd.notna(app_perms_lc.loc[riskiest_perm_idx, "intro_year"]):
            riskiest_perm_age_at_restrict = float(max(0, app_year - app_perms_lc.loc[riskiest_perm_idx, "intro_year"]))
            riskiest_perm_age             = riskiest_perm_age_at_restrict

    def _mean(series): return series.mean() if len(series) > 0 else np.nan
    def _min(series):  return series.min()  if len(series) > 0 else np.nan

    return pd.Series({
        "perm_age_mean":                    perm_ages.mean(),
        "perm_age_max":                     perm_ages.max(),
        "perm_age_min":                     perm_ages.min(),
        "perm_restricted_ratio":            is_past_restricted.mean(),
        "perm_deprecated_ratio":            is_deprecated.mean(),
        "perm_near_restrict_ratio":         is_near_restriction.mean(),
        "perm_ttl_restrict_mean":           _mean(years_to_restriction),
        "perm_ttl_restrict_min":            _min(years_to_restriction),
        "perm_ttl_deprecate_mean":          _mean(years_to_deprecation),
        "perm_ttl_deprecate_min":           _min(years_to_deprecation),
        "perm_announced_restrict_ratio":    is_valid_announced_restrict.mean(),
        "perm_ttl_announced_restrict_mean": _mean(years_to_announced_restrict),
        "perm_ttl_announced_restrict_min":  _min(years_to_announced_restrict),
        "perm_restricted_count":            int(is_past_restricted.sum()),
        "perm_near_restrict_count":         int(is_near_restriction.sum()),
        "perm_deprecated_count":            int(is_deprecated.sum()),
        "perm_has_any_restricted":          int(is_past_restricted.any()),
        "perm_has_any_near_restrict":       int(is_near_restriction.any()),
        "perm_worst_age":                   riskiest_perm_age if not np.isnan(riskiest_perm_age) else 0.0,
        "perm_worst_ttl_restrict":          min_years_to_restrict,
        "perm_worst_perm_age_at_restrict":  riskiest_perm_age_at_restrict,
    })

# ── Load & clean dataset ────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_FILE)
df["year"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.year
df = df.drop(columns=DROP_COLS + [DATE_COL], errors="ignore")
df = df[df["year"] >= MIN_YEAR].dropna(subset=["year"])
df["year"] = df["year"].astype(int)
print("App counts per year:\n", df["year"].value_counts().sort_index(), "\n")

# ── Load lifecycle table ────────────────────────────────────────────────────
lifecycle_df = pd.read_csv(LIFECYCLE_FILE)
for col in LIFECYCLE_YEAR_COLS:
    if col in lifecycle_df.columns:
        lifecycle_df[col] = pd.to_numeric(lifecycle_df[col], errors="coerce")

lifecycle_perm_cols = [c for c in df.columns if c in lifecycle_df["permission"].values]
print(f"Permissions matched to lifecycle table: {len(lifecycle_perm_cols)}/{len(lifecycle_df)}")

# ── Compute lifecycle features ──────────────────────────────────────────────
print("Computing lifecycle features (may take ~1 min)...")
lifecycle_feats = df.apply(
    lambda row: lifecycle_features(row, lifecycle_df, lifecycle_perm_cols), axis=1
)
df = pd.concat([df, lifecycle_feats], axis=1).copy()

# Fill NaN TTL cols with sentinel → transform to risk scores → drop raw TTL
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

# Interaction features
# riskiest_perm_age × near_restrict: how long the riskiest permission has been
# in use relative to its upcoming restriction (avoids dilution from age_mean).
df["perm_age_x_near_restrict"]       = df["perm_worst_age"] * df["perm_near_restrict_ratio"]
df["perm_worst_age_x_risk_restrict"] = df["perm_worst_perm_age_at_restrict"] * df["perm_risk_worst_restrict"]

# ── Whitelist columns ───────────────────────────────────────────────────────
TEMPORAL_COLS = [
    "perm_age_mean", "perm_age_max", "perm_age_min",
    "perm_restricted_ratio",  "perm_deprecated_ratio",
    "perm_near_restrict_ratio",
    "perm_risk_restrict_mean",  "perm_risk_restrict_min",
    "perm_risk_deprecate_mean", "perm_risk_deprecate_min",
    "perm_announced_restrict_ratio",
    "perm_risk_announced_restrict_mean", "perm_risk_announced_restrict_min",
    "perm_restricted_count", "perm_near_restrict_count", "perm_deprecated_count",
    "perm_has_any_restricted", "perm_has_any_near_restrict",
    "perm_worst_age", "perm_risk_worst_restrict", "perm_worst_perm_age_at_restrict",
    "perm_age_x_near_restrict", "perm_worst_age_x_risk_restrict",
]

permission_flag_cols = [c for c in df.columns if c.isupper()]
temporal_cols        = [c for c in TEMPORAL_COLS if c in df.columns]
df = df[[c for c in permission_flag_cols + temporal_cols + [LABEL_COL, "year"] if c in df.columns]]

print(f"\nColumns after whitelist: {len(df.columns)} "
      f"({len(permission_flag_cols)} permission flags, {len(temporal_cols)} temporal)\n")

# ── Coverage diagnostic ─────────────────────────────────────────────────────
print("Lifecycle feature coverage (% non-zero rows):")
for col in temporal_cols:
    print(f"  {col:<45} {(df[col] != 0).mean():.2%}")

# ── Feature sets ────────────────────────────────────────────────────────────
FEATURE_SETS = {
    "static_flags":  (permission_flag_cols,                       "[A] Static flags only"),
    "temporal_only": (temporal_cols,                              "[B] Temporal only"),
    "count_free":    (permission_flag_cols + temporal_cols,       "[C] Count-free full model"),
}

print("\nFeature sets:")
for feature_cols, model_label in FEATURE_SETS.values():
    print(f"  {model_label:<28}: {len(feature_cols)} features")

# ── Evaluation helper ───────────────────────────────────────────────────────
METRICS = ["f1_weighted", "f1_macro", "bal_acc"]

def evaluate(X_train, y_train, X_test, y_test, model_label, show_importances=False):
    model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   n_jobs=-1, class_weight="balanced",
                                   max_features=0.5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    f1_weighted  = f1_score(y_test, predictions, average="weighted")
    f1_macro     = f1_score(y_test, predictions, average="macro")
    balanced_acc = balanced_accuracy_score(y_test, predictions)

    print(f"\n── {model_label} {'─' * (45 - len(model_label))}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}  "
          f"Balanced: {balanced_acc:.4f}  "
          f"Precision: {precision_score(y_test, predictions, average='binary'):.4f}  "
          f"Recall: {recall_score(y_test, predictions, average='binary'):.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}  macro: {f1_macro:.4f}")
    print(classification_report(y_test, predictions))

    if show_importances:
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        print("Top 15 features:\n", feature_importances.nlargest(15).to_string())

    return {"f1_weighted": f1_weighted, "f1_macro": f1_macro, "bal_acc": balanced_acc}

# ── Forward-year temporal ablation ──────────────────────────────────────────
results = {model_key: {} for model_key in FEATURE_SETS}
auts    = {model_key: {metric_key: [] for metric_key in METRICS} for model_key in FEATURE_SETS}

for test_year in TEST_YEARS:
    train_mask = df["year"].isin(TRAIN_YEARS)
    test_mask  = df["year"] == test_year
    if not test_mask.any():
        print(f"[SKIP] No data for {test_year}")
        continue

    years_ahead       = test_year - max(TRAIN_YEARS)
    exclude_from_aut  = test_year in AUT_EXCLUDE_YEARS
    aut_exclusion_note = " [EXCLUDED FROM AUT — low benign count]" if exclude_from_aut else ""

    print(f"\n{'=' * 60}")
    print(f"Train: {min(TRAIN_YEARS)}–{max(TRAIN_YEARS)} | Test: {test_year} (k={years_ahead}){aut_exclusion_note}")
    print(f"Train: {train_mask.sum()} samples  |  Test: {test_mask.sum()} samples")
    print(f"Train malware: {df.loc[train_mask, LABEL_COL].mean():.2%}  |  "
          f"Test malware: {df.loc[test_mask, LABEL_COL].mean():.2%}")

    y_train = df.loc[train_mask, LABEL_COL]
    y_test  = df.loc[test_mask,  LABEL_COL]

    for model_key, (feature_cols, model_label) in FEATURE_SETS.items():
        fold_metrics = evaluate(
            df.loc[train_mask, feature_cols], y_train,
            df.loc[test_mask,  feature_cols], y_test,
            model_label=f"{model_label} (k={years_ahead})",
            show_importances=(years_ahead == 1),
        )
        results[model_key][test_year] = fold_metrics
        if not exclude_from_aut:
            for metric_key in METRICS:
                auts[model_key][metric_key].append(fold_metrics[metric_key])

# ── AUT summary ─────────────────────────────────────────────────────────────
evaluated_test_years = [y for y in TEST_YEARS if y in results["static_flags"]]
reliable_test_years  = [y for y in evaluated_test_years if y not in AUT_EXCLUDE_YEARS]
excluded_years_str   = ", ".join(str(y) for y in AUT_EXCLUDE_YEARS)

for metric_key, metric_label in [("f1_macro",    "Macro F1"),
                                   ("bal_acc",     "Balanced Accuracy"),
                                   ("f1_weighted", "Weighted F1")]:
    print(f"\n{'=' * 60}")
    print(f"Metric: {metric_label}  (AUT over k=1–{len(reliable_test_years)}, excluding {excluded_years_str})")
    header = (f"  {'Model':<28}"
              + "".join(f"  k={y - max(TRAIN_YEARS)}" for y in evaluated_test_years)
              + "    AUT")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for model_key, (_, model_label) in FEATURE_SETS.items():
        per_year_scores = [results[model_key][y][metric_key] for y in evaluated_test_years]
        aut_score       = np.mean(auts[model_key][metric_key])
        print(f"  {model_label:<28}"
              + "".join(f"  {score:.3f}" for score in per_year_scores)
              + f"  {aut_score:.4f}")
