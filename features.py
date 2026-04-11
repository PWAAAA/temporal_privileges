"""
features.py — TTL-to-risk transform and per-app lifecycle feature engineering.
"""
import numpy as np
import pandas as pd

from config import TTL_SENTINEL


def ttl_to_risk(ttl_series, sentinel=TTL_SENTINEL):
    """Remap TTL to a monotonic [0, 1.5] risk score."""
    risk        = pd.Series(0.0, index=ttl_series.index, dtype=float)
    is_sentinel = ttl_series == sentinel
    is_past     = (ttl_series <= 0) & ~is_sentinel
    is_future   = (ttl_series >  0) & ~is_sentinel
    risk[is_past]   = 1.0 / (ttl_series[is_past].abs() + 1) + 0.5
    risk[is_future] = 1.0 / (ttl_series[is_future]      + 1)
    return risk


def per_flag_risk_columns(df, lifecycle_df, drift_flags):
    """Compute per-flag temporal risk encoding for high-drift permission flags.

    For each flag in drift_flags, produces a continuous column (pfr_{PERM}):
      - App doesn't use permission (flag=0) → 0
      - App uses it, no lifecycle event     → 1.0 (preserves original signal)
      - App uses it, has lifecycle event     → ttl_to_risk(event_year - app_year)

    This encodes BOTH which permission is used AND its temporal context,
    so the model sees that READ_SMS=1 in 2010 is different from READ_SMS=1 in 2017.
    """
    lc = lifecycle_df.set_index("permission")
    result = {}

    for perm in drift_flags:
        if perm not in df.columns:
            continue

        col = f"pfr_{perm}"
        vals = pd.Series(0.0, index=df.index)
        used = df[perm] == 1

        if not used.any():
            result[col] = vals
            continue

        if perm not in lc.index:
            vals[used] = 1.0
            result[col] = vals
            continue

        row = lc.loc[perm]
        restrict_yr = row.get("restrict_year", np.nan)
        deprecate_yr = row.get("deprecate_year", np.nan)
        event_year = restrict_yr if pd.notna(restrict_yr) else deprecate_yr

        if pd.notna(event_year):
            ttl = event_year - df.loc[used, "year"]
            vals[used] = ttl_to_risk(ttl)
        else:
            vals[used] = 1.0

        result[col] = vals

    return pd.DataFrame(result, index=df.index)


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
    "perm_age_std": 0.0,
    "perm_restrict_density": 0,
    "perm_unrestricted_old_ratio": 0.0,
    # --- new features ---
    "perm_announced_deprecate_ratio": 0.0,
    "perm_ttl_announced_deprecate_mean": np.nan,
    "perm_ttl_announced_deprecate_min": np.nan,
    "perm_risk_deprecate_max": 0.0,
    "perm_risk_deprecate_std": 0.0,
    "perm_new_perm_ratio": 0.0,
    "perm_age_relative_max": 0.0,
    # --- data-driven features ---
    "perm_count": 0,
    "perm_will_restrict_count": 0,
    "perm_will_restrict_ratio": 0.0,
    "perm_ever_lifecycle_count": 0,
    "perm_ever_lifecycle_ratio": 0.0,
    "perm_count_x_risk_deprecate": 0.0,
    "perm_count_x_will_restrict_ratio": 0.0,
}


def lifecycle_features(row, lifecycle_df, lifecycle_perm_cols):
    """Compute temporal lifecycle features for a single app row."""
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

    is_deprecated        = app_perms_lc["deprecate_year"].notna() & (app_perms_lc["deprecate_year"] <= app_year)
    years_to_deprecation = app_perms_lc.loc[app_perms_lc["deprecate_year"].notna(), "deprecate_year"] - app_year

    years_to_announced_restrict = app_perms_lc.loc[is_valid_announced_restrict, "announced_restriction_year"] - app_year

    min_years_to_restrict         = years_to_restriction.min() if len(years_to_restriction) > 0 else np.nan
    riskiest_perm_age             = np.nan
    riskiest_perm_age_at_restrict = np.nan
    if len(years_to_restriction) > 0:
        riskiest_perm_idx = years_to_restriction.idxmin()
        if riskiest_perm_idx in app_perms_lc.index and pd.notna(app_perms_lc.loc[riskiest_perm_idx, "intro_year"]):
            riskiest_perm_age_at_restrict = float(max(0, app_year - app_perms_lc.loc[riskiest_perm_idx, "intro_year"]))
            riskiest_perm_age             = riskiest_perm_age_at_restrict

    def _mean(series): return series.mean() if len(series) > 0 else np.nan
    def _min(series):  return series.min()  if len(series) > 0 else np.nan

    perm_age_std = float(perm_ages.std()) if len(perm_ages) > 1 else 0.0

    restrict_window = (
        app_perms_lc["restrict_year"].notna() &
        app_perms_lc["restrict_year"].between(app_year - 1, app_year + 2)
    )
    restrict_density = int(restrict_window.sum())

    # --- Announced deprecation (category 1) ---
    is_valid_announced_deprecate = (
        app_perms_lc["announced_deprecation_year"].notna() &
        (app_perms_lc["announced_deprecation_year"] <= app_year) &
        (
            app_perms_lc["deprecate_year"].isna() |
            (app_perms_lc["deprecate_year"] > app_year)
        )
    )
    years_to_announced_deprecate = (
        app_perms_lc.loc[is_valid_announced_deprecate, "announced_deprecation_year"] - app_year
    )

    # --- Per-permission deprecation risk (category 2) ---
    has_deprecate_event = app_perms_lc["deprecate_year"].notna()
    per_perm_dep_ttl = (app_perms_lc.loc[has_deprecate_event, "deprecate_year"] - app_year)
    per_perm_dep_risk = ttl_to_risk(per_perm_dep_ttl) if len(per_perm_dep_ttl) > 0 else pd.Series(dtype=float)
    risk_deprecate_max = float(per_perm_dep_risk.max()) if len(per_perm_dep_risk) > 0 else 0.0
    risk_deprecate_std = float(per_perm_dep_risk.std()) if len(per_perm_dep_risk) > 1 else 0.0

    # --- Relative age features (category 3) ---
    NEW_PERM_THRESHOLD = 2
    new_perm_ratio = float((perm_ages <= NEW_PERM_THRESHOLD).mean()) if len(perm_ages) > 0 else 0.0
    ecosystem_age = max(app_year - 2008, 1)  # 2008 = earliest intro_year
    age_relative_max = float(perm_ages.max() / ecosystem_age) if len(perm_ages) > 0 else 0.0

    # --- Data-driven features ---
    n_perms = len(used_perms)

    # Permissions that WILL be restricted at any point (past, present, or future)
    has_any_restrict_ever = app_perms_lc["restrict_year"].notna()
    will_restrict_count = int(has_any_restrict_ever.sum())
    will_restrict_ratio = float(will_restrict_count / n_perms) if n_perms > 0 else 0.0

    # Permissions with ANY lifecycle event (restrict, deprecate, or announced)
    has_any_lifecycle = (
        app_perms_lc["restrict_year"].notna() |
        app_perms_lc["deprecate_year"].notna() |
        app_perms_lc["announced_restriction_year"].notna() |
        app_perms_lc["announced_deprecation_year"].notna()
    )
    ever_lifecycle_count = int(has_any_lifecycle.sum())
    ever_lifecycle_ratio = float(ever_lifecycle_count / n_perms) if n_perms > 0 else 0.0

    # Interactions: count amplifies risk signal
    mean_dep_risk = float(per_perm_dep_risk.mean()) if len(per_perm_dep_risk) > 0 else 0.0
    count_x_risk_deprecate = n_perms * mean_dep_risk
    count_x_will_restrict_ratio = n_perms * will_restrict_ratio

    OLD_AGE_THRESHOLD = 5
    is_old = perm_ages > OLD_AGE_THRESHOLD
    if is_old.any():
        old_idx = perm_ages[is_old].index
        old_not_restricted = ~(
            is_past_restricted.reindex(old_idx, fill_value=False) |
            is_deprecated.reindex(old_idx, fill_value=False)
        )
        unrestricted_old_ratio = float(old_not_restricted.mean())
    else:
        unrestricted_old_ratio = 0.0

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
        "perm_age_std":                     perm_age_std,
        "perm_restrict_density":            restrict_density,
        "perm_unrestricted_old_ratio":      unrestricted_old_ratio,
        # --- new features ---
        "perm_announced_deprecate_ratio":   is_valid_announced_deprecate.mean(),
        "perm_ttl_announced_deprecate_mean": _mean(years_to_announced_deprecate),
        "perm_ttl_announced_deprecate_min": _min(years_to_announced_deprecate),
        "perm_risk_deprecate_max":          risk_deprecate_max,
        "perm_risk_deprecate_std":          risk_deprecate_std,
        "perm_new_perm_ratio":              new_perm_ratio,
        "perm_age_relative_max":            age_relative_max,
        # --- data-driven features ---
        "perm_count":                       n_perms,
        "perm_will_restrict_count":         will_restrict_count,
        "perm_will_restrict_ratio":         will_restrict_ratio,
        "perm_ever_lifecycle_count":        ever_lifecycle_count,
        "perm_ever_lifecycle_ratio":        ever_lifecycle_ratio,
        "perm_count_x_risk_deprecate":      count_x_risk_deprecate,
        "perm_count_x_will_restrict_ratio": count_x_will_restrict_ratio,
    })
