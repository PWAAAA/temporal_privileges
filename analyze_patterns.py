"""
analyze_patterns.py -- Reverse-engineer what separates malware from benign
across time periods. Look at raw permission usage, lifecycle properties,
and any structural patterns that hold across train and test years.
"""
import numpy as np
import pandas as pd
from config import TRAIN_YEARS, TEST_YEARS, LABEL_COL, LIFECYCLE_FILE, LIFECYCLE_YEAR_COLS

# Load raw data (before feature engineering, so we can see everything)
from data import load_dataset
df, perm_cols, temp_cols = load_dataset(verbose=False)

lifecycle_df = pd.read_csv(LIFECYCLE_FILE)
for col in LIFECYCLE_YEAR_COLS:
    if col in lifecycle_df.columns:
        lifecycle_df[col] = pd.to_numeric(lifecycle_df[col], errors="coerce")

print(f"Dataset: {len(df)} apps, {len(perm_cols)} permission flags")
print(f"Train years: {TRAIN_YEARS}, Test years: {TEST_YEARS}")

# ======================================================================
# 1. Permission count distributions (malware vs benign, by year)
# ======================================================================
print(f"\n{'='*70}")
print("1. Permission count distributions")
print(f"{'='*70}")

df["perm_count"] = df[perm_cols].sum(axis=1)
for year in TRAIN_YEARS[-2:] + TEST_YEARS:
    mask = df["year"] == year
    for label, name in [(0, "benign"), (1, "malware")]:
        subset = df.loc[mask & (df[LABEL_COL] == label), "perm_count"]
        if len(subset) > 0:
            print(f"  {year} {name:>7}: n={len(subset):5d}  "
                  f"mean={subset.mean():.1f}  med={subset.median():.0f}  "
                  f"std={subset.std():.1f}  max={subset.max():.0f}")

# ======================================================================
# 2. Which permissions have the biggest malware/benign usage gap
#    IN TRAINING, and does that gap persist in test years?
# ======================================================================
print(f"\n{'='*70}")
print("2. Top permissions by malware-benign usage gap (train vs test)")
print(f"{'='*70}")

train_mask = df["year"].isin(TRAIN_YEARS)
train_mal = df.loc[train_mask & (df[LABEL_COL] == 1)]
train_ben = df.loc[train_mask & (df[LABEL_COL] == 0)]

gaps = {}
for p in perm_cols:
    mal_rate = train_mal[p].mean()
    ben_rate = train_ben[p].mean()
    gap = mal_rate - ben_rate
    gaps[p] = (mal_rate, ben_rate, gap)

# Top 20 most malware-indicative permissions
sorted_gaps = sorted(gaps.items(), key=lambda x: x[1][2], reverse=True)
print(f"\n  Top 20 malware-over-benign permissions (training):")
print(f"  {'Permission':<40} {'Mal%':>6} {'Ben%':>6} {'Gap':>6} | ", end="")
for y in TEST_YEARS[:3]:
    print(f"  {y} Gap", end="")
print()
print(f"  {'-'*100}")

for p, (mr, br, g) in sorted_gaps[:20]:
    print(f"  {p:<40} {mr:.3f}  {br:.3f}  {g:+.3f} | ", end="")
    for y in TEST_YEARS[:3]:
        ym = df.loc[(df["year"] == y) & (df[LABEL_COL] == 1)]
        yb = df.loc[(df["year"] == y) & (df[LABEL_COL] == 0)]
        if len(ym) > 0 and len(yb) > 0:
            test_gap = ym[p].mean() - yb[p].mean()
            print(f"  {test_gap:+.3f}  ", end="")
        else:
            print(f"    N/A  ", end="")
    print()

# Top 20 most benign-indicative
print(f"\n  Top 20 benign-over-malware permissions (training):")
print(f"  {'Permission':<40} {'Mal%':>6} {'Ben%':>6} {'Gap':>6} | ", end="")
for y in TEST_YEARS[:3]:
    print(f"  {y} Gap", end="")
print()
print(f"  {'-'*100}")

for p, (mr, br, g) in sorted_gaps[-20:]:
    print(f"  {p:<40} {mr:.3f}  {br:.3f}  {g:+.3f} | ", end="")
    for y in TEST_YEARS[:3]:
        ym = df.loc[(df["year"] == y) & (df[LABEL_COL] == 1)]
        yb = df.loc[(df["year"] == y) & (df[LABEL_COL] == 0)]
        if len(ym) > 0 and len(yb) > 0:
            test_gap = ym[p].mean() - yb[p].mean()
            print(f"  {test_gap:+.3f}  ", end="")
        else:
            print(f"    N/A  ", end="")
    print()

# ======================================================================
# 3. Lifecycle properties of malware-preferred vs benign-preferred perms
# ======================================================================
print(f"\n{'='*70}")
print("3. Lifecycle properties of malware vs benign preferred permissions")
print(f"{'='*70}")

# Classify permissions by their training-era gap
mal_preferred = [p for p, (_, _, g) in sorted_gaps if g > 0.05]
ben_preferred = [p for p, (_, _, g) in sorted_gaps if g < -0.05]

print(f"\n  Malware-preferred perms (gap > 0.05): {len(mal_preferred)}")
print(f"  Benign-preferred perms (gap < -0.05): {len(ben_preferred)}")

for label, perm_list in [("Malware-preferred", mal_preferred), ("Benign-preferred", ben_preferred)]:
    lc_match = lifecycle_df[lifecycle_df["permission"].isin(perm_list)]
    print(f"\n  {label} ({len(perm_list)} perms, {len(lc_match)} in lifecycle table):")
    if len(lc_match) > 0:
        has_restrict = lc_match["restrict_year"].notna().mean()
        has_deprecate = lc_match["deprecate_year"].notna().mean()
        has_announced = lc_match["announced_restriction_year"].notna().mean()
        mean_intro = lc_match["intro_year"].mean()
        print(f"    Mean intro year:      {mean_intro:.1f}")
        print(f"    % with restriction:   {has_restrict:.1%}")
        print(f"    % with deprecation:   {has_deprecate:.1%}")
        print(f"    % with announced:     {has_announced:.1%}")
        print(f"    Perms: {list(lc_match['permission'].values)}")

# ======================================================================
# 4. Permission "staleness" -- are malware apps using older/newer perms?
# ======================================================================
print(f"\n{'='*70}")
print("4. Permission intro year profile (malware vs benign by era)")
print(f"{'='*70}")

# For each app, compute the distribution of intro_years of its permissions
intro_lookup = dict(zip(lifecycle_df["permission"], lifecycle_df["intro_year"]))

for year in TRAIN_YEARS[-2:] + TEST_YEARS[:3]:
    print(f"\n  Year {year}:")
    for label, name in [(1, "malware"), (0, "benign")]:
        subset = df.loc[(df["year"] == year) & (df[LABEL_COL] == label)]
        if len(subset) == 0:
            continue
        intro_years_all = []
        for _, row in subset.iterrows():
            used = [p for p in perm_cols if row[p] == 1]
            intros = [intro_lookup[p] for p in used if p in intro_lookup and pd.notna(intro_lookup[p])]
            if intros:
                intro_years_all.append(np.mean(intros))
        if intro_years_all:
            arr = np.array(intro_years_all)
            print(f"    {name:>7}: mean_intro={arr.mean():.2f}  std={arr.std():.2f}  "
                  f"% pre-2010={np.mean(arr < 2010):.2%}")

# ======================================================================
# 5. Permission co-occurrence: do malware apps use specific combos?
# ======================================================================
print(f"\n{'='*70}")
print("5. Permission co-occurrence patterns")
print(f"{'='*70}")

# Find permission pairs with highest differential co-occurrence
# Use top malware-indicative perms to look for combos
top_mal_perms = [p for p, (_, _, g) in sorted_gaps[:30] if g > 0.02]
print(f"  Checking pairs among top {len(top_mal_perms)} malware-indicative permissions...")

pair_diffs = {}
for i, p1 in enumerate(top_mal_perms):
    for p2 in top_mal_perms[i+1:]:
        co_mal = ((train_mal[p1] == 1) & (train_mal[p2] == 1)).mean()
        co_ben = ((train_ben[p1] == 1) & (train_ben[p2] == 1)).mean()
        pair_diffs[(p1, p2)] = (co_mal, co_ben, co_mal - co_ben)

top_pairs = sorted(pair_diffs.items(), key=lambda x: x[1][2], reverse=True)[:15]
print(f"\n  Top 15 malware co-occurrence pairs:")
print(f"  {'Perm1':<30} {'Perm2':<30} {'Mal%':>6} {'Ben%':>6} {'Gap':>6}")
print(f"  {'-'*108}")
for (p1, p2), (cm, cb, g) in top_pairs:
    print(f"  {p1:<30} {p2:<30} {cm:.3f}  {cb:.3f}  {g:+.3f}")

# ======================================================================
# 6. Structural: how many "dangerous" perms does each app have?
# ======================================================================
print(f"\n{'='*70}")
print("6. Dangerous permission concentration")
print(f"{'='*70}")

# Android dangerous permissions (runtime permissions post API 23)
# Use lifecycle: restricted permissions as a proxy
restricted_perms = lifecycle_df.loc[lifecycle_df["restrict_year"].notna(), "permission"].tolist()
restricted_in_data = [p for p in restricted_perms if p in perm_cols]
deprecated_perms = lifecycle_df.loc[lifecycle_df["deprecate_year"].notna(), "permission"].tolist()
deprecated_in_data = [p for p in deprecated_perms if p in perm_cols]

print(f"  Restricted perms in data: {len(restricted_in_data)}")
print(f"  Deprecated perms in data: {len(deprecated_in_data)}")

for year in TRAIN_YEARS[-2:] + TEST_YEARS[:3]:
    print(f"\n  Year {year}:")
    for label, name in [(1, "malware"), (0, "benign")]:
        subset = df.loc[(df["year"] == year) & (df[LABEL_COL] == label)]
        if len(subset) == 0:
            continue
        r_count = subset[restricted_in_data].sum(axis=1)
        d_count = subset[deprecated_in_data].sum(axis=1)
        total = subset[perm_cols].sum(axis=1)
        r_ratio = r_count / total.replace(0, np.nan)
        d_ratio = d_count / total.replace(0, np.nan)
        print(f"    {name:>7}: restrict_count={r_count.mean():.2f}  "
              f"restrict_ratio={r_ratio.mean():.3f}  "
              f"deprec_count={d_count.mean():.2f}  "
              f"deprec_ratio={d_ratio.mean():.3f}  "
              f"total_perms={total.mean():.1f}")

# ======================================================================
# 7. Gap stability: which individual permissions have STABLE gaps?
# ======================================================================
print(f"\n{'='*70}")
print("7. Permissions with most stable malware/benign gap across ALL years")
print(f"{'='*70}")

all_years = TRAIN_YEARS[-3:] + TEST_YEARS[:3]
stable_perms = []
for p in perm_cols:
    year_gaps = []
    for y in all_years:
        ym = df.loc[(df["year"] == y) & (df[LABEL_COL] == 1), p]
        yb = df.loc[(df["year"] == y) & (df[LABEL_COL] == 0), p]
        if len(ym) > 10 and len(yb) > 10:
            year_gaps.append(ym.mean() - yb.mean())
    if len(year_gaps) >= 5:
        mean_gap = np.mean(year_gaps)
        std_gap = np.std(year_gaps)
        # Stability = consistent direction and magnitude
        if abs(mean_gap) > 0.02:
            stable_perms.append((p, mean_gap, std_gap, std_gap / abs(mean_gap), year_gaps))

# Sort by coefficient of variation (lower = more stable)
stable_perms.sort(key=lambda x: x[3])
print(f"\n  Top 20 most stable malware-indicative permissions:")
print(f"  {'Permission':<40} {'MeanGap':>8} {'StdGap':>8} {'CV':>6} | Year gaps")
print(f"  {'-'*110}")
for p, mg, sg, cv, yg in stable_perms[:20]:
    gaps_str = "  ".join(f"{g:+.3f}" for g in yg)
    print(f"  {p:<40} {mg:+.4f}   {sg:.4f}  {cv:.3f} | {gaps_str}")

print(f"\n  Top 20 most stable benign-indicative permissions:")
stable_perms_rev = [x for x in stable_perms if x[1] < 0]
stable_perms_rev.sort(key=lambda x: x[3])
for p, mg, sg, cv, yg in stable_perms_rev[:20]:
    gaps_str = "  ".join(f"{g:+.3f}" for g in yg)
    print(f"  {p:<40} {mg:+.4f}   {sg:.4f}  {cv:.3f} | {gaps_str}")

# ======================================================================
# 8. Meta-feature: count of "temporally stable" malware indicators
# ======================================================================
print(f"\n{'='*70}")
print("8. Meta-feature exploration")
print(f"{'='*70}")

# How many of the top-stable malware perms does each app use?
top_stable_mal = [p for p, mg, sg, cv, yg in stable_perms[:20] if mg > 0]
top_stable_ben = [p for p, mg, sg, cv, yg in stable_perms if mg < 0][:20]

print(f"  Stable malware indicators: {len(top_stable_mal)}")
print(f"  Stable benign indicators: {len(top_stable_ben)}")

for year in TRAIN_YEARS[-2:] + TEST_YEARS[:3]:
    print(f"\n  Year {year}:")
    for label, name in [(1, "malware"), (0, "benign")]:
        subset = df.loc[(df["year"] == year) & (df[LABEL_COL] == label)]
        if len(subset) == 0:
            continue
        mal_ind_count = subset[top_stable_mal].sum(axis=1) if top_stable_mal else pd.Series(0)
        ben_ind_count = subset[top_stable_ben].sum(axis=1) if top_stable_ben else pd.Series(0)
        total = subset[perm_cols].sum(axis=1)
        ratio = (mal_ind_count / total.replace(0, np.nan)) if len(top_stable_mal) > 0 else pd.Series(0)
        print(f"    {name:>7}: stable_mal_count={mal_ind_count.mean():.2f}  "
              f"stable_ben_count={ben_ind_count.mean():.2f}  "
              f"mal_ratio={ratio.mean():.3f}")
