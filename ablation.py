"""
ablation.py — Forward-year temporal ablation experiment and AUT summary.
"""
import numpy as np

from config import TRAIN_YEARS, TEST_YEARS, AUT_EXCLUDE_YEARS, LABEL_COL, METRICS
from evaluate import evaluate


def run_ablation(df, feature_sets, model_type="rf"):
    """Run forward-year temporal ablation and print AUT summary."""
    results = {key: {} for key in feature_sets}
    auts    = {key: {m: [] for m in METRICS} for key in feature_sets}

    for test_year in TEST_YEARS:
        train_mask = df["year"].isin(TRAIN_YEARS)
        test_mask  = df["year"] == test_year
        if not test_mask.any():
            print(f"[SKIP] No data for {test_year}")
            continue

        years_ahead      = test_year - max(TRAIN_YEARS)
        exclude_from_aut = test_year in AUT_EXCLUDE_YEARS
        note = " [EXCLUDED FROM AUT]" if exclude_from_aut else ""

        print(f"\n{'=' * 60}")
        print(f"Train: {min(TRAIN_YEARS)}-{max(TRAIN_YEARS)} | Test: {test_year} (k={years_ahead}){note}")
        print(f"Train: {train_mask.sum()} samples  |  Test: {test_mask.sum()} samples")
        print(f"Train malware: {df.loc[train_mask, LABEL_COL].mean():.2%}  |  "
              f"Test malware: {df.loc[test_mask, LABEL_COL].mean():.2%}")

        y_train = df.loc[train_mask, LABEL_COL]
        y_test  = df.loc[test_mask,  LABEL_COL]

        for model_key, (feature_cols, model_label) in feature_sets.items():
            fold_metrics = evaluate(
                df.loc[train_mask, feature_cols], y_train,
                df.loc[test_mask,  feature_cols], y_test,
                model_label=f"{model_label} (k={years_ahead})",
                model_type=model_type,
                show_importances=(years_ahead == 1),
            )
            results[model_key][test_year] = fold_metrics
            if not exclude_from_aut:
                for m in METRICS:
                    auts[model_key][m].append(fold_metrics[m])

    # -- AUT summary --------------------------------------------------------
    first_key = next(iter(results))
    evaluated_years = [y for y in TEST_YEARS if y in results[first_key]]
    reliable_years  = [y for y in evaluated_years if y not in AUT_EXCLUDE_YEARS]
    excluded_str    = ", ".join(str(y) for y in AUT_EXCLUDE_YEARS)

    for metric_key, metric_label in [("f1_macro", "Macro F1"),
                                      ("bal_acc", "Balanced Accuracy"),
                                      ("f1_weighted", "Weighted F1")]:
        print(f"\n{'=' * 60}")
        print(f"Metric: {metric_label}  (AUT over k=1-{len(reliable_years)}, excluding {excluded_str})")
        header = (f"  {'Model':<28}"
                  + "".join(f"  k={y - max(TRAIN_YEARS)}" for y in evaluated_years)
                  + "    AUT")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model_key, (_, model_label) in feature_sets.items():
            scores = [results[model_key][y][metric_key] for y in evaluated_years]
            aut    = np.mean(auts[model_key][metric_key])
            print(f"  {model_label:<28}"
                  + "".join(f"  {s:.3f}" for s in scores)
                  + f"  {aut:.4f}")

    return results, auts
