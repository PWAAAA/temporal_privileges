"""
top_features.py — Extract and display the top 15 most impactful features.

Usage:
  python top_features.py                          # RF, all feature sets, top 15
  python top_features.py --model lr               # Logistic Regression
  python top_features.py --top-n 20               # Top 20 instead of 15
  python top_features.py --feature-set count_free  # Only the combined set
"""
import argparse

from config import TRAIN_YEARS, LABEL_COL
from data import load_dataset
from feature_selection import build_feature_sets
from evaluate import make_model, get_top_features


def main():
    parser = argparse.ArgumentParser(description="Show top N most impactful features")
    parser.add_argument("--model", choices=["rf", "lr"], default="rf",
                        help="Model type: rf (Random Forest) or lr (Logistic Regression)")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Number of top features to show (default: 15)")
    parser.add_argument("--feature-set", choices=["static_flags", "temporal_only", "count_free"],
                        default=None,
                        help="Which feature set to analyze (default: all)")
    args = parser.parse_args()

    df, perm_cols, temp_cols = load_dataset(verbose=False)
    feature_sets, _ = build_feature_sets(df, perm_cols, temp_cols, prune=True)

    train_mask = df["year"].isin(TRAIN_YEARS)
    X_train_full = df.loc[train_mask]
    y_train = df.loc[train_mask, LABEL_COL]

    sets_to_run = {args.feature_set: feature_sets[args.feature_set]} if args.feature_set else feature_sets

    for key, (cols, label) in sets_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Top {args.top_n} features — {label} ({args.model.upper()})")
        print("=" * 60)

        model = make_model(args.model)
        model.fit(X_train_full[cols], y_train)
        top = get_top_features(model, cols, top_n=args.top_n)
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
