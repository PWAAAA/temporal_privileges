"""
plot_rf_comparison.py — Line graph: Static RF vs Best Temporal RF across k=1 to k=3.

Reads from paper_ablation_results.json (pre-generated).
Saves: rf_comparison.png in the same folder.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH  = os.path.join(SCRIPT_DIR, "paper_ablation_results.json")
OUT_PATH   = os.path.join(SCRIPT_DIR, "rf_comparison.png")

# ---------------------------------------------------------------------------
def load_series(results, key):
    """Return (ks, f1s, excl, aut) for a given results key."""
    rows = sorted(results[key]["years"].items(), key=lambda x: int(x[0]))
    ks, f1s, excl = zip(*((v["k"], v["f1_macro"], v["excluded"]) for _, v in rows))
    aut = np.mean([f for f, e in zip(f1s, excl) if not e])
    return list(ks), list(f1s), list(excl), aut


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)
    results = d["results"]

    ks_static, f1_static, excl_static, aut_static = load_series(results, "static_only_rf")
    ks_temp,   f1_temp,   excl_temp,   aut_temp   = load_series(results, "pfr_restrict_rf")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    BLUE  = "#2563EB"
    GREEN = "#16A34A"

    assert excl_static == excl_temp, "Exclusion masks differ between series"
    incl_idx = [i for i, e in enumerate(excl_static) if not e]

    series = [
        (ks_static, f1_static, BLUE,  "o", (-18, 7), f"Static RF  (AUT = {aut_static:.3f})"),
        (ks_temp,   f1_temp,   GREEN, "s", (  4, 7), f"Temporal RF — PFR + Restriction  (AUT = {aut_temp:.3f})"),
    ]

    for ks, f1s, color, marker, xytext, label in series:
        ks_incl  = [ks[i]  for i in incl_idx]
        f1s_incl = [f1s[i] for i in incl_idx]
        ax.plot(ks_incl, f1s_incl, color=color, linewidth=2.2,
                marker=marker, markersize=7, label=label)
        for k, f in zip(ks_incl, f1s_incl):
            ax.annotate(f"{f:.3f}", (k, f), textcoords="offset points",
                        xytext=xytext, fontsize=8, color=color)

    # Axes
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["k=1\n(2013)", "k=2\n(2014)", "k=3\n(2015)"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.60, 1.0)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_xlabel("Test Year", fontsize=11)
    ax.set_title("Random Forest: Static vs. Best Temporal Configuration\nPer-Year Macro F1",
                 fontsize=12, pad=12)

    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
