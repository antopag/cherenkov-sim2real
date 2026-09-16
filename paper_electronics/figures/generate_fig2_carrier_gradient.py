"""Figure 2: Main result — carrier gradient of DAUC."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, DOUBLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()

CARRIERS_ORDER = ["lr", "lgbm", "rf", "et", "mlp"]
CARRIER_LABELS = [
    "Logistic\nRegression",
    "LightGBM",
    "Random\nForest",
    "ExtraTrees",
    "MLP\n(shallow)",
]
# Reorder by marginal-sensitivity (LR < MLP < LGBM < ET < RF)
SENSITIVITY_ORDER = [0, 4, 1, 3, 2]  # indices into CARRIERS_ORDER


def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / "extracted_results.json"
    with open(data_path) as f:
        data = json.load(f)

    deltas = data["s0_deltas"]
    summary = data["s0_summary"]

    x_pos = np.arange(5)
    ordered_carriers = [CARRIERS_ORDER[i] for i in SENSITIVITY_ORDER]
    ordered_labels = [CARRIER_LABELS[i] for i in SENSITIVITY_ORDER]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45))

    for method, color, marker, label in [
        ("coral", COLORS["blue"], "o", "CORAL"),
        ("mean_matching", COLORS["orange"], "s", "Mean matching"),
    ]:
        y_vals = []
        y_errs = []
        annotations = []
        for clf in ordered_carriers:
            key = f"{method}__{clf}"
            d = deltas[key]
            y_vals.append(d["delta_auc"])
            # Error bar: half-width of the paired 95% t-interval
            lo, hi = d["delta_auc_ci95"]
            y_errs.append((hi - lo) / 2)
            annotations.append(f"{d['delta_auc']:+.3f}")

        ax.errorbar(
            x_pos, y_vals, yerr=y_errs,
            color=color, marker=marker, markersize=7,
            capsize=4, linewidth=1.5, label=label,
        )

        # Annotate CORAL deltas (only for CORAL to avoid clutter)
        if method == "coral":
            for xp, yp, ann in zip(x_pos, y_vals, annotations, strict=True):
                ax.annotate(
                    ann, (xp, yp),
                    textcoords="offset points", xytext=(10, 8),
                    ha="left", fontsize=8, color=color,
                )

    # Reference line
    ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)

    # Shading
    ax.axhspan(-0.01, 0, alpha=0.06, color=COLORS["red"], zorder=0)
    ax.axhspan(0, 0.04, alpha=0.06, color=COLORS["blue"], zorder=0)
    ax.text(4.6, -0.003, "DA hurts", fontsize=8, color=COLORS["neutral"], ha="right")
    ax.text(4.6, 0.002, "DA helps", fontsize=8, color=COLORS["neutral"], ha="right")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ordered_labels, fontsize=9)
    ax.set_ylabel(r"$\Delta$AUC (method $-$ source-only)")
    ax.set_xlabel("Carrier classifier (ordered by marginal-sensitivity)")
    ax.legend(loc="upper left", frameon=False)

    save_figure(fig, "fig2_carrier_gradient")
    print("Fig 2 saved.")


if __name__ == "__main__":
    main()
