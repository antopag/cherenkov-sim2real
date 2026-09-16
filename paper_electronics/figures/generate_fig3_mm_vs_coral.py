"""Figure 3: Mean matching vs CORAL on tree-based carriers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, SINGLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()


def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / "extracted_results.json"
    with open(data_path) as f:
        data = json.load(f)

    deltas = data["s0_deltas"]
    summary = data["s0_summary"]

    carriers = ["lgbm", "et", "rf"]
    carrier_labels = [
        "LightGBM\n(max_depth=6)",
        "ExtraTrees\n(max_depth=\u221e)",
        "Random Forest\n(max_depth=\u221e)",
    ]

    x_pos = np.arange(len(carriers))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 1.4, SINGLE_COL_WIDTH * 1.0))

    mm_vals, mm_errs = [], []
    coral_vals, coral_errs = [], []

    for clf in carriers:
        for method, vals, errs in [
            ("mean_matching", mm_vals, mm_errs),
            ("coral", coral_vals, coral_errs),
        ]:
            d = deltas[f"{method}__{clf}"]
            lo, hi = d["delta_auc_ci95"]
            vals.append(d["delta_auc"])
            errs.append((hi - lo) / 2)  # paired 95% CI half-width

    ax.bar(
        x_pos - bar_width / 2, mm_vals, bar_width,
        yerr=mm_errs, capsize=3,
        color=COLORS["orange"], label="Mean matching", alpha=0.85,
    )
    ax.bar(
        x_pos + bar_width / 2, coral_vals, bar_width,
        yerr=coral_errs, capsize=3,
        color=COLORS["blue"], label="CORAL", alpha=0.85,
    )

    # Annotate gap
    for i, _clf in enumerate(carriers):
        gap = coral_vals[i] - mm_vals[i]
        y_top = max(coral_vals[i], mm_vals[i]) + max(coral_errs[i], mm_errs[i]) + 0.001
        ax.text(
            x_pos[i], y_top + 0.0005,
            f"\u0394={gap:+.004f}" if abs(gap) > 0.0005 else "\u0394\u22480",
            ha="center", fontsize=8, color="#333333",
        )

    ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(carrier_labels, fontsize=9)
    ax.set_ylabel(r"$\Delta$AUC vs source-only")
    ax.legend(loc="upper left", frameon=False)

    save_figure(fig, "fig3_mm_vs_coral")
    print("Fig 3 saved.")


if __name__ == "__main__":
    main()
