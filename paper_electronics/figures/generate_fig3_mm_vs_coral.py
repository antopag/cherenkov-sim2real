"""Figure 3: Mean matching vs CORAL on tree-based carriers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, SINGLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()


def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / "paper_data.json"
    with open(data_path) as f:
        data = json.load(f)

    deltas = data["s0_deltas"]

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

    # Headroom for the gap annotations and the legend, so that neither can fall
    # on a bar or on an error bar.
    bar_top = max(
        v + e
        for vals, errs in ((mm_vals, mm_errs), (coral_vals, coral_errs))
        for v, e in zip(vals, errs, strict=True)
    )
    ax.set_ylim(0, bar_top * 1.30)

    # Annotate gap, all at the same height just above the tallest error bar.
    # The sign is a true minus (U+2212), not a hyphen.
    for i, _clf in enumerate(carriers):
        gap = coral_vals[i] - mm_vals[i]
        label = f"\u0394={gap:+.4f}" if abs(gap) > 0.0005 else "\u0394\u22480"
        ax.text(
            x_pos[i], bar_top * 1.06,
            label.replace("-", "\u2212"),
            ha="center", va="bottom", fontsize=8, color="#333333",
        )

    ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(carrier_labels, fontsize=9)
    ax.set_ylabel(r"$\Delta$AUC vs source-only")
    # Legend above the axes: inside, it sat on the LightGBM error bar.
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
        frameon=False, borderaxespad=0.0,
    )

    save_figure(fig, "fig3_mm_vs_coral")
    print("Fig 3 saved.")


if __name__ == "__main__":
    main()
