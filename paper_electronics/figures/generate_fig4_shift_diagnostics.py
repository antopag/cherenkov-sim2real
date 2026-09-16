"""Figure 4: Carrier marginal-sensitivity vs CORAL benefit.

The scalar shift diagnostics (kernel MMD, proxy A-distance, calibration
drift) are reported in a table in the manuscript; this figure shows the
only diagnostic with a carrier dimension: the per-carrier CORAL delta-AUC
against the ordinal marginal-sensitivity index.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, SINGLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()

# Ordinal marginal-sensitivity index: LR=1, MLP=2, LGBM=3, ET=4, RF=5
CARRIERS: list[tuple[str, str, int]] = [
    ("lr", "LR", 1),
    ("mlp", "MLP", 2),
    ("lgbm", "LightGBM", 3),
    ("et", "ExtraTrees", 4),
    ("rf", "RF", 5),
]


def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / "extracted_results.json"
    with open(data_path) as f:
        data = json.load(f)

    deltas = data["s0_deltas"]

    xs = np.array([idx for _, _, idx in CARRIERS], dtype=float)
    ys = np.array([deltas[f"coral__{clf}"]["delta_auc"] for clf, _, _ in CARRIERS])
    errs = np.array(
        [deltas[f"coral__{clf}"].get("delta_auc_std", 0.0) for clf, _, _ in CARRIERS]
    )

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 1.5, SINGLE_COL_WIDTH * 1.05))

    # Trend line + Pearson r
    z = np.polyfit(xs, ys, 1)
    x_line = np.linspace(0.5, 5.5, 50)
    ax.plot(x_line, np.polyval(z, x_line), color=COLORS["neutral"],
            linestyle="--", linewidth=0.9, zorder=2)
    r = float(np.corrcoef(xs, ys)[0, 1])

    ax.errorbar(xs, ys, yerr=errs if errs.any() else None, fmt="o",
                color=COLORS["blue"], ecolor=COLORS["blue"], capsize=3,
                markersize=6, zorder=3)
    for (_, label, idx), y in zip(CARRIERS, ys):
        ax.annotate(label, (idx, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="#333333")

    ax.axhline(0, color=COLORS["neutral"], linestyle=":", linewidth=0.6)
    ax.text(0.04, 0.92, f"Pearson $r$ = {r:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="#333333")

    ax.set_xticks(xs)
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(min(0.0, ys.min() - 0.005), ys.max() + 0.012)
    ax.set_xlabel("Carrier marginal-sensitivity (ordinal rank)")
    ax.set_ylabel(r"CORAL $\Delta$AUC vs source-only")

    fig.tight_layout()
    save_figure(fig, "fig4_shift_diagnostics")
    print(f"Fig 4 saved (r = {r:.3f}).")


if __name__ == "__main__":
    main()
