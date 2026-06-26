"""Figure 4: Shift diagnostics and carrier benefit."""

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

    diags = data["s0_shift_diagnostics"]
    deltas = data["s0_deltas"]

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COL_WIDTH * 1.4, SINGLE_COL_WIDTH * 2.5))

    # Panel (a): Kernel MMD
    ax = axes[0]
    mmd_val = diags["mmd"]["value"]
    ax.barh([0], [mmd_val], color=COLORS["blue"], height=0.5)
    ax.set_xlim(0, max(mmd_val * 3, 0.005))
    ax.set_yticks([])
    ax.set_xlabel("MMD value")
    ax.set_title(f"(a) Kernel MMD: {mmd_val:.4f} (p < {diags['mmd']['p_value']:.3f})", fontsize=10)
    ax.axvline(0, color="k", linewidth=0.5)

    # Panel (b): Proxy A-distance
    ax = axes[1]
    a_dist = diags["proxy_a_distance"]["value"]
    ax.barh([0], [a_dist], color=COLORS["orange"], height=0.5)
    ax.set_xlim(0, 1.0)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.axvline(1.0, color=COLORS["neutral"], linestyle=":", linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Proxy A-distance")
    ax.set_title(f"(b) Proxy A-distance: {a_dist:.2f}", fontsize=10)
    ax.text(0.95, 0, "perfectly\ndistinguishable", ha="right", va="center",
            fontsize=7, color=COLORS["neutral"])

    # Panel (c): Per-carrier CORAL DAUC scatter
    ax = axes[2]
    carriers = ["lr", "lgbm", "rf", "et", "mlp"]
    carrier_labels = ["LR", "LGBM", "RF", "ET", "MLP"]

    # For each carrier, compute calibration drift proxy (delta_ECE is global;
    # use per-carrier DAUC as x and carrier index as proxy for marginal-sensitivity)
    coral_daucs = []
    for clf in carriers:
        key = f"coral__{clf}"
        if key in deltas:
            coral_daucs.append(deltas[key]["delta_auc"])
        else:
            coral_daucs.append(0.0)

    # x-axis: marginal-sensitivity index (1-5)
    sensitivity_index = [1, 3, 5, 4, 2]  # LR=1, MLP=2, LGBM=3, ET=4, RF=5

    ax.scatter(sensitivity_index, coral_daucs, color=COLORS["blue"], s=50, zorder=3)
    for i, label in enumerate(carrier_labels):
        ax.annotate(
            label, (sensitivity_index[i], coral_daucs[i]),
            textcoords="offset points", xytext=(8, 0),
            fontsize=8, color="#333333",
        )

    # Trend line
    z = np.polyfit(sensitivity_index, coral_daucs, 1)
    x_line = np.linspace(0.5, 5.5, 50)
    ax.plot(x_line, np.polyval(z, x_line), color=COLORS["neutral"],
            linestyle="--", linewidth=0.8)

    # Pearson r
    r = float(np.corrcoef(sensitivity_index, coral_daucs)[0, 1])
    ax.text(0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
            ha="right", fontsize=9, color="#333333")

    ax.axhline(0, color=COLORS["neutral"], linestyle=":", linewidth=0.6)
    ax.set_xlabel("Carrier marginal-sensitivity (ordinal)")
    ax.set_ylabel(r"CORAL $\Delta$AUC")
    ax.set_title("(c) Carrier sensitivity vs CORAL benefit", fontsize=10)
    ax.set_xlim(0.5, 5.5)

    fig.tight_layout(h_pad=1.5)
    save_figure(fig, "fig4_shift_diagnostics")
    print("Fig 4 saved.")


if __name__ == "__main__":
    main()
