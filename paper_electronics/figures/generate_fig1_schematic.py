"""Figure 1: Conceptual 2x2 framework schematic."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from style import SINGLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()


def main() -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 1.6, SINGLE_COL_WIDTH * 1.3))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Cell colors: intensity reflects DA benefit magnitude
    cell_colors = {
        (0, 1): "#e8e8e8",  # robust + sym: DA neutral/hurts
        (1, 1): "#b3d9ff",  # sensitive + sym: DA helps
        (0, 0): "#b3d9ff",  # robust + asym: DA helps (predicted)
        (1, 0): "#4da6ff",  # sensitive + asym: DA helps strongly (predicted)
    }

    cell_text = {
        (0, 1): "DA NEUTRAL\n/ HURTS",
        (1, 1): "DA HELPS",
        (0, 0): "DA HELPS\n(predicted)",
        (1, 0): "DA HELPS\nSTRONGLY\n(predicted)",
    }

    cell_carriers = {
        (0, 1): "LR, MLP",
        (1, 1): "LightGBM,\nRF, ExtraTrees",
        (0, 0): "",
        (1, 0): "",
    }

    cell_evidence = {
        (0, 1): "S0 empirical",
        (1, 1): "S0 empirical",
        (0, 0): "S2/S3 pending",
        (1, 0): "S2/S3 pending",
    }

    for (col, row), color in cell_colors.items():
        rect = mpatches.FancyBboxPatch(
            (col * 1.0 + 0.02, row * 1.0 + 0.02),
            0.96, 0.96,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax.add_patch(rect)

        # Main text
        ax.text(
            col + 0.5, row + 0.58, cell_text[(col, row)],
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#333333",
        )
        # Carrier labels
        if cell_carriers[(col, row)]:
            ax.text(
                col + 0.5, row + 0.25, cell_carriers[(col, row)],
                ha="center", va="center", fontsize=8, fontstyle="italic",
                color="#555555",
            )
        # Evidence tag
        ax.text(
            col + 0.5, row + 0.10, cell_evidence[(col, row)],
            ha="center", va="center", fontsize=7, color="#888888",
        )

    # Axis labels
    ax.text(
        1.0, 2.12, "Carrier marginal-sensitivity",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
    ax.text(
        0.5, 2.04, "Marginal-robust", ha="center", va="bottom", fontsize=9,
    )
    ax.text(
        1.5, 2.04, "Marginal-sensitive", ha="center", va="bottom", fontsize=9,
    )

    ax.text(
        -0.15, 1.0, "Shift symmetry",
        ha="center", va="center", fontsize=11, fontweight="bold",
        rotation=90,
    )
    ax.text(
        -0.05, 1.5, "Class-\nsymmetric", ha="center", va="center", fontsize=9,
        rotation=90,
    )
    ax.text(
        -0.05, 0.5, "Class-\nasymmetric", ha="center", va="center", fontsize=9,
        rotation=90,
    )

    save_figure(fig, "fig1_schematic")
    print("Fig 1 saved.")


if __name__ == "__main__":
    main()
