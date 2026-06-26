"""Figure 5: Threshold-split mechanism schematic (conceptual)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, SINGLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()


def main() -> None:
    rng = np.random.default_rng(42)

    # Synthetic 2D data: two Gaussian blobs per class
    n = 80
    class0 = rng.normal([2.0, 2.0], 0.6, (n, 2))
    class1 = rng.normal([4.0, 4.0], 0.6, (n, 2))

    # Shifted version (uniform marginal shift: +1.5 on feature 1)
    shift = np.array([1.5, 0.0])
    class0_shifted = class0 + shift
    class1_shifted = class1 + shift

    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COL_WIDTH * 1.8, SINGLE_COL_WIDTH * 1.6))

    titles = [
        ["Linear: source", "Linear: shifted target"],
        ["Tree: source", "Tree: shifted target"],
    ]

    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            if col == 0:
                c0, c1 = class0, class1
            else:
                c0, c1 = class0_shifted, class1_shifted

            ax.scatter(c0[:, 0], c0[:, 1], c=COLORS["orange"], s=12, alpha=0.6, label="Hadron")
            ax.scatter(c1[:, 0], c1[:, 1], c=COLORS["blue"], s=12, alpha=0.6, label="Gamma")

            if row == 0:
                # Linear classifier: diagonal hyperplane (slope ≈ -1)
                # Trained on source, applied to both
                x_line = np.linspace(0, 8, 50)
                y_line = -x_line + 6.0  # hyperplane
                ax.plot(x_line, y_line, color="#333333", linewidth=1.5, linestyle="-")

                if col == 1:
                    # The hyperplane still separates — shift doesn't break it
                    ax.text(
                        0.5, 0.92, "Still separates",
                        transform=ax.transAxes, fontsize=8,
                        color=COLORS["green"], fontweight="bold",
                        ha="center",
                    )
            else:
                # Tree classifier: axis-aligned threshold at x=3.0
                ax.axvline(3.0, color="#333333", linewidth=1.5, linestyle="-")

                if col == 0:
                    ax.text(
                        0.5, 0.92, "Threshold separates",
                        transform=ax.transAxes, fontsize=8,
                        color=COLORS["green"], fontweight="bold",
                        ha="center",
                    )
                else:
                    ax.text(
                        0.5, 0.92, "Threshold misaligned",
                        transform=ax.transAxes, fontsize=8,
                        color=COLORS["red"], fontweight="bold",
                        ha="center",
                    )

            ax.set_xlim(0, 8)
            ax.set_ylim(0, 7)
            ax.set_title(titles[row][col], fontsize=9)
            ax.set_xlabel("Feature 1", fontsize=8)
            ax.set_ylabel("Feature 2", fontsize=8)

    # Legend in top-left panel only
    axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)

    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save_figure(fig, "fig5_threshold_schematic")
    print("Fig 5 saved.")


if __name__ == "__main__":
    main()
