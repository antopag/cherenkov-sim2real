"""Figure 9: Q-factor as a function of the decision threshold."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, DOUBLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()

PANELS = [("rf", "(a) Random Forest"), ("lgbm", "(b) LightGBM"), ("lr", "(c) Logistic regression")]
METHODS = [("srconly", COLORS["neutral"], "Source-only"), ("mean_matching", COLORS["orange"], "Mean matching"),
           ("coral", COLORS["blue"], "CORAL")]


def main() -> None:
    q = json.loads((Path(__file__).parent.parent / "data" / "heldout.json")
                   .read_text(encoding="utf-8"))["q"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.36))
    for ax, (tag, title) in zip(axes, PANELS, strict=True):
        for m, color, label in METHODS:
            g = q[tag][m]["q_grid_mean"]
            th = np.array([float(t) for t, v in g.items() if v is not None])
            qv = np.array([v for v in g.values() if v is not None])
            ax.plot(th, qv, color=color, linewidth=1.5, label=label)
            ts = q[tag][m]["thr_star_mean"]
            ax.plot([ts], [np.interp(ts, th, qv)], marker="*", color=color, markersize=10, zorder=4)
        ax.axvline(0.5, color="k", linestyle=":", linewidth=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Decision threshold $\theta$")
        ax.set_xlim(0.05, 0.95)
    axes[0].set_ylabel("Q-factor on target")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[2].text(0.97, 0.05, "star: threshold selected\non source validation\ndotted: default 0.5",
                 transform=axes[2].transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "fig9_q_threshold")
    print("Fig 9 saved.")


if __name__ == "__main__":
    main()
