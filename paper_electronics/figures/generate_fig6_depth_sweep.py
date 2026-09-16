"""Figure 6: Random-Forest max_depth sweep (revision ablation A)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, DOUBLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()


def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / "ablations.json"
    with open(data_path) as f:
        depth = json.load(f)["depth"]

    tags = sorted(depth, key=lambda t: (t == "none", int(t) if t != "none" else 0))
    x = np.arange(len(tags))
    labels = ["∞" if depth[t]["max_depth"] is None else str(depth[t]["max_depth"]) for t in tags]

    def series(method: str, key: str = "delta_auc") -> tuple[np.ndarray, np.ndarray]:
        vals = np.array([depth[t][method][key] for t in tags])
        ci = np.array([depth[t][method][f"{key}_ci95"] for t in tags])
        return vals, (ci[:, 1] - ci[:, 0]) / 2

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42)
    )

    # (a) DAUC of CORAL and mean matching vs depth
    for method, color, marker, label in [
        ("coral", COLORS["blue"], "o", "CORAL"),
        ("mean_matching", COLORS["orange"], "s", "Mean matching"),
    ]:
        v, e = series(method)
        ax1.errorbar(x, v, yerr=e, color=color, marker=marker, markersize=6,
                     capsize=3, linewidth=1.4, label=label)
    ax1.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Random Forest max_depth")
    ax1.set_ylabel(r"$\Delta$AUC vs source-only")
    ax1.set_title("(a) DA benefit vs tree depth", fontsize=10)
    ax1.legend(frameon=False, loc="upper left")

    # (b) CORAL minus mean matching, paired
    v, e = series("coral_minus_mm")
    ax2.errorbar(x, v, yerr=e, color=COLORS["green"], marker="D", markersize=6,
                 capsize=3, linewidth=1.4)
    ax2.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Random Forest max_depth")
    ax2.set_ylabel(r"$\Delta$AUC (CORAL $-$ mean matching)")
    ax2.set_title("(b) Covariance-alignment gain vs depth", fontsize=10)

    # secondary info: source-only AUC as text on (a)
    src = [depth[t]["srconly"]["auc_mean"] for t in tags]
    ax1.text(0.98, 0.13, "source-only AUC: " + ", ".join(f"{s:.2f}" for s in src),
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=6.5,
             color="#555555")

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "fig6_depth_sweep")
    print("Fig 6 saved.")


if __name__ == "__main__":
    main()
