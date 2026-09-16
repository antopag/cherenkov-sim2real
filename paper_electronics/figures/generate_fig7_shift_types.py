"""Figure 7: carrier gradient under different shift types / intensities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, DOUBLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()

CARRIERS = ["lr", "mlp", "lgbm", "et", "rf"]  # marginal-sensitivity order
CARRIER_LABELS = ["LR", "MLP", "LightGBM", "ExtraTrees", "RF"]
CONFIGS = [  # noqa: RUF001
    ("loc", "Location only\n(atmospheric ×1.0)", COLORS["purple"], "v"),
    ("nsb", "Noise only\n(NSB ×1.0)", COLORS["green"], "^"),
    ("head05", "Headline ×0.5", COLORS["orange"], "s"),
    ("head10", "Headline ×1.0 (main)", COLORS["blue"], "o"),
    ("head20", "Headline ×2.0", COLORS["red"], "D"),
]


def main() -> None:
    base = Path(__file__).parent.parent / "data"
    with open(base / "ablations.json") as f:
        shift = json.load(f)["shift"]
    with open(base / "extracted_results.json") as f:
        main_deltas = json.load(f)["s0_deltas"]

    # headline x1.0 from the main matrix (15 seeds) for reference
    shift["head10"] = {
        c: {"coral": {"delta_auc": main_deltas[f"coral__{c}"]["delta_auc"],
                      "delta_auc_ci95": main_deltas[f"coral__{c}"]["delta_auc_ci95"]}}
        for c in CARRIERS
    }

    x = np.arange(len(CARRIERS))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45))
    for i, (cfg, label, color, marker) in enumerate(CONFIGS):
        if cfg not in shift:
            continue
        v = np.array([shift[cfg][c]["coral"]["delta_auc"] for c in CARRIERS])
        ci = np.array([shift[cfg][c]["coral"]["delta_auc_ci95"] for c in CARRIERS])
        e = (ci[:, 1] - ci[:, 0]) / 2
        off = (i - 2) * 0.06
        ax.errorbar(x + off, v, yerr=e, color=color, marker=marker, markersize=6,
                    capsize=2.5, linewidth=1.3, label=label)
    ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CARRIER_LABELS)
    ax.set_xlabel("Carrier classifier (ordered by marginal-sensitivity)")
    ax.set_ylabel(r"CORAL $\Delta$AUC vs source-only")
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "fig7_shift_types")
    print("Fig 7 saved.")


if __name__ == "__main__":
    main()
