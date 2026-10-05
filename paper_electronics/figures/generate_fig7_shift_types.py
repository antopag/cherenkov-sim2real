"""Figure 7: shift damage and CORAL recovery under different shift types."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, DOUBLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()

CARRIERS = ["lr", "mlp", "lgbm", "et", "rf"]
LABELS = ["LR", "MLP", "LGBM", "ExtraTrees", "RF"]
CONFIGS = [
    ("loc", "Location only", COLORS["purple"], "v"),
    ("nsb", "Noise only", COLORS["green"], "^"),
    ("head05", "Headline ×0.5", COLORS["orange"], "s"),
    ("head10", "Headline ×1.0", COLORS["blue"], "o"),
    ("head20", "Headline ×2.0", COLORS["red"], "D"),
    ("pure_translation", "Pure translation", "#444444", "P"),
]


def half(ci):
    return (ci[1] - ci[0]) / 2


def main() -> None:
    base = Path(__file__).parent.parent / "data"
    ho = json.load(open(base / "heldout.json"))
    rc = json.load(open(base / "review_checks.json"))
    pd_ = json.load(open(base / "paper_data.json"))
    shift = ho["shift"]
    # pure translation: same schema as the shift block
    shift["pure_translation"] = {
        c: {"coral": {"delta_auc": rc["pure_translation"][c]["coral"]["delta_auc"]},
            "damage": rc["pure_translation"][c]["damage"]}
        for c in CARRIERS
    }
    # headline x1.0 comes from the main 15-seed matrix
    shift["head10"] = {
        c: {"coral": {"delta_auc": pd_["matrix"][c]["coral"]["delta_auc"]},
            "damage": pd_["damage"][c]}
        for c in CARRIERS
    }

    x = np.arange(len(CARRIERS))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.40))
    for i, (cfg, label, color, marker) in enumerate(CONFIGS):
        if cfg not in shift:
            continue
        off = (i - 2) * 0.07
        d = np.array([shift[cfg][c]["damage"]["mean"] for c in CARRIERS])
        de = np.array([half(shift[cfg][c]["damage"]["ci95"]) for c in CARRIERS])
        ax1.errorbar(x + off, d, yerr=de, color=color, marker=marker, markersize=5,
                     capsize=2, linewidth=1.2, label=label)
        g = np.array([shift[cfg][c]["coral"]["delta_auc"]["mean"] for c in CARRIERS])
        ge = np.array([half(shift[cfg][c]["coral"]["delta_auc"]["ci95"]) for c in CARRIERS])
        ax2.errorbar(x + off, g, yerr=ge, color=color, marker=marker, markersize=5,
                     capsize=2, linewidth=1.2, label=label)

    for ax, ylab, title in [
        (ax1, "Shift damage (AUC lost)", "(a) What the shift costs"),
        (ax2, r"CORAL $\Delta$AUC", "(b) What alignment gives back"),
    ]:
        ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=8.5)
        ax.set_xlabel("Carrier classifier")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "fig7_shift_types")
    print("Fig 7 saved.")


if __name__ == "__main__":
    main()
