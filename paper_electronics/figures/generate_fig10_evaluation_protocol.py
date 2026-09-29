"""Figure 10: the evaluation protocol, what it does and why.

(a) Source-only AUC on the full perturbed sample versus on the held-out
    target, with the unshifted held-out source for reference.
(b) The pooled AUC of the unadapted model, decomposed into the three
    populations of pairs it counts: both events from the training set,
    both from the held-out set, or one of each.
(c) Dose-response: the excess CORAL gain (full target minus held-out
    target) as a function of how much of the sample the carrier was
    trained on.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, DOUBLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()

CARRIERS = [("lr", "LR"), ("mlp", "MLP"), ("lgbm", "LGBM"), ("et", "ExtraTrees"), ("rf", "RF")]
FRACS = ["0.1", "0.3", "0.5", "0.7", "0.9"]


def main() -> None:
    base = Path(__file__).parent.parent / "data"
    with open(base / "extracted_results.json") as f:
        full = json.load(f)
    with open(base / "paper_data.json") as f:
        ho = json.load(f)
    with open(base / "overlap_doseresponse.json") as f:
        dose = json.load(f)

    x = np.arange(len(CARRIERS))
    w = 0.38
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.44))

    # (a) what the target set contains
    a_full = [full["s0_summary"][f"srconly__{c}"]["auc_mean"] for c, _ in CARRIERS]
    a_ho = [ho["matrix"][c]["srconly"]["auc"]["mean"] for c, _ in CARRIERS]
    a_src = [ho["damage"][c]["auc_source_heldout"] for c, _ in CARRIERS]
    ax1.bar(x - w / 2, a_full, w, color=COLORS["red"], alpha=0.85, label="Full target")
    ax1.bar(x + w / 2, a_ho, w, color=COLORS["blue"], alpha=0.85, label="Held-out target")
    ax1.plot(x, a_src, "k_", markersize=13, markeredgewidth=1.5, label="Held-out source")
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel("Source-only AUC")
    ax1.set_title("(a) What is scored", fontsize=10)
    ax1.legend(frameon=False, fontsize=7, loc="upper left")

    # (b) pooled-AUC decomposition into pair populations
    d7 = dose["0.7"]
    for key, color, mk, lab in [("within_train", COLORS["red"], "s", "both from training set"),
                                ("cross", COLORS["orange"], "D", "one of each"),
                                ("within_ho", COLORS["blue"], "o", "both held-out")]:
        ax2.plot(x, [d7[c][key] for c, _ in CARRIERS], color=color, marker=mk,
                 markersize=5, linewidth=1.3, label=lab)
    ax2.set_ylim(0.6, 1.02)
    ax2.set_ylabel("AUC of that pair population")
    ax2.set_title("(b) What the pooled AUC mixes", fontsize=10)
    ax2.legend(frameon=False, fontsize=7, loc="lower right")

    # (c) dose-response
    for (c, lab), color, mk in zip(CARRIERS,
                                   [COLORS["neutral"], COLORS["purple"], COLORS["orange"],
                                    COLORS["green"], COLORS["blue"]],
                                   ["o", "^", "s", "D", "v"], strict=True):
        y = [dose[f][c]["gain_full"] - dose[f][c]["gain_heldout"] for f in FRACS]
        ax3.plot([float(f) for f in FRACS], y, color=color, marker=mk, markersize=5,
                 linewidth=1.3, label=lab)
    ax3.axhline(0, color="k", linewidth=0.8)
    ax3.set_xlabel("Training fraction")
    ax3.set_ylabel(r"Excess gain (full $-$ held-out)")
    ax3.set_title("(c) Dose-response", fontsize=10)
    ax3.legend(frameon=False, fontsize=7, loc="upper left")

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in CARRIERS], fontsize=7.5, rotation=30)

    fig.tight_layout(w_pad=1.9)
    save_figure(fig, "fig10_evaluation_protocol")
    print("Fig 10 saved.")


if __name__ == "__main__":
    main()
