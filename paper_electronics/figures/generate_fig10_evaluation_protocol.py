"""Figure 10: what the evaluation protocol does to the measured DA benefit.

Left: source-only AUC on the full target (which contains perturbed copies of
the carrier's own training events) against the held-out target. Right: the
CORAL gain measured the two ways. Both are the same runs, the same models
and the same shift; only the set of target events scored differs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import COLORS, DOUBLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()

CARRIERS = [("lr", "LR"), ("mlp", "MLP"), ("lgbm", "LGBM"), ("et", "ExtraTrees"), ("rf", "RF")]


def main() -> None:
    base = Path(__file__).parent.parent / "data"
    full = json.load(open(base / "extracted_results.json"))
    ho = json.load(open(base / "paper_data.json"))

    x = np.arange(len(CARRIERS))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.40))

    # (a) source-only AUC, full target vs held-out target
    a_full = [full["s0_summary"][f"srconly__{c}"]["auc_mean"] for c, _ in CARRIERS]
    a_ho = [ho["matrix"][c]["srconly"]["auc"]["mean"] for c, _ in CARRIERS]
    a_src = [ho["damage"][c]["auc_source_heldout"] for c, _ in CARRIERS]
    ax1.bar(x - w / 2, a_full, w, color=COLORS["red"], alpha=0.85, label="Full target (seen + unseen events)")
    ax1.bar(x + w / 2, a_ho, w, color=COLORS["blue"], alpha=0.85, label="Held-out target (unseen only)")
    ax1.plot(x, a_src, "k_", markersize=16, markeredgewidth=1.6, label="Held-out source (no shift)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([lab for _, lab in CARRIERS], fontsize=8.5)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel("Source-only AUC")
    ax1.set_title("(a) What the target set contains", fontsize=10)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")

    # (b) CORAL gain, full target vs held-out target
    g_full = [full["s0_deltas"][f"coral__{c}"]["delta_auc"] for c, _ in CARRIERS]
    e_full = [(lambda ci: (ci[1] - ci[0]) / 2)(full["s0_deltas"][f"coral__{c}"]["delta_auc_ci95"]) for c, _ in CARRIERS]
    g_ho = [ho["matrix"][c]["coral"]["delta_auc"]["mean"] for c, _ in CARRIERS]
    e_ho = [(lambda ci: (ci[1] - ci[0]) / 2)(ho["matrix"][c]["coral"]["delta_auc"]["ci95"]) for c, _ in CARRIERS]
    ax2.bar(x - w / 2, g_full, w, yerr=e_full, capsize=2.5, color=COLORS["red"], alpha=0.85,
            label="Measured on full target")
    ax2.bar(x + w / 2, g_ho, w, yerr=e_ho, capsize=2.5, color=COLORS["blue"], alpha=0.85,
            label="Measured on held-out target")
    ax2.axhline(0, color=COLORS["neutral"], linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lab for _, lab in CARRIERS], fontsize=8.5)
    ax2.set_ylabel(r"CORAL $\Delta$AUC vs source-only")
    ax2.set_title("(b) What the DA benefit looks like", fontsize=10)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "fig10_evaluation_protocol")
    print("Fig 10 saved. inflation factor (coral dAUC full/heldout):",
          [f"{c}:{gf / gh:.1f}x" for (c, _), gf, gh in zip(CARRIERS, g_full, g_ho)])


if __name__ == "__main__":
    main()
