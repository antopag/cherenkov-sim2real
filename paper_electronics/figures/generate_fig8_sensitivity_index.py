"""Figure 8: DA-free marginal-sensitivity index vs CORAL gain."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from style import COLORS, DOUBLE_COL_WIDTH, apply_paper_style, save_figure

apply_paper_style()

CARRIERS = [("lr", "LR"), ("mlp", "MLP"), ("lgbm", "LightGBM"), ("et", "ExtraTrees"), ("rf", "RF")]
DELTAS = [0.1, 0.25, 0.5, 1.0]


def main() -> None:
    base = Path(__file__).parent.parent / "data"
    sens = json.loads((base / "revision2.json").read_text(encoding="utf-8"))["sens"]
    deltas = json.loads(
        (base / "paper_data.json").read_text(encoding="utf-8"))["s0_deltas"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    # (a) S(delta) curves per carrier
    palette = [COLORS["neutral"], COLORS["purple"], COLORS["orange"], COLORS["green"], COLORS["blue"]]
    markers = ["o", "^", "s", "D", "v"]
    for (tag, label), color, mk in zip(CARRIERS, palette, markers, strict=True):
        s = sens[tag]
        y = [s[f"auc_drop_{d}_mean"] for d in DELTAS]
        e = [s[f"auc_drop_{d}_std"] for d in DELTAS]
        ax1.errorbar(DELTAS, y, yerr=e, color=color, marker=mk, markersize=5, capsize=2,
                     linewidth=1.3, label=label)
    ax1.set_xscale("log")
    ax1.set_xticks(DELTAS)
    ax1.set_xticklabels([str(d) for d in DELTAS])
    ax1.set_xlabel(r"Uniform translation $\delta$ (standardised units)")
    ax1.set_ylabel(r"Held-out source AUC drop $S(\delta)$")
    ax1.set_title("(a) Sensitivity index, no DA involved", fontsize=10)
    ax1.legend(frameon=False, fontsize=8)

    # (b) S(0.25) vs CORAL dAUC, per seed and per carrier mean
    xs_all, ys_all = [], []
    for (tag, label), color, mk in zip(CARRIERS, palette, markers, strict=True):
        per = sens[tag]["per_seed"]
        x = np.array([p["auc_drop_0.25"] for p in per])
        y = np.array(deltas[f"coral__{tag}"]["delta_auc_per_seed"])
        ax2.scatter(x, y, color=color, marker=mk, s=14, alpha=0.35)
        ax2.scatter(x.mean(), y.mean(), color=color, marker=mk, s=70, edgecolor="k", linewidth=0.6, zorder=4)
        ax2.annotate(label, (x.mean(), y.mean()), textcoords="offset points", xytext=(7, -3), fontsize=8)
        xs_all.extend(x)
        ys_all.extend(y)
    xm = [np.mean([p["auc_drop_0.25"] for p in sens[t]["per_seed"]]) for t, _ in CARRIERS]
    ym = [deltas[f"coral__{t}"]["delta_auc"] for t, _ in CARRIERS]
    rs = spearmanr(xm, ym)[0]
    rp_pool = pearsonr(xs_all, ys_all)[0]
    ax2.text(0.04, 0.93, f"5 carriers: Spearman $\\rho$ = {rs:.2f}\n75 seed pairs: Pearson $r$ = {rp_pool:.2f}",
             transform=ax2.transAxes, va="top", fontsize=8.5)
    ax2.axhline(0, color=COLORS["neutral"], linestyle=":", linewidth=0.6)
    ax2.set_xlabel(r"Sensitivity index $S(0.25)$")
    ax2.set_ylabel(r"CORAL $\Delta$AUC vs source-only")
    ax2.set_title("(b) Index vs DA gain", fontsize=10)
    ax2.set_xlim(-0.005, 0.075)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "fig8_sensitivity_index")
    print(f"Fig 8 saved (rho={rs:.2f}, r_pooled={rp_pool:.2f}).")


if __name__ == "__main__":
    main()
