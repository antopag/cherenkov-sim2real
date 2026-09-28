"""Turn the held-out re-analysis into the single data file the paper uses.

Reads ``data/heldout.json`` (written by ``scripts/run_heldout_matrix.py``)
and writes ``data/paper_data.json`` with

  matrix[carrier][method]  auc/f1/q05 mean+std, paired deltas vs source-only,
                           and the recovery fraction R = gain / damage
  damage[carrier]          AUC lost to the shift, held-out source vs held-out
                           target, paired over seeds
  depth, shift, q, lgbm    pass-through of the corresponding held-out blocks
  s0_deltas, s0_summary    compatibility view in the schema the figure
                           scripts already consume

The recovery fraction is computed per seed as
    R = [AUC(method, target) - AUC(srconly, target)] / [AUC(srconly, source)
        - AUC(srconly, target)]
and then averaged, so that its confidence interval reflects seed-to-seed
variability of the ratio itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
CARRIERS = ["lr", "mlp", "lgbm", "et", "rf"]
METHODS = ["mean_matching", "coral", "mmd_rbf"]


def paired(d: np.ndarray) -> dict[str, Any]:
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], float)
    n = len(d)
    if n < 2:
        return {"mean": round(float(d.mean()), 4) if n else None, "ci95": None, "n": n}
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    return {
        "mean": round(float(d.mean()), 4),
        "ci95": [round(float(d.mean() - tc * se), 4), round(float(d.mean() + tc * se), 4)],
        "p": float(stats.ttest_1samp(d, 0.0).pvalue) if d.std(ddof=1) > 0 else 0.0,
        "n": n,
        "per_seed": [round(float(v), 4) for v in d],
    }


def recovery_ratio(gain: np.ndarray, dmg: np.ndarray, n_boot: int = 10000,
                   seed: int = 0) -> dict[str, Any]:
    """Recovery fraction as a ratio of means, with a bootstrap interval.

    The per-seed ratio gain/damage is unusable when the damage is small
    (the MLP), so the point estimate is mean(gain)/mean(damage) and the
    interval comes from resampling the seed pairs.
    """
    gain, dmg = np.asarray(gain, float), np.asarray(dmg, float)
    rng = np.random.default_rng(seed)
    n = len(gain)
    idx = rng.integers(0, n, size=(n_boot, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = gain[idx].mean(axis=1) / dmg[idx].mean(axis=1)
    boots = boots[np.isfinite(boots)]
    return {
        "mean": round(float(gain.mean() / dmg.mean()), 3),
        "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                 round(float(np.percentile(boots, 97.5)), 3)],
        "n": n,
    }


def main() -> None:
    ho = json.loads((HERE / "data" / "heldout.json").read_text(encoding="utf-8"))
    mat = ho["matrix"]
    out: dict[str, Any] = {"matrix": {}, "damage": {}, "s0_summary": {}, "s0_deltas": {}}

    for c in CARRIERS:
        src = mat[c]["srconly"]["per_seed"]
        auc_src_t = np.array([r["auc"] for r in src])
        auc_src_s = np.array([r["auc_src_heldout"] for r in src])
        dmg = auc_src_s - auc_src_t
        out["damage"][c] = paired(dmg)
        out["damage"][c]["auc_source_heldout"] = round(float(auc_src_s.mean()), 4)
        out["damage"][c]["auc_target_heldout"] = round(float(auc_src_t.mean()), 4)

        out["matrix"][c] = {}
        for m in ["srconly"] + METHODS:
            per = mat[c][m]["per_seed"]
            entry: dict[str, Any] = {
                k: {"mean": mat[c][m][k]["mean"], "std": mat[c][m][k]["std"]}
                for k in ("auc", "f1", "q05")
            }
            if m != "srconly":
                for k in ("auc", "f1", "q05"):
                    g = np.array([r[k] for r in per], float) - np.array(
                        [r[k] for r in src], float
                    )
                    entry[f"delta_{k}"] = paired(g)
                gain = np.array([r["auc"] for r in per]) - auc_src_t
                entry["recovery"] = recovery_ratio(gain, dmg)
            out["matrix"][c][m] = entry
            # compatibility view for the existing figure scripts
            out["s0_summary"][f"{m}__{c}"] = {
                "method": m,
                "carrier": c,
                "n_seeds": mat[c][m]["auc"]["n"],
                "auc_mean": mat[c][m]["auc"]["mean"],
                "auc_std": mat[c][m]["auc"]["std"],
                "f1_mean": mat[c][m]["f1"]["mean"],
                "f1_std": mat[c][m]["f1"]["std"],
                "q_mean": mat[c][m]["q05"]["mean"],
                "q_std": mat[c][m]["q05"]["std"],
            }
            if m != "srconly":
                e = out["matrix"][c][m]
                out["s0_deltas"][f"{m}__{c}"] = {
                    "method": m,
                    "carrier": c,
                    "delta_auc": e["delta_auc"]["mean"],
                    "delta_auc_ci95": e["delta_auc"]["ci95"],
                    "delta_auc_p": e["delta_auc"].get("p"),
                    "delta_auc_per_seed": e["delta_auc"]["per_seed"],
                    "delta_q": e["delta_q05"]["mean"],
                    "delta_q_ci95": e["delta_q05"]["ci95"],
                    "delta_f1": e["delta_f1"]["mean"],
                    "delta_f1_ci95": e["delta_f1"]["ci95"],
                    "recovery": e["recovery"]["mean"],
                    "recovery_ci95": e["recovery"]["ci95"],
                    "n_pairs": e["delta_auc"]["n"],
                }

    for blk in ("depth", "shift", "q", "lgbm"):
        if blk in ho:
            out[blk] = ho[blk]

    # shift-diagnostics block kept from the previous extraction (unchanged)
    prev = json.loads((HERE / "data" / "extracted_results.json").read_text(encoding="utf-8"))
    out["s0_shift_diagnostics"] = prev["s0_shift_diagnostics"]

    (HERE / "data" / "paper_data.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"{'carrier':8s} {'srcAUC':>7s} {'tgtAUC':>7s} {'damage':>8s} "
          f"{'mm':>16s} {'coral':>16s} {'recov(coral)':>14s}")
    for c in CARRIERS:
        d = out["damage"][c]
        mm = out["matrix"][c]["mean_matching"]["delta_auc"]
        co = out["matrix"][c]["coral"]["delta_auc"]
        rc = out["matrix"][c]["coral"]["recovery"]
        print(f"{c:8s} {d['auc_source_heldout']:7.4f} {d['auc_target_heldout']:7.4f} "
              f"{d['mean']:+8.4f} {mm['mean']:+8.4f} {str(mm['ci95']):>0s} "
              f"{co['mean']:+8.4f} {str(co['ci95']):>0s} "
              f"{rc['mean']:+6.2f} {rc['ci95']}")


if __name__ == "__main__":
    main()
