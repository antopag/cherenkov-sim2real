"""Round-3 computations A1-A3, from the existing per-seed manifests.

A1  Bootstrap 95% interval of the recovery fraction R for all three alignment
    methods on all five carriers, using exactly the procedure of Section 3.2
    (ratio of means, percentile bootstrap, B = 10^4, resampling the 15
    carrier-seed pairs). The CORAL column must reproduce paper_data.json.

A2  Paired MMD-rbf minus CORAL difference in held-out AUC: mean and 95%
    t-interval with 14 degrees of freedom, all five carriers.

A3  Range of the across-seed standard deviation of held-out AUC over the 20
    cells of Table 5 (five carriers x four methods including source-only).

Writes results/diagnostics/recovery_fraction_ci.{csv,md} and
results/diagnostics/mmd_minus_coral.{csv,md}.

    python scripts/run_interval_diagnostics.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper_electronics"))
from build_paper_data import CARRIERS, METHODS, recovery_ratio  # noqa: E402

OUT = ROOT / "results" / "diagnostics"
LABEL = {"lr": "Logistic regression", "mlp": "MLP-shallow", "lgbm": "LightGBM",
         "et": "ExtraTrees", "rf": "Random Forest"}
MLABEL = {"mean_matching": "Mean matching", "coral": "CORAL", "mmd_rbf": "MMD-rbf"}


def main() -> None:
    mat = json.loads((ROOT / "paper_electronics" / "data" / "heldout.json")
                     .read_text(encoding="utf-8"))["matrix"]
    pub = json.loads((ROOT / "paper_electronics" / "data" / "paper_data.json")
                     .read_text(encoding="utf-8"))["matrix"]

    # ------------------------------------------------------------------ A1
    rows_a1, mismatches = [], []
    for c in CARRIERS:
        src = mat[c]["srconly"]["per_seed"]
        auc_t = np.array([r["auc"] for r in src])
        dmg = np.array([r["auc_src_heldout"] for r in src]) - auc_t
        row = {"carrier": LABEL[c]}
        for m in METHODS:
            gain = np.array([r["auc"] for r in mat[c][m]["per_seed"]]) - auc_t
            rec = recovery_ratio(gain, dmg)
            row[f"R_{m}"] = rec["mean"]
            row[f"R_{m}_lo"], row[f"R_{m}_hi"] = rec["ci95"]
            ref = pub[c][m].get("recovery")
            if ref is not None and (ref["mean"] != rec["mean"] or ref["ci95"] != rec["ci95"]):
                mismatches.append((c, m, ref, rec))
        rows_a1.append(row)

    if mismatches:
        for c, m, ref, rec in mismatches:
            print(f"MISMATCH {c}/{m}: manifest {ref['mean']} {ref['ci95']} "
                  f"vs recomputed {rec['mean']} {rec['ci95']}")
        raise SystemExit("A1 does not reproduce the published intervals; stopping.")
    print("A1: all three methods reproduce paper_data.json exactly "
          f"({len(CARRIERS) * len(METHODS)} cells checked)")

    # ------------------------------------------------------------------ A2
    rows_a2 = []
    for c in CARRIERS:
        auc_t = np.array([r["auc"] for r in mat[c]["srconly"]["per_seed"]])
        a = np.array([r["auc"] for r in mat[c]["mmd_rbf"]["per_seed"]]) - auc_t
        b = np.array([r["auc"] for r in mat[c]["coral"]["per_seed"]]) - auc_t
        d = a - b
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = stats.t.ppf(0.975, len(d) - 1)
        rows_a2.append({"carrier": LABEL[c], "n_seeds": len(d),
                        "mean": round(float(d.mean()), 5),
                        "ci95_lo": round(float(d.mean() - t * se), 5),
                        "ci95_hi": round(float(d.mean() + t * se), 5),
                        "excludes_zero": bool((d.mean() - t * se) * (d.mean() + t * se) > 0)})

    # ------------------------------------------------------------------ A3
    sds = {f"{LABEL[c]} / {m}": mat[c][m]["auc"]["std"]
           for c in CARRIERS for m in ["srconly"] + METHODS}
    lo_k = min(sds, key=lambda k: sds[k])
    hi_k = max(sds, key=lambda k: sds[k])
    a3 = {"n_cells": len(sds), "min": sds[lo_k], "min_cell": lo_k,
          "max": sds[hi_k], "max_cell": hi_k}
    print(f"A3: {a3['n_cells']} cells, std of held-out AUC from {a3['min']:.4f} "
          f"({lo_k}) to {a3['max']:.4f} ({hi_k})")

    # ------------------------------------------------------------------ write
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "recovery_fraction_ci.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_a1[0]))
        w.writeheader()
        w.writerows(rows_a1)
    md = ["# Recovery fraction R with 95% bootstrap intervals", "",
          "Ratio of means, percentile bootstrap, B = 10^4, resampling the 15 "
          "carrier-seed pairs (Section 3.2).", "",
          "| carrier | mean matching | CORAL | MMD-rbf |", "|---|---|---|---|"]
    for r in rows_a1:
        cells = [f"{r[f'R_{m}']:.2f} [{r[f'R_{m}_lo']:.2f}, {r[f'R_{m}_hi']:.2f}]"
                 for m in METHODS]
        md.append("| " + r["carrier"] + " | " + " | ".join(cells) + " |")
    (OUT / "recovery_fraction_ci.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with open(OUT / "mmd_minus_coral.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_a2[0]))
        w.writeheader()
        w.writerows(rows_a2)
    md2 = ["# Paired MMD-rbf minus CORAL, held-out AUC gain over source-only", "",
           "Per-seed paired differences, 95% t-interval, 14 degrees of freedom.", "",
           "| carrier | difference | interval excludes zero |", "|---|---|---|"]
    for r in rows_a2:
        md2.append(f"| {r['carrier']} | {r['mean']:+.5f} "
                   f"[{r['ci95_lo']:+.5f}, {r['ci95_hi']:+.5f}] | "
                   f"{'yes' if r['excludes_zero'] else 'no'} |")
    (OUT / "mmd_minus_coral.md").write_text("\n".join(md2) + "\n", encoding="utf-8")

    (OUT / "interval_diagnostics.json").write_text(json.dumps(
        {"recovery_fraction": rows_a1, "mmd_minus_coral": rows_a2,
         "heldout_auc_std_range": a3, "heldout_auc_std_cells": sds}, indent=1),
        encoding="utf-8")

    print()
    print("\n".join(md))
    print()
    print("\n".join(md2))


if __name__ == "__main__":
    main()
