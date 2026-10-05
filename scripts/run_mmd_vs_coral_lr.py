"""A1: why RBF-kernel MMD beats CORAL on the linear carrier.

On logistic regression mean matching is exactly neutral, so the whole gain of
CORAL and of MMD-rbf comes from the non-translational part of their transform
matrix W. This script measures, over the 15 seeds of the main matrix:

(a) how close each method brings the per-feature standardised standard
    deviation of the aligned source to the target's, for the three features
    the shift actually touches in scale (WIDTH, LENGTH, SIZE);
(b) how far each W departs from the identity: Frobenius norm of W - I and the
    diagonal of W;
(c) the paired difference MMD-rbf minus CORAL on the held-out LR AUC, with a
    95% t-interval.

Writes results/diagnostics/mmd_vs_coral_lr.csv and .md.

    python scripts/run_mmd_vs_coral_lr.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cherenkov_sim2real.adaptation.coral import CORAL
from cherenkov_sim2real.adaptation.mean_matching import MeanMatching
from cherenkov_sim2real.adaptation.mmd_alignment import MMDAlignment
from run_ablations import (
    HEADLINE_COMPOSITE,
    SEEDS_15,
    TRAIN_RATIO,
    apply_composite,
    load_base,
    make_classifier,
    preprocess_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("cherenkov_sim2real").setLevel(logging.WARNING)
logger = logging.getLogger("mmd_vs_coral")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "diagnostics"
FOCUS = ["WIDTH", "LENGTH", "SIZE"]
METHODS = ["srconly", "mean_matching", "coral", "mmd_rbf"]


def ci(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    return float(d.mean()), float(d.mean() - t * se), float(d.mean() + t * se)


def main() -> None:
    x_all, y_all = load_base()
    cols = list(x_all.columns)
    ys = y_all.to_numpy()
    idx_focus = [cols.index(f) for f in FOCUS]

    sd_gap = {m: {f: [] for f in FOCUS} for m in METHODS}
    wdev = {m: [] for m in ("coral", "mmd_rbf")}
    wdiag = {m: [] for m in ("coral", "mmd_rbf")}
    auc = {m: [] for m in METHODS}

    for seed in SEEDS_15:
        xt = apply_composite(x_all, list(HEADLINE_COMPOSITE), seed=seed)
        xs_p, xt_p, _, _ = preprocess_features(
            pd.DataFrame(x_all.to_numpy(), columns=cols),
            pd.DataFrame(xt.to_numpy(), columns=cols))
        itr, iva = train_test_split(np.arange(len(xs_p)), train_size=TRAIN_RATIO,
                                    random_state=seed, stratify=ys)
        al = {"srconly": None,
              "mean_matching": MeanMatching().fit(xs_p, xt_p),
              "coral": CORAL(lambda_reg=1e-3).fit(xs_p, xt_p),
              "mmd_rbf": MMDAlignment(kernel="rbf", lambda_reg=1e-3,
                                      max_iter=100).fit(xs_p, xt_p)}
        sd_t = xt_p.std(axis=0)
        for m in METHODS:
            xa = xs_p if al[m] is None else al[m].transform(xs_p)
            sd_a = xa.std(axis=0)
            for f, j in zip(FOCUS, idx_focus, strict=True):
                sd_gap[m][f].append(float(abs(sd_a[j] - sd_t[j])))
            model = make_classifier("logistic_regression", seed)
            tr = xs_p[itr] if al[m] is None else al[m].transform(xs_p[itr])
            model.fit(tr, ys[itr])
            auc[m].append(float(roc_auc_score(ys[iva], model.predict_proba(xt_p[iva])[:, 1])))
        for m in ("coral", "mmd_rbf"):
            W = al[m]._transform_matrix if m == "coral" else al[m]._W
            wdev[m].append(float(np.linalg.norm(W - np.eye(W.shape[0]))))
            wdiag[m].append(np.diag(W).copy())
        logger.info("seed %d done", seed)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in METHODS:
        r = {"method": m}
        for f in FOCUS:
            r[f"sd_gap_{f}"] = round(float(np.mean(sd_gap[m][f])), 5)
        r["auc_lr"] = round(float(np.mean(auc[m])), 5)
        if m in wdev:
            r["W_minus_I_frob"] = round(float(np.mean(wdev[m])), 4)
            d = np.mean(wdiag[m], axis=0)
            r["W_diag_min"] = round(float(d.min()), 4)
            r["W_diag_max"] = round(float(d.max()), 4)
            r["W_diag_WIDTH"] = round(float(d[cols.index("WIDTH")]), 4)
            r["W_diag_LENGTH"] = round(float(d[cols.index("LENGTH")]), 4)
            r["W_diag_SIZE"] = round(float(d[cols.index("SIZE")]), 4)
        rows.append(r)

    diff = np.array(auc["mmd_rbf"]) - np.array(auc["coral"])
    mean, lo, hi = ci(diff)
    summary = {"metric": "AUC(MMD-rbf) - AUC(CORAL) on LR, held-out, paired",
               "mean": round(mean, 5), "ci95_lo": round(lo, 5), "ci95_hi": round(hi, 5),
               "n_seeds": len(diff)}

    keys = sorted({k for r in rows for k in r})
    with open(OUT / "mmd_vs_coral_lr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    md = ["# MMD-rbf versus CORAL on the linear carrier", "",
          "Mean over the 15 seeds of the main matrix. `sd_gap_X` is "
          "|std(aligned source) - std(target)| for feature X in the standardised space; "
          "smaller is better aligned.", "",
          "| method | sd_gap WIDTH | sd_gap LENGTH | sd_gap SIZE | LR AUC | ||W-I||_F | diag(W) WIDTH | LENGTH | SIZE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {} | {} | {} | {} |".format(
            r["method"], r["sd_gap_WIDTH"], r["sd_gap_LENGTH"], r["sd_gap_SIZE"], r["auc_lr"],
            r.get("W_minus_I_frob", "--"), r.get("W_diag_WIDTH", "--"),
            r.get("W_diag_LENGTH", "--"), r.get("W_diag_SIZE", "--")))
    md += ["", f"Paired difference on LR AUC: {summary['mean']:+.5f} "
               f"[{summary['ci95_lo']:+.5f}, {summary['ci95_hi']:+.5f}] over {summary['n_seeds']} seeds."]
    (OUT / "mmd_vs_coral_lr.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "mmd_vs_coral_lr.json").write_text(
        json.dumps({"rows": rows, "paired_difference": summary}, indent=1), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
