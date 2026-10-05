"""Computations required by the review fixes.

A5a  Per-feature source->target change of mean and standard deviation, in the
     standardised space, for the headline composite, the location-only
     (atmospheric) shift and the noise-only (NSB) shift. Shows that the
     "location-only" family is not a pure translation: it rescales WIDTH and
     LENGTH, which are not log-transformed, so their standard deviations move.

A5b  A genuinely pure translation. The target is the standardised source plus
     a constant per-feature offset equal to the mean shift of the headline
     composite, so that the covariance is unchanged by construction. Five
     carriers x {srconly, mean matching, CORAL} x 5 seeds, held-out protocol.
     Prediction: zero damage and zero gain on a linear carrier.

A2   Pooled-AUC pair decomposition at the 70% training fraction over the same
     15 seeds as the main matrix, so that the "both held-out" entry matches
     the held-out AUC of Table 1 rather than a 5-seed subset of it.

A6   Kernel-MMD permutation test with B = 1000, replacing the B = 200 run
     whose smallest attainable p-value was 1/201.

Writes paper_electronics/data/review_checks.json.

    python scripts/run_review_checks.py --which all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cherenkov_sim2real.adaptation.coral import CORAL
from cherenkov_sim2real.adaptation.mean_matching import MeanMatching
from cherenkov_sim2real.shift_diagnostics.kernel_mmd import compute_kernel_mmd
from run_ablations import (
    CARRIER_TAG,
    CARRIERS,
    HEADLINE_COMPOSITE,
    SEEDS_5,
    SEEDS_15,
    TRAIN_RATIO,
    apply_composite,
    load_base,
    make_classifier,
    preprocess_features,
)
from run_heldout_matrix import paired, summ
from run_revision2 import q_at

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("cherenkov_sim2real").setLevel(logging.WARNING)
logger = logging.getLogger("review")
OUT = Path(__file__).resolve().parent.parent / "paper_electronics" / "data" / "review_checks.json"
METHODS3 = ["srconly", "mean_matching", "coral"]
SHIFTS = {
    "headline": list(HEADLINE_COMPOSITE),
    "location_only": [("atmospheric_attenuation", 1.0)],
    "noise_only": [("nsb_injection", 1.0)],
}


def standardised(x_all, families, seed):
    """Return (source, target) in the standardised space used by the pipeline."""
    cols = list(x_all.columns)
    xt = apply_composite(x_all, families, seed=seed)
    xs_proc, xt_proc, _, _ = preprocess_features(
        pd.DataFrame(x_all.to_numpy(), columns=cols),
        pd.DataFrame(xt.to_numpy(), columns=cols))
    return xs_proc, xt_proc, cols


# --------------------------------------------------------------- A5a
def block_feature_stats(x_all, y_all):
    out: dict[str, Any] = {}
    for name, fam in SHIFTS.items():
        dmu, dsd = [], []
        for seed in SEEDS_5:
            xs_p, xt_p, cols = standardised(x_all, fam, seed)
            dmu.append(xt_p.mean(axis=0) - xs_p.mean(axis=0))
            dsd.append(xt_p.std(axis=0) - xs_p.std(axis=0))
        out[name] = {
            "features": cols,
            "d_mean": [round(float(v), 4) for v in np.mean(dmu, axis=0)],
            "d_std": [round(float(v), 4) for v in np.mean(dsd, axis=0)],
            "mean_shift_norm": round(float(np.linalg.norm(np.mean(dmu, axis=0))), 4),
        }
        logger.info("[A5a] %-14s |dmu|=%.4f  max|dsd|=%.4f", name,
                    out[name]["mean_shift_norm"], max(abs(v) for v in out[name]["d_std"]))
    return out


# --------------------------------------------------------------- A5b
def block_pure_translation(x_all, y_all):
    out: dict[str, Any] = {}
    ys = y_all.to_numpy()
    cols = list(x_all.columns)
    for clf in CARRIERS:
        tag = CARRIER_TAG[clf]
        recs: dict[str, list] = {m: [] for m in METHODS3}
        for seed in SEEDS_5:
            # offset = mean shift of the headline composite, same seed
            xs_p, xt_head, _ = standardised(x_all, list(HEADLINE_COMPOSITE), seed)
            offset = xt_head.mean(axis=0) - xs_p.mean(axis=0)
            xt_p = xs_p + offset                      # pure translation
            idx_tr, idx_val = train_test_split(np.arange(len(xs_p)), train_size=TRAIN_RATIO,
                                               random_state=seed, stratify=ys)
            tr = xs_p[idx_tr]
            al = CORAL(lambda_reg=1e-3).fit(xs_p, xt_p)
            mm = MeanMatching().fit(xs_p, xt_p)
            trs = {"srconly": tr, "mean_matching": mm.transform(tr), "coral": al.transform(tr)}
            for m in METHODS3:
                model = make_classifier(clf, seed).fit(trs[m], ys[idx_tr])
                p = model.predict_proba(xt_p[idx_val])[:, 1]
                rec = {"auc": float(roc_auc_score(ys[idx_val], p)),
                       "q05": q_at(ys[idx_val], p, 0.5)}
                if m == "srconly":
                    p0 = model.predict_proba(xs_p[idx_val])[:, 1]
                    rec["auc_src_heldout"] = float(roc_auc_score(ys[idx_val], p0))
                recs[m].append(rec)
        out[tag] = {}
        for m in METHODS3:
            out[tag][m] = summ([r["auc"] for r in recs[m]])
            if m != "srconly":
                out[tag][m]["delta_auc"] = paired([r["auc"] for r in recs[m]],
                                                  [r["auc"] for r in recs["srconly"]])
        out[tag]["damage"] = paired([r["auc_src_heldout"] for r in recs["srconly"]],
                                    [r["auc"] for r in recs["srconly"]])
        logger.info("[A5b] %-4s src=%.4f damage=%+.5f mm=%+.5f coral=%+.5f", tag,
                    out[tag]["srconly"]["mean"], out[tag]["damage"]["mean"],
                    out[tag]["mean_matching"]["delta_auc"]["mean"],
                    out[tag]["coral"]["delta_auc"]["mean"])
    return out


# --------------------------------------------------------------- A2
def pair_auc(y_a, s_a, y_b, s_b):
    g, h = s_a[y_a == 1], s_b[y_b == 0]
    if len(g) == 0 or len(h) == 0:
        return None
    gt = (g[:, None] > h[None, :]).sum()
    eq = (g[:, None] == h[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(g) * len(h)))


def block_pairs_15(x_all, y_all):
    out: dict[str, Any] = {}
    ys = y_all.to_numpy()
    cols = list(x_all.columns)
    for clf in CARRIERS:
        tag = CARRIER_TAG[clf]
        acc = {k: [] for k in ("within_train", "within_ho", "cross", "full_target")}
        for seed in SEEDS_15:
            xs_p, xt_p, _ = standardised(x_all, list(HEADLINE_COMPOSITE), seed)
            itr, iva = train_test_split(np.arange(len(xs_p)), train_size=TRAIN_RATIO,
                                        random_state=seed, stratify=ys)
            tr = xs_p[itr]
            m = make_classifier(clf, seed).fit(tr, ys[itr])
            p = m.predict_proba(xt_p)[:, 1]
            acc["within_train"].append(pair_auc(ys[itr], p[itr], ys[itr], p[itr]))
            acc["within_ho"].append(pair_auc(ys[iva], p[iva], ys[iva], p[iva]))
            acc["cross"].append(float(np.mean([pair_auc(ys[itr], p[itr], ys[iva], p[iva]),
                                               pair_auc(ys[iva], p[iva], ys[itr], p[itr])])))
            acc["full_target"].append(float(roc_auc_score(ys, p)))
        out[tag] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
        logger.info("[A2] %-4s train=%.4f ho=%.4f cross=%.4f full=%.4f", tag,
                    out[tag]["within_train"], out[tag]["within_ho"],
                    out[tag]["cross"], out[tag]["full_target"])
    return out


# --------------------------------------------------------------- A6
def block_permutation(x_all, y_all, n_perm=1000):
    xs_p, xt_p, _ = standardised(x_all, list(HEADLINE_COMPOSITE), 42)
    t0 = time.perf_counter()
    res = compute_kernel_mmd(xs_p, xt_p, n_permutations=n_perm, seed=42)
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    res["p_floor"] = round(1.0 / (n_perm + 1), 5)
    logger.info("[A6] MMD=%.6f p=%.4f (B=%d, floor %.5f) in %.0f s",
                res["value"], res["p_value"], n_perm, res["p_floor"], res["elapsed_s"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["a5a", "a5b", "a2", "a6", "all"], default="all")
    a = ap.parse_args()
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    x_all, y_all = load_base()
    blocks = {"a5a": ("feature_stats", block_feature_stats),
              "a5b": ("pure_translation", block_pure_translation),
              "a2": ("pairs_15seed", block_pairs_15),
              "a6": ("mmd_permutation", block_permutation)}
    for key, (name, fn) in blocks.items():
        if a.which in (key, "all"):
            res[name] = fn(x_all, y_all)
            OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
            logger.info("block %s written", name)
    logger.info("Done -> %s", OUT)


if __name__ == "__main__":
    main()
