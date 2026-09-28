"""Revision-2 analyses requested by the two Algorithms referees.

All on the S0 machinery (same data, split, preprocessing, alignment and
classifier hyperparameters as ``scripts/run_ablations.py``). Four blocks:

A. ``sens``   Independent, DA-free measure of carrier marginal sensitivity.
   A source-only model is trained on the 70 % source split and probed on the
   30 % held-out source split translated by a uniform offset delta (in
   standardised units) on every feature. We record, per delta and direction,
   the AUC degradation and the fraction of flipped hard predictions. The
   AUC-based index S(delta) is threshold-free and is the quantity later
   correlated with the CORAL gain; the flip rate documents the
   threshold-dependent behaviour.

B. ``q``      Q-factor as a function of the decision threshold, for
   srconly / mean matching / CORAL on every carrier: Q on the target at a
   grid of thresholds, at the fixed 0.5, at the threshold that maximises Q
   on the *source* validation split (transferred unchanged to the target),
   and the oracle maximum on the target.

C. ``cov``    Second-order statistics before and after alignment: Frobenius
   distance between the covariance (and correlation) of the aligned source
   and that of the target, for srconly / mean matching / CORAL.

D. ``lgbm``   LightGBM with unbounded depth (max_depth=-1, num_leaves=255),
   srconly / mean matching / CORAL, to test whether the depth cap of the
   main-matrix LightGBM drives its position in the gradient.

Outputs one JSON: ``paper_electronics/data/revision2.json``.

    python scripts/run_revision2.py --which all
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
from run_ablations import (  # noqa: E402
    CARRIER_TAG,
    CARRIERS,
    HEADLINE_COMPOSITE,
    SEEDS_15,
    TRAIN_RATIO,
    apply_composite,
    apply_log_transform,
    load_base,
    make_classifier,
    preprocess_features,
)
from cherenkov_sim2real.adaptation.coral import CORAL  # noqa: E402
from cherenkov_sim2real.adaptation.mean_matching import MeanMatching  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("revision2")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "paper_electronics" / "data" / "revision2.json"

DELTAS = [0.1, 0.25, 0.5, 1.0]
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
METHODS = ["srconly", "mean_matching", "coral"]
MIN_HADRONS = 50  # guard against Q blowing up as eps_h -> 0


def prepare(x_all: pd.DataFrame, y_all: pd.Series, seed: int) -> dict[str, Any]:
    """Split, preprocess and align exactly as run_ablations.run_cell."""
    fam = list(HEADLINE_COMPOSITE)
    x_target = apply_composite(x_all, fam, seed=seed)
    x_source, y_source = x_all.to_numpy(), y_all.to_numpy()
    y_target = y_source.copy()
    idx = np.arange(len(x_source))
    idx_tr, idx_val = train_test_split(
        idx, train_size=TRAIN_RATIO, random_state=seed, stratify=y_source
    )
    x_tr, x_val, y_tr, y_val = x_source[idx_tr], x_source[idx_val], y_source[idx_tr], y_source[idx_val]
    cols = list(x_all.columns)
    xs_proc, xt_proc, scaler, _ = preprocess_features(
        pd.DataFrame(x_source, columns=cols), pd.DataFrame(x_target.to_numpy(), columns=cols)
    )
    tr_proc = scaler.transform(apply_log_transform(pd.DataFrame(x_tr, columns=cols)).values)
    val_proc = scaler.transform(apply_log_transform(pd.DataFrame(x_val, columns=cols)).values)
    al = CORAL(lambda_reg=1e-3).fit(xs_proc, xt_proc)
    mm = MeanMatching().fit(xs_proc, xt_proc)
    return {
        "cols": cols,
        "xs_proc": xs_proc,
        "xt_proc": xt_proc,
        "y_target": y_target,
        "tr": {"srconly": tr_proc, "mean_matching": mm.transform(tr_proc), "coral": al.transform(tr_proc)},
        "val": {"srconly": val_proc, "mean_matching": mm.transform(val_proc), "coral": al.transform(val_proc)},
        "y_tr": y_tr,
        "y_val": y_val,
        "idx_val": idx_val,
        "aligned_full": {"srconly": xs_proc, "mean_matching": mm.transform(xs_proc), "coral": al.transform(xs_proc)},
    }


def q_at(y: np.ndarray, p: np.ndarray, thr: float) -> float | None:
    pred = p >= thr
    g, h = y == 1, y == 0
    eg = pred[g].mean()
    nh = int(pred[h].sum())
    if nh < MIN_HADRONS:
        return None
    eh = nh / h.sum()
    return float(eg / np.sqrt(eh))


# ----------------------------------------------------------------- A. sens
def block_sens(x_all: pd.DataFrame, y_all: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for clf in CARRIERS:
        tag = CARRIER_TAG[clf]
        out[tag] = {"per_seed": []}
        for seed in SEEDS_15:
            t0 = time.perf_counter()
            d = prepare(x_all, y_all, seed)
            model = make_classifier(clf, seed).fit(d["tr"]["srconly"], d["y_tr"])
            xv, yv = d["val"]["srconly"], d["y_val"]
            p0 = model.predict_proba(xv)[:, 1]
            auc0 = roc_auc_score(yv, p0)
            pred0 = p0 >= 0.5
            rec: dict[str, Any] = {"seed": seed, "auc0": round(float(auc0), 4)}
            for delta in DELTAS:
                drops, flips = [], []
                for sgn in (+1.0, -1.0):
                    p = model.predict_proba(xv + sgn * delta)[:, 1]
                    drops.append(auc0 - roc_auc_score(yv, p))
                    flips.append(float(np.mean((p >= 0.5) != pred0)))
                rec[f"auc_drop_{delta}"] = round(float(np.mean(drops)), 5)
                rec[f"flip_{delta}"] = round(float(np.mean(flips)), 5)
            out[tag]["per_seed"].append(rec)
            logger.info("[sens] %s seed=%d dropAUC(0.25)=%.4f flip(0.25)=%.3f (%.1fs)",
                        tag, seed, rec["auc_drop_0.25"], rec["flip_0.25"], time.perf_counter() - t0)
        for k in [f"auc_drop_{dl}" for dl in DELTAS] + [f"flip_{dl}" for dl in DELTAS]:
            v = np.array([r[k] for r in out[tag]["per_seed"]])
            out[tag][f"{k}_mean"] = round(float(v.mean()), 5)
            out[tag][f"{k}_std"] = round(float(v.std(ddof=1)), 5)
    return out


# ----------------------------------------------------------------- B. q
def block_q(x_all: pd.DataFrame, y_all: pd.Series, carriers: list[str] | None = None,
            lgbm_unbounded: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for clf in carriers or CARRIERS:
        tag = CARRIER_TAG[clf] + ("_deep" if lgbm_unbounded else "")
        out[tag] = {m: {"per_seed": []} for m in METHODS}
        for seed in SEEDS_15:
            t0 = time.perf_counter()
            d = prepare(x_all, y_all, seed)
            for m in METHODS:
                model = make_classifier(clf, seed)
                if lgbm_unbounded:
                    model.set_params(max_depth=-1, num_leaves=255)
                model.fit(d["tr"][m], d["y_tr"])
                pt = model.predict_proba(d["xt_proc"])[:, 1]
                pv = model.predict_proba(d["val"][m])[:, 1]
                yt, yv = d["y_target"], d["y_val"]
                q_grid = {str(t): q_at(yt, pt, t) for t in THRESHOLDS}
                qv_grid = {t: q_at(yv, pv, t) for t in THRESHOLDS}
                valid = {t: q for t, q in qv_grid.items() if q is not None}
                thr_star = max(valid, key=valid.get) if valid else 0.5
                q_target_vals = [q for q in q_grid.values() if q is not None]
                rec = {
                    "seed": seed,
                    "auc": round(float(roc_auc_score(yt, pt)), 4),
                    "q_at_05": q_at(yt, pt, 0.5),
                    "thr_star_source": thr_star,
                    "q_at_thr_star": q_at(yt, pt, thr_star),
                    "q_oracle_target": max(q_target_vals) if q_target_vals else None,
                    "q_grid": q_grid,
                }
                out[tag][m]["per_seed"].append(rec)
            logger.info("[q] %s seed=%d Q05 src=%.2f coral=%.2f | Q* src=%.2f coral=%.2f (%.1fs)", tag, seed,
                        out[tag]["srconly"]["per_seed"][-1]["q_at_05"] or -1,
                        out[tag]["coral"]["per_seed"][-1]["q_at_05"] or -1,
                        out[tag]["srconly"]["per_seed"][-1]["q_at_thr_star"] or -1,
                        out[tag]["coral"]["per_seed"][-1]["q_at_thr_star"] or -1,
                        time.perf_counter() - t0)
        for m in METHODS:
            recs = out[tag][m]["per_seed"]
            for k in ("auc", "q_at_05", "q_at_thr_star", "q_oracle_target"):
                v = np.array([r[k] for r in recs if r[k] is not None], dtype=float)
                out[tag][m][f"{k}_mean"] = round(float(v.mean()), 4) if len(v) else None
                out[tag][m][f"{k}_std"] = round(float(v.std(ddof=1)), 4) if len(v) > 1 else None
            out[tag][m]["thr_star_mean"] = round(float(np.mean([r["thr_star_source"] for r in recs])), 3)
            grid_mean = {}
            for t in THRESHOLDS:
                v = [r["q_grid"][str(t)] for r in recs if r["q_grid"][str(t)] is not None]
                grid_mean[str(t)] = round(float(np.mean(v)), 4) if len(v) == len(recs) else None
            out[tag][m]["q_grid_mean"] = grid_mean
    return out


# ----------------------------------------------------------------- E. heldout
def block_heldout(x_all: pd.DataFrame, y_all: pd.Series) -> dict[str, Any]:
    """Target metrics restricted to the perturbed copies of the 30 % held-out
    source events, i.e. events the carrier has never seen in any form."""
    out: dict[str, Any] = {}
    for clf in CARRIERS:
        tag = CARRIER_TAG[clf]
        out[tag] = {m: {"auc": [], "q05": [], "f1": []} for m in METHODS}
        for seed in SEEDS_15:
            d = prepare(x_all, y_all, seed)
            iv = d["idx_val"]
            xt_ho, yt_ho = d["xt_proc"][iv], d["y_target"][iv]
            for m in METHODS:
                model = make_classifier(clf, seed).fit(d["tr"][m], d["y_tr"])
                p = model.predict_proba(xt_ho)[:, 1]
                out[tag][m]["auc"].append(float(roc_auc_score(yt_ho, p)))
                out[tag][m]["q05"].append(q_at(yt_ho, p, 0.5))
                from sklearn.metrics import f1_score
                out[tag][m]["f1"].append(float(f1_score(yt_ho, p >= 0.5, average="macro")))
        for m in METHODS:
            for k in ("auc", "q05", "f1"):
                v = np.array([x for x in out[tag][m][k] if x is not None], dtype=float)
                out[tag][m][f"{k}_mean"] = round(float(v.mean()), 4)
                out[tag][m][f"{k}_std"] = round(float(v.std(ddof=1)), 4)
        a = np.array(out[tag]["srconly"]["auc"])
        for m in ("mean_matching", "coral"):
            diff = np.array(out[tag][m]["auc"]) - a
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            from scipy import stats
            tc = stats.t.ppf(0.975, len(diff) - 1)
            out[tag][m]["delta_auc"] = round(float(diff.mean()), 4)
            out[tag][m]["delta_auc_ci95"] = [round(float(diff.mean() - tc * se), 4), round(float(diff.mean() + tc * se), 4)]
        logger.info("[heldout] %s srconly=%.4f mm=%+.4f coral=%+.4f", tag, out[tag]["srconly"]["auc_mean"],
                    out[tag]["mean_matching"]["delta_auc"], out[tag]["coral"]["delta_auc"])
    return out


# ----------------------------------------------------------------- C. cov
def block_cov(x_all: pd.DataFrame, y_all: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {m: {"frob_cov": [], "frob_corr": [], "mean_l2": []} for m in METHODS}
    for seed in SEEDS_15:
        d = prepare(x_all, y_all, seed)
        ct = np.cov(d["xt_proc"], rowvar=False)
        rt = np.corrcoef(d["xt_proc"], rowvar=False)
        mt = d["xt_proc"].mean(axis=0)
        for m in METHODS:
            xa = d["aligned_full"][m]
            out[m]["frob_cov"].append(float(np.linalg.norm(np.cov(xa, rowvar=False) - ct)))
            out[m]["frob_corr"].append(float(np.linalg.norm(np.corrcoef(xa, rowvar=False) - rt)))
            out[m]["mean_l2"].append(float(np.linalg.norm(xa.mean(axis=0) - mt)))
    for m in METHODS:
        for k in ("frob_cov", "frob_corr", "mean_l2"):
            v = np.array(out[m][k])
            out[m][f"{k}_mean"] = round(float(v.mean()), 4)
            out[m][f"{k}_std"] = round(float(v.std(ddof=1)), 4)
    # one representative correlation matrix set (seed 42) for a figure
    d = prepare(x_all, y_all, 42)
    out["example_seed42"] = {
        "cols": d["cols"],
        "corr_target": np.corrcoef(d["xt_proc"], rowvar=False).round(3).tolist(),
        **{f"corr_{m}": np.corrcoef(d["aligned_full"][m], rowvar=False).round(3).tolist() for m in METHODS},
    }
    logger.info("[cov] frob_cov srconly=%.3f mm=%.3f coral=%.3f", out["srconly"]["frob_cov_mean"],
                out["mean_matching"]["frob_cov_mean"], out["coral"]["frob_cov_mean"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["sens", "q", "cov", "lgbm", "heldout", "all"], default="all")
    a = ap.parse_args()
    res: dict[str, Any] = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    x_all, y_all = load_base()
    if a.which in ("cov", "all"):
        res["cov"] = block_cov(x_all, y_all)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.which in ("sens", "all"):
        res["sens"] = block_sens(x_all, y_all)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.which in ("lgbm", "all"):
        res["lgbm_deep"] = block_q(x_all, y_all, carriers=["lightgbm"], lgbm_unbounded=True)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.which in ("heldout", "all"):
        res["heldout"] = block_heldout(x_all, y_all)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.which in ("q", "all"):
        res["q"] = block_q(x_all, y_all)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    logger.info("Done -> %s", OUT)


if __name__ == "__main__":
    main()
