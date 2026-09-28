"""Re-run of every quantitative result of the paper with HELD-OUT target
evaluation (revision 2).

Rationale. In the S0 design the target is a perturbed copy of the whole
sample, so 70 % of the target events are perturbed copies of the carrier's
training events. Deep trees memorise those events, and the full-target AUC
of a source-only Random Forest (0.909) is inflated relative to the AUC on
the 30 % of events it has never seen (0.759). Every result below is
therefore computed on the perturbed copies of the 30 % held-out source
events only. Training, preprocessing, alignment (fitted on the full
unlabeled source and target, as before) and hyperparameters are unchanged.

Blocks (all written to paper_electronics/data/heldout.json):
  matrix : 5 carriers x {srconly, mean_matching, coral, mmd_rbf} x 15 seeds,
           headline shift. AUC, macro-F1 and Q at 0.5, plus the shift
           damage D = AUC(source held-out, unperturbed) - AUC(target held-out).
  depth  : RF max_depth sweep x {srconly, mean_matching, coral} x 15 seeds.
  shift  : loc / nsb / head05 / head20 x 5 carriers x 3 methods x 5 seeds.
  q      : Q(theta) grid, source-selected theta*, oracle, 5 carriers x 3
           methods x 15 seeds.
  lgbm   : LightGBM unbounded depth x 3 methods x 15 seeds.

    python scripts/run_heldout_matrix.py --which all
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
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ablations import (  # noqa: E402
    CARRIER_TAG,
    CARRIERS,
    DEPTHS,
    HEADLINE_COMPOSITE,
    SEEDS_5,
    SEEDS_15,
    SHIFT_CONFIGS,
    TRAIN_RATIO,
    apply_composite,
    apply_log_transform,
    load_base,
    make_classifier,
    preprocess_features,
)
from run_revision2 import MIN_HADRONS, THRESHOLDS, q_at  # noqa: E402
from cherenkov_sim2real.adaptation.coral import CORAL  # noqa: E402
from cherenkov_sim2real.adaptation.mean_matching import MeanMatching  # noqa: E402
from cherenkov_sim2real.adaptation.mmd_alignment import MMDAlignment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("cherenkov_sim2real").setLevel(logging.WARNING)
logger = logging.getLogger("heldout")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "paper_electronics" / "data" / "heldout.json"
METHODS3 = ["srconly", "mean_matching", "coral"]
METHODS4 = METHODS3 + ["mmd_rbf"]


def prepare(x_all, y_all, seed, families=None, intensity=1.0, with_rbf=False):
    fam = [(n, i * intensity) for n, i in (families or HEADLINE_COMPOSITE)]
    x_target = apply_composite(x_all, fam, seed=seed)
    xs, ys = x_all.to_numpy(), y_all.to_numpy()
    idx_tr, idx_val = train_test_split(np.arange(len(xs)), train_size=TRAIN_RATIO,
                                       random_state=seed, stratify=ys)
    cols = list(x_all.columns)
    xs_proc, xt_proc, scaler, _ = preprocess_features(
        pd.DataFrame(xs, columns=cols), pd.DataFrame(x_target.to_numpy(), columns=cols))
    tr = scaler.transform(apply_log_transform(pd.DataFrame(xs[idx_tr], columns=cols)).values)
    val = scaler.transform(apply_log_transform(pd.DataFrame(xs[idx_val], columns=cols)).values)
    aligners = {"srconly": None, "mean_matching": MeanMatching().fit(xs_proc, xt_proc),
                "coral": CORAL(lambda_reg=1e-3).fit(xs_proc, xt_proc)}
    if with_rbf:
        aligners["mmd_rbf"] = MMDAlignment(kernel="rbf", lambda_reg=1e-3, max_iter=100).fit(xs_proc, xt_proc)
    tf = lambda a, x: x if a is None else a.transform(x)  # noqa: E731
    return {
        "tr": {m: tf(a, tr) for m, a in aligners.items()},
        "val_src": {m: tf(a, val) for m, a in aligners.items()},
        "y_tr": ys[idx_tr], "y_val": ys[idx_val],
        "xt_ho": xt_proc[idx_val], "yt_ho": ys[idx_val],
        "x_val_unperturbed": val,
    }


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {"mean": float(d.mean()) if n else None, "ci95": None, "n": n}
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(d.mean() - tc * se), 4),
            round(float(d.mean() + tc * se), 4)], "n": n, "per_seed": [round(float(v), 4) for v in d]}


def summ(v: list[float]) -> dict[str, float | None]:
    a = np.array([x for x in v if x is not None], float)
    return {"mean": round(float(a.mean()), 4) if len(a) else None,
            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else None, "n": int(len(a))}


def cell(clf, seed, d, m, max_depth=None, lgbm_unbounded=False):
    model = make_classifier(clf, seed, max_depth)
    if lgbm_unbounded:
        model.set_params(max_depth=-1, num_leaves=255)
    model.fit(d["tr"][m], d["y_tr"])
    p = model.predict_proba(d["xt_ho"])[:, 1]
    y = d["yt_ho"]
    rec = {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= 0.5, average="macro")),
           "q05": q_at(y, p, 0.5)}
    if m == "srconly":
        p0 = model.predict_proba(d["x_val_unperturbed"])[:, 1]
        rec["auc_src_heldout"] = float(roc_auc_score(d["y_val"], p0))
    return rec, model, p


def block_matrix(x_all, y_all):
    out = {}
    for clf in CARRIERS:
        tag = CARRIER_TAG[clf]
        recs = {m: [] for m in METHODS4}
        for seed in SEEDS_15:
            t0 = time.perf_counter()
            d = prepare(x_all, y_all, seed, with_rbf=True)
            for m in METHODS4:
                recs[m].append(cell(clf, seed, d, m)[0])
            logger.info("[matrix] %s seed=%d src=%.4f coral=%.4f (%.1fs)", tag, seed,
                        recs["srconly"][-1]["auc"], recs["coral"][-1]["auc"], time.perf_counter() - t0)
        out[tag] = {}
        for m in METHODS4:
            out[tag][m] = {k: summ([r[k] for r in recs[m]]) for k in ("auc", "f1", "q05")}
            out[tag][m]["per_seed"] = recs[m]
            if m != "srconly":
                for k in ("auc", "f1", "q05"):
                    out[tag][m][f"delta_{k}"] = paired([r[k] for r in recs[m]], [r[k] for r in recs["srconly"]])
        out[tag]["srconly"]["auc_src_heldout"] = summ([r["auc_src_heldout"] for r in recs["srconly"]])
        out[tag]["damage"] = paired([r["auc_src_heldout"] for r in recs["srconly"]],
                                    [r["auc"] for r in recs["srconly"]])
    return out


def block_depth(x_all, y_all):
    out = {}
    for depth in DEPTHS:
        tag = "none" if depth is None else f"{depth:02d}"
        recs = {m: [] for m in METHODS3}
        for seed in SEEDS_15:
            d = prepare(x_all, y_all, seed)
            for m in METHODS3:
                recs[m].append(cell("random_forest", seed, d, m, max_depth=depth)[0])
        out[tag] = {"max_depth": depth}
        for m in METHODS3:
            out[tag][m] = summ([r["auc"] for r in recs[m]])
            if m != "srconly":
                out[tag][m]["delta_auc"] = paired([r["auc"] for r in recs[m]], [r["auc"] for r in recs["srconly"]])
        out[tag]["coral_minus_mm"] = paired([r["auc"] for r in recs["coral"]], [r["auc"] for r in recs["mean_matching"]])
        logger.info("[depth] %s src=%.4f mm=%+.4f coral=%+.4f c-mm=%+.4f", tag, out[tag]["srconly"]["mean"],
                    out[tag]["mean_matching"]["delta_auc"]["mean"], out[tag]["coral"]["delta_auc"]["mean"],
                    out[tag]["coral_minus_mm"]["mean"])
    return out


def block_shift(x_all, y_all):
    out = {}
    for cfg, (families, intensity) in SHIFT_CONFIGS.items():
        out[cfg] = {}
        for clf in CARRIERS:
            tag = CARRIER_TAG[clf]
            recs = {m: [] for m in METHODS3}
            for seed in SEEDS_5:
                d = prepare(x_all, y_all, seed, families=families, intensity=intensity)
                for m in METHODS3:
                    recs[m].append(cell(clf, seed, d, m)[0])
            out[cfg][tag] = {}
            for m in METHODS3:
                out[cfg][tag][m] = summ([r["auc"] for r in recs[m]])
                if m != "srconly":
                    out[cfg][tag][m]["delta_auc"] = paired([r["auc"] for r in recs[m]], [r["auc"] for r in recs["srconly"]])
            out[cfg][tag]["damage"] = paired([r["auc_src_heldout"] for r in recs["srconly"]], [r["auc"] for r in recs["srconly"]])
            logger.info("[shift] %s %s src=%.4f dmg=%+.4f mm=%+.4f coral=%+.4f", cfg, tag, out[cfg][tag]["srconly"]["mean"],
                        out[cfg][tag]["damage"]["mean"], out[cfg][tag]["mean_matching"]["delta_auc"]["mean"],
                        out[cfg][tag]["coral"]["delta_auc"]["mean"])
    return out


def block_q(x_all, y_all, carriers=None, lgbm_unbounded=False):
    out = {}
    for clf in carriers or CARRIERS:
        tag = CARRIER_TAG[clf] + ("_deep" if lgbm_unbounded else "")
        recs = {m: [] for m in METHODS3}
        for seed in SEEDS_15:
            d = prepare(x_all, y_all, seed)
            for m in METHODS3:
                rec, model, pt = cell(clf, seed, d, m, lgbm_unbounded=lgbm_unbounded)
                pv = model.predict_proba(d["val_src"][m])[:, 1]
                yt, yv = d["yt_ho"], d["y_val"]
                grid = {str(t): q_at(yt, pt, t) for t in THRESHOLDS}
                valid = {t: q for t in THRESHOLDS if (q := q_at(yv, pv, t)) is not None}
                thr = max(valid, key=valid.get) if valid else 0.5
                vals = [q for q in grid.values() if q is not None]
                rec.update({"thr_star": thr, "q_thr_star": q_at(yt, pt, thr),
                            "q_oracle": max(vals) if vals else None, "q_grid": grid})
                recs[m].append(rec)
        out[tag] = {}
        for m in METHODS3:
            out[tag][m] = {k: summ([r[k] for r in recs[m]]) for k in ("auc", "q05", "q_thr_star", "q_oracle")}
            out[tag][m]["thr_star_mean"] = round(float(np.mean([r["thr_star"] for r in recs[m]])), 3)
            out[tag][m]["q_grid_mean"] = {str(t): (round(float(np.mean(v)), 4) if len(v := [r["q_grid"][str(t)] for r in recs[m] if r["q_grid"][str(t)] is not None]) == len(recs[m]) else None) for t in THRESHOLDS}
            if m != "srconly":
                for k in ("auc", "q05", "q_thr_star", "q_oracle"):
                    out[tag][m][f"delta_{k}"] = paired([r[k] for r in recs[m]], [r[k] for r in recs["srconly"]])
        logger.info("[q] %s Q05 src=%.2f coral=%.2f | Q* src=%.2f coral=%.2f", tag, out[tag]["srconly"]["q05"]["mean"] or -1,
                    out[tag]["coral"]["q05"]["mean"] or -1, out[tag]["srconly"]["q_thr_star"]["mean"] or -1,
                    out[tag]["coral"]["q_thr_star"]["mean"] or -1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["matrix", "depth", "shift", "q", "lgbm", "all"], default="all")
    a = ap.parse_args()
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    x_all, y_all = load_base()
    blocks = {"matrix": block_matrix, "depth": block_depth, "shift": block_shift, "q": block_q,
              "lgbm": lambda x, y: block_q(x, y, carriers=["lightgbm"], lgbm_unbounded=True)}
    for name, fn in blocks.items():
        if a.which in (name, "all"):
            res[name] = fn(x_all, y_all)
            OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
            logger.info("block %s written", name)
    logger.info("Done -> %s", OUT)


if __name__ == "__main__":
    main()
