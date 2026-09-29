"""Causal test of the overlap explanation: vary how much of the target the
carrier was trained on, and watch the inflation follow.

If the inflated full-target numbers are caused by the overlap between the
training set and the target set, then shrinking the training fraction must
shrink the inflation, and it must do so for the carriers that memorise and
not for the one that cannot. This script varies the training fraction over
{0.1, 0.3, 0.5, 0.7, 0.9} and, for each carrier, reports

    gain_full      CORAL AUC gain scored on the whole perturbed sample
    gain_heldout   the same, scored only on events excluded from training
    inflation      gain_full / gain_heldout

The held-out gain should be roughly independent of the training fraction
(beyond ordinary learning-curve effects), while the full-target gain and the
inflation should rise with it for high-capacity carriers only.

Also records the exact decomposition of the pooled AUC into the three
populations of pairs it counts, which identifies which of them the
inflation lives in:

    within-train    gamma and hadron both from the training events
    within-heldout  both from the held-out events
    cross           one from each

Writes paper_electronics/data/overlap_doseresponse.json.

    python scripts/run_overlap_doseresponse.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cherenkov_sim2real.adaptation.coral import CORAL
from run_ablations import (
    CARRIER_TAG,
    CARRIERS,
    HEADLINE_COMPOSITE,
    apply_composite,
    apply_log_transform,
    load_base,
    make_classifier,
    preprocess_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("cherenkov_sim2real").setLevel(logging.WARNING)
logger = logging.getLogger("overlap")
OUT = Path(__file__).resolve().parent.parent / "paper_electronics" / "data" / "overlap_doseresponse.json"
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = list(range(42, 47))


def pair_auc(y_a, s_a, y_b, s_b) -> float | None:
    """P(score of a gamma from A > score of a hadron from B), ties at 0.5."""
    g, h = s_a[y_a == 1], s_b[y_b == 0]
    if len(g) == 0 or len(h) == 0:
        return None
    gt = (g[:, None] > h[None, :]).sum()
    eq = (g[:, None] == h[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(g) * len(h)))


def main() -> None:
    x_all, y_all = load_base()
    xs, ys = x_all.to_numpy(), y_all.to_numpy()
    cols = list(x_all.columns)
    out: dict[str, Any] = {}

    for frac in FRACTIONS:
        key = f"{frac:.1f}"
        out[key] = {}
        for clf in CARRIERS:
            tag = CARRIER_TAG[clf]
            rec: dict[str, list[float]] = {k: [] for k in
                                           ("gain_full", "gain_heldout", "auc_full_src",
                                            "auc_ho_src", "within_train", "within_ho", "cross")}
            for seed in SEEDS:
                xt = apply_composite(x_all, list(HEADLINE_COMPOSITE), seed=seed)
                itr, iva = train_test_split(np.arange(len(xs)), train_size=frac,
                                            random_state=seed, stratify=ys)
                xs_p, xt_p, sc, _ = preprocess_features(
                    pd.DataFrame(xs, columns=cols), pd.DataFrame(xt.to_numpy(), columns=cols))
                tr = sc.transform(apply_log_transform(pd.DataFrame(xs[itr], columns=cols)).values)
                al = CORAL(lambda_reg=1e-3).fit(xs_p, xt_p)

                m0 = make_classifier(clf, seed).fit(tr, ys[itr])
                m1 = make_classifier(clf, seed).fit(al.transform(tr), ys[itr])
                p0, p1 = m0.predict_proba(xt_p)[:, 1], m1.predict_proba(xt_p)[:, 1]

                a0f, a1f = roc_auc_score(ys, p0), roc_auc_score(ys, p1)
                a0h, a1h = roc_auc_score(ys[iva], p0[iva]), roc_auc_score(ys[iva], p1[iva])
                rec["gain_full"].append(a1f - a0f)
                rec["gain_heldout"].append(a1h - a0h)
                rec["auc_full_src"].append(a0f)
                rec["auc_ho_src"].append(a0h)
                # pooled-AUC decomposition for the unadapted model
                rec["within_train"].append(pair_auc(ys[itr], p0[itr], ys[itr], p0[itr]) or np.nan)
                rec["within_ho"].append(pair_auc(ys[iva], p0[iva], ys[iva], p0[iva]) or np.nan)
                cross = [pair_auc(ys[itr], p0[itr], ys[iva], p0[iva]),
                         pair_auc(ys[iva], p0[iva], ys[itr], p0[itr])]
                rec["cross"].append(float(np.nanmean([c for c in cross if c is not None])))

            agg = {k: round(float(np.nanmean(v)), 4) for k, v in rec.items()}
            gh = agg["gain_heldout"]
            agg["inflation"] = round(agg["gain_full"] / gh, 2) if abs(gh) > 1e-6 else None
            out[key][tag] = agg
            logger.info("[dose] frac=%.1f %-4s gain_full=%+.4f gain_ho=%+.4f infl=%s | "
                        "AUC pairs: train=%.3f ho=%.3f cross=%.3f", frac, tag, agg["gain_full"],
                        agg["gain_heldout"], agg["inflation"], agg["within_train"],
                        agg["within_ho"], agg["cross"])

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info("Done -> %s", OUT)


if __name__ == "__main__":
    main()
