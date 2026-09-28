"""Why the full-target protocol inflates the measured DA benefit.

For each carrier and seed we train the source-only model on the 70 % source
split and then score it on four sets:

    train / unperturbed   the events it was fitted on
    train / perturbed     the same events after the headline shift
    holdout / unperturbed events excluded from training, no shift
    holdout / perturbed   the same events after the shift  <- honest target

The gap between the first and third column is memorisation. The full-target
figures reported in the submitted version are the 70/30 mixture of columns
two and four, so the higher a carrier's memorisation, the more its reported
target performance is really training performance, and the more the shift
appears to damage it, because a memorised fit is brittle under perturbation.
Feature alignment moves the perturbed training events back towards where
the carrier memorised them, which is where the inflated "benefit" comes
from.

Writes paper_electronics/data/memorisation.json.

    python scripts/run_memorisation_check.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("cherenkov_sim2real").setLevel(logging.WARNING)
logger = logging.getLogger("memorisation")
OUT = Path(__file__).resolve().parent.parent / "paper_electronics" / "data" / "memorisation.json"
KEYS = ["train_unpert", "train_pert", "holdout_unpert", "holdout_pert",
        "full_target", "train_pert_coral", "holdout_pert_coral", "full_target_coral"]


def main() -> None:
    x_all, y_all = load_base()
    xs, ys = x_all.to_numpy(), y_all.to_numpy()
    cols = list(x_all.columns)
    out: dict[str, Any] = {}

    for clf in CARRIERS:
        tag = CARRIER_TAG[clf]
        per: dict[str, list[float]] = {k: [] for k in KEYS}
        for seed in SEEDS_15:
            xt = apply_composite(x_all, list(HEADLINE_COMPOSITE), seed=seed)
            itr, iva = train_test_split(np.arange(len(xs)), train_size=TRAIN_RATIO,
                                        random_state=seed, stratify=ys)
            xs_p, xt_p, sc, _ = preprocess_features(
                pd.DataFrame(xs, columns=cols), pd.DataFrame(xt.to_numpy(), columns=cols))
            tr = sc.transform(apply_log_transform(pd.DataFrame(xs[itr], columns=cols)).values)
            al = CORAL(lambda_reg=1e-3).fit(xs_p, xt_p)

            m = make_classifier(clf, seed).fit(tr, ys[itr])
            auc = lambda X, Y: float(roc_auc_score(Y, m.predict_proba(X)[:, 1]))  # noqa: E731
            per["train_unpert"].append(auc(xs_p[itr], ys[itr]))
            per["train_pert"].append(auc(xt_p[itr], ys[itr]))
            per["holdout_unpert"].append(auc(xs_p[iva], ys[iva]))
            per["holdout_pert"].append(auc(xt_p[iva], ys[iva]))
            per["full_target"].append(auc(xt_p, ys))

            mc = make_classifier(clf, seed).fit(al.transform(tr), ys[itr])
            aucc = lambda X, Y: float(roc_auc_score(Y, mc.predict_proba(X)[:, 1]))  # noqa: E731
            per["train_pert_coral"].append(aucc(xt_p[itr], ys[itr]))
            per["holdout_pert_coral"].append(aucc(xt_p[iva], ys[iva]))
            per["full_target_coral"].append(aucc(xt_p, ys))

        out[tag] = {k: {"mean": round(float(np.mean(v)), 4),
                        "std": round(float(np.std(v, ddof=1)), 4)} for k, v in per.items()}
        out[tag]["per_seed"] = {k: [round(x, 4) for x in v] for k, v in per.items()}
        g_full = np.array(per["full_target_coral"]) - np.array(per["full_target"])
        g_ho = np.array(per["holdout_pert_coral"]) - np.array(per["holdout_pert"])
        g_tr = np.array(per["train_pert_coral"]) - np.array(per["train_pert"])
        out[tag]["gain_full_target"] = round(float(g_full.mean()), 4)
        out[tag]["gain_heldout"] = round(float(g_ho.mean()), 4)
        out[tag]["gain_on_training_events"] = round(float(g_tr.mean()), 4)
        out[tag]["memorisation_gap"] = round(float(np.mean(np.array(per["train_unpert"])
                                                          - np.array(per["holdout_unpert"]))), 4)
        logger.info("[mem] %-4s train=%.4f holdout=%.4f gap=%.4f | gain full=%+.4f heldout=%+.4f "
                    "on-train=%+.4f", tag, out[tag]["train_unpert"]["mean"],
                    out[tag]["holdout_unpert"]["mean"], out[tag]["memorisation_gap"],
                    out[tag]["gain_full_target"], out[tag]["gain_heldout"],
                    out[tag]["gain_on_training_events"])

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    logger.info("Done -> %s", OUT)


if __name__ == "__main__":
    main()
