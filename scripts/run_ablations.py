"""Reviewer-requested ablations on the S0 benchmark (revision round 1).

Two ablations, both on the S0 machinery (Prod5 baseline vs synthetic
perturbation, unsupervised feature alignment, carrier trained on aligned
source and evaluated on the perturbed target):

A. ``depth``  -- Random Forest max_depth sweep, headline composite at
   intensity 1.0, methods srconly / mean_matching / coral, 15 seeds.
   Tests the depth-modulation hypothesis (Discussion, "Tree Depth as a
   Modulator") within a single ensemble family.

B. ``shift``  -- shift type and intensity sweep over all five carriers,
   methods srconly / mean_matching / coral, 5 seeds:
     loc     : atmospheric_attenuation only (multiplicative -> location shift)
     nsb     : nsb_injection only (additive Poisson-like noise -> covariance)
     head05  : headline composite at intensity 0.5
     head20  : headline composite at intensity 2.0
   (headline at 1.0 is the main S0 matrix already on disk.)

The Prod5 sample is loaded once; everything else replicates
``scripts/run_experiment.py::_run_s0`` step by step (same split, same
preprocessing, same alignment, same classifier hyperparameters). One
``manifest.json`` per (config, method, seed) is written under
``results/abl_<name>/<timestamp>_seed<N>/`` so that the extraction and
reproducibility chain is the same as for the main matrix.

Usage (inside the ``cherenkov`` env, from the project root)::

    python scripts/run_ablations.py --which depth
    python scripts/run_ablations.py --which shift
    python scripts/run_ablations.py --which all
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from cherenkov_sim2real.adaptation.coral import CORAL
from cherenkov_sim2real.adaptation.mean_matching import MeanMatching
from cherenkov_sim2real.data.cta_prod5 import load_prod5_dl1
from cherenkov_sim2real.data.preprocessing import (
    apply_log_transform,
    apply_quality_cuts,
    preprocess_features,
)
from cherenkov_sim2real.metrics.physics import q_factor
from cherenkov_sim2real.perturbations import HEADLINE_COMPOSITE, apply_composite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ablations")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cta_prod5" / "dl1"
GAMMA_FILE = "gamma-diffuse_with_images_40.dl2.h5"
PROTON_FILE = "proton_with_images_00.dl2.h5"
TRAIN_RATIO = 0.70  # configs/experiment/s0_*.yaml split_ratios[0]

SEEDS_15 = list(range(42, 57))
SEEDS_5 = list(range(42, 47))
METHODS = ["srconly", "mean_matching", "coral"]
DEPTHS: list[int | None] = [2, 4, 6, 8, 12, 16, None]

SHIFT_CONFIGS: dict[str, tuple[list[tuple[str, float]], float]] = {
    "loc": ([("atmospheric_attenuation", 1.0)], 1.0),
    "nsb": ([("nsb_injection", 1.0)], 1.0),
    "head05": (list(HEADLINE_COMPOSITE), 0.5),
    "head20": (list(HEADLINE_COMPOSITE), 2.0),
}
CARRIERS = ["logistic_regression", "mlp_shallow", "lightgbm", "extratrees", "random_forest"]
CARRIER_TAG = {
    "logistic_regression": "lr",
    "mlp_shallow": "mlp",
    "lightgbm": "lgbm",
    "extratrees": "et",
    "random_forest": "rf",
}


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


def make_classifier(name: str, seed: int, max_depth: int | None = None) -> Any:
    """Same hyperparameters as scripts/run_experiment.py (Appendix A)."""
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            max_depth=6 if max_depth is None else max_depth,
            learning_rate=0.05,
            random_state=seed,
            verbose=-1,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            max_depth=max_depth,
            min_samples_split=2,
            random_state=seed,
            n_jobs=1,
        )
    if name == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=100,
            criterion="entropy",
            max_depth=max_depth,
            min_samples_split=2,
            random_state=seed,
            n_jobs=1,
        )
    if name == "mlp_shallow":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
    return LogisticRegression(max_iter=1000, random_state=seed)


def load_base() -> tuple[pd.DataFrame, pd.Series]:
    """Load Prod5 gamma + proton and apply the S0 quality cuts (once)."""
    t0 = time.perf_counter()
    x_g, y_g = load_prod5_dl1(DATA_DIR / GAMMA_FILE)
    x_p, y_p = load_prod5_dl1(DATA_DIR / PROTON_FILE)
    x_g, y_g = apply_quality_cuts(x_g, y_g)
    x_p, y_p = apply_quality_cuts(x_p, y_p)
    x_all = pd.concat([x_g, x_p], ignore_index=True)
    y_all = pd.concat([y_g, y_p], ignore_index=True)
    logger.info("Base sample: %d events (%.1f s)", len(x_all), time.perf_counter() - t0)
    return x_all, y_all


def run_cell(
    x_all: pd.DataFrame,
    y_all: pd.Series,
    *,
    families: list[tuple[str, float]],
    intensity: float,
    method: str,
    classifier: str,
    seed: int,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """One (config, method, classifier, seed) cell; mirrors _run_s0."""
    fam = [(name, inten * intensity) for name, inten in families]
    x_target = apply_composite(x_all, fam, seed=seed)
    x_source, y_source = x_all.to_numpy(), y_all.to_numpy()
    y_target = y_source.copy()

    x_src_train, _x_val, y_src_train, _y_val = train_test_split(
        x_source, y_source, train_size=TRAIN_RATIO, random_state=seed, stratify=y_source
    )
    cols = list(x_all.columns)
    x_source_proc, x_target_proc, scaler, _ = preprocess_features(
        pd.DataFrame(x_source, columns=cols), pd.DataFrame(x_target.to_numpy(), columns=cols)
    )
    x_train_proc = scaler.transform(
        apply_log_transform(pd.DataFrame(x_src_train, columns=cols)).values
    )

    if method == "coral":
        al = CORAL(lambda_reg=1e-3)
        al.fit(x_source_proc, x_target_proc)
        x_train_final = al.transform(x_train_proc)
    elif method == "mean_matching":
        mm = MeanMatching()
        mm.fit(x_source_proc, x_target_proc)
        x_train_final = mm.transform(x_train_proc)
    elif method == "srconly":
        x_train_final = x_train_proc
    else:
        raise ValueError(method)

    model = make_classifier(classifier, seed, max_depth)
    model.fit(x_train_final, y_src_train)
    y_pred = model.predict(x_target_proc)
    y_prob = model.predict_proba(x_target_proc)[:, 1]
    return {
        "f1_macro": round(float(f1_score(y_target, y_pred, average="macro")), 4),
        "auc": round(float(roc_auc_score(y_target, y_prob)), 4),
        "q_factor": round(float(q_factor(y_target, y_pred)), 4),
    }


def write_manifest(
    experiment_name: str,
    method: str,
    seed: int,
    metrics: dict[str, float],
    ablation: dict[str, Any],
    git_sha: str,
) -> None:
    ts = datetime.now(tz=UTC)
    run_dir = PROJECT_ROOT / "results" / experiment_name / f"{ts:%Y-%m-%d_%H-%M-%S}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": experiment_name,
        "scenario": "S0-ablation",
        "method": f"logreg-{method}",
        "seed": seed,
        "dataset_source": {"name": "cta_prod5_baseline", "version": "public"},
        "dataset_target": {
            "name": "cta_prod5_perturbed",
            "families": ablation["families"],
            "intensity": ablation["intensity"],
        },
        "hyperparameters": {"method": method, "lambda_reg": 1e-3 if method == "coral" else None},
        "ablation": ablation,
        "metrics": metrics,
        "git_commit": git_sha,
        "timestamp_utc": ts.isoformat(),
        "hydra_config_path": None,
        "notes": "scripts/run_ablations.py (revision round 1)",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_depth(x_all: pd.DataFrame, y_all: pd.Series, git_sha: str) -> None:
    n_total = len(DEPTHS) * len(METHODS) * len(SEEDS_15)
    k = 0
    for depth in DEPTHS:
        tag = "none" if depth is None else f"{depth:02d}"
        for method in METHODS:
            exp = f"abl_depth_{tag}_{method}_rf"
            for seed in SEEDS_15:
                k += 1
                t0 = time.perf_counter()
                m = run_cell(
                    x_all,
                    y_all,
                    families=list(HEADLINE_COMPOSITE),
                    intensity=1.0,
                    method=method,
                    classifier="random_forest",
                    seed=seed,
                    max_depth=depth,
                )
                write_manifest(
                    exp,
                    method,
                    seed,
                    m,
                    {
                        "kind": "depth",
                        "carrier": "rf",
                        "max_depth": depth,
                        "families": list(HEADLINE_COMPOSITE),
                        "intensity": 1.0,
                    },
                    git_sha,
                )
                logger.info(
                    "[depth %d/%d] %s seed=%d AUC=%.4f Q=%.3f (%.1fs)",
                    k,
                    n_total,
                    exp,
                    seed,
                    m["auc"],
                    m["q_factor"],
                    time.perf_counter() - t0,
                )


def run_shift(x_all: pd.DataFrame, y_all: pd.Series, git_sha: str) -> None:
    n_total = len(SHIFT_CONFIGS) * len(CARRIERS) * len(METHODS) * len(SEEDS_5)
    k = 0
    for cfg_name, (families, intensity) in SHIFT_CONFIGS.items():
        for clf in CARRIERS:
            for method in METHODS:
                exp = f"abl_shift_{cfg_name}_{method}_{CARRIER_TAG[clf]}"
                for seed in SEEDS_5:
                    k += 1
                    t0 = time.perf_counter()
                    m = run_cell(
                        x_all,
                        y_all,
                        families=families,
                        intensity=intensity,
                        method=method,
                        classifier=clf,
                        seed=seed,
                    )
                    write_manifest(
                        exp,
                        method,
                        seed,
                        m,
                        {
                            "kind": "shift",
                            "config": cfg_name,
                            "carrier": CARRIER_TAG[clf],
                            "families": families,
                            "intensity": intensity,
                        },
                        git_sha,
                    )
                    logger.info(
                        "[shift %d/%d] %s seed=%d AUC=%.4f Q=%.3f (%.1fs)",
                        k,
                        n_total,
                        exp,
                        seed,
                        m["auc"],
                        m["q_factor"],
                        time.perf_counter() - t0,
                    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["depth", "shift", "all"], default="all")
    args = ap.parse_args()
    git_sha = _git_commit()
    x_all, y_all = load_base()
    if args.which in ("depth", "all"):
        run_depth(x_all, y_all, git_sha)
    if args.which in ("shift", "all"):
        run_shift(x_all, y_all, git_sha)
    logger.info("Done.")


if __name__ == "__main__":
    main()
