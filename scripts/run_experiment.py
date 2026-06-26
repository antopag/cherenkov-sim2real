"""Hydra entry point for a single experiment cell.

Usage:
    python scripts/run_experiment.py experiment=smoke_test
    python scripts/run_experiment.py experiment=s0_srconly seed=42
    python scripts/run_experiment.py experiment=s0_coral seed=42

Supports two modes:
  - "smoke" scenario: single-dataset train/test split (UCI MAGIC, Prod5).
  - "S0" scenario: source/target from build_s0, with optional CORAL.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import hydra
import mlflow
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from cherenkov_sim2real.adaptation.coral import CORAL
from cherenkov_sim2real.adaptation.mean_matching import MeanMatching
from cherenkov_sim2real.adaptation.mmd_alignment import MMDAlignment
from cherenkov_sim2real.data.cta_prod5 import load_prod5_dl1
from cherenkov_sim2real.data.magic_uci import load_uci_magic
from cherenkov_sim2real.data.preprocessing import apply_log_transform, preprocess_features
from cherenkov_sim2real.metrics.physics import q_factor
from cherenkov_sim2real.scenarios.s0 import build_s0
from cherenkov_sim2real.tracking.manifest import validate_manifest
from cherenkov_sim2real.tracking.mlflow_setup import configure_mlflow, log_manifest

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _git_commit() -> str:
    """Return the current git SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return "unknown"


def _compute_metrics(
    y_test: np.ndarray,  # type: ignore[type-arg]
    y_pred: np.ndarray,  # type: ignore[type-arg]
    y_prob: np.ndarray,  # type: ignore[type-arg]
) -> dict[str, float]:
    """Compute F1, AUC, Q-factor. Returns dict with rounded values."""
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    n_classes = len(set(y_test))
    auc = float(roc_auc_score(y_test, y_prob)) if n_classes > 1 else float("nan")
    q = q_factor(y_test, y_pred)
    return {"f1_macro": round(f1, 4), "auc": round(auc, 4), "q_factor": round(q, 4)}


def _run_smoke(cfg: DictConfig, project_root: Path) -> dict[str, Any]:
    """Smoke-test mode: single dataset, train/test split."""
    # Load data
    if cfg.data.name == "cta_prod5":
        file_path = project_root / cfg.data.data_dir / cfg.data.file_name
        X, y = load_prod5_dl1(file_path)
    else:
        X, y = load_uci_magic(project_root / cfg.data.data_dir)

    ratios = list(cfg.data.split_ratios)
    _train_ratio, val_ratio, test_ratio = ratios
    n_classes = y.nunique()
    stratify_col = y if n_classes > 1 else None
    x_train, x_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_ratio + test_ratio, random_state=cfg.seed, stratify=stratify_col
    )
    val_frac_of_temp = val_ratio / (val_ratio + test_ratio)
    stratify_temp = y_temp if n_classes > 1 else None
    x_val, x_test, _y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=1 - val_frac_of_temp, random_state=cfg.seed, stratify=stratify_temp
    )
    logger.info("Train: %d, Val: %d, Test: %d", len(x_train), len(x_val), len(x_test))

    model = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    if n_classes > 1:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        logger.warning("Single-class data; predicting majority class.")
        y_pred = np.full(len(y_test), y_train.iloc[0])
        y_prob = np.ones(len(y_test))

    metrics = _compute_metrics(y_test.to_numpy(), y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "scenario": "smoke",
        "method": f"{cfg.model.name}-srconly",
        "metrics": metrics,
        "confusion_matrix": cm,
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "dataset_source": {"name": cfg.data.name, "version": "public"},
        "dataset_target": None,
        "hyperparameters": {"max_iter": 1000},
    }


def _run_s0(cfg: DictConfig, project_root: Path) -> dict[str, Any]:
    """S0 scenario: source-only or CORAL on Prod5 baseline vs perturbed."""
    data_dir = project_root / cfg.scenario.data_dir
    s0 = build_s0(
        data_dir,
        perturbation_intensity=cfg.scenario.perturbation_intensity,
        seed=cfg.seed,
        subsample=cfg.scenario.get("subsample"),
    )

    x_source = np.asarray(s0["X_source"])
    y_source = np.asarray(s0["y_source"])
    x_target = np.asarray(s0["X_target"])
    y_target = np.asarray(s0["y_target"])

    # Train/test split on source side for classifier training
    ratios = list(cfg.scenario.split_ratios)
    train_ratio = ratios[0]
    x_src_train, x_src_val, y_src_train, _y_src_val = train_test_split(
        x_source, y_source, train_size=train_ratio, random_state=cfg.seed, stratify=y_source
    )

    method = cfg.method.name

    # Preprocessing: log-transform heavy-tailed features + StandardScaler
    cols: list[str] = list(s0["X_source"].columns)  # type: ignore[union-attr]
    x_source_proc, x_target_proc, scaler, _log_feats = preprocess_features(
        pd.DataFrame(x_source, columns=cols),
        pd.DataFrame(x_target, columns=cols),
    )
    # Scale training subset with the same fitted scaler
    x_src_train_df = pd.DataFrame(x_src_train, columns=cols)
    x_src_train_log = apply_log_transform(x_src_train_df)
    x_src_train_proc = scaler.transform(x_src_train_log.values)

    if method == "coral":
        coral = CORAL(lambda_reg=cfg.method.get("lambda_reg", 1e-3))
        coral.fit(x_source_proc, x_target_proc)
        x_train_final = coral.transform(x_src_train_proc)
    elif method == "mean_matching":
        mm = MeanMatching()
        mm.fit(x_source_proc, x_target_proc)
        x_train_final = mm.transform(x_src_train_proc)
    elif method in ("mmd_linear", "mmd_rbf"):
        kernel: Literal["linear", "rbf"] = "linear" if method == "mmd_linear" else "rbf"
        mmd = MMDAlignment(
            kernel=kernel,
            lambda_reg=cfg.method.get("lambda_reg", 1e-3),
            max_iter=cfg.method.get("max_iter", 100),
        )
        mmd.fit(x_source_proc, x_target_proc)
        x_train_final = mmd.transform(x_src_train_proc)
    elif method == "srconly":
        x_train_final = x_src_train_proc
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    # Train classifier on (aligned) source, test on target
    classifier_name = cfg.get("classifier", "logistic_regression")
    clf_params = cfg.get("classifier_params", {})
    if classifier_name == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=clf_params.get("n_estimators", 300),
            max_depth=clf_params.get("max_depth", 6),
            learning_rate=clf_params.get("learning_rate", 0.05),
            random_state=cfg.seed,
            verbose=-1,
        )
    elif classifier_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=clf_params.get("n_estimators", 100),
            criterion=clf_params.get("criterion", "entropy"),
            max_depth=clf_params.get("max_depth", None),
            min_samples_split=clf_params.get("min_samples_split", 2),
            random_state=cfg.seed,
            n_jobs=1,
        )
    elif classifier_name == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(
            n_estimators=clf_params.get("n_estimators", 100),
            criterion=clf_params.get("criterion", "entropy"),
            max_depth=clf_params.get("max_depth", None),
            min_samples_split=clf_params.get("min_samples_split", 2),
            random_state=cfg.seed,
            n_jobs=1,
        )
    elif classifier_name == "mlp_shallow":
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(
            hidden_layer_sizes=tuple(clf_params.get("hidden_layer_sizes", [64, 32])),
            activation=clf_params.get("activation", "relu"),
            solver=clf_params.get("solver", "adam"),
            alpha=clf_params.get("alpha", 1e-4),
            batch_size=clf_params.get("batch_size", 256),
            max_iter=clf_params.get("max_iter", 200),
            early_stopping=clf_params.get("early_stopping", True),
            validation_fraction=clf_params.get("validation_fraction", 0.1),
            random_state=cfg.seed,
        )
    else:
        model = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    model.fit(x_train_final, y_src_train)

    y_pred = model.predict(x_target_proc)
    y_prob = model.predict_proba(x_target_proc)[:, 1]

    metrics = _compute_metrics(y_target, y_pred, y_prob)
    cm = confusion_matrix(y_target, y_pred).tolist()

    logger.info(
        "S0 %s: F1=%.4f, AUC=%.4f, Q=%.4f",
        method,
        metrics["f1_macro"],
        metrics["auc"],
        metrics["q_factor"],
    )

    return {
        "scenario": "S0",
        "method": f"logreg-{method}",
        "metrics": metrics,
        "confusion_matrix": cm,
        "n_train": len(x_src_train),
        "n_val": len(x_src_val),
        "n_test": len(x_target),
        "dataset_source": {"name": "cta_prod5_baseline", "version": "public"},
        "dataset_target": {"name": "cta_prod5_perturbed", "intensity": cfg.scenario.perturbation_intensity},
        "hyperparameters": {
            "max_iter": 1000,
            "method": method,
            "lambda_reg": cfg.method.get("lambda_reg", None),
        },
    }


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    _set_seeds(cfg.seed)

    project_root = Path(hydra.utils.get_original_cwd())
    configure_mlflow(
        experiment_name=cfg.experiment_name,
        tracking_uri=project_root / "results" / "_mlruns",
    )

    # Dispatch by scenario
    scenario_name = cfg.get("scenario", {}).get("name", "smoke")
    if scenario_name == "S0":
        run_result = _run_s0(cfg, project_root)
    else:
        run_result = _run_smoke(cfg, project_root)

    elapsed = time.perf_counter() - t0
    metrics = run_result["metrics"]

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "seed": cfg.seed,
            "experiment_name": cfg.experiment_name,
            "scenario": run_result["scenario"],
            "method": run_result["method"],
        })

        # Log metrics
        for k, v in metrics.items():
            if not np.isnan(v):
                mlflow.log_metric(k, v)
        mlflow.log_metric("wall_clock_seconds", round(elapsed, 2))

        # Build and log manifest
        manifest: dict = {  # type: ignore[type-arg]
            "experiment_name": cfg.experiment_name,
            "scenario": run_result["scenario"],
            "method": run_result["method"],
            "seed": cfg.seed,
            "dataset_source": run_result["dataset_source"],
            "dataset_target": run_result.get("dataset_target"),
            "hyperparameters": run_result["hyperparameters"],
            "metrics": metrics,
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "hydra_config_path": OmegaConf.to_yaml(cfg),
            "notes": "",
        }

        missing = validate_manifest(manifest)
        if missing:
            logger.error("Manifest missing required fields: %s", missing)

        hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)
        log_manifest(manifest, hydra_run_dir=hydra_run_dir)

    # JSON output
    output_dir = project_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": cfg.experiment_name,
        "scenario": run_result["scenario"],
        "method": run_result["method"],
        "seed": cfg.seed,
        "n_train": run_result["n_train"],
        "n_val": run_result["n_val"],
        "n_test": run_result["n_test"],
        "metrics": metrics,
        "confusion_matrix": run_result["confusion_matrix"],
        "wall_clock_seconds": round(elapsed, 2),
    }
    out_path = output_dir / f"{cfg.experiment_name}_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n=== {cfg.experiment_name} RESULTS ===")
    print(f"Method:    {run_result['method']}")
    print(f"F1 macro:  {metrics['f1_macro']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Q-factor:  {metrics['q_factor']:.4f}")
    print(f"Confusion matrix:\n{np.array(run_result['confusion_matrix'])}")
    print(f"Output:    {out_path}")
    print(f"Wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
