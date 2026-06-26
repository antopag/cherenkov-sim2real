"""MLflow configuration for offline, local-file-backend tracking.

All experiment runs log to a local ``results/_mlruns/`` directory.
No network calls are made. See CLAUDE.md §5 and PLAN.md §5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import mlflow

from cherenkov_sim2real.tracking.manifest import manifest_to_json

logger = logging.getLogger(__name__)


def configure_mlflow(
    experiment_name: str,
    tracking_uri: Path | None = None,
) -> None:
    """Set up MLflow for offline, file-based tracking.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name. Created if it does not exist.
    tracking_uri:
        Directory for the MLflow backend store. Defaults to
        ``results/_mlruns`` relative to the current working directory.
    """
    if tracking_uri is None:
        tracking_uri = Path("results/_mlruns")
    tracking_uri.mkdir(parents=True, exist_ok=True)

    uri = tracking_uri.resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI: %s, experiment: %s", uri, experiment_name)


def log_manifest(
    manifest: dict[str, Any],
    hydra_run_dir: Path | None = None,
) -> None:
    """Write a manifest as an MLflow artifact and optionally to disk.

    Parameters
    ----------
    manifest:
        The manifest dict to log.
    hydra_run_dir:
        If provided, also writes ``manifest.json`` here for direct
        inspection without MLflow tooling.
    """
    manifest_json = manifest_to_json(manifest)

    # Log as MLflow artifact
    run = mlflow.active_run()
    if run is not None:
        raw_uri = mlflow.get_artifact_uri().removeprefix("file:///").removeprefix("file:")
        tmp_dir = Path(unquote(raw_uri))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = tmp_dir / "manifest.json"
        artifact_path.write_text(manifest_json, encoding="utf-8")
        logger.info("Manifest logged as MLflow artifact: %s", artifact_path)
    else:
        logger.warning("No active MLflow run; manifest not logged as artifact.")

    # Also write to Hydra run dir for direct access
    if hydra_run_dir is not None:
        hydra_run_dir.mkdir(parents=True, exist_ok=True)
        disk_path = hydra_run_dir / "manifest.json"
        disk_path.write_text(manifest_json, encoding="utf-8")
        logger.info("Manifest written to %s", disk_path)
