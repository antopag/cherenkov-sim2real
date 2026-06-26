"""Tests for MLflow integration and manifest schema."""

from __future__ import annotations

from pathlib import Path

import mlflow

from cherenkov_sim2real.tracking.manifest import (
    manifest_from_json,
    manifest_to_json,
    validate_manifest,
)
from cherenkov_sim2real.tracking.mlflow_setup import configure_mlflow, log_manifest


def _minimal_manifest() -> dict:  # type: ignore[type-arg]
    """Return a manifest dict with all required fields populated."""
    return {
        "experiment_name": "test_experiment",
        "scenario": "smoke",
        "method": "logreg-srconly",
        "seed": 42,
        "dataset_source": {"name": "uci_magic", "version": "1.0"},
        "dataset_target": None,
        "hyperparameters": {"max_iter": 1000},
        "metrics": {"f1_macro": 0.75, "auc": 0.82},
        "git_commit": "abc123",
        "timestamp_utc": "2026-05-04T12:00:00Z",
        "hydra_config_path": "configs/experiment/smoke_test.yaml",
        "notes": "",
    }


def test_manifest_schema_required_fields() -> None:
    """All required fields must be present; missing ones are reported."""
    full = _minimal_manifest()
    assert validate_manifest(full) == []

    # Remove a required field
    partial = {k: v for k, v in full.items() if k != "seed"}
    missing = validate_manifest(partial)
    assert missing == ["seed"]

    # Remove multiple
    partial2 = {k: v for k, v in full.items() if k not in {"seed", "scenario"}}
    missing2 = validate_manifest(partial2)
    assert missing2 == ["scenario", "seed"]


def test_manifest_serialization_roundtrip() -> None:
    """Serialize to JSON and back; assert equality."""
    original = _minimal_manifest()
    text = manifest_to_json(original)
    restored = manifest_from_json(text)
    assert restored == original


def test_mlflow_offline_run(tmp_path: Path) -> None:
    """Configure MLflow with a tmp_path, run a mock experiment, verify."""
    mlruns_dir = tmp_path / "mlruns"
    configure_mlflow("test_offline", tracking_uri=mlruns_dir)

    manifest = _minimal_manifest()

    with mlflow.start_run() as run:
        mlflow.log_param("seed", 42)
        mlflow.log_metric("f1_macro", 0.75)
        log_manifest(manifest, hydra_run_dir=tmp_path / "hydra_output")

    # Verify run exists in local backend
    run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    fetched = client.get_run(run_id)
    assert fetched.data.params["seed"] == "42"
    assert fetched.data.metrics["f1_macro"] == 0.75

    # Verify manifest artifact was logged
    artifacts = client.list_artifacts(run_id)
    artifact_names = [a.path for a in artifacts]
    assert "manifest.json" in artifact_names

    # Verify manifest written to hydra_run_dir
    hydra_manifest = tmp_path / "hydra_output" / "manifest.json"
    assert hydra_manifest.exists()
    restored = manifest_from_json(hydra_manifest.read_text(encoding="utf-8"))
    assert restored["experiment_name"] == "test_experiment"
