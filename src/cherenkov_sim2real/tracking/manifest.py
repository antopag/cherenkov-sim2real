"""Structured manifest schema for experiment runs.

Every experiment run produces a manifest.json that captures the full
provenance of the result: config, data, hyperparameters, metrics, git
state, and timestamps. The manifest is the artifact the paper cites
(see CLAUDE.md §5 and PLAN.md §5).
"""

from __future__ import annotations

import json
from typing import Any, TypedDict


class DatasetInfo(TypedDict, total=False):
    """Dataset provenance record."""

    name: str
    version: str
    sha256: str
    doi: str


class Manifest(TypedDict, total=False):
    """Structured manifest for a single experiment run.

    Required fields are enforced by :func:`validate_manifest`.
    """

    experiment_name: str
    scenario: str  # "S0", "S1", "S2", "S3", "Img-A", "smoke"
    method: str  # e.g. "logreg-srconly", "coral-mlp"
    seed: int
    dataset_source: dict[str, Any]
    dataset_target: dict[str, Any] | None
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    git_commit: str
    timestamp_utc: str
    hydra_config_path: str
    notes: str


_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "experiment_name",
        "scenario",
        "method",
        "seed",
        "dataset_source",
        "hyperparameters",
        "metrics",
        "git_commit",
        "timestamp_utc",
        "hydra_config_path",
    }
)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return a list of missing required fields (empty if valid)."""
    return sorted(_REQUIRED_FIELDS - manifest.keys())


def manifest_to_json(manifest: dict[str, Any]) -> str:
    """Serialize a manifest dict to a JSON string."""
    return json.dumps(manifest, indent=2, ensure_ascii=False)


def manifest_from_json(text: str) -> dict[str, Any]:
    """Deserialize a manifest from a JSON string."""
    return json.loads(text)  # type: ignore[no-any-return]
