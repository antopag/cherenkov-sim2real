"""Tests for shift scenarios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cherenkov_sim2real.scenarios.s0 import build_s0

_DATA_DIR = Path("data/raw/cta_prod5/dl1")


@pytest.fixture()
def s0_data() -> dict:  # type: ignore[type-arg]
    """Build S0 with subsample for fast tests."""
    return build_s0(_DATA_DIR, subsample=2000, seed=42)


@pytest.mark.data
def test_s0_shape_consistency(s0_data: dict) -> None:  # type: ignore[type-arg]
    """Source and target have same schema and same n_rows."""
    xs = s0_data["X_source"]
    xt = s0_data["X_target"]
    assert isinstance(xs, pd.DataFrame)
    assert isinstance(xt, pd.DataFrame)
    assert list(xs.columns) == list(xt.columns)
    assert len(xs) == len(xt)


@pytest.mark.data
def test_s0_labels_identical(s0_data: dict) -> None:  # type: ignore[type-arg]
    """y_source equals y_target (same events, perturbed features only)."""
    ys = s0_data["y_source"]
    yt = s0_data["y_target"]
    pd.testing.assert_series_equal(ys, yt)


@pytest.mark.data
def test_s0_features_differ(s0_data: dict) -> None:  # type: ignore[type-arg]
    """X_target differs from X_source on perturbed columns."""
    xs = s0_data["X_source"]
    xt = s0_data["X_target"]
    # SIZE should be shifted (NSB + atmospheric attenuation)
    assert not np.allclose(xs["SIZE"].values, xt["SIZE"].values)
    # Mean SIZE should be lower in target (atmospheric attenuation)
    assert xt["SIZE"].mean() < xs["SIZE"].mean()


@pytest.mark.data
def test_s0_metadata_complete(s0_data: dict) -> None:  # type: ignore[type-arg]
    """Metadata dict has all required keys."""
    meta = s0_data["metadata"]
    required = {
        "scenario", "n_events", "n_gamma", "n_proton", "gamma_fraction",
        "perturbation_intensity", "headline_composite", "seed", "subsample",
        "schema",
    }
    assert required <= set(meta.keys())
    assert meta["scenario"] == "S0"
    assert meta["n_events"] == 2000
    assert meta["n_gamma"] + meta["n_proton"] == 2000


@pytest.mark.data
def test_s0_subsample() -> None:
    """subsample=1000 returns 1000 rows in source and target."""
    data = build_s0(_DATA_DIR, subsample=1000, seed=42)
    assert len(data["X_source"]) == 1000
    assert len(data["X_target"]) == 1000


@pytest.mark.data
def test_s0_determinism() -> None:
    """Same seed produces identical X_target and metadata."""
    d1 = build_s0(_DATA_DIR, subsample=500, seed=42)
    d2 = build_s0(_DATA_DIR, subsample=500, seed=42)
    pd.testing.assert_frame_equal(d1["X_target"], d2["X_target"])
    assert d1["metadata"] == d2["metadata"]
