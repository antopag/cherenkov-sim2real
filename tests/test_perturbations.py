"""Tests for synthetic perturbation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cherenkov_sim2real.perturbations import (
    HEADLINE_COMPOSITE,
    apply_composite,
    apply_perturbation,
)

_COLUMNS = ["LENGTH", "WIDTH", "SIZE", "CONC", "CONC1", "ASYM", "M3LONG", "DIST", "ALPHA", "M3TRANS"]

# Families that preserve event count (a-d) vs reduce it (e).
_PRESERVING = ["nsb_injection", "flat_field_uncertainty", "atmospheric_attenuation", "zenith_size_shift"]
_REDUCING = ["energy_threshold_shift"]
_ALL = [*_PRESERVING, *_REDUCING]


@pytest.fixture()
def synthetic_X() -> pd.DataFrame:
    """Small synthetic dataset with realistic-ish values."""
    rng = np.random.default_rng(99)
    n = 500
    return pd.DataFrame(
        {
            "LENGTH": rng.uniform(0.1, 1.0, n),
            "WIDTH": rng.uniform(0.05, 0.5, n),
            "SIZE": rng.uniform(50, 10000, n),
            "CONC": rng.uniform(0.1, 0.8, n),
            "CONC1": rng.uniform(0.05, 0.5, n),
            "ASYM": rng.normal(0, 0.5, n),
            "M3LONG": rng.normal(0, 1, n),
            "DIST": rng.uniform(0.1, 2.0, n),
            "ALPHA": rng.uniform(-np.pi, np.pi, n),
            "M3TRANS": rng.uniform(-np.pi, np.pi, n),
        }
    )


# --- Identity at zero intensity ---

@pytest.mark.parametrize("family", _ALL)
def test_identity_at_zero_intensity(synthetic_X: pd.DataFrame, family: str) -> None:
    result = apply_perturbation(family, synthetic_X, intensity=0.0, seed=42)
    pd.testing.assert_frame_equal(result, synthetic_X)


# --- Determinism ---

@pytest.mark.parametrize("family", _ALL)
def test_determinism_same_seed(synthetic_X: pd.DataFrame, family: str) -> None:
    r1 = apply_perturbation(family, synthetic_X, intensity=0.7, seed=42)
    r2 = apply_perturbation(family, synthetic_X, intensity=0.7, seed=42)
    pd.testing.assert_frame_equal(r1, r2)


@pytest.mark.parametrize("family", ["nsb_injection", "flat_field_uncertainty"])
def test_determinism_different_seeds(synthetic_X: pd.DataFrame, family: str) -> None:
    r1 = apply_perturbation(family, synthetic_X, intensity=0.7, seed=42)
    r2 = apply_perturbation(family, synthetic_X, intensity=0.7, seed=99)
    assert not r1.equals(r2), f"{family} should differ with different seeds"


# --- Purity ---

@pytest.mark.parametrize("family", _ALL)
def test_purity_no_mutation(synthetic_X: pd.DataFrame, family: str) -> None:
    original = synthetic_X.copy()
    apply_perturbation(family, synthetic_X, intensity=1.0, seed=42)
    pd.testing.assert_frame_equal(synthetic_X, original)


# --- Schema preserved ---

@pytest.mark.parametrize("family", _ALL)
def test_schema_preserved(synthetic_X: pd.DataFrame, family: str) -> None:
    result = apply_perturbation(family, synthetic_X, intensity=1.0, seed=42)
    assert list(result.columns) == _COLUMNS


# --- Event count ---

@pytest.mark.parametrize("family", _PRESERVING)
def test_event_count_preserved(synthetic_X: pd.DataFrame, family: str) -> None:
    result = apply_perturbation(family, synthetic_X, intensity=1.0, seed=42)
    assert len(result) == len(synthetic_X)


def test_event_count_decreases(synthetic_X: pd.DataFrame) -> None:
    result = apply_perturbation("energy_threshold_shift", synthetic_X, intensity=1.0, seed=42)
    assert len(result) < len(synthetic_X)


# --- Intensity monotonicity ---

def test_nsb_monotonicity(synthetic_X: pd.DataFrame) -> None:
    """Mean noise (|SIZE_perturbed - SIZE_original|) increases with intensity."""
    diffs = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("nsb_injection", synthetic_X, intensity=intensity, seed=42)
        diffs.append(np.mean(np.abs(r["SIZE"].values - synthetic_X["SIZE"].values)))
    assert diffs[0] <= diffs[1] <= diffs[2]


def test_atmospheric_monotonicity(synthetic_X: pd.DataFrame) -> None:
    """Mean SIZE decreases monotonically with intensity."""
    means = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("atmospheric_attenuation", synthetic_X, intensity=intensity, seed=42)
        means.append(r["SIZE"].mean())
    assert means[0] >= means[1] >= means[2]


def test_flatfield_monotonicity(synthetic_X: pd.DataFrame) -> None:
    """Std of SIZE/original_SIZE ratio increases with intensity."""
    stds = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("flat_field_uncertainty", synthetic_X, intensity=intensity, seed=42)
        ratio = r["SIZE"].values / synthetic_X["SIZE"].values
        stds.append(np.std(ratio))
    assert stds[0] <= stds[1] <= stds[2]


def test_zenith_monotonicity(synthetic_X: pd.DataFrame) -> None:
    """Mean SIZE decreases monotonically with intensity."""
    means = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("zenith_size_shift", synthetic_X, intensity=intensity, seed=42)
        means.append(r["SIZE"].mean())
    assert means[0] >= means[1] >= means[2]


def test_threshold_monotonicity(synthetic_X: pd.DataFrame) -> None:
    """Event count decreases monotonically with intensity."""
    counts = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("energy_threshold_shift", synthetic_X, intensity=intensity, seed=42)
        counts.append(len(r))
    assert counts[0] >= counts[1] >= counts[2]


# --- WIDTH / LENGTH propagation (families 1, 3, 4) ---

def test_nsb_width_increases(synthetic_X: pd.DataFrame) -> None:
    """Mean WIDTH increases monotonically with NSB intensity."""
    means = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("nsb_injection", synthetic_X, intensity=intensity, seed=42)
        means.append(r["WIDTH"].mean())
    assert means[0] <= means[1] <= means[2]


def test_nsb_length_increases(synthetic_X: pd.DataFrame) -> None:
    """Mean LENGTH increases monotonically with NSB intensity."""
    means = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("nsb_injection", synthetic_X, intensity=intensity, seed=42)
        means.append(r["LENGTH"].mean())
    assert means[0] <= means[1] <= means[2]


def test_atmospheric_width_decreases(synthetic_X: pd.DataFrame) -> None:
    """Mean WIDTH decreases monotonically with atmospheric intensity."""
    means = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("atmospheric_attenuation", synthetic_X, intensity=intensity, seed=42)
        means.append(r["WIDTH"].mean())
    assert means[0] >= means[1] >= means[2]


def test_atmospheric_length_decreases(synthetic_X: pd.DataFrame) -> None:
    """Mean LENGTH decreases monotonically with atmospheric intensity."""
    means = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("atmospheric_attenuation", synthetic_X, intensity=intensity, seed=42)
        means.append(r["LENGTH"].mean())
    assert means[0] >= means[1] >= means[2]


def test_zenith_length_increases(synthetic_X: pd.DataFrame) -> None:
    """Mean LENGTH increases monotonically with zenith intensity (oblique viewing)."""
    means = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("zenith_size_shift", synthetic_X, intensity=intensity, seed=42)
        means.append(r["LENGTH"].mean())
    assert means[0] <= means[1] <= means[2]


def test_zenith_width_increases(synthetic_X: pd.DataFrame) -> None:
    """Mean WIDTH increases monotonically with zenith intensity (secondary)."""
    means = []
    for intensity in [0.0, 0.5, 1.0]:
        r = apply_perturbation("zenith_size_shift", synthetic_X, intensity=intensity, seed=42)
        means.append(r["WIDTH"].mean())
    assert means[0] <= means[1] <= means[2]


# --- Family 2: WIDTH/LENGTH deliberately unchanged ---

def test_flatfield_width_unchanged(synthetic_X: pd.DataFrame) -> None:
    """Flat-fielding is photometric only — WIDTH must not change."""
    r = apply_perturbation("flat_field_uncertainty", synthetic_X, intensity=1.0, seed=42)
    np.testing.assert_array_equal(r["WIDTH"].values, synthetic_X["WIDTH"].values)


def test_flatfield_length_unchanged(synthetic_X: pd.DataFrame) -> None:
    """Flat-fielding is photometric only — LENGTH must not change."""
    r = apply_perturbation("flat_field_uncertainty", synthetic_X, intensity=1.0, seed=42)
    np.testing.assert_array_equal(r["LENGTH"].values, synthetic_X["LENGTH"].values)


# --- Composite ---

def test_composite_reproducibility(synthetic_X: pd.DataFrame) -> None:
    r1 = apply_composite(synthetic_X, HEADLINE_COMPOSITE, seed=42)
    r2 = apply_composite(synthetic_X, HEADLINE_COMPOSITE, seed=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_headline_composite_default(synthetic_X: pd.DataFrame) -> None:
    """Headline composite changes SIZE (NSB + atmospheric attenuation)."""
    result = apply_composite(synthetic_X, HEADLINE_COMPOSITE, seed=42)
    assert list(result.columns) == _COLUMNS
    assert len(result) == len(synthetic_X)
    # SIZE should be different from the original
    assert not np.allclose(result["SIZE"].values, synthetic_X["SIZE"].values)
    # Mean SIZE should be lower (atmospheric attenuation dominates)
    assert result["SIZE"].mean() < synthetic_X["SIZE"].mean()
