"""Tests for domain-shift diagnostics."""

from __future__ import annotations

import numpy as np

from cherenkov_sim2real.shift_diagnostics import compute_shift_diagnostics
from cherenkov_sim2real.shift_diagnostics.calibration_drift import compute_calibration_drift
from cherenkov_sim2real.shift_diagnostics.kernel_mmd import compute_kernel_mmd
from cherenkov_sim2real.shift_diagnostics.proxy_a_distance import compute_proxy_a_distance


def _matching_pair(n: int = 500, d: int = 5, seed: int = 42) -> tuple:  # type: ignore[type-arg]
    """Two samples from the same distribution."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, d))
    return x, y


def _shifted_pair(n: int = 500, d: int = 5, seed: int = 42) -> tuple:  # type: ignore[type-arg]
    """Two samples with different means."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, d)) + 2.0
    return x, y


def test_mmd_zero_when_match() -> None:
    """MMD near zero and p-value > 0.05 when distributions match."""
    xs, xt = _matching_pair()
    result = compute_kernel_mmd(xs, xt, n_permutations=100, seed=42)
    assert result["p_value"] > 0.05, f"p={result['p_value']}, expected >0.05"
    assert result["value"] < 0.05, f"MMD={result['value']}, expected near 0"


def test_mmd_positive_when_shifted() -> None:
    """MMD > 0 with p < 0.05 when distributions differ."""
    xs, xt = _shifted_pair()
    result = compute_kernel_mmd(xs, xt, n_permutations=100, seed=42)
    assert result["p_value"] < 0.05, f"p={result['p_value']}, expected <0.05"
    assert result["value"] > 0.01


def test_proxy_a_distance_zero_when_match() -> None:
    """A-distance near zero when distributions match."""
    xs, xt = _matching_pair()
    result = compute_proxy_a_distance(xs, xt, seed=42)
    assert result["value"] < 0.2, f"A-dist={result['value']}, expected near 0"


def test_proxy_a_distance_positive_when_shifted() -> None:
    """A-distance > 0 when distributions differ."""
    xs, xt = _shifted_pair()
    result = compute_proxy_a_distance(xs, xt, seed=42)
    assert result["value"] > 0.5, f"A-dist={result['value']}, expected >0.5"


def test_calibration_drift_zero_when_match() -> None:
    """Calibration drift near zero when source ≈ target."""
    rng = np.random.default_rng(42)
    xs = rng.standard_normal((500, 5))
    xt = rng.standard_normal((500, 5))
    ys = (xs[:, 0] > 0).astype(int)
    yt = (xt[:, 0] > 0).astype(int)
    result = compute_calibration_drift(xs, xt, ys, yt, seed=42)
    assert abs(result["delta"]) < 0.1, f"delta={result['delta']}, expected near 0"


def test_compute_shift_diagnostics_full() -> None:
    """Full API: all keys present, values within expected ranges."""
    rng = np.random.default_rng(42)
    xs = rng.standard_normal((300, 5))
    xt = rng.standard_normal((300, 5)) + 1.0
    ys = (xs[:, 0] > 0).astype(int)
    yt = (xt[:, 0] > 0).astype(int)

    result = compute_shift_diagnostics(xs, xt, ys, yt, seed=42)

    assert "mmd" in result
    assert "proxy_a_distance" in result
    assert "calibration_drift" in result
    assert result["mmd"]["value"] >= 0
    assert 0 <= result["mmd"]["p_value"] <= 1
    assert result["proxy_a_distance"]["value"] >= 0
    assert result["calibration_drift"] is not None
    assert "ece_source" in result["calibration_drift"]
    assert "ece_target" in result["calibration_drift"]
    assert "delta" in result["calibration_drift"]
