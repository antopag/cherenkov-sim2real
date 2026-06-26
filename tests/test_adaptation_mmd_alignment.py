"""Tests for MMD-based feature alignment."""

from __future__ import annotations

from typing import Any

import numpy as np

from cherenkov_sim2real.adaptation.coral import CORAL
from cherenkov_sim2real.adaptation.mmd_alignment import MMDAlignment


def _make_shifted_pair(
    n: int = 500, d: int = 5, seed: int = 42,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Source and target with different mean and covariance."""
    rng = np.random.default_rng(seed)
    mean_s = np.zeros(d)
    mean_t = np.ones(d) * 2.0
    cov_s = np.eye(d) + rng.uniform(0, 0.3, (d, d))
    cov_s = cov_s @ cov_s.T
    cov_t = np.eye(d) * 2.0 + rng.uniform(0, 0.2, (d, d))
    cov_t = cov_t @ cov_t.T
    return rng.multivariate_normal(mean_s, cov_s, n), rng.multivariate_normal(mean_t, cov_t, n)


def _empirical_mmd(X: np.ndarray[Any, Any], Y: np.ndarray[Any, Any], gamma: float = 1.0) -> float:
    """Simple empirical MMD^2 with RBF kernel for testing."""
    n = min(200, len(X), len(Y))
    xs, yt = X[:n], Y[:n]
    k_ss = np.exp(-gamma * np.sum((xs[:, None] - xs[None, :]) ** 2, axis=2))
    k_tt = np.exp(-gamma * np.sum((yt[:, None] - yt[None, :]) ** 2, axis=2))
    k_st = np.exp(-gamma * np.sum((xs[:, None] - yt[None, :]) ** 2, axis=2))
    return float(k_ss.mean() + k_tt.mean() - 2 * k_st.mean())


def test_mmd_reduces_distance() -> None:
    """Empirical MMD between transformed source and target is smaller."""
    X_s, X_t = _make_shifted_pair()
    mmd_before = _empirical_mmd(X_s, X_t)

    mmd_align = MMDAlignment(kernel="linear", lambda_reg=1e-3)
    X_aligned = mmd_align.fit_transform(X_s, X_t)
    mmd_after = _empirical_mmd(X_aligned, X_t)

    assert mmd_after < mmd_before, (
        f"MMD should decrease: before={mmd_before:.6f}, after={mmd_after:.6f}"
    )


def test_mmd_linear_determinism() -> None:
    """Same input produces same output."""
    X_s, X_t = _make_shifted_pair()
    r1 = MMDAlignment(kernel="linear", lambda_reg=1e-3).fit_transform(X_s, X_t)
    r2 = MMDAlignment(kernel="linear", lambda_reg=1e-3).fit_transform(X_s, X_t)
    np.testing.assert_array_equal(r1, r2)


def test_mmd_rbf_convergence() -> None:
    """RBF optimisation loss is finite and non-negative."""
    X_s, X_t = _make_shifted_pair(n=200, d=3)
    mmd_align = MMDAlignment(kernel="rbf", lambda_reg=1e-2, max_iter=50)
    X_aligned = mmd_align.fit_transform(X_s, X_t)
    assert np.isfinite(X_aligned).all()
    # After optimisation, MMD should be smaller
    mmd_before = _empirical_mmd(X_s, X_t)
    mmd_after = _empirical_mmd(X_aligned, X_t)
    assert mmd_after < mmd_before


def test_mmd_purity() -> None:
    """Input X is not mutated."""
    X_s, X_t = _make_shifted_pair()
    X_s_copy = X_s.copy()
    MMDAlignment(kernel="linear", lambda_reg=1e-3).fit_transform(X_s, X_t)
    np.testing.assert_array_equal(X_s, X_s_copy)


def test_mmd_linear_close_to_coral() -> None:
    """Linear MMD output is numerically close to CORAL output."""
    X_s, X_t = _make_shifted_pair(n=1000)
    coral_out = CORAL(lambda_reg=1e-3).fit_transform(X_s, X_t)
    mmd_out = MMDAlignment(kernel="linear", lambda_reg=1e-3).fit_transform(X_s, X_t)

    # Should be very close (same formula up to minor differences)
    mean_diff = np.abs(coral_out - mmd_out).mean()
    feature_std = np.std(X_s, axis=0).mean()
    relative_diff = mean_diff / feature_std
    assert relative_diff < 0.01, (
        f"Linear MMD and CORAL should match within 1%: relative_diff={relative_diff:.4f}"
    )
