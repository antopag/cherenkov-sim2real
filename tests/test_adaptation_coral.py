"""Tests for CORAL domain adaptation."""

from __future__ import annotations

from typing import Any

import numpy as np

from cherenkov_sim2real.adaptation.coral import CORAL


def _make_shifted_pair(
    n: int = 500, d: int = 5, seed: int = 42
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Create source and target with different mean and covariance."""
    rng = np.random.default_rng(seed)
    mean_s = np.zeros(d)
    mean_t = np.ones(d) * 2.0
    cov_s = np.eye(d) + rng.uniform(0, 0.3, (d, d))
    cov_s = cov_s @ cov_s.T  # ensure PSD
    cov_t = np.eye(d) * 2.0 + rng.uniform(0, 0.2, (d, d))
    cov_t = cov_t @ cov_t.T
    X_s = rng.multivariate_normal(mean_s, cov_s, n)
    X_t = rng.multivariate_normal(mean_t, cov_t, n)
    return X_s, X_t


def test_coral_identity_when_distributions_match() -> None:
    """When source ≈ target, transform is approximately identity."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 5))
    # Add tiny noise to make them "different" datasets from same dist
    X_s = X + rng.normal(0, 0.01, X.shape)
    X_t = X + rng.normal(0, 0.01, X.shape)

    coral = CORAL(lambda_reg=1e-3)
    X_aligned = coral.fit_transform(X_s, X_t)

    # Transformed source should be very close to original source
    np.testing.assert_allclose(X_aligned, X_s, atol=0.15)


def test_coral_aligns_means() -> None:
    """After transform, mean of transformed source is close to mean of target."""
    X_s, X_t = _make_shifted_pair()
    coral = CORAL(lambda_reg=1e-3)
    X_aligned = coral.fit_transform(X_s, X_t)

    mean_aligned = np.mean(X_aligned, axis=0)
    mean_target = np.mean(X_t, axis=0)
    np.testing.assert_allclose(mean_aligned, mean_target, atol=0.2)


def test_coral_aligns_covariances() -> None:
    """After transform, cov of transformed source is closer to cov of target."""
    X_s, X_t = _make_shifted_pair(n=1000)
    coral = CORAL(lambda_reg=1e-3)
    X_aligned = coral.fit_transform(X_s, X_t)

    cov_before = np.cov(X_s, rowvar=False)
    cov_after = np.cov(X_aligned, rowvar=False)
    cov_target = np.cov(X_t, rowvar=False)

    dist_before = np.linalg.norm(cov_before - cov_target, "fro")
    dist_after = np.linalg.norm(cov_after - cov_target, "fro")

    assert dist_after < dist_before, (
        f"Covariance should be closer after CORAL: before={dist_before:.4f}, after={dist_after:.4f}"
    )


def test_coral_determinism() -> None:
    """Same input produces same output."""
    X_s, X_t = _make_shifted_pair()
    r1 = CORAL(lambda_reg=1e-3).fit_transform(X_s, X_t)
    r2 = CORAL(lambda_reg=1e-3).fit_transform(X_s, X_t)
    np.testing.assert_array_equal(r1, r2)


def test_coral_handles_singular() -> None:
    """With lambda_reg > 0, runs without error on rank-deficient input."""
    rng = np.random.default_rng(42)
    # Create rank-deficient data: 10 features but only 3 independent
    base = rng.standard_normal((200, 3))
    proj = rng.standard_normal((3, 10))
    X_s = base @ proj + rng.normal(0, 0.01, (200, 10))
    X_t = (base + 1.0) @ proj + rng.normal(0, 0.01, (200, 10))

    coral = CORAL(lambda_reg=1e-2)
    X_aligned = coral.fit_transform(X_s, X_t)
    assert X_aligned.shape == X_s.shape
    assert np.isfinite(X_aligned).all()
