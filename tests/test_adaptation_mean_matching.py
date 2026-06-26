"""Tests for Mean Matching domain adaptation."""

from __future__ import annotations

import numpy as np

from cherenkov_sim2real.adaptation.mean_matching import MeanMatching


def test_mean_matching_aligns_means() -> None:
    """After transform, mean of transformed source equals mean of target."""
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((500, 5))
    X_t = rng.standard_normal((500, 5)) + 3.0

    mm = MeanMatching()
    X_aligned = mm.fit_transform(X_s, X_t)

    np.testing.assert_allclose(
        np.mean(X_aligned, axis=0), np.mean(X_t, axis=0), atol=1e-10
    )


def test_mean_matching_preserves_covariance() -> None:
    """Mean shift does not change covariance."""
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((500, 5))
    X_t = rng.standard_normal((500, 5)) + 3.0

    mm = MeanMatching()
    X_aligned = mm.fit_transform(X_s, X_t)

    cov_before = np.cov(X_s, rowvar=False)
    cov_after = np.cov(X_aligned, rowvar=False)
    np.testing.assert_allclose(cov_after, cov_before, atol=1e-10)


def test_mean_matching_determinism() -> None:
    """Same input produces same output."""
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((100, 5))
    X_t = rng.standard_normal((100, 5)) + 2.0

    r1 = MeanMatching().fit_transform(X_s, X_t)
    r2 = MeanMatching().fit_transform(X_s, X_t)
    np.testing.assert_array_equal(r1, r2)


def test_mean_matching_purity() -> None:
    """Input X is not mutated."""
    rng = np.random.default_rng(42)
    X_s = rng.standard_normal((100, 5))
    X_t = rng.standard_normal((100, 5)) + 2.0
    X_s_copy = X_s.copy()

    MeanMatching().fit_transform(X_s, X_t)
    np.testing.assert_array_equal(X_s, X_s_copy)
