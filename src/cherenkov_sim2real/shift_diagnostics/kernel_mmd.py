"""Kernel MMD with RBF kernel and permutation test.

Empirical Maximum Mean Discrepancy between two samples, with
median-heuristic bandwidth and a permutation test for statistical
significance.

All random state is fixed via the seed parameter for reproducibility.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


def compute_kernel_mmd(
    X_source: np.ndarray[Any, Any],
    X_target: np.ndarray[Any, Any],
    n_permutations: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute empirical MMD^2 with RBF kernel and permutation p-value.

    Returns
    -------
    {"value": float, "p_value": float, "n_permutations": int, "bandwidth": float}
    """
    rng = np.random.default_rng(seed)

    # Subsample for computational tractability
    max_n = 2000
    xs = X_source[:max_n] if len(X_source) > max_n else X_source
    xt = X_target[:max_n] if len(X_target) > max_n else X_target

    # Median heuristic bandwidth
    combined = np.vstack([xs[:500], xt[:500]])
    dists = pdist(combined)
    median_dist = np.median(dists)
    gamma = 1.0 / (2.0 * median_dist**2 + 1e-10)

    # Observed MMD^2
    mmd_observed = _mmd_squared(xs, xt, gamma)

    # Permutation test
    pooled = np.vstack([xs, xt])
    n_s = len(xs)
    n_perm_exceeding = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(pooled))
        xs_perm = pooled[perm[:n_s]]
        xt_perm = pooled[perm[n_s:]]
        mmd_perm = _mmd_squared(xs_perm, xt_perm, gamma)
        if mmd_perm >= mmd_observed:
            n_perm_exceeding += 1

    p_value = (n_perm_exceeding + 1) / (n_permutations + 1)

    logger.info(
        "Kernel MMD: value=%.6f, p=%.4f (n_perm=%d, gamma=%.4f)",
        mmd_observed, p_value, n_permutations, gamma,
    )

    return {
        "value": float(mmd_observed),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "bandwidth": float(1.0 / (2.0 * gamma)),
    }


def _mmd_squared(
    X: np.ndarray[Any, Any],
    Y: np.ndarray[Any, Any],
    gamma: float,
) -> float:
    """Compute biased empirical MMD^2 with RBF kernel."""
    k_xx = np.exp(-gamma * _pairwise_sq_dists(X, X))
    k_yy = np.exp(-gamma * _pairwise_sq_dists(Y, Y))
    k_xy = np.exp(-gamma * _pairwise_sq_dists(X, Y))
    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


def _pairwise_sq_dists(
    X: np.ndarray[Any, Any],
    Y: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Squared Euclidean distances between all pairs."""
    result: np.ndarray[Any, Any] = (
        np.sum(X**2, axis=1, keepdims=True)
        + np.sum(Y**2, axis=1)
        - 2 * X @ Y.T
    )
    return result
