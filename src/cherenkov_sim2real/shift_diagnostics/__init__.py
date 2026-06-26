"""Domain-shift diagnostics: quantify source-target divergence.

These diagnostics form the quantitative backbone of the decomposition
framework (PLAN.md §6.3). They operate in the same preprocessed feature
space as the classifier (log + standardized) for consistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cherenkov_sim2real.shift_diagnostics.calibration_drift import compute_calibration_drift
from cherenkov_sim2real.shift_diagnostics.kernel_mmd import compute_kernel_mmd
from cherenkov_sim2real.shift_diagnostics.proxy_a_distance import compute_proxy_a_distance


def compute_shift_diagnostics(
    X_source: np.ndarray[Any, Any],
    X_target: np.ndarray[Any, Any],
    y_source: np.ndarray[Any, Any] | None = None,
    y_target: np.ndarray[Any, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute all shift diagnostics between source and target.

    Parameters
    ----------
    X_source, X_target:
        Feature arrays in preprocessed (log + scaled) space.
    y_source, y_target:
        Labels. Required for calibration drift; optional for MMD and
        A-distance.
    seed:
        Random seed for permutation tests and CV splits.

    Returns
    -------
    Dict with keys: "mmd", "proxy_a_distance", "calibration_drift".
    """
    result: dict[str, Any] = {}
    result["mmd"] = compute_kernel_mmd(X_source, X_target, seed=seed)
    result["proxy_a_distance"] = compute_proxy_a_distance(X_source, X_target, seed=seed)

    if y_source is not None and y_target is not None:
        result["calibration_drift"] = compute_calibration_drift(
            X_source, X_target, y_source, y_target, seed=seed,
        )
    else:
        result["calibration_drift"] = None

    return result


__all__ = ["compute_shift_diagnostics"]
