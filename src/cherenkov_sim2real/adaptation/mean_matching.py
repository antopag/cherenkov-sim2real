"""Mean Matching: trivial domain adaptation via location shift.

Subtracts the source mean and adds the target mean. Catches location
shift exactly; does nothing about covariance or higher-order differences.

This is a baseline reference for location-shift scenarios. On S0 with
the headline composite (dominated by mean shifts), mean matching is
expected to perform well. On scenarios with covariance-dominated shift,
it is expected to fail.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MeanMatching:
    """Mean matching for unsupervised domain adaptation.

    Transform: ``X_source - mean_source + mean_target``.
    No hyperparameters.
    """

    def __init__(self) -> None:
        self._mean_source: np.ndarray[Any, Any] | None = None
        self._mean_target: np.ndarray[Any, Any] | None = None

    def fit(
        self,
        X_source: np.ndarray[Any, Any],
        X_target: np.ndarray[Any, Any],
    ) -> MeanMatching:
        """Compute source and target means."""
        self._mean_source = np.mean(X_source, axis=0)
        self._mean_target = np.mean(X_target, axis=0)
        return self

    def transform(self, X_source: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Shift source features to match target mean."""
        if self._mean_source is None or self._mean_target is None:
            msg = "MeanMatching.fit() must be called before transform()"
            raise RuntimeError(msg)
        result: np.ndarray[Any, Any] = X_source - self._mean_source + self._mean_target
        return result

    def fit_transform(
        self,
        X_source: np.ndarray[Any, Any],
        X_target: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Fit and transform in one step."""
        self.fit(X_source, X_target)
        return self.transform(X_source)
