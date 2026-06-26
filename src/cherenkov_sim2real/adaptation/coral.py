"""CORAL: CORrelation ALignment (Sun & Saenko 2016).

Aligns second-order feature statistics (covariance) of the source domain
to match the target domain. The transform whitens the source features
using the source covariance and then re-colours them using the target
covariance.

Reference: Sun, B. & Saenko, K. (2016). "Return of Frustratingly Easy
Domain Adaptation". AAAI. arXiv:1511.05547.

See PLAN.md §4.3 for the role of CORAL in the benchmark.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CORAL:
    """CORrelation ALignment for unsupervised domain adaptation.

    Parameters
    ----------
    lambda_reg:
        Regularisation added to the diagonal of covariance matrices
        to avoid ill-conditioning. Must be > 0.
    """

    def __init__(self, lambda_reg: float = 1e-3) -> None:
        if lambda_reg <= 0:
            msg = f"lambda_reg must be > 0, got {lambda_reg}"
            raise ValueError(msg)
        self.lambda_reg = lambda_reg
        self._mean_source: np.ndarray[Any, Any] | None = None
        self._mean_target: np.ndarray[Any, Any] | None = None
        self._transform_matrix: np.ndarray[Any, Any] | None = None

    def fit(
        self,
        X_source: np.ndarray[Any, Any],
        X_target: np.ndarray[Any, Any],
    ) -> CORAL:
        """Compute the CORAL transform from source to target.

        The transform is: ``(X - mean_s) @ Cs^{-1/2} @ Ct^{1/2} + mean_t``

        where Cs, Ct are the regularised covariance matrices.
        """
        self._mean_source = np.mean(X_source, axis=0)
        self._mean_target = np.mean(X_target, axis=0)

        eye = np.eye(X_source.shape[1]) * self.lambda_reg
        cov_s = np.cov(X_source, rowvar=False) + eye
        cov_t = np.cov(X_target, rowvar=False) + eye

        cs_inv_half = _matrix_power(cov_s, -0.5)
        ct_half = _matrix_power(cov_t, 0.5)

        self._transform_matrix = cs_inv_half @ ct_half

        logger.info(
            "CORAL fit: %d source, %d target, %d features, lambda=%.1e",
            len(X_source),
            len(X_target),
            X_source.shape[1],
            self.lambda_reg,
        )
        return self

    def transform(self, X_source: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Transform source features to align with target distribution."""
        if self._transform_matrix is None or self._mean_source is None or self._mean_target is None:
            msg = "CORAL.fit() must be called before transform()"
            raise RuntimeError(msg)

        centered = X_source - self._mean_source
        result: np.ndarray[Any, Any] = centered @ self._transform_matrix + self._mean_target
        return result

    def fit_transform(
        self,
        X_source: np.ndarray[Any, Any],
        X_target: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Fit on (source, target) and transform source in one step."""
        self.fit(X_source, X_target)
        return self.transform(X_source)


def _matrix_power(
    M: np.ndarray[Any, Any],
    power: float,
) -> np.ndarray[Any, Any]:
    """Compute matrix power via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    result: np.ndarray[Any, Any] = (eigenvectors * eigenvalues**power) @ eigenvectors.T
    return result
