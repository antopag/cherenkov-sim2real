"""MMD-based feature alignment via learned linear projection.

Learns a linear transform W that minimizes the Maximum Mean Discrepancy
between W @ X_source and X_target, with Tikhonov regularisation
lambda_reg * ||W - I||_F^2 to prevent trivial collapse.

Linear kernel: closed-form solution via covariance matching. This is
mathematically very close to CORAL — both align second-order statistics.
The difference: CORAL whitens source and re-colours with target
covariance (Cs^{-1/2} @ Ct^{1/2}), while linear-MMD minimises
||mean(W@Xs) - mean(Xt)||^2 + trace-penalty, yielding a similar but not
identical solution when lambda_reg is small. In practice, outputs are
nearly identical on well-conditioned data.

RBF kernel: gradient-based optimisation on W using scipy.optimize with
the kernel MMD criterion and median-heuristic bandwidth.

Reference: Gretton et al. 2012, "A Kernel Two-Sample Test", JMLR 13.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


class MMDAlignment:
    """MMD-based linear feature alignment.

    Parameters
    ----------
    kernel:
        Kernel for the MMD criterion. "linear" uses a closed-form
        solution; "rbf" uses gradient-based optimisation.
    gamma:
        RBF bandwidth parameter. If None, uses the median heuristic.
        Ignored for linear kernel.
    lambda_reg:
        Tikhonov regularisation on ||W - I||_F^2. Must be > 0.
    max_iter:
        Maximum iterations for RBF optimisation. Ignored for linear.
    """

    def __init__(
        self,
        kernel: Literal["linear", "rbf"] = "linear",
        gamma: float | None = None,
        lambda_reg: float = 1e-3,
        max_iter: int = 100,
    ) -> None:
        self.kernel = kernel
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self._W: np.ndarray[Any, Any] | None = None
        self._mean_source: np.ndarray[Any, Any] | None = None
        self._mean_target: np.ndarray[Any, Any] | None = None

    def fit(
        self,
        X_source: np.ndarray[Any, Any],
        X_target: np.ndarray[Any, Any],
    ) -> MMDAlignment:
        """Learn the projection W minimising MMD(W@Xs, Xt)."""
        self._mean_source = np.mean(X_source, axis=0)
        self._mean_target = np.mean(X_target, axis=0)

        if self.kernel == "linear":
            self._fit_linear(X_source, X_target)
        else:
            self._fit_rbf(X_source, X_target)
        return self

    def _fit_linear(
        self,
        X_source: np.ndarray[Any, Any],
        X_target: np.ndarray[Any, Any],
    ) -> None:
        """Closed-form linear-MMD alignment via covariance matching."""
        d = X_source.shape[1]
        eye = np.eye(d)
        cov_s = np.cov(X_source, rowvar=False) + self.lambda_reg * eye
        cov_t = np.cov(X_target, rowvar=False) + self.lambda_reg * eye

        # W = Cs^{-1/2} @ Ct^{1/2} (same as CORAL up to regularisation)
        cs_inv_half = _matrix_power(cov_s, -0.5)
        ct_half = _matrix_power(cov_t, 0.5)
        self._W = cs_inv_half @ ct_half

        logger.info(
            "MMD-linear fit: %d source, %d target, %d features",
            len(X_source), len(X_target), d,
        )

    def _fit_rbf(
        self,
        X_source: np.ndarray[Any, Any],
        X_target: np.ndarray[Any, Any],
    ) -> None:
        """Gradient-based RBF-MMD alignment."""
        from scipy.optimize import minimize

        d = X_source.shape[1]

        # Median heuristic for bandwidth
        if self.gamma is None:
            dists = pdist(X_source[:min(500, len(X_source))])
            median_dist = np.median(dists)
            gamma = 1.0 / (2.0 * median_dist**2 + 1e-10)
        else:
            gamma = self.gamma

        def _rbf_mmd_loss(w_flat: np.ndarray[Any, Any]) -> float:
            w_mat = w_flat.reshape(d, d)
            xs_proj = (X_source - self._mean_source) @ w_mat + self._mean_target
            # Subsample for speed
            n = min(500, len(xs_proj), len(X_target))
            xs = xs_proj[:n]
            xt = X_target[:n]
            # Kernel matrices
            k_ss = _rbf_kernel(xs, xs, gamma)
            k_tt = _rbf_kernel(xt, xt, gamma)
            k_st = _rbf_kernel(xs, xt, gamma)
            mmd2 = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
            # Regularisation
            reg = self.lambda_reg * np.sum((w_mat - np.eye(d))**2)
            return float(mmd2 + reg)

        w0 = np.eye(d).ravel()
        result = minimize(
            _rbf_mmd_loss, w0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-8},
        )
        self._W = result.x.reshape(d, d)
        self._loss_history = float(result.fun)

        logger.info(
            "MMD-rbf fit: %d source, %d target, %d features, "
            "gamma=%.4f, final_loss=%.6f, iters=%d",
            len(X_source), len(X_target), d,
            gamma, result.fun, result.nit,
        )

    def transform(self, X_source: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Transform source features."""
        if self._W is None or self._mean_source is None or self._mean_target is None:
            msg = "MMDAlignment.fit() must be called before transform()"
            raise RuntimeError(msg)
        centered = X_source - self._mean_source
        result: np.ndarray[Any, Any] = centered @ self._W + self._mean_target
        return result

    def fit_transform(
        self,
        X_source: np.ndarray[Any, Any],
        X_target: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Fit and transform in one step."""
        self.fit(X_source, X_target)
        return self.transform(X_source)


def _matrix_power(
    M: np.ndarray[Any, Any],
    power: float,
) -> np.ndarray[Any, Any]:
    """Matrix power via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    result: np.ndarray[Any, Any] = (eigenvectors * eigenvalues**power) @ eigenvectors.T
    return result


def _rbf_kernel(
    X: np.ndarray[Any, Any],
    Y: np.ndarray[Any, Any],
    gamma: float,
) -> np.ndarray[Any, Any]:
    """RBF kernel matrix K(X, Y)."""
    sq_dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * X @ Y.T
    result: np.ndarray[Any, Any] = np.exp(-gamma * sq_dists)
    return result
