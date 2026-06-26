"""Calibration drift: ECE difference between source and target.

Fits a logistic regression on labelled source, computes Expected
Calibration Error (ECE) on source held-out set and on target test set.
The delta ECE quantifies how much classifier calibration drifts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def compute_calibration_drift(
    X_source: np.ndarray[Any, Any],
    X_target: np.ndarray[Any, Any],
    y_source: np.ndarray[Any, Any],
    y_target: np.ndarray[Any, Any],
    seed: int = 42,
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute calibration drift between source and target.

    Returns
    -------
    {"ece_source": float, "ece_target": float, "delta": float}
    """
    # Train on source, test on held-out source and on target
    xs_train, xs_test, ys_train, ys_test = train_test_split(
        X_source, y_source, test_size=0.3, random_state=seed, stratify=y_source,
    )

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(xs_train, ys_train)

    # ECE on source held-out
    prob_source = clf.predict_proba(xs_test)[:, 1]
    ece_source = _expected_calibration_error(ys_test, prob_source, n_bins)

    # ECE on target
    prob_target = clf.predict_proba(X_target)[:, 1]
    ece_target = _expected_calibration_error(y_target, prob_target, n_bins)

    delta = ece_target - ece_source

    logger.info(
        "Calibration drift: ECE_src=%.4f, ECE_tgt=%.4f, delta=%.4f",
        ece_source, ece_target, delta,
    )

    return {
        "ece_source": float(ece_source),
        "ece_target": float(ece_target),
        "delta": float(delta),
    }


def _expected_calibration_error(
    y_true: np.ndarray[Any, Any],
    y_prob: np.ndarray[Any, Any],
    n_bins: int = 15,
) -> float:
    """Compute ECE with equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_accuracy - bin_confidence)
    return float(ece / len(y_true))
