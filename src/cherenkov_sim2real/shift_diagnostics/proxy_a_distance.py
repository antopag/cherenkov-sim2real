"""Proxy A-distance via domain classifier.

Trains a logistic regression to distinguish source from target events.
The proxy A-distance is 2 * (1 - 2 * cv_error_rate), computed with
5-fold cross-validation. Higher values indicate more distinguishable
domains.

Reference: Ben-David et al. 2010, "A theory of learning from different
domains", Machine Learning 79.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def compute_proxy_a_distance(
    X_source: np.ndarray[Any, Any],
    X_target: np.ndarray[Any, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Compute proxy A-distance between source and target.

    Returns
    -------
    {"value": float, "cv_error": float}
    """
    # Create domain labels: 0 = source, 1 = target
    n_s = len(X_source)
    n_t = len(X_target)
    x_combined = np.vstack([X_source, X_target])
    y_domain = np.concatenate([np.zeros(n_s), np.ones(n_t)])

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    scores = cross_val_score(clf, x_combined, y_domain, cv=5, scoring="accuracy")
    cv_accuracy = float(scores.mean())
    cv_error = 1.0 - cv_accuracy
    a_distance = 2.0 * (1.0 - 2.0 * cv_error)

    logger.info(
        "Proxy A-distance: value=%.4f (cv_accuracy=%.4f, cv_error=%.4f)",
        a_distance, cv_accuracy, cv_error,
    )

    return {"value": float(a_distance), "cv_error": float(cv_error)}
