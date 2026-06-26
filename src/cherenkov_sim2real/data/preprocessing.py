"""Feature preprocessing: log-transform of heavy-tailed features + scaling.

Standard IACT analysis practice uses log(SIZE) because the Cherenkov
photon yield (SIZE) follows a power-law energy spectrum, producing
extreme right-skew. Log-transform compresses this tail and makes the
feature suitable for linear methods.

The threshold for log-transform is CV > 2 on strictly positive features.
In the project's 10-feature Hillas schema, only SIZE qualifies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Features to log-transform. Determined by CV > 2 inspection on S0 source.
# Only SIZE qualifies (CV ≈ 8, strictly positive, power-law spectrum).
# Other high-CV features (ASYM, ALPHA, M3TRANS) are signed quantities
# centered near zero — log is physically inappropriate for them.
LOG_TRANSFORMED_FEATURES: list[str] = ["SIZE"]


def apply_log_transform(
    X: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Apply log10 transform to specified features.

    Parameters
    ----------
    X:
        Input DataFrame (not mutated).
    features:
        List of feature names to log-transform. Defaults to
        LOG_TRANSFORMED_FEATURES.

    Returns
    -------
    New DataFrame with specified features log10-transformed.

    Raises
    ------
    ValueError
        If any value in the specified features is negative.
    """
    if features is None:
        features = LOG_TRANSFORMED_FEATURES

    out = X.copy()
    for feat in features:
        values = out[feat].values
        if (values < 0).any():
            msg = f"Cannot log-transform feature '{feat}': contains negative values"
            raise ValueError(msg)
        # Epsilon to handle near-zero values (e.g. events with very low SIZE)
        out[feat] = np.log10(values + 1e-3)
    return out


def preprocess_features(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    log_features: list[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], StandardScaler, list[str]]:
    """Full preprocessing pipeline: log-transform + StandardScaler.

    1. Apply log10 to heavy-tailed features (fit-free, same transform
       on both domains).
    2. Fit StandardScaler on source, apply to both.

    Parameters
    ----------
    X_source, X_target:
        Raw DataFrames with the 10-feature Hillas schema.
    log_features:
        Features to log-transform. Defaults to LOG_TRANSFORMED_FEATURES.
    scaler:
        Pre-fitted scaler. If None, a new one is fitted on X_source.

    Returns
    -------
    (X_source_processed, X_target_processed, fitted_scaler, log_features_used)
    """
    if log_features is None:
        log_features = LOG_TRANSFORMED_FEATURES

    # Log-transform
    xs_log = apply_log_transform(X_source, log_features)
    xt_log = apply_log_transform(X_target, log_features)

    # Standardize
    if scaler is None:
        scaler = StandardScaler()
        xs_scaled: np.ndarray[Any, Any] = scaler.fit_transform(xs_log.values)
    else:
        xs_scaled = scaler.transform(xs_log.values)
    xt_scaled: np.ndarray[Any, Any] = scaler.transform(xt_log.values)

    logger.info(
        "Preprocessing: log(%s) + StandardScaler (fit on source, %d features)",
        log_features,
        xs_scaled.shape[1],
    )

    return xs_scaled, xt_scaled, scaler, log_features


def apply_quality_cuts(
    X: pd.DataFrame,
    y: pd.Series,
    size_min: float = 50.0,
    width_min: float = 0.0,
    length_min: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply quality pre-cuts on RAW (pre-log, pre-scale) features.

    Aligned with companion AP paper practice (Pagliaro et al.):
    SIZE > size_min, WIDTH > width_min, LENGTH > length_min.

    Note: the AP paper also applies NUMISLAND < 2 and LEAKAGE < 0.1,
    but these features are not in our 10-feature schema (they exist
    in the full ctapipe DL1 but are not extracted by the loader).
    This asymmetry is documented.

    Parameters
    ----------
    X:
        Raw DataFrame (not mutated).
    y:
        Labels Series (not mutated).
    size_min:
        Minimum SIZE (photo-electrons). Default 50 p.e.
    width_min:
        Minimum WIDTH. Default 0 (excludes degenerate events).
    length_min:
        Minimum LENGTH. Default 0 (excludes degenerate events).

    Returns
    -------
    (X_filtered, y_filtered) with events failing cuts removed.
    """
    mask = (X["SIZE"] > size_min) & (X["WIDTH"] > width_min) & (X["LENGTH"] > length_min)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info(
            "Quality cuts: removed %d / %d events (%.1f%%)",
            n_removed,
            len(X),
            100 * n_removed / len(X),
        )
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
