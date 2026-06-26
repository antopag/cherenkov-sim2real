"""Family 2: Flat-fielding coefficient uncertainty.

Physical motivation: residual flat-fielding errors after gain calibration
propagate into SIZE and image shape parameters.

Reference: Gaug et al. 2019, ApJS 243, 11 (arXiv:1907.04375);
           Daniel 2015 (arXiv:1508.06625).

Headline intensity=1.0 corresponds to sigma = 5% per-event multiplicative
Gaussian factor on SIZE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply(
    X: pd.DataFrame,
    intensity: float,
    seed: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Apply flat-fielding coefficient uncertainty.

    Multiplies SIZE by a per-event Gaussian factor with
    ``sigma = 0.05 * intensity``.
    """
    if intensity == 0.0:
        return X.copy()

    if rng is None:
        rng = np.random.default_rng(seed)

    out = X.copy()
    sigma = 0.05 * intensity
    factors = rng.normal(1.0, sigma, size=len(out))
    out["SIZE"] = out["SIZE"].values * factors
    return out
