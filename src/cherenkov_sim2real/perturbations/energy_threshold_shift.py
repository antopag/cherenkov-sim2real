"""Family 5: Effective energy threshold shift — selective event rejection.

Physical motivation: trigger rate instability from NSB variation,
hardware ageing, or PMT gain drift shifts the effective energy threshold.

Reference: Gaug et al. 2019 (atmospheric monitoring via trigger rates);
           LST-1 camera calibration (hal-05381165).

Headline intensity=1.0 raises the effective threshold by 30%
(cuts ~10% of events at low SIZE).

NOTE: Unlike families 1-4, this perturbation CHANGES the number of
events (selective rejection). Downstream code must handle this.
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
    """Apply effective energy threshold shift.

    Rejects events with SIZE below a threshold that increases with
    intensity. The threshold is computed as::

        threshold = SIZE_quantile(q) * (1 + 0.30 * intensity)

    where q = 0.10 (the 10th percentile of SIZE). At intensity=1.0,
    events below 130% of the 10th percentile SIZE are rejected.
    """
    if intensity == 0.0:
        return X.copy()

    size = X["SIZE"].values.astype(np.float64)
    base_threshold = np.quantile(np.abs(size), 0.10)
    threshold = base_threshold * (1.0 + 0.30 * intensity)

    mask = np.abs(size) >= threshold
    return X.loc[mask].reset_index(drop=True)
