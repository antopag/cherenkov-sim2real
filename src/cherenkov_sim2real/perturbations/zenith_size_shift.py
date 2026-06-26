"""Family 4: Zenith-equivalent SIZE shift.

Physical motivation: higher zenith = longer slant depth = more atmospheric
absorption; SIZE decreases at fixed true energy, with the strongest
effect on low-SIZE events. Additionally, oblique viewing elongates the
shower image in the camera plane: LENGTH increases (dominant geometric
effect), WIDTH increases slightly (secondary).

Reference: Sobczynska et al. 2019, Astropart. Phys. (arXiv:1902.03875).
TODO-CITE: LENGTH/WIDTH zenith dependence from CTA performance papers.

Headline intensity=1.0 mimics a 20 deg -> 40 deg zenith transition
(~15% global SIZE reduction, ~15% LENGTH increase, ~5% WIDTH increase).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Phenomenological geometric propagation at headline intensity=1.0.
# LENGTH increases by ~15% (oblique viewing elongates the image).
# WIDTH increases by ~5% (secondary effect).
_GAMMA_L = 0.15
_GAMMA_W = 0.05


def apply(
    X: pd.DataFrame,
    intensity: float,
    seed: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Apply zenith-equivalent SIZE shift.

    Reduces SIZE with a size-dependent factor (low-SIZE events are
    attenuated more). LENGTH increases (oblique viewing). WIDTH
    increases slightly.
    """
    if intensity == 0.0:
        return X.copy()

    out = X.copy()
    size = out["SIZE"].values.astype(np.float64)
    size_median = np.median(np.abs(size))
    if size_median == 0:
        size_median = 1.0

    # Size-dependent attenuation: stronger for low-SIZE events
    weight = 1.0 + size_median / (np.abs(size) + size_median)
    factor = 1.0 - 0.15 * intensity * weight
    factor = np.clip(factor, 0.01, 1.0)
    out["SIZE"] = size * factor

    # Geometric propagation: oblique viewing elongates the image
    out["LENGTH"] = out["LENGTH"].values * (1.0 + _GAMMA_L * intensity)
    out["WIDTH"] = out["WIDTH"].values * (1.0 + _GAMMA_W * intensity)

    return out
