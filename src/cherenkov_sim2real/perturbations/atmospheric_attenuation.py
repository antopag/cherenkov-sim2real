"""Family 3: Atmospheric attenuation — global SIZE multiplicative factor.

Physical motivation: aerosol loading, cloud layers, and Rayleigh
scattering variation reduce the Cherenkov light reaching the camera.
Fewer photons survive image cleaning, so the image ellipse shrinks:
SIZE, WIDTH, and LENGTH all decrease.

Reference: Fruck et al. 2022, A&A; Gaug 2017, EPJ Web Conf.
TODO-CITE: WIDTH/LENGTH propagation coefficients from H.E.S.S.
atmospheric monitoring literature (Bregeon et al.).

Headline intensity=1.0 corresponds to 20% attenuation (SIZE factor =
0.80, WIDTH factor ~ 0.92, LENGTH factor ~ 0.96).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Headline attenuation: 20% SIZE reduction at intensity=1.0.
_SIZE_ATTENUATION = 0.20

# Phenomenological geometric propagation at headline intensity=1.0.
# WIDTH decreases by ~8%, LENGTH by ~4%.
_BETA_W = 0.08
_BETA_L = 0.04


def apply(
    X: pd.DataFrame,
    intensity: float,
    seed: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Apply atmospheric attenuation.

    Multiplies SIZE by ``1 - 0.20 * intensity``, WIDTH by
    ``1 - 0.08 * intensity``, LENGTH by ``1 - 0.04 * intensity``
    (deterministic; no randomness involved).
    """
    if intensity == 0.0:
        return X.copy()

    out = X.copy()
    out["SIZE"] = out["SIZE"].values * (1.0 - _SIZE_ATTENUATION * intensity)
    out["WIDTH"] = out["WIDTH"].values * (1.0 - _BETA_W * intensity)
    out["LENGTH"] = out["LENGTH"].values * (1.0 - _BETA_L * intensity)
    return out
