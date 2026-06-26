"""Family 1: NSB injection — additive Poisson noise scaled by sqrt(SIZE).

Physical motivation: night-sky background variation is the dominant
run-to-run systematic in IACT observations. Higher NSB raises the image
cleaning threshold, truncating faint pixels at image edges. This
propagates to all Hillas parameters: SIZE increases (noise adds
photo-electrons), WIDTH and LENGTH increase (noise floor extends the
image ellipse tails), and CONC/CONC1 decrease (light is more diffuse).

Reference: Ahnen et al. 2017, Astropart. Phys. 94, 29 (arXiv:1704.00906).
TODO-CITE: geometric propagation coefficients from NSB systematics
literature (e.g. Aleksic et al. 2016 MAGIC stereo performance, or
Aharonian et al. 2006 H.E.S.S.).

Headline intensity=1.0 corresponds to half-moon NSB equivalent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Phenomenological propagation coefficients at headline intensity=1.0.
# WIDTH increases by ~10%, LENGTH by ~5% relative to the noise fraction.
_ALPHA_W = 0.10
_ALPHA_L = 0.05


def apply(
    X: pd.DataFrame,
    intensity: float,
    seed: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Apply NSB injection perturbation.

    Adds Poisson-distributed noise to SIZE proportional to
    ``intensity * sqrt(SIZE)``. CONC and CONC1 are recomputed to
    reflect the diluted concentration. WIDTH and LENGTH increase
    proportionally to the per-event noise fraction.
    """
    if intensity == 0.0:
        return X.copy()

    if rng is None:
        rng = np.random.default_rng(seed)

    out = X.copy()
    size = out["SIZE"].values.astype(np.float64)

    # Noise scale: intensity * sqrt(|SIZE|) — Poisson-like additive noise
    noise_scale = intensity * np.sqrt(np.abs(size))
    noise = rng.normal(0.0, noise_scale)
    new_size = size + noise

    # Per-event noise fraction (bounded to avoid extreme values)
    noise_frac = np.clip(np.abs(noise) / (np.abs(size) + 1.0), 0.0, 1.0)

    # Concentration dilution: original concentration * (old SIZE / new SIZE)
    ratio = np.where(new_size > 0, size / new_size, 1.0)
    out["SIZE"] = new_size
    out["CONC"] = out["CONC"].values * ratio
    out["CONC1"] = out["CONC1"].values * ratio

    # Geometric propagation: NSB extends the image ellipse
    out["WIDTH"] = out["WIDTH"].values * (1.0 + _ALPHA_W * noise_frac)
    out["LENGTH"] = out["LENGTH"].values * (1.0 + _ALPHA_L * noise_frac)

    return out
