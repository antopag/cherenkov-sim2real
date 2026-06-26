"""Synthetic perturbation module for S0 and Img-A scenarios.

Five physically motivated perturbation families applied at DL1 level.
See PLAN.md §3.3 for the full specification and citations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from cherenkov_sim2real.perturbations import (
    atmospheric_attenuation,
    energy_threshold_shift,
    flat_field_uncertainty,
    nsb_injection,
    zenith_size_shift,
)

# Registry: name -> apply function.
# Each function has signature: (X, intensity, seed, rng) -> DataFrame.
PERTURBATION_REGISTRY: dict[str, Any] = {
    "nsb_injection": nsb_injection.apply,
    "flat_field_uncertainty": flat_field_uncertainty.apply,
    "atmospheric_attenuation": atmospheric_attenuation.apply,
    "zenith_size_shift": zenith_size_shift.apply,
    "energy_threshold_shift": energy_threshold_shift.apply,
}

# Headline composite per PLAN.md S3.3: NSB injection at half-moon level +
# 20% atmospheric attenuation.
HEADLINE_COMPOSITE: list[tuple[str, float]] = [
    ("nsb_injection", 1.0),
    ("atmospheric_attenuation", 1.0),
]


def apply_perturbation(
    name: str,
    X: pd.DataFrame,
    intensity: float,
    seed: int,
) -> pd.DataFrame:
    """Apply a single named perturbation."""
    return PERTURBATION_REGISTRY[name](X, intensity=intensity, seed=seed, rng=None)


def apply_composite(
    X: pd.DataFrame,
    families: list[tuple[str, float]],
    seed: int,
) -> pd.DataFrame:
    """Apply a sequence of perturbation families.

    Parameters
    ----------
    X:
        Input DataFrame (not mutated).
    families:
        List of (family_name, intensity) pairs. Applied in order.
    seed:
        Base seed. Each family gets a deterministic child seed.

    Returns
    -------
    Perturbed DataFrame.
    """
    rng = np.random.default_rng(seed)
    result = X.copy()
    for name, intensity in families:
        fn = PERTURBATION_REGISTRY[name]
        child_seed = int(rng.integers(0, 2**31))
        child_rng = np.random.default_rng(child_seed)
        result = fn(result, intensity=intensity, seed=child_seed, rng=child_rng)
    return result


__all__ = [
    "HEADLINE_COMPOSITE",
    "PERTURBATION_REGISTRY",
    "apply_composite",
    "apply_perturbation",
]
