"""S0 scenario: controlled MC/MC shift via synthetic perturbations.

Source = CTA Prod5 baseline (gamma + proton, unperturbed).
Target = same Prod5 sample with headline composite perturbation applied.

Same labels on both sides — S0 measures feature-distribution shift only,
not label shift. See PLAN.md §3.6.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cherenkov_sim2real.data.cta_prod5 import load_prod5_dl1
from cherenkov_sim2real.data.preprocessing import apply_quality_cuts
from cherenkov_sim2real.perturbations import HEADLINE_COMPOSITE, apply_composite

logger = logging.getLogger(__name__)

_GAMMA_FILE = "gamma-diffuse_with_images_40.dl2.h5"
_PROTON_FILE = "proton_with_images_00.dl2.h5"


def build_s0(
    data_dir: Path,
    perturbation_intensity: float = 1.0,
    seed: int = 42,
    subsample: int | None = None,
) -> dict[str, pd.DataFrame | pd.Series | dict[str, Any]]:
    """Build the S0 scenario: baseline vs perturbed Prod5.

    Parameters
    ----------
    data_dir:
        Directory containing the Prod5 DL1 HDF5 files.
    perturbation_intensity:
        Intensity for headline composite perturbation. Default 1.0.
    seed:
        Random seed for perturbations and subsampling.
    subsample:
        If set, randomly subsample this many events from the combined
        dataset before perturbation. Useful for fast iteration.

    Returns
    -------
    Dict with keys: X_source, y_source, X_target, y_target, metadata.
    """
    # Load gamma and proton
    x_gamma, y_gamma = load_prod5_dl1(data_dir / _GAMMA_FILE)
    x_proton, y_proton = load_prod5_dl1(data_dir / _PROTON_FILE)

    n_gamma = len(x_gamma)
    n_proton = len(x_proton)
    logger.info("Loaded gamma: %d events, proton: %d events", n_gamma, n_proton)

    # Quality pre-cuts (aligned with AP paper: SIZE > 50, WIDTH > 0, LENGTH > 0)
    x_gamma, y_gamma = apply_quality_cuts(x_gamma, y_gamma)
    x_proton, y_proton = apply_quality_cuts(x_proton, y_proton)

    # Concatenate
    x_all = pd.concat([x_gamma, x_proton], ignore_index=True)
    y_all = pd.concat([y_gamma, y_proton], ignore_index=True)

    # Optional subsample
    if subsample is not None and subsample < len(x_all):
        idx = x_all.sample(n=subsample, random_state=seed).index
        x_all = x_all.loc[idx].reset_index(drop=True)
        y_all = y_all.loc[idx].reset_index(drop=True)

    # Source = baseline (unperturbed)
    x_source = x_all.copy()
    y_source = y_all.copy()

    # Target = perturbed copy
    families = [(name, intensity * perturbation_intensity) for name, intensity in HEADLINE_COMPOSITE]
    x_target = apply_composite(x_all, families, seed=seed)
    y_target = y_all.copy()

    gamma_frac = float(y_source.mean())
    logger.info(
        "S0 built: %d events, gamma fraction=%.3f, perturbation intensity=%.2f",
        len(x_source),
        gamma_frac,
        perturbation_intensity,
    )

    metadata: dict[str, Any] = {
        "scenario": "S0",
        "n_events": len(x_source),
        "n_gamma": int((y_source == 1).sum()),
        "n_proton": int((y_source == 0).sum()),
        "gamma_fraction": gamma_frac,
        "perturbation_intensity": perturbation_intensity,
        "headline_composite": HEADLINE_COMPOSITE,
        "seed": seed,
        "subsample": subsample,
        "schema": list(x_source.columns),
    }

    return {
        "X_source": x_source,
        "y_source": y_source,
        "X_target": x_target,
        "y_target": y_target,
        "metadata": metadata,
    }
