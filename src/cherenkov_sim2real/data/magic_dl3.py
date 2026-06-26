"""Loader for MAGIC DL3 PDR1 (Crab Nebula, dark conditions).

Reads GADF-compliant DL3 FITS event lists from the MAGIC PDR1 release
(Zenodo DOI 10.5281/zenodo.11108474, Nigro et al. 2024).

The DL3 schema is post-cut and minimal: EVENT_ID, TIME, RA, DEC, ENERGY.
No Hillas parameters, no gammaness score, no stereo quantities. This is
structurally different from the Prod5 DL1 Hillas schema — see PLAN.md
§3.6 for the S2 schema-projection rationale.

ON/OFF labelling heuristic: events within theta_on (default 0.14 deg)
of the Crab position are labelled as ON (gamma-candidate, label=1).
Events at similar angular distance from mirrored OFF positions are
labelled as OFF (background, label=0). Events in neither region are
excluded. This follows standard wobble-mode analysis practice in IACT.

CAVEATS (per PLAN.md §3.2):
- DL3 events are post-cut: upstream quality cuts are already applied.
- No per-event ground-truth gamma/hadron labels exist. ON/OFF labels
  are spatial proxies with residual contamination in both regions.
- Only Crab Nebula dark-conditions subset is used.

Required acknowledgement: cite Zenodo DOI 10.5281/zenodo.11108474
and Nigro et al. 2024 (arXiv:2409.18823).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord  # type: ignore[import-untyped]
from astropy.io import fits  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Crab Nebula J2000 position
_CRAB_RA = 83.63333
_CRAB_DEC = 22.01444

# Default ON region radius (degrees). Standard MAGIC theta^2 cut.
_THETA_ON = 0.14


def load_magic_dl3(
    file_path: Path,
    theta_on: float = _THETA_ON,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a single MAGIC DL3 PDR1 FITS file.

    Parameters
    ----------
    file_path:
        Path to a GADF DL3 FITS file.
    theta_on:
        ON region radius in degrees (default 0.14 deg).

    Returns
    -------
    X:
        DataFrame with columns: ENERGY (TeV), OFFSET (deg from camera
        center), THETA (deg from Crab).
    y:
        Series with binary labels (1 = ON/gamma-candidate,
        0 = OFF/background). Events outside both ON and OFF regions
        are excluded.
    """
    if not file_path.exists():
        msg = f"MAGIC DL3 file not found: {file_path}"
        raise FileNotFoundError(msg)

    with fits.open(file_path) as hdul:
        events = hdul["EVENTS"].data
        header = hdul["EVENTS"].header

    ra = events["RA"]
    dec = events["DEC"]
    energy = events["ENERGY"]

    # Pointing position
    ra_pnt = header["RA_PNT"]
    dec_pnt = header["DEC_PNT"]

    # Angular distances
    event_coords = SkyCoord(ra=ra, dec=dec, unit="deg")
    crab_coord = SkyCoord(ra=_CRAB_RA, dec=_CRAB_DEC, unit="deg")
    pnt_coord = SkyCoord(ra=ra_pnt, dec=dec_pnt, unit="deg")

    theta = event_coords.separation(crab_coord).deg
    offset = event_coords.separation(pnt_coord).deg

    # OFF region: mirror the Crab position through the pointing center
    off_ra = 2 * ra_pnt - _CRAB_RA
    off_dec = 2 * dec_pnt - _CRAB_DEC
    off_coord = SkyCoord(ra=off_ra, dec=off_dec, unit="deg")
    theta_off = event_coords.separation(off_coord).deg

    # ON/OFF labelling
    on_mask = theta < theta_on
    off_mask = theta_off < theta_on

    # Build feature DataFrame for ON and OFF events
    all_features = pd.DataFrame({
        "ENERGY": energy.astype(np.float64),
        "OFFSET": offset.astype(np.float64),
        "THETA": theta.astype(np.float64),
    })

    # ON events
    on_df = all_features.loc[on_mask].copy()
    on_labels = pd.Series(np.ones(on_mask.sum(), dtype=int), name="label")

    # OFF events
    off_df = all_features.loc[off_mask].copy()
    off_labels = pd.Series(np.zeros(off_mask.sum(), dtype=int), name="label")

    # Concatenate
    x_out = pd.concat([on_df, off_df], ignore_index=True)
    y_out = pd.concat([on_labels, off_labels], ignore_index=True)

    logger.info(
        "Loaded MAGIC DL3: %d total events, %d ON, %d OFF, "
        "energy range [%.3f, %.1f] TeV",
        len(events), on_mask.sum(), off_mask.sum(),
        energy.min(), energy.max(),
    )

    return x_out, y_out


def load_magic_dl3_directory(
    data_dir: Path,
    theta_on: float = _THETA_ON,
    max_files: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load all MAGIC DL3 FITS files from a directory.

    Parameters
    ----------
    data_dir:
        Directory containing *.fits files (e.g. the dark/single_offset
        subdirectory of the PDR1 release).
    theta_on:
        ON region radius in degrees.
    max_files:
        If set, load only the first N files (for fast iteration).

    Returns
    -------
    (X, y) concatenated across all files.
    """
    fits_files = sorted(data_dir.glob("*.fits"))
    if not fits_files:
        msg = f"No FITS files found in {data_dir}"
        raise FileNotFoundError(msg)

    if max_files is not None:
        fits_files = fits_files[:max_files]

    x_frames: list[pd.DataFrame] = []
    y_frames: list[pd.Series] = []

    for fpath in fits_files:
        x, y = load_magic_dl3(fpath, theta_on=theta_on)
        x_frames.append(x)
        y_frames.append(y)

    x_all = pd.concat(x_frames, ignore_index=True)
    y_all = pd.concat(y_frames, ignore_index=True)

    logger.info(
        "Loaded %d MAGIC DL3 files: %d events total (%d ON, %d OFF)",
        len(fits_files), len(x_all),
        (y_all == 1).sum(), (y_all == 0).sum(),
    )

    return x_all, y_all
