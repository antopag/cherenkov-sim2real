"""Loader for CTA Prod5 public DL1+DL2 Monte Carlo simulations.

Reads per-telescope Hillas parameters from the ctapipe 0.17 HDF5 schema
and returns a tabular DataFrame under a "largest-image LST per event"
projection. See PLAN.md §3.3 for the S2 single-tel projection rationale.

Required acknowledgement (for the paper):
  "This research has made use of the CTA DL1 and DL2 Event lists
  provided by the CTA Observatory and Consortium (version
  prod5-DL2-release1-DL2)"
  Zenodo DOI: 10.5281/zenodo.7298569. License: CC-BY-4.0.

NOTE: ctapipe.io is not used directly because ctapipe 0.17 has a
``pwd`` module dependency that is unavailable on Windows. We read the
HDF5 file directly using PyTables (``tables``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import tables

logger = logging.getLogger(__name__)

# CTA-North Alpha layout: LST telescope IDs (from subarray/layout).
_LST_TEL_IDS: frozenset[int] = frozenset({1, 2, 3, 4})

# Columns extracted from the ctapipe DL1 Hillas table, mapped to the
# closest UCI MAGIC schema equivalent. See
# docs/uci_magic_to_prod5_schema_mapping.md for the full mapping.
_HILLAS_COLS: list[str] = [
    "hillas_length",
    "hillas_width",
    "hillas_intensity",  # SIZE equivalent
    "concentration_cog",
    "concentration_core",
    "hillas_skewness",  # asymmetry proxy
    "hillas_kurtosis",  # no direct UCI equivalent
    "hillas_r",  # DIST equivalent
    "hillas_phi",  # ALPHA proxy (angular)
    "hillas_psi",  # orientation angle
]

# Rename to short names for downstream compatibility.
_RENAME_MAP: dict[str, str] = {
    "hillas_length": "LENGTH",
    "hillas_width": "WIDTH",
    "hillas_intensity": "SIZE",
    "concentration_cog": "CONC",
    "concentration_core": "CONC1",
    "hillas_skewness": "ASYM",
    "hillas_kurtosis": "M3LONG",
    "hillas_r": "DIST",
    "hillas_phi": "ALPHA",
    "hillas_psi": "M3TRANS",
}


def load_prod5_dl1(
    file_path: Path,
    projection: Literal["largest_lst"] = "largest_lst",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load CTA Prod5 DL1 Hillas parameters with largest-LST projection.

    For each event, finds the LST with the largest ``hillas_intensity``
    (SIZE) among those that triggered, and returns the Hillas parameters
    of that single telescope image.

    Parameters
    ----------
    file_path:
        Path to a Prod5 DL1+DL2 HDF5 file (ctapipe 0.17 schema).
    projection:
        Projection mode. Currently only ``"largest_lst"`` is supported.

    Returns
    -------
    X:
        DataFrame with 10 Hillas-derived features, one row per event.
    y:
        Series with binary labels (1 = gamma, 0 = hadron). Read from
        ``true_shower_primary_id`` (0 → gamma → label 1; 101 → proton
        → label 0).

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file is not a valid Prod5 DL1 HDF5 or expected tables
        are missing.
    """
    if not file_path.exists():
        msg = f"Prod5 DL1 file not found: {file_path}"
        raise FileNotFoundError(msg)

    try:
        h5 = tables.open_file(str(file_path), mode="r")
    except tables.HDF5ExtError as exc:
        msg = f"Not a valid HDF5 file: {file_path}"
        raise ValueError(msg) from exc

    try:
        return _load_largest_lst(h5)
    finally:
        h5.close()


def _load_largest_lst(h5: tables.File) -> tuple[pd.DataFrame, pd.Series]:
    """Core loader: largest-intensity LST per event."""
    # 1. Read shower table for labels and event index
    try:
        shower_table = h5.root.simulation.event.subarray.shower
    except tables.NoSuchNodeError as exc:
        msg = "Missing /simulation/event/subarray/shower table"
        raise ValueError(msg) from exc

    shower_df = pd.DataFrame(shower_table.read())
    shower_df = shower_df[["obs_id", "event_id", "true_shower_primary_id"]]

    # 2. Read DL1 Hillas for all LST telescopes
    params_group_path = "/dl1/event/telescope/parameters"
    try:
        h5.get_node(params_group_path)
    except tables.NoSuchNodeError as exc:
        msg = f"Missing {params_group_path} group"
        raise ValueError(msg) from exc

    lst_frames: list[pd.DataFrame] = []
    for tel_id in sorted(_LST_TEL_IDS):
        node_name = f"tel_{tel_id:03d}"
        try:
            tel_table = h5.get_node(params_group_path + "/" + node_name)
        except tables.NoSuchNodeError:
            logger.warning("LST tel_%03d not found in file; skipping", tel_id)
            continue

        needed_cols = ["obs_id", "event_id", "tel_id", *_HILLAS_COLS]
        available = set(tel_table.colnames)
        missing = set(needed_cols) - available
        if missing:
            msg = f"tel_{tel_id:03d} missing columns: {missing}"
            raise ValueError(msg)

        df = pd.DataFrame({col: tel_table.col(col) for col in needed_cols})
        lst_frames.append(df)

    if not lst_frames:
        msg = "No LST telescope data found in file"
        raise ValueError(msg)

    all_lst = pd.concat(lst_frames, ignore_index=True)
    all_lst = all_lst.dropna(subset=["hillas_intensity"]).reset_index(drop=True)

    # 3. For each event, pick the LST with largest hillas_intensity (SIZE)
    idx_max = all_lst.groupby(["obs_id", "event_id"])["hillas_intensity"].idxmax()
    largest = all_lst.loc[idx_max].reset_index(drop=True)

    # 4. Merge with shower table for labels
    merged = largest.merge(shower_df, on=["obs_id", "event_id"], how="inner")

    # 5. Build X with renamed columns
    x_df = merged[_HILLAS_COLS].rename(columns=_RENAME_MAP)

    # 6. Build y: true_shower_primary_id 0 → gamma (1), 101 → proton (0)
    y = merged["true_shower_primary_id"].map({0: 1, 101: 0})
    y.name = "label"

    # Handle unknown primary IDs
    unknown_mask = y.isna()
    if unknown_mask.any():
        unknown_ids = set(merged.loc[unknown_mask, "true_shower_primary_id"])
        logger.warning(
            "Dropping %d events with unknown primary IDs: %s",
            unknown_mask.sum(),
            unknown_ids,
        )
        x_df = x_df.loc[~unknown_mask].reset_index(drop=True)
        y = y.dropna().astype(int).reset_index(drop=True)

    logger.info(
        "Loaded Prod5 DL1: %d events, %d features, gamma fraction=%.3f",
        len(x_df),
        x_df.shape[1],
        y.mean() if len(y) > 0 else 0.0,
    )

    return x_df, y
