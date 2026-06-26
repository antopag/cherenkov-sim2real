"""Loader for the UCI MAGIC Gamma Telescope tabular dataset.

The UCI MAGIC dataset is **Monte Carlo** (not real observations).
See PLAN.md S3.1 for the clarification: UCI MAGIC is a source-side MC
in this benchmark, used in scenario S1 (realistic complex).

The dataset contains ~19 000 events with 10 Hillas-style features and
a binary gamma/hadron label. Features: fLength, fWidth, fSize, fConc,
fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist.

Reference: R.K. Bock et al., UCI ML Repository, 2004.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMN_NAMES: list[str] = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
    "class",
]

_EXPECTED_FEATURES: int = 10
_RAW_CSV: str = "magic04.data"


def load_uci_magic(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the UCI MAGIC Gamma Telescope dataset.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``magic04.data`` (e.g. ``data/raw/uci_magic``).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with 10 Hillas-style columns, shape (n_events, 10).
    y : pd.Series
        Binary labels: 1 = gamma, 0 = hadron.

    Raises
    ------
    FileNotFoundError
        If ``magic04.data`` is not found in *data_dir*.
    ValueError
        If the file has unexpected columns.
    """
    csv_path = data_dir / _RAW_CSV
    if not csv_path.exists():
        msg = (
            f"UCI MAGIC data file not found at {csv_path}. "
            f"Run `python scripts/download_magic_uci.py` first."
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path, header=None, names=_COLUMN_NAMES)

    feature_cols = _COLUMN_NAMES[:-1]
    if len(feature_cols) != _EXPECTED_FEATURES:
        msg = (
            f"Expected {_EXPECTED_FEATURES} feature columns, "
            f"got {len(feature_cols)}: {feature_cols}"
        )
        raise ValueError(msg)

    X = df[feature_cols].astype(float)
    y = df["class"].map({"g": 1, "h": 0})

    if y.isna().any():
        unexpected = df["class"][y.isna()].unique().tolist()
        msg = f"Unexpected class labels in UCI MAGIC data: {unexpected}"
        raise ValueError(msg)

    y = y.astype(int)
    logger.info(
        "Loaded UCI MAGIC: %d events, %d features, gamma fraction=%.3f",
        len(X),
        X.shape[1],
        y.mean(),
    )
    return X, y
