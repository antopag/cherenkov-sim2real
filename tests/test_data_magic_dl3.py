"""Tests for MAGIC DL3 PDR1 loader."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from cherenkov_sim2real.data.magic_dl3 import load_magic_dl3

_DL3_DIR = Path(
    "data/raw/magic_dl3/magic_dl3_pdr1-main/data/CrabNebula/dark/single_offset"
)


@pytest.fixture()
def dl3_single_file() -> Path:
    """Return the first dark-conditions FITS file."""
    fits_files = sorted(_DL3_DIR.glob("*.fits"))
    if not fits_files:
        pytest.skip("MAGIC DL3 data not downloaded")
    return fits_files[0]


@pytest.mark.data
def test_load_magic_dl3_smoke(dl3_single_file: Path) -> None:
    """Loads one file; asserts n_events > 0 and expected columns."""
    x, y = load_magic_dl3(dl3_single_file)
    assert len(x) > 0
    assert set(x.columns) == {"ENERGY", "OFFSET", "THETA"}
    assert len(y) == len(x)


@pytest.mark.data
def test_load_magic_dl3_labels(dl3_single_file: Path) -> None:
    """y has binary classes if both ON and OFF events are present."""
    _, y = load_magic_dl3(dl3_single_file)
    unique = set(y.unique())
    if unique == {0, 1}:
        pass  # expected
    else:
        warnings.warn(
            f"Expected both ON and OFF labels, got {unique}",
            stacklevel=1,
        )


@pytest.mark.data
def test_load_magic_dl3_no_nans(dl3_single_file: Path) -> None:
    """No NaNs in critical columns."""
    x, y = load_magic_dl3(dl3_single_file)
    assert not x.isna().any().any(), "NaN values found in X"
    assert not y.isna().any(), "NaN values found in y"


def test_load_magic_dl3_missing_file(tmp_path: Path) -> None:
    """Clear error on missing file."""
    with pytest.raises(FileNotFoundError, match="MAGIC DL3 file not found"):
        load_magic_dl3(tmp_path / "nonexistent.fits")
