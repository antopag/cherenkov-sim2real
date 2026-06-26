"""Tests for dataset loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from cherenkov_sim2real.data.magic_uci import load_uci_magic

_DATA_DIR = Path("data/raw/uci_magic")


@pytest.fixture()
def uci_magic() -> tuple:  # type: ignore[type-arg]
    """Load UCI MAGIC dataset once per test session."""
    return load_uci_magic(_DATA_DIR)


@pytest.mark.data
def test_load_uci_magic_shape(uci_magic: tuple) -> None:  # type: ignore[type-arg]
    X, y = uci_magic
    assert len(X) >= 19000, f"Expected >= 19000 rows, got {len(X)}"
    assert X.shape[1] == 10, f"Expected 10 features, got {X.shape[1]}"
    assert len(y) == len(X), "X and y length mismatch"


@pytest.mark.data
def test_load_uci_magic_labels(uci_magic: tuple) -> None:  # type: ignore[type-arg]
    _, y = uci_magic
    unique = set(y.unique())
    assert unique == {0, 1}, f"Expected labels {{0, 1}}, got {unique}"
    gamma_frac = y.mean()
    assert 0.55 <= gamma_frac <= 0.75, (
        f"Gamma fraction {gamma_frac:.3f} outside expected range [0.55, 0.75]"
    )


@pytest.mark.data
def test_load_uci_magic_no_nans(uci_magic: tuple) -> None:  # type: ignore[type-arg]
    X, y = uci_magic
    assert not X.isna().any().any(), "NaN values found in X"
    assert not y.isna().any(), "NaN values found in y"


def test_load_uci_magic_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"magic04\.data"):
        load_uci_magic(tmp_path)
