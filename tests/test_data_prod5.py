"""Tests for CTA Prod5 DL1 loader."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from cherenkov_sim2real.data.cta_prod5 import load_prod5_dl1

_DEV_SAMPLE = Path("data/raw/cta_prod5/dl1/gamma-diffuse_with_images_40.dl2.h5")

_EXPECTED_FEATURES = {
    "LENGTH",
    "WIDTH",
    "SIZE",
    "CONC",
    "CONC1",
    "ASYM",
    "M3LONG",
    "DIST",
    "ALPHA",
    "M3TRANS",
}


@pytest.fixture()
def prod5_data() -> tuple:  # type: ignore[type-arg]
    """Load Prod5 dev sample once per test session."""
    return load_prod5_dl1(_DEV_SAMPLE)


@pytest.mark.data
def test_load_prod5_smoke(prod5_data: tuple) -> None:  # type: ignore[type-arg]
    """Loads the dev sample; asserts shape and expected schema."""
    x, y = prod5_data
    assert len(x) > 0, "Expected non-empty DataFrame"
    assert set(x.columns) == _EXPECTED_FEATURES, (
        f"Column mismatch: got {set(x.columns)}"
    )
    assert len(y) == len(x), "X and y length mismatch"


@pytest.mark.data
def test_load_prod5_labels(prod5_data: tuple) -> None:  # type: ignore[type-arg]
    """Asserts correct label encoding."""
    _, y = prod5_data
    unique = set(y.unique())
    # Dev sample is gamma-diffuse only → single class (1 = gamma).
    # If proton file is added later, expect {0, 1}.
    if unique == {1}:
        warnings.warn(
            "Dev sample contains only gamma events (label=1). "
            "Proton file needed for binary classification.",
            stacklevel=1,
        )
    else:
        assert unique == {0, 1}, f"Expected labels {{0, 1}}, got {unique}"


@pytest.mark.data
def test_load_prod5_no_nans(prod5_data: tuple) -> None:  # type: ignore[type-arg]
    """No NaN values in features."""
    x, y = prod5_data
    assert not x.isna().any().any(), "NaN values found in X"
    assert not y.isna().any(), "NaN values found in y"


@pytest.mark.data
def test_load_prod5_largest_lst_invariant() -> None:
    """For each event, SIZE matches the maximum SIZE among LSTs."""
    import pandas as pd
    import tables

    if not _DEV_SAMPLE.exists():
        pytest.skip("Dev sample not downloaded")

    x, _ = load_prod5_dl1(_DEV_SAMPLE)

    # Re-read raw data to verify
    h5 = tables.open_file(str(_DEV_SAMPLE), mode="r")
    try:
        frames = []
        for tel_id in [1, 2, 3, 4]:
            node_path = f"/dl1/event/telescope/parameters/tel_{tel_id:03d}"
            try:
                tbl = h5.get_node(node_path)
            except tables.NoSuchNodeError:
                continue
            df = pd.DataFrame(
                {
                    "obs_id": tbl.col("obs_id"),
                    "event_id": tbl.col("event_id"),
                    "hillas_intensity": tbl.col("hillas_intensity"),
                }
            )
            frames.append(df)

        all_lst = pd.concat(frames, ignore_index=True)
        all_lst = all_lst.dropna(subset=["hillas_intensity"])
        max_per_event = all_lst.groupby(["obs_id", "event_id"])[
            "hillas_intensity"
        ].max()
    finally:
        h5.close()

    # The SIZE column in X should match the max intensity per event
    # (order may differ, so compare sorted arrays).
    import numpy as np

    x_sizes = np.sort(x["SIZE"].values)
    raw_maxes = np.sort(max_per_event.values)
    assert len(x_sizes) == len(raw_maxes), (
        f"Event count mismatch: loader={len(x_sizes)}, raw={len(raw_maxes)}"
    )
    np.testing.assert_allclose(x_sizes, raw_maxes, rtol=1e-6)


def test_load_prod5_missing_file(tmp_path: Path) -> None:
    """Clear error on missing file."""
    with pytest.raises(FileNotFoundError, match=r"Prod5 DL1 file not found"):
        load_prod5_dl1(tmp_path / "nonexistent.h5")
