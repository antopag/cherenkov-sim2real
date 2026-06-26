"""Tests for feature preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cherenkov_sim2real.data.preprocessing import (
    LOG_TRANSFORMED_FEATURES,
    apply_log_transform,
    apply_quality_cuts,
    preprocess_features,
)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Synthetic DataFrame mimicking Prod5 Hillas schema."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "LENGTH": rng.uniform(0.1, 1.0, n),
        "WIDTH": rng.uniform(0.05, 0.5, n),
        "SIZE": rng.uniform(50, 10000, n),
        "CONC": rng.uniform(0.1, 0.8, n),
        "CONC1": rng.uniform(0.05, 0.5, n),
        "ASYM": rng.normal(0, 0.5, n),
        "M3LONG": rng.normal(2.4, 0.9, n),
        "DIST": rng.uniform(0.1, 2.0, n),
        "ALPHA": rng.uniform(-180, 180, n),
        "M3TRANS": rng.uniform(-90, 90, n),
    })


def test_log_transform_applied(synthetic_df: pd.DataFrame) -> None:
    """Log-transformed SIZE has expected statistics (log10 scale)."""
    result = apply_log_transform(synthetic_df, ["SIZE"])
    # log10(50) ≈ 1.7, log10(10000) = 4.0, so mean should be ~2.5-3.5
    assert 1.5 < result["SIZE"].mean() < 4.0
    # std should be much smaller than raw (raw std ~3000, log std ~0.5-1)
    assert result["SIZE"].std() < 2.0
    # Non-logged features unchanged
    pd.testing.assert_series_equal(result["LENGTH"], synthetic_df["LENGTH"])


def test_log_transform_handles_negative_values() -> None:
    """Clear error on negative input."""
    df = pd.DataFrame({"SIZE": [-1.0, 100.0, 200.0]})
    with pytest.raises(ValueError, match="contains negative values"):
        apply_log_transform(df, ["SIZE"])


def test_preprocessing_full_pipeline(synthetic_df: pd.DataFrame) -> None:
    """Source and target after preprocessing have expected statistics."""
    # Create a shifted target
    target = synthetic_df.copy()
    target["SIZE"] = target["SIZE"] * 0.8  # 20% attenuation

    xs, xt, _scaler, log_feats = preprocess_features(synthetic_df, target)

    assert log_feats == LOG_TRANSFORMED_FEATURES
    assert xs.shape == (500, 10)
    assert xt.shape == (500, 10)

    # Source should be standardized (mean≈0, std≈1)
    np.testing.assert_allclose(xs.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(xs.std(axis=0), 1.0, atol=1e-10)

    # Target SIZE (column index 2) should have a visible negative shift
    # because log10(0.8*SIZE) = log10(SIZE) + log10(0.8) ≈ log10(SIZE) - 0.097
    # In scaled units: -0.097 / log_std. With log_std ≈ 0.7, shift ≈ -0.14
    size_col = list(synthetic_df.columns).index("SIZE")
    assert xt[:, size_col].mean() < -0.05, (
        f"Expected negative SIZE shift in target, got mean={xt[:, size_col].mean():.4f}"
    )


# --- Quality cuts ---


def test_quality_cuts_reduce_event_count() -> None:
    """Quality cuts remove some events."""
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "SIZE": rng.uniform(10, 1000, n),  # some below 50
        "WIDTH": rng.uniform(-0.01, 0.5, n),  # some <= 0
        "LENGTH": rng.uniform(-0.01, 1.0, n),  # some <= 0
        **{c: rng.uniform(0, 1, n) for c in ["CONC", "CONC1", "ASYM", "M3LONG", "DIST", "ALPHA", "M3TRANS"]},
    })
    y = pd.Series(rng.integers(0, 2, n))
    X_cut, y_cut = apply_quality_cuts(X, y)
    assert len(X_cut) < len(X)
    assert len(y_cut) == len(X_cut)


def test_quality_cuts_size_threshold() -> None:
    """Events with SIZE < 50 are excluded."""
    X = pd.DataFrame({
        "SIZE": [10.0, 49.9, 50.1, 100.0],
        "WIDTH": [0.1, 0.1, 0.1, 0.1],
        "LENGTH": [0.2, 0.2, 0.2, 0.2],
        **{c: [1.0] * 4 for c in ["CONC", "CONC1", "ASYM", "M3LONG", "DIST", "ALPHA", "M3TRANS"]},
    })
    y = pd.Series([0, 1, 0, 1])
    X_cut, _ = apply_quality_cuts(X, y)
    assert len(X_cut) == 2
    assert (X_cut["SIZE"] > 50).all()


def test_quality_cuts_pure() -> None:
    """Input X, y are not mutated."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame({
        "SIZE": rng.uniform(10, 1000, n),
        "WIDTH": rng.uniform(0.01, 0.5, n),
        "LENGTH": rng.uniform(0.01, 1.0, n),
        **{c: rng.uniform(0, 1, n) for c in ["CONC", "CONC1", "ASYM", "M3LONG", "DIST", "ALPHA", "M3TRANS"]},
    })
    y = pd.Series(rng.integers(0, 2, n))
    X_copy = X.copy()
    y_copy = y.copy()
    apply_quality_cuts(X, y)
    pd.testing.assert_frame_equal(X, X_copy)
    pd.testing.assert_series_equal(y, y_copy)
