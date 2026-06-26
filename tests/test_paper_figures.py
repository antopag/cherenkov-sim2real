"""Tests for paper figure generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DATA_PATH = Path("paper_electronics/data/extracted_results.json")
_OUTPUT_DIR = Path("paper_electronics/figures/output")


def test_extract_paper_data_completeness() -> None:
    """extracted_results.json contains all required keys."""
    if not _DATA_PATH.exists():
        pytest.skip("Run extract_paper_data.py first")

    with open(_DATA_PATH) as f:
        data = json.load(f)

    assert "s0_summary" in data
    assert "s0_deltas" in data
    assert "s0_shift_diagnostics" in data

    summary = data["s0_summary"]
    carriers = ["lr", "lgbm", "rf", "et", "mlp"]
    methods = ["srconly", "coral", "mean_matching", "mmd_linear", "mmd_rbf"]

    for method in methods:
        for clf in carriers:
            key = f"{method}__{clf}"
            assert key in summary, f"Missing cell: {key}"
            cell = summary[key]
            for metric in ["f1_mean", "f1_std", "auc_mean", "auc_std", "q_mean", "q_std"]:
                assert metric in cell, f"Missing metric {metric} in {key}"

    diags = data["s0_shift_diagnostics"]
    assert "mmd" in diags
    assert "proxy_a_distance" in diags
    assert "calibration_drift" in diags


def test_figures_exist() -> None:
    """All figure PDFs and PNGs exist in output directory."""
    expected = [
        "fig1_schematic", "fig2_carrier_gradient",
        "fig3_mm_vs_coral", "fig4_shift_diagnostics",
        "fig5_threshold_schematic",
    ]
    for name in expected:
        for ext in [".pdf", ".png"]:
            fpath = _OUTPUT_DIR / f"{name}{ext}"
            assert fpath.exists(), f"Missing: {fpath}"
            assert fpath.stat().st_size > 0, f"Empty: {fpath}"


def test_pdf_sizes_reasonable() -> None:
    """PDF files are < 500 KB each."""
    for pdf in _OUTPUT_DIR.glob("*.pdf"):
        size_kb = pdf.stat().st_size / 1024
        assert size_kb < 500, f"{pdf.name} is {size_kb:.0f} KB (>500 KB)"
