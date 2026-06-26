"""Publication styling for Electronics paper figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Tableau-colorblind palette
COLORS: dict[str, str] = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "purple": "#56B4E9",
    "neutral": "#949494",
}

# Figure dimensions (cm -> inches)
_CM_TO_INCH = 1 / 2.54
SINGLE_COL_WIDTH = 8.5 * _CM_TO_INCH  # ~3.35 inch
DOUBLE_COL_WIDTH = 17.8 * _CM_TO_INCH  # ~7.0 inch
STANDARD_HEIGHT = 6.5 * _CM_TO_INCH  # ~2.55 inch


def apply_paper_style() -> None:
    """Set matplotlib rcParams for publication-quality figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, name: str, output_dir: str = "output") -> None:
    """Save figure as both PDF and PNG."""
    from pathlib import Path

    out = Path(__file__).parent / output_dir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
