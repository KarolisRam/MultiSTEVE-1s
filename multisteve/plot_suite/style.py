from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#0072B2",
    "cyan": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "brown": "#8C6D31",
    "gray": "#6B7280",
    "black": "#111827",
}

NEURIPS_TEXT_WIDTH_IN = 5.5
NEURIPS_MAIN_FONT_PT = 10.0
NEURIPS_TITLE_FONT_PT = 11.0
SIDE_BY_SIDE_FIGURE_WIDTH_IN = 0.49 * NEURIPS_TEXT_WIDTH_IN


def apply_style(style_config: dict[str, Any] | None = None) -> None:
    """Apply the shared style for all paper plots."""
    cfg = style_config or {}
    base_font_size = float(cfg.get("font_size", NEURIPS_MAIN_FONT_PT))
    title_font_size = float(cfg.get("title_font_size", NEURIPS_TITLE_FONT_PT))
    label_font_size = float(cfg.get("label_font_size", NEURIPS_MAIN_FONT_PT))
    dpi = int(cfg.get("dpi", 220))
    pad_inches = float(cfg.get("pad_inches", 0.0))

    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": pad_inches,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "TeX Gyre Termes",
                "Tinos",
                "serif",
            ],
            "font.size": base_font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": label_font_size,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#2F3437",
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "grid.color": "#D2D6DA",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "legend.fontsize": base_font_size,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.6,
        }
    )


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    dpi: int | None = None,
    bbox_inches: str | None = "tight",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "dpi": dpi if dpi is not None else mpl.rcParams["savefig.dpi"],
        "bbox_inches": bbox_inches,
        "pad_inches": float(mpl.rcParams["savefig.pad_inches"]),
    }
    if bbox_inches is None:
        with mpl.rc_context({"savefig.bbox": None}):
            fig.savefig(output_path, **save_kwargs)
    else:
        fig.savefig(output_path, **save_kwargs)
    plt.close(fig)
