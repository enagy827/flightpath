"""Shared chart style for notebooks and the deck (validated reference palette, light mode)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#a3a29c"


def use() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "lines.linewidth": 2,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110,
    })


def label_end(ax, x, y, text, color, dx=4):
    """Direct label at the end of a line (text in text color, small colored marker)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", fontsize=9, color=TEXT_2)
    ax.plot([x], [y], "o", ms=4, color=color)
