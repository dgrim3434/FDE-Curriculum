"""AI Generated"""
"""Shared figure styling. One place to change colors, fonts and save paths."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                 # headless: we save files, never open windows
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_MUTED = "#52514e"
SERIES    = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order, never cycled

# MAGNITUDE (attention weights, 0..1): ONE hue, light -> dark. Never a rainbow.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])
# POLARITY (differences, signed): two hues + a neutral midpoint at zero.
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f1f1e"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_MUTED, "axes.titlecolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#d8d7d3", "grid.color": "#e8e7e3", "font.size": 9,
})

def save(fig, name):
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
    return path