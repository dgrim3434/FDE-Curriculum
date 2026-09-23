import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# categorical slots, fixed order
SERIES = {
    "train": "#2a78d6",     # slot 1, blue
    "val": "#eb6834",       # slot 2, orange
    "third": "#1baf7a",     # slot 3, aqua
}

INK = "#0b0b0b"             # primary text
INK_MUTED = "#52514e"       # secondary text, reference lines, annotations
GRID = "#e3e2de"
SURFACE = "#fcfcfb"

# single-hue sequential ramp for attention weights: light -> dark
ATTN_CMAP = LinearSegmentedColormap.from_list(
    "attn_blue", ["#f4f7fb", "#c6dbf2", "#8db8ea", "#4b8fdd", "#2a78d6", "#1b4f8f", "#0f2f57"]
)


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_MUTED,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,       # 2px lines
        "lines.markersize": 5,
        "text.color": INK,
    })


def smooth(values, window=25):
    """Centered moving average. ALWAYS plot the raw series underneath it --
    a smoothed-only curve hides the jaggedness that the pattern table reads."""
    if window <= 1 or len(values) < window:
        return list(values)
    out, run = [], 0.0
    from collections import deque
    q = deque()
    for v in values:
        q.append(v); run += v
        if len(q) > window:
            run -= q.popleft()
        out.append(run / len(q))
    return out