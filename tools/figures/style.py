"""Shared style of the figures: one palette, a light theme, and the few drawing helpers every script uses.

Sizes: every figure is ``WIDTH_IN`` inches wide and is shown about 880 px wide in the README, so 1 pt of text is
about 1.27 px on the page. Body text is 8.5 pt or more; console text (monospace) is 7 pt or more.
MR images are grey on black, as on a console. Row = y with the first row at the top, column = x.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "assets")
WIDTH_IN = 9.6
DPI = 160
MAX_PNG_KB = 600

SURFACE, PANEL = "#ffffff", "#f3f2ee"
INK, INK2, MUTED = "#0b0b0b", "#4a4946", "#807e78"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#d9591c", "#1b9e77"
TRACK = "#cde2fb"
GOOD, CRITICAL = "#0b7d0b", "#c62f2f"
OVERLAY_FOV, OVERLAY_TRUTH, OVERLAY_REGION = "#ff7a45", "#35c2ff", "#ffd23f"
MONO = "DejaVu Sans Mono"
CLASS_COLOURS = ["#000000", "#e6c9a8", "#f2d16b", "#d9d9d9", "#b8a070", "#3b6fb6", "#f4f1e8", "#9a9a9a", "#6f6f6f",
                 "#d9591c", "#1b9e77", "#8c6bb1", "#e7298a", "#66a61e"]


def apply() -> None:
    """Set the matplotlib defaults of the figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": False, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "svg.fonttype": "path", "svg.hashsalt": "mrgym", "legend.frameon": False, "figure.dpi": DPI,
    })


def sequential() -> LinearSegmentedColormap:
    """Light to dark; for T1, T2 and other positive maps."""
    cm = LinearSegmentedColormap.from_list("mrgym_seq", ["#f4f0d8", "#c9dc9a", "#7fc4a0", "#3e9fb3", "#2a6fb0", "#27408b", "#1d1b4f"])
    cm.set_bad(PANEL)
    return cm


def diverging() -> LinearSegmentedColormap:
    """Blue below, red above, neutral at the reference value; for B1+ around 1 and off-resonance around 0."""
    cm = LinearSegmentedColormap.from_list("mrgym_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a6a5", "#e34948", "#8f1f1f"])
    cm.set_bad(PANEL)
    return cm


def classes(n: int) -> ListedColormap:
    """One flat colour per tissue class; index 0 is the background."""
    return ListedColormap(CLASS_COLOURS[:n + 1])


def extent(shape, fov_yx_mm) -> tuple[float, float, float, float]:
    """``imshow`` extent in mm for a scanner image: pixel i of an n-point axis is centred at (i - n // 2) * FOV / n."""
    ny, nx = shape[-2:]
    dy, dx = fov_yx_mm[0] / ny, fov_yx_mm[1] / nx
    x0, y0 = (0 - nx // 2 - 0.5) * dx, (0 - ny // 2 - 0.5) * dy
    return (x0, x0 + fov_yx_mm[1], y0 + fov_yx_mm[0], y0)


def bare(ax, face: str = "black") -> None:
    """An image axis: no ticks, no frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(face)


def show(ax, img, fov_yx_mm, vmax: float | None = None, percentile: float = 99.7, view_mm=None, interpolation="nearest"):
    """Draw a magnitude image in grey with its true proportions. ``view_mm`` = (y_lo, y_hi, x_lo, x_hi) crops."""
    img = np.asarray(img, float)
    top = float(np.percentile(img, percentile)) if vmax is None else vmax
    im = ax.imshow(img, cmap="gray", vmin=0, vmax=top or 1.0, extent=extent(img.shape, fov_yx_mm), interpolation=interpolation)
    bare(ax)
    ax.set_aspect("equal")
    if view_mm is not None:
        ax.set_ylim(view_mm[1], view_mm[0])
        ax.set_xlim(view_mm[2], view_mm[3])
    return im


def wrap_header(header: str, width: int) -> list[str]:
    """The console header line broken at its ' | ' separators into lines of at most ``width`` characters (a single
    field longer than that keeps its own line). The text itself is not changed."""
    lines, cur = [], ""
    for part in header.split(" | "):
        part = " ".join(part.split())
        if cur and len(cur) + 3 + len(part) > width:
            lines.append(cur)
            cur = part
        else:
            cur = part if not cur else cur + " | " + part
    return lines + [cur] if cur else lines


def request_lines(protocol: dict) -> list[str]:
    """A protocol request as ``key: value`` lines, in the order the request gave them."""
    fmt = lambda v: f"{v:g}" if isinstance(v, float) else str(v).lower() if isinstance(v, bool) else str(v)  # noqa: E731
    return [f"{k}: {fmt(v)}" for k, v in protocol.items()]


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}" if seconds % 1 else f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def card(ax, x, y, w, h, colour: str = PANEL, edge: str | None = None, radius: float = 0.06, lw: float = 0.8, z: int = 0):
    """A rounded panel in data coordinates of ``ax``."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=colour, ec=edge or "none", lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def canvas(height_in: float):
    """A figure of the standard width with one invisible axis in inches: x in [0, WIDTH_IN], y in [0, height_in]."""
    fig = plt.figure(figsize=(WIDTH_IN, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_IN)
    ax.set_ylim(0, height_in)
    ax.axis("off")
    return fig, ax


def inset(fig, x, y, w, h):
    """An axis placed in inches from the lower left corner of ``fig``."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def save(fig, path: str, svg: bool = False, dpi: int = DPI) -> list[str]:
    """Write the PNG (and optionally an SVG next to it) and print the size. A PNG above the size limit is written
    again at a lower resolution, down to 120 dpi, until it fits."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = [path]
    for step in range(dpi, 119, -10):
        fig.savefig(path, dpi=step, metadata={"Software": "MR Gym tools/figures"})
        if os.path.getsize(path) <= MAX_PNG_KB * 1024:
            break
    if svg:
        out.append(os.path.splitext(path)[0] + ".svg")
        fig.savefig(out[-1], metadata={"Creator": "MR Gym tools/figures", "Date": None})
    width = int(round(fig.get_size_inches()[0] * step))
    plt.close(fig)
    for p in out:
        kb = os.path.getsize(p) / 1024
        note = f"  {width} px wide" if p.endswith(".png") else ""
        note += f"  (above the {MAX_PNG_KB} kB limit)" if p.endswith(".png") and kb > MAX_PNG_KB else ""
        print(f"{os.path.relpath(p, ROOT)}  {kb:.0f} kB{note}")
    return out
