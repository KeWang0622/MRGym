"""What the world holds against what one scan returns.

    python tools/figures/hidden_vs_received.py        -> assets/hidden_vs_received.png

Top row: four maps of the hidden world of a ``Scanner/Explore-v0`` session (brain phantom, 3 T): tissue classes, T1,
off-resonance and the transmit field. They are read from the environment object by this script; no tool returns them.
Bottom row: one ``scan`` call made through ``env.step`` and everything it returned: the image, the header line and
the region read-outs, as the text the agent reads.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import style
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle, Rectangle

import mrgym
from mrgym.scanner import anatomy

SEED = 1
PROTOCOL = dict(sequence="se", tr_ms=3000, te_ms=90, matrix=160, pe_dir="x", phase_fov=0.75)
REGIONS = [
    dict(id="wm", y_mm=-52.0, x_mm=22.0, radius_mm=5.0),
    dict(id="deep", y_mm=-8.0, x_mm=30.0, radius_mm=5.0),
    dict(id="csf", y_mm=30.0, x_mm=20.0, radius_mm=3.0),
    dict(id="lesion", y_mm=10.4, x_mm=-40.8, radius_mm=3.0),
    dict(id="air", y_mm=-100.0, x_mm=-70.0, radius_mm=8.0),
]
VIEW = 112.0                                        # half width of every panel, mm


def play() -> tuple[dict, dict]:
    """One session: a scan with read-out circles. Returns (what the agent received, the hidden world)."""
    env = mrgym.make("Scanner/Explore-v0", phantom="brain", field_t=3.0, lesion="ms", seed=SEED)
    env.reset()
    obs = env.step({"name": "scan", "arguments": {"protocol": PROTOCOL, "regions": REGIONS}}).observation
    if obs.is_error:
        raise SystemExit(obs.text)
    scan = env._scanner.scans[obs.data["scan_index"]]
    received = dict(text=obs.text, image=scan["magnitude"][0], fov_yx_mm=scan["fov_yx_mm"])
    world = env._session.world
    env.close()
    return received, world


def wrap_console(text: str, width: int) -> list[str]:
    """Console text with long lines broken at ' | ' or ' · ' separators; continuation lines are indented."""
    out = []
    for line in text.splitlines():
        sep = " | " if " | " in line else " · "
        cur = ""
        for part in line.split(sep):
            if cur and len(cur) + len(sep) + len(part) > width:
                out.append(cur)
                cur = "    " + part
            else:
                cur = part if not cur else cur + sep + part
        out.append(cur)
    return out


def draw(received: dict, world: dict):
    W = style.WIDTH_IN
    n, ext = world["n"], world["extent_mm"]
    d = ext / n
    extent = (-ext / 2 - d / 2, ext / 2 - d / 2, ext / 2 - d / 2, -ext / 2 - d / 2)
    body = world["weights"].sum(0) > 0.3
    names = world["names"]
    msize = 2.0
    H = 0.92 + (0.36 + msize + 0.74) + 0.2 + (0.36 + 2.3 + 0.12) + 0.1
    fig, ax = style.canvas(H)
    ax.text(0.2, H - 0.32, "What the world holds, and what a scan returns", fontsize=15, fontweight="bold", va="center")
    ax.text(0.2, H - 0.64, "The agent never receives a map. It receives images, a header line and the numbers of the regions it asks for, "
            "and pays scan time for each.", fontsize=8.8, color=style.INK2, va="center")

    # ---- hidden
    top = H - 0.92
    hh = 0.36 + msize + 0.74
    frame = style.card(ax, 0.2, top - hh, W - 0.4, hh, colour=style.SURFACE, edge=style.INK2, radius=0.08, lw=1.0)
    frame.set_linestyle((0, (4, 3)))
    ax.text(0.34, top - 0.19, "Hidden: the world of this session", fontsize=10, fontweight="bold", va="center")
    ax.text(W - 0.34, top - 0.19, "scanner and verifier side only", fontsize=8.6, color=style.INK2, va="center", ha="right")
    gap = (W - 0.4 - 0.28 - 4 * msize) / 3
    xs = [0.34 + k * (msize + gap) for k in range(4)]
    y_map = top - 0.36 - msize
    t1 = np.where(body, anatomy.param_map(world, "t1_ms"), np.nan)
    b0 = np.where(body, world["fields"]["b0_hz"] + world["system"]["cf_offset_hz"], np.nan)
    b1 = np.where(body, world["fields"]["b1"] * world["system"]["tx_gain"], np.nan)
    b0_lim = float(np.ceil(np.nanpercentile(np.abs(b0), 99.5) / 10) * 10)
    b1_dev = float(np.ceil(np.nanmax(np.abs(b1 - 1)) * 20) / 20)
    maps = [
        ("tissue classes", anatomy.label_map(world), style.classes(len(names)), Normalize(-0.5, len(names) + 0.5), None),
        ("T1 (ms)", t1, style.sequential(), LogNorm(300, 4500), [300, 1000, 4500]),
        ("off-resonance (Hz)", b0, style.diverging(), Normalize(-b0_lim, b0_lim), [-b0_lim, 0, b0_lim]),
        ("transmit field (B1+ ratio)", b1, style.diverging(), Normalize(1 - b1_dev, 1 + b1_dev), [1 - b1_dev, 1, 1 + b1_dev]),
    ]
    for x, (title, arr, cmap, norm, ticks) in zip(xs, maps):
        a = style.inset(fig, x, y_map, msize, msize)
        a.imshow(arr, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        style.bare(a, face="black" if ticks is None else style.PANEL)
        a.set_xlim(-VIEW, VIEW)
        a.set_ylim(VIEW, -VIEW)
        ax.text(x, y_map - 0.14, title, fontsize=8.8, fontweight="bold", va="center")
        if ticks is None:                                               # class legend: three columns
            for i, name in enumerate(names):
                cx, cy = x + (i % 3) * 0.68, y_map - 0.33 - (i // 3) * 0.125
                ax.add_patch(Rectangle((cx, cy - 0.04), 0.1, 0.08, fc=style.CLASS_COLOURS[i + 1], ec=style.AXIS, lw=0.4))
                ax.text(cx + 0.14, cy, name.replace("_", " "), fontsize=6.9, color=style.INK2, va="center")
        else:
            cb = style.inset(fig, x, y_map - 0.36, msize * 0.72, 0.09)
            grad = np.linspace(0, 1, 256)[None, :]
            cb.imshow(grad, cmap=cmap, aspect="auto", extent=(0, 1, 0, 1))
            style.bare(cb, face="none")
            for t in ticks:
                pos = float(norm(t))
                ax.text(x + pos * msize * 0.72, y_map - 0.47, f"{t:g}", fontsize=7.4, color=style.INK2, va="center",
                        ha="left" if pos < 0.05 else "right" if pos > 0.95 else "center")

    # ---- received
    top2 = top - hh - 0.2
    rh = 0.36 + 2.3 + 0.12
    style.card(ax, 0.2, top2 - rh, W - 0.4, rh)
    ax.text(0.34, top2 - 0.19, "Received: one scan call and its reply", fontsize=10, fontweight="bold", va="center")
    ax.text(W - 0.34, top2 - 0.19, "through env.step, scripted, no language model", fontsize=8.6, color=style.INK2, va="center", ha="right")
    y0 = top2 - 0.36 - 2.3
    # the request
    ax.text(0.34, top2 - 0.5, "request: scan", fontsize=8.4, color=style.MUTED, va="center")
    lines = style.request_lines(PROTOCOL) + ["regions:"] + [f" {r['id']}: y {r['y_mm']:g}, x {r['x_mm']:g}, r {r['radius_mm']:g}" for r in REGIONS]
    for i, line in enumerate(lines):
        ax.text(0.34, top2 - 0.68 - 0.138 * i, line, fontsize=7.4, family=style.MONO, va="center")
    # the image
    xi = 2.5
    a = style.inset(fig, xi, y0, 2.3, 2.3)
    style.show(a, received["image"], received["fov_yx_mm"], interpolation="bilinear")
    a.set_xlim(-VIEW - 3, VIEW + 3)
    a.set_ylim(VIEW + 3, -VIEW - 3)
    for r in REGIONS:
        a.add_patch(Circle((r["x_mm"], r["y_mm"]), r["radius_mm"], fill=False, ec=style.OVERLAY_REGION, lw=1.1))
        a.text(r["x_mm"] + r["radius_mm"] + 3, r["y_mm"], r["id"], fontsize=7.4, color=style.OVERLAY_REGION, va="center")
    # the reply
    xt = xi + 2.3 + 0.2
    ax.text(xt, top2 - 0.5, "reply: the text the agent reads",
            fontsize=8.4, color=style.MUTED, va="center")
    width = int((W - 0.34 - xt) * 72 / (0.602 * 6.9))
    for i, line in enumerate(wrap_console(received["text"], width)):
        ax.text(xt, top2 - 0.72 - 0.15 * i, line, fontsize=6.9, family=style.MONO, va="center")
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join(style.ASSETS, "hidden_vs_received.png"))
    a = ap.parse_args(argv)
    style.apply()
    style.save(draw(*play()), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
