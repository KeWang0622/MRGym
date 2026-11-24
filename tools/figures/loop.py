"""The loop: agent, tools, scanner (console and engine), hidden world, log, verifier and reward.

    python tools/figures/loop.py                      -> assets/loop.png and assets/loop.svg

A drawing, not a measurement. Every label names something in the package: the tools of ``mrgym.envs.scanner_env``,
the console and the k-space engine of ``mrgym.scanner``, the world components of ``mrgym.scanner.world``, the
hash-chained log and its replay in ``mrgym.scanner.session``.
"""
from __future__ import annotations

import argparse
import os

import style
from matplotlib.patches import FancyArrowPatch


def draw():
    W, H = style.WIDTH_IN, 4.5
    fig, ax = style.canvas(H)
    blue, ink, ink2 = style.BLUE, style.INK, style.INK2

    def box(x, y, w, h, title, lines, colour=style.PANEL, edge=style.AXIS, dashed=False, title_size=10.5, size=8.2):
        p = style.card(ax, x, y, w, h, colour=colour, edge=edge, radius=0.08, lw=1.0)
        if dashed:
            p.set_linestyle((0, (4, 3)))
        ax.text(x + 0.13, y + h - 0.21, title, fontsize=title_size, fontweight="bold", va="center")
        for k, ln in enumerate(lines):
            ax.text(x + 0.13, y + h - 0.47 - 0.175 * k, ln, fontsize=size, color=ink2, va="center")

    def arrow(p, q, colour=ink2, lw=1.5):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=lw, color=colour, shrinkA=0, shrinkB=0, zorder=4))

    def label(x, y, text, mono=False, colour=ink, size=8.2, ha="center"):
        ax.text(x, y, text, fontsize=size, color=colour, va="center", ha=ha, family=style.MONO if mono else "DejaVu Sans")

    # the boundary of the environment: only tool calls cross it
    frame = style.card(ax, 3.55, 0.16, W - 3.55 - 0.16, H - 0.62, colour="none", edge=style.MUTED, radius=0.1, lw=1.0)
    frame.set_linestyle((0, (2, 3)))
    label(3.55, H - 0.26, "the environment: only tool calls cross this line; the world never does", colour=ink2, size=8.4, ha="left")

    box(0.16, 0.5, 1.95, H - 1.3, "Agent", ["a language model", "or a scripted policy", "", "reads the task and", "the replies",
                                             "", "never sees the world,", "its seed or the truth"])
    # scanner: console and engine inside one box
    sx, sy, sw, sh = 3.75, 1.75, 2.75, H - 2.4
    style.card(ax, sx, sy, sw, sh, colour=style.PANEL, edge=style.AXIS, radius=0.08, lw=1.0)
    ax.text(sx + 0.13, sy + sh - 0.21, "Virtual scanner", fontsize=10.5, fontweight="bold", va="center")
    box(sx + 0.12, sy + 0.12, 1.2, sh - 0.52, "console", ["checks the", "protocol,", "prices it in", "scan time,", "keeps the", "budget"],
        colour=style.SURFACE, title_size=9.2, size=7.9)
    box(sx + 1.43, sy + 0.12, 1.2, sh - 0.52, "engine", ["k-space", "signal of", "the world", "for that", "protocol,", "plus noise"],
        colour=style.SURFACE, title_size=9.2, size=7.9)
    box(7.05, 1.75, W - 7.05 - 0.36, H - 2.4, "Hidden world", ["anatomy and tissue:", "PD, T1, T2, fat", "fields: B0, B1+, receive",
                                                                "system: gain and", "frequency errors, noise", "one 128-bit seed"],
        dashed=True, edge=ink2)
    box(3.75, 0.36, W - 3.75 - 0.36, 1.0, "Verifier", ["replays the log and checks the budget; scores the deliverable on the hidden world",
                                                        "reward = gates first, then a continuous score"], colour=style.SURFACE)

    # agent -> scanner: the tools
    x0, x1 = 2.11, 3.75
    xm = (x0 + x1) / 2
    rows = [(3.52, "scanner_info  plan", "free"), (2.98, "scan", "charged scan time"), (2.44, "read_regions  view", "free, on stored scans")]
    for y, name, note in rows:
        arrow((x0, y), (x1, y), blue)
        label(xm, y + 0.13, name, mono=True, size=8)
        label(xm, y - 0.13, note, colour=ink2, size=7.6)
    arrow((x1, 1.93), (x0, 1.93))
    label(xm, 2.06, "reply: text, images", size=8)
    label(xm, 1.80, "header line, read-outs, budget", colour=ink2, size=7.6)
    # console <-> engine <-> world
    arrow((sx + 1.32, sy + 0.95), (sx + 1.43, sy + 0.95), ink2, lw=1.2)
    arrow((6.5, 2.95), (7.05, 2.95))
    label(6.775, 3.08, "protocol", size=7.6)
    arrow((7.05, 2.45), (6.5, 2.45))
    label(6.775, 2.32, "signal", size=7.6)
    # log, verifier, reward
    arrow((4.45, 1.75), (4.45, 1.36))
    label(4.55, 1.555, "scanner log: every request, hash-chained", size=7.6, ha="left")
    arrow((8.1, 1.75), (8.1, 1.36))
    label(8.2, 1.555, "truth", size=7.6, ha="left")
    arrow((x0, 1.12), (3.75, 1.12), blue)
    ax.text(xm, 1.25, "submit", fontsize=8, family=style.MONO, va="center", ha="center",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=5)
    label(xm, 0.99, "ends the episode", colour=ink2, size=7.6)
    arrow((3.75, 0.62), (x0, 0.62))
    ax.text(xm, 0.75, "reward", fontsize=8, va="center", ha="center", bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=5)
    label(xm, 0.49, "0 until the end", colour=ink2, size=7.6)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join(style.ASSETS, "loop.png"))
    a = ap.parse_args(argv)
    style.apply()
    style.save(draw(), a.out, svg=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
