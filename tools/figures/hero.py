"""The first-screen figure: one scripted session on the brain phantom at 3 T.

    python tools/figures/hero.py                      -> assets/hero.png

Five protocols are planned and scanned through ``Scanner/Explore-v0`` with ``env.step``, exactly as an agent would
call the tools, so every header line, cost and budget figure in the picture is what the scanner returned. No language
model is involved: the list of requests is written in this file.
"""
from __future__ import annotations

import argparse
import os

import style

import mrgym

SEED = 1
BUDGET_S = 1200.0
GEOMETRY = dict(pe_dir="x", phase_fov=0.75)         # the head is narrower left to right: fewer phase-encode lines
SESSION = [
    ("localiser", dict(sequence="spgr", tr_ms=11, flip_deg=15, matrix=128)),
    ("T1-weighted", dict(sequence="se", tr_ms=500, matrix=192, **GEOMETRY)),
    ("T2-weighted", dict(sequence="se", tr_ms=3000, te_ms=90, matrix=160, **GEOMETRY)),
    ("FLAIR-like", dict(sequence="ir", tr_ms=6000, ti_ms=2000, te_ms=90, matrix=128, **GEOMETRY)),
    ("EPI", dict(sequence="epi", tr_ms=3000, matrix=96)),
]


def play() -> tuple[list[dict], dict, dict]:
    """Run the session through the tools. Returns (one record per scan, final state, package facts)."""
    env = mrgym.make("Scanner/Explore-v0", phantom="brain", field_t=3.0, lesion="ms", seed=SEED, budget_s=BUDGET_S)
    env.reset()
    env.step({"name": "scanner_info", "arguments": {}})
    out = []
    for label, protocol in SESSION:
        plan = env.step({"name": "plan", "arguments": {"protocol": protocol}}).observation
        if plan.is_error or not plan.data["fits_budget"]:
            raise SystemExit(f"the console does not accept {label}: {plan.text}")
        obs = env.step({"name": "scan", "arguments": {"protocol": protocol}}).observation
        scan = env._scanner.scans[obs.data["scan_index"]]
        out.append(dict(label=label, protocol=protocol, header=obs.data["header"], cost_s=obs.data["cost_s"],
                        image=scan["magnitude"][0], fov_yx_mm=scan["fov_yx_mm"], status=dict(obs.status)))
    state = env.state()
    env.step({"name": "finish", "arguments": {}})
    env.close()
    return out, state, dict(engine=mrgym.ENGINE_ID, version=mrgym.__version__)


def draw(scans: list[dict], state: dict, facts: dict):
    n = len(scans)
    W, H = style.WIDTH_IN, 6.72
    fig, ax = style.canvas(H)
    ax.text(0.2, H - 0.32, "One session on the virtual scanner", fontsize=15, fontweight="bold", va="center")
    ax.text(0.2, H - 0.64, "Brain phantom at 3 T. A request is a protocol in the scanner's own parameters; the reply is an image, "
            "a header line and a charge in scan time.", fontsize=8.8, color=style.INK2, va="center")
    ax.text(W - 0.2, H - 0.32, "scripted session, no language model", fontsize=9, va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc=style.PANEL, ec=style.AXIS, lw=0.8))

    # the filmstrip: request above, image below, charge in the corner
    gap, x0 = 0.1, 0.2
    cw = (W - 2 * x0 - (n - 1) * gap) / n
    img = cw - 0.16
    top = H - 0.86
    bottom = top - (0.36 + 7 * 0.135 + 0.08 + img + 0.08)
    for k, s in enumerate(scans):
        x = x0 + k * (cw + gap)
        style.card(ax, x, bottom, cw, top - bottom)
        ax.text(x + 0.09, top - 0.19, f"{k + 1}", fontsize=10, fontweight="bold", va="center", color=style.BLUE)
        ax.text(x + 0.25, top - 0.19, s["label"], fontsize=9.6, fontweight="bold", va="center")
        cost = f"{s['cost_s']:.1f} s" if s["cost_s"] < 100 else f"{s['cost_s']:.0f} s"
        ax.text(x + cw - 0.09, top - 0.19, cost, fontsize=8.8, va="center", ha="right", color=style.BLUE,
                fontweight="bold")
        for i, line in enumerate(style.request_lines(s["protocol"])):
            ax.text(x + 0.09, top - 0.45 - 0.135 * i, line, fontsize=7.8, family=style.MONO, va="center")
        a = style.inset(fig, x + 0.08, bottom + 0.08, img, img)
        style.show(a, s["image"], s["fov_yx_mm"], interpolation="bilinear")
        a.set_xlim(-120, 120)
        a.set_ylim(120, -120)

    # the console: the header line of every scan, verbatim
    dy = 0.136
    ch = 0.3 + 2 * dy * n + 0.05
    cy = bottom - 0.12 - ch
    style.card(ax, x0, cy, W - 2 * x0, ch, colour="#16181c", radius=0.05)
    ax.text(x0 + 0.1, cy + ch - 0.16, "console header line of each scan, as returned", fontsize=8, color="#a9adb5", va="center")
    ax.text(W - x0 - 0.1, cy + ch - 0.16, f"Scanner/Explore-v0 through env.step  ·  {facts['engine']}", fontsize=8, color="#a9adb5",
            va="center", ha="right")
    for k, s in enumerate(scans):
        fields = [" ".join(f.split()) for f in s["header"].split(" | ")]
        y = cy + ch - 0.37 - 2 * dy * k
        ax.text(x0 + 0.1, y, f"{k + 1}", fontsize=8, family=style.MONO, color="#7fb4f5", va="center", fontweight="bold")
        for i, part in enumerate((fields[:4], fields[4:])):
            ax.text(x0 + 0.3, y - dy * i, " | ".join(part), fontsize=8, family=style.MONO, color="#e9ebee", va="center")

    # the budget bar: one segment per scan, in the order they were charged
    total, used = state["scan_time_budget_s"], state["scan_time_used_s"]
    bx, bw, by, bh = x0, W - 2 * x0, 0.26, 0.28
    ax.text(bx, by + bh + 0.2, f"Scan-time budget: {style.mmss(round(used))} of {style.mmss(total)} min used", fontsize=10, va="center",
            fontweight="bold")
    ax.text(bx + bw, by + bh + 0.2, f"{style.mmss(round(total - used))} left  ·  planning is free  ·  the session cannot be reset",
            fontsize=8.8, va="center", ha="right", color=style.INK2)
    style.card(ax, bx, by, bw, bh, colour=style.TRACK, radius=0.05)
    t = 0.0
    for k, s in enumerate(scans):
        w = bw * s["cost_s"] / total
        xs = bx + bw * t / total
        ax.add_patch(style.plt.Rectangle((xs, by), max(w - 0.012, 0.012), bh, fc=style.BLUE, ec="none", zorder=2))
        if w > 0.9:
            ax.text(xs + w / 2, by + bh / 2, f"{k + 1} · {s['label']} · {style.mmss(s['cost_s'])}", fontsize=8.6, color="white",
                    ha="center", va="center", zorder=3)
        else:
            ax.text(xs + w / 2, by - 0.12, f"{k + 1}", fontsize=8.4, color=style.BLUE, ha="center", va="center", fontweight="bold")
        t += s["cost_s"]
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join(style.ASSETS, "hero.png"))
    a = ap.parse_args(argv)
    style.apply()
    scans, state, facts = play()
    style.save(draw(scans, state, facts), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
