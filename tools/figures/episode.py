"""One recorded ``Scanner/NoWrap-v0`` episode, next to a shortcut policy that fails on the same hidden world.

    python tools/figures/episode.py                   -> assets/episode_nowrap.png

Without arguments the script records two scripted episodes on the same world (level 2, seed 0): the reference policy
and ``size_only``, which measures the object's size and ignores that it lies off-centre. Everything drawn comes from
the two bundles: the localiser the agent received, the geometry it delivered, the hidden object and the verifier's
noise-free re-scan of the delivered protocol.

To show a language-model episode instead, pass its bundle first and a failing scripted episode of the same world
second (record the second with the level and seed the model episode was played on):

    mrgym record Scanner/NoWrap-v0 --policy size_only --level 2 --seed 0 --out runs/size_only
    python tools/figures/episode.py --bundle runs/model/episode_bundle.json runs/size_only/episode_bundle.json \
        --out assets/episode_nowrap_model.png

One bundle alone gives a single row.
"""
from __future__ import annotations

import argparse
import os
import textwrap

import numpy as np
import style
from matplotlib.patches import Rectangle

from mrgym import record

ENV_ID = "Scanner/NoWrap-v0"
GATES = dict(gate_coverage="whole object inside", gate_no_wrap="no wrap-around", gate_resolution="pixel size",
             gate_measured="measured first", gate_log="log replays")
SIZE, X_IMG, GAP = 1.93, 3.31, 0.13                # image size, left edge of the first image, gap, in inches
OUTSIDE = "#2b2e34"                                 # the area outside an image, so the edge of a field of view shows


def fov_box(delivered: dict) -> tuple[float, float, float, float]:
    """(x_lo, y_lo, width, height) in mm of the delivered field of view: pixel i is centred at (i - n // 2) * FOV / n."""
    fy, fx = delivered["fov_yx_mm"]
    r = delivered["realised"]
    ny, nx = (r["matrix_pe"], r["matrix"]) if r["pe_dir"] == "y" else (r["matrix"], r["matrix_pe"])
    return (-fx / 2 - fx / nx / 2, -fy / 2 - fy / ny / 2, fx, fy)


def agent_chip(agent: dict) -> str:
    kind = agent.get("kind", "agent")
    return "scripted policy" if kind == "scripted" else "language model" if kind == "model" else str(kind)


def row(fig, ax, b: dict, y0: float, h: float, lim: float) -> None:
    """One episode: text, the localiser, the delivered geometry on the hidden object, the verifier's re-scan."""
    W = style.WIDTH_IN
    r, sc, dv, hid = b["result"], b["result"]["score"] or {}, b["result"]["delivered"], b["hidden"]
    style.card(ax, 0.2, y0, W - 0.4, h)
    passed = bool(sc.get("passed"))
    tx, ty = 0.36, y0 + h - 0.26
    ax.text(tx, ty, b["agent"]["name"], fontsize=12, fontweight="bold", va="center")
    ax.text(tx + 2.62, ty, agent_chip(b["agent"]), fontsize=8.4, va="center", ha="right", color=style.INK2,
            bbox=dict(boxstyle="round,pad=0.3", fc=style.SURFACE, ec=style.AXIS, lw=0.7))
    y = ty - 0.27
    for line in textwrap.wrap(b["agent"].get("description") or "", 46)[:3]:
        ax.text(tx, y, line, fontsize=8.6, color=style.INK2, va="center")
        y -= 0.155
    used = (r.get("budget") or {}).get("used_s", 0.0)
    y -= 0.06
    ax.text(tx, y, f"looked first: {len(b['scans'])} scan, {used:.1f} of {b['task']['budget_s']:.0f} s of scan time", fontsize=8.6, va="center")
    y -= 0.22
    ax.text(tx, y, "delivered with submit", fontsize=8, color=style.MUTED, va="center")
    answer = r["answer"] or {}
    val = lambda k: f"{k}: {answer[k]:.4g}" if isinstance(answer[k], float) else f"{k}: {answer[k]}"  # noqa: E731
    for pair in (("pe_dir", "fov_mm"), ("matrix", "phase_fov")):
        y -= 0.145
        for i, k in enumerate(pair):
            if k in answer:
                ax.text(tx + 1.2 * i, y, val(k), fontsize=7.8, family=style.MONO, va="center")
    y -= 0.27
    if sc.get("scan_s") is not None:
        ax.text(tx, y, f"scan time {sc['scan_s']:.1f} s", fontsize=10, fontweight="bold", va="center")
        ax.text(tx + 1.5, y, f"fastest valid {sc['scan_s_fastest']:.1f} s", fontsize=8.2, color=style.INK2, va="center")
    y -= 0.27
    ax.text(tx, y, f"reward {r['reward']:.2f}", fontsize=10, fontweight="bold", va="center")
    ax.text(tx + 1.54, y, "pass" if passed else "fail", fontsize=8.8, fontweight="bold", color="white", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc=style.GOOD if passed else style.CRITICAL, ec="none"))
    y -= 0.25
    for key, name in GATES.items():
        if key in sc and (key in ("gate_coverage", "gate_no_wrap") or not sc[key]):
            ok = bool(sc[key])
            ax.add_patch(Rectangle((tx, y - 0.04), 0.08, 0.08, fc=style.GOOD if ok else style.CRITICAL, ec="none"))
            ax.text(tx + 0.14, y, f"{name}: {'ok' if ok else 'failed'}", fontsize=8.4, va="center",
                    color=style.INK if ok else style.CRITICAL)
            y -= 0.15

    size, x_img, gap = SIZE, X_IMG, GAP
    y_img = y0 + 0.47

    def caption(k, lines, colour=style.INK2):
        for i, line in enumerate(lines):
            ax.text(x_img + k * (size + gap), y_img - 0.13 - 0.15 * i, line, fontsize=7.8, color=colour, va="center")

    # 1. what the agent saw
    a1 = style.inset(fig, x_img, y_img, size, size)
    scan = b["scans"][-1] if b["scans"] else None
    if scan:
        mag, _ = record.scan_arrays(b, scan["index"])
        style.show(a1, mag[0], scan["fov_yx_mm"])
        a1.set_xlim(-lim, lim)
        a1.set_ylim(lim, -lim)
        boxes = [s["data"].get("bounding_box") for s in b["steps"] if s["tool"] == "view" and not s["is_error"]]
        cap = [f"{scan['sequence']} localiser, {scan['fov_yx_mm'][0]:.0f} x {scan['fov_yx_mm'][1]:.0f} mm"]
        if boxes and boxes[-1]:
            (ya, yb), (xa, xb) = boxes[-1]["y_mm"], boxes[-1]["x_mm"]
            a1.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, fill=False, ec=style.OVERLAY_REGION, lw=1.0, ls=(0, (3, 2))))
            cap = ["extent of the object read with view:", f"y {ya:.0f} to {yb:.0f} mm, x {xa:.0f} to {xb:.0f} mm"]
    else:
        style.bare(a1)
        cap = ["no scan"]
    caption(0, cap)
    # 2. the delivered field of view on the hidden object
    a2 = style.inset(fig, x_img + size + gap, y_img, size, size)
    m = hid["maps"]
    pd = record.hidden_map(b, "pd")
    half, d = m["extent_mm"] / 2, m["extent_mm"] / hid["world"]["master_grid"]
    ext = (-half - d / 2, half - d / 2, half - d / 2, -half - d / 2)
    a2.imshow(pd, cmap="gray", vmin=0, vmax=float(pd.max()) or 1.0, extent=ext, interpolation="nearest")
    style.bare(a2)
    e = hid["object_extent_mm"]
    cap2 = ["the console refuses this geometry"]
    if dv:
        bx, by, bw, bh = fov_box(dv)
        yy = -half + (np.arange(pd.shape[0]) + 0.5) * m["extent_mm"] / pd.shape[0]
        outside = (pd > 0.05 * pd.max()) & ~((yy[:, None] >= by) & (yy[:, None] <= by + bh) & (yy[None, :] >= bx) & (yy[None, :] <= bx + bw))
        if outside.any():
            tint = np.zeros(pd.shape + (4,))
            tint[outside] = (0.86, 0.16, 0.16, 0.9)
            a2.imshow(tint, extent=ext, interpolation="nearest")
        a2.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=style.OVERLAY_FOV, lw=1.6))
        cap2 = [f"delivered: {dv['fov_yx_mm'][0]:.0f} x {dv['fov_yx_mm'][1]:.0f} mm (y, x),", f"phase encoding along {dv['realised']['pe_dir']}"]
    a2.add_patch(Rectangle((e["x"][0], e["y"][0]), e["x"][1] - e["x"][0], e["y"][1] - e["y"][0], fill=False,
                           ec=style.OVERLAY_TRUTH, lw=1.0, ls=(0, (3, 2))))
    a2.plot([0], [0], marker="+", color="white", ms=6, mew=0.9)
    a2.set_xlim(-lim, lim)
    a2.set_ylim(lim, -lim)
    caption(1, cap2)
    # 3. the verifier's re-scan
    a3 = style.inset(fig, x_img + 2 * (size + gap), y_img, size, size)
    rs = hid.get("rescan")
    cap3, bad = ["no re-scan: nothing valid was delivered"], ["none"]
    if rs:
        img = record.decode_f32(rs["magnitude_f32_b64"], rs["shape"])[0]
        style.show(a3, img, rs["fov_yx_mm"], interpolation="bilinear")
        bx, by, bw, bh = fov_box(dv)
        a3.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=style.OVERLAY_FOV, lw=1.6))
        wrap = sc.get("wrap_fraction")
        bad = [n for k, n in (("gate_coverage", "object cut off"), ("gate_no_wrap", "wrap-around")) if sc.get(k) is False]
        cap3 = [" and ".join(bad)] + ([f"wrap energy {100 * wrap:.1f} % of the image"] if wrap is not None else []) if bad else \
            ["whole object, no wrap-around", f"scan time {dv['scan_s']:.1f} s"]
    else:
        style.bare(a3)
    a3.set_facecolor(OUTSIDE)
    a3.set_xlim(-lim, lim)
    a3.set_ylim(lim, -lim)
    caption(2, cap3, style.CRITICAL if bad else style.INK2)


def draw(bundles: list[dict]):
    main = bundles[0]
    if main["env"]["id"] != ENV_ID:
        raise SystemExit(f"this figure is drawn for {ENV_ID}; the bundle is {main['env']['id']}")
    same = all(b["episode"]["task_id"] == main["episode"]["task_id"] for b in bundles)
    W, rh, head = style.WIDTH_IN, 2.63, 1.6
    H = head + len(bundles) * (rh + 0.12) + 0.06
    fig, ax = style.canvas(H)
    ax.text(0.2, H - 0.32, f"One episode of {ENV_ID}, level {main['env']['level']}", fontsize=15, fontweight="bold", va="center")
    task = main["steps"][0]["text"].splitlines()[0].lstrip("# ").strip()
    ax.text(0.2, H - 0.64, f"Task: {task[0].lower() + task[1:]}. Size, angle and position are hidden; "
            "the field of view stays centred.", fontsize=8.8, color=style.INK2, va="center")
    kinds = {b["agent"].get("kind") for b in bundles}
    note = "scripted policies, no language model" if kinds == {"scripted"} else "recorded episodes"
    ax.text(W - 0.2, H - 0.32, note + (", same hidden world" if same and len(bundles) > 1 else ""), fontsize=9, va="center",
            ha="right", bbox=dict(boxstyle="round,pad=0.35", fc=style.PANEL, ec=style.AXIS, lw=0.8))
    size, x_img, gap = SIZE, X_IMG, GAP
    yh = H - head + 0.26
    heads = [("what the agent received", "the localiser it paid for"),
             ("what it delivered", "on the hidden object, never shown"),
             ("the verifier's re-scan", "delivered protocol, noise-free")]
    for k, (a, c) in enumerate(heads):
        ax.text(x_img + k * (size + gap), yh + 0.16, a, fontsize=8.8, fontweight="bold", va="center")
        ax.text(x_img + k * (size + gap), yh, c, fontsize=8, color=style.INK2, va="center")
    # legend of the overlays
    lx, ly = 0.36, yh + 0.36
    for colour, dash, text in ((style.OVERLAY_FOV, False, "delivered field of view"), (style.OVERLAY_TRUTH, True, "true extent of the object"),
                               ("#db2929", None, "object outside the image")):
        if dash is None:
            ax.add_patch(Rectangle((lx, ly - 0.045), 0.22, 0.09, fc=colour, ec="none"))
        else:
            ax.plot([lx, lx + 0.22], [ly, ly], color=colour, lw=1.8, ls=(0, (3, 2)) if dash else "-")
        ax.text(lx + 0.3, ly, text, fontsize=8, color=style.INK2, va="center")
        ly -= 0.15
    ax.plot([lx + 0.11], [ly], marker="+", color=style.INK, ms=6, mew=0.9)
    ax.text(lx + 0.3, ly, "isocentre", fontsize=8, color=style.INK2, va="center")
    lim = 136.0
    for k, b in enumerate(bundles):
        row(fig, ax, b, H - head - (k + 1) * (rh + 0.12) + 0.12, rh, lim)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bundle", nargs="+", default=None, help="one or two episode bundles; default: record two scripted episodes")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--policy", default="reference")
    ap.add_argument("--compare", default="size_only", help="the failing scripted policy recorded on the same world")
    ap.add_argument("--out", default=os.path.join(style.ASSETS, "episode_nowrap.png"))
    a = ap.parse_args(argv)
    style.apply()
    if a.bundle:
        bundles = [record.load(p) for p in a.bundle[:2]]
    else:
        bundles = [record.record(ENV_ID, policy=p, level=a.level, seed=a.seed) for p in (a.policy, a.compare)]
    style.save(draw(bundles), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
