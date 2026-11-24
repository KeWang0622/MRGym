"""One brain through the sequence set, and the abdomen with and without fat saturation.

    python tools/figures/gallery_sequences.py         -> assets/gallery_sequences.png

Every panel is one scan of the k-space engine on the same hidden phantom; the text under it is the console's header
line for that scan, unchanged.
"""
from __future__ import annotations

import argparse
import os

import scenes
import style

HEAD = dict(pe_dir="x", phase_fov=0.75, matrix=192)             # the head is narrower left to right
BODY = dict(matrix=256, phase_fov=0.75)
PANELS = [
    ("brain", "T1-weighted gradient echo", dict(sequence="spgr", tr_ms=18, flip_deg=30, **HEAD)),
    ("brain", "T1-weighted spin echo", dict(sequence="se", tr_ms=500, **HEAD)),
    ("brain", "T2-weighted spin echo", dict(sequence="se", tr_ms=3000, te_ms=90, **HEAD)),
    ("brain", "FLAIR-like inversion recovery", dict(sequence="ir", tr_ms=9000, ti_ms=2500, te_ms=90, **HEAD)),
    ("brain", "balanced SSFP", dict(sequence="bssfp", tr_ms=6, flip_deg=40, **HEAD)),
    ("brain", "single-shot echo-planar", dict(sequence="epi", tr_ms=3000, matrix=96)),
    ("abdomen", "abdomen, gradient echo", dict(sequence="spgr", tr_ms=120, flip_deg=70, **BODY)),
    ("abdomen", "abdomen, with fat saturation", dict(sequence="spgr", tr_ms=120, flip_deg=70, fat_sat=True, **BODY)),
]
VIEW = dict(brain=(-112, 112, -112, 112), abdomen=(-160, 160, -160, 160))


def draw():
    cols, rows = 4, 2
    W = style.WIDTH_IN
    x0, gap = 0.2, 0.12
    cw = (W - 2 * x0 - (cols - 1) * gap) / cols
    lines, dy = 6, 0.122
    rh = 0.3 + cw + 0.1 + lines * dy + 0.14
    H = 0.98 + rows * rh
    fig, ax = style.canvas(H)
    ax.text(x0, H - 0.32, "One phantom, the sequence set", fontsize=15, fontweight="bold", va="center")
    ax.text(x0, H - 0.64, "A brain slice with white-matter lesions and an abdomen, both hidden worlds at 3 T. "
            "Under each image: the console's header line for that scan.", fontsize=8.8, color=style.INK2, va="center")
    for k, (phantom, title, protocol) in enumerate(PANELS):
        s = scenes.scan(phantom, protocol)
        r, c = divmod(k, cols)
        x, top = x0 + c * (cw + gap), H - 0.98 - r * rh
        ax.text(x, top - 0.14, title, fontsize=9.2, fontweight="bold", va="center")
        a = style.inset(fig, x, top - 0.3 - cw, cw, cw)
        style.show(a, s["image"], s["fov_yx_mm"], view_mm=VIEW[phantom], interpolation="bilinear")
        for i, line in enumerate(style.wrap_header(s["header"], 38)[:lines]):
            ax.text(x, top - 0.3 - cw - 0.15 - dy * i, line, fontsize=6.9, family=style.MONO, color=style.INK2, va="center")
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join(style.ASSETS, "gallery_sequences.png"))
    a = ap.parse_args(argv)
    style.apply()
    style.save(draw(), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
