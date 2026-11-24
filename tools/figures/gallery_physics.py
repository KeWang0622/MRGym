"""Textbook behaviours of the k-space engine: what a protocol parameter does to the image.

    python tools/figures/gallery_physics.py           -> assets/gallery_physics.png

Five columns, one behaviour each; within a column only the parameter in the panel title changes. Every panel is one
scan of the engine on the gallery phantoms (3 T). Nothing is added after the scan: the shifts, the folding, the
ringing, the bands and the distortion come out of the Fourier integral of the hidden world.
"""
from __future__ import annotations

import argparse
import os
import textwrap

import scenes
import style

BRAIN = (-101, 101, -120, 120)                      # y_lo, y_hi, x_lo, x_hi in mm
COLUMNS = [
    dict(title="Chemical shift", phantom="abdomen", view=(-135, 135, -160, 160),
         text="Fat shifts along the readout by 434 Hz over the bandwidth per pixel; the readout polarity sets the side.",
         base=dict(sequence="se", tr_ms=500, matrix=256, phase_fov=0.75),
         panels=[("bandwidth_hz_per_px: 800", dict(bandwidth_hz_per_px=800)),
                 ("bandwidth_hz_per_px: 80", dict(bandwidth_hz_per_px=80)),
                 ("+ ro_polarity: -1", dict(bandwidth_hz_per_px=80, ro_polarity=-1))]),
    dict(title="Wrap-around", phantom="brain", view=BRAIN,
         text="Anatomy outside the phase field of view folds back into the image; phase oversampling removes it.",
         base=dict(sequence="se", tr_ms=500, matrix=192),
         panels=[("phase_fov: 1", dict(phase_fov=1.0)),
                 ("phase_fov: 0.6", dict(phase_fov=0.6)),
                 ("+ phase_oversampling: 0.7", dict(phase_fov=0.6, phase_oversampling=0.7))]),
    dict(title="Truncation (Gibbs)", phantom="brain", view=(-78, 48, -75, 75),
         text="Fewer k-space samples over the same field of view: ringing next to every sharp edge.",
         base=dict(sequence="se", tr_ms=3000, te_ms=90),
         panels=[("matrix: 256", dict(matrix=256)), ("matrix: 128", dict(matrix=128)), ("matrix: 64", dict(matrix=64))]),
    dict(title="bSSFP bands", phantom="brain", view=BRAIN,
         text="Signal nulls where off-resonance turns the phase by half a cycle per TR; a longer TR brings the bands closer.",
         base=dict(sequence="bssfp", flip_deg=40, matrix=192),
         panels=[("tr_ms: 6", dict(tr_ms=6)), ("tr_ms: 14", dict(tr_ms=14)), ("tr_ms: 28", dict(tr_ms=28))]),
    dict(title="EPI distortion", phantom="brain", view=BRAIN,
         text="Off-resonance near air displaces signal along the phase direction; the blip polarity reverses it.",
         base=dict(tr_ms=3000, matrix=96),
         panels=[("se (reference)", dict(sequence="se", te_ms=45)),
                 ("epi, pe_polarity: 1", dict(sequence="epi", pe_polarity=1)),
                 ("epi, pe_polarity: -1", dict(sequence="epi", pe_polarity=-1))]),
]


def draw():
    n = len(COLUMNS)
    W = style.WIDTH_IN
    x0, gap = 0.2, 0.1
    cw = (W - 2 * x0 - (n - 1) * gap) / n
    ph = cw / 1.187
    head, foot, th = 0.82, 0.42, 0.21
    H = 0.9 + head + 3 * (th + ph + 0.06) + foot
    fig, ax = style.canvas(H)
    ax.text(x0, H - 0.32, "What a protocol parameter does to the image", fontsize=15, fontweight="bold", va="center")
    ax.text(x0, H - 0.64, "One behaviour per column; from one row to the next only the parameter in the panel title changes. "
            "Abdomen and brain phantoms at 3 T.", fontsize=8.8, color=style.INK2, va="center")
    for c, col in enumerate(COLUMNS):
        x, top = x0 + c * (cw + gap), H - 0.9
        ax.text(x, top - 0.12, col["title"], fontsize=10, fontweight="bold", va="center")
        for i, line in enumerate(textwrap.wrap(col["text"], 30)[:4]):
            ax.text(x, top - 0.33 - 0.135 * i, line, fontsize=7.6, color=style.INK2, va="center")
        vmax = None
        for r, (title, change) in enumerate(col["panels"]):
            s = scenes.scan(col["phantom"], dict(col["base"], **change))
            y = top - head - r * (th + ph + 0.06)
            ax.text(x, y - 0.14, title, fontsize=7.8, family=style.MONO, va="baseline")
            a = style.inset(fig, x, y - th - ph, cw, ph)
            if col["title"].startswith("Truncation") and vmax is None:
                vmax = float(style.np.percentile(s["image"], 99.7))    # one grey scale down the column: the matrix changes the noise
            style.show(a, s["image"], s["fov_yx_mm"], vmax=vmax if col["title"].startswith("Truncation") else None,
                       view_mm=col["view"], interpolation="bilinear")
        base = " ".join(f"{k}={v}" for k, v in col["base"].items())
        for i, line in enumerate(textwrap.wrap(f"{col['phantom']}: {base}", 30)[:2]):
            ax.text(x, foot - 0.14 - 0.125 * i, line, fontsize=6.9, family=style.MONO, color=style.MUTED, va="center")
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join(style.ASSETS, "gallery_physics.png"))
    a = ap.parse_args(argv)
    style.apply()
    style.save(draw(), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
