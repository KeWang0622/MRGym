"""10 EPI distortion: off-resonance moves and stretches the image along phase encoding; reversing the blips reverses it.

A known world: a disc of fluid (radius 40 mm) in a field of 25 Hz + 0.5 Hz/mm x y, no noise. Two single-shot ``epi``
scans with ``pe_polarity`` +1 and -1. Along phase encoding the bandwidth per pixel is 1 / (echo spacing x lines), so a
spin at y appears at  y - polarity x off-resonance(y) x echo spacing x field of view  (Jezzard & Balaban 1995):
a constant offset shifts the disc, a field gradient stretches or compresses it by 1 -/+ gradient x echo spacing x FOV.
Signal is conserved, so the compressed image is brighter. The mean of the two positions is the true position.

    python examples/textbook/10_epi_distortion.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import kspace
from mrgym.scanner import world as W

OFFSET, GRADIENT, RADIUS = 25.0, 0.5, 40.0                           # Hz, Hz / mm, mm
FOV, MATRIX, ESP, UP = 240.0, 96, 1.0, 8                             # mm, pixels, ms


def main():
    args = C.parse_args(__doc__)
    phantom = W.shapes_phantom([("csf", lambda y, x: np.hypot(y, x) - RADIUS)])
    world = W.flat_world(phantom, cf_offset_hz=OFFSET)
    n, d = phantom["n"], phantom["extent_mm"] / phantom["n"]
    world["fields"]["b0_hz"] = GRADIENT * np.broadcast_to(((np.arange(n) - n // 2) * d)[:, None], (n, n)).copy()
    engine = kspace.Engine(world)
    t_fov = ESP * 1e-3 * FOV                                         # s x mm: displacement in mm per Hz
    t2s = world["tissues"]["csf"]["t2s_ms"]
    measured, images = {}, {}
    for polarity in (1, -1):
        out = engine.scan(dict(sequence="epi", tr_ms=3000, fov_mm=FOV, matrix=MATRIX, bandwidth_hz_per_px=2500,
                               echo_spacing_ms=ESP, pe_polarity=polarity, fat_sat=False))
        images[polarity] = np.abs(out["images"][0])
        fine = np.abs(C.upsample(out["images"][0], UP))
        pos = (np.arange(fine.shape[0]) - fine.shape[0] // 2) * FOV / fine.shape[0]
        column = fine[:, fine.shape[1] // 2 - UP // 2:fine.shape[1] // 2 + UP // 2].mean(1)
        above = pos[column > 0.5 * column.max()]
        lo, hi = float(above[0]), float(above[-1])
        plateau = float(column[(pos > lo + 10) & (pos < hi - 10)].mean())
        scale, shift = 1 - polarity * GRADIENT * t_fov, -polarity * OFFSET * t_fov
        measured[polarity] = (lo, hi, plateau)
        print(f"blips {polarity:+d}: disc from {lo:+.1f} to {hi:+.1f} mm (expected {-RADIUS * scale + shift:+.1f} to {RADIUS * scale + shift:+.1f}) | "
              f"centre {(lo + hi) / 2:+.2f} mm | height {hi - lo:.1f} mm")
        C.check(f"blips {polarity:+d}: shift of the centre", (lo + hi) / 2, shift, 0.3, "mm")
        C.check(f"blips {polarity:+d}: height / true height", (hi - lo) / (2 * RADIUS), scale, 0.01)
    up, down = measured[1], measured[-1]
    C.check("mean of the two centres (the true position)", (up[0] + up[1] + down[0] + down[1]) / 4, 0.0, 0.3, "mm")
    # brightness: 1 / scale from conservation of signal, times T2* decay over the echo shift that the gradient causes
    te = engine.scan(dict(sequence="epi", tr_ms=3000, fov_mm=FOV, matrix=MATRIX, bandwidth_hz_per_px=2500, echo_spacing_ms=ESP,
                          fat_sat=False))["plan"]["realised"]["te_ms"]
    echo_shift = [polarity * GRADIENT * te * 1e-3 * FOV / (1 - polarity * GRADIENT * t_fov) * ESP for polarity in (1, -1)]   # ms
    expected = (1 + GRADIENT * t_fov) / (1 - GRADIENT * t_fov) * np.exp(-(echo_shift[0] - echo_shift[1]) / t2s)
    C.check("brightness, compressed / stretched", up[2] / down[2], float(expected), 0.03, relative=True)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
        for ax, polarity in zip(axes, (1, -1)):
            ax.imshow(images[polarity], cmap="gray", extent=(-FOV / 2, FOV / 2, FOV / 2, -FOV / 2), vmin=0, vmax=images[1].max())
            ax.add_patch(plt.Circle((0, 0), RADIUS, fill=False, color="r", lw=0.8))
            ax.set(title=f"pe_polarity {polarity:+d}", xlabel="x (mm)", ylabel="y (mm), phase encoding", xlim=(-80, 80), ylim=(80, -80))
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
