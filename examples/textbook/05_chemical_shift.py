"""05 Chemical shift: fat moves along the readout by (fat frequency) / (bandwidth per pixel), and flips with polarity.

A known world: a disc of water at x = -50 mm and a disc of fat at x = +50 mm, perfect fields, no noise. Spin-echo
scans at four bandwidths, both readout polarities, both readout axes, 1.5 and 3 T. The main fat peak lies 3.4 ppm
below water, so with ``ro_polarity`` = +1 fat moves towards +readout by 3.4 ppm x gamma x B0 / bandwidth pixels,
which is the "WFS" figure in the header. The scanner images a six-peak fat spectrum; its centre of mass moves by
the amplitude-weighted mean shift, which is smaller. Water does not move.

    python examples/textbook/05_chemical_shift.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import kspace, tissue
from mrgym.scanner import signal as S
from mrgym.scanner import world as W

FOV, MATRIX, PIXEL = 256.0, 128, 2.0
BANDWIDTHS = (100.0, 200.0, 400.0, 800.0)


def disc(radius, cx):
    return lambda y, x: np.hypot(y, x - cx) - radius


def centre_of_mass_px(image, columns):
    """Centre of mass (y, x) in pixels from the image centre of the real part of ``image`` within ``columns``."""
    a = np.zeros(image.shape)
    a[:, columns] = image.real[:, columns]
    pos = np.arange(MATRIX) - MATRIX // 2
    return float((a.sum(1) * pos).sum() / a.sum()), float((a.sum(0) * pos).sum() / a.sum())


def main():
    args = C.parse_args(__doc__)
    fat_half, water_half = slice(MATRIX // 2, MATRIX), slice(0, MATRIX // 2)
    mean_ppm = sum(p * a for p, a in tissue.FAT_PEAKS) / sum(a for _, a in tissue.FAT_PEAKS)
    err_main = err_header = err_six = err_water = err_other = 0.0
    keep = {}
    for b0 in (1.5, 3.0):
        phantom = W.shapes_phantom([("wm", disc(30.0, -50.0)), ("subcut_fat", disc(30.0, 50.0))], b0_t=b0)
        f_fat = tissue.FAT_PPM * S.GAMMA_MHZ_PER_T * b0                                  # Hz, negative
        rows = []
        for model in ("single", "multipeak"):
            engine = kspace.Engine(W.flat_world(phantom), fat_model=model)
            for bw in BANDWIDTHS:
                for polarity, pe_dir in ((1, "y"), (-1, "y"), (1, "x")):
                    out = engine.scan(dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=FOV, matrix=MATRIX,
                                           bandwidth_hz_per_px=bw, ro_polarity=polarity, pe_dir=pe_dir))
                    image = out["images"][0]
                    fat, water = centre_of_mass_px(image, fat_half), centre_of_mass_px(image, water_half)
                    along, across = ((fat[1] - 50.0 / PIXEL, fat[0]) if pe_dir == "y" else (fat[0], fat[1] - 50.0 / PIXEL))
                    ppm = tissue.FAT_PPM if model == "single" else mean_ppm
                    expected = -polarity * ppm * S.GAMMA_MHZ_PER_T * b0 / bw
                    if model == "single":
                        err_main = max(err_main, abs(along - expected))
                        err_header = max(err_header, abs(out["plan"]["derived"]["water_fat_shift_px"] - abs(expected)))
                    else:
                        err_six = max(err_six, abs(along - expected))
                    err_other = max(err_other, abs(across))
                    err_water = max(err_water, abs(water[0]), abs(water[1] + 50.0 / PIXEL))
                    if (polarity, pe_dir) == (1, "y"):
                        rows.append(along)
                        keep[(b0, model, bw)] = image
        n = len(BANDWIDTHS)
        print(f"{b0} T, fat at {f_fat:.0f} Hz, bandwidth {' / '.join(f'{b:.0f}' for b in BANDWIDTHS)} Hz/px, polarity +1: main peak "
              + " / ".join(f"{v:+.2f}" for v in rows[:n]) + " px; six-peak centre of mass " + " / ".join(f"{v:+.2f}" for v in rows[n:]) + " px")
        assert all(v > 0 for v in rows), "fat must move towards +readout for positive polarity"
    C.check("main peak: largest |shift + polarity x f_fat / bandwidth|, 24 scans", err_main, 0.0, 0.03, "px")
    C.check("header WFS against the same formula", err_header, 0.0, 0.001, "px")
    C.check(f"six peaks: largest |centre of mass + polarity x ({mean_ppm:.2f} ppm) x gamma B0 / bandwidth|", err_six, 0.0, 0.05, "px")
    C.check("largest shift of fat along the phase-encode axis", err_other, 0.0, 0.01, "px")
    C.check("largest shift of water", err_water, 0.0, 0.01, "px")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(BANDWIDTHS), figsize=(3 * len(BANDWIDTHS), 3.2))
        for ax, bw in zip(axes, BANDWIDTHS):
            ax.imshow(np.abs(keep[(3.0, "multipeak", bw)]), cmap="gray", extent=(-FOV / 2, FOV / 2, FOV / 2, -FOV / 2))
            ax.axvline(50.0, color="r", lw=0.5)
            ax.set(title=f"{bw:.0f} Hz/px", xlabel="x (mm)")
        fig.suptitle("3 T, six-peak fat (right disc; red line = true centre), water (left disc)")
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
