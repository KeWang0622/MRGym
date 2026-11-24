"""07 Gibbs ringing: the overshoot at a sharp edge stays near 9 % of the step; only its width shrinks with the matrix.

A known world: a 120 mm square of white matter, perfect fields, no noise. Spin-echo scans at matrix 64, 128 and 256.
The profile along the phase-encode axis is pure truncation of k-space. For a continuous step the overshoot is
8.95 % of the step, the first maximum lies one pixel inside the edge and the ripple period is two pixels. Two
properties of this scanner lower the overshoot at small pixels: the object lives on a master grid of 0.4 mm, and the
edges of the shipped phantoms are 0.5 mm soft (half-width of a tanh).

    python examples/textbook/07_gibbs_ringing.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import anatomy, kspace
from mrgym.scanner import world as W

FOV, HALF, UP = 256.0, 60.0, 16


def square(y, x):
    return np.maximum(np.abs(y) - HALF, np.abs(x) - HALF)


def profile(engine, matrix):
    image = engine.scan(dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=FOV, matrix=matrix))["images"][0]
    column = np.abs(C.upsample(image, UP))[:, matrix * UP // 2]
    return (np.arange(len(column)) - len(column) // 2) * FOV / matrix / UP, column


def ringing(pos, column, pixel):
    """Overshoot in percent of the step, distance of the first maximum from the edge and ripple period, in pixels."""
    plateau = column[np.abs(pos) < 40].mean()
    first = int(np.argmax(np.where((pos > -HALF) & (pos < -HALF + 2 * pixel), column, 0)))
    second = int(np.argmax(np.where((pos > pos[first] + 0.5 * pixel) & (pos < pos[first] + 3.5 * pixel), column, 0)))
    return 100 * (column.max() / plateau - 1), (pos[first] + HALF) / pixel, (pos[second] - pos[first]) / pixel


def main():
    args = C.parse_args(__doc__)
    sharp = kspace.Engine(W.flat_world(W.shapes_phantom([("wm", square)], edge=0.02)))
    soft = kspace.Engine(W.flat_world(W.shapes_phantom([("wm", square)], edge=anatomy.EDGE_MM)))
    rows, profiles = {}, {}
    for matrix in (64, 128, 256):
        pixel = FOV / matrix
        profiles[matrix] = profile(sharp, matrix)
        over, first, period = ringing(*profiles[matrix], pixel)
        over_soft = ringing(*profile(soft, matrix), pixel)[0]
        rows[matrix] = (over, first, period, over_soft)
        print(f"matrix {matrix}, pixel {pixel:.0f} mm: overshoot {over:.2f} % | first maximum {first:.2f} px = {first * pixel:.1f} mm inside the edge | "
              f"ripple period {period:.2f} px | overshoot with a 0.5 mm soft edge {over_soft:.2f} %")
    C.check("overshoot at 4 mm pixels", rows[64][0], 8.95, 0.3, "%")
    C.check("largest |first maximum - 1 pixel|", max(abs(r[1] - 1) for r in rows.values()), 0.0, 0.1, "px")
    C.check("largest |ripple period - 2 pixels|", max(abs(r[2] - 2) for r in rows.values()), 0.0, 0.1, "px")
    assert rows[64][0] > rows[128][0] > rows[256][0] > 7.0, "the master grid lowers the overshoot slowly as pixels shrink"
    assert rows[64][3] > rows[128][3] > rows[256][3] and rows[256][3] < 3.0, "a soft edge stops ringing once pixels are as small as the edge"

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for matrix, (pos, column) in profiles.items():
            ax.plot(pos, column / column[np.abs(pos) < 40].mean(), lw=1, label=f"matrix {matrix}")
        ax.axhline(1.0895, color="k", ls=":", lw=0.8)
        ax.set(xlim=(-75, -35), ylim=(-0.05, 1.15), xlabel="position along phase encoding (mm)", ylabel="signal / plateau",
               title="sharp edge at -60 mm; dotted: 8.95 % overshoot")
        ax.legend(fontsize=8)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
