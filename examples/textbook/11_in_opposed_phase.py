"""11 In-phase and opposed-phase echo times of water and fat at 1.5 and 3 T.

A known world: a disc in which every voxel is half liver, half fat; perfect fields, no noise. Gradient-echo
(``spgr``) scans with TE stepped by 0.1 ms. Fat precesses 3.4 ppm x gamma x B0 slower than water, so the two
signals cancel at TE = (k + 1/2) / |f_fat| and add at TE = k / |f_fat|: 2.30 and 4.61 ms at 1.5 T, half of that at
3 T. T2* decay pulls the maxima a little earlier. The scanner's default fat has six peaks; its minima sit close to,
not on, the single-peak times and are deeper.

    python examples/textbook/11_in_opposed_phase.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import console, kspace, tissue
from mrgym.scanner import signal as S
from mrgym.scanner import world as W

PROTOCOL = dict(sequence="spgr", tr_ms=150, flip_deg=60, fov_mm=256, matrix=32, bandwidth_hz_per_px=3000)
ROI = [dict(id="centre", y_mm=0.0, x_mm=0.0, radius_mm=20.0)]


def extrema(te, s, sign):
    """Echo times of the local minima (sign -1) or maxima (+1), refined by a parabola through three samples."""
    idx = [i for i in range(1, len(s) - 1) if sign * s[i] > sign * s[i - 1] and sign * s[i] > sign * s[i + 1]]
    out = []
    for i in idx:
        a, b, _ = np.polyfit(te[i - 1:i + 2], s[i - 1:i + 2], 2)
        out.append(-b / (2 * a))
    return np.array(out)


def main():
    args = C.parse_args(__doc__)
    err_opposed = err_in = 0.0
    curves = {}
    for b0 in (1.5, 3.0):
        phantom = W.shapes_phantom([("liver", lambda y, x: np.hypot(y, x) - 40.0), ("subcut_fat", lambda y, x: np.hypot(y, x) - 40.0)],
                                   b0_t=b0, n=160)
        phantom["weights"][:] = 0.5 * phantom["weights"].sum(0)[None]              # every voxel: half water, half fat
        world = W.flat_world(phantom)
        period = 1e3 / (abs(tissue.FAT_PPM) * S.GAMMA_MHZ_PER_T * b0)               # ms
        te_min = console.plan(PROTOCOL, world["console"])["timing"]["te_min_ms"]
        te = np.arange(np.ceil(te_min * 10) / 10, 3.3 * period, 0.1)
        found = {}
        for model in ("single", "multipeak"):
            engine = kspace.Engine(world, tier=1, fat_model=model)
            s = np.array([C.engine_scan(engine, ROI, **dict(PROTOCOL, te_ms=float(t)))["roi_readouts"][0]["mean_magnitude"][0] for t in te])
            found[model] = (extrema(te, s, -1), extrema(te, s, +1), s.min() / s.max())
            curves[(b0, model)] = (te, s)
        lo, hi, depth = found["single"]
        want_lo, want_hi = (np.arange(len(lo)) + 0.5) * period, (np.arange(len(hi)) + 1) * period
        if lo[0] > period:                                                           # first minimum below the shortest TE
            want_lo = want_lo + period
        print(f"{b0} T, one fat peak: opposed phase at {', '.join(f'{v:.2f}' for v in lo)} ms ((k + 1/2) / |f_fat| = "
              f"{', '.join(f'{v:.2f}' for v in want_lo)}); in phase at {', '.join(f'{v:.2f}' for v in hi)} ms (k / |f_fat| = "
              f"{', '.join(f'{v:.2f}' for v in want_hi)}); minimum / maximum {depth:.2f}")
        print(f"{b0} T, six fat peaks: opposed phase at {', '.join(f'{v:.2f}' for v in found['multipeak'][0])} ms; in phase at "
              f"{', '.join(f'{v:.2f}' for v in found['multipeak'][1])} ms; minimum / maximum {found['multipeak'][2]:.2f}")
        err_opposed = max(err_opposed, float(np.abs(lo - want_lo).max()))
        err_in = max(err_in, float(np.abs(hi - want_hi).max()))
        assert np.all(hi < want_hi), "T2* decay moves the in-phase maxima to shorter TE"
        assert np.abs(found["multipeak"][0] - want_lo).max() < 0.2 and found["multipeak"][2] < depth
    C.check("one fat peak: largest |opposed-phase TE - (k + 1/2) / |f_fat||", err_opposed, 0.0, 0.02, "ms")
    C.check("one fat peak: largest |in-phase TE - k / |f_fat||", err_in, 0.0, 0.1, "ms")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, b0 in zip(axes, (1.5, 3.0)):
            for model in ("single", "multipeak"):
                ax.plot(*curves[(b0, model)], lw=1, label="one fat peak" if model == "single" else "six fat peaks")
            ax.set(xlabel="TE (ms)", ylabel="signal of a 50 / 50 water-fat voxel", title=f"{b0} T")
            ax.legend(fontsize=8)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
