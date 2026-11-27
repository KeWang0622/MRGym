"""09 bSSFP banding: dark bands where the off-resonance is (m + 1/2) / TR, so their spacing shrinks as TR grows.

A known world: a wide slab of white matter in a field that rises by 2 Hz per mm along x, no noise. ``bssfp`` scans
at TR 5, 8 and 10 ms. With the default RF phase cycle of 180 degrees the pass band is centred on resonance and the
signal collapses at off-resonance (m + 1/2) / TR. In a linear field the bands are 1 / (TR x gradient) apart. A phase
cycle of 0 degrees moves every band by half a spacing, so one sits on resonance; in general the bands sit at
(m + phase cycle / 360 degrees) / TR, so a phase cycle of 90 degrees puts one at +1 / (4 TR).

    python examples/textbook/09_bssfp_bands.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import kspace
from mrgym.scanner import world as W

GRADIENT, FOV, MATRIX, BANDWIDTH = 2.0, 256.0, 256, 1000.0        # Hz / mm; 1 mm pixels


def slab(y, x):
    return np.maximum(np.abs(y) - 40.0, np.abs(x) - 110.0)


def bands(engine, tr_ms, phase_cycle_deg=180.0):
    """Positions (mm) of the dark bands along x within |x| < 100 mm, and the profile."""
    image = np.abs(engine.scan(dict(sequence="bssfp", tr_ms=tr_ms, flip_deg=40, fov_mm=FOV, matrix=MATRIX,
                                    bandwidth_hz_per_px=BANDWIDTH, phase_cycle_deg=phase_cycle_deg))["images"][0])
    prof = image[MATRIX // 2 - 10:MATRIX // 2 + 10].mean(0)
    x = (np.arange(MATRIX) - MATRIX // 2) * FOV / MATRIX
    inside = np.abs(x) < 100
    low = 0.5 * np.median(prof[inside])
    dips = [i for i in range(1, MATRIX - 1) if inside[i] and prof[i] < prof[i - 1] and prof[i] <= prof[i + 1] and prof[i] < low]
    return np.array([x[i] for i in dips]), x, prof


def expected(tr_ms, phase_cycle_deg=180.0):
    """Null positions: off-resonance (m + cycle / 360) / TR, mapped through the field and the readout shift."""
    f = (np.arange(-8, 9) + phase_cycle_deg / 360.0) * 1e3 / tr_ms
    x = f / GRADIENT - f / BANDWIDTH * (FOV / MATRIX)               # readout displacement: -f / bandwidth pixels
    return x[np.abs(x) < 100]


def main():
    args = C.parse_args(__doc__)
    phantom = W.shapes_phantom([("wm", slab)])
    world = W.flat_world(phantom)
    n, d = phantom["n"], phantom["extent_mm"] / phantom["n"]
    world["fields"]["b0_hz"] = GRADIENT * np.broadcast_to((np.arange(n) - n // 2) * d, (n, n)).copy()
    engine = kspace.Engine(world)
    worst, profiles = 0.0, {}
    for tr in (5.0, 8.0, 10.0):
        found, x, prof = bands(engine, tr)
        profiles[tr] = (x, prof)
        want = expected(tr)
        assert len(found) == len(want), f"TR {tr}: {len(found)} bands found, {len(want)} expected"
        spacing = float(np.diff(found).mean())
        print(f"TR {tr:.0f} ms: bands at {', '.join(f'{v:+.0f}' for v in found)} mm | spacing {spacing:.1f} mm "
              f"(1 / (TR x gradient) = {1e3 / tr / GRADIENT:.1f} mm)")
        worst = max(worst, float(np.abs(found - want).max()))
    C.check("largest |band position - position of off-resonance (m + 1/2) / TR|", worst, 0.0, 1.0, "mm")
    found, _, _ = bands(engine, 5.0, phase_cycle_deg=0.0)
    print(f"TR 5 ms, phase cycle 0 deg: bands at {', '.join(f'{v:+.0f}' for v in found)} mm")
    C.check("phase cycle 0 deg: band nearest to resonance", float(found[np.argmin(np.abs(found))]), 0.0, 1.0, "mm")
    found, _, _ = bands(engine, 5.0, phase_cycle_deg=90.0)
    want = expected(5.0, 90.0)
    print(f"TR 5 ms, phase cycle 90 deg: bands at {', '.join(f'{v:+.0f}' for v in found)} mm")
    C.check("phase cycle 90 deg: band nearest to resonance, at +1 / (4 TR) = +50 Hz", float(found[np.argmin(np.abs(found))]),
            float(want[np.argmin(np.abs(want))]), 1.0, "mm")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for tr, (x, prof) in profiles.items():
            ax.plot(x, prof, lw=1, label=f"TR {tr:.0f} ms")
        ax.set(xlim=(-100, 100), xlabel="x (mm); off-resonance = 2 Hz/mm x x", ylabel="signal", title="bssfp, phase cycle 180 deg")
        ax.legend(fontsize=8)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
