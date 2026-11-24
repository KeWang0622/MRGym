"""04 T2* and a field map from one multi-echo gradient-echo scan.

One 8-echo ``spgr`` scan of the QA phantom. The mean of the pixel magnitudes decays with exp(-TE / T2*),
1 / T2* = 1 / T2 + 1 / T2'. The phase of a region grows by 2 pi x (off-resonance) x TE, with a positive slope for
positive off-resonance, which gives a field map. The field also varies across a region. The complex mean
therefore loses signal faster than the magnitude mean: a T2* fitted to ``complex_mean_magnitude`` is too short.

    python examples/textbook/04_t2star_field_map.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import anatomy
from mrgym.scanner import world as W


def main():
    args = C.parse_args(__doc__)
    env = C.session("qa", seed=1)
    world = C.hidden_world(env)
    rois, truth = C.water_regions(world), W.truth(world)
    scan = C.scan(env, rois, sequence="spgr", tr_ms=80, flip_deg=25, matrix=128, n_echoes=8, echo_spacing_ms=6.0)
    te = np.array(scan["realised"]["echo_times_ms"])
    magnitude = C.column(scan["roi_readouts"], "mean_magnitude")                           # [region, echo]
    complex_magnitude = C.column(scan["roi_readouts"], "complex_mean_magnitude")
    phase = np.unwrap(C.column(scan["roi_readouts"], "complex_mean_phase_rad"), axis=1)
    y, x = anatomy.axes(world["n"], world["extent_mm"])
    field = world["fields"]["b0_hz"] + world["system"]["cf_offset_hz"]                     # what the spins see, Hz
    print(scan["header"])
    err_t2s = err_field = 0.0
    shortening = []
    for k, r in enumerate(rois):
        t2s_true = truth[r["id"]]["t2s_ms"]
        t2s = -1.0 / np.polyfit(te, np.log(magnitude[k]), 1)[0]
        t2s_complex = -1.0 / np.polyfit(te, np.log(complex_magnitude[k]), 1)[0]
        hz = np.polyfit(te * 1e-3, phase[k], 1)[0] / (2 * np.pi)
        inside = (y - r["y_mm"]) ** 2 + (x - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2
        print(f"{r['id']}: T2* {t2s:.1f} ms from the magnitude mean (true {t2s_true:.1f}), {t2s_complex:.1f} ms from the complex mean | "
              f"off-resonance {hz:+.2f} Hz (true mean {field[inside].mean():+.2f}, spread {np.ptp(field[inside]):.1f} Hz across the region)")
        err_t2s, err_field = max(err_t2s, abs(t2s / t2s_true - 1)), max(err_field, abs(hz - field[inside].mean()))
        shortening.append(1 - t2s_complex / t2s)
    C.check("largest relative T2* error, magnitude mean", err_t2s, 0.0, 0.02)
    C.check("largest field-map error", err_field, 0.0, 0.5, "Hz")
    print(f"complex-mean T2* is shorter than the magnitude-mean T2* by {100 * min(shortening):.1f} to {100 * max(shortening):.1f} %")
    assert min(shortening) > 0 and max(shortening) > 0.05, "the complex mean must decay faster in a field gradient"

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4))
        for k, r in enumerate(rois):
            line, = a0.semilogy(te, magnitude[k], "o-", ms=4, lw=1, label=f"{r['id']} magnitude mean")
            a0.semilogy(te, complex_magnitude[k], "x--", ms=4, lw=1, color=line.get_color())
            a1.plot(te, phase[k] - phase[k, 0], "o-", ms=4, lw=1, color=line.get_color(), label=r["id"])
        a0.set(xlabel="TE (ms)", ylabel="region signal", title="circles: magnitude mean; crosses: complex mean")
        a1.set(xlabel="TE (ms)", ylabel="phase - phase of echo 1 (rad)", title="slope = 2 pi x off-resonance")
        a0.legend(fontsize=7)
        a1.legend(fontsize=7)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
