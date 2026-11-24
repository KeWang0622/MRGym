"""03 T2 from a spin-echo TE series.

Six ``se`` scans of the QA phantom at TR 10 s. A spin echo refocuses static dephasing, so the region signal decays
with exp(-TE / T2), not exp(-TE / T2*). A straight line through log(signal) against TE gives T2. The same regions
have a shorter T2* (example 04). The scanner has no fast spin echo: every phase-encode line costs one TR.

    python examples/textbook/03_t2_spin_echo.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import world as W

TR = 10000.0
TES = np.array([15.0, 30, 50, 80, 120, 180])


def main():
    args = C.parse_args(__doc__)
    env = C.session("qa", seed=1)
    world = C.hidden_world(env)
    rois, truth = C.water_regions(world), W.truth(world)
    scans = [C.scan(env, rois, sequence="se", tr_ms=TR, te_ms=float(te), matrix=64) for te in TES]
    signal = np.array([C.column(s["roi_readouts"], "complex_mean_magnitude")[:, 0] for s in scans]).T      # [region, TE]
    print("one scan:", scans[-1]["header"])
    worst, fits = 0.0, []
    for k, r in enumerate(rois):
        t = truth[r["id"]]
        slope, intercept = np.polyfit(TES, np.log(signal[k]), 1)
        t2 = -1.0 / slope
        fits.append((t2, np.exp(intercept)))
        print(f"{r['id']}: T2 {t2:.1f} ms (true {t['t2_ms']:.1f}; T2* of the same vial {t['t2s_ms']:.1f})")
        worst = max(worst, abs(t2 / t["t2_ms"] - 1))
    C.check("largest relative T2 error", worst, 0.0, 0.02)
    total = sum(s["cost_s"] for s in scans)
    print(f"scan time of the series: {total / 60:.0f} min for six 64 x 64 images (one line per TR)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        fine = np.linspace(0, TES[-1], 100)
        for k, r in enumerate(rois):
            line, = ax.semilogy(TES, signal[k], "o", ms=4, label=f"{r['id']}: T2 {fits[k][0]:.0f} ms")
            ax.semilogy(fine, fits[k][1] * np.exp(-fine / fits[k][0]), "-", color=line.get_color(), lw=1)
        ax.set(xlabel="TE (ms)", ylabel="region signal", title="se, TR 10 s: mono-exponential decay with T2")
        ax.legend(fontsize=7)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
