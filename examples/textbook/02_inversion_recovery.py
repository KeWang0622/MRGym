"""02 Inversion recovery: the null point and T1 from a TI series.

Nine ``ir`` scans of the QA phantom at TR 15 s. The signal is signed; the sign is in the phase image, so each region
is projected on its phase at the longest TI. A three-parameter fit S = a - b exp(-TI / T1) gives T1 and the
inversion efficiency kappa = b / a - 1. For TR >> T1 the signal crosses zero at TI = T1 ln(1 + kappa), which is
T1 ln 2 for a perfect inversion. One more scan at that TI shows the nulled region.

    python examples/textbook/02_inversion_recovery.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import world as W

TR = 15000.0
TIS = np.array([50.0, 150, 300, 500, 800, 1200, 2000, 3500, 6000])


def fit_t1(ti, s):
    """Least squares of s = a - b exp(-ti / T1): linear in (a, b) on a fine grid of T1. Returns (T1, a, b)."""
    best = None
    for t1 in np.exp(np.linspace(np.log(100.0), np.log(6000.0), 6000)):
        basis = np.stack([np.ones_like(ti), -np.exp(-ti / t1)], 1)
        coef = np.linalg.lstsq(basis, s, rcond=None)[0]
        res = float(np.sum((basis @ coef - s) ** 2))
        if best is None or res < best[0]:
            best = (res, float(t1), float(coef[0]), float(coef[1]))
    return best[1:]


def main():
    args = C.parse_args(__doc__)
    env = C.session("qa", seed=1)
    world = C.hidden_world(env)
    rois, truth, kappa = C.water_regions(world), W.truth(world), world["system"]["inv_eff"]
    scans = [C.scan(env, rois, sequence="ir", tr_ms=TR, ti_ms=float(ti), matrix=64) for ti in TIS]
    mean = np.array([C.complex_mean(s["roi_readouts"])[:, 0] for s in scans]).T          # [region, TI]
    signed = (mean * np.exp(-1j * np.angle(mean[:, -1:]))).real                           # sign from the phase
    print("one scan:", scans[-1]["header"])
    err_t1 = err_null = err_kappa = 0.0
    fits = []
    for k, r in enumerate(rois):
        t1_true = truth[r["id"]]["t1_ms"]
        t1, a, b = fit_t1(TIS, signed[k])
        null, null_true = t1 * np.log(b / a), t1_true * np.log(1 + kappa)
        fits.append((t1, a, b))
        print(f"{r['id']}: T1 {t1:.0f} ms (true {t1_true:.0f}) | null point {null:.0f} ms (T1 ln(1 + kappa) = {null_true:.0f}, "
              f"T1 ln 2 = {t1_true * np.log(2):.0f}) | kappa {b / a - 1:.3f}")
        err_t1, err_null = max(err_t1, abs(t1 / t1_true - 1)), max(err_null, abs(null / null_true - 1))
        err_kappa = max(err_kappa, abs(b / a - 1 - kappa))
    C.check("largest relative T1 error", err_t1, 0.0, 0.01)
    C.check("largest relative error of the null point against T1 ln(1 + kappa)", err_null, 0.0, 0.01)
    assert err_kappa <= 0.01, f"inversion efficiency off by {err_kappa}"
    k = 0                                                                                 # null the first region
    ti_null = fits[k][0] * np.log(fits[k][2] / fits[k][1])
    at_null = C.complex_mean(C.scan(env, rois, sequence="ir", tr_ms=TR, ti_ms=float(ti_null), matrix=64)["roi_readouts"])[:, 0]
    C.check(f"{rois[k]['id']} scanned at its null point, signal / recovered signal", abs(at_null[k]) / fits[k][1], 0.0, 0.02)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        fine = np.linspace(0, TIS[-1], 400)
        for k, r in enumerate(rois):
            t1, a, b = fits[k]
            line, = ax.plot(TIS, signed[k], "o", ms=4, label=f"{r['id']}: T1 {t1:.0f} ms")
            ax.plot(fine, a - b * np.exp(-fine / t1), "-", color=line.get_color(), lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="TI (ms)", ylabel="signed region signal", title="ir, TR 15 s: signed signal and three-parameter fit")
        ax.legend(fontsize=7)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
