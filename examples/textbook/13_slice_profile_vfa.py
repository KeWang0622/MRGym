"""13 Slice profile: why flip-angle methods read wrong on a 2D slice.

A slice-selective pulse does not give a slice one flip angle. The angle falls from its nominal value at the slice
centre to zero outside, and a 2D image is the integral of the signal over that profile. Two consequences are shown
on a vial of the QA phantom, with the excitation the console plays (``rf_pulse``) and with hard pulses and 3D
encoding (``excitation="nonselective"``):

1. Two-angle variable-flip-angle T1 (DESPOT1: S / sin a against S / tan a is a line of slope exp(-TR / T1)) with the
   nominal angles in the fit. Published for uncorrected 2D acquisitions: 40 to 80 % underestimation of T1 (Svedin &
   Parker 2017), (-63 +- 1.5) % (Dieringer 2014). Asserted here: inside -35 to -80 % for time-bandwidth products
   up to 4, and within 3 % of the truth in 3D with the transmit factor known.
2. The flip angle of maximum signal moves from the Ernst angle to a much larger nominal angle on a 2D slice.

The first part needs no scan: ``kspace.slice_signal`` is the function the scanner serves a voxel from (ideal
spoiling here, to isolate the profile). The second part scans.

    python examples/textbook/13_slice_profile_vfa.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import kspace
from mrgym.scanner import signal as S
from mrgym.scanner import world as W

TR, T1S, PAIRS = 20.0, (300.0, 1000.0, 2500.0), ((4, 20), (3, 17), (6, 35))


def two_angle_t1(s1, s2, a1, a2, tr):
    y1, y2, x1, x2 = s1 / np.sin(a1), s2 / np.sin(a2), s1 / np.tan(a1), s2 / np.tan(a2)
    return -tr / np.log((y2 - y1) / (x2 - x1))


def bias_table():
    """{time-bandwidth product: (smallest, largest) relative T1 error} of the two-angle fit with nominal angles."""
    out = {}
    for tbw in (2.0, 4.0, 8.0):
        errs = []
        for t1 in T1S:
            for pair in PAIRS:
                a = np.deg2rad(np.array(pair, float))
                s = np.abs(kspace.slice_signal(dict(sequence="spgr", tr_ms=TR), dict(t1_ms=t1), a, tbw, tier=1)[0])
                errs.append(two_angle_t1(s[0], s[1], a[0], a[1], TR) / t1 - 1)
        out[tbw] = (min(errs), max(errs))
    return out


def main():
    args = C.parse_args(__doc__)
    table = bias_table()
    print(f"two-angle T1 on a 2D slice with the nominal angles (TR {TR:g} ms, T1 300 to 2500 ms, ideal spoiling): "
          + "; ".join(f"time-bandwidth {tbw:g}: {100 * hi:+.0f} to {100 * lo:+.0f} %" for tbw, (lo, hi) in table.items()))
    for tbw in (2.0, 4.0):
        assert -0.80 <= table[tbw][0] and table[tbw][1] <= -0.35, "the published 2D bias is 40 to 80 % (Svedin & Parker 2017)"
    assert table[8.0][1] > table[4.0][1] > table[2.0][1], "a sharper profile is biased less"

    # the same fit on scans: 2D with the console's pulse, and 3D with hard pulses
    env = C.session("qa", seed=1)
    world = C.hidden_world(env)
    rois, truth = C.water_regions(world), W.truth(world)
    r = max(rois, key=lambda q: truth[q["id"]]["t1_ms"])                # the vial with the longest T1: the largest effect
    t = truth[r["id"]]
    worst3d, sweep = 0.0, {}
    for mode in ("selective", "nonselective"):
        sig = {f: C.column(C.scan(env, [r], sequence="spgr", tr_ms=TR, flip_deg=float(f), matrix=64, averages=4, spoiler_mt_ms_per_m=340.0,
                                  excitation=mode)["roi_readouts"], "complex_mean_magnitude")[0, 0] for f in (2, 4, 6, 9, 13, 20, 28, 38, 50, 65)}
        sweep[mode] = sig
        est = [two_angle_t1(sig[4], sig[20], np.deg2rad(4.0) * b1, np.deg2rad(20.0) * b1, TR) for b1 in (1.0, t["b1"])]
        if mode == "nonselective":
            worst3d = abs(est[1] / t["t1_ms"] - 1)
        print(f"{r['id']} (T1 {t['t1_ms']:.0f} ms, B1 {t['b1']:.3f}), {'2D, console pulse' if mode == 'selective' else '3D, hard pulse'}, scans at 4 and 20 deg: "
              f"T1 {est[0]:.0f} ms ({100 * (est[0] / t['t1_ms'] - 1):+.1f} %) with the nominal angles, {est[1]:.0f} ms ({100 * (est[1] / t['t1_ms'] - 1):+.1f} %) with angles x B1")
    C.check("3D, true angles: |T1 error|", worst3d, 0.0, 0.03)
    peak = {m: max(s, key=s.get) for m, s in sweep.items()}
    ernst = np.degrees(np.arccos(np.exp(-TR / t["t1_ms"]))) / t["b1"]
    print(f"flip angle of maximum signal on the sweep: 3D {peak['nonselective']} deg (Ernst angle / B1 {ernst:.1f} deg), 2D {peak['selective']} deg")
    assert peak["selective"] > peak["nonselective"], "on a 2D slice the signal peaks above the Ernst angle"
    assert abs(S.spgr(np.deg2rad(peak["nonselective"]) * t["b1"], TR, t["t1_ms"]) / S.spgr(np.deg2rad(ernst) * t["b1"], TR, t["t1_ms"]) - 1) < 0.1

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, s in sweep.items():
            f = np.array(sorted(s))
            ax.plot(f, np.array([s[k] for k in f]) / max(s.values()), "o-", lw=1, ms=3, label="2D, selective pulse" if mode == "selective" else "3D, hard pulse")
        ax.axvline(ernst, color="k", ls=":", lw=0.8)
        ax.set(xlabel="nominal flip angle (deg)", ylabel="signal / maximum", title=f"spgr TR {TR:g} ms, T1 {t['t1_ms']:.0f} ms; dotted: Ernst angle / B1")
        ax.legend(fontsize=8)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
