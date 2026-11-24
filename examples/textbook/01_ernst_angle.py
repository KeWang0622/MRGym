"""01 Ernst angle: the flip angle of maximum spoiled gradient-echo signal, and what the scanner serves instead.

A flip-angle sweep of ``spgr`` on the QA phantom at TR 15 ms. With ideal spoiling the peak sits at
arccos(exp(-TR / T1)) / B1 (Ernst & Anderson 1966; B1 = actual / nominal flip angle). The scanner serves the
steady state with RF spoiling, finite T2 and diffusion (tier 2), so its peak is shifted. The same world imaged with
the ideal-spoiling engine (tier 1) gives the textbook peak.

    python examples/textbook/01_ernst_angle.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import console, epg, kspace, tissue
from mrgym.scanner import signal as S
from mrgym.scanner import world as W

TR, FLIPS = 15.0, np.arange(2.0, 26.1, 1.5)


def peak(flips, s):
    """Flip angle of the maximum: a cubic through the seven samples around the largest one."""
    i = int(np.clip(np.argmax(s), 3, len(s) - 4))
    fine = np.linspace(flips[i - 3], flips[i + 3], 6001)
    return float(fine[np.argmax(np.polyval(np.polyfit(flips[i - 3:i + 4], s[i - 3:i + 4], 3), fine))])


def tier_gap(exam):
    """Range of (served steady state) / (Ernst formula) over six tissues at 3 T, flip 5 to 60 deg, TR 8 to 30 ms and
    three spoiler moments, for the protocols the console accepts (matrix 64, 1000 Hz/px)."""
    tab, out = tissue.table(3.0), {}
    for moment in (100.0, 300.0, 600.0):
        r = []
        for tr in (8.0, 10.0, 15.0, 30.0):
            for f in (5, 10, 20, 30, 45, 60):
                if not console.plan(dict(sequence="spgr", tr_ms=tr, flip_deg=f, spoiler_mt_ms_per_m=moment, matrix=64,
                                         bandwidth_hz_per_px=1000), exam)["accept"]:
                    continue
                for c in (tab[k] for k in ("wm", "gm", "csf", "liver", "muscle", "blood")):
                    bd = kspace.diffusion_bd(c["adc_um2_ms"], moment, tr)
                    r.append(abs(epg.spgr(np.deg2rad(f), tr, c["t1_ms"], c["t2_ms"], 117.0, bd)) / S.spgr(np.deg2rad(f), tr, c["t1_ms"]))
        out[moment] = (float(min(r)), float(max(r)))
    return out


def main():
    args = C.parse_args(__doc__)
    env = C.session("qa", seed=1)
    world = C.hidden_world(env)
    rois, truth = C.water_regions(world), W.truth(world)
    served = np.array([C.column(C.scan(env, rois, sequence="spgr", tr_ms=TR, flip_deg=float(f), matrix=64)["roi_readouts"],
                                "complex_mean_magnitude")[:, 0] for f in FLIPS]).T
    ideal_engine = kspace.Engine(world, tier=1)
    ideal = np.array([C.column(C.engine_scan(ideal_engine, rois, sequence="spgr", tr_ms=TR, flip_deg=float(f), matrix=64)["roi_readouts"],
                               "complex_mean_magnitude")[:, 0] for f in FLIPS]).T
    worst_ideal = worst_served = 0.0
    for k, r in enumerate(rois):
        t, cls = truth[r["id"]], world["tissues"][next(q["cls"] for q in world["regions"] if q["id"] == r["id"])]
        ernst = np.degrees(np.arccos(np.exp(-TR / t["t1_ms"]))) / t["b1"]
        fine = np.deg2rad(np.linspace(0.7 * ernst * t["b1"], 1.4 * ernst * t["b1"], 141))
        model = np.abs(epg.spgr(fine, TR, cls["t1_ms"], cls["t2_ms"], 117.0, kspace.diffusion_bd(cls["adc_um2_ms"], 100.0, TR)))
        predicted = float(np.degrees(fine[np.argmax(model)])) / t["b1"]
        p_ideal, p_served = peak(FLIPS, ideal[k]), peak(FLIPS, served[k])
        print(f"{r['id']}: T1 {t['t1_ms']:.0f} ms, B1 {t['b1']:.3f} | Ernst / B1 {ernst:.2f} deg | ideal-spoiling engine {p_ideal:.2f} "
              f"| served {p_served:.2f} (steady-state model {predicted:.2f})")
        worst_ideal, worst_served = max(worst_ideal, abs(p_ideal - ernst)), max(worst_served, abs(p_served - predicted))
    C.check("largest |ideal-spoiling peak - Ernst angle / B1|", worst_ideal, 0.0, 0.15, "deg")
    C.check("largest |served peak - peak of the served steady-state model|", worst_served, 0.0, 0.4, "deg")
    gap = tier_gap(world["console"])
    print("served / Ernst, six tissues at 3 T, flip 5-60 deg, TR 8-30 ms, RF increment 117 deg: "
          + "; ".join(f"spoiler {m:.0f} mT*ms/m {lo:.3f} to {hi:.3f}" for m, (lo, hi) in gap.items()))
    assert min(v[0] for v in gap.values()) < 0.95 and max(v[1] for v in gap.values()) > 1.05, "the two tiers must differ"
    timing = console.plan(dict(sequence="spgr", tr_ms=TR, flip_deg=10, matrix=64), world["console"])["timing"]
    print(f"spoiler lobe at the default moment: {timing['spoiler_ms']:.2f} ms of a {TR:.0f} ms TR")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, r in enumerate(rois):
            line, = ax.plot(FLIPS, served[k], "o", ms=3, label=f"{r['id']} served")
            ax.plot(FLIPS, ideal[k], "-", color=line.get_color(), lw=1)
            t = truth[r["id"]]
            ax.axvline(np.degrees(np.arccos(np.exp(-TR / t["t1_ms"]))) / t["b1"], color=line.get_color(), ls=":", lw=1)
        ax.set(xlabel="nominal flip angle (deg)", ylabel="region signal", title=f"spgr, TR {TR:.0f} ms: points served, lines ideal spoiling, dotted Ernst / B1")
        ax.legend(fontsize=7)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
