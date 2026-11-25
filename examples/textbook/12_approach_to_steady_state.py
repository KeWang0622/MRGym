"""12 Approach to steady state: what the first sampled line would see after the console's dummy cycles.

The scanner charges dummy time (4 s for ``spgr``, ``afi``, ``bssfp``; one cycle for ``se`` and ``ir``) and then
serves every line at the periodic steady state. This script uses the separately written Bloch chain of
``mrgym.scanner.bloch`` (ideal spoiling, start from equilibrium) to compute the magnetisation a real scan would
have at its first sampled line and at the centre of k-space, and compares it with the periodic state.

Textbook result for spoiled gradient echo: the longitudinal magnetisation before pulse n approaches its steady
state m as  m + (1 - m) (E1 cos a)^n,  E1 = exp(-TR / T1), so the time constant is 1 / (1 / T1 - ln(cos a) / TR).
For a spin echo with exact 90 and 180 degree pulses one dummy cycle is enough; with a transmit error it is not.
The chain has one flip angle per voxel, so the dummy cycles are those of ``excitation="nonselective"``.

    python examples/textbook/12_approach_to_steady_state.py [--plot FILE.png]
"""
import _common as C
import numpy as np

from mrgym.scanner import anatomy, bloch, console, tissue

EXAM = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
TRS, FLIPS = (8.0, 15.0, 30.0), (1, 2, 3, 5, 10, 20, 30)


def main():
    args = C.parse_args(__doc__)
    table = tissue.table(3.0)
    names = ["wm", "gm", "blood", "QA vial, longest", "csf"]
    t1 = np.array([table["wm"]["t1_ms"], table["gm"]["t1_ms"], table["blood"]["t1_ms"], anatomy.QA_SPEC["vial_t1_ms"][1], table["csf"]["t1_ms"]])
    first, centre, where, err_formula, dummies = np.zeros(len(t1)), np.zeros(len(t1)), [None] * len(t1), 0.0, {}
    for tr in TRS:
        for flip in FLIPS:
            plan = console.plan(dict(sequence="spgr", tr_ms=tr, flip_deg=flip, matrix=128, bandwidth_hz_per_px=600, excitation="nonselective"), EXAM)
            assert plan["accept"], plan["violations"]
            n, lines = plan["timing"]["dummy_cycles"], plan["realised"]["lines"]
            dummies[tr] = n
            a = np.full(len(t1), np.deg2rad(flip))
            periodic = bloch.spgr_ideal(a, tr, t1)
            line0 = bloch.spgr_ideal(a, tr, t1, n_brute=n) / periodic - 1
            line_c = bloch.spgr_ideal(a, tr, t1, n_brute=n + lines // 2) / periodic - 1
            e1 = np.exp(-tr / t1)
            m = (1 - e1) / (1 - e1 * np.cos(a))
            err_formula = max(err_formula, float(np.abs(line0 - (1 / m - 1) * (e1 * np.cos(a)) ** n).max()))
            for i in np.flatnonzero(line0 > first):
                first[i], centre[i], where[i] = line0[i], line_c[i], (tr, flip)
    print("spgr, ideal spoiling, dummy cycles played by the console: " + ", ".join(f"{n} at TR {tr:.0f} ms" for tr, n in dummies.items()))
    for i, name in enumerate(names):
        print(f"{name} (T1 {t1[i]:.0f} ms): first line {100 * first[i]:+.2f} % above the periodic state, centre of a 128-line scan "
              f"{100 * centre[i]:+.2f} % (worst case: TR {where[i][0]:.0f} ms, {where[i][1]} deg)")
    C.check("largest |first line - m (1 + (1 / m - 1) (E1 cos a)^n)| / m", err_formula, 0.0, 1e-12)
    long_run = bloch.spgr_ideal(np.full(len(t1), np.deg2rad(5.0)), 15.0, t1, n_brute=20000) / bloch.spgr_ideal(np.full(len(t1), np.deg2rad(5.0)), 15.0, t1)
    assert np.abs(long_run - 1).max() < 1e-9, "after 300 s of pulses the chain must sit on the periodic state"
    assert first[0] < 0.001 and 0.005 < first[1] < 0.03 and 0.10 < first[-1] < 0.20, "sizes quoted in docs/physics/limits.md"

    # spin echo: one dummy cycle, white matter, transmit factor 0.8 / 1.0 / 1.2
    b1 = np.array([0.8, 1.0, 1.2])
    a, t1_wm, t2 = b1 * np.pi / 2, np.full(3, table["wm"]["t1_ms"]), np.full(3, table["wm"]["t2_ms"])
    cycle = [bloch.rf_x(a), bloch.spoil(3), bloch.free(7.5, t1_wm, t2), bloch.rf_x(2 * a), bloch.spoil(3), bloch.free(500.0 - 7.5, t1_wm, t2)]
    one, steady = bloch.brute(cycle, 1)[:, 2], bloch.periodic(cycle)[:, 2]
    carry = np.cos(a) * np.cos(2 * a) * np.exp(-500.0 / t1_wm)
    print("se, TR 500 ms, TE 15 ms, white matter, one dummy cycle, transmit factor 0.8 / 1.0 / 1.2: first line "
          + " / ".join(f"{100 * v:+.1f}" for v in one / steady - 1) + " % from the periodic state; each further cycle multiplies the error by "
          + " / ".join(f"{v:+.2f}" for v in carry))
    assert abs(one[1] / steady[1] - 1) < 1e-12, "with exact 90 and 180 degree pulses one dummy cycle reaches the periodic state"
    assert np.allclose(one - steady, carry * (1 - steady), atol=1e-12), "error after one cycle = carry-over factor x initial error"

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        tr, flip = 15.0, 5.0
        a = np.full(len(t1), np.deg2rad(flip))
        periodic = bloch.spgr_ideal(a, tr, t1)
        pulses = np.unique(np.round(np.geomspace(1, 2000, 60)).astype(int))
        curve = np.array([bloch.spgr_ideal(a, tr, t1, n_brute=int(k)) / periodic for k in pulses])
        for i, name in enumerate(names):
            ax.semilogx(pulses * tr / 1e3, curve[:, i], lw=1, label=f"{name}, T1 {t1[i]:.0f} ms")
        ax.axvline(dummies[tr] * tr / 1e3, color="k", ls=":", lw=0.8)
        ax.set(xlabel="time since the first pulse (s)", ylabel="signal / periodic state", title=f"spgr TR {tr:.0f} ms, {flip:.0f} deg; dotted: sampling starts")
        ax.legend(fontsize=7)
        C.save(fig, args.plot)


if __name__ == "__main__":
    main()
