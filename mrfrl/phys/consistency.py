"""Tier-consistency suite: where the tiers must agree, and where they must differ.

    python -m mrfrl.phys.consistency            prints the measured numbers as JSON and exits 1 on any failed bound

Agreement (bounds asserted):
  A1  closed forms (spgr, afi, corrected ir incl. inversion efficiency) vs an independent matrix Bloch steady state
  A2  closed forms vs brute-force repetition from equilibrium
  A3  EPG steady state vs time-stepping EPG and vs an isochromat ensemble (spgr and afi patterns); with diffusion
      vs time-stepping EPG only (same b-factor formula in another frame)
  A4  tier 2 -> tier 1 when T2 << TR, when TR >> T2, and when diffusion spoiling is complete
  A5  the engine's B1-node interpolation vs direct evaluation
  A6  echo factor: T2* from T2 and T2', sign of the off-resonance phase
Difference (bounds asserted, so nobody "fixes" tier 2 to match the textbook):
  D1  spgr tier 2 / tier 1 over TR x flip for tissue-like classes; must exceed 3 % somewhere at TR <= 6 ms and stay
      within 0.5 % at flip <= 5 deg
  D2  the draft inversion-recovery formula (Mz = 0 after the readout) vs the corrected one at B1 != 1
"""
import itertools
import json
import sys
import time

import numpy as np

from . import bloch, closed, engine, epg, world as W

D = np.deg2rad


def _grid():
    t1 = np.array([250.0, 400, 700, 1100, 1800, 2800])
    b1 = np.round(np.arange(0.6, 1.41, 0.1), 2)
    g = np.array(list(itertools.product(t1, b1)))
    return g[:, 0].copy(), g[:, 1].copy()


def closed_vs_matrix():
    t1, b1 = _grid()
    worst = dict(spgr=0.0, spgr_brute=0.0, afi=0.0, afi_brute=0.0, ir=0.0, ir_brute=0.0, n=0)
    for flip, tr in itertools.product([1, 3, 5, 10, 20, 40, 60, 90], [4, 8, 15, 30, 100, 500, 3000, 15000]):
        a = b1 * D(flip)
        c = closed.spgr(a, tr, t1)
        worst["spgr"] = max(worst["spgr"], np.abs(c - bloch.spgr_ideal(a, tr, t1)).max())
        worst["n"] += c.size
        if tr >= 15 and flip >= 5:
            worst["spgr_brute"] = max(worst["spgr_brute"], np.abs(c - bloch.spgr_ideal(a, tr, t1, n_brute=6000)).max())
    for flip, tr1, n in itertools.product([10, 20, 40, 60, 80, 90], [4, 10, 20, 50, 150], [2, 3, 5, 10]):
        a = b1 * D(flip)
        c = np.array(closed.afi(a, tr1, n * tr1, t1))
        worst["afi"] = max(worst["afi"], np.abs(c - np.array(bloch.afi_ideal(a, tr1, n * tr1, t1))).max())
        worst["n"] += c.size
        if tr1 >= 10:
            worst["afi_brute"] = max(worst["afi_brute"],
                                     np.abs(c - np.array(bloch.afi_ideal(a, tr1, n * tr1, t1, n_brute=4000))).max())
    for ti, tr, k in itertools.product([20, 50, 100, 300, 700, 1500, 3000, 10000], [500, 1500, 3000, 6000, 15000],
                                       [1.0, 0.95, 0.8]):
        if ti + 2 >= tr:
            continue
        a = b1 * np.pi / 2
        c = closed.ir(a, ti, tr, t1, k)
        worst["ir"] = max(worst["ir"], np.abs(c - bloch.ir_ideal(a, ti, tr, t1, k)).max())
        worst["ir_brute"] = max(worst["ir_brute"], np.abs(c - bloch.ir_ideal(a, ti, tr, t1, k, n_brute=400)).max())
        worst["n"] += c.size
    return {k: float(v) for k, v in worst.items()}


def epg_vs_independent():
    out = dict(max_vs_stepping=0.0, max_vs_isochromats=0.0, cases=0)
    cases = [((1,), 20, 10, 1000, 80, 117), ((1,), 45, 5, 1000, 80, 50), ((1,), 30, 8, 600, 40, 117),
             ((1,), 25, 12, 1500, 200, 150), ((1,), 12, 6, 300, 40, 117), ((1, 1, 0, 0, 0, 0), 60, 20, 1000, 80, 129.3),
             ((1, 1, 0, 0), 45, 10, 600, 60, 117), ((1, 1, 0, 0, 0), 70, 15, 1500, 150, 39)]
    for pattern, flip, tr, t1, t2, phi in cases:
        s = epg.steady_state(D(flip), tr, t1, t2, phi, pattern)
        out["max_vs_stepping"] = max(out["max_vs_stepping"],
                                     np.abs(s - bloch.rf_spoiled_epg_stepping(D(flip), tr, t1, t2, phi, pattern)).max())
        out["max_vs_isochromats"] = max(out["max_vs_isochromats"], np.abs(
            s - bloch.rf_spoiled_isochromats(D(flip), tr, t1, t2, phi, pattern, n_iso=1024)).max())
        out["cases"] += 1
    out["max_vs_stepping_with_diffusion"] = 0.0
    for pattern, flip, tr, t1, t2, phi, bd in (((1,), 30, 6, 1000, 80, 117, 0.003), ((1,), 20, 20, 1500, 200, 117, 0.15),
                                               ((1, 1, 0, 0, 0, 0), 70, 30, 1200, 150, 45, 0.4)):
        s = epg.steady_state(D(flip), tr, t1, t2, phi, pattern, bd)
        ref = bloch.rf_spoiled_epg_stepping(D(flip), tr, t1, t2, phi, pattern, bd=bd)
        out["max_vs_stepping_with_diffusion"] = max(out["max_vs_stepping_with_diffusion"], np.abs(s - ref).max())
    return {k: float(v) for k, v in out.items()}


def tier2_limits():
    """Relative difference tier 2 vs tier 1 where they must coincide."""
    out = {}
    rel = lambda s, c: float(np.max(np.abs(np.asarray(s) - np.asarray(c)) / np.abs(c)))   # noqa: E731
    out["t2_much_less_than_tr"] = max(
        rel(epg.steady_state(D(f), tr, t1, 0.05)[0], closed.spgr(D(f), tr, t1))
        for f, tr, t1 in itertools.product([5, 30, 80], [5, 20], [300, 2500]))
    out["tr_much_longer_than_t2"] = max(
        rel(epg.steady_state(D(f), 600, t1, 40)[0], closed.spgr(D(f), 600, t1)) for f in (5, 30, 80) for t1 in (300, 2500))
    out["afi_t2_much_less_than_tr"] = max(
        rel(epg.steady_state(D(f), 20, t1, 0.05, 129.3, (1, 1) + (0,) * (n - 1)), closed.afi(D(f), 20, 20 * n, t1))
        for f in (30, 60, 90) for t1 in (300, 2500) for n in (2, 5))
    out["complete_diffusion_spoiling"] = max(
        rel(epg.steady_state(D(f), 5, 1000, 80, bd=200.0)[0], closed.spgr(D(f), 5, 1000)) for f in (5, 30, 80))
    return out


def node_interpolation():
    """Engine with its default B1 nodes vs 24 nodes per class, on a strong-B1 world."""
    w = W.make_world("5b" * 16, dict(b1_shape_amp=(0.2, 0.24)))
    worst = 0.0
    for prot in (dict(sequence="spgr", matrix=64, te_ms=2.5, tr_ms=6.0, flip_deg=30.0, spoiler_mt_ms_per_m=20.0,
                      rf_phase_increment_deg=117.0),
                 dict(sequence="afi", matrix=64, te_ms=2.5, tr_ms=20.0, tr_ratio=5, tr2_ms=100.0, flip_deg=60.0,
                      spoiler_mt_ms_per_m=300.0, rf_phase_increment_deg=129.3),
                 dict(sequence="ir", matrix=64, te_ms=2.5, tr_ms=2000.0, ti_ms=400.0, readout_flip_deg=90.0)):
        a, b = engine.Engine(w).clean(prot), engine.Engine(w, nodes=24).clean(prot)
        worst = max(worst, float(np.abs(a - b).max() / np.abs(b).max()))
    return dict(max_rel_image_difference=worst)


def echo_factor():
    te = np.array([2.0, 7.0, 12.0])
    e = closed.echo(te, 80.0, 120.0, b0_hz=25.0)
    return dict(t2star_ms=float(closed.t2star_ms(80.0, 120.0)),
                decay_error=float(np.abs(np.abs(e) - np.exp(-te / 48.0)).max()),
                phase_slope_hz=float(np.polyfit(te * 1e-3, np.unwrap(np.angle(e)), 1)[0] / (2 * np.pi)))


def gap_table():
    """Measured tier 2 / tier 1 magnitude ratio of spgr (117 deg increment). Rows TR, columns flip."""
    flips, trs = [3, 5, 10, 20, 30, 45, 60], [4, 6, 10, 15, 20, 30, 50]
    tissues = dict(t1_1000_t2_80=(1000, 80), t1_1500_t2_200=(1500, 200), t1_300_t2_40=(300, 40), t1_2500_t2_150=(2500, 150))
    table = {}
    for name, (t1, t2) in tissues.items():
        for label, area in (("spoiler_20", 20.0), ("spoiler_100", 100.0), ("spoiler_300", 300.0)):
            table[f"{name}/adc_1.0/{label}"] = [
                [round(float(abs(epg.steady_state(D(f), tr, t1, t2, 117.0, (1,), engine.diffusion_bd(1.0, area, tr))[0])
                             / closed.spgr(D(f), tr, t1)), 4) for f in flips] for tr in trs]
    vfa = {}
    for tr, area in ((5, 20.0), (10, 100.0), (15, 100.0), (25, 250.0)):
        for name, (t1, t2) in tissues.items():
            a = D(np.array([4.0, 18.0]))
            s = np.abs(epg.spgr(a, tr, t1, t2, 117.0, engine.diffusion_bd(1.0, area, tr)))
            y, x = s / np.sin(a), s / np.tan(a)                   # DESPOT1 line: y = E1 x + M0 (1 - E1)
            vfa[f"tr_{tr}_spoiler_{int(area)}/{name}"] = round(100 * (-tr / np.log((y[1] - y[0]) / (x[1] - x[0])) / t1 - 1), 2)
    afi = {}
    for n, phi, tr1, area in ((5, 117.0, 20, 300.0), (5, 129.3, 20, 300.0), (5, 129.3, 30, 450.0), (8, 47.0, 20, 300.0),
                              (8, 47.0, 30, 450.0)):
        for name, (t1, t2) in tissues.items():
            for adc in (0.6, 2.2):
                s1, s2 = epg.afi(D(60.0), tr1, n, t1, t2, phi, engine.diffusion_bd(adc, area, tr1))
                r = abs(s2) / abs(s1)
                b1 = np.degrees(np.arccos(np.clip((r * n - 1) / (n - r), -1, 1))) / 60.0
                i1, i2 = closed.afi(D(60.0), tr1, n * tr1, t1)
                b1_ideal = np.degrees(np.arccos((i2 / i1 * n - 1) / (n - i2 / i1))) / 60.0
                afi[f"n_{n}_phi_{phi}_tr1_{tr1}_spoiler_{int(area)}/{name}/adc_{adc}"] = round(100 * (b1 / b1_ideal - 1), 2)
    return dict(flips_deg=flips, trs_ms=trs, spgr_ratio=table, two_angle_vfa_t1_bias_percent=vfa,
                afi_b1_bias_percent_vs_ideal_spoiling=afi)


def must_differ():
    r_short = abs(epg.steady_state(D(30), 5, 1500, 200)[0]) / closed.spgr(D(30), 5, 1500)
    r_low = max(abs(abs(epg.steady_state(D(f), tr, 1000, 80)[0]) / closed.spgr(D(f), tr, 1000) - 1)
                for f in (3, 5) for tr in (5, 10, 20))
    t1, b1, tr, ti = 2800.0, 1.2, 1500.0, 400.0
    a = b1 * np.pi / 2
    draft = np.sin(a) * (1 - 2 * np.exp(-ti / t1) + np.exp(-tr / t1))
    return dict(spgr_tr5_flip30_t1_1500_t2_200=float(r_short), spgr_low_flip_max_dev=float(r_low),
                ir_draft_over_corrected_b1_1p2=float(draft / closed.ir(a, ti, tr, t1)))


BOUNDS = [
    ("closed_vs_matrix.spgr", "<=", 1e-12), ("closed_vs_matrix.afi", "<=", 1e-12), ("closed_vs_matrix.ir", "<=", 1e-12),
    ("closed_vs_matrix.spgr_brute", "<=", 1e-10), ("closed_vs_matrix.afi_brute", "<=", 1e-10),
    ("closed_vs_matrix.ir_brute", "<=", 1e-10),
    ("epg_vs_independent.max_vs_stepping", "<=", 1e-9), ("epg_vs_independent.max_vs_isochromats", "<=", 1e-9),
    ("epg_vs_independent.max_vs_stepping_with_diffusion", "<=", 1e-9),
    ("tier2_limits.t2_much_less_than_tr", "<=", 1e-6), ("tier2_limits.tr_much_longer_than_t2", "<=", 1e-6),
    ("tier2_limits.afi_t2_much_less_than_tr", "<=", 1e-6), ("tier2_limits.complete_diffusion_spoiling", "<=", 1e-6),
    ("node_interpolation.max_rel_image_difference", "<=", 1e-4),
    ("echo_factor.decay_error", "<=", 1e-12),
    ("must_differ.spgr_low_flip_max_dev", "<=", 5e-3),
]


def run(with_table=True):
    t0 = time.time()
    res = dict(closed_vs_matrix=closed_vs_matrix(), epg_vs_independent=epg_vs_independent(), tier2_limits=tier2_limits(),
               node_interpolation=node_interpolation(), echo_factor=echo_factor(), must_differ=must_differ())
    failed = []
    for path, _, bound in BOUNDS:
        sec, key = path.split(".")
        if not res[sec][key] <= bound:
            failed.append(f"{path} = {res[sec][key]:.3g} > {bound:g}")
    md = res["must_differ"]
    if not abs(md["spgr_tr5_flip30_t1_1500_t2_200"] - 1) >= 0.03:
        failed.append("tier 2 does not differ from tier 1 at TR 5 ms / 30 deg")
    if not abs(md["ir_draft_over_corrected_b1_1p2"] - 1) >= 0.05:
        failed.append("corrected IR does not differ from the draft formula at B1 = 1.2")
    if not abs(res["echo_factor"]["phase_slope_hz"] - 25.0) < 1e-6:
        failed.append("off-resonance phase has the wrong sign or slope")
    if with_table:
        res["gap_table"] = gap_table()
    res["failed"], res["seconds"] = failed, round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=1))
    sys.exit(1 if r["failed"] else 0)
