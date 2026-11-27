"""The external validation ladder: every rung that can be run here, as one command and one published table.

    python tools/external/run_all.py            # run, print the table
    python tools/external/run_all.py --check    # run; exit 1 if a row fails or the stored results are stale
    python tools/external/run_all.py --write    # run, store docs/physics/external/<rung>.json and docs/physics/external.md

Rungs (what each one can and cannot show is in the table it writes)
  L1  the scanner's equations solved by other code: ``independent_bloch.py`` (numpy only, written from the Bloch
      equations, imports nothing from the package) against the engine's signal models
  L2  another group's code: sycomore (C++ extended phase graphs). Its values for 17 cases are stored
      (``docs/physics/external/sycomore_reference.json``, written by ``compare_sycomore.py --write``), so the rung
      runs without sycomore; where sycomore is installed the stored values are recomputed and compared as well
  L3  published operating points and published biases, and closed forms from the literature
  L4  a standard object with published reference values: on its own page, docs/physics/measured.md (rungs M0, M1, M9)
  L5  a scanner: public phantom measurements on real scanners, on the same page (``tools/measured/run_all.py``)

A stored result is stale when its engine id is not the engine's, or when the rows it lists are not the rows this
script produces.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import independent_bloch as IB  # noqa: E402

from mrgym.scanner import epg, kspace, rf  # noqa: E402
from mrgym.scanner import signal as S  # noqa: E402

D = np.deg2rad
AUTHORS, OTHERS, PAPER, FORMULA = "yes", "no", "no (a publication)", "no (a closed form from the literature)"


def row(id, quantity, reference, by, cases, value, tolerance, unit="", note="", kind="max"):
    ok = bool(value <= tolerance) if kind == "max" else bool(value >= tolerance)
    return dict(id=id, quantity=quantity, reference=reference, written_by_the_authors=by, cases=cases, value=float(value), tolerance=float(tolerance),
                unit=unit, kind=kind, passed=ok, note=note)


# ------------------------------------------------------------------------------------------------ L1
def rung_l1():
    rows = []
    # hard pulses: RF-spoiled gradient echo and dual-TR, no diffusion
    cases = [((1,), 20, 10.0, 1000, 80, 117.0), ((1,), 45, 5.0, 1000, 80, 50.0), ((1,), 30, 8.0, 600, 40, 117.0), ((1,), 12, 6.0, 300, 40, 117.0),
             ((1, 1, 0, 0, 0, 0), 60, 20.0, 1000, 80, 129.3), ((1, 1, 0, 0), 45, 10.0, 600, 60, 117.0), ((1, 1, 0, 0, 0, 0), 55, 20.0, 838, 75, 46.0)]
    worst = max(np.abs(epg.steady_state(D(f), tr, t1, t2, phi, pat) - IB.spoiled_gre(D(f), tr, t1, t2, phi, pat, n_iso=2048)).max()
                for pat, f, tr, t1, t2, phi in cases)
    rows.append(row("L1.1", "RF-spoiled steady state of spgr and afi (hard pulses, no diffusion)",
                    "independent_bloch.spoiled_gre: 2,048 isochromats over one spoiler cycle, 3 x 3 rotations, time stepping from equilibrium",
                    AUTHORS, len(cases), worst, 1e-6, "M0"))
    # the shaped pulse: rotation of every position
    z = np.linspace(-3, 3, 241)
    worst = 0.0
    for tbw in (2.0, 4.0, 8.0):
        for deg, reph in ((10, 0.5), (60, 0.5), (90, 0.5), (180, 0.0)):
            a, b = rf.cayley_klein(D(deg), tbw, z, rephase=reph)
            a2, b2, ab = IB.slice_rotation_parameters(D(deg), tbw, z, rephase=reph)
            worst = max(worst, np.abs(a * a - a2).max(), np.abs(b * b - b2).max(), np.abs(np.conj(a) * b - ab).max())
    rows.append(row("L1.2", "rotation made by the windowed-sinc pulse at every position through the slice (Cayley-Klein parameters), 10 to 180 degrees, "
                    "time-bandwidth 2 / 4 / 8", "independent_bloch.shaped_pulse acting on the three unit vectors, sample by sample", AUTHORS, 12, worst, 1e-9))
    # slice-integrated spoiled gradient echo: the pulse stepped under its gradient for every isochromat in every repetition
    cases = [(20, 4.0, 20.0, 400.0, 50.0), (40, 2.0, 15.0, 300.0, 40.0), (12, 8.0, 20.0, 500.0, 50.0)]
    worst, during = 0.0, 0.0
    for flip, tbw, tr, t1, t2 in cases:
        rp = dict(sequence="spgr", tr_ms=tr, rf_phase_increment_deg=117.0, spoiler_mt_ms_per_m=100.0)
        got = kspace.slice_signal(rp, dict(t1_ms=t1, t2_ms=t2, adc_um2_ms=0.0), D(flip), tbw)[0][0]
        ref = IB.slice_gre(D(flip), tbw, tr, t1, t2, n_iso=96)
        worst = max(worst, abs(got - ref) / abs(ref))
        if tbw == 4.0:
            during = abs(IB.slice_gre(D(flip), tbw, tr, t1, t2, n_iso=96, rf_ms=2.0) / ref - 1)
    rows.append(row("L1.3", "RF-spoiled gradient echo integrated over the slice (time-bandwidth 2 / 4 / 8)",
                    "independent_bloch.slice_gre: isochromats over (position through the slice) x (96 spoiler phases), the 128-sample pulse stepped "
                    "under its slice gradient for each of them in every repetition", AUTHORS, len(cases), worst, 1e-3, "relative",
                    note=f"With T1 / T2 relaxation during a 2 ms pulse, which the scanner's model leaves out, the same integrator gives a signal "
                         f"{100 * during:.2f} % away (20 degrees, TR 20 ms, T1 400 ms, T2 50 ms): a declared difference, not a tolerance."))
    # spin echo and inversion recovery of a slice
    cases = [(1.0, 4.0, 500.0, 20.0, 800.0, None, 1.0), (0.8, 4.0, 300.0, 15.0, 1200.0, None, 1.0), (1.15, 2.0, 600.0, 20.0, 500.0, None, 1.0),
             (1.0, 8.0, 400.0, 20.0, 900.0, None, 1.0), (0.9, 4.0, 2000.0, 20.0, 1000.0, 400.0, 0.93)]
    worst, grating = 0.0, []
    for scale, tbw, tr, te, t1, ti, kap in cases:
        rp = dict(sequence="se" if ti is None else "ir", tr_ms=tr, te_ms=te, ti_ms=ti)
        got = kspace.slice_signal(rp, dict(t1_ms=t1), scale * np.pi / 2, tbw, inv_eff=kap)[0][0]
        worst = max(worst, abs(got - IB.slice_se(scale * np.pi / 2, tbw, tr, te, t1, ti_ms=ti, inv_eff=kap)) / abs(got))
        grating.append((IB.slice_se(scale * np.pi / 2, tbw, tr, te, t1, ti_ms=ti, inv_eff=kap, gratings=True) / got).real - 1)
    rows.append(row("L1.4", "spin echo and inversion recovery integrated over the slice: shaped 90 and 180 degree pulses, crushers, B1 80 to 115 %",
                    "independent_bloch.slice_se: isochromats over (position) x (32 crusher phases), both pulses stepped under their gradients; the "
                    "longitudinal magnetisation is averaged over the crusher phases after every echo, as the scanner's model states",
                    AUTHORS, len(cases), worst, 1e-3, "relative",
                    note=f"A refocusing pulse that is not 180 degrees everywhere stores a grating of longitudinal magnetisation, which identical "
                         f"crushers recall in the next repetition (a stimulated echo). The scanner's model leaves it out. With the grating kept, the same "
                         f"integrator gives {100 * min(grating):+.0f} to {100 * max(grating):+.0f} % of the served signal in these {len(cases)} cases "
                         f"(TR 300 to 2000 ms, T1 500 to 1200 ms): a declared difference, not a tolerance."))
    # balanced SSFP against off-resonance
    worst, n = 0.0, 0
    th = np.linspace(-np.pi, np.pi, 41)
    for flip, tr, t1, t2, cycle in ((40, 6.0, 800.0, 80.0, 180.0), (20, 4.0, 1500.0, 200.0, 180.0), (70, 8.0, 300.0, 40.0, 180.0),
                                    (40, 6.0, 800.0, 80.0, 90.0), (70, 8.0, 300.0, 40.0, 300.0)):
        ref = IB.bssfp(D(flip), tr, t1, t2, th, phase_cycle_deg=cycle)
        worst = max(worst, np.abs(S.bssfp(D(flip), th - D(cycle), tr, t1, t2) - ref).max())
        n += th.size
    rows.append(row("L1.5", "balanced SSFP against off-resonance (hard pulses, 41 precession angles per TR; RF phase advance 180, 90 and "
                    "300 degrees per TR)", "independent_bloch.bssfp: time stepping to the steady state, the phase turning the axis of every pulse",
                    AUTHORS, n, worst, 1e-6, "M0"))
    # inversion-recovery null
    worst, n = 0.0, 0
    for tr, t1, kap in ((6000.0, 800.0, 1.0), (2000.0, 800.0, 0.95), (1500.0, 1500.0, 0.85), (3000.0, 300.0, 0.9)):
        ti = np.linspace(20.0, tr - 60.0, 20001)
        s = S.ir_se(np.pi / 2, tr, 20.0, ti, t1, kap)
        k = int(np.flatnonzero(np.diff(np.sign(s)))[0])
        ours = ti[k] - s[k] * (ti[k + 1] - ti[k]) / (s[k + 1] - s[k])
        worst = max(worst, abs(ours / IB.ir_null(tr, 20.0, t1, kap) - 1))
        n += 1
    rows.append(row("L1.6", "inversion time that nulls an inversion-recovery spin echo at finite TR and inversion efficiency 0.85 to 1",
                    "independent_bloch.ir_null: the longitudinal chain stepped through 40 repetitions, bisection", AUTHORS, n, worst, 1e-6, "relative"))
    # engine 0.3.0: the calibration of the slice gradient and the saturation pulse before every excitation
    worst = max(abs(rf.fwhm_factor(tbw) / IB.slice_calibration(tbw) - 1) for tbw in (2.0, 4.0, 8.0))
    rows.append(row("L1.7", "slice gradient of the three selective pulses, in nominal bandwidths across the prescribed thickness (1.197 / 0.994 / 1.000)",
                    "independent_bloch.slice_calibration: the width at half maximum of the transverse magnetisation that a 0.0001 degree pulse leaves "
                    "on single isochromats, by bisection", AUTHORS, 3, worst, 1e-6, "relative"))
    cases = [(10, 30.0, 382.0, 68.0, 0.1, 6.5), (60, 30.0, 382.0, 68.0, 0.1, 6.5), (25, 15.0, 343.0, 58.0, 0.0, 7.0), (40, 13.0, 300.0, 40.0, 0.5, 8.0),
             (60, 500.0, 382.0, 68.0, 0.25, 7.0)]
    worst = max(abs(epg.steady_state(D(f), tr, t1, t2, sat=(q, delay))[0] - IB.spoiled_gre(D(f), tr, t1, t2, n_iso=1024, sat=(q, delay))[0])
                / abs(epg.steady_state(D(f), tr, t1, t2)[0]) for f, tr, t1, t2, q, delay in cases)
    rows.append(row("L1.8", "RF-spoiled steady state with a saturation pulse (Mz x q, 6.5 to 8 ms before every excitation; 50 to 100 % saturated)",
                    "independent_bloch.spoiled_gre with the saturation as a step of the time stepping: 1,024 isochromats", AUTHORS, len(cases), worst, 1e-6,
                    "of the unsaturated signal",
                    note="The saturation acts on the longitudinal magnetisation only, in the scanner and in the integrator: what a real pulse does "
                         "to the transverse magnetisation of earlier repetitions is left out of both (limits.md, block 9a)."))
    return dict(rung="L1", title="The scanner's equations, solved by other code", rows=rows,
                not_run=["per-pulse transient of the first 300 pulses: the scanner serves the periodic state only"])


# ------------------------------------------------------------------------------------------------ L2
def rung_l2():
    import compare_sycomore as CS
    path = os.path.join(ROOT, "docs", "physics", "external", "sycomore_reference.json")
    with open(path, encoding="utf-8") as f:
        ref = json.load(f)
    assert ref["spgr_cases"] == [list(c) for c in CS.SPGR_CASES] and ref["afi_cases"] == [list(c) for c in CS.AFI_CASES], "stored cases differ from the script's"
    rows_, nd, wd = CS.compare(ref["spgr"], ref["afi"])
    who = f"sycomore {ref['sycomore_version']} (Lamy and others), values stored {ref['date_utc']}"
    rows = [row("L2.1", "RF-spoiled steady state of spgr (7 cases) and afi, dual TR (2 cases), no diffusion", who, OTHERS,
                sum(not r["diffusion"] for r in rows_), nd, CS.BOUND_NO_DIFFUSION, "M0"),
            row("L2.2", "the same with diffusion: spgr (6 cases), afi (2 cases)", who, OTHERS, sum(r["diffusion"] for r in rows_), wd, CS.BOUND_DIFFUSION, "M0",
                note="Both codes use the constant-gradient b-factors of Weigel 2010: this checks the implementation, not the diffusion model.")]
    live = "not installed here: the stored values were used"
    try:
        spgr, afi = CS.sycomore_values()
        drift = max(float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in zip(spgr + afi, ref["spgr"] + ref["afi"]))
        rows.append(row("L2.3", "sycomore run now against its stored values", who, OTHERS, len(spgr) + len(afi), drift, 1e-12, "M0"))
        live = "installed: recomputed"
    except ImportError:
        pass
    return dict(rung="L2", title="Another group's code", rows=rows, sycomore=live,
                not_run=["the pulse's Cayley-Klein parameters against sigpy.mri.rf.sim.abrm (sigpy is not installed here)",
                         "an image of the flat QA world against MRzero-Core or KomaMRI (neither is installed here)"])


# ------------------------------------------------------------------------------------------------ L3
TISSUES = [("white matter 3 T", 838, 75, 0.75), ("grey matter 3 T", 1607, 95, 0.85), ("muscle", 1412, 50, 1.4), ("liver", 809, 34, 1.1), ("blood", 1932, 275, 2.5),
           ("gel 1000 / 200 / 2.0", 1000, 200, 2.0), ("vial 2500 / 250 / 0.6", 2500, 250, 0.6), ("vial 300 / 40 / 2.2", 300, 40, 2.2)]


def _vfa_t1(f1, f2, tr, area, phi, t1, t2, adc):
    a = D(np.array([f1, f2], float))
    s = np.array([abs(epg.steady_state(x, tr, t1, t2, phi, (1,), kspace.diffusion_bd(adc, area, tr))[0]) for x in a])
    y, x = s / np.sin(a), s / np.tan(a)
    return -tr / np.log((y[1] - y[0]) / (x[1] - x[0]))


def _afi_b1_per_pulse(flip, tr1, n, t1, t2, adc, area, phi_deg, cycles_extra=30):
    """Dual-TR B1 estimate from a time-stepping EPG in which the RF phase advances per PULSE (phi p (p + 1) / 2 for
    pulse number p), the scheme of the publication. Written here; the engine advances the phase per raster position."""
    e1, e2 = np.exp(-tr1 / t1), np.exp(-tr1 / t2)
    bd = adc * 1e-9 * (2 * np.pi * 42.577478518e6 * area * 1e-6) ** 2 * tr1 * 1e-3
    kmax = int(np.ceil(-np.log(1e-7) * t2 / tr1)) + 4
    k = np.arange(kmax)
    dz, dp, dm = np.exp(-bd * k * k), np.exp(-bd * (k * k + k + 1 / 3)), np.exp(-bd * (k * k - k + 1 / 3))
    fp, fm, z = np.zeros(kmax, complex), np.zeros(kmax, complex), np.zeros(kmax, complex)
    z[0] = 1
    s1, s2, p = [], [], 0
    c2, s_2, s, c = np.cos(flip / 2) ** 2, np.sin(flip / 2) ** 2, np.sin(flip), np.cos(flip)
    for j in range((int(np.ceil(8 * t1 / (tr1 * (n + 1)))) + cycles_extra) * (n + 1)):
        if j % (n + 1) in (0, 1):
            ph = D(phi_deg) * p * (p + 1) / 2
            fp, fm, z = (c2 * fp + np.exp(2j * ph) * s_2 * fm - 1j * np.exp(1j * ph) * s * z,
                         np.exp(-2j * ph) * s_2 * fp + c2 * fm + 1j * np.exp(-1j * ph) * s * z,
                         -0.5j * np.exp(-1j * ph) * s * fp + 0.5j * np.exp(1j * ph) * s * fm + c * z)
            (s1 if j % (n + 1) == 0 else s2).append(abs(fp[0]))
            p += 1
        z = z * e1 * dz
        z[0] += 1 - e1
        fp, fm = np.roll(fp * e2 * dp, 1), np.roll(fm * e2 * dm, -1)
        fm[-1] = 0
        fp[0] = np.conj(fm[0])
    r = np.mean(s2[-30:]) / np.mean(s1[-30:])
    return np.arccos(np.clip((n * r - 1) / (n - r), -1, 1)) / flip


def _afi_b1_engine(flip, tr1, n, t1, t2, adc, area, phi_deg):
    s = epg.steady_state(flip, tr1, t1, t2, phi_deg, (1, 1) + (0,) * (n - 1), kspace.diffusion_bd(adc, area, tr1))
    r = (s[1] / s[0]).real
    return np.arccos(np.clip((r * n - 1) / (n - r), -1, 1)) / flip


def rung_l3():
    from mrgym import validate
    rows = []
    stated = "Yarnykh 2010 (Magn Reson Med 63:1610), abstract: relative T1 and B1 errors below 2 % in a variety of tissues"
    for label, f1, f2, tr, area in (("TR 20 ms, spoiler 450 mT*ms/m", 4, 20, 20.0, 450.0), ("TR 15 ms, spoiler 280 mT*ms/m", 3, 17, 15.0, 280.0)):
        worst = max(abs(_vfa_t1(f1, f2, tr, area, 169.0, t1, t2, adc) / t1 - 1) for _, t1, t2, adc in TISSUES)
        rows.append(row(f"L3.{len(rows) + 1}", f"two-angle T1 with the published spoiling: {label}, RF increment 169 degrees ({f1} and {f2} degrees, hard pulses, "
                        f"{len(TISSUES)} materials)", stated, PAPER, len(TISSUES), worst, 0.02, "relative T1 error"))
    worst = max(abs(_afi_b1_per_pulse(D(55.0), 20.0, 5, t1, t2, adc, 450.0, 39.0) - 1) for _, t1, t2, adc in TISSUES[:3] + TISSUES[6:7])
    rows.append(row("L3.3", "dual-TR B1 with the published spoiling (TR 20 / 100 ms, spoiler 450 / 2250 mT*ms/m, RF increment 39 degrees counted per pulse), "
                    "55 degrees, 4 materials", stated + "; signal from a time-stepping EPG written for this row with the publication's phase scheme",
                    PAPER, 4, worst, 0.01, "relative B1 error"))
    worst = max(abs(_afi_b1_engine(D(55.0), 20.0, 5, t1, t2, adc, 450.0, 46.0) - 1) for _, t1, t2, adc in TISSUES[:3] + TISSUES[6:7])
    rows.append(row("L3.4", "the same with the scanner's own phase scheme (increment counted per raster position), at the equivalent increment of 46 degrees",
                    stated, PAPER, 4, worst, 0.02, "relative B1 error",
                    note="The scanner counts the RF phase per raster position of period TR1; a published per-pulse increment of 39 degrees corresponds to "
                         "about 46 degrees here. The instrument datasheet says so."))
    # the published bias of 2D variable-flip-angle T1
    bias = []
    for tbw in (2.0, 4.0):
        for t1 in (300.0, 1000.0, 2500.0):
            for f1, f2 in ((4, 20), (3, 17), (6, 35)):
                a = D(np.array([f1, f2], float))
                s = np.abs(kspace.slice_signal(dict(sequence="spgr", tr_ms=20.0), dict(t1_ms=t1), a, tbw, tier=1)[0])
                y, x = s / np.sin(a), s / np.tan(a)
                bias.append(-20.0 / np.log((y[1] - y[0]) / (x[1] - x[0])) / t1 - 1)
    rows.append(row("L3.5", f"two-angle T1 on a 2D slice with the nominal angles (time-bandwidth 2 and 4): {100 * max(bias):.0f} to {100 * min(bias):.0f} %",
                    "Svedin & Parker 2017: 40 to 80 % underestimation without profile correction; Dieringer 2014: (-63 +- 1.5) %; band -35 to -80 %",
                    PAPER, len(bias), max(0.0, -0.80 - min(bias), max(bias) + 0.35), 0.0, "outside the band"))
    by_id = {r["id"]: r for r in validate.run(["encoding"])[0]}
    for vid, what, who in (("K8", "overshoot at a sharp edge", "Gibbs phenomenon, 8.95 % of the step"),
                           ("K16", "magnitude of the background", "Rayleigh statistics: mean 1.2533 sigma, SD 0.6551 sigma (Gudbjartsson & Patz 1995)"),
                           ("K3", "water-fat shift along the readout", "3.4 ppm x gamma B0 / bandwidth per pixel"),
                           ("K12", "EPI displacement along phase encoding", "offset x echo spacing x lines (Jezzard & Balaban 1995)"),
                           ("K25", "position of the ghosts of a periodically pulsing source", "lines x frac(f x TR x averages) (Zaitsev 2015)")):
        v = by_id[vid]
        rows.append(row(f"L3.{len(rows) + 1}", f"{what} (row {vid} of `mrgym validate`): {v['measured']}", who, FORMULA, 1, v["value"], v["limit"]))
    return dict(rung="L3", title="Published operating points, published biases, closed forms", rows=rows, not_run=[])


RUNGS = (rung_l1, rung_l2, rung_l3)
NOT_BUILT = [
    ("L4", "A standard object with published reference values (the plates of the ISMRM/NIST system phantom; Stupic 2021) imaged with published "
           "inversion-recovery and spin-echo protocols", "on its own page: [measured.md](measured.md), rungs M0 (the plates against the published tables), M1 "
           "(inversion-recovery T1 of the T1 plate) and M9 (the T2 plate from raw data)"),
    ("L5", "A scanner: the same protocols on a real system", "on its own page: [measured.md](measured.md) compares statistics of images, region signals and fitted "
           "values of public phantom sessions on real scanners with the engine's model of the same protocol. No image of the six console sequences has been "
           "compared pixel by pixel with a scan of the same object"),
]
NOT_EXTERNAL = [
    "contrast of `se` and `ir` in tissue beyond the null point and the slice integral (the tissue values decide it; the curves of `se` and `ir` on phantoms "
    "are compared with measurements in [measured.md](measured.md), rungs M7 and M9)", "the transient of every sequence (not served)",
    "the Nyquist-ghost model of `epi`", "fat saturation (efficiency, band, offset)", "the diffusion model itself (sycomore uses the same b-factor formula)",
    "the periodic-motion model beyond the position and amplitude of its ghosts",
    "the 3D mode (it is the hard-pulse signal with a price; slab profile and slab wrap are not modelled)",
]


def markdown(results, engine_id):
    out = ["<!-- Generated by tools/external/run_all.py --write. -->", "# External validation", "",
           f"Engine `{engine_id}`. What this page is: the scanner compared with things it does not share code with, rung by rung. What it is not: a comparison "
           "with a measurement. Those are on their own page, [measured.md](measured.md) (rungs L4 and L5): public phantom sessions on real scanners against the "
           "engine's model of the same protocol, with the differences that were found. Nothing on either page shows that a protocol found on this scanner is right "
           "on a real one. "
           "The column \"written by the authors\" says who wrote the reference: a row whose reference the authors of MR Gym wrote shows that two programs solve "
           "the same equations, and is called verification here, not validation.", "",
           "Regenerate: `python tools/external/run_all.py --write`. Check: `python tools/external/run_all.py --check`.", ""]
    for r in results:
        out += [f"## {r['rung']}. {r['title']}", "", "| # | Quantity | Reference | Reference written by the authors of MR Gym? | Cases | Largest difference | Tolerance | |",
                "|---|---|---|---|---:|---|---|---|"]
        for q in r["rows"]:
            bound = ("≤ " if q["kind"] == "max" else "≥ ") + f"{q['tolerance']:g}"
            out.append(f"| {q['id']} | {q['quantity']} | {q['reference']} | {q['written_by_the_authors']} | {q['cases']} | {q['value']:.2g} {q['unit']} | {bound} | "
                       f"{'pass' if q['passed'] else '**FAIL**'} |")
        out.append("")
        out += [f"{q['id']}: {q['note']}" for q in r["rows"] if q["note"]]
        if r.get("sycomore"):
            out.append(f"sycomore on the machine that generated this page: {r['sycomore']}.")
        if r["not_run"]:
            out += ["", "Not run: " + "; ".join(r["not_run"]) + "."]
        out.append("")
    out += ["## Rungs on another page", "", "| Rung | What it is | State |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in NOT_BUILT]
    out += ["", "## Not checked against anything external", "", "\n".join(f"- {x}" for x in NOT_EXTERNAL), "",
            "Tissue values and console timing left this list: both are compared in [measured.md](measured.md) (rungs M5 and M0).", "",
            "The largest differences are from the machine that generated this page; values at rounding level differ between machines, the tolerances do not.", ""]
    return "\n".join(out)


def _strip(md):
    """The page without the measured column (it differs between machines at rounding level)."""
    out = []
    for line in md.splitlines():
        cells = line.split(" | ")
        if line.startswith("| L") and len(cells) >= 8:
            cells = cells[:5] + cells[6:]
        out.append(" | ".join(cells) if line.startswith("| L") else line)
    return "\n".join(x for x in out if not x.startswith("sycomore on the machine") and "With T1 / T2 relaxation during" not in x and "stores a grating of longitudinal" not in x
                     and not x.startswith("| L3.5") and not x.startswith("| L3.6") and not x.startswith("| L3.7") and not x.startswith("| L3.8")
                     and not x.startswith("| L3.9") and not x.startswith("| L3.10"))


def main(argv=None):
    import mrgym
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    results = []
    for fn in RUNGS:
        r = dict(fn(), engine=mrgym.ENGINE_ID, date_utc=date, command="python tools/external/run_all.py --write")
        results.append(r)
        print(f"== {r['rung']}. {r['title']}")
        for q in r["rows"]:
            print(f"   {q['id']:6s} {'ok  ' if q['passed'] else 'FAIL'} {q['value']:9.2e} {'<=' if q['kind'] == 'max' else '>='} {q['tolerance']:<7g} {q['quantity'][:110]}")
    for rung, _, state in NOT_BUILT:
        print(f"== {rung}. {state.replace('**', '')}")
    failed = [q["id"] for r in results for q in r["rows"] if not q["passed"]]
    md = markdown(results, mrgym.ENGINE_ID)
    ext = os.path.join(ROOT, "docs", "physics", "external")
    stale = []
    if a.write:
        os.makedirs(ext, exist_ok=True)
        for r in results:
            with open(os.path.join(ext, r["rung"] + ".json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1)
                f.write("\n")
        with open(os.path.join(ROOT, "docs", "physics", "external.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print("written: docs/physics/external.md and docs/physics/external/L1.json, L2.json, L3.json")
    elif a.check:
        for r in results:
            try:
                with open(os.path.join(ext, r["rung"] + ".json"), encoding="utf-8") as f:
                    old = json.load(f)
            except FileNotFoundError:
                stale.append(f"{r['rung']}: no stored result")
                continue
            if old["engine"] != mrgym.ENGINE_ID:
                stale.append(f"{r['rung']}: stored for {old['engine']}")
            elif [(q["id"], q["tolerance"], q["passed"]) for q in old["rows"]] != [(q["id"], q["tolerance"], q["passed"]) for q in r["rows"]]:
                stale.append(f"{r['rung']}: the stored rows differ from the rows of this run")
        try:
            with open(os.path.join(ROOT, "docs", "physics", "external.md"), encoding="utf-8") as f:
                if _strip(f.read()) != _strip(md):
                    stale.append("docs/physics/external.md differs from what this run would write")
        except FileNotFoundError:
            stale.append("docs/physics/external.md is missing")
    n = sum(len(r["rows"]) for r in results)
    print(f"{n - len(failed)} / {n} rows passed" + (f"; failed: {', '.join(failed)}" if failed else "") + ("; stale: " + "; ".join(stale) if stale else ""))
    return 1 if failed or stale else 0


if __name__ == "__main__":
    sys.exit(main())
