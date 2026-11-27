"""The comparison with measured data: every rung that can be run here, as one command and one stored table.

    python tools/measured/run_all.py                    # run, print the table
    python tools/measured/run_all.py --check            # run; exit 1 if a row fails, a declared difference has grown, or the stored table is stale
    python tools/measured/run_all.py --write            # run, store docs/physics/measured/results.json and docs/physics/measured.md
    python tools/measured/run_all.py --fetch --cache DIR [--write]
                                                        # first download the public originals into DIR (outside the repository, about 0.7 GB),
                                                        # check their checksums and rebuild the cached extracts from them

What is compared. Public phantom sessions on real scanners (``sources.py``: where, licence, citation) were reduced to
small extracts (``docs/physics/measured/*.json``; ``extract.py``). Each rung runs the engine's own model of the same
protocol with the same estimator and puts the two side by side: one row per quantity, with the measurement, its
spread and n, the engine, the difference, and either a tolerance taken from the spread of the measurement or the
words "declared difference". A row that fails is listed. A declared difference is a number the engine is known to
get wrong; it carries a ceiling and fails when it grows past it.

What is not compared: no image of the six console sequences was compared pixel by pixel with a scan of the same
object. The rungs compare statistics of images, signals of regions and fitted values.

Rungs
  M0   the NIST plate in the package against the published tables; console timing against three product sequences
  M1   inversion-recovery T1 on the NIST plate: 34 maps of the ISMRM 2020 T1 challenge
  M2   variable-flip-angle T1: the published rule and example
  M2b  two-angle T1 on a 2D slice without profile correction: a published measurement at 1.5 and 3 T
  M3b  T2 from a multi-echo spin echo: a published measurement against the design layer (the console has no train)
  M4 / M8  fat: the peak table, the published protocols, and the six measured echoes of 24 acquisitions
  M5   the tissue table against a literature compilation (runs with --fetch only: its licence is unclear)
  M6   ACR phantom: 14 real series of 9 scanners measured with the package's own QA functions
  M7   gel sphere: product inversion-recovery and spin-echo series
  M9   NIST T2 plate from raw k-space: spin echo at 12 TE, inversion recovery at 11 TI, B1 map

A stored table is stale when its engine id is not the engine's, when a cached extract is not the one it was computed
from, when the tissue table has changed since M5 was fetched, or when the rows of this run differ from the stored ones.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys

import numpy as np
import scipy.optimize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import fits  # noqa: E402
import page  # noqa: E402

from mrgym import qa  # noqa: E402
from mrgym.design import pulses  # noqa: E402
from mrgym.design import sequence as Q  # noqa: E402
from mrgym.scanner import anatomy, console, kspace, tissue  # noqa: E402
from mrgym.scanner import signal as S  # noqa: E402
from mrgym.scanner import world as W  # noqa: E402

STORE = os.path.join(ROOT, "docs", "physics", "measured")
RESULTS = os.path.join(STORE, "results.json")
EXTRACTS = dict(nist_tables="nist_tables.json", rrsg="M1_t1_challenge.json", acr="M6_acr.json", gel="M7_gel.json", pdff="M8_fat_phantom.json",
                scholand="M9_nist_t2_raw.json")
SLICE_FILE = "acr_off_centre_slice7.npz"
D = np.deg2rad


# ------------------------------------------------------------------------------------------------ rows
def test(id, quantity, measured, engine, difference, tolerance, ok, note=""):
    """A row with a tolerance: it passes or it fails."""
    return dict(id=id, quantity=quantity, measured=measured, engine=engine, difference=difference, tolerance=tolerance, declared=False,
                passed=bool(ok), note=note)


# Ceilings of the declared differences: the size on the day each was declared, rounded up. A declared difference
# that grows past its ceiling fails ``--check`` and tests/test_measured.py; one that shrinks is good news and a
# reason to lower the ceiling.
CEILINGS = {
    "M0.2": 0.75, "M0.3": 3.6, "M0.4": 21.0, "M0.6": 70.0, "M0.7": 43.0, "M0.8": 22.0, "M1.5": 5.5, "M1.6": 2.8, "M2.3": 11.0, "M3b.4": 1.0,
    "M4.2": 0.12, "M5.2": 17.0, "M6.3": 1.3, "M6.4": 1.0, "M6.5": 0.02, "M6.6": 0.22, "M6.8": 4.2, "M6.11": 21.0, "M6.12": 4.0, "M7.5": 4.6, "M8.3": 0.10,
    "M9.3": 160.0, "M9.4": 9.0, "M9.5": 4.0, "M9.6": 0.33, "M9.7": 1.3,
}


def declared(id, quantity, measured, engine, difference, size, known, note=""):
    """A row without a tolerance: a difference the engine is known to have. ``size`` is the number that must not
    grow past ``CEILINGS[id]``; ``known`` is its line in the list of known differences (None: about the object)."""
    return dict(id=id, quantity=quantity, measured=measured, engine=engine, difference=difference, tolerance="declared difference", declared=True,
                size=float(size), ceiling=float(CEILINGS[id]), passed=bool(size <= CEILINGS[id]), known=known, note=note)


def load(name):
    with open(os.path.join(STORE, EXTRACTS[name]), encoding="utf-8") as f:
        return json.load(f)


def rng_span(v, fmt="{:+.1f}"):
    v = np.asarray(v, float)
    return f"{fmt.format(v.min())} to {fmt.format(v.max())}"


# ------------------------------------------------------------------------------------------------ engine helpers
def seed(i):
    return f"{0x5eed0000000000000000000000000000 + i:032x}"


def nist_world(i, t1_ms=None, t2_ms=None, **spec):
    """The package's NIST T1-plate world (3 T); ``t1_ms`` / ``t2_ms`` replace the built-in values of the 14 spheres."""
    w = W.make_world(seed(i), dict(kind="nist_t1", b0_t=3.0, **spec))
    for k in range(14):
        c = w["tissues"][f"sphere{k + 1}"]
        if t1_ms is not None:
            c["t1_ms"] = float(t1_ms[k])
        if t2_ms is not None:
            c["t2_ms"] = float(t2_ms[k])
            c["t2s_ms"] = 1.0 / (1.0 / c["t2_ms"] + 1.0 / c["t2p_ms"])
    return w


def roi_masks(w, rp, radius_mm):
    """Boolean masks of the pixels whose centre lies within ``radius_mm`` of each region of the world."""
    n_pe, n_ro = rp["matrix_pe"], rp["matrix"]
    y = (np.arange(n_pe) - n_pe // 2) * rp["fov_pe_mm"] / n_pe
    x = (np.arange(n_ro) - n_ro // 2) * rp["fov_mm"] / n_ro
    return [np.hypot(y[:, None] - r["y_mm"], x[None, :] - r["x_mm"]) <= radius_mm for r in w["regions"]]


def realise(prot, cons, only=None):
    """The realised protocol; where the console cannot realise a published time, its own minimum is taken."""
    pl = console.plan(prot, cons)
    for _ in range(4):
        if pl["accept"]:
            break
        fix = {v["parameter"]: v["minimum"] for v in pl["violations"] if v.get("minimum") is not None and (only is None or v["parameter"] in only)}
        if not fix:
            break
        prot = dict(prot, **fix)
        pl = console.plan(prot, cons)
    if not pl["accept"]:
        raise ValueError("; ".join(v["message"] for v in pl["violations"]))
    return pl["realised"]


# ------------------------------------------------------------------------------------------------ M0
def rung_m0():
    t = load("nist_tables")
    a = anatomy.NIST_T1
    ang = 2 * np.pi * np.arange(a["ring_n"]) / a["ring_n"]
    g = a["grid_mm"] / 2
    plate = [(-a["ring_mm"] * np.cos(q), a["ring_mm"] * np.sin(q)) for q in ang] + [(-g, -g), (-g, g), (g, g), (g, -g)]      # (y, x) of ``nist_t1``
    dist = max(float(np.hypot(x - px, -y - pz)) for (y, x), (px, pz) in zip(plate, t["xz_mm"]))
    s1, s2 = np.array(t["table_s1_nicl2"]), np.array(t["table_s2_mncl2"])
    t1 = np.array(a["t1_ms"])
    d_t1 = t1 - s1[:, 2]
    k = int(np.abs(d_t1).argmax())
    rule = 100 * (t1 * a["t2_ms_sphere_12"] / t1[11] / s1[:, 3] - 1)
    water = [1e3 / v for v in t["table_s4_water_rate_per_s"]["3.0"]]
    fill = (100 * (a["fill"]["t1_ms"] / water[0] - 1), 100 * (a["fill"]["t2_ms"] / water[1] - 1))
    same = sum(int(np.array_equal(np.array(tab[key][b]), ref[:, col])) for tab, ref in ((anatomy.NIST_T1_PUBLISHED, s1), (anatomy.NIST_T2, s2))
               for key, b, col in (("t1_ms", 1.5, 0), ("t2_ms", 1.5, 1), ("t1_ms", 3.0, 2), ("t2_ms", 3.0, 3)))
    served = "nist_t2" in W.KINDS
    section = 2 * float(np.sqrt((a["shell_mm"] / 2) ** 2 - t["plate_y_mm"]["ni"] ** 2))      # the plate lies off the centre of the spherical shell
    rows = [
        test("M0.1", "places of the 14 spheres of the T1 plate (world `nist_t1`)", f"Table S1, stated accuracy ±{t['position_accuracy_mm']:g} mm",
             f"largest distance {dist:.2f} mm, same order", f"{dist:.2f} mm", f"≤ {t['position_accuracy_mm']:g} mm (the table's accuracy)", dist <= t["position_accuracy_mm"]),
        declared("M0.2", "T1 of the 14 spheres of `nist_t1` at 3 T", f"Table S1; sphere {k + 1}: {s1[k, 2]:g} ± {t['t1_3t_standard_deviation_ms_nicl2'][k]:g} ms",
                 f"sphere {k + 1}: {t1[k]:g} ms; the other 13 equal the table", f"{d_t1[k]:+.1f} ms", abs(d_t1[k]), 11),
        declared("M0.3", "T2 of the 14 spheres of `nist_t1` at 3 T", "Table S1: measured for every sphere", "T1 × 31.86 / 44.53, a declared rule",
                 f"{rng_span(rule)} %", np.abs(rule).max(), 9),
        declared("M0.4", "water round the spheres of `nist_t1` at 3 T: T1, T2", f"Table S4: {water[0]:.0f} ms, {water[1]:.0f} ms",
                 f"{a['fill']['t1_ms']:g} ms, {a['fill']['t2_ms']:g} ms (declared)", f"{fill[0]:+.0f} %, {fill[1]:+.0f} %", max(abs(v) for v in fill), 9),
        test("M0.5", "the new data: T1 and T2 of both plates at 1.5 and 3 T (`anatomy.NIST_S1`, `NIST_S2`; the plate `anatomy.nist_t2`)",
             "Tables S1 and S2, 112 values", f"{same} of 8 columns equal the table" + ("" if served else "; the scanner does not serve `nist_t2` as a world kind yet"),
             f"{8 - same} columns differ", "0 (a transcription)", same == 8),
    ]
    ex = dict(fov_mm=250.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
    base = dict(sequence="spgr", flip_deg=10.0, tr_ms=3.0, te_ms=0.5, fov_mm=250.0, matrix=256, excitation="nonselective")
    te, tr, text = [], [], []
    for vendor, bw, pub_tr, pub_te in (("Siemens", 280.0, 6.6, 2.44), ("GE", 488.0, 6.0, 1.4), ("Philips", 904.0, 6.6, 1.8)):   # Stupic 2021, Table S10
        rp = realise(dict(base, bandwidth_hz_per_px=bw), ex)
        te.append(100 * (rp["te_ms"] / pub_te - 1)), tr.append(100 * (rp["tr_ms"] / pub_tr - 1))
        text.append((f"{vendor} {pub_te:g} / {pub_tr:g} at {bw:g} Hz/px", f"{rp['te_ms']:g} / {rp['tr_ms']:g}"))
    rows += [
        declared("M0.6", "shortest TE of a 3D RF-spoiled gradient echo, matrix 256, 250 mm: TE / TR in ms", "; ".join(a for a, _ in text) + " (Table S10)",
                 "; ".join(b for _, b in text), f"TE {rng_span(te, '{:+.0f}')} %", max(te), 3),
        declared("M0.7", "shortest TR of the same three protocols", "as above", "as above", f"TR {rng_span(tr, '{:+.0f}')} %", max(tr), 3),
    ]
    section_t2 = 2 * float(np.sqrt((a["shell_mm"] / 2) ** 2 - t["plate_y_mm"]["mn"] ** 2))
    rows.append(declared("M0.8", "outline of the T1 plate (`nist_t1`): diameter of the water round the spheres",
                         f"{section:.1f} mm: the section of the {a['shell_mm']:g} mm shell at the plate, {t['plate_y_mm']['ni']:g} mm from its centre (Table S1, column Y)",
                         f"{a['shell_mm']:g} mm: the full shell", f"{100 * (a['shell_mm'] / section - 1):+.0f} %", 100 * (a["shell_mm"] / section - 1), 9,
                         note=f"Found while adding the T2 plate, whose outline is the section at its own height ({section_t2:.1f} mm); the outline measured in rung M9 "
                              "agrees with that."))
    return dict(rung="M0", title="The NIST plate of the package against the published tables; console timing", rows=rows, extracts=["nist_tables"],
                source="Stupic 2021, supporting information (CC BY)",
                notes=["The tables are NMR reference measurements at 20 °C. Rows M0.2 to M0.4 and M0.8 are about the older world `nist_t1`, which this branch does "
                       "not change; M0.5 is the data added here."])


# ------------------------------------------------------------------------------------------------ M1
TI_M1 = (50.0, 400.0, 1100.0, 2500.0)
M1_BASE = dict(sequence="ir", tr_ms=2550.0, te_ms=14.0, fov_mm=256.0, matrix=256, slice_mm=2.0, bandwidth_hz_per_px=130.0)
M1_SYSTEMS, M1_ROI_MM = 12, 5.3                       # 88 pixels of 1 mm^2, the commonest region of the database


def _ir_run(i, sigma0=None, noise=True, magnitude=False, **over):
    w = nist_world(i, **({} if sigma0 is None else dict(sigma0=sigma0, sigma0_log_sd=0.0)))
    eng = kspace.Engine(w)
    rng = np.random.default_rng(1000 + i) if noise else None
    ims = []
    for ti in TI_M1:
        rp = realise(dict(M1_BASE, ti_ms=ti, **over), w["console"])
        ims.append(eng.acquire(rp, rng)[0])
    ims = np.stack(ims, -1)
    truth = np.array([w["tissues"][f"sphere{k + 1}"]["t1_ms"] for k in range(14)])
    mean, cov = [], []
    for m in roi_masks(w, rp, M1_ROI_MM):
        t1 = fits.ir_rdnls(ims[m], TI_M1, magnitude=magnitude)
        mean.append(t1.mean()), cov.append(100 * t1.std(ddof=1) / t1.mean())
    return 100 * (np.array(mean) / truth - 1), np.array(cov)


def rung_m1():
    x = load("rrsg")
    q = lambda part, key: np.array(x[part][key])                 # noqa: E731  [q25, median, q75][sphere]
    e_all, c_all, e_c, c_c, e_m, c_m = (q(p, k) for p in ("all", "complex", "magnitude") for k in ("error_percent_q25_median_q75", "in_roi_cov_percent_q25_median_q75"))
    n_all, n_c, n_m = x["all"]["n"], x["complex"]["n"], x["magnitude"]["n"]
    clean = np.array([_ir_run(i, noise=False)[0] for i in range(4)] + [_ir_run(0, noise=False, rf_pulse=p, te_ms=16.0)[0] for p in ("fast", "sharp")])
    form = float(np.abs(clean)[:, :8].max())
    cd = np.median([_ir_run(i)[1] for i in range(M1_SYSTEMS)], 0)
    scale = float(np.mean(c_all[1][2:5]) / np.mean(cd[2:5]))        # ONE fitted number: the noise level, from the scatter inside spheres 3 to 5
    sig = W.DEFAULT_SPEC["sigma0"] * scale
    rc, rm = ([_ir_run(i, sigma0=sig, magnitude=mag) for i in range(M1_SYSTEMS)] for mag in (False, True))
    ec, cc = np.array([r[0] for r in rc]), np.median([r[1] for r in rc], 0)
    em, cm = np.median([r[0] for r in rm], 0), np.median([r[1] for r in rm], 0)
    ec_med, ec_iqr = np.median(ec, 0), np.percentile(ec, 75, 0) - np.percentile(ec, 25, 0)
    others = [0, 1, 5, 6, 7, 8]                                        # spheres 1, 2, 6 to 9: not used for the one fitted number
    d_cov = cc[others] - c_c[1][others]
    iqr_cov = c_c[2][others] - c_c[0][others]
    jump = [7, 8]                                                      # spheres 8 and 9: the magnitude fit breaks down
    in_jump = all(e_m[0][k] <= em[k] <= e_m[2][k] for k in jump)
    low = [10, 11, 12, 13]                                             # spheres 11 to 14: below 90 ms only TI 50 ms lies on the recovery
    in_low = sum(e_all[0][k] <= ec_med[k] <= e_all[2][k] for k in low)
    meas_iqr = float(np.median((e_all[2] - e_all[0])[:7]))
    eng_iqr = float(np.median(ec_iqr[:7]))
    ser = x["median_error_percent_spheres_1_to_7"]
    rows = [
        test("M1.1", "scatter of T1 inside a sphere, complex fit, spheres 1, 2, 6 to 9 (coefficient of variation, %)",
             ", ".join(f"{v:.1f}" for v in c_c[1][others]) + f" (median of {n_c} maps)", ", ".join(f"{v:.1f}" for v in cc[others]) + f" ({M1_SYSTEMS} systems)",
             f"{rng_span(d_cov)} points", "within the interquartile range of the maps, per sphere (" + ", ".join(f"{v:.1f}" for v in iqr_cov) + " points)",
             np.all(np.abs(d_cov) <= iqr_cov), note="One number was fitted: the noise level, so that the scatter inside spheres 3 to 5 equals the measured median "
                                                    f"(default noise × {scale:.2f}). The row is then a test of how noise propagates, not of the noise level."),
        test("M1.2", "magnitude fit where the null lies between the first two TI: T1 error of spheres 8 and 9, and scatter inside them",
             "; ".join(f"{e_m[1][k]:+.1f} % [{e_m[0][k]:+.1f}, {e_m[2][k]:+.1f}], scatter {c_m[1][k]:.1f} %" for k in jump) + f" ({n_m} maps)",
             "; ".join(f"{em[k]:+.1f} %, scatter {cm[k]:.1f} %" for k in jump), "; ".join(f"{em[k] - e_m[1][k]:+.1f} points" for k in jump),
             "error inside the interquartile range of the maps", in_jump,
             note=f"Complex fit of the same spheres: measured {e_c[1][7]:+.1f} / {e_c[1][8]:+.1f} %, engine {ec_med[7]:+.2f} / {ec_med[8]:+.2f} %."),
        test("M1.3", "collapse of the estimator below 90 ms: T1 error of spheres 11 to 14 (63, 45, 31, 22 ms)",
             "; ".join(f"{e_all[1][k]:+.0f} % [{e_all[0][k]:+.0f}, {e_all[2][k]:+.0f}]" for k in low) + f" ({n_all} maps)",
             "; ".join(f"{ec_med[k]:+.0f} %" for k in low), "; ".join(f"{ec_med[k] - e_all[1][k]:+.0f}" for k in low) + " points",
             "inside the interquartile range of the maps, each sphere", in_low == 4),
        test("M1.4", "the served form: T1 error without noise, spheres 1 to 8 (T1 175 ms and longer), 6 systems and pulse shapes", "form assumed by the "
             "challenge's estimator: a + b exp(−TI / T1)", f"largest error {form:.2f} %", f"{form:.2f} %", "≤ 0.5 % (a tenth of the spread between sites)", form <= 0.5),
        declared("M1.5", "spread of the T1 error between sites, spheres 1 to 7 (T1 250 to 2000 ms): interquartile range",
                 f"{meas_iqr:.1f} points; median error {rng_span(e_all[1][:7])} % ({n_all} maps); by phantom version {ser['serial_up_to_42']:+.1f} % "
                 f"(serial ≤ 42, {ser['n_up_to_42']} maps) and {ser['serial_from_43']:+.1f} % (serial ≥ 43, {ser['n_from_43']} maps)",
                 f"{eng_iqr:.1f} points; median error {rng_span(ec_med[:7], '{:+.2f}')} %", f"{meas_iqr - eng_iqr:.1f} points of spread missing", meas_iqr - eng_iqr, 7),
        declared("M1.6", "receiver noise: default noise of the package over the noise that gives the measured scatter", f"scatter of spheres 3 to 5: "
                 f"{np.mean(c_all[1][2:5]):.2f} % (median of {n_all} maps)", f"{np.mean(cd[2:5]):.2f} % at the default noise", f"default noise × {1 / scale:.1f} too high",
                 1 / scale, 2),
    ]
    return dict(rung="M1", title="Inversion-recovery T1 on the NIST plate", rows=rows, extracts=["rrsg"],
                source="ISMRM 2020 T1 challenge, Boudreau 2024 (data without a licence: derived numbers only)",
                notes=[f"Measured: {x['used']} of the {x['submissions_in_the_file']} maps of the database (left out: {x['left_out']}); some acquisitions were fitted "
                       "both as complex and as magnitude data. Reference of each map: the challenge's temperature-corrected value for that phantom version.",
                       f"Engine: world `nist_t1`, sequence `ir`, TR {M1_BASE['tr_ms']:g} ms, TE {M1_BASE['te_ms']:g} ms, TI " + " / ".join(f"{v:g}" for v in TI_M1)
                       + f" ms, matrix {M1_BASE['matrix']}, {M1_BASE['slice_mm']:g} mm, {M1_BASE['bandwidth_hz_per_px']:g} Hz per pixel, {M1_SYSTEMS} drawn systems, "
                       "the challenge's estimator, regions of 88 pixels.",
                       "The measured bias by phantom version is a reading of the database; that the reference list of the older units causes it is a hypothesis."])


# ------------------------------------------------------------------------------------------------ M2
FLIPS_M2 = (2.0, 5.0, 10.0, 20.0, 25.0, 30.0)
M2_BASE = dict(sequence="spgr", tr_ms=6.6, te_ms=2.44, fov_mm=250.0, matrix=256, phase_fov=0.75, bandwidth_hz_per_px=280.0, excitation="nonselective", averages=4)


def _ernst(s, flip_rad, tr):
    t0 = fits.vfa_linear(s[None, :], flip_rad, tr)[0]
    t0 = 500.0 if not np.isfinite(t0) else t0
    f = lambda p: p[0] * np.sin(flip_rad) * (1 - np.exp(-tr / p[1])) / (1 - np.exp(-tr / p[1]) * np.cos(flip_rad)) - s   # noqa: E731
    return scipy.optimize.least_squares(f, [s.max() * 10, t0], bounds=([0, 1], [np.inf, 20000])).x[1]


def _vfa(w, tier=None, **over):
    eng = kspace.Engine(w) if tier is None else kspace.Engine(w, tier=tier)
    ims = []
    for a in FLIPS_M2:
        rp = realise(dict(M2_BASE, flip_deg=a, **over), w["console"])
        ims.append(np.abs(eng.acquire(rp, None)[0]))
    ims = np.stack(ims, -1)
    t1 = np.array([_ernst(ims[m].mean(0), D(FLIPS_M2), rp["tr_ms"]) for m in roi_masks(w, rp, 5.0)])
    return t1, np.array([w["tissues"][f"sphere{k + 1}"]["t1_ms"] for k in range(14)]), rp


def rung_m2():
    t = load("nist_tables")
    s1 = np.array(t["table_s1_nicl2"])
    ideal = dict(tx_gain_dev=(0.0, 0.0), b1_shape_amp=(0.0, 0.0), sigma0=0.0)

    def world(dev=0.0):
        w = nist_world(0, s1[:, 2], s1[:, 3], **dict(ideal, tx_gain_dev=(abs(dev), abs(dev))))
        w["system"]["tx_gain"] = 1 + dev
        return w

    t1, truth, rp = _vfa(world())
    e0 = 100 * (t1 / truth - 1)
    under = _vfa(world(-0.10), tier=1)[0][0]                              # ideal spoiling, 10 % under-rotation, sphere 1
    slope = []
    for dev in (-0.10, -0.05, 0.05, 0.10):
        e = 100 * (_vfa(world(dev))[0] / truth - 1)
        slope.append(float(np.mean((e - e0)[:10]) / (100 * dev)))
    alt = {name: 100 * (_vfa(world(), rf_phase_increment_deg=phi, spoiler_mt_ms_per_m=area)[0] / truth - 1)
           for name, phi, area in (("50° / 100 mT·ms/m", 50.0, 100.0), ("169° / 450 mT·ms/m", 169.0, 450.0))}
    worst = float(e0[np.abs(e0).argmax()])
    rows = [
        test("M2.1", "T1 read by variable flip angle when the transmitter under-rotates by 10 %, true T1 2000 ms", "\"approximately 1600 ms\" (Keenan 2021)",
             f"{under:.0f} ms for {truth[0]:.0f} ms (sphere 1, ideal spoiling)", f"{under - 1600 * truth[0] / 2000:+.0f} ms from the scaled figure",
             "1550 to 1650 ms × 1989 / 2000 (the figure is given to two digits)", 1550 <= under * 2000 / truth[0] <= 1650),
        test("M2.2", "relative T1 error per relative flip-angle error, ±5 and ±10 %", "\"approximately twice\" (Keenan 2021)", f"{min(slope):.2f} to {max(slope):.2f} "
             "(spheres 1 to 10, six angles)", f"{100 * (min(slope) / 2 - 1):+.0f} to {100 * (max(slope) / 2 - 1):+.0f} % of the rule", "1.5 to 2.5 (the rule is given to one digit)",
             1.5 <= min(slope) and max(slope) <= 2.5),
        declared("M2.3", "T1 error with a perfect transmitter, from incomplete RF spoiling alone (NiCl2 spheres: T2 is 0.72 T1)",
                 "at 3 T two vendors read too long and one too short (figures of Keenan 2021 and Stupic 2021; no number per vendor)",
                 f"{worst:+.0f} % at the default 117° / 100 mT·ms/m; " + "; ".join(f"{rng_span(v, '{:+.0f}')} % at {k}" for k, v in alt.items()),
                 f"up to {worst:+.0f} % at the default", abs(worst), 10,
                 note="The engine predicts a sign that depends on the vendor's RF phase increment; no measured number per vendor was found, so this is not a test yet."),
    ]
    return dict(rung="M2", title="Variable-flip-angle T1: the published rule and example", rows=rows, extracts=["nist_tables"],
                source="Keenan 2021 (CC0), Stupic 2021 (CC BY): statements in the text; the per-sphere deviations are in figures only",
                notes=[f"Engine: world `nist_t1` with T1 and T2 of Table S1 (3 T), 3D `spgr`, {', '.join(f'{a:g}' for a in FLIPS_M2)} degrees, 280 Hz per pixel, matrix 256; "
                       f"realised TR {rp['tr_ms']:g} ms, TE {rp['te_ms']:g} ms (published {M2_BASE['tr_ms']:g} / {M2_BASE['te_ms']:g}: the console is slower, row M0.6); "
                       "fitted with the nominal angles as both papers do, no B1 map."])


# ------------------------------------------------------------------------------------------------ M2b
def rung_m2b():
    rows = []
    for k, (b0, t1, sd_ref, meas, sd) in enumerate(((1.5, 963.0, 7.0, 372.0, 28.0), (3.0, 1022.0, 10.0, 376.0, 23.0))):     # Dieringer 2014: IR-SE reference, 2D VFA
        got = []
        for t2 in (t1 / 10, t1 / 2):                                # T2 of the doped phantom is not stated: two guesses
            for tier in (1, 2):                                      # ideal spoiling, the engine's spoiling
                s = [abs(kspace.slice_signal(dict(sequence="spgr", tr_ms=5.0, flip_deg=a, spoiler_mt_ms_per_m=100.0, rf_phase_increment_deg=117.0),
                                             dict(t1_ms=t1, t2_ms=t2, adc_um2_ms=2.0), D([a]), 2.0, "hann", tier)[0][0]) for a in (2.0, 13.0)]
                got.append(float(fits.vfa_linear(np.array([s]), D((2.0, 13.0)), 5.0)[0]))
        z = max(abs(g - meas) for g in got) / sd
        rows.append(test(f"M2b.{k + 1}", f"T1 read by two-angle VFA on a 2D slice without profile correction, {b0:g} T (2° / 13°, TR 5 ms, 5 mm, 2 ms sinc of 1 kHz)",
                         f"{meas:.0f} ± {sd:.0f} ms for {t1:.0f} ± {sd_ref:.0f} ms by IR-SE ({100 * (meas / t1 - 1):+.1f} %)",
                         f"{min(got):.0f} to {max(got):.0f} ms ({100 * (min(got) / t1 - 1):+.1f} to {100 * (max(got) / t1 - 1):+.1f} %)",
                         f"{min(got) - meas:+.0f} to {max(got) - meas:+.0f} ms = {z:.1f} SD", "within 2 SD of the measurement", z <= 2.0))
    return dict(rung="M2b", title="Two-angle T1 on a 2D slice without slice-profile correction", rows=rows, extracts=[],
                source="Dieringer 2014, PLoS ONE 9:e91318 (CC BY): numbers of its phantom experiment, typed with the citation",
                notes=["Engine: the slice integral of `spgr` with `rf_pulse=fast` (time-bandwidth 2, Hann window), ideal and default spoiling, T2 = T1 / 10 and T1 / 2 "
                       "(the paper does not state T2); the four results span the range shown."])


# ------------------------------------------------------------------------------------------------ M3b
# Ben-Eliezer 2015 (Magn Reson Med 73:809, PMC4169365; no open licence), Table 1. Kept here: T1 and single-echo T2 of the
# nine tubes, which define the object the train is computed for, and of the measured multi-echo T2 only the mean and
# SD over the tubes of its error (derived numbers). The 27 measured values are not stored.
M3B_T1 = (1424.0, 877.0, 528.0, 363.0, 235.0, 214.0, 155.0, 120.0, 287.0)
M3B_T2 = (116.7, 68.5, 36.9, 23.5, 17.9, 14.4, 11.9, 9.7, 17.8)
M3B_MEASURED = {180: (50.5, 13.2), 150: (55.3, 15.7), 120: (82.6, 20.6)}       # mean, SD over 9 tubes of 100 (T2 multi-echo / T2 single-echo - 1)
M3B = dict(esp=12.0, n=18, tr=1500.0, thk=3.0, width=1.2)


def _train(refocus_deg, tbw, n_tr=3):
    z = np.linspace(-3 * M3B["thk"], 3 * M3B["thk"], 1201)
    wz = np.full(z.size, (z[1] - z[0]) / M3B["thk"])
    exc = pulses.sinc(90.0, 2.0, tbw, thickness_mm=M3B["thk"], kind="excitation", n=128)
    ref = pulses.sinc(float(refocus_deg), 3.0, tbw, thickness_mm=M3B["thk"] * M3B["width"], kind="refocusing", n=128)
    one = [Q.rf_pulse(exc, pos_mm=z, weights=wz)]
    for k in range(M3B["n"]):
        one += [Q.gradient(1, ms=M3B["esp"] / 2), Q.rf_pulse(ref, pos_mm=z, phase_deg=90.0), Q.gradient(1, ms=M3B["esp"] / 2), Q.readout(k + 1)]
    return (one + [Q.spoil(), Q.delay(M3B["tr"] - M3B["n"] * M3B["esp"])]) * n_tr


def rung_m3b():
    te = M3B["esp"] * np.arange(1, M3B["n"] + 1)
    rows, t2 = [], np.array(M3B_T2)
    for k, ang in enumerate((180, 150, 120)):
        mean = {}
        for tbw in (4.0, 2.0):
            blocks = _train(ang, tbw)
            est = [fits.decay(te, np.abs(Q.run(blocks, a, b).signal)[-M3B["n"]:], 30.0)[1] for a, b in zip(M3B_T1, M3B_T2)]      # the echoes of the last repetition
            mean[tbw] = float(np.mean(100 * (np.array(est) / t2 - 1)))
        m, sd = M3B_MEASURED[ang]
        lo, hi = min(mean.values()), max(mean.values())
        rows.append(test(f"M3b.{k + 1}", f"T2 error of an exponential fit to a multi-echo spin echo, refocusing {ang}° (mean of 9 tubes)",
                         f"{m:+.1f} % (SD over the tubes {sd:.1f}) against single-echo spin echo", f"{mean[4.0]:+.1f} % (time-bandwidth 4), {mean[2.0]:+.1f} % (time-bandwidth 2)",
                         f"{mean[4.0] - m:+.1f} and {mean[2.0] - m:+.1f} points", "the measured mean lies between the two pulse shapes (the product pulse is not published)",
                         lo <= m <= hi))
    rows.append(declared("M3b.4", "the same on the scanner (`mrgym scan`)", "a product sequence", "no echo train: the six console sequences have one echo per excitation",
                         "not served", 1.0, 8, note="Rows M3b.1 to M3b.3 are the design layer (`mrgym.design.sequence`), which no console sequence reaches."))
    return dict(rung="M3b", title="T2 from a multi-echo spin echo (design layer; the console has no echo train)", rows=rows, extracts=[],
                source="Ben-Eliezer 2015, Magn Reson Med 73:809, Table 1 (no open licence: the nine tubes' T1 and T2 as the object, and derived means of the measured error)",
                notes=[f"Protocol of the paper: 3 T, {M3B['n']} echoes of {M3B['esp']:g} ms, TR {M3B['tr']:g} ms, {M3B['thk']:g} mm, refocusing slice {M3B['width']:g} × the "
                       "excitation slice. Engine: the block-list EPG of the design layer through the slice profile of the console's windowed-sinc pulses; nothing is fitted."])


# ------------------------------------------------------------------------------------------------ M4 / M8
HAMILTON = ((5.3, 4.7), (4.2, 3.9), (2.75, 0.6), (2.1, 12.0), (1.3, 70.0), (0.9, 8.8))    # Hamilton 2011 (NMR Biomed 24:784), Table 3: ppm, % of the fat signal
M4_T1 = dict(water={1.5: 900.0, 3.0: 1000.0}, oil={1.5: 260.0, 3.0: 340.0})               # declared: the T1 of the emulsions is not published
M4_PROTOCOLS = ((1.5, dict(te_ms=2.30, echo_spacing_ms=2.31, flip_deg=3.0, tr_ms=16.6)), (1.5, dict(te_ms=1.20, echo_spacing_ms=1.98, flip_deg=3.0, tr_ms=14.9)),
                (3.0, dict(te_ms=1.15, echo_spacing_ms=1.17, flip_deg=2.0, tr_ms=9.2)), (3.0, dict(te_ms=1.24, echo_spacing_ms=1.00, flip_deg=2.0, tr_ms=8.0)))


def _vial_world(b0_t, pdff, equal_t1=False):
    n, ext = 512, 256.0
    y, x = anatomy.axes(n, ext)
    ang = 2 * np.pi * np.arange(len(pdff)) / len(pdff)
    cen = [(70 * np.sin(a), 70 * np.cos(a)) for a in ang]
    masks = [anatomy.sig(12.5 - np.hypot(y - cy, x - cx), 0.5) for cy, cx in cen]
    wts = np.stack([sum((1 - f) * m for f, m in zip(pdff, masks)), sum(f * m for f, m in zip(pdff, masks))]).astype(np.float32)
    mk = lambda t1, fat: dict(pd=1.0, t1_ms=t1, t2_ms=60.0, t2s_ms=40.0, t2p_ms=1 / (1 / 40.0 - 1 / 60.0), adc_um2_ms=2.0, fat=fat)   # noqa: E731
    body = wts.sum(0) > 0.5
    ph = dict(kind="brain", b0_t=b0_t, n=n, extent_mm=ext, names=["wat", "oil"], weights=wts,
              tissues=dict(wat=mk(M4_T1["water"][b0_t], False), oil=mk(M4_T1["water" if equal_t1 else "oil"][b0_t], True)), air=[],
              masks=dict(body=body, brain=body, lesion=np.zeros((n, n), np.float32)), size_mm=(200.0, 200.0), rot_deg=0.0)
    return W.flat_world(ph), cen


def _six_echoes(w, cen, prot):
    rp = realise(prot, w["console"], only=("tr_ms", "te_ms", "echo_spacing_ms"))
    im = kspace.Engine(w).acquire(rp, None)
    yy = (np.arange(rp["matrix_pe"]) - rp["matrix_pe"] // 2) * rp["fov_pe_mm"] / rp["matrix_pe"]
    xx = (np.arange(rp["matrix"]) - rp["matrix"] // 2) * rp["fov_mm"] / rp["matrix"]
    return np.array([[im[e][np.hypot(yy[:, None] - cy, xx[None, :] - cx) < 8].mean() for e in range(im.shape[0])] for cy, cx in cen]), np.array(rp["echo_times_ms"])


def _fat_model(p, te_ms, b0, peaks):
    return np.abs(p[0] + p[1] * S.fat_spectrum(te_ms * 1e-3, b0, peaks)) * np.exp(-p[2] * te_ms * 1e-3)


def _fat_fit(sig, te, b0, peaks, starts, r0):
    """Water, fat and one R2* fitted to six magnitudes: (fat fraction, rms residual over the largest echo)."""
    a0 = sig[0] * np.exp(r0 * te[0] * 1e-3)
    best = None
    for f0 in starts:
        sol = scipy.optimize.least_squares(lambda p: _fat_model(p, te, b0, peaks) - sig, [a0 * (1 - f0), a0 * f0, r0],
                                           bounds=([0, 0, 0], [np.inf, np.inf, 1000.0]), x_scale=[a0, a0, 50.0])
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x[1] / (best.x[0] + best.x[1]), float(np.sqrt(np.mean(best.fun ** 2)) / sig.max())


def _fat_fixed(sig, te, b0, peaks, pdff, r0):
    """Nothing fitted but a scale and R2*: the fat fraction is the known one. Rms residual over the largest echo."""
    sol = scipy.optimize.least_squares(lambda p: p[0] * _fat_model([1 - pdff, pdff, p[1]], te, b0, peaks) - sig, [sig[0], r0],
                                       bounds=([0, 0], [np.inf, 1000.0]), x_scale=[sig[0], 50.0])
    return float(np.sqrt(np.mean(sol.fun ** 2)) / sig.max())


def rung_m4_m8():
    x = load("pdff")
    pdff = np.array(x["pdff_percent"]) / 100
    engine = tuple(tissue.FAT_PEAKS)
    shifted = lambda d: tuple((ppm - d, a) for ppm, a in engine)      # noqa: E731  d > 0: fat further from water (a colder phantom)
    models = dict(engine=engine, room=shifted(0.10), one=((tissue.FAT_PPM, 1.0),))
    main = -min(engine, key=lambda p: -p[1])[0]                         # water to the main fat peak, ppm
    # ---- M4: the peak table and the published protocols
    eng_sorted = sorted(engine, key=lambda p: -p[0])
    d_amp = [100 * e[1] - pct for (_, pct), e in zip(HAMILTON, eng_sorted)]
    d_ppm = [e[0] - (ppm - 4.7) for (ppm, _), e in zip(HAMILTON, eng_sorted)]
    k = int(np.abs(d_ppm).argmax())
    bias, slope = [], []
    for b0, p in M4_PROTOCOLS:
        prot = dict(sequence="spgr", n_echoes=6, fov_mm=256.0, matrix=64, bandwidth_hz_per_px=3000.0, excitation="nonselective", **p)
        for equal in (False, True):
            w, cen = _vial_world(b0, pdff, equal)
            sig, te = _six_echoes(w, cen, prot)
            ff = np.array([_fat_fit(np.abs(s), te, b0, engine, (f,), 1e3 / 40.0)[0] for s, f in zip(sig, pdff)])
            (bias if not equal else slope).append((float(np.mean(100 * (ff - pdff))), float(np.polyfit(100 * pdff, 100 * ff, 1)[0])))
    b_eng, b_eq = [v[0] for v in bias], [v[0] for v in slope]
    # ---- M8: the measured echoes
    acq = x["acquisitions"]
    per = {name: [] for name in models}
    shipped, best = [], []
    for a in acq:
        te, b0, sig = np.array(a["te_ms"]), a["field_t"], np.array(a["signal"])
        r0 = np.clip(np.array(a["shipped_r2star_per_s"], float), 1.0, 500.0)
        f0 = np.clip(np.array(a["shipped_fat_fraction"], float), 0.0, 1.0)
        shipped.append(float(np.mean(100 * (np.array(a["shipped_fat_fraction"]) - pdff))))
        for name, peaks in models.items():
            r = [_fat_fit(sig[v], te, b0, peaks, (f0[v], pdff[v]), r0[v]) for v in range(len(pdff))]
            ff = np.array([q[0] for q in r])
            per[name].append(dict(bias=float(np.mean(100 * (ff - pdff))), slope=float(np.polyfit(100 * pdff, 100 * ff, 1)[0]),
                                  fixed=float(np.mean([_fat_fixed(sig[v], te, b0, peaks, pdff[v], r0[v]) for v in range(1, len(pdff))]))))
        grid = np.arange(-0.15, 0.4001, 0.025)                       # the shift the echoes of the vials with 10.5 % fat and more prefer, fat fraction free
        cost = [np.mean([_fat_fit(sig[v], te, b0, shifted(d), (f0[v], pdff[v]), r0[v])[1] ** 2 for v in range(4, len(pdff))]) for d in grid]
        j = int(np.argmin(cost))
        d = grid[j] + (0.5 * 0.025 * (cost[j - 1] - cost[j + 1]) / (cost[j - 1] - 2 * cost[j] + cost[j + 1]) if 0 < j < len(grid) - 1 else 0.0)
        best.append(main + d)
    best = np.array(best)
    fx = {name: 100 * float(np.mean([r["fixed"] for r in v])) for name, v in per.items()}
    b8, s8 = np.array([r["bias"] for r in per["engine"]]), np.array([r["slope"] for r in per["engine"]])
    b1 = np.array([r["bias"] for r in per["one"]])
    n = len(acq)
    pub = x["published"]
    rows = [
        test("M4.1", "share of the six fat peaks (% of the fat signal)", ", ".join(f"{p:g}" for _, p in HAMILTON) + " at " + ", ".join(f"{q:g}" for q, _ in HAMILTON)
             + " ppm (Hamilton 2011, liver, Table 3)", ", ".join(f"{100 * e[1]:.1f}" for e in eng_sorted), f"{rng_span(d_amp)} points",
             "≤ 1 point (declared by this tool: the table gives the shares to 0.1 point and no spread)", np.abs(d_amp).max() <= 1.0),
        declared("M4.2", "position of the six peaks relative to water", "the table's ppm − 4.7", ", ".join(f"{e[0]:+.2f}" for e in eng_sorted) + " ppm",
                 f"{HAMILTON[k][0]:g} ppm peak at {4.7 + eng_sorted[k][0]:.2f} ppm ({d_ppm[k]:+.2f}); the others within {np.sort(np.abs(d_ppm))[-2]:.2f}", np.abs(d_ppm).max(), 11,
                 note="The engine's list is the six-peak model of the Hernando papers, not the table it names."),
        test("M4.3", "fat fraction of the phantom with the published six-echo protocols run through the engine: mean error over the 11 vials",
             f"published {pub['bias_percent']:+.2f} % (95 % interval {pub['bias_ci_percent'][0]:g} to {pub['bias_ci_percent'][1]:g}); per acquisition of the shipped maps "
             f"{rng_span(shipped, '{:+.2f}')} % ({n} acquisitions)", f"{rng_span(b_eng, '{:+.2f}')} % (4 protocols); {rng_span(b_eq, '{:+.2f}')} % with equal T1 of oil and water",
             f"{min(b_eng) - pub['bias_percent']:+.2f} to {max(b_eng) - pub['bias_percent']:+.2f} points from the published mean",
             "inside the range of the acquisitions of the shipped maps", min(shipped) <= min(b_eng) and max(b_eng) <= max(shipped),
             note="All of the engine's error is T1 weighting, and the T1 of the emulsions is declared (water 900 / 1000 ms, oil 260 / 340 ms at 1.5 / 3 T), not published."),
        test("M8.1", "six echo magnitudes of the 10 vials with fat, the known fat fraction fixed, a scale and R2* free: rms residual / largest echo",
             f"{n} acquisitions; the same with the peaks 0.10 ppm further from water {fx['room']:.2f} %, with one peak {fx['one']:.2f} %", f"{fx['engine']:.2f} % with the engine's six peaks",
             f"{fx['engine'] - fx['room']:+.2f} points above the room-temperature model", "≤ 3 % (between the residual of the study's own model and that of one peak)", fx["engine"] <= 3.0),
        test("M8.2", "fat fraction fitted to the measured echoes with the engine's spectrum: mean error per acquisition",
             f"shipped maps (the study's model): mean {np.mean(shipped):+.2f} %, {rng_span(shipped, '{:+.2f}')} %; published {pub['bias_percent']:+.2f} %",
             f"mean {b8.mean():+.2f} %, {rng_span(b8, '{:+.2f}')} %; slope {s8.min():.3f} to {s8.max():.3f}; with one peak {b1.mean():+.2f} %",
             f"{b8.mean() - np.mean(shipped):+.2f} points in the mean", f"mean inside the range of the shipped maps; slope inside the published {pub['slope'][0]:g} to {pub['slope'][1]:g}",
             min(shipped) <= b8.mean() <= max(shipped) and pub["slope"][0] <= s8.min() and s8.max() <= pub["slope"][1],
             note="The spread per acquisition is wider than that of the shipped maps: an error that depends on the echo times, from the 0.09 ppm in row M8.3."),
        declared("M8.3", "water to the main fat peak, the shift the measured echoes fit best", f"{best.mean():.3f} ppm, SD {best.std(ddof=1):.3f}, {rng_span(best, '{:.3f}')} "
                 f"({n} acquisitions at room temperature; 1.5 T {best[[a['field_t'] < 2 for a in acq]].mean():.3f}, 3 T {best[[a['field_t'] > 2 for a in acq]].mean():.3f})",
                 f"{main:.3f} ppm, fixed (body temperature)", f"{main - best.mean():+.3f} ppm = {abs(main - best.mean()) / best.std(ddof=1):.1f} SD", abs(main - best.mean()), 1,
                 note="Hernando 2014 measured −0.01085 ppm per °C; the engine has no temperature. The best-fit shift assumes liver-fat amplitudes for peanut oil."),
    ]
    return dict(rung="M4 / M8", title="Fat: the peak table, the published protocols, six measured echoes", rows=rows, extracts=["pdff"],
                source="Hernando 2017 phantom data (CC0); Hamilton 2011 Table 3 and the abstract of Hernando 2017 (no open licence: twelve and four numbers, typed with the citation)",
                notes=[f"Measured: {n} of the {x['acquisitions_in_the_archive']} acquisitions of the archive (6 sites, 3 vendors, 1.5 and 3 T, 2 protocols); "
                       + (f"{len(x['left_out'])} were left out because 11 vials were not found by the segmentation" if x["left_out"] else "none left out")
                       + ". Vials were told apart by the fat-fraction map shipped with the data.",
                       "Not carried into this tool: the temperature confound of the research pass (a phantom at 20 °C fitted with the 37 °C model), which it reproduced only "
                       "by changing the engine's peak table in the process."])


# ------------------------------------------------------------------------------------------------ M5
ITIS_NAME = dict(wm="Brain (White Matter)", gm="Brain (Grey Matter)", deep_gm="Thalamus", csf="Cerebrospinal Fluid", scalp_fat="SAT (Subcutaneous Fat)",
                 diploe="Bone Marrow (Red)", muscle="Muscle", blood="Blood", liver="Liver", spleen="Spleen", kidney_cortex="Kidney (Cortex)",
                 kidney_medulla="Kidney (Medulla)", pancreas="Pancreas", abd_muscle="Muscle", marrow="Bone Marrow (Red)", subcut_fat="SAT (Subcutaneous Fat)",
                 visceral_fat="SAT (Subcutaneous Fat)", cord="Spinal Cord")
BLOOD_T1_3T = (1664.0, 14.0)                           # Lu 2004 (Magn Reson Med 52:679), abstract: arterial blood at 3 T, 37 degrees C


def tissue_digest() -> str:
    """Digest of the tissue values M5 compares (so that a change of ``tissue.py`` makes the stored M5 stale)."""
    vals = [(name, b0, tissue.table(b0)[name]["t1_ms"], tissue.table(b0)[name]["t2_ms"]) for name in tissue.NAMES for b0 in (1.5, 3.0)]
    return hashlib.sha256(json.dumps(vals).encode()).hexdigest()[:16]


def rung_m5(db):
    counts = dict(inside=0, outside=0, single=0, none=0)
    out = []
    for name in tissue.NAMES:
        for b0 in (1.5, 3.0):
            for q, key in (("t1", "t1_ms"), ("t2", "t2_ms")):
                v, c = tissue.table(b0)[name][key], (db.get(ITIS_NAME.get(name, "")) or {}).get((q, b0))
                if c is None:
                    counts["none"] += 1
                elif c["n"] <= 1 or c["lo"] == c["hi"]:
                    counts["single"] += 1
                else:
                    inside = c["lo"] <= v <= c["hi"]
                    counts["inside" if inside else "outside"] += 1
                    if not inside:
                        out.append(f"{name} {q} {b0:g} T: {v:g} against {c['lo']:g} to {c['hi']:g}")
    n = sum(counts.values())
    wm = db[ITIS_NAME["wm"]][("t1", 1.5)]
    blood = tissue.table(3.0)["blood"]["t1_ms"]
    rows = [
        test("M5.1", f"the {n} values of the tissue table (T1, T2 at 1.5 and 3 T) against a compilation of studies",
             f"{counts['inside'] + counts['outside']} values have a range from two or more studies; {counts['single']} have one study; {counts['none']} have none",
             f"{counts['inside']} inside the range, {counts['outside']} outside" + (": " + "; ".join(out) if out else ""), f"{counts['outside']} outside",
             "0 outside the range of the studies", counts["outside"] == 0,
             note=f"The ranges are wide (white-matter T1 at 1.5 T: {wm['lo']:.0f} to {wm['hi']:.0f} ms over {wm['n']} studies), and the compilation shares sources with "
                  "the table: where it holds one study the comparison would be circular and is not counted."),
        declared("M5.2", "T1 of blood at 3 T", f"{BLOOD_T1_3T[0]:.0f} ± {BLOOD_T1_3T[1]:.0f} ms (arterial blood at a haematocrit of 0.42; Lu 2004, abstract)",
                 f"{blood:g} ms (the table's source tag: an in-vitro value)",
                 f"{100 * (blood / BLOOD_T1_3T[0] - 1):+.0f} %", abs(100 * (blood / BLOOD_T1_3T[0] - 1)), None),
    ]
    return dict(rung="M5", title="The tissue table against a literature compilation", rows=rows, extracts=[], fetch_only=True, tissue_digest=tissue_digest(),
                source="IT'IS Foundation tissue properties database, relaxation times (licence unclear: nothing from it is stored; this rung runs with --fetch only)",
                notes=["The rows were computed when the database was last fetched; without --fetch they are shown as stored, and they go stale when the tissue table changes."])


# ------------------------------------------------------------------------------------------------ M6
M6_SEEDS = 6


def _acr_engine(field):
    import extract
    out = []
    for s in range(M6_SEEDS):
        sp, grads = qa.system_spec(field, s)
        im = {}
        for sec in (1, 5, 7):
            eng = qa.engine_of(sp, grads, sec, s)
            im[("t1", sec)] = qa.scan(eng, "t1", sec, s)
            if sec == 7:
                im[("t2", 7)] = qa.scan(eng, "t2", 7, s)
        t1 = extract.acr_measures(im[("t1", 7)][0], im[("t1", 7)][1], im[("t1", 1)][0], im[("t1", 5)][0])
        t2 = extract.acr_measures(im[("t2", 7)][0], im[("t2", 7)][1])
        out.append(dict(t1=t1, snr_ratio=t2["snr"] / t1["snr"]))
    return out


def rung_m6():
    x = load("acr")
    ser = x["series"]
    t1s = {k: v for k, v in ser.items() if v["tr_ms"] < 1000}
    by = lambda f: [v for v in t1s.values() if v["field_t"] == f]         # noqa: E731
    eng = {f: _acr_engine(f) for f in (1.5, 3.0)}
    e = lambda f, k: np.array([r["t1"][k] for r in eng[f]])               # noqa: E731
    chords = lambda v: [c for c in v["d1"][1:] + v["d5"] if c > 180]      # noqa: E731  not the top-bottom chord of slice 1 (the wedges lie on it), nor a chord cut short
    cut1 = sum(v["d1"][0] <= 180 for v in t1s.values())
    cut5 = sum(any(c <= 180 for c in v["d5"]) for v in t1s.values())
    rows = []
    bws = [float(round(v["bandwidth_hz_per_px"])) for v in t1s.values()]
    mode = max(set(bws), key=bws.count)
    rows.append(test("M6.1", "bandwidth of the ACR T1 series (declared in the package)", f"{mode:g} Hz per pixel on {bws.count(mode)} of {len(bws)} scanners; the others "
                     + ", ".join(f"{b:g}" for b in sorted(set(bws)) if b != mode), f"{qa.SERIES['t1']['bandwidth_hz_per_px']:g} Hz per pixel", "equal to the commonest",
                     "the commonest measured value", qa.SERIES["t1"]["bandwidth_hz_per_px"] == mode))
    dd = np.array([c - qa.A["diameter_mm"] for v in t1s.values() for c in chords(v)])
    worst = max(abs(c - qa.A["diameter_mm"]) for f in eng for r in eng[f] for c in r["t1"]["d1"] + r["t1"]["d5"])
    rows.append(test("M6.2", "diameter error, slices 1 and 5", f"{rng_span(dd)} mm, mean {dd.mean():+.2f}, SD {dd.std(ddof=1):.2f} ({dd.size} chords, {len(t1s)} scanners)",
                     f"largest error {worst:.2f} mm ({M6_SEEDS} healthy systems per field)", f"{worst / dd.std(ddof=1):.1f} SD of the real chords",
                     "largest error ≤ 2 SD of the real chords", worst <= 2 * dd.std(ddof=1),
                     note="Real images spread three to four times wider than the engine's healthy gradient error; they carry distortion correction and edge shading."))
    for k, f in enumerate((1.5, 3.0)):
        piu = np.array([v["piu"] for v in by(f)])
        ep = e(f, "piu")
        text = (f"percent integral uniformity, T1 series, {f:g} T", f"{rng_span(piu, '{:.1f}')} (median {np.median(piu):.1f}; {piu.size} scanners)",
                f"{ep.mean():.1f} ({rng_span(ep, '{:.1f}')})")
        if piu.min() <= ep.mean() <= piu.max():
            rows.append(test(f"M6.{3 + k}", *text, "inside", "inside the range of the real scanners", True))
        else:
            rows.append(declared(f"M6.{3 + k}", *text, f"{ep.mean() - piu.max():+.1f} points above the most uniform scanner", ep.mean() - piu.max(), 4))
    for k, (f, known) in enumerate(((1.5, 4), (3.0, 5))):
        cr = np.array([v["centre_over_rim"] for v in by(f)])
        ec = e(f, "centre_over_rim").mean()
        low = min(by(f), key=lambda v: v["centre_over_rim"])
        extra = "" if f == 1.5 else f"; lowest {low['centre_over_rim']:.2f} on the {low['vendor'].capitalize()} scanner, a receive array"
        size = max(cr.min() - ec, ec - cr.max(), 0.0) if f == 1.5 else ec - cr.min()
        rows.append(declared(f"M6.{5 + k}", f"signal at the centre over the signal near the rim, slice 7, {f:g} T", f"{rng_span(cr, '{:.2f}')} ({cr.size} scanners){extra}", f"{ec:.2f}",
                             (f"{size:.3f} outside the real range" if f == 1.5 else f"{size:+.2f} against the receive array: the sign is wrong for an array"), size, known))
    gh = np.array([v["ghost"] for v in t1s.values()])
    eg = np.concatenate([e(f, "ghost") for f in eng])
    rows.append(test("M6.7", "ghosting ratio, slice 7", f"{rng_span(gh, '{:.4f}')} ({gh.size} scanners)", f"{eg.mean():.4f} (largest {eg.max():.4f})", "inside",
                     "inside the range of the real scanners", gh.min() <= eg.mean() <= gh.max()))
    snr = {f: sorted((v["vendor"], v["coil"] or "", v["snr"]) for v in by(f) if v["snr"]) for f in (1.5, 3.0)}
    head = lambda f: [s for vnd, _, s in snr[f] if vnd == "GE"]            # noqa: E731
    array = lambda f: [s for vnd, _, s in snr[f] if vnd.upper() == "SIEMENS"]                  # noqa: E731
    es = {f: e(f, "snr").mean() for f in eng}
    ratio = max(max(array(f)) / es[f] for f in eng if array(f))
    rows.append(declared("M6.8", "SNR of the T1 series, slice 7 (signal of the large region over the corner noise)",
                         "; ".join(f"{f:g} T: GE (coil HEAD) " + ", ".join(f"{s:.0f}" for s in head(f)) + "; Siemens " + ", ".join(f"{s:.0f}" for s in array(f)) for f in snr),
                         "; ".join(f"{f:g} T: {es[f]:.0f}" for f in es), f"the Siemens scanners up to × {ratio:.1f} the engine", ratio, 2,
                         note="The noise level of the QA system is that of the GE scanners with the coil HEAD. Of the Siemens series one names its coil (HeadNeck_20_TCS); "
                              "the research pass calls them 20-channel arrays. Corner noise is not valid for arrays or masked backgrounds (the two Philips series gave "
                              "no value)."))
    lo, hi = head(3.0)[0] / max(head(1.5)), head(3.0)[0] / min(head(1.5))
    er = es[3.0] / es[1.5]
    rows.append(test("M6.9", "SNR at 3 T over SNR at 1.5 T, same vendor", f"{head(3.0)[0] / np.mean(head(1.5)):.2f} (GE; {lo:.2f} to {hi:.2f} over its three 1.5 T scanners), "
                     f"{array(3.0)[0] / np.mean(array(1.5)):.2f} (Siemens)", f"{er:.2f}", f"{100 * (er / (head(3.0)[0] / np.mean(head(1.5))) - 1):+.0f} %",
                     "inside the range over the three 1.5 T scanners of one vendor", lo <= er <= hi))
    pairs = [k[:-3] for k in ser if k.endswith("_T2") and k[:-3] + "_T1" in ser]
    pr = np.array([ser[p + "_T2"]["snr"] / ser[p + "_T1"]["snr"] for p in pairs])
    epr = float(np.mean([r["snr_ratio"] for f in eng for r in eng[f]]))
    rows.append(test("M6.10", "SNR of the T2 series over SNR of the T1 series, slice 7", ", ".join(f"{v:.3f}" for v in np.sort(pr)) + f" ({pr.size} scanners)", f"{epr:.3f}", "inside",
                     "inside the range of the real scanners", pr.min() <= epr <= pr.max(),
                     note="The declared relaxation times of the ACR solution (T1 150 ms, T2 130 ms) set this ratio."))
    off = np.array([float(np.hypot(*v["offset_mm"])) for v in t1s.values()])
    eoff = max(float(np.hypot(*r["t1"]["offset_mm"])) for f in eng for r in eng[f])
    rows.append(declared("M6.11", "distance of the phantom from the centre of the image", f"{rng_span(off, '{:.1f}')} mm; {int((off > 3).sum())} of {off.size} scanners above 3 mm",
                         f"up to {eoff:.1f} mm (drawn within ±{anatomy.ACR_SPEC['offset_mm']:g} mm per axis)", f"{off.max() - eoff:.0f} mm of real positioning not drawn", off.max() - eoff, 6))
    o = ser[x["off_centre_slice"]["series"]]
    rows.append(test("M6.13", f"ghosting ratio of the real image whose phantom lies {np.hypot(*o['offset_mm']):.0f} mm off centre, measured by `qa.ghosting`",
                     f"{o['ghost']:.4f} with the regions placed round the phantom (the figure of the research pass, found by re-centring the image by hand)",
                     f"{o['ghost_regions_from_isocentre']:.2f} before this branch (regions placed from the isocentre, one of them on the phantom)", "fixed here",
                     "the function returns the hand-computed figure to 4 decimals", round(o["ghost"], 4) == 0.0016 and o["ghost_regions_from_isocentre"] > 0.1))
    rows.insert(-1, declared("M6.12", "structures of the phantom the real images show and the world does not draw", "vial and slice-thickness insert (slices 1 to 4), positioning "
                             "wedges (slices 1 and 11), holder ring of the low-contrast sheet (slices 8 to 11)", "none of the four", "4 structures absent", 4.0, 12,
                             note=f"Seen on the real images by the research pass. Row M6.2 leaves out the top-bottom chord of slice 1 on the real images, where the "
                                  f"wedges lie on it (on {cut1} of the {len(t1s)} scanners it is more than 10 mm short), and {cut5} chord of slice 5 that is cut short "
                                  "likewise."))
    return dict(rung="M6", title="ACR phantom: real images measured with the package's own QA functions", rows=rows, extracts=["acr"],
                source="hazen project, tests/data/acr (Apache-2.0): derived figures, and the pixels of one slice",
                notes=[f"Measured: {x['files']} DICOM files, {len(ser)} series (T1 and T2 axial) of {x['scanners']} scanners (GE, Philips, Siemens; 1.5 and 3 T), slices 1, 5 and 7. "
                       f"Engine: world `acr`, the series of `mrgym.qa` (spin echo 500 / 20 and 2000 / 80 ms, matrix 256, 250 mm, 5 mm), {M6_SEEDS} healthy systems per field.",
                       "The provenance of the series is unknown: scanner state, image filters, coil combination, and whether a set was chosen because it fails a test."])


# ------------------------------------------------------------------------------------------------ M7
def rung_m7():
    x = load("gel")
    ir, se, pub = x["ir"], x["se"], x["published"]
    ti, s = np.array(ir["times_ms"]), fits.restore_polarity(ir["times_ms"], ir["mean"])
    a, b, t1, res = fits.ir_three(ti, s, starts=(800.0,))
    t1_px, t1_sd = ir["t1_pixelwise_ms_mean_sd"]
    te, s2 = np.array(se["times_ms"]), np.array(se["mean"])
    s0, t2, res2 = fits.decay(te, s2, 80.0)
    q = -b / a                                                            # periodic inversion recovery with a saturating read-out: S = M (1 - (1 + k) exp(-TI / T1) + k exp(-TR / T1))
    kappa = (q - 1) / (1 - q * np.exp(-ir["tr_ms"] / t1))
    rms, worst = 100 * float(np.sqrt(np.mean(res ** 2)) / abs(a)), 100 * float(np.abs(res).max() / abs(a))
    worst2 = 100 * float(np.abs(res2).max() / s0)
    snr = s[-1] / ir["corner_noise_sd"]
    # the engine: the same protocol on a gel disc of the measured T1 and T2
    radius, fov = ir["phantom_radius_mm"], ir["pixel_mm"] * ir["matrix"]
    ph = W.shapes_phantom([("wm", lambda yy, xx: np.hypot(yy, xx) - radius)], b0_t=3.0, n=384, extent_mm=256.0,
                          tissues=dict(wm=dict(pd=1.0, t1_ms=float(t1), t2_ms=float(t2), t2s_ms=60.0, t2p_ms=1 / (1 / 60.0 - 1 / t2), adc_um2_ms=2.0, fat=False)))
    eng_res, eng_snr = [], None
    for system in (dict(), dict(sigma0=W.DEFAULT_SPEC["sigma0"], inv_eff=0.95, tx_gain=1.08)):
        w = W.flat_world(ph, **system)
        w["console"]["fov_mm"] = fov
        e = kspace.Engine(w)
        rng = np.random.default_rng(3) if system else None
        sig = []
        for t in ti:
            rp = realise(dict(sequence="ir", tr_ms=ir["tr_ms"], te_ms=ir["te_ms"], ti_ms=float(t), fov_mm=fov, matrix=ir["matrix"], slice_mm=ir["slice_mm"],
                              bandwidth_hz_per_px=ir["bandwidth_hz_per_px"]), w["console"])
            im = np.abs(e.acquire(rp, rng)[0])
            n = im.shape[0]
            yy, xx = np.mgrid[:n, :n]
            sig.append(im[np.hypot(yy - n // 2, xx - n // 2) * fov / n < 0.6 * radius].mean())
        ae, _, _, re = fits.ir_three(ti, fits.restore_polarity(ti, sig), starts=(800.0,))
        eng_res.append(100 * float(np.sqrt(np.mean(re ** 2)) / abs(ae)))
        if system:
            eng_snr = sig[-1] / e.noise_sd(rp)
    lo, hi = W.DEFAULT_SPEC["inv_eff"]
    rows = [
        test("M7.1", "the reading: T1 and T2 fitted here to the stored signals against the values the article fitted to the same images",
             f"article: {pub['t1_ms'][0]:g} ± {pub['t1_ms'][1]:g} ms, {pub['t2_ms'][0]:g} ± {pub['t2_ms'][1]:g} ms", f"this tool: {t1_px:.1f} ± {t1_sd:.1f} ms pixel by pixel "
             f"({t1:.1f} ms from the mean of the region), {t2:.1f} ms (no engine in this row)", f"{t1_px - pub['t1_ms'][0]:+.1f} ms, {t2 - pub['t2_ms'][0]:+.1f} ms",
             "within 2 SD of the article's values", abs(t1_px - pub["t1_ms"][0]) <= 2 * pub["t1_ms"][1] and abs(t2 - pub["t2_ms"][0]) <= 2 * pub["t2_ms"][1]),
        test("M7.2", "inversion recovery against the form the engine serves, a + b exp(−TI / T1): residual over the asymptote, 13 TI",
             f"rms {rms:.2f} %, largest {worst:.2f} % (product IR-TSE, {ir['echo_train']} echoes per excitation)", f"exact: {eng_res[0]:.2f} % without noise, {eng_res[1]:.2f} % with",
             f"{rms:.2f} % rms", f"≤ {100 * pub['t1_ms'][1] / pub['t1_ms'][0]:.2f} % (the relative uncertainty the article gives for T1)", rms <= 100 * pub["t1_ms"][1] / pub["t1_ms"][0]),
        test("M7.3", "spin-echo decay against one exponential, TE 7.5 to 250 ms: largest residual over S0, 10 TE", f"{worst2:.2f} %", "one exponential (T1 of the gel is short "
             "against TR)", f"{worst2:.2f} %", f"≤ {100 * pub['t2_ms'][1] / pub['t2_ms'][0]:.2f} % (the relative uncertainty the article gives for T2)",
             worst2 <= 100 * pub["t2_ms"][1] / pub["t2_ms"][0]),
        test("M7.4", "inversion efficiency, from b / a of the measured curve", f"{kappa:.3f}", f"drawn from {lo:g} to {hi:g}", "inside", "inside the range the engine draws",
             lo <= kappa <= hi, note="Assumes a read-out that saturates; the sequence has four echoes per excitation. Stupic 2021 reports about 0.85 on its scanner."),
        declared("M7.5", f"SNR at the last TI, {ir['pixel_mm']:.0f} × {ir['pixel_mm']:.0f} × {ir['slice_mm']:g} mm, {ir['bandwidth_hz_per_px']:g} Hz per pixel",
                 f"{snr:.0f} ({ir['coil']} coil, corner noise)", f"{eng_snr:.0f} at the default noise", f"default noise × {snr / eng_snr:.1f} too high for this coil", snr / eng_snr, 2),
    ]
    return dict(rung="M7", title="Gel sphere: product inversion-recovery and spin-echo series", rows=rows, extracts=["gel"], source="Chen 2025, data on Zenodo (CC BY 4.0)",
                notes=[f"Measured: {ir['scanner']}, {ir['field_t']:g} T; inversion recovery TR {ir['tr_ms']:g} ms, TE {ir['te_ms']:g} ms, {len(ti)} TI from {ti[0]:g} to {ti[-1]:g} ms; "
                       f"spin echo TR {se['tr_ms']:g} ms, {len(te)} TE; mean of a central disc of {ir['pixels']} pixels. Engine: sequence `ir` with the same TR, TI, matrix, "
                       "slice and bandwidth on a disc with the fitted T1 and T2."])


# ------------------------------------------------------------------------------------------------ M9
M9_FLOOR = 0.04                                        # of the brightest sphere: four times the signal that real scans leave inside a sphere that has decayed


def _usable(s):
    """Echoes that are fitted: above 5 % of the sphere's first echo and above ``M9_FLOOR`` of the brightest sphere
    (signals in units of the brightest sphere)."""
    s = np.asarray(s)
    return (s > 0.05 * s[0]) & (s > M9_FLOOR)


def _t2_of(te, s):
    """(S0, T2, residuals / S0) of the usable echoes, or None when fewer than three are usable."""
    ok = _usable(s)
    if ok.sum() < 3:
        return None
    s0, t2, res = fits.decay(te[ok], np.asarray(s)[ok])
    return s0, t2, res / s0


def _kappa(a, b, t1, tr):
    """Inversion efficiency from the fitted a + b exp(-TI / T1) of a periodic inversion recovery with a saturating read-out."""
    q = -b / a
    return (q - 1) / (1 - q * np.exp(-tr / t1))


def _m9_world(coef, t1, t2, inv_eff):
    ph = anatomy.nist_t2(np.random.default_rng(0), 3.0, spec=dict(offset_mm=0.0, rotation_deg=0.0))
    for k in range(14):
        c = ph["tissues"][f"sphere{k + 1}"]
        c["t1_ms"], c["t2_ms"] = float(t1[k]), float(t2[k])
        c["t2s_ms"] = 1.0 / (1.0 / c["t2_ms"] + 1.0 / c["t2p_ms"])
    w = W.flat_world(ph, inv_eff=float(inv_eff))
    if coef is not None:                                               # the measured B1: its second-order fit in plate coordinates (v = image y, u = image x)
        y, x = anatomy.axes(ph["n"], ph["extent_mm"])
        v, u = y / 100.0, x / 100.0
        w["fields"]["b1"] = np.clip(coef[0] + coef[1] * v + coef[2] * u + coef[3] * v * v + coef[4] * v * u + coef[5] * u * u, 0.3, 1.7)
    return w


def _m9_scan(w, fov, te_list, ti_list):
    """Region signals of the engine for the two series, in units of the brightest spin-echo mean: ([sphere, TE], [sphere, TI])."""
    eng = kspace.Engine(w)
    cons = dict(w["console"], fov_mm=fov)
    se, ir = [], []
    for te in te_list:
        rp = realise(dict(sequence="se", tr_ms=8000.0, te_ms=float(te), fov_mm=fov, matrix=256, slice_mm=5.0), cons)
        se.append(eng.acquire(rp, None)[0])
    for ti in ti_list:
        rp = realise(dict(sequence="ir", tr_ms=8000.0, te_ms=15.0, ti_ms=float(ti), fov_mm=fov, matrix=256, slice_mm=5.0), cons)
        ir.append(eng.acquire(rp, None)[0])
    masks = roi_masks(w, rp, 5.0)
    ref = ir[-1]
    sig_se = np.array([[abs(im[m].mean()) for im in se] for m in masks])
    sig_ir = np.array([[(im[m] * np.conj(ref[m]) / np.abs(ref[m])).mean().real for im in ir] for m in masks])
    return sig_se / sig_se.max(), sig_ir / sig_se.max()


def rung_m9():
    x, tab = load("scholand"), load("nist_tables")
    st, sp = x["stated"], x["spheres"]
    te, ti, tr = np.array(st["te_ms"]), np.array(st["ti_ms"]), st["tr_ms"]
    s2 = np.array(tab["table_s2_mncl2"])
    cert_t1, cert_t2 = s2[:, 2], s2[:, 3]
    coeff = tab["temperature_coefficient_mncl2_3t_percent_per_k"]
    warm = st["temperature_c"] - 20.0                                    # the certificate is at 20 degrees C
    ref_t1, ref_t2 = cert_t1 * (1 + coeff["t1"] * warm / 100), cert_t2 * (1 + coeff["t2"] * warm / 100)
    m_se, m_ir = np.array([s["se_mean"] for s in sp]), np.array([s["ir_mean"] for s in sp])
    fit_se = [_t2_of(te, s) for s in m_se]
    nine = [k for k in range(14) if fit_se[k] is not None]               # the spheres whose T2 can be fitted
    m_t2 = np.array([f[1] if f else np.nan for f in fit_se])
    fit_ir = [fits.ir_three(ti, s) for s in m_ir]
    m_t1 = np.array([f[2] for f in fit_ir])
    kappa = np.array([_kappa(f[0], f[1], f[2], tr) for f in fit_ir])
    res_ir = np.array([100 * f[3] / abs(f[0]) for f in fit_ir])          # [sphere, TI], % of the asymptote
    fov = x["plate"]["fov_mm"]
    cons = dict(fov_mm=fov, min_pixel_mm=0.8, b1rms_limit_ut=1e3, b0_t=3.0)
    served = np.array([console.plan(dict(sequence="se", tr_ms=tr, te_ms=float(v), fov_mm=fov, matrix=256, slice_mm=5.0), cons)["accept"] for v in te])
    coef = x["b1"]["polynomial"]["coefficients"]
    # (a) the object as measured: T1 and T2 of each sphere as fitted to the measurement (the published T2 where none can be fitted), the measured B1
    e_se, e_ir = _m9_scan(_m9_world(coef, m_t1, np.where(np.isfinite(m_t2), m_t2, cert_t2), np.median(kappa[nine])), fov, te[served], ti)
    d_se, n_cmp = {}, 0
    for k in nine:
        ok = _usable(m_se[k]) & served
        es0 = _t2_of(te[served], e_se[k])[0]
        d_se[k] = 100 * (m_se[k][ok] / fit_se[k][0] - e_se[k][_usable(m_se[k])[served]] / es0)
        n_cmp += int(ok.sum())
    w_se = max(float(np.abs(v).max()) for v in d_se.values())
    long_t2 = [k for k in nine if _usable(m_se[k]).all()]                # spheres that are fitted at every TE
    common = np.mean([100 * fit_se[k][2] for k in long_t2], 0)           # what they share at one echo time: the amplitude of that scan
    rep_se = float(common.std(ddof=1))
    rep_ir = float(res_ir[nine].mean(0).std(ddof=1))
    rms_ir = np.sqrt((res_ir[nine] ** 2).mean(1))
    e_fit = [fits.ir_three(ti, s) for s in e_ir]
    e_rms = max(float(np.sqrt(np.mean((100 * f[3] / abs(f[0])) ** 2))) for f in e_fit)
    # (b) the object as published: Table S2 in the world, with the measured B1 and without it
    p_se, p_ir = _m9_scan(_m9_world(coef, cert_t1, cert_t2, 1.0), fov, te[served], ti)
    q_se, q_ir = _m9_scan(_m9_world(None, cert_t1, cert_t2, 1.0), fov, te[served], ti)
    t2_of = lambda sig: np.array([f[1] if f else np.nan for f in (_t2_of(te[served], s) for s in sig)])     # noqa: E731
    e_t2, b_t2 = t2_of(p_se), t2_of(q_se)
    e_t1, b_t1 = (np.array([fits.ir_three(ti, s)[2] for s in sig]) for sig in (p_ir, q_ir))
    dev_t2, dev_t1 = 100 * (m_t2 / ref_t2 - 1), 100 * (m_t1 / ref_t1 - 1)
    eng_t2, eng_t1 = 100 * (e_t2 / cert_t2 - 1), 100 * (e_t1 / cert_t1 - 1)
    odd = [k for k in nine if abs(dev_t2[k]) > 20]                       # spheres of this unit that do not hold the published solution
    rest = [k for k in nine if k not in odd and np.isfinite(e_t2[k])]
    short = [k for k in range(14) if k not in nine]
    b1 = np.array([s["b1"] for s in sp])
    both = np.isfinite(e_t2) & np.isfinite(b_t2)
    b1_effect = 100 * max(float(np.abs(e_t2[both] / b_t2[both] - 1).max()), float(np.abs(e_t1 / b_t1 - 1).max()))
    gone = [k for k in short if cert_t2[k] < 20.0]                        # at TE 95 ms and longer their own signal is below 0.3 % of its start
    dark = m_se[gone][:, te >= 95.0]
    e_dark = e_se[gone][:, te[served] >= 95.0]
    lo, hi = W.DEFAULT_SPEC["inv_eff"]
    section = 2 * float(np.sqrt((anatomy.NIST_T2["shell_mm"] / 2) ** 2 - anatomy.NIST_T2["plate_y_mm"] ** 2))
    name = lambda ks: ", ".join(str(k + 1) for k in ks)                  # noqa: E731
    rows = [
        test("M9.1", f"spin-echo signal against TE, measured minus served, each scaled by its own fitted amplitude: spheres {name(nine)}, the echoes that are fitted and "
             f"served ({n_cmp} values, TE {te[served][0]:g} to {te[served][-1]:g} ms)", f"repeatability from scan to scan {rep_se:.2f} % of the amplitude (what the "
             f"spheres fitted at all 12 TE share at each TE; each TE is a scan of its own)", "served with the T2 fitted to each sphere and the measured B1",
             f"largest {w_se:.2f} %", f"≤ 3 × the repeatability = {3 * rep_se:.2f} %", w_se <= 3 * rep_se,
             note=f"The console's TE ends at 300 ms: {int((~served).sum())} of the {te.size} measured echo times are not served."),
        test("M9.2", f"inversion recovery against the form the engine serves, a + b exp(−TI / T1), spheres {name(nine)}, {ti.size} TI from {ti[0]:g} to {ti[-1]:g} ms: "
             "rms residual over the asymptote", f"{rng_span(rms_ir, '{:.2f}')} % per sphere; repeatability from scan to scan {rep_ir:.2f} %",
             f"exact: {e_rms:.3f} % on the engine's own images", f"largest {rms_ir.max():.2f} %", f"≤ 3 × the repeatability = {3 * rep_ir:.2f} %", rms_ir.max() <= 3 * rep_ir),
        declared("M9.3", f"spheres of this unit that do not hold the published solution: T2 (and T1) of sphere{'s' if len(odd) > 1 else ''} {name(odd)}",
                 "; ".join(f"sphere {k + 1}: {m_t2[k]:.0f} ms ({m_t1[k]:.0f} ms)" for k in odd), "; ".join(f"{cert_t2[k]:g} ms ({cert_t1[k]:g} ms)" for k in odd) + " in Table S2",
                 "; ".join(f"{dev_t2[k]:+.0f} % ({dev_t1[k]:+.0f} %)" for k in odd), max(abs(dev_t2[k]) for k in odd), None,
                 note="A property of the measured phantom, not of the engine: the reference values that the authors of the data set list for these two spheres agree with "
                      "the fits here (T2 1.45 s and 0.175 s). The engine's plate is the published one."),
        declared("M9.4", f"T2 of the other fitted spheres ({name(rest)}): fitted value against the published one",
                 f"{rng_span(dev_t2[rest])} % of Table S2 at {st['temperature_c']:g} °C (median {np.median(dev_t2[rest]):+.1f} %)", f"{rng_span(eng_t2[rest], '{:+.2f}')} % (the "
                 "same estimator on the engine's images of the published plate)", f"a spread of {np.ptp(dev_t2[rest]):.1f} points the engine does not have", np.abs(dev_t2[rest]).max(), 7,
                 note=f"Reference at {st['temperature_c']:g} °C = Table S2 (20 °C) × (1 + {coeff['t2']:g} % per °C for T2, {coeff['t1']:g} % for T1), the coefficients the "
                      "phantom paper gives for this array; the article of the data set states the room temperature as about 21 °C."),
        declared("M9.5", f"T1 of the same spheres ({name(rest)}): fitted value against the published one",
                 f"{rng_span(dev_t1[rest])} % of Table S2 at {st['temperature_c']:g} °C (median {np.median(dev_t1[rest]):+.1f} %); spheres {name(short[:-1])}: "
                 f"{rng_span(dev_t1[short[:-1]])} %", f"{rng_span(eng_t1[rest], '{:+.2f}')} %", f"a spread of {np.ptp(dev_t1[rest]):.1f} points the engine does not have",
                 np.abs(dev_t1[rest]).max(), 7),
        declared("M9.6", "inversion efficiency, from b / a of the measured curve of each sphere",
                 f"{kappa[nine[0]]:.2f} to {kappa[nine[-1]]:.2f} over spheres {nine[0] + 1} to {nine[-1] + 1} (T2 {m_t2[nine[0]]:.0f} to {m_t2[nine[-1]]:.0f} ms), then "
                 + ", ".join(f"{kappa[k]:.2f}" for k in short[:-1]) + f" for spheres {name(short[:-1])} (published T2 {cert_t2[short[0]]:.0f} to {cert_t2[short[-2]]:.0f} ms)",
                 f"one value per session for every tissue, drawn from {lo:g} to {hi:g}", f"falls by {kappa[nine].max() - kappa[short[:-1]].min():.2f} with T2; the engine's does not",
                 kappa[nine].max() - kappa[short[:-1]].min(), 13, note=f"Median of spheres {name(nine)}: {np.median(kappa[nine]):.3f}; {int(((kappa[nine] >= lo) & (kappa[nine] <= hi)).sum())} "
                 "of them lie inside the range the engine draws. Relaxation during the inversion pulse is the usual cause; the pulse of this sequence was not looked up."),
        declared("M9.7", f"signal left inside a sphere that has decayed (spheres {name(gone)}, TE 95 ms and longer), in % of the brightest sphere",
                 f"median {100 * np.median(dark):.1f} %, {rng_span(100 * dark, '{:.1f}')} %; alike in the {len(gone)} spheres at one TE, different from TE to TE",
                 f"{rng_span(100 * e_dark, '{:.2f}')} %", f"{100 * (np.median(dark) - np.median(e_dark)):.1f} points of artefact level the healthy engine does not have",
                 100 * (np.median(dark) - np.median(e_dark)), 14, note="Ringing and ghosts of the bright water round the spheres. It biases T2 of a dark sphere upwards, which is why "
                 f"echoes below {100 * M9_FLOOR:g} % of the brightest sphere are not fitted on either side."),
        test("M9.8", "relative B1 at the 14 spheres", f"{rng_span(b1, '{:.2f}')} (second-order fit of the map in the water round the spheres, rms residual "
             f"{x['b1']['polynomial']['rms_residual']:.3f})", f"with this B1 in the world the fitted T1 and T2 of the published plate move by at most {b1_effect:.2f} %",
             f"{b1_effect:.2f} %", "≤ 0.5 % (a single-echo spin echo and an inversion recovery at TR 8 s do not depend on B1 to first order)", b1_effect <= 0.5),
    ]
    return dict(rung="M9", title="NIST T2 plate from raw k-space: spin echo at 12 TE, inversion recovery at 11 TI, B1 map", rows=rows, extracts=["scholand", "nist_tables"],
                source="Scholand 2023, data on Zenodo (CC BY 4.0)",
                notes=[f"Measured: {st['scanner']}; TR {tr:g} ms; {x['reconstruction']}.",
                       f"Field of view: the description of the data set says {st['fov_mm']:g} mm. The places of the 14 spheres (Table S2) give {fov:.1f} mm "
                       f"({100 * (fov / st['fov_mm'] - 1):+.1f} %), the B1 scan gives {x['b1']['pixel_mm'] * 192:.1f} mm, and the outline is then "
                       f"{np.mean(x['plate']['outline_mm']):.0f} mm across, against {section:.1f} mm for the section of the 200 mm shell at the plate. The fitted value is used.",
                       f"The plate: {x['plate']['how']}. Labels chosen: {x['plate']['labels']} (misfit: " + ", ".join(f"{k} {v:g}" for k, v in x["plate"]["label_misfit"].items()) + "). "
                       "`qa.register` could not place the regions: it knows the ACR diameter only and no rotation.",
                       "Engine: the plate `anatomy.nist_t2` at 3 T in a world with ideal fields except the measured B1, sequences `se` and `ir` with the stated TR, TE and TI, "
                       "matrix 256, 5 mm, the fitted field of view, no noise; regions of 5 mm radius on both sides.",
                       f"T2 is fitted where three or more echoes lie above 5 % of the sphere's first echo and above {100 * M9_FLOOR:g} % of the brightest sphere: spheres {name(nine)}. "
                       f"Spheres {name(short)} (published T2 22 ms and shorter, first TE 15 ms) have no T2 here; T1 of sphere 14 is not reported (its signal is "
                       f"{100 * m_se[13, 0]:.0f} % of the brightest at TE 15 ms).",
                       "The B1 map inside a sphere with short T1 is misread by the mapping method; the fit of the water round the spheres is used."])


RUNGS = (("M0", rung_m0), ("M1", rung_m1), ("M2", rung_m2), ("M2b", rung_m2b), ("M3b", rung_m3b), ("M4 / M8", rung_m4_m8), ("M5", None), ("M6", rung_m6),
         ("M7", rung_m7), ("M9", rung_m9))                                # M5 (``rung_m5``) runs with --fetch only
LADDER_NOT_RUN = [
    ("M3", "Stupic 2021 T2 series: the MnCl2 array with the published 32-echo train", "run by the research pass against a figure; not in this tool: the measured side is "
     "qualitative, and M9 now gives numbers for the same spheres"),
    ("M10", "VENUS: the NIST phantom with an open vendor-neutral VFA sequence on three scanners (64.5 MB, CC BY)", "not run: needs a NIfTI reader and the RF spoiling "
     "parameters of the sequence"),
    ("M11", "Pulseq tutorials: `.seq` files with Siemens raw data (CC0)", "not run: needs a `.seq` reader into the block-list core and a raw-data reader"),
    ("M12", "OSI² ONE: T1 and T2 maps at 46 mT from `.seq` and raw data (CC BY)", "not run: needs M11 and a third field strength"),
    ("M13", "ACR phantom, spin-echo EPI at 0.55 T with reversed phase encoding (CC BY)", "not run: needs spin-echo EPI and 0.55 T"),
    ("M14", "A scan of one system phantom with the six console sequences", "**not done.** No image of the console's sequences has been compared pixel by pixel with a scan "
     "of the same object; this is the only rung that would close the gap"),
]

# The twelve known differences (section 4 of the comparison), in plain words. ``rows``: where the number comes from;
# ``closes``: the engine work that would close it.
KNOWN = [
    (1, "The engine has no temperature. The fat-water shift of a phantom is the body's, and the relaxation times of the phantom solutions are those of 20 °C.",
     ("M8.3", "M8.1"), "a world temperature that moves the fat spectrum and the NiCl2 / MnCl2 values (engine package: temperature)"),
    (2, "The default receiver noise is too high for a 3 T head array, and the QA noise level is that of the scanners with the lowest SNR.",
     ("M1.6", "M7.5", "M6.8"), "a noise level per coil class (engine package: receive array)"),
    (3, "The console is slower than product sequences: no partial echo, no overlap of the spoiler with the read-out ramp.",
     ("M0.6", "M0.7"), "console timing with partial echo and overlapping gradients (engine package: console timing)"),
    (4, "The ACR phantom is too uniform at 1.5 T and its systems are too alike: the declared field amplitudes are too small.",
     ("M6.3", "M6.5"), "field amplitudes of `world.ACR_FIELDS` drawn from the measured range (engine package: field model)"),
    (5, "One receive channel: the shading has the wrong sign for a receive array, which is brighter at the rim.",
     ("M6.6",), "several receive channels with sensitivity maps (engine package: receive array)"),
    (6, "The phantom is drawn within 1 mm of the isocentre; real phantoms lie up to 21 mm off. The QA regions that were placed from the isocentre are fixed in this branch.",
     ("M6.11", "M6.13"), "a positioning error of the phantom drawn from the measured range (`anatomy.ACR_SPEC`; ordinary engine work). The measurement side is closed here"),
    (7, "The engine cannot say how T1 and T2 of one phantom differ between sites and units: its objects hold exactly the published values, and nothing in `ir` or `se` changes them.",
     ("M1.5", "M9.4", "M9.5"), "a spread of the phantom values per unit and temperature, and transverse coherence across TR (engine packages: temperature, block-list core)"),
    (8, "The console has no echo train, so it cannot say the T2 error of the multi-echo spin echo that published T2 protocols are. The design layer computes it.",
     ("M3b.4", "M3b.1"), "multi-echo spin echo in the console through the design layer's phase graph (engine package: block-list core)"),
    (9, "The NIST world exists at 3 T only, with a declared T2 rule and declared water. The published T1 and T2 of both plates at both fields are in the package since this branch.",
     ("M0.3", "M0.4", "M0.8", "M0.5"), "`nist_t2` and the 1.5 T plates as world kinds, `nist_t1` on the published T2, water and outline (the engine fix in progress and "
     "one branch of `world.py`)"),
    (10, "Incomplete RF spoiling moves variable-flip-angle T1 at the default phase increment, which is one vendor's style. This is real physics, but no measured number per vendor was found.",
     ("M2.3",), "RF-spoiling presets per vendor and a measured series (rung M10)"),
    (11, "Transcription items: one T1 of the NIST list, one fat peak position, and the name of the fat model.",
     ("M0.2", "M4.2"), "the engine fix in progress (it bumps the engine id)"),
    (12, "The ACR world lacks structures every real image shows.", ("M6.12",), "the inserts and wedges of the phantom through the slice (engine package: 3D world)"),
    # found by rung M9, after the twelve of the research pass
    (13, "Inversion efficiency is one number per session. In the measurement it falls with T2: the inversion is less complete in a tissue that relaxes during the pulse.",
     ("M9.6",), "relaxation during the inversion pulse (engine package: block-list core)"),
    (14, "A healthy engine system leaves no signal inside a dark object. Real scans leave about 1 % of the brightest signal there, different from scan to scan.",
     ("M9.7",), "a ringing and ghost level of the healthy system (ordinary engine work on `qa.HEALTHY` and the noise model)"),
]


# ------------------------------------------------------------------------------------------------ run, store, check
def tool_digest() -> str:
    """Digest of the code that computes the table (this folder), so that a stored table says which code made it."""
    h = hashlib.sha256()
    for name in sorted(n for n in os.listdir(HERE) if n.endswith(".py")):
        with open(os.path.join(HERE, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def source_rows() -> list:
    """One line per cached extract (and one for the source that is never stored): what the page lists."""
    import sources
    out = []
    for name, file in EXTRACTS.items():
        x = load(name)
        out.append(dict(file=file, **{k: x[k] for k in ("source", "url", "licence", "citation", "retrieved", "sha256", "sha256_of", "stored")}))
    out.append(dict(file=None, **sources.provenance("itis")))
    return out


def extract_digests() -> dict:
    """sha256 of every cached extract as it is stored (text files with their line ends as the repository has them)."""
    out = {}
    for name in sorted(set(EXTRACTS.values()) | {SLICE_FILE}):
        with open(os.path.join(STORE, name), "rb") as f:
            data = f.read()
        out[name] = hashlib.sha256(data.replace(b"\r\n", b"\n") if name.endswith(".json") else data).hexdigest()
    return out


def dump(obj) -> str:
    """JSON with one level of indentation per key and lists of numbers on one line."""
    text = json.dumps(obj, indent=1, ensure_ascii=False)
    return re.sub(r"\[\s+((?:-?[\d.eE+-]+|null|true|false)(?:,\s+(?:-?[\d.eE+-]+|null|true|false))*)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", text) + "\n"


def rebuild_extracts(cache: str) -> dict:
    """Fetch the originals and write the cached extracts again. Returns the tissue compilation (never stored)."""
    import extract
    os.makedirs(STORE, exist_ok=True)

    def put(name, data):
        with open(os.path.join(STORE, EXTRACTS[name]), "w", encoding="utf-8") as f:
            f.write(dump(data))
        print(f"   extract written: docs/physics/measured/{EXTRACTS[name]}")

    put("nist_tables", extract.nist_tables(cache))
    put("rrsg", extract.rrsg(cache))
    data, pixels, _ = extract.acr(cache)
    np.savez_compressed(os.path.join(STORE, SLICE_FILE), pixels=pixels)
    with open(os.path.join(STORE, SLICE_FILE), "rb") as f:
        data["off_centre_slice"]["sha256"] = hashlib.sha256(f.read()).hexdigest()
    put("acr", data)
    try:
        put("gel", extract.gel(cache))
    except RuntimeError as e:                                          # no 7z reader on this machine: the stored extract stays
        print(f"   gel: {e}")
    put("pdff", extract.pdff(cache))
    put("scholand", extract.scholand(cache))
    return extract.itis(cache)


def stored() -> dict | None:
    try:
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def known_list(results) -> list:
    by_id = {q["id"]: q for r in results for q in r["rows"]}
    out = []
    for n, what, ids, closes in KNOWN:
        out.append(dict(n=n, what=what, rows=list(ids), numbers=[f"{by_id[i]['difference']} ({i})" for i in ids if i in by_id], closes=closes))
    return out


def key(q):
    """What must be the same in a stored row and in a fresh one for the stored table to be current. The texts of a row
    hold numbers that can differ in the last digit between machines, so they are not compared; the size of a declared
    difference is, to 2 %."""
    return (q["id"], q["tolerance"], q["passed"])


def same_rows(old, new) -> bool:
    return [key(q) for q in old] == [key(q) for q in new] and all(
        abs(a["size"] - b["size"]) <= 0.02 * max(abs(a["size"]), abs(b["size"]), 1e-9) for a, b in zip(old, new) if a["declared"])


def main(argv=None):
    import mrgym
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="download the public originals and rebuild the cached extracts (needs --cache)")
    ap.add_argument("--cache", help="folder for the downloads; it must lie outside the repository")
    ap.add_argument("--only", help="comma-separated rungs to run (the others are taken from the stored table), e.g. M0,M7")
    a = ap.parse_args(argv)
    old = stored()
    db = None
    if a.fetch:
        if not a.cache or os.path.abspath(a.cache).startswith(ROOT + os.sep):
            ap.error("--fetch needs --cache DIR, a folder outside the repository")
        print("== fetching the originals and rebuilding the extracts")
        db = rebuild_extracts(os.path.abspath(a.cache))
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    only = set(a.only.split(",")) if a.only else None
    old_by = {r["rung"]: r for r in (old or {}).get("rungs", [])}
    results = []
    for name, fn in RUNGS:
        if fn is None:
            r = dict(rung_m5(db), engine=mrgym.ENGINE_ID, date_utc=date) if db is not None else old_by.get("M5")
            if r is None:
                r = dict(rung="M5", title="The tissue table against a literature compilation", rows=[], extracts=[], fetch_only=True, tissue_digest=None, engine=None, date_utc=None,
                         source="IT'IS Foundation tissue properties database (licence unclear: this rung runs with --fetch only)", notes=["Not run yet: run with --fetch."])
        else:
            if only is not None and not (set(name.split(" / ")) & only) and name in old_by:
                r = old_by[name]
            else:
                r = dict(fn(), engine=mrgym.ENGINE_ID, date_utc=date)
        results.append(r)
        print(f"== {r['rung']}. {r['title']}" + (" (as stored: runs with --fetch only)" if r.get("fetch_only") and db is None else ""))
        for q in r["rows"]:
            mark = ("decl" if q["passed"] else "GREW") if q["declared"] else ("ok  " if q["passed"] else "FAIL")
            print(f"   {q['id']:6s} {mark}  {q['quantity'][:88]:88s} | {q['difference'][:70]}")
    table = dict(schema="mrgym-measured/1", engine=mrgym.ENGINE_ID, command="python tools/measured/run_all.py --write", tool=tool_digest(), extracts=extract_digests(),
                 sources=source_rows(), rungs=results,
                 not_run=[dict(rung=a_, what=b, state=c) for a_, b, c in LADDER_NOT_RUN], known=known_list(results),
                 sessions=dict(m6=load("acr")["scanners"], m1=load("rrsg")["used"], m8=len(load("pdff")["acquisitions"]), m7=1, m9=1))
    failed = [q["id"] for r in results for q in r["rows"] if not q["declared"] and not q["passed"]]
    grown = [q["id"] for r in results for q in r["rows"] if q["declared"] and not q["passed"]]
    stale = []
    m5 = next(r for r in results if r["rung"] == "M5")
    if m5["rows"] and (m5.get("engine") != mrgym.ENGINE_ID or m5.get("tissue_digest") != tissue_digest()):
        stale.append("M5: the engine or the tissue table has changed since it was fetched; run with --fetch")
    if a.write:
        with open(RESULTS, "w", encoding="utf-8") as f:
            f.write(dump(table))
        with open(os.path.join(ROOT, "docs", "physics", "measured.md"), "w", encoding="utf-8") as f:
            f.write(page.markdown(table))
        print("written: docs/physics/measured/results.json and docs/physics/measured.md")
    elif a.check:
        if old is None:
            stale.append("no stored table")
        else:
            if old["engine"] != mrgym.ENGINE_ID:
                stale.append(f"the table is stored for {old['engine']}")
            if old["extracts"] != table["extracts"]:
                stale.append("a cached extract is not the one the table was computed from")
            if old.get("tool") != table["tool"]:
                stale.append("tools/measured/ has changed since the table was stored")
            for r in results:
                o = old_by.get(r["rung"])
                if o is None or not same_rows(o["rows"], r["rows"]):
                    stale.append(f"{r['rung']}: the stored rows differ from the rows of this run")
            if old.get("known") != table["known"] or old.get("not_run") != table["not_run"]:
                stale.append("the lists of known differences or of rungs not run differ from the stored ones")
        try:
            with open(os.path.join(ROOT, "docs", "physics", "measured.md"), encoding="utf-8") as f:
                if old is not None and f.read() != page.markdown(old):
                    stale.append("docs/physics/measured.md is not the page of the stored table")
        except FileNotFoundError:
            stale.append("docs/physics/measured.md is missing")
    n = sum(len([q for q in r["rows"] if not q["declared"]]) for r in results)
    nd = sum(len([q for q in r["rows"] if q["declared"]]) for r in results)
    print(f"{n - len(failed)} / {n} rows with a tolerance passed" + (f"; failed: {', '.join(failed)}" if failed else "") + f"; {nd} declared differences"
          + (f"; grown past their ceiling: {', '.join(grown)}" if grown else ", none grown") + ("; stale: " + "; ".join(stale) if stale else ""))
    return 1 if failed or grown or stale else 0


if __name__ == "__main__":
    sys.exit(main())
