"""Control policies of ``Scanner/ShiftFree``. Every policy plays through the agent's interface only.

Must pass
  reference          one scan per readout direction at one low bandwidth (level 3: on both axes), a ``profile`` through
                     every region; per fat border the two readings give its true position (their mean) and the
                     displacement at the probe bandwidth (half their difference); then the direction (and axis) with
                     the most room and the bandwidth whose displacement is 0.9 x (room + 0.25 mm)
Must fail
  prior              no scan; the console defaults (300 Hz per pixel, ro_polarity 1, pe_dir y)
  default_bandwidth  one token scan, then the console defaults
  highest_bandwidth  one token scan, then the highest bandwidth the gradients allow at this geometry
  fixed_recipe       one token scan, then one bandwidth, readout direction and axis for every region, tuned on other worlds
  sweep              three scans at fixed bandwidths with ro_polarity 1; per region the lowest of them at which the
                     region's signal has not risen; no model of the displacement
  wrong_units        the reference's bandwidth in kHz per pixel
Ablations (must lose where the step they leave out matters)
  one_direction      measures like the reference, always delivers ro_polarity 1 (levels 1 to 3)
  fixed_axis         measures along x only and keeps pe_dir y (level 3)
Reported
  open_loop          the reference: its scan list and its estimator are fixed, no scan depends on an earlier one
  header_shift       the reference with the displacement taken from the water-fat shift that ``plan`` prints instead of
                     the measured one (it leaves out the local off-resonance and where on the fat line the border sits)
"""
from __future__ import annotations

import numpy as np

from ...scanner import console
from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "default_bandwidth", "highest_bandwidth", "fixed_recipe", "sweep", "wrong_units")
ABLATIONS = {"one_direction": (1, 2, 3), "fixed_axis": (3,)}
REPORTED = ("open_loop", "header_shift")
DESCRIPTIONS = dict(
    reference="two scans of opposite readout direction per axis at one low bandwidth, profiles through every region: room and displacement per fat border, then the bandwidth that uses 0.9 of the room",
    prior="no scan; the console defaults (300 Hz per pixel, ro_polarity 1, pe_dir y)",
    default_bandwidth="one token scan, then the console defaults",
    highest_bandwidth="one token scan, then the highest bandwidth the gradients allow at this geometry",
    fixed_recipe="one token scan, then one bandwidth, readout direction and axis for every region, tuned on other worlds",
    sweep="three scans at fixed bandwidths with ro_polarity 1; per region the lowest of them at which its signal has not risen",
    wrong_units="the reference's bandwidth in kHz per pixel",
    one_direction="measures like the reference, always delivers ro_polarity 1",
    fixed_axis="measures along x only and keeps pe_dir y",
    open_loop="the reference: a fixed scan list and a fixed estimator",
    header_shift="the reference with the displacement from the printed water-fat shift instead of the measured one",
)
PROBE = dict(sequence="se", tr_ms=150.0, matrix=task.TARGET["matrix"], fov_mm=task.TARGET["fov_mm"], bandwidth_hz_per_px=150.0)
TOKEN = dict(sequence="spgr", tr_ms=12.0, flip_deg=10.0, matrix=64)
AIM = 0.9                               # the delivered displacement as a share of (room + task.REACH_MM)
PROFILE_HALF_MM, PROFILE_WIDTH_MM = 40.0, 2.5
SWEEP_BW = (80.0, 226.0, 640.0)
BW_LIMITS = (console.SEQUENCES["se"]["bandwidth_hz_per_px"][0], task.BW_TOP)


def _task(sc):
    return sc.info()["task"]


def _axes(t):
    return ("y", "x") if "pe_dir" in t["free"] else (t["target_scan"]["pe_dir"],)


def _scan_of(t, bandwidth, ro_polarity, pe_dir="y"):
    full = dict(bandwidth_hz_per_px=round(float(np.clip(bandwidth, *BW_LIMITS)), 1), ro_polarity=int(ro_polarity), pe_dir=pe_dir)
    return {k: full[k] for k in t["free"]}


def _crossing(t, v, level, start):
    """First position along ``t`` (ascending) at or after ``start`` where ``v`` rises through ``level`` (linear
    interpolation between samples); None when it never does."""
    for k in range(1, len(t)):
        if t[k] >= start and v[k - 1] < level <= v[k]:
            return float(t[k - 1] + (level - v[k - 1]) * (t[k] - t[k - 1]) / (v[k] - v[k - 1]))
    return None


def _border(t, v, gel, approaches, fat=None):
    """Where the fat begins along the outward coordinate ``t`` (mm beyond the edge of the region). A receding disc
    leaves a dark gap in front of it: the border is where the signal rises through half the fat signal, and the fat
    signal is read behind it. An approaching disc overlaps the gel: the border is where the signal rises through
    gel + half the fat signal (``fat``: read on the scan in which the disc recedes). Returns (border, fat signal)."""
    if not approaches:
        rough = _crossing(t, v, 0.5 * float(v[(t > -1.0) & (t < 26.0)].max()), -1.0)
        if rough is None:
            return None, None
        inside = (t > rough + 3.0) & (t < rough + 8.0)
        fat = float(np.median(v[inside])) if inside.any() else float(v.max())
        gap = (t > -2.0) & (t < rough + 1.0)                             # the dark gap lies in front of the border
        start = float(t[gap][int(np.argmin(v[gap]))]) if gap.any() else -2.0
        return _crossing(t, v, 0.5 * fat, start), fat
    return _crossing(t, v, gel + 0.5 * fat, -task.REGION_MM + 1.0), fat


def _measure(sc, t, axes):
    """Room and displacement per region and side: {region id: {side: dict(room_mm, shift_mm)}} at the probe bandwidth,
    and the probe's realised protocols per axis."""
    out = {r["id"]: {} for r in t["regions"]}
    probes = {}
    for pe in axes:
        ro = "x" if pe == "y" else "y"
        scans = {pol: sc.scan(dict(PROBE, pe_dir=pe, ro_polarity=pol)) for pol in (1, -1)}
        probes[pe] = scans[1]
        for r in t["regions"]:
            centre, line = (r["x_mm"], r["y_mm"]) if ro == "x" else (r["y_mm"], r["x_mm"])
            prof = {pol: sc.profile(s["scan_index"], ro, position_mm=line, width_mm=PROFILE_WIDTH_MM,
                                    start_mm=centre - PROFILE_HALF_MM, stop_mm=centre + PROFILE_HALF_MM) for pol, s in scans.items()}
            for sign, side in ((-1.0, ro + "-"), (1.0, ro + "+")):
                near = 1 if sign < 0 else -1                             # the readout direction that moves this side's fat towards the region
                cut = {}
                for pol, p in prof.items():
                    u = np.asarray(p["positions_mm"]) - centre
                    v = np.asarray(p["values"])
                    order = np.argsort(sign * u)
                    cut[pol] = (sign * u[order] - r["radius_mm"], v[order], float(np.median(v[np.abs(u) <= 2.0])))
                far, fat = _border(cut[-near][0], cut[-near][1], cut[-near][2], approaches=False)
                close = None if far is None else _border(cut[near][0], cut[near][1], cut[near][2], approaches=True, fat=fat)[0]
                if far is None or close is None:
                    continue
                out[r["id"]][side] = dict(room_mm=0.5 * (far + close), shift_mm=0.5 * (far - close))
    return out, probes


def _choose(t, found, axes, directions=(1, -1), shift_of=None):
    """Per region: the allowed setting whose approaching fat has the most room, with the bandwidth whose
    displacement is ``AIM`` x (room + reach)."""
    scans = {}
    for r in t["regions"]:
        cand = []
        for pe in axes:
            for pol in directions:
                m = found[r["id"]].get(task.approaching(pe, pol))
                if m is None or m["room_mm"] <= 0 or m["shift_mm"] <= 0:
                    continue
                shift = shift_of(pe) if shift_of else m["shift_mm"]
                cand.append((PROBE["bandwidth_hz_per_px"] * shift / (AIM * (m["room_mm"] + task.REACH_MM)), pol, pe))
        bw, pol, pe = min(cand) if cand else (300.0, 1, axes[0])
        scans[r["id"]] = _scan_of(t, bw, pol, pe)
    return dict(scans=scans)


def reference(sc):
    t = _task(sc)
    found, _ = _measure(sc, t, _axes(t))
    return _choose(t, found, _axes(t))


def one_direction(sc):
    t = _task(sc)
    found, _ = _measure(sc, t, _axes(t))
    return _choose(t, found, _axes(t), directions=(1,))


def fixed_axis(sc):
    t = _task(sc)
    found, _ = _measure(sc, t, ("y",))
    return _choose(t, found, ("y",))


def header_shift(sc):
    t = _task(sc)
    found, probes = _measure(sc, t, _axes(t))
    printed = {pe: sc.plan(dict(PROBE, pe_dir=pe))["derived"]["water_fat_shift_px"] * task.PIXEL_MM for pe in probes}
    return _choose(t, found, _axes(t), shift_of=lambda pe: printed[pe])


def _constant(t, bandwidth, ro_polarity=1, pe_dir="y"):
    return dict(scans={r["id"]: _scan_of(t, bandwidth, ro_polarity, pe_dir) for r in t["regions"]})


def prior(sc):
    return _constant(_task(sc), 300.0)


def default_bandwidth(sc):
    t = _task(sc)
    sc.scan(dict(TOKEN))
    return _constant(t, 300.0)


def highest_bandwidth(sc):
    t = _task(sc)
    sc.scan(dict(TOKEN))
    return _constant(t, task.BW_TOP)


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        sc.scan(dict(TOKEN))
        return _constant(t, recipe["bandwidth_hz_per_px"], recipe.get("ro_polarity", 1), recipe.get("pe_dir", "y"))
    return policy


DEFAULT_RECIPE = dict(bandwidth_hz_per_px=110.0, ro_polarity=1, pe_dir="y")


def sweep(sc):
    """Three bandwidths, no model: per region the lowest one at which its signal is within 5 % of its signal at the highest."""
    t = _task(sc)
    seen = []
    for bw in SWEEP_BW:
        r = sc.scan(dict(PROBE, bandwidth_hz_per_px=bw, pe_dir="y", ro_polarity=1), rois=t["regions"])
        seen.append({row["id"]: row["mean_magnitude"][0] for row in r["roi_readouts"]})
    scans = {}
    for r in t["regions"]:
        ok = [bw for bw, s in zip(SWEEP_BW, seen) if s[r["id"]] <= 1.05 * seen[-1][r["id"]]]
        scans[r["id"]] = _scan_of(t, min(ok), 1, "y")
    return dict(scans=scans)


def wrong_units(sc):
    a = reference(sc)
    return dict(scans={i: dict(s, bandwidth_hz_per_px=s["bandwidth_hz_per_px"] / 1000.0) for i, s in a["scans"].items()})


def policies(recipe=None):
    return dict(reference=reference, prior=prior, default_bandwidth=default_bandwidth, highest_bandwidth=highest_bandwidth,
                fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), sweep=sweep, wrong_units=wrong_units,
                one_direction=one_direction, fixed_axis=fixed_axis, open_loop=reference, header_shift=header_shift)


def tune(cfgs):
    """The strongest constant answer on other worlds: one bandwidth, readout direction and axis for every region, by
    the per-region pass rate under the verifier's rule as the drawn gaps predict it (the widest gap names the setting;
    its bandwidth is the main fat line's displacement at gap + ``task.REACH_MM``; the window reaches up by 1 / NEAR_BEST^2).
    The local off-resonance, which moves the true window by a few percent, is left out of the prediction."""
    rows = []
    for c in cfgs:
        f_main = abs(console.FAT_PPM) * 1e-6 * console.GAMMA_HZ_PER_T * c["world_spec"]["b0_t"]
        for s in c["truth"]["sites"]:
            side = max(s["gap_mm"], key=s["gap_mm"].get)
            pe, pol = ("y" if side[0] == "x" else "x"), (1 if side[1] == "-" else -1)
            rows.append((pe, pol, f_main * task.PIXEL_MM / (s["gap_mm"][side] + task.REACH_MM)))
    best = None
    for pe, pol in sorted({(r[0], r[1]) for r in rows}):
        for cand in sorted({round(r[2] * 1.07, 1) for r in rows}):
            rate = float(np.mean([r[0] == pe and r[1] == pol and r[2] <= cand <= r[2] / task.NEAR_BEST ** 2 for r in rows]))
            if best is None or rate > best[0]:
                best = (rate, cand, pol, pe)
    return dict(bandwidth_hz_per_px=best[1], ro_polarity=best[2], pe_dir=best[3], tuned_region_pass_rate=round(best[0], 3))
