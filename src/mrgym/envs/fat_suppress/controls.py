"""Control policies of ``Scanner/FatSuppress``. Every policy plays through the agent's interface only.

Must pass
  reference          tries fat saturation: a fast spin-echo pair with and without it gives what it leaves in every
                     region; where that is not enough, four signed inversion-recovery probes at a repetition time the
                     budget allows, a fit of a - b exp(-TI / T1) per region, and the null carried over to the target
                     repetition time with the recovery model (the fit gives the inversion efficiency too)
Must fail
  prior              no scan; fat saturation everywhere
  wrong_units        the reference's inversion times in seconds (regions it gives to fat saturation stay as they are)
  fixed_recipe       one token scan, then the best answer that ignores the object, tuned on other worlds by the gate:
                     one method for all regions or a method by the region's distance from the isocentre, and one
                     inversion time or one fraction of the (told) repetition time
  constant           the same, tuned once on 300 other worlds per level (``python -m mrgym.envs.fat_suppress.controls``)
  brute_force        tries fat saturation, then probes on a fixed grid of inversion times at a repetition time the
                     budget allows and delivers the one with the smallest signal; no model
Ablations (the reference with one decision removed; each must lose where the levels say so)
  fat_sat_only       fat saturation everywhere (a scan is made, its result ignored)
  stir_only          inversion recovery everywhere, with the reference's inversion times
  field_map          does not try fat saturation: predicts it from a field map and the nominal band of the pulse
  assumed_inversion  the reference's fit of T1, then the null of the recovery model with an assumed inversion efficiency
  textbook_null      the reference's fit of T1, then TI = ln 2 x T1 (ignores the repetition time and the inversion
                     efficiency; nearly right while the repetition time is long against the T1 of fat and the inversion
                     is nearly complete, which is levels 1 and 2)
Reported
  open_loop          a fixed scan list (the pair, four probes at 600 ms) and the reference's estimator

For ``mrgym check <id> --learnability``
  blind_answers      one method for every region, the inversion time constant or a fraction of the repetition time:
                     answers that use the task card only
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "wrong_units", "fixed_recipe", "constant", "brute_force")
ABLATIONS = {"fat_sat_only": (1, 2, 3), "stir_only": (1, 2, 3), "field_map": (2, 3), "assumed_inversion": (3,), "textbook_null": (3,)}
REPORTED = ("open_loop",)
LADDER = {1: "field_map", 2: "assumed_inversion"}        # the policy that suffices at a level and must not at the next
DESCRIPTIONS = dict(
    reference="tries fat saturation (a fast pair with and without); where it is not enough, four signed inversion probes, "
              "a fit per region and the null carried over to the target repetition time",
    prior="no scan; fat saturation everywhere",
    wrong_units="the reference's inversion times in seconds",
    fixed_recipe="one token scan, then the best answer that ignores the object (method for all or by distance from the "
                 "isocentre, inversion time constant or a fraction of the repetition time), tuned on other worlds by the gate",
    constant="one token scan, then the best answer that ignores the object, tuned once on 300 other worlds",
    brute_force="tries fat saturation, then a fixed grid of inversion times; delivers the smallest signal, no model",
    fat_sat_only="fat saturation everywhere",
    stir_only="inversion recovery everywhere, with the reference's inversion times",
    field_map="predicts fat saturation from a field map and the nominal pulse band instead of trying it",
    assumed_inversion="the reference's fit of T1, then the null with an assumed inversion efficiency",
    textbook_null="the reference's fit of T1, then TI = ln 2 x T1",
    open_loop="a fixed scan list (the pair, four probes at a repetition time of 600 ms) and the reference's estimator",
)
# the pair: the geometry of the target image (the same pixels and the same displacement of the fat, so a region shows
# the same spins), at a short repetition time
PAIR = dict(task.GEOMETRY, sequence="se", tr_ms=60.0)
PROBE = dict(sequence="ir", matrix=48, partial_fourier=0.6, bandwidth_hz_per_px=600.0)
FIELD_MAP = dict(sequence="spgr", flip_deg=10.0, tr_ms=12.0, matrix=96, bandwidth_hz_per_px=3000.0, n_echoes=3, averages=3)
SWITCH = 0.5 * (task.FAT_MAX + task.FAT_SAT_SUFFICIENT)      # fat saturation is taken where it leaves less than this
ASSUMED_INVERSION = 0.95
# field_map: three numbers of a fixed estimator, set on other worlds of level 1
FAT_APPARENT_HZ = 309.2                                      # phase of fat between the first two echoes of FIELD_MAP, as a frequency
BAND_HALF_WIDTH_HZ = 76.0                                    # fat saturation is predicted while the off-resonance stays inside
EDGE_FACTOR = 0.9                                            # ... at the centre plus this much of its change to the edge


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _complex(scan):
    return {row["id"]: row["complex_mean_magnitude"][0] * np.exp(1j * row["complex_mean_phase_rad"][0]) for row in scan["roi_readouts"]}


# ------------------------------------------------------------------------------------------------ fat saturation
def _tried(sc, t):
    """What fat saturation leaves in every region: a fast spin echo with it over the same without. The saturation acts
    before the excitation, so the ratio is the one of the target scan. The saturated scan is the weak one and is
    averaged twice; its value is taken along the phase of the plain scan, which has no noise floor."""
    plain = _complex(sc.scan(dict(PAIR), rois=t["regions"]))
    sat = _complex(sc.scan(dict(PAIR, fat_sat=True, averages=2), rois=t["regions"]))
    return {i: float((sat[i] * np.conj(plain[i])).real) / max(abs(plain[i]) ** 2, 1e-30) for i in _ids(t)}


def _field(sc, t):
    """Off-resonance at the centre of every region and the size of its change from the centre to the edge, in Hz, from
    the phase between two gradient echoes read inside the vial. In fat that phase carries the chemical shift: at these
    echo times it is the off-resonance plus ``FAT_APPARENT_HZ`` (measured on other worlds), modulo 1 / echo spacing.
    {region id: (off-resonance, change across the radius)}."""
    r = sc.scan(dict(FIELD_MAP), rois=t["regions"])
    te = r["realised"]["echo_times_ms"]
    period = 1e3 / (te[1] - te[0])
    wrap = lambda f: (f + period / 2) % period - period / 2             # noqa: E731
    f12 = lambda row: float(np.angle(np.exp(1j * (row["complex_mean_phase_rad"][1] - row["complex_mean_phase_rad"][0])))   # noqa: E731
                            / (2 * np.pi * (te[1] - te[0]) * 1e-3))
    centre = {row["id"]: f12(row) for row in r["roi_readouts"]}
    out = {}
    for reg in t["regions"]:
        h = 0.55 * reg["radius_mm"]
        sub = [dict(id=str(k), y_mm=reg["y_mm"] + dy * h, x_mm=reg["x_mm"] + dx * h, radius_mm=0.45 * reg["radius_mm"])
               for k, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1)))]
        d = [wrap(f12(row) - centre[reg["id"]]) for row in sc.read_regions(r["scan_index"], sub)]
        out[reg["id"]] = (wrap(centre[reg["id"]] - FAT_APPARENT_HZ), math.hypot(d[1] - d[0], d[3] - d[2]) / (2 * h) * reg["radius_mm"])
    return out


def _predicted(sc, t):
    """Where fat saturation should work: where the off-resonance stays inside the nominal pulse band over the region."""
    return {i: abs(f) + EDGE_FACTOR * across < BAND_HALF_WIDTH_HZ for i, (f, across) in _field(sc, t).items()}


# ------------------------------------------------------------------------------------------------ the null
def _probe_tr(sc, t, n_probes):
    """The repetition time of the probes: the target's when ``n_probes`` of them fit what is left of the budget, else
    shorter."""
    tr, left = t["target_scan"]["tr_ms"], sc.budget()["remaining_s"]
    while tr > 300.0:
        p = sc.plan(dict(PROBE, tr_ms=tr, ti_ms=50.0))
        if p["accept"] and n_probes * p["scan_s"] <= left:
            return tr
        tr = round(tr * 0.9 / 10.0) * 10.0
    return 300.0


def _fit(ti, s):
    """Least squares of s = a - b exp(-ti / T1) over T1 on a grid (a and b are linear). Returns (a, b, T1)."""
    ti, s = np.asarray(ti, float), np.asarray(s, float)
    best = None
    for t1 in np.exp(np.linspace(math.log(80.0), math.log(3000.0), 600)):
        m = np.stack([np.ones_like(ti), -np.exp(-ti / t1)], 1)
        coef = np.linalg.lstsq(m, s, rcond=None)[0]
        err = float(np.sum((m @ coef - s) ** 2))
        if best is None or err < best[0]:
            best = (err, float(coef[0]), float(coef[1]), float(t1))
    return best[1:]


def _probes(sc, t, tis, tr):
    """Signed probe values per region: the sign is the phase relative to the probe with the longest inversion time,
    which lies beyond the null of any fat."""
    vals = [_complex(sc.scan(dict(PROBE, tr_ms=tr, ti_ms=ti), rois=t["regions"])) for ti in tis]
    out = {}
    for i in _ids(t):
        phase = vals[-1][i] / abs(vals[-1][i])
        out[i] = [float((v[i] * np.conj(phase)).real) for v in vals]
    return out


def _null_at(a, b, t1, tr_probe, tr_target, te):
    """Inversion time of the null at ``tr_target`` from a fit made at ``tr_probe``.

    Model: S(TI) proportional to 1 + D - (1 + k) exp(-TI / T1), D = k exp(-(TR - TE / 2) / T1). The fit gives
    b / a = (1 + k) / (1 + D_probe); solve for the inversion efficiency k, then evaluate D at the target TR."""
    q = b / a if a > 0 else 2.0
    if q <= 1.0:
        return 20.0
    e_p, e_t = math.exp(-(tr_probe - te / 2) / t1), math.exp(-(tr_target - te / 2) / t1)
    k = min(max((q - 1.0) / (1.0 - q * e_p), 0.5), 1.0)
    return t1 * math.log((1.0 + k) / (1.0 + k * e_t))


def _nulls(sc, t, tr=None, n_probes=4, kind="fit"):
    """Inversion time that nulls every region at the target repetition time, from signed probes and a fit per region.
    kind 'fit': the model above; 'assumed': the same with the inversion efficiency assumed instead of taken from
    the fit; 'textbook': ln 2 x the fitted T1."""
    tr_t = t["target_scan"]["tr_ms"]
    tr = tr or _probe_tr(sc, t, n_probes)
    te = sc.plan(dict(PROBE, tr_ms=tr, ti_ms=50.0))["realised"]["te_ms"]
    hi = min(tr - te - 25.0, 900.0)
    tis = [round(float(v), 1) for v in np.exp(np.linspace(math.log(60.0), math.log(hi), n_probes))]
    vals = _probes(sc, t, tis, tr)
    out = {}
    for i in _ids(t):
        a, b, t1 = _fit(tis, vals[i])
        if kind == "textbook":
            out[i] = math.log(2.0) * t1
        elif kind == "assumed":
            k = ASSUMED_INVERSION
            out[i] = t1 * math.log((1.0 + k) / (1.0 + k * math.exp(-(tr_t - te / 2) / t1)))
        else:
            out[i] = _null_at(a, b, t1, tr, tr_t, te)
    lim = tr_t - te - 25.0
    return {i: round(float(np.clip(v, 20.0, lim)), 1) for i, v in out.items()}


# ------------------------------------------------------------------------------------------------ policies
def _deliver(t, fat_sat_ok, ti):
    return {i: dict(method="fat_sat") if fat_sat_ok[i] else dict(method="stir", ti_ms=ti[i]) for i in _ids(t)}


def reference(sc, probe_tr=None, kind="fit"):
    t = _task(sc)
    left = _tried(sc, t)
    ok = {i: left[i] < SWITCH for i in _ids(t)}
    ti = {} if all(ok.values()) else _nulls(sc, t, tr=probe_tr, kind=kind)
    return _deliver(t, ok, ti)


def open_loop(sc):
    t = _task(sc)
    left = _tried(sc, t)
    ti = _nulls(sc, t, tr=600.0)
    return _deliver(t, {i: left[i] < SWITCH for i in _ids(t)}, ti)


def assumed_inversion(sc):
    return reference(sc, kind="assumed")


def textbook_null(sc):
    return reference(sc, kind="textbook")


def field_map(sc):
    t = _task(sc)
    ok = _predicted(sc, t)
    ti = {} if all(ok.values()) else _nulls(sc, t)
    return _deliver(t, ok, ti)


def fat_sat_only(sc):
    t = _task(sc)
    _tried(sc, t)
    return {i: dict(method="fat_sat") for i in _ids(t)}


def stir_only(sc):
    t = _task(sc)
    return _deliver(t, {i: False for i in _ids(t)}, _nulls(sc, t))


def prior(sc):
    return {i: dict(method="fat_sat") for i in _ids(_task(sc))}


def wrong_units(sc):
    return {i: dict(e, ti_ms=e["ti_ms"] / 1000.0) if e["method"] == "stir" else e for i, e in reference(sc).items()}


def brute_force(sc):
    t = _task(sc)
    left = _tried(sc, t)
    tr = _probe_tr(sc, t, 5)
    cost = sc.plan(dict(PROBE, tr_ms=tr, ti_ms=50.0))["scan_s"]
    n = max(2, int(sc.budget()["remaining_s"] // cost))
    tis = [round(float(v), 1) for v in np.linspace(120.0, 400.0, n)]
    mags = [{k: abs(v) for k, v in _complex(sc.scan(dict(PROBE, tr_ms=tr, ti_ms=ti), rois=t["regions"])).items()} for ti in tis]
    ti = {i: tis[int(np.argmin([m[i] for m in mags]))] for i in _ids(t)}
    return _deliver(t, {i: left[i] < SWITCH for i in _ids(t)}, ti)


def recipe_answer(recipe, t):
    """The answer of a recipe that ignores the object. ``fat_sat_within_mm``: fat saturation for regions closer to the
    isocentre than this (0 = nowhere, a large number = everywhere); the others get stir with a constant inversion
    time or a constant fraction of the repetition time."""
    ti = recipe["ti_over_tr"] * t["target_scan"]["tr_ms"] if recipe.get("ti_over_tr") else recipe["ti_ms"]
    return {r["id"]: dict(method="fat_sat") if math.hypot(r["y_mm"], r["x_mm"]) < recipe["fat_sat_within_mm"]
            else dict(method="stir", ti_ms=round(float(ti), 1)) for r in t["regions"]}


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        sc.scan(dict(PAIR), rois=t["regions"])
        return recipe_answer(recipe, t)
    return policy


# tuned by ``python -m mrgym.envs.fat_suppress.controls`` on 300 worlds per level of the tune split (seeds constant:<i>),
# which the gate never plays; engine 0.2.0
CONSTANTS = {1: {"fat_sat_within_mm": 54.0, "ti_ms": 283.0, "tuned_pass_rate": 0.01},
             2: {"fat_sat_within_mm": 1000.0, "ti_ms": 100.0, "tuned_pass_rate": 0.02},
             3: {"fat_sat_within_mm": 1000.0, "ti_ms": 100.0, "tuned_pass_rate": 0.02}}


def constant(sc):
    t = _task(sc)
    sc.scan(dict(PAIR), rois=t["regions"])
    return recipe_answer(CONSTANTS[t["level"]], t)


def blind_answers(t: dict) -> dict:
    """Answers that use the task card only: one method for every region, the inversion time as a constant or as a
    fraction of the told repetition time. {label: answer}; the same labels on every instance."""
    tr = t["target_scan"]["tr_ms"]
    out = {"fat_sat": {i: dict(method="fat_sat") for i in _ids(t)}}
    grid = {f"stir,ti_ms={v:g}": float(v) for v in (180.0, 210.0, 240.0, 270.0, 300.0, 340.0)}
    grid.update({f"stir,ti_over_tr={f:g}": f * tr for f in (0.08, 0.10, 0.12, 0.15)})
    out.update({label: {i: dict(method="stir", ti_ms=round(v, 1)) for i in _ids(t)} for label, v in grid.items()})
    return out


def policies(recipe=None):
    return dict(reference=reference, prior=prior, wrong_units=wrong_units, fixed_recipe=fixed_recipe(recipe or CONSTANTS[2]),
                constant=constant, brute_force=brute_force, fat_sat_only=fat_sat_only, stir_only=stir_only, field_map=field_map,
                assumed_inversion=assumed_inversion, textbook_null=textbook_null, open_loop=open_loop)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds, by the pass rate under the two pass rules as the
    independent oracle evaluates them (closed form, no scans). Candidates: fat saturation inside a radius around the
    isocentre (nowhere, everywhere, or in between), stir elsewhere with a constant inversion time or a constant
    fraction of the (told) repetition time."""
    from ...scanner import world as W
    from . import oracle
    grid = np.arange(100.0, 460.0, 1.0)
    worlds = []
    for c in cfgs:
        t, truth = c["task"], oracle.truth(c, W.make_world(c["seed"], c["world_spec"]))
        rows = []
        for r in t["regions"]:
            row = truth[r["id"]]
            nulled = grid[[oracle.left_with(row, dict(method="stir", ti_ms=float(ti))) <= t["fat_max"] for ti in grid]]
            rows.append((math.hypot(r["y_mm"], r["x_mm"]), row["fat_sat_left"] <= t["fat_max"], row["fat_sat_left"] > t["fat_sat_sufficient"],
                         nulled.min() if len(nulled) else np.inf, nulled.max() if len(nulled) else -np.inf))
        worlds.append((t["target_scan"]["tr_ms"], np.array(rows)))
    radii = [0.0, 1000.0] + sorted({round(float(d) + 0.5) for _, rows in worlds for d in rows[:, 0]})[::3]
    best = None
    for radius in radii:
        for kind, values in (("ti_ms", grid[::3]), ("ti_over_tr", np.arange(0.04, 0.26, 0.002))):
            for v in values:
                n = 0
                for tr, rows in worlds:
                    ti = v if kind == "ti_ms" else v * tr
                    sat = rows[:, 0] < radius
                    n += bool(np.all(np.where(sat, rows[:, 1], rows[:, 2] * (rows[:, 3] <= ti) * (ti <= rows[:, 4]))))
                if best is None or n > best[0]:
                    best = (n, {"fat_sat_within_mm": float(radius), kind: round(float(v), 4)})
            if radius >= 1000.0:
                break
    return dict(best[1], tuned_pass_rate=round(best[0] / len(worlds), 3))


if __name__ == "__main__":
    import mrgym
    from mrgym import gate
    spec = mrgym.spec("Scanner/FatSuppress-v0")
    print("CONSTANTS = {" + ",\n             ".join(
        f"{lv}: {tune([gate._instance(spec, lv, f'constant:{i}') for i in range(300)])}" for lv in spec.levels) + "}")
