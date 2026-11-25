"""Control policies of ``Scanner/NullTissue``. Every policy plays through the agent's interface only.

Must pass
  reference      three coarse probes at the target's own repetition time when the budget pays them (else four probes
                 at a shorter one and a transfer model), signed with the phase image; fits
                 S(TI) = a - b exp(-TI / T1); when the probes were run at another repetition time than the target,
                 carries the fit over with the recovery model; then one check scan and a linear correction
Must fail
  prior          no scan; TI = ln 2 x the middle of the plausible T1 range
  fixed_recipe   one token scan, then the best answer that ignores the object: one constant inversion time, or one
                 constant fraction of the (told) repetition time, tuned on other worlds
  brute_force    probes on a fixed grid of inversion times and delivers the one with the smallest signal; no model
  wrong_units    the reference's answer in seconds
Ablation (must lose at levels 2 and 3, where the confounders it ignores are present)
  textbook       measures T1 like the reference, then delivers TI = ln 2 x T1 (ignores the repetition time and the
                 inversion efficiency): the policy that the levels are built to defeat one confounder at a time

For ``mrgym check <id> --learnability`` (``mrgym.learn.check``)
  policy_space   a small parametric family around the reference: how many probes, at which share of the target
                 repetition time, at which resolution, whether the fit is carried to the target repetition time,
                 whether a check scan is made. The reference and the textbook ablation are two of its corners
  blind_answers  constant inversion times and constant fractions of the (told) repetition time: the answers that use
                 the task card only, on a grid (bounds what ``progress`` gives without looking at the object)
"""
from __future__ import annotations

import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "brute_force", "wrong_units")
ABLATIONS = {"textbook": (2, 3)}        # the reference without the repetition-time and inversion-efficiency model
REPORTED = ()
DESCRIPTIONS = dict(
    reference="three signed probes, fit of a - b exp(-TI / T1), transfer to the target repetition time, one check scan",
    prior="no scan; TI = ln 2 x the middle of the plausible T1 range",
    fixed_recipe="one token scan, then one constant inversion time tuned on other worlds",
    brute_force="probes on a fixed grid of inversion times, delivers the one with the smallest signal",
    wrong_units="the reference's inversion time in seconds",
    textbook="measures T1, then TI = ln 2 x T1 (ignores the repetition time and the inversion efficiency)",
)
PROBE = dict(sequence="ir", matrix=64, partial_fourier=0.6, bandwidth_hz_per_px=300.0)
T1_MID_MS = 630.0


def _task(sc):
    return sc.info()["task"]


def _ids(t):
    return [r["id"] for r in t["regions"]]


def _probe(sc, t, tr_ms, ti_ms, matrix=PROBE["matrix"]):
    """One low-resolution inversion-recovery probe: {region id: complex mean}."""
    r = sc.scan(dict(PROBE, matrix=matrix, tr_ms=tr_ms, ti_ms=ti_ms), rois=t["regions"])
    return {row["id"]: row["complex_mean_magnitude"][0] * np.exp(1j * row["complex_mean_phase_rad"][0]) for row in r["roi_readouts"]}


def _probe_tr(sc, t, n_probes, tr_fraction=1.0, matrix=PROBE["matrix"]):
    """The repetition time of the probes: ``tr_fraction`` of the target's when ``n_probes`` of them fit the budget, else
    shorter."""
    tr, left = t["target_scan"]["tr_ms"], sc.budget()["remaining_s"]
    if tr_fraction != 1.0:
        tr = max(round(tr * tr_fraction / 10.0) * 10.0, 300.0)
    while tr > 300.0:
        p = sc.plan(dict(PROBE, matrix=matrix, tr_ms=tr, ti_ms=50.0))
        if p["accept"] and n_probes * p["scan_s"] <= left:
            return tr
        tr = round(tr * 0.8 / 10.0) * 10.0
    return 300.0


def _fit(ti, s):
    """Least squares of s = a - b exp(-ti / T1) over T1 on a grid (a and b are linear). Returns (a, b, T1)."""
    ti, s = np.asarray(ti, float), np.asarray(s, float)
    best = None
    for t1 in np.exp(np.linspace(math.log(60.0), math.log(8000.0), 700)):
        m = np.stack([np.ones_like(ti), -np.exp(-ti / t1)], 1)
        coef = np.linalg.lstsq(m, s, rcond=None)[0]
        err = float(np.sum((m @ coef - s) ** 2))
        if best is None or err < best[0]:
            best = (err, float(coef[0]), float(coef[1]), float(t1))
    return best[1:]


def _measure(sc, t, n_probes=4, reserve=1, tr_fraction=1.0, matrix=PROBE["matrix"]):
    """Signed probes and one fit per region. The sign comes from the phase relative to the probe with the longest
    inversion time, which lies beyond the null of any material. Returns (fits {id: dict(a, b, t1, phase)}, TR, TE)."""
    tr = _probe_tr(sc, t, n_probes + reserve, tr_fraction, matrix)
    te = sc.plan(dict(PROBE, matrix=matrix, tr_ms=tr, ti_ms=50.0))["realised"]["te_ms"]
    tis = [round(float(v), 1) for v in np.exp(np.linspace(math.log(60.0), math.log(tr - te - 25.0), n_probes))]
    vals = [_probe(sc, t, tr, ti, matrix) for ti in tis]
    fits = {}
    for i in _ids(t):
        phase = vals[-1][i] / abs(vals[-1][i])
        signed = [float((v[i] * np.conj(phase)).real) for v in vals]
        a, b, t1 = _fit(tis, signed)
        fits[i] = dict(a=a, b=b, t1=t1, phase=phase, tis=list(tis), signed=signed)
    return fits, tr, te


def _null_at(m, tr_probe, tr_target, te):
    """Inversion time of the null at ``tr_target`` from a fit made at ``tr_probe``.

    Model: S(TI) proportional to 1 + D - (1 + k) exp(-TI / T1), D = k exp(-(TR - TE / 2) / T1).
    The fit gives b / a = (1 + k) / (1 + D_probe); solve for k, then evaluate D at the target repetition time."""
    q, t1 = m["b"] / m["a"], m["t1"]
    if q <= 1.0:
        return 20.0
    if tr_probe == tr_target:
        return t1 * math.log(q)
    e_p, e_t = math.exp(-(tr_probe - te / 2) / t1), math.exp(-(tr_target - te / 2) / t1)
    k = min(max((q - 1.0) / (1.0 - q * e_p), 0.5), 1.0)
    return t1 * math.log((1.0 + k) / (1.0 + k * e_t))


def _reference_transfer(sc, t):
    """Probes at a shorter repetition time than the target's (all the budget pays), carried over with the recovery model."""
    tr_t = t["target_scan"]["tr_ms"]
    fits, tr_p, te = _measure(sc, t)
    lim = tr_t - te - 25.0
    ti = {i: float(np.clip(_null_at(m, tr_p, tr_t, te), 20.0, lim)) for i, m in fits.items()}
    if tr_p == tr_t:                                                     # check scans at the estimates, same repetition time
        for i in _ids(t):
            p = sc.plan(dict(PROBE, tr_ms=tr_t, ti_ms=ti[i]))
            if not (p["accept"] and p["scan_s"] <= sc.budget()["remaining_s"]):
                break
            m = fits[i]
            resid = float((_probe(sc, t, tr_t, ti[i])[i] * np.conj(m["phase"])).real)
            slope = m["b"] / m["t1"] * math.exp(-ti[i] / m["t1"])
            if slope > 0:
                ti[i] = float(np.clip(ti[i] - resid / slope, 20.0, lim))
    return dict(ti_ms={i: round(v, 1) for i, v in ti.items()})


TARGET_TR_PROBES, TARGET_TR_MATRIX = 3, 32


def reference(sc):
    """Measure at the target's own repetition time whenever the budget pays three coarse probes there: the fit
    S(TI) = a - b exp(-TI / T1) then holds exactly for that scan (no model of the recovery between repetition times, of
    the inversion efficiency or of the slice profile is needed), and what is left of the budget goes to check scans at
    the estimates, each read in every region. Otherwise: probes at a shorter repetition time and the transfer model."""
    t = _task(sc)
    tr_t, ids = t["target_scan"]["tr_ms"], _ids(t)
    p = sc.plan(dict(PROBE, matrix=TARGET_TR_MATRIX, tr_ms=tr_t, ti_ms=50.0))
    if not (p["accept"] and TARGET_TR_PROBES * p["scan_s"] <= sc.budget()["remaining_s"]):
        return _reference_transfer(sc, t)
    fits, _, te = _measure(sc, t, n_probes=TARGET_TR_PROBES, reserve=0, matrix=TARGET_TR_MATRIX)
    lim = tr_t - te - 25.0
    points = {i: (list(fits[i]["tis"]), list(fits[i]["signed"])) for i in ids}
    ti = {i: float(np.clip(m["t1"] * math.log(m["b"] / m["a"]) if m["b"] > m["a"] > 0 else 200.0, 20.0, lim)) for i, m in fits.items()}
    for i in ids:
        p = sc.plan(dict(PROBE, matrix=TARGET_TR_MATRIX, tr_ms=tr_t, ti_ms=ti[i]))
        if not (p["accept"] and p["scan_s"] <= sc.budget()["remaining_s"]):
            break
        seen = _probe(sc, t, tr_t, ti[i], TARGET_TR_MATRIX)
        for j in ids:                                                    # one scan is a data point for every region
            points[j][0].append(ti[i])
            points[j][1].append(float((seen[j] * np.conj(fits[j]["phase"])).real))
        _, b, t1 = _fit(*points[i])
        slope = b / t1 * math.exp(-ti[i] / t1)
        if slope > 0:
            ti[i] = float(np.clip(ti[i] - points[i][1][-1] / slope, 20.0, lim))
    return dict(ti_ms={i: round(v, 1) for i, v in ti.items()})


def prior(sc):
    return dict(ti_ms={i: round(task.LN2 * T1_MID_MS, 1) for i in _ids(_task(sc))})


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _probe(sc, t, 500.0, 100.0)
        ti = float(recipe["ti_over_tr"] * t["target_scan"]["tr_ms"]) if recipe.get("ti_over_tr") else float(recipe["ti_ms"])
        return dict(ti_ms={i: ti for i in _ids(t)})
    return policy


DEFAULT_RECIPE = dict(ti_ms=400.0)


def brute_force(sc):
    t = _task(sc)
    tr = _probe_tr(sc, t, 6)
    te = sc.plan(dict(PROBE, tr_ms=tr, ti_ms=50.0))["realised"]["te_ms"]
    tis = [round(float(v), 1) for v in np.exp(np.linspace(math.log(60.0), math.log(tr - te - 25.0), 6))]
    vals = [_probe(sc, t, tr, ti) for ti in tis]
    return dict(ti_ms={i: tis[int(np.argmin([abs(v[i]) for v in vals]))] for i in _ids(t)})


def wrong_units(sc):
    return dict(ti_ms={i: v / 1000.0 for i, v in reference(sc)["ti_ms"].items()})


def textbook(sc):
    t = _task(sc)
    return dict(ti_ms={i: round(task.LN2 * m["t1"], 1) for i, m in _measure(sc, t)[0].items()})


# ------------------------------------------------------------------------------------------------ learnability
def parametric(params: dict):
    """The policy of ``policy_space`` with the given parameter values. ``n_probes=4, probe_tr_fraction=1,
    probe_matrix=64, finite_tr_model=True, check_scan=True`` plays exactly like ``reference``; ``finite_tr_model=False``
    without a check scan is ``textbook``; ``n_probes=0`` is ``prior``."""
    n, frac, matrix = int(params["n_probes"]), float(params["probe_tr_fraction"]), int(params["probe_matrix"])

    def policy(sc):
        if n == 0:
            return prior(sc)
        t = _task(sc)
        tr_t = t["target_scan"]["tr_ms"]
        fits, tr_p, te = _measure(sc, t, n, 1 if params["check_scan"] else 0, frac, matrix)
        lim = tr_t - te - 25.0
        if params["finite_tr_model"]:
            ti = {i: float(np.clip(_null_at(m, tr_p, tr_t, te), 20.0, lim)) for i, m in fits.items()}
        else:
            ti = {i: float(np.clip(task.LN2 * m["t1"], 20.0, lim)) for i, m in fits.items()}
        if params["check_scan"] and tr_p == tr_t:                        # as in ``reference``
            for i in _ids(t):
                p = sc.plan(dict(PROBE, matrix=matrix, tr_ms=tr_t, ti_ms=ti[i]))
                if not (p["accept"] and p["scan_s"] <= sc.budget()["remaining_s"]):
                    break
                m = fits[i]
                resid = float((_probe(sc, t, tr_t, ti[i], matrix)[i] * np.conj(m["phase"])).real)
                slope = m["b"] / m["t1"] * math.exp(-ti[i] / m["t1"])
                if slope > 0:
                    ti[i] = float(np.clip(ti[i] - resid / slope, 20.0, lim))
        return dict(ti_ms={i: round(v, 1) for i, v in ti.items()})
    return policy


def policy_space() -> dict:
    """The parametric policy family of the learnability check (the contract is in ``mrgym.learn.check``).

    ``starts`` are distributions over the parameters: a list of probabilities for a choice, P(True) for a bool,
    (mean, standard deviation) for a float. ``weak`` measures enough in 15 % of its episodes and carries the fit to the
    target repetition time in 5 %: below 2 % of passes, where a pass / fail reward has almost no group with spread.
    ``in_band`` does each of the two about two times in three: an entry-rung start."""
    return dict(
        parameters=[
            dict(name="n_probes", kind="choice", values=(0, 2, 3, 4), what="inversion times probed (0: no scan)"),
            dict(name="probe_tr_fraction", kind="float", low=0.4, high=1.0, step=0.2,
                 what="repetition time of the probes as a share of the target's (shortened further if the budget needs it)"),
            dict(name="finite_tr_model", kind="bool", what="carry the fit to the target repetition time (else TI = ln 2 x T1)"),
            dict(name="check_scan", kind="bool", what="keep budget for a check scan at the estimate and correct it"),
            dict(name="probe_matrix", kind="choice", values=(32, 64), what="matrix of the probes"),
        ],
        build=parametric,
        reference=dict(n_probes=4, probe_tr_fraction=1.0, finite_tr_model=True, check_scan=True, probe_matrix=64),
        starts=dict(
            weak=dict(n_probes=[0.45, 0.40, 0.10, 0.05], probe_tr_fraction=(0.6, 0.2), finite_tr_model=0.05, check_scan=0.5,
                      probe_matrix=[0.5, 0.5]),
            in_band=dict(n_probes=[0.15, 0.15, 0.35, 0.35], probe_tr_fraction=(0.7, 0.2), finite_tr_model=0.65, check_scan=0.5,
                         probe_matrix=[0.5, 0.5])),
        procedure=dict(measures_enough=lambda p: p["n_probes"] >= 3, models_repetition_time=lambda p: bool(p["finite_tr_model"])),
        shortcuts=dict(no_scan=lambda p: p["n_probes"] == 0, too_few_probes=lambda p: p["n_probes"] == 2,
                       textbook=lambda p: p["n_probes"] > 0 and not p["finite_tr_model"]),
    )


def blind_answers(t: dict) -> dict:
    """Answers that use the task card only: one inversion time for every region, as a constant or as a fraction of the
    told repetition time. {label: answer}; the same labels on every instance."""
    tr, ids = t["target_scan"]["tr_ms"], _ids(t)
    lim = tr - 60.0
    grid = {f"ti_ms={v:g}": float(v) for v in (150.0, 220.0, 300.0, 400.0, 520.0, 680.0, 880.0, 1150.0)}
    grid.update({f"ti_over_tr={f:g}": f * tr for f in (0.08, 0.12, 0.17, 0.23, 0.30, 0.38)})
    return {label: dict(ti_ms={i: round(min(v, lim), 1) for i in ids}) for label, v in grid.items()}


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), brute_force=brute_force,
                wrong_units=wrong_units, textbook=textbook)


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: the best constant inversion time and the best
    constant fraction of the (told) repetition time, by the per-region pass rate under the verifier's own rule."""
    from ...scanner import kspace
    from ...scanner import world as W
    rows = []
    for c in cfgs:
        w = W.make_world(c["seed"], c["world_spec"])
        e = kspace.Engine(w)
        rows += [(c["task"]["target_scan"]["tr_ms"], task.true_null_ms(c, r["id"], w, e)) for r in c["task"]["regions"]]
    tr, null = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    tr, null = tr[null == null], null[null == null]
    width = 0.05 * null                                                 # a pass window of about +- 5 % of the null
    best = None
    for kind, cand in (("ti_ms", null), ("ti_over_tr", null / tr)):
        for v in cand:
            guess = v if kind == "ti_ms" else v * tr
            rate = float(np.mean(np.abs(guess - null) <= width))
            if best is None or rate > best[0]:
                best = (rate, kind, float(v))
    return {best[1]: round(best[2], 4), "tuned_region_pass_rate": round(best[0], 3)}
