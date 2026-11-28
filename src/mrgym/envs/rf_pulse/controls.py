"""Control policies of ``Scanner/RFPulse``. Every policy plays through the agent's interface only (``info`` / ``design``
/ ``simulate`` / ``plan`` / ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world. What a
policy may use: the task card, the instrument manual and the public physics (``mrgym.design``, ``model.py``).

How the reference measures (the calibration of the stated method, level 1)
  transmit scale   one thick, long, low-amplitude pulse and the same pulse twice as strong, each imaged with ``rfscan``
                   (the slice-select gradient on the readout axis); a circle in the flat middle of the stripe, in the
                   doped water where the signal is sin(flip angle). S2 / S1 = 2 cos(s a).
  amplifier        (levels 2 and 3) a third pulse of the same shape, short enough to reach 95 % of the peak limit. Its
                   read-out against the first one's gives its actual flip angle; the saturation level A is the one at
                   which the public model (the declared amplifier law) gives that angle.

Must pass
  reference        measures, then runs the reference search of ``model.py`` on the chain it measured, asking for validity
                   at the corners of its measurement error as well; delivers that design. Level 1: the stated method to
                   the letter (no search)
Must fail
  prior            no scan: the best design of the same search on the IDEAL chain (the measurement gate fails it)
  ideal_tuned      one token scan, then that same design: tuned to the simulator, and failed by the scanner
  default_pulse    one token scan, then the console's own excitation: a Hann-windowed sinc with the time-bandwidth
                   product of the pulse ``normal``, at the specified angle and thickness, as short as the limits allow
  fixed_recipe     one token scan, then the best constant correction, tuned on other worlds: one amplitude, one stretch
                   of the duration, for every scanner
  random_search    one token scan, then generator parameters drawn at random within the tool-call limit, each judged
                   by ``simulate``; the shortest one that meets the specification on the simulator (else the nearest)
  wrong_units      the reference's design with the slice thickness in centimetres (cut at the tool's 1 mm)
Ablations (the reference with one hidden quantity ignored; each must lose at the levels it names)
  ignores_transmit   the reference with the transmit scale taken as 1 (every level)
  ignores_amplifier  the reference with a linear amplifier (levels 2 and 3)
Reported
  open_loop        a fixed list of scans (the calibration scans of the level, in their order) and a fixed design rule
                   (the search on the measured chain and its corners). The reference never chooses a scan from the result
                   of an earlier one, so at levels 2 and 3 this is the reference itself and at level 1 the search in place
                   of the stated method: the family is a measurement task by the 25-point rule, and says so
"""
from __future__ import annotations

import math

import numpy as np

from ...design import hardware, tool
from ...scanner import console
from . import model, task

REFERENCE = "reference"
MUST_FAIL = ("prior", "ideal_tuned", "default_pulse", "fixed_recipe", "random_search", "wrong_units")
ABLATIONS = {"ignores_transmit": (1, 2, 3), "ignores_amplifier": (2, 3)}
REPORTED = ("open_loop",)
DESCRIPTIONS = dict(
    prior="no scan; the best design of the search on the ideal simulator",
    reference="two or three calibration scans give the transmit scale and the amplifier's knee; then the reference search on that chain",
    ideal_tuned="one token scan, then the best design of the same search on the ideal simulator",
    default_pulse="one token scan, then the console's own pulse: a Hann sinc of time-bandwidth 4, as short as the limits allow",
    fixed_recipe="one token scan, then one constant amplitude and stretch for every scanner, tuned on other worlds",
    random_search="one token scan, then random generator parameters within the call limit, judged by the simulator",
    wrong_units="the reference's design with the slice thickness in centimetres",
    ignores_transmit="the reference with the transmit scale taken as 1",
    ignores_amplifier="the reference with a linear amplifier",
    open_loop="the calibration scans in their fixed order and the search on the measured chain (levels 2, 3: the reference itself)",
)
TOKEN_SCAN = dict(sequence="spgr", tr_ms=12.0, flip_deg=10.0, matrix=64)
SCALE_ERROR, KNEE_ERROR, DF_HZ = 0.008, 0.02, 5.0          # the corners of the reference: relative errors of its two estimates, off-resonance
KNEE_PEAK, KNEE_FLIP_DEG = 0.95, 35.0                       # the third calibration pulse: part of the peak limit it reaches, angle it would turn
RANDOM_PAIRS = 12                                           # design + simulate pairs of ``random_search``


def _task(sc):
    return sc.info()["task"]


def _read(resp) -> float:
    return float(resp["roi_readouts"][0]["complex_mean_magnitude"][0])


def _disc(pulse, chain, radius_mm=task.CAL_CIRCLE["radius_mm"]) -> float:
    """What the read-out circle sees of a stripe on the public model: sin(flip angle) averaged over a disc of the
    given radius centred on the stripe (the chord at every position weighs it)."""
    x = np.linspace(-radius_mm, radius_mm, 81)
    w = np.sqrt(np.maximum(radius_mm ** 2 - x ** 2, 0.0))
    return float(np.sum(w * np.sin(np.radians(hardware.flip_deg(pulse, chain, x)))) / np.sum(w))


def _scan(sc, number):
    return _read(sc.scan(dict(task.CAL_SCAN, rf_design=number), [task.CAL_CIRCLE]))


def measure_scale(sc):
    """The two calibration scans of the stated method: (transmit scale, signal of a spin turned by 90 degrees)."""
    one = sc.design(dict(task.CAL))
    two = sc.design(dict(task.CAL, amplitude=task.CAL_AMPLITUDES[1]))
    s1, s2 = _scan(sc, one["design"]), _scan(sc, two["design"])
    a = one["facts"]["flip_deg"]
    scale = math.degrees(math.acos(float(np.clip(s2 / (2 * s1), -1.0, 1.0)))) / a
    return scale, s1 / math.sin(math.radians(scale * a))


def measure_knee(sc, t, scale, unit):
    """The third calibration scan: the saturation level of the amplifier (uT), or None when the pulse came through
    uncompressed. The calibration shape, played so strongly and so briefly that it would turn the centre by
    ``KNEE_FLIP_DEG`` on a linear chain of the measured scale while its amplitude peaks at ``KNEE_PEAK`` of the peak
    limit (the duration is adjusted on the peak the ``design`` tool reports, as an operator would)."""
    limits = model.limits_of(t)
    low = tool.build(dict(task.CAL), limits)[0]
    lo, hi = tool.PARAMETERS["amplitude"][:2]
    amplitude = round(float(np.clip(KNEE_FLIP_DEG / (scale * low.flip_deg()), lo, hi)), 4)
    want = KNEE_PEAK * limits.b1_peak_ut
    floor = max(0.3, task.CAL["tbw"] * 1e3 / (model.HZ_PER_MT_M_MM * task.CAL["thickness_mm"] * limits.g_max_mt_m))
    up = lambda x: round(math.ceil(max(x, floor) / 0.01 - 1e-9) * 0.01, 2)   # noqa: E731
    duration = up(low.duration_ms * amplitude * low.peak_b1_ut / want)
    for _ in range(4):
        got = sc.design(dict(task.CAL, duration_ms=duration, amplitude=amplitude))
        peak = got["facts"]["peak_b1_ut"]
        longer = up(duration * peak / want)
        if peak <= limits.b1_peak_ut and (abs(peak / want - 1) < 0.03 or longer == duration):
            break
        duration = longer
    pulse = tool.build(got["call"], limits)[0]
    target = float(np.clip(_scan(sc, got["design"]) / unit, 0.0, 1.0))

    def seen(a_sat):
        return _disc(pulse, hardware.Chain(scale, a_sat))
    if target >= seen(None) * (1 - 5e-3):
        return None
    lo, hi = 2.0, 400.0                                                  # the compression falls as A rises
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if seen(mid) < target else (lo, mid)
    return math.sqrt(lo * hi)


def _chains(scale, knee, corners=True):
    """The measured chain and, with ``corners``, what the measurement leaves open around it."""
    centre = hardware.Chain(scale, knee)
    if not corners:
        return [centre]
    out = [centre]
    for ds, dk, df in ((1, -1, 1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1)):
        out.append(hardware.Chain(scale * (1 + ds * SCALE_ERROR), None if knee is None else knee * (1 + dk * KNEE_ERROR), 0.0, df * DF_HZ))
    return out


def _searched(t, chains) -> dict:
    """The call of the reference search on ``chains``."""
    best = model.search(model.spec_of(t), model.limits_of(t), chains)
    if best is None:
        raise RuntimeError("the search found no design for this chain")
    return best["call"]


def _stated(sc, t, scale) -> dict:
    """Level 1, to the letter: the generator call of the method text, at the shortest duration by its rule."""
    spec, lim, m = t["design"]["specification"], t["design"]["limits"], t["method"]
    call = dict(kind="slr", flip_deg=spec["flip_deg"], thickness_mm=spec["thickness_mm"], amplitude=round(1.0 / scale, 4),
                tol_in_deg=round(m["share"][0] * spec["tol_in_deg"], 4), max_out_deg=round(m["share"][1] * spec["max_out_deg"], 4), tbw=m["tbw"])
    up = lambda x: round(math.ceil(x / 0.01 - 1e-9) * 0.01, 2)           # noqa: E731
    call["duration_ms"] = up(m["tbw"] / (0.042577 * spec["thickness_mm"] * lim["g_max_mt_m"]))
    for _ in range(4):
        peak = sc.design(call)["facts"]["peak_b1_ut"]
        if peak <= lim["b1_peak_ut"]:
            break
        call["duration_ms"] = up(call["duration_ms"] * peak / lim["b1_peak_ut"])
    return call


def _amplifier(t) -> bool:
    return "amplifier" in t["design"]["hidden"]


def _reference_call(sc, corners=True, transmit=True, amplifier=True) -> tuple:
    """(task card, the call the reference would deliver) with parts of its measurement switched off."""
    t = _task(sc)
    scale, unit = measure_scale(sc)
    if t.get("method") and transmit:
        return t, _stated(sc, t, scale)
    knee = measure_knee(sc, t, scale, unit) if amplifier and _amplifier(t) else None
    return t, _searched(t, _chains(scale if transmit else 1.0, knee, corners))


def _store(sc, call) -> dict:
    return dict(design=sc.design(call)["design"])


def reference(sc):
    return _store(sc, _reference_call(sc)[1])


def open_loop(sc):
    """The calibration scans in their fixed order, then one fixed rule: the search on the measured chain and its
    corners. At levels 2 and 3 this IS the reference; at level 1 it differs from the stated method by the search."""
    t = _task(sc)
    scale, unit = measure_scale(sc)
    knee = measure_knee(sc, t, scale, unit) if _amplifier(t) else None
    return _store(sc, _searched(t, _chains(scale, knee)))


def ignores_transmit(sc):
    return _store(sc, _reference_call(sc, transmit=False)[1])


def ignores_amplifier(sc):
    return _store(sc, _reference_call(sc, amplifier=False)[1])


def _token(sc):
    sc.scan(dict(TOKEN_SCAN))


def ideal_tuned(sc):
    t = _task(sc)
    _token(sc)
    return _store(sc, _searched(t, [hardware.IDEAL]))


def default_pulse(sc):
    t = _task(sc)
    _token(sc)
    spec, lim = t["design"]["specification"], t["design"]["limits"]
    tbw = float(console.RF.PULSES["normal"][1])
    call = dict(kind="sinc", flip_deg=spec["flip_deg"], tbw=tbw, thickness_mm=spec["thickness_mm"], window="hann")
    duration = model.gradient_ms(spec, model.limits_of(t), tbw)
    got = sc.design(dict(call, duration_ms=duration))
    for _ in range(4):
        f = got["facts"]
        need = max(f["peak_b1_ut"] / lim["b1_peak_ut"], (f["b1rms_ut"] / lim["b1rms_ut"]) ** 2)
        if need <= 1.0:
            break
        duration = round(math.ceil(duration * need / 0.01 - 1e-9) * 0.01, 2)
        got = sc.design(dict(call, duration_ms=duration))
    return dict(design=got["design"])


DEFAULT_RECIPE = dict(amplitude=1.0, stretch=1.0)


def recipe_tbw(t):
    """The time-bandwidth product a constant recipe uses for a task: the smallest of the stated shares on the ideal model."""
    return model.smallest_tbw(model.spec_of(t), model.limits_of(t), task.METHOD_SHARE, [hardware.IDEAL])


def recipe_call(t, recipe, tbw=None):
    """The design of a constant recipe for a task: one amplitude and one stretch of the shortest duration. None when it
    cannot be built."""
    tbw = recipe_tbw(t) if tbw is None else tbw
    got = None if tbw is None else model.shortest(model.spec_of(t), model.limits_of(t), tbw, task.METHOD_SHARE, recipe["amplitude"], recipe["stretch"])
    return None if got is None else got[1]


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _token(sc)
        call = recipe_call(t, recipe) or recipe_call(t, DEFAULT_RECIPE)
        return dict(design=sc.design(call)["design"])
    return policy


def random_search(sc):
    """Random generator parameters, each judged by the simulator (which is ideal): the shortest design that meets the
    specification there, else the one nearest to it."""
    t = _task(sc)
    _token(sc)
    spec = t["design"]["specification"]
    seed = int(1e6 * spec["thickness_mm"] + 1e3 * spec["flip_deg"] + 100 * spec["transition"])      # a function of the task card only
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(RANDOM_PAIRS):
        tbw = round(float(rng.uniform(2.0, 10.0)), 1)
        call = dict(kind="slr", flip_deg=spec["flip_deg"], thickness_mm=spec["thickness_mm"], tbw=tbw,
                    amplitude=round(float(np.exp(rng.uniform(math.log(0.5), math.log(2.0)))), 3),
                    tol_in_deg=round(float(rng.uniform(0.3, 1.0)) * spec["tol_in_deg"], 3), max_out_deg=round(float(rng.uniform(0.5, 1.0)) * spec["max_out_deg"], 3),
                    duration_ms=round(math.ceil(model.gradient_ms(spec, model.limits_of(t), tbw) * float(rng.uniform(1.0, 3.0)) / 0.01) * 0.01, 2))
        try:
            got = sc.design(call)
        except Exception:  # noqa: BLE001  a refused call is a wasted draw
            continue
        if not got["facts"]["limits"]["ok"]:
            continue
        sim = sc.simulate(got["design"])
        ok = all(sim["checks"].values())
        key = (0 if ok else 1, got["facts"]["duration_ms"] if ok else -min(sim["margins"].values()))
        if best is None or key < best[0]:
            best = (key, got["design"])
    if best is None:
        return _store(sc, _searched(t, [hardware.IDEAL]))
    return dict(design=best[1])


def wrong_units(sc):
    """The reference's design with the slice thickness in centimetres. The tool takes 1 to 50 mm, so the value is
    cut at 1 mm; the duration is lengthened by the same factor, which keeps the gradient where it was."""
    t, call = _reference_call(sc, corners=False)
    thin = max(call["thickness_mm"] / 10.0, tool.PARAMETERS["thickness_mm"][0])
    duration = round(math.ceil(call["duration_ms"] * call["thickness_mm"] / thin / 0.01 - 1e-9) * 0.01, 2)
    return _store(sc, dict(call, thickness_mm=thin, duration_ms=min(duration, tool.MAX_DURATION_MS)))


def prior(sc):
    """What ``mrgym.policies.run`` delivers when a policy raises: the design for the ideal chain, no scan."""
    return _store(sc, _searched(_task(sc), [hardware.IDEAL]))


def policies(recipe=None):
    return dict(reference=reference, prior=prior, ideal_tuned=ideal_tuned, default_pulse=default_pulse, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE),
                random_search=random_search, wrong_units=wrong_units, ignores_transmit=ignores_transmit, ignores_amplifier=ignores_amplifier,
                open_loop=open_loop)


def tune(cfgs):
    """The strongest constant recipe on other worlds: one amplitude and one stretch of the duration for every
    scanner, by the share of worlds on which the design passes the verifier's rule."""
    from ...scanner import world as W
    worlds = []
    for c in cfgs:
        w = W.make_world(c["seed"], c["world_spec"])
        task.reference(c, w)                                             # once per world, before the candidates
        worlds.append((c, w, recipe_tbw(c["task"])))
    stretches = (1.0, 1.4, 1.8, 2.3) if cfgs[0]["world_spec"]["rf_hardware"].get("amp_sat_ut") else (1.0,)
    best = None
    for amplitude in np.round(np.exp(np.linspace(math.log(0.55), math.log(1.8), 9)), 3):
        for stretch in stretches:
            recipe, passed = dict(amplitude=float(amplitude), stretch=float(stretch)), 0
            for c, w, tbw in worlds:
                call = recipe_call(c["task"], recipe, tbw)
                if call is None:
                    continue
                limits = model.limits_of(c["task"])
                w["console"]["designs"] = {1: dict(index=1, call=call, call_json="", facts=tool.facts(tool.build(call, limits)[0], limits))}
                passed += bool(task.score(c, dict(design=1), w)["passed"])
            rate = passed / len(worlds)
            if best is None or rate > best[0]:
                best = (rate, recipe)
    return dict(best[1], tuned_pass_rate=round(best[0], 3))
