"""Control policies of ``Scanner/RFPulse-v1``. Every policy plays through the agent's interface only (``info`` / ``design``
/ ``simulate`` / ``plan`` / ``scan`` / ``budget``) and returns the deliverable. None reads the hidden world. What a
policy may use: the task card, the manual and the public physics (``mrgym.design``, ``model.py``).

How the reference measures (three scans of one wide-band calibration pulse; level 1: the first two)
  transmit scale   the pulse and the same pulse twice as strong, a circle in the flat middle of the stripe, in the
                   doped water where the signal is sin(flip angle): S2 / S1 = 2 cos(s a)
  gradient delay   the phase of the first scan along x across the middle of the stripe (the ``profile`` tool, free,
                   pixel by pixel): its slope is 2 pi gamma G delay per mm (the calibration pulse is symmetric in
                   time, its own phase is flat). Two circles would do, but a circle of a few pixels does not sit
                   where it is asked for: its phase is that of its pixels' centres
  off-resonance    the first scan again at an echo time 0.4 ms later: the phase of the centre circle runs by
                   2 pi x offset x 0.4 ms
Then ``model.design_rule``: the shortest ``minphase`` pulse within the limits, with the three corrections.

Must pass
  reference        as above; level 1: the stated method to the letter
Must fail
  prior            no scan: the design of the rule for an ideal chain (the measurement gate fails it)
  ideal_tuned      one token scan, then that same design: tuned to the simulator, and failed by the scanner
  default_pulse    one token scan, then the console's own excitation: a Hann-windowed sinc with the time-bandwidth
                   product of the pulse ``normal``, at the specified angle and thickness, at the gradient limit
  fixed_recipe     one token scan, then the best constant correction, tuned on other worlds: one amplitude, one
                   frequency offset, one delay, one part of the gradient limit, for every scanner
  random_search    one token scan, then generator parameters drawn at random within the tool-call limit, each judged
                   by ``simulate``; the shortest one that meets the specification on the simulator (else the nearest)
  wrong_units      the reference's design with the slice thickness in centimetres
Ablations (the reference with one hidden quantity ignored; each must lose at the levels it names)
  ignores_transmit   the transmit scale taken as 1
  ignores_offset     the off-resonance taken as 0 (the third scan is not made)
  ignores_delay      the gradient delay taken as 0
  v0_reference       the reference of ``Scanner/RFPulse-v0``: it measures the transmit scale and nothing else, so it
                     ignores the two quantities this id adds. At level 1, where they are small, it is the reference
Reported
  open_loop        the reference's scan list, which is fixed, and a design rule that does not branch on a measurement:
                   the measured scale, offset and delay enter the amplitude, the frequency and the rephasing lobe by
                   their formulas, and the gradient is one number chosen from the task card alone: the part of the
                   gradient limit with the best chance over the told range of the transmit scale, by the public
                   physics (``expected_bound``). It cannot ask which limit binds on THIS scanner. A rule that may (lower the gradient until
                   the limits hold) on the same fixed scans is the reference itself
"""
from __future__ import annotations

import math

import numpy as np

from ...design import bound, tool
from ...design.units import RAD_PER_MT_M_MM_MS
from ...scanner import console
from . import model, task

REFERENCE = "reference"
MUST_FAIL = ("prior", "ideal_tuned", "default_pulse", "fixed_recipe", "random_search", "wrong_units")
ABLATIONS = {"ignores_transmit": (1, 2, 3), "ignores_offset": (2, 3), "ignores_delay": (2, 3), "v0_reference": (2, 3)}
REPORTED = ("open_loop",)
DESCRIPTIONS = dict(
    prior="no scan; the shortest design of the rule for an ideal chain",
    reference="three calibration scans give the transmit scale, the off-resonance and the gradient delay; then the shortest minphase pulse within the limits",
    ideal_tuned="one token scan, then the shortest design of the rule for an ideal chain",
    default_pulse="one token scan, then the console's own pulse: a Hann sinc of time-bandwidth 4 at the gradient limit",
    fixed_recipe="one token scan, then one constant amplitude, frequency, delay and gradient for every scanner, tuned on other worlds",
    random_search="one token scan, then random generator parameters within the call limit, judged by the simulator",
    wrong_units="the reference's design with the slice thickness in centimetres",
    v0_reference="the reference of Scanner/RFPulse-v0: the transmit scale is measured, the off-resonance and the delay are not",
    ignores_transmit="the reference with the transmit scale taken as 1",
    ignores_offset="the reference with the off-resonance taken as 0",
    ignores_delay="the reference with the gradient delay taken as 0",
    open_loop="the reference's fixed scans; the corrections by formula; one gradient chosen from the task card, not from the measurement",
)
TOKEN_SCAN = dict(sequence="spgr", tr_ms=12.0, flip_deg=10.0, matrix=64)
RANDOM_PAIRS = 12                                           # design + simulate pairs of ``random_search``
GRADIENT_PARTS = (1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.35, 0.3, 0.25)   # of the gradient limit: what ``open_loop`` chooses from
SCALE_GRID = 13                                             # transmit scales of the told range on which ``open_loop`` counts passes
PHASE_LINE = dict(half_mm=10.0, width_mm=10.0)              # the line of the first scan whose phase slope gives the gradient delay


def _task(sc):
    return sc.info()["task"]


def _rows(resp) -> dict:
    return {r["id"]: (float(r["complex_mean_magnitude"][0]), float(r["complex_mean_phase_rad"][0])) for r in resp["roi_readouts"]}


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def measure(sc, offset=True):
    """The calibration scans: dict(scale, delay_us, df_hz). ``offset`` False: the third scan is not made (df_hz 0)."""
    one = sc.design(dict(task.CAL))
    two = sc.design(dict(task.CAL, amplitude=task.CAL_AMPLITUDES[1]))
    resp = sc.scan(dict(task.CAL_SCAN, rf_design=one["design"]), task.CAL_CIRCLES)
    first = _rows(resp)
    line = sc.profile(resp["scan_index"], "x", kind="phase", width_mm=PHASE_LINE["width_mm"], start_mm=-PHASE_LINE["half_mm"], stop_mm=PHASE_LINE["half_mm"])
    slope = float(np.polyfit(np.array(line["positions_mm"]), np.unwrap(np.array(line["values"])), 1)[0])       # rad per mm
    second = _rows(sc.scan(dict(task.CAL_SCAN, rf_design=two["design"]), task.CAL_CIRCLES))
    a = one["facts"]["flip_deg"]
    scale = math.degrees(math.acos(float(np.clip(second["centre"][0] / (2 * first["centre"][0]), -1.0, 1.0)))) / a
    delay_us = 1e3 * slope / (RAD_PER_MT_M_MM_MS * one["facts"]["gradient_mt_m"])
    df = 0.0
    if offset:
        late = _rows(sc.scan(dict(task.CAL_SCAN, rf_design=one["design"], te_ms=task.CAL_TE2_MS), task.CAL_CIRCLES))
        df = _wrap(late["centre"][1] - first["centre"][1]) / (2 * math.pi * (task.CAL_TE2_MS - task.CAL_SCAN["te_ms"]) * 1e-3)
    return dict(scale=scale, delay_us=delay_us, df_hz=df)


def measure_scale(sc) -> float:
    """The two calibration scans of the stated method (level 1, and all that ``-v0`` measured)."""
    one = sc.design(dict(task.CAL))
    two = sc.design(dict(task.CAL, amplitude=task.CAL_AMPLITUDES[1]))
    s1 = _rows(sc.scan(dict(task.CAL_SCAN, rf_design=one["design"]), task.CAL_CIRCLES))["centre"][0]
    s2 = _rows(sc.scan(dict(task.CAL_SCAN, rf_design=two["design"]), task.CAL_CIRCLES))["centre"][0]
    return math.degrees(math.acos(float(np.clip(s2 / (2 * s1), -1.0, 1.0)))) / one["facts"]["flip_deg"]


def _full(t) -> bool:
    return "off_resonance" in t["design"]["hidden"]


def _rule(t, scale=1.0, df_hz=0.0, delay_us=0.0) -> dict:
    return model.design_rule(model.spec_of(t), model.limits_of(t), scale, df_hz, delay_us)[1]


def _store(sc, call) -> dict:
    return dict(design=sc.design(call)["design"])


def _stated(t, scale) -> dict:
    """Level 1, to the letter: the generator call of the method text."""
    spec = t["design"]["specification"]
    return dict(kind="minphase", flip_deg=spec["flip_deg"], thickness_mm=spec["thickness_mm"], transition=spec["transition"], **t["method"],
                amplitude=round(1.0 / scale, 4))


def _reference_call(sc, transmit=True, offset=True, delay=True) -> dict:
    t = _task(sc)
    if not _full(t):
        scale = measure_scale(sc)
        return _stated(t, scale) if transmit and t.get("method") else _rule(t, scale if transmit else 1.0)
    m = measure(sc, offset)
    return _rule(t, m["scale"] if transmit else 1.0, m["df_hz"], m["delay_us"] if delay else 0.0)


def reference(sc):
    return _store(sc, _reference_call(sc))


def ignores_transmit(sc):
    return _store(sc, _reference_call(sc, transmit=False))


def ignores_offset(sc):
    return _store(sc, _reference_call(sc, offset=False))


def ignores_delay(sc):
    return _store(sc, _reference_call(sc, delay=False))


def v0_reference(sc):
    return _store(sc, _rule(_task(sc), measure_scale(sc)))


def expected_bound(t, scale: float) -> int:
    """What a policy expects the bound to be on a scanner of the transmit scale ``scale``, without the certificates:
    the samples of the shortest pulse for the whole of both tolerances at the gradient limit, times gradient limit /
    largest gradient the energy and the peak arguments of the bound leave (``bound.gradient_cap``; the samples a mask
    needs are inversely proportional to the gradient; three rounds, since the cap depends a little on the count)."""
    spec, limits = model.spec_of(t), model.limits_of(t)
    mask = bound.Mask.of(spec)
    n = first = model.shortest(spec, limits.g_max_mt_m, (1.0, 1.0))
    for _ in range(3):
        cap = bound.gradient_cap(n, mask, scale, limits.b1_peak_ut, model.energy_limit(limits)) / bound.precession(1.0)
        n = int(math.ceil(first * max(1.0, limits.g_max_mt_m / cap))) if cap > 0 else bound.MAX_SAMPLES
    return n


def gradient_part(t) -> float:
    """What ``open_loop`` plays for a task card: the part of the gradient limit at which the shortest pulse, played
    1 / s times as strong, is within the RF limits and short enough on the largest share of the told range of the
    transmit scale (log-uniform), by ``expected_bound``. A function of the task card."""
    spec, limits = model.spec_of(t), model.limits_of(t)
    scales = np.exp(np.linspace(*np.log(task.TX_RANGE), SCALE_GRID))
    bounds = [expected_bound(t, float(s)) for s in scales]
    full, _ = model.design_at(spec, limits, limits.g_max_mt_m)          # at a part p of the gradient limit: energy and peak x p, samples / p
    best = (-1, 1.0)
    for part in GRADIENT_PARTS:
        ok = sum(bool(part * full.energy_ut2_ms / s ** 2 <= model.energy_limit(limits) and part * full.peak_b1_ut / s <= limits.b1_peak_ut
                      and math.ceil(full.n / part) <= t["duration_factor"] * n) for s, n in zip(scales, bounds))
        if ok > best[0]:
            best = (ok, part)
    return best[1]


def open_loop(sc):
    t = _task(sc)
    m = measure(sc) if _full(t) else dict(scale=measure_scale(sc), df_hz=0.0, delay_us=0.0)
    spec, limits = model.spec_of(t), model.limits_of(t)
    return _store(sc, model.design_at(spec, limits, gradient_part(t) * limits.g_max_mt_m, m["scale"], m["df_hz"], m["delay_us"])[1])


def _token(sc):
    sc.scan(dict(TOKEN_SCAN))


def prior(sc):
    """What ``mrgym.policies.run`` delivers when a policy raises: the design for the ideal chain, no scan."""
    return _store(sc, _rule(_task(sc)))


def ideal_tuned(sc):
    t = _task(sc)
    _token(sc)
    return _store(sc, _rule(t))


def default_pulse(sc):
    t = _task(sc)
    _token(sc)
    spec, lim = t["design"]["specification"], t["design"]["limits"]
    tbw = float(console.RF.PULSES["normal"][1])
    duration = round(math.ceil(tbw * 1e3 / (model.HZ_PER_MT_M_MM * spec["thickness_mm"] * lim["g_max_mt_m"]) / 0.01 - 1e-9) * 0.01, 2)
    return _store(sc, dict(kind="sinc", flip_deg=spec["flip_deg"], tbw=tbw, thickness_mm=spec["thickness_mm"], window="hann", duration_ms=duration))


DEFAULT_RECIPE = dict(amplitude=1.0, freq_hz=0.0, delay_us=0.0, part=1.0)


def recipe_call(t, recipe) -> dict:
    """The design of a constant recipe for a task."""
    spec, limits = model.spec_of(t), model.limits_of(t)
    return model.design_at(spec, limits, recipe["part"] * limits.g_max_mt_m, 1.0 / recipe["amplitude"], recipe["freq_hz"], recipe["delay_us"])[1]


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _token(sc)
        return _store(sc, recipe_call(t, recipe))
    return policy


def random_search(sc):
    """Random generator parameters, each judged by the simulator (which is ideal): the shortest design that meets the
    specification there, else the one nearest to it."""
    t = _task(sc)
    _token(sc)
    spec, lim = t["design"]["specification"], t["design"]["limits"]
    seed = int(1e6 * spec["thickness_mm"] + 1e3 * spec["flip_deg"] + 100 * spec["transition"])      # a function of the task card only
    rng = np.random.default_rng(seed)
    best = None
    shortest = model.shortest(spec, lim["g_max_mt_m"], (1.0, 1.0))
    for _ in range(RANDOM_PAIRS):
        n = int(round(shortest * float(rng.uniform(0.9, 2.5))))
        g = lim["g_max_mt_m"] * float(rng.uniform(0.4, 1.0))
        full = _full(t)
        call = model.call_of(spec, n, g, (float(rng.uniform(0.5, 1.0)),) * 2, float(np.exp(rng.uniform(math.log(0.5), math.log(2.0)))),
                             float(rng.uniform(-task.OFFSET_HZ, task.OFFSET_HZ)) if full else 0.0)
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
        return _store(sc, _rule(t))
    return dict(design=best[1])


def wrong_units(sc):
    """The reference's design with the slice thickness in centimetres (cut at the tool's 1 mm); the duration is
    lengthened by the same factor, which keeps the gradient where it was."""
    call = _reference_call(sc)
    thin = max(call["thickness_mm"] / 10.0, tool.PARAMETERS["thickness_mm"][0])
    duration = round(math.ceil(call["duration_ms"] * call["thickness_mm"] / thin / 0.01 - 1e-9) * 0.01, 2)
    return _store(sc, dict({k: v for k, v in call.items() if k != "rephase_fraction"}, thickness_mm=thin, duration_ms=min(duration, tool.MAX_DURATION_MS)))


def policies(recipe=None):
    return dict(reference=reference, prior=prior, ideal_tuned=ideal_tuned, default_pulse=default_pulse, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE),
                random_search=random_search, wrong_units=wrong_units, v0_reference=v0_reference, ignores_transmit=ignores_transmit,
                ignores_offset=ignores_offset, ignores_delay=ignores_delay, open_loop=open_loop)


def recipes(full: bool, binding: bool) -> list:
    """The constant recipes ``tune`` chooses from."""
    amplitudes = [float(a) for a in np.round(np.exp(np.linspace(math.log(0.6), math.log(1.7), 9)), 3)]
    freqs = (0.0, -150.0, 150.0) if full else (0.0,)
    delays = (0.0, -90.0, 90.0) if full else (0.0,)
    parts = (1.0, 0.6) if binding else (1.0,)
    return [dict(amplitude=a, freq_hz=f, delay_us=d, part=p) for a in amplitudes for f in freqs for d in delays for p in parts]


def passes(cfg, world, call) -> bool:
    """Whether the design of ``call`` passes the verifier's rule on ``world`` (for tuning: no episode is played)."""
    limits = model.limits_of(cfg["task"])
    world["console"]["designs"] = {1: dict(index=1, call=call, call_json="", facts=tool.facts(tool.build(call, limits)[0], limits))}
    return bool(task.score(cfg, dict(design=1), world)["passed"])


def tune(cfgs):
    """The strongest constant recipe on other worlds: by the share of worlds on which its design passes the verifier's rule."""
    from ...scanner import world as W
    worlds = [(c, W.make_world(c["seed"], c["world_spec"])) for c in cfgs]
    lv = task.LEVELS[cfgs[0]["level"]]
    best = None
    for recipe in recipes(len(lv["hidden"]) > 1, lv["power"] == "binding"):
        passed = 0
        for c, w in worlds:
            try:
                passed += passes(c, w, recipe_call(c["task"], recipe))
            except (ValueError, tool.Refusal):
                continue
        if best is None or passed > best[0]:
            best = (passed, recipe)
    return dict(best[1], tuned_pass_rate=round(best[0] / len(worlds), 3))

