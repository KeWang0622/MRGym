"""``Scanner/RFPulse``: instances and the verifier. The first design task: the agent delivers a piece of a sequence.

The job: design a slice-selective excitation pulse that meets a flip-angle specification on THIS scanner, within
hardware limits, and not much longer than it has to be. Designs are made with the ``design`` tool (a generator and
its parameters), looked at on the public model with ``simulate``, played on the scanner with ``scan`` (sequence
``rfscan``) and delivered by number.

Why it cannot be done on the simulator alone. The simulator is ideal. The scanner's transmit chain is not
(``mrgym.design.hardware``), and its sizes are hidden:
  transmit scale   the transmitter is not calibrated: the flip angle a spin gets is the nominal one times a factor
                   between 0.6 and 1.6 (log-uniform; every level). The specification is on the FLIP ANGLE, within 3 or
                   4 %: an uncorrected design misses it. One hidden number in a known range leaves a constant correction
                   a chance of about 2 x margin / ln(1.6 / 0.6): the gate row ``fixed_recipe`` measures it. On |Mxy| the specification would not
                   need the scale at all near 90 degrees: a 10 % transmit error moves |Mxy| by 1 % there.
  amplifier        (levels 2 and 3) amplitudes are compressed, a -> a / (1 + (a / A)^6)^(1/6), A between 11 and 17 uT
                   with a peak limit of 20 uT. The shortest pulse of the ideal model runs at the peak limit; there the
                   peak of the main lobe loses 19 to 45 % and the side lobes do not, and the profile leaves the mask. The pulse has
                   to be about twice as long, and how long depends on A.
The RF-to-gradient delay of the chain is modelled and is zero in this family: its generators play a constant
gradient, which no delay changes.

Deliverable    {"design": 3}        the number of one stored design
Pass rule      on the hidden scanner, at the isocentre (the exact rotation of the delivered samples through the
               scanner's chain, on positions 1/100 of the thickness apart, ``mrgym.design.tool.figures``):
                 1. every limit holds, recomputed from the delivered samples: peak B1, gradient, slew rate, RF power
                    (B1rms over the stated repetition time), duration;
                 2. the mask holds: flip angle within the tolerance in the slice, below the bound outside it out to
                    the end of the object, and the in-slice signal after the rephasing lobe at least 95 % of what the
                    flip angles give;
                 3. a design that reaches the flip angle of the slice is not shorter than the floor (one that is, is an
                    engine defect: it fails, and is logged);
                 4. the duration is at most ``factor`` (level 1: 1.5, level 2: 1.35, level 3: 1.25) times the duration
                    of the reference search on the hidden scanner (``model.search``).
               THE PASS RULE HAS A REFERENCE IN IT. Clause 4 compares with the best design a stated search over one
               generator finds, not with a bound: no proven bound tighter than the floor of clause 3 is known to this
               package (docs/physics/design.md, section 4), and the floor is 3 to 18 times below the reference
               search on the instances looked at. A design shorter than the reference passes; ``merit_reference`` > 1 records it.
Never paid     ``info["score"]``: duration, the reference and the floor, ``merit_reference`` = reference / duration,
               ``merit_floor`` = floor / duration, and the key of the specification class for a record table.

What the mask does not look at (known, stated): the band between the slice and the outside is free (two designs of
5.7 and 4.7 mm full width at half maximum both pass a 5 mm specification with a transition of 0.5); positions beyond
the end of the object; relaxation during the pulse. The positions are 1/100 of the thickness apart and never fewer
than four per period of the finest oscillation the pulse's selection area allows (``tool.positions``).

What a pass means: the design meets this specification on the modelled scanner. Nothing more: the model has no
relaxation during the pulse, no eddy currents, one transmit channel, and its limits are prices of this package, not
an assessment of any kind for a person or a real system.
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np

from ...design import hardware, tool
from ...scanner import rfscan
from ...scanner import world as W
from .. import family as F
from . import model

FAMILY = "rf_pulse"
ENV_ID = "Scanner/RFPulse-v0"
SUBMIT_DOC = "Deliver the number of one stored design: the excitation pulse for the specification."
DESIGN_TOOLS = dict(kinds=("sinc", "slr", "points"))        # constant-gradient generators: see the module docstring
OUTCOMES = ("limit_violated", "mask_in_slice", "mask_out_of_slice", "mask_phase", "below_floor", "too_long")   # worst first
COMPETENCIES = ("qc.transmit_gain", "parameter.flip_angle")
REFOCUSED_MIN = 0.95
EXTENT_MM = rfscan.OBJECT_HALF_MM
TR_LOOSE_MS, TR_BINDING_MS = 30.0, 10.0
TX_RANGE = (0.6, 1.6)                                       # of the worlds (log-uniform), and told
AMP_RANGE_UT = (11.0, 17.0)                                 # of the worlds; told as 10 to 18
AMP_TOLD_UT = (10.0, 18.0)
INSERT = dict(radius_mm=25.0, t1_ms=83.7, t2_ms=60.0)       # the doped-water cylinder at the isocentre; told: its radius, and
T1_TOLD_BELOW_MS = 100.0                                    # that its T1 is below this (a T1 itself is never told: ``mrgym check``)
# Calibration scans of the stated method (level 1) and of the control policies: a thick, long, low-amplitude slice, so
# that the stripe is flat over the read-out circle and the amplifier does nothing to it.
CAL = dict(kind="slr", flip_deg=40.0, duration_ms=8.0, tbw=6.0, thickness_mm=30.0, tol_in_deg=0.2, max_out_deg=1.0)
CAL_AMPLITUDES = (1.0, 2.0)                                 # the same pulse twice as strong turns twice as far at the slice centre
CAL_SCAN = dict(sequence="rfscan", tr_ms=500.0, te_ms=8.0, matrix=96)        # one echo time for every calibration scan: T2* must cancel
CAL_CIRCLE = dict(id="centre", y_mm=0.0, x_mm=0.0, radius_mm=4.0)
METHOD_SHARE = (0.5, 0.8)                                   # the parts of the two tolerances the stated method builds to
BASE = dict(kind="qa", b0_t=3.0, b1_shape_amp=(0.0, 0.0), b0_hz=2.0, cf_offset_hz=3.0, console_sequences=["rfscan"],
            qa=dict(p_centre=0.0, inserts=[dict(y_mm=0.0, x_mm=0.0, radius_mm=INSERT["radius_mm"],
                                                tissue=dict(pd=0.9, t1_ms=INSERT["t1_ms"], t2_ms=INSERT["t2_ms"], t2p_ms=150.0, adc_um2_ms=2.0))]))
LEVELS = {
    # Level 1 is a practice level: it states the method, and it hides one number in a known range, which leaves the best
    # constant correction a few episodes in a hundred (0, 5 and 1 of 60 for three tuning sets at the development gate)
    1: dict(world_spec=dict(BASE, rf_hardware=dict(tx_scale=TX_RANGE)), flip_deg=(20.0, 40.0), factor=1.5, power="loose", method=True, scans=2,
            kind="practice",
            summary="one hidden quantity, the transmit scale; the method is stated: two calibration scans, the generator and its parameters"),
    2: dict(world_spec=dict(BASE, rf_hardware=dict(tx_scale=TX_RANGE, amp_sat_ut=AMP_RANGE_UT)), flip_deg=(55.0, 90.0), factor=1.35, power="loose", method=False, scans=3,
            summary="the method is withheld and the amplifier compresses: the shortest pulse of the ideal model leaves the mask on the scanner"),
    # 1.25 and not less: at 1.15 the reference, which leaves a margin for its own measurement error, passed 52 of 60
    # (development gate); a windowed sinc needs about 1.8 times the reference
    3: dict(world_spec=dict(BASE, rf_hardware=dict(tx_scale=TX_RANGE, amp_sat_ut=AMP_RANGE_UT)), flip_deg=(45.0, 90.0), factor=1.25, power="binding", method=False, scans=3,
            summary="as level 2 with the duration within 25 % of the reference and an RF power limit that binds: a windowed sinc cannot pass"),
}
SCAN_S = 55.0                                               # one calibration scan (CAL_SCAN): 96 lines of 500 ms, 3 s of adjustments, 4 s of dummy cycles


def _spec(rng, lv) -> dict:
    flip = float(np.round(rng.uniform(*lv["flip_deg"])))
    d = float(rng.choice([4.0, 5.0, 6.0, 8.0]))
    w = float(rng.choice([0.25, 0.3, 0.35, 0.4, 0.5]))
    tol, out = float(rng.choice([0.03, 0.04])), float(rng.choice([0.03, 0.04]))
    return dict(thickness_mm=d, flip_deg=flip, tol_in_deg=round(tol * flip, 1), max_out_deg=round(out * flip, 1), transition=w,
                extent_mm=EXTENT_MM, refocused_min=REFOCUSED_MIN)


@functools.lru_cache(maxsize=1024)
def _instance(seed: str, level: int):
    """(specification, limits, stated method or None) of an instance. A draw for which the public model finds no design
    (none was met on the gate instances) is drawn again from the next attempt of the seed (``family.reseed``)."""
    lv = LEVELS[level]
    for attempt in range(8):
        rng = W.rng_for(F.reseed(seed, attempt), "task")
        spec = _spec(rng, lv)
        limits = dict(b1_peak_ut=20.0, g_max_mt_m=32.0, slew_t_m_s=120.0, b1rms_ut=3.0, tr_ms=TR_LOOSE_MS)
        loose = model.limits_of(dict(design=dict(limits=limits)))
        rho = float(rng.uniform(1.2, 1.7))                  # drawn at every level, so that the levels share their specifications' stream
        if lv["power"] == "binding":
            # the power limit is set from the public model: the shortest pulse of the ideal chain would have to be ``rho``
            # times as long to stay within it (its energy falls as 1 / duration)
            ideal = model.search(spec, loose, hardware.IDEAL)
            if ideal is None:
                continue
            b1rms = math.floor(math.sqrt(ideal["pulse"].energy_ut2_ms / (rho * TR_BINDING_MS)) / 0.05) * 0.05
            limits.update(b1rms_ut=round(max(b1rms, 0.5), 2), tr_ms=TR_BINDING_MS)
        method = None
        if lv["method"]:
            # what level 1 states: the tbw of the stated shares on the ideal model (it does not depend on the transmit scale:
            # the amplitude corrects that), with a tenth to spare, and the number D = tbw x transition behind it
            tbw = model.smallest_tbw(spec, loose, METHOD_SHARE, [hardware.IDEAL])
            if tbw is None:
                continue
            tbw = round(math.ceil((tbw + 0.1) * 10 - 1e-9) / 10, 1)
            method = dict(tbw=tbw, d_number=round(tbw * spec["transition"], 2), share=list(METHOD_SHARE))
        return spec, limits, method
    raise RuntimeError("no instance with a design on the public model in eight draws")


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    spec, limits, method = _instance(seed, level)
    hidden = ["transmit_scale"] + (["amplifier"] if lv["world_spec"]["rf_hardware"].get("amp_sat_ut") else [])
    # the field of this object is flat and its frequency adjustment good (within 5 Hz at the isocentre): told, so that
    # the two quantities a level hides are the ones that decide the design
    told = [f"the transmit scale s: the B1 a spin gets is s times the B1 of the design, one number between {TX_RANGE[0]:g} and {TX_RANGE[1]:g} "
            "(the transmitter is not calibrated)"]
    if lv["world_spec"]["rf_hardware"].get("amp_sat_ut"):
        told.append("the RF amplifier compresses: an amplitude a of the design comes out as a / (1 + (a / A)^6)^(1/6), A between "
                    f"{AMP_TOLD_UT[0]:g} and {AMP_TOLD_UT[1]:g} uT (then times s)")
    task = dict(family=FAMILY, design=dict(specification=dict(spec), limits=dict(limits, duration_max_ms=tool.MAX_DURATION_MS), unknown=told,
                                           hidden=hidden),
                duration_factor=lv["factor"], object=dict(insert_radius_mm=INSERT["radius_mm"], insert_t1_below_ms=T1_TOLD_BELOW_MS, half_length_mm=EXTENT_MM),
                regions=[], deliverable={"design": "<number of a stored design>"})
    if method:
        task["method"] = dict(method, share=list(method["share"]))
    return dict(task_id=task_id or "rfpulse-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=lv["world_spec"],
                budget_s=round(lv["scans"] * SCAN_S + 20.0, 1), max_requests=40, task=task)


# ------------------------------------------------------------------------------------------------ the task text
def _bands(spec: dict) -> tuple:
    d, w = spec["thickness_mm"], spec["transition"]
    return (1 - w) * d / 2, (1 + w) * d / 2


def method_text(task: dict) -> str:
    m = task.get("method")
    if not m:
        return ""
    spec, lim = task["design"]["specification"], task["design"]["limits"]
    cal = ", ".join(f'"{k}": {v:g}' if not isinstance(v, str) else f'"{k}": "{v}"' for k, v in CAL.items())
    t_grad = m["tbw"] * 1e3 / (model.HZ_PER_MT_M_MM * spec["thickness_mm"] * lim["g_max_mt_m"])
    return f"""How to do it (this level states the procedure).
  1. Make two calibration pulses with `design`: {{{cal}}}
     and the same with "amplitude": {CAL_AMPLITUDES[1]:g}. Note the "flip on resonance" of the first one: a.
  2. Scan each: {{"sequence": "rfscan", "rf_design": <number>, "tr_ms": {CAL_SCAN['tr_ms']:g}, "te_ms": {CAL_SCAN['te_ms']:g}, "matrix": {CAL_SCAN['matrix']},
     "circles": [{{"y_mm": 0, "x_mm": 0, "radius_mm": {CAL_CIRCLE['radius_mm']:g}}}]}}. Take |complex_mean| of the circle: S1 and S2.
     (The image is the pulse's profile, a 30 mm stripe through the isocentre; the circle lies in its flat middle,
     in the doped water, where the signal is sin(flip angle) to 0.3 % at this repetition time.)
  3. The second pulse turns twice as far as the first, so S2 / S1 = 2 cos(s a) and the transmit scale is
     s = arccos(S2 / (2 S1)) [in degrees] / a.
  4. Make the pulse to deliver with `design`, kind slr:
       flip_deg = {spec['flip_deg']:g}, thickness_mm = {spec['thickness_mm']:g}
       amplitude = 1 / s                        (the scanner multiplies every B1 by s)
       tol_in_deg = {m['share'][0] * spec['tol_in_deg']:g}, max_out_deg = {m['share'][1] * spec['max_out_deg']:g}      ({m['share'][0]:g} and {m['share'][1]:g} of the two tolerances; the rest is your margin)
       tbw = {m['tbw']:g}                              (an equiripple pulse with these ripples has a transition of at most D / tbw
                                               of the thickness, D = {m['d_number']:g}; it has to be at most {spec['transition']:g})
       duration_ms: the shortest the limits allow. The gradient is tbw / (0.042577 x thickness_mm x duration_ms) mT/m, so
       the gradient limit gives {t_grad:.4f} ms: round UP to 0.01 ms. If the answer of `design` then shows a peak B1 above its
       limit, multiply the duration by peak / limit, round up and make the design again (peak B1 falls as 1 / duration).
  5. Deliver the number of that design."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    spec, lim = t["design"]["specification"], t["design"]["limits"]
    z_in, z_out = _bands(spec)
    scans = LEVELS[cfg["level"]]["scans"]
    unknown = "\n".join(f"  {u}" for u in t["design"]["unknown"])
    method = method_text(t)
    return f"""# Design a slice-selective excitation pulse for this scanner
Deliver one RF pulse design that meets the specification below on THIS scanner. Make designs with `design` (a generator
and its parameters; a stored design gets a number and never changes), look at one on the public model with `simulate`,
play one on the scanner with `scan` (sequence rfscan, rf_design = its number), deliver one number with `submit`.

Specification: the flip angle against the position z along the slice axis, on the scanner, at the isocentre
  slice thickness {spec['thickness_mm']:g} mm
  in the slice, |z| <= {z_in:g} mm: {spec['flip_deg']:g} +- {spec['tol_in_deg']:g} degrees
  outside, |z| from {z_out:g} to {spec['extent_mm']:g} mm (the end of the object): at most {spec['max_out_deg']:g} degrees
  after the rephasing lobe of the design the in-slice signal is at least {100 * spec['refocused_min']:g} % of what the flip angles give
Limits, checked on the waveform of the design: peak B1 {lim['b1_peak_ut']:g} uT, gradient {lim['g_max_mt_m']:g} mT/m, slew rate {lim['slew_t_m_s']:g} T/m/s,
B1rms {lim['b1rms_ut']:g} uT over a repetition time of {lim['tr_ms']:g} ms (one pulse per repetition), duration {lim['duration_max_ms']:g} ms.
Duration: at most {t['duration_factor']:g} times the shortest pulse that a stated search finds for this scanner (Shinnar-Le Roux pulses
that meet the mask on it; scanner_info topic design). A longer pulse fails although it meets the mask.

The simulator is ideal. This scanner is not, and what it does to a design is unknown until measured:
{unknown}
The object: a gel phantom with a cylinder of doped water at the isocentre (radius {t['object']['insert_radius_mm']:g} mm, T1 below {t['object']['insert_t1_below_ms']:g} ms); it
reaches {t['object']['half_length_mm']:g} mm either side along the slice axis. The scan-time budget pays {scans} scans of {SCAN_S:g} s (rfscan, tr_ms {CAL_SCAN['tr_ms']:g},
matrix {CAL_SCAN['matrix']}), not {scans + 1}. Signals of two scans compare only at the same te_ms. Your scans may use any sequence and any parameters the console accepts; only the delivered design is evaluated.

The frequency adjustment of this session is good: off resonance at the isocentre is below 5 Hz.

Passing: on the scanner the design meets the specification and the limits and is short enough. The scanner's log must show
that you measured. An answer in the wrong unit is wrong.""" + ("\n\n" + method if method else "")


def submit_schema(cfg: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["design"],
            "properties": {"design": {"type": "integer", "minimum": 1, "maximum": tool.MAX_DESIGNS, "description": "number of a stored design"}}}


def check(cfg: dict, answer, exam: dict):
    if not isinstance(answer, dict) or set(answer) != {"design"} or isinstance(answer["design"], bool) or not isinstance(answer["design"], int):
        return 'the deliverable is {"design": <number of a stored design>}'
    table = exam.get("designs") or {}
    k = answer["design"]
    if k not in table:
        stored = ", ".join(str(i) for i in sorted(table)) or "none"
        return dict(message=f"no stored design {k} (stored: {stored})", parameter="design", requested=k,
                    **({"nearest_valid": max(table), "fix": {"design": max(table)}} if table else {}))
    lim = table[k]["facts"]["limits"]
    if not lim["ok"]:
        v = lim["violations"][0]
        return dict(message=f"design {k} violates a limit ({tool.LIMIT_WORDS[v['code']]} {v['value']:g} against {v['limit']:g} {v['unit']})",
                    parameter="design", requested=k)
    return None


# ------------------------------------------------------------------------------------------------ verifier side
def chain_of(world: dict):
    """The transmit chain of the hidden scanner at the isocentre: where the specification is evaluated."""
    n = world["n"]
    df = float(world["fields"]["b0_hz"][n // 2, n // 2] + world["system"]["cf_offset_hz"])
    return rfscan.chain_of(world, W.b1_at(world, 0.0, 0.0), df)


_REFERENCE = {}


def reference(cfg: dict, world: dict) -> dict | None:
    """The reference search on the hidden scanner (``model.search``): once per instance and process."""
    key = (cfg["seed"], cfg["level"])
    if key not in _REFERENCE:
        if len(_REFERENCE) >= 4096:
            _REFERENCE.clear()
        got = model.search(model.spec_of(cfg["task"]), model.limits_of(cfg["task"]), chain_of(world))
        _REFERENCE[key] = None if got is None else {k: v for k, v in got.items() if k != "pulse"}
    return _REFERENCE[key]


def record_class(task: dict) -> str:
    """Key of a record table: the public specification and limits of the instance, on their own lattice."""
    s, lim = task["design"]["specification"], task["design"]["limits"]
    return (f"excitation|d{s['thickness_mm']:g}|flip{s['flip_deg']:g}|in{s['tol_in_deg']:g}|out{s['max_out_deg']:g}|w{s['transition']:g}"
            f"|b1{lim['b1_peak_ut']:g}|g{lim['g_max_mt_m']:g}|rms{lim['b1rms_ut']:g}@{lim['tr_ms']:g}")


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem if isinstance(problem, str) else problem["message"], error=None)
    t = cfg["task"]
    spec, limits = model.spec_of(t), model.limits_of(t)
    entry = world["console"]["designs"][answer["design"]]
    pulse, _ = tool.build(entry["call"], limits)
    if pulse.thickness_mm is None or pulse.kind != "excitation":
        return F.result(dict(deliverable_valid=False), "the design is not a slice-selective excitation", error=None)
    facts = tool.facts(pulse, limits)                                    # every limit recomputed from the samples
    chain = chain_of(world)
    fig = tool.figures(pulse, chain, spec)
    ref = reference(cfg, world)
    floor = model.floor_ms(spec, limits, chain.tx_scale)
    duration = round(pulse.duration_ms, 6)
    ref_ms = None if ref is None else ref["duration_ms"]
    violated = [v["code"] for v in facts["limits"]["violations"]]
    checks = dict(deliverable_valid=True, limits=not violated, in_slice=fig["checks"]["in_slice"], out_of_slice=fig["checks"]["out_of_slice"],
                  phase=fig["checks"]["phase"], not_below_floor=duration >= floor * (1 - 1e-9) or not fig["checks"]["in_slice"],
                  short_enough=ref_ms is None or duration <= t["duration_factor"] * ref_ms * (1 + 1e-9))
    why = []
    if violated:
        why.append("limit violated: " + ", ".join(violated))
    if not checks["in_slice"]:
        why.append(f"in the slice the flip angle is {fig['in_min_deg']:.2f} to {fig['in_max_deg']:.2f} degrees")
    if not checks["out_of_slice"]:
        why.append(f"outside the slice the flip angle reaches {fig['out_max_deg']:.2f} degrees at z = {fig['out_max_at_mm']:.2f} mm")
    if not checks["phase"]:
        why.append(f"{100 * fig['refocused']:.1f} % of the in-slice signal is left after the rephasing lobe")
    if not checks["not_below_floor"]:
        why.append(f"{duration:g} ms is below the floor of {floor:.4f} ms: an engine defect, not a design")
    if not checks["short_enough"]:
        why.append(f"{duration:g} ms is {duration / ref_ms:.2f} times the reference of {ref_ms:g} ms")
    return F.result(
        checks, "; ".join(why), error=None if ref_ms is None else duration / ref_ms - 1.0,
        duration_ms=duration, reference_ms=ref_ms, floor_ms=round(floor, 6), duration_factor=t["duration_factor"],
        merit_reference=None if ref_ms is None else round(ref_ms / duration, 4), merit_floor=round(floor / duration, 4),
        reference_design=None if ref is None else dict(call=ref["call"], tbw=ref["tbw"], share=list(ref["share"]), stretch=ref["stretch"]),
        bound=dict(kind="theorem", statement="flip angle <= 2 pi gamma x transmit scale x peak B1 limit x duration (hard-pulse floor)", value_ms=round(floor, 6)),
        record=dict(specification_class=record_class(t), duration_ms=duration, design=entry["call"]),
        profile={k: fig[k] for k in ("centre_flip_deg", "in_min_deg", "in_max_deg", "out_max_deg", "out_max_at_mm", "refocused")},
        design_facts={k: facts[k] for k in ("peak_b1_ut", "energy_ut2_ms", "gradient_mt_m", "slew_t_m_s", "b1rms_ut")},
        distance=dict(in_slice_margin_deg=round(fig["margins"]["in_slice"], 4), out_of_slice_margin_deg=round(fig["margins"]["out_of_slice"], 4),
                      duration_over_reference=None if ref_ms is None else round(duration / ref_ms, 4)))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured design that fails: the first requirement it misses, in the order of
    the pass rule."""
    c = score["checks"]
    for key, label in (("limits", "limit_violated"), ("in_slice", "mask_in_slice"), ("out_of_slice", "mask_out_of_slice"), ("phase", "mask_phase"),
                       ("not_below_floor", "below_floor")):
        if not c.get(key, True):
            return label
    return "too_long"


def task_kind(stored: dict | None = None) -> dict:
    from ... import gate
    return F.kind_by_open_loop(stored or gate.load(ENV_ID))


def card() -> dict:
    from ... import learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = task_kind()
    return dict(
        id=ENV_ID, title="Design a slice-selective excitation pulse for this scanner",
        given="A flip-angle specification for a slice (thickness, angle, tolerances, transition), hardware limits (peak B1, gradient, slew "
              "rate, RF power at a stated repetition time), an ideal simulator, and a scan-time budget of two or three calibration scans",
        deliver="The number of one stored design",
        scored_by="The exact rotation of the delivered samples through the hidden scanner's transmit chain: the mask and the limits hold "
                  "and the pulse is at most 1.5 / 1.35 / 1.25 times as long as the best design of a stated reference search on that scanner",
        summary="The first design task: the agent makes a piece of a sequence. Designs are generator calls or control points, never "
                "samples or code. The simulator the agent is given is ideal; the scanner's transmit scale (and from level 2 its "
                "amplifier's compression) is hidden and has to be measured by playing calibration pulses, or the design misses the "
                "mask. The pass rule has a reference in it: the duration is compared with the best Shinnar-Le Roux design a stated "
                "search finds on the hidden scanner, because no proven bound tighter than the hard-pulse floor is known here. A pass "
                "means the design meets the specification on the modelled scanner, nothing more. " + kind["sentence"],
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom at 3 T with a cylinder of doped water at the isocentre", sequences_useful=["rfscan"],
                   agent_side_formulas="rotation of a piecewise-constant pulse under its gradient (the simulate tool); gradient = tbw / "
                                       "(gamma x thickness x duration); peak B1 and energy fall as 1 / duration; double angle: "
                                       "S2 / S1 = 2 cos(s a); amplifier a / (1 + (a / A)^6)^(1/6)"),
        hidden=["the transmit scale (0.6 to 1.6: the transmitter is not calibrated)", "levels 2 and 3: the saturation level of the RF amplifier (11 to 17 uT)",
                "the error of the frequency adjustment and the residual field at the isocentre", "noise level", "the 128-bit world seed"],
        told=["the specification and the limits", "the duration factor", "that a transmit scale (and an amplifier law) exists, with its range",
              "the object", "level 1: the calibration pulses, the scans, the formula and the parameters of the generator",
              "the instrument manual incl. the topic design"],
        levels={lv: dict(summary=d["summary"], budget_s=[round(d["scans"] * SCAN_S + 20.0, 1)] * 2) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=40, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"design": "<number of a stored design>"}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the delivered number is a stored design that violates no limit"],
                    error="duration / reference duration - 1",
                    score="none paid: the reward is 1 for a pass and 0 otherwise; merit_reference and merit_floor are stored",
                    passing="all gates, the limits, the mask (in slice, outside, phase), duration >= the floor and <= factor x reference"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier computes rotations, it scans nothing",
    )
