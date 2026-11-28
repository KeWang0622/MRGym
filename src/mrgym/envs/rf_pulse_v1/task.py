"""``Scanner/RFPulse-v1``: instances and the verifier. A design task whose pass rule has a proven bound in it.

The job: the shortest slice-selective excitation pulse that meets a flip-angle mask on THIS scanner within its
limits. Designs are generator calls (``design``), looked at on the ideal model (``simulate``), played on the scanner
(``scan``, sequence ``rfscan``) and delivered by number.

The class of designs (stated in the task; ``mrgym.design.bound``): samples on the 0.01 ms raster, played as hard
sub-pulses under one constant slice gradient that the design chooses through ``tbw``. A variable-rate pulse is another
class and is not offered. For this class, and for every transmit chain of this family, the bound is a theorem:
no pulse with fewer than N* samples meets the mask at the positions the verifier looks at, with a gradient within
the gradient limit and nominal samples within the RF power and peak limits. N* depends on the hidden transmit scale
through the two RF limits only.

Hidden hardware (``mrgym.design.hardware.Chain`` with ``model="split"``, ``guard=True``)
  transmit scale   0.6 to 1.6, log-uniform, every level: the flip angle is the nominal one times it
  off-resonance    (levels 2, 3) up to +-400 Hz at the slice: moves the slice by offset / (gamma G)
  gradient delay   (levels 2, 3) 60 to 150 us, early or late: the plateau of the slice gradient outlasts the pulse on
                   both sides, so the gradient during the pulse is the same and the area G x delay behind it is not
                   rewound: a tilt of the phase across the slice
The amplifier compression of ``-v0`` is NOT in this id: its remedy there was a longer pulse, a peak cap is bounded
only through the energy (loose by the square of the crest factor), and no level with it could have a tight factor.

Deliverable    {"design": 3}        the number of one stored design
Pass rule      on the hidden scanner, at the isocentre (the exact rotation of the delivered samples through the
               scanner's chain, on the positions of ``bound.positions``):
                 1. every limit holds, recomputed from the delivered samples: gradient, peak B1, slew rate, RF power
                    (B1rms over the stated repetition time), duration;
                 2. the mask holds: flip angle within the tolerance in the slice; at most the stated angle outside,
                    out to half the repeat distance of the profile; the band between is FREE;
                 3. after the rephasing lobe at least 95 % of the in-slice signal adds up. The bound does not use this
                    requirement (it is on the flip-angle magnitude): the constructive design meets it, so on the
                    instances looked at it does not move the bound (declared, not proven);
                 4. duration <= x_k times the bound N* x raster of THIS scanner. x_k: ``FACTORS``, fixed from the
                    reference's margins on worlds of the split ``tune`` before any gate was run (``factors.py``).
               A design that meets 1 and 2 with fewer samples than N* contradicts the theorem: it fails with the
               outcome ``below_bound`` and is a defect of the bound or of the simulator, never a record.
Never paid     ``info["score"]``: duration, bound, ``duration_over_bound`` (>= 1), energy, the class of the specification.

What a pass means: the design meets this specification on the modelled scanner. Nothing more: no relaxation during
the pulse (no level has a short-T2 material), no eddy currents, one transmit channel, hard sub-pulses, and limits that
are prices of this package, not an assessment of any kind for a person or a real system.
"""
from __future__ import annotations

import functools
import importlib.resources
import json
import math
import os

import numpy as np

from ...design import bound, hardware, pulses, tool
from ...design.units import HZ_PER_MT_M_MM, RASTER_MS
from ...scanner import rfscan
from ...scanner import world as W
from .. import family as F
from . import model

FAMILY = "rf_pulse_v1"
ENV_ID = "Scanner/RFPulse-v1"
SUBMIT_DOC = "Deliver the number of one stored design: the excitation pulse for the specification."
CHAIN = dict(model="split", guard=True)                     # how this family's scanner and simulator play a design
DESIGN_TOOLS = dict(kinds=("sinc", "minphase"), chain=CHAIN)
OUTCOMES = ("limit_violated", "mask_in_slice", "mask_out_of_slice", "mask_phase", "below_bound", "too_long")   # worst first
COMPETENCIES = ("qc.transmit_gain", "parameter.flip_angle")
REFOCUSED_MIN = 0.95
TX_RANGE = (0.6, 1.6)                                       # of the worlds (log-uniform), and told
OFFSET_HZ = 400.0                                           # of the worlds: uniform in +- this (levels 2, 3), and told
DELAY_US = (60.0, 150.0)                                    # of the worlds, either sign (levels 2, 3), and told
INSERT = dict(radius_mm=25.0, t1_ms=83.7, t2_ms=60.0)       # the doped-water cylinder at the isocentre; told: its radius, and
T1_TOLD_BELOW_MS = 100.0                                    # that its T1 is below this
LIMITS = dict(b1_peak_ut=40.0, g_max_mt_m=32.0, slew_t_m_s=120.0, tr_ms=10.0)
B1RMS_LOOSE_UT = 6.0                                        # levels 1, 2: the RF power limit never binds
POWER_SCALE = (1.0, 1.35)                                   # level 3: the transmit scale below which the RF power limit binds, drawn per instance
# Calibration pulses of the stated method and of the control policies: a wide band (8 kHz over 50 mm) with a flat top
# (time-bandwidth 16), so that the read-out circles stay in the flat middle of the stripe although the off-resonance
# moves the stripe (2.5 mm at most) and its image along the read-out
CAL = dict(kind="sinc", flip_deg=30.0, duration_ms=2.0, tbw=16.0, thickness_mm=50.0)
CAL_AMPLITUDES = (1.0, 2.0)
CAL_SCAN = dict(sequence="rfscan", tr_ms=500.0, te_ms=8.0, matrix=96)
CAL_TE2_MS = 8.4                                            # the first scan again at this echo time: the phase runs by 2 pi x offset x 0.4 ms
CAL_CIRCLES = [dict(id="centre", y_mm=0.0, x_mm=0.0, radius_mm=4.0)]
SCAN_S = 55.0                                               # one calibration scan: 96 lines of 500 ms, 3 s of adjustments, 4 s of dummy cycles
PROVISIONAL_FACTOR = 2.0                                    # only while ``factors.py`` has not been run: never a served value
BASE = dict(kind="qa", b0_t=3.0, b1_shape_amp=(0.0, 0.0), b0_hz=2.0, cf_offset_hz=3.0, console_sequences=["rfscan"],
            qa=dict(p_centre=0.0, inserts=[dict(y_mm=0.0, x_mm=0.0, radius_mm=INSERT["radius_mm"],
                                                tissue=dict(pd=0.9, t1_ms=INSERT["t1_ms"], t2_ms=INSERT["t2_ms"], t2p_ms=150.0, adc_um2_ms=2.0))]))
_QUIET = dict(tx_scale=TX_RANGE, **CHAIN)
_FULL = dict(_QUIET, grad_delay_us=DELAY_US, grad_delay_sign=True)
LEVELS = {
    # Level 1 states the method and hides one number in a known range: a practice level, as in ``-v0``
    1: dict(world_spec=dict(BASE, rf_hardware=_QUIET), power="loose", method=True, scans=2, kind="practice", transitions=(0.2, 0.25, 0.3, 0.4),
            hidden=("transmit_scale",),
            summary="one hidden quantity, the transmit scale; the method is stated: two calibration scans, the generator and its parameters"),
    2: dict(world_spec=dict(BASE, cf_offset_hz=OFFSET_HZ, rf_hardware=_FULL), power="loose", method=False, scans=3, transitions=(0.2, 0.25, 0.3, 0.4),
            hidden=("transmit_scale", "off_resonance", "gradient_delay"),
            summary="the method is withheld; transmit scale, off-resonance of the slice and gradient delay are hidden; the gradient limit decides the duration"),
    # transitions up to 0.3: the energy argument of the bound leaves out most of the transition band, and with a wider one
    # the reference stands further above the bound where the RF power limit decides (1.75 to 1.88 times at a transition of
    # 0.4 on development worlds, before the factors were derived)
    3: dict(world_spec=dict(BASE, cf_offset_hz=OFFSET_HZ, rf_hardware=_FULL), power="binding", method=False, scans=3, transitions=(0.2, 0.25, 0.3),
            hidden=("transmit_scale", "off_resonance", "gradient_delay"),
            summary="as level 2 with an RF power limit that binds on scanners with a low transmit scale: which limit decides the gradient "
                    "is known only after the scale is measured"),
}


def _factors() -> dict:
    try:
        text = importlib.resources.files("mrgym.envs").joinpath("results", "Scanner-RFPulse-v1.factors.json").read_text(encoding="utf-8")
        return {int(k): float(v["factor"]) for k, v in json.loads(text)["levels"].items()}
    except (FileNotFoundError, OSError):
        return {}


FACTORS = {lv: _factors().get(lv, PROVISIONAL_FACTOR) for lv in LEVELS}


def _spec(rng, widths) -> dict:
    flip = float(rng.integers(20, 41))
    d = float(rng.choice([4.0, 5.0, 6.0]))
    w = float(rng.choice(widths))
    tol, out = float(rng.choice([0.04, 0.05])), float(rng.choice([0.03, 0.04]))
    return dict(thickness_mm=d, flip_deg=flip, tol_in_deg=round(tol * flip, 1), max_out_deg=round(out * flip, 1), transition=w,
                extent_mm="repeat", refocused_min=REFOCUSED_MIN)


@functools.lru_cache(maxsize=1024)
def _instance(seed: str, level: int):
    """(specification, limits, stated method or None) of an instance."""
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    spec = _spec(rng, lv["transitions"])
    rho = float(rng.uniform(*POWER_SCALE))                  # drawn at every level, so that the levels share their specifications' stream
    limits = dict(LIMITS, b1rms_ut=B1RMS_LOOSE_UT)
    loose = model.limits_of(dict(design=dict(limits=limits)))
    samples = model.shortest(spec, limits["g_max_mt_m"])
    if lv["power"] == "binding":
        # the shortest pulse of the public model at the gradient limit, played 1 / rho times as strong, is at the limit
        ideal, _ = tool.build(model.call_of(spec, samples, limits["g_max_mt_m"]), loose)
        limits["b1rms_ut"] = round(math.floor(math.sqrt(ideal.energy_ut2_ms / (rho * rho * limits["tr_ms"])) / 0.01) * 0.01, 2)
    method = None
    if lv["method"]:
        method = {k: v for k, v in model.call_of(spec, samples, limits["g_max_mt_m"]).items() if k in ("duration_ms", "tbw", "tol_in_deg", "max_out_deg")}
    return spec, limits, method


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    spec, limits, method = _instance(seed, level)
    task = dict(family=FAMILY, design=dict(specification=dict(spec), limits=dict(limits, duration_max_ms=tool.MAX_DURATION_MS), hidden=list(lv["hidden"])),
                duration_factor=FACTORS[level], raster_ms=RASTER_MS,
                object=dict(insert_radius_mm=INSERT["radius_mm"], insert_t1_below_ms=T1_TOLD_BELOW_MS), regions=[],
                deliverable={"design": "<number of a stored design>"})
    if method:
        task["method"] = method
    return dict(task_id=task_id or "rfpulse1-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=lv["world_spec"],
                budget_s=round(lv["scans"] * SCAN_S + 20.0, 1), max_requests=40, task=task)


# ------------------------------------------------------------------------------------------------ the task text
def _json(d: dict) -> str:
    return json.dumps(d, separators=(", ", ": "))


def method_text(task: dict) -> str:
    m = task.get("method")
    if not m:
        return ""
    spec = task["design"]["specification"]
    scan = dict(CAL_SCAN, rf_design="<number>", circles=[{k: v for k, v in CAL_CIRCLES[0].items() if k != "id"}])
    call = dict(kind="minphase", flip_deg=spec["flip_deg"], thickness_mm=spec["thickness_mm"], transition=spec["transition"], **m)
    return f"""How to do it (this level states the procedure).
  1. `design` {_json(CAL)} and the same with "amplitude": {CAL_AMPLITUDES[1]:g}. Note "flip on resonance" of the first: a.
  2. `scan` each: {_json(scan)}. |complex mean| of the circle: S1, S2.
  3. The second pulse turns twice as far: S2 / S1 = 2 cos(s a), so s = arccos(S2 / (2 S1)) [degrees] / a.
  4. `design` {_json(call)[:-1]}, "amplitude": 1 / s}}
  5. `submit` its number."""


HIDDEN_TEXT = dict(
    transmit_scale=f"transmit scale s, {TX_RANGE[0]:g} to {TX_RANGE[1]:g}: a spin gets s times the B1 of the design",
    off_resonance=f"off-resonance of the slice, up to +-{OFFSET_HZ:g} Hz: it moves the slice by offset / ({HZ_PER_MT_M_MM:.3f} x G) mm",
    gradient_delay=f"the slice gradient is early or late against the RF by {DELAY_US[0]:g} to {DELAY_US[1]:g} us: its plateau outlasts the pulse, "
                   "so the area G x delay stays behind the pulse and tilts the phase across the slice")


def statement(cfg: dict) -> str:
    t = cfg["task"]
    spec, lim = t["design"]["specification"], t["design"]["limits"]
    m = bound.Mask.of(spec)
    lv = LEVELS[cfg["level"]]
    unknown = "\n".join(f"  {HIDDEN_TEXT[h]}" for h in t["design"]["hidden"])
    power = (" On a scanner with a low s the shortest pulse passes the B1rms limit when played 1 / s times as strong; the energy "
             "is proportional to the gradient.") if lv["power"] == "binding" else ""
    method = method_text(t)
    return f"""# Design the shortest slice-selective excitation pulse for this scanner
Make designs with `design`, look at one on the ideal model with `simulate`, play one with `scan` (sequence rfscan,
rf_design = its number), deliver one number with `submit`.

Specification: flip angle against the position z along the slice axis, on THIS scanner, at the isocentre
  in the slice, |z| <= {m.z_in:g} mm: {spec['flip_deg']:g} +- {spec['tol_in_deg']:g} degrees
  outside, |z| >= {m.z_out:g} mm, out to half the repeat distance of the profile: at most {spec['max_out_deg']:g} degrees
  between the two: free
  after the rephasing lobe at least {100 * spec['refocused_min']:g} % of the in-slice signal adds up
Class of pulses: samples of {t['raster_ms']:g} ms played as hard sub-pulses under one constant slice gradient
G = tbw / ({HZ_PER_MT_M_MM * 1e-3:.6f} x thickness_mm x duration_ms) mT/m; the profile repeats every {1e3 / (HZ_PER_MT_M_MM * t['raster_ms']):.1f} / G mm.
Limits on the samples of the design: gradient {lim['g_max_mt_m']:g} mT/m, peak B1 {lim['b1_peak_ut']:g} uT, B1rms {lim['b1rms_ut']:g} uT over a repetition
time of {lim['tr_ms']:g} ms, duration {lim['duration_max_ms']:g} ms.
Duration: at most {t['duration_factor']:g} times the bound, the shortest duration any pulse of this class can have on this scanner
within the limits (proven; scanner_info topic design). A longer pulse fails although it meets the mask.

This scanner is not ideal. Unknown until measured:
{unknown}{power}
The object: a gel phantom, doped water at the isocentre (radius {t['object']['insert_radius_mm']:g} mm, T1 below {t['object']['insert_t1_below_ms']:g} ms). The scan-time budget pays {lv['scans']} scans of {SCAN_S:g} s (rfscan, tr_ms {CAL_SCAN['tr_ms']:g}, matrix {CAL_SCAN['matrix']}), not {lv['scans'] + 1}.
With rf_axis readout the image shows the pulse's profile along x; magnitudes of two scans compare only at the same te_ms.
Your scans may use any sequence and any parameters the console accepts; only the delivered design is evaluated.

Passing: on the scanner the design meets the specification and the limits and is short enough. The scanner's log
must show that you measured. An answer in the wrong unit is wrong.""" + ("\n\n" + method if method else "")


DESIGN_MANUAL = f"""Designed pulses on this console (Scanner/RFPulse-v1)

A design is a generator call; `design` stores it under a number and prints its record, `simulate` its flip-angle
profile on the model, `scan` with sequence rfscan plays it (rf_axis readout: the image is the profile along x).

How a design is played. Every sample of {RASTER_MS:g} ms is a hard sub-pulse; the slice gradient is constant, G = tbw /
({HZ_PER_MT_M_MM * 1e-3:.6f} x thickness_mm x duration_ms) mT/m, and its plateau begins before the pulse and ends after it.
The profile of a sampled pulse repeats every {1e3 / (HZ_PER_MT_M_MM * RASTER_MS):.1f} / G mm. After the pulse a rephasing lobe of
rephase_fraction x G x duration is played.

What a scanner does to a design, and the correction
  transmit scale s        every B1 is multiplied by s                              amplitude = 1 / s
  off-resonance f (Hz)    the slice moves by f / ({HZ_PER_MT_M_MM:.3f} x G) mm                     freq_hz = f
  gradient late by t      the area G x t stays behind the pulse: the phase of the  rephase_fraction = best + t / duration
                          signal tilts by 2 pi x {HZ_PER_MT_M_MM:.3f} x G x t[s] rad per mm          (t negative: early)

Generators
  sinc      windowed sinc; flip_deg at the slice centre
  minphase  for a mask (thickness, transition W, tol_in_deg, max_out_deg): the best pulse of its duration and
            gradient; when a pulse of that length fits the mask, this one does. The record prints its largest
            deviation in units of the two tolerances (at most 1: met on the ideal model) and the best
            rephase_fraction. Shorter duration or weaker gradient: a larger deviation. Its energy and peak B1 are
            proportional to the gradient.

The duration bound. For samples of {RASTER_MS:g} ms under one constant gradient the flip angle is a function of
|B|^2, a trigonometric polynomial of degree (samples - 1) in the precession per sample. Whether such a polynomial
fits a mask is decided exactly; the bound is the smallest number of samples for which one fits at some gradient
that the limits allow. The gradient limit is stated. The RF power and the peak limit are on the design's samples, which
are 1 / s times what the spins need, so they cap the gradient at a value that grows with s^2 (energy is proportional
to the gradient). The bound is therefore N(largest allowed gradient) x {RASTER_MS:g} ms: it is the duration of minphase at
the gradient limit when the RF limits leave that gradient, and longer when they do not.

Measuring. At tr_ms of five T1 the signal of a spin is sin(flip angle). The same pulse twice as strong turns twice as far.
The phase of a circle's complex mean runs by 2 pi x off-resonance x te_ms / 1000 between two echo times. Across the stripe of a
pulse that is symmetric in time the phase is flat on an ideal scanner."""


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


def bound_of(task: dict, tx_scale: float) -> dict:
    """The duration bound of an instance on a scanner with the transmit scale ``tx_scale`` (``bound.duration_bound``)."""
    lim = model.limits_of(task)
    return bound.duration_bound(model.spec_of(task), lim.g_max_mt_m, tx_scale, lim.b1_peak_ut, model.energy_limit(lim))


def profile(pulse, chain, spec: dict, cells=()) -> dict:
    """Figures of the flip-angle profile of ``pulse`` through ``chain`` on the positions of the pass rule."""
    u = bound.precession(float(pulse.g_mt_m[0]), float(pulse.dt_ms[0]))
    z_in, z_out = bound.positions(spec, pulse.n, u, cells)
    a, b = hardware.rotate(pulse, chain, np.concatenate([z_in, z_out]))
    flip = np.degrees(2 * np.arcsin(np.clip(np.abs(b), 0, 1)))
    f_in, f_out = flip[:z_in.size], flip[z_in.size:]
    worst = int(np.argmax(f_out))
    lo, hi = spec["flip_deg"] - spec["tol_in_deg"], spec["flip_deg"] + spec["tol_in_deg"]
    out = dict(in_min_deg=float(f_in.min()), in_max_deg=float(f_in.max()), out_max_deg=float(f_out[worst]), out_max_at_mm=float(z_out[worst]),
               refocused=pulses.refocused((2 * np.conj(a) * b)[:z_in.size], z_in), positions=int(flip.size))
    out["checks"] = dict(in_slice=bool(out["in_min_deg"] >= lo and out["in_max_deg"] <= hi), out_of_slice=bool(out["out_max_deg"] <= spec["max_out_deg"]),
                         phase=bool(out["refocused"] >= spec["refocused_min"]))
    out["margins"] = dict(in_slice=float(min(out["in_min_deg"] - lo, hi - out["in_max_deg"])), out_of_slice=float(spec["max_out_deg"] - out["out_max_deg"]))
    return out


def in_class(pulse) -> bool:
    """Whether a pulse is of the class of the bound: a slice-selective excitation on the raster under one constant gradient."""
    return bool(pulse.kind == "excitation" and pulse.thickness_mm is not None and np.ptp(pulse.g_mt_m) == 0 and pulse.g_mt_m[0] > 0
                and np.ptp(pulse.dt_ms) < 1e-12 and abs(float(pulse.dt_ms[0]) - RASTER_MS) < 1e-9)


def record_class(task: dict, level: int) -> str:
    """Key of the record table: the level (which limits bind) and the geometry of the mask."""
    s = task["design"]["specification"]
    return f"level {level} | thickness {s['thickness_mm']:g} mm | transition {s['transition']:g}"


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem if isinstance(problem, str) else problem["message"], error=None)
    t = cfg["task"]
    spec, limits = model.spec_of(t), model.limits_of(t)
    entry = world["console"]["designs"][answer["design"]]
    pulse, _ = tool.build(entry["call"], limits)
    if not in_class(pulse):
        return F.result(dict(deliverable_valid=False), "the design is not a slice-selective excitation under one constant gradient", error=None)
    facts = tool.facts(pulse, limits)                                    # every limit recomputed from the samples
    chain = chain_of(world)
    b = bound_of(t, chain.tx_scale)
    fig = profile(pulse, chain, spec, b["cells"] if pulse.n < b["samples"] else ())
    duration, bound_ms = round(pulse.duration_ms, 6), b["duration_ms"]
    violated = [v["code"] for v in facts["limits"]["violations"]]
    valid = not violated and fig["checks"]["in_slice"] and fig["checks"]["out_of_slice"]
    checks = dict(deliverable_valid=True, limits=not violated, in_slice=fig["checks"]["in_slice"], out_of_slice=fig["checks"]["out_of_slice"],
                  phase=fig["checks"]["phase"], not_below_bound=not (valid and pulse.n < b["samples"]),
                  short_enough=duration <= t["duration_factor"] * bound_ms * (1 + 1e-9))
    why = []
    if violated:
        why.append("limit violated: " + ", ".join(violated))
    if not checks["in_slice"]:
        why.append(f"in the slice the flip angle is {fig['in_min_deg']:.2f} to {fig['in_max_deg']:.2f} degrees")
    if not checks["out_of_slice"]:
        why.append(f"outside the slice the flip angle reaches {fig['out_max_deg']:.2f} degrees at z = {fig['out_max_at_mm']:.2f} mm")
    if not checks["phase"]:
        why.append(f"{100 * fig['refocused']:.1f} % of the in-slice signal is left after the rephasing lobe")
    if not checks["not_below_bound"]:
        why.append(f"{pulse.n} samples meet the mask and the limits where {b['samples']} are proven necessary: a defect of the bound or of the simulator")
    if not checks["short_enough"]:
        why.append(f"{duration:g} ms is {duration / bound_ms:.2f} times the bound of {bound_ms:g} ms")
    return F.result(
        checks, "; ".join(why), error=duration / bound_ms - 1.0,
        duration_ms=duration, bound_ms=bound_ms, duration_over_bound=round(duration / bound_ms, 4), duration_factor=t["duration_factor"],
        bound=dict(kind="theorem", statement="no pulse of the class with fewer samples meets the mask within the limits on this scanner "
                                             "(mrgym.design.bound; docs/physics/design.md, section 4a)",
                   samples=b["samples"], limited_by=b["limited_by"], gradient_cap_mt_m=round(min(b["gradient_cap_mt_m"], limits.g_max_mt_m), 4),
                   certificates=len(b["cells"]), tight=b["samples"] == b["found_samples"]),
        quarantine=None if checks["not_below_bound"] else "beats_bound",
        record=dict(specification_class=record_class(t, cfg["level"]), duration_over_bound=round(duration / bound_ms, 4), design=entry["call"]),
        profile={k: fig[k] for k in ("in_min_deg", "in_max_deg", "out_max_deg", "out_max_at_mm", "refocused", "positions")},
        design_facts={k: facts[k] for k in ("peak_b1_ut", "energy_ut2_ms", "gradient_mt_m", "slew_t_m_s", "b1rms_ut")},
        energy_over_limit=round(pulse.energy_ut2_ms / model.energy_limit(limits), 4),
        distance=dict(in_slice_margin_deg=round(fig["margins"]["in_slice"], 4), out_of_slice_margin_deg=round(fig["margins"]["out_of_slice"], 4),
                      duration_over_bound=round(duration / bound_ms, 4)))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured design that fails: the first requirement it misses, in the order of
    the pass rule."""
    c = score["checks"]
    for key, label in (("limits", "limit_violated"), ("in_slice", "mask_in_slice"), ("out_of_slice", "mask_out_of_slice"), ("phase", "mask_phase"),
                       ("not_below_bound", "below_bound")):
        if not c.get(key, True):
            return label
    return "too_long"


def task_kind(stored: dict | None = None) -> dict:
    """Measurement or decision, per level, by the gate's 25-point rule on the control ``open_loop``, with what that
    control is in this family said next to it: the scan list of the reference is fixed, so the rule is about the choice
    of the REMEDY from the measurement, not about the choice of the next scan."""
    from ... import gate
    kind = F.kind_by_open_loop(stored or gate.load(ENV_ID))
    if not kind["levels"]:
        return kind
    levels = ", ".join(f"level {lv} is a {k} task" for lv, k in sorted(kind["levels"].items()))
    return dict(kind, sentence=f"By level: {levels}, in this sense: the control open_loop keeps the reference's scan list, which is fixed, puts what "
                               "it measured into the three corrections by their formulas, and takes its gradient from the task card alone; 25 points or "
                               f"more below the reference make a decision task ({kind['evidence']}). The decision is which limit sets the gradient on "
                               "this scanner. No level asks for the next scan to be chosen from the last result: the reference itself plays a fixed "
                               "list of scans.")


def card() -> dict:
    from ... import learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = task_kind()
    factors = " / ".join(f"{FACTORS[lv]:g}" for lv in LEVELS)
    return dict(
        id=ENV_ID, title="Design the shortest slice-selective excitation pulse for this scanner",
        given="A flip-angle mask for a slice (thickness, angle, tolerances, a free transition band), the class of pulses (0.01 ms samples, "
              "one constant gradient), hardware limits (gradient, peak B1, RF power at a stated repetition time), an ideal simulator, and "
              "a scan-time budget of two or three calibration scans",
        deliver="The number of one stored design",
        scored_by="The exact rotation of the delivered samples through the hidden scanner's transmit chain: the limits and the mask hold, "
                  f"the in-slice signal refocuses, and the pulse is at most {factors} times (levels 1 / 2 / 3) as long as a proven lower bound "
                  "on the duration of any pulse of the class on that scanner",
        summary="A design task whose pass rule has no reference in it. Designs are generator calls, never samples or code. The scanner's "
                "transmit scale, the off-resonance of the slice and the delay of the slice gradient are hidden and have to be measured with "
                "calibration pulses, or the design misses the mask. The duration is compared with a bound that is a theorem for the stated "
                "class of pulses (mrgym.design.bound): certificates show that no shorter pulse meets the mask at any allowed gradient, and "
                "the generator minphase reaches the bound on an ideal scanner where the gradient limit decides; where the RF power "
                "limit decides, the bound is proven but not tight, and the factor of that level is wider. A pass means the design meets this specification on the modelled scanner, "
                "nothing more. " + kind["sentence"],
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom at 3 T with a cylinder of doped water at the isocentre", sequences_useful=["rfscan"],
                   agent_side_formulas="G = tbw / (gamma x thickness x duration); slice shift = offset / (gamma G); phase tilt = 2 pi gamma G delay "
                                       "per mm; double angle: S2 / S1 = 2 cos(s a); energy and peak B1 of a pulse are proportional to G"),
        hidden=[f"the transmit scale ({TX_RANGE[0]:g} to {TX_RANGE[1]:g})", f"levels 2 and 3: the off-resonance of the slice (up to {OFFSET_HZ:g} Hz)",
                f"levels 2 and 3: the delay of the slice gradient ({DELAY_US[0]:g} to {DELAY_US[1]:g} us, either sign)", "noise level", "the 128-bit world seed"],
        told=["the specification, the class of pulses and the limits", "the duration factor and that the bound exists (its value on this scanner "
              "depends on the hidden transmit scale through the RF limits)", "which quantities are hidden, with their ranges", "the object",
              "level 1: the calibration pulses, the scans, the formula and the generator call", "the manual of designed pulses (scanner_info topic design)"],
        levels={lv: dict(summary=d["summary"], budget_s=[round(d["scans"] * SCAN_S + 20.0, 1)] * 2) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=40, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"design": "<number of a stored design>"}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the delivered number is a stored design of the class that violates no limit"],
                    error="duration / bound - 1",
                    score="none paid: the reward is 1 for a pass and 0 otherwise; duration_over_bound and the energy are stored",
                    passing="all gates, the limits, the mask (in slice, outside), the refocused signal, duration >= the bound and <= factor x bound"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier computes rotations, it scans nothing",
    )
