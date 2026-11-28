"""Designs as an agent makes them: a generator call with a handful of numbers, never samples and never code.

``build(call, limits)``     a call ``{"kind": "sinc" | "slr" | "minphase" | "sech" | "points" | "verse", ...}`` -> ``pulses.Pulse``.
                            Every parameter has a unit and a range (``PARAMETERS``); a value outside it is refused
                            with the parameter and the nearest valid value (``Refusal``), as the console refuses a
                            protocol. Nothing is clipped silently.
``canonical(call)``         the call with its defaults filled in and every number a float: the form that is logged.
                            The same canonical call always builds the same pulse, so a log that holds the call
                            holds the design.
``facts(pulse, limits)``    duration, peak B1, energy, B1rms over the stated repetition time, gradient, slew, and
                            the limit check recomputed from the samples, with the violated limits in a fixed order
                            (``LIMIT_ORDER``).
``summary(index, ...)``     what an agent reads about a stored design.
``simulate(pulse, chain)``  the flip-angle profile of a design on the model the agent is told about: the rotation of
                            ``hardware.rotate`` with the transmit chain the caller assumes (the ideal one by default),
                            as figures of merit, a verdict against a specification when there is one, and one row of
                            numbers. Never the samples.

A slice specification (``spec``) is a dict: ``thickness_mm``, ``flip_deg``, ``tol_in_deg`` (the flip angle in the
slice is within flip_deg +- tol_in_deg), ``max_out_deg`` (at most that outside), ``transition`` (fraction W of the
thickness: the slice is |z| <= (1 - W) thickness / 2, outside is |z| >= (1 + W) thickness / 2), ``extent_mm`` (how far
outside is looked at, on either side) and optionally ``refocused_min`` (the part of the in-slice signal that must
survive the phase left after the rephasing lobe). ``extent_mm`` may be the word ``"repeat"``: outside then reaches to
half the repeat distance of the pulse's profile, 1 / (2 gamma G raster), the mask of ``mrgym.design.bound``.
"""
from __future__ import annotations

import dataclasses
import json
import math

import numpy as np

from . import hardware, pulses
from .units import HZ_PER_MT_M_MM, RAD_PER_MT_M_MM_MS, RASTER_MS

KINDS = ("sinc", "slr", "minphase", "sech", "points", "verse")
MAX_DESIGNS = 32                                     # stored designs per episode
MAX_DURATION_MS = 20.0
PROFILE_BINS, PROFILE_STEP = 24, 0.125               # the printed row: flip angle every 0.125 thickness from the centre outwards
POINTS_PER_THICKNESS = 100                           # density of the positions a profile is evaluated on
DEFAULT_EXTENT = 4.0                                 # thicknesses looked at on either side when no specification says how far
LIMIT_ORDER = ("DURATION_LIMIT", "B1_PEAK_LIMIT", "GRADIENT_LIMIT", "SLEW_LIMIT", "RF_POWER_LIMIT")
# name: (low, high, unit, what it is). Lists and choices are described in ``SCHEMA_EXTRA``.
PARAMETERS = dict(
    flip_deg=(1.0, 179.0, "deg", "flip angle in the slice: at the centre (sinc), the middle of the ripple (slr)"),
    duration_ms=(0.1, MAX_DURATION_MS, "ms", f"a multiple of the raster {RASTER_MS:g} ms"),
    tbw=(1.0, 24.0, "", "time-bandwidth product: bandwidth = tbw / duration"),
    thickness_mm=(1.0, 50.0, "mm", "slice thickness: the gradient is bandwidth / (gamma x thickness)"),
    tol_in_deg=(0.05, 30.0, "deg", "slr: the flip angle in the slice stays within flip_deg +- this"),
    max_out_deg=(0.05, 30.0, "deg", "slr: the flip angle outside the slice stays below this"),
    b1_peak_ut=(0.5, 40.0, "uT", "sech: peak amplitude; verse: the amplitude no sample may pass"),
    beta_rad_per_ms=(0.1, 20.0, "rad/ms", "sech: rate of the modulation"),
    mu=(0.0, 20.0, "", "sech: the frequency sweep is +- mu beta"),
    bandwidth_hz=(100.0, 50000.0, "Hz", "points: the frequency span the gradient spreads over the thickness"),
    rephase_fraction=(0.0, 1.0, "", "points: area of the rephasing lobe / selection area"),
    amplitude=(0.2, 5.0, "", "factor on every sample of the pulse; default 1. What corrects a transmit scale s: 1 / s"),
    g_max_mt_m=(1.0, 80.0, "mT/m", "verse: the gradient no sample may pass"),
    slew_t_m_s=(1.0, 400.0, "T/m/s", "verse: the slew rate between neighbouring samples"),
    transition=(0.05, 0.95, "", "minphase: W; the slice is |z| <= (1 - W) thickness / 2, outside starts at (1 + W) thickness / 2"),
    freq_hz=(-5000.0, 5000.0, "Hz", "minphase: frequency offset of the pulse; the slice lies where spins precess this much faster"),
)
# A range that differs for one generator: (kind, name) -> (low, high). The rephasing lobe of a minimum-phase pulse is
# short, and a slice gradient that is EARLY asks for less of it than nothing: a lobe of the gradient's own sign.
KIND_RANGES = {("minphase", "rephase_fraction"): (-1.0, 1.0)}
WINDOWS = pulses.WINDOWS
INTERPS = ("cubic", "linear")
POINT_RANGE_UT = 40.0
REQUIRED = dict(sinc=("flip_deg", "duration_ms", "tbw", "thickness_mm"),
                slr=("flip_deg", "duration_ms", "tbw", "thickness_mm", "tol_in_deg", "max_out_deg"),
                minphase=("flip_deg", "duration_ms", "tbw", "thickness_mm", "transition", "tol_in_deg", "max_out_deg"),
                sech=("b1_peak_ut", "duration_ms", "beta_rad_per_ms", "mu"),
                points=("amplitude_ut", "duration_ms", "bandwidth_hz", "thickness_mm"),
                verse=("of", "b1_peak_ut"))
OPTIONAL = dict(sinc=("window", "amplitude"), slr=("amplitude",), minphase=("amplitude", "freq_hz", "rephase_fraction"), sech=("thickness_mm",), points=("phase_deg", "interp", "rephase_fraction"),
                verse=("g_max_mt_m", "slew_t_m_s"))


class Refusal(ValueError):
    """A design call that cannot be built: ``message``, the ``parameter`` and what would be valid."""

    def __init__(self, message, parameter=None, **extra):
        super().__init__(message)
        self.message, self.parameter, self.extra = message, parameter, extra


@dataclasses.dataclass(frozen=True)
class Limits:
    """What a design is checked against. ``b1rms_ut`` over ``tr_ms`` with one pulse per repetition (None: not checked)."""
    b1_peak_ut: float = 20.0
    g_max_mt_m: float = 32.0
    slew_t_m_s: float = 120.0
    b1rms_ut: float | None = None
    tr_ms: float | None = None
    duration_max_ms: float = MAX_DURATION_MS

    def public(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


# ---------------------------------------------------------------------------------------------- the call
def range_of(kind, name) -> tuple:
    """(low, high) of a numeric parameter for a generator."""
    return KIND_RANGES.get((kind, name), PARAMETERS[name][:2])


def _number(call, name, kind=None):
    v = call[name]
    unit = PARAMETERS[name][2]
    lo, hi = range_of(kind, name)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Refusal(f"{name} must be a number" + (f" [{unit}]" if unit else ""), name, low=lo, high=hi)
    if not lo <= v <= hi:
        raise Refusal(f"{name} = {v:g} is outside {lo:g} to {hi:g}" + (f" {unit}" if unit else ""), name, requested=v, low=lo, high=hi,
                      nearest_valid=lo if v < lo else hi)
    return float(v)


def _numbers(call, name, bound, count=None):
    v = call[name]
    if not isinstance(v, list) or not 1 <= len(v) <= pulses.MAX_CONTROL_POINTS \
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v):
        raise Refusal(f"{name} must be a list of 1 to {pulses.MAX_CONTROL_POINTS} numbers", name)
    if count is not None and len(v) != count:
        raise Refusal(f"{name} has {len(v)} values and amplitude_ut {count}: one phase per control point", name)
    bad = next((x for x in v if abs(x) > bound), None)
    if bad is not None:
        raise Refusal(f"{name} holds {bad:g}; every value must lie within +-{bound:g}", name, requested=bad, low=-bound, high=bound,
                      nearest_valid=math.copysign(bound, bad))
    return [float(x) for x in v]


def canonical(call, lookup=None) -> dict:
    """The call with defaults filled in, checked against ``PARAMETERS``. ``lookup(index)`` returns the canonical call
    of a stored design (for ``verse``, whose ``of`` is replaced by that call: the result needs no store)."""
    if not isinstance(call, dict):
        raise Refusal("a design is a JSON object {\"kind\": ..., ...}", "kind")
    kind = call.get("kind")
    if kind not in KINDS:
        raise Refusal(f"kind must be one of {', '.join(KINDS)}", "kind", allowed=list(KINDS))
    known = ("kind",) + REQUIRED[kind] + OPTIONAL[kind]
    extra = sorted(str(k)[:40] for k in call if k not in known)
    if extra:
        raise Refusal(f"{kind} has no parameter {extra[0]}; it takes {', '.join(known[1:])}", extra[0], allowed=list(known[1:]))
    missing = [k for k in REQUIRED[kind] if call.get(k) is None]
    if missing:
        lo, hi, unit, what = PARAMETERS.get(missing[0], (None, None, "", ""))
        raise Refusal(f"{kind} needs {missing[0]}" + (f" [{unit}]" if unit else "") + (f": {what}" if what else ""), missing[0])
    out = dict(kind=kind)
    for name in REQUIRED[kind] + OPTIONAL[kind]:
        if call.get(name) is None:
            continue
        if name in PARAMETERS:
            out[name] = _number(call, name, kind)
    if "duration_ms" in out:
        n = out["duration_ms"] / RASTER_MS
        if abs(n - round(n)) > 1e-6:
            near = round(round(n) * RASTER_MS, 6)
            raise Refusal(f"duration_ms = {out['duration_ms']:g} is not a multiple of the raster {RASTER_MS:g} ms", "duration_ms",
                          requested=out["duration_ms"], nearest_valid=near)
        out["duration_ms"] = round(round(n) * RASTER_MS, 6)
    if kind in ("sinc", "slr", "minphase"):
        out.setdefault("amplitude", 1.0)
    if kind == "minphase":
        out.setdefault("freq_hz", 0.0)
    if kind == "sinc":
        out["window"] = call.get("window") or "hann"
        if out["window"] not in WINDOWS:
            raise Refusal(f"window must be one of {', '.join(WINDOWS)}", "window", allowed=list(WINDOWS))
    elif kind in ("slr", "minphase"):
        if out["tol_in_deg"] >= out["flip_deg"] or out["flip_deg"] + out["tol_in_deg"] > 180.0:
            raise Refusal("tol_in_deg must be smaller than flip_deg and flip_deg + tol_in_deg at most 180", "tol_in_deg",
                          requested=out["tol_in_deg"], nearest_valid=round(min(0.5 * out["flip_deg"], 180.0 - out["flip_deg"]), 3))
    elif kind == "points":
        out["amplitude_ut"] = _numbers(call, "amplitude_ut", POINT_RANGE_UT)
        if call.get("phase_deg") is not None:
            out["phase_deg"] = _numbers(call, "phase_deg", 360.0, len(out["amplitude_ut"]))
        out["interp"] = call.get("interp") or "cubic"
        if out["interp"] not in INTERPS:
            raise Refusal(f"interp must be one of {', '.join(INTERPS)}", "interp", allowed=list(INTERPS))
        out.setdefault("rephase_fraction", 0.5)
    elif kind == "verse":
        of = call["of"]
        if isinstance(of, dict):                     # a canonical call carries the base design itself
            out["of"] = canonical(of)
        else:
            base = lookup(of) if lookup is not None and isinstance(of, int) and not isinstance(of, bool) else None
            if base is None:
                raise Refusal(f"of = {str(of)[:20]} is not a stored design", "of")
            out["of"] = base
        if out["of"]["kind"] in ("verse", "sech"):
            raise Refusal(f"verse rescales a pulse under a constant gradient, not a {out['of']['kind']} design", "of")
    return out


_BUILT = {}                                          # canonical call (JSON) -> pulse: a pulse is immutable, the call decides it


def build(call, limits: Limits | None = None, lookup=None):
    """(pulse, canonical call) of a design call. Raises ``Refusal``. ``limits``: what ``verse`` may be asked for (its
    gradient and slew arguments may not pass the console's); every other check of a design is ``facts``."""
    limits = limits or Limits()
    c = canonical(call, lookup)
    kind = c["kind"]
    if kind == "verse":
        for name, top in (("g_max_mt_m", limits.g_max_mt_m), ("slew_t_m_s", limits.slew_t_m_s)):
            if c.setdefault(name, float(top)) > top * (1 + 1e-9):
                raise Refusal(f"{name} = {c[name]:g} is above the console's {top:g}", name, requested=c[name], high=top, nearest_valid=top)
    key = json.dumps(c, sort_keys=True)
    if key in _BUILT:
        return _BUILT[key], c
    try:
        if kind == "sinc":
            p = pulses.sinc(c["flip_deg"], c["duration_ms"], c["tbw"], c["thickness_mm"], window=c["window"])
        elif kind == "slr":
            p = pulses.slr_flip(c["flip_deg"], c["duration_ms"], c["tbw"], c["thickness_mm"], c["tol_in_deg"], c["max_out_deg"])
        elif kind == "minphase":
            p = pulses.minphase(c["flip_deg"], c["duration_ms"], c["tbw"], c["thickness_mm"], c["transition"], c["tol_in_deg"], c["max_out_deg"],
                                c.get("rephase_fraction"), c["freq_hz"])
        if kind in ("sinc", "slr", "minphase") and c["amplitude"] != 1.0:   # the designed shape, played c["amplitude"] times as strong
            p = p.replace(b1_ut=p.b1_ut * c["amplitude"])
        if kind == "sech":
            p = pulses.sech(c["b1_peak_ut"], c["duration_ms"], c["beta_rad_per_ms"], c["mu"], c.get("thickness_mm"))
        elif kind == "points":
            amp = np.array(c["amplitude_ut"], float)
            pts = amp * np.exp(1j * np.radians(c["phase_deg"])) if "phase_deg" in c else amp.astype(complex)
            p = pulses.control_points(pts, c["duration_ms"], c["bandwidth_hz"], c["thickness_mm"], interp=c["interp"],
                                      rephase_fraction=c["rephase_fraction"])
        elif kind == "verse":
            base, _ = build(c["of"], limits)
            if not (np.ptp(base.g_mt_m) == 0 and base.g_mt_m[0] > 0):
                raise Refusal("verse needs a base design with a slice-select gradient", "of")
            p = pulses.verse(base, c["b1_peak_ut"], c["g_max_mt_m"], c["slew_t_m_s"])
    except Refusal:
        raise
    except ValueError as e:
        text = str(e)
        name = next((k for k in ("tbw", "duration_ms", "thickness_mm", "tol_in_deg", "max_out_deg") if k.split("_")[0] in text.lower()), None)
        if kind == "slr" and "ripples" in text:
            name = "tbw"
        if kind == "minphase":
            name = "tbw" if "gradient is too strong" in text else "duration_ms"
        raise Refusal(f"{kind} cannot be built from these numbers: {text[:220]}", name) from None
    if len(_BUILT) >= 512:                           # a cache, not a store: emptied when full (clear is atomic under threads)
        _BUILT.clear()
    _BUILT[key] = p
    return p, c


# ---------------------------------------------------------------------------------------------- facts and limits
def facts(pulse, limits: Limits | None = None) -> dict:
    """The record of a design: every number recomputed from the samples."""
    limits = limits or Limits()
    g, dt = np.abs(pulse.g_mt_m), pulse.dt_ms
    centres = np.cumsum(dt) - 0.5 * dt
    slew = float((np.abs(np.diff(pulse.g_mt_m)) / np.diff(centres)).max()) if pulse.n > 1 else 0.0
    out = dict(kind=pulse.kind, samples=pulse.n, duration_ms=round(pulse.duration_ms, 6), flip_deg=round(pulse.flip_deg(), 3),
               peak_b1_ut=round(pulse.peak_b1_ut, 4), energy_ut2_ms=round(pulse.energy_ut2_ms, 4),
               gradient_mt_m=round(float(g.max()), 4), slew_t_m_s=round(slew, 3), thickness_mm=pulse.thickness_mm,
               bandwidth_hz=None if pulse.bandwidth_hz is None else round(pulse.bandwidth_hz, 3),
               rephase_mt_ms_m=round(pulse.rephase_mt_ms_m, 6), variable_rate=bool(np.ptp(dt) > 1e-12),
               transition=pulse.call.get("design", {}).get("transition_fraction"))
    if pulse.call.get("generator") == "minphase":                        # the part of its two tolerances the pulse uses, and its rephasing lobe
        out.update(mask_use=round(pulse.call["design"]["use"], 4), rephase_fraction=round(pulse.call["rephase_fraction"], 4),
                   rephase_best=pulse.call["design"]["rephase_fraction"])
    bad = []
    tol = 1 + 1e-9
    if pulse.duration_ms > limits.duration_max_ms * tol:
        bad.append(dict(code="DURATION_LIMIT", value=out["duration_ms"], limit=limits.duration_max_ms, unit="ms"))
    if pulse.peak_b1_ut > limits.b1_peak_ut * tol:
        bad.append(dict(code="B1_PEAK_LIMIT", value=out["peak_b1_ut"], limit=limits.b1_peak_ut, unit="uT"))
    if float(g.max()) > limits.g_max_mt_m * tol:
        bad.append(dict(code="GRADIENT_LIMIT", value=out["gradient_mt_m"], limit=limits.g_max_mt_m, unit="mT/m"))
    if slew > limits.slew_t_m_s * tol:
        bad.append(dict(code="SLEW_LIMIT", value=out["slew_t_m_s"], limit=limits.slew_t_m_s, unit="T/m/s"))
    if limits.b1rms_ut is not None and limits.tr_ms is not None:
        out["tr_ms"] = float(limits.tr_ms)
        out["b1rms_ut"] = round(math.sqrt(pulse.energy_ut2_ms / limits.tr_ms), 4)
        if pulse.energy_ut2_ms > limits.b1rms_ut ** 2 * limits.tr_ms * tol:
            bad.append(dict(code="RF_POWER_LIMIT", value=out["b1rms_ut"], limit=limits.b1rms_ut, unit="uT"))
    out["limits"] = dict(ok=not bad, violations=bad, **limits.public())
    return out


LIMIT_WORDS = dict(DURATION_LIMIT="duration", B1_PEAK_LIMIT="peak B1", GRADIENT_LIMIT="gradient", SLEW_LIMIT="slew rate",
                   RF_POWER_LIMIT="B1rms")


def call_text(c: dict) -> str:
    """A canonical call in a few words."""
    kind = c["kind"]
    if kind == "verse":
        return f"verse of [{call_text(c['of'])}] under {c['b1_peak_ut']:g} uT, {c['g_max_mt_m']:g} mT/m, {c['slew_t_m_s']:g} T/m/s"
    if kind == "points":
        return (f"{len(c['amplitude_ut'])} control points ({c['interp']}), {c['duration_ms']:g} ms, {c['bandwidth_hz']:g} Hz over "
                f"{c['thickness_mm']:g} mm")
    if kind == "sech":
        return f"sech {c['b1_peak_ut']:g} uT, {c['duration_ms']:g} ms, beta {c['beta_rad_per_ms']:g} rad/ms, mu {c['mu']:g}"
    tail = f", {c['window']}" if kind == "sinc" else f", flip within +-{c['tol_in_deg']:g}, outside below {c['max_out_deg']:g} deg"
    if kind == "minphase":
        tail += f", transition {c['transition']:g}" + (f", {c['freq_hz']:+g} Hz" if c["freq_hz"] else "")
    if c["amplitude"] != 1.0:
        tail += f", amplitude x {c['amplitude']:g}"
    return f"{kind} flip {c['flip_deg']:g} deg, {c['duration_ms']:g} ms, tbw {c['tbw']:g}, slice {c['thickness_mm']:g} mm{tail}"


def summary(index: int, c: dict, f: dict) -> str:
    """What an agent reads about a stored design (at most about 220 tokens)."""
    lim = f["limits"]
    lines = [f"design {index} stored · {call_text(c)}"]
    row = [f"{f['duration_ms']:g} ms ({f['samples']} samples{', unequal' if f['variable_rate'] else ''})", f"flip on resonance {f['flip_deg']:g} deg",
           f"peak B1 {f['peak_b1_ut']:.4g} uT (limit {lim['b1_peak_ut']:g})", f"energy {f['energy_ut2_ms']:.4g} uT2*ms"]
    if "b1rms_ut" in f:
        row.append(f"B1rms {f['b1rms_ut']:.4g} uT over TR {f['tr_ms']:g} ms (limit {lim['b1rms_ut']:g})")
    lines.append(" · ".join(row))
    if f["thickness_mm"] is not None:
        row = [f"gradient {f['gradient_mt_m']:.4g} mT/m (limit {lim['g_max_mt_m']:g})"]
        if f["variable_rate"]:
            row.append(f"slew {f['slew_t_m_s']:.4g} T/m/s (limit {lim['slew_t_m_s']:g})")
        if f["bandwidth_hz"] is not None:
            row.append(f"bandwidth {f['bandwidth_hz']:.5g} Hz")
        if f.get("mask_use") is not None:
            row.append(f"largest deviation {f['mask_use']:.3f} of its tolerances ({'met' if f['mask_use'] <= 1 else 'NOT met: too short'})")
            row.append(f"rephasing {f['rephase_fraction']:g} (best on the ideal model {f['rephase_best']:g})")
        elif f.get("transition") is not None:
            row.append(f"transition {f['transition']:.3f} of the thickness")
        lines.append(" · ".join(row))
    if lim["ok"]:
        lines.append("limits: all met")
    else:
        v = lim["violations"]
        first = v[0]
        more = f" (also: {', '.join(LIMIT_WORDS[x['code']] for x in v[1:])})" if len(v) > 1 else ""
        power_only = all(x["code"] == "RF_POWER_LIMIT" for x in v)       # a scan has its own repetition time and its own power check
        lines.append(f"limits: VIOLATED, first {LIMIT_WORDS[first['code']]} {first['value']:.4g} against {first['limit']:g} {first['unit']}{more}; "
                     + ("it can be simulated and scanned, not delivered" if power_only else "it can be simulated, not scanned or delivered"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- the profile
def positions(spec: dict, area_mt_ms_m: float = 0.0):
    """(positions in the slice, positions outside it on both sides) on which a specification is looked at, mm:
    1 / ``POINTS_PER_THICKNESS`` of the thickness apart, and never fewer than four per period of the finest oscillation
    a pulse of selection area ``area_mt_ms_m`` can put into its profile (gamma x area cycles per mm)."""
    d, w = float(spec["thickness_mm"]), float(spec["transition"])
    step = d / POINTS_PER_THICKNESS
    if area_mt_ms_m > 0:
        step = min(step, 1.0 / (4.0 * HZ_PER_MT_M_MM * 1e-3 * area_mt_ms_m))
    half_in, start_out = (1 - w) * d / 2, (1 + w) * d / 2
    n_in = max(3, int(math.ceil(2 * half_in / step)) + 1)
    z_in = np.linspace(-half_in, half_in, n_in)
    n_out = max(2, int(math.ceil((float(spec["extent_mm"]) - start_out) / step)) + 1)
    side = np.linspace(start_out, float(spec["extent_mm"]), n_out)
    return z_in, np.concatenate([-side[::-1], side])


def figures(pulse, chain: hardware.Chain = hardware.IDEAL, spec: dict | None = None) -> dict:
    """Figures of merit of the flip-angle profile of ``pulse`` played through ``chain``. With ``spec``: on its bands,
    with the verdict of each requirement (``checks``) and the margins. Without: on the design's own thickness, with
    the transition the design states (0.5 when it states none)."""
    if pulse.thickness_mm is None:
        return dict(centre_flip_deg=float(hardware.flip_deg(pulse, chain, 0.0)), slice=False)
    own = dict(thickness_mm=pulse.thickness_mm, transition=pulse.call.get("design", {}).get("transition_fraction") or 0.5,
               extent_mm=DEFAULT_EXTENT * pulse.thickness_mm)
    s = dict(own, **(spec or {}))
    if s["extent_mm"] == "repeat":                                       # half the repeat distance of a sampled pulse under its gradient
        s["extent_mm"] = round(math.pi / (RAD_PER_MT_M_MM_MS * abs(float(pulse.g_mt_m[0])) * float(pulse.dt_ms[0])), 3)
    z_in, z_out = positions(s, abs(pulse.selection_area_mt_ms_m))
    a, b = hardware.rotate(pulse, chain, np.concatenate([z_in, z_out]))
    flip = np.degrees(2 * np.arcsin(np.clip(np.abs(b), 0, 1)))
    f_in, f_out = flip[:z_in.size], flip[z_in.size:]
    mxy = (2 * np.conj(a) * b)[:z_in.size]
    weight = float(np.abs(mxy).sum())
    worst = int(np.argmax(f_out))
    out = dict(slice=True, thickness_mm=s["thickness_mm"], transition=s["transition"], extent_mm=s["extent_mm"],
               centre_flip_deg=float(hardware.flip_deg(pulse, chain, 0.0)), in_min_deg=float(f_in.min()), in_max_deg=float(f_in.max()),
               out_max_deg=float(f_out[worst]), out_max_at_mm=float(z_out[worst]),
               refocused=float(np.abs(mxy.sum()) / weight) if weight > 0 else 0.0)
    if spec is not None and "flip_deg" in spec:
        lo, hi = spec["flip_deg"] - spec["tol_in_deg"], spec["flip_deg"] + spec["tol_in_deg"]
        checks = dict(in_slice=bool(out["in_min_deg"] >= lo and out["in_max_deg"] <= hi), out_of_slice=bool(out["out_max_deg"] <= spec["max_out_deg"]))
        if spec.get("refocused_min") is not None:
            checks["phase"] = bool(out["refocused"] >= spec["refocused_min"])
        out.update(checks=checks, in_low_deg=lo, in_high_deg=hi,
                   margins=dict(in_slice=float(min(out["in_min_deg"] - lo, hi - out["in_max_deg"])), out_of_slice=float(spec["max_out_deg"] - out["out_max_deg"])))
    return out


def chain_text(chain: hardware.Chain) -> str:
    return ", ".join([f"b1_scale {chain.tx_scale:g}", "amplifier linear" if chain.amp_sat_ut is None else f"amplifier saturating at {chain.amp_sat_ut:g} uT",
                      "no gradient delay" if not chain.grad_delay_us else f"gradient delay {chain.grad_delay_us:g} us",
                      "on resonance" if not chain.df_hz else f"off resonance {chain.df_hz:g} Hz"])


BRIEF_BINS, BRIEF_EVERY = 12, 2                      # the printed row of the short answer: out to 1.5 thicknesses, every second position


def _brief(pulse, chain, spec, index, f) -> str:
    """The short answer of ``simulate``: one line per requirement and one row of seven flip angles (two rows when the
    profile is not symmetric), below 220 tokens. The data of the answer hold the profile at all 25 positions."""
    said = ([f"b1_scale {chain.tx_scale:g}"] if chain.tx_scale != 1.0 else []) \
        + ([f"amplifier saturating at {chain.amp_sat_ut:g} uT"] if chain.amp_sat_ut is not None else []) \
        + ([f"delay {chain.grad_delay_us:g} us"] if chain.grad_delay_us else []) + ([f"off resonance {chain.df_hz:g} Hz"] if chain.df_hz else [])
    lines = [f"design {index} on the model, {', '.join(said) or 'ideal'}"]
    d, w = f["thickness_mm"], f["transition"]
    word = lambda key: "" if "checks" not in f else ": met" if f["checks"][key] else ": NOT met"   # noqa: E731
    asked = (f" (asked {f['in_low_deg']:g} to {f['in_high_deg']:g}{word('in_slice')})", f" (asked {spec['max_out_deg']:g}{word('out_of_slice')})",
             f" (asked {100 * spec['refocused_min']:g}{word('phase')})" if "checks" in f and "phase" in f["checks"] else "") if "checks" in f else ("", "", "")
    lines.append(f"slice |z| <= {(1 - w) * d / 2:.4g} mm: {f['in_min_deg']:.2f} to {f['in_max_deg']:.2f} deg{asked[0]}")
    lines.append(f"outside, to {f['extent_mm']:.4g} mm: max {f['out_max_deg']:.2f} deg at z = {f['out_max_at_mm']:.2f}{asked[1]}")
    lines.append(f"refocused {100 * f['refocused']:.1f} %{asked[2]}")
    right, left = np.array(f["profile"]["flip_deg"][:BRIEF_BINS + 1:BRIEF_EVERY]), f["profile"].get("flip_deg_negative_z")
    fmt = lambda v: " ".join(f"{x:.0f}" if x >= 10 else f"{x:.1f}" for x in v)   # noqa: E731
    lines.append(f"flip deg every {BRIEF_EVERY * PROFILE_STEP * d:g} mm, centre {'to +z' if left else 'outwards'}: {fmt(right)}")
    if left:
        lines.append(f"to -z: {fmt(left[:BRIEF_BINS + 1:BRIEF_EVERY])}")
    return "\n".join(lines)


def simulate(pulse, chain: hardware.Chain = hardware.IDEAL, spec: dict | None = None, index=None, brief: bool = False) -> tuple[str, dict]:
    """(text, data) of a design on the model. ``data`` holds the figures and the printed row at full precision.
    ``brief``: the short text (``_brief``); the data are the same."""
    f = figures(pulse, chain, spec)
    if brief and f["slice"]:
        z = np.arange(PROFILE_BINS + 1) * PROFILE_STEP * f["thickness_mm"]
        right, left = hardware.flip_deg(pulse, chain, z), hardware.flip_deg(pulse, chain, -z)
        f["profile"] = dict(z_mm=[round(float(x), 4) for x in z], flip_deg=[float(x) for x in right])
        if np.abs(right - left).max() > 0.05:
            f["profile"]["flip_deg_negative_z"] = [float(x) for x in left]
        return _brief(pulse, chain, spec, index, f), f
    head = f"design {index} on the model" if index is not None else "on the model"
    lines = [f"{head} ({chain_text(chain)})"]
    if not f["slice"]:
        lines.append(f"no slice-select gradient · flip {f['centre_flip_deg']:.2f} deg")
        return "\n".join(lines), f
    d, w = f["thickness_mm"], f["transition"]
    lines.append(f"flip at the centre {f['centre_flip_deg']:.2f} deg · in the slice (|z| <= {(1 - w) * d / 2:.3g} mm) {f['in_min_deg']:.2f} to "
                 f"{f['in_max_deg']:.2f} deg · outside (|z| from {(1 + w) * d / 2:.3g} to {f['extent_mm']:g} mm) at most {f['out_max_deg']:.2f} deg, "
                 f"at z = {f['out_max_at_mm']:.2f} mm")
    lines.append(f"in-slice signal after the rephasing lobe: {100 * f['refocused']:.1f} % of what the flip angles give")
    if "checks" in f:
        word = lambda ok: "met" if ok else "NOT met"                     # noqa: E731
        parts = [f"in the slice {f['in_low_deg']:g} to {f['in_high_deg']:g} deg: {word(f['checks']['in_slice'])}",
                 f"outside at most {spec['max_out_deg']:g} deg: {word(f['checks']['out_of_slice'])}"]
        if "phase" in f["checks"]:
            parts.append(f"signal at least {100 * spec['refocused_min']:g} %: {word(f['checks']['phase'])}")
        lines.append("specification on this model · " + " · ".join(parts))
    z = np.arange(PROFILE_BINS + 1) * PROFILE_STEP * d
    right, left = hardware.flip_deg(pulse, chain, z), hardware.flip_deg(pulse, chain, -z)
    fmt = lambda v: " ".join(f"{x:.0f}" if x >= 10 else f"{x:.1f}" for x in v)   # noqa: E731
    f["profile"] = dict(z_mm=[round(float(x), 4) for x in z], flip_deg=[float(x) for x in right])
    step = f"steps of {PROFILE_STEP:g} thickness = {PROFILE_STEP * d:g} mm"
    if np.abs(right - left).max() > 0.05:
        f["profile"]["flip_deg_negative_z"] = [float(x) for x in left]
        lines += [f"flip (deg) from the centre towards +z, {step}: {fmt(right)}", f"towards -z: {fmt(left)}"]
    else:
        lines.append(f"flip (deg) from the centre outwards, {step}: {fmt(right)}")
    return "\n".join(lines), f
