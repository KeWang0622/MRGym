"""Public physics of ``Scanner/RFPulse-v1``: what the task card, the instrument manual and the design layer let anyone
compute. Nothing here reads a world.

The class of designs (stated in the task): samples on the 0.01 ms raster, played as hard sub-pulses under ONE constant
slice gradient, whose amplitude the design chooses through ``tbw``: G = tbw / (gamma x thickness x duration). For
this class ``mrgym.design.bound`` proves a lower bound on the duration, and the generator ``minphase`` reaches it on
an ideal chain. What is left to a designer:
  share      the parts of the two tolerances the pulse is built to; the rest pays for the errors of the measurement
  gradient   the largest one the limits allow. The gradient limit is stated. The RF power limit is on the NOMINAL
             waveform, whose amplitude is 1 / (transmit scale) times the designed one: on a scanner with a low
             transmit scale the energy of the shortest pulse passes the limit, and only a weaker gradient (a
             narrower band, a longer pulse) brings it back: energy is proportional to the gradient
  three corrections   amplitude 1 / s, frequency offset = the off-resonance, rephasing fraction + delay / duration
``design_rule`` does this; it is what the reference policy calls with what it measured.
"""
from __future__ import annotations

import functools
import math

from ...design import bound, tool
from ...design.units import HZ_PER_MT_M_MM, RASTER_MS

SHARE = (0.7, 0.9)                                   # the parts of the in-slice and of the out-of-slice tolerance the reference builds to
SPEC_KEYS = ("thickness_mm", "transition", "flip_deg", "tol_in_deg", "max_out_deg")
GRADIENT_STEP = 0.995                                # how far below the energy-limited gradient the rule stays
MIN_SAMPLES = 16


def spec_of(task: dict) -> dict:
    return dict(task["design"]["specification"])


def limits_of(task: dict) -> tool.Limits:
    lim = task["design"]["limits"]
    return tool.Limits(b1_peak_ut=lim["b1_peak_ut"], g_max_mt_m=lim["g_max_mt_m"], slew_t_m_s=lim["slew_t_m_s"], b1rms_ut=lim["b1rms_ut"],
                       tr_ms=lim["tr_ms"])


def energy_limit(limits: tool.Limits) -> float:
    """The largest energy of a nominal waveform, uT^2 ms."""
    return limits.b1rms_ut ** 2 * limits.tr_ms


def tbw_of(g_mt_m: float, thickness_mm: float, duration_ms: float) -> float:
    """The time-bandwidth product that gives the gradient ``g_mt_m``, cut at 1e-6 so that the gradient is never above it."""
    return math.floor(HZ_PER_MT_M_MM * g_mt_m * thickness_mm * duration_ms * 1e-3 * 1e6) / 1e6


@functools.lru_cache(maxsize=4096)
def _shortest(key: tuple, g_mt_m: float, share: tuple) -> int:
    spec = dict(zip(SPEC_KEYS, key))
    mask = bound.Mask.of(dict(spec, tol_in_deg=share[0] * spec["tol_in_deg"], max_out_deg=share[1] * spec["max_out_deg"]))
    u = bound.precession(g_mt_m)
    hi = MIN_SAMPLES
    while not bound.fits(hi, mask, u):
        hi *= 2
        if hi > bound.MAX_SAMPLES:
            raise ValueError("no pulse of up to 2048 samples meets this mask at this gradient")
    lo = hi // 2 if hi > MIN_SAMPLES else MIN_SAMPLES - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if bound.fits(mid, mask, u) else (mid, hi)
    return hi


def shortest(spec: dict, g_mt_m: float, share=SHARE) -> int:
    """Samples of the shortest ``minphase`` pulse that stays within the parts ``share`` of the two tolerances of ``spec`` at the
    gradient ``g_mt_m`` (a bisection on the design of ``bound.best``; at least ``MIN_SAMPLES``)."""
    return _shortest(tuple(float(spec[k]) for k in SPEC_KEYS), round(float(g_mt_m), 9), (float(share[0]), float(share[1])))


def call_of(spec: dict, samples: int, g_mt_m: float, share=SHARE, amplitude: float = 1.0, freq_hz: float = 0.0,
            rephase_fraction: float | None = None) -> dict:
    """The ``design`` call of the ``minphase`` pulse of ``samples`` samples at the gradient ``g_mt_m``."""
    t = round(samples * RASTER_MS, 6)
    call = dict(kind="minphase", flip_deg=float(spec["flip_deg"]), duration_ms=t, tbw=tbw_of(g_mt_m, spec["thickness_mm"], t),
                thickness_mm=float(spec["thickness_mm"]), transition=float(spec["transition"]), tol_in_deg=round(share[0] * spec["tol_in_deg"], 4),
                max_out_deg=round(share[1] * spec["max_out_deg"], 4), amplitude=round(float(amplitude), 4), freq_hz=round(float(freq_hz), 2))
    if rephase_fraction is not None:
        call["rephase_fraction"] = round(float(rephase_fraction), 4)
    return call


def design_at(spec: dict, limits: tool.Limits, g_mt_m: float, scale: float = 1.0, df_hz: float = 0.0, delay_us: float = 0.0,
              share=SHARE):
    """(pulse, call) of the shortest pulse at the gradient ``g_mt_m`` with the three corrections for a chain of the
    given transmit scale, off-resonance and gradient delay. The limits are not looked at."""
    n = shortest(spec, g_mt_m, share)
    lo, hi = tool.PARAMETERS["amplitude"][:2]
    amplitude = min(max(1.0 / scale, lo), hi)
    base, _ = tool.build(call_of(spec, n, g_mt_m, share, amplitude, df_hz), limits)
    fraction = base.call["design"]["rephase_fraction"] + delay_us * 1e-3 / base.duration_ms
    lo, hi = tool.range_of("minphase", "rephase_fraction")
    return tool.build(call_of(spec, n, g_mt_m, share, amplitude, df_hz, min(max(fraction, lo), hi)), limits)


def design_rule(spec: dict, limits: tool.Limits, scale: float = 1.0, df_hz: float = 0.0, delay_us: float = 0.0, share=SHARE):
    """(pulse, call): the shortest ``minphase`` pulse within the limits for a chain of the given transmit scale,
    off-resonance and gradient delay. The gradient starts at its limit and is lowered in proportion while the
    nominal waveform passes the RF power or the peak limit (both fall with the gradient)."""
    g = limits.g_max_mt_m
    for _ in range(12):
        pulse, call = design_at(spec, limits, g, scale, df_hz, delay_us, share)
        over = max(pulse.energy_ut2_ms / energy_limit(limits), pulse.peak_b1_ut / limits.b1_peak_ut)
        if over <= 1.0 and tool.facts(pulse, limits)["limits"]["ok"]:
            return pulse, call
        g *= GRADIENT_STEP / max(over, 1.0)
    raise ValueError("no design within the limits")
