"""Gradient programs: moments, k-space, b-value, hardware limits recomputed from the waveform, and the shortest lobe.

A program is, per axis ("x", "y", "z"), a list of at most 32 vertices (time in ms, amplitude in mT/m) with increasing
times, joined by straight lines; before its first and after its last vertex an axis is at zero. ``rasterise`` puts it
on the gradient raster: the value at every raster time, and the played waveform is the straight-line interpolation of
those values. Everything below is computed from the rasterised waveform, exactly for that piecewise-linear function
(no quadrature error): a vertex between two raster times, or a first vertex that is not at zero, shows up as what the
hardware would have to play.

Definitions
  sign          s(t) = +1 before the first refocusing pulse, and changes sign at every refocusing time given: the
                phase a spin has at the end is that of the "effective" gradient s G (a 180 degree pulse negates the
                phase gathered before it; up to a common sign this is the same as flipping what comes after).
  moment        m_n = integral of s(t) G(t) (t - t_ref)^n dt          mT/m ms^(n+1)   (n = 0 area, 1 velocity, 2 acceleration)
  k(t)          gamma x integral from 0 to t of s G                    cycles/mm; the phase of a spin at r is +2 pi k.r
                (a positive gradient gives a spin at positive r a positive frequency; ``rotation`` docstring)
  b-tensor      B_ij = (2 pi gamma)^2 x integral of F_i F_j dt, F = integral of s G;  b = trace(B)       s/mm^2
                (Stejskal & Tanner 1965 for a pulse pair; Basser et al. 1994 for the tensor). Signal exp(-sum B_ij D_ij).
  heating       integral of G^2 dt per axis, (mT/m)^2 ms: proportional to the resistive loss of an ideal coil.

Limits, per raster sample (formed arithmetic on the waveform; the numbers of a limit are a hardware record)
  amplitude     |G_i| per axis and |G| as a vector
  slew          |G_i(t + dt) - G_i(t)| / dt per axis and as a vector
  rule          ``Hardware.rule`` = "axis": each physical axis has its own amplifier, the limits apply per axis (a
                program on logical axes must be rotated to the physical ones first); "vector": the limits apply to the
                magnitude, which is what a program must meet to be playable at ANY orientation.

Peripheral nerve stimulation: a DECLARED MODEL, not formed physics. The rectangular strength-duration law that
IEC 60601-2-33 uses for whole-body gradients, as reproduced in the open literature (the standard itself was not
read): threshold dB/dt = rheobase x (1 + chronaxie / t_eff), rheobase 20 T/s, chronaxie 0.36 ms; t_eff of a monotonic
ramp = its total field change divided by its largest rate of change; normal operating mode = 80 % of the threshold.
dB/dt is the slew rate times a declared distance from the isocentre (``pns_radius_m``), per axis, and the axes are
combined as a root sum of squares. Real thresholds depend on the coil and the subject; vendors use coil-specific
models. The percentage is a consistent price for fast switching, never a safety statement.

The shortest lobe
  ``trapezoid``      closed form: the time-optimal triangle or trapezoid of a given area under amplitude and slew
                     limits, 2 sqrt(A / S) or A / G + G / S (the formula of ``scanner/console.py::_lobe_ms``).
  ``optimal_lobe``   linear programme on the raster: the smallest number of raster steps for which a waveform with
                     zero ends, |G| <= Gmax and |slew| <= S has the required moments. The constraints are linear in
                     the samples, feasibility is decided exactly by an LP (``scipy.optimize.linprog``, HiGHS), and a
                     waveform that is feasible in N steps is feasible in N + 1 (pad a zero on the side away from the
                     reference time), so bisection on N finds the optimum of the raster class and certifies it:
                     N - 1 steps are infeasible.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import scipy.optimize

from .units import B_PER_MT_M_MS_SQUARED_MS, CYCLES_PER_MM_PER_MT_M_MS, RASTER_MS

AXES = ("x", "y", "z")
MAX_VERTICES = 32
_GAUSS_X = np.array([0.5 - math.sqrt(0.15), 0.5, 0.5 + math.sqrt(0.15)])     # 3-point Gauss-Legendre on [0, 1]:
_GAUSS_W = np.array([5.0, 8.0, 5.0]) / 18.0                                    # exact for polynomials of degree <= 5


@dataclasses.dataclass(frozen=True)
class Hardware:
    """Limits of a gradient system and its RF peak. ``console()`` reads the ones of the virtual scanner."""
    g_max_mt_m: float = 40.0
    slew_t_m_s: float = 150.0
    raster_ms: float = RASTER_MS
    b1_peak_ut: float = 20.0
    rule: str = "axis"                       # "axis" or "vector"
    pns_rheobase_t_s: float = 20.0
    pns_chronaxie_ms: float = 0.36
    pns_radius_m: float = 0.25               # declared: distance from the isocentre at which dB/dt is taken
    pns_limit_percent: float = 80.0          # normal operating mode

    def __post_init__(self):
        if self.rule not in ("axis", "vector"):
            raise ValueError("rule must be 'axis' or 'vector'")

    @classmethod
    def console(cls, **changes):
        """The limits of ``mrgym.scanner.console.SYSTEM`` (amplitude, slew, raster, peak B1), read when called."""
        from ..scanner import console
        s = console.SYSTEM
        return cls(**dict(dict(g_max_mt_m=float(s["g_max_mt_m"]), slew_t_m_s=float(s["slew_t_m_s"]), raster_ms=float(s["raster_ms"]),
                               b1_peak_ut=float(s["b1_peak_ut"])), **changes))


# ---------------------------------------------------------------------------------------------- program
def program(**axes):
    """A gradient program from vertex lists, e.g. ``program(x=[(0, 0), (0.3, 30), (1.3, 30), (1.6, 0)])``.
    Returns ``{axis: ((t_ms, g_mt_m), ...)}`` after checking it: known axes, at most 32 vertices per axis, finite
    numbers, non-negative and strictly increasing times."""
    out = {}
    for name, vertices in axes.items():
        if name not in AXES:
            raise ValueError(f"axis must be one of {AXES}, not {name!r}")
        v = np.asarray(vertices, float)
        if v.ndim != 2 or v.shape[1] != 2 or not 1 <= v.shape[0] <= MAX_VERTICES:
            raise ValueError(f"axis {name}: 1 to {MAX_VERTICES} vertices (t_ms, g_mt_m)")
        if not np.all(np.isfinite(v)) or v[0, 0] < 0 or not np.all(np.diff(v[:, 0]) > 0):
            raise ValueError(f"axis {name}: times must be finite, not negative and strictly increasing")
        out[name] = tuple((float(t), float(g)) for t, g in v)
    if not out:
        raise ValueError("a program needs at least one axis")
    return out


def on_raster(prog, raster_ms=RASTER_MS, tol=1e-6):
    """True when every vertex time is a multiple of the raster (within ``tol`` of a raster step)."""
    return all(abs(t / raster_ms - round(t / raster_ms)) <= tol for v in prog.values() for t, _ in v)


def rasterise(prog, raster_ms=RASTER_MS, t_end_ms=None):
    """(t [N + 1] in ms, g [N + 1, 3] in mT/m, columns x, y, z): the program at the raster times 0, dt, ..., N dt, with
    N dt >= the last vertex time (or ``t_end_ms``). The played waveform is the straight-line interpolation."""
    prog = program(**prog)
    last = max(v[-1][0] for v in prog.values())
    if t_end_ms is not None:
        last = max(last, t_end_ms)
    n = max(1, int(math.ceil(last / raster_ms - 1e-9)))
    t = np.arange(n + 1) * raster_ms
    g = np.zeros((n + 1, 3))
    tol = 1e-6 * raster_ms                                      # k x raster and a decimal vertex time differ in the last bit
    for name, v in prog.items():
        tv, gv = np.array(v).T
        g[:, AXES.index(name)] = np.where((t < tv[0] - tol) | (t > tv[-1] + tol), 0.0, np.interp(t, tv, gv))
    return t, g


def _waveform(t_ms, g):
    t, g = np.asarray(t_ms, float), np.asarray(g, float)
    if g.ndim == 1:
        g = g[:, None]
    if t.ndim != 1 or g.ndim != 2 or g.shape[0] != t.size or t.size < 2 or not np.all(np.diff(t) > 0):
        raise ValueError("a waveform is increasing times [N + 1] and amplitudes [N + 1] or [N + 1, axes]")
    return t, g


def _segments(t, g, refocus_ms):
    """Segments of the waveform split at the refocusing times: start times, lengths, start and end amplitudes
    [segments, axes], and the sign of every segment."""
    cuts = np.asarray(sorted(float(r) for r in np.atleast_1d(refocus_ms)), float) if np.size(refocus_ms) else np.zeros(0)
    inner = cuts[(cuts > t[0]) & (cuts < t[-1])]
    if inner.size:
        new = inner[~np.isin(inner, t)]
        if new.size:
            order = np.argsort(np.concatenate([t, new]), kind="stable")
            g = np.concatenate([g, np.stack([np.interp(new, t, g[:, i]) for i in range(g.shape[1])], 1)])[order]
            t = np.concatenate([t, new])[order]
    mid = 0.5 * (t[:-1] + t[1:])
    sign = 1.0 - 2.0 * (np.searchsorted(cuts, mid, side="right") % 2)
    return t[:-1], np.diff(t), g[:-1], g[1:], sign, t


def moments(t_ms, g, t_ref_ms, order=2, refocus_ms=()):
    """Moments m_0 .. m_order about ``t_ref_ms``, [order + 1, axes], in mT/m ms^(n+1). Orders 0 to 4 (the segment
    quadrature is exact up to there)."""
    if not 0 <= int(order) <= 4:
        raise ValueError("moments are computed for orders 0 to 4")
    t, g = _waveform(t_ms, g)
    t0, h, g0, g1, sign, _ = _segments(t, g, refocus_ms)
    out = np.zeros((order + 1, g.shape[1]))
    for x, w in zip(_GAUSS_X, _GAUSS_W):
        tau = t0 + x * h - t_ref_ms
        val = (g0 + x * (g1 - g0)) * (w * h * sign)[:, None]
        for n in range(order + 1):
            out[n] += (val * (tau ** n)[:, None]).sum(0)
    return out


def k_trajectory(t_ms, g, refocus_ms=()):
    """(times, k) with k [M, axes] in cycles/mm at the times of the waveform (and at the refocusing times, where the
    sign changes): gamma x the running integral of s G."""
    t, g = _waveform(t_ms, g)
    _, h, g0, g1, sign, tt = _segments(t, g, refocus_ms)
    area = np.concatenate([np.zeros((1, g.shape[1])), np.cumsum(0.5 * (g0 + g1) * (h * sign)[:, None], 0)])
    return tt, CYCLES_PER_MM_PER_MT_M_MS * area


def b_tensor(t_ms, g, refocus_ms=()):
    """b-tensor [axes, axes] in s/mm^2 of the waveform (exact for the piecewise-linear waveform)."""
    t, g = _waveform(t_ms, g)
    _, h, g0, g1, sign, _ = _segments(t, g, refocus_ms)
    s = sign[:, None]
    start = np.concatenate([np.zeros((1, g.shape[1])), np.cumsum(0.5 * (g0 + g1) * h[:, None] * s, 0)])[:-1]
    out = np.zeros((g.shape[1], g.shape[1]))
    for x, w in zip(_GAUSS_X, _GAUSS_W):
        u = x * h[:, None]
        f = start + s * (g0 * u + (g1 - g0) * u * u / (2 * h[:, None]))            # F at the Gauss point of every segment
        out += np.einsum("s,si,sj->ij", w * h, f, f)
    return B_PER_MT_M_MS_SQUARED_MS * out


def b_value(t_ms, g, refocus_ms=()):
    """b-value in s/mm^2: the trace of the b-tensor."""
    return float(np.trace(b_tensor(t_ms, g, refocus_ms)))


def heating(t_ms, g):
    """Integral of G^2 dt per axis, (mT/m)^2 ms."""
    t, g = _waveform(t_ms, g)
    h = np.diff(t)[:, None]
    return (h * (g[:-1] ** 2 + g[:-1] * g[1:] + g[1:] ** 2) / 3.0).sum(0)


# ---------------------------------------------------------------------------------------------- limits
def pns_percent(t_ms, g, hardware=None):
    """Stimulation figure per axis, in percent of the declared threshold (module docstring), and where it occurs.
    Returns (percent [axes], time of the worst ramp's start [axes] in ms)."""
    hw = hardware or Hardware()
    t, g = _waveform(t_ms, g)
    dt = np.diff(t)
    pct, when = np.zeros(g.shape[1]), np.zeros(g.shape[1])
    for i in range(g.shape[1]):
        dg = np.diff(g[:, i])
        sgn = np.sign(dg)
        idx = np.nonzero(sgn)[0]
        if idx.size == 0:
            continue
        brk = np.nonzero((np.diff(idx) > 1) | (np.diff(sgn[idx]) != 0))[0] + 1      # a pause or a reversal ends a ramp
        for run in np.split(idx, brk):
            rate = np.abs(dg[run]) / dt[run]                                         # mT/m/ms = T/m/s
            t_eff = float(np.abs(dg[run]).sum() / rate.max())                        # ms
            value = 100.0 * rate.max() * hw.pns_radius_m / (hw.pns_rheobase_t_s * (1 + hw.pns_chronaxie_ms / t_eff))
            if value > pct[i]:
                pct[i], when[i] = value, t[run[0]]
    return pct, when


def check(t_ms, g, hardware=None, raster_ms=None):
    """Limits recomputed from the waveform, per raster sample.

    Returns a dictionary: per-axis and vector peaks of amplitude (mT/m) and slew (T/m/s), stimulation percent per
    axis and combined, heating per axis, whether the samples sit on the raster and start and end at zero, and
    ``violations``: a list of ``dict(code, axis, t_ms, value, limit)`` for the first offending sample of every broken
    rule, in the order GRADIENT_LIMIT, SLEW_LIMIT, STIMULATION_LIMIT, NOT_ON_RASTER, NOT_ZERO_AT_ENDS. ``ok`` is True
    when the list is empty. Amplitude and slew are judged by ``hardware.rule``."""
    hw = hardware or Hardware()
    raster = hw.raster_ms if raster_ms is None else raster_ms
    t, g = _waveform(t_ms, g)
    names = AXES[:g.shape[1]] if g.shape[1] <= 3 else tuple(str(i) for i in range(g.shape[1]))
    dt = np.diff(t)
    slew = np.abs(np.diff(g, axis=0)) / dt[:, None]
    g_vec, s_vec = np.sqrt((g * g).sum(1)), np.sqrt((np.diff(g, axis=0) ** 2).sum(1)) / dt
    pns, pns_t = pns_percent(t, g, hw)
    tol = 1 + 1e-9
    bad = []

    def first(code, values, limit, times, vector):
        """values [samples, axes] or [samples]; record the first sample over the limit."""
        over = values > limit * tol
        if over.any():
            i = int(np.argmax(over.any(1) if over.ndim == 2 else over))
            worst = 0 if vector else int(np.argmax(values[i]))          # of the axes over the limit at that sample, the largest
            axis = "vector" if vector else names[worst]
            v = values[i] if vector else values[i, worst]
            bad.append(dict(code=code, axis=axis, t_ms=round(float(times[i]), 6), value=round(float(v), 6), limit=float(limit)))

    if hw.rule == "axis":
        first("GRADIENT_LIMIT", np.abs(g), hw.g_max_mt_m, t, False)
        first("SLEW_LIMIT", slew, hw.slew_t_m_s, t[:-1], False)
    else:
        first("GRADIENT_LIMIT", g_vec, hw.g_max_mt_m, t, True)
        first("SLEW_LIMIT", s_vec, hw.slew_t_m_s, t[:-1], True)
    pns_all = float(np.sqrt((pns * pns).sum()))
    if pns_all > hw.pns_limit_percent * tol:
        worst = int(np.argmax(pns))
        bad.append(dict(code="STIMULATION_LIMIT", axis=names[worst], t_ms=round(float(pns_t[worst]), 6), value=round(pns_all, 4),
                        limit=float(hw.pns_limit_percent)))
    steps = t / raster
    uniform = bool(np.all(np.abs(steps - np.round(steps)) < 1e-6) and np.all(np.abs(dt - raster) < 1e-6 * raster))
    if not uniform:
        bad.append(dict(code="NOT_ON_RASTER", axis="all", t_ms=round(float(t[0]), 6), value=float(dt.min()), limit=float(raster)))
    floor = 1e-9 * max(1.0, float(np.abs(g).max()))                          # an end value below this is rounding
    if np.any(np.abs(g[0]) > floor) or np.any(np.abs(g[-1]) > floor):
        end = 0 if np.any(np.abs(g[0]) > floor) else -1
        bad.append(dict(code="NOT_ZERO_AT_ENDS", axis=names[int(np.argmax(np.abs(g[end])))], t_ms=round(float(t[end]), 6),
                        value=round(float(np.abs(g[end]).max()), 6), limit=0.0))
    return dict(ok=not bad, violations=bad, rule=hw.rule, duration_ms=round(float(t[-1] - t[0]), 6),
                peak_mt_m={n: round(float(v), 6) for n, v in zip(names, np.abs(g).max(0))}, peak_vector_mt_m=round(float(g_vec.max()), 6),
                slew_t_m_s={n: round(float(v), 6) for n, v in zip(names, slew.max(0))}, slew_vector_t_m_s=round(float(s_vec.max()), 6),
                pns_percent={n: round(float(v), 4) for n, v in zip(names, pns)}, pns_combined_percent=round(pns_all, 4),
                heating_mt2_m2_ms={n: round(float(v), 6) for n, v in zip(names, heating(t, g))}, on_raster=uniform)


def check_pulse_gradient(pulse, hardware=None):
    """The same limits for the slice-select gradient of a ``pulses.Pulse``, whose samples are constant during their
    own (possibly unequal) durations: amplitude per sample, slew between the centres of neighbouring samples. The
    ramps to and from zero around the pulse are not part of the pulse and are not checked. Returns
    ``dict(ok, violations, peak_mt_m, slew_t_m_s, peak_b1_ut)`` with codes GRADIENT_LIMIT, SLEW_LIMIT, B1_PEAK_LIMIT."""
    hw = hardware or Hardware()
    g, dt = np.asarray(pulse.g_mt_m, float), np.asarray(pulse.dt_ms, float)
    centres = np.cumsum(dt) - 0.5 * dt
    slew = np.abs(np.diff(g)) / np.diff(centres) if g.size > 1 else np.zeros(1)
    peak, peak_b1 = float(np.abs(g).max()), float(np.abs(pulse.b1_ut).max())
    bad = []
    for code, value, limit, when in (("GRADIENT_LIMIT", peak, hw.g_max_mt_m, centres[int(np.argmax(np.abs(g)))]),
                                     ("SLEW_LIMIT", float(slew.max()), hw.slew_t_m_s, centres[int(np.argmax(slew))]),
                                     ("B1_PEAK_LIMIT", peak_b1, hw.b1_peak_ut, centres[int(np.argmax(np.abs(pulse.b1_ut)))])):
        if value > limit * (1 + 1e-9):
            bad.append(dict(code=code, axis="slice", t_ms=round(float(when), 6), value=round(value, 6), limit=float(limit)))
    return dict(ok=not bad, violations=bad, peak_mt_m=round(peak, 6), slew_t_m_s=round(float(slew.max()), 6), peak_b1_ut=round(peak_b1, 6))


# ---------------------------------------------------------------------------------------------- shortest lobes
def trapezoid(area_mt_ms_m, g_max_mt_m, slew_t_m_s, raster_ms=None, start_ms=0.0):
    """The shortest lobe of the given area (mT/m ms, signed) under an amplitude and a slew limit.

    Closed form (``raster_ms`` None): a triangle of peak sqrt(|A| S) and duration 2 sqrt(|A| / S) when
    |A| <= G^2 / S, otherwise a trapezoid at G with ramps G / S and duration |A| / G + G / S.
    With a raster: ramp and plateau are rounded UP to the raster and the peak is lowered so that the area is exact;
    this is a valid lobe on the raster, at most three raster steps longer than the closed form, but not always the
    shortest one on the raster (``optimal_lobe`` is). Returns ``dict(vertices, duration_ms, peak_mt_m, ramp_ms, flat_ms)``."""
    if not (g_max_mt_m > 0 and slew_t_m_s > 0):
        raise ValueError("limits must be positive")
    a, sign = abs(float(area_mt_ms_m)), (1.0 if area_mt_ms_m >= 0 else -1.0)
    if a == 0:
        return dict(vertices=((start_ms, 0.0),), duration_ms=0.0, peak_mt_m=0.0, ramp_ms=0.0, flat_ms=0.0)
    if a <= g_max_mt_m ** 2 / slew_t_m_s:
        ramp, flat = math.sqrt(a / slew_t_m_s), 0.0
    else:
        ramp, flat = g_max_mt_m / slew_t_m_s, a / g_max_mt_m - g_max_mt_m / slew_t_m_s
    if raster_ms is not None:                                   # whole numbers of raster steps, so that the vertices sit on it
        steps = lambda x: int(math.ceil(x / raster_ms - 1e-9))               # noqa: E731
        ramp_n = max(1, steps(ramp))
        flat_n = steps(max(0.0, a / g_max_mt_m - ramp_n * raster_ms)) if a / (ramp_n * raster_ms) > g_max_mt_m * (1 + 1e-12) else 0
        peak = a / ((ramp_n + flat_n) * raster_ms)
        counts = [0, ramp_n, ramp_n + flat_n, 2 * ramp_n + flat_n] if flat_n else [0, ramp_n, 2 * ramp_n]
        amps = [0.0, peak, peak, 0.0] if flat_n else [0.0, peak, 0.0]
        return dict(vertices=tuple((start_ms + c * raster_ms, sign * g) for c, g in zip(counts, amps)),
                    duration_ms=(2 * ramp_n + flat_n) * raster_ms, peak_mt_m=sign * peak, ramp_ms=ramp_n * raster_ms, flat_ms=flat_n * raster_ms)
    peak = a / (ramp + flat)
    times = [0.0, ramp, ramp + flat, 2 * ramp + flat] if flat > 0 else [0.0, ramp, 2 * ramp]
    amps = [0.0, peak, peak, 0.0] if flat > 0 else [0.0, peak, 0.0]
    return dict(vertices=tuple((start_ms + t, sign * g) for t, g in zip(times, amps)), duration_ms=2 * ramp + flat,
                peak_mt_m=sign * peak, ramp_ms=ramp, flat_ms=flat)


def _moment_weights(n_steps, raster_ms, t_ref_ms, order):
    """w [order + 1, n_steps + 1] with m_n = sum_i w[n, i] g_i for the piecewise-linear waveform on the raster."""
    t0 = np.arange(n_steps) * raster_ms
    w = np.zeros((order + 1, n_steps + 1))
    for x, gw in zip(_GAUSS_X, _GAUSS_W):
        tau = t0 + x * raster_ms - t_ref_ms
        for n in range(order + 1):
            w[n, :-1] += gw * raster_ms * (1 - x) * tau ** n
            w[n, 1:] += gw * raster_ms * x * tau ** n
    return w


def _lobe_lp(n_steps, targets, g_max, slew, raster_ms, ref, offset_ms):
    """Feasibility LP for a lobe of ``n_steps`` raster steps; minimises the peak. Returns the samples or None."""
    if n_steps < 2:
        return None
    t_ref = n_steps * raster_ms + offset_ms if ref == "end" else -offset_ms
    order = max(targets)
    w = _moment_weights(n_steps, raster_ms, t_ref, order)[:, 1:-1]           # interior samples; the ends are zero
    m = n_steps - 1
    rows = sorted(targets)
    scale = np.array([max(np.abs(w[n]).max(), 1e-300) for n in rows])
    a_eq = np.hstack([w[rows] / scale[:, None], np.zeros((len(rows), 1))])
    b_eq = np.array([targets[n] for n in rows]) / scale
    eye = np.eye(m)
    diff = np.zeros((m + 1, m))                                              # g_1 - 0, g_2 - g_1, ..., 0 - g_(m)
    diff[np.arange(m), np.arange(m)] = 1.0
    diff[np.arange(1, m + 1), np.arange(m)] -= 1.0
    step = slew * raster_ms
    a_ub = np.vstack([np.hstack([eye, -np.ones((m, 1))]), np.hstack([-eye, -np.ones((m, 1))]),
                      np.hstack([diff, np.zeros((m + 1, 1))]), np.hstack([-diff, np.zeros((m + 1, 1))])])
    b_ub = np.concatenate([np.zeros(2 * m), np.full(2 * (m + 1), step)])
    cost = np.zeros(m + 1)
    cost[-1] = 1.0
    res = scipy.optimize.linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                                 bounds=[(None, None)] * m + [(0.0, g_max)], method="highs")
    if res.status != 0:
        return None
    return np.concatenate([[0.0], res.x[:m], [0.0]])


def optimal_lobe(targets, g_max_mt_m, slew_t_m_s, raster_ms=RASTER_MS, ref="end", offset_ms=0.0, max_steps=4096):
    """The shortest waveform on the raster with the required moments: a certified optimum (module docstring).

    ``targets``   {order: value}: moments in mT/m ms^(order + 1), e.g. {0: 5.0} (area), {0: 0, 1: 2.0} (bipolar
                  velocity encoding), {0: 5.0, 1: 0.0} (flow-compensated lobe).
    ``ref``       where the moments are taken: "end" = ``offset_ms`` after the end of the lobe (an echo that follows
                  it), "start" = ``offset_ms`` before its start (an excitation that precedes it).
    The waveform starts and ends at zero. Among the waveforms of the least duration the one of least peak amplitude
    is returned. Returns ``dict(steps, duration_ms, t_ms, g_mt_m, peak_mt_m, moments, residual, certificate)``;
    ``certificate`` names the infeasible LP one raster step shorter. Raises if nothing is feasible in ``max_steps``."""
    if ref not in ("end", "start"):
        raise ValueError("ref must be 'end' or 'start'")
    targets = {int(k): float(v) for k, v in targets.items()}
    if not targets or min(targets) < 0 or max(targets) > 4:
        raise ValueError("targets are {order: value} with orders 0 to 4")
    if all(v == 0 for v in targets.values()):
        raise ValueError("all moments are zero: the shortest waveform is no waveform")

    def solve(n):
        return _lobe_lp(n, targets, g_max_mt_m, slew_t_m_s, raster_ms, ref, offset_ms)

    lo, hi = 1, 2                                                             # lo infeasible, hi to be shown feasible
    g = solve(hi)
    while g is None:
        if hi >= max_steps:
            raise ValueError(f"no waveform with these moments within {max_steps} raster steps")
        lo, hi = hi, min(hi * 2, max_steps)
        g = solve(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        trial = solve(mid)
        if trial is None:
            lo = mid
        else:
            hi, g = mid, trial
    t = np.arange(hi + 1) * raster_ms
    t_ref = hi * raster_ms + offset_ms if ref == "end" else -offset_ms
    got = moments(t, g, t_ref, max(targets))[:, 0]
    return dict(steps=hi, duration_ms=hi * raster_ms, t_ms=t, g_mt_m=g, peak_mt_m=float(np.abs(g).max()),
                moments={n: float(got[n]) for n in sorted(targets)},
                residual=max(abs(got[n] - targets[n]) for n in targets),
                certificate=f"the linear programme with {lo} raster steps is infeasible; with {hi} it is feasible")
