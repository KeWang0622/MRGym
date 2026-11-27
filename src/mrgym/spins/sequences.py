"""Spectroscopy sequences from protocol-level parameters, as block lists for ``evolve.run``.

Every builder returns a ``Sequence``: the blocks, the time the detected magnetisation spends transverse and stored
along z before the acquisition (for the relaxation factors), and the times of the refocusing pulses (for the
recovery between repetitions). All times are between pulse CENTRES; a shaped pulse of duration T takes T / 2 from the
delay on either side, and a builder raises when the pulses do not fit.

  pulse_acquire   90x, acquire
  spin_echo       90x - TE/2 - 180y - TE/2 - acquire
  press           90x - TE1/2 - 180y - (TE1 + TE2)/2 - 180y - TE2/2 - acquire          (Bottomley 1987)
  steam           90x - TE/2 - 90x - TM - 90(-x) - TE/2 - acquire                        (Frahm et al. 1987)
  slaser          90x - four adiabatic full passages, in pairs on two axes - acquire    (Scheenen et al. 2008)
  mega_press      press with two frequency-selective 180s half an echo time apart       (Mescher et al. 1998)
  jres, cosy      lists of sequences over an incremented time, for ``simulate_2d``

Phases: excitation about +x, refocusing about +y, so that the echo of an uncoupled spin has the sign of its free
induction decay and the spectrum is a positive absorption line (``evolve.spectrum``).

Crushers are ideal. Around a refocusing pulse they keep the coherences whose order changes sign (``pathway=
"refocus"``); around an editing pulse those whose order is unchanged (``"keep"``); in the mixing time of STEAM those
of order zero, which includes zero-quantum coherence; in the two TE/2 periods of STEAM the pair +1 then -1.

Slice-selective refocusing. Give ``refocus`` a ``mrgym.design.pulses.Pulse`` designed with a thickness: the pulse is
then played under its gradient on an ensemble of positions along its own axis ("x" for the first, "y" for the
second), every site at its own frequency, and the ensemble is summed with the weight spacing / thickness. The
signal is therefore relative to an ideal voxel of the nominal size. The sample is uniform and extends ``extent``
slice thicknesses along the axis (default 2.5): the result DEPENDS on that number, because a pulse refocuses its
stop-band ripple (1 % of |b|^2 for the default Shinnar-Le Roux design) of everything in that range; a singlet under
PRESS with such pulses gives 1.03 at 2.5 and 1.08 at 8 thicknesses. The chemical-shift displacement (a site whose frequency differs from the
carrier by df has its slice moved by thickness x df / bandwidth) is formed, including the compartments in which
one coupled partner is refocused and the other is not. The excitation pulse is ideal: the displacement of the
excited slice along the third axis is not modelled.

Every builder uses the pulse model "product" unless told otherwise (``model="full"``); for the sequences of the
validation the two differ by 1 to 3 % of the signal of a coupled group (rows S2b, S6b).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..design import pulses as design_pulses
from ..design import rotation
from ..design.units import RAD_PER_UT_MS
from . import evolve, operators


@dataclasses.dataclass(frozen=True, eq=False)
class Sequence:
    name: str
    blocks: tuple
    transverse_ms: float = 0.0
    longitudinal_ms: float = 0.0
    inversions_ms: tuple = ()
    resets_ms: tuple = ()             # times after the excitation at which a pulse leaves no z magnetisation (STEAM)
    params: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------------------------- pulses
def gaussian(flip_deg, duration_ms, truncation=0.01, n=None, raster_ms=0.05):
    """Gaussian pulse, truncated where its amplitude is ``truncation`` of the peak: the editing pulse of MEGA.
    Returns a ``mrgym.design.pulses.Pulse`` without gradient; ``bandwidth_hz`` is the full width at half maximum of
    its inversion profile (Mz), found numerically."""
    n = int(round(duration_ms / raster_ms)) if n is None else int(n)
    t = (np.arange(n) + 0.5) / n - 0.5
    shape = np.exp(4 * math.log(truncation) * t * t)
    dt = duration_ms / n
    b1 = shape / shape.sum() * math.radians(flip_deg) / (RAD_PER_UT_MS * dt)
    df = np.linspace(0.0, 4e3 / duration_ms, 4001)
    a, b = rotation.rotate(b1, dt, None, None, df)
    mz = np.abs(a) ** 2 - np.abs(b) ** 2
    half = 0.5 * (mz[0] + 1.0)
    k = int(np.argmax(mz > half))
    edge = df[k - 1] + (half - mz[k - 1]) / (mz[k] - mz[k - 1]) * (df[k] - df[k - 1]) if k > 0 else float("nan")
    return design_pulses.Pulse(b1, dt, 0.0, "inversion", 0.0, 2 * edge, None,
                               dict(generator="spins.gaussian", flip_deg=float(flip_deg), duration_ms=float(duration_ms),
                                    truncation=float(truncation), n=n))


def refocusing_pulse(bandwidth_hz, thickness_mm, tbw=6.0, raster_ms=0.02, **slr_args):
    """A slice-selective refocusing pulse from its bandwidth: ``mrgym.design.pulses.slr`` (180 degrees, kind
    "refocusing") of duration tbw / bandwidth. A larger time-bandwidth product gives sharper edges and a longer pulse."""
    duration = 1e3 * tbw / bandwidth_hz
    n = max(int(round(duration / raster_ms)), 16)
    return design_pulses.slr(180.0, duration, tbw, thickness_mm=thickness_mm, kind="refocusing", n=n, **slr_args)


def adiabatic_pulse(bandwidth_hz, duration_ms, thickness_mm, truncation=0.01, b1_margin=1.1, n=None, raster_ms=0.02):
    """A hyperbolic-secant full passage from its bandwidth and duration (``mrgym.design.pulses.sech``): beta from the
    truncation (amplitude at the ends / peak), mu = pi x bandwidth / beta, and a peak B1 of ``b1_margin`` times the
    value gamma B1 = bandwidth / 2 above which the inversion no longer depends on B1."""
    beta = 2 * math.acosh(1 / truncation) / duration_ms
    mu = math.pi * 1e-3 * bandwidth_hz / beta
    n = int(round(duration_ms / raster_ms)) if n is None else n
    return design_pulses.sech(b1_margin * mu * beta / RAD_PER_UT_MS, duration_ms, beta, mu, thickness_mm=thickness_mm, n=n)


def slice_positions(pulse, n=32, extent=2.5):
    """Positions (mm) and weights for the ensemble of a slice-selective pulse: ``n`` midpoints over ``extent``
    thicknesses, weight = spacing / thickness."""
    if pulse.thickness_mm is None:
        raise ValueError("the pulse has no thickness: design it with thickness_mm")
    width = extent * pulse.thickness_mm
    pos = (np.arange(n) + 0.5) / n * width - width / 2
    return pos, np.full(n, width / n / pulse.thickness_mm)


def _refocus(pulse, axis, carrier_ppm, n_pos, extent, model, close=True, phase_deg=90.0):
    """(block, duration) of a refocusing pulse: ideal when ``pulse`` is None."""
    if pulse is None:
        return evolve.rf(180.0, phase_deg, pathway="refocus"), 0.0
    if pulse.thickness_mm is None:
        return evolve.rf_pulse(pulse.b1_ut, pulse.dt_ms, None, phase_deg, carrier_ppm, pathway="refocus", model=model), pulse.duration_ms
    pos, w = slice_positions(pulse, n_pos, extent)
    return (evolve.rf_pulse(pulse.b1_ut, pulse.dt_ms, pulse.g_mt_m, phase_deg, carrier_ppm, axis, pos, w, "refocus", model, close=close),
            pulse.duration_ms)


# ---------------------------------------------------------------------------------------------- sequences
def pulse_acquire(flip_deg=90.0):
    return Sequence("pulse_acquire", (evolve.rf(flip_deg), evolve.acquire()), params=dict(flip_deg=float(flip_deg)))


def spin_echo(te_ms, refocus=None, carrier_ppm=None, n_pos=32, extent=2.5, model="product"):
    blk, dur = _refocus(refocus, "x", carrier_ppm, n_pos, extent, model)
    d = te_ms / 2 - dur / 2
    return Sequence("spin_echo", (evolve.rf(90.0), evolve.delay(d), blk, evolve.delay(d), evolve.acquire()), float(te_ms), 0.0,
                    (te_ms / 2,), (), dict(te_ms=float(te_ms)))


def press(te_ms=None, te1_ms=None, te2_ms=None, refocus=None, carrier_ppm=None, n_pos=32, extent=2.5, model="product"):
    """PRESS. Give ``te_ms`` (then TE1 = TE2 = TE / 2) or ``te1_ms`` and ``te2_ms``: TE1 from the excitation to the
    first echo, TE2 from the first echo to the second. ``refocus``: None (ideal 180s) or a design ``Pulse``, used for
    both refocusing pulses; ``carrier_ppm`` the frequency it is played at (default: the reference)."""
    if te1_ms is None or te2_ms is None:
        if te_ms is None:
            raise ValueError("give te_ms, or te1_ms and te2_ms")
        te1_ms = te_ms / 2 if te1_ms is None else te1_ms
        te2_ms = te_ms - te1_ms
    b1, dur = _refocus(refocus, "x", carrier_ppm, n_pos, extent, model)
    b2, _ = _refocus(refocus, "y", carrier_ppm, n_pos, extent, model)
    blocks = (evolve.rf(90.0), evolve.delay(te1_ms / 2 - dur / 2), b1, evolve.delay((te1_ms + te2_ms) / 2 - dur), b2,
              evolve.delay(te2_ms / 2 - dur / 2), evolve.acquire())
    return Sequence("press", blocks, float(te1_ms + te2_ms), 0.0, (te1_ms / 2, te1_ms + te2_ms / 2), (),
                    dict(te_ms=float(te1_ms + te2_ms), te1_ms=float(te1_ms), te2_ms=float(te2_ms)))


def steam(te_ms, tm_ms):
    """STEAM with ideal pulses. The crusher pair of the two TE/2 periods keeps order +1 before the second pulse
    (and the detector reads order -1 after the third); the crusher of the mixing time keeps order zero. An uncoupled
    spin gives half the signal of a spin echo."""
    blocks = (evolve.rf(90.0), evolve.delay(te_ms / 2), evolve.select((1,)), evolve.rf(90.0), evolve.spoil(), evolve.delay(tm_ms),
              evolve.rf(90.0, 180.0), evolve.delay(te_ms / 2), evolve.acquire())
    return Sequence("steam", blocks, float(te_ms), float(tm_ms), (), (te_ms / 2 + tm_ms,), dict(te_ms=float(te_ms), tm_ms=float(tm_ms)))


def steam_explicit(te_ms, tm_ms, n=8, tm_cycles=3):
    """The same sequence with gradients as explicit dephasing: one cycle in each TE/2 period on one ensemble,
    ``tm_cycles`` in the mixing time on another."""
    blocks = (evolve.rf(90.0), evolve.delay(te_ms / 2), evolve.gradient(1, "te", n), evolve.rf(90.0), evolve.gradient(tm_cycles, "tm", n),
              evolve.delay(tm_ms), evolve.rf(90.0, 180.0), evolve.gradient(1, "te", n), evolve.delay(te_ms / 2), evolve.acquire())
    return Sequence("steam", blocks, float(te_ms), float(tm_ms), (), (te_ms / 2 + tm_ms,), dict(te_ms=float(te_ms), tm_ms=float(tm_ms)))


def slaser(te_ms, afp=None, carrier_ppm=None, n_pos=32, extent=2.5, model="product"):
    """Semi-LASER: an ideal excitation and two pairs of adiabatic full passages at TE x (1, 3, 5, 7) / 8, the first
    pair on the axis "x", the second on "y". ``afp``: a pulse of ``mrgym.design.pulses.sech`` with a thickness, or
    None for ideal 180s. The ensemble of an axis stays open between the two pulses of its pair, because the phase
    that one adiabatic pulse leaves across the slice is removed only by the second."""
    tau = te_ms / 8
    if afp is None:
        pulses, dur = [evolve.rf(180.0, 90.0, pathway="refocus")] * 4, 0.0
    else:
        pulses = [_refocus(afp, axis, carrier_ppm, n_pos, extent, model, close)[0] for axis, close in (("x", False), ("x", True), ("y", False), ("y", True))]
        dur = afp.duration_ms
    blocks = [evolve.rf(90.0), evolve.delay(tau - dur / 2)]
    for k, p in enumerate(pulses):
        blocks += [p, evolve.delay((2 * tau - dur) if k < 3 else (tau - dur / 2))]
    return Sequence("slaser", tuple(blocks) + (evolve.acquire(),), float(te_ms), 0.0, tuple(tau * k for k in (1, 3, 5, 7)), (), dict(te_ms=float(te_ms)))


def mega_press(te_ms, edit_ppm, te1_ms=None, edit=None, edit_width_ppm=0.4, refocus=None, carrier_ppm=None, n_pos=32, extent=2.5,
               model="product"):
    """MEGA-PRESS, one of the two acquisitions. The editing pulses sit a quarter of the echo time before and after
    the second refocusing pulse, half an echo time apart, so that a coupling to the spins they invert is refocused
    at the echo. ``edit_ppm``: where the editing pulse is played; None plays no pulse (the "off" acquisition), a
    frequency away from the spectrum is the usual "off". ``edit``: None for an ideal inversion of the sites within
    ``edit_width_ppm`` / 2 of ``edit_ppm``, or a design ``Pulse`` (``gaussian``). The crushers of an editing pulse
    keep the coherences whose order it does not change."""
    te1_ms = te_ms / 4 if te1_ms is None else te1_ms
    te2_ms = te_ms - te1_ms
    if te2_ms / 2 - te_ms / 4 < 0:
        raise ValueError("te1_ms must be at most te_ms / 2: the second editing pulse would come after the echo")
    b1, dur = _refocus(refocus, "x", carrier_ppm, n_pos, extent, model)
    b2, _ = _refocus(refocus, "y", carrier_ppm, n_pos, extent, model)
    if edit_ppm is None:
        ed, edur = None, 0.0
    elif edit is None:
        ed, edur = evolve.rf(180.0, 0.0, (edit_ppm - edit_width_ppm / 2, edit_ppm + edit_width_ppm / 2), "keep"), 0.0
    else:
        ed, edur = evolve.rf_pulse(edit.b1_ut, edit.dt_ms, None, 0.0, edit_ppm, pathway="keep", model=model), edit.duration_ms
    q = te_ms / 4
    blocks = [evolve.rf(90.0), evolve.delay(te1_ms / 2 - dur / 2), b1, evolve.delay(q - dur / 2 - edur / 2)]
    blocks += ([ed] if ed else []) + [evolve.delay(q - dur / 2 - edur / 2), b2, evolve.delay(q - dur / 2 - edur / 2)]
    blocks += ([ed] if ed else []) + [evolve.delay(te2_ms / 2 - q - edur / 2), evolve.acquire()]
    return Sequence("mega_press", tuple(blocks), float(te_ms), 0.0, (te1_ms / 2, te1_ms + te2_ms / 2), (),
                    dict(te_ms=float(te_ms), te1_ms=float(te1_ms), te2_ms=float(te2_ms), edit_ppm=edit_ppm))


def jres(te0_ms, n1, dwell1_ms, **press_args):
    """2D J-resolved: PRESS at the echo times te0 + k dwell1, k = 0 .. n1 - 1 (the indirect time t1 is the echo
    time). After a Fourier transform over t1 a multiplet is spread along F1 by its couplings."""
    return [press(te_ms=te0_ms + k * dwell1_ms, **press_args) for k in range(n1)]


def cosy(n1, dwell1_ms, second_flip_deg=90.0, gradient_selected=True):
    """COSY: 90x - t1 - 90x - acquire, t1 = k dwell1. With ``gradient_selected`` a pair of equal crushers around the
    second pulse keeps the pathway +1 -> -1 (the echo, or N-type, signal)."""
    out = []
    for k in range(n1):
        mid = (evolve.select((1,)),) if gradient_selected else ()
        out.append(Sequence("cosy", (evolve.rf(90.0), evolve.delay(k * dwell1_ms)) + mid + (evolve.rf(second_flip_deg), evolve.acquire()),
                            float(k * dwell1_ms), 0.0, (), (), dict(t1_ms=float(k * dwell1_ms))))
    return out


# ---------------------------------------------------------------------------------------------- relaxation factors
def recovery(tr_ms, t1_ms, inversions_ms=(), flip_deg=90.0, resets_ms=()):
    """Mz just before the excitation in the steady state of a repetition: Mz is multiplied by cos(flip) by the
    excitation, inverted at each of the times ``inversions_ms`` after it (ideal 180s), set to zero at each of the
    times ``resets_ms`` (a 90 degree pulse after which the crushers leave nothing that returns to z: the third
    pulse of STEAM) and recovers with T1 in between; all transverse magnetisation is gone before the next
    excitation. Exact for an uncoupled spin and ideal pulses; applied to a coupled group as its T1. With flip = 90
    degrees and nothing else: 1 - exp(-TR / T1). Not in it: the inversion of a group by editing pulses."""
    events = sorted([(float(t), "invert") for t in inversions_ms] + [(float(t), "reset") for t in resets_ms]) + [(float(tr_ms), "end")]
    if len(events) > 1 and events[-2][0] > tr_ms:
        raise ValueError("a pulse after the end of the repetition")
    a, b, now = math.cos(math.radians(flip_deg)), 0.0, 0.0      # Mz = a x Mz(before the excitation) + b
    for t, what in events:
        e = math.exp(-(t - now) / t1_ms)
        a, b, now = a * e, b * e + (1 - e), t
        if what == "invert":
            a, b = -a, -b
        elif what == "reset":
            a, b = 0.0, 0.0
    return b / (1 - a)


def relaxation_factors(sequence, relaxation, tr_ms=None, groups=None):
    """{group: factor} for a dict ``relaxation`` group -> (T1 ms, T2 ms): exp(-transverse / T2) exp(-mixing / T1),
    times the recovery between repetitions when ``tr_ms`` is given. Groups without an entry get 1."""
    out = {}
    for g in (groups if groups is not None else relaxation):
        if g not in relaxation:
            out[g] = 1.0
            continue
        t1, t2 = relaxation[g]
        f = math.exp(-sequence.transverse_ms / t2) * math.exp(-sequence.longitudinal_ms / t1)
        if tr_ms is not None:
            f *= recovery(tr_ms, t1, sequence.inversions_ms, sequence.params.get("flip_deg", 90.0), sequence.resets_ms)
        out[g] = f
    return out


# ---------------------------------------------------------------------------------------------- simulation
def _systems(target):
    if isinstance(target, operators.SpinSystem):
        return (target,)
    if hasattr(target, "systems"):
        return tuple(target.systems)
    return tuple(s for t in target for s in _systems(t))


def simulate(target, sequence, b0_t=3.0, ref_ppm=4.7, coupling="strong", composite=True, relaxation=None, tr_ms=None):
    """Lines of the signal of ``target`` (a ``SpinSystem``, a metabolite, or a list of either) under ``sequence``,
    from the start of the acquisition. One proton at equilibrium gives a line of amplitude 1 in a pulse-acquire
    experiment. ``relaxation``: dict group -> (T1 ms, T2 ms), applied as factors on the lines (module docstring of
    ``evolve``); the decay during the acquisition is an argument of ``Lines.fid``."""
    parts = []
    for system in _systems(target):
        compiled = operators.compile(system, b0_t, ref_ppm, coupling, composite)
        acquired = evolve.run(compiled, sequence.blocks)
        if len(acquired) != 1:
            raise ValueError("simulate needs a sequence with exactly one acquire block")
        lines = acquired[0]
        if relaxation is not None:
            lines = lines.weighted(relaxation_factors(sequence, relaxation, tr_ms, lines.groups))
        parts.append(lines)
    return evolve.concat(parts)


def simulate_2d(target, sequences, n2, dwell2_ms, t2_ms=None, **args):
    """Time-domain matrix [len(sequences), n2] of a list of sequences over an incremented time (``jres``, ``cosy``)."""
    return np.array([simulate(target, seq, **args).fid(n2, dwell2_ms, t2_ms) for seq in sequences])
