"""Sequences as data: functions that return a block list (``blocks.Sequence``) from protocol-level parameters.

The six sequences the scanner serves, written as the model the engine computes today (``scanner/signal.py``,
``scanner/epg.py``, ``scanner/kspace.py``; ``mrgym.core.validate`` rows Q compare the numbers):

  spgr    RF-spoiled gradient echo. spoiling="rf": the engine's model of ``spgr`` (tier 2): one cycle of dephasing
          spread over the whole TR, quadratic RF phase, the sample right after the pulse (the echo-time decay is
          applied outside, as the engine does). spoiling="ideal": an ideal spoiler before every pulse (the Ernst
          formula: the engine's tier 1 and its ``epi``); the sample may then sit at any echo time.
  afi     dual-TR gradient echo: pulses on raster positions 0 and 1 of a raster of n + 1 intervals of TR1.
  epi     one excitation per TR with an ideal spoiler (the Ernst steady state).
  se      90 - 180 spin echo; the refocusing pulse (twice the excitation angle, phase 90 degrees) is the IDEAL
          crushed pulse and an ideal spoiler ends the TR, so Mz of a column evolves as one number.
  ir      the same after an inversion of efficiency ``inv_eff`` that acts on Mz only.
  bssfp   balanced SSFP, sample at TR / 2, RF phase advancing by ``phase_inc_deg`` per TR.
``served`` maps a realised protocol of the console to these, and to the block lists of the sequences a console serves
from the core (fse, mese, megre, dwse; ``scanner/served.py``): it is the one source of both.
Saturation (``sat=(q, delay_ms)`` of spgr, epi, se, ir): the engine's model of a spectrally selective saturation
pulse for ONE resonance line, an event of every repetition that multiplies the longitudinal magnetisation by q
``delay_ms`` before the excitation (engine 0.3.0; q = 1 - efficiency x the share of the band the line lies in).
The lines of fat are saturated differently, so a fat tissue is one block list per line.

Eleven more sequences, to show that a new sequence is data (a console that lists them serves fse, mese, megre and dwse
from these block lists):
  fse, mese          CPMG echo train (constant or variable refocusing angles, real or ideal crushers); the echoes
                     feed k-space lines (``echo_of_line``) or a T2 fit
  megre              several gradient echoes after one pulse
  spgr_lobe          RF-spoiled gradient echo with a real spoiler lobe after the readout and any phase increment
  dess               the free induction decay and the echo of an unbalanced steady state in one TR
  mprage             inversion, a train of small-angle pulses, a recovery time (Look-Locker with n pulses)
  t2prep             T2 preparation (90 - 180 - 90 back) before a readout pulse
  satrec             saturation recovery
  dwse, steam        Stejskal-Tanner diffusion weighting in a spin echo and in a stimulated echo
  mt_spgr            gradient echo with an off-resonance saturation pulse acting on a bound pool

Times in ms, angles in degrees. A slice-selective excitation is ``selective=blocks.Selective(tbw, ...)``; ``se`` and
``ir`` then refocus on a slice 1.5 times as thick (``REFOCUS_WIDTH``), as the engine does. Every periodic sequence is
ONE repeat block, so ``periodic.steady_state`` takes it as it is and ``epg.run`` steps ``repetitions`` of it.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..design.units import CYCLES_PER_MM_PER_MT_M_MS, GAMMA_HZ_PER_T
from . import blocks as B

REFOCUS_WIDTH = 1.5


def spoiler_cycle_mm(spoiler_mt_ms_per_m):
    """Length (mm) over which a gradient lobe of the given area (mT ms / m) dephases by one cycle: 1 / (gamma A)."""
    return 1e3 / (GAMMA_HZ_PER_T * spoiler_mt_ms_per_m * 1e-6)


def _refocus(sel):
    return None if sel is None else dataclasses.replace(sel, rephase=0.0, width=sel.width * REFOCUS_WIDTH)


def refocusing_pulse(sel):
    """The refocusing pulse that goes with the excitation pulse ``sel`` in every spin-echo sequence of this module:
    the same shape on a slice ``REFOCUS_WIDTH`` times as thick, without a rephasing lobe."""
    return _refocus(sel)


# ---------------------------------------------------------------------------------------------- the six served
def _saturation(sat, room_ms):
    """The blocks that end a repetition with a saturation pulse: ([Mz -> q Mz, delay], the delay). ``room_ms``: the
    time between the last event before it and the next excitation."""
    if sat is None:
        return [], 0.0
    q, delay = float(sat[0]), float(sat[1])
    if not 0 <= delay <= room_ms:
        raise ValueError("the saturation pulse does not fit before the excitation")
    return [B.mz(q), B.delay(delay)], delay


def spgr(tr_ms, flip_deg, te_ms=0.0, spoiling="rf", phase_inc_deg=117.0, spoiler_mt_ms_per_m=100.0, selective=None, repetitions=1, sat=None):
    """``sat``: with spoiling="rf" the saturation multiplies the longitudinal states of every order, the spoiler has
    finished its cycle before it, and nothing else happens to the transverse states. Without diffusion this is the
    engine's model exactly (row Q16b); the engine spreads the cycle over the whole TR, so with diffusion the two
    differ slightly (row Q16d)."""
    exc = B.rf(flip_deg, selective=selective)
    tail, d = _saturation(sat, tr_ms - te_ms)
    if spoiling == "ideal":
        body = [exc, B.delay(te_ms), B.adc("echo", static_ms=te_ms), B.delay(tr_ms - te_ms - d), B.spoil()] + tail
        return B.sequence([B.repeat(body, repetitions)], name="spgr-ideal")
    if spoiling != "rf" or te_ms:
        raise ValueError("spgr: spoiling is 'rf' (sample right after the pulse) or 'ideal'; for a real spoiler lobe use spgr_lobe")
    body = [exc, B.adc("echo"), B.gradient(1, ms=tr_ms - d)] + tail
    return B.sequence([B.repeat(body, repetitions, spoil_inc_deg=phase_inc_deg)], cycle_mm=spoiler_cycle_mm(spoiler_mt_ms_per_m), name="spgr")


def afi(tr1_ms, tr_ratio, flip_deg, spoiling="rf", phase_inc_deg=117.0, spoiler_mt_ms_per_m=100.0, selective=None, repetitions=1):
    """Samples labelled ("s", 0) (after the pulse that follows the long interval: S1) and ("s", 1) (S2); with
    spoiling="rf" the other raster positions of a period carry empty samples."""
    n = int(tr_ratio)
    if n != tr_ratio or n < 2:
        raise ValueError("tr_ratio is a whole number >= 2")
    if spoiling == "ideal":
        body = [B.rf(flip_deg, selective=selective), B.adc(("s", 0)), B.delay(tr1_ms), B.spoil(),
                B.rf(flip_deg, selective=selective), B.adc(("s", 1)), B.delay(n * tr1_ms), B.spoil()]
        return B.sequence([B.repeat(body, repetitions)], name="afi-ideal")
    body = [B.rf(selective=selective, flip_table="pattern"), B.adc("s"), B.gradient(1, ms=tr1_ms)]
    rep = B.repeat(body, repetitions * (n + 1), spoil_inc_deg=phase_inc_deg, tables=dict(pattern=(flip_deg, flip_deg) + (0.0,) * (n - 1)))
    return B.sequence([rep], cycle_mm=spoiler_cycle_mm(spoiler_mt_ms_per_m), name="afi")


def epi(tr_ms, flip_deg=90.0, selective=None, repetitions=1, sat=None):
    seq = spgr(tr_ms, flip_deg, spoiling="ideal", selective=selective, repetitions=repetitions, sat=sat)
    return dataclasses.replace(seq, name="epi")


def se(tr_ms, te_ms, flip_deg=90.0, selective=None, crusher_cycles=None, repetitions=1, sat=None):
    """``crusher_cycles`` None: the ideal crushed refocusing pulse and an ideal spoiler at the end of the TR (the
    engine's model). A number: real crushers of that many cycles either side of the refocusing pulse and nothing
    else, which keeps the stimulated echoes of the following repetitions (docs/physics/limits.md, 8a)."""
    exc = B.rf(flip_deg, 0.0, selective=selective)
    if crusher_cycles is None:
        tail, d = _saturation(sat, tr_ms - te_ms)
        body = [exc, B.delay(te_ms / 2), B.rf(2 * flip_deg, 90.0, crushed=True, selective=_refocus(selective)), B.delay(te_ms / 2),
                B.adc("echo"), B.delay(tr_ms - te_ms - d), B.spoil()] + tail
    else:
        if sat is not None:
            raise ValueError("se: the saturation pulse is modelled with the ideal crushed refocusing pulse only")
        body = [exc, B.delay(te_ms / 2), B.gradient(crusher_cycles), B.rf(2 * flip_deg, 90.0, selective=_refocus(selective)),
                B.gradient(crusher_cycles), B.delay(te_ms / 2), B.adc("echo"), B.delay(tr_ms - te_ms)]
    return B.sequence([B.repeat(body, repetitions)], name="se")


def ir(tr_ms, te_ms, ti_ms, flip_deg=90.0, inv_eff=1.0, selective=None, repetitions=1, sat=None):
    """``sat``: the saturation pulse lies between the inversion and the excitation."""
    prep, d = _saturation(sat, ti_ms)
    body = [B.mz(-inv_eff), B.delay(ti_ms - d)] + prep + [B.rf(flip_deg, 0.0, selective=selective), B.delay(te_ms / 2),
            B.rf(2 * flip_deg, 90.0, crushed=True, selective=_refocus(selective)), B.delay(te_ms / 2), B.adc("echo"),
            B.delay(tr_ms - ti_ms - te_ms), B.spoil()]
    return B.sequence([B.repeat(body, repetitions)], name="ir")


def bssfp(tr_ms, flip_deg, phase_inc_deg=180.0, selective=None, repetitions=1):
    """With a selective pulse the slice gradient is balanced: half of the rewinding area before the pulse, half after."""
    if selective is not None:
        selective = dataclasses.replace(selective, prephase=selective.rephase)
    body = [B.rf(flip_deg, selective=selective), B.delay(tr_ms / 2), B.adc("echo"), B.delay(tr_ms / 2)]
    return B.sequence([B.repeat(body, repetitions, phase_inc_deg=phase_inc_deg)], name="bssfp")


def ir_gre(tr_ms, ti_ms, flip_deg, inv_eff=1.0, repetitions=1):
    """Inversion, TI, one gradient-echo pulse, ideal spoiler (``scanner/signal.py::ir_gre``)."""
    body = [B.mz(-inv_eff), B.delay(ti_ms), B.rf(flip_deg), B.adc("echo"), B.delay(tr_ms - ti_ms), B.spoil()]
    return B.sequence([B.repeat(body, repetitions)], name="ir-gre")


def served(rp, selective=None, inv_eff=1.0, tier=2, sat=None):
    """The block list of a realised protocol of the console (``console.plan(...)["realised"]``): (sequence, labels of
    the samples that are the engine's contrasts, in order). ``selective``: the excitation pulse, or None for the hard
    pulse. ``sat`` = (q, delay_ms): the saturation of ONE resonance line by the fat-saturation pulse of the protocol
    (spgr, epi, se, ir; the caller gives q of the line and the delay the console planned, ``fatsat_delay_ms``); None:
    a line the pulse does not reach, or a protocol without fat saturation. ``phase_cycle_deg`` of bssfp is the advance
    of the RF phase per TR (rows Q10, Q10b; engine 0.2.0 counted it the other way and this function followed it until
    the engine was corrected)."""
    name = rp["sequence"]
    if sat is not None and name in ("afi", "bssfp"):
        raise ValueError(f"no saturation pulse in the block list of {name}")
    if name == "spgr":
        kw = dict(spoiling="rf", phase_inc_deg=rp["rf_phase_increment_deg"], spoiler_mt_ms_per_m=rp["spoiler_mt_ms_per_m"]) if tier == 2 else dict(spoiling="ideal")
        return spgr(rp["tr_ms"], rp["flip_deg"], selective=selective, sat=sat, **kw), ("echo",)
    if name == "afi":
        kw = dict(spoiling="rf", phase_inc_deg=rp["rf_phase_increment_deg"], spoiler_mt_ms_per_m=rp["spoiler_mt_ms_per_m"]) if tier == 2 else dict(spoiling="ideal")
        return afi(rp["tr_ms"], rp["tr_ratio"], rp["flip_deg"], selective=selective, **kw), (("s", 0), ("s", 1))
    if name == "epi":
        return epi(rp["tr_ms"], rp["flip_deg"], selective=selective, sat=sat), ("echo",)
    if name == "se":
        return se(rp["tr_ms"], rp["te_ms"], rp["flip_deg"], selective=selective, sat=sat), ("echo",)
    if name == "ir":
        return ir(rp["tr_ms"], rp["te_ms"], rp["ti_ms"], rp["flip_deg"], inv_eff, selective=selective, sat=sat), ("echo",)
    if name == "bssfp":
        return bssfp(rp["tr_ms"], rp["flip_deg"], rp["phase_cycle_deg"], selective=selective), ("echo",)
    if sat is not None:
        raise ValueError(f"no saturation pulse in the block list of {name}")
    # ---- the sequences a console serves from the core (``scanner/served.py``)
    if name in ("fse", "mese"):                                     # the train: real crushers, the stimulated echoes kept
        n = rp["etl"] if name == "fse" else rp["n_echoes"]
        make = fse if name == "fse" else mese
        # ``refocus_deg``: one angle from the console; a realised protocol written by hand (the design layer) may
        # carry one angle per echo, which ``fse`` takes as it is
        return make(rp["tr_ms"], rp["echo_spacing_ms"], n, rp["refocus_deg"], excitation_deg=rp["flip_deg"], selective=selective), \
            tuple(("echo", i) for i in range(n))
    if name == "megre":                                             # the steady state of spgr; the echoes are read from it
        kw = dict(spoiling="rf", phase_inc_deg=rp["rf_phase_increment_deg"], spoiler_mt_ms_per_m=rp["spoiler_mt_ms_per_m"]) if tier == 2 else dict(spoiling="ideal")
        return dataclasses.replace(spgr(rp["tr_ms"], rp["flip_deg"], selective=selective, **kw), name="megre"), ("echo",)
    if name == "dwse":
        if not rp["b_value"]:                                       # no lobes: the spin echo between perfect crushers
            return dataclasses.replace(se(rp["tr_ms"], rp["te_ms"], rp["flip_deg"], selective=selective), name="dwse"), ("echo",)
        return dwse(rp["tr_ms"], rp["te_ms"], rp["diffusion_g_mt_m"], rp["diffusion_delta_ms"], rp["diffusion_separation_ms"], rp["diffusion_ramp_ms"],
                    rp["diffusion_start_ms"], flip_deg=rp["flip_deg"], selective=selective), ("echo",)
    raise ValueError(f"no block list for sequence {name!r}")


# ---------------------------------------------------------------------------------------------- echo trains
def fse(tr_ms, esp_ms, etl, refocus_deg=180.0, excitation_deg=90.0, crusher_cycles=1, ideal_crushers=False, selective=None, repetitions=1):
    """CPMG train: excitation about x, ``etl`` refocusing pulses about y spaced ``esp_ms`` apart, echo i labelled
    ("echo", i). ``refocus_deg``: one angle or one per echo (a variable flip-angle train). Equal crushers either side
    of every refocusing pulse (``ideal_crushers``: the ideal crushed pulse, which keeps the spin-echo pathway only).
    An ideal crusher ends the TR."""
    flips = tuple(float(v) for v in np.broadcast_to(np.asarray(refocus_deg, float), (int(etl),)))
    if tr_ms < etl * esp_ms:
        raise ValueError("the echo train is longer than TR")
    ref = _refocus(selective)
    if ideal_crushers:
        echo = [B.delay(esp_ms / 2), B.rf(phase_deg=90.0, crushed=True, selective=ref, flip_table="refocus"), B.delay(esp_ms / 2), B.adc("echo")]
    else:
        echo = [B.delay(esp_ms / 2), B.gradient(crusher_cycles), B.rf(phase_deg=90.0, selective=ref, flip_table="refocus"),
                B.gradient(crusher_cycles), B.delay(esp_ms / 2), B.adc("echo")]
    body = [B.rf(excitation_deg, 0.0, selective=selective), B.repeat(echo, etl, tables=dict(refocus=flips)), B.delay(tr_ms - etl * esp_ms), B.crush()]
    return B.sequence([B.repeat(body, repetitions)], name="fse")


def mese(tr_ms, esp_ms, n_echoes, refocus_deg=180.0, **kw):
    """Multi-echo spin echo for T2 mapping: the train of ``fse`` with every echo read as its own image. With real
    crushers and a refocusing angle below 180 degrees the echoes carry stimulated echoes and do not decay as
    exp(-TE / T2)."""
    return dataclasses.replace(fse(tr_ms, esp_ms, n_echoes, refocus_deg, **kw), name="mese")


def echo_of_line(lines, etl, esp_ms, te_eff_ms, order="linear", centre_line=None):
    """Which echo of an ``fse`` train fills every phase-encode line, and the effective echo time realised.

    The ``lines`` lines are split into ``etl`` bands of ceil(lines / etl) neighbouring lines (one line of every band
    per shot). "linear": the band of the centre line is filled by the echo nearest to ``te_eff_ms`` and the bands
    follow k-space monotonically, wrapping round; "centric": the first echo fills the centre band and later echoes
    alternate outwards. ``centre_line``: the index of the line through the centre of k-space among the ``lines``
    acquired ones (None: lines // 2; with partial Fourier it lies off the middle). Returns (echo index [lines],
    realised effective TE in ms, shots)."""
    lines, etl = int(lines), int(etl)
    shots = int(math.ceil(lines / etl))
    band = np.minimum(np.arange(lines) // shots, etl - 1)
    mid = lines // 2 if centre_line is None else int(centre_line)
    centre = int(band[mid])
    if order == "linear":
        target = int(np.clip(round(te_eff_ms / esp_ms) - 1, 0, etl - 1))
        rank = (np.arange(etl) - centre + target) % etl
    elif order == "centric":
        dist = np.arange(etl) - centre
        rank = np.empty(etl, int)
        rank[np.argsort(2 * np.abs(dist) + (dist < 0), kind="stable")] = np.arange(etl)
    else:
        raise ValueError("order is linear or centric")
    echo = rank[band]
    return echo, float((echo[mid] + 1) * esp_ms), shots


# ---------------------------------------------------------------------------------------------- gradient echoes
def megre(tr_ms, flip_deg, te_ms, selective=None, repetitions=1):
    """Several gradient echoes at the times ``te_ms`` after one pulse, ideal spoiler at the end of the TR; echo i is
    labelled ("echo", i). Rewound readouts: no net dephasing between the echoes."""
    te = [float(t) for t in te_ms]
    if any(b <= a for a, b in zip([0.0] + te, te)) or te[-1] > tr_ms:
        raise ValueError("echo times increase and end before TR")
    body, last = [B.rf(flip_deg, selective=selective)], 0.0
    for i, t in enumerate(te):
        body += [B.delay(t - last), B.adc(("echo", i), static_ms=t)]
        last = t
    body += [B.delay(tr_ms - last), B.spoil()]
    return B.sequence([B.repeat(body, repetitions)], name="megre")


def spgr_lobe(tr_ms, flip_deg, te_ms, phase_inc_deg=117.0, spoiler_cycles=1, spoiler_ms=1.0, cycle_mm=None, selective=None, repetitions=1):
    """RF-spoiled gradient echo without the ideal-spoiling assumption and with the gradients where a scanner plays
    them: pulse, TE, sample, a spoiler lobe of ``spoiler_cycles`` cycles lasting ``spoiler_ms``, rest of the TR.
    ``phase_inc_deg`` 0 is the gradient-spoiled steady state (FISP); ``cycle_mm`` switches diffusion on."""
    if te_ms + spoiler_ms > tr_ms:
        raise ValueError("TE + spoiler exceeds TR")
    body = [B.rf(flip_deg, selective=selective), B.delay(te_ms), B.adc("echo", static_ms=te_ms), B.gradient(spoiler_cycles, ms=spoiler_ms),
            B.delay(tr_ms - te_ms - spoiler_ms)]
    return B.sequence([B.repeat(body, repetitions, spoil_inc_deg=phase_inc_deg)], cycle_mm=cycle_mm, name="spgr-lobe")


def dess(tr_ms, flip_deg, cycle_mm=None, selective=None, repetitions=1):
    """Dual echo in the steady state: "fid" right after the pulse, "echo" right before the next one, one cycle of
    dephasing between them (spread over the TR)."""
    body = [B.rf(flip_deg, selective=selective), B.adc("fid"), B.gradient(1, ms=tr_ms), B.adc("echo")]
    return B.sequence([B.repeat(body, repetitions)], cycle_mm=cycle_mm, name="dess")


def mprage(ti_ms, n_pulses, esp_ms, flip_deg, td_ms, inv_eff=1.0, repetitions=1):
    """Magnetisation-prepared gradient echo: inversion, ``ti_ms`` to the first pulse, ``n_pulses`` pulses ``esp_ms``
    apart (ideal spoiler after each; sample ("line", i)), recovery ``td_ms``, ideal crusher. Cycle time
    ti + n esp + td."""
    train = [B.rf(flip_deg), B.adc("line"), B.delay(esp_ms), B.spoil()]
    body = [B.mz(-inv_eff), B.delay(ti_ms), B.repeat(train, n_pulses), B.delay(td_ms), B.crush()]
    return B.sequence([B.repeat(body, repetitions)], name="mprage")


def t2prep_module(te_prep_ms, n_refocus=1):
    """90 (x), ``n_refocus`` 180 (y) pulses evenly spaced, 90 (-x), ideal spoiler: Mz -> Mz exp(-te_prep / T2) for
    exact pulses on resonance."""
    tau = te_prep_ms / (2 * n_refocus)
    out = [B.rf(90.0, 0.0)]
    for _ in range(n_refocus):
        out += [B.delay(tau), B.rf(180.0, 90.0), B.delay(tau)]
    return out + [B.rf(90.0, 180.0), B.spoil()]


def t2prep(tr_ms, te_prep_ms, flip_deg=90.0, n_refocus=1, repetitions=1):
    """T2 preparation, one readout pulse right after it, recovery for the rest of the TR, ideal crusher."""
    body = t2prep_module(te_prep_ms, n_refocus) + [B.rf(flip_deg), B.adc("echo"), B.delay(tr_ms - te_prep_ms), B.crush()]
    return B.sequence([B.repeat(body, repetitions)], name="t2prep")


def satrec(tr_ms, ts_ms, flip_deg=90.0, sat_eff=1.0, repetitions=1):
    """Saturation recovery: saturation of efficiency ``sat_eff`` (Mz -> (1 - e) Mz, ideal spoiler), ``ts_ms``, one
    pulse, sample, rest of the TR, ideal spoiler."""
    body = [B.mz(1.0 - sat_eff), B.spoil(), B.delay(ts_ms), B.rf(flip_deg), B.adc("echo"), B.delay(tr_ms - ts_ms), B.spoil()]
    return B.sequence([B.repeat(body, repetitions)], name="satrec")


# ---------------------------------------------------------------------------------------------- diffusion
def diffusion_lobe(g_mt_m, delta_ms, ramp_ms):
    """A trapezoid lobe of plateau amplitude ``g_mt_m``: ramp up, flat until ``delta_ms`` after its start, ramp down
    (``delta_ms`` = ramp + flat, the usual delta). Returns dict(t, g: its vertices; area; duration; shape;
    cycle_mm: the length over which the lobe dephases by exactly one cycle)."""
    if not 0 <= ramp_ms <= delta_ms:
        raise ValueError("0 <= ramp_ms <= delta_ms")
    if ramp_ms > 0:
        t, g = np.array([0.0, ramp_ms, delta_ms, delta_ms + ramp_ms]), np.array([0.0, g_mt_m, g_mt_m, 0.0])
        if delta_ms == ramp_ms:
            t, g = t[[0, 1, 3]], g[[0, 1, 3]]
        area, dur, shape = B.lobe_shape(t, g)
    else:
        t, g = np.array([0.0, delta_ms]), np.array([g_mt_m, g_mt_m])
        area, dur, shape = g_mt_m * delta_ms, delta_ms, B.LINEAR
    return dict(t=t, g=g, area=area, duration=dur, shape=shape, cycle_mm=1.0 / (CYCLES_PER_MM_PER_MT_M_MS * abs(area)))


def dwse(tr_ms, te_ms, g_mt_m, delta_ms, big_delta_ms, ramp_ms=0.0, start_ms=0.0, repetitions=1, flip_deg=90.0, selective=None):
    """Stejskal-Tanner spin echo: excitation, a lobe starting ``start_ms`` later, the 180 at TE / 2, the same lobe
    starting ``big_delta_ms`` after the first, sample at TE, ideal crusher at the end of the TR. The cycle length of
    the sequence is the one over which a lobe dephases by one cycle, so the lobes are gradient blocks of 1 cycle with
    the shape moments of the trapezoid. ``waveform`` gives the same program for ``mrgym.design.gradients.b_value``.
    ``flip_deg``: the excitation angle, the refocusing pulse has twice that; ``selective``: the excitation pulse of a
    slice, the refocusing pulse then plays on a slice 1.5 times as thick (as ``se``)."""
    lobe = diffusion_lobe(g_mt_m, delta_ms, ramp_ms)
    dur = lobe["duration"]
    gaps = (start_ms, te_ms / 2 - start_ms - dur, start_ms + big_delta_ms - te_ms / 2, te_ms - start_ms - big_delta_ms - dur, tr_ms - te_ms)
    if min(gaps) < -1e-9:
        raise ValueError("the lobes do not fit: need start + duration <= TE / 2 <= start + Delta and start + Delta + duration <= TE <= TR")
    gaps = [max(0.0, v) for v in gaps]
    grad = B.gradient(1, ms=dur, shape=lobe["shape"])
    body = [B.rf(flip_deg, 0.0, selective=selective), B.delay(gaps[0]), grad, B.delay(gaps[1]),
            B.rf(2 * flip_deg, 90.0, selective=_refocus(selective)), B.delay(gaps[2]), grad, B.delay(gaps[3]),
            B.adc("echo"), B.delay(gaps[4]), B.crush()]
    return B.sequence([B.repeat(body, repetitions)], cycle_mm=lobe["cycle_mm"], name="dwse")


def steam(tr_ms, te_ms, tm_ms, g_mt_m, delta_ms, ramp_ms=0.0, flips_deg=(90.0, 90.0, 90.0), tm_crusher_cycles=2, repetitions=1):
    """Stimulated echo with diffusion weighting: pulse 1, a lobe ending at TE / 2, pulse 2, mixing time with a crusher
    on a second axis, pulse 3, the same lobe, sample TE / 2 after pulse 3; ideal crusher at the end of the TR. From
    lobe start to lobe start: Delta = TM + the duration of a lobe."""
    lobe = diffusion_lobe(g_mt_m, delta_ms, ramp_ms)
    dur = lobe["duration"]
    if dur > te_ms / 2 or te_ms + tm_ms > tr_ms:
        raise ValueError("the lobe must fit into TE / 2 and TE + TM into TR")
    grad = B.gradient((1, 0), ms=dur, shape=lobe["shape"])
    body = [B.rf(flips_deg[0], 0.0), B.delay(te_ms / 2 - dur), grad, B.rf(flips_deg[1], 0.0), B.gradient((0, tm_crusher_cycles), ms=tm_ms),
            B.rf(flips_deg[2], 0.0), grad, B.delay(te_ms / 2 - dur), B.adc("echo"), B.delay(tr_ms - te_ms - tm_ms), B.crush()]
    return B.sequence([B.repeat(body, repetitions)], cycle_mm=(lobe["cycle_mm"], lobe["cycle_mm"]), name="steam")


def diffusion_waveform(kind, te_ms, g_mt_m, delta_ms, big_delta_ms=None, tm_ms=None, ramp_ms=0.0, start_ms=0.0):
    """(t [ms], g [mT/m], refocusing times) of the two lobes of ``dwse`` (kind "se") or ``steam`` (kind "steam") as
    one waveform, for ``mrgym.design.gradients.b_value``: an independent b-value of the same program. For the
    stimulated echo the magnetisation is longitudinal during TM; the sign of the effective gradient changes once,
    which ``b_value`` takes as a refocusing time in the middle of TM."""
    lobe = diffusion_lobe(g_mt_m, delta_ms, ramp_ms)
    if kind == "se":
        first, second, flip = start_ms, start_ms + big_delta_ms, te_ms / 2
    elif kind == "steam":
        first = te_ms / 2 - lobe["duration"]
        second, flip = te_ms / 2 + tm_ms, te_ms / 2 + tm_ms / 2
    else:
        raise ValueError("kind is se or steam")
    eps = 1e-9                                                       # a rectangular lobe: edges of one nanosecond
    lt, lg = (lobe["t"], lobe["g"]) if ramp_ms > 0 else (np.array([0.0, eps, delta_ms, delta_ms + eps]), np.array([0.0, g_mt_m, g_mt_m, 0.0]))
    t = np.concatenate([[0.0] if first > 0 else [], first + lt, second + lt])
    g = np.concatenate([[0.0] if first > 0 else [], lg, lg])
    return t, g, (flip,)


# ---------------------------------------------------------------------------------------------- magnetisation transfer
def mt_spgr(tr_ms, flip_deg, mt_energy_ut2_ms, mt_offset_hz, gap_ms=2.0, spoiling="ideal", phase_inc_deg=117.0, spoiler_mt_ms_per_m=100.0,
            repetitions=1):
    """Gradient echo with a saturation pulse: an off-resonance pulse of the given energy (it acts on the bound pool
    only: its direct effect on free water is left out), ``gap_ms`` later the excitation and the sample, then the rest
    of the TR with an ideal spoiler (or, spoiling="rf", one cycle of dephasing and quadratic RF phase)."""
    head = [B.rf(energy_ut2_ms=mt_energy_ut2_ms, offset_hz=mt_offset_hz), B.delay(gap_ms), B.rf(flip_deg), B.adc("echo")]
    if spoiling == "ideal":
        return B.sequence([B.repeat(head + [B.delay(tr_ms - gap_ms), B.spoil()], repetitions)], name="mt-spgr")
    body = head + [B.gradient(1, ms=tr_ms - gap_ms)]
    return B.sequence([B.repeat(body, repetitions, spoil_inc_deg=phase_inc_deg)], cycle_mm=spoiler_cycle_mm(spoiler_mt_ms_per_m), name="mt-spgr-rf")


SERVED = dict(spgr=spgr, afi=afi, epi=epi, se=se, ir=ir, bssfp=bssfp)
NEW = dict(fse=fse, mese=mese, megre=megre, spgr_lobe=spgr_lobe, dess=dess, mprage=mprage, t2prep=t2prep, satrec=satrec, dwse=dwse, steam=steam,
           mt_spgr=mt_spgr)

__all__ = ["NEW", "REFOCUS_WIDTH", "SERVED", "afi", "bssfp", "dess", "diffusion_lobe", "diffusion_waveform", "dwse", "echo_of_line", "epi", "fse",
           "ir", "ir_gre", "megre", "mese", "mprage", "mt_spgr", "refocusing_pulse", "satrec", "se", "served", "spgr", "spgr_lobe", "spoiler_cycle_mm", "steam", "t2prep",
           "t2prep_module"]
