"""Signal models of the console's sequences, per tissue class. Every function states the idealisation it makes.

All take the ACTUAL flip angle ``a`` in rad (nominal angle times the local B1+ ratio) and times in ms, and return
transverse magnetisation per unit M0 at the echo WITHOUT the echo-time decay (that part is ``time_function``, because
it continues through the readout). Arguments broadcast.

  spgr      spoiled gradient echo with IDEAL spoiling: the Ernst formula (Ernst & Anderson 1966). The scanner serves
            the EPG steady state with RF spoiling, finite T2 and diffusion instead (``epg``; Weigel 2015)
  afi       actual flip-angle imaging, dual TR, IDEAL spoiling (Yarnykh 2007)
  se        spin echo 90 - 180 with crushers: longitudinal steady state of the two-pulse cycle, refocusing efficiency
            sin^2(beta / 2) (Hennig 1988; Bernstein, King & Zhou 2004, ch. 14)
  ir_se     inversion-recovery spin echo (FLAIR, STIR): the same chain with an inversion of efficiency kappa
  bssfp     balanced SSFP steady state with off-resonance (Freeman & Hill 1971; Scheffler & Lehnhardt 2003), echo at TR / 2
  epi       single-shot gradient-echo EPI: one excitation per TR, so the Ernst steady state; all the EPI physics is
            in the encoding (``kspace``)
"""
from __future__ import annotations

import numpy as np

GAMMA_MHZ_PER_T = 42.577478518


def t2star_ms(t2_ms, t2p_ms):
    """1/T2* = 1/T2 + 1/T2'. T2 is irreversible, T2' the reversible (static, Lorentzian) part."""
    return 1.0 / (1.0 / np.asarray(t2_ms, float) + 1.0 / np.asarray(t2p_ms, float))


def mz_steady(events, t1_ms):
    """Periodic longitudinal magnetisation just before the first event of a cycle of ('rf', factor) and ('wait', ms)
    events, with every transverse component crushed. Mz -> factor * Mz at a pulse; Mz -> 1 + (Mz - 1) E during a wait."""
    a, b = 1.0, 0.0
    for kind, v in events:
        if kind == "rf":
            a, b = a * v, b * v
        else:
            e = np.exp(-v / t1_ms)
            a, b = a * e, b * e + (1 - e)
    return b / (1 - a)


def spgr(a, tr_ms, t1_ms):
    """Spoiled gradient echo, IDEAL spoiling (no transverse magnetisation survives a repetition).

    S = sin a (1 - E1) / (1 - E1 cos a),  E1 = exp(-TR / T1)
    """
    e1 = np.exp(-np.asarray(tr_ms, float) / t1_ms)
    return np.sin(a) * (1 - e1) / (1 - e1 * np.cos(a))


def afi(a, tr1_ms, tr2_ms, t1_ms):
    """Actual flip-angle imaging (dual-TR spoiled gradient echo, same pulse), IDEAL spoiling. Returns (S1, S2).

    S1 is the echo in the short interval TR1 (after the pulse that follows the long delay TR2), S2 the echo in TR2.
    S1 = sin a (1 - E2 + (1 - E1) E2 cos a) / (1 - E1 E2 cos^2 a),  S2: E1 <-> E2
    """
    e1, e2 = np.exp(-np.asarray(tr1_ms, float) / t1_ms), np.exp(-np.asarray(tr2_ms, float) / t1_ms)
    c = np.cos(a)
    den = 1 - e1 * e2 * c * c
    return (np.sin(a) * (1 - e2 + (1 - e1) * e2 * c) / den,
            np.sin(a) * (1 - e1 + (1 - e2) * e1 * c) / den)


def ir_gre(a_read, ti_ms, tr_ms, t1_ms, inv_eff=1.0):
    """Inversion recovery with one gradient-echo readout pulse per inversion; signed signal.

    Cycle (period TR): inversion Mz -> -inv_eff * Mz, delay TI, readout pulse of actual angle ``a_read``, delay
    TR - TI. Idealisations: no transverse magnetisation survives to the next pulse, the inversion acts on Mz only.
    The readout leaves Mz cos(a_read) behind, so with k = inv_eff:

    S = sin a (1 - (1 + k) E_TI + k E_TR) / (1 + k cos a E_TR),  E_x = exp(-x / T1)
    """
    eti, etr = np.exp(-np.asarray(ti_ms, float) / t1_ms), np.exp(-np.asarray(tr_ms, float) / t1_ms)
    return np.sin(a_read) * (1 - (1 + inv_eff) * eti + inv_eff * etr) / (1 + inv_eff * np.cos(a_read) * etr)


def se(a, tr_ms, te_ms, t1_ms):
    """Spin echo with excitation a and refocusing 2a (both scale with B1+). Ideal pulses: 1 - 2 E(TR - TE/2) + E(TR)."""
    beta = 2 * a
    mz = mz_steady([("rf", np.cos(a)), ("wait", te_ms / 2), ("rf", np.cos(beta)), ("wait", tr_ms - te_ms / 2)], t1_ms)
    return mz * np.sin(a) * np.sin(beta / 2) ** 2


def ir_se(a, tr_ms, te_ms, ti_ms, t1_ms, inv_eff=1.0):
    """Inversion (adiabatic, efficiency kappa) - TI - spin echo. Signed."""
    beta = 2 * a
    mz = mz_steady([("rf", np.cos(a)), ("wait", te_ms / 2), ("rf", np.cos(beta)), ("wait", tr_ms - ti_ms - te_ms / 2),
                    ("rf", -inv_eff), ("wait", ti_ms)], t1_ms)
    return mz * np.sin(a) * np.sin(beta / 2) ** 2


def bssfp(a, theta, tr_ms, t1_ms, t2_ms):
    """Balanced SSFP, complex transverse magnetisation right after the pulse, in the frame of that pulse.

    ``theta`` (rad) = precession per TR in the frame that follows the RF phase cycle: 2 pi df TR + (phase increment).
    With the usual 180 degree increment the pass band is centred on df = 0 and the nulls sit at df = (n + 1/2) / TR.
        M+ = M0 (1 - E1) sin a (1 - E2 exp(-i theta)) / (C cos theta + D)
        C = E2 (E1 - 1)(1 + cos a),  D = (1 - E1 cos a) - (E1 - cos a) E2^2
    Convention: demodulated with the pulse phase, real and positive at the centre of the pass band. The echo at
    TE = TR / 2 carries exp(-TE / T2) exp(i theta_free / 2), applied by the caller.
    """
    e1, e2 = np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms)
    c = e2 * (e1 - 1) * (1 + np.cos(a))
    d = (1 - e1 * np.cos(a)) - (e1 - np.cos(a)) * e2 * e2
    return (1 - e1) * np.sin(a) * (1 - e2 * np.exp(-1j * theta)) / (c * np.cos(theta) + d)


def fat_spectrum(t_s, b0_t, peaks):
    """Sum_p alpha_p exp(2 pi i f_p t) of a fat species; f_p = ppm * gamma * B0."""
    f0 = GAMMA_MHZ_PER_T * b0_t
    return sum(amp * np.exp(2j * np.pi * ppm * f0 * t_s) for ppm, amp in peaks) / sum(amp for _, amp in peaks)


def time_function(cls, tau_s, tau_spec_s, te_ms, echo, b0_t, fat_peaks):
    """Factor of one tissue class at time ``tau`` relative to the echo centre (array, s).

    echo 'gre': exp(-(TE + tau) / T2*) * spectrum(TE + tau)            (decay and chemical-shift phase from the pulse)
    echo 'se' : exp(-(TE + tau) / T2) * exp(-|tau| / T2') * spectrum(tau)   (static dephasing refocused at the echo)
    ``tau_spec_s`` is the time used for the chemical-shift evolution (the same array; zeros switch the shift off).
    """
    t_ms = np.maximum(te_ms + 1e3 * tau_s, 0.0)                 # samples before the pulse do not exist
    if echo == "gre":
        f = np.exp(-t_ms / cls["t2s_ms"])
        return f * fat_spectrum(1e-3 * te_ms + tau_spec_s, b0_t, fat_peaks) if cls["fat"] else f.astype(complex)
    f = np.exp(-t_ms / cls["t2_ms"]) * np.exp(-np.abs(1e3 * tau_s) / cls["t2p_ms"])
    return f * fat_spectrum(tau_spec_s, b0_t, fat_peaks) if cls["fat"] else f.astype(complex)
