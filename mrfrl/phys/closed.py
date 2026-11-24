"""Tier 1: closed-form signals. Every function states the idealisation under which it is exact.

All take the ACTUAL flip angle ``a`` in rad (nominal angle times the local B1+ ratio), times in ms, and return the
transverse magnetisation per unit M0 at TE = 0. Arguments broadcast.
"""
import numpy as np


def t2star_ms(t2_ms, t2p_ms):
    """1/T2* = 1/T2 + 1/T2'. T2 is irreversible, T2' the reversible (static, Lorentzian) part."""
    return 1.0 / (1.0 / np.asarray(t2_ms, float) + 1.0 / np.asarray(t2p_ms, float))


def spgr(a, tr_ms, t1_ms):
    """Spoiled gradient echo, IDEAL spoiling (no transverse magnetisation survives a repetition); Ernst formula.

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


def ir(a_read, ti_ms, tr_ms, t1_ms, inv_eff=1.0):
    """Inversion recovery, one readout pulse per inversion; signed signal.

    Cycle (period TR): inversion Mz -> -inv_eff * Mz, delay TI, readout pulse of actual angle ``a_read``, delay
    TR - TI. Idealisations: no transverse magnetisation survives to the next pulse (crushers; TR - TI >> T2), the
    inversion acts on Mz only. The readout leaves Mz cos(a_read) behind, so with k = inv_eff:

    S = sin a (1 - (1 + k) E_TI + k E_TR) / (1 + k cos a E_TR),  E_x = exp(-x / T1)

    (The draft's S = sin a (1 - 2 E_TI + E_TR) silently assumed Mz = 0 after the readout, i.e. a = 90 deg exactly.)
    """
    eti, etr = np.exp(-np.asarray(ti_ms, float) / t1_ms), np.exp(-np.asarray(tr_ms, float) / t1_ms)
    return np.sin(a_read) * (1 - (1 + inv_eff) * eti + inv_eff * etr) / (1 + inv_eff * np.cos(a_read) * etr)


def echo(te_ms, t2_ms, t2p_ms, b0_hz=0.0):
    """Gradient-echo factor at echo time TE: exp(-TE / T2*) * exp(+2 pi i * dB0 * TE). Multi-echo: pass an array of TE.

    Sign convention of this package: a spin that precesses ``b0_hz`` FASTER than the receiver reference gains
    POSITIVE phase with time.
    """
    te = np.asarray(te_ms, float)
    return np.exp(-te / t2star_ms(t2_ms, t2p_ms)) * np.exp(2j * np.pi * np.asarray(b0_hz, float) * te * 1e-3)
