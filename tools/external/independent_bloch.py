"""A Bloch integrator written from the equations, for checking the scanner from outside.

It imports numpy and nothing from the package. Everything is an ensemble of isochromats (Mx, My, Mz) moved by three
operators, in the rotating frame:

    rotate    right-handed rotation by an angle about an axis in the transverse plane (a hard pulse, or one sample of
              a shaped pulse); from equilibrium it leaves  Mx + i My = -i exp(i phase) sin(angle)
    precess   rotation about z (off-resonance, a gradient): Mx + i My -> (Mx + i My) exp(i angle)
    relax     Mx, My -> E2 Mx, E2 My;  Mz -> 1 + (Mz - 1) E1      (operator splitting: exact between pulses)

A dephasing gradient is a set of isochromats with precession angles spread evenly over one turn (the spoiler of a
gradient echo, the crushers of a spin echo); a slice is a set of isochromats at positions through the slice, each
precessing under the slice gradient while the shaped pulse plays, sample by sample.

Sequences
    spoiled_gre     RF-spoiled gradient echo and its dual-TR variant (actual flip-angle imaging), hard pulses
    slice_gre       the same through a slice: the windowed-sinc pulse stepped under its slice gradient for every
                    isochromat, a rephasing lobe of half the selection area, optional relaxation during the pulse
    slice_se        spin echo / inversion recovery through a slice: shaped 90 and 180, crushers, spoiler
    bssfp           balanced SSFP with off-resonance, hard pulses
    ir_null         inversion time that nulls the signal of an inversion-recovery spin echo
    slice_rotation_parameters   Cayley-Klein parameters of the shaped pulse from its 3 x 3 rotation

The pulse is the one the instrument datasheet states: 128 samples of a Hann-windowed sinc with a time-bandwidth
product, the samples summing to the flip angle at the slice centre.

Signals are complex and demodulated with the phase of the pulse that made them; a single pulse from equilibrium
gives +sin(angle).
"""
import numpy as np


# ------------------------------------------------------------------------------------------------ the three operators
def rotate(m, angle, phase=0.0):
    c, s, nx, ny = np.cos(angle), np.sin(angle), np.cos(phase), np.sin(phase)
    along = nx * m[0] + ny * m[1]
    return np.stack([m[0] * c + ny * m[2] * s + nx * along * (1 - c),
                     m[1] * c - nx * m[2] * s + ny * along * (1 - c),
                     m[2] * c + (nx * m[1] - ny * m[0]) * s])


def precess(m, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([m[0] * c - m[1] * s, m[0] * s + m[1] * c, m[2]])


def relax(m, t_ms, t1_ms, t2_ms):
    e1, e2 = np.exp(-t_ms / t1_ms), (np.exp(-t_ms / t2_ms) if np.isfinite(t2_ms) else 1.0)
    return np.stack([m[0] * e2, m[1] * e2, 1 + (m[2] - 1) * e1])


def equilibrium(shape):
    m = np.zeros((3,) + tuple(shape))
    m[2] = 1.0
    return m


def demodulated(m, phase):
    """Mean transverse magnetisation, referred to a pulse of the given phase: +sin(angle) for one pulse."""
    return 1j * np.exp(-1j * phase) * (m[0] + 1j * m[1])


# ------------------------------------------------------------------------------------------------ the shaped pulse
def windowed_sinc(tbw, n=128):
    t = (np.arange(n) + 0.5) / n - 0.5
    x = np.pi * tbw * t
    b = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(np.abs(x) < 1e-12, 1.0, x)) * (0.5 + 0.5 * np.cos(2 * np.pi * t))
    return b / b.sum()


def shaped_pulse(m, alpha, phase, tbw, zeta, rephase=0.5, n=128, sample_ms=0.0, t1_ms=np.inf, t2_ms=np.inf):
    """Play the pulse on isochromats at positions ``zeta`` (z / slice thickness; broadcast against m[0]): every sample
    is a small rotation between two half-steps of precession under the slice gradient (2 pi tbw zeta / n per sample),
    then the rephasing lobe. ``sample_ms`` > 0 adds relaxation during every sample."""
    step = 2 * np.pi * tbw * zeta / n
    for b in windowed_sinc(tbw, n) * alpha:
        m = precess(m, step / 2)
        m = rotate(m, b, phase)
        m = precess(m, step / 2)
        if sample_ms:
            m = relax(m, sample_ms, t1_ms, t2_ms)
    return precess(m, -rephase * n * step)


def slice_rotation_parameters(alpha, tbw, zeta, rephase=0.5, n=128):
    """The rotation of the pulse at ``zeta`` as (a^2, b^2, conj(a) b) of its Cayley-Klein parameters, read off its
    action on the three unit vectors: with M+ = Mx + i My a rotation gives
    M+' = conj(a)^2 M+ - b^2 conj(M+) + 2 conj(a) b Mz   (a pulse about x: a = cos(alpha / 2), b = -i sin(alpha / 2))."""
    zeta = np.asarray(zeta, float)
    plus = []
    for k in range(3):
        m = np.zeros((3, zeta.size))
        m[k] = 1.0
        m = shaped_pulse(m, alpha, 0.0, tbw, zeta, rephase, n)
        plus.append(m[0] + 1j * m[1])
    return np.conj(0.5 * (plus[0] - 1j * plus[1])), -0.5 * (plus[0] + 1j * plus[1]), 0.5 * plus[2]


# ------------------------------------------------------------------------------------------------ sequences
def spoiled_gre(alpha, tr_ms, t1_ms, t2_ms, phi_deg=117.0, pattern=(1,), n_iso=2048, n_periods=None):
    """RF-spoiled gradient echo, hard pulses. ``pattern`` gives the pulse amplitude (in units of alpha) at every
    position of a raster of period ``tr_ms``; (1,) is ordinary spoiled gradient echo, (1, 1, 0, 0, 0, 0) the dual-TR
    scheme with TR2 = 5 TR1. The pulse at raster position j has phase phi j (j + 1) / 2, and every raster interval
    one full turn of spoiler dephasing. Returns the echo right after each pulse of the last period."""
    p = len(pattern)
    if n_periods is None:
        n_periods = int(np.ceil(14 * t1_ms / (tr_ms * p))) + 20
    spoil = 2 * np.pi * np.arange(n_iso) / n_iso
    m, phi, out = equilibrium((n_iso,)), np.deg2rad(phi_deg), {}
    for j in range(n_periods * p):
        if pattern[j % p]:
            ph = phi * j * (j + 1) / 2
            m = rotate(m, alpha * pattern[j % p], ph)
            out[j % p] = demodulated(m, ph).mean()
        m = relax(precess(m, spoil), tr_ms, t1_ms, t2_ms)
    return np.array([out[k] for k in sorted(out)])


def slice_gre(alpha, tbw, tr_ms, t1_ms, t2_ms, phi_deg=117.0, z_max=None, per_thickness=None, n_iso=96, n_tr=None, rf_ms=0.0):
    """RF-spoiled gradient echo of a slice: signal per unit slice thickness right after the rephasing lobe, extrapolated
    to the centre of the pulse. Isochromats over positions on +- z_max thicknesses x spoiler phases; the shaped pulse
    is stepped for all of them in every repetition. ``rf_ms`` > 0: the pulse has that duration and the spins relax
    while it plays (the scanner's model has none: 0)."""
    z_max = z_max or 0.5 + 12.0 / tbw
    n_z = int(round(2 * z_max * (per_thickness or 6 * tbw))) + 1
    zeta = np.linspace(-z_max, z_max, n_z)[:, None]
    w = np.full(n_z, 2 * z_max / (n_z - 1))
    w[[0, -1]] /= 2
    spoil = (2 * np.pi * np.arange(n_iso) / n_iso)[None, :]
    if n_tr is None:
        n_tr = int(np.ceil(10 * t1_ms / tr_ms)) + 10
    m, phi, sig = equilibrium((n_z, n_iso)), np.deg2rad(phi_deg), 0j
    for j in range(n_tr):
        ph = phi * j * (j + 1) / 2
        m = shaped_pulse(m, alpha, ph, tbw, zeta, sample_ms=rf_ms / 128, t1_ms=t1_ms, t2_ms=t2_ms)
        sig = np.sum(w * demodulated(m, ph).mean(1)) * np.exp(0.5 * rf_ms / t2_ms)
        m = relax(precess(m, spoil), tr_ms - rf_ms, t1_ms, t2_ms)
    return sig


def slice_se(alpha, tbw, tr_ms, te_ms, t1_ms, ti_ms=None, inv_eff=1.0, width=1.5, n_iso=32, n_tr=None, gratings=False):
    """Spin echo of a slice (``ti_ms``: preceded by a non-selective inversion of efficiency ``inv_eff``): excitation
    ``alpha`` about x, refocusing 2 alpha about x on a slice ``width`` times as thick between two equal crushers, a
    spoiler after the echo. Echo per unit slice thickness, without T2 decay; the echo of two pulses about the same
    axis forms on the opposite side of the excitation, and is returned positive.

    An imperfect refocusing pulse stores part of the crushed transverse magnetisation along z, as a grating over
    the crusher phase. ``gratings`` False (the scanner's stated model: every transverse component is crushed for
    good) replaces Mz by its mean over the crusher phases after the echo. True keeps the grating until the next
    repetition, where the same crushers bring part of it back (a stimulated-echo pathway)."""
    z_max = 0.5 + 12.0 / tbw
    n_z = int(round(2 * z_max * 6 * tbw)) + 1
    zeta = np.linspace(-z_max, z_max, n_z)[:, None]
    w = np.full(n_z, 2 * z_max / (n_z - 1))
    w[[0, -1]] /= 2
    crush = (2 * np.pi * np.arange(n_iso) / n_iso)[None, :]
    if n_tr is None:
        n_tr = int(np.ceil(10 * t1_ms / tr_ms)) + 6
    m, sig = equilibrium((n_z, n_iso)), 0j
    for _ in range(n_tr):
        if ti_ms is not None:
            m[2] *= -inv_eff
            m = relax(m, ti_ms, t1_ms, np.inf)
        m = shaped_pulse(m, alpha, 0.0, tbw, zeta)
        m = precess(relax(m, te_ms / 2, t1_ms, np.inf), crush)
        m = precess(shaped_pulse(m, 2 * alpha, 0.0, tbw, zeta / width, rephase=0.0), crush)
        sig = -np.sum(w * demodulated(m, 0.0).mean(1))
        m[0], m[1] = 0.0, 0.0                                           # spoiler
        if not gratings:
            m[2] = m[2].mean(1, keepdims=True)
        m = relax(m, tr_ms - te_ms / 2 - (ti_ms or 0.0), t1_ms, np.inf)
    return sig


def bssfp(alpha, tr_ms, t1_ms, t2_ms, precession, phase_cycle_deg=180.0, n_tr=None):
    """Balanced SSFP, hard pulses: transverse magnetisation right after the pulse in the frame of that pulse, for
    precession angles per TR ``precession`` (rad, array)."""
    precession = np.asarray(precession, float)
    if n_tr is None:
        n_tr = int(np.ceil(14 * t1_ms / tr_ms))
    m, step, out = equilibrium(precession.shape), np.deg2rad(phase_cycle_deg), None
    for j in range(n_tr):
        m = rotate(m, alpha, j * step)
        out = demodulated(m, j * step)
        m = relax(precess(m, precession), tr_ms, t1_ms, t2_ms)
    return out


def ir_signal(ti_ms, tr_ms, te_ms, t1_ms, inv_eff=1.0, alpha=np.pi / 2, n_tr=40):
    """Signed signal of inversion - TI - 90 - TE/2 - 180 - echo with hard pulses and ideal crushers (no T2 decay)."""
    mz = 1.0
    for _ in range(n_tr):
        mz = 1 + (-inv_eff * mz - 1) * np.exp(-ti_ms / t1_ms)
        sig = mz * np.sin(alpha) * np.sin(alpha) ** 2
        mz = 1 + (mz * np.cos(alpha) - 1) * np.exp(-te_ms / 2 / t1_ms)
        mz = 1 + (mz * np.cos(2 * alpha) - 1) * np.exp(-(tr_ms - ti_ms - te_ms / 2) / t1_ms)
    return sig


def ir_null(tr_ms, te_ms, t1_ms, inv_eff=1.0):
    """Inversion time at which ``ir_signal`` changes sign (bisection)."""
    lo, hi = 1.0, tr_ms - te_ms - 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ir_signal(mid, tr_ms, te_ms, t1_ms, inv_eff) < 0 else (lo, mid)
    return 0.5 * (lo + hi)
