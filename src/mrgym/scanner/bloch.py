"""Bloch calculations written separately from the engine. Used by ``mrgym validate`` and the tests only; nothing
here feeds the scanner.

Three methods that share no code with ``signal`` or ``epg``:
A. Matrix steady state under ideal spoiling. Every sequence is a periodic chain of affine maps M -> A M + b on
   (Mx, My, Mz): hard-pulse rotation, relaxation, ideal spoiler (Mx = My = 0), inversion. The periodic state solves
   (I - A_cycle) M = b_cycle.
B. Isochromat ensemble of RF-spoiled gradient echo: N spins with dephasing angles 2 pi m / N per raster interval,
   real 3 x 3 rotations about the pulse axis, quadratic pulse phase, time stepping from equilibrium.
C. Plain time-stepping EPG in the laboratory frame (explicit pulse phases), the textbook 30-line version.
D. Slice-selective pulses as a cascade of real 3 x 3 rotations (hard sub-pulse about x, precession about z under the
   slice gradient), and the RF-spoiled gradient echo of a slice as an ensemble of isochromats over (position through
   the slice) x (spoiler phase), each turned by the cascade of its position.
"""
import numpy as np


# ------------------------------------------------------------------ A. affine operators, batched over voxels [V]
def rf_x(angle):
    """Rotation about x by ``angle`` [V] (rad): My' = My cos + Mz sin, Mz' = -My sin + Mz cos."""
    v = len(angle)
    a = np.zeros((v, 3, 3))
    c, s = np.cos(angle), np.sin(angle)
    a[:, 0, 0] = 1
    a[:, 1, 1], a[:, 1, 2], a[:, 2, 1], a[:, 2, 2] = c, s, -s, c
    return a, np.zeros((v, 3))


def free(t, t1, t2):
    v = len(t1)
    e1, e2 = np.exp(-t / t1), np.exp(-t / t2)
    a = np.zeros((v, 3, 3))
    a[:, 0, 0], a[:, 1, 1], a[:, 2, 2] = e2, e2, e1
    b = np.zeros((v, 3))
    b[:, 2] = 1 - e1
    return a, b


def spoil(v):
    a = np.zeros((v, 3, 3))
    a[:, 2, 2] = 1
    return a, np.zeros((v, 3))


def invert(v, eff=1.0):
    """Inversion of efficiency ``eff`` (Mz -> -eff Mz) followed by a spoiler."""
    a = np.zeros((v, 3, 3))
    a[:, 2, 2] = -eff
    return a, np.zeros((v, 3))


def chain(ops):
    a, b = ops[0]
    for a2, b2 in ops[1:]:
        a, b = a2 @ a, np.einsum("vij,vj->vi", a2, b) + b2
    return a, b


def apply(op, m):
    return np.einsum("vij,vj->vi", op[0], m) + op[1]


def periodic(cycle):
    a, b = chain(cycle)
    return np.linalg.solve(np.eye(3)[None] - a, b[..., None])[..., 0]


def brute(cycle, n):
    v = cycle[0][0].shape[0]
    m = np.zeros((v, 3))
    m[:, 2] = 1.0
    op = chain(cycle)
    for _ in range(n):
        m = apply(op, m)
    return m


def spgr_ideal(a, tr, t1, n_brute=0):
    """Signal at TE = 0 (My right after the pulse). ``a`` [V] actual flip in rad; ideal spoiler each TR."""
    v, t2 = len(t1), np.full(len(t1), 50.0)
    cyc = [rf_x(a), free(tr, t1, t2), spoil(v)]
    m = brute(cyc, n_brute) if n_brute else periodic(cyc)
    return apply(rf_x(a), m)[:, 1]


def afi_ideal(a, tr1, tr2, t1, n_brute=0):
    v, t2 = len(t1), np.full(len(t1), 50.0)
    first = [rf_x(a), free(tr1, t1, t2), spoil(v)]
    cyc = first + [rf_x(a), free(tr2, t1, t2), spoil(v)]      # starts just before the pulse that follows TR2
    m = brute(cyc, n_brute) if n_brute else periodic(cyc)
    return apply(rf_x(a), m)[:, 1], apply(rf_x(a), apply(chain(first), m))[:, 1]


def ir_ideal(a_read, ti, tr, t1, inv_eff=1.0, n_brute=0):
    """Inversion (efficiency inv_eff) - TI - readout pulse a_read - (TR - TI) - repeat; signed signal at TE = 0."""
    v, t2 = len(t1), np.full(len(t1), 50.0)
    cyc = [invert(v, inv_eff), free(ti, t1, t2), rf_x(a_read), free(tr - ti, t1, t2), spoil(v)]
    m = brute(cyc, n_brute) if n_brute else periodic(cyc)
    return apply(rf_x(a_read), apply(chain([invert(v, inv_eff), free(ti, t1, t2)]), m))[:, 1]


# ------------------------------------------------------------------ B. isochromat ensemble, RF-spoiled
def rf_spoiled_isochromats(a, tr_ms, t1_ms, t2_ms, phi_deg=117.0, pattern=(1,), n_iso=2048, n_periods=None):
    """Echo amplitudes (TE = 0, demodulated with the pulse phase) of the last period, one per pulsed raster position.

    Each isochromat precesses by 2 pi m / n_iso per raster interval (one spoiler cycle across the voxel). n_iso must
    exceed twice the number of intervals a coherence survives (about 14 T2 / TR for 1e-6); it is the caller's job.
    """
    p = len(pattern)
    if n_periods is None:
        n_periods = int(np.ceil(12 * t1_ms / (tr_ms * p))) + 20
    theta = 2 * np.pi * np.arange(n_iso) / n_iso
    cth, sth = np.cos(theta), np.sin(theta)
    e1, e2 = np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms)
    mx, my, mz = np.zeros(n_iso), np.zeros(n_iso), np.ones(n_iso)
    phi, out = np.deg2rad(phi_deg), np.zeros(p, complex)
    for j in range(n_periods * p):
        fl = a * pattern[j % p]
        ph = phi * j * (j + 1) / 2
        if fl:
            # right-handed rotation by fl about the axis (cos ph, sin ph, 0): Mz -> transverse -i exp(i ph) sin(fl)
            nx, ny = np.cos(ph), np.sin(ph)
            c, s = np.cos(fl), np.sin(fl)
            dot = nx * mx + ny * my
            mx, my, mz = (mx * c + ny * mz * s + nx * dot * (1 - c),
                          my * c - nx * mz * s + ny * dot * (1 - c),
                          mz * c + (nx * my - ny * mx) * s)
            if j >= (n_periods - 1) * p:
                out[j % p] = 1j * np.mean(mx + 1j * my) * np.exp(-1j * ph)
        mx, my, mz = e2 * (mx * cth - my * sth), e2 * (mx * sth + my * cth), 1 + (mz - 1) * e1
    return out[[i for i in range(p) if pattern[i]]]


# ------------------------------------------------------------------ C. time-stepping EPG, laboratory frame
def rf_spoiled_epg_stepping(a, tr_ms, t1_ms, t2_ms, phi_deg=117.0, pattern=(1,), n_periods=None, k_states=None, bd=0.0):
    """``bd`` = D * dk^2 * TR adds diffusion with the constant-gradient b-factors of Weigel 2010 (the same formula as
    the engine, applied here in the laboratory frame, so it checks the engine's frame change, not the formula)."""
    p = len(pattern)
    if n_periods is None:
        n_periods = int(np.ceil(12 * t1_ms / (tr_ms * p))) + 20
    if k_states is None:
        k_states = int(np.ceil(16 * t2_ms / tr_ms)) + 4
    e1, e2 = np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms)
    fp, fm, z = np.zeros(k_states, complex), np.zeros(k_states, complex), np.zeros(k_states, complex)
    z[0] = 1.0
    k = np.arange(k_states)
    d_l, d_up, d_dn = np.exp(-bd * k * k), np.exp(-bd * (k * k + k + 1 / 3)), np.exp(-bd * (k * k - k + 1 / 3))
    phi, out = np.deg2rad(phi_deg), np.zeros(p, complex)
    for j in range(n_periods * p):
        fl = a * pattern[j % p]
        ph = phi * j * (j + 1) / 2
        if fl:
            c2, s2, s, c = np.cos(fl / 2) ** 2, np.sin(fl / 2) ** 2, np.sin(fl), np.cos(fl)
            fp, fm, z = (c2 * fp + np.exp(2j * ph) * s2 * fm - 1j * np.exp(1j * ph) * s * z,
                         np.exp(-2j * ph) * s2 * fp + c2 * fm + 1j * np.exp(-1j * ph) * s * z,
                         -0.5j * np.exp(-1j * ph) * s * fp + 0.5j * np.exp(1j * ph) * s * fm + c * z)
            if j >= (n_periods - 1) * p:
                out[j % p] = 1j * fp[0] * np.exp(-1j * ph)
        fp, fm, z = fp * e2 * d_up, fm * e2 * d_dn, z * e1 * d_l
        z[0] += 1 - e1
        fp = np.roll(fp, 1)
        fm = np.roll(fm, -1)
        fm[-1] = 0
        fp[0] = np.conj(fm[0])
    return out[[i for i in range(p) if pattern[i]]]


# ------------------------------------------------------------------ D. slice-selective pulse, 3 x 3 rotations
def _sinc_samples(tbw, n, window="hann"):
    """Sub-pulse amplitudes of a windowed sinc, summing to 1 (written here again on purpose: no import from ``rf``)."""
    t = (np.arange(n) + 0.5) / n - 0.5
    env = np.where(np.abs(tbw * t) < 1e-12, 1.0, np.sin(np.pi * tbw * t) / (np.pi * tbw * t + 1e-300))
    env = env * ((0.5 + 0.5 * np.cos(2 * np.pi * t)) if window == "hann" else (0.54 + 0.46 * np.cos(2 * np.pi * t)))
    return env / env.sum()


def slice_rotation(alpha, tbw, zeta, n=128, rephase=0.5, window="hann"):
    """Rotation matrices [len(zeta), 3, 3] of a slice-selective pulse of centre flip angle ``alpha`` (rad) at
    positions ``zeta`` (z / slice thickness): n sub-pulses about x, each between two half-steps of precession about z
    (2 pi tbw zeta / n per sample), then a rephasing lobe of ``rephase`` times the selection area.
    Conventions as in ``rf_x``: a positive angle about x takes Mz to +My; precession by phi takes Mx to Mx cos - My sin."""
    zeta = np.asarray(zeta, float)
    dphi = 2 * np.pi * tbw * zeta / n

    def rz(phi):
        m = np.zeros((zeta.size, 3, 3))
        m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1], m[:, 2, 2] = np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi), 1.0
        return m

    half, total = rz(dphi / 2), np.broadcast_to(np.eye(3), (zeta.size, 3, 3)).copy()
    for w in _sinc_samples(tbw, n, window) * alpha:
        rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(w), np.sin(w)], [0.0, -np.sin(w), np.cos(w)]])
        total = half @ (rx[None] @ (half @ total))
    return rz(-rephase * dphi * n) @ total


def slice_profile(alpha, tbw, zeta, n=128, rephase=0.5, window="hann"):
    """(flip angle theta(z) in rad, complex Mxy(z) from equilibrium) of the cascade."""
    m = slice_rotation(alpha, tbw, zeta, n, rephase, window)[:, :, 2]            # image of (0, 0, 1)
    return np.arccos(np.clip(m[:, 2], -1.0, 1.0)), m[:, 0] + 1j * m[:, 1]


def slice_spgr_isochromats(alpha, tbw, tr_ms, t1_ms, t2_ms, phi_deg=117.0, n_z=161, z_max=2.0, n_iso=256, n_tr=None, window="hann"):
    """RF-spoiled gradient echo of a slice: complex signal per unit slice thickness right after the pulse (TE = 0),
    demodulated with the pulse phase and referred to a hard pulse about the same axis (+1 x sin(alpha) for an ideal
    rectangular slice and full recovery).

    Isochromats over ``n_z`` positions on +- ``z_max`` thicknesses times ``n_iso`` spoiler phases; every repetition
    applies the position's 3 x 3 pulse rotation about an axis turned by the quadratic RF phase, then one spoiler
    cycle of in-plane dephasing and T1 / T2 relaxation. Time stepping from equilibrium; no relaxation during the pulse."""
    if n_tr is None:
        n_tr = int(np.ceil(12 * t1_ms / tr_ms)) + 20
    zeta = np.linspace(-z_max, z_max, n_z)
    rot = slice_rotation(alpha, tbw, zeta, window=window)                        # [z, 3, 3]
    th = 2 * np.pi * np.arange(n_iso) / n_iso
    cth, sth = np.cos(th)[None, :], np.sin(th)[None, :]
    e1, e2 = np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms)
    m = np.zeros((3, n_z, n_iso))
    m[2] = 1.0
    phi, sig = np.deg2rad(phi_deg), 0j
    for j in range(n_tr):
        ph = phi * j * (j + 1) / 2
        c, s = np.cos(ph), np.sin(ph)
        mx, my = c * m[0] + s * m[1], -s * m[0] + c * m[1]                       # into the frame of the pulse
        v = np.einsum("zij,jzk->izk", rot, np.stack([mx, my, m[2]]))
        m = np.stack([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])            # and back
        if j == n_tr - 1:
            w = np.full(n_z, zeta[1] - zeta[0])
            w[[0, -1]] /= 2
            sig = np.sum(w * (m[0] + 1j * m[1]).mean(1)) * np.exp(-1j * ph) / 1j   # Mxy = +i sin(alpha) at the centre
        m = np.stack([e2 * (m[0] * cth - m[1] * sth), e2 * (m[0] * sth + m[1] * cth), 1 + (m[2] - 1) * e1])
    return sig


def slice_se_isochromats(alpha, tbw, tr_ms, te_ms, t1_ms, ti_ms=None, inv_eff=1.0, width=1.5, n_z=None, n_iso=48, n_tr=None):
    """Spin echo (``ti_ms`` given: inversion recovery with an adiabatic, non-selective inversion of efficiency
    ``inv_eff``) of a slice: echo per unit slice thickness, without T2 decay, referred to +1 for ideal pulses, a
    rectangular slice and full recovery (the sign convention of ``signal.se``).

    Excitation of centre angle ``alpha`` (with its rephasing lobe), refocusing pulse of centre angle 2 alpha on a
    slice ``width`` times as thick, between two equal crushers; isochromats over positions through the slice times
    ``n_iso`` crusher phases; a spoiler removes the transverse magnetisation after the echo. Time stepping."""
    z_max = 0.5 + 12.0 / tbw
    n_z = n_z or int(round(2 * z_max * 12 * tbw)) + 1
    zeta = np.linspace(-z_max, z_max, n_z)
    r_ex = slice_rotation(alpha, tbw, zeta)
    r_ref = slice_rotation(2 * alpha, tbw, zeta / width, rephase=0.0)
    th = 2 * np.pi * np.arange(n_iso) / n_iso
    cth, sth = np.cos(th)[None, :], np.sin(th)[None, :]
    if n_tr is None:
        n_tr = int(np.ceil(10 * t1_ms / tr_ms)) + 6

    def relax(mz, t):
        e = np.exp(-t / t1_ms)
        return 1 + (mz - 1) * e

    def crush(m):
        return np.stack([m[0] * cth - m[1] * sth, m[0] * sth + m[1] * cth, m[2]])

    mz, sig = np.ones(n_z), 0j
    for _ in range(n_tr):
        if ti_ms is not None:
            mz = relax(-inv_eff * mz, ti_ms)
        m = np.einsum("zij,j...->iz...", r_ex, np.array([0.0, 0.0, 1.0])) * mz              # [3, z]
        m = np.repeat(m[:, :, None], n_iso, axis=2)
        m[2] = relax(m[2], te_ms / 2)
        m = crush(np.einsum("zij,jzk->izk", r_ref, crush(m)))
        w = np.full(n_z, zeta[1] - zeta[0])
        w[[0, -1]] /= 2
        sig = np.sum(w * (m[0] + 1j * m[1]).mean(1)) / -1j           # both pulses about x: the echo forms along -My; reported as +1
        mz = relax(m[2].mean(1), tr_ms - te_ms / 2 - (ti_ms or 0.0))
    return sig
