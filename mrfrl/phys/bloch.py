"""Independent Bloch calculations. Used by the consistency suite only; nothing here feeds the scanner.

Three methods that share no code with ``closed`` or ``epg``:
A. Matrix steady state under ideal spoiling. Every sequence is a periodic chain of affine maps M -> A M + b on
   (Mx, My, Mz): hard-pulse rotation, relaxation, ideal spoiler (Mx = My = 0), inversion. The periodic state solves
   (I - A_cycle) M = b_cycle.
B. Isochromat ensemble of RF-spoiled gradient echo: N spins with dephasing angles 2 pi m / N per raster interval,
   real 3 x 3 rotations about the pulse axis, quadratic pulse phase, time stepping from equilibrium.
C. Plain time-stepping EPG in the laboratory frame (explicit pulse phases), the textbook 30-line version.
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
