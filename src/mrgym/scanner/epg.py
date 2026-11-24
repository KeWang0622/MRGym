"""Tier 2: steady state of RF-spoiled gradient echo with finite T2, by extended phase graphs (EPG).

Sequence model
- Hard pulses on a regular raster of period ``tr_ms``. The pulse at raster position j has flip ``a * pattern[j % P]``
  and phase  phi_j = phi * j (j + 1) / 2  (quadratic RF spoiling; raster positions without a pulse still count).
- Every raster interval carries the same spoiler moment: one unit of dephasing (>= one cycle across the voxel).
- ``pattern = (1,)`` is ordinary spoiled gradient echo. ``pattern = (1, 1, 0, ..., 0)`` of length n + 1 is actual
  flip-angle imaging with TR2 = n * TR1 and spoiler moments and RF phase increments in the ratio TR1 : TR2
  (the equidistant "virtual pulse" scheme; Nehrke 2009).
- Relaxation T1, T2 during every interval. Optional isotropic diffusion through ``bd`` = D * dk^2 * TR
  (dimensionless; dk = spoiler moment in rad/m). No T2': a static off-resonance distribution multiplies the state
  of order k by a factor that is 1 for k = 0 at the pulse, so it only enters as exp(-TE / T2') at the echo. For the
  same reason a uniform off-resonance does not change the k = 0 amplitude.

Method. States F_k (k >= 0), F_-k^* and Z_k as in Weigel 2015. Substituting
    F_k = exp(i phi_j) exp(-i k j phi) G_k,   Z_k = exp(-i k j phi) W_k
removes the pulse phase from the RF operator and makes the interval operator independent of j:
    G_k'  = exp(i (k-1) phi) E2 G_(k-1)+      (k >= 1)
    G_-k' = exp(-i (k+1) phi) E2 G_-(k+1)+     (stored as conj: Gm_k' = exp(i (k+1) phi) E2 Gm_(k+1)+)
    W_k'  = exp(i k phi) (E1 W_k+ + (1 - E1) [k = 0])
so the periodic state is the solution of one banded real linear system (no time stepping, no convergence question).
The echo right after pulse j, demodulated with phi_j, is i * G_0+ (real and equal to the Ernst value when T2 -> 0).
An order-k state has spent at least k intervals in the transverse plane, so amplitudes fall as E2^k and the
truncation K with E2^K <= tol is controlled.

Checked by ``mrgym validate`` against plain time-stepping EPG and an isochromat ensemble (``bloch.py``), and against
the SSFP-FID closed form for a phase increment of zero.
"""
import functools

import numpy as np
import scipy.linalg
import scipy.sparse as sp


def _realify(n, lin, anti=None):
    """Real 2n x 2n sparse matrix of  x -> L x + C conj(x)  from complex COO arrays (rows, cols, values)."""
    r, c, v = [], [], []
    for triple, sign in ((lin, 1.0), (anti, -1.0)):
        if triple is None:
            continue
        rows, cols, vals = (np.concatenate([np.atleast_1d(x) for x in part]) for part in triple)
        a, b = vals.real, vals.imag
        r += [2 * rows, 2 * rows, 2 * rows + 1, 2 * rows + 1]
        c += [2 * cols, 2 * cols + 1, 2 * cols, 2 * cols + 1]
        v += [a, -sign * b, b, sign * a]
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(2 * n, 2 * n))


def _interval(k_states, e1, e2, phi, bd):
    """Relaxation + diffusion + spoiler shift + frame phase of one raster interval. Returns (S, b)."""
    k = np.arange(k_states)
    d_l = np.exp(-bd * k * k)
    d_up = np.exp(-bd * (k * k + k + 1 / 3))            # order k -> k + 1
    d_dn = np.exp(-bd * (k * k - k + 1 / 3))            # order -k -> -k + 1
    lin = ([3 * k[1:], 3 * k[:-1] + 1, 3 * k + 2],
           [3 * (k[1:] - 1), 3 * (k[:-1] + 1) + 1, 3 * k + 2],
           [np.exp(1j * phi * (k[1:] - 1)) * e2 * d_up[:-1], np.exp(1j * phi * (k[:-1] + 1)) * e2 * d_dn[1:],
            np.exp(1j * phi * k) * e1 * d_l])
    anti = ([0], [4], [np.exp(-1j * phi) * e2 * d_dn[1]])             # F_0' = conj(Gm_0')
    b = np.zeros(6 * k_states)
    b[4] = 1 - e1                                         # Re W_0
    return _realify(3 * k_states, lin, anti), b


def _rf(k_states, a):
    k = 3 * np.arange(k_states)
    c2, s2, s, c = np.cos(a / 2) ** 2, np.sin(a / 2) ** 2, np.sin(a), np.cos(a)
    blk = np.array([[c2, s2, -1j * s], [s2, c2, 1j * s], [-0.5j * s, 0.5j * s, c]])
    ij = [(i, j) for i in range(3) for j in range(3)]
    return _realify(3 * k_states, ([k + i for i, _ in ij], [k + j for _, j in ij],
                                   [np.full(k_states, blk[i, j]) for i, j in ij]))


def _solve_banded(m, rhs):
    coo = m.tocoo()
    lo, up = int(max(0, (coo.row - coo.col).max())), int(max(0, (coo.col - coo.row).max()))
    ab = np.zeros((lo + up + 1, m.shape[0]))
    np.add.at(ab, (up + coo.row - coo.col, coo.col), coo.data)
    return scipy.linalg.solve_banded((lo, up), ab, rhs)


def n_states(tr_ms, t2_ms, tol=1e-6, k_cap=1500):
    return int(min(k_cap, max(8, np.ceil(-np.log(tol) * t2_ms / tr_ms) + 2)))


def steady_state(a, tr_ms, t1_ms, t2_ms, phi_deg=117.0, pattern=(1,), bd=0.0, tol=1e-6):
    """Complex echo amplitudes (TE = 0, per unit M0) right after every pulsed raster position of one period.

    ``a``: actual flip angle in rad (scalar). Returns an array with one entry per non-zero element of ``pattern``.
    """
    kk = n_states(tr_ms, t2_ms, tol)
    phi = np.deg2rad(phi_deg)
    s_op, b = _interval(kk, np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms), phi, bd)
    rfs = {p: _rf(kk, a * p) for p in set(pattern) if p}
    steps = [(s_op @ rfs[p]) if p else s_op for p in pattern]
    m, c = sp.identity(6 * kk, format="csr"), np.zeros(6 * kk)
    for a_j in steps:
        m, c = a_j @ m, a_j @ c + b
    u = _solve_banded(sp.identity(6 * kk, format="csr") - m, c)
    out = []
    for p, a_j in zip(pattern, steps):
        if p:
            v = rfs[p] @ u
            out.append(1j * (v[0] + 1j * v[1]))
        u = a_j @ u + b
    return np.array(out)


@functools.lru_cache(maxsize=200000)
def _cached(a, tr_ms, t1_ms, t2_ms, phi_deg, pattern, bd):
    return steady_state(a, tr_ms, t1_ms, t2_ms, phi_deg, pattern, bd)


def spgr(a, tr_ms, t1_ms, t2_ms, phi_deg=117.0, bd=0.0):
    """RF-spoiled gradient echo; ``a`` may be an array. Complex."""
    a = np.asarray(a, float)
    flat = [_cached(float(x), float(tr_ms), float(t1_ms), float(t2_ms), float(phi_deg), (1,), float(bd))[0]
            for x in a.ravel()]
    return np.array(flat).reshape(a.shape)


def afi(a, tr1_ms, n, t1_ms, t2_ms, phi_deg=117.0, bd=0.0):
    """Dual-TR RF-spoiled gradient echo with TR2 = n * TR1 (integer n >= 2). Returns (S1, S2), complex."""
    a = np.asarray(a, float)
    pattern = (1, 1) + (0,) * (int(n) - 1)
    flat = np.array([_cached(float(x), float(tr1_ms), float(t1_ms), float(t2_ms), float(phi_deg), pattern, float(bd))
                     for x in a.ravel()])
    return flat[:, 0].reshape(a.shape), flat[:, 1].reshape(a.shape)
