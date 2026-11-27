"""One general RF rotation: what an arbitrary B1(t) played under an arbitrary gradient G(t) does to a spin.

Input: complex B1 samples (uT) on a raster, the gradient on the same raster (mT/m, one to three axes), and for every
isochromat a position (mm), an off-resonance (Hz) and a B1 scale. Output: the Cayley-Klein parameters (a, b) of the
rotation (``rotate``, no relaxation), or the magnetisation after the pulse with T1 / T2 relaxation (``bloch``).

Conventions (the ones of ``mrgym.scanner.rf``; ``mrgym.design.validate`` rows D4, D5 compare the two)
  rotation   by the angle phi about the unit axis n, right-handed:
                 a = cos(phi / 2) - i n_z sin(phi / 2),     b = -i (n_x + i n_y) sin(phi / 2)
             and the spinor matrix is Q = [[a, -conj(b)], [b, conj(a)]]. Rotations compose as matrices:
             first (a1, b1), then (a2, b2) gives (a2 a1 - conj(b2) b1,  b2 a1 + conj(a2) b1).
  nutation   sample j of duration dt turns by  theta = 2 pi gamma |B1_j| scale dt  about the axis
             (cos p, sin p, 0), p = arg(B1_j). A real positive B1 turns Mz towards -y: Mxy = -i sin(theta).
  precession a spin with off-resonance df at position r under the gradient G turns by
                 psi = 2 pi (gamma G.r + df) dt
             about +z: a positive frequency gains positive phase, Mxy -> Mxy exp(+i psi)  (docs/physics/conventions.md,
             section 5). The sign of a gradient is therefore the sign of the frequency it gives a spin at positive r.
  magnetisation of a spin that starts at (Mxy, Mz), after the rotation (a, b):
                 Mxy' = conj(a)^2 Mxy - b^2 conj(Mxy) + 2 conj(a) b Mz
                 Mz'  = (|a|^2 - |b|^2) Mz - 2 Re(a b conj(Mxy))
             from equilibrium Mxy = 2 conj(a) b, Mz = |a|^2 - |b|^2, flip angle 2 asin|b|; a refocusing pulse between
             equal crushers keeps -b^2 conj(Mxy).
  time       sample j covers [t_j, t_j + dt_j]; the rotation is the one from the start of the first sample to the end
             of the last. Nothing is referred to the pulse centre: a rephasing lobe is a further sample with B1 = 0.

Two models of one sample
  ``split``      (default) half the precession, then the whole nutation as an instantaneous (hard) sub-pulse, then
                 half the precession. This is the model of ``scanner/rf.py`` (Pauly et al. 1991, the hard-pulse
                 approximation behind the Shinnar-Le Roux transform): for a constant precession per sample, a and b
                 are polynomials in exp(i psi). It is exact for a train of hard sub-pulses and second order in dt for
                 a B1 that is constant during the sample.
  ``piecewise``  B1 and gradient constant during the sample: one rotation about the effective field
                 (w1 cos p, w1 sin p, dw), by |w_eff| dt. Exact for a piecewise-constant waveform.
Row D3 of ``mrgym.design.validate`` shows the two converging to each other as dt^2.

Relaxation (``bloch``): symmetric (Strang) operator splitting of every sample,
    [precession psi / 2 and relaxation dt / 2]  ->  [rotation by the RF]  ->  [precession psi / 2 and relaxation dt / 2]
Precession about z and relaxation commute, so the two halves are exact; only the RF rotation and the relaxation are
split. The error of one sample is third order in dt and the error of a pulse second order (row D9, against an
independent integration of the Bloch equations).
"""
from __future__ import annotations

import numpy as np

from .units import RAD_PER_HZ_MS, RAD_PER_MT_M_MM_MS, RAD_PER_UT_MS

MODELS = ("split", "piecewise")


# ---------------------------------------------------------------------------------------------- spinor algebra
def compose(a1, b1, a2, b2):
    """The rotation (a1, b1) followed by (a2, b2)."""
    return a2 * a1 - np.conj(b2) * b1, b2 * a1 + np.conj(a2) * b1


def inverse(a, b):
    """The rotation that undoes (a, b)."""
    return np.conj(a), -b


def axis_angle(axis, angle):
    """(a, b) of the right-handed rotation by ``angle`` (rad) about ``axis`` (3 numbers, normalised here)."""
    n = np.asarray(axis, float)
    n = n / np.linalg.norm(n)
    s = np.sin(np.asarray(angle, float) / 2)
    return np.cos(np.asarray(angle, float) / 2) - 1j * n[2] * s, -1j * (n[0] + 1j * n[1]) * s


def matrix(a, b):
    """Real 3 x 3 rotation matrices [..., 3, 3] acting on (Mx, My, Mz) for Cayley-Klein parameters (a, b)."""
    a, b = np.broadcast_arrays(np.asarray(a, complex), np.asarray(b, complex))
    r = np.empty(a.shape + (3, 3))
    p, q, c = np.conj(a) ** 2, b * b, 2 * np.conj(a) * b       # Mxy' = p Mxy - q conj(Mxy) + c Mz
    r[..., 0, 0], r[..., 0, 1], r[..., 0, 2] = (p - q).real, -(p + q).imag, c.real
    r[..., 1, 0], r[..., 1, 1], r[..., 1, 2] = (p - q).imag, (p + q).real, c.imag
    ab = a * b                                                  # Mz' = (|a|^2 - |b|^2) Mz - 2 Re(ab conj(Mxy))
    r[..., 2, 0], r[..., 2, 1], r[..., 2, 2] = -2 * ab.real, -2 * ab.imag, np.abs(a) ** 2 - np.abs(b) ** 2
    return r


def apply(a, b, mxy=0.0, mz=1.0):
    """Magnetisation (Mxy, Mz) after the rotation (a, b) of a spin that starts at (``mxy``, ``mz``)."""
    mxy, mz = np.asarray(mxy, complex), np.asarray(mz, float)
    return (np.conj(a) ** 2 * mxy - b * b * np.conj(mxy) + 2 * np.conj(a) * b * mz,
            (np.abs(a) ** 2 - np.abs(b) ** 2) * mz - 2 * (a * b * np.conj(mxy)).real)


def flip_angle(b):
    """Flip angle (rad) of a rotation: the angle between +z and where the rotation takes it."""
    return 2 * np.arcsin(np.clip(np.abs(b), 0.0, 1.0))


def z_rotation(a, b, angle):
    """(a, b) followed by a rotation by ``angle`` (rad) about +z, e.g. a rephasing lobe: angle = 2 pi gamma A.r."""
    e = np.exp(-0.5j * np.asarray(angle, float))
    return e * a, np.conj(e) * b


# ---------------------------------------------------------------------------------------------- inputs
def _samples(b1_ut, dt_ms, grad_mt_m):
    b1 = np.asarray(b1_ut, complex)
    if b1.ndim != 1 or b1.size == 0:
        raise ValueError("b1_ut must be a one-dimensional array of at least one sample")
    n = b1.size
    dt = np.broadcast_to(np.asarray(dt_ms, float), (n,))
    if not np.all(dt > 0) or not np.all(np.isfinite(dt)):
        raise ValueError("dt_ms must be positive")
    if not np.all(np.isfinite(b1)):
        raise ValueError("b1_ut must be finite")
    if grad_mt_m is None:
        g = np.zeros((n, 1))
    else:
        g = np.asarray(grad_mt_m, float)
        if g.ndim == 0:
            g = np.full((n, 1), float(g))
        elif g.ndim == 1:
            if g.size != n:
                raise ValueError(f"the gradient has {g.size} samples, the pulse {n}")
            g = g[:, None]
        elif g.ndim != 2 or g.shape[0] != n or not 1 <= g.shape[1] <= 3:
            raise ValueError("grad_mt_m must be a number, [n] or [n, axes] with 1 to 3 axes")
        if not np.all(np.isfinite(g)):
            raise ValueError("grad_mt_m must be finite")
    return b1, dt, g


def _isochromats(g, pos_mm, df_hz, b1_scale):
    """Positions as [m, axes]; off-resonance and B1 scale as [m]; the common shape of the three inputs."""
    axes = g.shape[1]
    pos = np.zeros(()) if pos_mm is None else np.asarray(pos_mm, float)
    if axes == 1:
        pos = pos[..., None]                                    # one axis: positions have the shape of the isochromats
    elif pos_mm is None:
        pos = np.zeros(axes)
    elif pos.ndim == 0 or pos.shape[-1] != axes:
        raise ValueError(f"with {axes} gradient axes the last dimension of pos_mm must have length {axes}")
    df, sc = np.asarray(df_hz, float), np.asarray(b1_scale, float)
    shape = np.broadcast_shapes(pos.shape[:-1], df.shape, sc.shape)
    pos = np.broadcast_to(pos, shape + (axes,)).reshape(-1, axes)
    return pos, np.broadcast_to(df, shape).ravel(), np.broadcast_to(sc, shape).ravel(), shape


def _precession(g, dt, pos, df):
    """Constant-gradient case: (None, psi [m]); otherwise (psi_fn(j) -> [m], None). psi = precession per sample."""
    const = np.all(g == g[0]) and np.all(dt == dt[0])
    if const:
        return RAD_PER_MT_M_MM_MS * dt[0] * (pos @ g[0]) + RAD_PER_HZ_MS * dt[0] * df
    kz = RAD_PER_MT_M_MM_MS * (pos @ (g * dt[:, None]).T)      # [m, n]
    return kz + RAD_PER_HZ_MS * df[:, None] * dt[None, :]


# ---------------------------------------------------------------------------------------------- the rotation
def rotate(b1_ut, dt_ms, grad_mt_m=None, pos_mm=None, df_hz=0.0, b1_scale=1.0, model="split", normalise=True):
    """Cayley-Klein parameters (a, b) of the pulse for every isochromat. No relaxation.

    ``b1_ut``      complex [n], microtesla.
    ``dt_ms``      duration of a sample: a number or [n].
    ``grad_mt_m``  None (no gradient), a number, [n] (one axis) or [n, axes] with up to three axes.
    ``pos_mm``     positions: shape S for one axis, S + (axes,) otherwise.
    ``df_hz``, ``b1_scale``  broadcast against the positions.
    Returns two complex arrays of the common shape S. A product of n unitary steps drifts from |a|^2 + |b|^2 = 1 by
    rounding (about 1e-13 after 512 samples, row D8); ``normalise`` divides the result by sqrt(|a|^2 + |b|^2).
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    b1, dt, g = _samples(b1_ut, dt_ms, grad_mt_m)
    pos, df, sc, shape = _isochromats(g, pos_mm, df_hz, b1_scale)
    m = pos.shape[0]
    a, b = np.ones(m, complex), np.zeros(m, complex)
    if m == 0:
        return a.reshape(shape), b.reshape(shape)
    psi = _precession(g, dt, pos, df)
    varying = psi.ndim == 2
    theta = RAD_PER_UT_MS * np.abs(b1) * dt                    # nominal nutation per sample (scale 1)
    phase = np.exp(1j * np.angle(b1))
    uniform_scale = bool(np.all(sc == sc.flat[0])) if sc.size else True
    if model == "split":
        if uniform_scale:
            cs, sn = np.cos(0.5 * sc.flat[0] * theta), -1j * phase * np.sin(0.5 * sc.flat[0] * theta)
        half = np.exp(-0.25j * (psi[:, 0] if varying else psi))
        a *= half
        full = None if varying else half * half
        for j in range(b1.size):
            if uniform_scale:
                c, s = cs[j], sn[j]
            else:
                c, s = np.cos(0.5 * sc * theta[j]), -1j * phase[j] * np.sin(0.5 * sc * theta[j])
            a, b = c * a - np.conj(s) * b, s * a + c * b
            if j + 1 < b1.size:                                 # the two half steps between sub-pulses, as one
                e = np.exp(-0.25j * (psi[:, j] + psi[:, j + 1])) if varying else full
            else:
                e = np.exp(-0.25j * psi[:, j]) if varying else half
            a, b = e * a, np.conj(e) * b
    else:
        for j in range(b1.size):
            w1 = sc * theta[j]                                  # rad per sample about the B1 axis
            dz = psi[:, j] if varying else psi                  # rad per sample about z
            w = np.sqrt(w1 * w1 + dz * dz)
            half = 0.5 * w
            sinc = np.where(w > 0, np.sin(half) / np.where(w > 0, w, 1.0), 0.5)      # sin(w / 2) / w
            aj = np.cos(half) - 1j * dz * sinc
            bj = -1j * phase[j] * w1 * sinc
            a, b = aj * a - np.conj(bj) * b, bj * a + np.conj(aj) * b
    if normalise:
        norm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
        a, b = a / norm, b / norm
    return a.reshape(shape), b.reshape(shape)


def bloch(b1_ut, dt_ms, grad_mt_m=None, pos_mm=None, df_hz=0.0, b1_scale=1.0, t1_ms=np.inf, t2_ms=np.inf,
          mxy0=0.0, mz0=1.0, m_eq=1.0, model="split"):
    """Magnetisation (Mxy complex, Mz) after the pulse, with relaxation towards ``m_eq`` during the pulse.

    Inputs as ``rotate``; ``t1_ms``, ``t2_ms``, ``mxy0``, ``mz0``, ``m_eq`` broadcast against the isochromats (``np.inf`` = no
    relaxation, and then the result equals ``apply(*rotate(...))``). Splitting of one sample: half of the precession
    and of the relaxation, the RF rotation, the other half (see the module docstring).
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    b1, dt, g = _samples(b1_ut, dt_ms, grad_mt_m)
    t1, t2 = np.asarray(t1_ms, float), np.asarray(t2_ms, float)
    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("t1_ms and t2_ms must be positive")
    extra = np.broadcast_shapes(t1.shape, t2.shape, np.shape(mxy0), np.shape(mz0), np.shape(m_eq))
    pos, df, sc, shape = _isochromats(g, pos_mm, np.broadcast_to(np.asarray(df_hz, float), np.broadcast_shapes(np.shape(df_hz), extra)), b1_scale)
    t1, t2 = np.broadcast_to(t1, shape).ravel(), np.broadcast_to(t2, shape).ravel()
    mxy = np.array(np.broadcast_to(np.asarray(mxy0, complex), shape).ravel())
    mz = np.array(np.broadcast_to(np.asarray(mz0, float), shape).ravel())
    m_eq = np.broadcast_to(np.asarray(m_eq, float), shape).ravel()
    psi = _precession(g, dt, pos, df)
    varying = psi.ndim == 2
    theta = RAD_PER_UT_MS * np.abs(b1) * dt
    phase = np.exp(1j * np.angle(b1))
    same_dt = bool(np.all(dt == dt[0]))
    if same_dt:
        e1h, e2h = np.exp(-0.5 * dt[0] / t1), np.exp(-0.5 * dt[0] / t2)
    zh_const = None if varying else np.exp(0.5j * psi)
    for j in range(b1.size):
        if not same_dt:
            e1h, e2h = np.exp(-0.5 * dt[j] / t1), np.exp(-0.5 * dt[j] / t2)
        w1 = sc * theta[j]
        if model == "split":                                    # hard sub-pulse between two halves of the precession
            zh = np.exp(0.5j * psi[:, j]) if varying else zh_const
            aj, bj, f = np.cos(0.5 * w1), -1j * phase[j] * np.sin(0.5 * w1), e2h * zh
        else:                                                   # rotation about the effective field: it holds the
            dz = psi[:, j] if varying else psi                  # precession, the halves hold the relaxation only
            w = np.sqrt(w1 * w1 + dz * dz)
            sinc = np.where(w > 0, np.sin(0.5 * w) / np.where(w > 0, w, 1.0), 0.5)
            aj, bj, f = np.cos(0.5 * w) - 1j * dz * sinc, -1j * phase[j] * w1 * sinc, e2h
        mxy, mz = f * mxy, m_eq + (mz - m_eq) * e1h
        mxy, mz = (np.conj(aj) ** 2 * mxy - bj * bj * np.conj(mxy) + 2 * np.conj(aj) * bj * mz,
                   (np.abs(aj) ** 2 - np.abs(bj) ** 2) * mz - 2 * (aj * bj * np.conj(mxy)).real)
        mxy, mz = f * mxy, m_eq + (mz - m_eq) * e1h
    return mxy.reshape(shape), mz.reshape(shape)
