"""The reference evaluator: an ensemble of isochromats that integrates the Bloch equation block by block.

Every column of spins is an ensemble of isochromats at sub-voxel positions x (in cycle lengths, on a regular grid
through one cycle per gradient axis) and, optionally, at a set of static off-resonances (``spread``: T2' as an
ensemble). Each isochromat is a real vector (Mx, My, Mz). A block acts on it exactly:
  rf        a real 3 x 3 rotation: Rodrigues' formula about (cos p, sin p, 0) for a hard pulse; the rotation matrix
            of the Cayley-Klein parameters for a selective or designed pulse
  free      rotation about z by 2 pi (n . x + df t), decay exp(-t / T2), recovery towards M0 with exp(-t / T1); with
            two pools the matrix exponential of the exchange generator (``scipy.linalg.expm``)
  spoil     Mx = My = 0;   crush  also Mz -> its mean over the positions;   mz  Mz x factor
  adc       the mean of Mx + i My over the positions (weighted over the off-resonances)
Nothing here uses Fourier states, so the agreement with ``epg`` tests the phase-graph algebra. For whole cycles the
mean over N equidistant positions equals F_0 exactly when N exceeds the largest order present; the default N is the
reach of the gradients + 1.

Diffusion (``diffusion_substeps`` > 0): isochromats do not diffuse, so the Bloch-Torrey equation is integrated on the
position grid instead. Diffusion is a Gaussian blur along x, applied in the Fourier domain of the grid; while a
gradient is on, the pattern is the periodic pattern of the grid times the part of the block's dephasing played so
far, exp(2 pi i f x), so the component q of the grid decays with the wavenumber q + f. Every free block is split
into substeps and f is advanced between two half steps of blur (the trapezoid rule in time): second order in the
step. It checks the closed-form diffusion factors of ``epg`` without using them.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg

from ..design import rotation as design_rotation
from . import blocks as B
from . import operators as O
from .epg import Result


def rodrigues(axis, angle):
    """Right-handed rotation matrices [..., 3, 3] by ``angle`` (array) about the unit vectors ``axis`` [..., 3]."""
    n = np.asarray(axis, float)
    ang = np.asarray(angle, float)
    c, s = np.cos(ang)[..., None, None], np.sin(ang)[..., None, None]
    cross = np.zeros(np.broadcast_shapes(n.shape[:-1], ang.shape) + (3, 3))
    cross[..., 0, 1], cross[..., 0, 2] = -n[..., 2], n[..., 1]
    cross[..., 1, 0], cross[..., 1, 2] = n[..., 2], -n[..., 0]
    cross[..., 2, 0], cross[..., 2, 1] = -n[..., 1], n[..., 0]
    return c * np.eye(3) + s * cross + (1 - c) * (n[..., :, None] * n[..., None, :])


def positions(n_sub):
    """Sub-voxel positions [points, axes] in cycle lengths: the midpoints of ``n_sub[i]`` equal cells of [-1/2, 1/2)."""
    axes = [(np.arange(n) + 0.5) / n - 0.5 for n in n_sub]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], 1)


def run(seq, sp, n_sub=None, spread=None, diffusion_substeps=0, demodulation=1j, start=None):
    """Integrate the sequence for the columns ``sp`` from equilibrium (or from ``start`` = the ``state`` of an
    earlier result). ``n_sub``: isochromats per gradient axis (a number or one per axis). ``spread``: (x, w) of
    ``operators.lorentzian_spread``: every isochromat is repeated at the off-resonances x / (2 pi T2') of its column;
    the adc then does not apply the ``static_ms`` factor (the ensemble is that dephasing).
    Returns ``epg.Result`` whose ``state`` is dict(m [3, points, offsets, columns], mb, x [points, axes])."""
    prog = seq if isinstance(seq, B.Program) else B.compile_ops(seq, frame=False)
    axes, unit = prog.axes, prog.unit
    rate = O.diffusion_rate(prog, sp) if diffusion_substeps else None
    if rate is not None and (sp.pool is not None or any(u != 1 for u in unit)):
        raise ValueError("diffusion in the isochromat ensemble: one pool and whole cycles only")
    if n_sub is None:
        n_sub = tuple((2 * r + 2 if rate is not None else r + 1) if u == 1 else max(64, 4 * r + 1) for r, u in zip(prog.reach, unit))
    n_sub = tuple(int(v) for v in (n_sub if isinstance(n_sub, (tuple, list)) else (n_sub,) * axes))
    x = positions(n_sub)                                                # [P, axes]
    npos = x.shape[0]
    if spread is None:
        off, wgt = np.zeros((1, sp.n)), np.ones(1)
    else:
        with np.errstate(divide="ignore"):
            width = np.where(np.isfinite(sp.t2p_ms), 1e3 / (2 * math.pi * sp.t2p_ms), 0.0)
        off, wgt = np.asarray(spread[0], float)[:, None] * width[None, :], np.asarray(spread[1], float)
    df = sp.df_hz[None, :] + off                                        # [S, columns]
    nspr = df.shape[0]
    two, cx = sp.pool is not None, sp.pool == "cx"
    if start is None:
        m = np.zeros((3, npos, nspr, sp.n))
        m[2] = sp.m0
        mb = None
        if two:
            mb = np.zeros((3, npos, nspr, sp.n))
            mb[2] = sp.m0 * sp.f_b
    else:
        m, mb = start["m"].copy(), None if start.get("mb") is None else start["mb"].copy()
    r1, r2 = 1.0 / sp.t1_ms, 1.0 / sp.t2_ms
    if two:
        kf = sp.k_ab_per_s * 1e-3
        kr, r1b = kf / sp.f_b, 1.0 / sp.t1b_ms
        gen_z = np.zeros((sp.n, 2, 2))
        gen_z[:, 0, 0], gen_z[:, 0, 1], gen_z[:, 1, 0], gen_z[:, 1, 1] = -r1 - kf, kr, kf, -r1b - kr
        m_inf = np.stack([sp.m0, sp.m0 * sp.f_b], 1)                    # [columns, 2]
        if cx:
            r2b, wb = 1.0 / sp.t2b_ms, 2 * math.pi * 1e-3 * sp.shift_b_hz
            gen_t = np.zeros((sp.n, 4, 4))                              # (Mxa, Mya, Mxb, Myb); the shift of b relative to a
            gen_t[:, 0, 0] = gen_t[:, 1, 1] = -r2 - kf
            gen_t[:, 2, 2] = gen_t[:, 3, 3] = -r2b - kr
            gen_t[:, 0, 2] = gen_t[:, 1, 3] = kr
            gen_t[:, 2, 0] = gen_t[:, 3, 1] = kf
            gen_t[:, 2, 3], gen_t[:, 3, 2] = -wb, wb
    qs = None
    if rate is not None:                                                # integer spatial frequencies of the position grid
        qs = [np.fft.fftfreq(n, 1.0 / n).reshape([n if j == i else 1 for j in range(axes)] + [1, 1]) for i, n in enumerate(n_sub)]
    out, cache = [], {}

    def turn(v, ang):
        """Rotate the transverse part of ``v`` [3, ...] about z by ``ang``."""
        c, s = np.cos(ang), np.sin(ang)
        return c * v[0] - s * v[1], s * v[0] + c * v[1]

    def blur(v, t, frac):
        """Free diffusion for ``t`` ms of the pattern ``v`` [points, offsets, columns] x exp(2 pi i frac . x): the
        Fourier component q of ``v`` has the wavenumber q + frac and decays as exp(-D kappa^2 (q + frac)^2 t)."""
        spec = np.fft.fftn(v.reshape(n_sub + v.shape[1:]), axes=tuple(range(axes)))
        spec *= np.exp(-t * sum(rate[i] * (qs[i] + frac[i]) ** 2 for i in range(axes)))
        return np.fft.ifftn(spec, axes=tuple(range(axes))).reshape(v.shape)

    for op in prog.ops:
        kind = op.kind
        if kind == "rf":
            sat = O.saturation(op, sp, cache)
            if sat is not None:
                mb[2] *= sat
            if op.block.offset_hz:
                continue
            blk = op.block
            if blk.crushed:
                a, b = O.rotation(op, sp)
                for v in ((m, mb) if cx else (m,)):
                    mxy = -(b * b) * (v[0] - 1j * v[1])
                    v[0], v[1], v[2] = mxy.real, mxy.imag, (np.abs(a) ** 2 - np.abs(b) ** 2) * v[2]
                continue
            if blk.ck is None and blk.selective is None:                # hard pulse: axis and angle, no spinors
                p = math.radians(op.phase_deg)
                rot = rodrigues(np.array([math.cos(p), math.sin(p), 0.0]), math.radians(op.flip_deg) * sp.b1)
            elif blk.selective is not None and blk.selective.model == "flip":
                _, b = O.selective_rotation(blk.selective, math.radians(op.flip_deg) * sp.b1, sp.zeta)
                p = math.radians(op.phase_deg)
                rot = rodrigues(np.array([math.cos(p), math.sin(p), 0.0]), 2 * np.arcsin(np.clip(np.abs(b), 0.0, 1.0)))
            else:
                rot = design_rotation.matrix(*O.rotation(op, sp))
            m = np.einsum("cij,jpsc->ipsc", rot, m)
            if cx:
                mb = np.einsum("cij,jpsc->ipsc", rot, mb)
        elif kind == "free":
            t = op.ms
            ramp = 2 * math.pi * (x @ (np.asarray(op.n, float) * np.asarray(unit)))[:, None, None]      # [P, 1, 1]
            if t > 0:
                phase = 2 * math.pi * 1e-3 * df[None, :, :] * t
                if cx:
                    et = scipy.linalg.expm(gen_t * t)                   # [columns, 4, 4]
                    tr = np.einsum("cij,jpsc->ipsc", et, np.stack([m[0], m[1], mb[0], mb[1]]))
                    m[0], m[1], mb[0], mb[1] = tr[0], tr[1], tr[2], tr[3]
                    m[0], m[1] = turn(m, phase)
                    mb[0], mb[1] = turn(mb, phase)
                else:
                    e2 = np.exp(-t * r2)
                    m[0], m[1] = turn(m, phase)
                    m[0] *= e2
                    m[1] *= e2
                if two:
                    ez = scipy.linalg.expm(gen_z * t)                   # [columns, 2, 2]
                    dev = np.stack([m[2] - m_inf[:, 0], mb[2] - m_inf[:, 1]])
                    dev = np.einsum("cij,jpsc->ipsc", ez, dev)
                    m[2], mb[2] = dev[0] + m_inf[:, 0], dev[1] + m_inf[:, 1]
                else:
                    m[2] = sp.m0 + (m[2] - sp.m0) * np.exp(-t * r1)
            if rate is not None and t > 0:
                if any(op.n) and tuple(op.shape) != B.LINEAR:
                    raise ValueError("diffusion in the isochromat ensemble: constant gradients only (shape moments 1/2, 1/3)")
                steps = int(diffusion_substeps)
                mxy = m[0] + 1j * m[1]
                for j in range(steps):                                  # the part of the block's dephasing played so far
                    mxy = blur(mxy, 0.5 * t / steps, [j / steps * v for v in op.n])
                    mxy = blur(mxy, 0.5 * t / steps, [(j + 1) / steps * v for v in op.n])
                mxy = mxy * np.exp(1j * ramp)
                m[0], m[1] = mxy.real, mxy.imag
                m[2] = sp.m0 + blur(m[2] - sp.m0, t, [0] * axes).real
            elif any(op.n):
                m[0], m[1] = turn(m, ramp)
                if cx:
                    mb[0], mb[1] = turn(mb, ramp)
        elif kind == "spoil" or kind == "crush":
            m[0], m[1] = 0.0, 0.0
            if cx:
                mb[0], mb[1] = 0.0, 0.0
            if kind == "crush":
                m[2] = m[2].mean(0, keepdims=True)
                if two:
                    mb[2] = mb[2].mean(0, keepdims=True)
        elif kind == "mz":
            m[2] *= op.factor
        elif kind == "frame":
            raise ValueError("the isochromat ensemble runs in the laboratory frame")
        else:
            mxy = m[0] + 1j * m[1]
            if cx:
                mxy = mxy + mb[0] + 1j * mb[1]
            s = np.tensordot(wgt, mxy.mean(0), axes=(0, 0)) * (np.exp(-1j * math.radians(op.phase_deg)) * demodulation)
            if op.static_ms and spread is None:
                s = s * np.exp(-abs(op.static_ms) / sp.t2p_ms)
            out.append(s)
    return Result(np.array(out).reshape(len(out), sp.n), prog.labels, dict(m=m, mb=mb, x=x), prog)


__all__ = ["positions", "rodrigues", "run"]
