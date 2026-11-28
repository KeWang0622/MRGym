"""Extended phase graphs: the block list stepped in time, for all columns of spins at once.

State. Inside one cycle length of the gradients the magnetisation of a column is periodic, so it is a Fourier
series: Mxy(x) = sum_k F_k exp(2 pi i k x), Mz(x) = sum_k Z_k exp(2 pi i k x) (``operators`` docstring). The arrays
``f`` and ``z`` hold F_k and Z_k for every k of a lattice -K .. K per gradient axis (both signs are stored; Z_-k =
conj(Z_k) is kept by every operation), with the columns as the last axis.

What each operation does to the states
  rf        F_k' = conj(a)^2 F_k - b^2 conj(F_-k) + 2 conj(a) b Z_k
            Z_k' = (|a|^2 - |b|^2) Z_k - a b conj(F_-k) - conj(a b) F_k          (the rotation, order by order)
            crushed: F_k' = -b^2 conj(F_-k), Z_k' = (|a|^2 - |b|^2) Z_k
  free      for t ms with n cycles of dephasing: F_k x exp(-t / T2) exp(2 pi i df t), Z_k x exp(-t / T1),
            Z_0 + (1 - exp(-t / T1)) M0; diffusion: F_k x exp(-D t sum_axes kappa^2 (k^2 + 2 m1 k n + m2 n^2)) for the
            order k it enters with, Z_k x exp(-D t sum kappa^2 k^2); then F_k moves to F_(k+n)
  spoil     F = 0;   crush  F = 0 and Z_k = 0 for k != 0;   mz  Z_k x factor
  adc       sum_k F_k sinc(k) (F_0 for whole cycles) x exp(-i receiver phase) x demodulation x exp(-|static| / T2')
  frame     F_k x exp(i turn) exp(i k.shift), Z_k x exp(i k.shift)   (the pulse frame of ``blocks.compile_ops``)
With two pools the longitudinal states of both pools (and for chemical exchange the transverse ones) evolve under
the 2 x 2 exchange matrix of the delay, exactly (``operators.exp2``), with the diffusion of each order inside it.

Truncation. ``kmax`` (a number or one per axis) bounds the lattice; states that a gradient moves beyond it are
dropped. Without ``kmax`` every state the gradients can reach is kept and nothing is dropped. The operations act
on the part of the lattice that has been reached so far, so a short sequence costs little whatever the bound.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import blocks as B
from . import operators as O


@dataclasses.dataclass(eq=False)
class State:
    f: np.ndarray                     # [lattice..., columns], complex: F_k of the free pool
    z: np.ndarray
    fb: np.ndarray | None             # second pool (chemical exchange only)
    zb: np.ndarray | None             # second pool (magnetisation transfer and chemical exchange)
    kmax: tuple                       # lattice half-width per axis
    unit: tuple                       # cycles per lattice step
    active: tuple | None = None       # half-width of the part of the lattice that can be non-zero (None: all of it)

    def copy(self):
        return State(self.f.copy(), self.z.copy(), None if self.fb is None else self.fb.copy(),
                     None if self.zb is None else self.zb.copy(), self.kmax, self.unit, self.active)


@dataclasses.dataclass(eq=False)
class Result:
    signal: np.ndarray                # [samples, columns] complex
    labels: tuple
    state: State | None
    program: B.Program | None = None

    def of(self, label):
        """Samples whose label is ``label`` or starts with it (a label inside a repeat is (label, repetition, ...))."""
        head = label if isinstance(label, tuple) else (label,)
        idx = [i for i, lab in enumerate(self.labels) if lab == label or (isinstance(lab, tuple) and lab[:len(head)] == head)]
        return self.signal[idx]


def equilibrium(sp, kmax, unit):
    sizes = tuple(2 * k + 1 for k in kmax)
    centre = tuple(kmax)
    f, z = np.zeros(sizes + (sp.n,), complex), np.zeros(sizes + (sp.n,), complex)
    z[centre] = sp.m0
    fb = zb = None
    if sp.pool is not None:
        zb = np.zeros_like(z)
        zb[centre] = sp.m0 * sp.f_b
        fb = np.zeros_like(f) if sp.pool == "cx" else None
    return State(f, z, fb, zb, tuple(kmax), tuple(unit), (0,) * len(kmax))


def _embed(state, kmax):
    """``state`` on a lattice of half-widths ``kmax`` (zero padded or cut)."""
    if tuple(state.kmax) == tuple(kmax):
        return state.copy()

    def fit(x):
        if x is None:
            return None
        out = np.zeros(tuple(2 * k + 1 for k in kmax) + x.shape[-1:], complex)
        keep = [min(a, b) for a, b in zip(state.kmax, kmax)]
        src = tuple(slice(a - k, a + k + 1) for a, k in zip(state.kmax, keep))
        dst = tuple(slice(b - k, b + k + 1) for b, k in zip(kmax, keep))
        out[dst] = x[src]
        return out

    active = None if state.active is None else tuple(min(a, k) for a, k in zip(state.active, kmax))
    return State(fit(state.f), fit(state.z), fit(state.fb), fit(state.zb), tuple(kmax), state.unit, active)


def _k(act, unit, axis, axes):
    """Dephasing (cycles) along ``axis`` of the active lattice, shaped to broadcast against [lattice..., columns]."""
    shape = [1] * (axes + 1)
    shape[axis] = 2 * act[axis] + 1
    return (np.arange(-act[axis], act[axis] + 1) * unit[axis]).reshape(shape)


def run(seq, sp, kmax=None, state=None, frame=False, demodulation=1j, repetitions=None, keep_state=True, prune=False):
    """Step the sequence for the columns ``sp`` from equilibrium, or from ``state`` (the ``State`` of an earlier
    result). Returns ``Result``: the samples of the adc blocks in order, [samples, columns], and the final state.

    ``frame`` runs the RF phase increments of repeats in the pulse frame (``blocks.compile_ops``); the samples are
    the same. ``repetitions`` compiles that many repetitions of a sequence that is one repeat block.
    ``prune``: stop carrying states whose order exceeds the dephasing that the rest of the sequence can undo; they
    cannot return to order 0, so every sample is unchanged, but the final state is then valid at order 0 only. A
    tuple of orders per axis instead of True keeps what can still return to that box of orders: the final state is
    then valid inside the box (a sample is taken at order 0, which every box holds)."""
    prog = seq if isinstance(seq, B.Program) else B.compile_ops(seq, frame=frame, repetitions=repetitions)
    axes, unit = prog.axes, prog.unit
    if state is not None and tuple(state.unit) != tuple(unit):
        raise ValueError("the state was made on another lattice")
    kcap = None if kmax is None else tuple(int(k) for k in (kmax if isinstance(kmax, (tuple, list)) else (kmax,) * axes))
    start = (0,) * axes if state is None else tuple(state.kmax)
    big = tuple(r + s0 for r, s0 in zip(prog.reach, start))
    kk = big if kcap is None else tuple(min(b, c) for b, c in zip(big, kcap))
    st = equilibrium(sp, kk, unit) if state is None else _embed(state, kk)
    f, z, fb, zb = st.f, st.z, st.fb, st.zb
    centre = tuple(kk)
    act = [min(s0, k) for s0, k in zip(start if st.active is None else st.active, kk)]
    whole = all(u == 1 for u in unit)
    rate = O.diffusion_rate(prog, sp)                                   # [axes, columns] or None
    r1, r2 = 1.0 / sp.t1_ms, 1.0 / sp.t2_ms
    two, cx = sp.pool is not None, sp.pool == "cx"
    if two:
        kf = sp.k_ab_per_s * 1e-3
        kr, r1b = kf / sp.f_b, 1.0 / sp.t1b_ms
        m0b = sp.m0 * sp.f_b
    has_df = bool(np.any(sp.df_hz != 0))
    flip = (slice(None, None, -1),) * axes
    cache, out = {}, []
    remain = None
    if prune is not False and prune is not None:                        # dephasing still to come after every operation
        steps = np.array([[abs(m) for m in op.n] if op.kind == "free" else [0] * axes for op in prog.ops]).reshape(len(prog.ops), axes)
        box = (0,) * axes if prune is True else tuple(int(b) for b in prune)
        remain = steps[::-1].cumsum(0)[::-1] - steps + np.asarray(box)

    def window():
        return tuple(slice(k - a, k + a + 1) for k, a in zip(kk, act))

    for i_op, op in enumerate(prog.ops):
        kind = op.kind
        win = window()
        if kind == "rf":
            sat = O.saturation(op, sp, cache)
            if sat is not None:
                zb[win] *= sat
            if op.block.offset_hz:
                continue
            a, b = O.rotation(op, sp, cache)
            pools = ((f, z), (fb, zb)) if cx else ((f, z),)
            for ft, zt in pools:
                fw, zw = ft[win], zt[win]
                fr = np.conj(fw[flip])
                if op.block.crushed:
                    ft[win], zt[win] = -(b * b) * fr, (np.abs(a) ** 2 - np.abs(b) ** 2) * zw
                else:
                    ac, ab = np.conj(a), a * b
                    ft[win], zt[win] = (ac * ac * fw - (b * b) * fr + (2 * ac * b) * zw,
                                        (np.abs(a) ** 2 - np.abs(b) ** 2) * zw - ab * fr - np.conj(ab) * fw)
        elif kind == "free":
            t, n = op.ms, op.n
            if t > 0:
                key = ("free", t, n, op.shape)
                fac = cache.get(key)
                if fac is None:                                         # the factors of this interval on the whole lattice: equal blocks share them
                    df_exp = dz_rate = None
                    if rate is not None:
                        kax = [_k(kk, unit, i, axes) for i in range(axes)]
                        m1, m2 = op.shape
                        df_exp = sum(rate[i] * t * (kax[i] ** 2 + 2 * m1 * kax[i] * (n[i] * unit[i]) + m2 * (n[i] * unit[i]) ** 2) for i in range(axes))
                        dz_rate = sum(rate[i] * kax[i] ** 2 for i in range(axes))                   # 1/ms, [lattice..., columns]
                    rot = np.exp(2j * math.pi * 1e-3 * sp.df_hz * t) if has_df else 1.0
                    fac = dict(df_exp=df_exp, dz_rate=dz_rate, rot=rot)
                    if not cx:
                        fac["f"] = (np.exp(-t * r2) * rot) if df_exp is None else np.exp(-t * r2 - df_exp) * rot
                    if not two:
                        e1 = np.exp(-t * r1)
                        fac["z"], fac["grow"] = (e1 if dz_rate is None else np.exp(-t * (r1 + dz_rate))), (1 - e1) * sp.m0
                    cache[key] = fac
                df_exp, dz_rate = fac["df_exp"], fac["dz_rate"]
                on = (lambda a, win=win: a[win]) if rate is not None else (lambda a: a)    # noqa: E731  (a factor of the lattice, or one per column)
                if cx:
                    w = 2j * math.pi * 1e-3
                    e11, e12, e21, e22 = O.exp2(-r2 - kf + w * sp.df_hz, kr, kf, -1.0 / sp.t2b_ms - kr + w * (sp.df_hz + sp.shift_b_hz), t)
                    fw, gw = f[win], fb[win]
                    f[win], fb[win] = e11 * fw + e12 * gw, e21 * fw + e22 * gw
                    if df_exp is not None:
                        damp = np.exp(-df_exp[win])
                        f[win] *= damp
                        fb[win] *= damp
                else:
                    f[win] *= on(fac["f"])
                if two:
                    da = 0.0 if dz_rate is None else dz_rate[win]
                    e11, e12, e21, e22 = O.exp2(-r1 - kf - da, kr, kf, -r1b - kr - (da if cx else 0.0), t)
                    zw, yw = z[win], zb[win]
                    z[win], zb[win] = e11 * zw + e12 * yw, e21 * zw + e22 * yw
                    g11, g12, g21, g22 = O.exp2(-r1 - kf, kr, kf, -r1b - kr, t)
                    z[centre] += sp.m0 - g11 * sp.m0 - g12 * m0b
                    zb[centre] += m0b - g21 * sp.m0 - g22 * m0b
                else:
                    z[win] *= on(fac["z"])
                    z[centre] += fac["grow"]
            if any(n):
                new = [min(k, a + abs(m)) for k, a, m in zip(kk, act, n)]
                dst = tuple(slice(max(0, k - a + m), min(2 * k, k + a + m) + 1) for k, a, m in zip(kk, act, n))
                src = tuple(slice(s.start - m, s.stop - m) for s, m in zip(dst, n))
                for ft in ((f, fb) if cx else (f,)):
                    moved = ft[src].copy() if all(s.stop > s.start for s in dst) else None
                    ft[win] = 0
                    if moved is not None:
                        ft[dst] = moved
                act = new if remain is None else [min(a, int(r)) for a, r in zip(new, remain[i_op])]
        elif kind == "spoil" or kind == "crush":
            f[win] = 0
            if cx:
                fb[win] = 0
            if kind == "crush":
                for zt in ((z, zb) if two else (z,)):
                    keep = zt[centre].copy()
                    zt[win] = 0
                    zt[centre] = keep
                act = [0] * axes                                        # nothing but Z_0 is left
        elif kind == "mz":
            z[win] *= op.factor
        elif kind == "frame":
            ramp = 1.0
            if any(op.shift):
                ramp = np.exp(1j * sum(op.shift[i] * _k(act, unit, i, axes) for i in range(axes)))
            for ft, zt in (((f, z), (fb, zb)) if cx else ((f, z),)):
                ft[win] *= ramp * np.exp(1j * op.turn)
                zt[win] *= ramp
            if two and not cx:
                zb[win] *= ramp
        else:                                                           # adc
            if whole:
                s = f[centre] + (fb[centre] if cx else 0.0)
            else:
                wgt = 1.0
                for i in range(axes):
                    wgt = wgt * np.sinc(_k(act, unit, i, axes))
                s = (f[win] * wgt).reshape(-1, sp.n).sum(0)
                if cx:
                    s = s + (fb[win] * wgt).reshape(-1, sp.n).sum(0)
            s = s * (np.exp(-1j * math.radians(op.phase_deg)) * demodulation)
            if op.static_ms:
                s = s * np.exp(-abs(op.static_ms) / sp.t2p_ms)
            out.append(s)
    sig = np.array(out).reshape(len(out), sp.n)
    st.active = tuple(act)
    return Result(sig, prog.labels, st if keep_state else None, prog)


def magnetisation(state, x):
    """(Mxy, Mz) of the free pool at the positions ``x`` (cycle lengths; [points] for one axis or [points, axes]) for
    every column: the Fourier series of the state. For comparison with an isochromat ensemble."""
    axes = len(state.kmax)
    x = np.asarray(x, float).reshape(-1, axes)
    e = 1.0
    for i in range(axes):
        k = np.arange(-state.kmax[i], state.kmax[i] + 1) * state.unit[i]
        shape = [x.shape[0]] + [1] * axes
        shape[1 + i] = k.size
        e = e * np.exp(2j * math.pi * x[:, i][:, None] * k[None, :]).reshape(shape)
    lat = tuple(range(1, axes + 1))
    mxy = np.tensordot(e, state.f, axes=(lat, tuple(range(axes))))
    mz = np.tensordot(e, state.z, axes=(lat, tuple(range(axes)))).real
    return mxy, mz


__all__ = ["Result", "State", "equilibrium", "magnetisation", "run"]
