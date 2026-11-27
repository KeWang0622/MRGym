"""The steady state of a periodic block list as a fixed point, without stepping to it.

One period of a sequence is an affine map of the state, s -> M s + c: every block is linear in the magnetisation
except for the regrowth of Mz, which adds a constant. The periodic state is the solution of (1 - M) s = c.

The period. A sequence that is one ``repeat`` block is periodic with the period of its tables (the least common
multiple of their lengths, times the body); any other block list is taken as one period. RF phase increments of the
repeat are handled in the pulse frame (``blocks.compile_ops``): there the map of one repetition is the same in
every repetition, also with quadratic RF spoiling, so the fixed point exists.

Two methods
  ``matrix``   M is assembled as a sparse matrix on the extended-phase-graph states of all columns at once and the
               system is solved directly (banded LU). The unknowns per order k are F_k, conj(F_-k) and Z_k (plus the
               second pool): with conj(F_-k) as its own unknown every operator is complex-linear. This code shares
               the numbers of the operators with ``epg`` (``operators``) but not their application.
  ``reduced``  when the period contains an ideal crusher after its last pulse and gradient, the state at that point
               is Mz alone (one number per column and pool). The period is stepped with ``epg`` from Mz = 0 and from
               Mz = 1 (and once more for a second pool): Mz at the end and every sample are affine in the starting
               Mz, so the fixed point and the samples at it follow from those runs, whatever the length of an echo
               train. States above half the reach of the period cannot return to order 0 before the crusher and
               are not carried.
``auto`` takes ``reduced`` when it applies. Truncation: ``kmax``, or for the matrix method the rule of
``operators.auto_kmax`` per tissue (states of high order have decayed by exp(-t / T2) below ``tol``).

``approach`` returns the transient: the samples of every period from equilibrium next to the periodic ones.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg
import scipy.sparse as sps
import scipy.sparse.linalg

from . import blocks as B
from . import epg
from . import operators as O

MAX_UNKNOWNS = 150_000              # per solve; the columns are processed in chunks
MAX_BAND_ENTRIES = 12_000_000       # banded storage above this: sparse LU instead


def period(seq):
    """(program of one period in the pulse frame, repetitions per period or None)."""
    blocks = seq.blocks if isinstance(seq, B.Sequence) else tuple(seq)
    if len(blocks) == 1 and blocks[0].kind == "repeat":
        reps = 1
        for _, values in blocks[0].tables:
            reps = math.lcm(reps, len(values))
        return B.compile_ops(seq, frame=True, repetitions=reps), reps
    return B.compile_ops(seq, frame=True), None


def _reducible(prog):
    """True when an ideal crusher follows the last pulse and the last dephasing of the period."""
    last = max([i for i, op in enumerate(prog.ops) if op.kind == "rf" or (op.kind == "free" and any(op.n))], default=-1)
    return any(op.kind == "crush" for op in prog.ops[last + 1:])


def steady_state(seq, sp, kmax=None, tol=1e-6, method="auto", demodulation=1j):
    """Samples of one period in the periodic state: ``epg.Result`` with ``signal`` [samples, columns] in the order of
    the adc blocks of the period, and ``state`` = the periodic state at the start of the period (pulse frame).
    A sample that an RF-spoiled repeat takes away from zero dephasing has a periodic magnitude and a phase that
    advances from repetition to repetition (``blocks`` docstring); the value returned is that of the first period."""
    prog, _ = period(seq)
    if method not in ("auto", "matrix", "reduced"):
        raise ValueError("method is auto, matrix or reduced")
    if method == "reduced" or (method == "auto" and _reducible(prog)):
        if not _reducible(prog):
            raise ValueError("reduced: the period must end with an ideal crusher after its last pulse and gradient")
        return _reduced(prog, sp, kmax, demodulation)
    if kmax is not None:
        kcap = tuple(int(k) for k in (kmax if isinstance(kmax, (tuple, list)) else (kmax,) * prog.axes))
        groups = [(np.arange(sp.n), tuple(min(k, r) if r == 0 else k for k, r in zip(kcap, prog.reach)))]
    else:
        per_col = {}
        for t2 in np.unique(sp.t2_ms):
            idx = np.flatnonzero(sp.t2_ms == t2)
            k = O.auto_kmax(prog, O.take(sp, idx), tol)
            if k is None:
                raise ValueError("the gradients of this period take no time: give kmax (the reach of the states is not bounded by T2)")
            per_col.setdefault(k, []).append(idx)
        groups = [(np.concatenate(v), k) for k, v in per_col.items()]
    sig = np.zeros((len(prog.labels), sp.n), complex)
    state = None
    for idx, k in groups:
        nk = int(np.prod([2 * v + 1 for v in k]))
        ncomp = {None: 3, "mt": 4, "cx": 6}[sp.pool]
        chunk = max(1, MAX_UNKNOWNS // (nk * ncomp))
        for lo in range(0, idx.size, chunk):
            part = idx[lo:lo + chunk]
            s, u = _matrix(prog, O.take(sp, part), k, demodulation)
            sig[:, part] = s
            if len(groups) == 1 and chunk >= idx.size:
                state = u
    return epg.Result(sig, prog.labels, state, prog)


def approach(seq, sp, periods, kmax=None, tol=1e-6, demodulation=1j):
    """The transient: (samples [periods, samples per period, columns] stepped from equilibrium, the periodic samples
    [samples per period, columns]). ``seq`` must be one repeat block; ``periods`` counts periods of its tables."""
    prog, reps = period(seq)
    if reps is None:
        raise ValueError("approach needs a sequence that is one repeat block")
    fixed = steady_state(seq, sp, kmax=kmax, tol=tol, demodulation=demodulation)
    if kmax is None and prog.reach != (0,) * prog.axes:
        kmax = O.auto_kmax(prog, sp, tol)
    stepped = epg.run(seq, sp, kmax=kmax, frame=True, repetitions=reps * int(periods), demodulation=demodulation, keep_state=False)
    n = fixed.signal.shape[0]
    return stepped.signal.reshape(int(periods), n, sp.n), fixed.signal


# ---------------------------------------------------------------------------------------------- reduced: Mz only
def _reduced(prog, sp, kmax, demodulation):
    cut = max(i for i, op in enumerate(prog.ops) if op.kind == "crush")
    # start the period right after its last crusher: there the state is Mz alone
    rolled = B.Program(prog.ops[cut + 1:] + prog.ops[:cut + 1], prog.unit, prog.reach, prog.cycle_mm, prog.labels, prog.frame, prog.period_ms)
    # a state above half the reach of the period cannot come back to order 0 before the crusher: nothing is lost
    half = tuple((r + 1) // 2 for r in prog.reach)
    kk = half if kmax is None else tuple(min(r, int(k)) for r, k in zip(half, (kmax if isinstance(kmax, (tuple, list)) else (kmax,) * prog.axes)))
    centre = tuple(kk)
    two = sp.pool is not None

    def end(za, zb):
        st = epg.equilibrium(sp, kk, prog.unit)
        st.z[centre] = za
        if two:
            st.zb[centre] = zb
        res = epg.run(rolled, sp, kmax=kk, state=st, demodulation=demodulation, prune=True)
        return res, res.state.z[centre].real.copy(), (res.state.zb[centre].real.copy() if two else None)

    zero, one = np.zeros(sp.n), np.ones(sp.n)
    r0, ca, cb = end(zero, zero)
    r1, a1, b1 = end(one, zero)
    if two:
        r2, a2, b2 = end(zero, one)
        mat = np.zeros((sp.n, 2, 2))
        mat[:, 0, 0], mat[:, 0, 1], mat[:, 1, 0], mat[:, 1, 1] = 1 - (a1 - ca), -(a2 - ca), -(b1 - cb), 1 - (b2 - cb)
        sol = np.linalg.solve(mat, np.stack([ca, cb], 1)[..., None])[..., 0]
        za, zb = sol[:, 0], sol[:, 1]
        signal = r0.signal + za * (r1.signal - r0.signal) + zb * (r2.signal - r0.signal)
    else:
        za, zb = ca / (1 - (a1 - ca)), None
        signal = r0.signal + za * (r1.signal - r0.signal)                # the samples are affine in the starting Mz too
    # samples in the order of the period as written: those before the crusher come last in the rolled run
    n_before = sum(op.kind == "adc" for op in prog.ops[:cut + 1])
    n_all = len(prog.labels)
    order = list(range(n_all - n_before, n_all)) + list(range(n_all - n_before))
    start = epg.equilibrium(sp, kk, prog.unit)
    start.z[centre] = za
    if two:
        start.zb[centre] = zb
    tail = tuple(op for op in prog.ops[cut + 1:] if op.kind != "adc")
    if tail:                                                # the state at the START of the period as written
        start = epg.run(B.Program(tail, prog.unit, prog.reach, prog.cycle_mm, (), prog.frame, 0.0), sp, kmax=kk, state=start).state
    return epg.Result(signal[order], prog.labels, start, prog)


# ---------------------------------------------------------------------------------------------- matrix method
def _lattice(kmax, unit):
    """(k [nk, axes] in cycles, integer index [nk, axes], sizes)."""
    sizes = tuple(2 * k + 1 for k in kmax)
    idx = np.stack([g.ravel() for g in np.indices(sizes)], 1) - np.asarray(kmax)
    return idx * np.asarray(unit, float), idx, sizes


def _components(pool):
    """Indices of the components per order: (transverse F, their conjugate partners, longitudinal)."""
    if pool == "cx":
        return (0, 3), (1, 4), (2, 5)
    return (0,), (1,), ((2, 3) if pool == "mt" else (2,))


def _rf_block(op, sp, cache):
    nc = {None: 3, "mt": 4, "cx": 6}[sp.pool]
    blk = np.zeros((sp.n, 1, nc, nc), complex)
    blk[:, 0] = np.eye(nc)
    sat = O.saturation(op, sp, cache)
    if sat is not None:
        blk[:, 0, 3, 3] = sat
    if op.block.offset_hz:
        return blk
    a, b = O.rotation(op, sp, cache)
    rot = np.zeros((sp.n, 3, 3), complex)
    if op.block.crushed:
        rot[:, 0, 1], rot[:, 1, 0], rot[:, 2, 2] = -b * b, -np.conj(b * b), np.abs(a) ** 2 - np.abs(b) ** 2
    else:
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 0, 2] = np.conj(a) ** 2, -b * b, 2 * np.conj(a) * b
        rot[:, 1, 0], rot[:, 1, 1], rot[:, 1, 2] = -np.conj(b * b), a * a, 2 * a * np.conj(b)
        rot[:, 2, 0], rot[:, 2, 1], rot[:, 2, 2] = -np.conj(a * b), -a * b, np.abs(a) ** 2 - np.abs(b) ** 2
    blk[:, 0, :3, :3] = rot
    if sp.pool == "cx":
        blk[:, 0, 3:, 3:] = rot
    return blk


def _free_block(op, sp, prog, kc, rate):
    """Relaxation, precession, exchange and diffusion of a free operation, before its shift: (blocks [columns, nk,
    nc, nc], constant [columns, nk, nc])."""
    nc = {None: 3, "mt": 4, "cx": 6}[sp.pool]
    nk, t = kc.shape[0], op.ms
    blk = np.zeros((sp.n, nk, nc, nc), complex)
    const = np.zeros((sp.n, nk, nc), complex)
    zero = int(np.flatnonzero(np.all(kc == 0, 1))[0])
    step = np.asarray(op.n, float) * np.asarray(prog.unit, float)       # dephasing of the block, cycles per axis
    m1, m2 = op.shape
    if rate is not None:                                                # [columns, nk]
        up = np.exp(-t * np.einsum("ac,ka->ck", rate, kc ** 2 + 2 * m1 * kc * step + m2 * step ** 2))
        dn = np.exp(-t * np.einsum("ac,ka->ck", rate, kc ** 2 - 2 * m1 * kc * step + m2 * step ** 2))
        dz = np.einsum("ac,ka->ck", rate, kc ** 2)                      # 1/ms
    else:
        up = dn = np.ones((sp.n, nk))
        dz = np.zeros((sp.n, nk))
    r1, r2 = 1.0 / sp.t1_ms, 1.0 / sp.t2_ms
    w = 2j * math.pi * 1e-3
    if sp.pool == "cx":
        kf = sp.k_ab_per_s * 1e-3
        kr = kf / sp.f_b
        e = O.exp2(-r2 - kf + w * sp.df_hz, kr, kf, -1.0 / sp.t2b_ms - kr + w * (sp.df_hz + sp.shift_b_hz), t)
        for (i, j), v in zip(((0, 0), (0, 3), (3, 0), (3, 3)), e):
            blk[:, :, i, j] = v[:, None] * up
            blk[:, :, i + 1, j + 1] = np.conj(v)[:, None] * dn
    else:
        e2 = np.exp(-t * r2 + w * sp.df_hz * t)
        blk[:, :, 0, 0] = e2[:, None] * up
        blk[:, :, 1, 1] = np.conj(e2)[:, None] * dn
    if sp.pool is None:
        e1 = np.exp(-t * r1)
        blk[:, :, 2, 2] = np.exp(-t * (r1[:, None] + dz))
        const[:, zero, 2] = (1 - e1) * sp.m0
    else:
        kf = sp.k_ab_per_s * 1e-3
        kr, r1b, m0b = kf / sp.f_b, 1.0 / sp.t1b_ms, sp.m0 * sp.f_b
        zb = 5 if sp.pool == "cx" else 3
        e = O.exp2((-r1 - kf)[:, None] - dz, kr[:, None], kf[:, None], (-r1b - kr)[:, None] - (dz if sp.pool == "cx" else 0.0), t)
        blk[:, :, 2, 2], blk[:, :, 2, zb], blk[:, :, zb, 2], blk[:, :, zb, zb] = e
        g11, g12, g21, g22 = O.exp2(-r1 - kf, kr, kf, -r1b - kr, t)
        const[:, zero, 2] = sp.m0 - g11 * sp.m0 - g12 * m0b
        const[:, zero, zb] = m0b - g21 * sp.m0 - g22 * m0b
    return blk, const


def _diag_block(op, sp, kc):
    nc = {None: 3, "mt": 4, "cx": 6}[sp.pool]
    tr, cj, lg = _components(sp.pool)
    nk = kc.shape[0]
    d = np.ones((1, nk, nc), complex)
    if op.kind in ("spoil", "crush"):
        d[:, :, list(tr + cj)] = 0
        if op.kind == "crush":
            d[:, np.any(kc != 0, 1)] = 0
    elif op.kind == "mz":
        d[:, :, 2] = op.factor
    else:                                                               # frame
        ramp = np.exp(1j * (kc @ np.asarray(op.shift, float))) if any(op.shift) else np.ones(nk)
        d = d * ramp[None, :, None]
        d[:, :, list(tr)] *= np.exp(1j * op.turn)
        d[:, :, list(cj)] *= np.exp(-1j * op.turn)
    blk = np.zeros((1, nk, nc, nc), complex)
    blk[:, :, np.arange(nc), np.arange(nc)] = d
    return blk


def _shift_maps(n, ki, sizes, kmax, pool):
    """Source and target (order index, component) of a dephasing by ``n`` lattice steps: F_k -> F_(k+n), the partner
    conj(F_-k) moves the other way, longitudinal components stay. States that leave the lattice are dropped."""
    tr, cj, lg = _components(pool)
    src_k, dst_k, comp = [], [], []
    base = np.arange(ki.shape[0])
    for comps, sign in ((tr, 1), (cj, -1), (lg, 0)):
        tgt = ki + sign * np.asarray(n)
        ok = np.all(np.abs(tgt) <= np.asarray(kmax), 1)
        lin = np.ravel_multi_index(tuple((tgt[ok] + np.asarray(kmax)).T), sizes)
        for c in comps:
            src_k.append(base[ok])
            dst_k.append(lin)
            comp.append(np.full(lin.size, c))
    return np.concatenate(src_k), np.concatenate(dst_k), np.concatenate(comp)


def _matrix(prog, sp, kmax, demodulation):
    """Solve the periodic state of the columns ``sp`` on the lattice ``kmax``; (samples [adc, columns], state)."""
    nc = {None: 3, "mt": 4, "cx": 6}[sp.pool]
    kc, ki, sizes = _lattice(kmax, prog.unit)
    nk, ncol = kc.shape[0], sp.n
    n_unk = ncol * nk * nc
    rate = O.diffusion_rate(prog, sp)
    cache = {}
    zero = int(np.flatnonzero(np.all(ki == 0, 1))[0])

    # the period as a list of steps: ("local", blocks, constant) and ("shift", maps)
    steps, pending = [], None

    def add_local(blk, const=None):
        nonlocal pending
        if pending is None:
            pending = [blk, const]
        else:
            old, old_c = pending
            new_c = None if old_c is None else np.matmul(blk, old_c[..., None])[..., 0]
            if const is not None:
                new_c = const if new_c is None else new_c + const
            pending = [np.matmul(blk, old), new_c]

    def flush():
        nonlocal pending
        if pending is not None:
            steps.append(("local", pending[0], pending[1]))
            pending = None

    for op in prog.ops:
        if op.kind == "rf":
            add_local(_rf_block(op, sp, cache))
        elif op.kind == "free":
            if op.ms > 0:
                add_local(*_free_block(op, sp, prog, kc, rate))
            if any(op.n):
                flush()
                steps.append(("shift", _shift_maps(op.n, ki, sizes, kmax, sp.pool)))
        elif op.kind == "adc":
            flush()
            steps.append(("adc", op))
        else:
            add_local(_diag_block(op, sp, kc))
    flush()

    def apply(step, vec):
        """One step on a state [columns, nk, nc]."""
        if step[0] == "local":
            out = np.matmul(step[1], vec[..., None])[..., 0]
            return out if step[2] is None else out + step[2]
        src, dst, comp = step[1]
        out = np.zeros_like(vec)
        out[:, dst, comp] = vec[:, src, comp]
        return out

    # assemble M and c
    mat, const = None, np.zeros((ncol, nk, nc), complex)
    offs = (np.arange(ncol) * nk)[:, None]
    for step in steps:
        if step[0] == "adc":
            continue
        const = apply(step, const)
        if step[0] == "local":
            data = np.ascontiguousarray(np.broadcast_to(step[1], (ncol, nk, nc, nc))).reshape(ncol * nk, nc, nc)
            op_m = sps.bsr_matrix((data, np.arange(ncol * nk), np.arange(ncol * nk + 1)), shape=(n_unk, n_unk)).tocsr()
        else:
            src, dst, comp = step[1]
            rows = ((offs + dst[None, :]) * nc + comp[None, :]).ravel()
            cols = ((offs + src[None, :]) * nc + comp[None, :]).ravel()
            op_m = sps.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n_unk, n_unk))
        mat = op_m if mat is None else op_m @ mat
    system = (sps.identity(n_unk, format="csr", dtype=complex) - mat).tocoo()
    lo, up = int(max(0, (system.row - system.col).max())), int(max(0, (system.col - system.row).max()))
    rhs = const.reshape(-1)
    if (2 * lo + up + 1) * n_unk <= MAX_BAND_ENTRIES:
        ab = np.zeros((lo + up + 1, n_unk), complex)
        ab[up + system.row - system.col, system.col] = system.data     # a matrix in COO form made from CSR has no duplicates
        sol = scipy.linalg.solve_banded((lo, up), ab, rhs, overwrite_ab=True, overwrite_b=True, check_finite=False)
        if not np.all(np.isfinite(sol)):
            raise FloatingPointError("the periodic system is singular: the sequence has no unique periodic state")
    else:
        sol = scipy.sparse.linalg.spsolve(system.tocsc(), rhs)
    u = sol.reshape(ncol, nk, nc)

    # samples of one period from the fixed point
    tr = _components(sp.pool)[0]
    whole = all(v == 1 for v in prog.unit)
    wgt = np.prod(np.sinc(kc), 1)
    out, vec = [], u
    for step in steps:
        if step[0] != "adc":
            vec = apply(step, vec)
            continue
        op = step[1]
        s = sum(vec[:, zero, c] for c in tr) if whole else sum((vec[:, :, c] * wgt[None, :]).sum(1) for c in tr)
        s = s * (np.exp(-1j * math.radians(op.phase_deg)) * demodulation)
        if op.static_ms:
            s = s * np.exp(-abs(op.static_ms) / sp.t2p_ms)
        out.append(s)
    lat = lambda c: np.ascontiguousarray(u[:, :, c].T).reshape(sizes + (ncol,))       # noqa: E731
    state = epg.State(lat(0), lat(2), lat(3) if sp.pool == "cx" else None,
                      lat(5) if sp.pool == "cx" else (lat(3) if sp.pool == "mt" else None), tuple(kmax), tuple(prog.unit))
    return np.array(out).reshape(len(out), ncol), state


__all__ = ["approach", "period", "steady_state"]
