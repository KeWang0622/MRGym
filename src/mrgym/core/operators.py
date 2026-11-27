"""What the three evaluators share: the spins, the conventions, and the numbers of every operator.

Conventions (those of docs/physics/design.md, section 2; ``mrgym.core.validate`` rows C1 to C3 compare them with
``mrgym.design`` and ``mrgym.scanner``)
  rotation     right-handed, by the angle phi about the unit axis n:  a = cos(phi/2) - i n_z sin(phi/2),
               b = -i (n_x + i n_y) sin(phi/2). A spin (Mxy, Mz) becomes
                   Mxy' = conj(a)^2 Mxy - b^2 conj(Mxy) + 2 conj(a) b Mz
                   Mz'  = (|a|^2 - |b|^2) Mz - 2 Re(a b conj(Mxy)).
               A hard pulse of flip angle alpha and phase p has a = cos(alpha/2), b = -i exp(i p) sin(alpha/2): it
               turns about (cos p, sin p, 0) and takes +z to Mxy = -i exp(i p) sin(alpha).
  precession   a positive frequency gains positive phase: Mxy -> Mxy exp(+2 pi i df t).
  dephasing    a gradient block of +n cycles gives the spin at position x (in cycles lengths, -1/2 <= x < 1/2 across
               one voxel when the cycle length is the voxel) the phase +2 pi n x.
  states       Mxy(x) = sum_k F_k exp(2 pi i k x),  Mz(x) = sum_k Z_k exp(2 pi i k x),  Z_-k = conj(Z_k). A gradient of
               +n cycles moves F_k to F_(k+n); Z does not move.
  sample       adc returns  (sum_k F_k sinc(k)) exp(-i receiver phase) x demodulation: the mean of Mxy over one cycle
               length. For whole cycles this is F_0. The default demodulation +i makes the signal of a pulse about
               x real and positive, as the engine reports it.
  crushed rf   the pulse between a perfect crusher pair: Mxy' = -b^2 conj(Mxy), Mz' = (|a|^2 - |b|^2) Mz.
  time         pulses are instantaneous; a block's duration is the time of its free evolution.

Two pools (option). ``pool="mt"``: free water a (M0a = m0) and a bound pool b without transverse magnetisation
(M0b = f_b m0), exchanging longitudinal magnetisation at k_ab (a to b) and k_ab / f_b (b to a):
    dZa/dt = (M0a - Za) / T1a - k_ab Za + (k_ab / f_b) Zb,     dZb/dt = (M0b - Zb) / T1b + k_ab Za - (k_ab / f_b) Zb.
An rf block with the energy E = integral of |B1|^2 dt multiplies Zb by exp(-pi w1^2 tau g(offset)), where
w1^2 tau = (2 pi gamma)^2 E x (B1 scale)^2 and g is the absorption line of the bound pool (``lineshape``). The bound
pool does not diffuse. ``pool="cx"``: chemical exchange (Bloch-McConnell): pool b also has transverse magnetisation
with its own T2 and a frequency shift; both pools see every pulse as the same rotation and diffuse alike.
"""
from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np

from ..design import pulses as design_pulses
from ..design import rotation as design_rotation
from ..design.units import GAMMA_HZ_PER_T

POOLS = (None, "mt", "cx")
LINESHAPES = ("superlorentzian", "lorentzian", "gaussian")
RAD2_PER_S_PER_UT2_MS = (2 * math.pi * GAMMA_HZ_PER_T) ** 2 * 1e-15       # w1^2 tau (rad^2 / s) per uT^2 ms


# ---------------------------------------------------------------------------------------------- the spins
@dataclasses.dataclass(frozen=True, eq=False)
class Spins:
    """Independent columns of spins: one entry per column in every array. ``shape`` is the grid the columns were
    made from (``grid``: tissue x slice position x B1 scale), for reshaping results."""
    t1_ms: np.ndarray
    t2_ms: np.ndarray
    t2p_ms: np.ndarray
    df_hz: np.ndarray
    b1: np.ndarray
    m0: np.ndarray
    adc_um2_ms: np.ndarray
    zeta: np.ndarray                  # position through the slice in reference slice thicknesses
    column: np.ndarray                # index of the slice column (for rf blocks given by ck)
    pool: str | None = None
    f_b: np.ndarray | None = None
    k_ab_per_s: np.ndarray | None = None
    t1b_ms: np.ndarray | None = None
    t2b_ms: np.ndarray | None = None
    shift_b_hz: np.ndarray | None = None
    lineshape: str = "superlorentzian"
    shape: tuple = ()

    @property
    def n(self):
        return int(self.t1_ms.size)


def spins(t1_ms, t2_ms, t2p_ms=np.inf, df_hz=0.0, b1=1.0, m0=1.0, adc_um2_ms=0.0, zeta=0.0, column=0, pool=None, f_b=0.0,
          k_ab_per_s=0.0, t1b_ms=1000.0, t2b_ms=0.012, shift_b_hz=0.0, lineshape="superlorentzian", shape=None):
    """Columns of spins from numbers or arrays that broadcast to one common one-dimensional shape."""
    if pool not in POOLS or lineshape not in LINESHAPES:
        raise ValueError(f"pool in {POOLS}, lineshape in {LINESHAPES}")
    vals = [np.asarray(v, float) for v in (t1_ms, t2_ms, t2p_ms, df_hz, b1, m0, adc_um2_ms, zeta, f_b, k_ab_per_s, t1b_ms, t2b_ms, shift_b_hz)]
    col = np.asarray(column, int)
    full = np.broadcast_shapes(col.shape, *(v.shape for v in vals))
    n = int(np.prod(full)) if full else 1
    flat = [np.ascontiguousarray(np.broadcast_to(v, full).reshape(n)) for v in vals]
    t1, t2, t2p, df, sc, m, adc, z, fb, kab, t1b, t2b, shb = flat
    if np.any(t1 <= 0) or np.any(t2 <= 0) or np.any(t2p <= 0) or np.any(adc < 0):
        raise ValueError("t1_ms, t2_ms, t2p_ms are positive and adc_um2_ms is not negative")
    extra = {}
    if pool is not None:
        if np.any(fb <= 0) or np.any(kab < 0) or np.any(t1b <= 0) or np.any(t2b <= 0):
            raise ValueError("two pools: f_b > 0, k_ab_per_s >= 0, t1b_ms > 0, t2b_ms > 0")
        extra = dict(f_b=fb, k_ab_per_s=kab, t1b_ms=t1b, t2b_ms=t2b, shift_b_hz=shb)
    return Spins(t1, t2, t2p, df, sc, m, adc, z, np.ascontiguousarray(np.broadcast_to(col, full).reshape(n)), pool=pool,
                 lineshape=lineshape, shape=tuple(shape) if shape is not None else (full or (1,)), **extra)


def grid(tissues, zeta=(0.0,), b1=(1.0,), df_hz=0.0, **common):
    """Columns for every tissue x slice position x B1 scale; ``Spins.shape`` = (tissues, positions, scales).
    ``tissues``: dicts with t1_ms, t2_ms and optionally t2p_ms, adc_um2_ms, m0, df_hz and the two-pool numbers."""
    tissues, zeta, b1 = list(tissues), np.atleast_1d(np.asarray(zeta, float)), np.atleast_1d(np.asarray(b1, float))
    shp = (len(tissues), zeta.size, b1.size)
    keys = ("t1_ms", "t2_ms", "t2p_ms", "adc_um2_ms", "m0", "df_hz", "f_b", "k_ab_per_s", "t1b_ms", "t2b_ms", "shift_b_hz")
    kw = dict(common)
    for key in keys:
        if any(key in t for t in tissues):
            default = dict(t2p_ms=np.inf, adc_um2_ms=0.0, m0=1.0, df_hz=0.0, shift_b_hz=0.0).get(key)
            vals = [t.get(key, default) for t in tissues]
            if any(v is None for v in vals):
                raise ValueError(f"{key} is given for some tissues only")
            kw[key] = np.broadcast_to(np.asarray(vals, float)[:, None, None], shp)
    kw["df_hz"] = kw.get("df_hz", 0.0) + np.asarray(df_hz, float)
    return spins(zeta=np.broadcast_to(zeta[None, :, None], shp), b1=np.broadcast_to(b1[None, None, :], shp),
                 column=np.broadcast_to(np.arange(zeta.size)[None, :, None], shp), shape=shp, **kw)


def take(sp, index):
    """The columns ``index`` (an index array or mask) of ``sp``."""
    out = {}
    for f in dataclasses.fields(sp):
        v = getattr(sp, f.name)
        out[f.name] = v[index] if isinstance(v, np.ndarray) else v
    out["shape"] = (int(out["t1_ms"].size),)
    return Spins(**out)


# ---------------------------------------------------------------------------------------------- slice columns
def slice_columns(tbw, two_sided=True, points_per_thickness=6, wing=12.0):
    """(zeta, weights): positions through the slice in slice thicknesses and the weights of the integral of the
    signal over them, per slice thickness: the trapezoid rule with ``points_per_thickness`` x tbw points per thickness
    out to 0.5 + ``wing`` / tbw thicknesses (the rule of ``scanner/rf.py::quadrature``). ``two_sided`` False: z >= 0
    only, every weight counting the position and its mirror image (valid for a signal that is even in z)."""
    h = 1.0 / (points_per_thickness * tbw)
    k = int(round((0.5 + wing / tbw) / h))
    if two_sided:
        w = np.full(2 * k + 1, h)
        w[[0, -1]] = h / 2
        return np.arange(-k, k + 1) * h, w
    w = np.full(k + 1, 2 * h)
    w[[0, -1]] = h
    return np.arange(k + 1) * h, w


@functools.lru_cache(maxsize=32)
def _sinc(tbw, window, samples):
    """The windowed sinc of the design layer with unit flip angle (1 rad at B1 scale 1), 1 ms long, on a 1 mm slice."""
    return design_pulses.sinc(math.degrees(1.0), 1.0, tbw, 1.0, window=window, kind="refocusing", n=samples)


def selective_rotation(sel, centre_flip_rad, zeta):
    """(a, b) of the selective pulse ``sel`` for flip angles at the slice centre and positions (arrays of one shape),
    after its rephasing lobe, for a pulse about x. Computed for the distinct (position, angle) pairs only."""
    flip, pos = np.broadcast_arrays(np.asarray(centre_flip_rad, float), np.asarray(zeta, float) / sel.width)
    pairs, inverse = np.unique(np.stack([pos.ravel(), flip.ravel()], 1), axis=0, return_inverse=True)
    pulse = _sinc(float(sel.tbw), sel.window, int(sel.samples))
    a, b = design_rotation.rotate(pulse.b1_ut, pulse.dt_ms, pulse.g_mt_m, pairs[:, 0], 0.0, pairs[:, 1])
    if sel.rephase:
        a, b = design_rotation.z_rotation(a, b, -sel.rephase * 2 * math.pi * sel.tbw * pairs[:, 0])
    if sel.prephase:                                                    # a rotation about z BEFORE the pulse: (a e, b e)
        e = np.exp(0.5j * sel.prephase * 2 * math.pi * sel.tbw * pairs[:, 0])
        a, b = a * e, b * e
    inverse = np.asarray(inverse).reshape(-1)
    return a[inverse].reshape(flip.shape), b[inverse].reshape(flip.shape)


def column_phase(sel, flip_deg, sp):
    """The transverse phase factor that the selective pulse leaves in every column, +1 at the slice centre:
    i exp(i arg(2 conj(a) b)). With ``Selective(model="flip")`` the signal of a column is multiplied by it."""
    a, b = selective_rotation(sel, math.radians(flip_deg) * sp.b1, sp.zeta)
    return 1j * np.exp(1j * np.angle(2 * np.conj(a) * b))


def rotation(op, sp, cache=None):
    """Cayley-Klein parameters (a, b) of the rf operation ``op`` for every column of ``sp``."""
    blk = op.block
    key = (blk.selective, blk.ck, op.flip_deg)
    hit = None if cache is None else cache.get(key)
    if hit is None:
        if blk.ck is not None:
            ck = np.asarray(blk.ck)
            if sp.column.max() >= ck.shape[0]:
                raise ValueError(f"the rf block holds {ck.shape[0]} slice columns; the spins ask for column {int(sp.column.max())}")
            row = ck[sp.column]
            hit = (row[:, 0] + 1j * row[:, 1], row[:, 2] + 1j * row[:, 3])
        elif blk.selective is not None:
            a, b = selective_rotation(blk.selective, math.radians(op.flip_deg) * sp.b1, sp.zeta)
            if blk.selective.model == "flip":
                half = np.arcsin(np.clip(np.abs(b), 0.0, 1.0))
                a, b = np.cos(half) + 0j, -1j * np.sin(half)
            hit = (a, b)
        else:
            half = 0.5 * math.radians(op.flip_deg) * sp.b1
            hit = (np.cos(half) + 0j, -1j * np.sin(half))
        if cache is not None:
            cache[key] = hit
    return hit[0], hit[1] * np.exp(1j * math.radians(op.phase_deg))


# ---------------------------------------------------------------------------------------------- bound pool
def lineshape(kind, offset_hz, t2b_ms):
    """Absorption line g of the bound pool at the frequency offset, in seconds, normalised to unit area over the
    angular frequency (attributed to Morrison & Henkelman 1995; the paper was not opened, row M1 checks the normalisation only):
        lorentzian        T2b / pi / (1 + (w T2b)^2)
        gaussian          T2b / sqrt(2 pi) x exp(-(w T2b)^2 / 2)
        superlorentzian   integral over u in [0, 1] of sqrt(2 / pi) T2b / |3 u^2 - 1| x exp(-2 (w T2b / (3 u^2 - 1))^2)
    with w = 2 pi offset. The super-Lorentzian diverges on resonance: ask for another line there."""
    t2b = np.asarray(t2b_ms, float) * 1e-3
    x = 2 * math.pi * np.asarray(offset_hz, float) * t2b
    if kind == "lorentzian":
        return t2b / math.pi / (1 + x * x)
    if kind == "gaussian":
        return t2b / math.sqrt(2 * math.pi) * np.exp(-0.5 * x * x)
    if kind != "superlorentzian":
        raise ValueError(f"lineshape in {LINESHAPES}")
    if np.any(x == 0):
        raise ValueError("the super-Lorentzian line is not defined on resonance; use lineshape='gaussian' or 'lorentzian'")
    u, w = np.polynomial.legendre.leggauss(400)
    out = np.zeros(np.shape(x))
    for lo, hi in ((0.0, 1 / math.sqrt(3)), (1 / math.sqrt(3), 1.0)):         # the integrand vanishes at 3 u^2 = 1
        uu, ww = 0.5 * (hi - lo) * u + 0.5 * (hi + lo), 0.5 * (hi - lo) * w
        c = np.abs(3 * uu * uu - 1)
        out = out + np.sum(ww * math.sqrt(2 / math.pi) / c * np.exp(-2 * (np.asarray(x)[..., None] / c) ** 2), -1)
    return t2b * out


def saturation(op, sp, cache=None):
    """Factor by which the rf operation multiplies the longitudinal magnetisation of the bound pool (None = none)."""
    energy = op.block.energy_ut2_ms
    if sp.pool != "mt" or not energy:
        return None
    key = ("saturation", energy, op.block.offset_hz)
    if cache is not None and key in cache:
        return cache[key]
    g = lineshape(sp.lineshape, op.block.offset_hz, sp.t2b_ms)
    out = np.exp(-math.pi * RAD2_PER_S_PER_UT2_MS * energy * sp.b1 ** 2 * g)
    if cache is not None:
        cache[key] = out
    return out


# ---------------------------------------------------------------------------------------------- free evolution
def exp2(p, q, r, s, t):
    """exp(t [[p, q], [r, s]]) element by element (arrays broadcast; real or complex): (e11, e12, e21, e22)."""
    mu, dl = 0.5 * (p + s), 0.5 * (p - s)
    root = np.sqrt(dl * dl + q * r + 0j)
    if not (np.iscomplexobj(p) or np.iscomplexobj(q) or np.iscomplexobj(r) or np.iscomplexobj(s)):
        root = root.real if np.all((dl * dl + q * r) >= 0) else root
    up, dn = np.exp((mu + root) * t), np.exp((mu - root) * t)
    small = np.abs(root * t) < 1e-7
    safe = np.where(small, 1.0, root)
    sh = np.where(small, np.exp(mu * t) * t, 0.5 * (up - dn) / safe)        # exp(mu t) sinh(root t) / root
    ch = 0.5 * (up + dn)
    return ch + dl * sh, q * sh, r * sh, ch - dl * sh


def diffusion_rate(program, sp):
    """D kappa_i^2 per axis in 1/ms for every column, [axes, columns]; None without diffusion."""
    if program.cycle_mm is None or not np.any(sp.adc_um2_ms):
        return None
    kappa2 = (2 * math.pi / np.asarray(program.cycle_mm, float)) ** 2      # rad^2 / mm^2
    return kappa2[:, None] * (sp.adc_um2_ms * 1e-6)[None, :]               # um^2/ms = 1e-6 mm^2/ms


def lorentzian_spread(n):
    """(x [n], w [n]): an ensemble of static off-resonances that stands for T2': offsets x / (2 pi T2') (in Hz for
    T2' in seconds) with weights w, the n equal-probability midpoints of a Lorentzian of half width 1 / T2' in
    angular frequency. Its free induction decay tends to exp(-t / T2') as n grows (slowly: the tails are heavy)."""
    u = (np.arange(n) + 0.5) / n
    return np.tan(math.pi * (u - 0.5)), np.full(n, 1.0 / n)


def auto_kmax(program, sp, tol=1e-6, cap=1500):
    """Largest lattice index worth keeping, per axis, for a PERIODIC use of the compiled period ``program``.

    Between two pulses (or ideal spoilers) a state is either transverse for the whole interval or longitudinal, and
    only a transverse state changes its order. An interval of duration d whose gradients add up to g cycles therefore
    costs exp(-d / T2) per g orders: a state of order k has decayed at least by exp(-k / (r T2)), r = the largest g / d
    of the period, and orders above -ln(tol) r T2 are dropped. With one cycle per TR this is the rule of
    ``scanner/epg.py::n_states``: ceil(-ln(tol) T2 / TR) + 2 orders. None when an interval with a gradient takes no
    time (then relaxation does not bound the reach: give ``kmax``)."""
    ops = list(program.ops)
    first = next((i for i, op in enumerate(ops) if op.kind == "rf"), None)
    if first is None:
        return (0,) * program.axes
    ops = ops[first:] + ops[:first]
    segments, dur, gain = [], 0.0, [0] * program.axes
    for op in ops + [None]:
        if op is None or op.kind in ("rf", "spoil", "crush"):
            segments.append((dur, tuple(gain)))
            dur, gain = 0.0, [0] * program.axes
        elif op.kind == "free":
            dur += op.ms
            gain = [g + abs(n) for g, n in zip(gain, op.n)]
    out = []
    for i in range(program.axes):
        used = [(d, g[i]) for d, g in segments if g[i]]
        if not used:
            out.append(0)
            continue
        if any(d <= 0 for d, _ in used):
            return None
        rate, most = max(g / d for d, g in used), max(g for _, g in used)
        out.append(int(min(cap, max(8, np.ceil(-math.log(tol) * float(sp.t2_ms.max()) * rate) + 2 * most))) - 1)
    return tuple(out)


__all__ = ["LINESHAPES", "POOLS", "Spins", "auto_kmax", "column_phase", "diffusion_rate", "exp2", "grid", "lineshape", "lorentzian_spread",
           "rotation", "saturation", "selective_rotation", "slice_columns", "spins", "take"]
