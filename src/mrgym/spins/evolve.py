"""Liouville-von Neumann propagation of the density matrix over a block list, and the signal it gives.

The state is the reduced density matrix rho of one ``operators.Sector`` in the eigenbasis of its Hamiltonian; thermal
equilibrium is rho = Fz (one unit of z magnetisation per proton). Every block is a unitary map rho -> U rho U^+ or
a projection onto coherence orders. A sequence is a list of blocks made by the functions below; ``run`` returns the
signal as a list of spectral lines (``Lines``), from which a free induction decay or a spectrum is made.

Blocks (kept close to ``mrgym.design.sequence``: rf, rf_pulse, delay, gradient, spoil, readout)
  rf(flip, phase)        instantaneous pulse about (cos p, sin p, 0) on every site, or only on the sites whose
                         chemical shift lies in ``band_ppm`` (an ideal frequency-selective pulse)
  rf_pulse(b1, dt, ...)  a B1(t) waveform (the arrays of a ``mrgym.design.pulses.Pulse``), optionally under its
                         slice-select gradient for an ensemble of positions along a named axis
  delay(ms)              free precession under the full Hamiltonian
  select(orders)         ideal crusher: keeps the matrix elements of the listed coherence orders
  spoil()                select((0,)): what a strong gradient leaves. It keeps z magnetisation AND zero-quantum
                         coherence, which no gradient dephases in a homonuclear system
  gradient(cycles, axis) explicit dephasing: an ensemble of positions along ``axis`` over which the gradient winds
                         ``cycles`` whole turns; every later block acts on every member
  average(axis)          sums the ensemble of an axis with its weights
  acquire(phase)         the signal from this instant on, as lines

Pathways. A refocusing pulse between two equal crushers keeps the elements that change the sign of their order,
p -> -p; an editing pulse between two opposite crushers keeps p -> p. ``rf`` and ``rf_pulse`` take
``pathway="refocus"`` or ``"keep"`` and apply  sum_p P_(+-p) [U (P_p rho) U^+]  (P_p = projection on order p), which
is what the ensemble average over the crusher gives when it is taken at once. Row E8 of the validation compares it
with the explicit ensemble.

Shaped pulses, two models
  ``product``  every site turns by its own Bloch rotation, computed by ``mrgym.design.rotation.rotate`` at the
               site's frequency (and position), and the coupling acts as if the pulse were instantaneous at its
               centre:  U = F(T/2) [prod_i P_i] F(T/2),  F = the free evolution under the full Hamiltonian,
               P_i = the rotation of site i with the precession at the site's own frequency taken out (half before,
               half after). Exact without coupling; the error is the coupling evolution that the pulse interrupts,
               of the order J x T. Fast: no matrix product per raster sample.
  ``full``     the pulse is integrated under the whole Hamiltonian, sample by sample, with the hard-pulse splitting
               of ``rotation.rotate(model="split")``: half the free evolution, the sub-pulse, the other half.
               Second order in the raster. Cost: one matrix product per sample and position.
  The carrier of a pulse may differ from the reference (``carrier_hz``); its phase is referred to the CENTRE of
  the pulse, so a spin on the carrier is refocused without a phase. Timing: a pulse of duration T occupies T; the
  sequence builders shorten the delays around it by T / 2.

Detection. s(t) = i exp(-i receiver phase) Tr(rho(t) F+) / 2^(n-2): the +i makes the signal of a pulse about +x
real and positive, as the engine reports it. In the eigenbasis  Tr(rho(t) F+) = sum rho[r, s] F+[s, r]
exp(2 pi i (E_s - E_r) t): one line of frequency E_s - E_r per pair of levels, grouped by the relaxation group of
the site that carries F+.

Relaxation is NOT part of the propagation. It is applied to the lines (``Lines.weighted``, ``Lines.fid``): T2 of a
line's group as exp(-t / T2) during acquisition and as a factor exp(-(time spent transverse) / T2) before it, T1 as
a factor for a mixing time and for incomplete recovery between repetitions. That is exact for uncoupled spins and
ideal pulses (rows E9 to E9c), and for a coupled network whose coherences all decay with one rate; it leaves out
the different rates of antiphase and multiple-quantum terms (zero-quantum coherence in a mixing time is given T1),
relaxation during a shaped pulse, cross-relaxation and cross-correlation.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..design import rotation
from ..design.units import HZ_PER_MT_M_MM, RAD_PER_UT_MS
from . import operators

KINDS = ("rf", "rf_pulse", "delay", "select", "gradient", "average", "acquire")
PATHWAYS = (None, "refocus", "keep")
MODELS = ("product", "full")


@dataclasses.dataclass(frozen=True, eq=False)
class Block:
    kind: str
    flip_rad: float = 0.0
    phase_rad: float = 0.0            # rf, rf_pulse: pulse phase; acquire: receiver phase
    band_ppm: tuple = None            # rf: sites the pulse acts on
    pathway: str = None
    ms: float = 0.0
    orders: tuple = None
    cycles: float = 0.0
    axis: str = None
    n: int = 0
    b1_ut: np.ndarray = None
    dt_ms: np.ndarray = None
    g_mt_m: np.ndarray = None
    carrier_ppm: float = None
    pos_mm: np.ndarray = None
    weights: np.ndarray = None
    model: str = "product"
    b1_scale: float = 1.0
    close: bool = True                # rf_pulse with an axis: sum the ensemble after the pulse


def rf(flip_deg, phase_deg=0.0, band_ppm=None, pathway=None):
    if pathway not in PATHWAYS:
        raise ValueError(f"pathway must be one of {PATHWAYS}")
    return Block("rf", flip_rad=math.radians(flip_deg), phase_rad=math.radians(phase_deg), pathway=pathway,
                 band_ppm=None if band_ppm is None else (float(min(band_ppm)), float(max(band_ppm))))


def rf_pulse(b1_ut, dt_ms, g_mt_m=None, phase_deg=0.0, carrier_ppm=None, axis=None, pos_mm=None, weights=None,
             pathway=None, model="product", b1_scale=1.0, close=True):
    """A waveform. ``b1_ut`` complex [n] (uT), ``dt_ms`` a number or [n], ``g_mt_m`` the gradient during the pulse
    (None, a number or [n]); a ``mrgym.design.pulses.Pulse`` p is passed as (p.b1_ut, p.dt_ms, p.g_mt_m).
    ``carrier_ppm`` the frequency the pulse is played at (None: the reference). With ``axis`` and ``pos_mm`` the
    pulse acts on an ensemble of positions (mm from the centre of the slice), summed afterwards with ``weights``
    (default: spacing of the positions, so that the sum is an integral in mm; give spacing / thickness for the
    signal relative to an ideal slice) unless ``close`` is False (the next pulse on the same axis then meets the
    same ensemble)."""
    if pathway not in PATHWAYS or model not in MODELS:
        raise ValueError(f"pathway must be one of {PATHWAYS} and model one of {MODELS}")
    b1 = np.asarray(b1_ut, complex)
    dt = np.array(np.broadcast_to(np.asarray(dt_ms, float), b1.shape))
    g = np.array(np.broadcast_to(np.zeros(()) if g_mt_m is None else np.asarray(g_mt_m, float), b1.shape))
    if b1.ndim != 1 or b1.size == 0 or not np.all(dt > 0):
        raise ValueError("b1_ut must be a one-dimensional array and dt_ms positive")
    if (axis is None) != (pos_mm is None):
        raise ValueError("give axis and pos_mm together, or neither")
    if pos_mm is not None:
        pos_mm = np.asarray(pos_mm, float).ravel()
        if weights is None:
            weights = np.gradient(pos_mm) if pos_mm.size > 1 else np.ones(1)
        weights = np.broadcast_to(np.asarray(weights, float), pos_mm.shape).copy()
    return Block("rf_pulse", phase_rad=math.radians(phase_deg), pathway=pathway, b1_ut=b1, dt_ms=dt, g_mt_m=g, carrier_ppm=carrier_ppm,
                 axis=axis, pos_mm=pos_mm, weights=weights, model=model, b1_scale=float(b1_scale), close=bool(close))


def rf_design(pulse, rephase=True, **args):
    """``rf_pulse`` for a ``mrgym.design.pulses.Pulse``. With ``rephase`` the rephasing lobe of an excitation pulse
    is appended as one sample without RF, 1e-6 ms long, whose gradient has the area of the lobe: as in the design
    layer, no time passes in it. The other arguments are those of ``rf_pulse``."""
    b1, dt, g = np.asarray(pulse.b1_ut), np.asarray(pulse.dt_ms), np.asarray(pulse.g_mt_m)
    if rephase and pulse.rephase_mt_ms_m:
        b1, dt, g = np.append(b1, 0.0), np.append(dt, 1e-6), np.append(g, pulse.rephase_mt_ms_m / 1e-6)
    return rf_pulse(b1, dt, g, **args)


def delay(ms):
    if not ms >= -1e-9:
        raise ValueError(f"a delay of {ms} ms: the pulses do not fit into the timing")
    return Block("delay", ms=max(float(ms), 0.0))


def select(orders):
    return Block("select", orders=tuple(int(p) for p in np.atleast_1d(orders)))


def spoil():
    return select((0,))


def gradient(cycles, axis="g", n=32):
    """Dephasing by ``cycles`` turns across the ensemble of ``axis``; the first gradient of an axis creates the
    ensemble of ``n`` positions k / n. The selection is exact when n exceeds every sum of order x cycles that the
    sequence can form."""
    return Block("gradient", cycles=float(cycles), axis=str(axis), n=int(n))


def average(axis):
    return Block("average", axis=str(axis))


def acquire(receiver_phase_deg=0.0):
    return Block("acquire", phase_rad=math.radians(receiver_phase_deg))


# ---------------------------------------------------------------------------------------------- propagators
def _dagger(u):
    return np.conj(np.swapaxes(u, -1, -2))


def sandwich(u, rho):
    return u @ rho @ _dagger(u)


def free_phase(sec, ms):
    """exp(-2 pi i E t) of every eigenstate for a time ``ms``."""
    return np.exp(-2e-3j * np.pi * sec.energy_hz * ms)


def free(sec, rho, ms):
    """Free precession: rho[r, s] exp(-2 pi i (E_r - E_s) t)."""
    ph = free_phase(sec, ms)
    return rho * (ph[:, None] * np.conj(ph)[None, :])


def z_rotate(sec, rho, angle_rad):
    """Rotation about z by ``angle_rad`` (a number, or an array whose shape matches the leading shape of rho)."""
    return rho * np.exp(-1j * np.asarray(angle_rad, float)[..., None, None] * sec.order)


def select_orders(sec, rho, orders):
    return rho * np.isin(sec.order, orders)


def hard_propagator(sec, compiled, flip_rad, phase_rad, band_ppm=None):
    shifts = np.asarray(compiled.system.shifts_ppm)
    on = np.ones(shifts.size, bool) if band_ppm is None else (shifts >= band_ppm[0]) & (shifts <= band_ppm[1])
    a = np.where(on, math.cos(flip_rad / 2), 1.0)
    b = np.where(on, -1j * np.exp(1j * phase_rad) * math.sin(flip_rad / 2), 0.0)
    return sec.rotation(a, b)


def pulse_propagator(sec, compiled, b1_ut, dt_ms, g_mt_m=None, pos_mm=None, carrier_hz=0.0, phase_rad=0.0,
                     model="product", b1_scale=1.0):
    """Propagator of a waveform in the eigenbasis: [d, d], or [positions, d, d] with ``pos_mm``."""
    b1 = np.asarray(b1_ut, complex) * np.exp(1j * phase_rad)
    dt = np.broadcast_to(np.asarray(dt_ms, float), b1.shape)
    g = np.broadcast_to(np.zeros(()) if g_mt_m is None else np.asarray(g_mt_m, float), b1.shape)
    pos = np.zeros(1) if pos_mm is None else np.asarray(pos_mm, float).ravel()
    total = float(dt.sum())
    if model == "product":
        a = np.empty((pos.size, compiled.freq_hz.size), complex)
        b = np.empty_like(a)
        for i, f in enumerate(compiled.freq_hz):
            a[:, i], b[:, i] = rotation.rotate(b1, dt, g, pos, f - carrier_hz, b1_scale)
        # the rotation without the precession of the site's own frequency: that part belongs to the free evolution
        a = a * np.exp(1j * np.pi * 1e-3 * (compiled.freq_hz[None, :] - carrier_hz) * total)
        half = free_phase(sec, total / 2)
        u = half[:, None] * sec.rotation(a, b) * half[None, :]
    elif model == "full":
        u = _full_propagator(sec, b1, dt, g, pos, carrier_hz, b1_scale)
    else:
        raise ValueError(f"model must be one of {MODELS}")
    return u[0] if pos_mm is None else u


def _full_propagator(sec, b1, dt, g, pos, carrier_hz, b1_scale):
    """Hard-pulse splitting in the frame of the carrier: prod_j [F_j^(1/2) R_j F_j^(1/2)], F_j the free evolution of
    sample j (diagonal in the eigenbasis: energies, carrier offset, gradient), R_j = exp(-i theta_j (Fx cos p_j +
    Fy sin p_j)) from the eigenvectors of Fx. Then back to the frame of the reference."""
    fx = sum(sec.site_operator(i, "x") for i in range(len(sec.js))).real
    mu, w = np.linalg.eigh(fx)
    theta = RAD_PER_UT_MS * np.abs(b1) * dt * b1_scale
    p = np.angle(b1)
    # frequency of every eigenstate at every position, in the frame of the carrier (Hz)
    level = (sec.energy_hz - carrier_hz * sec.m)[None, :]
    slope = HZ_PER_MT_M_MM * pos[:, None] * sec.m[None, :]              # x gradient
    u = np.broadcast_to(np.eye(sec.dim, dtype=complex), (pos.size, sec.dim, sec.dim)).copy()
    half_prev = None
    for j in range(b1.size):
        half = np.exp(-1e-3j * np.pi * (level + g[j] * slope) * dt[j])  # exp(-2 pi i f dt / 2), [positions, d]
        step = half if half_prev is None else half * half_prev
        u = step[:, :, None] * u
        if theta[j] != 0.0:
            zw = np.exp(-1j * p[j] * sec.m)[:, None] * w
            u = ((zw * np.exp(-1j * theta[j] * mu)) @ zw.conj().T) @ u
        half_prev = half
    u = half_prev[:, :, None] * u
    z = np.exp(-1e-3j * np.pi * carrier_hz * float(dt.sum()) * sec.m)
    return z[:, None] * u * z[None, :]


def pathway_sandwich(sec, u, rho, pathway):
    """sum_p P_(sign p) [U (P_p rho) U^+], sign -1 for "refocus" and +1 for "keep"; orders absent from rho are skipped."""
    sign = -1 if pathway == "refocus" else 1
    out = np.zeros(np.broadcast_shapes(u.shape, rho.shape), complex)
    top = int(sec.order.max())
    for p in range(-top, top + 1):
        mask = sec.order == p
        part = rho * mask
        if not np.any(part):
            continue
        out += sandwich(u, part) * (sec.order == sign * p)
    return out


# ---------------------------------------------------------------------------------------------- lines
@dataclasses.dataclass(frozen=True, eq=False)
class Lines:
    """A signal as a sum of lines: s(t) = sum_k amp_k exp(2 pi i freq_k t) x decay of the line's group."""
    freq_hz: np.ndarray
    amp: np.ndarray
    group: np.ndarray                 # index into ``groups``
    groups: tuple

    def weighted(self, factor):
        """Lines with every amplitude multiplied by ``factor``: a number or a dict group name -> number."""
        if isinstance(factor, dict):
            f = np.array([factor[g] for g in self.groups], complex)[self.group] if self.groups else 1.0
        else:
            f = factor
        return Lines(self.freq_hz, self.amp * f, self.group, self.groups)

    def shifted(self, hz):
        return Lines(self.freq_hz + hz, self.amp, self.group, self.groups)

    def fid(self, n, dwell_ms, t2_ms=None, lorentz_hz=0.0, gauss_hz=0.0, t0_ms=0.0):
        """Samples at t0 + k dwell, k = 0 .. n - 1 (ms). ``t2_ms``: None (no decay), a number or a dict by group
        name; the decay runs from t = 0. ``lorentz_hz`` / ``gauss_hz``: full widths at half maximum of a further
        Lorentzian / Gaussian line shape (field inhomogeneity as a line shape): exp(-pi L |t|) exp(-(pi G t)^2 / (4 ln 2))."""
        t = 1e-3 * (t0_ms + dwell_ms * np.arange(int(n)))
        if self.freq_hz.size == 0:
            return np.zeros(int(n), complex)
        if t2_ms is None:
            rate = np.zeros(len(self.groups))
        elif isinstance(t2_ms, dict):
            rate = np.array([1e3 / t2_ms[g] for g in self.groups])
        else:
            rate = np.full(len(self.groups), 1e3 / float(t2_ms))
        z = 2j * np.pi * self.freq_hz - rate[self.group]
        out = np.zeros(int(n), complex)
        for lo in range(0, z.size, 256):                         # powers of exp(z dwell) by doubling: no exp per sample
            zz = z[lo:lo + 256]
            pw = np.exp(zz * t[0])[:, None]
            while pw.shape[1] < n:
                k = pw.shape[1]
                pw = np.concatenate([pw, pw * np.exp(zz * 1e-3 * dwell_ms * k)[:, None]], axis=1)
            out += self.amp[lo:lo + 256] @ pw[:, :int(n)]
        if lorentz_hz or gauss_hz:
            out = out * np.exp(-np.pi * lorentz_hz * np.abs(t) - (np.pi * gauss_hz * t) ** 2 / (4 * math.log(2)))
        return out

    def total(self, group=None):
        """Sum of the amplitudes (the signal at t = 0), of all lines or of one group."""
        if group is None:
            return complex(self.amp.sum())
        return complex(self.amp[self.group == self.groups.index(group)].sum())

    def window(self, lo_hz, hi_hz):
        keep = (self.freq_hz >= lo_hz) & (self.freq_hz <= hi_hz)
        return Lines(self.freq_hz[keep], self.amp[keep], self.group[keep], self.groups)


def concat(parts):
    """One ``Lines`` from several (different metabolites or networks)."""
    groups = tuple(dict.fromkeys(g for p in parts for g in p.groups))
    index = {g: k for k, g in enumerate(groups)}
    remap = [np.array([index[g] for g in p.groups], int)[p.group] if p.freq_hz.size else np.zeros(0, int) for p in parts]

    def cat(xs, dtype):
        return np.concatenate(xs).astype(dtype) if xs else np.zeros(0, dtype)

    return Lines(cat([p.freq_hz for p in parts], float), cat([p.amp for p in parts], complex), cat(remap, int), groups)


def merge(lines, tol_hz=1e-6):
    """Lines of one group closer than ``tol_hz`` summed; amplitudes below 1e-12 of the largest dropped."""
    if lines.freq_hz.size == 0:
        return lines
    key = np.round(lines.freq_hz / tol_hz).astype(np.int64)
    order = np.lexsort((key, lines.group))
    k, g = key[order], lines.group[order]
    start = np.concatenate([[True], (k[1:] != k[:-1]) | (g[1:] != g[:-1])])
    idx = np.cumsum(start) - 1
    amp = np.zeros(idx[-1] + 1, complex)
    np.add.at(amp, idx, lines.amp[order])
    freq, grp = lines.freq_hz[order][start], g[start]
    keep = np.abs(amp) > 1e-12 * max(np.abs(amp).max(), 1e-300)
    return Lines(freq[keep], amp[keep], grp[keep], lines.groups)


def detect(sec, compiled, rho, receiver_phase_rad=0.0):
    """Lines of the signal that ``rho`` (no ensemble dimension) gives from now on."""
    scale = 1j * np.exp(-1j * receiver_phase_rad) * sec.weight * compiled.system.scale / compiled.norm
    freq, amp, grp = [], [], []
    for k, (r, s, coef, f) in enumerate(sec.detect):
        freq.append(f)
        amp.append(scale * rho[r, s] * coef)
        grp.append(np.full(f.size, k))
    return Lines(np.concatenate(freq), np.concatenate(amp), np.concatenate(grp).astype(int), compiled.groups)


def magnetisation(sec, compiled, rho, site):
    """(Mxy, Mz) of one site per proton of the site, summed over nothing: of this sector only, weight included."""
    c = sec.weight / (compiled.norm * compiled.system.counts[site])
    return c * np.trace(rho @ sec.site_operator(site, "+")), (c * np.trace(rho @ sec.site_operator(site, "z"))).real


# ---------------------------------------------------------------------------------------------- the runner
class _State:
    """rho with ensemble dimensions: shape [n_axis_1, ..., d, d]."""

    def __init__(self, sec):
        self.sec, self.rho, self.axes = sec, sec.fz(), []       # axes: [name, coordinates, weights]

    def _dim(self, name):
        return [a[0] for a in self.axes].index(name)

    def open(self, name, coords, weights):
        if name in [a[0] for a in self.axes]:
            if len(self.axes[self._dim(name)][1]) != len(coords):
                raise ValueError(f"the ensemble of axis {name!r} is already open with another size")
            return
        self.axes.append([name, np.asarray(coords, float), np.asarray(weights, float)])
        d = self.sec.dim
        self.rho = np.repeat(self.rho.reshape(self.rho.shape[:-2] + (1, d, d)), len(coords), axis=-3)

    def along(self, name, values):
        """``values`` [n, ...] laid along the axis ``name`` so that it broadcasts against rho."""
        k, extra = self._dim(name), values.ndim - 1
        shape = [1] * len(self.axes)
        shape[k] = values.shape[0]
        return values.reshape(tuple(shape) + values.shape[1:]) if extra else values.reshape(tuple(shape))

    def close(self, name):
        k = self._dim(name)
        w = self.axes[k][2]
        self.rho = np.tensordot(w, self.rho, axes=([0], [k]))
        del self.axes[k]


def run_sector(sec, compiled, blocks):
    """Lines of every ``acquire`` block for one sector."""
    st, out = _State(sec), []
    hz_per_ppm = operators.larmor_hz_per_ppm(compiled.b0_t)
    for blk in blocks:
        if blk.kind == "delay":
            if blk.ms:
                st.rho = free(sec, st.rho, blk.ms)
        elif blk.kind == "rf":
            u = hard_propagator(sec, compiled, blk.flip_rad, blk.phase_rad, blk.band_ppm)
            st.rho = sandwich(u, st.rho) if blk.pathway is None else pathway_sandwich(sec, u, st.rho, blk.pathway)
        elif blk.kind == "rf_pulse":
            carrier = 0.0 if blk.carrier_ppm is None else (blk.carrier_ppm - compiled.ref_ppm) * hz_per_ppm
            u = pulse_propagator(sec, compiled, blk.b1_ut, blk.dt_ms, blk.g_mt_m, blk.pos_mm, carrier, blk.phase_rad, blk.model, blk.b1_scale)
            if blk.axis is not None:
                st.open(blk.axis, blk.pos_mm, blk.weights)
                u = st.along(blk.axis, u)
            st.rho = sandwich(u, st.rho) if blk.pathway is None else pathway_sandwich(sec, u, st.rho, blk.pathway)
            if blk.axis is not None and blk.close:
                st.close(blk.axis)
        elif blk.kind == "select":
            st.rho = select_orders(sec, st.rho, blk.orders)
        elif blk.kind == "gradient":
            st.open(blk.axis, np.arange(blk.n) / blk.n, np.full(blk.n, 1.0 / blk.n))
            angle = 2 * np.pi * blk.cycles * st.axes[st._dim(blk.axis)][1]
            st.rho = st.rho * np.exp(-1j * st.along(blk.axis, angle)[..., None, None] * sec.order)
        elif blk.kind == "average":
            st.close(blk.axis)
        elif blk.kind == "acquire":
            rho = st.rho
            for k in range(len(st.axes) - 1, -1, -1):
                rho = np.tensordot(st.axes[k][2], rho, axes=([0], [k]))
            out.append(detect(sec, compiled, rho, blk.phase_rad))
        else:
            raise ValueError(f"unknown block kind {blk.kind!r}")
    return out, st


def run(compiled, blocks):
    """Lines of every ``acquire`` block (a list), summed over the sectors of the compiled system."""
    per_sector = [run_sector(sec, compiled, blocks)[0] for sec in compiled.sectors]
    return [merge(concat([p[k] for p in per_sector])) for k in range(len(per_sector[0]))] if per_sector and per_sector[0] else []


def final_state(compiled, blocks):
    """The density matrix after the blocks (ensembles summed) for a system with one sector: for tests of operators."""
    if len(compiled.sectors) != 1:
        raise ValueError("final_state needs a system with one sector (compile with composite=False)")
    st = run_sector(compiled.sectors[0], compiled, blocks)[1]
    rho = st.rho
    for k in range(len(st.axes) - 1, -1, -1):
        rho = np.tensordot(st.axes[k][2], rho, axes=([0], [k]))
    return rho


# ---------------------------------------------------------------------------------------------- spectrum
def spectrum(fid, dwell_ms, b0_t=None, ref_ppm=4.7, zero_fill=1, lb_hz=0.0, gb_hz=0.0, phase0_deg=0.0, phase1_ms=0.0):
    """Spectrum of a free induction decay sampled from t = 0.

    Returns (axis, S): the axis in Hz relative to the reference, ascending, or in ppm when ``b0_t`` is given
    (ppm = ref_ppm + Hz / (1e-6 gamma B0); still ascending: plot it reversed, by the convention of spectroscopy).
        S(f) = dwell [s] x sum_k w_k s_k exp(-2 pi i f t_k),   w_0 = 1/2, the other w_k = apodisation
    so that a line  A exp(2 pi i f0 t - t / T2)  appears at +f0 with a positive real (absorption) part of height
    A T2, full width 1 / (pi T2) and area A / 2 (the area over Hz of the real part is half the first point of the
    decay). ``lb_hz`` / ``gb_hz``: Lorentzian / Gaussian apodisation, the full width each adds to a line.
    ``zero_fill``: the decay is extended with zeros to zero_fill x n points. Phasing: S is multiplied by
    exp(-i (phase0 + 2 pi f phase1)), which removes a constant phase ``phase0_deg`` and a delay ``phase1_ms`` of the
    first sample."""
    s = np.asarray(fid, complex).copy()
    n = s.size
    t = 1e-3 * dwell_ms * np.arange(n)
    if lb_hz or gb_hz:
        s *= np.exp(-np.pi * lb_hz * t - (np.pi * gb_hz * t) ** 2 / (4 * math.log(2)))
    s[0] *= 0.5
    big = int(round(zero_fill * n))
    f = np.fft.fftshift(np.fft.fftfreq(big, 1e-3 * dwell_ms))
    spec = np.fft.fftshift(np.fft.fft(s, big)) * 1e-3 * dwell_ms
    spec = spec * np.exp(-1j * (math.radians(phase0_deg) + 2 * np.pi * f * 1e-3 * phase1_ms))
    if b0_t is not None:
        f = ref_ppm + f / operators.larmor_hz_per_ppm(b0_t)
    return f, spec


def lorentzian_spectrum(lines, f_hz, t2_ms=None, lorentz_hz=0.0):
    """The continuous spectrum of the lines at the frequencies ``f_hz`` (no sampling, no truncation):
    sum_k amp_k / (R_k + 2 pi i (f - f_k)),  R_k = 1 / T2 + pi x lorentz_hz. Same scale as ``spectrum``."""
    f_hz = np.asarray(f_hz, float)
    if t2_ms is None:
        rate = np.zeros(len(lines.groups))
    elif isinstance(t2_ms, dict):
        rate = np.array([1e3 / t2_ms[g] for g in lines.groups])
    else:
        rate = np.full(len(lines.groups), 1e3 / float(t2_ms))
    r = rate[lines.group] + np.pi * lorentz_hz
    if np.any(r <= 0):
        raise ValueError("a continuous spectrum needs a positive line width")
    out = np.zeros(f_hz.shape, complex)
    for lo in range(0, lines.freq_hz.size, 256):
        out += (lines.amp[lo:lo + 256, None] / (r[lo:lo + 256, None] + 2j * np.pi * (f_hz[None, :] - lines.freq_hz[lo:lo + 256, None]))).sum(axis=0)
    return out
