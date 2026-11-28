"""Image formation in k-space for a 2D Cartesian readout (line by line, or single-shot EPI).

The hidden world lives on a master grid that never changes with the protocol. A scan computes the Fourier integral of
the magnetisation at the k-space points the protocol prescribes,

    S(k_pe, k_ro) = dA dz * sum_r  m(r, t(k)) exp(-2 pi i (k_pe y + k_ro x)),      k = index / FOV,

so the field of view, matrix, bandwidth, phase-encode direction, partial Fourier and oversampling act where they act
on a scanner: an object larger than the phase FOV wraps, truncation rings, and everything that evolves during the
readout is displaced or blurred along the axis that is traversed slowly.

How the time dependence m(r, t) is made cheap (t = TE + tau, tau = time of the sample relative to the echo centre):
  1. Tissue is piecewise constant, so decay and chemical shift are functions of the CLASS: g_c(tau). The matrix
     [class, sample] is compressed by an SVD into a few terms  g_c(tau) = sum_l coef[c, l] basis_l(tau); that many
     images are transformed and weighted by basis_l along the time axis. T2* blurring, the water-fat shift, the
     in / opposed-phase behaviour of a multi-peak fat spectrum and the EPI fat displacement all come out of this.
  2. The off-resonance field is smooth in space, not constant per class: exp(2 pi i df(r) tau) with tau proportional
     to k is a DISPLACEMENT of each spin by df / (bandwidth per pixel) pixels. It is applied to the master grid as a
     mass-conserving Kaiser-Bessel splat, so stretching, compression and pile-up are in the data. Along the readout
     for line-by-line scans; along the phase direction (bandwidth 1 / (lines * echo spacing)) for EPI.
  3. The non-commensurate Fourier sums are chirp-z transforms (scipy.signal.ZoomFFT), exact for the master grid.
Noise is added per k-space sample with variance proportional to the receiver bandwidth; the SNR laws then follow.

Per-class echo amplitude (``Engine.amplitudes``), at the local flip angle (nominal * B1+ * transmit gain):
  spgr, afi   tier 2: EPG steady state with RF spoiling, finite T2 and diffusion (``epg``), evaluated on a few
              Chebyshev nodes of the flip angle and interpolated;  tier 1: the ideal-spoiling closed forms
  se, ir      longitudinal steady state of the pulse chain, refocusing efficiency sin^2(beta / 2)
  bssfp       closed form with the local off-resonance;  epi  Ernst steady state
Slice-selective excitation (``rf_tbw`` of the realised protocol is a number; the default). The pulse is a windowed
sinc (``rf``); the flip angle and the transverse phase vary through the slice, and the amplitude served is the
integral of the sequence's signal over independent columns through the slice (``rf.quadrature``). The slice gradient
is calibrated per pulse: the full width at half maximum of the small-flip-angle profile is the prescribed thickness.
The local flip angle above is then the flip angle at the slice centre. ``rf_tbw`` None (``excitation:
nonselective``) is the hard pulse: one flip angle per voxel, the signal of a 3D acquisition.

Fat saturation (``fat_sat``). A spectrally selective pulse ``fatsat_delay_ms`` before every excitation multiplies the
longitudinal magnetisation of every resonance LINE by q = 1 - efficiency x band(line frequency - pulse centre): the
six lines of fat separately, water as one line. Each line then has its own longitudinal steady state, driven by the
saturation in every repetition: Mz = (beta0 + q beta1) / (1 - q alpha) at the excitation (``signal.mz_saturated``),
so what is left depends on TR, T1 and the flip angle, and a line inside the band recovers for the delay. In the
RF-spoiled steady state the saturation multiplies the longitudinal states of every order (``epg``). A class whose
lines are saturated differently is imaged line by line: each line has its own amplitude map and its own evolution.

Adjustments of the realised protocol: ``cf_shift_hz`` (the scanner's centre frequency is raised by it: every spin is
that much less off resonance, for the displacement, the echo phase, the bSSFP bands and the fat-saturation band),
``shim_y_hz_per_mm`` / ``shim_x_hz_per_mm`` (a linear field in patient coordinates added to the off-resonance) and
``tx_scale`` (a factor on the transmit amplitude: every excitation and refocusing angle).

Reconstruction. A fully sampled scan: the inverse FFT. Partial Fourier: homodyne detection (Noll 1991). The lines
acquired on both sides of the centre weigh 1, the lines that exist on one side only weigh 2 (they stand in for their
missing conjugates), and the image is the part of that transform that is in phase with a low-resolution image made
from the symmetric lines. Where the phase of the object is smooth the resolution is the nominal one; the noise of a
pixel rises with the sum of the squared weights (``Engine.noise_sd``), and the phase of the image is the
low-resolution phase.

Field of view. The centre of the field of view moves the chirp-z transforms; a rotation re-evaluates the world on
turned axes (``world.turned``) and images that: no interpolation anywhere. Periodic motion (``world["motion"]``)
modulates the k-space lines of one tissue class with the time at which each line is acquired.

Sequences of the registry (``console.REGISTRY``). The six above have a closed form (``CLOSED_FORMS``) and are formed
by ``Engine._form_steady``: one amplitude map per class, the same on every phase-encode line. A sequence whose entry
says ``serves = "core"`` (fse, mese, megre, dwse; a console offers them when its card lists them) is formed by
``scanner/served.py``: amplitudes from its block list through ``mrgym.core``, and for echo trains an amplitude that
changes from line to line (``FORMATION``).

Frame of the acquisition: arrays [pe, ro]. Master arrays are [y, x]; pe_dir = 'x' transposes them.
Sign convention: a spin that precesses df faster than the reference gains positive phase; with ro_polarity = +1 it is
displaced by -df / bandwidth_hz_per_px pixels along the readout axis (fat, df < 0, moves towards +readout).
"""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.signal
import scipy.sparse as sp

from . import console, epg, rf, rfscan, tissue
from . import signal as S
from . import world as W

ALL = frozenset({"t2star", "chemshift", "b0shift", "dropout", "noise", "rfi", "ghost", "aa_filter", "motion"})
GAMMA_MHZ = S.GAMMA_MHZ_PER_T
GAMMA_RAD = 2 * np.pi * console.GAMMA_HZ_PER_T
KB_W, KB_BETA = 4, 8.996            # Kaiser-Bessel kernel for a grid >= 2x finer than the acquired pixel (Beatty 2005)
SVD_TOL = 1e-4
PROFILE_NODES = 16                  # Chebyshev nodes of the flip angle on which a steady state is tabulated for the slice integral
PF_COARSE, PF_BLEND = 3, 0.02       # homodyne phase reference: lines either side of the coarse image, and its weight
SAT_EDGE_HZ = 8.0                   # softness of the edges of the fat-saturation band
SAT_NODES = 13                      # nodes of the saturation factor q on which the EPG's departure from ideal spoiling is tabulated
SAT_GRID = (33, 8193)               # points of q on which the slice integral of a saturated line is tabulated (fewest, most)
TURNED_ENGINES = 3                  # engines of turned fields of view kept per engine


def cheb_nodes(lo, hi, m):
    """Chebyshev points of the first kind on [lo, hi] and their barycentric weights."""
    t = np.pi * (np.arange(m) + 0.5) / m
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(t), (-1.0) ** np.arange(m) * np.sin(t)


def bary(nodes, weights, values, x):
    """Barycentric polynomial interpolation of ``values`` (given on ``nodes``) at ``x``. Deterministic by
    construction (no random node permutation), which replay needs."""
    d = x[:, None] - nodes[None, :]
    hit = d == 0
    d[hit] = 1.0
    q = weights / d
    out = (q @ values) / q.sum(axis=1)
    rows, cols = np.nonzero(hit)
    out[rows] = values[cols]
    return out


def lobatto_nodes(lo, hi, m):
    """Chebyshev points of the second kind on [lo, hi] (both ends are nodes) and their barycentric weights."""
    t = np.pi * np.arange(m) / (m - 1)
    w = (-1.0) ** np.arange(m)
    w[[0, -1]] *= 0.5
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(t), w


def hermite(x0, h, f, df, x):
    """Piecewise cubic through the values ``f`` and slopes ``df`` ([..., n] on the uniform grid x0 + h * arange(n)),
    evaluated at ``x`` [N]: [..., N]."""
    i = np.clip(np.floor((x - x0) / h).astype(np.int64), 0, f.shape[-1] - 2)
    t = (x - x0) / h - i
    return (f[..., i] * ((1 + 2 * t) * (1 - t) ** 2) + h * df[..., i] * (t * (1 - t) ** 2)
            + f[..., i + 1] * (t * t * (3 - 2 * t)) + h * df[..., i + 1] * (t * t * (t - 1)))


def sat_band(off_hz, bw_hz):
    """Share of the saturation that a line ``off_hz`` from the centre of the pulse receives: 1 inside the band of
    width ``bw_hz``, 0 outside, edges ``SAT_EDGE_HZ`` soft."""
    return 0.5 * (1 - np.tanh((np.abs(off_hz) - bw_hz / 2) / SAT_EDGE_HZ))


def sat_delay_ms(rp):
    """Centre of the saturation pulse to centre of the excitation of a realised protocol: what the console planned
    (``fatsat_delay_ms``), or the same rule from the console's constants for a protocol that was written by hand."""
    if rp.get("fatsat_delay_ms") is not None:
        return rp["fatsat_delay_ms"]
    return console.SYSTEM["fatsat_ms"] / 2 + console.SYSTEM["fatsat_gap_ms"] + rp.get("rf_ms", console.SYSTEM["rf_ms"]) / 2


def bary_matrix(nodes, weights, x):
    """[len(x), len(nodes)] matrix Q of the barycentric interpolation: values at ``x`` = Q @ values on ``nodes``."""
    d = x[:, None] - nodes[None, :]
    hit = d == 0
    d[hit] = 1.0
    q = weights / d
    q /= q.sum(axis=1, keepdims=True)
    rows, cols = np.nonzero(hit)
    q[rows] = 0.0
    q[rows, cols] = 1.0
    return q


def diffusion_bd(adc_um2_ms, spoiler_mt_ms_per_m, tr_ms):
    """Dimensionless D * dk^2 * TR for the EPG: D in um^2/ms (= 1e-9 m^2/s), spoiler area in mT*ms/m, TR in ms."""
    return adc_um2_ms * 1e-9 * (GAMMA_RAD * spoiler_mt_ms_per_m * 1e-6) ** 2 * tr_ms * 1e-3


def gre_steady(rp, cls, theta, tier=2, nodes=None, sat=None):
    """Steady state of spgr / afi / epi at flip angles ``theta`` (rad, any shape): list over contrasts, complex.
    ``sat`` = (q, delay_ms): a saturation Mz -> q Mz that long before every excitation (spgr, epi).

    Tier 2 (spgr, afi): the EPG steady state, solved on Chebyshev nodes and interpolated. ``nodes`` None: 5 to 10
    nodes that span ``theta`` (a narrow range: one tissue under a smooth B1 pattern). ``nodes`` = m: m nodes on
    [0, max(theta)] for the flip angles through a slice profile; there the difference between the EPG and the
    ideal-spoiling closed form is interpolated, which is smooth where the signal itself peaks sharply at small angles."""
    seq = rp["sequence"]
    if seq == "afi":
        ideal = lambda t: [np.asarray(v, complex) for v in S.afi(t, rp["tr_ms"], rp["tr2_ms"], cls["t1_ms"])]   # noqa: E731
    else:
        ideal = lambda t: [S.spgr(t, rp["tr_ms"], cls["t1_ms"], sat).astype(complex)]   # noqa: E731
    if seq == "epi" or tier != 2:
        return ideal(theta)
    lo, hi = float(theta.min()), float(theta.max())
    if nodes is None:
        pad = 0.01 * (hi - lo) + 1e-4
        tn, tw = cheb_nodes(lo - pad, hi + pad, int(np.clip(np.ceil(30 * (hi - lo) / (hi + lo)) + 4, 5, 10)))
    else:
        tn, tw = cheb_nodes(0.0, 1.02 * hi + 1e-3, nodes)
    bd = diffusion_bd(cls["adc_um2_ms"], rp["spoiler_mt_ms_per_m"], rp["tr_ms"])
    if seq == "spgr":
        vals = [epg.spgr(tn, rp["tr_ms"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"], bd, sat)]
    else:
        vals = epg.afi(tn, rp["tr_ms"], rp["tr_ratio"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"], bd)
    if nodes is None:
        return [bary(tn, tw, np.asarray(v), theta.ravel()).reshape(theta.shape) for v in vals]
    return [i + bary(tn, tw, np.asarray(v) - j, theta.ravel()).reshape(theta.shape)
            for v, i, j in zip(vals, ideal(theta), ideal(tn))]


def slice_profile(centre_flip, tbw, window="hann", refocus=False):
    """Tables of a slice-selective pulse for flip angles ``centre_flip`` (rad, 1D) at the slice centre, on the
    positions of ``rf.quadrature(tbw)``: ``theta`` [flip, position], ``w`` = quadrature weight x real part of the
    transverse phase factor (the two sides of the slice added), and with ``refocus`` the echo factor ``-b^2`` of the
    refocusing pulse of ``se`` / ``ir`` (centre angle 2 x the excitation's, real and >= 0 for a symmetric pulse)."""
    zeta, wz = rf.quadrature(tbw, window)
    theta, ph = rf.excitation(centre_flip[:, None], tbw, zeta[None, :], window)
    out = dict(theta=theta, w=wz * ph.real)
    if refocus:
        b = rf.refocusing(2 * centre_flip[:, None], tbw, zeta[None, :], window)
        out["echo"] = (-b * b).real
    return out


def cycle_events(rp, theta, refocus=None, inv_eff=1.0):
    """(events of one repetition as ``signal.mz_chain`` takes them, starting at the excitation; factor that turns the
    longitudinal magnetisation at the excitation into the echo) for flip angles ``theta`` under ideal spoiling.
    ``refocus``: echo factor of the refocusing pulse of se / ir through the slice (None: a hard pulse of 2 theta)."""
    seq = rp["sequence"]
    if seq in ("spgr", "epi"):
        return S.spgr_events(theta, rp["tr_ms"]), np.sin(theta)
    # se, ir: excitation theta; crushed refocusing pulse: echo = -b^2 conj(Mxy), Mz -> (1 - 2 |b|^2) Mz
    echo = np.sin(theta) ** 2 if refocus is None else refocus
    if seq == "se":
        return S.se_events(theta, rp["tr_ms"], rp["te_ms"], 1 - 2 * echo), np.sin(theta) * echo
    return S.ir_events(theta, rp["tr_ms"], rp["te_ms"], rp["ti_ms"], inv_eff, 1 - 2 * echo), np.sin(theta) * echo


def slice_table(rp, cls, prof, tier=2, inv_eff=1.0, sat=None):
    """Slice-integrated signal per unit slice thickness for every centre flip angle of ``prof`` (``slice_profile``):
    list over contrasts of complex arrays. Independent columns through the slice, each with its own flip angle and
    transverse phase. Not for bssfp, whose signal also depends on the voxel's off-resonance. ``sat`` = (q, delay_ms):
    a saturation Mz -> q Mz that long before every excitation."""
    seq, theta = rp["sequence"], prof["theta"]
    if seq in ("spgr", "afi", "epi"):
        return [(prof["w"] * v).sum(1) for v in gre_steady(rp, cls, theta, tier, PROFILE_NODES, sat)]
    events, factor = cycle_events(rp, theta, prof["echo"], inv_eff)
    mz = S.mz_steady(events, cls["t1_ms"]) if sat is None else S.mz_saturated(events, cls["t1_ms"], sat[0], sat[1])
    return [(prof["w"] * mz * factor).sum(1).astype(complex)]


def slice_signal(rp, cls, centre_flip, tbw, window="hann", tier=2, inv_eff=1.0, sat=None):
    """``slice_table`` for given centre flip angles (rad, 1D): what the scanner serves for a voxel of one material."""
    centre_flip = np.atleast_1d(np.asarray(centre_flip, float))
    return slice_table(rp, cls, slice_profile(centre_flip, tbw, window, rp["sequence"] in ("se", "ir")), tier, inv_eff, sat)


def zoom_dft(x, axis, n_out, cyc_per_sample, offset=0.0, centre=0.0):
    """sum_j x_j exp(-2 pi i q (j - n // 2 - centre) c) for q = -n_out/2 + offset, ..., n_out/2 - 1 + offset;
    c = cycles/sample, ``centre`` = position of the centre of the field of view in samples from the middle of the grid."""
    n = x.shape[axis]
    f = (np.arange(n_out) - n_out // 2 + offset) * cyc_per_sample
    z = scipy.signal.ZoomFFT(n, [f[0], f[0] + n_out * cyc_per_sample], m=n_out, fs=1.0, endpoint=False)
    out = z(x, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = n_out
    return out * np.exp(2j * np.pi * f * (n // 2 + centre)).reshape(shape)


def _splat_matrix(shift_px, axis, keep):
    """Sparse operator that moves the content of every master pixel by ``shift_px`` (float, pixels) along ``axis``.

    A type-1 non-uniform Fourier transform in one dimension: every source is spread on KB_W grid points with a
    Kaiser-Bessel kernel; the k-space is later divided by the kernel's transform (``_deapodise``). Mass is conserved,
    so compression piles signal up and stretching thins it out.
    """
    n = shift_px.shape[0]
    src = np.flatnonzero(keep.ravel())
    row, col = np.divmod(src, n)
    pos = (col if axis == 1 else row) + shift_px.ravel()[src]
    i0 = np.floor(pos).astype(np.int64) - (KB_W // 2 - 1)
    rows, cols, vals = [], [], []
    for off in range(KB_W):
        i = i0 + off
        u = (i - pos) / (KB_W / 2)
        wgt = np.i0(KB_BETA * np.sqrt(np.clip(1 - u * u, 0, None))) / np.i0(KB_BETA)
        ok = (i >= 0) & (i < n)
        tgt = (row[ok] * n + i[ok]) if axis == 1 else (i[ok] * n + col[ok])
        rows.append(tgt), cols.append(src[ok]), vals.append(wgt[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))


def _deapodise(n_out, cyc_per_sample):
    """1 / (Fourier transform of the Kaiser-Bessel kernel) at the sampled spatial frequencies."""
    nu = (np.arange(n_out) - n_out // 2) * cyc_per_sample
    z = np.sqrt((KB_BETA ** 2 - (np.pi * KB_W * nu) ** 2).astype(complex))
    return np.i0(KB_BETA) / (KB_W * (np.sinh(z) / z).real)


def physics_key(rp, averages=False):
    """The part of a realised protocol that changes the noise-free data: everything but the averages, unless
    something in the world moves (``averages`` True: the lines of each average are then acquired at other times).
    An adjustment at its default is left out: such a protocol has the key it had before the adjustments existed
    (the phase of a periodic mover is a hash of this key)."""
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(rp.items())
                 if (averages or k != "averages") and not (k in console.ADJUSTMENTS and v == console.ADJUSTMENTS[k]))


class Engine:
    """One hidden world behind ``acquire(realised protocol, rng) -> complex images``. Caches the noise-free k-space.

    ``rf``: an excitation pulse other than the one the console plays, for sensitivity studies (``mrgym validate
    --perturb``): dict with ``tbw`` (time-bandwidth product), ``window`` ("hann" or "hamming") or ``profile="rect"``
    (the ideal rectangular profile). It acts on slice-selective protocols only."""

    def __init__(self, world: dict, tier: int = 2, fat_model: str = "multipeak", effects=None, cache: int = 48, rf=None):
        self.world = self.w = world
        self.tier, self.fat_model, self.cache_size = tier, fat_model, cache
        self.effects = frozenset(ALL if effects is None and world.get("effects") is None
                                 else effects if effects is not None else world["effects"])
        self.rf = dict(rf or {})
        self.n, self.d = world["n"], world["extent_mm"] / world["n"]
        self.sup = [wc > 1e-4 for wc in world["weights"]]
        self.body = world["weights"].sum(0) > 1e-4
        self.df = world["fields"]["b0_hz"] + world["system"]["cf_offset_hz"]
        self.b1 = world["fields"]["b1"] * world["system"]["tx_gain"]
        self.b1_range = (float(self.b1[self.body].min()), float(self.b1[self.body].max())) if self.body.any() else (1.0, 1.0)
        self._clean, self._splat, self._turned, self._prof = {}, {}, {}, {}
        self._yx, self._adjusted = None, (None, None)

    # ------------------------------------------------------------------ the adjustments of a scan
    def _df(self, rp):
        """Off-resonance of every master-grid point in the scan ``rp`` (Hz): the hidden field and the error of the
        frequency adjustment, minus the shift of the centre frequency, plus the linear shim terms (patient
        coordinates, so a turned field of view sees the same field)."""
        key = (rp.get("cf_shift_hz", 0.0), rp.get("shim_y_hz_per_mm", 0.0), rp.get("shim_x_hz_per_mm", 0.0))
        if not any(key):
            return self.df
        if self._adjusted[0] != key:
            if self._yx is None:
                self._yx = W.grid(self.w)
            self._adjusted = (key, self.df - key[0] + key[1] * self._yx[0] + key[2] * self._yx[1])
        return self._adjusted[1]

    # ------------------------------------------------------------------ field of view turned in the plane
    def _frame(self, rp):
        """(engine, realised protocol) that image ``rp``: this engine, or for a turned field of view an engine around
        the world re-evaluated on the turned axes, with the centre of the field of view expressed in that frame."""
        rot = float(rp.get("rotation_deg", 0.0) or 0.0)
        if not rot:
            return self, rp
        key = round(rot, 6)
        if key not in self._turned:
            if len(self._turned) >= TURNED_ENGINES:
                self._turned.pop(next(iter(self._turned)))
            self._turned[key] = Engine(W.turned(self.w, rot), tier=self.tier, fat_model=self.fat_model, effects=self.effects,
                                       cache=self.cache_size, rf=self.rf)
        c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
        cy, cx = rp.get("centre_y_mm", 0.0), rp.get("centre_x_mm", 0.0)
        return self._turned[key], dict(rp, rotation_deg=0.0, centre_y_mm=cy * c + cx * s, centre_x_mm=-cy * s + cx * c)

    # ------------------------------------------------------------------ per-class echo amplitude on the master grid
    def pulse(self, rp):
        """(time-bandwidth product, window) of the excitation of ``rp``; (None, None) for a hard pulse."""
        tbw = rp.get("rf_tbw")
        if tbw is None or self.rf.get("profile") == "rect":
            return None, None
        return float(self.rf.get("tbw", tbw)), self.rf.get("window", "hann")

    def _hard(self, rp, cls, a, free):
        """One flip angle ``a`` per voxel (hard pulse; the signal of a 3D acquisition)."""
        seq = rp["sequence"]
        if seq in ("spgr", "afi", "epi"):
            return gre_steady(rp, cls, a, self.tier)
        if seq == "se":
            return [S.se(a, rp["tr_ms"], rp["te_ms"], cls["t1_ms"]).astype(complex)]
        if seq == "ir":
            return [S.ir_se(a, rp["tr_ms"], rp["te_ms"], rp["ti_ms"], cls["t1_ms"], self.w["system"]["inv_eff"]).astype(complex)]
        return [S.bssfp(a, free - np.deg2rad(rp["phase_cycle_deg"]), rp["tr_ms"], cls["t1_ms"], cls["t2_ms"]) * np.exp(0.5j * free)]

    def _profile(self, rp, tbw, window):
        """What the pulse of ``rp`` does through the slice, for flip angles at the slice centre that span the whole
        object (Chebyshev nodes ``gn``): one table per protocol, shared by the tissue classes."""
        tx = rp.get("tx_scale", 1.0)
        key = (rp["flip_deg"], tx, tbw, window, rp["sequence"] in ("se", "ir"))
        if key not in self._prof:
            if len(self._prof) >= 8:
                self._prof.pop(next(iter(self._prof)))
            lo, hi = (np.deg2rad(rp["flip_deg"]) * tx * v for v in self.b1_range)
            pad = 0.01 * (hi - lo) + 1e-4
            gn, gw = cheb_nodes(lo - pad, hi + pad, int(np.clip(np.ceil(30 * (hi - lo) / (hi + lo)) + 4, 5, 16)))
            self._prof[key] = dict(slice_profile(gn, tbw, window, refocus=key[3]), gn=gn, gw=gw)
        return self._prof[key]

    def _bssfp_selective(self, rp, cls, a, free, prof):
        """bSSFP closed form integrated over the slice, per voxel: S = K sin(t) / (p + q cos(t)) with K, p, q functions
        of the voxel's precession per TR and t = theta(z); sin and cos of theta(z) are interpolated over the flip
        angle at the slice centre."""
        s_tab, c_tab = prof["w"] * np.sin(prof["theta"]), np.cos(prof["theta"])
        e1, e2 = np.exp(-rp["tr_ms"] / cls["t1_ms"]), np.exp(-rp["tr_ms"] / cls["t2_ms"])
        th = free - np.deg2rad(rp["phase_cycle_deg"])                    # precession per TR seen from the pulse, whose phase advances
        cth = np.cos(th)
        p, q = e2 * (e1 - 1) * cth + 1 - e1 * e2 * e2, e2 * (e1 - 1) * cth - e1 + e2 * e2
        out = np.empty(a.shape)
        for i in range(0, a.size, 16384):                                # bounded memory: [chunk, positions]
            sl = slice(i, i + 16384)
            qm = bary_matrix(prof["gn"], prof["gw"], a[sl])
            out[sl] = ((qm @ s_tab) / (p[sl, None] + q[sl, None] * (qm @ c_tab))).sum(1)
        return [out * (1 - e1) * (1 - e2 * np.exp(-1j * th)) * np.exp(0.5j * free)]

    # ------------------------------------------------------------------ fat saturation: every line has its own steady state
    def _lines(self, cls):
        """Resonance lines of a class: ((ppm relative to water, share of the signal), ...)."""
        if not cls["fat"]:
            return ((0.0, 1.0),)
        if self.fat_model != "multipeak":
            return ((tissue.FAT_PPM, 1.0),)
        total = sum(a for _, a in tissue.FAT_PEAKS)
        return tuple((ppm, a / total) for ppm, a in tissue.FAT_PEAKS)

    def _spoiling_gap(self, rp, cls, tn, qn, tbw=None, prof=None):
        """[len(tn) or centre angles of ``prof``, len(qn)] table of what the RF-spoiled steady state of spgr adds to the
        ideal-spoiling form under a saturation q (tier 2; zeros otherwise)."""
        delay = sat_delay_ms(rp)
        if rp["sequence"] != "spgr" or self.tier != 2:
            return None
        if prof is not None:
            return np.stack([slice_table(rp, cls, prof, 2, sat=(q, delay))[0] - slice_table(rp, cls, prof, 1, sat=(q, delay))[0] for q in qn], 1)
        bd = diffusion_bd(cls["adc_um2_ms"], rp["spoiler_mt_ms_per_m"], rp["tr_ms"])
        return np.stack([epg.spgr(tn, rp["tr_ms"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"], bd, (q, delay))
                         - S.spgr(tn, rp["tr_ms"], cls["t1_ms"], (q, delay)) for q in qn], 1)

    def _saturated(self, rp, cls, a, qs, tbw, window):
        """Echo amplitudes of the lines of one class under fat saturation: ``a`` [N] flip angle at the slice centre,
        ``qs`` list of [N] saturation factors (one per line). Returns a list of complex [N].

        Ideal spoiling: Mz at the excitation is (beta0 + q beta1) / (1 - q alpha) with the terms of the sequence's
        cycle (``signal.mz_saturated_terms``). Hard pulse: evaluated per voxel. Selective pulse: the terms are
        tabulated on (centre flip angle node, position through the slice); the slice integral is a sum of such
        fractions in q, tabulated with its slope on a grid of q fine against the distance of its nearest pole
        (1 - E1) and interpolated by cubics, then interpolated over the centre flip angle as without saturation.
        spgr, tier 2: the departure of the RF-spoiled steady state from ideal spoiling is added from a table over
        (flip angle node, ``SAT_NODES`` values of q)."""
        t1, delay, inv = cls["t1_ms"], sat_delay_ms(rp), self.w["system"]["inv_eff"]
        q_lo = min(float(q.min()) for q in qs)
        q_lo = min(q_lo, 1.0 - 1e-6)
        qn, qw = lobatto_nodes(q_lo, 1.0, SAT_NODES)
        if tbw is None:
            events, factor = cycle_events(rp, a, None, inv)
            alpha, beta0, beta1 = S.mz_saturated_terms(events, t1, delay)
            lo, hi = float(a.min()), float(a.max())
            pad = 0.01 * (hi - lo) + 1e-4
            tn, tw = cheb_nodes(lo - pad, hi + pad, int(np.clip(np.ceil(30 * (hi - lo) / (hi + lo)) + 4, 5, 10)))
            gap = self._spoiling_gap(rp, cls, tn, qn)
            qa = bary_matrix(tn, tw, a) if gap is not None else None
            out = []
            for q in qs:
                v = (factor * (beta0 + q * beta1) / (1 - q * alpha)).astype(complex)
                if gap is not None:
                    v = v + np.einsum("ri,ij,rj->r", qa, gap, bary_matrix(qn, qw, q))
                out.append(v)
            return out
        prof = self._profile(rp, tbw, window)
        events, factor = cycle_events(rp, prof["theta"], prof.get("echo"), inv)
        alpha, beta0, beta1 = (np.broadcast_to(v, prof["theta"].shape) for v in S.mz_saturated_terms(events, t1, delay))   # [node, position]
        wgt = prof["w"] * factor
        pole = max(1.0 - float(np.max(alpha)), 1e-9)                         # distance of the nearest pole of Mz(q) from q = 1
        n_q = int(np.clip(np.ceil(12 * (1 - q_lo) / pole) + 1, *SAT_GRID))
        qg = np.linspace(q_lo, 1.0, n_q)
        f, df = np.empty((len(prof["gn"]), n_q)), np.empty((len(prof["gn"]), n_q))
        for i in range(len(prof["gn"])):                                     # bounded memory: [q, position]
            den = 1 - qg[:, None] * alpha[i][None, :]
            f[i] = (wgt[i] * (beta0[i] + qg[:, None] * beta1[i]) / den).sum(1)
            df[i] = (wgt[i] * (beta1[i] + alpha[i] * beta0[i]) / den ** 2).sum(1)
        gap = self._spoiling_gap(rp, cls, None, qn, tbw, prof)
        out = []
        for q in qs:
            v = np.empty(a.shape, complex)
            for k in range(0, a.size, 32768):
                sl = slice(k, k + 32768)
                qa = bary_matrix(prof["gn"], prof["gw"], a[sl])
                v[sl] = np.einsum("ri,ir->r", qa, hermite(q_lo, qg[1] - qg[0], f, df, q[sl]))
                if gap is not None:
                    v[sl] += np.einsum("ri,ij,rj->r", qa, gap, bary_matrix(qn, qw, q[sl]))
            out.append(v)
        return out

    def _species(self, rp, c):
        """(lines, amplitudes) of class ``c`` over its support. ``lines`` None: the class evolves as one species and
        the amplitudes are a list over contrasts (afi: 2, otherwise 1) of arrays [support]. Under fat saturation a
        class with several lines that are saturated differently is returned line by line: ``lines`` =
        ((ppm, share), ...) and each amplitude array is [line, support], without the share."""
        w, name = self.w, self.w["names"][c]
        cls, sup = w["tissues"][name], self.sup[c]
        a = np.deg2rad(rp["flip_deg"]) * rp.get("tx_scale", 1.0) * self.b1[sup]      # flip angle at the slice centre
        seq = rp["sequence"]
        scale = w["weights"][c][sup] * cls["pd"]
        tbw, window = self.pulse(rp)
        if rp["fat_sat"]:                                               # spectrally selective saturation, line by line
            sy, f0 = w["system"], GAMMA_MHZ * w["b0_t"]
            lines = self._lines(cls)
            centre = tissue.FAT_PPM * f0 + sy.get("fatsat_offset_hz", 0.0) + rp.get("fatsat_offset_hz", 0.0)      # of the pulse, from water
            df = self._df(rp)[sup]
            qs = [1 - sy["fatsat_eff"] * sat_band(df + ppm * f0 - centre, sy["fatsat_bw_hz"]) for ppm, _ in lines]
            if a.size and min(float(q.min()) for q in qs) < 1 - 1e-12:
                amp = [v * scale for v in self._saturated(rp, cls, a, qs, tbw, window)]
                return (None, [amp[0]]) if len(lines) == 1 else (lines, [np.array(amp)])
        f_fat = tissue.FAT_PPM * GAMMA_MHZ * w["b0_t"] if cls["fat"] else 0.0
        free = 2 * np.pi * (self._df(rp)[sup] + f_fat) * rp["tr_ms"] * 1e-3 if seq == "bssfp" else None
        if tbw is None:
            out = self._hard(rp, cls, a, free)
        elif seq == "bssfp":
            out = self._bssfp_selective(rp, cls, a, free, self._profile(rp, tbw, window))
        else:
            prof = self._profile(rp, tbw, window)
            out = [bary(prof["gn"], prof["gw"], g, a) for g in slice_table(rp, cls, prof, self.tier, w["system"]["inv_eff"])]
        return None, [s * scale for s in out]

    def amplitudes(self, rp, c):
        """Echo amplitudes of class ``c`` over its support: list over contrasts (afi: 2, otherwise 1) of arrays. For a
        fat class under fat saturation: the sum over its lines, each with its share (the amplitude at the centre of a
        spin echo, where the lines are in phase). A sequence served from the block-list core: one array per sample
        of its cycle (every echo of a train), a spin echo with its T2 decay up to the echo."""
        if console.REGISTRY[rp["sequence"]].serves == "core":            # one map per sample of the cycle (``served.amplitudes``)
            return served.amplitudes(self, rp, c)
        lines, out = self._species(rp, c)
        if lines is None:
            return out
        share = np.array([wgt for _, wgt in lines])
        return [np.tensordot(share, o, 1) for o in out]

    # ------------------------------------------------------------------ one contrast -> k-space [lines_full, os * matrix]
    def _shift_operator(self, rp, epi, fov_pe_os):
        swap = rp["pe_dir"] == "x"
        df = self._df(rp)
        df, body = (df.T, self.body.T) if swap else (df, self.body)
        adjusted = (rp.get("cf_shift_hz", 0.0), rp.get("shim_y_hz_per_mm", 0.0), rp.get("shim_x_hz_per_mm", 0.0))
        if epi:
            key = ("epi", swap, rp["pe_polarity"], rp["echo_spacing_ms"], fov_pe_os, adjusted)
            shift, axis = -rp["pe_polarity"] * df * rp["echo_spacing_ms"] * 1e-3 * fov_pe_os / self.d, 0
        else:
            key = ("line", swap, rp["ro_polarity"], rp["bandwidth_hz_per_px"], rp["fov_mm"], rp["matrix"], adjusted)
            shift, axis = -rp["ro_polarity"] * df / rp["bandwidth_hz_per_px"] * (rp["fov_mm"] / rp["matrix"]) / self.d, 1
        if key not in self._splat:
            self._splat = {key: _splat_matrix(shift, axis, body)}        # keep the last one: echoes and contrasts share it
        return self._splat[key], axis

    def kspace(self, rp, te_ms, amps):
        """Noise-free k-space of one image. ``amps``: per-class amplitude arrays (one contrast of ``_species``): [support],
        or [line, support] for a class that is imaged line by line."""
        w, n, d, eff = self.w, self.n, self.d, self.effects
        entry = console.REGISTRY[rp["sequence"]]
        epi, echo = entry.readout == "epi", entry.echo                   # how the lines are read, how a class evolves around the echo
        ro_os = console.SYSTEM["ro_os"]
        k_ro, k_pe = ro_os * rp["matrix"], rp["lines_full"]
        fov_ro_os = ro_os * rp["fov_mm"]
        fov_pe_os = rp["fov_pe_mm"] * k_pe / rp["matrix_pe"]
        # ---- time of every sample along the slow axis of the chemical-shift / decay physics
        if epi:
            tau = rp["pe_polarity"] * (np.arange(k_pe) - k_pe // 2) * rp["echo_spacing_ms"] * 1e-3
        else:
            tau = rp["ro_polarity"] * (np.arange(k_ro) - k_ro // 2) / (k_ro * rp["bandwidth_hz_per_px"])
        peaks = tissue.FAT_PEAKS if self.fat_model == "multipeak" else ((tissue.FAT_PPM, 1.0),)
        t_dec, t_spec = tau if "t2star" in eff else 0 * tau, tau if "chemshift" in eff else 0 * tau
        rows, source = [], []                                            # one row per species: a class, or one line of a class
        for c, nm in enumerate(w["names"]):
            cls = w["tissues"][nm]
            if np.ndim(amps[c]) == 2:
                for (ppm, share), a in zip(self._lines(cls), amps[c]):
                    rows.append(share * S.time_function(cls, t_dec, t_spec, te_ms, echo, w["b0_t"], ((ppm, 1.0),)))
                    source.append((c, a))
            else:
                rows.append(S.time_function(cls, t_dec, t_spec, te_ms, echo, w["b0_t"], peaks))
                source.append((c, amps[c]))
        g = np.array(rows)
        u, sv, vh = np.linalg.svd(g, full_matrices=False)
        rank = max(1, int(np.sum(sv > SVD_TOL * sv[0])))
        coef, basis = u[:, :rank] * sv[:rank], vh[:rank]
        # ---- component images on the master grid
        comp = np.zeros((rank, n, n), complex)
        for row, (c, a) in enumerate(source):
            comp[:, self.sup[c]] += coef[row][:, None] * a[None, :]
        common = w["fields"]["rx"].astype(complex)
        if echo == "gre":
            common = common * np.exp(2j * np.pi * self._df(rp) * te_ms * 1e-3)
            if "dropout" in eff:
                # through-slice dephasing: cycles of phase across one slice thickness at the echo
                cyc = w["fields"]["dbdz_hz_per_mm"] * rp["slice_mm"] * te_ms * 1e-3
                tbw, window = self.pulse(rp)
                if tbw is None:                                          # hard pulse: Gaussian profile, FWHM = thickness
                    common = common * np.exp(-(np.pi * cyc) ** 2 / (4 * np.log(2)))
                else:                                                    # the small-tip profile of the shaped pulse
                    common = common * rf.dephasing_factor(cyc, tbw, window)
        comp *= common
        # ---- displacement by off-resonance
        if rp["pe_dir"] == "x":
            comp = np.ascontiguousarray(comp.transpose(0, 2, 1))
        axis = None
        if "b0shift" in eff:
            a, axis = self._shift_operator(rp, epi, fov_pe_os)
            comp = np.asarray((a @ comp.reshape(rank, -1).T).T).reshape(rank, n, n)
        # ---- centre of the field of view, in the frame of the acquisition (mm from the isocentre)
        cy, cx = rp.get("centre_y_mm", 0.0), rp.get("centre_x_mm", 0.0)
        p0, r0 = (cx, cy) if rp["pe_dir"] == "x" else (cy, cx)
        if "aa_filter" in eff:                                           # receiver filter: nothing beyond the sampled band
            x = (np.arange(n) - n // 2) * d
            comp[:, :, np.abs(x - r0) > fov_ro_os / 2] = 0
        # ---- Fourier integrals
        c_ro, c_pe = d / fov_ro_os, d / fov_pe_os
        if epi:
            k1 = np.einsum("lpx,lp->px", zoom_dft(comp, 1, k_pe, c_pe, centre=p0 / d), basis)
            eps = (w["system"]["grad_delay_us"] * 1e-6 * rp["matrix"] * rp["bandwidth_hz_per_px"] * ro_os
                   if "ghost" in eff else 0.0)
            ks = zoom_dft(k1, 1, k_ro, c_ro, +eps, centre=r0 / d)
            if eps:
                acquired = self._acquired(rp)
                odd = (np.cumsum(acquired) % 2 == 0) & acquired
                ks[odd] = zoom_dft(k1[odd], 1, k_ro, c_ro, -eps, centre=r0 / d)
        else:
            k1 = np.einsum("lpx,lx->px", zoom_dft(comp, 2, k_ro, c_ro, centre=r0 / d), basis)
            ks = zoom_dft(k1, 0, k_pe, c_pe, centre=p0 / d)
        if axis is not None:
            ks *= _deapodise(k_ro, c_ro)[None, :] if axis == 1 else _deapodise(k_pe, c_pe)[:, None]
        return ks * (d * d * rp["slice_mm"])

    @staticmethod
    def line_weights(rp):
        """Weight of every phase-encode line in the reconstruction, [lines_full]. Fully sampled: 1. Partial Fourier
        (homodyne detection): with n = lines - lines_full / 2, the lines -n .. +n around the centre are acquired on
        both sides and weigh 1; the lines beyond +n (in the order of acquisition) exist on one side only and weigh 2;
        the others are not acquired and weigh 0. The one line without a conjugate partner on the lattice (index
        -lines_full / 2) is among those not acquired, so exactly ``lines`` lines are."""
        k_pe = rp["lines_full"]
        if rp["lines"] >= k_pe:
            return np.ones(k_pe)
        line = np.arange(k_pe) - k_pe // 2
        order = rp.get("pe_polarity", 1) * line if rp["sequence"] == "epi" else line
        n = rp["lines"] - k_pe // 2
        return np.where(line == -(k_pe // 2), 0.0, np.where(order > n, 2.0, np.where(order >= -n, 1.0, 0.0)))

    @classmethod
    def _acquired(cls, rp):
        return cls.line_weights(rp) > 0

    # ------------------------------------------------------------------ periodic motion
    def _line_times(self, rp, offset_ms=0.0):
        """Time (s) at which every phase-encode line is acquired in every average, [lines_full, averages], counted
        from the first acquired line. Line by line: linear order, the averages of a line in succession. EPI: all
        lines of a shot ``echo_spacing_ms`` apart, one shot per average."""
        k_pe, nex = rp["lines_full"], rp["averages"]
        line = np.arange(k_pe) - k_pe // 2
        order = (rp.get("pe_polarity", 1) * line if rp["sequence"] == "epi" else line) + (rp["lines"] - k_pe // 2)
        if rp["sequence"] == "epi":
            t = order[:, None] * rp["echo_spacing_ms"] + np.arange(nex)[None, :] * rp["tr_ms"]
        else:
            cycle = rp["tr_ms"] * (1 + rp["tr_ratio"]) if rp["sequence"] == "afi" else rp["tr_ms"]
            t = (order[:, None] * nex + np.arange(nex)[None, :]) * cycle
        return (t + offset_ms) * 1e-3

    def _motion(self, ks, rp, te_ms, contrast, amps, key, times=None, of_class=None):
        """Add what periodic motion does to the k-space ``ks`` of one image. Every mover is one tissue class (or the
        whole object) whose k-space lines are modulated with the time of their acquisition:
          pulse   amplitude x (1 + depth cos(2 pi f t + phase))
          shift   rigid displacement amp_mm cos(2 pi f t + phase) along a patient axis: exp(-2 pi i k . delta(t))
        averaged over the averages of the line. The phase is a hash of the mover's key and the protocol, so a scan
        is reproducible and its noise-free twin is defined. No spin history, no inflow, nothing through the plane.
        ``times`` [lines_full, averages] (s): the time of every line where it is not one line per cycle (an echo
        train, ``served.line_times``); ``of_class``: class index -> the k-space of that class alone, where the image
        is not formed by ``kspace`` (``served.form``)."""
        k_pe, k_ro = ks.shape
        offset = te_ms + (rp["tr_ms"] if rp["sequence"] == "afi" and contrast == 1 else 0.0)
        t = self._line_times(rp, offset) if times is None else times
        out = ks.copy()
        rot = np.deg2rad(self.w.get("rot_deg", 0.0))
        for mv in self.w["motion"]:
            if mv["cls"] == "*":
                ks_m = ks
            else:
                c = self.w["names"].index(mv["cls"])
                ks_m = self.kspace(rp, te_ms, [v if i == c else np.zeros_like(v) for i, v in enumerate(amps)]) if of_class is None else of_class(c)
            h = hashlib.sha256(repr((mv["phase_key"], key)).encode()).digest()
            arg = 2 * np.pi * (mv["f_hz"] * t + int.from_bytes(h[:4], "big") / 2 ** 32)
            if mv["kind"] == "pulse":
                out += (mv["depth"] * np.cos(arg)).mean(1)[:, None] * ks_m
                continue
            row, col = (np.cos(rot), -np.sin(rot)) if mv["dir"] == "y" else (np.sin(rot), np.cos(rot))    # image rows, columns
            d_pe, d_ro = (col, row) if rp["pe_dir"] == "x" else (row, col)
            ro_os = console.SYSTEM["ro_os"]
            kp = (np.arange(k_pe) - k_pe // 2) / (rp["fov_pe_mm"] * k_pe / rp["matrix_pe"])
            kr = (np.arange(k_ro) - k_ro // 2) / (ro_os * rp["fov_mm"])
            delta = mv["amp_mm"] * np.cos(arg)                               # [line, average], mm
            fac = np.exp(-2j * np.pi * d_pe * kp[:, None] * delta).mean(1)[:, None] if abs(d_ro) < 1e-12 else \
                np.mean([np.exp(-2j * np.pi * (d_pe * kp[:, None] + d_ro * kr[None, :]) * delta[:, i:i + 1])
                         for i in range(delta.shape[1])], axis=0)
            out += (fac - 1) * ks_m
        return out

    def clean_kspace(self, rp):
        """Noise-free k-space of every image of a scan, [n_images, lines_full, os * matrix]; cached per physics key."""
        eng, rp = self._frame(rp)
        if eng is not self:
            return eng.clean_kspace(rp)
        moving = bool(self.w.get("motion")) and "motion" in self.effects
        key = physics_key(rp, averages=moving)
        if key not in self._clean:
            ks = FORMATION[console.REGISTRY[rp["sequence"]].serves](self, rp, key, moving)
            if len(self._clean) >= self.cache_size:
                self._clean.pop(next(iter(self._clean)))
            self._clean[key] = np.array(ks)
        return self._clean[key]

    def _form_steady(self, rp, key, moving):
        """Image formation of a sequence whose every image is one sample of a periodic state (the six with a closed
        form): the amplitude of a class is one map, the same on every phase-encode line. List of k-spaces, in the
        order of the sampling plan (``console.Sampling.images``)."""
        n_c = len(self.w["names"])
        amps = [self._species(rp, c)[1] for c in range(n_c)]             # [class][sample of the cycle]
        ks = []
        for k, te in console.sampling(rp).images:
            one = [amps[c][k] for c in range(n_c)]
            ks.append(self.kspace(rp, te, one))
            if moving:
                ks[-1] = self._motion(ks[-1], rp, te, k, one, key)
        return ks

    def recon(self, ks, rp):
        """The console reconstruction, readout and phase oversampling removed. Complex [y, x].

        Fully sampled: the inverse FFT. Partial Fourier: homodyne detection. The k-space is weighted line by line
        (``line_weights``) and transformed; a low-resolution image gives the phase; the image is the component of the
        weighted transform along that phase, with that phase. For an object whose phase is smooth this is the image
        of the full k-space; the component across the phase, which holds the error of the synthesis and half of the
        noise, is discarded. The low-resolution image is made from the symmetric lines under a Hann taper, and the
        same band along the readout, so that it has no ringing of its own; where it has no signal (next to an edge,
        where the full-resolution image still rings) the phase is that of a coarser image of ``PF_COARSE`` lines
        either side, added with the weight ``PF_BLEND``: the phase of the object is continued into its surroundings."""
        k_pe, k_ro = ks.shape
        if rp["lines"] < k_pe:
            wgt = self.line_weights(rp)
            line, col = np.arange(k_pe) - k_pe // 2, np.arange(k_ro) - k_ro // 2
            n = rp["lines"] - k_pe // 2
            per_line = (console.SYSTEM["ro_os"] * rp["fov_mm"]) / (rp["fov_pe_mm"] * k_pe / rp["matrix_pe"])      # readout samples per line spacing

            def taper(half):                                             # Hann over +- half lines, and over the same band along the readout
                pe = np.where((wgt == 1.0) & (np.abs(line) <= half), 0.5 * (1 + np.cos(np.pi * line / (half + 1))), 0.0)
                ro = np.where(np.abs(col) < (half + 1) * per_line, 0.5 * (1 + np.cos(np.pi * col / ((half + 1) * per_line))), 0.0)
                return pe[:, None] * ro[None, :]
            ift = lambda k: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))   # noqa: E731
            img, ref = ift(ks * wgt[:, None]), ift(ks * (taper(n) + PF_BLEND * taper(min(n, PF_COARSE))))
            mag = np.abs(ref)
            phase = np.where(mag > 0, ref / np.where(mag > 0, mag, 1.0), 1.0)
            img = (img * np.conj(phase)).real * phase * (k_pe * k_ro)
        else:
            img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks))) * (k_pe * k_ro)
        img /= (console.SYSTEM["ro_os"] * rp["fov_mm"]) * (rp["fov_pe_mm"] * k_pe / rp["matrix_pe"])
        lo_p, lo_r = k_pe // 2 - rp["matrix_pe"] // 2, k_ro // 2 - rp["matrix"] // 2
        img = img[lo_p:lo_p + rp["matrix_pe"], lo_r:lo_r + rp["matrix"]]
        return img.T if rp["pe_dir"] == "x" else img

    def acquire_kspace(self, rp, rng=None):
        """k-space as the receiver delivers it: interference line, noise per sample, unacquired lines zeroed."""
        eng, rp = self._frame(rp)
        if eng is not self:
            return eng.acquire_kspace(rp, rng)
        ks = self.clean_kspace(rp).copy()
        sy, nex, eff = self.w["system"], rp["averages"], self.effects
        k_pe, k_ro = ks.shape[1:]
        acquired = self._acquired(rp)
        for i in range(ks.shape[0]):
            if rng is not None and "rfi" in eff and sy.get("rfi"):
                f = sy["rfi"]["freq_hz"]
                if abs(f) < k_ro * rp["bandwidth_hz_per_px"] / 2:
                    t_s = rp["ro_polarity"] * (np.arange(k_ro) - k_ro // 2) / (k_ro * rp["bandwidth_hz_per_px"])
                    ph = np.exp(2j * np.pi * rng.uniform(size=(nex, k_pe))).mean(0)
                    ks[i] += sy["rfi"]["amp"] * ph[:, None] * np.exp(2j * np.pi * f * t_s)[None, :]
            if rng is not None and "noise" in eff and sy["sigma0"] > 0:
                sd = sy["sigma0"] * np.sqrt(k_ro * rp["bandwidth_hz_per_px"] / (nex * rp.get("partitions", 1)))
                ks[i] += sd * (rng.standard_normal((k_pe, k_ro)) + 1j * rng.standard_normal((k_pe, k_ro)))
            ks[i][~acquired] = 0
        return ks

    def noise_sd(self, rp) -> float:
        """Standard deviation of the noise in ONE component (real or imaginary) of a reconstructed pixel of ``rp``:
        the receiver noise per k-space sample, summed over the acquired samples by the console reconstruction with
        the weight of each line squared (a non-selective 3D scan averages over its partitions as well). Partial
        Fourier: the component in phase with the signal, which is the noise of the magnitude where there is signal
        (the component across it is discarded by the reconstruction; the background is not Rayleigh distributed).
        Verifier side (it holds the hidden noise density)."""
        k_ro = console.SYSTEM["ro_os"] * rp["matrix"]
        sd = self.w["system"]["sigma0"] * np.sqrt(k_ro * rp["bandwidth_hz_per_px"] / (rp["averages"] * rp.get("partitions", 1)))
        area = (console.SYSTEM["ro_os"] * rp["fov_mm"]) * (rp["fov_pe_mm"] * rp["lines_full"] / rp["matrix_pe"])
        return float(sd * np.sqrt(console.noise_lines(rp["lines_full"], rp["lines"]) * k_ro) / area)

    def acquire(self, rp, rng=None):
        """What one scan returns: complex images [n_images, y, x]. ``rng`` None = no noise and no interference."""
        return np.array([self.recon(k, rp) for k in self.acquire_kspace(rp, rng)])

    def scan(self, protocol, rng=None):
        """Convenience for checks, figures and examples: protocol -> dict(images, header, plan). Raises ValueError on
        a refused protocol. Episodes go through ``session.ScannerSession``, which logs and charges."""
        pl = console.plan(protocol, self.w["console"])
        if not pl["accept"]:
            raise ValueError("; ".join(v["message"] for v in pl["violations"]))
        return dict(images=self.acquire(pl["realised"], rng), header=pl["header"], plan=pl)


# How the k-space of a scan is formed, by where the registry says the amplitude of the sequence comes from
# (``console.Sequence.serves``).
FORMATION = {"closed form": Engine._form_steady}
# The closed form behind each of the six (``Engine._species`` evaluates them at the local flip angle or through the
# slice); the block list of the same sequence through ``mrgym.core`` is compared with it in ``mrgym validate``, rows J1 to J6.
CLOSED_FORMS = dict(
    spgr="kspace.gre_steady: RF-spoiled steady state on the phase graph (scanner/epg.py), ideal spoiling at tier 1 (signal.spgr)",
    afi="kspace.gre_steady: dual-TR RF-spoiled steady state (scanner/epg.py), ideal spoiling at tier 1 (signal.afi)",
    epi="kspace.gre_steady: the Ernst steady state (signal.spgr)",
    se="signal.se / kspace.cycle_events: longitudinal steady state of the pulse chain, crushed refocusing pulse",
    ir="signal.ir_se / kspace.cycle_events: the same after an inversion of the world's efficiency",
    bssfp="signal.bssfp at the local off-resonance",
)

from . import served  # noqa: E402  (registers FORMATION["core"]; it needs the engine above)

FORMATION["design"] = rfscan.form        # a designed excitation (``scanner/rfscan.py``; imported with the console)
