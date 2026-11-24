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

Frame of the acquisition: arrays [pe, ro]. Master arrays are [y, x]; pe_dir = 'x' transposes them.
Sign convention: a spin that precesses df faster than the reference gains positive phase; with ro_polarity = +1 it is
displaced by -df / bandwidth_hz_per_px pixels along the readout axis (fat, df < 0, moves towards +readout).
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.sparse as sp

from . import console, epg, tissue
from . import signal as S

ALL = frozenset({"t2star", "chemshift", "b0shift", "dropout", "noise", "rfi", "ghost", "aa_filter"})
GAMMA_MHZ = S.GAMMA_MHZ_PER_T
GAMMA_RAD = 2 * np.pi * console.GAMMA_HZ_PER_T
KB_W, KB_BETA = 4, 8.996            # Kaiser-Bessel kernel for a grid >= 2x finer than the acquired pixel (Beatty 2005)
SVD_TOL = 1e-4


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


def diffusion_bd(adc_um2_ms, spoiler_mt_ms_per_m, tr_ms):
    """Dimensionless D * dk^2 * TR for the EPG: D in um^2/ms (= 1e-9 m^2/s), spoiler area in mT*ms/m, TR in ms."""
    return adc_um2_ms * 1e-9 * (GAMMA_RAD * spoiler_mt_ms_per_m * 1e-6) ** 2 * tr_ms * 1e-3


def zoom_dft(x, axis, n_out, cyc_per_sample, offset=0.0):
    """sum_j x_j exp(-2 pi i q (j - n // 2) c) for q = -n_out/2 + offset, ..., n_out/2 - 1 + offset; c = cycles/sample."""
    n = x.shape[axis]
    f = (np.arange(n_out) - n_out // 2 + offset) * cyc_per_sample
    z = scipy.signal.ZoomFFT(n, [f[0], f[0] + n_out * cyc_per_sample], m=n_out, fs=1.0, endpoint=False)
    out = z(x, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = n_out
    return out * np.exp(2j * np.pi * f * (n // 2)).reshape(shape)


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


def physics_key(rp):
    """The part of a realised protocol that changes the noise-free data (not the averages)."""
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(rp.items()) if k != "averages")


class Engine:
    """One hidden world behind ``acquire(realised protocol, rng) -> complex images``. Caches the noise-free k-space."""

    def __init__(self, world: dict, tier: int = 2, fat_model: str = "multipeak", effects=None, cache: int = 48):
        self.world = self.w = world
        self.tier, self.fat_model, self.cache_size = tier, fat_model, cache
        self.effects = frozenset(ALL if effects is None and world.get("effects") is None
                                 else effects if effects is not None else world["effects"])
        self.n, self.d = world["n"], world["extent_mm"] / world["n"]
        self.sup = [wc > 1e-4 for wc in world["weights"]]
        self.body = world["weights"].sum(0) > 1e-4
        self.df = world["fields"]["b0_hz"] + world["system"]["cf_offset_hz"]
        self.b1 = world["fields"]["b1"] * world["system"]["tx_gain"]
        self._clean, self._splat = {}, {}

    # ------------------------------------------------------------------ per-class echo amplitude on the master grid
    def amplitudes(self, rp, c):
        """Echo amplitudes of class ``c`` over its support: list over contrasts (afi: 2, otherwise 1) of arrays."""
        w, name = self.w, self.w["names"][c]
        cls, sup = w["tissues"][name], self.sup[c]
        a = np.deg2rad(rp["flip_deg"]) * self.b1[sup]
        seq = rp["sequence"]
        f_fat = tissue.FAT_PPM * GAMMA_MHZ * w["b0_t"] if cls["fat"] else 0.0
        if seq in ("spgr", "afi") and self.tier == 2:
            lo, hi = float(a.min()), float(a.max())
            pad = 0.01 * (hi - lo) + 1e-4
            nodes, wts = cheb_nodes(lo - pad, hi + pad, int(np.clip(np.ceil(30 * (hi - lo) / (hi + lo)) + 4, 5, 10)))
            bd = diffusion_bd(cls["adc_um2_ms"], rp["spoiler_mt_ms_per_m"], rp["tr_ms"])
            if seq == "spgr":
                vals = [epg.spgr(nodes, rp["tr_ms"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"], bd)]
            else:
                vals = epg.afi(nodes, rp["tr_ms"], rp["tr_ratio"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"], bd)
            out = [bary(nodes, wts, np.asarray(v), a) for v in vals]
        elif seq == "afi":
            out = [np.asarray(v, complex) for v in S.afi(a, rp["tr_ms"], rp["tr2_ms"], cls["t1_ms"])]
        elif seq in ("spgr", "epi"):
            out = [S.spgr(a, rp["tr_ms"], cls["t1_ms"]).astype(complex)]
        elif seq == "se":
            out = [S.se(a, rp["tr_ms"], rp["te_ms"], cls["t1_ms"]).astype(complex)]
        elif seq == "ir":
            out = [S.ir_se(a, rp["tr_ms"], rp["te_ms"], rp["ti_ms"], cls["t1_ms"], w["system"]["inv_eff"]).astype(complex)]
        else:                                                           # bssfp
            free = 2 * np.pi * (self.df[sup] + f_fat) * rp["tr_ms"] * 1e-3
            out = [S.bssfp(a, free + np.deg2rad(rp["phase_cycle_deg"]), rp["tr_ms"], cls["t1_ms"], cls["t2_ms"])
                   * np.exp(0.5j * free)]
        scale = w["weights"][c][sup] * cls["pd"]
        if rp["fat_sat"]:                                               # spectrally selective saturation at the fat line
            sy = w["system"]
            off = self.df[sup] + (0.0 if cls["fat"] else -tissue.FAT_PPM * GAMMA_MHZ * w["b0_t"])
            hit = 0.5 * (1 - np.tanh((np.abs(off) - sy["fatsat_bw_hz"] / 2) / 8.0))
            scale = scale * (1 - sy["fatsat_eff"] * hit)
        return [s * scale for s in out]

    # ------------------------------------------------------------------ one contrast -> k-space [lines_full, os * matrix]
    def _shift_operator(self, rp, epi, fov_pe_os):
        swap = rp["pe_dir"] == "x"
        df, body = (self.df.T, self.body.T) if swap else (self.df, self.body)
        if epi:
            key = ("epi", swap, rp["pe_polarity"], rp["echo_spacing_ms"], fov_pe_os)
            shift, axis = -rp["pe_polarity"] * df * rp["echo_spacing_ms"] * 1e-3 * fov_pe_os / self.d, 0
        else:
            key = ("line", swap, rp["ro_polarity"], rp["bandwidth_hz_per_px"], rp["fov_mm"], rp["matrix"])
            shift, axis = -rp["ro_polarity"] * df / rp["bandwidth_hz_per_px"] * (rp["fov_mm"] / rp["matrix"]) / self.d, 1
        if key not in self._splat:
            self._splat = {key: _splat_matrix(shift, axis, body)}        # keep the last one: echoes and contrasts share it
        return self._splat[key], axis

    def kspace(self, rp, te_ms, amps):
        """Noise-free k-space of one image. ``amps``: per-class amplitude arrays (one contrast of ``amplitudes``)."""
        w, n, d, eff = self.w, self.n, self.d, self.effects
        seq, epi = rp["sequence"], rp["sequence"] == "epi"
        echo = "gre" if seq in ("spgr", "afi", "epi") else "se"
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
        g = np.array([S.time_function(w["tissues"][nm], tau if "t2star" in eff else 0 * tau,
                                      tau if "chemshift" in eff else 0 * tau, te_ms, echo, w["b0_t"], peaks)
                      for nm in w["names"]])
        u, sv, vh = np.linalg.svd(g, full_matrices=False)
        rank = max(1, int(np.sum(sv > SVD_TOL * sv[0])))
        coef, basis = u[:, :rank] * sv[:rank], vh[:rank]
        # ---- component images on the master grid
        comp = np.zeros((rank, n, n), complex)
        for c in range(len(w["names"])):
            comp[:, self.sup[c]] += coef[c][:, None] * amps[c][None, :]
        common = w["fields"]["rx"].astype(complex)
        if echo == "gre":
            common = common * np.exp(2j * np.pi * self.df * te_ms * 1e-3)
            if "dropout" in eff:
                # through-slice dephasing for a Gaussian slice profile of FWHM = slice thickness
                common = common * np.exp(-(np.pi * w["fields"]["dbdz_hz_per_mm"] * rp["slice_mm"] * te_ms * 1e-3) ** 2
                                         / (4 * np.log(2)))
        comp *= common
        # ---- displacement by off-resonance
        if rp["pe_dir"] == "x":
            comp = np.ascontiguousarray(comp.transpose(0, 2, 1))
        axis = None
        if "b0shift" in eff:
            a, axis = self._shift_operator(rp, epi, fov_pe_os)
            comp = np.asarray((a @ comp.reshape(rank, -1).T).T).reshape(rank, n, n)
        if "aa_filter" in eff:                                           # receiver filter: nothing beyond the sampled band
            x = (np.arange(n) - n // 2) * d
            comp[:, :, np.abs(x) > fov_ro_os / 2] = 0
        # ---- Fourier integrals
        c_ro, c_pe = d / fov_ro_os, d / fov_pe_os
        if epi:
            k1 = np.einsum("lpx,lp->px", zoom_dft(comp, 1, k_pe, c_pe), basis)
            eps = (w["system"]["grad_delay_us"] * 1e-6 * rp["matrix"] * rp["bandwidth_hz_per_px"] * ro_os
                   if "ghost" in eff else 0.0)
            ks = zoom_dft(k1, 1, k_ro, c_ro, +eps)
            if eps:
                acquired = self._acquired(rp)
                odd = (np.cumsum(acquired) % 2 == 0) & acquired
                ks[odd] = zoom_dft(k1[odd], 1, k_ro, c_ro, -eps)
        else:
            k1 = np.einsum("lpx,lx->px", zoom_dft(comp, 2, k_ro, c_ro), basis)
            ks = zoom_dft(k1, 0, k_pe, c_pe)
        if axis is not None:
            ks *= _deapodise(k_ro, c_ro)[None, :] if axis == 1 else _deapodise(k_pe, c_pe)[:, None]
        return ks * (d * d * rp["slice_mm"])

    @staticmethod
    def _acquired(rp):
        k_pe = rp["lines_full"]
        line = np.arange(k_pe) - k_pe // 2
        order = rp.get("pe_polarity", 1) * line if rp["sequence"] == "epi" else line
        return order >= -(rp["lines"] - k_pe // 2)

    def clean_kspace(self, rp):
        """Noise-free k-space of every image of a scan, [n_images, lines_full, os * matrix]; cached per physics key."""
        key = physics_key(rp)
        if key not in self._clean:
            n_c = len(self.w["names"])
            amps = [self.amplitudes(rp, c) for c in range(n_c)]          # [class][contrast]
            ks = [self.kspace(rp, te, [amps[c][k] for c in range(n_c)])
                  for k in range(len(amps[0])) for te in rp["echo_times_ms"]]
            if len(self._clean) >= self.cache_size:
                self._clean.pop(next(iter(self._clean)))
            self._clean[key] = np.array(ks)
        return self._clean[key]

    def recon(self, ks, rp):
        """The console reconstruction: zero-filled inverse FFT, readout and phase oversampling removed. Complex [y, x]."""
        k_pe, k_ro = ks.shape
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks))) * (k_pe * k_ro)
        img /= (console.SYSTEM["ro_os"] * rp["fov_mm"]) * (rp["fov_pe_mm"] * k_pe / rp["matrix_pe"])
        lo_p, lo_r = k_pe // 2 - rp["matrix_pe"] // 2, k_ro // 2 - rp["matrix"] // 2
        img = img[lo_p:lo_p + rp["matrix_pe"], lo_r:lo_r + rp["matrix"]]
        return img.T if rp["pe_dir"] == "x" else img

    def acquire_kspace(self, rp, rng=None):
        """k-space as the receiver delivers it: interference line, noise per sample, unacquired lines zeroed."""
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
                sd = sy["sigma0"] * np.sqrt(k_ro * rp["bandwidth_hz_per_px"] / nex)
                ks[i] += sd * (rng.standard_normal((k_pe, k_ro)) + 1j * rng.standard_normal((k_pe, k_ro)))
            ks[i][~acquired] = 0
        return ks

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
