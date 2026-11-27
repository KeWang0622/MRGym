"""Reconstruction mathematics for receive arrays: functions of arrays of k-space and images, nothing of the scanner.

Conventions (every function)
    k-space   [channel, line, sample], k = 0 at index n // 2 of both axes; lines are the phase-encode axis
    images    [channel, row, column], the centre of the field of view at index n // 2; ``ifft2c(fft2c(x)) = x``
    psi       [channel, channel] Hermitian positive definite: the covariance of the channels' noise up to a common
              factor, psi_ij ~ E[n_i conj(n_j)]. Where an absolute level is needed it is ``sd``: the standard
              deviation of ONE component (real or imaginary) of a channel with psi_ii = 1.
    SNR       |signal| / standard deviation of one component of the noise. An image "in SNR units" has that standard
              deviation equal to 1 in every pixel (Kellman & McVeigh 2005).
    maps      [channel, row, column] complex sensitivities S on the pixels of the image that is reconstructed.

What is here
    noise        ``noise_covariance``, ``whitener``, ``prewhiten``, ``correlated_noise``, ``pseudo_replica``
    combination  ``sos`` (root sum of squares), ``roemer`` (the matched filter with known maps: optimal), ``adaptive``
                 (Walsh: the filter estimated from the data), their noise: ``roemer_noise``, ``sos_snr``, ``chi_moments``
    maps         ``estimate_maps`` from a low-resolution prescan or from the calibration lines of a scan
    sampling     ``sampling_mask`` (every R-th line, with or without calibration lines), ``acceleration_of``
    SENSE        ``sense_plan`` (unfolding matrices, g-factor, noise), ``sense_unfold`` / ``sense``, ``sense_snr_units``,
                 ``sense_cg`` (any set of lines, iterative)
    GRAPPA       not built: ``grappa`` is the seam (it takes what ``sampling_mask`` and ``calibration_block`` give)

SENSE in four lines. A scan that keeps every R-th line folds the image: pixel p of the folded coil image c is
    a_c = sum_m S_c(p_m) rho(p_m),   p_m = the R pixels a field of view / R apart                         a = A rho
With noise covariance psi the best linear unbiased estimate and its covariance are
    rho = U a,   U = (A^H psi^-1 A)^-1 A^H psi^-1,   cov(rho) = (A^H psi^-1 A)^-1
At full sampling pixel p alone gives the variance 1 / (A^H psi^-1 A)_pp, with R times as many samples. Hence
    SNR_R(p) = SNR_full(p) / (g(p) sqrt(R)),   g(p) = sqrt([(A^H psi^-1 A)^-1]_pp [A^H psi^-1 A]_pp) >= 1
(Pruessmann 1999). ``validate()`` checks these against noise replicas on the engine.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.special


# ------------------------------------------------------------------------------------------------ transforms
def fft2c(img):
    """Centred forward transform of the last two axes (no normalisation)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1))), axes=(-2, -1))


def ifft2c(ks):
    """Centred inverse transform of the last two axes (divides by the number of samples)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks, axes=(-2, -1))), axes=(-2, -1))


def resize_k(ks, shape):
    """Crop or zero-pad the last two axes of k-space to ``shape``, keeping k = 0 at index n // 2."""
    out = np.zeros(ks.shape[:-2] + tuple(shape), ks.dtype)
    src, dst = [], []
    for n, m in zip(ks.shape[-2:], shape):
        k = min(n, m)
        src.append(slice(n // 2 - k // 2, n // 2 - k // 2 + k))
        dst.append(slice(m // 2 - k // 2, m // 2 - k // 2 + k))
    out[(..., *dst)] = ks[(..., *src)]
    return out


# ------------------------------------------------------------------------------------------------ noise
def noise_covariance(noise):
    """psi from noise-only samples [channel, ...]: psi_ij = sum_k n_ik conj(n_jk) / (2 N). Its diagonal is the
    variance of one component of each channel."""
    n = np.asarray(noise).reshape(len(noise), -1)
    return n @ n.conj().T / (2 * n.shape[1])


def whitener(psi):
    """W with W psi W^H = 1 (the inverse of the Cholesky factor): ``W @ data`` has uncorrelated channels of equal
    noise. A reconstruction of whitened data with whitened maps (``W @ maps``) needs no psi."""
    return np.linalg.inv(np.linalg.cholesky(np.asarray(psi, complex)))


def prewhiten(data, psi):
    """``data`` [channel, ...] (k-space, images or maps) with the channels mixed so that their noise is white."""
    return np.tensordot(whitener(psi), data, 1)


def correlated_noise(shape, psi, sd=1.0, rng=None):
    """Complex Gaussian noise [channel, *shape] with covariance psi: every component of a channel with psi_ii = 1 has
    the standard deviation ``sd``."""
    rng = np.random.default_rng() if rng is None else rng
    c = len(psi)
    white = rng.standard_normal((c,) + tuple(shape)) + 1j * rng.standard_normal((c,) + tuple(shape))
    return sd * np.tensordot(np.linalg.cholesky(np.asarray(psi, complex)), white, 1)


def pseudo_replica(reconstruct, shape, psi, sd=1.0, n=100, rng=None, lines=None, data=None):
    """Noise of a reconstruction by pseudo multiple replicas (Robson 2008): ``n`` times, noise with the covariance of
    the receiver is put on the acquired samples and passed through ``reconstruct(k-space [channel, line, sample])``.

    ``lines``: boolean [line], the acquired lines (noise exists only there). ``data``: the measured k-space to which
    the noise is added; None for a linear reconstruction, whose noise does not depend on the signal. Returns the
    standard deviation of one component per pixel for a complex result (pooled over real and imaginary part), the
    standard deviation for a real result (a magnitude image; then give ``data``)."""
    rng = np.random.default_rng() if rng is None else rng
    acc = acc2 = 0.0
    for _ in range(n):
        nz = correlated_noise(shape, psi, sd, rng)
        if lines is not None:
            nz[:, ~np.asarray(lines, bool)] = 0
        r = reconstruct(nz if data is None else data + nz)
        acc, acc2 = acc + r, acc2 + np.abs(r) ** 2
    var = (acc2 - np.abs(acc) ** 2 / n) / (n - 1)
    return np.sqrt(np.maximum(var, 0) / (2 if np.iscomplexobj(acc) else 1))


def chi_moments(n_channels, sd=1.0):
    """(mean, standard deviation) of the root sum of squares of ``n_channels`` complex channels of pure noise, each
    component with the standard deviation ``sd``: a chi distribution with 2 n degrees of freedom (Constantinides
    1997),  mean = sd sqrt(2) Gamma(n + 1/2) / Gamma(n),  second moment = 2 n sd^2. One channel: Rayleigh."""
    n = n_channels
    mean = sd * np.sqrt(2.0) * np.exp(scipy.special.gammaln(n + 0.5) - scipy.special.gammaln(n))
    return float(mean), float(np.sqrt(max(2 * n * sd * sd - mean * mean, 0.0)))


# ------------------------------------------------------------------------------------------------ combination
def _psi_inv(psi, c):
    return np.eye(c) if psi is None else np.linalg.inv(np.asarray(psi, complex))


def sos(imgs, psi=None):
    """Root sum of squares of coil images. With ``psi`` the channels are whitened first: the sum is then the
    matched filter with the data as its own map, in units of the noise."""
    d = imgs if psi is None else prewhiten(imgs, psi)
    return np.sqrt((np.abs(d) ** 2).sum(0))


def roemer(imgs, maps, psi=None, snr_units=False):
    """Optimal combination with known maps (Roemer 1990): the matched filter

        rho = S^H psi^-1 d / (S^H psi^-1 S)                          unit sensitivity: an estimate of the magnetisation
        rho = S^H psi^-1 d / sqrt(S^H psi^-1 S)   (``snr_units``)    one component of its noise = that of a channel with psi_ii = 1

    Pixels where every map is zero are returned as zero."""
    pi = _psi_inv(psi, len(maps))
    num = np.einsum("c...,cd,d...->...", maps.conj(), pi, imgs)
    den = np.einsum("c...,cd,d...->...", maps.conj(), pi, maps).real
    ok = den > 0
    return np.where(ok, num / np.where(ok, np.sqrt(den) if snr_units else den, 1.0), 0.0)


def roemer_noise(maps, psi=None):
    """Standard deviation of one component of the unit-sensitivity ``roemer`` image, in units of that of a channel
    with psi_ii = 1: 1 / sqrt(S^H psi^-1 S). Its inverse is the SNR per unit magnetisation."""
    den = np.einsum("c...,cd,d...->...", maps.conj(), _psi_inv(psi, len(maps)), maps).real
    return 1.0 / np.sqrt(np.where(den > 0, den, np.inf))


def sos_snr(maps, psi=None):
    """SNR per unit magnetisation of the root sum of squares at high SNR, in the units of ``1 / roemer_noise``:
    S^H S / sqrt(S^H psi S). Never above the matched filter's (Cauchy-Schwarz); equal to it when psi = 1."""
    p = np.eye(len(maps)) if psi is None else np.asarray(psi, complex)
    num = (np.abs(maps) ** 2).sum(0)
    den = np.einsum("c...,cd,d...->...", maps.conj(), p, maps).real
    return num / np.sqrt(np.where(den > 0, den, np.inf))


def adaptive(imgs, psi=None, block=5, reference=None):
    """Adaptive combination (Walsh 2000): no maps. In every pixel the filter is the dominant eigenvector of the
    signal correlation of the (whitened) coil images over a ``block`` x ``block`` neighbourhood: an estimate of the
    direction of the sensitivity vector there. Its phase is fixed by one channel (``reference``; default the channel
    with the most energy), so the result carries that channel's phase, and that channel's noise where it is weak:
    a magnitude combination. Returns (image, weights [row, column, channel]); the image is ``conj(weights) .
    whitened data``, unit-norm weights: in units of the noise."""
    d = np.asarray(imgs if psi is None else prewhiten(imgs, psi))
    c = len(d)
    rs = np.empty(d.shape[1:] + (c, c), complex)
    for i in range(c):
        for j in range(i, c):
            p = d[i] * d[j].conj()
            rs[..., i, j] = scipy.ndimage.uniform_filter(p.real, block, mode="nearest") + 1j * scipy.ndimage.uniform_filter(p.imag, block, mode="nearest")
            rs[..., j, i] = rs[..., i, j].conj()
    v = np.linalg.eigh(rs)[1][..., -1]                                   # [row, column, channel], unit norm
    ref = int(np.argmax((np.abs(d) ** 2).reshape(c, -1).sum(1))) if reference is None else int(reference)
    v = v * np.exp(-1j * np.angle(v[..., ref]))[..., None]
    return np.einsum("yxc,cyx->yx", v.conj(), d), v


# ------------------------------------------------------------------------------------------------ sensitivity maps
def calibration_block(ks, lines):
    """The block of consecutive acquired lines around k = 0 of an accelerated scan, [channel, lines of the block,
    sample], centred: what ``estimate_maps`` (and a GRAPPA fit) works on."""
    lines = np.asarray(lines, bool)
    c = len(lines) // 2
    if not lines[c]:
        raise ValueError("the line k = 0 was not acquired")
    lo = hi = c
    while lo > 0 and lines[lo - 1]:
        lo -= 1
    while hi + 1 < len(lines) and lines[hi + 1]:
        hi += 1
    half = min(c - lo, hi - c)                                           # symmetric about k = 0 (2 half + 1 lines)
    return ks[..., c - half:c + half + 1, :] if half < c else ks


def estimate_maps(cal, ref=None, shape=None, smooth=1.0, threshold=0.05):
    """Sensitivity maps from low-resolution data: a prescan of the array, or ``calibration_block`` of a scan.

    ``cal``     k-space [channel, line, sample] of the same field of view as the image to be reconstructed
    ``ref``     k-space [line, sample] of a reference coil with the same lattice (a body coil): the maps are then
                relative to it, and a reconstruction with them shows the object as that coil would. None: the
                reference is the root sum of squares with the phase of the strongest virtual channel; the
                reconstruction is then weighted with the array's own profile.
    ``shape``   (rows, columns) of the image the maps are for (None: the lattice of ``cal``)
    Steps (Pruessmann 1999, section "Sensitivity assessment", with a weighted smoothing in place of the polynomial
    fit): a Hann window over the block, zero-filling to ``shape``, images c and r; then
    S = G[c conj(r)] / G[|r|^2] with G a Gaussian of ``smooth`` pixels of the image: a local average of c / r
    weighted with the reference's signal, which continues smoothly into holes and past the edge. Noisy calibration
    data need a wider G; the window has already halved their resolution.
    Returns (maps [channel, row, column], support): support = where the smoothed reference exceeds ``threshold`` of
    its maximum; elsewhere the maps are extrapolations."""
    cal = np.asarray(cal)
    ny, nx = cal.shape[-2:]
    shape = (ny, nx) if shape is None else tuple(shape)
    win = np.outer(np.hanning(ny + 2)[1:-1] if ny > 1 else np.ones(1), np.hanning(nx + 2)[1:-1] if nx > 1 else np.ones(1))
    c = ifft2c(resize_k(cal * win, shape))
    if ref is None:
        flat = c.reshape(len(c), -1)
        v = np.linalg.eigh(flat @ flat.conj().T)[1][:, -1]
        r = np.sqrt((np.abs(c) ** 2).sum(0)) * np.exp(1j * np.angle(np.tensordot(v.conj(), c, 1)))
    else:
        r = ifft2c(resize_k(np.asarray(ref) * win, shape))
    sigma = float(smooth)

    def g(a):
        if not sigma:
            return a
        return scipy.ndimage.gaussian_filter(a.real, sigma, mode="nearest") + 1j * scipy.ndimage.gaussian_filter(a.imag, sigma, mode="nearest")

    den = g(np.abs(r) ** 2 + 0j).real
    maps = np.array([g(ci * r.conj()) for ci in c]) / np.maximum(den, 1e-30 * den.max())
    return maps, den > threshold ** 2 * den.max()


# ------------------------------------------------------------------------------------------------ sampling
def sampling_mask(n_lines, acceleration=1, acs=0, offset=0):
    """Boolean [n_lines]: the lines a regularly undersampled scan acquires. Line index i has k = i - n_lines // 2;
    kept are k = offset (mod acceleration), and the ``acs`` consecutive calibration lines centred on k = 0."""
    k = np.arange(n_lines) - n_lines // 2
    keep = (k - offset) % int(acceleration) == 0
    if acs:
        keep |= (k >= -(acs // 2)) & (k < acs - acs // 2)
    return keep


def acceleration_of(lines):
    """(acceleration, offset, calibration lines) of a mask made by ``sampling_mask``: the acceleration is the largest
    gap between acquired lines (never the mean density, which calibration lines lower), the calibration lines are
    the run of consecutive lines through k = 0 (0 when there is none or the scan is not accelerated)."""
    lines = np.asarray(lines, bool)
    k = np.flatnonzero(lines) - len(lines) // 2
    if len(k) < 2:
        raise ValueError("fewer than two acquired lines")
    gaps = np.diff(k)
    r = int(gaps.max())
    off = int(k[np.argmax(gaps)] % r)
    lattice = sampling_mask(len(lines), r, 0, off)
    run, c = 0, len(lines) // 2
    lo = hi = c
    if r > 1 and lines[c]:
        while lo > 0 and lines[lo - 1]:
            lo -= 1
        while hi + 1 < len(lines) and lines[hi + 1]:
            hi += 1
        run = hi - lo + 1
    extra = np.flatnonzero(lines & ~lattice)
    if not np.all(lines[lattice]) or (extra.size and (run < 2 or extra.min() < lo or extra.max() > hi)):
        raise ValueError("the lines are not a regular lattice plus a calibration block")
    return r, off, (run if run > 1 else 0)


# ------------------------------------------------------------------------------------------------ SENSE
def sense_plan(maps, psi=None, acceleration=2, offset=0, reg=0.0, support=None):
    """Unfolding matrices of a scan that keeps the lines k = offset (mod R) of a lattice of ``maps.shape[1]`` lines.

    ``maps`` [channel, row, column] on the full image (R must divide the number of rows). ``reg``: Tikhonov
    regularisation, U = (A^H psi^-1 A + reg m 1)^-1 A^H psi^-1 with m the mean diagonal of A^H psi^-1 A of the pixel set
    (dimensionless; 0 = the unbiased estimate). ``support`` boolean [row, column]: pixels outside it are known to be
    empty and are left out of the unfolding (their result is 0, their g undefined: nan).

    Returns a dict:
        rows           [rows / R, R] the rows that fold onto each other
        unfold         [rows / R, column, R, channel]: rho(rows[j, m], x) = sum_c unfold[j, x, m, c] a_c(j, x), with
                       a = R x the coil images of the zero-filled k-space (``sense_unfold`` does this)
        g              [row, column] the g-factor sqrt([(A^H psi^-1 A)^-1]_pp [A^H psi^-1 A]_pp) (of the unbiased estimate)
        unfold_noise   [row, column] sqrt([U psi U^H]_pp): one component of the noise of rho per unit of a's. With
                       sd_z the noise of the zero-filled coil images, the noise of rho is R sd_z unfold_noise; at
                       reg = 0 this is sd_full / sqrt([A^H psi^-1 A]_pp) g sqrt(R), sd_full the coil-image noise of the
                       full scan
        full_noise     [row, column] 1 / sqrt([A^H psi^-1 A]_pp): ``roemer_noise`` of the full scan
        g_reg          [row, column] unfold_noise / full_noise: the noise amplification actually paid (= g at reg = 0)
    """
    maps = np.asarray(maps, complex)
    c, ny, nx = maps.shape
    r = int(acceleration)
    if r < 1 or ny % r:
        raise ValueError(f"the acceleration {r} must divide the number of lines {ny}")
    if r > c and support is None:
        raise ValueError(f"{r} folded pixels cannot be told apart by {c} channels")
    m = ny // r
    rows = (np.arange(m)[:, None] + np.arange(r)[None, :] * m) % ny
    phase = np.exp(-2j * np.pi * (offset % r) * np.arange(r) / r)         # of the replica m rows further on
    a = maps[:, rows, :].transpose(1, 3, 0, 2) * phase                   # [j, x, channel, alias]
    if support is not None:
        inside = np.asarray(support, bool)[rows, :].transpose(0, 2, 1)   # [j, x, alias]
        a = a * inside[:, :, None, :]
    else:
        inside = np.ones((m, nx, r), bool)
    pi = _psi_inv(psi, c)
    p = np.eye(c) if psi is None else np.asarray(psi, complex)
    mm = np.einsum("jxca,cd,jxdb->jxab", a.conj(), pi, a)
    diag = np.einsum("jxaa->jxa", mm).real
    empty = diag <= 0                                                    # no channel sees this pixel, or it is outside the support
    lam = reg * np.where(inside & ~empty, diag, 0).sum(-1) / np.maximum((inside & ~empty).sum(-1), 1)
    eye = np.eye(r)
    solve = mm + eye * (lam[..., None, None] + empty[..., None, :] * 1.0)
    inv = np.linalg.inv(solve)
    u = np.einsum("jxab,jxcb,cd->jxad", inv, a.conj(), pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv0 = inv if not reg else np.linalg.inv(mm + eye * (empty[..., None, :] * 1.0))
        g = np.sqrt(np.einsum("jxaa->jxa", inv0).real * diag)
        noise = np.sqrt(np.einsum("jxac,cd,jxad->jxa", u, p, u.conj()).real)
        full = 1.0 / np.sqrt(np.where(empty, np.nan, diag))
    out = {k: np.full((ny, nx), np.nan) for k in ("g", "unfold_noise", "full_noise")}
    for k, v in (("g", np.where(empty, np.nan, g)), ("unfold_noise", np.where(empty, np.nan, noise)), ("full_noise", full)):
        for i in range(r):
            out[k][rows[:, i], :] = v[..., i]
    return dict(acceleration=r, offset=int(offset) % r, rows=rows, unfold=u, g=out["g"], unfold_noise=out["unfold_noise"],
                full_noise=out["full_noise"], g_reg=out["unfold_noise"] / out["full_noise"], reg=float(reg))


def sense_unfold(plan, aliased):
    """The unfolded image [row, column] from the coil images [channel, row, column] of the zero-filled k-space that
    holds the lines of the plan's lattice and zeros elsewhere (any other line must be removed first: ``sense``)."""
    r, rows = plan["acceleration"], plan["rows"]
    v = np.einsum("jxad,djx->ajx", plan["unfold"], r * aliased[:, :rows.shape[0], :])
    out = np.zeros(aliased.shape[1:], complex)
    for i in range(r):
        out[rows[:, i], :] = v[i]
    return out


def sense(ks, plan):
    """SENSE reconstruction of k-space [channel, line, sample] on the full lattice: the lines of the plan's lattice
    are used, every other line (calibration lines in between, zeros) is ignored."""
    keep = sampling_mask(ks.shape[-2], plan["acceleration"], 0, plan["offset"])
    return sense_unfold(plan, ifft2c(ks * keep[:, None]))


def sense_snr_units(plan, aliased, sd_z):
    """``sense_unfold`` in SNR units: divided by its own noise R sd_z unfold_noise, ``sd_z`` = the standard deviation
    of one component of the zero-filled coil images for a channel with psi_ii = 1 (psi as given to the plan)."""
    return sense_unfold(plan, aliased) / (plan["acceleration"] * sd_z * plan["unfold_noise"])


def sense_cg(ks, lines, maps, psi=None, reg=0.0, n_iter=100, tol=1e-8):
    """SENSE for any set of acquired lines (calibration lines included, an acceleration that does not divide the
    lattice, partial Fourier): conjugate gradients on  (E^H psi^-1 E + reg m) rho = E^H psi^-1 d,  E = lines x FFT x
    maps, m = the mean of S^H psi^-1 S, preconditioned with 1 / sqrt(S^H psi^-1 S) (the intensity correction of
    Pruessmann 2001). Returns the image [row, column]; pixels no channel sees are zero. Its noise has no closed form
    here: use ``pseudo_replica``."""
    lines = np.asarray(lines, bool)
    w = np.eye(len(maps)) if psi is None else whitener(psi)
    s, d = np.tensordot(w, maps, 1), np.tensordot(w, ks, 1) * lines[:, None]
    power = (np.abs(s) ** 2).sum(0)
    lam = reg * float(power.mean())
    pre = np.where(power > 0, 1.0 / np.sqrt(np.where(power > 0, power, 1.0)), 0.0)

    def normal(y):
        x = pre * y
        return pre * ((s.conj() * ifft2c(fft2c(s * x) * lines[:, None])).sum(0) + lam * x)

    b = pre * (s.conj() * ifft2c(d)).sum(0)
    y = np.zeros_like(b)
    res = b.copy()
    p, rr = res.copy(), np.vdot(res, res).real
    r0 = rr
    for _ in range(n_iter):
        if rr <= tol ** 2 * r0 or rr == 0:
            break
        ap = normal(p)
        alpha = rr / np.vdot(p, ap).real
        y, res = y + alpha * p, res - alpha * ap
        new = np.vdot(res, res).real
        p, rr = res + (new / rr) * p, new
    return pre * y


def grappa(ks, lines, acs_block=None, kernel=(2, 3)):
    """Seam for the next package (GRAPPA, Griswold 2002): fill the lines a regular scan skipped from their acquired
    neighbours with weights fitted on the calibration block, per channel; return k-space [channel, line, sample] on
    the full lattice. Inputs are what this module already produces: ``lines`` (``sampling_mask``; read R and the
    offset with ``acceleration_of``) and ``acs_block`` (``calibration_block``; None = cut it from ``ks``). The
    result goes through ``roemer``, ``sos`` or ``adaptive``; its noise through ``pseudo_replica``."""
    raise NotImplementedError("GRAPPA is the next work package (A3); use recon.sense or recon.sense_cg")


def validate(markdown=False, groups=None):
    """Print the validation table of the array formation and of this module (``array_checks``); return the rows."""
    from . import array_checks
    return array_checks.main((["--markdown"] if markdown else []) + list(groups or ()), exit=False)
