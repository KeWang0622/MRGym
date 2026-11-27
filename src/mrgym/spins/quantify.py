"""What a measurement needs: fitting a spectrum with a simulated basis, the Cramer-Rao bounds of that fit,
water-referenced concentration, line width and signal-to-noise ratio.

Model of the fit (linear-combination modelling in its smallest form; no baseline, no macromolecules, one line
shape for all metabolites):

    y(t_k) = exp(i phi) x exp(2 pi i d t_k - pi w t_k) x sum_m a_m b_m(t_k)  +  noise

b_m the basis decays (``sequences.simulate(...).fid(...)``, one per metabolite, for the unit of concentration the
caller chooses), a_m real amplitudes, d a common frequency shift (Hz), w a common extra Lorentzian width (Hz, full
width at half maximum), phi a zero-order phase. The noise is complex white Gaussian with the standard deviation
``sigma`` in each of the real and the imaginary part of every sample.

``fit`` minimises the sum of squares by variable projection (Golub & Pereyra 1973): for given (d, w, phi) the
amplitudes are the solution of a linear least-squares problem, and ``scipy.optimize.least_squares`` searches the
three non-linear parameters. With ``band_hz`` the residual is taken in the frequency domain on that band only (the
discrete Fourier transform is unitary, so without a band the two domains give the same fit).

``crb`` is the Cramer-Rao lower bound on the standard deviation of every parameter of this model for white Gaussian
noise (Cavassila et al. 2001): the square root of the diagonal of the inverse Fisher matrix F = Re(D^H D) / sigma^2,
D = the derivatives of the model with respect to (a_1 .. a_M, d, w, phi). It is a bound for THIS model: an
unmodelled baseline or line shape makes the true uncertainty larger.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import scipy.optimize

WATER_MOLAL_MM = 55510.0      # mmol of water per kg of water (1000 g / 18.015 g/mol)


# ---------------------------------------------------------------------------------------------- model
def _time(n, dwell_ms):
    return 1e-3 * dwell_ms * np.arange(n)


def _to_domain(x, band):
    """Rows of x in the fitting domain: time, or the bins of the unitary transform inside the band."""
    if band is None:
        return x
    return (np.fft.fft(x, axis=-1) / math.sqrt(x.shape[-1]))[..., band]


def _band(n, dwell_ms, band_hz):
    if band_hz is None:
        return None
    f = np.fft.fftfreq(n, 1e-3 * dwell_ms)
    keep = (f >= min(band_hz)) & (f <= max(band_hz))
    if keep.sum() < 4:
        raise ValueError("the band holds fewer than 4 points")
    return keep


def model(basis, dwell_ms, amplitudes, shift_hz=0.0, width_hz=0.0, phase_rad=0.0):
    """The model decay for a basis [M, n]."""
    basis = np.atleast_2d(np.asarray(basis, complex))
    t = _time(basis.shape[1], dwell_ms)
    return np.exp(1j * phase_rad + (2j * np.pi * shift_hz - np.pi * width_hz) * t) * (np.asarray(amplitudes, float) @ basis)


def _derivatives(basis, dwell_ms, amplitudes, shift_hz, width_hz, phase_rad):
    basis = np.atleast_2d(np.asarray(basis, complex))
    t = _time(basis.shape[1], dwell_ms)
    g = np.exp(1j * phase_rad + (2j * np.pi * shift_hz - np.pi * width_hz) * t)
    y = g * (np.asarray(amplitudes, float) @ basis)
    return np.concatenate([g * basis, [2j * np.pi * t * y, -np.pi * t * y, 1j * y]])       # [M + 3, n]


def fisher(basis, dwell_ms, amplitudes, sigma, shift_hz=0.0, width_hz=0.0, phase_rad=0.0, band_hz=None):
    """Fisher matrix of (a_1 .. a_M, shift, width, phase)."""
    d = _to_domain(_derivatives(basis, dwell_ms, amplitudes, shift_hz, width_hz, phase_rad), _band(np.shape(basis)[-1], dwell_ms, band_hz))
    return (d.conj() @ d.T).real / float(sigma) ** 2


def crb(basis, dwell_ms, amplitudes, sigma, shift_hz=0.0, width_hz=0.0, phase_rad=0.0, band_hz=None, fixed=()):
    """Cramer-Rao bounds (standard deviations) as a dict: ``amplitudes`` [M], ``shift_hz``, ``width_hz``,
    ``phase_rad``. ``fixed``: names among ("shift", "width", "phase") that are known and not fitted."""
    f = fisher(basis, dwell_ms, amplitudes, sigma, shift_hz, width_hz, phase_rad, band_hz)
    m = f.shape[0] - 3
    keep = list(range(m)) + [m + k for k, name in enumerate(("shift", "width", "phase")) if name not in fixed]
    cov = np.linalg.inv(f[np.ix_(keep, keep)])
    sd = np.full(m + 3, 0.0)
    sd[keep] = np.sqrt(np.diag(cov))
    return dict(amplitudes=sd[:m], shift_hz=float(sd[m]), width_hz=float(sd[m + 1]), phase_rad=float(sd[m + 2]))


# ---------------------------------------------------------------------------------------------- fit
@dataclasses.dataclass(frozen=True, eq=False)
class Fit:
    amplitudes: np.ndarray
    shift_hz: float
    width_hz: float
    phase_rad: float
    residual_sd: float            # standard deviation of the residual per real / imaginary part
    fitted: np.ndarray            # the model decay
    success: bool


def _stack(z):
    return np.concatenate([z.real, z.imag], axis=-1)


def _amplitudes(basis_mod, data, band):
    a = _stack(_to_domain(basis_mod, band)).T
    y = _stack(_to_domain(data, band))
    sol = np.linalg.lstsq(a, y, rcond=None)[0]
    return sol, y - a @ sol


def fit(fid, basis, dwell_ms, band_hz=None, start=(0.0, 0.0, 0.0), max_shift_hz=20.0, max_width_hz=30.0, fixed=()):
    """Fit of the model to ``fid`` [n] with the basis [M, n]. ``start`` = (shift Hz, width Hz, phase rad);
    ``fixed``: names among ("shift", "width", "phase") held at their start value. The width is searched between
    -max_width_hz and +max_width_hz: a negative value means the data are narrower than the basis."""
    fid, basis = np.asarray(fid, complex), np.atleast_2d(np.asarray(basis, complex))
    n = fid.size
    if basis.shape[1] != n:
        raise ValueError("the basis and the data must have the same number of points")
    band, t = _band(n, dwell_ms, band_hz), _time(n, dwell_ms)
    names = ("shift", "width", "phase")
    free = [k for k, name in enumerate(names) if name not in fixed]

    def unpack(x):
        p = np.array(start, float)
        p[free] = x
        return p

    def modulated(p):
        return np.exp(1j * p[2] + (2j * np.pi * p[0] - np.pi * p[1]) * t) * basis

    def residual(x):
        return _amplitudes(modulated(unpack(x)), fid, band)[1]

    lo = np.array([-max_shift_hz, -max_width_hz, -np.pi])[free]
    hi = np.array([max_shift_hz, max_width_hz, np.pi])[free]
    x0 = np.clip(np.array(start, float)[free], lo, hi)
    if free:
        res = scipy.optimize.least_squares(residual, x0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        p, ok = unpack(res.x), bool(res.success)
    else:
        p, ok = np.array(start, float), True
    amp, r = _amplitudes(modulated(p), fid, band)
    dof = max(r.size - amp.size - len(free), 1)
    return Fit(amp, float(p[0]), float(p[1]), float(p[2]), float(math.sqrt(np.sum(r * r) / dof)), amp @ modulated(p), ok)


def add_noise(fid, sigma, rng):
    """``fid`` plus complex white Gaussian noise of standard deviation ``sigma`` per real and imaginary part."""
    fid = np.asarray(fid, complex)
    return fid + sigma * (rng.standard_normal(fid.shape) + 1j * rng.standard_normal(fid.shape))


# ---------------------------------------------------------------------------------------------- concentration
def relaxation_factor(te_ms, tr_ms, t1_ms, t2_ms):
    """exp(-TE / T2) (1 - exp(-TR / T1)): the attenuation the consensus formula corrects for."""
    return math.exp(-te_ms / t2_ms) * (1.0 - math.exp(-tr_ms / t1_ms))


def water_referenced(s_met, s_water, protons_met, te_ms, tr_ms, t1_met_ms, t2_met_ms, compartments, te_water_ms=None, tr_water_ms=None,
                     water_mm=WATER_MOLAL_MM, metabolite_free=("csf",)):
    """Concentration of a metabolite from its signal and the signal of the unsuppressed water of the same voxel
    (Gasparovic et al. 2006, the form of the consensus of Near et al. 2021):

        [M] = (S_M / S_W) x (2 / n_H) x  sum_i f_i R_W,i / ((1 - f_csf) R_M)  x [water]

    f_i = the fraction of the voxel's water that is in compartment i = v_i d_i / sum_j v_j d_j (v volume fraction,
    d water content relative to pure water), R = exp(-TE / T2) (1 - exp(-TR / T1)) of the water of the compartment
    and of the metabolite, [water] = 55,510 mmol per kg. The result is in mmol per kg of tissue water (molal), in
    the tissue that holds the metabolite: the compartments named in ``metabolite_free`` hold none.
    ``compartments``: dict name -> dict(volume, water, t1_ms, t2_ms). ``s_met`` and ``s_water`` are amplitudes on
    one scale (same receiver gain, same number of averages); ``protons_met`` the protons behind ``s_met``."""
    te_w = te_ms if te_water_ms is None else te_water_ms
    tr_w = tr_ms if tr_water_ms is None else tr_water_ms
    total = sum(c["volume"] * c["water"] for c in compartments.values())
    if not total > 0:
        raise ValueError("the voxel holds no water")
    frac = {k: c["volume"] * c["water"] / total for k, c in compartments.items()}
    visible = sum(frac[k] * relaxation_factor(te_w, tr_w, c["t1_ms"], c["t2_ms"]) for k, c in compartments.items())
    tissue = 1.0 - sum(frac[k] for k in compartments if k in metabolite_free)
    if not tissue > 0:
        raise ValueError("the voxel holds no tissue")
    return float(s_met / s_water * 2.0 / protons_met * visible / (tissue * relaxation_factor(te_ms, tr_ms, t1_met_ms, t2_met_ms)) * water_mm)


# ---------------------------------------------------------------------------------------------- spectral quality
def linewidth_hz(f_hz, absorption, peak_hz):
    """Full width at half maximum (Hz) of the highest point of the real spectrum ``absorption`` inside the range
    ``peak_hz`` = (lo, hi), by linear interpolation of the two half-height crossings next to it. The baseline is
    taken as zero."""
    f, a = np.asarray(f_hz, float), np.asarray(absorption, float)
    inside = np.nonzero((f >= min(peak_hz)) & (f <= max(peak_hz)))[0]
    k = inside[np.argmax(a[inside])]
    half = 0.5 * a[k]
    i = k
    while i > 0 and a[i] > half:
        i -= 1
    j = k
    while j < a.size - 1 and a[j] > half:
        j += 1
    if a[i] > half or a[j] > half:
        raise ValueError("the line does not fall to half its height inside the spectrum")
    left = f[i] + (half - a[i]) / (a[i + 1] - a[i]) * (f[i + 1] - f[i])
    right = f[j - 1] + (half - a[j - 1]) / (a[j] - a[j - 1]) * (f[j] - f[j - 1])
    return float(right - left)


def snr(f_hz, absorption, peak_hz, noise_hz, detrend=True):
    """Signal-to-noise ratio of a spectrum in the frequency domain: the height of the highest point of the real
    spectrum inside ``peak_hz`` divided by the standard deviation of the real spectrum inside ``noise_hz``, a range
    without signal (after removing a straight line when ``detrend``)."""
    f, a = np.asarray(f_hz, float), np.asarray(absorption, float)
    peak = a[(f >= min(peak_hz)) & (f <= max(peak_hz))].max()
    sel = (f >= min(noise_hz)) & (f <= max(noise_hz))
    if sel.sum() < 16:
        raise ValueError("the noise range holds fewer than 16 points")
    noise = a[sel]
    if detrend:
        noise = noise - np.polyval(np.polyfit(f[sel], noise, 1), f[sel])
    return float(peak / noise.std(ddof=2 if detrend else 1))


def spectral_noise_sd(sigma, n, dwell_ms, zero_fill=1, lb_hz=0.0):
    """Standard deviation of the real part of ``evolve.spectrum`` for time-domain noise ``sigma`` per component:
    sigma x dwell x sqrt(sum of the squared apodisation weights), the first point counted with 1/2."""
    t = _time(n, dwell_ms)
    w = np.exp(-np.pi * lb_hz * t)
    w[0] *= 0.5
    return float(sigma * 1e-3 * dwell_ms * math.sqrt(np.sum(w * w)))
