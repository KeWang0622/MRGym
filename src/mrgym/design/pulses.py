"""Pulse generators: what an agent calls with a handful of numbers, and what comes back.

Every generator returns a ``Pulse``: complex B1 samples (uT), the duration of every sample (ms), the slice-select
gradient during every sample (mT/m; zero when no thickness was asked for), the area of the rephasing lobe that follows
the pulse, and the generator call that made it. ``Pulse.record(tr_ms)`` is the small record: duration, peak B1, energy
(the integral of |B1|^2 dt), B1rms over the stated TR, flip angle on resonance.

Generators
  ``sinc``            windowed sinc (time-bandwidth product, window). With n = 128 and a Hann or Hamming window it is
                      the console's pulse, sample for sample (``scanner/rf.py::sinc_pulse``).
  ``slr``             Shinnar-Le Roux design (Pauly, Le Roux, Nishimura, Macovski 1991): a linear-phase FIR filter
                      (equiripple by ``scipy.signal.remez`` or least squares by ``scipy.signal.firls``) is taken as
                      the polynomial B, the minimum-phase A is found from |A|^2 = 1 - |B|^2, and the inverse transform
                      gives the sub-pulse angles. Types: excitation, refocusing, inversion, saturation.
  ``sech``            hyperbolic-secant adiabatic inversion (Silver, Joseph, Hoult 1985): B1 = B1max sech(beta t)^(1 + i mu).
  ``control_points``  at most 64 complex control points, interpolated onto the raster (rule in the docstring).
  ``verse``           variable-rate rescaling of a pulse and its gradient (Conolly et al. 1988) under a peak-B1 limit.

The slice-select gradient for a thickness is  G = bandwidth / (gamma x thickness), bandwidth = tbw / duration for
``sinc`` and ``slr`` and mu beta / pi for ``sech``: the definition of ``scanner/console.py``.

The Shinnar-Le Roux polynomials used here. With psi the precession per sample and x = exp(i psi), the rotation of
``rotation.rotate`` (model ``split``) for n samples is
    a = exp(-i n psi / 2) A(x),      b = exp(-i (n - 1) psi / 2) B(x),      A, B polynomials of degree n - 1 in x,
built by   A_j = c_j A_(j-1) - conj(s_j) x B_(j-1),   B_j = s_j A_(j-1) + c_j x B_(j-1),   A_0 = 1, B_0 = 0,
c_j = cos(theta_j / 2), s_j = -i exp(i phase_j) sin(theta_j / 2). The constant terms give the LAST sub-pulse:
s_n / c_n = B_n(0) / A_n(0); that is the inverse transform (``inverse_slr``). A(0) = prod c_j > 0, and Jensen's formula
makes prod c_j largest, hence the RF energy least, when A has no zero inside the unit circle: the minimum-phase A.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal

from . import rotation
from .units import HZ_PER_MT_M_MM, RAD_PER_MT_M_MM_MS, RAD_PER_UT_MS, RASTER_MS

KINDS = ("excitation", "refocusing", "inversion", "saturation")
WINDOWS = ("hann", "hamming", "blackman", "rect")
MAX_SAMPLES = 4096
MAX_CONTROL_POINTS = 64
PEAK_MARGIN = 1e-5          # a 180 degree design keeps max |B| = 1 - PEAK_MARGIN: the inverse transform needs |B| < 1


# ---------------------------------------------------------------------------------------------- the pulse object
def _frozen(x, dtype):
    out = np.array(x, dtype=dtype)
    out.setflags(write=False)
    return out


@dataclasses.dataclass(frozen=True, eq=False)
class Pulse:
    """An RF pulse with its slice-select gradient. Arrays are read-only; make a changed pulse with ``replace``."""
    b1_ut: np.ndarray                 # complex [n]
    dt_ms: np.ndarray                 # [n], duration of every sample
    g_mt_m: np.ndarray                # [n], slice-select gradient during every sample
    kind: str = "excitation"
    rephase_mt_ms_m: float = 0.0      # area of the rephasing lobe after the pulse (signed), mT/m x ms
    bandwidth_hz: float | None = None
    thickness_mm: float | None = None
    call: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        b1 = _frozen(self.b1_ut, complex)
        if b1.ndim != 1 or not 1 <= b1.size <= MAX_SAMPLES:
            raise ValueError(f"a pulse has 1 to {MAX_SAMPLES} samples")
        dt = _frozen(np.broadcast_to(np.asarray(self.dt_ms, float), b1.shape), float)
        g = _frozen(np.broadcast_to(np.asarray(self.g_mt_m, float), b1.shape), float)
        if not (np.all(np.isfinite(b1)) and np.all(np.isfinite(g)) and np.all(np.isfinite(dt)) and np.all(dt > 0)):
            raise ValueError("samples must be finite and every sample duration positive")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        for name, value in (("b1_ut", b1), ("dt_ms", dt), ("g_mt_m", g)):
            object.__setattr__(self, name, value)

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    # ---- the record
    @property
    def n(self):
        return int(self.b1_ut.size)

    @property
    def duration_ms(self):
        return float(self.dt_ms.sum())

    @property
    def peak_b1_ut(self):
        return float(np.abs(self.b1_ut).max())

    @property
    def energy_ut2_ms(self):
        """Integral of |B1|^2 dt, uT^2 x ms."""
        return float(np.sum(np.abs(self.b1_ut) ** 2 * self.dt_ms))

    @property
    def selection_area_mt_ms_m(self):
        return float(np.sum(self.g_mt_m * self.dt_ms))

    def b1rms_ut(self, tr_ms, pulses_per_tr=1):
        """Root mean square of B1 over a repetition of ``tr_ms`` that holds ``pulses_per_tr`` of this pulse."""
        if tr_ms < self.duration_ms * pulses_per_tr:
            raise ValueError("the repetition time is shorter than the pulses it holds")
        return math.sqrt(pulses_per_tr * self.energy_ut2_ms / tr_ms)

    def flip_deg(self, b1_scale=1.0):
        """Flip angle on resonance at the slice centre, degrees."""
        _, b = rotation.rotate(self.b1_ut, self.dt_ms, None, None, 0.0, b1_scale)
        return float(np.degrees(rotation.flip_angle(b)))

    def record(self, tr_ms):
        """The small record of the pulse. ``tr_ms`` is the repetition over which B1rms is stated (one pulse per TR)."""
        return dict(kind=self.kind, samples=self.n, duration_ms=round(self.duration_ms, 6), peak_b1_ut=round(self.peak_b1_ut, 6),
                    energy_ut2_ms=round(self.energy_ut2_ms, 6), tr_ms=float(tr_ms), b1rms_ut=round(self.b1rms_ut(tr_ms), 6),
                    flip_deg=round(self.flip_deg(), 4), bandwidth_hz=None if self.bandwidth_hz is None else round(self.bandwidth_hz, 4),
                    thickness_mm=self.thickness_mm, gradient_mt_m=round(float(np.abs(self.g_mt_m).max()), 6),
                    rephase_mt_ms_m=round(self.rephase_mt_ms_m, 6))

    # ---- the rotation
    def rotate(self, pos_mm=None, df_hz=0.0, b1_scale=1.0, rephase=True, model="split"):
        """(a, b) at positions ``pos_mm`` along the slice axis, after the rephasing lobe when ``rephase``. The lobe is
        a rotation about z by the gradient area only: no time passes in it (no off-resonance phase, no relaxation)."""
        a, b = rotation.rotate(self.b1_ut, self.dt_ms, self.g_mt_m, pos_mm, df_hz, b1_scale, model)
        if rephase and self.rephase_mt_ms_m and pos_mm is not None:
            a, b = rotation.z_rotation(a, b, RAD_PER_MT_M_MM_MS * self.rephase_mt_ms_m * np.asarray(pos_mm, float) + 0 * a.real)
        return a, b

    def bloch(self, pos_mm=None, df_hz=0.0, b1_scale=1.0, t1_ms=np.inf, t2_ms=np.inf, mxy0=0.0, mz0=1.0, rephase=True, model="split"):
        """(Mxy, Mz) after the pulse with relaxation during it (``rotation.bloch``), then the rephasing lobe."""
        mxy, mz = rotation.bloch(self.b1_ut, self.dt_ms, self.g_mt_m, pos_mm, df_hz, b1_scale, t1_ms, t2_ms, mxy0, mz0, model=model)
        if rephase and self.rephase_mt_ms_m and pos_mm is not None:
            mxy = mxy * np.exp(1j * RAD_PER_MT_M_MM_MS * self.rephase_mt_ms_m * np.asarray(pos_mm, float))
        return mxy, mz


def _n_samples(duration_ms, n, raster_ms):
    if not (duration_ms > 0 and math.isfinite(duration_ms)):
        raise ValueError("duration_ms must be positive")
    if n is None:
        n = int(round(duration_ms / raster_ms))
        if abs(n * raster_ms - duration_ms) > 1e-9 * max(1.0, duration_ms):
            raise ValueError(f"duration_ms = {duration_ms} is not a multiple of the raster {raster_ms} ms; pass n to choose the samples")
    n = int(n)
    if not 1 <= n <= MAX_SAMPLES:
        raise ValueError(f"a pulse has 1 to {MAX_SAMPLES} samples, not {n}")
    return n


def _gradient(bandwidth_hz, thickness_mm):
    """Slice-select gradient (mT/m) that spreads ``bandwidth_hz`` over ``thickness_mm``; 0 without a thickness."""
    if thickness_mm is None:
        return 0.0
    if not thickness_mm > 0:
        raise ValueError("thickness_mm must be positive")
    return bandwidth_hz / (HZ_PER_MT_M_MM * thickness_mm)


def _rephase(kind, g, duration_ms, fraction):
    return -fraction * g * duration_ms if kind == "excitation" else 0.0


def _window(name, t):
    """Window on t in [-1/2, 1/2]."""
    if name == "hann":
        return 0.5 * (1 + np.cos(2 * np.pi * t))
    if name == "hamming":
        return 0.54 + 0.46 * np.cos(2 * np.pi * t)
    if name == "blackman":
        return 0.42 + 0.5 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t)
    if name == "rect":
        return np.ones_like(t)
    raise ValueError(f"window must be one of {WINDOWS}")


# ---------------------------------------------------------------------------------------------- windowed sinc
def sinc(flip_deg, duration_ms, tbw, thickness_mm=None, window="hann", kind="excitation", n=None, raster_ms=RASTER_MS):
    """Windowed sinc with ``tbw`` zero crossings (time-bandwidth product). Sample j sits at the centre of its interval,
    (j + 1/2) / n - 1/2 of the duration; the samples are scaled so that their sum is the flip angle on resonance.
    Bandwidth = tbw / duration. An excitation is followed by a rephasing lobe of half the selection area."""
    n = _n_samples(duration_ms, n, raster_ms)
    if not tbw > 0:
        raise ValueError("tbw must be positive")
    t = (np.arange(n) + 0.5) / n - 0.5
    shape = np.sinc(tbw * t) * _window(window, t)
    dt = duration_ms / n
    b1 = shape / shape.sum() * math.radians(flip_deg) / (RAD_PER_UT_MS * dt)
    bw = 1e3 * tbw / duration_ms
    g = _gradient(bw, thickness_mm)
    return Pulse(b1, dt, g, kind, _rephase(kind, g, duration_ms, 0.5), bw, thickness_mm,
                 dict(generator="sinc", flip_deg=float(flip_deg), duration_ms=float(duration_ms), tbw=float(tbw), thickness_mm=thickness_mm,
                      window=window, kind=kind, n=n))


# ---------------------------------------------------------------------------------------------- Shinnar-Le Roux
def forward_slr(theta):
    """Coefficients (A, B) of the polynomials of the module docstring for complex sub-pulse angles ``theta`` [n]
    (rad; the angle of the complex number is the phase of the sub-pulse). Index k = coefficient of x^k."""
    theta = np.asarray(theta, complex)
    n = theta.size
    a, b = np.zeros(n, complex), np.zeros(n, complex)
    a[0] = 1.0
    for th in theta:
        c, s = np.cos(abs(th) / 2), -1j * np.exp(1j * np.angle(th)) * np.sin(abs(th) / 2)
        xb = np.roll(b, 1)                                      # x B; the top coefficient of B is still zero here
        a, b = c * a - np.conj(s) * xb, s * a + c * xb
    return a, b


def min_phase_a(b, oversample=256):
    """Coefficients of the polynomial A with |A|^2 = 1 - |B|^2 on the unit circle, no zero inside it and A(0) > 0.

    log A is analytic inside the unit circle, so its coefficients are those of log|A| = log(1 - |B|^2) / 2 with the
    negative powers folded onto the positive ones (the real cepstrum; Oppenheim & Schafer). Computed on
    ``oversample`` x n points, a power of two. Needs max |B| < 1.
    """
    b = np.asarray(b, complex)
    n = b.size
    big = 1 << int(math.ceil(math.log2(max(16, oversample * n))))
    mag2 = 1.0 - np.abs(np.fft.ifft(b, big) * big) ** 2        # B at x = exp(2 pi i m / big)
    if mag2.min() <= 0:
        raise ValueError("|B| must stay below 1 on the unit circle")
    cep = np.fft.fft(0.5 * np.log(mag2)) / big                  # coefficient of x^k of log|A|
    cep[1:big // 2] *= 2
    cep[big // 2 + 1:] = 0
    a = np.fft.fft(np.exp(np.fft.ifft(cep) * big)) / big
    return a[:n]


def inverse_slr(b, a=None, oversample=256):
    """Sub-pulse angles (complex, rad) whose rotation has the polynomial B with coefficients ``b`` (index = power of
    x) and the minimum-phase A, or the given ``a``. The recursion peels the last sub-pulse off the constant terms."""
    b = np.array(b, complex)
    a = min_phase_a(b, oversample) if a is None else np.array(a, complex)
    n = b.size
    theta = np.zeros(n, complex)
    for j in range(n - 1, -1, -1):
        r = b[0] / a[0]
        c = 1.0 / math.sqrt(1.0 + abs(r) ** 2)
        s = c * r
        theta[j] = 2 * math.atan2(abs(s), c) * (1j * s / abs(s) if abs(s) > 0 else 1.0)
        a, b = c * a + np.conj(s) * b, (-s * a + c * b)
        b = np.append(b[1:], 0.0)                              # divide by x: the constant term has just been cancelled
    return theta


def _peak_gain(h, amp):
    """Largest |H| on the unit circle of the filter ``h``: the largest value on the grid ``amp`` (|FFT|), refined
    between the grid points around every local maximum near it."""
    n, big = h.size, amp.size
    k = np.arange(n)
    step = 2 * np.pi / big
    best = float(amp.max())
    for i in np.nonzero((amp >= np.roll(amp, 1)) & (amp >= np.roll(amp, -1)) & (amp > 0.98 * best))[0]:
        res = scipy.optimize.minimize_scalar(lambda psi: -abs(np.sum(h * np.exp(-1j * k * psi))), method="bounded",
                                             bounds=((i - 1) * step, (i + 1) * step), options=dict(xatol=1e-13))
        best = max(best, -float(res.fun))
    return best


def dinf(d1, d2):
    """Transition-width measure D_inf of an equiripple low-pass filter with pass / stop ripples d1 / d2: the product
    (number of taps) x (transition width in cycles per sample), fitted by Herrmann, Rabiner & Chan 1973; used by
    Pauly et al. 1991 as  time-bandwidth = D_inf / (fractional transition width). An empirical fit, not a bound."""
    l1, l2 = math.log10(d1), math.log10(d2)
    return (5.309e-3 * l1 * l1 + 7.114e-2 * l1 - 4.761e-1) * l2 + (-2.66e-3 * l1 * l1 - 5.941e-1 * l1 - 4.278e-1)


def _response(kind, beta):
    """What the profile specification is written on, as a function of |b|: |Mxy| after an excitation, Mz after a
    saturation or an inversion, the echo factor |b|^2 of a refocusing pulse between crushers."""
    if kind == "excitation":
        return np.sin(2 * np.arcsin(np.clip(beta, 0, 1)))
    if kind == "refocusing":
        return np.asarray(beta, float) ** 2
    return 1 - 2 * np.asarray(beta, float) ** 2


def polynomial_ripples(kind, flip_deg, d1, d2):
    """Ripples of the filter B that give the ripples ``d1`` (in the slice) and ``d2`` (outside) of the profile, in
    units of M0, for a pulse of the given kind and flip angle. Returns (scale of B, filter pass ripple, filter stop
    ripple, clipped). The filter has gain 1 +- pass ripple in the slice; B = scale x filter. When the flip angle is so
    large that scale x (1 + ripple) would pass 1, B is normalised to a peak just below 1 instead (``clipped``; the
    largest flip angle is then 2 asin(1 - PEAK_MARGIN), 179.5 degrees) and the deviation is still counted from the
    profile value of the flip angle asked for. For 90 degree excitation / saturation and 180 degree refocusing / inversion
    these are the relations of Pauly et al. 1991: sqrt(d1 / 2), d1 / 2, d1 / 4, d1 / 8 in the slice and d2 / sqrt(2),
    sqrt(d2), sqrt(d2), sqrt(d2 / 2) outside, to first order in the ripples (checked by a test)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not (0 < d1 < 0.5 and 0 < d2 < 0.5):
        raise ValueError("ripples must be between 0 and 0.5")
    beta0 = math.sin(math.radians(flip_deg) / 2)
    if not 0 < flip_deg <= 180:
        raise ValueError("flip_deg must be in (0, 180]")
    if kind == "excitation":
        x = math.sin(math.asin(min(1.0, d2)) / 2)
    elif kind == "refocusing":
        x = math.sqrt(d2)
    else:
        x = math.sqrt(d2 / 2)
    stop = x / beta0
    grid = np.linspace(-1.0, 1.0, 401)

    def deviation(delta):
        h = 1 + delta * grid
        clipped = beta0 * (1 + delta) >= 1
        beta = h / (1 + delta) if clipped else beta0 * h
        return float(np.abs(_response(kind, beta) - _response(kind, beta0)).max()), clipped

    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if deviation(mid)[0] <= d1:
            lo = mid
        else:
            hi = mid
    clipped = deviation(lo)[1]
    if lo > 0.45 or stop > 0.45:                                # ripples in units of M0 that a small flip angle cannot give meaning to
        raise ValueError(f"ripples of {d1:g} / {d2:g} of M0 are too large for a {flip_deg:g} degree {kind} pulse: the filter would have "
                         f"ripples of {min(lo, 0.5):.2f} / {stop:.2f} of its own gain; ask for smaller ripples")
    return beta0, lo, stop, clipped


def slr_filter(n, tbw, kind="excitation", flip_deg=90.0, filter="pm", d1=0.01, d2=0.01):
    """The polynomial B of a Shinnar-Le Roux design: coefficients (index = power of x, rotation about +x, so that
    B(1) = -i sin(flip / 2) up to the pass-band ripple) and the record of the filter design."""
    scale, p_ripple, s_ripple, clipped = polynomial_ripples(kind, flip_deg, d1, d2)
    big = 1 << int(math.ceil(math.log2(64 * n)))
    freq = np.abs(np.fft.fftfreq(big))

    def design(width):
        """Filter for a fractional transition width; its measured ripples relative to the mid pass-band gain."""
        f_pass, f_stop = (1 - width) * tbw / (2 * n), (1 + width) * tbw / (2 * n)
        if f_stop >= 0.5:
            raise ValueError(f"{n} samples are too few for time-bandwidth {tbw}: the stop band starts beyond the Nyquist frequency")
        if filter == "pm":
            h = scipy.signal.remez(n, [0, f_pass, f_stop, 0.5], [1, 0], weight=[1, p_ripple / s_ripple], fs=1.0, maxiter=500)
        else:
            h = scipy.signal.firls(n, [0, f_pass, f_stop, 0.5], [1, 1, 0, 0], weight=[1, (p_ripple / s_ripple) ** 2], fs=1.0)
        amp = np.abs(np.fft.fft(h, big))
        in_band = amp[freq <= f_pass]
        mid = 0.5 * (in_band.max() + in_band.min())
        return h, amp, mid, (in_band.max() - in_band.min()) / (2 * mid), amp[freq >= f_stop].max() / mid, f_pass, f_stop

    if filter not in ("pm", "ls"):
        raise ValueError("filter must be 'pm' or 'ls'")
    if filter == "ls" and n % 2 == 0:
        raise ValueError("filter='ls' needs an odd number of samples")
    if not tbw > 0:
        raise ValueError("tbw must be positive")
    width = dinf(p_ripple, s_ripple) / tbw
    if not 0 < width < 1:
        raise ValueError(f"time-bandwidth {tbw} cannot hold these ripples: the transition would be {width:.2f} of the bandwidth")
    if filter == "pm":                       # D_inf is a fit: widen or narrow the transition until the ripples are met
        def met(w):
            out = design(w)
            return out[3] <= p_ripple * (1 + 1e-6) and out[4] <= s_ripple * (1 + 1e-6)
        lo, hi = 0.5 * width, min(0.98, 2.0 * width)
        if not met(hi):
            raise ValueError(f"time-bandwidth {tbw} cannot hold these ripples within a transition of {hi:.2f} of the bandwidth")
        if met(lo):
            hi = lo
        else:
            for _ in range(10):
                mid_w = 0.5 * (lo + hi)
                lo, hi = (lo, mid_w) if met(mid_w) else (mid_w, hi)
        width = hi
    h, amp, mid, pass_measured, stop_measured, f_pass, f_stop = design(width)
    peak = _peak_gain(h, amp)
    if clipped or scale / mid * peak >= 1 - PEAK_MARGIN:
        b = h / peak * (1 - PEAK_MARGIN)                        # largest |B| on the unit circle just below 1
    else:
        b = h * scale / mid
    return -1j * b, dict(filter_pass_ripple=float(pass_measured), filter_stop_ripple=float(stop_measured), filter_pass_target=p_ripple,
                         filter_stop_target=s_ripple, transition_fraction=float(width), pass_edge=float(f_pass), stop_edge=float(f_stop))


def slr(flip_deg, duration_ms, tbw, thickness_mm=None, kind="excitation", filter="pm", d1=0.01, d2=0.01, n=None,
        raster_ms=RASTER_MS, oversample=256):
    """Shinnar-Le Roux pulse with a linear-phase filter.

    ``flip_deg``  flip angle in the slice (180 for refocusing and inversion; smaller angles are allowed).
    ``tbw``       time-bandwidth product: bandwidth = tbw / duration, measured between the middles of the transitions.
    ``d1, d2``    ripples of the PROFILE in units of M0: |Mxy| (excitation), Mz (saturation, inversion), the echo
                  factor |b|^2 (refocusing between crushers), in the slice and outside. The fractional transition
                  width is  W = D_inf(filter ripples) / tbw  and must be below 1: a small tbw cannot hold small ripples.
    ``filter``    "pm" equiripple (Parks-McClellan, ``scipy.signal.remez``): D_inf is an empirical fit, so the
                  transition is then adjusted (between half and twice the fit) to the narrowest one whose filter meets
                  both ripples. An equiripple filter can carry a large first and last sample (they hold the stop-band
                  ripple up); they count towards the peak B1. "ls" least squares (``scipy.signal.firls``; needs an odd
                  number of samples): W from the fit, the ripples are targets, not bounds, and the ends are smooth.
    The rotation is about +x on resonance (b = -i sin(flip / 2)). An excitation is followed by a rephasing lobe of
    half the selection area. The record of the design is in ``Pulse.call``: band edges, filter ripples, and the
    largest difference on the unit circle between the designed polynomial B and the B of the pulse that came out.
    """
    n = _n_samples(duration_ms, n, raster_ms)
    b, record = slr_filter(n, tbw, kind, flip_deg, filter, d1, d2)
    theta = inverse_slr(b, oversample=oversample)
    b_sim = forward_slr(theta)[1]
    record["inverse_error"] = float(np.abs(np.fft.fft(b_sim - b, 1 << int(math.ceil(math.log2(64 * n))))).max())
    record["pass_edge_hz"] = 1e3 * record.pop("pass_edge") * n / duration_ms
    record["stop_edge_hz"] = 1e3 * record.pop("stop_edge") * n / duration_ms
    dt = duration_ms / n
    b1 = theta / (RAD_PER_UT_MS * dt)
    bw = 1e3 * tbw / duration_ms
    g = _gradient(bw, thickness_mm)
    return Pulse(b1, dt, g, kind, _rephase(kind, g, duration_ms, 0.5), bw, thickness_mm,
                 dict(generator="slr", flip_deg=float(flip_deg), duration_ms=float(duration_ms), tbw=float(tbw), thickness_mm=thickness_mm,
                      kind=kind, filter=filter, d1=float(d1), d2=float(d2), n=n, design=record))


# ---------------------------------------------------------------------------------------------- hyperbolic secant
def sech(b1_peak_ut, duration_ms, beta_rad_per_ms, mu, thickness_mm=None, n=None, raster_ms=RASTER_MS):
    """Hyperbolic-secant adiabatic inversion:  B1(t) = B1max sech(beta t)^(1 + i mu),  -T/2 <= t <= T/2.

    The amplitude is B1max sech(beta t) and the frequency sweeps as  -mu beta tanh(beta t)  (rad/ms): from +mu beta
    to -mu beta, bandwidth mu beta / pi. ``beta_rad_per_ms`` x duration / 2 sets the truncation: the amplitude at the
    ends is sech(beta T / 2) of the peak. For the untruncated pulse the inversion is known in closed form
    (``sech_mz``); on resonance it needs  gamma B1max > about beta sqrt(mu^2 - ...) and no longer depends on B1 above
    that: the adiabatic threshold."""
    n = _n_samples(duration_ms, n, raster_ms)
    if not (b1_peak_ut > 0 and beta_rad_per_ms > 0 and mu >= 0):
        raise ValueError("b1_peak_ut and beta_rad_per_ms must be positive, mu non-negative")
    dt = duration_ms / n
    t = (np.arange(n) + 0.5) * dt - duration_ms / 2
    b1 = b1_peak_ut / np.cosh(beta_rad_per_ms * t) * np.exp(-1j * mu * np.log(np.cosh(beta_rad_per_ms * t)))
    bw = 1e3 * mu * beta_rad_per_ms / math.pi
    g = _gradient(bw, thickness_mm) if bw > 0 else 0.0
    return Pulse(b1, dt, g, "inversion", 0.0, bw, thickness_mm,
                 dict(generator="sech", b1_peak_ut=float(b1_peak_ut), duration_ms=float(duration_ms), beta_rad_per_ms=float(beta_rad_per_ms),
                      mu=float(mu), thickness_mm=thickness_mm, n=n, truncation=float(1 / math.cosh(beta_rad_per_ms * duration_ms / 2))))


def sech_mz(b1_peak_ut, beta_rad_per_ms, mu, df_hz=0.0, b1_scale=1.0):
    """Mz after an UNTRUNCATED hyperbolic-secant pulse from equilibrium, without relaxation (the exact solution of
    the Bloch equations for this pulse; Silver, Joseph, Hoult 1985, in the form of Hioe 1984):

        Mz = tanh(pi (d + mu) / 2) tanh(pi (d - mu) / 2) + cos(pi sqrt(x^2 - mu^2)) sech(pi (d + mu) / 2) sech(pi (d - mu) / 2)

    x = gamma B1max / beta, d = offset / beta (both in rad). For x < mu the cosine is a hyperbolic cosine. On
    resonance with mu = 0 this is cos(pi x), the flip of a real sech pulse of area pi x."""
    x = RAD_PER_UT_MS * b1_peak_ut * np.asarray(b1_scale, float) / beta_rad_per_ms
    d = 2 * np.pi * 1e-3 * np.asarray(df_hz, float) / beta_rad_per_ms
    root = np.sqrt((x * x - mu * mu).astype(complex))
    osc = np.cos(np.pi * root).real
    p, m = 0.5 * np.pi * (d + mu), 0.5 * np.pi * (d - mu)
    return np.tanh(p) * np.tanh(m) + osc / (np.cosh(p) * np.cosh(m))


# ---------------------------------------------------------------------------------------------- control points
def control_points(points_ut, duration_ms, bandwidth_hz=None, thickness_mm=None, kind="excitation", interp="linear",
                   rephase_fraction=0.5, n=None, raster_ms=RASTER_MS):
    """A pulse from at most 64 complex control points (uT).

    Rule: control point k of K sits at the time  k T / (K - 1)  (the first at the start of the pulse, the last at its
    end; one point alone is a constant pulse). The real and the imaginary part are interpolated separately,
    ``linear`` between neighbours or by the natural ``cubic`` spline through all points, and evaluated at the centre
    (j + 1/2) dt of every raster sample. ``bandwidth_hz`` is the frequency span that the slice-select gradient
    spreads over ``thickness_mm`` (both or neither); an excitation is followed by a rephasing lobe of
    ``rephase_fraction`` of the selection area."""
    pts = np.asarray(points_ut, complex).ravel()
    if not 1 <= pts.size <= MAX_CONTROL_POINTS:
        raise ValueError(f"1 to {MAX_CONTROL_POINTS} control points")
    if not np.all(np.isfinite(pts)):
        raise ValueError("control points must be finite")
    if (bandwidth_hz is None) != (thickness_mm is None):
        raise ValueError("give bandwidth_hz and thickness_mm together, or neither")
    n = _n_samples(duration_ms, n, raster_ms)
    dt = duration_ms / n
    t = (np.arange(n) + 0.5) * dt
    if pts.size == 1:
        b1 = np.full(n, pts[0])
    else:
        tk = np.linspace(0.0, duration_ms, pts.size)
        if interp == "linear":
            b1 = np.interp(t, tk, pts.real) + 1j * np.interp(t, tk, pts.imag)
        elif interp == "cubic":
            b1 = (scipy.interpolate.CubicSpline(tk, pts.real, bc_type="natural")(t)
                  + 1j * scipy.interpolate.CubicSpline(tk, pts.imag, bc_type="natural")(t))
        else:
            raise ValueError("interp must be 'linear' or 'cubic'")
    g = 0.0 if thickness_mm is None else _gradient(bandwidth_hz, thickness_mm)
    return Pulse(b1, dt, g, kind, _rephase(kind, g, duration_ms, rephase_fraction), bandwidth_hz, thickness_mm,
                 dict(generator="control_points", points_ut=[[float(p.real), float(p.imag)] for p in pts], duration_ms=float(duration_ms),
                      bandwidth_hz=bandwidth_hz, thickness_mm=thickness_mm, kind=kind, interp=interp,
                      rephase_fraction=float(rephase_fraction), n=n))


# ---------------------------------------------------------------------------------------------- variable rate
def _slew_cap(s, c):
    """Largest rate allowed next to a sample of rate ``s`` under the slew limit (c = slew x dt0 / G0): the root of
    r - c / (2 r) = s + c / (2 s). Below s = sqrt(c / 2) the true cap grows again as s falls; it is held at its
    minimum there, which keeps the cap non-decreasing (what the two-pass rule needs) at the price of a conservative
    schedule in that range."""
    s = max(s, math.sqrt(c / 2))
    q = s + c / (2 * s)
    return 0.5 * (q + math.sqrt(q * q + 2 * c))


def verse(pulse, b1_peak_ut, g_max_mt_m, slew_t_m_s=None):
    """Variable-rate rescaling (VERSE; Conolly, Nishimura, Macovski, Glover 1988) of a pulse and its gradient.

    Sample j is played ``r_j`` times faster with B1 and gradient both multiplied by ``r_j``: the sub-pulse angle
    B1 dt and the gradient area G dt of every sample are unchanged, so the rotation of a spin ON resonance is the
    same at every position (exactly, in the ``split`` model; row D16). A spin off resonance precesses for a different
    time in every sample, so its rotation changes (row D17). The rate is the largest allowed:

        r_j = min(b1_peak_ut / |B1_j|,  g_max_mt_m / |G_j|)

    and, with ``slew_t_m_s`` (constant-gradient, constant-raster pulses only), rates lowered by one forward and one
    backward pass until |G_(j+1) - G_j| <= slew x (dt_j + dt_(j+1)) / 2 holds between neighbouring samples. The result
    always meets the three limits. It is the fastest such schedule when every rate stays above sqrt(c / 2),
    c = slew x raster / gradient of the input pulse (rates of order 1 and c well below 1: the usual case). Below
    that, a slower sample lasts so long that it would allow a LARGER step to its neighbour; the pass does not use
    this and is then slower than the best schedule (a 6-sample example was 3 % slower than a general optimiser).
    The gradient at the two ends is free (the ramps to and from zero are outside the pulse). The result has samples
    of unequal duration. The rephasing area is unchanged."""
    if not (b1_peak_ut > 0 and g_max_mt_m > 0):
        raise ValueError("b1_peak_ut and g_max_mt_m must be positive")
    amp, g = np.abs(pulse.b1_ut), np.abs(pulse.g_mt_m)
    if not np.all(g > 0):
        raise ValueError("variable-rate rescaling needs a non-zero slice-select gradient in every sample")
    with np.errstate(divide="ignore"):
        rate = np.minimum(np.where(amp > 0, b1_peak_ut / np.where(amp > 0, amp, 1.0), np.inf), g_max_mt_m / g)
    if slew_t_m_s is not None:
        if not (np.all(g == g[0]) and np.all(pulse.dt_ms == pulse.dt_ms[0])):
            raise ValueError("the slew limit is implemented for a constant gradient on a constant raster")
        c = slew_t_m_s * pulse.dt_ms[0] / g[0]
        rate = rate.copy()
        for j in range(1, rate.size):
            rate[j] = min(rate[j], _slew_cap(rate[j - 1], c))
        for j in range(rate.size - 2, -1, -1):
            rate[j] = min(rate[j], _slew_cap(rate[j + 1], c))
    return pulse.replace(b1_ut=pulse.b1_ut * rate, dt_ms=pulse.dt_ms / rate, g_mt_m=pulse.g_mt_m * rate,
                         call=dict(generator="verse", of=pulse.call, b1_peak_ut=float(b1_peak_ut), g_max_mt_m=float(g_max_mt_m),
                                   slew_t_m_s=None if slew_t_m_s is None else float(slew_t_m_s)))
