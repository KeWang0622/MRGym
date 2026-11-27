"""Figures of merit of a slice profile, a profile mask, and lower bounds on RF energy and pulse duration.

A profile is a response r(z) on positions z (mm, increasing). ``response(kind, a, b)`` gives the quantity a
specification is written on:

    excitation    Mxy = 2 conj(a) b            complex; magnitude sin(flip) in the slice
    refocusing    -b^2                         complex echo factor between equal crushers; magnitude 1 in the slice
    inversion     |b|^2 = (1 - Mz) / 2         1 in the slice (Mz = -1), 0 outside
    saturation    |b|^2 = (1 - Mz) / 2         1/2 in the slice (Mz = 0), 0 outside

Bands. With slice thickness d and fractional transition width W, the pass band is |z| <= (1 - W) d / 2 and the stop
band |z| >= (1 + W) d / 2: the convention of Pauly et al. 1991, bandwidth measured between the middles of the
transitions.

The figures are plain array computations: they are exact for the sampled profile and blind between the samples. Use
at least 20 positions per transition; ``sample_positions`` gives a grid.
"""
from __future__ import annotations

import math

import numpy as np

from . import pulses, rotation
from .units import RAD_PER_UT_MS


def response(kind, a, b):
    """The quantity of the table above for Cayley-Klein parameters (a, b)."""
    if kind == "excitation":
        return 2 * np.conj(a) * b
    if kind == "refocusing":
        return -b * b
    if kind in ("inversion", "saturation"):
        return np.abs(b) ** 2
    raise ValueError(f"kind must be one of {pulses.KINDS}")


def sample_positions(thickness_mm, extent=4.0, per_thickness=200):
    """Positions from -extent to +extent thicknesses, ``per_thickness`` points per thickness, symmetric about 0."""
    half = int(round(extent * per_thickness))
    return np.arange(-half, half + 1) * (thickness_mm / per_thickness)


def _check(z, r):
    z, r = np.asarray(z, float), np.asarray(r)
    if z.ndim != 1 or z.shape != r.shape or z.size < 3 or not np.all(np.diff(z) > 0):
        raise ValueError("positions must be one-dimensional and increasing, with one response per position")
    return z, r


def _bands(z, thickness_mm, transition):
    if not (thickness_mm > 0 and 0 <= transition < 1):
        raise ValueError("thickness_mm must be positive and the transition fraction in [0, 1)")
    inside = np.abs(z) <= (1 - transition) * thickness_mm / 2 * (1 + 1e-12)
    outside = np.abs(z) >= (1 + transition) * thickness_mm / 2 * (1 - 1e-12)
    if not inside.any() or not outside.any():
        raise ValueError("the positions do not reach into both the pass band and the stop band")
    return inside, outside


def _crossings(z, m, level):
    """Positions where the piecewise-linear magnitude crosses ``level``, with the direction (+1 rising)."""
    d = m - level
    idx = np.nonzero((d[:-1] < 0) != (d[1:] < 0))[0]
    pos = z[idx] + (z[idx + 1] - z[idx]) * d[idx] / (d[idx] - d[idx + 1])
    return pos, np.where(d[idx + 1] > d[idx], 1, -1)


def fwhm(z_mm, r, reference=None):
    """Full width at half maximum (mm): the distance between the outermost crossings of half of ``reference`` (the
    largest magnitude by default), with linear interpolation between samples. NaN when there is no pair of crossings."""
    z, r = _check(z_mm, r)
    m = np.abs(r)
    pos, _ = _crossings(z, m, 0.5 * (m.max() if reference is None else reference))
    return float(pos[-1] - pos[0]) if pos.size >= 2 else float("nan")


def passband_ripple(z_mm, r, thickness_mm, transition, target=None):
    """Largest |magnitude - target| in the pass band. Without ``target``: half of the peak-to-peak variation there."""
    z, r = _check(z_mm, r)
    m = np.abs(r)[_bands(z, thickness_mm, transition)[0]]
    return float(0.5 * (m.max() - m.min()) if target is None else np.abs(m - target).max())


def stopband_ripple(z_mm, r, thickness_mm, transition):
    """Largest magnitude in the stop band."""
    z, r = _check(z_mm, r)
    return float(np.abs(r)[_bands(z, thickness_mm, transition)[1]].max())


def transition_width(z_mm, r, lo=0.1, hi=0.9, reference=None):
    """Width of the edges (mm), the mean of the two sides: on each side the distance from the outermost crossing of
    ``hi`` x reference to the nearest crossing of ``lo`` x reference beyond it. Reference = the largest magnitude by
    default. NaN when a side has no such pair."""
    z, r = _check(z_mm, r)
    m = np.abs(r)
    ref = m.max() if reference is None else reference
    hi_pos, _ = _crossings(z, m, hi * ref)
    lo_pos, _ = _crossings(z, m, lo * ref)
    if hi_pos.size < 2:
        return float("nan")
    left, right = lo_pos[lo_pos < hi_pos[0]], lo_pos[lo_pos > hi_pos[-1]]
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(0.5 * ((hi_pos[0] - left[-1]) + (right[0] - hi_pos[-1])))


def out_of_slice_fraction(z_mm, r, thickness_mm):
    """Integral of the magnitude over |z| > thickness / 2 divided by its integral over all positions given (the
    magnitude taken as piecewise linear; the two slice edges are inserted as positions)."""
    z, r = _check(z_mm, r)
    m = np.abs(r)
    edges = np.array([e for e in (-thickness_mm / 2, thickness_mm / 2) if z[0] < e < z[-1] and e not in z])
    zz = np.sort(np.concatenate([z, edges]))
    mm = np.interp(zz, z, m)
    seg = 0.5 * (mm[1:] + mm[:-1]) * np.diff(zz)
    outside = np.abs(0.5 * (zz[1:] + zz[:-1])) > thickness_mm / 2
    total = float(seg.sum())
    return float(seg[outside].sum() / total) if total > 0 else float("nan")


def phase_figures(z_mm, r, thickness_mm, transition=0.0):
    """Phase of a complex response inside the pass band.

    ``slope_rad_per_mm``    slope of the magnitude-weighted straight line through the unwrapped phase
    ``linear_rms_rad``      weighted rms of the phase about that line (what no rephasing lobe can remove)
    ``rms_rad``             weighted rms of the phase about its weighted mean (what is left as delivered)
    ``refocused_fraction``  |integral of r| / integral of |r| over the pass band: the part of the in-slice signal that
                            survives the phase as delivered (1 = perfectly rephased)
    A slope s means the rephasing lobe is off by the area  -s / (2 pi gamma)  (``units.RAD_PER_MT_M_MM_MS``).
    """
    z, r = _check(z_mm, r)
    if not np.iscomplexobj(r):
        raise ValueError("the phase needs a complex response")
    keep = np.abs(z) <= (1 - transition) * thickness_mm / 2 * (1 + 1e-12)
    zz, rr = z[keep], r[keep]
    if zz.size < 3:
        raise ValueError("fewer than three positions in the pass band")
    w = np.abs(rr)
    if not w.sum() > 0:
        return dict(slope_rad_per_mm=float("nan"), linear_rms_rad=float("nan"), rms_rad=float("nan"), refocused_fraction=float("nan"))
    ph = np.unwrap(np.angle(rr))
    zc = zz - np.sum(w * zz) / w.sum()
    pc = ph - np.sum(w * ph) / w.sum()
    slope = float(np.sum(w * zc * pc) / np.sum(w * zc * zc))
    return dict(slope_rad_per_mm=slope, linear_rms_rad=float(np.sqrt(np.sum(w * (pc - slope * zc) ** 2) / w.sum())),
                rms_rad=float(np.sqrt(np.sum(w * pc * pc) / w.sum())), refocused_fraction=float(np.abs(rr.sum()) / w.sum()))


def flip_distribution(z_mm, b, thickness_mm):
    """Distribution of the flip angle 2 asin|b| (degrees) over the nominal slice |z| <= thickness / 2 (every position
    weighted equally), and the largest flip angle outside it."""
    z, b = _check(z_mm, b)
    th = np.degrees(rotation.flip_angle(b))
    inside = np.abs(z) <= thickness_mm / 2 * (1 + 1e-12)
    if not inside.any():
        raise ValueError("no position inside the slice")
    t = th[inside]
    q = np.percentile(t, [5, 25, 50, 75, 95])
    return dict(centre_deg=float(th[np.argmin(np.abs(z))]), mean_deg=float(t.mean()), std_deg=float(t.std()), min_deg=float(t.min()),
                max_deg=float(t.max()), percentiles_deg={p: float(v) for p, v in zip((5, 25, 50, 75, 95), q)},
                outside_max_deg=float(th[~inside].max()) if (~inside).any() else float("nan"))


# ---------------------------------------------------------------------------------------------- mask
def standard_mask(thickness_mm, transition, level, d1, d2):
    """The three bands of a slice specification: magnitude within ``level`` +- ``d1`` in the pass band, at most ``d2``
    in the stop band on either side. Bands are dictionaries (name, z_min, z_max, lower, upper) in mm."""
    p, s = (1 - transition) * thickness_mm / 2, (1 + transition) * thickness_mm / 2
    return [dict(name="passband", z_min=-p, z_max=p, lower=level - d1, upper=level + d1),
            dict(name="stopband_low", z_min=-math.inf, z_max=-s, lower=0.0, upper=d2),
            dict(name="stopband_high", z_min=s, z_max=math.inf, lower=0.0, upper=d2)]


def check_mask(z_mm, r, bands):
    """Pass / fail of a magnitude profile against a list of bands, each ``dict(name, z_min, z_max, lower, upper)``.
    Returns ``dict(passed, first_violation, margins)``: the name of the first band, in the order given, with a sample
    outside its limits (None when all hold), and per band the smallest distance to a limit (negative = violated).
    A band without any sample fails: a mask that is not looked at is not met."""
    z, r = _check(z_mm, r)
    m = np.abs(r)
    margins, first = {}, None
    for band in bands:
        sel = (z >= band["z_min"] - 1e-12) & (z <= band["z_max"] + 1e-12)
        margin = float(min((m[sel] - band["lower"]).min(), (band["upper"] - m[sel]).min())) if sel.any() else float("-inf")
        margins[band["name"]] = margin
        if margin < 0 and first is None:
            first = band["name"]
    return dict(passed=first is None, first_violation=first, margins=margins)


def figures(pulse, z_mm=None, transition=None, df_hz=0.0, b1_scale=1.0):
    """All figures of merit of a pulse with a slice-select gradient, on resonance unless ``df_hz`` is given.
    ``transition``: fractional transition width of the bands (default: the one stored by ``pulses.slr``, else 0.5)."""
    if pulse.thickness_mm is None:
        raise ValueError("the pulse has no slice thickness")
    d = pulse.thickness_mm
    z = sample_positions(d) if z_mm is None else np.asarray(z_mm, float)
    if transition is None:
        transition = pulse.call.get("design", {}).get("transition_fraction", 0.5)
    a, b = pulse.rotate(z, df_hz, b1_scale)
    r = response(pulse.kind, a, b)
    out = dict(kind=pulse.kind, thickness_mm=d, transition_fraction=float(transition), fwhm_mm=fwhm(z, r),
               passband_ripple=passband_ripple(z, r, d, transition), passband_mean=float(np.abs(r)[_bands(z, d, transition)[0]].mean()),
               stopband_ripple=stopband_ripple(z, r, d, transition), transition_width_mm=transition_width(z, r),
               out_of_slice_fraction=out_of_slice_fraction(z, r, d), flip=flip_distribution(z, b, d))
    if np.iscomplexobj(r):
        out["phase"] = phase_figures(z, r, d, transition)
    return out


# ---------------------------------------------------------------------------------------------- bounds
_GL_X, _GL_W = np.polynomial.legendre.leggauss(10)


def _log_integral(f, lo, hi, singular):
    """Integral of ln f from lo to hi for a function with an integrable logarithmic singularity at ``singular`` (one of
    the two ends): Gauss-Legendre on intervals that halve towards the singular end."""
    length = hi - lo
    edges = length * 0.5 ** np.arange(0, 53)                  # distances from the singular end
    a, b = edges[1:], edges[:-1]
    x = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * _GL_X[None, :]
    pts = singular + x if singular == lo else singular - x
    vals = np.log(np.maximum(f(pts.ravel()), 1e-300)).reshape(pts.shape)
    return float(np.sum(0.5 * (b - a)[:, None] * _GL_W[None, :] * vals))


def jensen_integral(pulse, points=1 << 16):
    """J = -(1 / 4 pi) x the integral over one turn of psi of  ln(1 - |B(exp(i psi))|^2) = ln|A|^2,  A and B the
    Shinnar-Le Roux polynomials of the pulse (``pulses.forward_slr``). Positive.

    Midpoint rule on at least ``points`` equally spaced psi (at least 64 per sample). Where the flip angle reaches
    180 degrees A has a zero on the unit circle and the integrand a logarithmic singularity; a zero that falls on a
    grid point would be counted as ln(0). Around every minimum of |A|^2 that the grid does not resolve (the logarithm
    changes by more than 0.5 within two points) the grid cells are replaced by a quadrature graded towards the
    minimum, located by a bounded search on the polynomial itself. What is left of the midpoint rule next to such a
    minimum errs on the LOW side (the logarithm is concave there), by about 1e-7 at the default number of points and
    in proportion to 1 / points: a floor computed from J is slightly too low, never too high."""
    import scipy.optimize
    theta = RAD_PER_UT_MS * pulse.b1_ut * pulse.dt_ms
    coeff, _ = pulses.forward_slr(theta)
    n = coeff.size
    big = 1 << int(math.ceil(math.log2(max(points, 64 * n))))
    step = 2 * math.pi / big
    k = np.arange(n)

    def mag2(psi):
        psi = np.atleast_1d(psi)
        out = np.empty(psi.size)
        for lo in range(0, psi.size, 4096):
            out[lo:lo + 4096] = np.abs(np.exp(-1j * np.outer(psi[lo:lo + 4096], k)) @ coeff) ** 2
        return out

    val = np.abs(np.fft.fft(coeff, big)) ** 2                 # |A|^2 at psi_m = m step
    logv = np.log(np.maximum(val, 1e-300))
    total = float(logv.sum() * step)
    low = (val <= np.roll(val, 1)) & (val <= np.roll(val, -1))
    rough = np.maximum(np.roll(logv, 2), np.roll(logv, -2)) - logv > 0.5
    reach = 8
    done = np.zeros(big, bool)
    for m in np.nonzero(low & rough)[0]:
        if done[m]:
            continue
        cells = np.arange(m - reach, m + reach + 1) % big
        done[cells] = True
        total -= float(logv[cells].sum() * step)
        res = scipy.optimize.minimize_scalar(lambda x: mag2(x)[0], method="bounded", bounds=((m - 1) * step, (m + 1) * step),
                                             options=dict(xatol=1e-14))
        zero = float(res.x)
        left, right = (m - reach - 0.5) * step, (m + reach + 0.5) * step
        total += _log_integral(mag2, left, zero, zero) + _log_integral(mag2, zero, right, zero)
    return -total / (4 * math.pi)


def _log_cos_coefficient(theta_max):
    """c with  -ln cos(theta / 2) <= c theta^2  for |theta| <= theta_max < pi (the ratio grows with theta); 1/8 at 0."""
    if not 0 <= theta_max < math.pi:
        raise ValueError("the largest sub-pulse angle must be below 180 degrees")
    return 0.125 if theta_max < 1e-4 else -math.log(math.cos(theta_max / 2)) / theta_max ** 2


def energy_floor_ut2_ms(jensen, dt_ms, theta_max_rad):
    """Least RF energy (uT^2 ms) of ANY pulse on a raster ``dt_ms`` with sub-pulse angles up to ``theta_max_rad`` whose
    Shinnar-Le Roux polynomial B has the Jensen integral ``jensen``: see ``duration_bound_ms`` for the derivation."""
    return jensen / (_log_cos_coefficient(theta_max_rad) * RAD_PER_UT_MS ** 2 * dt_ms)


def pulse_energy_floor_ut2_ms(pulse):
    """The energy floor of the pulse's own profile magnitude: no pulse on the same raster with sub-pulse angles up to
    this pulse's largest one produces the same |B| with less energy. Reached by a minimum-phase A (``pulses.slr``)."""
    if not (np.all(pulse.dt_ms == pulse.dt_ms[0]) and np.all(pulse.g_mt_m == pulse.g_mt_m[0])):
        raise ValueError("the energy floor holds for a constant raster and a constant gradient")
    theta_max = float(RAD_PER_UT_MS * np.abs(pulse.b1_ut).max() * pulse.dt_ms[0])
    return energy_floor_ut2_ms(jensen_integral(pulse), float(pulse.dt_ms[0]), theta_max)


def duration_bound_ms(flip_min_deg, passband_hz, b1_peak_ut, raster_ms):
    """Lower bound on the duration (ms) of a pulse that turns every spin in a frequency band of width ``passband_hz``
    by at least ``flip_min_deg``, with |B1| <= ``b1_peak_ut``, on a raster ``raster_ms``, under a CONSTANT gradient.

    Returns ``dict(bound_ms, hard_pulse_ms, energy_ms, energy_floor_ut2_ms)``; the bound is the larger of two floors.

    1. Hard-pulse floor. A product of rotations cannot turn a vector further than the sum of their angles (the
       triangle inequality for the angular distance on the sphere; the precessions about z between the sub-pulses do
       not move +z). So the flip angle anywhere is at most the sum of |theta_j| <= 2 pi gamma B1max T:
           T >= flip_min / (2 pi gamma B1max).
       This floor also holds under a time-varying gradient (it does not look at the gradient).
    2. Energy floor. In the hard-pulse model of ``rotation.rotate`` (``split``) on a uniform raster under a constant
       gradient, a and b are polynomials A, B in x = exp(i psi) with |A|^2 + |B|^2 = 1 on the unit circle and
       A(0) = prod cos(theta_j / 2) (``pulses`` docstring). Jensen's formula for A,
           (1 / 2 pi) integral ln|A(exp(i psi))| d psi = ln|A(0)| + sum over zeros inside the circle of ln(1 / |zero|),
       gives   -sum_j ln cos(theta_j / 2) >= J = -(1 / 4 pi) integral ln(1 - |B|^2) d psi,   with equality when A has
       no zero inside the circle (minimum phase). The band occupies the fraction passband_hz x dt of the turn, and
       there 1 - |B|^2 <= cos^2(flip_min / 2), elsewhere the integrand is >= 0, so
           J >= -passband_hz dt ln cos(flip_min / 2).
       With |theta_j| <= theta_max = 2 pi gamma B1max dt and -ln cos(t / 2) <= c(theta_max) t^2 (c = 1/8 for small
       angles), the energy E = sum theta_j^2 / ((2 pi gamma)^2 dt) is at least J / (c (2 pi gamma)^2 dt), and
       E <= B1max^2 T:
           T >= E_floor / B1max^2.

    Assumptions, all needed: one transmit channel; no relaxation; the profile is a function of frequency, i.e. the
    gradient is constant (a variable-rate pulse is outside the class and only floor 1 applies to it); the pulse is a
    train of hard sub-pulses on a uniform raster (the simulator's model), each of less than 180 degrees at B1max
    (otherwise the energy floor is empty and only floor 1 is returned); the band is narrower than the raster's
    Nyquist range. An exact 180 degrees over a band of non-zero width needs A = 0 on an arc, hence a sub-pulse of 180
    degrees: without one the energy floor is infinite, which is correct. The bound uses the pass band only: it is not tight, and it says nothing about the transition width.
    """
    if not (0 < flip_min_deg <= 180 and passband_hz >= 0 and b1_peak_ut > 0 and raster_ms > 0):
        raise ValueError("flip_min_deg in (0, 180], passband_hz >= 0, b1_peak_ut > 0, raster_ms > 0")
    if passband_hz * raster_ms * 1e-3 > 1:
        raise ValueError("the pass band is wider than one turn of the raster")
    theta_max = RAD_PER_UT_MS * b1_peak_ut * raster_ms
    half = math.radians(flip_min_deg) / 2
    hard = math.radians(flip_min_deg) / (RAD_PER_UT_MS * b1_peak_ut)
    if flip_min_deg >= 180:
        jensen = math.inf if passband_hz > 0 else 0.0
    else:
        jensen = -passband_hz * raster_ms * 1e-3 * math.log(math.cos(half))
    # a raster sample that can itself turn by 180 degrees (to within rounding) makes the energy floor empty
    floor = energy_floor_ut2_ms(jensen, raster_ms, theta_max) if theta_max < math.pi * (1 - 1e-9) else 0.0
    energy = floor / b1_peak_ut ** 2
    return dict(bound_ms=max(hard, energy), hard_pulse_ms=hard, energy_ms=energy, energy_floor_ut2_ms=floor)


def variable_rate_floor_ms(pulse, b1_peak_ut):
    """Shortest duration any variable-rate rescaling of this pulse can reach under the peak limit:
    sum |theta_j| / (2 pi gamma B1max), reached when |B1| is at the limit throughout (and the gradient follows)."""
    return float(np.sum(np.abs(pulse.b1_ut) * pulse.dt_ms) / b1_peak_ut)
