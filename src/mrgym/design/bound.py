"""A proven lower bound on the duration of a slice-selective pulse that meets a flip-angle mask, and the pulse that
comes closest to it. Derivation, with every assumption: docs/physics/design.md, section 4a.

The class. A pulse of ``n`` samples on one raster ``dt``, played as ``n`` hard sub-pulses (``rotation.rotate``, model
``split``) under ONE constant slice gradient G. Then the rotation of a spin at position z is (a, b) with
|b|^2 = P(psi), psi = u z + const, u = 2 pi gamma G dt (the precession per sample and per mm), and P an arbitrary
trigonometric polynomial of degree n - 1 with 0 <= P <= 1 (Shinnar-Le Roux; Fejer-Riesz for the converse). A transmit
scale, an amplifier law, an off-resonance and a gradient that is late by a constant time change the sub-pulse angles
or shift psi; they do not leave the class. A variable-rate pulse and a pulse whose samples hold B1 while the
gradient acts (model ``piecewise``) are NOT in the class.

The mask, on the flip angle theta = 2 asin sqrt(P), symmetric about the slice centre:
    in the slice   |z| <= z_in = (1 - W) d / 2                theta within flip +- tol_in
    outside        z_out = (1 + W) d / 2 <= |z| <= pi / u     theta <= max_out        (pi / u: half the repeat distance)
    between        free

Three statements, each a theorem for the class:
  1. certificate   ``n + 1`` positions and signed weights lambda with  sum_i lambda_i P(psi_i) = 0  for EVERY
                   trigonometric polynomial of degree n - 1 (a null vector of the cosine matrix), while the mask
                   forces the same sum below zero. No ``n``-sample pulse meets the mask at those positions
                   (``Certificate``, ``check``). The floating-point residual of the null vector is charged against
                   the margin with |coefficients of P| <= 1.
  2. every gradient the mask of a cell of gradients [u_lo, u_hi], weakened to the pass edge of u_lo and the stop edge
                   of u_hi, is implied by the mask of every u in the cell; a certificate for the weakened mask holds
                   for the cell. Below u = (theta_lo - theta_out) / ((n - 1) (z_out - z_in)) the Bernstein-Szego
                   inequality |d theta / d psi| <= n - 1 forbids the transition. A finite cover (``cover``).
  3. energy        Jensen's formula for the polynomial A:  sum_j -ln cos^2(theta_j / 2) >= -(1 / 2 pi) int ln(1 - P),
                   which bounds the gradient from above when the RF energy (B1rms over a repetition) or the peak B1
                   is limited (``gradient_cap``).
``duration_bound`` puts them together: the smallest ``n`` for which statements 1 to 3 leave a gradient.

The pulse that comes closest (``equiripple_b``): the weighted-Chebyshev-optimal P for a mask (a Remez exchange in
the cosine basis), its minimum-phase factor B (real cepstrum), the minimum-phase A and the inverse Shinnar-Le Roux
transform. ``tightness`` builds it at the sample count of the bound and plays it on the verifier's positions.
"""
from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np
import scipy.linalg
import scipy.signal

from .units import RAD_PER_MT_M_MM_MS, RAD_PER_UT_MS, RASTER_MS

SLACK_RAD = 0.002            # eta: between two neighbouring in-slice positions of the verifier the flip angle cannot dip further below the mask
MARGIN = 1e-9                # a certificate counts when its margin exceeds its residual by this (weights normalised to sum |lambda| = 1)
CELL_RATIO_MIN = 1.00001     # the narrowest cell of gradients a certificate is asked for
DENSITY = 16                 # grid points per extremum spacing pi / n of the exchange
FLOOR = 1e-4                 # the designed P stays above this part of the stop level, so that its logarithm exists
MAX_SAMPLES = 2048


# ---------------------------------------------------------------------------------------------- the mask
def precession(g_mt_m: float, raster_ms: float = RASTER_MS) -> float:
    """u: precession per sample and per mm of position under the gradient ``g_mt_m``, rad/mm."""
    return RAD_PER_MT_M_MM_MS * float(g_mt_m) * raster_ms


def p_of(flip_deg) -> float:
    """|b|^2 of a flip angle."""
    return math.sin(math.radians(flip_deg) / 2) ** 2


def _g(x: float) -> float:
    """-ln cos^2(x / 2): what a sub-pulse of angle x adds to -ln A(0)^2, and a flip angle x to -ln(1 - P)."""
    return -2.0 * math.log(math.cos(x / 2))


def _g_integral(theta: float, steps: int = 4000) -> float:
    """A lower bound of the integral of ``_g`` from 0 to theta: the left Riemann sum of an increasing function."""
    h = theta / steps
    return float(sum(_g(i * h) for i in range(steps)) * h)


@dataclasses.dataclass(frozen=True)
class Mask:
    """A flip-angle mask in the units the proof uses: positions in mm, levels as |b|^2."""
    z_in: float
    z_out: float
    lo: float
    hi: float
    stop: float
    theta_lo: float              # rad: the smallest flip angle the slice allows
    theta_out: float             # rad: the largest one outside

    @classmethod
    def of(cls, spec: dict) -> Mask:
        d, w = float(spec["thickness_mm"]), float(spec["transition"])
        flip, tol, out = float(spec["flip_deg"]), float(spec["tol_in_deg"]), float(spec["max_out_deg"])
        if not (0 < w < 1 and d > 0 and 0 < tol < flip and flip + tol < 180 and 0 < out < flip - tol):
            raise ValueError("a mask needs 0 < transition < 1, 0 < tol_in < flip, flip + tol_in < 180 and 0 < max_out < flip - tol_in")
        return cls((1 - w) * d / 2, (1 + w) * d / 2, p_of(flip - tol), p_of(flip + tol), p_of(out), math.radians(flip - tol), math.radians(out))

    def on_circle(self, u_lo: float, u_hi: float | None = None) -> Problem:
        """The mask in psi for the gradient u_lo, or weakened so that it is implied by the mask of every u in [u_lo, u_hi]."""
        return Problem(u_lo * self.z_in, (u_lo if u_hi is None else u_hi) * self.z_out, self.lo, self.hi, self.stop)

    def slope_precession(self, n: int) -> float:
        """Below this u no ``n``-sample pulse makes the transition (Bernstein-Szego): (n - 1) u (z_out - z_in) < theta_lo - theta_out."""
        return math.inf if n < 2 else (self.theta_lo - self.theta_out) / ((n - 1) * (self.z_out - self.z_in))


@dataclasses.dataclass(frozen=True)
class Problem:
    """lo <= P(psi) <= hi for |psi| <= a, 0 <= P(psi) <= stop for b <= |psi| <= pi."""
    a: float
    b: float
    lo: float
    hi: float
    stop: float


# ---------------------------------------------------------------------------------------------- the best polynomial
@dataclasses.dataclass(frozen=True)
class Fit:
    q: np.ndarray                # P(psi) = sum_k q[k] cos(k psi), k = 0 .. n - 1
    level: float                 # the levelled error of the last reference, in units of the tolerances (a lower bound of the best, when exact)
    worst: float                 # the largest error of P on the grid, same units: <= 1 means the grid mask is met
    reference: np.ndarray        # psi of the n + 1 reference points


def _grid(n: int, p: Problem, density: int):
    step = math.pi / (n * density)
    m_pass = max(4, int(math.ceil(p.a / step)) + 1)
    m_stop = max(4, int(math.ceil((math.pi - p.b) / step)) + 1)
    psi = np.concatenate([np.linspace(0.0, p.a, m_pass), np.linspace(p.b, math.pi, m_stop)])
    return psi, m_pass


def _extrema(e: np.ndarray, m_pass: int, count: int):
    """Indices of ``count`` alternating extrema of the error ``e`` on the two bands, or None."""
    cand = []
    for lo, hi in ((0, m_pass), (m_pass, e.size)):
        seg = np.abs(e[lo:hi])
        left = np.concatenate([[True], seg[1:] >= seg[:-1]])
        right = np.concatenate([seg[:-1] > seg[1:], [True]])
        cand += list(lo + np.nonzero(left & right)[0])
    out = []                                                    # one extremum per run of equal sign: the largest
    for i in cand:
        if out and (e[i] > 0) == (e[out[-1]] > 0):
            if abs(e[i]) > abs(e[out[-1]]):
                out[-1] = i
        else:
            out.append(i)
    while len(out) > count:
        if len(out) - count == 1 or (e[out[0]] > 0) == (e[out[-1]] > 0):   # an end goes: the smaller one
            out.pop(0 if abs(e[out[0]]) < abs(e[out[-1]]) else -1)
        else:                                                   # a neighbouring pair goes: the one whose smaller member is smallest
            j = int(np.argmin([min(abs(e[out[k]]), abs(e[out[k + 1]])) for k in range(len(out) - 1)]))
            del out[j:j + 2]
    return np.array(out) if len(out) == count else None


def _error(q: np.ndarray, psi: np.ndarray, p: Problem, floor: float):
    """(error of P at the points ``psi`` in units of the tolerance of their band, target, tolerance)."""
    inside = psi <= p.a
    centre = np.where(inside, 0.5 * (p.lo + p.hi), 0.5 * p.stop * (1 + floor))
    half = np.where(inside, 0.5 * (p.hi - p.lo), 0.5 * p.stop * (1 - floor))
    return (np.cos(np.outer(psi, np.arange(q.size))) @ q - centre) / half, centre, half


def _refine(q: np.ndarray, x: np.ndarray, lo: np.ndarray, hi: np.ndarray, p: Problem, floor: float) -> np.ndarray:
    """Each point of ``x`` moved to where |error| is largest within [lo, hi] (its neighbours on the grid, inside its
    band): four rounds of nine trial points, the interval shrinking fourfold around the best."""
    lo, hi = lo.copy(), hi.copy()
    for _ in range(4):
        trial = lo[None, :] + (hi - lo)[None, :] * np.linspace(0.0, 1.0, 9)[:, None]
        e = np.abs(_error(q, trial.ravel(), p, floor)[0]).reshape(9, -1)
        x = trial[np.argmax(e, 0), np.arange(x.size)]
        width = (hi - lo) / 8
        lo, hi = np.maximum(lo, x - width), np.minimum(hi, x + width)
    return x


def best(n: int, p: Problem, floor: float = 0.0, density: int = DENSITY, rounds: int = 40) -> Fit | None:
    """The even trigonometric polynomial of degree n - 1 that is closest to the mask ``p`` in the weighted Chebyshev
    sense: centre and half-width of each band are the target and the unit of the error (``floor``: the stop band is
    floor x stop <= P <= stop). A Remez exchange started from ``scipy.signal.remez``: the extrema of the error are
    found on a grid and then moved off it to where the error is largest (``_refine``), and the polynomial that
    levels the error on them is solved for, until the largest error is the level. None when ``n`` or the bands leave
    nothing to solve. Nothing proven rests on this function: ``check`` and the profile of the built pulse are the judges."""
    if n < 2 or not (0 < p.a < p.b < math.pi):
        return None
    psi, m_pass = _grid(n, p, density)
    cos = np.cos(np.outer(psi, np.arange(n)))
    centre, half = _error(np.zeros(n), psi, p, floor)[1:]
    q = None
    try:
        h = scipy.signal.remez(2 * n - 1, [0, p.a / (2 * math.pi), p.b / (2 * math.pi), 0.5], [centre[0], centre[-1]],
                               weight=[1 / half[0], 1 / half[-1]], fs=1.0, maxiter=60, grid_density=density)
        if np.all(np.isfinite(h)):
            q = h[n - 1:].copy()
            q[1:] *= 2
    except (ValueError, RuntimeError):                          # the exchange below starts from a least-squares fit instead
        pass
    if q is None:
        q = np.linalg.lstsq(cos / half[:, None], centre / half, rcond=None)[0]
    sign = (-1.0) ** np.arange(n + 1)
    level, ref, kept = 0.0, None, None
    for _ in range(rounds):
        idx = _extrema(cos @ q * (1 / half) - centre / half, m_pass, n + 1)
        if idx is None:
            break
        first, last = np.where(idx < m_pass, 0, m_pass), np.where(idx < m_pass, m_pass - 1, psi.size - 1)
        x = _refine(q, psi[idx], psi[np.maximum(idx - 1, first)], psi[np.minimum(idx + 1, last)], p, floor)
        e, c, hw = _error(q, x, p, floor)
        if ref is not None and float(np.abs(e).max()) <= level * (1 + 1e-9):   # the largest error is the level: the best polynomial
            break
        try:
            sol = np.linalg.solve(np.hstack([np.cos(np.outer(x, np.arange(n))), (sign * hw)[:, None]]), c)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(sol)):
            break
        kept = (q, level, ref)
        q, level, ref = sol[:n], abs(float(sol[n])), x
    if ref is None:
        return None
    fit = Fit(q, level, _worst(q, p, floor), ref)
    if kept is not None and kept[2] is not None and _worst(kept[0], p, floor) < fit.worst:   # a last step that made it worse is not taken
        fit = Fit(kept[0], kept[1], _worst(kept[0], p, floor), kept[2])
    return fit


def _on_circle(q: np.ndarray, points: int) -> np.ndarray:
    """P = sum_k q[k] cos(k psi) at psi = 2 pi m / points, m = 0 .. points - 1 (one FFT)."""
    full = np.zeros(points)
    full[0] = q[0]
    full[1:q.size] = 0.5 * q[1:]
    full[points - q.size + 1:] = 0.5 * q[:0:-1]
    return np.fft.fft(full).real


def _worst(q: np.ndarray, p: Problem, floor: float = 0.0) -> float:
    """The largest error of P against the mask ``p`` in units of its tolerances, on 128 points per extremum spacing
    and at the two band edges."""
    n = q.size
    big = 1 << int(math.ceil(math.log2(256 * n)))
    psi = 2 * math.pi * np.arange(big // 2 + 1) / big
    val = _on_circle(q, big)[:big // 2 + 1]
    edge = np.cos(np.outer([p.a, p.b], np.arange(n))) @ q
    in_pass = np.concatenate([val[psi <= p.a], edge[:1]])
    in_stop = np.concatenate([val[psi >= p.b], edge[1:]])
    return float(max(np.abs(in_pass - 0.5 * (p.lo + p.hi)).max() / (0.5 * (p.hi - p.lo)),
                     np.abs(in_stop - 0.5 * p.stop * (1 + floor)).max() / (0.5 * p.stop * (1 - floor))))


# ---------------------------------------------------------------------------------------------- the certificate
@dataclasses.dataclass(frozen=True)
class Certificate:
    """No trigonometric polynomial P of degree n - 1 with 0 <= P <= 1 has lower_i <= P(+-psi_i) <= upper_i for all i."""
    n: int
    psi: np.ndarray              # n + 1 points in [0, pi]
    weight: np.ndarray           # lambda, sum |lambda| = 1
    lower: np.ndarray
    upper: np.ndarray
    in_slice: np.ndarray         # bool: which band a point lies in


COS_ROUNDING = 8e-13         # of one cosine: k psi rounds by at most half a unit in the last place of a number below 6434, the cosine by one unit
VERIFIER_ROUNDING = 1e-10    # of |b|^2 as the verifier computes it at the position psi_i / u (a product of n rotations, the division, the conversion to degrees)


def check(c: Certificate) -> float:
    """Margin of a certificate: positive (above ``MARGIN``) proves it. In floating point, with the rounding charged.

    For every even P = sum_k q_k cos(k psi) of degree n - 1:  sum_i lambda_i P(psi_i) = sum_k q_k r_k,
    r_k = sum_i lambda_i cos(k psi_i). 0 <= P <= 1 gives 0 <= q_0 <= 1 and |q_k| <= 2 (q_k / 2 is a Fourier coefficient
    of a function between 0 and 1), so the sum is at least -(|r_0| + 2 sum |r_k|). The mask gives at most
    U = sum_(lambda > 0) lambda_i upper_i + sum_(lambda < 0) lambda_i lower_i. Contradiction when U < -(|r_0| + 2 sum |r_k|).
    A P that is not even meets the same mask with its even part, since the points come in pairs +-psi_i.
    Charged on top (sum |lambda| = 1): every computed cosine is within ``COS_ROUNDING`` of the cosine of the stored
    psi_i, every sum of n + 1 terms within (n + 1) x 2.3e-16 of its terms' sum; and the values of P the verifier
    computes at these positions are within ``VERIFIER_ROUNDING`` of the polynomial's."""
    n = c.n
    if not (c.psi.shape == c.weight.shape == c.lower.shape == c.upper.shape == (n + 1,)) or n > MAX_SAMPLES:
        return -math.inf
    if not (np.all(c.psi >= 0) and np.all(c.psi <= math.pi) and np.all(c.lower <= c.upper) and np.all(c.lower >= 0) and np.all(c.upper <= 1)):
        return -math.inf
    lam = c.weight / np.abs(c.weight).sum()
    r = lam @ np.cos(np.outer(c.psi, np.arange(n)))
    rounding = COS_ROUNDING + (n + 1) * 2.3e-16
    residual = abs(r[0]) + rounding + 2 * float(np.abs(r[1:]).sum() + (n - 1) * rounding)
    u = float(lam[lam > 0] @ c.upper[lam > 0] + lam[lam < 0] @ c.lower[lam < 0])
    return -u - residual - VERIFIER_ROUNDING - (n + 1) * 2.3e-16


def certificate(n: int, p: Problem) -> Certificate | None:
    """A certificate that no ``n``-sample pulse meets ``p``, or None (there may be such a pulse, or the arithmetic gave up)."""
    fit = best(n, p)
    if fit is None or fit.level <= 1.0:
        return None
    psi = fit.reference
    cos = np.cos(np.outer(psi, np.arange(n)))
    lam = scipy.linalg.qr(cos, mode="full")[0][:, -1]           # orthogonal to the n columns: lambda^T cos = 0
    lam = lam / np.abs(lam).sum()
    inside = psi <= p.a
    lower, upper = np.where(inside, p.lo, 0.0), np.where(inside, p.hi, p.stop)
    out = None
    for s in (1.0, -1.0):
        c = Certificate(n, psi, s * lam, lower, upper, inside)
        if out is None or check(c) > check(out):
            out = c
    return out if check(out) > MARGIN else None


def cover(n: int, mask: Mask, u_top: float) -> list | None:
    """Certificates that no ``n``-sample pulse meets ``mask`` under ANY gradient with precession u <= u_top:
    [(u_lo, u_hi, Certificate)] for cells that reach from the slope limit to u_top, or None when that is not shown.
    A cell that is not certified is halved (geometrically) until it is narrower than ``CELL_RATIO_MIN``."""
    if n < 1:
        return []
    low = mask.slope_precession(n)
    if u_top < low:                                             # the transition alone forbids it
        return []
    if n < 2 or u_top * mask.z_out >= math.pi:
        return None
    cells = []

    def certify(lo, hi):
        c = certificate(n, mask.on_circle(lo, hi))
        if c is not None:
            cells.append((lo, hi, c))
            return True
        if hi / lo < CELL_RATIO_MIN:
            return False
        mid = math.sqrt(lo * hi)
        return certify(mid, hi) and certify(lo, mid)            # the upper half first: that is where it fails
    return sorted(cells, key=lambda c: c[0]) if certify(low * (1 - 1e-9), u_top) else None


def fits(n: int, mask: Mask, u: float) -> bool:
    """Whether the exchange finds an ``n``-sample polynomial inside ``mask`` at the gradient u (numerical; a search aid)."""
    if u * mask.z_out >= math.pi:
        return False
    f = best(n, mask.on_circle(u))
    return f is not None and f.worst <= 1.0


# ---------------------------------------------------------------------------------------------- energy and peak
def gradient_cap(n: int, mask: Mask, tx_scale: float = 1.0, b1_peak_ut: float | None = None, energy_ut2_ms: float | None = None,
                 raster_ms: float = RASTER_MS) -> float:
    """The largest precession u an ``n``-sample pulse of the class can have when it meets the in-slice mask on
    positions ``SLACK_RAD`` fine, its NOMINAL samples stay below ``b1_peak_ut`` and its nominal energy below
    ``energy_ut2_ms``, on a chain that delivers at most ``tx_scale`` times the nominal amplitude. inf when neither
    limit is given.

    theta_j: the delivered sub-pulse angles, |theta_j| <= x = gamma tx_scale b1_peak dt. Jensen:
        sum_j g(theta_j) >= L = -(1 / 2 pi) int ln(1 - P) d psi,      g(t) = -ln cos^2(t / 2).
    Right side: in the slice theta >= theta_lo - eta, and next to it theta falls no faster than n - 1 per unit of psi:
        L >= (u z_in g(theta_lo - eta) + I(theta_lo) / (n - 1)) / pi,      I(t) = int_0^t g.
    Left side: at most n g(x) (peak), and at most c(x) / 4 x sum theta_j^2 = c(x) / 4 x gamma^2 tx_scale^2 dt x energy,
    c(x) = g(x) / (x^2 / 4) >= g(t) / (t^2 / 4) for |t| <= x (g / t^2 increases)."""
    if b1_peak_ut is None and energy_ut2_ms is None:
        return math.inf
    if b1_peak_ut is None:
        raise ValueError("the energy argument needs a peak limit (it bounds the sub-pulse angle)")
    x = RAD_PER_UT_MS * tx_scale * b1_peak_ut * raster_ms
    if not x < math.pi or n < 1:
        return math.inf if n >= 1 else -math.inf
    lam = _g(mask.theta_lo - SLACK_RAD)
    most = n * _g(x)                                            # the largest sum_j g(theta_j) the limits leave
    if energy_ut2_ms is not None:
        most = min(most, _g(x) / (x * x / 4) / 4 * (RAD_PER_UT_MS * tx_scale) ** 2 * raster_ms * energy_ut2_ms)
    plain = math.pi * most / (mask.z_in * lam)                  # from the slice alone: u z_in g(theta_lo - eta) / pi <= most
    if plain * mask.z_in > math.pi:                             # the argument needs the slice to fit on the circle: no cap is claimed
        return math.inf
    if n < 2:
        return plain
    tailed = plain - _g_integral(mask.theta_lo) / (n - 1) / (mask.z_in * lam)
    # the two tails next to the slice count only when they fit on the circle beside it: u z_in + theta_lo / (n - 1) <= pi for
    # every u up to the plain cap (then also for every u up to the smaller one)
    return tailed if plain * mask.z_in + mask.theta_lo / (n - 1) <= math.pi else plain


# ---------------------------------------------------------------------------------------------- the bound
def _key(spec: dict) -> tuple:
    return tuple(float(spec[k]) for k in ("thickness_mm", "transition", "flip_deg", "tol_in_deg", "max_out_deg"))


@functools.lru_cache(maxsize=512)
def _bound(key: tuple, g_max: float, tx_scale: float, b1_peak, energy, raster_ms: float):
    mask = Mask.of(dict(zip(("thickness_mm", "transition", "flip_deg", "tol_in_deg", "max_out_deg"), key)))
    u_max = precession(g_max, raster_ms)

    def cap(n):
        return min(u_max, gradient_cap(n, mask, tx_scale, b1_peak, energy, raster_ms))

    def found(n):
        return cap(n) > 0 and fits(n, mask, cap(n))
    hi = 8
    while not found(hi):
        hi *= 2
        if hi > MAX_SAMPLES:
            raise ValueError("no pulse of up to 2048 samples meets this mask under these limits")
    lo = hi // 2 if hi > 8 else 1                               # not found at lo (or lo = 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if found(mid) else (mid, hi)
    first = hi                                                  # the smallest n at which the exchange finds a polynomial
    for n in range(first, max(first - 4, 1), -1):               # the largest n whose predecessor is certified impossible
        cells = cover(n - 1, mask, cap(n - 1)) if cap(n - 1) > 0 else []
        if cells is not None:
            return dict(samples=n, duration_ms=round(n * raster_ms, 6), found_samples=first, cells=cells, cap=cap(n - 1),
                        gradient_cap_mt_m=cap(n) / (RAD_PER_MT_M_MM_MS * raster_ms), mask=mask, limited_by="gradient" if cap(n - 1) >= u_max else "rf")
    n = 1                                                       # no certificate: the slope of the transition alone
    while cap(n) <= 0 or cap(n) < mask.slope_precession(n):
        n += 1
    return dict(samples=n, duration_ms=round(n * raster_ms, 6), found_samples=first, cells=[], cap=cap(n - 1) if n > 1 else 0.0,
                gradient_cap_mt_m=cap(n) / (RAD_PER_MT_M_MM_MS * raster_ms), mask=mask, limited_by="slope")


TOLERANCE = 2e-7             # the limits are taken this much wider than stated: what a limit check and a raster check let through


def duration_bound(spec: dict, g_max_mt_m: float, tx_scale: float = 1.0, b1_peak_ut: float | None = None, energy_ut2_ms: float | None = None,
                   raster_ms: float = RASTER_MS) -> dict:
    """The bound: no pulse of the class with fewer than ``samples`` samples meets the mask of ``spec`` (on the
    positions of ``positions``) with a gradient of at most ``g_max_mt_m`` and nominal samples within the two RF limits,
    on a chain that delivers ``tx_scale`` (or less) times the nominal amplitude. The three limits and the raster
    are taken ``TOLERANCE`` wider than stated (the precession per sample up to (1 + 2 TOLERANCE) times that of the
    gradient limit on the stated raster), so that a check of the limits with a relative tolerance of 1e-9 and a
    raster within 1e-7 of the stated one are inside the hypotheses.

    ``samples``, ``duration_ms``; ``cells``: the certificates for ``samples - 1`` ([] when the slope or the energy alone
    decide); ``cap``: the precession up to which they reach; ``found_samples``: the smallest count at which the exchange
    finds a polynomial (the bound is tight when the two are equal); ``limited_by``: gradient, rf or slope."""
    wide = 1 + TOLERANCE
    return dict(_bound(_key(spec), float(g_max_mt_m) * wide ** 2, float(tx_scale), None if b1_peak_ut is None else float(b1_peak_ut) * wide ** 2,
                       None if energy_ut2_ms is None else float(energy_ut2_ms) * wide ** 4, float(raster_ms)))


def positions(spec: dict, n: int, u: float, cells=()) -> tuple:
    """(in-slice positions, positions outside) on which the mask of ``spec`` is looked at for a pulse of ``n`` samples
    with precession u, mm. In the slice so close that the flip angle cannot dip ``SLACK_RAD`` below its values between
    two of them (spacing <= 2 eta / ((n - 1) u)); outside 1/100 of the thickness apart, out to half the repeat
    distance pi / u; and the positions of the certificate of the cell that holds u, when ``cells`` has one."""
    m = Mask.of(spec)
    d = float(spec["thickness_mm"])
    step_in = min(d / 100, 2 * SLACK_RAD / (max(n - 1, 1) * u))
    z_in = np.linspace(-m.z_in, m.z_in, 2 * int(math.ceil(m.z_in / step_in)) + 1)
    far = math.pi / u
    side = np.linspace(m.z_out, far, max(2, int(math.ceil((far - m.z_out) / (d / 100))) + 1)) if far > m.z_out else np.array([m.z_out])
    extra_in, extra_out = [], []
    for lo, hi, c in cells:
        if lo <= u <= hi:
            z = c.psi / u
            extra_in += list(z[c.in_slice])
            extra_out += list(z[~c.in_slice])
            break
    z_in = np.concatenate([z_in, extra_in, [-z for z in extra_in]])
    z_out = np.concatenate([-side[::-1], side, extra_out, [-z for z in extra_out]])
    return z_in, z_out


def mask_met(pulse, spec: dict, chain=None, cells=()) -> tuple:
    """(in the slice, outside): whether ``pulse``, played through ``chain`` (default: the ideal chain of hard
    sub-pulses), meets the two bands of the mask of ``spec`` on ``positions``."""
    from . import hardware
    chain = chain or hardware.Chain(model="split", guard=True)
    z_in, z_out = positions(spec, pulse.n, precession(float(pulse.g_mt_m[0]), float(pulse.dt_ms[0])), cells)
    f_in, f_out = hardware.flip_deg(pulse, chain, z_in), hardware.flip_deg(pulse, chain, z_out)
    lo, hi = spec["flip_deg"] - spec["tol_in_deg"], spec["flip_deg"] + spec["tol_in_deg"]
    return bool(f_in.min() >= lo and f_in.max() <= hi), bool(f_out.max() <= spec["max_out_deg"])


def grid(flips=(20.0, 30.0, 40.0), thicknesses=(4.0, 5.0, 6.0), transitions=(0.2, 0.3, 0.4), tolerances=((0.04, 0.03), (0.05, 0.04))) -> list:
    """A grid of specifications: what the validation rows and the tightness table run over."""
    return [dict(thickness_mm=d, transition=w, flip_deg=f, tol_in_deg=round(t * f, 1), max_out_deg=round(o * f, 1))
            for f in flips for d in thicknesses for w in transitions for t, o in tolerances]


def tightness(specs=None, g_max_mt_m: float = 32.0, raster_ms: float = RASTER_MS) -> list:
    """Per specification: the bound at the gradient limit without RF limits, and the constructive design at exactly
    that many samples: dict(spec, samples, found_samples, certificates, built (samples of the shortest ``minphase`` pulse
    from ``samples`` upwards that meets the mask on ``positions``), ratio = built / samples, use, refocused)."""
    from . import hardware, pulses
    from .units import HZ_PER_MT_M_MM
    out = []
    for spec in specs or grid():
        b = duration_bound(spec, g_max_mt_m, raster_ms=raster_ms)
        for n in range(b["samples"], b["samples"] + 40):
            t = round(n * raster_ms, 6)
            tbw = math.floor(HZ_PER_MT_M_MM * g_max_mt_m * spec["thickness_mm"] * t * 1e-3 * 1e6) / 1e6
            try:
                p = pulses.minphase(spec["flip_deg"], t, tbw, spec["thickness_mm"], spec["transition"], spec["tol_in_deg"], spec["max_out_deg"])
            except ValueError:
                continue
            if all(mask_met(p, spec)):
                break
        m = Mask.of(spec)
        z = np.linspace(-m.z_in, m.z_in, 401)
        a_, b_ = hardware.rotate(p, hardware.Chain(model="split", guard=True), z)
        out.append(dict(spec=spec, samples=b["samples"], found_samples=b["found_samples"], certificates=len(b["cells"]), built=n,
                        ratio=n / b["samples"], use=p.call["design"]["use"], refocused=pulses.refocused(2 * np.conj(a_) * b_, z), pulse=p))
    return out


# ---------------------------------------------------------------------------------------------- the pulse that comes closest
def factor(q: np.ndarray, oversample: int = 512) -> np.ndarray:
    """Coefficients (index = power of x) of the minimum-phase polynomial B with |B(exp(i psi))|^2 = P(psi) =
    sum_k q[k] cos(k psi) > 0: the real cepstrum of P folded onto the positive powers."""
    n = q.size
    big = 1 << int(math.ceil(math.log2(max(64, oversample * n))))
    p = _on_circle(q, big)
    if p.min() <= 0:
        raise ValueError("the polynomial is not positive on the unit circle")
    cep = np.fft.ifft(0.5 * np.log(p))
    cep[1:big // 2] *= 2
    cep[big // 2 + 1:] = 0
    return np.fft.ifft(np.exp(np.fft.fft(cep)))[:n]


def equiripple_b(n: int, spec: dict, u: float, share=(1.0, 1.0)):
    """(B coefficients, record) of the ``n``-sample pulse for the mask of ``spec`` (tolerances scaled by ``share``) at
    precession u: the weighted-Chebyshev-optimal |B|^2 and its minimum-phase factor, a rotation about +x at the slice
    centre. When no polynomial of this length fits, the tolerances are widened in steps until one does and the
    pulse for those is returned. ``record["use"]``: the largest deviation of the result from the mask in units of
    the two tolerances asked for (in the slice |flip - flip_deg| / tol_in, outside flip / max_out), on 128 points
    per extremum spacing; at most 1 exactly when the pulse meets the mask there. Raises ValueError when nothing can be built."""
    tol_in, max_out, flip = share[0] * float(spec["tol_in_deg"]), share[1] * float(spec["max_out_deg"]), float(spec["flip_deg"])
    widen, fit = 1.0, None
    for _ in range(8):
        m = Mask.of(dict(spec, tol_in_deg=min(widen * tol_in, 0.98 * flip, 0.98 * (180 - flip)), max_out_deg=min(widen * max_out, 0.9 * (flip - min(widen * tol_in, 0.98 * flip)))))
        if u * m.z_out >= math.pi:
            raise ValueError("the gradient is too strong for the raster: the stop band starts beyond half the repeat distance")
        fit = best(n, m.on_circle(u), floor=FLOOR)
        if fit is None:
            raise ValueError("too few samples for a slice profile")
        if fit.worst <= 1.0:
            break
        widen *= 1.05 * max(1.0, math.sqrt(fit.worst))
    big = 1 << int(math.ceil(math.log2(256 * n)))
    p = _on_circle(fit.q, big)
    if fit.worst > 1.0 or p.min() <= 0 or p.max() >= 1:
        raise ValueError("too short for this mask: the best profile of this length leaves the range of a flip angle")
    psi = 2 * math.pi * np.arange(big // 2 + 1) / big
    theta = np.degrees(2 * np.arcsin(np.sqrt(p[:big // 2 + 1])))
    use = max(float(np.abs(theta[psi <= u * m.z_in] - flip).max()) / tol_in, float(theta[psi >= u * m.z_out].max()) / max_out)
    b = factor(fit.q)
    return -1j * b * np.exp(-1j * np.angle(b.sum())), dict(use=use, level=fit.level, widened=widen)


def tightness_text(rows=None) -> str:
    """The sentence of docs/physics/design.md, section 4a, on how tight the bound is (from ``tightness``)."""
    rows = rows or tightness()
    ratios = np.array([r["ratio"] for r in rows])
    return (f"built / bound in samples: smallest {ratios.min():.3f}, median {np.median(ratios):.3f}, largest {ratios.max():.3f} "
            f"({int((ratios == 1).sum())} of {len(rows)} specifications reach the bound exactly; bounds of {min(r['samples'] for r in rows)} to "
            f"{max(r['samples'] for r in rows)} samples, each with {min(r['certificates'] for r in rows)} to {max(r['certificates'] for r in rows)} "
            f"certificates; refocused signal of the built pulses {min(r['refocused'] for r in rows):.3f} to {max(r['refocused'] for r in rows):.3f}).")


if __name__ == "__main__":                                       # python -m mrgym.design.bound
    print(tightness_text())
