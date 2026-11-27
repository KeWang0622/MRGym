"""Public physics of ``Scanner/ErnstFlip``: what the datasheet lets anyone compute about a 2D spoiled gradient echo.

Nothing here reads a world or the engine. The module holds the two formulas of datasheet sections 1 and 5, written
again on purpose (the scanner serves the EPG steady state integrated over the slice with its own code; sharing that
code would let errors cancel):

  ernst          S = sin(theta) (1 - E) / (1 - E cos(theta)),  E = exp(-TR / T1): the ideal-spoiling steady state of one
                 column of spins that all see the flip angle theta
  slice_profile  the selective pulse (a Hann-windowed sinc of 128 samples with time-bandwidth product 4, a rephasing
                 lobe of half the selection area) as a cascade of small rotations with precession in between: the flip
                 angle theta(z) and the transverse phase at every height z, for a flip angle alpha at the slice centre
  curve          the 2D signal against the flip angle at the slice centre: the sum of ``ernst`` over the profile
  optimum        the flip angle at the slice centre at which ``curve`` is largest

The 2D optimum is not the Ernst angle arccos(E). The wings of the profile see smaller angles than the slice centre and
are below their own optimum when the centre is at arccos(E), so the whole slice gains from a larger nominal angle: about
1.5 times the Ernst angle with this pulse (``tests/test_ernst_flip.py`` checks the numbers against the engine).

Used by the task for the table that level 1 prints (the stated method) and by the control policies.
"""
from __future__ import annotations

import functools
import math

import numpy as np

D2R = math.pi / 180.0
PULSE_SAMPLES, TBW = 128, 4.0                                   # datasheet section 1: the ``normal`` pulse
ALPHA_GRID = np.linspace(0.0, 150.0 * D2R, 301)                 # flip angle at the slice centre on which the profile is tabulated
FLIP_GRID_DEG = np.linspace(1.0, 90.0, 357)                     # nominal flip angles the console accepts, 0.25 degrees apart


def ernst(theta, tr_ms, t1_ms):
    """Ideal-spoiling steady state of a spoiled gradient echo per unit M0 (datasheet section 5)."""
    e1 = np.exp(-tr_ms / t1_ms)
    return np.sin(theta) * (1.0 - e1) / (1.0 - e1 * np.cos(theta))


def ernst_angle_deg(tr_ms, t1_ms) -> float:
    """The flip angle of the largest ``ernst`` signal: arccos(exp(-TR / T1)). One flip angle per voxel (a hard pulse)."""
    return float(math.degrees(math.acos(math.exp(-tr_ms / t1_ms))))


@functools.lru_cache(maxsize=2)
def slice_profile(tbw: float = TBW):
    """(theta [alpha, z], weight [alpha, z]) of the selective pulse with time-bandwidth product ``tbw``. ``theta`` is
    the flip angle at height z for a flip angle alpha (``ALPHA_GRID``) at the slice centre; ``weight`` is the
    quadrature weight of z times the cosine of the transverse phase after the rephasing lobe (the two halves of the
    slice added). z is counted in nominal bandwidths of the pulse across the slice gradient. Since engine 0.3.0 the
    console calibrates that gradient so that the profile of a small flip angle is one slice thickness wide at half
    its maximum, which for this pulse is 0.994 nominal bandwidths: the sum over z is the signal per 0.994 slice
    thicknesses. The factor is the same at every flip angle and T1, so it drops out of every ratio, fit and optimum
    this module is used for (a fit has a free amplitude).

    The pulse is ``PULSE_SAMPLES`` hard sub-pulses about x with the amplitudes of a Hann-windowed sinc; between them a
    spin at height z precesses about z in proportion to z. Each step is a 2 x 2 unitary matrix; the product is
    accumulated for every (alpha, z) at once."""
    t = (np.arange(PULSE_SAMPLES) + 0.5) / PULSE_SAMPLES - 0.5
    amp = np.sinc(tbw * t) * 0.5 * (1.0 + np.cos(2 * np.pi * t))
    amp /= amp.sum()
    reach = 0.5 + 12.0 / tbw                                     # nominal bandwidths: the wings matter (see the module docstring)
    z = np.linspace(0.0, reach, int(round(8 * tbw * reach)) + 1)
    wz = np.full(len(z), 2 * (z[1] - z[0]))
    wz[[0, -1]] /= 2
    alpha = ALPHA_GRID[:, None]
    step = 2 * np.pi * tbw * z[None, :] / PULSE_SAMPLES          # precession between two sub-pulses
    half = np.exp(-0.25j * step)
    a, b = np.ones((len(ALPHA_GRID), len(z)), complex), np.zeros((len(ALPHA_GRID), len(z)), complex)
    for w in amp:
        a, b = half * a, np.conj(half) * b
        c, s = np.cos(w * alpha / 2), -1j * np.sin(w * alpha / 2)
        a, b = c * a - np.conj(s) * b, s * a + c * b
        a, b = half * a, np.conj(half) * b
    back = np.exp(0.25j * step * PULSE_SAMPLES)                  # rephasing lobe: half the selection area
    a, b = back * a, np.conj(back) * b
    theta = 2 * np.arcsin(np.clip(np.abs(b), 0.0, 1.0))
    phase = 1j * np.exp(1j * np.angle(2 * np.conj(a) * b))       # +1 at the slice centre
    return theta, wz * phase.real


def curve(alpha_rad, tr_ms, t1_ms):
    """2D signal per unit M0 (and per 0.994 slice thicknesses: ``slice_profile``) for flip angles ``alpha_rad`` (1D) at
    the slice centre and T1 values ``t1_ms`` (a number or 1D): array [t1, alpha]."""
    theta, weight = slice_profile()
    x = np.clip(np.asarray(alpha_rad, float) / (ALPHA_GRID[1] - ALPHA_GRID[0]), 0.0, len(ALPHA_GRID) - 1.000001)
    i = x.astype(int)
    f = (x - i)[:, None]
    th, ww = theta[i] * (1 - f) + theta[i + 1] * f, weight[i] * (1 - f) + weight[i + 1] * f          # [alpha, z]
    t1 = np.atleast_1d(np.asarray(t1_ms, float))[:, None, None]
    return (ww[None] * ernst(th[None], tr_ms, t1)).sum(-1)


def _peak(flips_deg, values) -> float:
    """Position of the largest value on an equidistant grid, refined with the parabola through its neighbours."""
    j = int(np.argmax(values))
    if 0 < j < len(values) - 1:
        y0, y1, y2 = values[j - 1], values[j], values[j + 1]
        d = y0 - 2 * y1 + y2
        if d < 0:
            return float(flips_deg[j] + 0.5 * (y0 - y2) / d * (flips_deg[1] - flips_deg[0]))
    return float(flips_deg[j])


def optimum_deg(tr_ms, t1_ms, scale: float = 1.0) -> float:
    """Nominal flip angle (degrees, within the console's 1 to 90) of the largest 2D signal for one T1. ``scale``: the
    flip angle at the slice centre is ``scale`` times the nominal one."""
    return _peak(FLIP_GRID_DEG, curve(FLIP_GRID_DEG * D2R * scale, tr_ms, t1_ms)[0])


T1_FIT_GRID_MS = np.geomspace(20.0, 30000.0, 300)


def fit(flips_deg, signals, tr_ms, profile: bool = True):
    """Least squares of  M x signal(flip; T1)  to read-outs at nominal flip angles ``flips_deg`` (two or more): T1 on a
    zooming grid, M linear. ``profile`` False: the one-angle Ernst formula instead of the 2D curve.
    Returns (t1_ms, m). With an unknown transmit scale the fitted T1 is an effective one: the curve is fitted against
    the NOMINAL angle, and that is the axis the answer is delivered on."""
    a, s = np.asarray(flips_deg, float) * D2R, np.asarray(signals, float)
    model = (lambda t1: curve(a, tr_ms, t1)) if profile else (lambda t1: ernst(a[None, :], tr_ms, np.atleast_1d(t1)[:, None]))
    grid = T1_FIT_GRID_MS
    for _ in range(4):
        m = model(grid)                                           # [t1, n]
        num, den = m @ s, (m * m).sum(1)
        sse = (s * s).sum() - num * num / np.maximum(den, 1e-300)
        k = int(np.argmin(sse))
        best = (float(grid[k]), float(num[k] / max(den[k], 1e-300)))
        grid = np.geomspace(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)], 25)
    return best


def table(tr_ms, flip_lo_deg, flip_hi_deg, t1_ms):
    """Rows (signal ratio S(flip_hi) / S(flip_lo), flip angle of the largest signal) of the 2D curve for the T1 values
    ``t1_ms``: what level 1 prints. The ratio falls as T1 grows; the rows are returned by rising ratio."""
    s = curve(np.array([flip_lo_deg, flip_hi_deg]) * D2R, tr_ms, t1_ms)
    rows = [(float(s[k, 1] / s[k, 0]), optimum_deg(tr_ms, t)) for k, t in enumerate(t1_ms)]
    return sorted(rows)
