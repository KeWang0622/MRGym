"""Slice-selective excitation: the windowed-sinc pulse of the console and the rotation it makes at every position
through the slice.

A slice-selective pulse is a shaped B1 waveform played under a constant slice gradient. A spin at height z precesses
about z at a rate proportional to z while the pulse turns it about x, so the flip angle depends on z: the slice
profile. The profile is not a rectangle and it changes shape with the flip angle (the Bloch equations are not linear
in B1). A 2D scan measures the integral of the signal over that profile, which is not the signal at the nominal
angle; methods that invert a signal equation for the flip angle (variable-flip-angle T1, actual-flip-angle imaging)
are biased by it (Malik 2011; Dieringer 2014; Svedin & Parker 2017).

Model
  pulse      ``n`` hard sub-pulses (128) with the amplitudes of a sinc of ``tbw`` zero crossings (time-bandwidth
             product) under a Hann window; the samples sum to 1, so the flip angle at the slice centre is the nominal
             angle. Nominal bandwidth = tbw / duration.
  thickness  the console calibrates the slice gradient per pulse: one "slice thickness" is the full width at half
             maximum of the small-flip-angle profile (the Fourier transform of the pulse). The gradient spans
             ``fwhm_factor(tbw)`` nominal bandwidths across it: 1.197 for time-bandwidth 2, 0.994 for 4, 1.000 for 8.
             A pulse of low time-bandwidth product therefore has the prescribed width and a wide transition band, not
             a thicker slice. At large flip angles the profile broadens (the Bloch equations are not linear in B1);
             that is in the rotation below and is not calibrated away.
  rotation   forward Shinnar-Le Roux transform (Pauly 1991): the product of the sub-pulse rotations and the
             precession between them, as Cayley-Klein parameters (a, b) of an SU(2) rotation, exact for a
             piecewise-constant B1. No relaxation during the pulse. A rephasing lobe of half the selection area
             follows an excitation pulse.
  use        from equilibrium  Mxy = 2 conj(a) b,  Mz = |a|^2 - |b|^2,  flip angle theta(z) = 2 asin|b|;
             a refocusing pulse between crushers  Mxy+ = -b^2 conj(Mxy-),  Mz+ = (1 - 2 |b|^2) Mz-.
Convention: a rotation by phi about the unit axis n has  a = cos(phi / 2) - i n_z sin(phi / 2),
b = -i (n_x + i n_y) sin(phi / 2). A pulse about x at the slice centre gives Mxy = -i sin(alpha).

Positions are in units of the slice thickness (``zeta`` = z / thickness). The engine integrates the signal through
the slice with ``quadrature(tbw)``: the trapezoid rule with 6 x tbw points per nominal bandwidth out to 0.5 + 12 / tbw
nominal bandwidths (divided by ``fwhm_factor`` in thicknesses). The pulse is symmetric in time, so the flip angle is
even in z and the transverse phase factor at -z is the complex conjugate of the one at +z: only z >= 0 is tabulated
and the two sides are added, which leaves the real part of the phase factor. The range and the density are set by the wings of the profile, not by its core: where the
flip angle in a wing passes the Ernst angle of a long-T1 material at short TR, the wing gives more signal per
millimetre than the slice centre, and lobes of opposite sign nearly cancel (``mrgym validate`` rows R3, R4).

Nothing here is specific to a sequence. ``mrgym validate --group rf`` checks this file against a separately written
3 x 3 rotation cascade (``bloch.slice_profile``) and against the small-tip Fourier relation.
"""
from __future__ import annotations

import functools

import numpy as np
import scipy.optimize

N_SAMPLES = 128
POINTS_PER_THICKNESS, WING = 6, 12.0      # quadrature, in nominal bandwidths: 6 tbw points per unit; range = 0.5 + WING / tbw
REFOCUS_WIDTH = 1.5                 # thickness of the refocusing slice of ``se`` / ``ir`` relative to the excitation slice
PULSES = dict(fast=(1.0, 2.0), normal=(2.0, 4.0), sharp=(4.0, 8.0))      # name: (duration of an excitation in ms, tbw)
WINDOWS = ("hann", "hamming")


def sinc_pulse(tbw, n=N_SAMPLES, window="hann"):
    """Amplitudes of the ``n`` sub-pulses of a windowed sinc with time-bandwidth product ``tbw``; they sum to 1."""
    t = (np.arange(n) + 0.5) / n - 0.5
    if window == "hann":
        w = 0.5 * (1 + np.cos(2 * np.pi * t))
    elif window == "hamming":
        w = 0.54 + 0.46 * np.cos(2 * np.pi * t)
    else:
        raise ValueError(f"window must be one of {WINDOWS}")
    b = np.sinc(tbw * t) * w
    return b / b.sum()


@functools.lru_cache(maxsize=64)
def fwhm_factor(tbw, window="hann", n=N_SAMPLES):
    """Full width at half maximum of the small-flip-angle slice profile, in nominal bandwidths (tbw / duration) of the
    slice gradient. The profile of a small flip angle is the Fourier transform of the pulse; the half-maximum point is
    found on it. The console sets the slice gradient to ``fwhm_factor x tbw / (duration x gamma x slice thickness)``,
    so that this width is the prescribed thickness for every pulse."""
    b = sinc_pulse(tbw, n, window)
    t = (np.arange(n) + 0.5) / n - 0.5
    over_half = lambda u: abs(np.sum(b * np.exp(-2j * np.pi * tbw * u * t))) - 0.5        # noqa: E731  the samples sum to 1
    hi = 0.05
    while over_half(hi) > 0:                               # the main lobe falls monotonically to its first zero
        hi += 0.05
    return float(2 * scipy.optimize.brentq(over_half, hi - 0.05, hi, xtol=1e-13))


def cayley_klein(alpha, tbw, zeta, n=N_SAMPLES, rephase=0.5, window="hann"):
    """Rotation made by the pulse at positions ``zeta`` (z / slice thickness) for a flip angle ``alpha`` (rad) at the
    slice centre. ``alpha`` and ``zeta`` broadcast. Returns (a, b) after the pulse and a rephasing lobe of ``rephase``
    times the selection area (0.5 for an excitation, 0 for a refocusing pulse between crushers)."""
    alpha, zeta = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(zeta, float))
    bj = sinc_pulse(tbw, n, window)
    dphi = 2 * np.pi * tbw * fwhm_factor(tbw, window, n) * zeta / n      # precession per sample under the calibrated gradient
    a, b = np.ones(dphi.shape, complex), np.zeros(dphi.shape, complex)
    ez = np.exp(-0.25j * dphi)                             # half a sample of precession
    for w in bj:
        a, b = ez * a, np.conj(ez) * b
        c, s = np.cos(w * alpha / 2), -1j * np.sin(w * alpha / 2)
        a, b = c * a - np.conj(s) * b, s * a + c * b
        a, b = ez * a, np.conj(ez) * b
    e = np.exp(0.5j * rephase * dphi * n)
    return e * a, np.conj(e) * b


@functools.lru_cache(maxsize=16)
def quadrature(tbw, window="hann"):
    """(zeta, weights) of the integral through the slice, in slice thicknesses: positions z >= 0 only, each weight
    counting the position and its mirror image. Sum of the weights = 2 x the range. The grid is laid out in nominal
    bandwidths of the slice gradient (where the lobes of the profile sit, whatever the calibration) and divided by
    ``fwhm_factor``."""
    h = 1.0 / (POINTS_PER_THICKNESS * tbw)
    k = int(round((0.5 + WING / tbw) / h))
    w = np.full(k + 1, 2 * h)
    w[[0, -1]] = h
    cal = fwhm_factor(tbw, window)
    return np.arange(k + 1) * h / cal, w / cal


def excitation(alpha, tbw, zeta, window="hann"):
    """What an excitation of centre flip angle ``alpha`` leaves at ``zeta``: (flip angle theta in rad, phase factor of
    the transverse magnetisation, +1 at the slice centre)."""
    a, b = cayley_klein(alpha, tbw, zeta, window=window)
    theta = 2 * np.arcsin(np.clip(np.abs(b), 0.0, 1.0))
    return theta, 1j * np.exp(1j * np.angle(2 * np.conj(a) * b))


def refocusing(alpha, tbw, zeta, window="hann", width=REFOCUS_WIDTH):
    """Cayley-Klein b of a refocusing pulse of centre flip angle ``alpha`` (rad) whose slice is ``width`` times the
    excitation slice; ``zeta`` is in units of the excitation slice. The echo is -b^2 conj(Mxy), Mz -> (1 - 2|b|^2) Mz."""
    return cayley_klein(alpha, tbw, np.asarray(zeta, float) / width, rephase=0.0, window=window)[1]


@functools.lru_cache(maxsize=64)
def energy_factor(tbw, window="hann", n=N_SAMPLES):
    """Integral of B1^2 over the shaped pulse relative to a hard pulse of the same duration and flip angle."""
    b = sinc_pulse(tbw, n, window)
    return float(n * np.sum(b * b))


@functools.lru_cache(maxsize=64)
def peak_factor(tbw, window="hann", n=N_SAMPLES):
    """Peak B1 of the shaped pulse relative to a hard pulse of the same duration and flip angle."""
    return float(n * sinc_pulse(tbw, n, window).max())


def dephasing_factor(cycles, tbw, window="hann"):
    """Signal left when the phase runs through ``cycles`` turns across one slice thickness (through-slice field
    gradient x thickness x echo time). For a small flip angle the profile is the Fourier transform of the pulse, so
    the integral of profile x phase ramp is the pulse shape itself at t = c / tbw of its duration, with c the turns
    across one nominal bandwidth of the slice gradient (``cycles / fwhm_factor``): sinc(c) x window(c / tbw), and
    nothing beyond the end of the pulse (|c| >= tbw / 2)."""
    c = np.asarray(cycles, float) / fwhm_factor(tbw, window)
    t = c / tbw
    w = 0.5 * (1 + np.cos(2 * np.pi * t)) if window == "hann" else 0.54 + 0.46 * np.cos(2 * np.pi * t)
    return np.where(np.abs(t) < 0.5, np.sinc(c) * w, 0.0)
