"""Electrical properties from the transmit field: the inverse of the field model, in homogeneous regions.

Where permittivity and conductivity do not change with position every Cartesian component of B obeys the Helmholtz
equation, and so does B1+ = (Bx + i By) / 2:

    laplace(B1+) + k^2 B1+ = 0,      k^2 = w^2 mu0 eps0 eps_r - i w mu0 sigma      (time dependence exp(+i w t)),

so (Katscher et al. 2009, IEEE Trans Med Imaging 28:1365; Voigt et al. 2011, Magn Reson Med 66:456)

    eps_r = -Re(laplace(B1+) / B1+) / (w^2 mu0 eps0),        sigma = Im(laplace(B1+) / B1+) / (w mu0).

With B1+ = |B1+| exp(i phi): Im(laplace(B1+) / B1+) = laplace(phi) + 2 grad(ln |B1+|) . grad(phi) and
Re(...) = laplace(|B1+|) / |B1+| - |grad(phi)|^2, which give the two common approximations: conductivity from the
phase alone, sigma ~ laplace(phi) / (w mu0), and permittivity from the magnitude alone,
eps_r ~ -laplace(|B1+|) / (|B1+| w^2 mu0 eps0).

Where it fails. (1) At a boundary between tissues the equation has further terms (the gradient of the properties
times the field): the estimate is wrong within the reach of the derivative kernel of every boundary. (2) A scanner
measures |B1+| and the transceive phase, the sum of the transmit phase and the receive phase; taking half of it as
the transmit phase is exact for a body with circular symmetry on the axis of an ideal quadrature current sheet that
transmits in one sense and receives in the other, close to exact for a birdcage of many rungs, and an approximation
otherwise: few rungs, a linear coil, a body without that symmetry (``transceive_phase``, ``from_transceive``; rows
E3 to E3c of ``validate``). (3) In three dimensions the Laplacian needs the
second derivative along z; in this two-dimensional model it is zero by construction, which a real slice is not.
(4) Second derivatives amplify noise: the error of the estimate grows as noise / (h^2 x window^3).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.ndimage

from .units import EPS0, MU0


def laplacian_kernel(half_width=2):
    """Kernel [2 m + 1, 2 m + 1] that returns h^2 x the Laplacian of the least-squares quadratic surface through the
    window (a two-dimensional Savitzky-Golay filter). Exact for polynomials up to the third degree; m = 1 uses a
    3 x 3 window. A larger window averages noise and widens the band lost at boundaries."""
    m = int(half_width)
    if m < 1:
        raise ValueError("the half-width must be at least 1")
    v, u = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    basis = np.stack([np.ones(v.size), v.ravel(), u.ravel(), v.ravel() ** 2, v.ravel() * u.ravel(), u.ravel() ** 2], axis=1)
    pinv = np.linalg.pinv(basis)
    return (2.0 * (pinv[3] + pinv[5])).reshape(v.shape)


def laplacian(field, h_m, half_width=2):
    """Laplacian of a real or complex map [y, x] on a grid of pitch ``h_m`` with ``laplacian_kernel``. The outer
    ``half_width`` pixels are not valid."""
    k = laplacian_kernel(half_width)
    f = np.asarray(field)
    if np.iscomplexobj(f):
        out = scipy.ndimage.correlate(f.real, k, mode="nearest") + 1j * scipy.ndimage.correlate(f.imag, k, mode="nearest")
    else:
        out = scipy.ndimage.correlate(f.astype(float), k, mode="nearest")
    return out / h_m ** 2


def valid_region(mask, half_width=2):
    """The pixels of a homogeneous region whose whole derivative window lies inside the region."""
    size = 2 * int(half_width) + 1
    return scipy.ndimage.binary_erosion(mask, structure=np.ones((size, size), bool), border_value=0)


def helmholtz(b1p, f_hz, h_m, half_width=2):
    """Permittivity and conductivity maps from the complex B1+ map: dict(eps_r, sigma [S/m], k2 [1/m^2]).
    Meaningful inside homogeneous regions, ``half_width`` pixels away from every boundary (``valid_region``)."""
    w = 2 * math.pi * f_hz
    b = np.asarray(b1p, complex)
    ratio = laplacian(b, h_m, half_width) / np.where(b == 0, np.nan, b)
    return dict(eps_r=-np.real(ratio) / (w * w * MU0 * EPS0), sigma=np.imag(ratio) / (w * MU0), k2=-ratio)


def unwrap(phase):
    """A phase map without 2 pi jumps along either axis (one-dimensional unwrapping along x in every row, then the
    rows are aligned along y). Enough for the smooth phases of this model; not a general two-dimensional unwrapper."""
    p = np.unwrap(np.asarray(phase, float), axis=1)
    mid = p.shape[1] // 2
    shift = np.unwrap(p[:, mid]) - p[:, mid]
    return p + shift[:, None]


def sigma_from_phase(phase, f_hz, h_m, half_width=2):
    """Phase-only conductivity: laplace(phi) / (w mu0), with phi the (unwrapped) phase of B1+ in rad."""
    return laplacian(unwrap(phase), h_m, half_width) / (2 * math.pi * f_hz * MU0)


def eps_from_magnitude(magnitude, f_hz, h_m, half_width=2):
    """Magnitude-only permittivity: -laplace(|B1+|) / (|B1+| w^2 mu0 eps0)."""
    w = 2 * math.pi * f_hz
    m = np.asarray(magnitude, float)
    return -laplacian(m, h_m, half_width) / np.where(m == 0, np.nan, m) / (w * w * MU0 * EPS0)


def transceive_phase(b1p_transmit, rx_receive):
    """What a scanner can measure: the phase of the image, arg(B1+) of the transmitting coil plus the phase of the
    receive sensitivity conj(B1-) of the receiving coil (``coils.receive_maps``), in rad."""
    return np.angle(np.asarray(b1p_transmit) * np.asarray(rx_receive))


def from_transceive(magnitude, phase_transceive, f_hz, h_m, half_width=2):
    """The Helmholtz estimate under the transceive-phase assumption: B1+ = |B1+| exp(i phase_transceive / 2)."""
    return helmholtz(np.asarray(magnitude) * np.exp(0.5j * unwrap(phase_transceive)), f_hz, h_m, half_width)
