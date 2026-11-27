"""Coils made of line currents along z, for transmission and, by reciprocity, for reception.

A channel is a list of line currents ``(y_m, x_m, current_a)``; ``solver.Solver.fields`` takes a list of channels and
returns the fields each makes for unit drive. Angles are measured from +x towards +y.

Transmit. A birdcage in its mode 1 is N rungs at phi_k = phi0 + 2 pi k / N with the currents
I exp(-i (phi_k - phi0)) (quadrature: the current pattern turns from +x towards +y and makes B1+) or
I cos(phi_k - phi0) (linear). The end rings that close the currents are not in a two-dimensional model. A transmit
array is N separate elements on a ring: single rungs (``element="line"``; the return path is the shield or
infinity) or loops long in z, which show in the slice as two opposite currents (``element="loop"``).

Receive (Hoult 2000, Concepts Magn Reson 12:173). The voltage that a unit transverse magnetisation precessing from
+x towards +y induces in a coil is proportional to w conj(B1-), where B1- = (Bx - i By)* / 2 is taken from the field
the coil makes with unit current: ``receive_maps``. The thermal noise of the body couples the elements through the
loss integrals R_ij = integral of sigma E_i conj(E_j) (Roemer et al. 1990, Magn Reson Med 16:192):
``noise_resistance``. (The voltage a current density J induces in coil i is the integral of E_i . J, without a
conjugate; thermal currents are uncorrelated between places, which gives E_i conj(E_j). This is the transpose of the
matrix of absorbed power in transmission, ``solver.Solver.power_matrix``.) With the sensitivities s and R the signal-to-noise ratio of the best combination is
sqrt(s^H R^-1 s) and the noise amplification of an unfolding is the g-factor of Pruessmann et al. 1999 (Magn Reson
Med 42:952): ``snr_optimal``, ``g_factor``. Nothing here reconstructs an undersampled image.

Units. Per-channel maps are per ampere. In two dimensions R is per metre of coil length (ohm/m) and the induced
voltage is per metre as well, so the ratios of ``snr_optimal`` are relative: T per ampere per sqrt(ohm per metre).
"""
from __future__ import annotations

import math

import numpy as np

from . import analytic


def ring(radius_m, n, phi0=0.0):
    """Angles and (y, x) positions of n points on a circle about the isocentre: (phi [n], y [n], x [n])."""
    phi = phi0 + 2 * math.pi * np.arange(n) / n
    return phi, radius_m * np.sin(phi), radius_m * np.cos(phi)


def birdcage(radius_m=0.14, n_rungs=16, drive="quadrature", current_a=1.0, phi0=0.0):
    """A birdcage driven in one mode, as ONE channel: ``n_rungs`` line currents with the mode-1 currents of
    ``analytic.rung_currents`` (drive "quadrature", "reverse" or "linear")."""
    _, y, x = ring(radius_m, n_rungs, phi0)
    cur = analytic.rung_currents(n_rungs, drive, current_a, phi0)
    return [[(float(a), float(b), complex(c)) for a, b, c in zip(y, x, cur)]]


def birdcage_ports(radius_m=0.14, n_rungs=16, current_a=1.0, phi0=0.0):
    """The two linear modes of a birdcage as TWO channels: currents I cos(phi_k - phi0) and I sin(phi_k - phi0).
    Weights (1, -i) make the quadrature mode, (1, +i) the reverse mode, (1, 0) the linear one."""
    phi, y, x = ring(radius_m, n_rungs, phi0)
    return [[(float(a), float(b), complex(current_a * f(p - phi0))) for a, b, p in zip(y, x, phi)] for f in (np.cos, np.sin)]


def array(radius_m=0.14, n_channels=8, element="line", width_rad=None, current_a=1.0, phi0=0.0):
    """A ring of ``n_channels`` separate elements.

    line   one rung per channel at phi_c. A single rung carries a net current; its return path is the shield. Without
           a shield the common mode of the array (all rungs in phase) is a line current that radiates: use line
           elements inside a shield, loops in an open coil
    loop   two currents +I at phi_c - width / 2 and -I at phi_c + width / 2 (default width: 0.8 of the pitch): a loop
           long in z, whose field at the isocentre points along the radius through phi_c"""
    phi, y, x = ring(radius_m, n_channels, phi0)
    if element == "line":
        return [[(float(a), float(b), complex(current_a))] for a, b in zip(y, x)]
    if element == "loop":
        half = 0.5 * (0.8 * 2 * math.pi / n_channels if width_rad is None else width_rad)
        return [[(radius_m * math.sin(p - half), radius_m * math.cos(p - half), complex(current_a)),
                 (radius_m * math.sin(p + half), radius_m * math.cos(p + half), complex(-current_a))] for p in phi]
    raise ValueError("element must be 'line' or 'loop'")


def mode_weights(n_channels, mode=1, sense=1):
    """Channel weights exp(-i sense mode 2 pi c / N) of a ring array: mode 1 with sense +1 is the birdcage-like
    (circularly polarised) mode, which adds the B1+ of all elements in phase at the isocentre."""
    return np.exp(-1j * sense * mode * 2 * math.pi * np.arange(n_channels) / n_channels)


# ------------------------------------------------------------------------------------------------ receive
def receive_maps(fields):
    """Receive sensitivity of every channel, [channel, y, x], T per ampere: conj(B1-) = (Bx - i By) / 2 of the field
    the channel makes with unit current."""
    return np.conj(fields["b1m"])


def noise_resistance(solver, ez):
    """Noise resistance matrix of the channels, ohm per metre: R[i, j] = integral of sigma E_i conj(E_j) over the
    slice, for unit currents. Hermitian, positive semi-definite. The noise voltages of the channels have the
    covariance <v_i conj(v_j)> = 4 k T df R[i, j]. It is the transpose of ``solver.power_matrix``."""
    return solver.power_matrix(ez).T


def noise_covariance(resistance, temperature_k=310.0, bandwidth_hz=1.0):
    """Covariance of the noise voltages, V^2 per metre: 4 k T df R (body noise only: no coil, no preamplifier)."""
    from .units import K_BOLTZMANN
    return 4.0 * K_BOLTZMANN * temperature_k * bandwidth_hz * np.asarray(resistance)


def _solve_psd(psi, rhs):
    """psi^-1 rhs for a Hermitian positive-definite psi (Cholesky)."""
    c = np.linalg.cholesky(np.asarray(psi, complex))
    return np.linalg.solve(c.conj().T, np.linalg.solve(c, rhs))


def snr_optimal(rx, psi):
    """Signal-to-noise ratio of the best linear combination of the channels at every pixel (Roemer et al. 1990):
    sqrt(s^H psi^-1 s) with s the sensitivities ``rx`` [channel, ...] at the pixel and ``psi`` the noise matrix."""
    rx = np.asarray(rx, complex)
    s = rx.reshape(rx.shape[0], -1)
    return np.sqrt(np.real(np.sum(np.conj(s) * _solve_psd(psi, s), axis=0))).reshape(rx.shape[1:])


def snr_combined(rx, psi, weights):
    """Signal-to-noise ratio of the fixed combination sum_c conj(w_c) v_c of the channel voltages:
    |w^H s| / sqrt(w^H psi w), with psi[i, j] = <v_i conj(v_j)>. Never above ``snr_optimal``."""
    rx = np.asarray(rx, complex)
    w = np.asarray(weights, complex)
    noise = math.sqrt(float(np.real(np.conj(w) @ np.asarray(psi) @ w)))
    return np.abs(np.tensordot(np.conj(w), rx, 1)) / noise


def g_factor(s, psi):
    """Geometry factor of an unfolding (Pruessmann et al. 1999).

    ``s``: sensitivities [channel, n] of the n positions that fold onto one pixel; ``psi``: noise matrix. Returns
    g [n] = sqrt([(S^H psi^-1 S)^-1]_kk [S^H psi^-1 S]_kk), at least 1 up to rounding; infinite when the positions
    cannot be told apart."""
    s = np.asarray(s, complex)
    m = np.conj(s).T @ _solve_psd(psi, s)
    if np.linalg.cond(m) > 1e12:
        return np.full(s.shape[1], np.inf)
    return np.sqrt(np.real(np.diag(np.linalg.inv(m))) * np.real(np.diag(m)))


def g_map(rx, psi, accel, mask):
    """g-factor map of regular undersampling by the integer factor ``accel`` along y (axis 0) on the grid of the
    maps: positions n_y / accel apart fold together; only positions inside ``mask`` take part. Returns a map
    [y, x] (0 outside the mask)."""
    rx = np.asarray(rx, complex)
    n_y = rx.shape[1]
    if n_y % accel:
        raise ValueError("the number of rows must be a multiple of the acceleration")
    step = n_y // accel
    out = np.zeros(rx.shape[1:])
    for i in range(step):
        rows = i + step * np.arange(accel)
        for j in np.flatnonzero(mask[rows].any(axis=0)):
            sel = rows[mask[rows, j]]
            out[sel, j] = g_factor(rx[:, sel, j], psi)
    return out
