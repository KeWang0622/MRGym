"""Constants and conventions of the radio-frequency field model. SI units throughout (m, s, A, V, T, S/m, kg).

Phasors: a field is ``Re{F exp(+i w t)}`` with F the complex amplitude (a peak value, not rms), so a time average of
a product is ``Re{F G*} / 2``. With this sign an outgoing wave is ``exp(-i k r)`` and a lossy medium has the complex
relative permittivity ``eps_r - i sigma / (w eps0)``.

Rotating components (Hoult 2000, Concepts Magn Reson 12:173): ``B1+ = (Bx + i By) / 2`` is the amplitude of the part
of the transverse field that rotates from +x towards +y (positive about +z), ``B1- = (Bx - i By)* / 2`` that of the
part rotating the other way. The engine's magnetisation precesses from +x towards +y (a positive frequency gains
positive phase: docs/physics/conventions.md, section 5), so B1+ is the field that excites, and its phase p is the
axis (cos p, sin p, 0) of docs/physics/design.md. The signal received by a coil is proportional to
``conj(B1-) = (Bx - i By) / 2`` of the field that coil makes with unit current.

What the engine's convention means in the laboratory. The engine turns magnetisation right-handed about the field
(dM/dt = gamma B x M: docs/physics/design.md, section 2); row A8b of ``validate`` integrates that equation in the
laboratory frame and finds the flip angle and phase that B1+ predicts. A proton turns left-handed about B0
(dM/dt = gamma M x B). The two pictures are mirror images: the formulas above describe a proton with B0 along -z, and
with B0 along +z the patterns of B1+ and B1- change places. Nothing the engine reports depends on the choice.
"""
from __future__ import annotations

import math

MU0 = 1.25663706212e-6                          # vacuum permeability, H/m (CODATA 2018)
C0 = 299_792_458.0                              # speed of light, m/s (exact)
EPS0 = 1.0 / (MU0 * C0 * C0)                    # vacuum permittivity, F/m
ETA0 = MU0 * C0                                 # impedance of free space, ohm
GAMMA_HZ_PER_T = 42.577478518e6                 # proton gyromagnetic ratio / 2 pi (the engine's value)
K_BOLTZMANN = 1.380649e-23                      # J/K (exact)

# Nominal proton frequencies used in the tables of this package (1.5 T, 3 T, 7 T)
F_NOMINAL_HZ = {1.5: 64.0e6, 3.0: 128.0e6, 7.0: 298.0e6}


def larmor_hz(b0_t: float) -> float:
    """Proton frequency of a field strength: gamma B0 (63.87, 127.73 and 298.04 MHz at 1.5, 3 and 7 T)."""
    return GAMMA_HZ_PER_T * float(b0_t)


def complex_permittivity(f_hz, eps_r, sigma):
    """Relative complex permittivity ``eps_r - i sigma / (w eps0)`` (time dependence exp(+i w t))."""
    return eps_r - 1j * sigma / (2.0 * math.pi * f_hz * EPS0)


def wavenumber(f_hz, eps_r=1.0, sigma=0.0):
    """Complex wavenumber k = (w / c) sqrt(eps_r - i sigma / (w eps0)), with Re k > 0 and Im k <= 0, in rad/m."""
    import numpy as np
    return 2.0 * math.pi * f_hz / C0 * np.sqrt(np.asarray(complex_permittivity(f_hz, eps_r, sigma), complex))


def wavelength_m(f_hz, eps_r=1.0, sigma=0.0):
    """Wavelength in a medium: 2 pi / Re k = c / (f Re sqrt(eps_complex))."""
    import numpy as np
    return 2.0 * math.pi / np.real(wavenumber(f_hz, eps_r, sigma))


def skin_depth_m(f_hz, eps_r=1.0, sigma=0.0):
    """Distance over which a plane wave falls to 1/e in amplitude: -1 / Im k (infinite without loss)."""
    import numpy as np
    im = -np.imag(wavenumber(f_hz, eps_r, sigma))
    return np.where(im > 0, 1.0 / np.maximum(im, 1e-300), np.inf)
