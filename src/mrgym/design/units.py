"""Constants and unit factors of the design layer.

Units at the interface of every function in ``mrgym.design``:

    time            ms              raster 0.01 ms unless stated
    B1              microtesla      complex: the angle of the number is the phase of the pulse
    gradient        mT/m            slew rate T/m/s (numerically equal to mT/m/ms)
    position        mm
    frequency       Hz
    angle           rad inside, degrees where a name ends in ``_deg``
    moment m_n      mT/m x ms^(n+1)
    k               cycles/mm
    b               s/mm^2;  diffusion coefficient in um^2/ms (= 1e-3 mm^2/s)
    RF energy       uT^2 x ms       (the integral of |B1|^2 dt)

The gyromagnetic ratio is the value of ``mrgym.scanner.console.GAMMA_HZ_PER_T``; it is repeated here so that importing
the design layer does not import the engine, and a test asserts that the two are equal.
"""
from __future__ import annotations

import math

GAMMA_HZ_PER_T = 42.577478518e6
TWO_PI = 2.0 * math.pi

# nutation: rad per (uT x ms);  precession: rad per (mT/m x mm x ms);  off-resonance: rad per (Hz x ms)
RAD_PER_UT_MS = TWO_PI * GAMMA_HZ_PER_T * 1e-6 * 1e-3
RAD_PER_MT_M_MM_MS = TWO_PI * GAMMA_HZ_PER_T * 1e-3 * 1e-3 * 1e-3
RAD_PER_HZ_MS = TWO_PI * 1e-3
# frequency of a spin: Hz per (mT/m x mm);  k-space: cycles/mm per (mT/m x ms)
HZ_PER_MT_M_MM = GAMMA_HZ_PER_T * 1e-3 * 1e-3
CYCLES_PER_MM_PER_MT_M_MS = GAMMA_HZ_PER_T * 1e-3 * 1e-3 * 1e-3
# b-value: s/mm^2 per ((mT/m x ms)^2 x ms) = gamma^2 [rad/s/T]^2 x (1e-6 T s/m)^2 x 1e-3 s x 1e-6 m^2/mm^2
B_PER_MT_M_MS_SQUARED_MS = (TWO_PI * GAMMA_HZ_PER_T) ** 2 * 1e-12 * 1e-3 * 1e-6

RASTER_MS = 0.01
