"""What a scanner does to a designed pulse that the ideal simulator leaves out.

A design is a nominal waveform: complex B1 samples and the slice-select gradient during each of them. ``Chain`` holds
the four declared imperfections of the transmit chain, and ``played`` returns the waveform a spin sees:

  transmit scale   the B1 a spin receives is ``tx_scale`` times what the amplifier delivers (the error of the transmit
                   adjustment times the transmit pattern at that place).
  amplifier        the RF amplifier compresses large amplitudes. Declared law (Rapp 1991, the usual model of a
                   solid-state amplifier): an amplitude a comes out as

                       f(a) = a / (1 + (a / A)^(2 p))^(1 / (2 p)),        p = 3 (``RAPP_P``),

                   A = ``amp_sat_ut``, the level the output can never pass. The phase is unchanged. f(a) <= a, and
                   f(a) / a is 0.9974 at a = A / 2, 0.962 at 0.8 A and 0.891 at a = A. ``None`` = a linear amplifier.
                   The scale acts after the amplifier: delivered = tx_scale x f(a).
  gradient delay   the slice-select gradient arrives ``grad_delay_us`` later than the RF: a spin sees G(t - delay)
                   while it sees B1(t). Before the first and after the last sample of the pulse the gradient holds its
                   end value (the ramps are outside the pulse). A pulse under a constant gradient is not changed by
                   any delay; a variable-rate pulse is.
  off-resonance    ``df_hz``: the spin precesses that much faster than the transmitter (the error of the frequency
                   adjustment plus the local field).

The delayed gradient is given per RF sample as its mean over that sample, which is its area: with the hard-pulse
picture (precession, sub-pulse, precession) the area is all a sample's gradient does. With B1 held constant during
a sample (``model="piecewise"``, what a digital-to-analogue converter with a hold plays) the gradient inside a sample
is taken as that mean; a delay that is not a whole number of samples is then exact in area and first order within the
sample.

Two further switches of a chain, both off by default (``Scanner/RFPulse-v1`` turns both on):
  ``model``        how one raster sample is played: ``piecewise`` (B1 held during the sample, the default) or ``split``
                   (a hard sub-pulse in the middle of the sample: the model of the Shinnar-Le Roux transform, in which
                   the duration bound of ``mrgym.design.bound`` is a theorem).
  ``guard``        the plateau of the slice gradient begins before the pulse and ends after it, by more than any delay.
                   The gradient is then constant during the pulse whatever the delay, and a gradient that is late by
                   tau leaves the area G tau behind the pulse that the rephasing lobe of the design does not take
                   back: a rotation about z by 2 pi gamma G tau z after the pulse, a tilt of the phase across the slice.

Not here, and so in neither the ideal simulator nor the scanner: relaxation during the pulse, the finite bandwidth of
the gradient chain (eddy currents, a slew-dependent lag), a phase distortion of the amplifier, more than one transmit
channel.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import rotation
from .units import RAD_PER_MT_M_MM_MS

RAPP_P = 3.0                     # smoothness of the amplifier's knee (declared, the same on every scanner)
MODEL = "piecewise"              # B1 constant during a raster sample


def amplifier(amp_ut, sat_ut, p: float = RAPP_P):
    """Output amplitude of the RF amplifier for the input amplitude ``amp_ut`` (uT, >= 0); ``sat_ut`` None: linear."""
    a = np.asarray(amp_ut, float)
    if sat_ut is None or not np.isfinite(sat_ut):
        return a
    if not sat_ut > 0:
        raise ValueError("the saturation level of the amplifier must be positive")
    return a / (1.0 + (a / sat_ut) ** (2 * p)) ** (1.0 / (2 * p))


def delayed_gradient(g_mt_m, dt_ms, delay_us: float):
    """Mean of the gradient G(t - delay) over every sample of the pulse. ``g_mt_m`` [n] is constant during its own
    sample; outside the pulse the gradient holds its first / last value."""
    g, dt = np.asarray(g_mt_m, float), np.asarray(dt_ms, float)
    if not delay_us or np.all(g == g[0]):
        return g
    t = np.concatenate([[0.0], np.cumsum(dt)])
    area = np.concatenate([[0.0], np.cumsum(g * dt)])

    def cumulative(x):
        out = np.interp(x, t, area)
        out = np.where(x < 0, g[0] * x, out)
        return np.where(x > t[-1], area[-1] + g[-1] * (x - t[-1]), out)
    tau = delay_us * 1e-3
    return (cumulative(t[1:] - tau) - cumulative(t[:-1] - tau)) / dt


@dataclasses.dataclass(frozen=True)
class Chain:
    """The transmit chain between a design and a spin. The default is the ideal one."""
    tx_scale: float = 1.0
    amp_sat_ut: float | None = None
    grad_delay_us: float = 0.0
    df_hz: float = 0.0
    model: str = MODEL
    guard: bool = False

    def __post_init__(self):
        if self.model not in rotation.MODELS:
            raise ValueError(f"model must be one of {rotation.MODELS}")
        if not (self.tx_scale > 0 and np.isfinite(self.tx_scale)):
            raise ValueError("tx_scale must be positive")
        if not np.isfinite(self.grad_delay_us) or not np.isfinite(self.df_hz):
            raise ValueError("grad_delay_us and df_hz must be finite")


IDEAL = Chain()


def played(pulse, chain: Chain = IDEAL):
    """(B1 [n] complex uT, sample durations [n] ms, gradient [n] mT/m) as a spin with ``chain`` receives the pulse."""
    amp = np.abs(pulse.b1_ut)
    out = amplifier(amp, chain.amp_sat_ut)
    b1 = chain.tx_scale * np.where(amp > 0, out / np.where(amp > 0, amp, 1.0), 1.0) * pulse.b1_ut
    return b1, pulse.dt_ms, delayed_gradient(pulse.g_mt_m, pulse.dt_ms, chain.grad_delay_us)


def rotate(pulse, chain: Chain = IDEAL, pos_mm=None, rephase: bool = True, model: str | None = None):
    """Cayley-Klein parameters (a, b) of the pulse as played through ``chain`` at positions ``pos_mm`` along the slice
    axis, after the rephasing lobe when ``rephase`` (a rotation about z by the lobe's area: no time passes in it).
    ``model``: the chain's own unless given."""
    b1, dt, g = played(pulse, chain)
    a, b = rotation.rotate(b1, dt, g, pos_mm, chain.df_hz, 1.0, model or chain.model)
    area = pulse.rephase_mt_ms_m if rephase else 0.0
    if rephase and chain.guard and chain.grad_delay_us:                 # what a late plateau leaves behind the pulse
        area += float(pulse.g_mt_m[-1]) * chain.grad_delay_us * 1e-3
    if area and pos_mm is not None:
        a, b = rotation.z_rotation(a, b, RAD_PER_MT_M_MM_MS * area * np.asarray(pos_mm, float))
    return a, b


def flip_deg(pulse, chain: Chain = IDEAL, pos_mm=0.0, model: str | None = None):
    """Flip angle (degrees) at positions ``pos_mm``."""
    return np.degrees(rotation.flip_angle(rotate(pulse, chain, pos_mm, rephase=False, model=model)[1]))
