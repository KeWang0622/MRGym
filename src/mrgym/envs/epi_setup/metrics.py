"""What the verifier of ``Scanner/EPISetup`` measures on a delivered protocol: noise-free scans of the hidden world, and
the same scans with one mechanism of the engine switched off (their *twins*).

    displacement   how far the material of a marked region lies from where it lies without the off-resonance field: the
                   centre of mass, along the phase-encode axis, of the image of that material alone in the delivered scan
                   minus its centre of mass in the twin without ``b0shift``. Both images are formed by the engine in
                   k-space from the material's density alone (no decay, no echo-time phase, no gradient delay) on a phase
                   field of view long enough that nothing wraps; the first moment of such an image is the mean position
                   of the material. An estimate from the full images (a cross-correlation, a centroid of the magnitude)
                   differs from it by several percent wherever the field also stretches the vial, and differs between
                   estimators; this one is the mean displacement itself.
    ghost          the Nyquist ghost of a marked region: |delivered - twin without the gradient delay| half a phase
                   field of view away from the region, summed over the region's pixels, relative to the region's own
                   signal in the twin. Complex images: what the delay adds there, whatever else lies there.
    wrap           energy of (delivered - twin with a phase field of view that holds the object) over the energy of that
                   twin, both without the gradient delay and without the chemical shift (where the displaced image of
                   fat lands is the business of ``fat``). The twin has more phase-encode lines at a proportionally
                   shorter echo spacing: the same pixel grid, the same displacement and the same decay along the train.
    fat            what displaced fat adds to a marked region: sum over its pixels of | |delivered| - |twin without
                   ``chemshift``| | relative to the twin (the twin reads fat out at the water frequency).
    snr            mean magnitude of a marked region in the delivered scan over the analytic noise of one pixel.

A region's pixels are those whose centres lie inside its circle, as in the console's own read-out. Nothing here is an
image-domain overlay: every image is formed by the engine in k-space.
"""
from __future__ import annotations

import math

import numpy as np

from ...scanner import kspace
from ...scanner import session as core

TWIN_FOV_MM = 320.0                 # the phase field of view of the no-wrap twin is at least this (the master grid spans 256 mm)
WRAP_OFF = ("ghost", "chemshift")                                       # wrap is judged without the ghost and with fat read out in place
DENSITY_OFF = ("ghost", "t2star", "chemshift", "dropout", "motion")       # what a density image leaves out


class Scans:
    """Noise-free complex images of one hidden world, with any set of engine effects switched off; engines are kept."""

    def __init__(self, world: dict, engine=None):
        self.world = world
        self._engines = {frozenset(): engine or kspace.Engine(world)}

    def engine(self, *off):
        key = frozenset(off)
        if key not in self._engines:
            self._engines[key] = kspace.Engine(self.world, effects=kspace.ALL - key, cache=8)
        return self._engines[key]

    def image(self, rp: dict, *off) -> np.ndarray:
        return self.engine(*off).acquire(rp, None)[0]

    def density_image(self, rp: dict, cls: str, *off) -> np.ndarray:
        """Complex image of the density of one tissue class alone, at echo time zero, without decay, chemical shift,
        through-slice dephasing, motion and gradient delay, and without the mechanisms in ``off``."""
        eng = self.engine(*DENSITY_OFF, *off)
        c = self.world["names"].index(cls)
        amps = [np.asarray(self.world["weights"][k][eng.sup[k]], float) if k == c else np.zeros(int(eng.sup[k].sum()))
                for k in range(len(self.world["names"]))]
        return eng.recon(eng.kspace(rp, 0.0, amps), rp)


def mask(rp: dict, region: dict) -> np.ndarray:
    """Pixels of the image of ``rp`` whose centres lie inside the circle ``region`` ([y, x], boolean)."""
    shape = (rp["matrix_pe"], rp["matrix"]) if rp["pe_dir"] == "y" else (rp["matrix"], rp["matrix_pe"])
    y, x = core.pixel_positions(shape, core.fov_yx(rp), core.centre_yx(rp))
    return (y[:, None] - region["y_mm"]) ** 2 + (x[None, :] - region["x_mm"]) ** 2 <= region["radius_mm"] ** 2


def centre_of_mass_mm(img: np.ndarray, rp: dict) -> float:
    """First moment of a complex image along the phase-encode axis, mm from the isocentre (real part: the image of a
    density is real up to the smooth phase of the receive profile)."""
    y, x = core.pixel_positions(img.shape, core.fov_yx(rp), core.centre_yx(rp))
    pos = y[:, None] if rp["pe_dir"] == "y" else x[None, :]
    return float(((pos * img).sum() / img.sum()).real)


def displacement_mm(scans: Scans, rp: dict, cls: str) -> float:
    """Signed shift (mm, along the phase-encode axis) of the centre of mass of the material ``cls`` against the twin
    without ``b0shift`` (see the module docstring)."""
    wide = wrap_twin(rp) or rp
    wide = dict(wide, matrix_pe=wide["lines_full"], fov_pe_mm=rp["fov_pe_mm"] * wide["lines_full"] / rp["matrix_pe"])   # keep every line of the long field of view
    return centre_of_mass_mm(scans.density_image(wide, cls), wide) - centre_of_mass_mm(scans.density_image(wide, cls, "b0shift"), wide)


def ghost(scans: Scans, rp: dict, region: dict) -> float:
    """The ghost of ``region`` half a phase field of view away, relative to the region's signal without the delay."""
    a, b = scans.image(rp), scans.image(rp, "ghost")
    m = mask(rp, region)
    axis = 0 if rp["pe_dir"] == "y" else 1
    there = np.roll(a - b, -(rp["matrix_pe"] // 2), axis=axis)           # the value half a field of view away, brought to the region
    return float(np.abs(there[m]).sum() / max(np.abs(b[m]).sum(), 1e-30))


def wrap_twin(rp: dict) -> dict | None:
    """The realised protocol with phase-encode lines added until the phase field of view is ``TWIN_FOV_MM`` or more, at
    an echo spacing shortened by the same factor; None when the delivered field of view is already that large."""
    px = rp["fov_pe_mm"] / rp["matrix_pe"]
    full = int(math.ceil(TWIN_FOV_MM / px / 2.0) * 2)
    if full <= rp["lines_full"]:
        return None
    return dict(rp, lines_full=full, lines=full, echo_spacing_ms=rp["echo_spacing_ms"] * rp["lines_full"] / full)


def wrap(scans: Scans, rp: dict) -> float:
    """Energy of what wrap-around changes in the delivered image, relative to the image without it."""
    twin = wrap_twin(rp)
    if twin is None:
        return 0.0
    a, b = scans.image(rp, *WRAP_OFF), scans.image(twin, *WRAP_OFF)
    return float(np.sum(np.abs(a - b) ** 2) / max(np.sum(np.abs(b) ** 2), 1e-30))


def fat(scans: Scans, rp: dict, region: dict) -> float:
    """What the chemical shift changes in the magnitude of ``region``, relative to the twin without it."""
    a, b = np.abs(scans.image(rp)), np.abs(scans.image(rp, "chemshift"))
    m = mask(rp, region)
    return float(np.abs(a[m] - b[m]).sum() / max(b[m].sum(), 1e-30))


def signal(scans: Scans, rp: dict, region: dict) -> float:
    """Mean magnitude of ``region`` in the delivered scan."""
    return float(np.abs(scans.image(rp)[mask(rp, region)]).mean())


def snr(scans: Scans, rp: dict, region: dict) -> float:
    return signal(scans, rp, region) / scans.engine().noise_sd(rp)
