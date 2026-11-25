"""Verifier core of the artefact clinic: counterfactual twins of a protocol on the hidden world, and one error measure.

A protocol is judged on a noise-free re-scan of the hidden world. The same protocol is scanned again with the artefact
mechanisms switched off (its *twins*), and the two magnitude images are compared inside a marked region:

    E(a, b; T) = sum_T m |a - b| / sum_T m b              m = fraction of each pixel that lies inside the circle T

    mechanism (switch)      twin of the delivered protocol
    chemshift               ``Engine(world, effects=ALL - {"chemshift"})``: fat is read out at the water frequency
    fatphase                the phase of the fat signal relative to water at the echo centre is removed (gradient
                            echo); the displacement during the readout is kept
    wrap                    the phase-encode field of view is extended (extra lines, acquired and discarded) until it
                            covers the object; pixel size, image grid and everything else are unchanged. Where
                            something in the object pulses, the extra lines are given the times that keep its
                            modulation the same function of k (``twin_line_spacing``) and the phase of the delivered
                            scan (``twin_of``): the ghost copies stay where they are, only the folding goes
    truncation              the object itself at the echo time, averaged over the area of each delivered pixel: what a
                            scan with unlimited k-space coverage would show on this pixel grid
    <engine effect name>    any further name in ``kspace.ALL`` (``dropout``, ``ghost``, ``motion`` ...) is switched by
                            ``effects``; a catalogue entry that names it needs nothing else here

Nothing here is an image-domain overlay or a detector: every image is formed by the engine in k-space.
"""
from __future__ import annotations

import math

import numpy as np

from ...scanner import console, kspace, tissue
from ...scanner import signal as S
from ...scanner import world as W

OWN = ("fatphase", "wrap", "truncation")          # switches implemented here; every other name is an engine effect
SUBSAMPLE = 5
WRAP_COVER = 1.05


class TwinEngine(kspace.Engine):
    """The engine with a set of mechanisms switched off. ``off``: names from the catalogue (see the module docstring)."""

    def __init__(self, world, off=(), **kw):
        self.off = frozenset(off)
        super().__init__(world, effects=kspace.ALL - (self.off - set(OWN)), **kw)

    def kspace(self, rp, te_ms, amps):
        if "fatphase" in self.off and rp["sequence"] in ("spgr", "afi", "epi"):
            w = self.w
            peaks = tissue.FAT_PEAKS if self.fat_model == "multipeak" else ((tissue.FAT_PPM, 1.0),)
            rot = np.exp(-1j * np.angle(S.fat_spectrum(1e-3 * te_ms, w["b0_t"], peaks)))
            amps = [a * rot if w["tissues"][nm]["fat"] else a for a, nm in zip(amps, w["names"])]
        return super().kspace(rp, te_ms, amps)

    def _line_times(self, rp, offset_ms=0.0):
        """Line times of a twin with extra phase-encode lines (``without_wrap``): the time between neighbouring lines
        is that of the delivered scan times ``twin_line_spacing`` (delivered lines per line of the twin), so that a
        periodic modulation is the same function of k and its copies lie where they lie in the delivered scan."""
        t = super()._line_times(rp, offset_ms)
        spacing = rp.get("twin_line_spacing")
        if spacing is None or rp["sequence"] == "epi":
            return t
        first = t[:, :1] - 1e-3 * offset_ms
        return first * spacing + (t - t[:, :1]) + 1e-3 * offset_ms

    def _motion(self, ks, rp, te_ms, contrast, amps, key):
        return super()._motion(ks, rp, te_ms, contrast, amps, rp.get("twin_of", key))   # the phase of the delivered scan


def realise(protocol, world):
    pl = console.plan(protocol, world["console"])
    if not pl["accept"]:
        raise ValueError("; ".join(v["message"] for v in pl["violations"]))
    return pl["realised"]


def without_wrap(rp, world):
    """The realised protocol with phase-encode lines added (and discarded by the reconstruction) until the encoded
    field of view covers the object from the centre of this field of view. Built directly: the console's
    ``phase_oversampling`` stops at 1."""
    box = W.extent_mm(world)
    lo, hi = box["x"] if rp["pe_dir"] == "x" else box["y"]
    c = rp.get("centre_x_mm", 0.0) if rp["pe_dir"] == "x" else rp.get("centre_y_mm", 0.0)
    need = 2.0 * WRAP_COVER * max(abs(hi - c), abs(lo - c))
    px_pe = rp["fov_pe_mm"] / rp["matrix_pe"]
    full = max(rp["lines_full"], int(math.ceil(need / px_pe / 2.0) * 2))
    if full == rp["lines_full"]:
        return rp
    twin = dict(rp, lines_full=full, lines=int(math.ceil(full * rp["partial_fourier"])))
    if world.get("motion"):                                               # see ``TwinEngine._line_times``
        twin.update(twin_line_spacing=rp["lines_full"] / full, twin_of=kspace.physics_key(rp, averages=True))
    return twin


def pixel_axes(rp):
    """Centres of the delivered pixels along (y, x), mm from the isocentre."""
    fy, fx = (rp["fov_pe_mm"], rp["fov_mm"]) if rp["pe_dir"] == "y" else (rp["fov_mm"], rp["fov_pe_mm"])
    ny, nx = (rp["matrix_pe"], rp["matrix"]) if rp["pe_dir"] == "y" else (rp["matrix"], rp["matrix_pe"])
    return (rp.get("centre_y_mm", 0.0) + (np.arange(ny) - ny // 2) * fy / ny,
            rp.get("centre_x_mm", 0.0) + (np.arange(nx) - nx // 2) * fx / nx)


def area_mask(rp, region, ss=SUBSAMPLE):
    """Fraction of each delivered pixel inside the circle ``region`` (``ss`` x ``ss`` sub-pixel samples)."""
    y, x = pixel_axes(rp)
    out = np.zeros((len(y), len(x)))
    o = (np.arange(ss) + 0.5) / ss - 0.5
    sel = []
    for axis, c in ((y, region["y_mm"]), (x, region["x_mm"])):
        step = axis[1] - axis[0]
        near = np.flatnonzero(np.abs(axis - c) <= region["radius_mm"] + step)
        if not len(near):
            return out
        sel.append((near, (axis[near][:, None] + o[None, :] * step).ravel()))
    (iy, yy), (ix, xx) = sel
    m = (yy[:, None] - region["y_mm"]) ** 2 + (xx[None, :] - region["x_mm"]) ** 2 <= region["radius_mm"] ** 2
    out[np.ix_(iy, ix)] = m.reshape(len(iy), ss, len(ix), ss).mean((1, 3))
    return out


def truth_image(engine, rp):
    """The object at the echo time on the delivered pixel grid: per class |decay at TE| x |echo amplitude|, summed
    as magnitudes (fat in phase with water), times the receive profile and the slice thickness, without any
    off-resonance, averaged over the area of each delivered pixel."""
    w = engine.w
    echo = "gre" if rp["sequence"] in ("spgr", "afi", "epi") else "se"
    zero = np.zeros(1)
    comp = np.zeros((engine.n, engine.n))
    for c, nm in enumerate(w["names"]):
        g = S.time_function(w["tissues"][nm], zero, zero, rp["te_ms"], echo, w["b0_t"], tissue.FAT_PEAKS)[0]
        comp[engine.sup[c]] += abs(g) * np.abs(engine.amplitudes(rp, c)[0])
    comp *= np.abs(w["fields"]["rx"]) * rp["slice_mm"]
    y, x = pixel_axes(rp)
    pos = (np.arange(engine.n) - engine.n // 2) * engine.d
    iy = np.floor((pos - y[0]) / (y[1] - y[0]) + 0.5).astype(int)
    ix = np.floor((pos - x[0]) / (x[1] - x[0]) + 0.5).astype(int)
    oy, ox = (iy >= 0) & (iy < len(y)), (ix >= 0) & (ix < len(x))
    out, cnt = np.zeros((len(y), len(x))), np.zeros((len(y), len(x)))
    np.add.at(out, (iy[oy][:, None], ix[ox][None, :]), comp[np.ix_(oy, ox)])
    np.add.at(cnt, (iy[oy][:, None], ix[ox][None, :]), 1.0)
    return out / np.maximum(cnt, 1.0)


def rel_l1(a, b, mask):
    """E(a, b; T): mask-weighted mean absolute difference relative to the mask-weighted mean of ``b``."""
    return float(np.sum(mask * np.abs(a - b)) / max(float(np.sum(mask * b)), 1e-30))


def block_average(fine, rp_fine, rp):
    """A finer image averaged over the area of each pixel of the grid of ``rp`` (matrix ratio 2: weights 1/4, 1/2,
    1/4 per axis around the fine pixel that shares the centre of the coarse one)."""
    out = fine
    for axis, (cf, cc) in enumerate(zip(pixel_axes(rp_fine), pixel_axes(rp))):
        j = np.clip(np.round((cc - cf[0]) / (cf[1] - cf[0])).astype(int), 0, out.shape[axis] - 1) + 1
        pad = np.pad(out, [(1, 1) if a == axis else (0, 0) for a in range(2)], mode="edge")
        out = 0.25 * np.take(pad, j - 1, axis) + 0.5 * np.take(pad, j, axis) + 0.25 * np.take(pad, j + 1, axis)
    return out


def change(a, b, mask, shape_only=False, noise_sd=0.0):
    """What an experimenter measures between two scans of one finding: E(a, b) inside ``mask``. ``shape_only``
    first scales ``a`` to the mean of ``b`` (a uniform change of signal, such as decay between two echo times, is
    not a change of the finding). ``noise_sd``: SD of the noise of a - b; its expected contribution is removed in
    quadrature."""
    area = float(np.sum(mask))
    mean_b = max(float(np.sum(mask * b)) / area, 1e-30)
    if shape_only:
        a = a * mean_b / max(float(np.sum(mask * a)) / area, 1e-30)
    raw = float(np.sum(mask * np.abs(a - b))) / area / mean_b
    floor = 0.8 * noise_sd / mean_b
    return math.sqrt(max(raw * raw - floor * floor, 0.0))


class Twins:
    """Magnitude images of one realised protocol on one hidden world, formed on demand and kept:
    ``image(off)`` with the mechanisms in ``off`` switched off, ``truth()``. Engines are shared between protocols."""

    def __init__(self, world, mechanisms):
        self.world, self.mech = world, tuple(mechanisms)
        self._engines, self._images = {}, {}

    def engine(self, off=()):
        key = frozenset(off) - {"wrap", "truncation"}
        if key not in self._engines:
            self._engines[key] = TwinEngine(self.world, off=key, cache=4)
        return self._engines[key]

    def image(self, rp, off=()):
        off = frozenset(off) & set(self.mech)
        key = (kspace.physics_key(rp, averages=True), off)
        if key not in self._images:
            scan = without_wrap(rp, self.world) if "wrap" in off else rp
            self._images[key] = np.abs(self.engine(off).acquire(scan, None)[0])
        return self._images[key]

    def truth(self, rp):
        key = (kspace.physics_key(rp, averages=True), "truth")
        if key not in self._images:
            self._images[key] = truth_image(self.engine(), rp)
        return self._images[key]

    def forget(self):
        self._images.clear()

    # ---------------------------------------------------------------------------------------- the error measures
    def switched(self):
        return [m for m in self.mech if m != "truncation"]

    def errors(self, rp, region):
        """All mechanisms at once on one region: ``mech`` = image against the twin with every switched mechanism off,
        ``trunc`` = that twin against the pixel-averaged object."""
        m = area_mask(rp, region)
        off = self.image(rp, self.switched())
        out = dict(mech=rel_l1(self.image(rp), off, m))
        if "truncation" in self.mech:
            out["trunc"] = rel_l1(off, self.truth(rp), m)
        return out

    def solo(self, rp, region):
        """Per mechanism: the error it makes on the region when it is the only one switched on, {name: E}."""
        m = area_mask(rp, region)
        sw = self.switched()
        off = self.image(rp, sw)
        out = {k: rel_l1(self.image(rp, [j for j in sw if j != k]), off, m) for k in sw}
        if "truncation" in self.mech:
            out["truncation"] = rel_l1(off, self.truth(rp), m)
        return out


def snr(engine, rp, region):
    """Mean noise-free magnitude in the region over the SD of the noise in one component of a pixel."""
    img = np.abs(engine.acquire(rp, None)[0])
    m = area_mask(rp, region)
    return float(np.sum(m * img) / np.sum(m) / engine.noise_sd(rp))
