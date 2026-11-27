"""Receive arrays: several coil elements around the object, and the multi-channel k-space of a scan.

A world with ``world["array"]`` has N receive elements in addition to its single receive profile
(``fields["rx"]``, which plays the body coil). The key is additive: a world without it is the world of before, and
nothing in ``kspace`` reads it. The array is formed here, from outside the engine, through two of its functions: the
amplitude maps of every tissue class, line by line where fat saturation treats the lines of fat differently
(``Engine._species``, the maps behind ``Engine.amplitudes``), and ``Engine.kspace``.

The array (``world["array"]``, a dict; hidden from the agent except ``n``, see ``public``)
    n, names     channel count, channel names
    maps         callable ``(y, x, z=0.0) -> complex [n, ...]``: the receive sensitivity of every element at patient
                 positions in mm (broadcastable). By reciprocity it is the transverse field a unit current in the
                 element would make there, B1- = (Bx - i By) / 2 with B0 along z (``loop_field``).
    psi          [n, n] Hermitian positive definite, dimensionless: the noise covariance in units of the world's
                 noise density. A k-space sample of channel i has noise n_i with  E[n_i conj(n_j)] = 2 sd^2 psi_ij,
                 sd = the standard deviation of ONE component of a single-channel sample (``Engine.acquire_kspace``:
                 sigma0 sqrt(readout samples x bandwidth per pixel / (averages x partitions))).
    elements, faults, rfi_coupling, healthy     the hidden layout and state (verifier side)

Declared now (``make_array``): circular loops on a ring, a row or two opposite rows around the object, each with the
Biot-Savart field of a loop in the quasi-static limit (closed form with elliptic integrals: magnitude fall-off and
phase), a cable phase and a small gain spread per element; ``psi`` from the overlap of the fields in the object,
reduced by a decoupling factor. Given from outside (``given``): any callable or a grid of maps, and any covariance:
the route by which computed fields (``mrgym.em``) replace the declared ones. Faults a quality-control task can hide
(``with_faults``): a dead element, a noisy element, a gain error.

Formation (``ArrayEngine``), [image, channel, phase line, readout sample]:
    per coil   the element's map replaces the receive profile on the master grid; one engine k-space per element.
               Slow, and the definition: the object is weighted where the spins are, before anything is encoded.
    fast       the object's k-space once (receive profile 1) on a lattice a little wider than the acquired one, to the
               image of that lattice, times the maps sampled there, back to k-space. The map is taken where the
               spins are: inside the object, with the off-resonance displacement along the readout undone. Fat is
               formed without its chemical shift, weighted, and only then given its spectrum, so that every fat
               peak is weighted at the position of the fat. Under fat saturation (engine 0.3.0) every line of fat
               has its own amplitude map and is formed, weighted and given its own frequency separately. What is
               left is the variation of a map inside the blur of one acquired sample (``ArrayEngine.fast``;
               measured by ``ArrayEngine.formation_error``).
Noise: white noise per channel, drawn channel by channel in the order of the single-channel receiver (real part, then
imaginary part), mixed with the Cholesky factor of ``psi``. One element with ``psi = [[1]]`` draws the numbers of
``Engine.acquire_kspace`` in the same order.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.special

from . import console, kspace, tissue
from . import signal as S
from . import world as W

MARGIN = 16                   # matrix units formed beyond the acquired band on every side (fast formation)
WIRE_MM = 2.0                 # the field of a loop is not followed closer to the conductor than this
PSI_LATTICE = 41              # points per axis of the lattice over the object on which the overlap is summed
SPEC = dict(
    n=8,                      # elements
    layout="ring",            # "ring" | "row" | "pair" (two opposite rows), or a list of these: one is drawn
    side=None,                # row / pair: "posterior" | "anterior" | "left" | "right"; None = drawn
    clearance_mm=(15.0, 30.0),   # distance from the object to the loop centres: a number or a (low, high) range
    loop_mm=None,             # loop radius; None = neighbours overlap critically (centres 1.5 radii apart)
    jitter_mm=3.0,            # placement error of every element (uniform, each axis)
    tilt_deg=5.0,             # error of every element's direction
    gain_log_sd=0.03,         # spread of the element gains
    coupling=0.5,             # share of the field overlap that survives decoupling as noise correlation
    centre_snr=1.0,           # optimal-combination sensitivity at the isocentre, in units of a receive profile of 1
    faults=None,              # dict(dead=, noisy=, gain=): True (the element is drawn), an index, or see ``with_faults``
)
LAYOUTS = ("ring", "row", "pair")
SIDES = ("posterior", "anterior", "left", "right")
NOISY_FACTOR = (4.0, 16.0)    # variance of a noisy element / its normal variance
GAIN_ERROR = (0.2, 0.5)       # relative size of a gain error (the sign is drawn)


# ------------------------------------------------------------------------------------------------ the field of a loop
def loop_field(a, rho, zeta):
    """Magnetic field of a circular loop of radius ``a`` carrying a steady current, at the distance ``rho`` from its
    axis and the height ``zeta`` above its plane (same length unit): (B_axial, B_radial) in units of
    mu0 I / (2 pi length). The exact Biot-Savart integral (complete elliptic integrals K, E of m = k^2):

        B_axial  = [K + (a^2 - rho^2 - zeta^2) / ((a - rho)^2 + zeta^2) E] / sqrt((a + rho)^2 + zeta^2)
        B_radial = zeta / rho [-K + (a^2 + rho^2 + zeta^2) / ((a - rho)^2 + zeta^2) E] / sqrt((a + rho)^2 + zeta^2)

    On the axis: pi a^2 / (a^2 + zeta^2)^(3/2). Closer to the conductor than ``WIRE_MM`` the field is held."""
    rho, zeta = np.abs(np.asarray(rho, float)), np.asarray(zeta, float)
    q = (a + rho) ** 2 + zeta ** 2
    m = np.clip(4 * a * rho / q, 0.0, 1 - 1e-12)
    den = np.maximum((a - rho) ** 2 + zeta ** 2, WIRE_MM ** 2)
    k, e = scipy.special.ellipk(m), scipy.special.ellipe(m)
    root = np.sqrt(q)
    b_ax = (k + (a * a - rho * rho - zeta * zeta) / den * e) / root
    near = rho < 1e-6 * a                                               # the bracket cancels there: first term of the series
    b_rad = zeta / np.where(near, 1.0, rho) * (-k + (a * a + rho * rho + zeta * zeta) / den * e) / root
    b_rad = np.where(near, 1.5 * np.pi * a * a * zeta * rho / (a * a + zeta * zeta) ** 2.5, b_rad)
    return b_ax, b_rad


class LoopMaps:
    """Sensitivity maps of loops whose axes lie in the image plane: ``(y, x, z=0) -> complex [n, ...]``.

    Element: dict(y_mm, x_mm, z_mm, normal_yx, radius_mm). The field of unit current at a point is B_axial along the
    loop's axis plus B_radial away from it; the map is its transverse part (Bx - i By) / 2 (x = patient left, y =
    posterior, z = superior and the direction of B0: a right-handed frame). In the quasi-static limit the field is
    real, so the phase of a map is minus the direction angle of the field in the plane."""

    def __init__(self, elements):
        self.elements = [dict(e) for e in elements]

    def __call__(self, y, x, z=0.0):
        y, x, z = np.broadcast_arrays(np.asarray(y, float), np.asarray(x, float), np.asarray(z, float))
        out = np.empty((len(self.elements),) + y.shape, complex)
        for k, el in enumerate(self.elements):
            ny, nx = el["normal_yx"]
            dy, dx, dz = y - el["y_mm"], x - el["x_mm"], z - el.get("z_mm", 0.0)
            zeta = dy * ny + dx * nx
            py, px = dy - zeta * ny, dx - zeta * nx
            rho = np.sqrt(py * py + px * px + dz * dz)
            b_ax, b_rad = loop_field(el["radius_mm"], rho, zeta)
            b_rad = b_rad / np.where(rho > 0, rho, 1.0)
            out[k] = 0.5 * ((b_ax * nx + b_rad * px) - 1j * (b_ax * ny + b_rad * py))
        return out


class GridMaps:
    """Maps given on a regular grid of patient positions, [n, len(y_mm), len(x_mm)] complex: cubic interpolation,
    held at the edge. The form in which a field solver hands over B1-."""

    def __init__(self, maps, y_mm, x_mm):
        self.maps = np.asarray(maps, complex)
        self.y_mm, self.x_mm = np.asarray(y_mm, float), np.asarray(x_mm, float)
        if self.maps.ndim != 3 or self.maps.shape[1:] != (self.y_mm.size, self.x_mm.size):
            raise ValueError("maps must be [n, len(y_mm), len(x_mm)]")

    def __call__(self, y, x, z=0.0):
        y, x = np.broadcast_arrays(np.asarray(y, float), np.asarray(x, float))
        iy = (y - self.y_mm[0]) / (self.y_mm[1] - self.y_mm[0])
        ix = (x - self.x_mm[0]) / (self.x_mm[1] - self.x_mm[0])
        out = np.empty((self.maps.shape[0],) + y.shape, complex)
        for k, m in enumerate(self.maps):
            out[k] = (scipy.ndimage.map_coordinates(m.real, [iy, ix], order=3, mode="nearest")
                      + 1j * scipy.ndimage.map_coordinates(m.imag, [iy, ix], order=3, mode="nearest"))
        return out


class ScaledMaps:
    """``factor[k] * base(y, x, z)[k]``: element gains, a common scale, faults."""

    def __init__(self, base, factor):
        self.base, self.factor = base, np.asarray(factor, complex)

    def __call__(self, y, x, z=0.0):
        out = self.base(y, x, z)
        return out * self.factor.reshape((-1,) + (1,) * (out.ndim - 1))


# ------------------------------------------------------------------------------------------------ the noise covariance
def object_lattice(size_mm, n=PSI_LATTICE):
    """Points (y, x) inside the ellipse of the object's declared size, on an n x n lattice over its bounding box."""
    ry, rx = size_mm[0] / 2, size_mm[1] / 2
    c = (np.arange(n) + 0.5) / n * 2 - 1
    v, u = np.meshgrid(c, c, indexing="ij")
    keep = v * v + u * u <= 1.0
    return ry * v[keep], rx * u[keep]


def overlap_psi(maps, size_mm, coupling=0.5):
    """Declared noise correlation of an array from the overlap of its elements' fields in the object:

        c_ij = Re sum_r S_i conj(S_j) / sqrt(sum |S_i|^2 sum |S_j|^2)        (= the normalised sum of B_i . B_j)
        psi  = (1 - coupling) 1 + coupling c

    Unit diagonal, real, symmetric and positive definite for coupling < 1 (a Gram matrix mixed with the identity).
    The physical quantity is the loss integral of the electric fields, sigma E_i . E_j over the object; the overlap of
    the magnetic fields stands in for it until fields are computed."""
    s = maps(*object_lattice(size_mm))
    gram = (s @ s.conj().T).real
    d = np.sqrt(np.diag(gram))
    d = np.where(d > 0, d, 1.0)
    c = gram / np.outer(d, d)
    n = len(c)
    return (1 - coupling) * np.eye(n) + coupling * (c - np.diag(np.diag(c)) + np.eye(n))


def check_psi(psi, n):
    psi = np.asarray(psi, complex)
    if psi.shape != (n, n) or not np.allclose(psi, psi.conj().T, atol=1e-12 * np.abs(psi).max()):
        raise ValueError(f"psi must be a Hermitian [{n}, {n}] matrix")
    if np.linalg.eigvalsh(psi).min() <= 0:
        raise ValueError("psi must be positive definite")
    return 0.5 * (psi + psi.conj().T)


# ------------------------------------------------------------------------------------------------ building an array
def given(maps, psi=None, names=None, y_mm=None, x_mm=None, size_mm=None, coupling=0.5, rfi_coupling=None):
    """An array from outside: ``maps`` a callable ``(y, x, z) -> [n, ...]`` or a grid [n, len(y_mm), len(x_mm)];
    ``psi`` a covariance in units of the world's noise density, or None: the overlap rule over the ellipse
    ``size_mm`` when that is given, else the identity. Attach it with ``attach``."""
    if not callable(maps):
        maps = GridMaps(maps, y_mm, x_mm)
    n = int(np.asarray(maps(np.zeros(1), np.zeros(1))).shape[0])
    if psi is None:
        psi = overlap_psi(maps, size_mm, coupling) if size_mm is not None else np.eye(n)
    return dict(n=n, names=list(names) if names else [f"ch{k + 1}" for k in range(n)], maps=maps, psi=check_psi(psi, n),
                model="given", elements=None, faults={},
                rfi_coupling=np.ones(n, complex) if rfi_coupling is None else np.asarray(rfi_coupling, complex))


def attach(world, array):
    """The world with this array (a shallow copy; the world itself is not changed)."""
    return dict(world, array=array)


def _perimeter(a, b):
    h = ((a - b) / (a + b)) ** 2
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))             # Ramanujan


def make_array(spec, ph, rng):
    """The declared array of a world, from ``world_spec["array"]`` (see ``SPEC``), the object ``ph`` (its ``size_mm``)
    and a generator of its own. Every draw is made in a fixed order whatever the spec asks for, so a fault changes
    its own term and nothing else."""
    unknown = set(spec) - set(SPEC)
    if unknown:
        raise ValueError(f"array: unknown key(s) {sorted(unknown)}")
    sp = dict(SPEC, **spec)
    n = int(sp["n"])
    if not 1 <= n <= 64:
        raise ValueError("array: n must be 1 to 64")
    u = rng.uniform(size=8)                                              # clearance, rotation, layout, side, 3 fault elements, sign
    jit, tilt = rng.uniform(-1, 1, (n, 2)), rng.uniform(-1, 1, n)
    gain = np.exp(sp["gain_log_sd"] * rng.standard_normal(n)) * np.exp(2j * np.pi * rng.uniform(size=n))
    rfi = rng.uniform(0.5, 1.5, n) * np.exp(2j * np.pi * rng.uniform(size=n))
    fault_u = rng.uniform(size=2)                                        # size of a noisy element's excess, of a gain error

    def pick(value, choices, uu, what):
        """A named choice, or one drawn from a list of them (None = all)."""
        if value is None or isinstance(value, (list, tuple)):
            pool = tuple(value) if value else choices
            value = pool[min(int(uu * len(pool)), len(pool) - 1)]
        if value not in choices:
            raise ValueError(f"array: {what} must be one of {choices}")
        return value

    layout, side = pick(sp["layout"], LAYOUTS, u[2], "layout"), pick(sp["side"], SIDES, u[3], "side")
    cl = sp["clearance_mm"]
    clear = float(cl[0] + u[0] * (cl[1] - cl[0])) if isinstance(cl, (list, tuple)) else float(cl)
    ry, rx = ph["size_mm"][0] / 2 + clear, ph["size_mm"][1] / 2 + clear
    pos = []                                                             # (y, x, normal_y, normal_x)
    if layout == "ring":
        spacing = _perimeter(ry, rx) / n
        for k in range(n):
            t = 2 * np.pi * (k + u[1]) / n
            ny, nx = -np.sin(t) / ry, -np.cos(t) / rx
            pos.append((ry * np.sin(t), rx * np.cos(t), ny, nx))
    else:
        along_x = side in ("posterior", "anterior")
        sign = 1.0 if side in ("posterior", "left") else -1.0
        rows = [(sign, n)] if layout == "row" else [(sign, n - n // 2), (-sign, n // 2)]
        length = 2 * (rx if along_x else ry)
        spacing = length / max(rows[0][1], 1)
        for sg, m in rows:
            for k in range(m):
                c = (k - (m - 1) / 2) * length / m
                pos.append((sg * ry, c, -sg, 0.0) if along_x else (c, sg * rx, 0.0, -sg))
    radius = float(sp["loop_mm"]) if sp["loop_mm"] else spacing / 1.5
    elements = []
    for k, (y0, x0, ny, nx) in enumerate(pos):
        ang = np.arctan2(ny, nx) + np.deg2rad(sp["tilt_deg"]) * tilt[k]
        elements.append(dict(y_mm=float(y0 + sp["jitter_mm"] * jit[k, 0]), x_mm=float(x0 + sp["jitter_mm"] * jit[k, 1]), z_mm=0.0,
                             normal_yx=(float(np.sin(ang)), float(np.cos(ang))), radius_mm=radius))
    loops = LoopMaps(elements)
    corr = overlap_psi(loops, ph["size_mm"], sp["coupling"])
    s0 = loops(np.zeros(1), np.zeros(1))[:, 0]
    scale = sp["centre_snr"] / np.sqrt((s0.conj() @ np.linalg.solve(corr, s0)).real)
    for el, g in zip(elements, gain):
        el["gain"] = complex(g)
    arr = dict(n=n, names=[f"ch{k + 1}" for k in range(n)], maps=ScaledMaps(loops, scale * gain),
               psi=check_psi(gain[:, None] * corr * gain.conj()[None, :], n), model="loops", elements=elements, faults={},
               rfi_coupling=rfi, layout=layout, side=side if layout != "ring" else None, clearance_mm=clear)
    faults = {}
    for i, name in enumerate(("dead", "noisy", "gain")):
        v = (sp["faults"] or {}).get(name)
        if v is None or v is False:
            continue
        el = min(int(u[4 + i] * n), n - 1) if v is True else int(v["element"] if isinstance(v, dict) else v)
        if name == "dead":
            faults["dead"] = [el]
        elif name == "noisy":
            f = v.get("factor") if isinstance(v, dict) else None
            faults["noisy"] = {el: float(f) if f else NOISY_FACTOR[0] + fault_u[0] * (NOISY_FACTOR[1] - NOISY_FACTOR[0])}
        else:
            g = v.get("gain") if isinstance(v, dict) else None
            size = GAIN_ERROR[0] + fault_u[1] * (GAIN_ERROR[1] - GAIN_ERROR[0])
            faults["gain"] = {el: complex(g) if g else 1.0 + (size if u[7] < 0.5 else -size)}
    return with_faults(arr, **faults) if faults else arr


def with_faults(array, dead=(), noisy=None, gain=None):
    """The array with faults; the healthy one is kept under ``healthy``.

        dead    element indices: no signal; the channel still delivers noise at its level, uncorrelated with the others
        noisy   {index: factor}: the variance of that channel is multiplied by ``factor`` (uncorrelated excess noise)
        gain    {index: g}: signal and noise of that channel are multiplied by the complex gain ``g``
    """
    n, psi = array["n"], np.array(array["psi"], complex)
    dead, noisy, gain = sorted(int(k) for k in dead), dict(noisy or {}), dict(gain or {})
    factor = np.ones(n, complex)
    for k in dead:
        d = psi[k, k]
        psi[k, :], psi[:, k], psi[k, k] = 0, 0, d
        factor[k] = 0
    for k, f in noisy.items():
        if f < 1:
            raise ValueError("a noisy element has a variance factor of at least 1")
        psi[k, k] *= f
    g = np.ones(n, complex)
    for k, v in gain.items():
        g[k] = v
    psi = g[:, None] * psi * g.conj()[None, :]
    healthy = array.get("healthy") or dict(maps=array["maps"], psi=array["psi"])
    return dict(array, maps=ScaledMaps(array["maps"], factor * g), psi=check_psi(psi, n), healthy=healthy,
                faults=dict(dead=dead, noisy={int(k): float(v) for k, v in noisy.items()},
                            gain={int(k): complex(v) for k, v in gain.items()}))


def public(world) -> dict:
    """What the agent is told about the receiver: the number of channels (1 without an array)."""
    return dict(channels=int(world["array"]["n"]) if world.get("array") else 1)


# ------------------------------------------------------------------------------------------------ geometry
def patient_coords(world, v, u):
    """Patient positions (y, x) in mm of the points (v, u) (mm along image rows and columns from the isocentre) of a
    world's frame: turned by ``world["rot_deg"]`` and passed through its coordinate map, as ``world.grid`` does for
    the master grid."""
    v, u = np.broadcast_arrays(np.asarray(v, float), np.asarray(u, float))
    rot = float(world.get("rot_deg", 0.0) or 0.0)
    if rot:
        co, si = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
        v, u = v * co - u * si, v * si + u * co
    warp = W._warp(world.get("gnl"), (world.get("system") or {}).get("grad_scale"))
    return warp(v, u) if warp is not None else (v, u)


def lattice(rp):
    """The acquired lattice of a realised protocol, in the frame of the acquisition [phase, readout]."""
    ro_os = console.SYSTEM["ro_os"]
    k_pe, k_ro = rp["lines_full"], ro_os * rp["matrix"]
    fov_pe, fov_ro = rp["fov_pe_mm"] * k_pe / rp["matrix_pe"], ro_os * rp["fov_mm"]
    cy, cx = rp.get("centre_y_mm", 0.0), rp.get("centre_x_mm", 0.0)
    p0, r0 = (cx, cy) if rp["pe_dir"] == "x" else (cy, cx)
    return dict(k_pe=k_pe, k_ro=k_ro, fov_pe=fov_pe, fov_ro=fov_ro, p0=p0, r0=r0, swap=rp["pe_dir"] == "x",
                lo_p=k_pe // 2 - rp["matrix_pe"] // 2, lo_r=k_ro // 2 - rp["matrix"] // 2)


# ------------------------------------------------------------------------------------------------ formation
class ArrayEngine:
    """Multi-channel data of one world with an array: ``acquire_kspace(realised protocol, rng)``.

    ``method``: "fast", "per_coil", or "auto" = fast where it is valid (a line-by-line readout of an object that does
    not move, with the receiver's band filter) and per coil otherwise (EPI: the displacement along the phase
    direction is many pixels; motion: the engine's line modulation acts on the coil-weighted object)."""

    def __init__(self, world, tier=2, fat_model="multipeak", effects=None, cache=4, margin=MARGIN, method="auto"):
        if not world.get("array"):
            raise ValueError("this world has no receive array (world_spec['array'], or array.attach)")
        self.world = self.w = world
        self.arr = world["array"]
        self.n_coils = int(self.arr["n"])
        self.maps, self.psi = self.arr["maps"], np.asarray(self.arr["psi"], complex)
        self.chol = np.linalg.cholesky(self.psi)
        self.margin, self.method, self.cache_size = int(margin), method, cache
        self._kw = dict(tier=tier, fat_model=fat_model, effects=effects)
        self.effects = kspace.Engine(world, **self._kw).effects
        # The engine of the fast formation: receive profile 1, fat as one line without its chemical shift (``fast``
        # weights the object with the maps first and applies the spectrum afterwards). It also serves the geometry.
        self.engine = kspace.Engine(self._with_rx(lambda t: np.ones((t["n"], t["n"]), complex)), tier=tier, fat_model="single",
                                    effects=self.effects - {"chemshift"})
        # The same world with the lines of the fat model: it gives the amplitude maps (under fat saturation one per line).
        self.species = kspace.Engine(self.engine.w, tier=tier, fat_model=fat_model, effects=self.effects)
        self.fat_model = fat_model
        self._clean, self._maps, self._near = {}, {}, {}

    def _with_rx(self, rx_of):
        """The world with another receive profile ``rx_of(world) -> [n, n]``, on a turned grid as well."""
        def hook(t):
            t["fields"] = dict(t["fields"], rx=rx_of(t))
        w = dict(self.world, post_build=[*self.world.get("post_build", ()), hook])
        hook(w)
        return w

    # ---------------------------------------------------------------- which formation
    def method_for(self, rp, method=None):
        method = method or self.method
        if method != "auto":
            return method
        moving = bool(self.w.get("motion")) and "motion" in self.effects
        return "per_coil" if rp["sequence"] == "epi" or moving or "aa_filter" not in self.effects else "fast"

    # ---------------------------------------------------------------- per coil: the definition
    def per_coil(self, rp):
        """[image, channel, line, sample]: one engine per element, its map in place of the receive profile."""
        out, held = [], {}

        def rx(t, j):
            key = (t["n"], float(t.get("rot_deg", 0.0)))                 # the maps on one grid, evaluated once for all elements
            if key not in held:
                if len(held) >= 2:                                       # the upright grid and one turned grid
                    held.pop(next(iter(held)))
                held[key] = self.maps(*W.grid(t))
            return held[key][j]

        for j in range(self.n_coils):
            out.append(kspace.Engine(self._with_rx(lambda t, j=j: rx(t, j)), cache=1, **self._kw).clean_kspace(rp))
        return np.stack(out, 1)

    # ---------------------------------------------------------------- fast: the object once, the coils in image space
    def _extended(self, rp, eng):
        """The protocol on a k-space lattice with the same spacing and sample times, ``margin`` matrix units more on
        every side, and a phase lattice refined until its field of view holds the whole master grid (the object folds
        after the coil has weighted it, as on a scanner); the positions of that lattice; where to find the acquired
        samples on it."""
        g, m = lattice(rp), self.margin
        ro_os = console.SYSTEM["ro_os"]
        kpe1, mat2 = g["k_pe"] + 2 * m, rp["matrix"] + 2 * m
        f = max(1, int(np.ceil((eng.w["extent_mm"] + 2 * abs(g["p0"])) / g["fov_pe"] - 1e-9)))
        rp_f = dict(rp, matrix=mat2, bandwidth_hz_per_px=rp["bandwidth_hz_per_px"] * rp["matrix"] / mat2,
                    lines_full=f * kpe1, lines=f * kpe1, matrix_pe=rp["matrix_pe"] * kpe1 / g["k_pe"])
        n_pe, n_ro = f * kpe1, ro_os * mat2
        p = g["p0"] + (np.arange(n_pe) - n_pe // 2) * (f * g["fov_pe"] / n_pe)
        r = g["r0"] + (np.arange(n_ro) - n_ro // 2) * (g["fov_ro"] / n_ro)
        pick_pe = (f * (np.arange(g["k_pe"]) - g["k_pe"] // 2)) % n_pe      # indices of the acquired samples, k = 0 at index 0
        pick_ro = (np.arange(g["k_ro"]) - g["k_ro"] // 2) % n_ro
        return rp_f, p, r, pick_pe, pick_ro, g

    def _lattice_maps(self, eng, rp, p, r, g):
        """Maps on the extended lattice, k = 0 ordering of the FFT (index 0 = the centre of the field of view), taken
        where the spins are: the image shows a spin displaced along the readout by -polarity x off-resonance /
        bandwidth pixels (the chemical shift is not in the images that are multiplied, see ``fast``)."""
        key = (round(float(eng.w.get("rot_deg", 0.0)), 6), g["swap"], p.size, r.size, float(p[0]), float(r[0]), float(p[1] - p[0]),
               float(r[1] - r[0]), rp["ro_polarity"], rp["bandwidth_hz_per_px"], rp["fov_mm"], rp["matrix"])
        if key not in self._maps:
            if len(self._maps) >= 4:
                self._maps.pop(next(iter(self._maps)))
            pp, rr = np.meshgrid(p, r, indexing="ij")
            if "b0shift" in self.effects:
                d, n = eng.d, eng.n
                vv, uu = (rr, pp) if g["swap"] else (pp, rr)
                df = scipy.ndimage.map_coordinates(eng.df, [vv / d + n // 2, uu / d + n // 2], order=1, mode="nearest")
                rr = rr + rp["ro_polarity"] * df / rp["bandwidth_hz_per_px"] * (rp["fov_mm"] / rp["matrix"])
            v, u = (rr, pp) if g["swap"] else (pp, rr)
            self._maps[key] = np.fft.ifftshift(self.maps(*patient_coords(eng.w, *self._inside(eng, v, u))), axes=(1, 2))
        return self._maps[key]

    def _inside(self, eng, v, u):
        """Positions (mm along rows and columns of the engine's grid) moved into the object: a point outside it
        becomes the nearest grid point of the object. Outside the object a map multiplies nothing but the ringing of
        the band-limited image, which belongs to the spins at the edge; the field of a loop next to its conductor,
        which a wide lattice reaches, must not."""
        key = round(float(eng.w.get("rot_deg", 0.0)), 6)                 # the grid of this engine: the world turned by this angle
        if key not in self._near:
            self._near = {key: scipy.ndimage.distance_transform_edt(~eng.body, return_distances=False, return_indices=True).astype(np.int32)}
        n, d = eng.n, eng.d
        iv = np.clip(np.rint(v / d).astype(np.int64) + n // 2, 0, n - 1)
        iu = np.clip(np.rint(u / d).astype(np.int64) + n // 2, 0, n - 1)
        out = ~eng.body[iv, iu]
        near = self._near[key]
        return (np.where(out, (near[0][iv, iu] - n // 2) * d, v), np.where(out, (near[1][iv, iu] - n // 2) * d, u))

    def fast(self, rp):
        """[image, channel, line, sample] by the fast formation (module docstring).

        Water: object k-space on the extended lattice -> image -> x maps -> k-space. Fat: the same, formed without
        its chemical shift (``self.engine``), so that fat is weighted where it is; the spectrum of the fat model then
        multiplies the readout samples, spectrum(t0 + tau) with tau the time of the sample: every peak is displaced
        by its own shift, exactly as in the engine, after the coil has weighted it. Under fat saturation the lines of
        a fat class are saturated by their place in the band of the pulse and have their own amplitude maps
        (``Engine._species``): each line is then formed and weighted by itself and given its own frequency."""
        (seng, _), (eng, rp) = self.species._frame(rp), self.engine._frame(rp)
        w, eff = eng.w, self.effects
        rp_f, p, r, pick_pe, pick_ro, g = self._extended(rp, eng)
        n_c = len(w["names"])
        species = [seng._species(rp, c) for c in range(n_c)]            # [class] (lines or None, [contrast])
        fat = [bool(w["tissues"][nm]["fat"]) for nm in w["names"]]
        maps = self._lattice_maps(eng, rp, p, r, g)
        echo = "gre" if rp["sequence"] in ("spgr", "afi", "epi") else "se"
        tau = rp["ro_polarity"] * (np.arange(g["k_ro"]) - g["k_ro"] // 2) / (g["k_ro"] * rp["bandwidth_hz_per_px"])
        peaks = tissue.FAT_PEAKS if self.fat_model == "multipeak" else ((tissue.FAT_PPM, 1.0),)
        n_lines = max([len(lines) for lines, _ in species if lines is not None], default=0)
        # the groups that are formed separately: water; fat that evolves as one species; line j of the fat that does not
        groups = [("water", None), ("fat", None)] + [("line", j) for j in range(n_lines)]
        out = []
        for k in range(len(species[0][1])):
            for te in rp["echo_times_ms"]:
                acc = 0.0
                t0 = 1e-3 * te if echo == "gre" else 0.0
                t = t0 + (tau if "chemshift" in eff else 0 * tau)
                carrier = S.fat_spectrum(np.array(t0), w["b0_t"], ((tissue.FAT_PPM, 1.0),))      # what the one-line engine put in at the echo
                for kind, j in groups:
                    amps, spec = [None] * n_c, None
                    for c in range(n_c):
                        lines, amp = species[c]
                        if kind == "line" and lines is not None:
                            amps[c] = lines[j][1] * amp[k][j]                    # share of the line x its amplitude map
                            spec = S.fat_spectrum(t, w["b0_t"], ((lines[j][0], 1.0),))
                        elif lines is None and (kind == "fat") == fat[c] and kind != "line":
                            amps[c] = amp[k]
                            spec = S.fat_spectrum(t, w["b0_t"], peaks) if fat[c] else None
                    if all(a is None for a in amps):
                        continue
                    k0 = eng.kspace(rp_f, te, [np.zeros(int(np.count_nonzero(eng.sup[c])), complex) if a is None else a for c, a in enumerate(amps)])
                    part = np.fft.fft2(np.fft.ifft2(np.fft.ifftshift(k0))[None] * maps, axes=(1, 2))[:, pick_pe][:, :, pick_ro]
                    if spec is not None:
                        part = part * (spec / carrier)[None, None, :]
                    acc = acc + part
                out.append(acc)
        return np.array(out)

    def formation_error(self, rp) -> float:
        """rms difference of the fast formation from the per-coil formation for this protocol, relative to the rms of
        the data: the check to run before a new coil geometry or protocol family is scored with the fast formation."""
        ref = self.per_coil(rp)
        return float(np.sqrt((np.abs(self.fast(rp) - ref) ** 2).sum() / (np.abs(ref) ** 2).sum()))

    def clean_kspace(self, rp, method=None):
        """Noise-free multi-channel k-space of every image of a scan, [image, channel, line, sample]; cached."""
        method = self.method_for(rp, method)
        moving = bool(self.w.get("motion")) and "motion" in self.effects
        key = (method, kspace.physics_key(rp, averages=moving))
        if key not in self._clean:
            if len(self._clean) >= self.cache_size:
                self._clean.pop(next(iter(self._clean)))
            self._clean[key] = self.per_coil(rp) if method == "per_coil" else self.fast(rp)
        return self._clean[key]

    # ---------------------------------------------------------------- the receiver
    def sample_sd(self, rp) -> float:
        """Standard deviation of one component of a k-space sample of a channel with psi_ii = 1: the single-channel
        receiver's (``Engine.acquire_kspace``)."""
        k_ro = console.SYSTEM["ro_os"] * rp["matrix"]
        return float(self.w["system"]["sigma0"] * np.sqrt(k_ro * rp["bandwidth_hz_per_px"] / (rp["averages"] * rp.get("partitions", 1))))

    def noise(self, rp, rng):
        """Noise of one image, [channel, line, sample]: per channel the two draws of the single-channel receiver, in
        its order, then mixed with the Cholesky factor of psi."""
        g = lattice(rp)
        white = np.empty((self.n_coils, g["k_pe"], g["k_ro"]), complex)
        for c in range(self.n_coils):
            white[c] = rng.standard_normal((g["k_pe"], g["k_ro"])) + 1j * rng.standard_normal((g["k_pe"], g["k_ro"]))
        white = self.sample_sd(rp) * white
        return white if self.n_coils == 1 and self.chol[0, 0] == 1 else np.einsum("cd,dpr->cpr", self.chol, white)

    def acquire_kspace(self, rp, rng=None, lines=None, method=None):
        """Multi-channel k-space as the receiver delivers it, [image, channel, line, sample]: interference, noise,
        lines that were not acquired zeroed. ``lines``: boolean [lines_full], the lines an accelerated scan acquires
        (``recon.sampling_mask``); the random stream is consumed as for the full scan. ``rng`` None: no noise."""
        ks = self.clean_kspace(rp, method).copy()
        sy, nex, eff = self.w["system"], rp["averages"], self.effects
        k_pe, k_ro = ks.shape[2:]
        acquired = self.engine._acquired(rp)
        if lines is not None:
            acquired = acquired & np.asarray(lines, bool)
        for i in range(ks.shape[0]):
            if rng is not None and "rfi" in eff and sy.get("rfi"):
                f = sy["rfi"]["freq_hz"]
                if abs(f) < k_ro * rp["bandwidth_hz_per_px"] / 2:
                    t_s = rp["ro_polarity"] * (np.arange(k_ro) - k_ro // 2) / (k_ro * rp["bandwidth_hz_per_px"])
                    ph = np.exp(2j * np.pi * rng.uniform(size=(nex, k_pe))).mean(0)
                    line = sy["rfi"]["amp"] * ph[:, None] * np.exp(2j * np.pi * f * t_s)[None, :]
                    ks[i] += line if self.n_coils == 1 and self.arr["rfi_coupling"][0] == 1 else self.arr["rfi_coupling"][:, None, None] * line
            if rng is not None and "noise" in eff and sy["sigma0"] > 0:
                ks[i] += self.noise(rp, rng)
            ks[i][:, ~acquired] = 0
        return ks

    # ---------------------------------------------------------------- images and the verifier's noise map
    def coil_images(self, ks, rp, full_pe=False):
        """Coil images of k-space [..., line, sample] with the console's scaling. Default: as ``Engine.recon`` returns
        an image, [..., y, x]. ``full_pe``: in the frame of the acquisition on the whole phase lattice (phase
        oversampling kept, readout oversampling removed), [..., line, readout pixel]: what an unfolding works on."""
        _, rp = self.engine._frame(rp)
        g = lattice(rp)
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks, axes=(-2, -1))), axes=(-2, -1)) * (g["k_pe"] * g["k_ro"] / (g["fov_ro"] * g["fov_pe"]))
        img = img[..., g["lo_r"]:g["lo_r"] + rp["matrix"]]
        return img if full_pe else self.to_image(img, rp)

    def to_image(self, img, rp):
        """An image on the whole phase lattice [..., line, readout pixel] as the console shows it, [..., y, x]."""
        g = lattice(rp)
        img = img[..., g["lo_p"]:g["lo_p"] + rp["matrix_pe"], :]
        return np.swapaxes(img, -2, -1) if g["swap"] else img

    def pixel_coords(self, rp):
        """Patient positions (y, x) of the pixel centres of the whole phase lattice, [line, readout pixel]."""
        eng, rp = self.engine._frame(rp)
        g = lattice(rp)
        p = g["p0"] + (np.arange(g["k_pe"]) - g["k_pe"] // 2) * (g["fov_pe"] / g["k_pe"])
        r = g["r0"] + (np.arange(rp["matrix"]) - rp["matrix"] // 2) * (rp["fov_mm"] / rp["matrix"])
        pp, rr = np.meshgrid(p, r, indexing="ij")
        return patient_coords(eng.w, *((rr, pp) if g["swap"] else (pp, rr)))

    def pixel_maps(self, rp, healthy=False):
        """The true maps at the pixel centres of the whole phase lattice, [channel, line, readout pixel]."""
        maps = self.arr["healthy"]["maps"] if healthy and self.arr.get("healthy") else self.maps
        return maps(*self.pixel_coords(rp))

    def image_sd(self, rp, lines=None) -> float:
        """Standard deviation of one component of a coil-image pixel of a channel with psi_ii = 1 (console scaling),
        from the samples actually acquired: ``Engine.noise_sd`` for the lines of this scan."""
        g = lattice(rp)
        acquired = self.engine._acquired(rp)
        if lines is not None:
            acquired = acquired & np.asarray(lines, bool)
        return float(self.sample_sd(rp) * np.sqrt(acquired.sum() * g["k_ro"]) / (g["fov_ro"] * g["fov_pe"]))

    def noise_sd_map(self, rp, acceleration=1, reg=0.0):
        """Verifier side: standard deviation of one component of every pixel of the optimal reconstruction of this
        scan with the true maps, [y, x]: sigma_full(p) at full sampling (Roemer), sigma_full(p) g(p) sqrt(R) for a
        scan that keeps every R-th line (SENSE). With ``reg`` it is the noise of the regularised unfolding."""
        from . import recon
        plan = recon.sense_plan(self.pixel_maps(rp), self.psi, acceleration, reg=reg)
        sd = self.image_sd(rp, recon.sampling_mask(rp["lines_full"], acceleration)) * acceleration * plan["unfold_noise"]
        return self.to_image(sd, self.engine._frame(rp)[1])
