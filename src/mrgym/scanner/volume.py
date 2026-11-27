"""The hidden world in three dimensions: a seeded recipe of analytic shapes, evaluated exactly on any plane.

A 2D world (``world.py``) is one axial section. A volume world is an object with structure along the third axis:
``head3d`` (a head) and ``qa3d`` (a cylinder test object). Nothing is stored as voxels. A world is a RECIPE: a few
dozen shapes (signed distances of ellipsoids, tubes, planes, layered shells) behind a smooth coordinate warp, all
drawn from the secret seed. The recipe is evaluated where it is needed: at points (``eval_points``), on a plane of
any position, angle and grid (``eval_plane``), or on a coarse lattice (``lattice``). No scan is formed here.

COORDINATES (stated once, used everywhere in this module)
  Patient coordinates in mm from the isocentre. A vector is written (y, x, z), the order of the engine's arrays:
      +y  posterior (the back of the head)     +x  the patient's left     +z  superior (towards the top of the head)
  This is the 2D convention of ``anatomy.py`` (+y posterior, +x left) with the head-foot axis added; (x, y, z) is
  the right-handed DICOM patient system (L, P, S). The static field B0 is along +z.
  A plane is a ``Frame``: an origin and three orthonormal vectors ``row``, ``col``, ``normal``. Grid point (i, j)
  of an n x n grid of extent E lies at  origin + (i - n // 2) (E / n) row + (j - n // 2) (E / n) col : the engine's
  in-plane rule. ``row`` is the direction of growing row index (down on a displayed image), ``col`` of growing
  column index (to the right). The three standard planes:
      axial      row +y (anterior at the top), col +x (the patient's left on the right), normal +z
      coronal    row -z (the top of the head at the top), col +x, normal +y
      sagittal   row -z, col +y (anterior on the left), normal +x
  The normal of a standard plane is the positive patient axis, so a position along it is the patient coordinate.
  ``Frame.plane("axial")`` at z = 0 is the grid of every 2D world (``anatomy.axes``), and ``rot_deg`` turns it as
  ``world.turned`` does (rows from +y towards +x).

CLASSES
  ``names``    the tissue classes that give signal: the order of ``world["names"]`` and of ``eval_plane``
  ``solid``    matter that gives no signal (the plastic of a test object); empty for the head
  ``matter``   names + solid: the order of ``lattice`` and of ``eval_*(..., solid=True)``
  Where there is matter the weights of ``matter`` sum to 1; where there is air they sum to less (0 outside the object
  and inside an air cavity). Boundaries are soft with the half-width of the 2D worlds (``anatomy.EDGE_MM``).

INTERFACE FOR THE FIELD BUILDER AND THE SLICE ENGINE (kept stable; ``vol = world["volume"]``)
  vol.eval_plane(frame, n, extent_mm) -> float32 [len(names), n, n]      class weights on a plane (signal classes)
  vol.eval_grid(frame, v_mm, u_mm)    -> float32 [len(names), nv, nu]    the same on any in-plane coordinates
  vol.eval_points(y, x, z)            -> float64 [len(names), ...]       the recipe at points: the reference
  vol.lattice(spacing_mm)             -> Lattice(weights, origin_mm, spacing_mm)
        weights    float32 [len(matter), nz, ny, nx]: the value at each voxel centre; ``supersample=k`` or
                   (kz, ky, kx): the mean over the voxel; ``labels=True``: uint8 [nz, ny, nx], 0 = no matter,
                   k + 1 = matter[k]
        origin_mm  (z, y, x) of voxel [0, 0, 0]; spacing_mm (dz, dy, dx): voxel [k, i, j] is centred at
                   origin + (k dz, i dy, j dx). Both follow the ARRAY axes (z, y, x), not the vector order.
  vol.susceptibility_ppm              dict: class of ``matter`` -> volume susceptibility relative to soft tissue,
                                      ppm (SI), plus the key "air"
  vol.chi_ppm(weights)                susceptibility of a weight array of all matter classes (class axis first):
                                      sum_c chi_c w_c + chi_air (1 - sum_c w_c)
  vol.organ                           the classes a shim is fitted over; vol.organ_mask(weights) -> bool
  vol.regions, vol.truth(), vol.layout, vol.inside(region, y, x, z), vol.bounds_mm
  Frame.plane(name, pos_mm, tilt_deg, rot_deg), frame.shifted(mm), frame.points(v, u), frame.coords(y, x, z)

WHAT IS EXACT. ``eval_points`` evaluates the recipe directly in float64. ``eval_plane`` is the same recipe on a
plane with two shortcuts that change no value beyond rounding: a small shape is evaluated only inside the
rectangle its bounding box can reach on the plane, and a smooth random field (a sum of plane cosines) is one small
matrix product, because on a plane cos(a + b v + c u) = Re(e^{ia} e^{ibv} e^{icu}). The head is evaluated in float32
on a plane (plane against points: a few 1e-5, tests/test_volume_world.py); the test object in float64. The matrix
product is the one place where a linear-algebra library enters: the recipe, the regions and ``eval_points`` use
elementwise arithmetic only.

RANDOMNESS. The recipe draws from the ``geometry`` and ``tissue`` streams of the seed. Every option (lesions,
vessels, the 3D inserts of the test object, the pose) draws from a sub-stream of its own, so switching one on or
off moves no other number. ``qa3d`` draws the layout of its vial section exactly as ``anatomy.qa`` does: its axial
plane through the isocentre is the 2D ``qa`` world of the same seed and spec, bit for bit.

WHERE THINGS ARE. The shapes of the head are written behind its warp, so a centre in the recipe is not where the
structure is. ``layout`` and ``regions`` give patient positions (the warp inverted); sizes in ``layout`` are those
of the recipe (the warp changes a length by a few per cent). Vessels are not warped: they are straight.

SUSCEPTIBILITIES (``CHI_PPM``). Air against soft tissue is the engine's own ``world.DCHI_AIR`` (9.4 ppm, Schenck
1996 as cited in world.py). Every other value is DECLARED [D]: chosen by the authors of this package (bone -2.0,
fatty marrow -0.6, fat +0.6, deep grey matter +0.10, blood +0.40 as for venous blood, lesion +-0.3 to 1.2 with a
hidden sign), of the size the literature reports, with no source opened for this module; plastic and the gels of
the test object are set to 0. docs/physics/volume.md lists them as declared.

Thread safety: a ``Volume`` holds numbers and is never changed by an evaluation (the one cached lattice is replaced,
not edited). A world is a function of (seed, spec) alone (tests/test_volume_world.py, as test_thread_determinism).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import anatomy, tissue

KINDS = ("head3d", "qa3d")
GRID = dict(head3d=(640, 256.0), qa3d=(640, 256.0))                     # master grid of the reference plane, as ``world.GRID``
EXAM_FOV_MM = dict(head3d=240.0, qa3d=240.0)
VOLUME_SPEC = dict(
    vessels=3,              # head3d: number of straight vessels (0 = none)
    pose_deg=None,          # the object is turned about its centre: a number or (low, high) per axis; None = upright
    inserts=None,           # placed shapes on top of everything inside the object, see ``_insert_shapes``
    features=True,          # qa3d: the sphere, the wedge pair and the ramp block (False: the vial section alone)
)
# Volume susceptibility relative to soft tissue (water-like), ppm, SI. "air" from world.DCHI_AIR; the rest declared [D].
CHI_PPM = dict(air=9.4, skull=-2.0, diploe=-0.6, scalp_fat=0.6, deep_gm=0.10, blood=0.40, plastic=0.0)
LESION_CHI_PPM = (0.3, 1.2)             # [D] size of the susceptibility of a lesion; its sign is drawn (hidden)
PURITY_MIN = 0.95                       # a region is listed only if its class fills it at least this far
LATTICE_MAX_BYTES = 1.5e9               # ``lattice`` refuses a float32 weight array larger than this (use labels=True)
STRIP_POINTS = 82000                    # points of a strip of a float32 plane evaluation (128 rows of 640)
THIN_EDGE = 0.3                         # soft edge of a thin sheet (the ramps), as a share of the world's half-width


# ---------------------------------------------------------------------------------------------- frames
def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if v.shape != (3,) or not n > 0:
        raise ValueError("a direction is three numbers (y, x, z), not all zero")
    return v / n


class Frame:
    """A plane in the patient: ``origin`` (y, x, z) in mm and the orthonormal vectors ``row``, ``col``, ``normal``
    (see the module docstring). ``normal`` defaults to the cross product of the (y, x, z) triples of row and col,
    which is +z for the axial plane; it may be given, with either sign."""

    __slots__ = ("origin", "row", "col", "normal")

    def __init__(self, origin=(0.0, 0.0, 0.0), row=(1.0, 0.0, 0.0), col=(0.0, 1.0, 0.0), normal=None):
        self.origin = np.asarray(origin, float)
        self.row, self.col = _unit(row), _unit(col)
        self.normal = np.cross(self.row, self.col) if normal is None else _unit(normal)
        if self.origin.shape != (3,):
            raise ValueError("origin is three numbers (y, x, z) in mm")
        if max(abs(float(self.row @ self.col)), abs(float(self.row @ self.normal)), abs(float(self.col @ self.normal))) > 1e-9:
            raise ValueError("row, col and normal of a frame must be perpendicular to each other")

    @classmethod
    def plane(cls, name="axial", pos_mm=0.0, tilt_deg=(0.0, 0.0), rot_deg=0.0, origin=None):
        """A standard plane through the point ``pos_mm`` on its patient axis (or through ``origin``), turned in the
        plane by ``rot_deg`` (rows from the row axis towards the column axis, as ``world.turned``) and then tilted
        by ``tilt_deg`` = (about its row axis, about its column axis)."""
        std = dict(axial=((1, 0, 0), (0, 1, 0), (0, 0, 1)), coronal=((0, 0, -1), (0, 1, 0), (1, 0, 0)),
                   sagittal=((0, 0, -1), (1, 0, 0), (0, 1, 0)))
        if name not in std:
            raise ValueError(f"plane must be one of {sorted(std)}")
        r, c, n = (np.array(v, float) for v in std[name])
        o = n * float(pos_mm) if origin is None else np.asarray(origin, float)
        if rot_deg:
            co, si = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
            r, c = r * co + c * si, c * co - r * si
        a, b = (np.deg2rad(float(t)) for t in tilt_deg)
        if a:
            c, n = c * np.cos(a) + n * np.sin(a), n * np.cos(a) - c * np.sin(a)
        if b:
            r, n = r * np.cos(b) + n * np.sin(b), n * np.cos(b) - r * np.sin(b)
        return cls(o, r, c, n)

    def shifted(self, mm):
        """The parallel plane ``mm`` further along the normal."""
        return Frame(self.origin + float(mm) * self.normal, self.row, self.col, self.normal)

    def points(self, v_mm, u_mm):
        """Patient coordinates (y, x, z), each [len(v), len(u)], of the in-plane positions v (along row), u (along col)."""
        v, u = np.asarray(v_mm, float)[:, None], np.asarray(u_mm, float)[None, :]
        return [self.origin[i] + v * self.row[i] + u * self.col[i] for i in range(3)]

    def coords(self, y, x, z):
        """(v, u, distance from the plane along the normal) of patient points."""
        d = [np.asarray(a, float) - self.origin[i] for i, a in enumerate((y, x, z))]
        return [sum(d[i] * e[i] for i in range(3)) for e in (self.row, self.col, self.normal)]

    def key(self):
        return tuple(round(float(v), 6) for v in np.concatenate([self.origin, self.row, self.col, self.normal]))

    def __repr__(self):
        o, r, c, n = (np.round(v, 4).tolist() for v in (self.origin, self.row, self.col, self.normal))
        return f"Frame(origin={o}, row={r}, col={c}, normal={n})"


def grid_axis(n, extent_mm):
    """In-plane positions of the points of an n-point axis: the engine's rule, index i at (i - n // 2) * extent / n."""
    return (np.arange(int(n)) - int(n) // 2) * (float(extent_mm) / int(n))


def rotation(about_y_deg=0.0, about_x_deg=0.0, about_z_deg=0.0):
    """Rotation matrix acting on (y, x, z) triples: about the y axis, then the x axis, then the z axis."""
    a, b, c = (np.deg2rad(float(v)) for v in (about_y_deg, about_x_deg, about_z_deg))
    ry = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rx = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return ry @ rx @ rz


# ---------------------------------------------------------------------------------------------- signed distances
# All take arrays of one shape and dtype and keep the dtype (parameters are Python floats). Negative inside, mm.
def sd_sphere(y, x, z, c, radius):
    return np.sqrt((y - c[0]) ** 2 + (x - c[1]) ** 2 + (z - c[2]) ** 2) - radius


def sd_ellipsoid(y, x, z, c, ax, rot=None):
    """First-order signed distance to an ellipsoid with centre ``c``, semi-axes ``ax`` (along y, x, z before the
    rotation ``rot``, a 3 x 3 matrix on (y, x, z) triples): (r - 1) / |grad r| of the implicit function r."""
    u = [y - c[0], x - c[1], z - c[2]]
    if rot is not None:
        u = [rot[i][0] * u[0] + rot[i][1] * u[1] + rot[i][2] * u[2] for i in range(3)]
    r2 = (u[0] * (1.0 / ax[0])) ** 2 + (u[1] * (1.0 / ax[1])) ** 2 + (u[2] * (1.0 / ax[2])) ** 2
    g2 = (u[0] * (1.0 / ax[0] ** 2)) ** 2 + (u[1] * (1.0 / ax[1] ** 2)) ** 2 + (u[2] * (1.0 / ax[2] ** 2)) ** 2
    r = np.sqrt(r2)
    return np.where(r > 0.05, (r - 1.0) * r / np.sqrt(np.maximum(g2, 1e-12)), (r - 1.0) * min(ax))


def sd_capsule(y, x, z, p0, p1, radius):
    """Signed distance to a tube of the given radius around the segment p0 - p1 (round ends)."""
    d = [p1[i] - p0[i] for i in range(3)]
    t = ((y - p0[0]) * d[0] + (x - p0[1]) * d[1] + (z - p0[2]) * d[2]) * (1.0 / (d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
    t = np.clip(t, 0.0, 1.0)
    return np.sqrt((y - p0[0] - t * d[0]) ** 2 + (x - p0[1] - t * d[1]) ** 2 + (z - p0[2] - t * d[2]) ** 2) - radius


def sd_cylinder(y, x, z, c, axis, radius, half_length=None):
    """Signed distance to a cylinder about the line through ``c`` along the unit vector ``axis``; with
    ``half_length`` it has flat ends (the larger of the radial and the axial distance: exact inside, a lower bound
    at the outer rim)."""
    d = [y - c[0], x - c[1], z - c[2]]
    t = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2]
    rad = np.sqrt(np.maximum((d[0] - t * axis[0]) ** 2 + (d[1] - t * axis[1]) ** 2 + (d[2] - t * axis[2]) ** 2, 0.0)) - radius
    return rad if half_length is None else np.maximum(rad, np.abs(t) - half_length)


def sd_halfspace(y, x, z, point, normal):
    """Signed distance to the half-space behind the plane through ``point``: negative where ``normal`` points away from."""
    return (y - point[0]) * normal[0] + (x - point[1]) * normal[1] + (z - point[2]) * normal[2]


def sd_slit(da, ds):
    """Signed distance to the corner region {da < 0 and ds < 0} (a slit of a half-width and a depth)."""
    return np.hypot(np.clip(da, 0, None), np.clip(ds, 0, None)) + np.minimum(np.maximum(da, ds), 0)


def _dist(sh, y, x, z):
    k = sh["kind"]
    if k == "ell":
        return sd_ellipsoid(y, x, z, sh["c"], sh["ax"], sh.get("rot"))
    if k == "ball":
        return sd_sphere(y, x, z, sh["c"], sh["radius"])
    if k == "tube":
        return sd_capsule(y, x, z, sh["p0"], sh["p1"], sh["radius"])
    if k == "cyl":
        return sd_cylinder(y, x, z, sh["c"], sh["axis"], sh["radius"], sh.get("half_length"))
    raise ValueError(f"unknown shape {k!r}")


def _f(v):
    return float(v)


def _t(v):
    return tuple(float(a) for a in v)


def _shape(kind, **kw):
    """A shape record with plain Python numbers and its axis-aligned bounding box (``lo``, ``hi``)."""
    sh = dict(kind=kind)
    for k, v in kw.items():
        if k == "rot":
            sh[k] = None if v is None else tuple(_t(r) for r in np.asarray(v, float))
        elif k in ("c", "ax", "p0", "p1", "axis"):
            sh[k] = _t(v)
        elif k in ("radius", "half_length"):
            sh[k] = None if v is None else _f(v)
        else:
            sh[k] = v
    if kind == "ell":
        m = np.eye(3) if sh.get("rot") is None else np.asarray(sh["rot"])
        half = np.sqrt(((m.T * np.asarray(sh["ax"])) ** 2).sum(1))      # half-extent of the rotated ellipsoid along each patient axis
        lo, hi = np.asarray(sh["c"]) - half, np.asarray(sh["c"]) + half
    elif kind == "ball":
        lo, hi = np.asarray(sh["c"]) - sh["radius"], np.asarray(sh["c"]) + sh["radius"]
    elif kind == "tube":
        lo = np.minimum(sh["p0"], sh["p1"]) - sh["radius"]
        hi = np.maximum(sh["p0"], sh["p1"]) + sh["radius"]
    else:                                                               # cyl
        a, hl = np.asarray(sh["axis"]), sh.get("half_length")
        if hl is None:
            raise ValueError("a placed cylinder needs half_length_mm")
        half = hl * np.abs(a) + sh["radius"] * np.sqrt(np.maximum(1.0 - a * a, 0.0))
        lo, hi = np.asarray(sh["c"]) - half, np.asarray(sh["c"]) + half
    sh["lo"], sh["hi"] = _t(lo), _t(hi)
    return sh


# ---------------------------------------------------------------------------------------------- smooth fields
class Field:
    """A smooth random function of position: a sum of plane cosines with RMS amplitude ``amp`` (the 3D form of
    ``anatomy.smooth_field``). ``bound``: the largest value it can take anywhere."""

    __slots__ = ("k", "ph", "a", "bound")

    def __init__(self, rng, n_modes, wl_lo, wl_hi, amp):
        wl = np.exp(rng.uniform(np.log(wl_lo), np.log(wl_hi), n_modes))
        d = rng.normal(0, 1, (n_modes, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.k = 2 * np.pi * d / wl[:, None]
        self.ph = rng.uniform(0, 2 * np.pi, n_modes)
        a = rng.normal(0, 1, n_modes)
        self.a = a * amp * np.sqrt(2.0) / np.sqrt(np.sum(a * a))
        self.bound = float(np.sum(np.abs(self.a)))

    def at(self, y, x, z, grad=False):
        """The field at points (arrays of one shape; the dtype is kept); with ``grad`` also |gradient|."""
        out = np.zeros(y.shape, y.dtype)
        g = [np.zeros(y.shape, y.dtype) for _ in range(3)] if grad else None
        for (ky, kx, kz), ph, a in zip(self.k.tolist(), self.ph.tolist(), self.a.tolist()):
            arg = ky * y + kx * x + kz * z + ph
            out += a * np.cos(arg)
            if grad:
                s = a * np.sin(arg)
                g[0] -= ky * s
                g[1] -= kx * s
                g[2] -= kz * s
        return (out, np.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])) if grad else out


# ---------------------------------------------------------------------------------------------- where a recipe is evaluated
def _rect(mask):
    """Bounding rectangle (two slices) of the True entries of a 2D mask, or None."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if rows.size == 0:
        return None
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def _join(outer, inner):
    """The rectangle ``inner`` (given inside ``outer``) in the coordinates ``outer`` is given in."""
    if outer is None:
        return inner
    return tuple(slice(o.start + i.start, o.start + i.stop) for o, i in zip(outer, inner))


def _hull(a, b):
    if a is None:
        return b
    return tuple(slice(min(p.start, q.start), max(p.stop, q.stop)) for p, q in zip(a, b))


class _Points:
    """Points at which a recipe is evaluated: three 2D arrays of one shape and dtype (object coordinates), the soft
    indicator of this evaluation, and two services a recipe may use: ``smooth`` (a smooth field at the points of a
    rectangle) and ``near`` (the rectangle a bounding box can reach). Here: any points, so every shape is
    evaluated everywhere and every field directly."""

    def __init__(self, y, x, z, edge):
        self.y, self.x, self.z = y, x, z
        self.shape, self.dt = y.shape, y.dtype
        self.edge = float(edge)
        # beyond this distance a soft edge is exactly 0 or 1 in this dtype (tanh saturates at 9.1 / 19.1)
        self.tail = (20.0 if self.dt == np.float64 else 10.0) * self.edge
        self._full = (slice(0, self.shape[0]), slice(0, self.shape[1]))

    def sig(self, d, e=None):
        """Soft indicator of d > 0: the rule of ``anatomy.sig``."""
        return 0.5 * (1.0 + np.tanh(d / (self.edge if e is None else e)))

    def smooth(self, f, sub, grad=False):
        return f.at(self.y[sub], self.x[sub], self.z[sub], grad=grad)

    def near(self, lo, hi, pad=0.0, sub=None):
        """The rectangle (slices inside ``sub``, or in the whole array) that can lie within ``pad`` + the soft tail
        of the box lo .. hi, or None when no point can. Here: everything."""
        s = self._full if sub is None else sub
        return (slice(0, s[0].stop - s[0].start), slice(0, s[1].stop - s[1].start))


class _Plane(_Points):
    """The points origin + v row + u col of a plane (v, u: ascending 1D positions). Smooth fields are matrix
    products of 1D vectors; ``near`` projects the corners of a box on the plane."""

    def __init__(self, origin, row, col, normal, v, u, edge, dt):
        self.o, self.r, self.c, self.n = (np.asarray(a, float) for a in (origin, row, col, normal))
        self.v, self.u = np.asarray(v, float), np.asarray(u, float)
        y, x, z = [(self.o[i] + self.v[:, None] * self.r[i] + self.u[None, :] * self.c[i]).astype(dt) for i in range(3)]
        super().__init__(y, x, z, edge)

    def smooth(self, f, sub, grad=False):
        """sum_m a_m cos(k_m . p + ph_m) at p = o + v r + u c is Re(sum_m [A_m e^{i b_m v}] [e^{i c_m u}]): a product of a
        [nv, 2 modes] and a [2 modes, nu] REAL matrix (real and imaginary parts side by side; measured three times
        faster than the complex product)."""
        v, u, k = self.v[sub[0]], self.u[sub[1]], f.k
        ev = np.exp(1j * np.outer(v, k @ self.r))                                  # [nv, modes]
        eu = np.exp(1j * np.outer(k @ self.c, u))                                  # [modes, nu]
        right = np.concatenate([eu.real, eu.imag], 0).astype(self.dt)
        amp = f.a * np.exp(1j * (k @ self.o + f.ph))                               # [modes]

        def product(left):                                                         # Re(left @ eu)
            return np.concatenate([left.real, -left.imag], 1).astype(self.dt) @ right

        val = product(ev * amp)
        if not grad:
            return val
        g2 = np.zeros(val.shape, self.dt)                                          # d/dr of a cos(arg) = Re(i k a e^{i arg})
        for j in range(3):
            gj = product(ev * (1j * k[:, j] * amp))
            g2 += gj * gj
        return val, np.sqrt(g2)

    def near(self, lo, hi, pad=0.0, sub=None):
        m = pad + self.tail
        lo, hi = np.asarray(lo, float) - m, np.asarray(hi, float) + m
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]) - self.o
        zeta = corners @ self.n
        if zeta.min() > 0 or zeta.max() < 0:
            return None
        pv, pu = corners @ self.r, corners @ self.c
        s = self._full if sub is None else sub
        out = []
        for p, axis, sl in ((pv, self.v, s[0]), (pu, self.u, s[1])):
            a = max(int(np.searchsorted(axis, p.min(), "left")), sl.start)
            b = min(int(np.searchsorted(axis, p.max(), "right")), sl.stop)
            if b <= a:
                return None
            out.append(slice(a - sl.start, b - sl.start) if sub is not None else slice(a, b))
        return tuple(out)


def _cover(P, sub, shapes, y, x, z, pad=0.0):
    """Soft union of shapes on the arrays y, x, z (the rectangle ``sub`` of the evaluation): the largest soft
    indicator, each shape evaluated only where its box can reach. Returns (weight, rectangle touched or None)."""
    w, box = np.zeros(y.shape, y.dtype), None
    for sh in shapes:
        sl = P.near(sh["lo"], sh["hi"], pad, sub)
        if sl is None:
            continue
        w[sl] = np.maximum(w[sl], P.sig(-_dist(sh, y[sl], x[sl], z[sl])))
        box = _hull(box, sl)
    return w, box


def _place(P, out, R, y, x, z, sub, pad=0.0):
    """Placed inserts (``spec["volume"]["inserts"]``): each sits on top of everything inside the object."""
    for k, sh in enumerate(R.get("inserts") or ()):
        sl = P.near(sh["lo"], sh["hi"], pad, sub)
        if sl is None:
            continue
        m = sh["weight"] * P.sig(-_dist(sh, y[sl], x[sl], z[sl]))
        view = out[(slice(None),) + _join(sub, sl)]
        total = view.sum(0)
        view *= (1.0 - m)
        view[R["index"][f"placed{k}"]] = m * total


# ---------------------------------------------------------------------------------------------- the head
HEAD_NAMES = ("skin", "scalp_fat", "skull", "diploe", "csf", "wm", "gm", "deep_gm", "muscle", "blood")


def _head_recipe(seed, sp, vs, W):
    """The recipe of a head from the secret seed: numbers only (a few kB), no arrays on a grid."""
    g = W.rng_for(seed, "geometry")
    U = lambda a, b: float(g.uniform(a, b))                             # noqa: E731
    R = dict(kind="head3d")
    R["warp"] = [Field(g, 5, 90, 300, 1.2) for _ in range(3)]
    R["wmax"] = float(np.sqrt(sum(f.bound ** 2 for f in R["warp"])))    # the farthest the warp moves a point
    sc = U(0.95, 1.05)
    ax, ay, az = sc * U(72, 78), sc * U(90, 97), sc * U(84, 92)
    R.update(ax=ax, ay=ay, az=az, sc=sc)
    R["neck"] = U(0.58, 0.68)                                           # cross-section of the neck relative to the head's largest
    R["az_low"] = U(95, 115)                                            # how fast the head narrows below its centre
    R["gmax"] = float(np.sqrt(1.0 - R["neck"] ** 2))
    zc = R["zc"] = U(10, 20)
    R["egg"] = U(0.07, 0.13)                                            # wider posteriorly
    at = R["atrophy"] = U(0, 1)
    R["b_skin"], R["fat"], R["fat_field"] = 1.6, U(3.0, 6.0), Field(g, 4, 60, 250, 1.0)
    R["bone"], R["bone_field"] = U(6.0, 9.0), Field(g, 4, 40, 200, 1.0)
    R["cav_low"] = U(50, 60)                                            # the cranial cavity below its centre
    m = R["b_skin"] + R["fat"] + R["bone"]
    R["cav"] = (ay - m, ax - m, az - m)                                 # semi-axes of the cranial cavity (y, x, z above the centre)
    R["b_csf"] = U(1.6, 2.6) + 1.2 * at
    R["sulc"], R["sulc_half"] = Field(g, 10, 13, 28, 1.0), 0.6 * (0.55 + 0.9 * at)
    R["sulc_depth"], R["sulc_depth_field"] = U(9, 14), Field(g, 4, 30, 120, 1.0)
    R["t_gm"] = U(2.6, 3.4) * (1 - 0.2 * at)
    R["fissure"] = 0.7 + 0.6 * at
    v = U(0.8, 1.15) * (1 + 0.6 * at)
    rot3 = lambda deg: rotation(*g.uniform(-deg, deg, 3))               # noqa: E731
    vent, deep = [], []
    for side, sgn in (("R", -1.0), ("L", 1.0)):
        j = [float(a) for a in g.uniform(0.92, 1.08, 8)]
        vent.append(_shape("ell", c=(-26 * sc, sgn * (4.0 * v + 1.3) * j[0], zc - 6), ax=(12.5 * j[1] * sc, 4.0 * v, 9.0 * j[6]), rot=rot3(18), name=f"frontal_horn_{side}"))
        vent.append(_shape("ell", c=(8 * sc, sgn * 12 * sc * j[2], zc + 4), ax=(26 * j[3] * sc, 4.6 * v, 6.5 * j[7]), rot=rot3(15), name=f"ventricle_body_{side}"))
        vent.append(_shape("ell", c=(34 * sc, sgn * 24 * sc * j[2], zc - 8), ax=(11 * j[3] * sc, 5.0 * v, 13.0), rot=rot3(20), name=f"atrium_{side}"))
        deep.append(_shape("ell", c=(-24 * sc, sgn * (8.0 * v + 7.5), zc - 8), ax=(9.5 * j[4], 4.8, 9.0), rot=rot3(12), name=f"caudate_{side}"))
        deep.append(_shape("ell", c=(-6 * sc, sgn * 27 * sc, zc - 12), ax=(16.5 * j[5], 7.0, 11.0), rot=rot3(12), name=f"lentiform_{side}"))
        deep.append(_shape("ell", c=(9 * sc, sgn * 11.5 * sc, zc - 10), ax=(13.0, 8.5, 9.5), rot=rot3(10), name=f"thalamus_{side}"))
    vent.append(_shape("ell", c=(7 * sc, 0.0, zc - 14), ax=(11.0, 1.0 + 0.9 * v, 9.0), name="third_ventricle"))
    R["vent"], R["deep"] = vent, deep
    R["callosum"] = ((2.0, 0.0, zc - 4), (U(36, 42), 60.0, U(20, 26)))  # no interhemispheric fissure inside this ellipsoid
    # air: frontal sinus, sphenoid sinus, two mastoid regions, the airway (a tube that leaves the imaged volume downwards)
    R["air"] = [
        _shape("ell", c=(-(ay - U(14, 20)), U(-6, 6), zc - U(30, 38)), ax=(U(8, 12), U(12, 18), U(9, 14)), name="frontal_sinus"),
        _shape("ell", c=(-U(8, 16), 0.0, zc - R["cav_low"] - U(13, 18)), ax=(U(10, 14), U(10, 14), U(8, 11)), name="sphenoid_sinus"),
        _shape("ell", c=(U(18, 28), -(ax - 16), zc - U(46, 54)), ax=(U(9, 13), U(7, 10), U(10, 14)), name="mastoid_R"),
        _shape("ell", c=(U(18, 28), (ax - 16), zc - U(46, 54)), ax=(U(9, 13), U(7, 10), U(10, 14)), name="mastoid_L"),
        _shape("tube", p0=(-(ay - 38), 0.0, zc - 70.0), p1=(-U(15, 30), 0.0, zc - 400.0), radius=U(8, 11), name="airway")]
    # the eyes: balls of vitreous (drawn with the CSF row) in the face below the floor of the cranial cavity, just behind
    # the skin and the fat of the anterior surface at their height
    R["eyes"] = []
    for side, sgn in (("R", -1.0), ("L", 1.0)):
        xe, ze, re = sgn * U(29, 33), zc - U(50, 56), U(11.0, 12.5)
        gz = R["gmax"] * np.tanh((zc - ze) / (R["az_low"] * R["gmax"]))
        front = -ay * float(np.sqrt(max(1.0 - gz * gz - (xe / (ax * (1 - 0.75 * R["egg"]))) ** 2, 0.05)))       # the skin surface there
        R["eyes"].append(_shape("ball", c=(front + R["b_skin"] + 1.45 * R["fat"] + re + U(0.5, 2.5), xe, ze), radius=re, name=f"eye_{side}"))
    R["box_lo"], R["box_hi"] = (-ay, -ax * (1 + R["egg"]), -1e6), (ay, ax * (1 + R["egg"]), zc + az)
    R["vessels"], R["lesions"], R["inserts"] = [], [], []
    names = list(HEAD_NAMES) + (["lesion_ms"] if sp["lesion"] == "ms" else [])
    R["names"] = names
    return R


def _head_vessels(R, seed, n, W):
    """Straight vessels from the neck into the brain, each with a hidden direction and speed (plug flow along the
    tube; the sign says which way the blood runs along ``direction``). They are straight in the object: the warp
    that shapes the head does not bend them. A sub-stream of its own."""
    g = W.rng_for(seed, "geometry", 2)
    out = []
    for i in range(int(n)):
        p0 = np.array([g.uniform(-0.25, 0.25) * R["ay"], g.uniform(-0.35, 0.35) * R["ax"], R["zc"] - 150.0])
        d = np.array([g.uniform(-0.18, 0.18), g.uniform(-0.18, 0.18), 1.0])
        d /= np.linalg.norm(d)
        length, radius = float(g.uniform(190, 240)), float(g.uniform(1.6, 2.6))
        speed = float(g.uniform(40, 250)) * (1.0 if g.uniform() < 0.5 else -1.0)
        out.append(dict(_shape("tube", p0=p0, p1=p0 + d * length, radius=radius, name=f"vessel{i}"), direction=_t(d), speed_mm_s=speed))
    return out


def _head_lesions(vol, R, seed, W):
    """4 to 8 small lesions seeded at points that the recipe itself reports as white matter, and the hidden sign
    and size of their susceptibility. A sub-stream of its own: the head without lesions is the same head."""
    g = W.rng_for(seed, "geometry", 1)
    zc = R["zc"]
    cand = np.stack([g.uniform(-60, 60, 6000), g.uniform(-55, 55, 6000), g.uniform(zc - 30, zc + 50, 6000)], 1)
    wm = vol._object_points(cand[:, 0], cand[:, 1], cand[:, 2])[R["index"]["wm"]]
    ok = np.flatnonzero(wm > 0.98)
    n = int(g.integers(4, 9))
    chi = float(g.uniform(*LESION_CHI_PPM)) * (1.0 if g.uniform() < 0.5 else -1.0)
    seeds = vol._warped(cand[ok[:n]])                                   # shapes live behind the warp: the lesion sits ON the tested point
    out = []
    for k in range(len(seeds)):
        out.append(_shape("ell", c=seeds[k], ax=(g.uniform(1.8, 5.0), g.uniform(1.5, 3.5), g.uniform(1.8, 4.5)), rot=rotation(*g.uniform(-90, 90, 3)),
                          name=f"lesion{k}"))
    return out, chi


def _head_classes(R, P):
    """Class weights [matter, n1, n2] of the head at the points ``P`` (object coordinates). A partition by
    construction: skin and fat by depth, then everything deeper is either cranial cavity or shell; the shell is
    bone, marrow, muscle, eye or air; the cavity is CSF or brain; the brain is split multiplicatively."""
    o, sig, tail = R["index"], P.sig, P.tail
    out = np.zeros((len(R["matter"]),) + P.shape, P.dt)
    S0 = P.near(R["box_lo"], R["box_hi"], R["wmax"])
    if S0 is None:
        return out
    pad = R["wmax"]                                                     # shapes live in warped coordinates
    y = P.y[S0] + P.smooth(R["warp"][0], S0)
    x = P.x[S0] + P.smooth(R["warp"][1], S0)
    z = P.z[S0] + P.smooth(R["warp"][2], S0)
    ax, ay, az, egg = R["ax"], R["ay"], R["az"], R["egg"]
    # ---- outer surface: an egg above its centre; below, it narrows smoothly into a neck that goes on downwards
    zz = z - R["zc"]
    up = zz >= 0
    tl = np.tanh(zz * (1.0 / (R["az_low"] * R["gmax"])))
    gz = np.where(up, zz * (1.0 / az), R["gmax"] * tl)                  # the term of the third axis in the implicit function
    dgz = np.where(up, 1.0 / az, (1.0 - tl * tl) * (1.0 / R["az_low"]))
    axe = ax * (1.0 + (egg / ay) * y)
    X, Y = x / axe, y * (1.0 / ay)
    r = np.sqrt(X * X + Y * Y + gz * gz)
    gy = Y * (1.0 / ay) - X * X / axe * (ax * egg / ay)
    grad = np.sqrt((X / axe) ** 2 + gy * gy + (gz * dgz) ** 2)
    tdep = np.where(r > 0.05, (1.0 - r) * r / np.maximum(grad, 1e-9), ax)      # depth below the skin surface, mm
    h = _rect(tdep > -tail)
    if h is None:
        return out
    H = _join(S0, h)
    y, x, z, zz, up, tdep = y[h], x[h], z[h], zz[h], up[h], tdep[h]
    fatb = R["b_skin"] + R["fat"] * (1.0 + 0.25 * P.smooth(R["fat_field"], H))
    s0, s1, s2 = sig(tdep), sig(tdep - R["b_skin"]), sig(tdep - fatb)
    out[o["skin"]][H], out[o["scalp_fat"]][H] = s0 - s1, s1 - s2
    # ---- cranial cavity: an ellipsoid with a shorter lower half, never closer to the skin than skin + fat + bone
    bone = R["bone"] * (1.0 + 0.2 * P.smooth(R["bone_field"], H))
    cy, cx, cz = R["cav"]
    caz = np.where(up, 1.0 / cz, 1.0 / R["cav_low"])
    cxe = cx * (1.0 + (egg / ay) * y)
    Xc, Yc, Zc = x / cxe, y * (1.0 / cy), zz * caz
    rc = np.sqrt(Xc * Xc + Yc * Yc + Zc * Zc)
    gyc = Yc * (1.0 / cy) - Xc * Xc / cxe * (cx * egg / ay)
    gc = np.sqrt((Xc / cxe) ** 2 + gyc * gyc + (Zc * caz) ** 2)
    sc = np.where(rc > 0.05, (1.0 - rc) * rc / np.maximum(gc, 1e-9), min(cx, R["cav_low"]))     # depth inside the cavity
    sc = np.minimum(sc, tdep - fatb - bone)
    cav = s2 * sig(sc)
    shell = s2 - cav
    air, _ = _cover(P, H, R["air"], y, x, z, pad)
    shell = shell * (1.0 - air)
    bw = shell * sig(sc + bone)                                         # bone: within its thickness of the cavity
    dip = bw * sig(-sc - 1.5) * sig(sc + bone - 1.7)                    # fatty marrow between the two tables
    rest = shell - bw                                                   # face and neck
    eye, _ = _cover(P, H, R["eyes"], y, x, z, pad)
    out[o["skull"]][H], out[o["diploe"]][H], out[o["muscle"]][H] = bw - dip, dip, rest * (1.0 - eye)
    s = sc - R["b_csf"]                                                 # depth below the brain surface
    brain = cav * sig(s)
    csf = rest * eye + (cav - brain)
    # ---- brain: only the rectangle that holds the points inside the pial surface
    b = _rect(s > -tail)
    if b is not None:
        HB = _join(H, b)
        yb, xb, zb, sb, brain_b = y[b], x[b], z[b], s[b], brain[b]
        fv, fg = P.smooth(R["sulc"], HB, grad=True)
        depth = R["sulc_depth"] * (1.0 + 0.3 * P.smooth(R["sulc_depth_field"], HB))
        f_sulc = sd_slit(np.abs(fv) / np.maximum(fg, 1e-3) - R["sulc_half"], sb - depth)
        cc, ca = R["callosum"]
        f_fiss = sd_slit(np.abs(xb - np.sin(yb * (1.0 / 23.0))) - R["fissure"], -sd_ellipsoid(yb, xb, zb, cc, ca))
        dn = np.minimum(np.minimum(sb, f_sulc), f_fiss)                 # distance into tissue from the pial surface
        sdn, sgm = sig(dn), sig(dn - R["t_gm"])
        deep, _ = _cover(P, HB, R["deep"], yb, xb, zb, pad)
        deep = deep * sgm
        vent, _ = _cover(P, HB, R["vent"], yb, xb, zb, pad)
        vent = vent * sig(dn - 1.0)
        keep = (1.0 - deep) * (1.0 - vent)
        wm, gm, dg, cs = sgm * keep, (sdn - sgm) * keep, deep * (1.0 - vent), (1.0 - sdn) * keep + vent
        if R["lesions"]:
            les, box = _cover(P, HB, R["lesions"], yb, xb, zb, pad)
            if box is not None:
                out[o["lesion_ms"]][HB] = brain_b * les * (1.0 - cs)
                wm, gm, dg = wm * (1.0 - les), gm * (1.0 - les), dg * (1.0 - les)
        out[o["wm"]][HB], out[o["gm"]][HB], out[o["deep_gm"]][HB] = brain_b * wm, brain_b * gm, brain_b * dg
        csf[b] += brain_b * cs
    out[o["csf"]][H] = csf
    # ---- vessels: on top of everything inside the fat layer except air; straight tubes, so not behind the warp
    if R["vessels"]:
        bl, box = _cover(P, H, R["vessels"], P.y[H], P.x[H], P.z[H])
        if box is not None:
            HV, blv = _join(H, box), bl[box]
            for k in R["inner"]:
                out[k][HV] *= (1.0 - blv)
            out[o["blood"]][HV] = blv * (shell[box] + cav[box])
    if R["inserts"]:                                                    # placed where the caller says: not behind the warp
        _place(P, out, R, P.y[H], P.x[H], P.z[H], H)
    return out


def _head_regions(vol, R, seed, W):
    """Scored regions of the head: balls inside the deep nuclei, the ventricles, white matter, the lesions and the
    neck muscle, and a cylinder along each vessel. Ids are a random permutation; ``name`` (hidden) says what it is."""
    g = W.rng_for(seed, "geometry", 3)
    zc, cand = R["zc"], []
    def here(c):                                                         # a point of a shape -> object coordinates
        return _t(vol._unwarped(np.asarray(c, float)[None])[0])

    for sh in R["deep"]:
        cand.append(dict(name=sh["name"], cls="deep_gm", shape="sphere", c=here(sh["c"]), radius=0.45 * min(sh["ax"])))
    for sh in R["vent"]:
        if sh["name"].startswith("ventricle_body"):
            cand.append(dict(name=sh["name"], cls="csf", shape="sphere", c=here(sh["c"]), radius=0.5 * min(sh["ax"])))
    pts = np.stack([g.uniform(-55, 55, 400), g.uniform(-50, 50, 400), g.uniform(zc - 15, zc + 45, 400)], 1)
    wm = vol._object_points(pts[:, 0], pts[:, 1], pts[:, 2])[R["index"]["wm"]]
    for k, i in enumerate(np.flatnonzero(wm > 0.999)[:8]):
        cand.append(dict(name=f"white_matter{k}", cls="wm", shape="sphere", c=_t(pts[i]), radius=3.0, limit=2))
    cand.append(dict(name="neck_muscle", cls="muscle", shape="sphere", c=(0.5 * R["neck"] * R["ay"], 0.0, zc - 115.0), radius=5.0))
    for sh in R["lesions"]:
        cand.append(dict(name=sh["name"], cls="lesion_ms", shape="sphere", c=here(sh["c"]), radius=0.35 * min(sh["ax"])))
    for sh in R["vessels"]:
        p0, d = np.asarray(sh["p0"]), np.asarray(sh["direction"])
        c = p0 + d * ((zc + 10.0 - p0[2]) / d[2])                       # where the vessel crosses the level of the brain's centre
        cand.append(dict(name=sh["name"], cls="blood", shape="cylinder", c=_t(c), radius=0.5 * sh["radius"], axis=sh["direction"], half_length=12.0))
    return _regions(vol, cand, g, prefix="V")


# ---------------------------------------------------------------------------------------------- the test object
def _qa_layout(g, sp):
    """The geometry draws of ``anatomy.qa`` in its order (body, ring, centre vial, vials, ids, oil): the vial
    section of ``qa3d`` is the 2D ``qa`` phantom of the same seed. ``_qa_recipe`` checks this against what
    ``anatomy.qa`` itself returns for the same stream."""
    by, bx = (float(v) for v in g.uniform(-sp["offset_mm"], sp["offset_mm"], 2))
    ba, bb, bth = float(g.uniform(*sp["body_radius_mm"])), float(g.uniform(*sp["body_radius_mm"])), float(g.uniform(0, 180))
    if sp.get("body_aspect") is not None:
        bb = ba / float(g.uniform(*sp["body_aspect"]))
    if sp["body_rotation_deg"] is not None:
        bth = float(sp["body_rotation_deg"][int(bth // (180.0 / len(sp["body_rotation_deg"])))])
    n_ring = int(g.integers(sp["n_ring"][0], sp["n_ring"][1] + 1))
    ring, start = float(g.uniform(*sp["ring_radius_mm"])), float(g.uniform(0, 2 * np.pi))
    ang = start + 2 * np.pi * np.arange(n_ring) / n_ring + g.uniform(-0.06, 0.06, n_ring)
    centres = [(by + ring * np.sin(a), bx + ring * np.cos(a)) for a in ang]
    if g.uniform() < sp["p_centre"]:
        centres.append((by + float(g.uniform(-2.4, 2.4)), bx + float(g.uniform(-2.4, 2.4))))
    vials = []
    for cy, cx in centres:
        r, e = g.uniform(*sp["vial_radius_mm"]), g.uniform(-0.05, 0.05)
        vials.append(dict(y=float(cy), x=float(cx), a=float(r * (1 + e)), b=float(r * (1 - e)), rot=float(g.uniform(0, 180))))
    return dict(y=by, x=bx, a=ba, b=bb, rot=bth), vials


def _qa_recipe(seed, sp, vs, W):
    """The recipe of the cylinder test object. Along its axis (z): a vial section about the isocentre (the 2D
    ``qa`` layout, uniform along z), and beyond it on one side a ball and a pair of wedges, on the other side a
    block of plastic with two crossed ramps. Which side is which, every position, size and angle are drawn."""
    b0_t = float(sp["b0_t"])
    qsp = dict(anatomy.QA_SPEC, **(sp["qa"] or {}))
    flat = anatomy.qa(W.rng_for(seed, "geometry"), b0_t=b0_t, spec=sp["qa"], tissue_rng=W.rng_for(seed, "tissue"), layout_only=True)
    body, vials = _qa_layout(W.rng_for(seed, "geometry"), qsp)
    seen = [(r["cls"], r["y_mm"], r["x_mm"]) for r in flat["regions"]]
    mine = sorted((f"vial{i}", round(v["y"], 2), round(v["x"], 2)) for i, v in enumerate(vials))
    if body != flat["body"] or sorted(seen) != mine:
        raise RuntimeError("volume._qa_layout no longer draws what anatomy.qa draws: the vial section of qa3d must follow the 2D rule")
    g = W.rng_for(seed, "geometry", 4)
    U = lambda a, b: float(g.uniform(a, b))                             # noqa: E731
    R = dict(kind="qa3d", body=body, vials=vials, flat=flat)
    R["cz"] = U(-qsp["offset_mm"], qsp["offset_mm"])                    # the centre of the object along z
    R["half_length"] = U(70.0, 78.0)
    R["vial_half"] = U(18.0, 22.0)                                      # the vial section: |z - cz| below this
    side = 1.0 if g.uniform() < 0.5 else -1.0                           # the side of the ball and the wedges; the ramp block is on the other
    wv, cz, by, bx = R["vial_half"], R["cz"], body["y"], body["x"]
    phi = U(0, 2 * np.pi)
    rs = U(9.0, 14.0)
    rho = U(28.0, 40.0)
    R["sphere"] = dict(c=(by + rho * np.sin(phi), bx + rho * np.cos(phi), cz + side * (wv + 6.0 + rs + U(0.0, 4.0))), radius=rs)
    phw, rhw = phi + np.pi + U(-0.4, 0.4), U(28.0, 40.0)
    R["wedges"] = dict(c=(by + rhw * np.sin(phw), bx + rhw * np.cos(phw), cz + side * (wv + 6.0 + 10.0 + U(0.0, 4.0))), rot_deg=U(0.0, 180.0),
                       angle_deg=U(35.0, 55.0), half=(18.0, 5.0, 10.0), lane_mm=6.0)
    slope = U(0.09, 0.115)                                              # rise of a ramp per mm along it (the ACR ramps: 1 in 10)
    reach = 60.0                                                        # half-length of a ramp
    hw = reach * slope + 3.0
    R["block"] = dict(c=(by + U(-6.0, 6.0), bx + U(-6.0, 6.0), cz - side * (wv + 8.0 + hw + U(0.0, 4.0))), rot_deg=U(0.0, 180.0),
                      half=(9.5, reach + 4.0, hw), slope=slope, reach=reach, sheet_mm=1.0, band_mm=5.0, band_centre_mm=4.0)
    R["side"] = side
    R["features"] = bool(vs["features"])
    R["names"] = list(flat["names"]) + (["sphere", "wedge", "slot"] if R["features"] else [])
    R["box_lo"] = (by - max(body["a"], body["b"]), bx - max(body["a"], body["b"]), cz - R["half_length"])
    R["box_hi"] = (by + max(body["a"], body["b"]), bx + max(body["a"], body["b"]), cz + R["half_length"])
    R["flat_inserts"] = list(qsp.get("inserts") or ())
    R["inserts"] = []
    return R, qsp


def _turn(y, x, c, rot_deg):
    """In-plane coordinates (v, u) of (y, x) in a frame centred on ``c`` and turned by ``rot_deg``."""
    co, si = float(np.cos(np.deg2rad(rot_deg))), float(np.sin(np.deg2rad(rot_deg)))
    dy, dx = y - c[0], x - c[1]
    return dy * co + dx * si, dx * co - dy * si


def _qa_classes(R, P):
    """Class weights of the test object. Inside the vial section this is line for line the rule of ``anatomy.qa``
    (the same ``ell``, the same soft indicator, the same order of sums), times soft slabs along z that are exactly
    1 there; the 3D inserts sit on top of the gel outside that section."""
    o, sig = R["index"], P.sig
    out = np.zeros((len(R["matter"]),) + P.shape, P.dt)
    body, cz = R["body"], R["cz"]
    ratio = max(body["a"], body["b"]) / min(body["a"], body["b"])       # ``ell`` scales its distance by the smaller semi-axis
    S0 = P.near(R["box_lo"], R["box_hi"], P.tail * (ratio - 1.0) + 1e-6)
    if S0 is None:
        return out
    Y, X, Z = P.y[S0], P.x[S0], P.z[S0]
    w = Z - cz
    cyl = sig(-anatomy.ell(Y, X, body["y"], body["x"], body["a"], body["b"], body["rot"])) * sig(R["half_length"] - np.abs(w))
    inside, wv, any_vial = np.zeros(Y.shape, P.dt), R["vial_half"], False
    flat = [(f"vial{i}", v["y"], v["x"], v["a"], v["b"], v["rot"], 1.0) for i, v in enumerate(R["vials"])]
    flat += [(f"insert{k}", s["y_mm"], s["x_mm"], s["radius_mm"], s["radius_mm"], 0.0, float(s.get("weight", 1.0))) for k, s in enumerate(R["flat_inserts"])]
    for name, vy, vx, va, vb, rot, weight in flat:
        any_vial = True
        big = max(va, vb)
        sl = P.near((vy - big, vx - big, cz - wv), (vy + big, vx + big, cz + wv), P.tail * (big / min(va, vb) - 1.0) + 1e-6, S0)
        if sl is None:
            continue
        m = sig(-anatomy.ell(Y[sl], X[sl], vy, vx, va, vb, rot)) * sig(wv - np.abs(w[sl]))
        if weight != 1.0:
            m = weight * m
        out[o[name]][S0][sl] = m
        inside[sl] += m
    out[o["body"]][S0] = cyl * (1 - np.clip(inside, 0, 1)) if any_vial else cyl * (1 - 0.0)
    if R["features"]:
        def put(name, m, sl):                                           # later shapes sit on top of the gel and of each other
            view = out[(slice(None),) + _join(S0, sl)]
            view *= (1.0 - m)
            view[o[name]] += m * cyl[sl]

        sp_ = R["sphere"]
        r = sp_["radius"]
        sl = P.near(tuple(c - r for c in sp_["c"]), tuple(c + r for c in sp_["c"]), 0.0, S0)
        if sl is not None:
            put("sphere", sig(-sd_sphere(Y[sl], X[sl], Z[sl], sp_["c"], r)), sl)
        wd = R["wedges"]
        hv, hu, hz = wd["half"]
        far = float(np.hypot(hv, hu + wd["lane_mm"]))
        sl = P.near((wd["c"][0] - far, wd["c"][1] - far, wd["c"][2] - hz), (wd["c"][0] + far, wd["c"][1] + far, wd["c"][2] + hz), 0.0, S0)
        if sl is not None:
            v, u = _turn(Y[sl], X[sl], wd["c"], wd["rot_deg"])
            t = Z[sl] - wd["c"][2]
            sb, cb = float(np.sin(np.deg2rad(wd["angle_deg"]))), float(np.cos(np.deg2rad(wd["angle_deg"])))
            box = sig(hv - np.abs(v)) * sig(hz - np.abs(t))
            m = box * (sig(hu - np.abs(u + wd["lane_mm"])) * sig(v * sb - t * cb) + sig(hu - np.abs(u - wd["lane_mm"])) * sig(v * sb + t * cb))
            put("wedge", m, sl)
        bk = R["block"]
        hv, hu, hz = bk["half"]
        far = float(np.hypot(hv, hu))
        sl = P.near((bk["c"][0] - far, bk["c"][1] - far, bk["c"][2] - hz), (bk["c"][0] + far, bk["c"][1] + far, bk["c"][2] + hz), 0.0, S0)
        if sl is not None:
            v, u = _turn(Y[sl], X[sl], bk["c"], bk["rot_deg"])
            t = Z[sl] - bk["c"][2]
            put("plastic", sig(hv - np.abs(v)) * sig(hu - np.abs(u)) * sig(hz - np.abs(t)), sl)
            ca = 1.0 / float(np.hypot(1.0, bk["slope"]))
            thin = THIN_EDGE * P.edge
            ramps = sig(bk["reach"] - np.abs(u)) * (
                sig(0.5 * bk["band_mm"] - np.abs(v + bk["band_centre_mm"])) * sig(0.5 * bk["sheet_mm"] - np.abs((t - bk["slope"] * u) * ca), thin)
                + sig(0.5 * bk["band_mm"] - np.abs(v - bk["band_centre_mm"])) * sig(0.5 * bk["sheet_mm"] - np.abs((t + bk["slope"] * u) * ca), thin))
            put("slot", ramps, sl)
    if R["inserts"]:
        _place(P, out, R, Y, X, Z, S0)
    return out


def _qa_tissues(R, qsp, seed, b0_t, W):
    """Tissue values: body and vials exactly as the 2D phantom draws them; the ball and the wedges like vials, the
    fluid of the ramps like the body gel, from a sub-stream of their own."""
    tis = {k: dict(v) for k, v in R["flat"]["tissues"].items()}
    if not R["features"]:
        return tis
    t = W.rng_for(seed, "tissue", 1)
    for name in ("sphere", "wedge"):
        t1 = anatomy._logu(t, *qsp["vial_t1_ms"])
        tis[name] = dict(pd=float(t.uniform(*qsp["vial_pd"])), t1_ms=t1, t2_ms=min(anatomy._logu(t, *qsp["vial_t2_ms"]), qsp["t2_over_t1_max"] * t1),
                         t2p_ms=anatomy._logu(t, *qsp["vial_t2p_ms"]), adc_um2_ms=float(t.uniform(*qsp["vial_adc_um2_ms"])), fat=False)
    tis["slot"] = dict(pd=float(t.uniform(0.85, 1.0)), t1_ms=float(t.uniform(700, 1300)), t2_ms=float(t.uniform(50, 110)),
                       t2p_ms=float(t.uniform(60, 200)), adc_um2_ms=float(t.uniform(0.8, 2.0)), fat=False)
    for name in ("sphere", "wedge", "slot"):
        tis[name]["t2s_ms"] = 1.0 / (1.0 / tis[name]["t2_ms"] + 1.0 / tis[name]["t2p_ms"])
    return tis


def _qa_regions(vol, R, qsp, seed, W):
    """The vial regions of the 2D phantom (same ids, now cylinders along the axis), then the ball and a wedge."""
    cz, wv = R["cz"], R["vial_half"]
    cand = []
    for r in R["flat"]["regions"]:
        cand.append(dict(id=r["id"], name=r["cls"], cls=r["cls"], shape="cylinder", c=(r["y_mm"], r["x_mm"], cz), radius=r["radius_mm"],
                         axis=(0.0, 0.0, 1.0), half_length=wv - 6.0, keep=True))
    if R["features"]:
        s = R["sphere"]
        cand.append(dict(name="sphere", cls="sphere", shape="sphere", c=s["c"], radius=0.6 * s["radius"]))
        wd = R["wedges"]
        co, si = np.cos(np.deg2rad(wd["rot_deg"])), np.sin(np.deg2rad(wd["rot_deg"]))
        v, u = 12.0, -wd["lane_mm"]                                     # well inside the first wedge, below its tilted face
        cand.append(dict(name="wedge", cls="wedge", shape="sphere", c=(wd["c"][0] + v * co - u * si, wd["c"][1] + v * si + u * co, wd["c"][2] - 5.0), radius=3.0))
    return _regions(vol, cand, W.rng_for(seed, "geometry", 5), prefix="R", first=len(R["flat"]["regions"]) + 1)


# ---------------------------------------------------------------------------------------------- regions
def _samples(shape, n=48):
    """Deterministic points inside the unit ball (sphere) or the unit cylinder (radius 1, |t| <= 1 along its axis)."""
    i = np.arange(n) + 0.5
    ga = np.pi * (3.0 - np.sqrt(5.0)) * i
    if shape == "sphere":
        ct = 1.0 - 2.0 * i / n
        st = np.sqrt(1.0 - ct * ct)
        rad = np.where(np.arange(n) % 2 == 0, 1.0, 0.6)
        return np.stack([rad * st * np.cos(ga), rad * st * np.sin(ga), rad * ct], 1)
    rad = np.sqrt(i / n)
    return np.stack([rad * np.cos(ga), rad * np.sin(ga), np.where(np.arange(n) % 3 == 0, -1.0, np.where(np.arange(n) % 3 == 1, 0.0, 1.0))], 1)


def _region_points(r, n=48):
    """Sample points (object coordinates [n, 3]) of a region record with centre ``c``."""
    c, s = np.asarray(r["c"], float), _samples(r["shape"], n)
    if r["shape"] == "sphere":
        return c + r["radius"] * s
    a = np.asarray(r["axis"], float)
    e1 = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return c + r["radius"] * (s[:, :1] * e1 + s[:, 1:2] * e2) + r["half_length"] * s[:, 2:] * a


def _regions(vol, cand, g, prefix, first=1):
    """Keep the candidates whose class fills them (``PURITY_MIN``, measured on sample points of the recipe), give
    the new ones ids in a random order, and express them in patient coordinates."""
    kept, limit = [], {}
    for r in cand:
        p = _region_points(r)
        purity = float(vol._object_points(p[:, 0], p[:, 1], p[:, 2])[vol.matter.index(r["cls"])].mean())
        group = r["name"].rstrip("0123456789")
        if not r.get("keep") and (purity < PURITY_MIN or limit.get(group, 0) >= r.get("limit", 99)):
            continue
        limit[group] = limit.get(group, 0) + 1
        kept.append(dict(r, purity=round(purity, 4)))
    new = [r for r in kept if "id" not in r]
    for k, i in enumerate(g.permutation(len(new))):
        new[int(i)]["id"] = f"{prefix}{first + k}"
    out = []
    for r in kept:
        c = vol.to_patient(r["c"])
        row = dict(id=r["id"], cls=r["cls"], name=r["name"], shape=r["shape"], y_mm=round(float(c[0]), 2), x_mm=round(float(c[1]), 2),
                   z_mm=round(float(c[2]), 2), radius_mm=round(float(r["radius"]), 2), purity=r["purity"])
        if r["shape"] == "cylinder":
            row.update(axis=tuple(round(float(v), 6) for v in vol.to_patient(r["axis"], direction=True)), half_length_mm=round(float(r["half_length"]), 2))
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------- placed inserts
def _insert_shapes(inserts, b0_t):
    """``spec["volume"]["inserts"]``: shapes placed by the caller in OBJECT coordinates, on top of everything inside
    the object (no draw is made for them). Each a dict: ``shape`` "sphere" or "cylinder"; ``y_mm, x_mm, z_mm``;
    ``radius_mm``; for a cylinder ``axis`` (y, x, z) and ``half_length_mm``; ``tissue`` ("oil": the fat row of the
    tissue table, a name of the tissue table, or a dict with pd, t1_ms, t2_ms, t2p_ms, adc_um2_ms and optionally
    fat); optionally ``weight`` (the share of the shape this material fills) and ``chi_ppm``. Classes ``placed0``,
    ``placed1``, ... after the others."""
    shapes, tissues, chi = [], {}, {}
    tab = tissue.table(b0_t)
    for k, s in enumerate(inserts or ()):
        c = (s["y_mm"], s["x_mm"], s["z_mm"])
        if s.get("shape", "sphere") == "sphere":
            sh = _shape("ball", c=c, radius=s["radius_mm"])
        elif s["shape"] == "cylinder":
            sh = _shape("cyl", c=c, axis=_unit(s.get("axis", (0.0, 0.0, 1.0))), radius=s["radius_mm"], half_length=s["half_length_mm"])
        else:
            raise ValueError("a placed insert is a 'sphere' or a 'cylinder'")
        sh["weight"] = float(s.get("weight", 1.0))
        shapes.append(sh)
        t = s["tissue"]
        cls = dict(tab["subcut_fat"], fat=True) if t == "oil" else dict(tab[t]) if isinstance(t, str) else dict(dict(adc_um2_ms=1.5, fat=False), **t)
        cls["t2s_ms"] = 1.0 / (1.0 / cls["t2_ms"] + 1.0 / cls["t2p_ms"])
        tissues[f"placed{k}"] = cls
        chi[f"placed{k}"] = float(s.get("chi_ppm", CHI_PPM["scalp_fat"] if cls["fat"] else 0.0))
    return shapes, tissues, chi


# ---------------------------------------------------------------------------------------------- the volume
class Lattice(NamedTuple):
    """What ``Volume.lattice`` returns: unpack it as (weights, origin_mm, spacing_mm). See the module docstring."""
    weights: np.ndarray
    origin_mm: tuple
    spacing_mm: tuple

    def axes(self):
        """The positions (z, y, x), three 1D arrays in mm, of the voxel centres."""
        shape = self.weights.shape[-3:]
        return tuple(self.origin_mm[a] + self.spacing_mm[a] * np.arange(shape[a]) for a in range(3))


class Volume:
    """A 3D object: a recipe and the ways to evaluate it. Build one with ``make_volume`` (seeded kinds) or
    ``shapes_volume`` (hand-made, for checks). See the module docstring for the conventions and the interface."""

    def __init__(self, kind, b0_t, recipe, classes, names, solid, tissues, chi_ppm, edge_mm, bounds_mm, organ, dtype, grid, pose=None):
        self.kind, self.b0_t = kind, float(b0_t)
        self.names, self.solid = tuple(names), tuple(solid)
        self.matter = self.names + self.solid
        self.tissues = tissues
        self.susceptibility_ppm = dict({k: 0.0 for k in self.matter}, **chi_ppm)
        self.susceptibility_ppm.setdefault("air", CHI_PPM["air"])
        self.edge_mm, self.bounds_mm, self.organ = float(edge_mm), bounds_mm, tuple(organ)
        self.grid = grid                                                 # (n, extent_mm) of the reference plane
        self.regions, self.layout = [], {}
        self._recipe, self._classes, self._dtype = recipe, classes, np.dtype(dtype)
        self._pose = pose                                                # (centre, rotation): object = centre + rot (patient - centre)
        self._lattice = None
        recipe["matter"] = self.matter
        recipe["index"] = {k: i for i, k in enumerate(self.matter)}

    # ------------------------------------------------------------------ coordinates
    def to_object(self, p, direction=False):
        """Patient -> object coordinates of a point (or of a direction). The identity for an upright object."""
        p = np.asarray(p, float)
        if self._pose is None:
            return p
        c, m = self._pose
        return m @ p if direction else c + m @ (p - c)

    def to_patient(self, q, direction=False):
        q = np.asarray(q, float)
        if self._pose is None:
            return q
        c, m = self._pose
        return m.T @ q if direction else c + m.T @ (q - c)

    def _warped(self, q):
        """Object coordinates [n, 3] -> the coordinates the shapes of the recipe are written in (behind its smooth warp)."""
        q = np.atleast_2d(np.asarray(q, float))
        warp = self._recipe.get("warp")
        if not warp:
            return q.copy()
        return q + np.concatenate([f.at(q[:, 0:1], q[:, 1:2], q[:, 2:3]) for f in warp], 1)

    def _unwarped(self, s):
        """The inverse: where in the object points of shapes lie (fixed-point iteration: the warp is a few mm over
        wavelengths of 90 mm and more, a contraction by less than 0.3 per step; tests/test_volume_world.py)."""
        s = np.atleast_2d(np.asarray(s, float))
        q = s.copy()
        if self._recipe.get("warp"):
            for _ in range(24):
                q = s - (self._warped(q) - q)
        return q

    def where(self, points):
        """Patient coordinates [n, 3] of points given in the coordinates of the recipe's shapes."""
        q = self._unwarped(points)
        return q if self._pose is None else np.array([self.to_patient(p) for p in q])

    def _frame(self, frame):
        if self._pose is None:
            return frame
        return Frame(self.to_object(frame.origin), *(self.to_object(v, direction=True) for v in (frame.row, frame.col, frame.normal)))

    # ------------------------------------------------------------------ evaluation
    def _object_points(self, y, x, z, edge_mm=None):
        """All matter classes at points given in OBJECT coordinates: float64 [matter, ...]."""
        y, x, z = np.broadcast_arrays(*(np.asarray(a, np.float64) for a in (y, x, z)))
        shape = y.shape
        P = _Points(*(np.ascontiguousarray(a).reshape(-1, 1) for a in (y, x, z)), self.edge_mm if edge_mm is None else edge_mm)
        return self._classes(self._recipe, P).reshape((len(self.matter),) + shape)

    def eval_points(self, y, x, z, solid=False, edge_mm=None):
        """Class weights at patient points (arrays that broadcast): float64 [classes, ...]. The recipe evaluated
        directly, with no shortcut: the reference for every other evaluation and the verifier's truth.
        ``solid``: all of ``matter`` instead of the signal classes ``names``."""
        y, x, z = np.broadcast_arrays(*(np.asarray(a, np.float64) for a in (y, x, z)))
        if self._pose is not None:
            c, m = self._pose
            d = [y - c[0], x - c[1], z - c[2]]
            y, x, z = [c[i] + m[i, 0] * d[0] + m[i, 1] * d[1] + m[i, 2] * d[2] for i in range(3)]
        w = self._object_points(y, x, z, edge_mm)
        return w if solid else w[:len(self.names)]

    def eval_grid(self, frame, v_mm, u_mm, solid=False, dtype=None, edge_mm=None):
        """Class weights float32 [classes, len(v), len(u)] at the in-plane positions v (along row) and u (along
        col) of ``frame``, both ascending. ``dtype``: the precision of the evaluation (None: float32 for the head,
        float64 for the test object); the result is float32 either way, as the weights of a 2D world."""
        f, dt = self._frame(frame), self._dtype if dtype is None else np.dtype(dtype)
        v, u = np.asarray(v_mm, float), np.asarray(u_mm, float)
        edge, keep = self.edge_mm if edge_mm is None else edge_mm, slice(None) if solid else slice(0, len(self.names))
        # In float32 the plane is evaluated in strips of rows that stay in the cache (measured: 640 x 640 in 52 to 65 ms
        # against 71 to 105 ms in one piece). A strip is a plane of its own, so the values are the same up to rounding.
        rows = len(v) if dt == np.float64 else max(32, STRIP_POINTS // max(len(u), 1))
        out = np.empty((len(self.matter[keep]), len(v), len(u)), np.float32)
        for i in range(0, len(v), rows):
            out[:, i:i + rows] = self._classes(self._recipe, _Plane(f.origin, f.row, f.col, f.normal, v[i:i + rows], u, edge, dt))[keep]
        return out

    def eval_plane(self, frame, n=None, extent_mm=None, solid=False, dtype=None, edge_mm=None):
        """Class weights float32 [classes, n, n] on the engine's grid in the plane ``frame`` (default: the grid of
        the world's reference plane): point (i, j) at origin + (i - n // 2) d row + (j - n // 2) d col, d = extent / n."""
        n, extent_mm = (self.grid[0] if n is None else int(n)), (self.grid[1] if extent_mm is None else float(extent_mm))
        c = grid_axis(n, extent_mm)
        return self.eval_grid(frame, c, c, solid=solid, dtype=dtype, edge_mm=edge_mm)

    def labels(self, weights):
        """uint8 labels of a weight array of all matter classes (class axis first): 0 where less than half is
        matter, else 1 + the index in ``matter`` of the largest class."""
        return np.where(weights.sum(0) > 0.5, weights.argmax(0) + 1, 0).astype(np.uint8)

    def chi_ppm(self, weights):
        """Volume susceptibility (ppm, relative to soft tissue) of a weight array of ALL matter classes (class axis
        first): what is not matter is air."""
        chi = np.array([self.susceptibility_ppm[k] for k in self.matter], weights.dtype)
        return np.tensordot(chi, weights, 1) + weights.dtype.type(self.susceptibility_ppm["air"]) * (1 - weights.sum(0))

    def organ_mask(self, weights):
        """Where the organ classes (``organ``: brain tissue; every signal class of a test object) fill more than half."""
        idx = [self.matter.index(k) for k in self.organ]
        return weights[idx].sum(0) > 0.5

    def lattice(self, spacing_mm=2.0, box=None, labels=False, edge_mm=None, supersample=1):
        """The object on a lattice: ``Lattice(weights, origin_mm, spacing_mm)`` (module docstring).

        ``spacing_mm``: a number or (dz, dy, dx). Voxel centres are whole multiples of the spacing, so the isocentre
        is a lattice point. ``box``: dict(z=(lo, hi), y=(lo, hi), x=(lo, hi)) in mm (default ``bounds_mm``: the
        object inside the imaged cube). ``labels``: uint8 labels instead of weights (built plane by plane, so a
        fine lattice never holds its float weights). ``supersample`` k or (kz, ky, kx): each voxel is the mean of
        that many samples per axis. The default 1 is the value at the voxel centre: fast, and exactly the recipe
        there, but a face that is parallel to the lattice planes (the flat end of a cylinder) is then located only
        to within the spacing; docs/physics/volume.md gives the measured sizes. ``edge_mm``: draw the boundaries with
        another half-width. The last lattice is kept; a float lattice above ``LATTICE_MAX_BYTES`` is refused."""
        sp = tuple(float(v) for v in (spacing_mm if isinstance(spacing_mm, (tuple, list)) else (spacing_mm,) * 3))
        box = dict(self.bounds_mm, **(box or {}))
        kz, ky, kx = (int(v) for v in (supersample if isinstance(supersample, (tuple, list)) else (supersample,) * 3))
        key = (sp, tuple((a, tuple(box[a])) for a in "zyx"), bool(labels), edge_mm, (kz, ky, kx))
        if self._lattice is not None and self._lattice[0] == key:
            return self._lattice[1]
        i0 = [int(np.ceil(box[a][0] / s - 1e-9)) for a, s in zip("zyx", sp)]
        i1 = [int(np.floor(box[a][1] / s + 1e-9)) for a, s in zip("zyx", sp)]
        nz, ny, nx = (b - a + 1 for a, b in zip(i0, i1))
        if min(nz, ny, nx) < 1:
            raise ValueError("the box holds no lattice point")
        size = nz * ny * nx * (1 if labels else 4 * len(self.matter))
        if size > LATTICE_MAX_BYTES:
            raise ValueError(f"a lattice of {nz} x {ny} x {nx} voxels and {len(self.matter)} classes is {size / 1e9:.1f} GB: "
                             "ask for labels=True, a coarser spacing or a smaller box")
        sub = lambda k: (np.arange(k) + 0.5) / k - 0.5                  # noqa: E731  sample positions inside a voxel, in voxels
        ys = ((i0[1] + np.arange(ny))[:, None] + sub(ky)[None, :]).ravel() * sp[1]
        xs = ((i0[2] + np.arange(nx))[:, None] + sub(kx)[None, :]).ravel() * sp[2]
        out = np.empty((nz, ny, nx), np.uint8) if labels else np.empty((len(self.matter), nz, ny, nx), np.float32)
        for iz in range(nz):
            acc = None
            for dz in sub(kz):
                w = self.eval_grid(Frame.plane("axial", (i0[0] + iz + dz) * sp[0]), ys, xs, solid=True, edge_mm=edge_mm)
                acc = w if acc is None else acc + w
            if kz * ky * kx > 1:
                acc = acc.reshape(len(self.matter), ny, ky, nx, kx).mean((2, 4)) / kz
            if labels:
                out[iz] = self.labels(acc)
            else:
                out[:, iz] = acc
        res = Lattice(out, tuple(i * s for i, s in zip(i0, sp)), sp)
        self._lattice = (key, res)
        return res

    # ------------------------------------------------------------------ regions and truth
    def inside(self, region, y, x, z):
        """Whether patient points lie in a region (a record of ``regions``, or its id): a ball, or a cylinder
        along ``axis`` with flat ends."""
        r = next(q for q in self.regions if q["id"] == region) if isinstance(region, str) else region
        d = [np.asarray(a, float) - r[k] for a, k in zip((y, x, z), ("y_mm", "x_mm", "z_mm"))]
        if r.get("shape", "sphere") == "sphere":
            return d[0] ** 2 + d[1] ** 2 + d[2] ** 2 <= r["radius_mm"] ** 2
        t = sum(d[i] * r["axis"][i] for i in range(3))
        return (sum(d[i] ** 2 for i in range(3)) - t * t <= r["radius_mm"] ** 2) & (np.abs(t) <= r["half_length_mm"])

    def truth(self):
        """Scanner and verifier side only: per scored region the true values of its class, as ``world.truth`` gives
        them for a 2D world, and its susceptibility."""
        out = {}
        for r in self.regions:
            c = self.tissues[r["cls"]]
            out[r["id"]] = dict(t1_ms=c["t1_ms"], t2_ms=c["t2_ms"], t2s_ms=c["t2s_ms"], pd=c["pd"], fat=c["fat"],
                                chi_ppm=self.susceptibility_ppm[r["cls"]])
        return out


def _pose(seed, vs, centre, W):
    """The pose of the object: turned about ``centre`` by angles drawn within ``pose_deg`` (a number: all three
    axes; three entries: about y, x, z; an entry is a number, read as +- that, or a (low, high) range)."""
    p = vs.get("pose_deg")
    if not p:
        return None, (0.0, 0.0, 0.0)
    g = W.rng_for(seed, "geometry", 6)
    ranges = [p] * 3 if isinstance(p, (int, float)) or (len(p) == 2 and not isinstance(p[0], (tuple, list))) else list(p)
    ang = [float(g.uniform(*r)) if isinstance(r, (tuple, list)) else float(g.uniform(-r, r)) for r in ranges]
    return (np.asarray(centre, float), rotation(*ang)), tuple(ang)


def _plain(obj):
    """A recipe value as plain JSON types."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj.item() if hasattr(obj, "item") else obj


def make_volume(seed: str, spec: dict) -> Volume:
    """The volume of (seed, full world spec). ``spec["volume"]``: the options of ``VOLUME_SPEC``."""
    from . import world as W
    kind, b0_t = spec["kind"], float(spec["b0_t"])
    vs = dict(VOLUME_SPEC, **(spec.get("volume") or {}))
    edge = float(spec.get("edge_mm") or anatomy.EDGE_MM)
    n, extent = GRID[kind]
    grid = (int(spec.get("n") or n), extent)
    half = extent / 2
    cube = lambda lo, hi: (max(lo, -half), min(hi, half - extent / grid[0]))     # noqa: E731  the imaged cube
    placed, placed_tis, placed_chi = _insert_shapes(vs.get("inserts"), b0_t)
    if kind == "head3d":
        if spec["lesion"] not in (None, "ms"):
            raise ValueError("head3d: lesion is None or 'ms'")
        R = _head_recipe(seed, spec, vs, W)
        names = R["names"] + [f"placed{k}" for k in range(len(placed))]
        tis = dict(anatomy._jitter(tissue.table(b0_t), R["names"], W.rng_for(seed, "tissue")), **placed_tis)
        chi = dict({k: v for k, v in CHI_PPM.items() if k in names or k == "air"}, **placed_chi)
        m = R["wmax"] + 10 * edge
        bounds = dict(z=cube(-1e6, R["zc"] + R["az"] + m), y=cube(-R["ay"] - m, R["ay"] + m), x=cube(-R["ax"] * (1 + R["egg"]) - m, R["ax"] * (1 + R["egg"]) + m))
        pose, angles = _pose(seed, vs, (0.0, 0.0, R["zc"]), W)
        if pose is not None:
            bounds = {a: cube(-1e6, 1e6) for a in "zyx"}
        vol = Volume(kind, b0_t, R, _head_classes, names, (), tis, chi, edge, bounds, [k for k in ("wm", "gm", "deep_gm", "lesion_ms") if k in names],
                     np.float32, grid, pose)
        R["inner"] = [vol.matter.index(k) for k in vol.matter if k not in ("skin", "scalp_fat", "blood") and not k.startswith("placed")]
        if spec["lesion"] == "ms":
            R["lesions"], chi_les = _head_lesions(vol, R, seed, W)
            vol.susceptibility_ppm["lesion_ms"] = chi_les
        R["vessels"] = _head_vessels(R, seed, vs["vessels"], W)
        R["inserts"] = placed
        vol.size_mm = (2 * R["ay"], 2 * R["ax"])
        vol.regions = _head_regions(vol, R, seed, W)
        keys = ("ax", "ay", "az", "zc", "neck", "az_low", "egg", "atrophy", "b_skin", "fat", "bone", "cav", "cav_low", "b_csf", "t_gm", "sulc_depth")
        groups = dict(ventricles=R["vent"], deep_nuclei=R["deep"], air=R["air"], eyes=R["eyes"], lesions=R["lesions"])
        marks = [(s, k) for shapes in groups.values() for s in shapes for k in ("c", "p0", "p1") if k in s]
        at = iter(vol.where([s[k] for s, k in marks]))                   # where each shape is in the patient (the warp inverted)

        def row(s):                                                      # a shape: its place in the patient, and its sizes in the recipe
            out = dict(name=s["name"], **{k: s[k] for k in ("ax", "radius") if k in s})
            out.update({k + "_mm": next(at) for k in ("c", "p0", "p1") if k in s})
            return out

        vessels = [dict(name=s["name"], radius=s["radius"], p0_mm=vol.to_patient(s["p0"]), p1_mm=vol.to_patient(s["p1"]),
                        direction=vol.to_patient(s["direction"], direction=True), speed_mm_s=s["speed_mm_s"]) for s in R["vessels"]]
        vol.layout = _plain(dict({k: R[k] for k in keys}, pose_deg=angles, **{name: [row(s) for s in shapes] for name, shapes in groups.items()},
                                 vessels=vessels, lesion_chi_ppm=vol.susceptibility_ppm.get("lesion_ms")))
        return vol
    if kind == "qa3d":
        R, qsp = _qa_recipe(seed, spec, vs, W)
        names = R["names"] + [f"placed{k}" for k in range(len(placed))]
        tis = dict(_qa_tissues(R, qsp, seed, b0_t, W), **placed_tis)
        solid = ("plastic",) if R["features"] else ()
        chi = dict({k: CHI_PPM["scalp_fat"] for k in R["names"] if tis[k]["fat"]}, **{k: CHI_PPM[k] for k in solid}, **placed_chi)
        pose, angles = _pose(seed, vs, (R["body"]["y"], R["body"]["x"], R["cz"]), W)
        m = 20 * edge
        bounds = {a: cube(R["box_lo"][i] - m, R["box_hi"][i] + m) if pose is None else cube(-1e6, 1e6) for i, a in enumerate("yxz")}
        vol = Volume(kind, b0_t, R, _qa_classes, names, solid, tis, chi, edge, bounds, names, np.float64, grid, pose)
        R["inserts"] = placed
        vol.size_mm = R["flat"]["size_mm"]
        vol.regions = _qa_regions(vol, R, qsp, seed, W)
        vol.layout = _plain(dict(body=dict(R["body"], z=R["cz"], half_length=R["half_length"]), vial_half_length=R["vial_half"], vials=R["vials"],
                                 side_of_sphere_and_wedges=R["side"], pose_deg=angles,
                                 **(dict(sphere=R["sphere"], wedges=R["wedges"], ramp_block=R["block"]) if R["features"] else {})))
        return vol
    raise ValueError(f"volume kind must be one of {KINDS}")


def shapes_volume(shapes, b0_t=3.0, tissues=None, chi_ppm=None, edge_mm=anatomy.EDGE_MM, solid=(), bounds_mm=None, n=640, extent_mm=256.0) -> Volume:
    """A hand-made volume for checks, the 3D form of ``world.shapes_phantom``. ``shapes``: list of (class name,
    callable(y, x, z) -> signed distance in mm, negative inside); later shapes sit on top of earlier ones. Names
    in ``solid`` give no signal. Evaluated in float64, every shape everywhere."""
    order = list(dict.fromkeys(name for name, _ in shapes))
    names = [k for k in order if k not in solid]
    tab = dict(tissue.table(b0_t), **(tissues or {}))

    def classes(R, P):
        out = np.zeros((len(R["matter"]),) + P.shape, P.dt)
        for name, f in shapes:
            m = P.sig(-f(P.y, P.x, P.z))
            out *= (1.0 - m)
            out[R["index"][name]] += m
        return out

    half = extent_mm / 2
    bounds = bounds_mm or {a: (-half, half - extent_mm / n) for a in "zyx"}
    vol = Volume("shapes", b0_t, dict(kind="shapes"), classes, names, [k for k in order if k in solid], {k: dict(tab[k]) for k in names},
                 dict(chi_ppm or {}), edge_mm, bounds, names, np.float64, (int(n), float(extent_mm)))
    vol.size_mm = (extent_mm, extent_mm)
    return vol


# ---------------------------------------------------------------------------------------------- the world of a volume
def build(seed, sp, layout_only=False, rot_deg=0.0, base=None):
    """The world dict of a volume kind (called by ``world._build`` before any draw of the 2D kinds).

    It carries the volume (``world["volume"]``) and, so that everything written for 2D worlds keeps working, the
    REFERENCE PLANE: the axial plane through the isocentre on the master grid (``weights``, ``masks``), turned by
    ``rot_deg`` as ``world.turned`` asks. It has no ``fields``: the fields of a volume come from its lattice
    (``scanner/fields3d.py``), and until a slice engine joins the two no scan is formed of a volume world
    (``kspace.Engine`` needs ``fields``)."""
    from . import world as W
    kind, b0_t = sp["kind"], float(sp["b0_t"])
    if b0_t not in W.B1RMS_LIMIT_UT:
        raise ValueError("field strength must be 1.5 or 3.0 T")
    for key in ("gnl", "grad_scale", "motion", "dbdz"):
        if sp.get(key):
            raise ValueError(f"world kind {kind!r}: {key} is a model of the 2D worlds and is not defined for a volume yet")
    vol = base["volume"] if base is not None else make_volume(seed, sp)
    n, extent = vol.grid
    ph = dict(kind=kind, b0_t=b0_t, n=n, extent_mm=extent, names=list(vol.names), tissues=vol.tissues, regions=vol.regions, layout=vol.layout,
              size_mm=vol.size_mm, air=[], volume=vol)
    system = W.make_system(ph, W.rng_for(seed, "system"), sp, lambda: W.rng_for(seed, "system", 1))
    if layout_only:
        return dict(ph, schema=W.SCHEMA, system=system)
    frame = Frame.plane("axial", rot_deg=rot_deg)
    weights = vol.eval_plane(frame, n, extent, dtype=np.float64)
    lesion = weights[vol.names.index("lesion_ms")].copy() if "lesion_ms" in vol.names else np.zeros((n, n), np.float32)
    w = dict(ph, weights=weights, masks=dict(body=weights.sum(0) > 0.5, brain=vol.organ_mask(weights), lesion=lesion), frame=frame,
             schema=W.SCHEMA, system=system)
    w["effects"] = None if sp["effects"] is None else sorted(sp["effects"])
    w["console"] = dict(fov_mm=EXAM_FOV_MM[kind], min_pixel_mm=round(2.0 * extent / n, 3), b1rms_limit_ut=W.B1RMS_LIMIT_UT[b0_t], b0_t=b0_t)
    if sp.get("console_system"):
        w["console"]["system"] = dict(sp["console_system"])
    w.update(rot_deg=float(rot_deg), gnl=None, source=dict(kind="seed", seed=seed, spec=sp))
    return w
