"""Fields in three dimensions: the static field that the anatomy itself induces, shimming on it, closed-form
inclusions, declared transmit and receive profiles along z, and the forward and reference inverse steps of
susceptibility mapping. A library: no scan of this package uses it yet (docs/physics/fields3d.md).

The model. Matter of volume susceptibility chi (SI, |chi| << 1) in a field B0 along the unit vector b is magnetised,
M = chi B0 / mu0, and the magnetisation adds a field. To first order in chi the component along b that a nucleus
sees is a convolution with the field of a point dipole; in k-space it is a product (Salomir 2003; Marques & Bowtell
2005):

    delta(k) = D(k) chi(k),    D(k) = 1/3 - (k . b)^2 / |k|^2,    D(0) = 0,        delta = (B_nucleus - B0) . b / B0

Conventions, each fixed in one place of this file:
  sign      delta > 0 is a spin that precesses faster: positive off-resonance, as everywhere in the engine
            (docs/physics/conventions.md section 5). Hz = delta [ppm] x 1e-6 x gamma x B0 (``Field3D.hz_per_ppm``).
  Lorentz   the 1/3 is the sphere of Lorentz: the nucleus sits in a small sphere cut out of the continuum, and the
            near neighbours inside it add nothing on average, so it sees the macroscopic field minus (2/3) chi(r) B0.
            Included (``lorentz=True``). It is a local term, constant within one material. For one material in air it
            is a constant that the frequency adjustment removes; between two materials it changes the step of the
            field at their interface and is therefore kept. ``lorentz=False`` returns the macroscopic field
            (D = 1 - (k . b)^2 / |k|^2), the quantity of the classical closed forms "without the correction".
  k = 0     D(k) has no limit at k = 0 (its value depends on the direction of approach: on the shape of the object
            at infinity), so the mean of the field over the transformed box is not determined by the model. It is
            set to zero. Two consequences: only differences of chi matter (a constant added to chi changes
            nothing), and the result carries an unknown constant, which a scanner removes with its frequency
            adjustment and this file with the zeroth shim term.
  period    the transform makes the lattice periodic: every object has images at the box length. The lattice is
            therefore zero-padded (``pad``, 1.5 by default; the padding is filled with the susceptibility of the
            surroundings), and the leading field of the images, which is quadratic in the position and known in
            closed form from the dipole moment of the object, is taken off (``images``). What remains is measured
            by ``validate``. A lattice that is meant to be periodic (an infinite cylinder) sets ``images=False``.
  B0        any direction. Along a lattice axis (the scanner's case: B0 is patient z whatever the slice) D is the
            kernel above. Oblique to the lattice the mixed products k_i k_j are replaced by their lattice form
            (``dipole_kernel``): the plain product rings.
  axes      lattices are arrays [z, y, x]; ``spacing`` and ``Lattice.origin`` follow the array axes (dz, dy, dx).
            Vectors (``b0_dir``, a frame's origin, row and column, centres of inclusions) are patient components
            (y, x, z) in mm from the isocentre, the order of the engine's grids and of ``world["air"]``; +y posterior,
            +x left, +z superior. Index i of an axis of ``Lattice.centred`` sits at (i - n // 2) x spacing.
  units     chi and delta in ppm relative to soft tissue (water); air is +9.4 ppm (``world.DCHI_AIR``).

The surface problem. The field is discontinuous wherever chi is: across an interface with unit normal n it steps by
(1/3 - (n . b)^2) x (step of chi), up to 800 Hz at an air surface at 3 T. A lattice holds the mean of both sides in
its surface cells, and interpolating that puts a false ramp of hundreds of Hz into the skin. The step is known in
closed form, so it is taken out before interpolating and put back with the susceptibility of the material the signal
comes from (``Field3D``):

    delta(r) = I{delta_lattice - kappa chi_lattice}(r) + I{kappa}(r) chi_seen(r),    kappa = 1/3 - (n . b)^2

with I trilinear interpolation, n the direction of the gradient of chi_lattice, and chi_seen the susceptibility of
the tissue at r: from the class weights of the image plane when the caller has them, otherwise from the lattice's
own tissue-only mean, continued outwards through the surface cells. The relation is exact for a plane interface and
any layering parallel to it; the remainder is of the order (cell / radius of curvature).

Lattice requirement: class weights must be cell MEANS (volume fractions), not samples at cell centres
(``volume_fraction``). A staircase moves the surface by up to half a cell.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import scipy.fft as sfft
import scipy.ndimage as ndi

from .world import B1_SHAPE, DCHI_AIR, GAMMA_HZ_PER_T

DCHI_AIR_PPM = DCHI_AIR * 1e6                          # air minus soft tissue, ppm (9.4)
# A structure enters the lattice only if it is at least this many cells across (radius >= 2 cells); a smaller one is
# added in closed form (``Field3D.inclusions``), never both. The error of either choice: validate rows F21, F22.
MIN_LATTICE_RADIUS_CELLS = 2.0
SHIM_TERMS = ("F0", "X", "Y", "Z", "Z2", "ZX", "ZY", "X2-Y2", "XY")
SHIM_ORDER = (0, 1, 1, 1, 2, 2, 2, 2, 2)
# Largest field each shim channel can add, in microtesla per metre^order. DECLARED sizes of the order of a whole-body
# clinical system, not a product's data: first order is the gradient set and is not the limit in practice.
SHIM_LIMITS_UT = {"X": 1000.0, "Y": 1000.0, "Z": 1000.0, "Z2": 4000.0, "ZX": 2000.0, "ZY": 2000.0, "X2-Y2": 1000.0, "XY": 1000.0}


# ------------------------------------------------------------------------------------------------ lattice
class Lattice(NamedTuple):
    """Geometry of a lattice [nz, ny, nx]: ``spacing`` (dz, dy, dx) in mm, ``origin`` (z, y, x) in mm of the centre
    of cell [0, 0, 0]. Array-axis order throughout."""
    shape: tuple
    spacing: tuple
    origin: tuple

    @classmethod
    def centred(cls, shape, spacing, centre=(0.0, 0.0, 0.0)):
        """The engine's grid rule on every axis: index i at centre + (i - n // 2) x spacing. ``centre`` is (z, y, x)."""
        sp = _three(spacing)
        return cls(tuple(int(n) for n in shape), sp, tuple(float(c) - (n // 2) * d for c, n, d in zip(centre, shape, sp)))

    def axes(self):
        """Cell-centre coordinates (z, y, x), three 1D arrays in mm."""
        return tuple(o + d * np.arange(n) for o, d, n in zip(self.origin, self.spacing, self.shape))

    def mesh(self):
        """Cell-centre coordinates as three broadcastable arrays (y, x, z): the argument order of every function of
        position in this file."""
        z, y, x = self.axes()
        return y[None, :, None], x[None, None, :], z[:, None, None]

    def index(self, y, x, z):
        """Fractional array indices [3, ...] (axis order z, y, x) of the patient positions (y, x, z)."""
        y, x, z = np.broadcast_arrays(*(np.asarray(v, float) for v in (y, x, z)))
        return np.stack([(v - o) / d for v, o, d in zip((z, y, x), self.origin, self.spacing)])


def _three(v):
    return tuple(float(a) for a in (v if np.ndim(v) else (v, v, v)))


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if v.shape != (3,) or not n > 0:
        raise ValueError("a direction is three patient components (y, x, z), not all zero")
    return v / n


def volume_fraction(sdf, lattice: Lattice, oversample=6):
    """Cell means of the indicator of an implicit shape: the fraction of each lattice cell that lies inside.

    ``sdf(y, x, z)``: signed distance to the surface in mm, negative inside (it may underestimate the distance, it
    must not overestimate it by more than a half). Cells whose centre is farther from the surface than the cell's
    half-diagonal are 0 or 1; the others are sampled on ``oversample``^3 points."""
    y, x, z = lattice.mesh()
    s = np.asarray(sdf(y, x, z), float)
    s = np.broadcast_to(s, lattice.shape)
    w = (s < 0).astype(np.float32)
    half = 0.5 * float(np.linalg.norm(lattice.spacing))
    near = np.argwhere(np.abs(s) < 1.5 * half)
    if len(near) == 0:
        return w
    o = (np.arange(oversample) + 0.5) / oversample - 0.5
    oz, oy, ox = (v.ravel() for v in np.meshgrid(o * lattice.spacing[0], o * lattice.spacing[1], o * lattice.spacing[2], indexing="ij"))
    az, ay, ax = lattice.axes()
    for a in range(0, len(near), 16384):
        i, j, k = near[a:a + 16384].T
        inside = sdf(ay[j][:, None] + oy[None, :], ax[k][:, None] + ox[None, :], az[i][:, None] + oz[None, :]) < 0
        w[i, j, k] = inside.mean(1)
    return w


def susceptibility(weights, chi_ppm, background_ppm=DCHI_AIR_PPM):
    """Lattice susceptibility from class weights [class, nz, ny, nx] (cell means) and one value per class (ppm
    relative to soft tissue): what is not covered by a class is the surroundings (air).

    Returns (chi, tissue, chi_tissue): the mean susceptibility of each cell, the fraction of the cell that holds
    signal-bearing material, and the mean susceptibility of that material alone (0 where there is none)."""
    w = np.asarray(weights, np.float32)
    c = np.asarray(chi_ppm, np.float32)
    if w.ndim != 4 or c.shape != (w.shape[0],):
        raise ValueError("weights must be [class, nz, ny, nx] with one susceptibility per class")
    tissue = np.clip(w.sum(0), 0.0, 1.0)
    part = np.tensordot(c, w, 1)
    chi = part + (1.0 - tissue) * np.float32(background_ppm)
    return chi.astype(np.float32), tissue, np.divide(part, tissue, out=np.zeros_like(part), where=tissue > 1e-3)


# ------------------------------------------------------------------------------------------------ the dipole kernel
def padded_shape(shape, pad=1.5):
    """Shape of the transform: an explicit shape, or every axis times ``pad`` rounded up to a fast transform length."""
    if np.ndim(pad):
        out = tuple(int(p) for p in pad)
        if len(out) != 3 or any(p < n for p, n in zip(out, shape)):
            raise ValueError("pad must be a factor >= 1 or a shape not smaller than the lattice")
        return out
    if pad < 1:
        raise ValueError("pad must be a factor >= 1 or a shape not smaller than the lattice")
    return tuple(int(sfft.next_fast_len(int(math.ceil(n * pad)), real=True)) for n in shape)


def dipole_kernel(shape, spacing, b0_dir=(0.0, 0.0, 1.0), lorentz=True, dtype=np.float32):
    """D(k) on the half-spectrum of ``scipy.fft.rfftn`` for a lattice [nz, ny, nx]; D(0) = 0.

    (k . b)^2 = sum_i b_i^2 k_i^2 + 2 sum_{i<j} b_i b_j s_i s_j with s_i = sin(2 pi k_i h_i) / (2 pi h_i) in the mixed
    terms. A product k_i k_j is odd in each component, so on a periodic lattice it would jump at the edge of the
    zone (k_i = +-1 / (2 h_i) are the same sample); its lattice form has to vanish there, and the symbol of a central
    difference is the simplest function that does and equals k_i at small k. With the plain product a field with B0
    oblique to the lattice rings at the cell scale, ten times the error of an axis-aligned B0 (validate F16c). For
    B0 along a lattice axis there is no mixed term and D is the textbook kernel."""
    sp = _three(spacing)
    b = _unit(b0_dir)[[2, 0, 1]]                                         # components along the array axes z, y, x
    freq = (sfft.fftfreq(shape[0], sp[0]), sfft.fftfreq(shape[1], sp[1]), sfft.rfftfreq(shape[2], sp[2]))
    form = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))
    k = [f.astype(dtype).reshape(form[i]) for i, f in enumerate(freq)]
    k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2]
    k2[0, 0, 0] = 1.0
    kb = np.zeros(k2.shape, dtype)
    for i in range(3):
        if b[i]:
            kb += dtype(b[i] * b[i]) * (k[i] * k[i])
    for i in range(3):
        for j in range(i + 1, 3):
            if b[i] and b[j]:
                si, sj = ((np.sin(2 * np.pi * freq[a] * sp[a]) / (2 * np.pi * sp[a])).astype(dtype).reshape(form[a]) for a in (i, j))
                kb += dtype(2 * b[i] * b[j]) * (si * sj)
    kb /= k2
    d = dtype(1.0 / 3.0 if lorentz else 1.0) - kb
    d[0, 0, 0] = 0.0
    return d


def _background(chi):
    """Susceptibility of the surroundings when the caller names none: the median of the six faces of the lattice."""
    faces = [chi[0], chi[-1], chi[:, 0], chi[:, -1], chi[:, :, 0], chi[:, :, -1]]
    return float(np.median(np.concatenate([f.ravel() for f in faces])))


def image_hessian(lengths, b0_dir=(0.0, 0.0, 1.0), shells=12):
    """Second derivatives H[i, j] (array axes z, y, x; 1 / mm^5) at the origin of the field that the periodic images
    of a point dipole of unit moment (chi x volume = 1 mm^3) make there, for a cell of edge ``lengths`` (Lz, Ly, Lx):
    the images at R_n, n != 0, of f(x) = (3 (x . b)^2 - |x|^2) / (4 pi |x|^5). The sum of second differences converges
    absolutely (as R^-5); ``shells`` cells in every direction leave less than 0.1 %. H is symmetric and traceless.
    Cubic cell, B0 along z: H_zz = 5.936 / L^5 = -2 H_xx = -2 H_yy."""
    L = np.asarray(lengths, float)
    b = _unit(b0_dir)[[2, 0, 1]]
    g = np.arange(-shells, shells + 1)
    R = np.stack(np.meshgrid(g * L[0], g * L[1], g * L[2], indexing="ij"), -1).reshape(-1, 3)
    R = R[np.any(R != 0, 1)]

    def f(x):
        r2 = (x * x).sum(-1)
        return (3 * (x @ b) ** 2 - r2) / (4 * np.pi * r2 ** 2.5)
    h = 0.02 * float(L.min())
    e = np.eye(3) * h
    f0, H = f(-R), np.zeros((3, 3))
    for i in range(3):
        H[i, i] = (f(e[i] - R) - 2 * f0 + f(-e[i] - R)).sum() / h ** 2
        for j in range(i + 1, 3):
            H[i, j] = H[j, i] = (f(e[i] + e[j] - R) - f(e[i] - e[j] - R) - f(-e[i] + e[j] - R) + f(-e[i] - e[j] - R)).sum() / (4 * h ** 2)
    return H


def _image_field(q, spacing, lengths, b0_dir):
    """Leading field of the periodic images of the source ``q`` (chi minus surroundings, [nz, ny, nx]) on its own
    lattice: (1/2) sum_src q (r - r')^T H (r - r') = (1/2) p r^T H r - r^T H m1 + (1/2) tr(H M2), with the zeroth,
    first and second moments p, m1, M2 of the source."""
    H = image_hessian(lengths, b0_dir)
    q = q.astype(np.float64) * float(np.prod(spacing))
    ax = [np.arange(n) * d for n, d in zip(q.shape, spacing)]
    marg = [q.sum((1, 2)), q.sum((0, 2)), q.sum((0, 1))]
    pair = {(0, 1): q.sum(2), (0, 2): q.sum(1), (1, 2): q.sum(0)}
    p = float(marg[0].sum())
    m1 = np.array([float(m @ a) for m, a in zip(marg, ax)])
    m2 = np.zeros((3, 3))
    for i in range(3):
        m2[i, i] = float(marg[i] @ (ax[i] * ax[i]))
        for j in range(i + 1, 3):
            m2[i, j] = m2[j, i] = float(ax[i] @ pair[(i, j)] @ ax[j])
    r = [ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :]]
    out = np.full(q.shape, 0.5 * float(np.sum(H * m2)))
    hm = H @ m1
    for i in range(3):
        out -= r[i] * hm[i]
        out += 0.5 * p * H[i, i] * r[i] * r[i]
        for j in range(i + 1, 3):
            out += p * H[i, j] * r[i] * r[j]
    return out


def dipole_field(chi, spacing, b0_dir=(0.0, 0.0, 1.0), pad=1.5, background=None, lorentz=True, extrude=(), images=True):
    """Relative field shift delta = dB / B0 (unit of ``chi``) on the lattice of the susceptibility ``chi`` [nz, ny, nx].

    spacing     (dz, dy, dx) in mm, or one number
    b0_dir      direction of B0, patient components (y, x, z); any direction
    pad         factor (1.5) or shape of the zero-padded transform, against the images of the periodic lattice
    background  susceptibility that fills the padding (None: the median of the lattice's faces)
    lorentz     False: the macroscopic field, without the sphere of Lorentz
    extrude     faces through which the object goes on, e.g. ("z-",) for a neck: the face plane is repeated through
                half of the padding on that side (a body that ends at the lattice would end in a flat cut)
    images      take off the leading (quadratic) field of the periodic images (``image_hessian``). Not done for an
                extruded object, whose images touch it. False for a lattice that is meant to be periodic.
    Before the image term the mean over the padded box is zero (module docstring, "k = 0"); the constant of the
    result has no meaning. The transform runs in the precision of ``chi``."""
    chi = np.asarray(chi)
    if chi.ndim != 3:
        raise ValueError("chi must be a lattice [nz, ny, nx]")
    dtype = np.float64 if chi.dtype == np.float64 else np.float32
    shape = padded_shape(chi.shape, pad)
    bg = _background(chi) if background is None else float(background)
    buf = np.zeros(shape, dtype)
    nz, ny, nx = chi.shape
    buf[:nz, :ny, :nx] = chi - dtype(bg)
    for face in extrude:
        axis, side = "zyx".index(face[0]), face[1]
        n, m = chi.shape[axis], (shape[axis] - chi.shape[axis]) // 2
        if face not in ("z-", "z+", "y-", "y+", "x-", "x+"):
            raise ValueError("extrude names faces: 'z-', 'z+', 'y-', 'y+', 'x-', 'x+'")
        src = [slice(0, nz), slice(0, ny), slice(0, nx)]
        dst = list(src)
        src[axis] = slice(0, 1) if side == "-" else slice(n - 1, n)
        dst[axis] = slice(shape[axis] - m, shape[axis]) if side == "-" else slice(n, n + m)   # "below" index 0 is the end of the buffer
        buf[tuple(dst)] = buf[tuple(src)]
    d = dipole_kernel(shape, spacing, b0_dir, lorentz, dtype)
    out = np.ascontiguousarray(sfft.irfftn(sfft.rfftn(buf, workers=1) * d, s=shape, workers=1)[:nz, :ny, :nx])
    if images and not extrude:
        sp = _three(spacing)
        out -= _image_field(buf[:nz, :ny, :nx], sp, [n * v for n, v in zip(shape, sp)], b0_dir).astype(dtype)
    return out


# ------------------------------------------------------------------------------------------------ closed forms
def sphere_field(y, x, z, centre, radius, dchi=1.0, b0_dir=(0.0, 0.0, 1.0), lorentz=True):
    """Field of a sphere of susceptibility ``dchi`` above its surroundings (Schenck 1996): inside 0 (2/3 dchi without
    the sphere of Lorentz), outside (dchi / 3)(a / r)^3 (3 cos^2 t - 1), t from B0. Exact for any radius."""
    b = _unit(b0_dir)
    p = [np.asarray(v, float) - c for v, c in zip((y, x, z), centre)]
    r2 = p[0] ** 2 + p[1] ** 2 + p[2] ** 2
    pb = p[0] * b[0] + p[1] * b[1] + p[2] * b[2]
    safe = np.maximum(r2, radius * radius)
    out = dchi / 3.0 * radius ** 3 * (3.0 * pb * pb / safe - 1.0) / safe ** 1.5
    return np.where(r2 < radius * radius, 0.0 if lorentz else 2.0 * dchi / 3.0, out)


def cylinder_field(y, x, z, point, axis, radius, dchi=1.0, b0_dir=(0.0, 0.0, 1.0), lorentz=True):
    """Field of an infinite circular cylinder through ``point`` along ``axis`` at the angle t to B0 (Schenck 1996):
    inside (dchi / 6)(3 cos^2 t - 1), outside (dchi / 2) sin^2 t (a / rho)^2 cos 2 phi, with phi the azimuth in the
    plane normal to the cylinder, counted from the projection of B0 onto that plane."""
    b, t = _unit(b0_dir), _unit(axis)
    cos_t = float(b @ t)
    bp = b - cos_t * t                                                  # projection of B0 onto the normal plane, length sin t
    p = [np.asarray(v, float) - c for v, c in zip((y, x, z), point)]
    along = p[0] * t[0] + p[1] * t[1] + p[2] * t[2]
    q = [p[i] - along * t[i] for i in range(3)]
    rho2 = q[0] ** 2 + q[1] ** 2 + q[2] ** 2
    qb = q[0] * bp[0] + q[1] * bp[1] + q[2] * bp[2]
    safe = np.maximum(rho2, radius * radius)
    out = dchi / 2.0 * radius ** 2 * (2.0 * qb * qb - safe * float(bp @ bp)) / safe ** 2
    inside = dchi / 6.0 * (3.0 * cos_t ** 2 - 1.0) + (0.0 if lorentz else 2.0 * dchi / 3.0)
    return np.where(rho2 < radius * radius, inside, out)


def shell_field(y, x, z, centre, r_inner, r_outer, dchi=1.0, b0_dir=(0.0, 0.0, 1.0)):
    """Field of a spherical shell of susceptibility ``dchi`` above what is inside and outside it: two spheres."""
    return sphere_field(y, x, z, centre, r_outer, dchi, b0_dir) - sphere_field(y, x, z, centre, r_inner, dchi, b0_dir)


def demagnetising_factors(semi_axes):
    """Demagnetising factors (N_1, N_2, N_3; sum 1) of an ellipsoid with the given semi-axes (Osborn 1945):
    N_i = (a b c / 2) int_0^inf ds / ((s + a_i^2) sqrt((s + a^2)(s + b^2)(s + c^2)))."""
    import scipy.integrate
    a = np.asarray(semi_axes, float) / float(np.max(semi_axes))
    out = []
    for i in range(3):
        f = lambda u, i=i: 1.0 / ((u / (1 - u) + a[i] ** 2) * math.sqrt(np.prod(u / (1 - u) + a ** 2)) * (1 - u) ** 2)   # noqa: E731, s = u / (1 - u)
        out.append(0.5 * float(np.prod(a)) * scipy.integrate.quad(f, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13, limit=400)[0])
    return tuple(out)


def ellipsoid_inside(semi_axes, b_in_axes, dchi=1.0, lorentz=True):
    """Uniform field inside an ellipsoid: dchi (1/3 - sum_i N_i b_i^2), b_i the components of B0 along its axes."""
    b = _unit(b_in_axes)
    n = np.array(demagnetising_factors(semi_axes))
    return dchi * ((1.0 / 3.0 if lorentz else 1.0) - float(n @ (b * b)))


def inclusion_field(inc, y, x, z, b0_dir=(0.0, 0.0, 1.0)):
    """Field (ppm) of one closed-form inclusion: dict(kind="sphere", centre=(y, x, z), radius_mm, dchi_ppm) or
    dict(kind="cylinder", point=(y, x, z), axis=(y, x, z), radius_mm, dchi_ppm). ``dchi_ppm`` is relative to the
    material around the inclusion."""
    if inc["kind"] == "sphere":
        return sphere_field(y, x, z, inc["centre"], inc["radius_mm"], inc["dchi_ppm"], b0_dir)
    if inc["kind"] == "cylinder":
        return cylinder_field(y, x, z, inc["point"], inc["axis"], inc["radius_mm"], inc["dchi_ppm"], b0_dir)
    raise ValueError("an inclusion is a 'sphere' or a 'cylinder'")


def lattice_or_closed_form(radius_mm, spacing):
    """The rule for a structure of the given radius: "lattice" when it is at least ``MIN_LATTICE_RADIUS_CELLS`` of
    the largest spacing, otherwise "closed form" (an inclusion; it must then be left out of the lattice)."""
    return "lattice" if radius_mm >= MIN_LATTICE_RADIUS_CELLS * max(_three(spacing)) else "closed form"


# ------------------------------------------------------------------------------------------------ shim
def shim_basis(y, x, z, order=2, b0_dir=(0.0, 0.0, 1.0)):
    """Solid harmonics up to ``order`` (0, 1 or 2) at the patient positions (y, x, z) in mm: [terms, ...], in the
    order of ``SHIM_TERMS``: 1; x, y, z; z^2 - (x^2 + y^2) / 2, z x, z y, x^2 - y^2, x y, in mm^order. Each has zero
    Laplacian, as the field of a coil outside the volume must. The harmonics are those of the magnet, whose z axis
    is B0: for another ``b0_dir`` the coordinates are turned so that z runs along it."""
    y, x, z = np.broadcast_arrays(*(np.asarray(v, float) for v in (y, x, z)))
    b = _unit(b0_dir)
    if abs(b[2] - 1.0) > 1e-12:                                         # magnet frame: z' = b, x' = patient x made normal to b
        ex = np.array([0.0, 1.0, 0.0]) if abs(b[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        ex = ex - (ex @ b) * b
        ex /= np.linalg.norm(ex)
        ey = np.cross(b[[1, 0, 2]], ex[[1, 0, 2]])[[1, 0, 2]]           # cross product in (x, y, z), back to (y, x, z)
        y, x, z = (y * e[0] + x * e[1] + z * e[2] for e in (ey, ex, b))
    terms = [np.ones_like(x)]
    if order >= 1:
        terms += [x, y, z]
    if order >= 2:
        terms += [z * z - 0.5 * (x * x + y * y), z * x, z * y, x * x - y * y, x * y]
    if order not in (0, 1, 2):
        raise ValueError("shim order must be 0, 1 or 2")
    return np.stack(terms)


def shim_limits_hz(limits_ut=None):
    """``limits_ut`` ({term: microtesla per metre^order}, default ``SHIM_LIMITS_UT``) as the largest |coefficient| of
    each term of ``SHIM_TERMS`` in Hz / mm^order (infinite for F0 and for a term that is not named)."""
    lim = SHIM_LIMITS_UT if limits_ut is None else limits_ut
    return np.array([lim[t] * 1e-6 * GAMMA_HZ_PER_T * 1e-3 ** o if t in lim else np.inf for t, o in zip(SHIM_TERMS, SHIM_ORDER)])


def shim_fit(basis, values, limit=None):
    """Least-squares shim: coefficients c that minimise |values - sum_j c_j basis_j|^2, with |c_j| <= limit_j.

    ``basis`` [terms, n] (``shim_basis`` at the points of the volume of interest), ``values`` [n] in Hz, ``limit``
    [terms] or None. Returns dict(coef [terms] in Hz / mm^order, residual [n], sd_before, sd_after, at_limit (indices),
    kkt (largest violation of the optimality conditions, relative; 0 is optimal)). The coefficients are ABSOLUTE:
    the field handed in must be the field without any shim."""
    a = np.asarray(basis, float).reshape(len(basis), -1).T
    v = np.asarray(values, float).ravel()
    scale = np.sqrt((a * a).mean(0))
    live = scale > 0                                                    # a term that vanishes on the points gets no current
    s = np.where(live, scale, 1.0)
    an = a[:, live] / s[live]
    coef = np.zeros(a.shape[1])
    if limit is None or not np.any(np.isfinite(np.asarray(limit, float)[live])):
        coef[live] = np.linalg.lstsq(an, v, rcond=None)[0]
        bound = np.full(a.shape[1], np.inf)
    else:
        import scipy.optimize
        bound = np.asarray(limit, float)
        q, r = np.linalg.qr(an)                                         # same minimiser on a square system
        hi = bound[live] * s[live]
        coef[live] = scipy.optimize.lsq_linear(r, q.T @ v, bounds=(-hi, hi), method="bvls", tol=1e-13).x
    coef[live] = coef[live] / s[live]
    resid = v - a @ coef
    grad = (an.T @ resid) / max(np.linalg.norm(v) * math.sqrt(len(v)), 1e-300)      # minus the gradient, scaled columns
    at = np.isfinite(bound[live]) & (np.abs(coef[live]) >= bound[live] * (1 - 1e-9))
    sign = np.sign(coef[live])
    viol = np.where(at, np.maximum(-grad * sign, 0.0), np.abs(grad))    # at a limit the pull must point outwards
    return dict(coef=coef, residual=resid, sd_before=float(v.std()), sd_after=float(resid.std()),
                at_limit=[int(i) for i in np.flatnonzero(live)[at]], kkt=float(viol.max()) if viol.size else 0.0)


# ------------------------------------------------------------------------------------------------ the field of a world
def _continue(values, known, reach=2):
    """Continue ``values`` from the cells where ``known`` outwards through ``reach`` cells: the mean of the known
    cells within that distance along each axis."""
    m = known.astype(np.float32)
    g = np.where(known, values, 0).astype(np.float32)
    num, den = ndi.uniform_filter(g, 2 * reach + 1, mode="nearest"), ndi.uniform_filter(m, 2 * reach + 1, mode="nearest")
    new = ~known & (den > 1e-4)
    g[new] = num[new] / den[new]
    return g


def _trilinear(arrays, idx):
    """Trilinear interpolation of several lattices at the same fractional indices ``idx`` [3, ...] (clamped)."""
    shape = arrays[0].shape
    i0, f = [], []
    for a in range(3):
        v = np.clip(idx[a], 0.0, shape[a] - 1.0)
        i = np.minimum(v.astype(np.intp), shape[a] - 2) if shape[a] > 1 else np.zeros(v.shape, np.intp)
        i0.append(i)
        f.append((v - i).astype(np.float32))
    sy, sx = shape[1] * shape[2], shape[2]
    base = i0[0] * sy + i0[1] * sx + i0[2]
    step = (sy if shape[0] > 1 else 0, sx if shape[1] > 1 else 0, 1 if shape[2] > 1 else 0)
    out = [np.zeros(base.shape, np.float32) for _ in arrays]
    for dz in (0, 1):
        wz = f[0] if dz else 1 - f[0]
        for dy in (0, 1):
            wzy = wz * (f[1] if dy else 1 - f[1])
            for dx in (0, 1):
                w = wzy * (f[2] if dx else 1 - f[2])
                at = base + dz * step[0] + dy * step[1] + dx * step[2]
                for o, arr in zip(out, arrays):
                    o += w * arr.ravel()[at]
    return out


class Field3D:
    """The static field of one world on a lattice, evaluable at any patient position (module docstring).

    Attributes (lattices [nz, ny, nx], ppm): ``delta`` the dipole field, ``chi`` the susceptibility, ``kappa`` the
    step factor 1/3 - (n . b)^2, ``smooth`` = delta - kappa chi (what is interpolated), ``chi_seen`` the susceptibility
    of the signal-bearing material, continued outwards. ``shim_coef``: coefficients of ``SHIM_TERMS`` in Hz / mm^order
    that ``hz`` subtracts (zero until ``shim`` is called). ``inclusions``: closed-form terms added at the points."""

    def __init__(self, chi, lattice: Lattice, b0_t, b0_dir=(0.0, 0.0, 1.0), tissue=None, chi_tissue=None,
                 background_ppm=DCHI_AIR_PPM, pad=1.5, extrude=(), inclusions=(), images=True):
        self.lattice = Lattice(tuple(lattice.shape), _three(lattice.spacing), tuple(float(v) for v in lattice.origin))
        self.b0_t, self.b0_dir = float(b0_t), _unit(b0_dir)
        self.chi = np.asarray(chi, np.float32)
        if self.chi.shape != self.lattice.shape:
            raise ValueError("chi does not have the shape of the lattice")
        self.delta = dipole_field(self.chi, self.lattice.spacing, self.b0_dir, pad, background_ppm, True, extrude, images).astype(np.float32)
        b = self.b0_dir
        g = [ndi.uniform_filter(c, 3, mode="nearest") for c in np.gradient(self.chi, *self.lattice.spacing)]
        g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2]
        gb = g[1] * b[0] + g[2] * b[1] + g[0] * b[2]
        steep = g2 > 1e-8 * float(g2.max()) if g2.size and g2.max() > 0 else np.zeros(g2.shape, bool)
        self.kappa = np.where(steep, 1.0 / 3.0 - gb * gb / np.where(steep, g2, 1.0), 0.0).astype(np.float32)
        self.smooth = self.delta - self.kappa * self.chi
        if tissue is None:                                              # no material named: every cell sees its own mean (plain interpolation)
            self.tissue, self.chi_seen = np.ones(self.chi.shape, np.float32), self.chi
        else:
            self.tissue = np.asarray(tissue, np.float32)
            seen = np.zeros_like(self.chi) if chi_tissue is None else np.asarray(chi_tissue, np.float32)
            self.chi_seen = _continue(seen, self.tissue > 0.02)
        self.inclusions = [dict(i) for i in inclusions]
        self.shim_coef = np.zeros(len(SHIM_TERMS))
        self.shim_report = None

    @property
    def hz_per_ppm(self):
        return GAMMA_HZ_PER_T * self.b0_t * 1e-6

    def ppm(self, y, x, z, chi_seen=None):
        """delta in ppm at patient positions, before any shim. ``chi_seen``: the susceptibility (ppm) of the material
        that gives the signal at each point (``seen_chi`` of the plane's class weights); None: the lattice's own."""
        idx = self.lattice.index(y, x, z)
        if chi_seen is None:
            smooth, kappa, seen = _trilinear((self.smooth, self.kappa, self.chi_seen), idx)
        else:
            (smooth, kappa), seen = _trilinear((self.smooth, self.kappa), idx), np.asarray(chi_seen, np.float32)
        out = smooth + kappa * seen
        for inc in self.inclusions:
            out = out + inclusion_field(inc, y, x, z, self.b0_dir)
        return out

    def hz(self, y, x, z, chi_seen=None, shim=True):
        """Off-resonance in Hz at the world's field strength at patient positions, after the stored shim."""
        out = self.ppm(y, x, z, chi_seen) * self.hz_per_ppm
        if shim and np.any(self.shim_coef):
            out = out - np.tensordot(self.shim_coef, shim_basis(y, x, z, 2, self.b0_dir), 1)
        return out

    def lattice_hz(self, shim=False):
        """The field of the signal-bearing material at the cell centres, Hz: what a field map of the object shows."""
        y, x, z = self.lattice.mesh()
        out = (self.smooth + self.kappa * self.chi_seen).astype(np.float64)
        for inc in self.inclusions:
            out = out + inclusion_field(inc, y, x, z, self.b0_dir)
        out = out * self.hz_per_ppm
        if shim and np.any(self.shim_coef):
            out = out - np.tensordot(self.shim_coef, shim_basis(y, x, z, 2, self.b0_dir), 1)
        return out

    def shim(self, voi, order=1, limits_ut=None, bounded=True):
        """Fit the shim over the cells of the boolean lattice ``voi`` and store it: terms up to ``order``, each within
        its limit (``SHIM_LIMITS_UT`` or ``limits_ut``; ``bounded=False``: no limits). The fit is made on the field
        without shim, so the stored coefficients are absolute. Returns the report of ``shim_fit`` plus ``terms``,
        ``coef_ut`` (microtesla per metre^order) and ``at_limit`` as names."""
        voi = np.asarray(voi, bool)
        if voi.shape != self.lattice.shape or not voi.any():
            raise ValueError("the volume of interest is a boolean lattice with at least one cell")
        y, x, z = (np.broadcast_to(v, voi.shape)[voi] for v in self.lattice.mesh())
        n = len(shim_basis(0.0, 0.0, 0.0, order))
        rep = shim_fit(shim_basis(y, x, z, order, self.b0_dir), self.lattice_hz()[voi], shim_limits_hz(limits_ut)[:n] if bounded else None)
        self.shim_coef = np.zeros(len(SHIM_TERMS))
        self.shim_coef[:n] = rep["coef"]
        rep.update(terms=SHIM_TERMS[:n], at_limit=[SHIM_TERMS[i] for i in rep["at_limit"]], order=order,
                   coef_ut=[c / (1e-6 * GAMMA_HZ_PER_T * 1e-3 ** o) for c, o in zip(rep["coef"], SHIM_ORDER)])
        self.shim_report = {k: v for k, v in rep.items() if k != "residual"}
        return rep


def from_weights(weights, chi_ppm, lattice: Lattice, b0_t, b0_dir=(0.0, 0.0, 1.0), background_ppm=DCHI_AIR_PPM, **kw) -> Field3D:
    """The field of a world from its class weights [class, nz, ny, nx] (cell means) and one susceptibility per class
    in ppm relative to soft tissue. Keywords as ``Field3D`` (``pad``, ``extrude``, ``inclusions``, ``images``)."""
    chi, tissue, chi_tissue = susceptibility(weights, chi_ppm, background_ppm)
    return Field3D(chi, lattice, b0_t, b0_dir, tissue, chi_tissue, background_ppm, **kw)


def seen_chi(weights, chi_ppm):
    """Susceptibility of the signal-bearing material at each point of an image plane: the weighted mean over the
    classes ``weights`` [class, ...] with ``chi_ppm`` per class (0 where there is no material)."""
    w = np.asarray(weights, np.float32)
    total = w.sum(0)
    return np.divide(np.tensordot(np.asarray(chi_ppm, np.float32), w, 1), total, out=np.zeros_like(total), where=total > 1e-3)


# ------------------------------------------------------------------------------------------------ image planes
def _frame(frame):
    """(origin, row, column, normal) of a plane, patient components (y, x, z): a ``volume.Frame`` (attributes
    ``origin, row, col, normal``), an object with ``o, r, c`` (and optionally ``n``), or a tuple (origin, row, column
    [, normal]). Without a normal it is the cross product of row and column in the component order y, x, z."""
    if hasattr(frame, "origin"):
        parts = (frame.origin, frame.row, frame.col, getattr(frame, "normal", None))
    elif hasattr(frame, "o"):
        parts = (frame.o, frame.r, frame.c, getattr(frame, "n", None))
    else:
        parts = tuple(frame) + (None,) * (4 - len(frame))
    o, r, c = np.asarray(parts[0], float), _unit(parts[1]), _unit(parts[2])
    n = np.cross(r, c) if parts[3] is None else _unit(parts[3])
    if max(abs(float(r @ c)), abs(float(r @ n)), abs(float(c @ n))) > 1e-9:
        raise ValueError("the row, the column and the normal of a frame must be perpendicular")
    return o, r, c, n


def plane(name, pos_mm=0.0):
    """Frame (origin, row, column, normal) of a standard plane through ``pos_mm`` on its patient axis, as
    ``volume.Frame.plane``: "axial": rows along y, columns along x, normal +z (the engine's slice); "coronal": rows
    from superior to inferior, columns along x, normal +y; "sagittal": rows from superior to inferior, columns along
    y, normal +x."""
    row, col, n = dict(axial=((1, 0, 0), (0, 1, 0), (0, 0, 1)), coronal=((0, 0, -1), (0, 1, 0), (1, 0, 0)),
                       sagittal=((0, 0, -1), (1, 0, 0), (0, 1, 0)))[name]
    return tuple(float(v) * pos_mm for v in n), tuple(float(v) for v in row), tuple(float(v) for v in col), tuple(float(v) for v in n)


def plane_points(frame, grid, fov, offset_mm=0.0):
    """Patient positions (y, x, z), each [grid, grid], of the engine's master grid on the plane ``frame`` moved
    ``offset_mm`` along its normal: index i of an axis at (i - grid // 2) x fov / grid."""
    o, r, c, n = _frame(frame)
    u = (np.arange(grid) - grid // 2) * (fov / grid)
    o = o + offset_mm * n
    return tuple(o[i] + u[:, None] * r[i] + u[None, :] * c[i] for i in range(3))


def sample(field: Field3D, frame, grid, fov, thickness, chi_seen=None, offsets=None):
    """The field on an image plane, in the form of ``world["fields"]``: dict(b0_hz [grid, grid] in Hz at the world's
    field strength, after the stored shim; dbdz_hz_per_mm: its gradient along the plane's normal).

    The gradient is the secant across the slice, (f(+thickness / 2) - f(-thickness / 2)) / thickness: the linear
    term the engine's dephasing model asks for, taken over the range it is applied to. ``chi_seen`` [grid, grid]:
    the susceptibility of the tissue at each grid point (``seen_chi``), which makes the field exact at an interface
    on the grid's own resolution; None uses the lattice's. The same ``chi_seen`` is used on the planes off the
    centre: the field there is the continuation of the field of the slice's own tissue. ``offsets`` (mm along the
    normal): also return ``stack`` [len(offsets), grid, grid], the field on each of those planes (for a phase per
    plane of a profile). Positions outside the lattice take the value of its nearest face."""
    if not thickness > 0:
        raise ValueError("thickness must be positive")
    f = lambda off: field.hz(*plane_points(frame, grid, fov, off), chi_seen=chi_seen)      # noqa: E731
    out = dict(b0_hz=f(0.0), dbdz_hz_per_mm=(f(0.5 * thickness) - f(-0.5 * thickness)) / thickness)
    if offsets is not None:
        out["stack"] = np.stack([f(float(o)) for o in offsets])
    return out


# ------------------------------------------------------------------------------------------------ transmit and receive
class Profile3D:
    """Declared transmit and receive shapes in three dimensions. Each is a polynomial of the patient position p =
    (y, x, z): B1+ ratio and receive magnitude c0 + g . p + p . Q p (clipped), receive phase linear. The interface
    is the two methods ``b1(y, x, z)`` and ``rx(y, x, z)``: a computed map (``mrgym.em``, on its own grid) replaces
    this object through an adapter with the same two methods (not written)."""

    def __init__(self, b1, rx_mag, rx_phase, b1_clip=(0.3, 1.7), rx_clip=None):
        self.b1_poly, self.rx_poly, self.rx_phase, self.b1_clip, self.rx_clip = b1, rx_mag, rx_phase, b1_clip, rx_clip

    @staticmethod
    def _poly(poly, p):
        c0, g, q = poly
        out = c0 + g[0] * p[0] + g[1] * p[1] + g[2] * p[2]
        for i in range(3):
            for j in range(3):
                if q[i][j]:
                    out = out + q[i][j] * p[i] * p[j]
        return out

    def b1(self, y, x, z):
        p = np.broadcast_arrays(*(np.asarray(v, float) for v in (y, x, z)))
        return np.clip(self._poly(self.b1_poly, p), *self.b1_clip)

    def rx(self, y, x, z):
        p = np.broadcast_arrays(*(np.asarray(v, float) for v in (y, x, z)))
        mag = self._poly(self.rx_poly, p)
        if self.rx_clip:
            mag = np.clip(mag, *self.rx_clip)
        ph0, gp = self.rx_phase
        return mag * np.exp(1j * (ph0 + gp[0] * p[0] + gp[1] * p[1] + gp[2] * p[2]))

    def on_plane(self, frame, grid, fov):
        """dict(b1, rx) on the engine's master grid of a plane."""
        p = plane_points(frame, grid, fov)
        return dict(b1=self.b1(*p), rx=self.rx(*p))


def declared_profiles(kind, b0_t, size_mm, rng, rng_z=None, z_centre_mm=0.0) -> Profile3D:
    """The declared transmit and receive shapes of ``world.anatomy_fields`` ("brain", "abdomen") continued along z.

    ``size_mm`` = (extent along y, along x, along z) of the object. The in-plane parameters are drawn from ``rng``
    in the order of ``anatomy_fields`` (after its smooth field term), so the same generator gives the same shapes:
    on the plane z = ``z_centre_mm`` the result equals the 2D world's ``b1`` and ``rx``. Along z (a DECLARED
    continuation, like the in-plane shapes themselves): the radial terms count (z - z_centre) / (size_z / 2) as a
    third coordinate, and the tilts and the phase ramp get a z component drawn from ``rng_z`` (None: no z tilt)."""
    if kind not in B1_SHAPE:
        raise ValueError(f"declared profiles exist for {tuple(B1_SHAPE)}")
    ry, rx_, rz = (0.5 * float(s) for s in size_mm)
    inv = np.array([1 / ry, 1 / rx_, 1 / rz])
    amp = B1_SHAPE[kind][float(b0_t)] * rng.uniform(0.8, 1.2)
    quad = np.zeros((3, 3))
    if kind == "abdomen":                                              # quadrupolar shading in the axial plane
        phi0 = rng.uniform(-0.5, 0.5)
        c2, s2 = math.cos(2 * phi0), math.sin(2 * phi0)
        quad[0, 0], quad[1, 1], quad[0, 1] = -amp * c2 - 0.3 * amp, amp * c2 - 0.3 * amp, 2 * amp * s2
        quad[2, 2] = -0.3 * amp
        c0, tilt = 1 + 0.15 * amp, np.zeros(3)
    else:                                                              # central brightening
        tilt = 0.3 * amp * np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0])
        quad[0, 0] = quad[1, 1] = quad[2, 2] = -2 * amp
        c0 = 1 + amp
    bowl = rng.uniform(0.1, 0.3)
    rtilt = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.0])
    ph0 = rng.uniform(0, 2 * np.pi)
    ptilt = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0])
    if rng_z is not None:
        tilt[2] = 0.0 if kind == "abdomen" else 0.3 * amp * rng_z.uniform(-1, 1)
        rtilt[2], ptilt[2] = rng_z.uniform(-0.1, 0.1), rng_z.uniform(-1, 1)
    centre = np.array([0.0, 0.0, float(z_centre_mm)])

    def poly(c, g, q):                                                  # scaled coordinates (p - centre) * inv -> polynomial of p
        g, q = np.asarray(g) * inv, np.asarray(q) * np.outer(inv, inv)
        return (float(c - g @ centre + centre @ q @ centre), g - (q + q.T) @ centre, q)

    rq = bowl * np.eye(3)
    gp = ptilt * inv
    return Profile3D(poly(c0, tilt, quad), poly(1.0, rtilt, rq), (float(ph0 - gp @ centre), gp))


# ------------------------------------------------------------------------------------------------ susceptibility mapping
def to_phase(field_hz, te_ms, wrap=True, phase0=0.0):
    """Phase (rad) of a gradient echo at each echo time: phase0 + 2 pi f TE, the engine's sign (a faster spin gains
    positive phase). ``te_ms``: one or several echo times; returns [echoes, ...]. ``wrap``: into (-pi, pi]."""
    te = np.atleast_1d(np.asarray(te_ms, float)) * 1e-3
    ph = phase0 + 2 * np.pi * te.reshape((-1,) + (1,) * np.ndim(field_hz)) * np.asarray(field_hz, float)[None]
    return np.angle(np.exp(1j * ph)) if wrap else ph


def tkd(delta, spacing, b0_dir=(0.0, 0.0, 1.0), threshold=0.1):
    """Reference inversion of the dipole convolution by truncated k-space division (Shmueli 2009): chi(k) =
    delta(k) / D(k) where |D| > threshold, delta(k) sign(D) / threshold elsewhere. The field of the whole lattice
    must be given (no mask). It underestimates: the mean factor is ``tkd_factor(threshold)``."""
    delta = np.asarray(delta)
    d = dipole_kernel(delta.shape, spacing, b0_dir, True, np.float64)
    inv = np.where(np.abs(d) > threshold, 1.0 / np.where(d == 0, 1.0, d), np.sign(d) / threshold)
    return sfft.irfftn(sfft.rfftn(delta, workers=1) * inv, s=delta.shape, workers=1)


def tkd_factor(threshold):
    """Mean of D(k) x (truncated inverse) over all directions of k: the factor by which truncated division scales
    the susceptibility of a compact object. With u = cos(angle to B0) uniform on (0, 1) and D = 1/3 - u^2:
    1 - (u2 - u1) + (1 / threshold) int_{u1}^{u2} |1/3 - u^2| du,  u1,2 = sqrt(1/3 -+ threshold)."""
    t = float(threshold)
    if not 0 < t < 1.0 / 3.0:
        raise ValueError("threshold must lie between 0 and 1/3")
    u1, u0, u2 = math.sqrt(1 / 3 - t), math.sqrt(1 / 3), math.sqrt(1 / 3 + t)
    prim = lambda u: u / 3 - u ** 3 / 3                                 # noqa: E731
    return 1 - (u2 - u1) + ((prim(u0) - prim(u1)) + (prim(u0) - prim(u2))) / t


def validate(markdown=False):
    """Print the validation table of this module (``python -m mrgym.scanner.fields3d_validate``); True if all pass."""
    from . import fields3d_validate
    return fields3d_validate.main(["--markdown"] if markdown else []) == 0
