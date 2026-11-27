"""Validation of ``mrgym.scanner.fields3d``: every function against a closed form or an independent computation.

    python -m mrgym.scanner.fields3d_validate               the table as text; exit status 1 if a row fails
    python -m mrgym.scanner.fields3d_validate --markdown    the table of docs/physics/fields3d.md
    python -m mrgym.scanner.fields3d_validate --head        the head figures of docs/physics/fields3d.md

Errors of a field are in units of the susceptibility step (dchi = 1) unless a unit is printed. The lattice of a row
is part of its text: 128^3 cells of 2 mm, transformed at 192^3, unless the row says otherwise. Shapes enter the
lattice as cell means (``fields3d.volume_fraction``). The closed forms are those of Schenck 1996 (sphere, cylinder)
and of the demagnetising factors of an ellipsoid; the rows show that the code computes them, not that a head obeys
them.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import scipy.fft as sfft
import scipy.ndimage as ndi

from . import fields3d as F
from .world import GAMMA_HZ_PER_T

GROUPS = dict(kernel="The dipole kernel against closed forms", lattice="The periodic lattice: padding, spacing, cut faces",
              surface="Surfaces and image planes", shim="Shim", extras="Inclusions, profiles, susceptibility mapping, time")
MAGIC = math.acos(math.sqrt(1.0 / 3.0))
A = 24.0                                                # radius of the test sphere and cylinder, mm


def _row(id, group, quantity, reference, measured, tolerance, kind="max", unit=""):
    """kind "max": passes when measured <= tolerance; "min": when measured >= tolerance."""
    measured = float(measured)
    ok = bool(np.isfinite(measured) and (measured <= tolerance if kind == "max" else measured >= tolerance))
    return dict(id=id, group=group, quantity=quantity, reference=reference, measured=measured, tolerance=float(tolerance), kind=kind,
                unit=unit, ok=ok)


def _rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def ball(lat, radius, centre=(0.0, 0.0, 0.0)):
    """Cell means of a sphere; ``centre`` (y, x, z)."""
    return F.volume_fraction(lambda y, x, z: np.sqrt((y - centre[0]) ** 2 + (x - centre[1]) ** 2 + (z - centre[2]) ** 2) - radius, lat)


def rod(lat, radius, axis):
    """Cell means of an infinite cylinder through the origin along the lattice axis ``axis`` ("z", "y" or "x")."""
    def sdf(y, x, z):
        p = dict(y=y, x=x, z=z)
        return np.sqrt(sum(p[k] ** 2 for k in "zyx" if k != axis) + 0 * p[axis]) - radius
    return F.volume_fraction(sdf, lat)


def spheroid(lat, a, c, tilt=0.0):
    """Cell means of a spheroid, semi-axis ``c`` along its axis and ``a`` across; the axis lies in the y-z plane at
    ``tilt`` from z. The implicit function is not a distance: it is scaled to underestimate it."""
    ct, st = math.cos(tilt), math.sin(tilt)

    def sdf(y, x, z):
        u, v = z * ct + y * st, y * ct - z * st
        return (np.sqrt((u / c) ** 2 + (v / a) ** 2 + (x / a) ** 2) - 1.0) * min(a, c)
    return F.volume_fraction(sdf, lat)


def _soft_ball(lat, radius, edge):
    """The lattice of the specification's prototype: a tanh edge of half-width ``edge`` sampled at the cell centres."""
    y, x, z = lat.mesh()
    return (0.5 * (1 + np.tanh((radius - np.sqrt(y * y + x * x + z * z)) / edge))).astype(np.float32)


def _shell_points(rng, n, r_lo, r_hi):
    """Random points (y, x, z) with a uniform direction and a radius uniform in [r_lo, r_hi]."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * rng.uniform(r_lo, r_hi, n)[:, None]).T


# ---------------------------------------------------------------------------------------------- kernel
def check_kernel():
    rows = []
    lat = F.Lattice.centred((128,) * 3, 2.0)
    y, x, z = lat.mesh()
    r = np.sqrt(y * y + x * x + z * z) + 0 * (y + x + z)
    w = ball(lat, A)
    f = F.dipole_field(w, 2.0, pad=(192,) * 3, background=0.0)
    exact = F.sphere_field(y, x, z, (0, 0, 0), A)
    outside, inside = (r > A + 3) & (r < 3 * A), r < A - 3
    rows.append(_row("F1", "kernel", "sphere, radius 24 mm (12 cells): field outside, from 3 mm off the surface to 3 radii, rms error",
                     "(1/3)(a/r)^3 (3 cos^2 t - 1); the specification reached 0.0004 (0.00022 padded)", _rms((f - exact)[outside]), 2.5e-4))
    rows.append(_row("F2", "kernel", "the same sphere: field inside, more than 3 mm from the surface, standard deviation (mean "
                     f"{abs(float(f[inside].mean())):.0e})", "0 with the sphere of Lorentz: uniform", float(f[inside].std()), 2e-3))
    soft = _soft_ball(lat, A, 1.0)
    fs = F.dipole_field(soft, 2.0, pad=1.0, background=0.0, images=False)
    rows.append(_row("F3", "kernel", "the specification's own lattice (edge of half-width 1 mm sampled at the cell centres, no padding): "
                     "larger of the outside rms error / 0.0004 and the inside standard deviation / 0.0008",
                     "the specification's figures 0.00038 and 0.00073", max(_rms((fs - exact)[outside]) / 4e-4, float(fs[inside].std()) / 8e-4), 1.0,
                     unit="of the bound"))
    c = 64
    on = [f[c + k, c, c] - F.sphere_field(0, 0, 2.0 * k, (0, 0, 0), A) for k in (18, 24)]
    eq = [f[c, c, c + k] - F.sphere_field(0, 2.0 * k, 0, (0, 0, 0), A) for k in (18, 24)]
    rows.append(_row("F4", "kernel", "the same sphere at four lattice points: on the B0 axis and on the equator at 1.5 and 2 radii",
                     "0.19753, 0.08333 (along B0); -0.09877, -0.04167 (across)", max(abs(float(v)) for v in on + eq), 1e-3))
    fm = F.dipole_field(w, 2.0, pad=(192,) * 3, background=0.0, lorentz=False, images=False)
    frac = float(w.sum()) / 192 ** 3
    rows.append(_row("F5", "kernel", "the same sphere without the sphere of Lorentz: mean inside",
                     "2/3 (Schenck: B inside = B0 (1 + 2 chi / 3)), less 2/3 of the volume fraction of the box, the mean that k = 0 removes",
                     abs(float(fm[inside].mean()) - 2.0 / 3.0 * (1 - frac)), 2e-4))
    # ---- infinite cylinder: along a lattice axis, the box periodic along it; the angle is that of B0
    lat2 = F.Lattice.centred((16, 128, 128), 2.0)                        # 16 cells along the axis are as good as any number
    wz = rod(lat2, A, "z")
    y2, x2, z2 = lat2.mesh()
    rho = np.sqrt(y2 * y2 + x2 * x2) + 0 * z2
    area = float(wz.sum()) / wz.shape[0]                                 # cells per cross-section
    for id_, name, theta in (("F6", "parallel to B0", 0.0), ("F7", "at 30 degrees to B0", math.radians(30.0)),
                             ("F8", "at the magic angle, 54.74 degrees", MAGIC), ("F9", "perpendicular to B0", math.pi / 2)):
        b = (math.sin(theta), 0.0, math.cos(theta))                     # B0 tilted from the cylinder (z) towards y
        fc = F.dipole_field(wz, 2.0, b, pad=(16, 256, 256), background=0.0, images=False)
        ex = F.cylinder_field(y2, x2, z2, (0, 0, 0), (0, 0, 1), A, 1.0, b)
        v_in = (3 * math.cos(theta) ** 2 - 1) / 6
        off = -area / 256 ** 2 * v_in                                    # the offset of a periodic square cell: -(area fraction) x inside
        m_in, m_out = rho < A - 3, (rho > A + 3) & (rho < 2.2 * A)
        rows.append(_row(id_, "kernel", f"infinite cylinder, radius 24 mm, {name}, cross-section of the box 512 mm: larger of the error of the "
                         f"mean inside and the rms error outside (3 mm off the surface to 2.2 radii)",
                         f"inside (3 cos^2 t - 1) / 6 = {v_in:+.5f}, outside (1/2) sin^2 t (a/rho)^2 cos 2 phi, both plus the offset "
                         f"-(area fraction) x inside = {off:+.5f}", max(abs(float(fc[m_in].mean()) - v_in - off), _rms((fc - ex - off)[m_out])), 1e-3))
    # the offset and the images of the periodic cell, unpadded cross-section (256 mm): what the closed form does not have
    b = (1.0, 0.0, 0.0)
    fc = F.dipole_field(wz, 2.0, b, pad=1.0, background=0.0, images=False)
    ex = F.cylinder_field(y2, x2, z2, (0, 0, 0), (0, 0, 1), A, 1.0, b)
    fr = area / 128 ** 2
    m_out = (rho > A + 3) & (rho < 2.2 * A)
    resid = (fc - ex - fr / 6)[m_out]
    shape = ((rho / 256.0) ** 2 * (2 * y2 * y2 / np.maximum(rho * rho, 1e-9) - 1))[m_out]      # (rho / L)^2 cos 2 phi, phi from B0
    coef = float(resid @ shape / (shape @ shape)) / fr
    rows.append(_row("F10", "kernel", "the same cylinder across B0 in an unpadded box (256 mm): the lattice minus the closed form outside is "
                     "an offset plus the field of the images; fitted amplitude of (rho / L)^2 cos 2 phi over the area fraction",
                     "offset -(area fraction) x inside, exact for a square cell; images 1.505 x area fraction x sin^2 t (rho / L)^2 cos 2 phi "
                     "(lattice sum of line dipoles, derived separately)", abs(coef / 1.505 - 1), 0.05, unit="relative"))
    rows.append(_row("F11", "kernel", "the same case after both terms: rms of what is left outside", "0", _rms(resid - 1.505 * fr * shape), 1e-3))
    # a cell that is not square: the constant depends on the direction of B0 in the cell
    lat6 = F.Lattice.centred((8, 160, 256), 2.0)                         # cross-section 320 mm (y) x 512 mm (x)
    wr = rod(lat6, A, "z")
    y6, x6, z6 = lat6.mesh()
    core = (np.sqrt(y6 * y6 + x6 * x6) + 0 * z6) < A - 12
    fr6 = float(wr.sum()) / 8 / (160 * 256)
    s_cell = np.pi * 1.6 * (1.0 / 3.0 - 2.0 * sum(1.0 / math.sinh(np.pi * n * 1.6) ** 2 for n in range(1, 30)))
    worst = 0.0
    for b, cos2a in (((1, 0, 0), 1.0), ((0, 1, 0), -1.0)):                # B0 along the short edge, along the long edge
        got = float(F.dipole_field(wr, 2.0, b, pad=1.0, background=0.0, images=False)[core].mean()) + 1.0 / 6.0
        worst = max(worst, abs(got - fr6 * (1.0 / 6.0 + 0.5 * cos2a * (s_cell - 1.0))))
    rows.append(_row("F11b", "kernel", "the same cylinder across B0 in a cell of 320 x 512 mm, B0 along the short and along the long edge: "
                     f"constant inside against its formula (it is {fr6 * (1 / 6 + 0.5 * (s_cell - 1)):+.5f} and "
                     f"{fr6 * (1 / 6 - 0.5 * (s_cell - 1)):+.5f}; the square-cell rule would give {fr6 / 6:+.5f} for both)",
                     "(area fraction) x [-(3 cos^2 t - 1) / 6 + (1/2) sin^2 t cos 2 alpha (s - 1)], s = (pi L2 / L1) [1/3 - 2 sum_n sinh^-2(pi n "
                     "L2 / L1)], alpha the angle of B0 from the edge L1 (derived separately; s = 1 for a square)", worst, 1e-5))
    # ---- shell
    ws = ball(lat, A) - ball(lat, 16.0)
    fsh = F.dipole_field(ws, 2.0, pad=(192,) * 3, background=0.0)
    exs = F.shell_field(y, x, z, (0, 0, 0), 16.0, A)
    rows.append(_row("F12", "kernel", "spherical shell, radii 16 and 24 mm (4 cells thick): larger of the rms error in the cavity (r < 13 mm), "
                     "in the shell (19 to 21 mm) and outside (27 to 72 mm)",
                     "cavity 0; shell -(1/3)(a1/r)^3 (3 cos^2 t - 1); outside (1/3)((a2^3 - a1^3)/r^3)(3 cos^2 t - 1)",
                     max(_rms((fsh - exs)[r < 13]), _rms((fsh - exs)[(r > 19) & (r < 21)]), _rms((fsh - exs)[outside])), 6e-3))
    # ---- superposition, linearity, the reference of chi
    lat3 = F.Lattice.centred((48,) * 3, 2.0)
    w1, w2 = ball(lat3, 12.0, (6.0, -4.0, 2.0)).astype(float), ball(lat3, 8.0, (-14.0, 10.0, -8.0)).astype(float)
    f1, f2 = (F.dipole_field(v, 2.0, background=0.0) for v in (w1, w2))
    both = F.dipole_field(2 * w1 - 3 * w2, 2.0, background=0.0)
    shifted = F.dipole_field(2 * w1 - 3 * w2 + 5.0, 2.0, background=5.0)
    rows.append(_row("F13", "kernel", "superposition and linearity: field of 2 chi1 - 3 chi2 against 2 field1 - 3 field2; and a constant added to "
                     "chi and to its surroundings (double precision)", "equal; no change (only differences of chi matter)",
                     max(np.abs(both - 2 * f1 + 3 * f2).max(), np.abs(shifted - both).max()), 1e-12))
    # ---- direction of B0
    obj = spheroid(lat3, 10.0, 20.0, 0.0).astype(float) + w2
    fx = F.dipole_field(obj, 2.0, (0, 1, 0), background=0.0)             # B0 along x
    fz = F.dipole_field(obj.transpose(2, 1, 0), 2.0, (0, 0, 1), background=0.0).transpose(2, 1, 0)   # the object turned instead (x <-> z)
    fy = F.dipole_field(obj, 2.0, (1, 0, 0), background=0.0)
    fz2 = F.dipole_field(obj.transpose(1, 0, 2), 2.0, (0, 0, 1), background=0.0).transpose(1, 0, 2)
    rows.append(_row("F14", "kernel", "B0 along x, and along y, against the object turned by 90 degrees with B0 along z (an object without "
                     "symmetry, double precision)", "equal: turning B0 is turning the object", max(np.abs(fx - fz).max(), np.abs(fy - fz2).max()), 1e-12))
    lat4 = F.Lattice.centred((96,) * 3, 2.0)
    y4, x4, z4 = lat4.mesh()
    worst_closed, worst_pair = 0.0, 0.0
    n_c, n_a = F.demagnetising_factors((18.0, 18.0, 36.0))[2], F.demagnetising_factors((18.0, 18.0, 36.0))[0]
    core0 = spheroid(lat4, 12.0, 30.0, 0.0) > 0.999                     # well inside the upright spheroid
    for tilt in (math.radians(30.0), MAGIC, math.pi / 2):
        closed = 1.0 / 3.0 - n_c * math.cos(tilt) ** 2 - n_a * math.sin(tilt) ** 2
        f_b = F.dipole_field(spheroid(lat4, 18.0, 36.0, 0.0), 2.0, (math.sin(tilt), 0.0, math.cos(tilt)), pad=(144,) * 3, background=0.0)
        f_o = F.dipole_field(spheroid(lat4, 18.0, 36.0, tilt), 2.0, (0, 0, 1), pad=(144,) * 3, background=0.0)
        m_b, m_o = float(f_b[core0].mean()), float(f_o[spheroid(lat4, 12.0, 30.0, tilt) > 0.999].mean())
        worst_closed, worst_pair = max(worst_closed, abs(m_b - closed), abs(m_o - closed)), max(worst_pair, abs(m_b - m_o))
    rows.append(_row("F15", "kernel", "prolate spheroid (semi-axes 36 and 18 mm) at 30, 54.74 and 90 degrees to B0: mean inside, once with B0 "
                     "turned and once with the object turned", "1/3 - N_c cos^2 t - N_a sin^2 t, N_c = 0.17356 (demagnetising factors)",
                     worst_closed, 1.5e-3))
    rows.append(_row("F16", "kernel", "the same three angles: B0 turned against the object turned", "equal", worst_pair, 1.5e-3))
    # ---- B0 oblique to the lattice: the mixed terms of the kernel
    off = ball(lat, A, (10.0, -6.0, 4.0))
    r_off = np.sqrt((y - 10.0) ** 2 + (x + 6.0) ** 2 + (z - 4.0) ** 2) + 0 * r
    shell, core = (r_off > A + 3) & (r_off < 3 * A), r_off < A - 3
    worst_out, worst_in, plain = 0.0, 0.0, 0.0
    for b in ((1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.2, -0.6, 0.7)):
        ex = F.sphere_field(y, x, z, (10.0, -6.0, 4.0), A, 1.0, b)
        fo = F.dipole_field(off, 2.0, b, pad=(192,) * 3, background=0.0)
        d = (fo - ex)[shell]
        worst_out, worst_in = max(worst_out, _rms(d - d.mean())), max(worst_in, float(fo[core].std()))
        kz, ky, kx = np.meshgrid(np.fft.fftfreq(192, 2.0), np.fft.fftfreq(192, 2.0), np.fft.rfftfreq(192, 2.0), indexing="ij", sparse=True)
        u = F._unit(b)
        k2 = kz * kz + ky * ky + kx * kx
        k2[0, 0, 0] = 1.0
        textbook = 1.0 / 3.0 - (ky * u[0] + kx * u[1] + kz * u[2]) ** 2 / k2          # the plain product in the mixed terms
        textbook[0, 0, 0] = 0.0
        buf = np.zeros((192,) * 3)
        buf[:128, :128, :128] = off
        d = (sfft.irfftn(sfft.rfftn(buf, workers=1) * textbook, s=(192,) * 3, workers=1)[:128, :128, :128] - ex)[shell]
        plain = max(plain, _rms(d - d.mean()))
    rows.append(_row("F16b", "kernel", "B0 oblique to the lattice (along the body diagonal, a face diagonal and a general direction), the "
                     "sphere off the lattice centre: larger of the rms error outside (3 mm off the surface to 3 radii, constant removed) / "
                     "0.0003 and the standard deviation inside / 0.002", "the closed form with that direction: as good as along an axis (F1, F2)",
                     max(worst_out / 3e-4, worst_in / 2e-3), 1.0, unit="of the bound"))
    rows.append(_row("F16c", "kernel", "the same three directions with the textbook kernel, the plain product k_i k_j in the mixed terms: rms "
                     "error outside", "this row fails if the plain product is good enough: it jumps at the edge of the zone and the field "
                     "rings", plain, 1e-3, kind="min"))
    # ---- anisotropic cells
    lat5 = F.Lattice.centred((86, 128, 172), (3.0, 2.0, 1.5))
    y5, x5, z5 = lat5.mesh()
    r5 = np.sqrt(y5 * y5 + x5 * x5 + z5 * z5)
    f5 = F.dipole_field(ball(lat5, A), lat5.spacing, pad=(128, 192, 256), background=0.0)
    rows.append(_row("F17", "kernel", "the sphere on cells of 3 x 2 x 1.5 mm (z, y, x): rms error outside, 3 mm off the surface to 3 radii",
                     "the closed form", _rms((f5 - F.sphere_field(y5, x5, z5, (0, 0, 0), A))[(r5 > A + 3) & (r5 < 3 * A)]), 6e-4))
    return rows


# ---------------------------------------------------------------------------------------------- lattice
def _finite_rod_on_axis(d_top, d_bottom, radius):
    """On the axis of a cylinder magnetised along its axis, at the distances d_top and d_bottom from its two flat ends
    (inside): (c1 + c2) / 2 - 2/3 with c = d / sqrt(d^2 + a^2): the two end faces carry the surface charge +-M."""
    c1, c2 = (d / np.sqrt(d * d + radius * radius) for d in (d_top, d_bottom))
    return 0.5 * (c1 + c2) - 2.0 / 3.0


def check_lattice():
    rows = []
    lat = F.Lattice.centred((128,) * 3, 2.0)
    y, x, z = lat.mesh()
    r = np.sqrt(y * y + x * x + z * z) + 0 * (y + x + z)
    w = ball(lat, A)
    near = r < 3 * A
    ref = F.dipole_field(w, 2.0, pad=(320,) * 3, background=0.0, images=False)         # box 640 mm
    cos2 = np.divide(z * z, r * r, out=np.zeros_like(r), where=r > 0)
    image = lambda length: (6.216 * (A / length) ** 3 * (r / length) ** 2 * (3 * cos2 - 1))[near]      # noqa: E731
    for id_, pad, tol in (("F18", 128, 0.10), ("F19", 192, 0.25)):
        length = 2.0 * pad
        diff = (F.dipole_field(w, 2.0, pad=(pad,) * 3, background=0.0, images=False) - ref)[near]
        rows.append(_row(id_, "lattice", f"padding: the sphere transformed at {pad}^3 (box {length:.0f} mm) against 320^3 (640 mm), within 3 "
                         f"radii: largest difference {np.abs(diff).max():.1e}; rms of (difference - image formula) over rms of the difference",
                         "images of a cubic cell: 6.216 (a / L)^3 (r / L)^2 (3 cos^2 t - 1) (lattice sum of dipoles, derived separately), for "
                         "both box lengths", _rms(diff - image(length) + image(640.0)) / _rms(diff), tol, unit="relative"))
    worst = float(np.abs((F.dipole_field(w, 2.0, pad=(192,) * 3, background=0.0, images=False) - ref)[near]).max())
    rows.append(_row("F20", "lattice", "padding 1.5 (the default) without the image term: largest image error of the sphere within 3 radii",
                     "bound 2 x 6.216 (a / L)^3 (3 a / L)^2 = 1.1e-4 at L = 384 mm", worst, 1.1e-4))
    off = ball(lat, A, (10.0, -6.0, 4.0))                               # off the lattice centre: the first-moment term is needed
    near2 = np.sqrt((y - 10.0) ** 2 + (x + 6.0) ** 2 + (z - 4.0) ** 2) + 0 * r < 3 * A
    truth = F.dipole_field(off, 2.0, pad=(320,) * 3, background=0.0, images=False)
    left = []
    for pad in (1.0, 1.5):
        d = (F.dipole_field(off, 2.0, pad=pad, background=0.0) - truth)[near2]
        left.append(float(np.abs(d - d.mean()).max()))
    tilted = (1.0, 1.0, 1.0)                                            # B0 along the body diagonal of the cell
    d = (F.dipole_field(off, 2.0, tilted, pad=1.5, background=0.0) - F.dipole_field(off, 2.0, tilted, pad=(320,) * 3, background=0.0, images=False))[near2]
    left.append(float(np.abs(d - d.mean()).max()))
    rows.append(_row("F20b", "lattice", f"the image term taken off (the default), a sphere off the centre of the lattice, against the transform "
                     f"at 320^3, within 3 radii, constant removed: largest difference without padding ({left[0]:.1e} in place of 8e-4), with "
                     "padding 1.5, and with padding 1.5 and B0 along the body diagonal; the larger of the last two",
                     "0: the quadratic term of the images is (1/2) sum_src q (r - r')^T H (r - r'), H the lattice "
                     "sum of second derivatives of the dipole field (``image_hessian``)", max(left[1:]), 2e-5))
    h_zz = F.image_hessian((256.0,) * 3)[0, 0] * 256.0 ** 5
    rows.append(_row("F20c", "lattice", "that lattice sum for a cubic cell, B0 along z: H_zz L^5 against 4 x 6.216 x 3 / (4 pi)",
                     "5.936 (the image formula of F18, derived separately)", abs(h_zz / (4 * 6.216 * 3 / (4 * np.pi)) - 1), 2e-3, unit="relative"))
    # ---- a head-sized object: a homogeneous ellipsoid has a uniform field inside, so what the brain shows is error
    hl, hw, _, brain = head_lattice([], (180.0, 150.0))
    air = F.DCHI_AIR_PPM * (1 - hw.sum(0))
    hz = 3.0 * GAMMA_HZ_PER_T * 1e-6
    plain = float((hz * F.dipole_field(air, 2.0, images=False))[brain].std())
    rows.append(_row("F20d", "lattice", "a homogeneous ellipsoidal head (180 x 150 x 220 mm) in air on the specification's lattice, padding 1.5, "
                     f"at 3 T: standard deviation of the field over the brain; without the image term it is {plain:.2f} Hz",
                     "0: the field inside an ellipsoid is uniform", float((hz * F.dipole_field(air, 2.0))[brain].std()), 0.3, unit="Hz"))
    # ---- convergence with the spacing: the same sphere in a box of 160 mm
    errs = []
    for n, h in ((40, 4.0), (80, 2.0), (160, 1.0)):
        lt = F.Lattice.centred((n,) * 3, h, (h / 2,) * 3 if n % 2 == 0 else (0.0,) * 3)     # the same box for every spacing
        yy, xx, zz = lt.mesh()
        rr = np.sqrt(yy * yy + xx * xx + zz * zz) + 0 * (yy + xx + zz)
        ff = F.dipole_field(ball(lt, A), h, pad=(int(1.5 * n),) * 3, background=0.0)            # the box is 240 mm at every spacing
        errs.append(_rms((ff - F.sphere_field(yy, xx, zz, (0, 0, 0), A))[(rr > A + 8) & (rr < 2.5 * A)]))
    order = float(np.mean([math.log2(errs[i] / errs[i + 1]) for i in range(2)]))
    rows.append(_row("F21", "lattice", f"convergence with the cell size: rms error of the sphere outside (8 mm off the surface to 2.5 radii), "
                     f"box 240 mm, at 4, 2, 1 mm = {errs[0]:.1e}, {errs[1]:.1e}, {errs[2]:.1e}; "
                     "order", "second order (cell means carry the exact dipole moment; what is left is the shape within a cell)", order, 1.5,
                     kind="min"))
    # ---- a body that leaves the lattice: a rod along B0 through the lower face
    lat2 = F.Lattice.centred((96, 96, 96), 2.0)
    y2, x2, z2 = lat2.mesh()
    top = 20.0                                                          # the rod ends 20 mm above the centre and leaves through z-
    rodw = F.volume_fraction(lambda yy, xx, zz: np.maximum(np.sqrt(yy * yy + xx * xx) - 16.0, zz - top), lat2)
    zc = lat2.axes()[0]
    sel = (zc > -60) & (zc < top - 6)
    cut = F.dipole_field(rodw, 2.0, pad=(192, 144, 144), background=0.0, images=False)[sel, 48, 48]
    ext = F.dipole_field(rodw, 2.0, pad=(192, 144, 144), background=0.0, extrude=("z-",))[sel, 48, 48]
    bottom_cut, bottom_ext = zc[0] - 1.0, zc[0] - 1.0 - 48 * 2.0        # the lower end: the face, or 48 cells further
    rel = lambda v: v - v[-1]                                           # noqa: E731, the profile along the axis; its constant is not determined (k = 0)
    exact_ext = rel(_finite_rod_on_axis(top - zc[sel], zc[sel] - bottom_ext, 16.0))
    exact_cut = rel(_finite_rod_on_axis(top - zc[sel], zc[sel] - bottom_cut, 16.0))
    exact_long = rel(_finite_rod_on_axis(top - zc[sel], 1e9, 16.0))
    rows.append(_row("F22", "lattice", "a rod along B0 (radius 16 mm) that leaves the lattice through its lower face, continued through the "
                     "padding (``extrude``): field along its axis from 37 mm above the face to 6 mm below its upper end, relative to the "
                     "upper point, against a rod that ends 96 mm below the face", "(c1 + c2) / 2 - 2/3, c = d / sqrt(d^2 + a^2) for the "
                     "distances d to the two ends", float(np.abs(rel(ext) - exact_ext).max()), 3e-3))
    rows.append(_row("F23", "lattice", "the same rod cut at the face: largest error of that profile against the rod that goes on for ever; "
                     f"with the continuation it is {np.abs(rel(ext) - exact_long).max():.3f}", "a flat cut is an end face with the full surface "
                     f"charge: the closed form gives {np.abs(exact_cut - exact_long).max():.3f}",
                     float(np.abs(rel(cut) - exact_long).max()), 0.025, kind="min"))
    return rows


# ---------------------------------------------------------------------------------------------- surfaces and planes
def _surface_errors(radius, cavity, rng, lo=0.0, hi=1.0, n=128):
    """rms error at points ``lo`` to ``hi`` mm from the surface on the tissue side of a sphere: (one-sided, plain
    interpolation). ``cavity``: air sphere in tissue (points outside); else a tissue sphere in air (points inside)."""
    lat = F.Lattice.centred((n,) * 3, 2.0)
    w = ball(lat, radius)
    air = w if cavity else 1 - w
    args = dict(lattice=lat, b0_t=3.0, background_ppm=0.0 if cavity else 1.0, pad=1.5)
    good, naive = F.Field3D(air, tissue=1 - air, **args), F.Field3D(air, tissue=None, **args)
    py, px, pz = _shell_points(rng, 20000, radius + lo, radius + hi) if cavity else _shell_points(rng, 20000, radius - hi, radius - lo)
    exact = F.sphere_field(py, px, pz, (0, 0, 0), radius) if cavity else float(good.delta[n // 2, n // 2, n // 2])
    return _rms(good.ppm(py, px, pz) - exact), _rms(naive.ppm(py, px, pz) - exact)


def image_pair(one_sided=True, grids=(320, 640)):
    """The image-level check of the specification: one slice through a head-like object (a tissue ball of 80 mm with
    an air ball of 14 mm 20 mm under the slice and off-centre), imaged by the engine from master grids of 0.8 and
    0.4 mm with the field of this module. Returns the rms difference of the two images over the rms of the image."""
    from . import console, kspace, world
    lat = F.Lattice.centred((112,) * 3, 2.0)
    centre = (-30.0, 10.0, -20.0)
    tissue = ball(lat, 80.0) - ball(lat, 14.0, centre)
    field = F.Field3D(F.DCHI_AIR_PPM * (1 - tissue), lat, 3.0, tissue=tissue if one_sided else None)
    images = []
    for n in grids:
        shapes = [("wm", lambda y, x: np.sqrt(y * y + x * x) - 80.0)]
        w = world.flat_world(world.shapes_phantom(shapes, n=n, extent_mm=256.0))
        seen = np.zeros((n, n), np.float32) if one_sided else None
        s = F.sample(field, F.plane("axial"), n, 256.0, 4.0, chi_seen=seen)
        w["fields"]["b0_hz"], w["fields"]["dbdz_hz_per_mm"] = s["b0_hz"].astype(float), s["dbdz_hz_per_mm"].astype(float)
        rp = console.plan(dict(sequence="spgr", tr_ms=20, te_ms=6, flip_deg=15, fov_mm=256, matrix=64, slice_mm=4, excitation="nonselective"),
                          dict(w["console"], min_pixel_mm=4.0))["realised"]
        eng = kspace.Engine(w)
        images.append(eng.recon(eng.clean_kspace(rp)[0], rp))
    return float(np.sqrt((np.abs(images[0] - images[1]) ** 2).mean() / (np.abs(images[1]) ** 2).mean()))


def check_surface():
    rows = []
    rng = np.random.default_rng(20261126)
    for id_, radius, cavity, name, tol in (("F24", 12.0, True, "an air sphere of 12 mm in tissue (a sinus): points in the tissue", 0.03),
                                           ("F25", 24.0, False, "a tissue sphere of 24 mm in air: points inside", 0.012)):
        good, naive = _surface_errors(radius, cavity, rng)
        rows.append(_row(id_, "surface", f"{name} within 1 mm of the surface, rms error of the field at arbitrary positions (9.4 ppm at 3 T: "
                         "x 1200 Hz)", "the closed form on the tissue side", good, tol))
        rows.append(_row(id_ + "b", "surface", "the same points with plain interpolation of the lattice field (no step taken out)",
                         "the error the specification found: the step smeared into the tissue; this row fails if it is small", naive, 0.08, kind="min"))
    good, _ = _surface_errors(12.0, True, rng, 2.0, 8.0)
    rows.append(_row("F26", "surface", "the air sphere of 12 mm: points in the tissue 2 to 8 mm from the surface, rms error", "the closed form",
                     good, 2.5e-3))
    # ---- an image plane against the closed form: 8 mm above the sphere
    lat = F.Lattice.centred((96,) * 3, 2.0)
    w = ball(lat, 12.0)
    field = F.Field3D(F.DCHI_AIR_PPM * w, lat, 3.0, tissue=1 - w, background_ppm=0.0)
    fr = F.plane("axial", 20.0)
    s = F.sample(field, fr, 320, 128.0, 5.0)
    py, px, pz = F.plane_points(fr, 320, 128.0)
    hz = F.DCHI_AIR_PPM * field.hz_per_ppm
    exact = hz * F.sphere_field(py, px, pz, (0, 0, 0), 12.0)
    secant = hz * (F.sphere_field(py, px, pz + 2.5, (0, 0, 0), 12.0) - F.sphere_field(py, px, pz - 2.5, (0, 0, 0), 12.0)) / 5.0
    rows.append(_row("F27", "surface", f"``sample`` on an axial plane 8 mm above an air sphere of 12 mm at 3 T (peak {exact.max():.0f} Hz), grid of "
                     "0.4 mm from the 2 mm lattice: largest error of the field", "the closed form; trilinear interpolation alone allows h^2 / 8 x "
                     "the second derivative, 9 Hz at the peak", float(np.abs(s["b0_hz"] - exact).max()), 10.0, unit="Hz"))
    rows.append(_row("F28", "surface", f"the same plane: through-plane gradient over a slice of 5 mm (peak {np.abs(secant).max():.0f} Hz/mm), "
                     "largest error", "the secant of the closed form across the slice", float(np.abs(s["dbdz_hz_per_mm"] - secant).max()), 2.0,
                     unit="Hz/mm"))
    for id_, name, b, frame in (("F29", "a coronal plane", (0, 0, 1), F.plane("coronal", 20.0)),
                                ("F30", "a sagittal plane", (0, 0, 1), F.plane("sagittal", -20.0)),
                                ("F31", "an axial plane, B0 along y,", (1, 0, 0), F.plane("axial", 20.0))):
        fld = F.Field3D(F.DCHI_AIR_PPM * w, lat, 3.0, b, tissue=1 - w, background_ppm=0.0)
        s = F.sample(fld, frame, 160, 128.0, 5.0)
        py, px, pz = F.plane_points(frame, 160, 128.0)
        rows.append(_row(id_, "surface", f"``sample`` on {name} 20 mm from the centre of the sphere: largest error",
                         "the closed form with the same direction of B0",
                         float(np.abs(s["b0_hz"] - hz * F.sphere_field(py, px, pz, (0, 0, 0), 12.0, 1.0, b)).max()), 10.0, unit="Hz"))
    rows.append(_row("F32", "surface", "one slice imaged by the engine from master grids of 0.8 and 0.4 mm (spoiled gradient echo, TE 6 ms, "
                     "matrix 64; a tissue ball with an air ball 6 mm under the slice): rms difference of the two images",
                     "the specification's acceptance: at most 1 % (its first version: 1.7 % to 3.4 %)", image_pair(True), 0.01, unit="relative"))
    rows.append(_row("F32b", "surface", "the same pair of images with plain interpolation of the lattice field",
                     "above the 1 % of the acceptance: this row fails if the check above cannot see the smeared step", image_pair(False), 0.01,
                     kind="min", unit="relative"))
    return rows


# ---------------------------------------------------------------------------------------------- shim
def engine_shim_pair(seed="0123456789abcdef0123456789abcdef"):
    """The engine's fitted shim of a brain world against ``fields3d.shim_fit`` on the same field, the closed-form air
    spheres of the world inside the brain mask, first order: the constant and the two in-plane terms from the plane
    z = 0 (the z term vanishes there and gets no current), the z term from the planes z = -1, 0, 1 mm. Returns the
    largest relative difference of the four coefficients (constant, y, x and the mean through-slice gradient)."""
    from . import world
    w = world.make_world(seed, dict(kind="brain", n=320))
    y, x = world.grid(w)
    m = w["masks"]["brain"]
    flat = F.shim_fit(F.shim_basis(y[m], x[m], 0.0, order=1), world.dipole_hz(w, y, x, 0.0)[m])
    pts = [np.concatenate([v[m]] * 3) for v in (y, x)] + [np.repeat([-1.0, 0.0, 1.0], int(m.sum()))]
    raw = np.concatenate([world.dipole_hz(w, y, x, zz)[m] for zz in (-1.0, 0.0, 1.0)])
    slab = F.shim_fit(F.shim_basis(*pts, order=1), raw)
    mine = np.array([flat["coef"][0], flat["coef"][2], flat["coef"][1], slab["coef"][3]])   # F0, Y, X, Z
    theirs = np.array(list(w["fields"]["shim"]["coef"]) + [w["fields"]["shim"]["dz_mean"]])
    return float(np.max(np.abs(mine - theirs) / np.abs(theirs)))


def check_shim():
    rows = []
    rng = np.random.default_rng(7)
    y, x, z = (rng.uniform(-80, 80, 4000) for _ in range(3))
    keep = (y / 80) ** 2 + (x / 70) ** 2 + (z / 60) ** 2 < 1
    y, x, z = y[keep] + 5.0, x[keep] - 3.0, z[keep] + 12.0              # a volume of interest off the isocentre
    basis = F.shim_basis(y, x, z, 2)
    worst = 0.0
    for n in (4, 9):                                                    # first order, second order
        c = rng.uniform(-1, 1, n) * np.array([50.0, 1, 1, 1, 0.02, 0.02, 0.02, 0.02, 0.02][:n])
        v = c @ basis[:n]
        rep = F.shim_fit(basis[:n], v)
        worst = max(worst, float(np.abs(rep["residual"]).max() / np.abs(v).max()), float(np.abs(rep["coef"] - c).max() / np.abs(c).max()))
    rows.append(_row("F33", "shim", "a pure first-order field and a pure second-order field over an ellipsoidal volume off the isocentre: "
                     "largest residual and largest error of a coefficient, relative", "removed to rounding", worst, 1e-11))
    h, p = 1e-2, rng.uniform(-50, 50, (3, 200))
    lap = sum(F.shim_basis(*(p + h * np.eye(3)[i][:, None]), 2) + F.shim_basis(*(p - h * np.eye(3)[i][:, None]), 2) for i in range(3)) - 6 * F.shim_basis(*p, 2)
    rows.append(_row("F34", "shim", "Laplacian of the nine basis functions (second differences at 200 points)",
                     "0: a field made by currents outside the volume is harmonic", float(np.abs(lap / h ** 2).max()), 1e-6, unit="mm^-2 x term"))
    # ---- limits: a quadratic field that needs more Z2 than the channel has
    lim = F.shim_limits_hz()
    c = np.zeros(9)
    c[4], c[1] = 2.5 * lim[4], 0.4
    v = c @ basis
    free, held = F.shim_fit(basis, v), F.shim_fit(basis, v, lim)
    ok = held["at_limit"] == [4] and abs(held["coef"][4] - lim[4]) < 1e-9 * lim[4] and held["sd_after"] > 10 * free["sd_after"]
    rows.append(_row("F35", "shim", "a field that needs 2.5 times the limit of the Z2 channel: the bounded fit puts Z2 at its limit and no other "
                     "channel; largest violation of the optimality conditions (the gradient is zero for a free channel and points outwards at "
                     "a limit)", "0 (Karush-Kuhn-Tucker)", held["kkt"] if ok else np.inf, 1e-9, unit="relative"))
    # an independent check of the bounded optimum: no feasible point near it is better
    best = float(np.sum(held["residual"] ** 2))
    trial = min(float(np.sum((v - np.clip(held["coef"] + rng.normal(0, 1e-3, 9) * np.abs(held["coef"]).max(), -lim, lim) @ basis) ** 2)) for _ in range(200))
    rows.append(_row("F36", "shim", "the same fit: smallest sum of squares among 200 feasible points drawn around the solution, over that of "
                     "the solution", "at least 1", trial / best, 1.0, kind="min", unit="ratio"))
    rows.append(_row("F37", "shim", "first order against the engine: the fitted shim of a brain world (``world.anatomy_fields``: constant, y, "
                     "x over the brain, and the mean through-slice gradient) against this module's fit of the same closed-form field on the "
                     "planes z = -1, 0, 1 mm: largest relative difference of the four coefficients", "equal: the engine's shim is a first-order "
                     "fit over a thin slab", engine_shim_pair(), 1e-9, unit="relative"))
    # ---- a z-uniform object: the lattice fit over a volume against the engine's procedure on one slice of it
    lat = F.Lattice.centred((24, 128, 128), 2.0)
    yl, xl, zl = lat.mesh()
    tis = F.volume_fraction(lambda yy, xx, zz: np.sqrt((yy / 90.0) ** 2 + (xx / 75.0) ** 2) * 75.0 - 75.0 + 0 * zz, lat)
    air = F.volume_fraction(lambda yy, xx, zz: np.sqrt((yy + 50.0) ** 2 + (xx - 5.0) ** 2) - 12.0 + 0 * zz, lat)
    tissue = tis - air
    fld = F.Field3D(F.DCHI_AIR_PPM * (1 - tissue), lat, 3.0, (1, 0, 0), tissue=tissue, pad=(24, 256, 256), images=False)     # B0 along y: across the cylinders
    voi = (np.sqrt((yl / 70.0) ** 2 + (xl / 58.0) ** 2) < 1) & (np.sqrt((yl + 50.0) ** 2 + (xl - 5.0) ** 2) > 18.0) & (zl == zl)
    py, px, pz = F.plane_points(F.plane("axial"), 640, 256.0)
    m = (np.sqrt((py / 70.0) ** 2 + (px / 58.0) ** 2) < 1) & (np.sqrt((py + 50.0) ** 2 + (px - 5.0) ** 2) > 18.0)
    raw = F.sample(fld, F.plane("axial"), 640, 256.0, 5.0)["b0_hz"]
    a = np.stack([np.ones(int(m.sum())), py[m], px[m]], 1)
    after2d = raw[m] - a @ np.linalg.lstsq(a, raw[m].astype(float), rcond=None)[0]     # the engine's procedure (``anatomy_fields``)
    fld.shim(voi, 1, bounded=False)
    after3d = F.sample(fld, F.plane("axial"), 640, 256.0, 5.0)["b0_hz"][m]
    along_z = float(fld.shim_coef[:4] @ (F.shim_basis(0.0, 0.0, 1.0, 1, fld.b0_dir) - F.shim_basis(0.0, 0.0, 0.0, 1, fld.b0_dir)))
    rows.append(_row("F38", "shim", f"an object that does not change along z (an elliptic cylinder with an air cylinder, B0 across them; field "
                     f"over the region {raw[m].std():.1f} Hz sd): first-order shim over the volume on the 2 mm lattice against the engine's "
                     f"fit on the slice's 0.4 mm grid: rms difference of the two shimmed fields over the sd after shim ({after2d.std():.1f} Hz); "
                     f"the shim's gradient along z is {abs(along_z):.0e} Hz/mm", "the same least-squares problem on two samplings of one region",
                     _rms(after3d - after2d) / float(after2d.std()), 0.02, unit="relative"))
    return rows


# ---------------------------------------------------------------------------------------------- extras
def profile_pair(kind="brain", seed="0123456789abcdef0123456789abcdef", n=160):
    """Largest difference between the declared 3D profiles on the plane z = 0 and the fields of the engine's 2D
    world of the same seed (transmit ratio, complex receive profile)."""
    from . import anatomy, world
    w = world.make_world(seed, dict(kind=kind, n=n))
    rng = world.rng_for(seed, "fields")
    anatomy.smooth_field(rng, 4, 150, 500, 1.0)                         # the draws of the smooth field term come first
    prof = F.declared_profiles(kind, w["b0_t"], (*w["size_mm"], 200.0), rng, world.rng_for(seed, "fields", 2))
    y, x = world.grid(w)
    return max(float(np.abs(prof.b1(y, x, 0.0) - w["fields"]["b1"]).max()), float(np.abs(prof.rx(y, x, 0.0) - w["fields"]["rx"]).max()))


def tkd_sphere(threshold, lattice="specification"):
    """Mean recovered susceptibility of a unit sphere by truncated division. "specification": its lattice (128^3 at
    2 mm, edge of 1 mm sampled at the centres, no padding), mean over r < a - 3 mm. "weighted": cell means, the mean
    weighted with the sphere itself, which is what ``tkd_factor`` predicts."""
    lat = F.Lattice.centred((128,) * 3, 2.0)
    y, x, z = lat.mesh()
    r = np.sqrt(y * y + x * x + z * z) + 0 * (y + x + z)
    w = _soft_ball(lat, A, 1.0) if lattice == "specification" else ball(lat, A)
    chi = F.tkd(F.dipole_field(w, 2.0, pad=1.0, background=0.0, images=False), 2.0, threshold=threshold)
    return float(chi[r < A - 3].mean()) if lattice == "specification" else float((chi * w).sum() / (w * w).sum())


def timing():
    """CPU seconds of this process (best of 3; ``time.process_time``, so that a busy machine does not fail the row) to build the field of a world on the lattice the specification chose (136 x 128 x 128 cells
    of 2 mm), and to sample one plane on the 0.4 mm grid."""
    lat = F.Lattice.centred((136, 128, 128), 2.0, (-24.0, 0.0, 0.0))
    y, x, z = lat.mesh()
    tissue = F.volume_fraction(lambda yy, xx, zz: (np.sqrt((yy / 96.0) ** 2 + (xx / 78.0) ** 2 + ((zz + 20.0) / 110.0) ** 2) - 1) * 78.0, lat)
    chi = F.DCHI_AIR_PPM * (1 - tissue)
    build, plane = [], []
    for _ in range(3):
        t0 = time.process_time()
        fld = F.Field3D(chi, lat, 3.0, tissue=tissue)
        build.append(time.process_time() - t0)
        t0 = time.process_time()
        F.sample(fld, F.plane("axial", 10.0), 640, 256.0, 5.0)
        plane.append(time.process_time() - t0)
    return min(build), min(plane)


def check_extras():
    rows = []
    # ---- small structures: rasterised, or closed form
    lat = F.Lattice.centred((96,) * 3, 2.0)
    y, x, z = lat.mesh()
    r = np.sqrt(y * y + x * x + z * z) + 0 * (y + x + z)
    rel = {}
    for cells in (1, 2, 4):
        a = 2.0 * cells
        ex = F.sphere_field(y, x, z, (0, 0, 0), a)
        m = (r > a + 2.0) & (r < 3 * a + 2.0)
        rel[cells] = _rms((F.dipole_field(ball(lat, a), 2.0, background=0.0) - ex)[m]) / _rms(ex[m])
    rows.append(_row("F39", "extras", f"a sphere of radius 2 cells on the lattice (cell means): rms error outside, from one cell off the "
                     f"surface to three radii further, over the rms of the field there (1 cell: {rel[1]:.3f}; 4 cells: {rel[4]:.3f})",
                     "the rule: at least 2 cells of radius on the lattice (``MIN_LATTICE_RADIUS_CELLS``), smaller in closed form", rel[2], 0.04,
                     unit="relative"))
    big = ball(lat, 30.0)
    inc = dict(kind="sphere", centre=(5.3, -2.1, 3.7), radius_mm=1.5, dchi_ppm=3.0)
    fld = F.Field3D(F.DCHI_AIR_PPM * (1 - big), lat, 3.0, tissue=big, inclusions=[inc])
    bare = F.Field3D(F.DCHI_AIR_PPM * (1 - big), lat, 3.0, tissue=big)
    py, px, pz = np.random.default_rng(3).uniform(-12, 12, (3, 5000))
    rows.append(_row("F40", "extras", "a closed-form inclusion (a sphere of 1.5 mm, 3 ppm) added to a lattice field: the field with it minus "
                     "the field without, at 5000 points around and inside it", "the exact dipole outside, 0 inside: added once, in ppm",
                     float(np.abs(fld.ppm(py, px, pz) - bare.ppm(py, px, pz) - F.sphere_field(py, px, pz, inc["centre"], 1.5, 3.0)).max()), 1e-5,
                     unit="ppm"))
    rows.append(_row("F41", "extras", "declared transmit and receive profiles on the plane z = 0 against the engine's brain world of the same "
                     "seed (B1+ ratio, complex receive profile)", "equal: the 3D shapes continue the 2D ones", profile_pair("brain"), 1e-12))
    # ---- susceptibility mapping
    got = tkd_sphere(0.1)
    rows.append(_row("F42", "extras", "truncated k-space division at threshold 0.1 on the specification's sphere: mean recovered susceptibility "
                     "inside against the specification's 0.905", "0.905 (true value 1: the underestimate)", abs(got - 0.905), 0.003))
    worst = max(abs(tkd_sphere(t, "weighted") - F.tkd_factor(t)) for t in (0.1, 0.2))
    rows.append(_row("F43", "extras", f"the same inversion at thresholds 0.1 and 0.2 on cell means: recovered susceptibility weighted with the "
                     f"sphere against the angular mean of D x its truncated inverse ({F.tkd_factor(0.1):.4f}, {F.tkd_factor(0.2):.4f})",
                     "``tkd_factor``: 1 - (u2 - u1) + (1 / t) x integral of abs(1/3 - u^2) between u1,2 = sqrt(1/3 -+ t); the lattice is lower "
                     "by the k = 0 term (the volume fraction, 0.003) and by the few directions a 128-cell box has at small k", worst, 0.01))
    f = np.array([[-37.5, 0.0], [12.25, 180.0]])
    te = np.array([4.0, 9.0, 14.0])
    ph = F.to_phase(f, te, wrap=False)
    wrapped = F.to_phase(f, te)
    slope = np.polyfit(te * 1e-3, ph.reshape(3, -1), 1)[0].reshape(2, 2) / (2 * np.pi)
    err = max(float(np.abs(slope - f).max()), float(np.abs(np.exp(1j * wrapped) - np.exp(1j * ph)).max()),
              0.0 if wrapped.max() <= np.pi and wrapped.min() > -np.pi and ph[1, 1, 1] > 0 else 1.0)
    rows.append(_row("F44", "extras", "multi-echo phase: slope over the echo times / 2 pi against the field; wrapped phase in (-pi, pi]; a "
                     "positive field gives a positive phase", "phase = 2 pi f TE, the sign of the engine", err, 1e-9, unit="Hz or rad"))
    build, plane = timing()
    rows.append(_row("F45", "extras", "time to build the field of a world on the specification's lattice (136 x 128 x 128 cells of 2 mm, "
                     f"transformed at 216 x 192 x 192), one thread; one plane on the 0.4 mm grid then takes {plane:.2f} s",
                     "at most 0.8 s", build, 0.8, unit="s"))
    return rows


# ---------------------------------------------------------------------------------------------- a head
HEAD = dict(half_z=110.0, centre_z=-20.0, scalp=6.0, skull=7.0, chi_skull=-2.1, floor_z=-10.0, heights=(4.0, 8.0, 12.0, 20.0))
SEED = "0123456789abcdef0123456789abcdef"
# Whole-brain standard deviations after shim, Hz at 7 T, as printed in the sources of docs/physics/fields3d.md section 13
IN_VIVO_7T = (("Stockmann & Wald 2018, one subject, first and second order", 51.9), ("the same paper, eight subjects (Kim 2016)", 29.72),
              ("Juchem 2011, orders 0 to 3, computed", 43.0))


def head_lattice(air, size_mm, spacing=2.0, shape=(136, 128, 128), oversample=6):
    """A head of nested ellipsoids: soft tissue, a skull shell (-2.1 ppm: Koch 2009 uses -11.3 ppm for bone against
    -9.2 ppm for tissue), soft tissue inside, and air spheres ``air`` [(y, x, z, radius)]. ``size_mm``: extent along
    y and x; the extent along z, the layer thicknesses and the floor of the brain are declared in ``HEAD``. The
    brain (the volume of interest) is the inner ellipsoid less 3 mm, above the floor, at least one cell away from air.
    Returns (lattice, weights [soft, skull], chi per class, brain mask)."""
    n = tuple(int(round(s * 2.0 / spacing)) for s in shape)
    lat = F.Lattice.centred(n, spacing, (-24.0, 0.0, 0.0))
    ay, ax, az = 0.5 * size_mm[0], 0.5 * size_mm[1], HEAD["half_z"]

    def ell(shrink):
        k = min(ay, ax, az) - shrink
        return lambda y, x, z: (np.sqrt((y / (ay - shrink)) ** 2 + (x / (ax - shrink)) ** 2 + ((z - HEAD["centre_z"]) / (az - shrink)) ** 2) - 1) * k
    head, outer, inner = (F.volume_fraction(ell(s), lat, oversample) for s in (0.0, HEAD["scalp"], HEAD["scalp"] + HEAD["skull"]))
    holes = sum(F.volume_fraction(lambda y, x, z, c=c: np.sqrt((y - c[0]) ** 2 + (x - c[1]) ** 2 + (z - c[2]) ** 2) - c[3], lat, oversample) for c in air)
    holes = np.clip(holes, 0.0, 1.0)
    skull = (outer - inner) * (1 - holes)
    soft = (head - (outer - inner)) * (1 - holes)
    brain = (F.volume_fraction(ell(HEAD["scalp"] + HEAD["skull"] + 3.0), lat, oversample) > 0.999) & (ndi.uniform_filter(holes, 3) < 1e-6)
    brain &= (lat.mesh()[2] > HEAD["floor_z"])
    return lat, np.stack([soft, skull]), (0.0, HEAD["chi_skull"]), brain


def head_numbers(seed=SEED, fine=True):
    """The figures of docs/physics/fields3d.md section 8, at 3 T. The engine's brain world of ``seed`` gives the size
    of the head and its three declared air spheres; the same spheres are cut out of a head of nested ellipsoids and
    the field is computed on the 2 mm lattice. ``fine``: also compare with a 1 mm lattice (1.7 GB, 10 s)."""
    from . import world
    e = world.make_world(seed, dict(kind="brain", n=320))
    air, size = [tuple(float(v) for v in a) for a in e["air"]], e["size_mm"]
    lat, w, chi, brain = head_lattice(air, size)
    fld = F.from_weights(w, chi, lat, 3.0)
    raw = fld.lattice_hz()
    mean = float(raw[brain].mean())
    hz = fld.hz_per_ppm
    sy, sx, sz, sr = air[0]                                              # the frontal sphere
    out = dict(radius=sr, depth=-(sz + sr), size=(float(size[0]), float(size[1])), raw_sd=float(raw[brain].std()),
               raw_range=(float(raw[brain].min() - mean), float(raw[brain].max() - mean)), above=[], above_sphere=[])
    for h in HEAD["heights"]:
        out["above"].append(float(fld.hz(sy, sx, sz + sr + h, chi_seen=0.0)) - mean)
        out["above_sphere"].append(float(hz * F.DCHI_AIR_PPM * F.sphere_field(sy, sx, sz + sr + h, (sy, sx, sz), sr)))
    for order in (1, 2):
        rep = fld.shim(brain, order)
        after = fld.lattice_hz(shim=True)[brain]
        out[f"sd{order}"], out[f"max{order}"], out[f"over20_{order}"] = float(after.std()), float(np.abs(after).max()), float((np.abs(after) > 20.0).mean())
        out[f"limit{order}"], out[f"kkt{order}"] = rep["at_limit"], rep["kkt"]
        out[f"above{order}"] = [float(fld.hz(sy, sx, sz + sr + h, chi_seen=0.0)) for h in HEAD["heights"]]
    fld.shim_coef[:] = 0
    # ---- the slice z = 0 on the engine's own grid and brain mask: the engine's procedure applied to both fields
    y, x = world.grid(e)
    inner = [0.5 * size[0] - 16.0, 0.5 * size[1] - 16.0, HEAD["half_z"] - 16.0]         # this head's brain on the slice
    m = e["masks"]["brain"] & ((y / inner[0]) ** 2 + (x / inner[1]) ** 2 + (HEAD["centre_z"] / inner[2]) ** 2 < 1)
    mine = F.sample(fld, F.plane("axial"), 320, e["extent_mm"], 5.0, chi_seen=np.zeros((320, 320), np.float32))
    theirs = world.dipole_hz(e, y, x, 0.0)
    a = np.stack([np.ones(int(m.sum())), y[m], x[m]], 1)
    fit = lambda v: v[m] - a @ np.linalg.lstsq(a, v[m].astype(float), rcond=None)[0]     # noqa: E731
    r_mine, r_theirs = fit(mine["b0_hz"]), fit(theirs)
    dz_theirs = (world.dipole_hz(e, y, x, 1.0) - world.dipole_hz(e, y, x, -1.0))[m] / 2
    out.update(slice_raw_sd=float(mine["b0_hz"][m].std()), slice_sd=float(r_mine.std()), slice_max=float(np.abs(r_mine).max()),
               slice_over20=float((np.abs(r_mine) > 20).mean()), spheres_raw_sd=float(theirs[m].std()), spheres_sd=float(r_theirs.std()),
               spheres_max=float(np.abs(r_theirs).max()), diff_rms=_rms(r_mine - r_theirs), corr=float(np.corrcoef(r_mine, r_theirs)[0, 1]),
               dz_sd=float((mine["dbdz_hz_per_mm"][m] - mine["dbdz_hz_per_mm"][m].mean()).std()), dz_sd_spheres=float((dz_theirs - dz_theirs.mean()).std()),
               engine_sd=float(e["fields"]["b0_hz"][m].std()), engine_max=float(np.abs(e["fields"]["b0_hz"][m] - e["fields"]["b0_hz"][m].mean()).max()),
               engine_over20=float((np.abs(e["fields"]["b0_hz"][m] - e["fields"]["b0_hz"][m].mean()) > 20).mean()))
    # a homogeneous head without the skull shell and without air: the part of the field the outer shape alone makes
    lat0, w0, _, _ = head_lattice([], size)
    shape_only = F.from_weights(w0, (0.0, 0.0), lat0, 3.0).lattice_hz()[brain]
    out["shape_sd"] = float(shape_only.std())
    out["skull_sd"] = float((F.from_weights(w0, chi, lat0, 3.0).lattice_hz()[brain] - shape_only).std())
    if fine:
        lat1, w1, _, _ = head_lattice(air, size, 1.0, oversample=3)
        d = (mine["b0_hz"] - F.sample(F.from_weights(w1, chi, lat1, 3.0), F.plane("axial"), 320, e["extent_mm"], 5.0,
                                      chi_seen=np.zeros((320, 320), np.float32))["b0_hz"])[m]
        out["fine_rms"], out["fine_max"] = _rms(d - d.mean()), float(np.abs(d - d.mean()).max())
    return out


def head3d_numbers(seed=SEED):
    """The seeded 3D head of ``volume.py`` (``head3d``: face, neck, skull, cavities) at 3 T on a 2 mm lattice of cell
    means (2 samples per axis): the field over its brain classes before and after shim, for three treatments of the
    neck, which leaves the lattice through its lower face. Returns (rows, sd of cut minus continued over the brain,
    fraction of the lower face that holds matter)."""
    from . import world
    vol = world.make_world(seed, dict(kind="head3d", b0_t=3.0))["volume"]
    L = vol.lattice(2.0, supersample=2)
    lat = F.Lattice(L.weights.shape[1:], L.spacing_mm, L.origin_mm)
    chi = [vol.susceptibility_ppm[k] for k in vol.matter]
    organ = vol.organ_mask(L.weights)
    rows, raw = [], {}
    for name, kw in (("neck cut at the lower face of the lattice", {}), ("neck continued through half the padding (`extrude`), padding 1.5", dict(extrude=("z-",))),
                     ("the same, padding 2 (the continuation is longer)", dict(extrude=("z-",), pad=2.0))):
        fld = F.from_weights(L.weights, chi, lat, 3.0, background_ppm=vol.susceptibility_ppm["air"], **kw)
        raw[name] = fld.lattice_hz()[organ]
        row = [name, float(raw[name].std())]
        for order in (1, 2):
            fld.shim(organ, order)
            v = fld.lattice_hz(True)[organ]
            row += [float(v.std()), float(np.abs(v).max())]
        rows.append(row)
    names = list(raw)
    return rows, float((raw[names[0]] - raw[names[1]]).std()), float(L.weights.sum(0)[0].mean())


def head_markdown(h=None):
    h = h or head_numbers()
    hz = lambda v: " / ".join(f"{q:+.0f}" for q in v)                    # noqa: E731
    lines = [
        f"Head: extent {h['size'][0]:.0f} mm (y) x {h['size'][1]:.0f} mm (x) x {2 * HEAD['half_z']:.0f} mm (z); frontal air sphere of radius "
        f"{h['radius']:.1f} mm, its top {h['depth']:.1f} mm below the slice z = 0; seed `{SEED}`.",
        "",
        "| Quantity at 3 T | Dipole lattice: head of nested ellipsoids with the three air spheres | The three declared spheres alone (the engine today) |",
        "|---|---|---|",
        f"| Field {' / '.join(f'{q:.0f}' for q in HEAD['heights'])} mm above the top of the frontal sphere, before shim (lattice: relative to the "
        f"brain mean) | {hz(h['above'])} Hz | {hz(h['above_sphere'])} Hz (that sphere's closed form) |",
        f"| The same points after a first-order shim over the whole brain; after a second-order shim | {hz(h['above1'])} Hz; {hz(h['above2'])} Hz | - |",
        f"| Whole brain before shim: standard deviation; range about the mean | {h['raw_sd']:.1f} Hz; {h['raw_range'][0]:+.0f} to "
        f"{h['raw_range'][1]:+.0f} Hz | one slice only |",
        f"| Whole brain after a first-order shim: standard deviation; largest; share of cells beyond 20 Hz | {h['sd1']:.1f} Hz; "
        f"{h['max1']:.0f} Hz; {100 * h['over20_1']:.1f} % | - |",
        f"| Whole brain after a second-order shim (channels at a limit of `SHIM_LIMITS_UT`: {', '.join(h['limit2']) or 'none'}) | "
        f"{h['sd2']:.1f} Hz; {h['max2']:.0f} Hz; {100 * h['over20_2']:.1f} % | no second order |",
        f"| Slice z = 0, the brain of both models, before shim: standard deviation | {h['slice_raw_sd']:.1f} Hz | {h['spheres_raw_sd']:.1f} Hz |",
        f"| The same slice after the engine's shim (constant, y, x fitted on the slice): standard deviation; largest; share beyond 20 Hz | "
        f"{h['slice_sd']:.1f} Hz; {h['slice_max']:.0f} Hz; {100 * h['slice_over20']:.1f} % | {h['spheres_sd']:.1f} Hz; {h['spheres_max']:.0f} Hz "
        f"(with the engine's smooth random term: {h['engine_sd']:.1f} Hz; {h['engine_max']:.0f} Hz; {100 * h['engine_over20']:.1f} %) |",
        f"| The two shimmed slice fields against each other: rms difference; correlation | {h['diff_rms']:.1f} Hz; {h['corr']:.3f} | - |",
        f"| Through-slice gradient on the slice, mean removed: standard deviation | {h['dz_sd']:.2f} Hz/mm | {h['dz_sd_spheres']:.2f} Hz/mm |",
        f"| Whole brain, the same head without air: field of the outer shape alone (an ellipsoid: uniform in theory, so this is the error "
        f"of the lattice); added by the skull shell (standard deviations) | {h['shape_sd']:.2f} Hz; {h['skull_sd']:.2f} Hz | absent |",
    ]
    if "fine_rms" in h:
        lines.append(f"| The 2 mm lattice against a 1 mm lattice on the slice z = 0 (brain, constant removed): rms; largest | {h['fine_rms']:.2f} Hz; "
                     f"{h['fine_max']:.1f} Hz | - |")
    rows, cut_sd, face = head3d_numbers()
    lines += ["", f"The seeded 3D head (`head3d` of `volume.py`, seed `{SEED}`, 3 T, 2 mm lattice of cell means); matter covers "
              f"{100 * face:.0f} % of the lower face of its lattice:", "",
              "| Whole brain, standard deviation (largest) | before shim | after a first-order shim | after a second-order shim |", "|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]:.1f} Hz | {r[2]:.1f} Hz ({r[3]:.0f} Hz) | {r[4]:.1f} Hz ({r[5]:.0f} Hz) |" for r in rows]
    lines += ["", f"The first two fields differ over the brain by {cut_sd:.1f} Hz (standard deviation, before shim)."]
    hz = 3.0 * GAMMA_HZ_PER_T * 1e-6
    lines += ["", f"Scale: 1 ppm is {hz:.1f} Hz at 3 T; the largest step at an air surface, (2/3) x {F.DCHI_AIR_PPM:.1f} ppm, is "
              f"{2 / 3 * F.DCHI_AIR_PPM * hz:.0f} Hz. Published whole-brain standard deviations at 7 T, scaled to 3 T by the field strength: "
              + "; ".join(f"{v:g} Hz x 3 / 7 = {v * 3 / 7:.1f} Hz ({name})" for name, v in IN_VIVO_7T) + "."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- run and print
CHECKS = dict(kernel=check_kernel, lattice=check_lattice, surface=check_surface, shim=check_shim, extras=check_extras)


def run(groups=None):
    """Run every check, or those of the named groups. Returns (rows, seconds)."""
    t0 = time.process_time()
    rows = []
    for name in (groups or CHECKS):
        rows += CHECKS[name]()
    return rows, time.process_time() - t0


def _number(x):
    if x == 0:
        return "0"
    if abs(x) >= 1e-3 and abs(x) < 1e4:
        return f"{x:.4g}"
    return f"{x:.1e}"


def markdown_table(rows):
    """The validation table of docs/physics/fields3d.md."""
    out = []
    for group, title in GROUPS.items():
        part = [r for r in rows if r["group"] == group]
        if not part:
            continue
        out += [f"### {title}", "", "| # | Quantity | Closed form or independent method | Error | Tolerance | |", "|---|---|---|---|---|---|"]
        for r in part:
            sign = "≤" if r["kind"] == "max" else "≥"
            unit = f" {r['unit']}" if r["unit"] else ""
            out.append(f"| {r['id']} | {r['quantity']} | {r['reference']} | {_number(r['measured'])}{unit} | {sign} {_number(r['tolerance'])} | "
                       f"{'pass' if r['ok'] else 'FAIL'} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--head" in argv:
        print(head_markdown())
        return 0
    rows, seconds = run([a for a in argv if a in CHECKS] or None)
    if "--markdown" in argv:
        print(markdown_table(rows))
    else:
        for r in rows:
            sign = "<=" if r["kind"] == "max" else ">="
            print(f"{r['id']:5s} {'pass' if r['ok'] else 'FAIL'}  {_number(r['measured']):>9s} {sign} {_number(r['tolerance']):<8s} {r['unit']:14s} {r['quantity'][:120]}")
        print(f"{sum(r['ok'] for r in rows)} of {len(rows)} rows pass; {seconds:.1f} s; gamma / 2 pi = {GAMMA_HZ_PER_T / 1e6:.6f} MHz/T")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
