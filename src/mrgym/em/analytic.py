"""Closed-form RF fields of a layered lossy dielectric cylinder in a coil (two dimensions, E along z).

The problem. Everything is invariant along z: an infinitely long circular cylinder of concentric homogeneous layers
(relative permittivity eps_r, conductivity sigma, permeability mu0) sits on the axis of a coil whose conductors run
along z at the radius ``source_radius_m``, optionally inside a perfectly conducting shield. With the time dependence
exp(+i w t) Maxwell's equations reduce to one scalar equation for Ez,

    laplace(Ez) + k^2 Ez = i w mu0 Jz,      k^2 = w^2 mu0 eps0 (eps_r - i sigma / (w eps0)),

and the magnetic field lies in the plane: Bx = (i / w) dEz/dy, By = -(i / w) dEz/dx.

Solution by separation of variables (Bessel series; the cylinder analogue of the Mie series, as used for MRI by
Glover et al. 1985, J Magn Reson 64:255, and for the sphere by Keltner et al. 1991, Magn Reson Med 22:467). A
surface current K(phi) = sum_n K_n exp(i n phi) (A/m) on the source radius drives each azimuthal order on its own:

    Ez(rho, phi) = sum_n K_n g_|n|(rho) exp(i n phi)

where g_m solves Bessel's equation of order m in every layer (J_m alone in the innermost layer; J_m and Y_m in the
others; the outgoing Hankel function H_m^(2) outside an unshielded coil, or the combination that vanishes on the
shield), with Ez and dEz/drho continuous at every interface and dEz/drho jumping by i w mu0 at the source for unit
K_n. The rotating components follow from the polar form of the curl:

    B1+        = (Bx + i By) / 2 =  (1 / 2w) sum_n K_n [g' - (n / rho) g] exp(i (n + 1) phi)
    conj(B1-)  = (Bx - i By) / 2 = -(1 / 2w) sum_n K_n [g' + (n / rho) g] exp(i (n - 1) phi)

so the order n = -1 alone (the quadrature birdcage mode) gives a B1+ that depends on rho only: J_0(k rho) in a
homogeneous cylinder, and a B1- that grows as J_2(k rho) from zero on the axis.

Sources provided: an ideal birdcage current sheet (order 1, quadrature or linear), a line current, and a birdcage of
N line currents. Angles are measured from +x towards +y; arrays are evaluated at any (y, x) in metres.
Powers are per metre of length (W/m) for currents in ampere (peak phasor amplitudes).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special

from .units import MU0, wavenumber

_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


class LayeredCylinder:
    """A cylinder of concentric layers on the axis of a coil.

    ``radii_m``: outer radius of every layer, increasing; ``eps_r``, ``sigma``: one value per layer, from the inside
    out (empty lists: an empty coil). ``source_radius_m``: radius of the coil conductors, larger than the cylinder.
    ``shield_radius_m``: radius of a perfectly conducting shield (Ez = 0), or None for an open coil that radiates.
    """

    def __init__(self, f_hz, radii_m=(), eps_r=(), sigma=(), source_radius_m=0.14, shield_radius_m=None):
        self.f_hz = float(f_hz)
        self.w = 2.0 * math.pi * self.f_hz
        self.radii = [float(r) for r in radii_m]
        self.eps_r = [float(e) for e in eps_r]
        self.sigma = [float(s) for s in sigma]
        if not (len(self.radii) == len(self.eps_r) == len(self.sigma)):
            raise ValueError("radii_m, eps_r and sigma need one entry per layer")
        if any(b <= a for a, b in zip([0.0] + self.radii, self.radii)):
            raise ValueError("layer radii must be positive and increasing")
        self.rs = float(source_radius_m)
        if self.radii and self.rs <= self.radii[-1]:
            raise ValueError("the source radius must be larger than the cylinder")
        self.shield = None if shield_radius_m is None else float(shield_radius_m)
        if self.shield is not None and self.shield <= self.rs:
            raise ValueError("the shield must be outside the source radius")
        self.k0 = complex(wavenumber(self.f_hz))
        # regions from the axis outwards: the layers, vacuum up to the source, vacuum beyond the source
        self.bounds = np.array([0.0] + self.radii + [self.rs, self.shield if self.shield is not None else np.inf])
        self.k = [complex(wavenumber(self.f_hz, e, s)) for e, s in zip(self.eps_r, self.sigma)] + [self.k0, self.k0]
        self._cache = {}

    # ------------------------------------------------------------------------------------------ radial functions
    def _basis(self, m, region):
        """Scaled radial functions of one region and their derivatives with respect to rho: a list of (f, df)."""
        k, r_in, r_out = self.k[region], self.bounds[region], self.bounds[region + 1]
        last = region == len(self.k) - 1

        def scaled(fun, dfun, ref):
            c = fun(m, k * ref)
            if not np.isfinite(c) or c == 0:
                raise ValueError(f"Bessel functions of order {m} are out of range for this geometry: lower the number of orders")
            return (lambda rho: fun(m, k * rho) / c), (lambda rho: k * dfun(m, k * rho) / c)

        if last and self.shield is None:
            return [scaled(special.hankel2, special.h2vp, r_in)]
        if last:
            jb, yb = special.jv(m, k * r_out), special.yv(m, k * r_out)
            c = special.jv(m, k * r_in) * yb - special.yv(m, k * r_in) * jb
            if not np.isfinite(c) or c == 0:
                raise ValueError(f"Bessel functions of order {m} are out of range for this geometry: lower the number of orders")
            return [((lambda rho: (special.jv(m, k * rho) * yb - special.yv(m, k * rho) * jb) / c),
                     (lambda rho: k * (special.jvp(m, k * rho) * yb - special.yvp(m, k * rho) * jb) / c))]
        out = [scaled(special.jv, special.jvp, r_out)]
        if region > 0:
            out.append(scaled(special.yv, special.yvp, r_in))
        return out

    def radial(self, m):
        """The radial function g_m of the order |n| = m for a unit surface-current coefficient K_n = 1 A/m.

        Returns dict(basis, coef, outer): ``coef[region]`` are the amplitudes of the scaled functions ``basis[region]``;
        ``outer`` is the amplitude C of the outgoing wave C H_m^(2)(k0 rho) beyond the source (open coil only)."""
        m = abs(int(m))
        if m in self._cache:
            return self._cache[m]
        nreg = len(self.k)
        basis = [self._basis(m, r) for r in range(nreg)]
        first = np.cumsum([0] + [len(b) for b in basis])
        a = np.zeros((first[-1], first[-1]), complex)
        rhs = np.zeros(first[-1], complex)
        for t in range(nreg - 1):                      # interface between region t and t + 1
            b = self.bounds[t + 1]
            for side, sign in ((t, 1.0), (t + 1, -1.0)):
                for j, (f, df) in enumerate(basis[side]):
                    a[2 * t, first[side] + j] = sign * f(b)
                    a[2 * t + 1, first[side] + j] = sign * df(b)
        rhs[2 * (nreg - 2) + 1] = -1j * self.w * MU0   # dEz/drho(outside) - dEz/drho(inside) = i w mu0 K at the source
        scale = np.abs(a).max(axis=0)
        c = np.linalg.solve(a / scale, rhs) / scale
        coef = [c[first[r]:first[r + 1]] for r in range(nreg)]
        outer = None if self.shield is not None else coef[-1][0] / special.hankel2(m, self.k0 * self.rs)
        self._cache[m] = dict(basis=basis, coef=coef, outer=outer)
        return self._cache[m]

    def g(self, m, rho):
        """g_m and dg_m/drho at the radii ``rho`` (any shape, metres)."""
        res = self.radial(m)
        rho = np.asarray(rho, float)
        g, dg = np.zeros(rho.shape, complex), np.zeros(rho.shape, complex)
        region = np.clip(np.searchsorted(self.bounds, rho, side="right") - 1, 0, len(self.k) - 1)
        if self.shield is not None:
            region = np.where(rho >= self.shield, -1, region)       # behind the shield: no field
        for r in range(len(self.k)):
            sel = region == r
            if not sel.any():
                continue
            for c, (f, df) in zip(res["coef"][r], res["basis"][r]):
                g[sel] += c * f(rho[sel])
                dg[sel] += c * df(rho[sel])
        return g, dg

    # ------------------------------------------------------------------------------------------ fields
    def fields(self, modes, y, x):
        """Fields of the surface current sum_n K_n exp(i n phi), ``modes`` = {n: K_n in A/m}, at (y, x) in metres.

        Returns dict(ez [V/m], bx, by, b1p, b1m [T], rho, phi). On the axis the limit is taken at 1e-9 m."""
        y, x = np.broadcast_arrays(np.asarray(y, float), np.asarray(x, float))
        rho = np.maximum(np.hypot(y, x), 1e-9)
        phi = np.arctan2(y, x)
        ez = np.zeros(rho.shape, complex)
        b1p = np.zeros(rho.shape, complex)
        b1m_conj = np.zeros(rho.shape, complex)
        for n, kn in modes.items():
            if kn == 0:
                continue
            g, dg = self.g(n, rho)
            ez += kn * g * np.exp(1j * n * phi)
            b1p += kn / (2 * self.w) * (dg - n / rho * g) * np.exp(1j * (n + 1) * phi)
            b1m_conj += -kn / (2 * self.w) * (dg + n / rho * g) * np.exp(1j * (n - 1) * phi)
        return dict(ez=ez, bx=b1p + b1m_conj, by=(b1p - b1m_conj) / 1j, b1p=b1p, b1m=np.conj(b1m_conj), rho=rho, phi=phi)

    def property_maps(self, y, x):
        """eps_r and sigma of the cylinder at (y, x): the layer values, vacuum outside."""
        rho = np.hypot(*np.broadcast_arrays(np.asarray(y, float), np.asarray(x, float)))
        idx = np.searchsorted(np.array(self.radii), rho, side="right") if self.radii else np.zeros(rho.shape, int)
        eps = np.array(self.eps_r + [1.0])[idx]
        sig = np.array(self.sigma + [0.0])[idx]
        return eps, sig

    def cell_maps(self, y, x, h_m):
        """eps_r and sigma averaged over square cells of side ``h_m`` centred at (y, x): the layer values weighted
        with the share of the cell each layer covers (``disc_fraction``). This is what a pixel grid holds, and with
        E along z the mean is the mixing rule of the long-wavelength limit (``solver``, module docstring)."""
        y, x = np.broadcast_arrays(np.asarray(y, float), np.asarray(x, float))
        eps, sig = np.ones(y.shape), np.zeros(y.shape)
        outer_e, outer_s = self.eps_r[1:] + [1.0], self.sigma[1:] + [0.0]
        for a, e_in, s_in, e_out, s_out in zip(self.radii, self.eps_r, self.sigma, outer_e, outer_s):
            frac = disc_fraction(y, x, h_m, a)
            eps += (e_in - e_out) * frac
            sig += (s_in - s_out) * frac
        return eps, sig

    # ------------------------------------------------------------------------------------------ powers (W per metre)
    def absorbed_power(self, modes):
        """Power absorbed in the cylinder per metre: integral of sigma |Ez|^2 / 2. The azimuthal orders are
        orthogonal, the radial integral is a 96-point Gauss-Legendre rule per layer."""
        total = 0.0
        for layer, (r_in, r_out) in enumerate(zip([0.0] + self.radii[:-1], self.radii)):
            if self.sigma[layer] == 0:
                continue
            rho = 0.5 * (r_out - r_in) * _GL_X + 0.5 * (r_out + r_in)
            wgt = 0.5 * (r_out - r_in) * _GL_W
            for n, kn in modes.items():
                g, _ = self.g(n, rho)
                total += 0.5 * self.sigma[layer] * 2 * math.pi * abs(kn) ** 2 * float(np.sum(wgt * rho * np.abs(g) ** 2))
        return total

    def source_power(self, modes):
        """Power delivered by the source per metre: -Re integral of Ez K* / 2 over the source circle."""
        return float(sum(-0.5 * 2 * math.pi * self.rs * abs(kn) ** 2 * np.real(self.g(n, np.array([self.rs]))[0][0])
                         for n, kn in modes.items()))

    def radiated_power(self, modes):
        """Power radiated to infinity per metre (open coil): sum_n |C_n|^2 2 / (w mu0), from the large-argument form
        of the Hankel function. Zero inside a shield."""
        if self.shield is not None:
            return 0.0
        return float(sum(abs(kn * self.radial(n)["outer"]) ** 2 * 2.0 / (self.w * MU0) for n, kn in modes.items()))


def disc_fraction(y, x, h_m, radius_m, n=64):
    """Share of each square cell (side ``h_m``, centre (y, x)) that lies inside the disc of radius ``radius_m`` about
    the origin: 0 or 1 away from the circle, and for the cells the circle cuts the integral of the chord length
    across the cell by the midpoint rule on ``n`` lines."""
    y, x = np.broadcast_arrays(np.asarray(y, float), np.asarray(x, float))
    rho = np.hypot(y, x)
    frac = (rho < radius_m).astype(float)
    cut = np.abs(rho - radius_m) < h_m
    if cut.any():
        yc, xc = y[cut][:, None], x[cut][:, None]
        xs = xc + ((np.arange(n) + 0.5) / n - 0.5) * h_m
        half = np.sqrt(np.maximum(radius_m ** 2 - xs ** 2, 0.0))
        length = np.clip(half, yc - h_m / 2, yc + h_m / 2) - np.clip(-half, yc - h_m / 2, yc + h_m / 2)
        frac[cut] = np.clip(length.mean(axis=1) / h_m, 0.0, 1.0)
    return frac


# ------------------------------------------------------------------------------------------------ sources
def sheet_modes(drive="quadrature", k0_a_per_m=1.0, phi0=0.0):
    """Fourier coefficients of the ideal birdcage current sheet of order 1.

    quadrature   K(phi) = K0 exp(-i (phi - phi0)): the pattern cos(w t - phi) turns from +x towards +y and makes B1+
    reverse      K(phi) = K0 exp(+i (phi - phi0)): the opposite sense, B1- only in an empty coil
    linear       K(phi) = K0 cos(phi - phi0): equal amounts of both"""
    if drive == "quadrature":
        return {-1: k0_a_per_m * np.exp(1j * phi0)}
    if drive == "reverse":
        return {1: k0_a_per_m * np.exp(-1j * phi0)}
    if drive == "linear":
        return {-1: 0.5 * k0_a_per_m * np.exp(1j * phi0), 1: 0.5 * k0_a_per_m * np.exp(-1j * phi0)}
    raise ValueError("drive must be 'quadrature', 'reverse' or 'linear'")


def line_modes(source_radius_m, phi, current_a=1.0, n_max=40):
    """Fourier coefficients of a line current I at (source radius, phi): K_n = I exp(-i n phi) / (2 pi rho'),
    truncated at |n| <= n_max. The series converges as (rho / rho')^n inside the source radius."""
    c = current_a / (2 * math.pi * source_radius_m)
    return {n: c * np.exp(-1j * n * phi) for n in range(-n_max, n_max + 1)}


def add_modes(*parts):
    """Sum of several mode dictionaries."""
    out = {}
    for p in parts:
        for n, v in p.items():
            out[n] = out.get(n, 0) + v
    return out


def birdcage_modes(source_radius_m, n_rungs=16, drive="quadrature", current_a=1.0, phi0=0.0, n_max=40):
    """A birdcage of ``n_rungs`` line currents at phi_k = phi0 + 2 pi k / N with the currents of ``rung_currents``."""
    phis = phi0 + 2 * math.pi * np.arange(n_rungs) / n_rungs
    return add_modes(*[line_modes(source_radius_m, p, c, n_max) for p, c in zip(phis, rung_currents(n_rungs, drive, current_a, phi0))])


def rung_currents(n_rungs=16, drive="quadrature", current_a=1.0, phi0=0.0):
    """Rung currents of the birdcage mode 1 at phi_k = phi0 + 2 pi k / N: I exp(-i (phi_k - phi0)) (quadrature, the
    sense that makes B1+), I exp(+i (phi_k - phi0)) (reverse), I cos(phi_k - phi0) (linear)."""
    d = 2 * math.pi * np.arange(n_rungs) / n_rungs
    if drive == "quadrature":
        return current_a * np.exp(-1j * d)
    if drive == "reverse":
        return current_a * np.exp(1j * d)
    if drive == "linear":
        return current_a * np.cos(d).astype(complex)
    raise ValueError("drive must be 'quadrature', 'reverse' or 'linear'")


# ------------------------------------------------------------------------------------------------ closed forms used as references
def free_space_line(f_hz, y, x, y0=0.0, x0=0.0, current_a=1.0):
    """Ez of a line current in free space: -(w mu0 I / 4) H_0^(2)(k0 |r - r0|) (Harrington 1961, section 5-6)."""
    w = 2 * math.pi * f_hz
    r = np.hypot(np.asarray(y, float) - y0, np.asarray(x, float) - x0)
    return -w * MU0 * current_a / 4.0 * special.hankel2(0, np.real(wavenumber(f_hz)) * r)


def quasi_static_b1(k0_a_per_m=1.0, n_rungs=None, current_a=1.0, source_radius_m=0.14, shield_radius_m=None):
    """|B1+| on the axis of an empty quadrature birdcage at zero frequency (magnetostatics).

    Current sheet K0 exp(-i phi): mu0 K0 / 2, uniform inside the coil. N rungs of current I: the same with
    K0 = N I / (2 pi R), exact on the axis for N >= 3. A shield at radius S multiplies by 1 - (R / S)^2
    (image currents)."""
    k0 = k0_a_per_m if n_rungs is None else n_rungs * current_a / (2 * math.pi * source_radius_m)
    shield = 1.0 if shield_radius_m is None else 1.0 - (source_radius_m / shield_radius_m) ** 2
    return MU0 * k0 / 2.0 * shield


def homogeneous_b1p_profile(f_hz, eps_r, sigma, rho):
    """B1+(rho) / B1+(0) inside a homogeneous cylinder under quadrature drive: J_0(k rho)."""
    return special.jv(0, wavenumber(f_hz, eps_r, sigma) * np.asarray(rho, float))


def homogeneous_b1m_profile(f_hz, eps_r, sigma, rho):
    """|B1-(rho)| / |B1+(0)| inside a homogeneous cylinder under quadrature drive: |J_2(k rho)|."""
    return np.abs(special.jv(2, wavenumber(f_hz, eps_r, sigma) * np.asarray(rho, float)))
