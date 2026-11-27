"""Finite-difference frequency-domain solver for the RF field on the slice (two dimensions, E along z).

Equation (time dependence exp(+i w t), see ``units``):

    laplace(Ez) + k0^2 eps_c Ez = i w mu0 Jz,        eps_c = eps_r - i sigma / (w eps0)

on a square grid of pitch h with Ez at the cell centres and any eps_r(y, x), sigma(y, x). The five-point stencil is
second order. A pixel that holds two materials takes the volume average of eps_c: with E along z the field is
tangential to every boundary in the plane, and for a tangential field the arithmetic mean is the mixing rule in the
limit of a pixel much smaller than the wavelength; the error is second order in pixel / wavelength.

Sources are line currents along z (A, peak phasor amplitude) at any position; a current is shared between the four
nearest cell centres with bilinear weights, and fields at a source are read with the same weights, which keeps the
discrete problem reciprocal. A channel is a list of line currents (one rung, a pair of opposite rungs, a whole
birdcage); the matrix is factorised once (``scipy.sparse.linalg.splu``) and reused for every channel.

Boundary. Open coil: a perfectly matched layer of ``n_pml`` cells on every side, a complex stretch of the
coordinates s(d) = 1 - i beta_max (d / L)^order written in the symmetric form
d/dx((s_y / s_x) dEz/dx) + d/dy((s_x / s_y) dEz/dy) + k0^2 eps_c s_x s_y Ez, with beta_max set from the round-trip
reflection asked for, R = exp(-2 k0 beta_max L / (order + 1)); behind it Ez = 0. The error this leaves is measured
against the free-space line current in ``validate`` (rows S1, S1b, S2); at low frequency the layer has to be thicker. Shielded coil: Ez = 0 on a circle. A cell next to
the circle uses a ghost value extrapolated linearly through the circle, which changes the diagonal only (the matrix
stays symmetric) and keeps second-order accuracy; cells outside the circle hold no field.

The magnetic field comes from centred differences of Ez: Bx = (i / w) dEz/dy, By = -(i / w) dEz/dx, then
B1+ = (Bx + i By) / 2 and B1- = (Bx - i By)* / 2.

Powers are per metre of length (W/m): absorbed = sum of sigma |Ez|^2 h^2 / 2; delivered by the sources =
-Re sum of Ez conj(I) / 2. SAR = sigma |Ez|^2 / (2 rho) in W/kg does not depend on the length.

Arrays are [y, x]; index i of an axis sits at (i - n // 2) h + origin, the engine's rule.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.ndimage
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .units import MU0, complex_permittivity, wavenumber

TEN_GRAM_SIDE_M = 0.0215443469       # side of a cube of 10 g at 1000 kg/m^3: (1e-5 m^3)^(1/3)
PML_MIN_HZ = 10.0e6                  # lowest frequency at which the absorbing layer is used at all (row S1b of validate)
PML_THIN_MIN_HZ = 30.0e6             # below this frequency the layer needs at least 24 cells


class Solver:
    """One frequency, one body, one factorisation.

    ``eps_r``, ``sigma``: square arrays [y, x] on the solver grid, including the rim of ``n_pml`` cells, which has to
    be free space. ``h_m``: pitch. ``origin_m``: (y, x) of the grid point n // 2. ``shield_radius_m``: a perfectly
    conducting circle about (0, 0); with a shield the absorbing layer is not needed (``n_pml=0``).
    """

    def __init__(self, f_hz, eps_r, sigma, h_m, origin_m=(0.0, 0.0), n_pml=16, shield_radius_m=None,
                 pml_reflection=1e-8, pml_order=3, permc_spec="MMD_ATA"):
        eps_r, sigma = np.asarray(eps_r, float), np.asarray(sigma, float)
        if eps_r.ndim != 2 or eps_r.shape[0] != eps_r.shape[1] or eps_r.shape != sigma.shape:
            raise ValueError("eps_r and sigma must be square arrays of the same shape")
        self.f_hz, self.w, self.h = float(f_hz), 2 * math.pi * float(f_hz), float(h_m)
        self.n = n = eps_r.shape[0]
        self.n_pml = int(n_pml)
        self.origin = (float(origin_m[0]), float(origin_m[1]))
        self.eps_r, self.sigma = eps_r, sigma
        self.k0 = float(np.real(wavenumber(self.f_hz)))
        self._order, self._refl = int(pml_order), float(pml_reflection)
        if self.n_pml and (self.f_hz < PML_MIN_HZ or (self.f_hz < PML_THIN_MIN_HZ and self.n_pml < 24)):
            raise ValueError(f"the absorbing layer is validated from {PML_THIN_MIN_HZ / 1e6:.0f} MHz with 16 cells and from "
                             f"{PML_MIN_HZ / 1e6:.0f} MHz with 24 cells (below, the stretch it needs is too steep for the grid): "
                             "use a thicker layer, or a shield and n_pml=0")
        self.shield = None if shield_radius_m is None else float(shield_radius_m)
        y, x = self.axes()
        self.y, self.x = np.meshgrid(y, x, indexing="ij")
        rim = np.ones((n, n), bool)
        if self.n_pml:
            rim[self.n_pml:n - self.n_pml, self.n_pml:n - self.n_pml] = False
        else:
            rim[:] = False
        if rim.any() and (np.abs(eps_r[rim] - 1).max() > 0 or np.abs(sigma[rim]).max() > 0):
            raise ValueError("the absorbing rim of the grid must be free space: enlarge the grid")
        self.physical = ~rim                                   # cells where the fields mean something
        inside = np.ones((n, n), bool) if self.shield is None else np.hypot(self.y, self.x) < self.shield
        if self.shield is not None:
            self.physical &= inside
            if (np.abs(eps_r[~inside] - 1).max(initial=0) > 0) or (np.abs(sigma[~inside]).max(initial=0) > 0):
                raise ValueError("the body reaches the shield")
        self.inside = inside
        self._lu = spla.splu(self._matrix().tocsc(), permc_spec=permc_spec)

    # ------------------------------------------------------------------------------------------ grid
    def axes(self):
        """Coordinates (y, x) in metres of the grid points along each axis."""
        i = (np.arange(self.n) - self.n // 2) * self.h
        return i + self.origin[0], i + self.origin[1]

    def _stretch(self, u):
        """Coordinate stretch s of the absorbing layer at the (fractional) index u along one axis."""
        if not self.n_pml:
            return np.ones(np.shape(u), complex)
        depth = np.maximum(np.maximum(self.n_pml - 0.5 - u, u - (self.n - self.n_pml - 0.5)), 0.0) / self.n_pml
        beta = -(self._order + 1) * math.log(self._refl) / (2 * self.k0 * self.n_pml * self.h)
        return 1.0 - 1j * beta * depth ** self._order

    def _matrix(self):
        """The system matrix, multiplied by h^2 (complex symmetric)."""
        n = self.n
        idx = np.arange(n * n).reshape(n, n)
        u = np.arange(n, dtype=float)
        s_c, s_h = self._stretch(u), self._stretch(u[:-1] + 0.5)           # at cell centres, at the faces between cells
        s_lo, s_hi = self._stretch(np.array([-0.5])), self._stretch(np.array([n - 0.5]))
        # link coefficients: along x between (i, j) and (i, j + 1): s_y(i) / s_x(j + 1/2); along y likewise
        ax = s_c[:, None] / s_h[None, :]                                   # [n, n - 1]
        ay = s_c[None, :] / s_h[:, None]                                   # [n - 1, n]
        diag = self.k0 ** 2 * self.h ** 2 * complex_permittivity(self.f_hz, self.eps_r, self.sigma) * (s_c[:, None] * s_c[None, :])
        diag = diag.astype(complex)
        diag[:, :-1] -= ax
        diag[:, 1:] -= ax
        diag[:-1, :] -= ay
        diag[1:, :] -= ay
        # the outer wall (Ez = 0 half a cell beyond the last cell centre is not used: the ghost cell is zero)
        diag[:, 0] -= s_c / s_lo
        diag[:, -1] -= s_c / s_hi
        diag[0, :] -= s_c / s_lo
        diag[-1, :] -= s_c / s_hi
        link_x = ax.copy()
        link_y = ay.copy()
        if self.shield is not None:
            ins = self.inside
            for links, a, b, axis in ((link_x, ins[:, :-1], ins[:, 1:], 1), (link_y, ins[:-1, :], ins[1:, :], 0)):
                cut = a != b                                               # the link crosses the shield
                theta = self._crossing(axis)[cut]                          # share of the link that lies inside
                # the inside cell sees a ghost value -(1 - theta) / theta E_c: its diagonal gains 1 - 1 / theta
                extra = np.zeros(links.shape, complex)
                extra[cut] = links[cut] * (1.0 - 1.0 / np.maximum(theta, 1e-6))
                lo = [slice(None), slice(None)]
                hi = [slice(None), slice(None)]
                lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                diag[tuple(lo)] += np.where(a & cut, extra, 0)
                diag[tuple(hi)] += np.where(b & cut, extra, 0)
                links[cut | (~a & ~b)] = 0.0
            diag[~ins] = 1.0
        rows = [idx.ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel()]
        cols = [idx.ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel()]
        vals = [diag.ravel(), link_x.ravel(), link_x.ravel(), link_y.ravel(), link_y.ravel()]
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n)).tocsr()

    def _crossing(self, axis):
        """For every link along ``axis``: the share of the link, measured from its end inside the shield, that lies
        inside the shield circle (meaningful only where the link crosses the circle)."""
        y, x, r = self.y, self.x, self.shield
        if axis == 1:
            p0, p1, q = x[:, :-1], x[:, 1:], y[:, :-1]
        else:
            p0, p1, q = y[:-1, :], y[1:, :], x[:-1, :]
        edge = np.sqrt(np.maximum(r * r - q * q, 0.0))                    # the circle meets this grid line at +-edge
        first_inside = p0 * p0 + q * q < r * r                             # which end of the link is the inner one
        # inner end at p0: the link runs towards +, it leaves the circle at +edge; inner end at p1: at -edge
        return np.clip(np.where(first_inside, edge - p0, p1 + edge) / self.h, 0.0, 1.0)

    # ------------------------------------------------------------------------------------------ solve
    def _weights(self, y_m, x_m):
        """The four cells around a point and their bilinear weights."""
        fy = (y_m - self.origin[0]) / self.h + self.n // 2
        fx = (x_m - self.origin[1]) / self.h + self.n // 2
        i0, j0 = int(math.floor(fy)), int(math.floor(fx))
        if not (self.n_pml <= i0 < self.n - self.n_pml - 1 and self.n_pml <= j0 < self.n - self.n_pml - 1):
            raise ValueError("a source or probe lies outside the physical part of the grid")
        ty, tx = fy - i0, fx - j0
        return [(i0, j0, (1 - ty) * (1 - tx)), (i0, j0 + 1, (1 - ty) * tx), (i0 + 1, j0, ty * (1 - tx)), (i0 + 1, j0 + 1, ty * tx)]

    def sample(self, field, y_m, x_m):
        """A field (last two axes [y, x]) at a point, with the bilinear weights of the sources."""
        return sum(wgt * field[..., i, j] for i, j, wgt in self._weights(y_m, x_m))

    def solve(self, channels):
        """Ez of every channel for unit drive. ``channels``: a list of channels, each a list of line currents
        (y_m, x_m, current in A, complex). Returns Ez [channel, y, x] in V/m."""
        n = self.n
        rhs = np.zeros((n * n, len(channels)), complex)
        for c, lines in enumerate(channels):
            for y_m, x_m, cur in lines:
                for i, j, wgt in self._weights(y_m, x_m):
                    if not self.inside[i, j]:
                        raise ValueError("a source lies on or outside the shield")
                    rhs[i * n + j, c] += 1j * self.w * MU0 * cur * wgt
        ez = self._lu.solve(rhs)
        return np.ascontiguousarray(ez.T).reshape(len(channels), n, n)

    def b_fields(self, ez):
        """Bx, By, B1+ and B1- (tesla) from Ez (last two axes [y, x]) by centred differences."""
        d_y = np.gradient(ez, self.h, axis=-2)
        d_x = np.gradient(ez, self.h, axis=-1)
        bx, by = 1j / self.w * d_y, -1j / self.w * d_x
        return dict(bx=bx, by=by, b1p=(bx + 1j * by) / 2, b1m=np.conj(bx - 1j * by) / 2)

    def fields(self, channels):
        """Per-channel fields for unit drive: dict(ez [V/m per unit], bx, by, b1p, b1m [T per unit]), each
        [channel, y, x]. Values in the absorbing rim and outside the shield are not physical (``self.physical``)."""
        ez = self.solve(channels)
        return dict(self.b_fields(ez), ez=ez, channels=channels)

    # ------------------------------------------------------------------------------------------ powers, W per metre
    def power_matrix(self, ez):
        """R[i, j] = sum of sigma conj(E_i) E_j h^2 (ohm per metre): the power absorbed for channel weights w is
        Re(w^H R w) / 2. Hermitian and positive semi-definite. It is also the noise resistance matrix of the same
        elements used as receivers (Roemer et al. 1990)."""
        e = ez.reshape(ez.shape[0], -1)
        return (np.conj(e) * self.sigma.ravel()) @ e.T * self.h ** 2

    def absorbed_power(self, ez_total, mask=None):
        """Power absorbed per metre by a combined field: sum of sigma |Ez|^2 h^2 / 2 (over ``mask`` if given)."""
        p = 0.5 * self.sigma * np.abs(ez_total) ** 2 * self.h ** 2
        return float(p.sum() if mask is None else p[mask].sum())

    def source_power(self, ez, channels, weights=None):
        """Power delivered by the sources per metre for channel weights w: -Re sum Ez_total conj(w_c I) / 2, with
        Ez_total read at each line current."""
        w = np.ones(len(channels), complex) if weights is None else np.asarray(weights, complex)
        total = np.tensordot(w, ez, 1)
        return float(sum(-0.5 * np.real(self.sample(total, y_m, x_m) * np.conj(w[c] * cur))
                         for c, lines in enumerate(channels) for y_m, x_m, cur in lines))

    def box(self, half_width_m):
        """Mask of the square of cells within ``half_width_m`` of the grid centre (the box of ``outward_flux``)."""
        c, m = self.n // 2, int(round(half_width_m / self.h))
        if c - m - 1 < self.n_pml or c + m + 1 >= self.n - self.n_pml:
            raise ValueError("the box reaches the absorbing rim")
        out = np.zeros((self.n, self.n), bool)
        out[c - m:c + m + 1, c - m:c + m + 1] = True
        return out

    def outward_flux(self, ez_total, half_width_m):
        """Power leaving a square box of cells about the grid centre (W per metre): the Poynting vector
        Re(E x H*) / 2 through the faces of the outermost cells of the box, with H on a face from the difference of
        the two cells next to it and Ez their mean. With this choice sources = absorbed in the box + flux holds for
        the difference equations themselves, to rounding (the discrete form of Poynting's theorem); the flux is then
        the same through every box that encloses the sources and the body."""
        box = self.box(half_width_m)
        rows = np.flatnonzero(box.any(1))
        lo, hi = int(rows[0]), int(rows[-1])
        e = ez_total
        s = slice(lo, hi + 1)
        pairs = [(e[s, hi], e[s, hi + 1]), (e[s, lo], e[s, lo - 1]), (e[hi, s], e[hi + 1, s]), (e[lo, s], e[lo - 1, s])]
        return float(sum(np.sum(np.imag(e_in * np.conj(e_out))) for e_in, e_out in pairs)) / (2 * self.w * MU0)

    # ------------------------------------------------------------------------------------------ SAR
    def point_sar(self, ez_total, density):
        """SAR at every pixel, W/kg: sigma |Ez|^2 / (2 rho); zero where the density is zero (air)."""
        rho = np.asarray(density, float)
        return np.where(rho > 0, 0.5 * self.sigma * np.abs(ez_total) ** 2 / np.maximum(rho, 1e-12), 0.0)

    def averaging_kernel(self, side_m=TEN_GRAM_SIDE_M):
        """Weights along one axis of the averaging square of side ``side_m`` about a pixel centre: the share of each
        pixel that lies inside the square (1 for a pixel wholly inside, a fraction for the two at the ends). The
        square is never smaller than one pixel. The two-dimensional weights are the outer product."""
        half = max(side_m / self.h / 2.0, 0.5)
        m = int(math.ceil(half - 0.5 - 1e-12))
        j = np.arange(-m, m + 1, dtype=float)
        return np.clip(np.minimum(j + 0.5, half) - np.maximum(j - 0.5, -half), 0.0, 1.0)

    def _window_sum(self, a, kernel):
        """Sum of a map over the averaging square about every pixel (zero outside the grid)."""
        out = scipy.ndimage.correlate1d(np.asarray(a, float), kernel, axis=0, mode="constant")
        return scipy.ndimage.correlate1d(out, kernel, axis=1, mode="constant")

    def averaged_sar(self, ez_total, density, side_m=TEN_GRAM_SIDE_M, min_fill=0.5):
        """SAR averaged over a square of side ``side_m`` centred on every pixel, W/kg: the power absorbed in the
        square divided by the mass in the square.

        The two-dimensional stand-in for the 10 g average: the fields do not change along z, so the average over a
        cube of side L equals the average over its L x L cross-section; L = 21.5 mm is the side of a 10 g cube at
        1000 kg/m^3. Pixels cut by the edge of the square count with the share inside it, so the result does not
        depend on how L compares with the pixel. The square has a fixed size (it is not grown until it holds 10 g),
        and the value is defined only where the square holds at least ``min_fill`` of the mass it would hold if it
        were full of tissue at 1000 kg/m^3 (elsewhere 0)."""
        kernel = self.averaging_kernel(side_m)
        rho = np.asarray(density, float)
        power = self._window_sum(0.5 * self.sigma * np.abs(ez_total) ** 2, kernel)
        mass = self._window_sum(rho, kernel)
        ok = mass >= min_fill * 1000.0 * kernel.sum() ** 2
        return np.where(ok, power / np.maximum(mass, 1e-12), 0.0)

    def sar_support(self, density, side_m=None, min_fill=0.5):
        """The pixels at which local SAR has a value: for point SAR (``side_m=None``) those that hold mass, for the
        averaged SAR those whose square is full enough. A limit on local SAR has to be imposed on all of them."""
        rho = np.asarray(density, float)
        if side_m is None:
            return rho > 0
        kernel = self.averaging_kernel(side_m)
        return self._window_sum(rho, kernel) >= min_fill * 1000.0 * kernel.sum() ** 2

    def sar_matrices(self, ez, density, mask, side_m=None, min_fill=0.5):
        """The quadratic forms of local SAR: Q[p] (W/kg per A^2, Hermitian, [pixel, channel, channel]) with
        SAR_p(w) = Re(w^H Q[p] w) for the pixels of ``mask``.

        ``side_m=None``: point SAR, Q[p] = sigma_p conj(e_p) e_p^T / (2 rho_p). Otherwise the average over the
        square of ``averaged_sar`` (pixels whose square is too empty get a zero matrix). A limit on local SAR holds
        only at the pixels of ``mask``: pass ``sar_support`` (every pixel at which the map has a value), not the
        body mask of a world, which leaves out the partly filled pixels of the surface."""
        rho = np.asarray(density, float)
        nc = ez.shape[0]
        q = np.zeros((int(mask.sum()), nc, nc), complex)
        if side_m is None:
            scale = np.where(rho > 0, 0.5 * self.sigma / np.maximum(rho, 1e-12), 0.0)[mask]
            e = ez[:, mask]
            return np.einsum("p,ip,jp->pij", scale, np.conj(e), e)
        kernel = self.averaging_kernel(side_m)
        mass = self._window_sum(rho, kernel)
        ok = (mass >= min_fill * 1000.0 * kernel.sum() ** 2)[mask]
        inv = np.where(ok, 1.0 / np.maximum(mass[mask], 1e-12), 0.0)
        for i in range(nc):
            for j in range(i, nc):
                prod = 0.5 * self.sigma * np.conj(ez[i]) * ez[j]
                avg = (self._window_sum(prod.real, kernel) + 1j * self._window_sum(prod.imag, kernel))[mask] * inv
                q[:, i, j] = avg
                q[:, j, i] = np.conj(avg)
        return q


def combine(fields, weights):
    """Fields of a set of complex channel weights: the weighted sum of the per-channel maps.
    Returns dict(ez, bx, by, b1p, b1m). Note that B1- is the conjugate of a linear combination: for transmit weights
    w the counter-rotating field is conj(sum w_c conj(B1-_c))."""
    w = np.asarray(weights, complex)
    out = {k: np.tensordot(w, fields[k], 1) for k in ("ez", "bx", "by", "b1p")}
    out["b1m"] = np.conj(np.tensordot(w, np.conj(fields["b1m"]), 1))
    return out


# ------------------------------------------------------------------------------------------------ worlds
def block_mean(a, k):
    """Mean of an array over k x k blocks (the array's sides must be multiples of k)."""
    n0, n1 = a.shape
    return a.reshape(n0 // k, k, n1 // k, k).mean(axis=(1, 3))


def embed(maps, d_m, factor, half_width_m, n_pml, fill):
    """Coarsen master-grid maps by ``factor`` and place them in a larger solver grid.

    ``maps``: arrays [n, n] on a grid of pitch ``d_m`` whose point n // 2 is the isocentre; ``fill``: the value of
    each map outside the master grid (free space). The solver pitch is factor x d; the solver grid covers at least
    +-half_width_m plus ``n_pml`` cells on every side. Returns (arrays, h, origin, offset) where ``offset`` is the
    solver index of the first coarse cell of the master grid."""
    n = maps[0].shape[0]
    if n % factor:
        raise ValueError(f"the grid of {n} points cannot be coarsened by {factor}")
    m, h = n // factor, d_m * factor
    # the centre of coarse cell b is ((b + 1/2) factor - 1/2 - n // 2) d from the isocentre
    first = (0.5 * factor - 0.5 - n // 2) * d_m
    pad = max(0, int(math.ceil((half_width_m + first) / h))) + n_pml      # cells added on the low side
    pad_hi = max(0, int(math.ceil((half_width_m - (first + (m - 1) * h)) / h))) + n_pml
    pad = max(pad, pad_hi)
    size = m + 2 * pad
    out = []
    for a, f in zip(maps, fill):
        big = np.full((size, size), f, float)
        big[pad:pad + m, pad:pad + m] = block_mean(np.asarray(a, float), factor)
        out.append(big)
    origin = first + (size // 2 - pad) * h
    return out, h, (origin, origin), pad


def to_master(field, solver, n, d_m, order=1):
    """A solver-grid map on the master grid (n points of pitch d, point n // 2 at the isocentre), by spline
    interpolation of the given order (real and imaginary part separately).

    ``order=1`` (bilinear) never leaves the range of the values around a point: use it for maps with jumps (SAR,
    conductivity, density). ``order=3`` is more accurate for the smooth maps (B1+, B1-, Ez) and overshoots at jumps:
    a cubic spline of a point-SAR map goes negative next to a boundary."""
    pos = ((np.arange(n) - n // 2) * d_m - solver.origin[0]) / solver.h + solver.n // 2
    pos_x = ((np.arange(n) - n // 2) * d_m - solver.origin[1]) / solver.h + solver.n // 2
    coords = np.meshgrid(pos, pos_x, indexing="ij")
    f = np.asarray(field)
    if np.iscomplexobj(f):
        return (scipy.ndimage.map_coordinates(f.real, coords, order=order, mode="nearest")
                + 1j * scipy.ndimage.map_coordinates(f.imag, coords, order=order, mode="nearest"))
    return scipy.ndimage.map_coordinates(f, coords, order=order, mode="nearest")


class WorldModel:
    """The RF problem of one engine world at one frequency: a ``Solver`` on the coarsened, enlarged grid, the
    density map and the masks on that grid, and the way back to the master grid.

    ``factor``: the solver pitch in master-grid pixels (the properties are block means, the long-wavelength mixing
    rule here). ``reach_m``: half-width of the physical part of the solver grid; it has to hold the coil.
    ``shield_radius_m``: a shield about the isocentre (then no absorbing layer is used). ``overrides``: see
    ``properties.world_maps``."""

    def __init__(self, world, f_hz, factor=5, reach_m=0.17, n_pml=16, shield_radius_m=None, overrides=None):
        from . import properties
        self.n_master, self.d = int(world["n"]), float(world["extent_mm"]) * 1e-3 / int(world["n"])
        self.factor = int(factor)
        eps, sig, rho = properties.world_maps(world, f_hz, overrides)
        if shield_radius_m is not None:
            n_pml, reach_m = 0, max(reach_m, shield_radius_m + 2 * self.d * self.factor)
        body = np.asarray(world["masks"]["body"], float)
        organ = np.asarray(world["masks"]["brain"], float)
        (eps, sig, rho, body, organ), h, origin, self.offset = embed(
            [eps, sig, rho, body, organ], self.d, self.factor, reach_m, n_pml, [1.0, 0.0, 0.0, 0.0, 0.0])
        self.density = rho
        # body, organ: the masks of the world (more than half of a solver pixel); tissue: every pixel that holds mass
        self.masks = dict(body=body > 0.5, organ=organ > 0.5, tissue=rho > 0)
        self.solver = Solver(f_hz, eps, sig, h, origin, n_pml=n_pml, shield_radius_m=shield_radius_m)

    def to_world(self, field, order=1):
        """A map of the solver grid on the master grid of the world (``to_master``: order 1 for maps with jumps such
        as SAR, order 3 for the smooth field maps)."""
        return to_master(field, self.solver, self.n_master, self.d, order)
