"""Validation of the radio-frequency field model: every function against a closed form or an independent computation.

    python -m mrgym.em.validate                 the table as text; exit status 1 if a row fails
    python -m mrgym.em.validate --markdown      the validation table of docs/physics/em.md
    python -m mrgym.em.validate --tissues       the tissue table of docs/physics/em.md
    python -m mrgym.em.validate --demo          the demonstration table of docs/physics/em.md (brain world)
    python -m mrgym.em.validate --timing        one factorisation and 8 channel solves on the brain world at 298 MHz
    python -m mrgym.em.validate --cost3d        what a three-dimensional time-domain solver would cost (an estimate)
    python -m mrgym.em.validate --write-doc docs/physics/em.md      refresh the generated blocks of that page

A row has a quantity, the reference it is compared with, the largest error found and a tolerance. Three kinds of
reference are used: a closed form (Bessel and Hankel functions, magnetostatics); an independent computation that
shares no code with the function under test (the Bessel series against the finite-difference solver, a direct sum
over wires, numerical derivatives of the analytic fields, a brute-force loop, a dense scan of a two-channel shim);
and a published or tabulated number (the calculator of IFAC-CNR for the tissue model, the size of the central
brightening). The references were written by the same authors as the code: the rows show that the code solves the
stated two-dimensional equations. They do not show that a two-dimensional model describes a head in a coil; what
that costs is in docs/physics/em.md, section 7.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
from scipy import special

from . import analytic, coils, ept, properties, shim, solver
from .units import MU0, wavelength_m, wavenumber

GROUPS = dict(properties="Tissue properties", analytic="Closed-form solutions", solver="Finite-difference solver",
              coils="Coils, reciprocity and reception", shim="RF shimming", ept="Electrical properties tomography")

COIL_R, SHIELD_R, RUNGS = 0.14, 0.19, 16
FREQS = (64e6, 128e6, 298e6)
WORLD_SEED = "0123456789abcdef0123456789abcdef"


def _row(id, group, quantity, reference, measured, tolerance, kind="max", unit=""):
    """kind "max": passes when measured <= tolerance; "min": when measured >= tolerance."""
    measured = float(measured)
    ok = bool(np.isfinite(measured) and (measured <= tolerance if kind == "max" else measured >= tolerance))
    return dict(id=id, group=group, quantity=quantity, reference=reference, measured=measured, tolerance=float(tolerance), kind=kind,
                unit=unit, ok=ok)


def _order(errors):
    """Convergence order from errors at successively halved steps: the mean of log2 of the ratios."""
    return float(np.mean([math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]))


def bodies(f_hz):
    """The three test cylinders at one frequency, with the tissue values of ``properties``: (radii, eps_r, sigma)."""
    gm, wm = properties.cole_cole("Brain (Grey Matter)", f_hz), properties.cole_cole("Brain (White Matter)", f_hz)
    csf, fat, mus = (properties.cole_cole(k, f_hz) for k in ("Cerebro Spinal Fluid", "Fat (Not Infiltrated)", "Muscle"))
    brain = (0.5 * (gm[0] + wm[0]), 0.5 * (gm[1] + wm[1]))
    return dict(homogeneous=([0.09], [brain[0]], [brain[1]]),
                fat_muscle=([0.08, 0.095], [mus[0], fat[0]], [mus[1], fat[1]]),
                csf_brain=([0.075, 0.09], [brain[0], csf[0]], [brain[1], csf[1]]))


def _grid(h, shield=None, n_pml=16, reach=0.17):
    """Square solver grid for the test coil: (n, n_pml, Y, X)."""
    n_pml = n_pml if shield is None else 0
    half = reach if shield is None else shield + 2 * h
    n = 2 * int(round(half / h)) + 1 + 2 * n_pml
    y = (np.arange(n) - n // 2) * h
    yy, xx = np.meshgrid(y, y, indexing="ij")
    return n, n_pml, yy, xx


def cylinder_case(f_hz, body, h, shield=None, drive="quadrature", n_pml=16, stride=1):
    """The solver and the Bessel series on the same problem: a layered cylinder in a 16-rung birdcage.

    Returns the relative errors of the solver inside the cylinder (on every ``stride``-th grid point): Ez (largest,
    relative to the largest |Ez|), B1+ and B1- (rms relative to rms; B1+ largest away from the layer boundaries and
    largest anywhere, relative to the largest |B1+|), absorbed and delivered power; and the objects."""
    radii, eps, sig = body
    cyl = analytic.LayeredCylinder(f_hz, radii, eps, sig, COIL_R, shield)
    n, n_pml, yy, xx = _grid(h, shield, n_pml)
    e_map, s_map = cyl.cell_maps(yy, xx, h)
    sol = solver.Solver(f_hz, e_map, s_map, h, n_pml=n_pml, shield_radius_m=shield)
    channels = coils.birdcage(COIL_R, RUNGS, drive)
    f = sol.fields(channels)
    modes = analytic.birdcage_modes(COIL_R, RUNGS, drive, n_max=40)
    rho = np.hypot(yy, xx)
    sel = rho < radii[-1]
    keep = np.zeros_like(sel)
    keep[::stride, ::stride] = True
    sel &= keep
    ref = cyl.fields(modes, yy[sel], xx[sel])
    away = np.ones(int(sel.sum()), bool)
    for a in radii:
        away &= np.abs(rho[sel] - a) > 1.5 * h
    d = np.abs(f["b1p"][0][sel] - ref["b1p"])
    top = np.abs(ref["b1p"]).max()
    p_abs, p_src = sol.absorbed_power(f["ez"][0]), sol.source_power(f["ez"], channels)
    return dict(
        ez=np.abs(f["ez"][0][sel] - ref["ez"]).max() / np.abs(ref["ez"]).max(),
        b1p_rms=math.sqrt(np.mean(d ** 2) / np.mean(np.abs(ref["b1p"]) ** 2)), b1p_away=d[away].max() / top, b1p_max=d.max() / top,
        b1m_rms=math.sqrt(np.mean(np.abs(f["b1m"][0][sel] - ref["b1m"]) ** 2) / np.mean(np.abs(ref["b1m"]) ** 2)),
        p_abs=abs(p_abs / cyl.absorbed_power(modes) - 1), p_src=abs(p_src / cyl.source_power(modes) - 1),
        solver=sol, fields=f, cyl=cyl, modes=modes, channels=channels, yy=yy, xx=xx)


_world_cache = {}


def brain_world():
    """The brain world used by the rows and the demonstration (a fixed seed, 3 T)."""
    if "w" not in _world_cache:
        from ..scanner import world
        _world_cache["w"] = world.make_world(WORLD_SEED, dict(kind="brain", b0_t=3.0))
    return _world_cache["w"]


def brain_model(f_hz, factor=5, shield=None):
    key = (f_hz, factor, shield)
    if key not in _world_cache:
        _world_cache[key] = solver.WorldModel(brain_world(), f_hz, factor=factor, shield_radius_m=shield)
    return _world_cache[key]


# ---------------------------------------------------------------------------------------------- tissue properties
def check_properties():
    rows = []
    dev = [0.0, 0.0, 0.0, 0.0]
    for row, by_f in properties.CHECK_VALUES.items():
        for mhz, (eps, sig, lam, depth) in by_f.items():
            e, s = properties.cole_cole(row, mhz * 1e6)
            got = (e, s, float(wavelength_m(mhz * 1e6, e, s)), -1.0 / float(np.imag(wavenumber(mhz * 1e6, e, s))))
            dev = [max(a, abs(g / r - 1)) for a, g, r in zip(dev, got, (eps, sig, lam, depth))]
    rows.append(_row("P1", "properties", f"four-term Cole-Cole model: permittivity and conductivity of {len(properties.CHECK_VALUES)} tissue rows at 64, 128 "
                     "and 298 MHz", "the values returned by the IFAC-CNR calculator for the same rows (six digits)", max(dev[:2]), 1e-4, unit="relative"))
    rows.append(_row("P2", "properties", "wavelength 2 pi / Re k = c / (f Re sqrt(eps_complex)) and penetration depth -1 / Im k of the same rows",
                     "the wavelength and penetration depth returned by that calculator", max(dev[2:]), 1e-4, unit="relative"))
    # every engine class has a row, a density and finite positive values; mixing is linear
    bad = 0
    for name in properties.CLASS_ROWS:
        for f in FREQS:
            rec = properties.tissue(name, f)
            bad += not (rec["eps_r"] > 1 and rec["sigma"] > 0 and 800 < rec["density"] < 2000)
    from ..scanner import tissue as engine_tissue
    missing = [k for k in engine_tissue.NAMES if k not in properties.CLASS_ROWS]
    rows.append(_row("P3", "properties", f"every tissue class of the engine ({len(engine_tissue.NAMES)}) has a row and a density; permittivity above 1, "
                     "conductivity above 0, density 800 to 2000 kg/m^3 at the three frequencies", "by construction", bad + len(missing), 0))
    # a hand-made world of four pixels: pure grey matter, pure CSF, half and half, air
    tiny = dict(kind="brain", names=["gm", "csf"], weights=np.array([[[1.0, 0.0, 0.5, 0.0]], [[0.0, 1.0, 0.5, 0.0]]]), tissues={})
    eps, sig, rho = properties.world_maps(tiny, 128e6)
    g, c = properties.tissue("gm", 128e6), properties.tissue("csf", 128e6)
    want = np.array([[g[k], c[k], 0.5 * (g[k] + c[k]), v] for k, v in (("eps_r", 1.0), ("sigma", 0.0), ("density", 0.0))])
    err = max(np.abs(got[0] / w_ - 1).max() if w_.min() > 0 else np.abs(got[0] - w_).max() for got, w_ in zip((eps, sig, rho), want))
    over = properties.world_maps(tiny, 128e6, overrides=dict(gm=dict(sigma=1.0)))[1][0, 0]
    rows.append(_row("P4", "properties", "property maps of a hand-made world of four pixels (grey matter, CSF, half of each, air), and an override of "
                     "one conductivity", "the two records, their mean, and permittivity 1 without conductivity or mass; the overriding value",
                     max(err, abs(over - 1.0)), 1e-12))
    return rows


# ---------------------------------------------------------------------------------------------- closed forms
def check_analytic():
    rows = []
    yy, xx = np.meshgrid(np.linspace(-0.1, 0.1, 41), np.linspace(-0.1, 0.1, 41), indexing="ij")
    inside = np.hypot(yy, xx) < 0.1

    # A1 addition theorem: the series of a line current against the Hankel function
    err = 0.0
    for f in FREQS:
        phi0 = 0.7
        cyl = analytic.LayeredCylinder(f, source_radius_m=COIL_R)
        got = cyl.fields(analytic.line_modes(COIL_R, phi0, 1.0, 60), yy[inside], xx[inside])["ez"]
        ref = analytic.free_space_line(f, yy[inside], xx[inside], COIL_R * math.sin(phi0), COIL_R * math.cos(phi0))
        err = max(err, np.abs(got - ref).max() / np.abs(ref).max())
    rows.append(_row("A1", "analytic", "Ez of one line current in an empty coil: Bessel series to order 60, inside 0.71 of the source radius, "
                     "64 / 128 / 298 MHz", "-(w mu0 I / 4) H0^(2)(k0 |r - r'|) (Graf's addition theorem)", err, 1e-8, unit="relative"))

    # A2 quasi-static limit of the current sheet
    f = 1e5
    err = 0.0
    for sh in (None, SHIELD_R):
        cyl = analytic.LayeredCylinder(f, source_radius_m=COIL_R, shield_radius_m=sh)
        fld = cyl.fields(analytic.sheet_modes("quadrature", 1.0), yy[inside], xx[inside])
        ref = analytic.quasi_static_b1(1.0, source_radius_m=COIL_R, shield_radius_m=sh)
        err = max(err, np.abs(np.abs(fld["b1p"]) - ref).max() / ref, np.abs(fld["b1m"]).max() / ref)
    rows.append(_row("A2", "analytic", "empty quadrature current sheet at 100 kHz, open and shielded: |B1+| over the bore and |B1-|",
                     "magnetostatics: B1+ = mu0 K0 / 2, uniform, times 1 - (R / S)^2 inside a shield; B1- = 0", err, 1e-6, unit="relative"))

    # A3 a birdcage of N rungs against a direct sum over wires
    cyl = analytic.LayeredCylinder(f, source_radius_m=COIL_R)
    near = np.hypot(yy, xx) < 0.09
    fld = cyl.fields(analytic.birdcage_modes(COIL_R, RUNGS, "quadrature", n_max=40), yy[near], xx[near])
    phi, ys, xs = coils.ring(COIL_R, RUNGS)
    cur = analytic.rung_currents(RUNGS, "quadrature")
    bx, by = np.zeros(int(near.sum()), complex), np.zeros(int(near.sum()), complex)
    for y0, x0, c in zip(ys, xs, cur):                     # field of a straight wire: mu0 I / (2 pi r), around the wire
        dy, dx = yy[near] - y0, xx[near] - x0
        r2 = dy * dy + dx * dx
        bx += -MU0 * c / (2 * math.pi) * dy / r2
        by += MU0 * c / (2 * math.pi) * dx / r2
    err = np.abs(fld["b1p"] - (bx + 1j * by) / 2).max() / np.abs(fld["b1p"]).max()
    centre = abs(cyl.fields(analytic.birdcage_modes(COIL_R, RUNGS, "quadrature", n_max=40), np.zeros(1), np.zeros(1))["b1p"][0])
    err = max(err, abs(centre / analytic.quasi_static_b1(n_rungs=RUNGS, source_radius_m=COIL_R) - 1))
    rows.append(_row("A3", "analytic", "birdcage of 16 rungs at 100 kHz: B1+ inside 0.64 of the coil radius, and on the axis",
                     "direct sum of the fields mu0 I / (2 pi r) of 16 straight wires; on the axis mu0 N I / (4 pi R)", err, 1e-6, unit="relative"))

    # A4 homogeneous lossy cylinder: J0 and J2
    err = 0.0
    for f in FREQS:
        radii, eps, sig = bodies(f)["homogeneous"]
        cyl = analytic.LayeredCylinder(f, radii, eps, sig, COIL_R)
        sel = np.hypot(yy, xx) < radii[0]
        fld = cyl.fields(analytic.sheet_modes("quadrature"), yy[sel], xx[sel])
        c = cyl.fields(analytic.sheet_modes("quadrature"), np.zeros(1), np.zeros(1))["b1p"][0]
        rho = np.hypot(yy[sel], xx[sel])
        err = max(err, np.abs(fld["b1p"] / c - analytic.homogeneous_b1p_profile(f, eps[0], sig[0], rho)).max(),
                  np.abs(np.abs(fld["b1m"] / c) - analytic.homogeneous_b1m_profile(f, eps[0], sig[0], rho)).max())
    rows.append(_row("A4", "analytic", "homogeneous brain-like cylinder (radius 9 cm) under the quadrature sheet, 64 / 128 / 298 MHz: B1+ and |B1-| "
                     "relative to B1+ on the axis", "J0(k rho) and |J2(k rho)|", err, 1e-12))

    # A5 degenerate layers
    err = 0.0
    modes = analytic.birdcage_modes(COIL_R, RUNGS, "quadrature", n_max=40)
    for f in FREQS:
        radii, eps, sig = bodies(f)["homogeneous"]
        one = analytic.LayeredCylinder(f, radii, eps, sig, COIL_R).fields(modes, yy[inside], xx[inside])
        two = analytic.LayeredCylinder(f, [0.04, 0.07, radii[0]], eps * 3, sig * 3, COIL_R).fields(modes, yy[inside], xx[inside])
        empty = analytic.LayeredCylinder(f, source_radius_m=COIL_R).fields(modes, yy[inside], xx[inside])
        vac = analytic.LayeredCylinder(f, [0.05, 0.09], [1.0, 1.0], [0.0, 0.0], COIL_R).fields(modes, yy[inside], xx[inside])
        for a, b in ((one, two), (empty, vac)):
            err = max(err, max(np.abs(a[k] - b[k]).max() / np.abs(a[k]).max() for k in ("ez", "b1p")))
    rows.append(_row("A5", "analytic", "three layers of one material; two layers of free space", "the homogeneous cylinder; the empty coil", err, 1e-10,
                     unit="relative"))

    # A6 power balance of the series solution
    err = 0.0
    for f in FREQS:
        for name in ("fat_muscle", "csf_brain"):
            radii, eps, sig = bodies(f)[name]
            for sh in (None, SHIELD_R):
                cyl = analytic.LayeredCylinder(f, radii, eps, sig, COIL_R, sh)
                for drive in ("quadrature", "linear"):
                    m = analytic.birdcage_modes(COIL_R, RUNGS, drive, n_max=40)
                    ps = cyl.source_power(m)
                    err = max(err, abs(ps - cyl.absorbed_power(m) - cyl.radiated_power(m)) / ps)
    rows.append(_row("A6", "analytic", "power balance of the series solution: two-layer cylinders, 16 rungs, quadrature and linear, open and shielded, "
                     "64 / 128 / 298 MHz", "delivered by the sources = absorbed (Gauss-Legendre quadrature of sigma |Ez|^2 / 2) + radiated (far field)",
                     err, 1e-9, unit="relative"))

    # A7 the fields obey Maxwell's equations: numerical derivatives of the series
    err_h, err_b = 0.0, 0.0
    d = 2.5e-4
    for f in FREQS:
        radii, eps, sig = bodies(f)["csf_brain"]
        cyl = analytic.LayeredCylinder(f, radii, eps, sig, COIL_R)
        m = analytic.birdcage_modes(COIL_R, RUNGS, "linear", n_max=40)
        rng = np.random.default_rng(7)
        for lo, hi, e_r, s_c in ((0.0, radii[0], eps[0], sig[0]), (radii[0], radii[1], eps[1], sig[1]), (radii[1], COIL_R, 1.0, 0.0)):
            r = rng.uniform(lo + 3 * d, hi - 3 * d, 40)
            p = rng.uniform(0, 2 * math.pi, 40)
            y0, x0 = r * np.sin(p), r * np.cos(p)
            off = np.arange(-2, 3) * d
            ez_y = np.stack([cyl.fields(m, y0 + o, x0)["ez"] for o in off])
            ez_x = np.stack([cyl.fields(m, y0, x0 + o)["ez"] for o in off])
            c2 = np.array([-1, 16, -30, 16, -1]) / (12 * d * d)            # fourth-order second derivative
            c1 = np.array([1, -8, 0, 8, -1]) / (12 * d)                   # fourth-order first derivative
            lap = c2 @ ez_y + c2 @ ez_x
            k2 = wavenumber(f, e_r, s_c) ** 2
            err_h = max(err_h, np.abs(lap + k2 * ez_y[2]).max() / np.abs(k2 * ez_y[2]).max())
            here = cyl.fields(m, y0, x0)
            w = 2 * math.pi * f
            bx, by = 1j / w * (c1 @ ez_y), -1j / w * (c1 @ ez_x)
            scale = np.abs(here["b1p"]).max()
            err_b = max(err_b, np.abs(here["bx"] - bx).max() / scale, np.abs(here["by"] - by).max() / scale,
                        np.abs(here["b1p"] - (bx + 1j * by) / 2).max() / scale, np.abs(here["b1m"] - np.conj(bx - 1j * by) / 2).max() / scale)
    rows.append(_row("A7", "analytic", "Helmholtz equation in each layer and in the gap (CSF / brain cylinder, linear drive, 120 random points per "
                     "frequency): laplace(Ez) + k^2 Ez", "zero; the Laplacian by fourth-order differences of the series on a 0.25 mm stencil", err_h, 1e-5,
                     unit="relative"))
    rows.append(_row("A7b", "analytic", "Bx, By, B1+ and B1- of the series at the same points", "Bx = (i / w) dEz/dy, By = -(i / w) dEz/dx by fourth-order "
                     "differences of Ez; B1+ = (Bx + i By) / 2, B1- = (Bx - i By)* / 2", err_b, 1e-6, unit="relative"))

    # A8 B1+ is the component that rotates with the magnetisation
    w = 2 * math.pi * 128e6
    radii, eps, sig = bodies(128e6)["fat_muscle"]
    cyl = analytic.LayeredCylinder(128e6, radii, eps, sig, COIL_R)
    fld = cyl.fields(analytic.birdcage_modes(COIL_R, RUNGS, "linear", n_max=40), np.array([0.03, -0.05]), np.array([0.04, 0.02]))
    t = np.arange(4096) / 4096 * 2 * math.pi / w
    err = 0.0
    for i in range(2):
        bxt, byt = np.real(fld["bx"][i] * np.exp(1j * w * t)), np.real(fld["by"][i] * np.exp(1j * w * t))
        plus = np.mean((bxt + 1j * byt) * np.exp(-1j * w * t))             # in the frame turning from +x towards +y
        minus = np.mean((bxt + 1j * byt) * np.exp(1j * w * t))             # in the frame turning the other way
        err = max(err, abs(plus - fld["b1p"][i]) / abs(fld["b1p"][i]), abs(minus - fld["b1m"][i]) / abs(fld["b1p"][i]))
    rows.append(_row("A8", "analytic", "the rotating components of a linearly driven coil at two off-centre points: the laboratory field "
                     "Re{B exp(i w t)} averaged over one period in the frame turning from +x towards +y, and in the frame turning the other way",
                     "B1+ = (Bx + i By) / 2 and B1- = (Bx - i By)* / 2 as returned (the definitions restated from Hoult 2000)", err, 1e-12,
                     unit="relative"))

    # A8b B1+ is the component that excites, with the phase convention of the design layer: the Bloch equation in the laboratory
    import scipy.integrate
    w0 = 2 * math.pi                                                       # precession: one turn per unit of time
    w1 = 5e-4 * w0                                                         # nutation rate gamma |B1+|
    flip = math.radians(40.0)
    t_end = flip / w1
    err = 0.0
    for bx, by in ((1.0 * np.exp(0.3j), 0.6 * np.exp(-1.1j)), (0.2 + 0j, 0.9j), (1.0 + 0j, 0j)):
        b1p, b1m = (bx + 1j * by) / 2, np.conj(bx - 1j * by) / 2
        for which, amp in (("plus", b1p), ("minus", b1m)):
            scale = w1 / abs(amp)                                          # gamma, so that the tested component nutates at w1
            if which == "minus":                                           # the counter-rotating part alone: mirror the field
                fx, fy = scale * np.conj(b1m), scale * 1j * np.conj(b1m)   # Bx + i By = 0, (Bx - i By)* / 2 = b1m
            else:
                fx, fy = scale * bx, scale * by

            def rhs(t, m, fx=complex(fx), fy=complex(fy)):                 # dM/dt = gamma B x M, B = (Re fx e^{iwt}, Re fy e^{iwt}, B0)
                c, sn = math.cos(w0 * t), math.sin(w0 * t)
                b0, b1 = fx.real * c - fx.imag * sn, fy.real * c - fy.imag * sn
                return [b1 * m[2] - w0 * m[1], w0 * m[0] - b0 * m[2], b0 * m[1] - b1 * m[0]]
            sol = scipy.integrate.solve_ivp(rhs, (0.0, t_end), [0.0, 0.0, 1.0], method="DOP853", rtol=1e-10, atol=1e-12)
            mx, my, mz = sol.y[:, -1]
            rot = (mx + 1j * my) * np.exp(-1j * w0 * t_end)                # transverse magnetisation in the rotating frame
            if which == "plus":
                err = max(err, abs(rot - (-1j) * np.exp(1j * np.angle(b1p)) * math.sin(flip)), abs(mz - math.cos(flip)))
            else:
                err = max(err, abs(rot), abs(mz - 1.0))                    # a field turning the other way does not excite
    rows.append(_row("A8b", "analytic", "the Bloch equation dM/dt = gamma B x M integrated in the laboratory frame (`scipy.integrate.solve_ivp`, "
                     "DOP853) for three elliptically polarised fields at the precession frequency, nutation 2,000 times slower than precession, "
                     "40 degrees: Mxy in the rotating frame and Mz; and for fields with B1- only", "the convention of design.md: Mxy = -i exp(i p) "
                     "sin(flip), Mz = cos(flip) with flip = gamma |B1+| T and p the phase of B1+; no excitation by B1- (both up to a ripple of the order "
                     "of the ratio of the two rates, 5e-4)", err, 1e-3))

    # A10 SAR and the square of the frequency
    radius, sig = 0.09, 0.6
    quasi = {}
    for f in (1e6, 2e6, 64e6, 128e6, 298e6):
        cyl = analytic.LayeredCylinder(f, [radius], [60.0], [sig], COIL_R)
        m = analytic.sheet_modes("quadrature")
        b0 = abs(cyl.fields(m, np.zeros(1), np.zeros(1))["b1p"][0])
        quasi[f] = cyl.absorbed_power(m) / b0 ** 2 / (math.pi * sig * (2 * math.pi * f) ** 2 * radius ** 4 / 4)
    rows.append(_row("A10", "analytic", "power absorbed per unit B1+^2 on the axis in a cylinder (radius 9 cm, permittivity 60, 0.6 S/m) at 1 and 2 MHz",
                     "quasi-static closed form pi sigma w^2 |B1+|^2 a^4 / 4 (the field E = w B1+ rho of a uniform rotating B1+): power for a fixed "
                     "B1+ grows as the square of the frequency", max(abs(quasi[1e6] - 1), abs(quasi[2e6] - 1)), 1e-3, unit="relative"))
    rows.append(_row("A10b", "analytic", "the same ratio to the quasi-static law at 64 MHz", "the law still holds within 15 %", abs(quasi[64e6] - 1), 0.15,
                     unit="relative"))
    rows.append(_row("A10c", "analytic", "the same ratio at 298 MHz", "the law fails: with the wavelength in the body the field on the axis is no longer "
                     "the field everywhere, and the power is below half of the law's", quasi[298e6], 0.5))
    # A11 wavelengths against published numbers
    err = max(abs(float(wavelength_m(f, 78.0)) / lam - 1) for f, lam in PUBLISHED_WAVELENGTH_WATER_M.items())
    rows.append(_row("A11", "analytic", "wavelength c / (f sqrt(eps)) in water (permittivity 78) at 94, 125 and 138 MHz", "published: 36, 27 and 24 cm "
                     "(Collins et al. 2005, J Magn Reson Imaging 21:192; two digits)", err, 0.03, unit="relative"))
    gm, wm = properties.cole_cole("Brain (Grey Matter)", 300e6), properties.cole_cole("Brain (White Matter)", 300e6)
    lam = float(wavelength_m(300e6, 0.5 * (gm[0] + wm[0]), 0.5 * (gm[1] + wm[1])))
    rows.append(_row("A11b", "analytic", "wavelength at 300 MHz in the mean of the grey and white matter rows", "published: \"approximately 13.4 cm\" in brain "
                     "tissue (Collins and Wang 2011, Magn Reson Med; PubMed Central PMC3078983)", abs(lam / 0.134 - 1), 0.03, unit="relative"))
    rows.append(_row("A11c", "analytic", "half a wavelength at 64 MHz in a medium of permittivity 100", "published: \"about 25 cm\" (Glover et al. 1985, "
                     "J Magn Reson 64:255)", abs(float(wavelength_m(64e6, 100.0)) / 2 / 0.25 - 1), 0.1, unit="relative"))
    return rows


PUBLISHED_WAVELENGTH_WATER_M = {94e6: 0.36, 125e6: 0.27, 138e6: 0.24}


# ---------------------------------------------------------------------------------------------- solver
def check_solver():
    rows = []
    h = 0.002

    # S1, S2 the absorbing layer: a line current in free space
    def free_space(f, n_pml):
        n, n_pml, yy, xx = _grid(h, None, n_pml)
        sol = solver.Solver(f, np.ones((n, n)), np.zeros((n, n)), h, n_pml=n_pml)
        y0, x0 = COIL_R * math.sin(0.7), COIL_R * math.cos(0.7)
        ez = sol.solve([[(y0, x0, 1.0)]])[0]
        ref = analytic.free_space_line(f, yy, xx, y0, x0)
        sel = np.hypot(yy, xx) < 0.1
        return np.abs(ez - ref)[sel].max() / np.abs(ref)[sel].max()
    rows.append(_row("S1", "solver", "absorbing layer of 16 cells (2 mm grid): Ez of a line current in free space, inside 10 cm, at 64 / 128 / 298 MHz",
                     "-(w mu0 I / 4) H0^(2)(k0 |r - r'|)", max(free_space(f, 16) for f in FREQS), 3e-4, unit="relative"))
    rows.append(_row("S1b", "solver", "a layer of 24 cells at 10 and 20 MHz, the range in which the solver asks for the thicker layer", "the same",
                     max(free_space(f, 24) for f in (10e6, 20e6)), 3e-4, unit="relative"))
    rows.append(_row("S2", "solver", "a layer of 8 cells at 64 / 128 / 298 MHz: the error a thinner layer leaves", "the same",
                     max(free_space(f, 8) for f in FREQS), 1e-2, unit="relative"))

    # S3 cylinders at 2 mm, open coil
    worst = dict(ez=0.0, b1p_rms=0.0, b1p_away=0.0, b1p_max=0.0, b1m_rms=0.0, p_abs=0.0, p_src=0.0)
    for f in FREQS:
        for body in bodies(f).values():
            res = cylinder_case(f, body, h, stride=2)
            worst = {k: max(v, res[k]) for k, v in worst.items()}
    what = "homogeneous, fat / muscle and CSF / brain cylinders in an open 16-rung birdcage, 2 mm grid, 64 / 128 / 298 MHz"
    rows.append(_row("S3", "solver", f"{what}: Ez inside the cylinder", "the Bessel series for the same 16 line currents", worst["ez"], 4e-3,
                     unit="of the largest |Ez|"))
    rows.append(_row("S3b", "solver", "the same cases: B1+ and B1-, rms error over the cylinder", "the series", max(worst["b1p_rms"], worst["b1m_rms"]), 1e-2,
                     unit="of the rms"))
    rows.append(_row("S3c", "solver", "the same cases: B1+ more than 1.5 pixels away from a layer boundary", "the series", worst["b1p_away"], 2.5e-3,
                     unit="of the largest |B1+|"))
    rows.append(_row("S3d", "solver", "the same cases: B1+ in the pixels next to a layer boundary, where the curvature of Ez jumps and the centred "
                     "difference is first order", "the series", worst["b1p_max"], 3e-2, unit="of the largest |B1+|"))
    rows.append(_row("S3e", "solver", "the same cases: absorbed power and power delivered by the rungs", "the series", max(worst["p_abs"], worst["p_src"]),
                     3e-3, unit="relative"))

    # S4 convergence with the grid at 298 MHz
    f = 298e6
    orders = dict(ez=[], b1p_away=[], b1p_max=[])
    finest = 0.0
    for name in ("homogeneous", "fat_muscle", "csf_brain"):
        errs = [cylinder_case(f, bodies(f)[name], hh, SHIELD_R, stride=s) for hh, s in ((0.004, 1), (0.002, 1), (0.001, 2))]
        for k in orders:
            orders[k].append(_order([e[k] for e in errs]))
        finest = max(finest, errs[-1]["ez"])
    rows.append(_row("S4", "solver", "order of convergence of Ez from 4 to 2 to 1 mm at 298 MHz (the three cylinders in the shielded coil): the "
                     "smallest of the three", "second order", min(orders["ez"]), 1.7, kind="min"))
    rows.append(_row("S4b", "solver", "the same: B1+ away from layer boundaries", "second order", min(orders["b1p_away"]), 1.7, kind="min"))
    rows.append(_row("S4c", "solver", "the same: B1+ in the pixels next to a layer boundary, distance of the order from 1", "first order",
                     max(abs(o - 1) for o in orders["b1p_max"]), 0.35))
    rows.append(_row("S4d", "solver", "Ez error on the 1 mm grid in those cases", "the series", finest, 1e-3, unit="of the largest |Ez|"))
    errs = [cylinder_case(f, bodies(f)["fat_muscle"], hh)["ez"] for hh in (0.004, 0.002)]
    rows.append(_row("S4e", "solver", "the open coil with its absorbing layer, fat / muscle cylinder, from 4 to 2 mm: order of convergence of Ez",
                     "second order", _order(errs), 1.7, kind="min"))

    # S5 shield and the quasi-static limit
    worst = 0.0
    for f in FREQS:
        res = cylinder_case(f, bodies(f)["homogeneous"], h, SHIELD_R, stride=2)
        worst = max(worst, res["ez"], res["b1p_away"], res["p_abs"])
    rows.append(_row("S5", "solver", "homogeneous cylinder in a shielded birdcage (shield radius 19 cm), 2 mm grid, 64 / 128 / 298 MHz: Ez, B1+ away "
                     "from the boundary, absorbed power", "the series with Ez = 0 on the shield", worst, 4e-3, unit="relative"))
    n, _, yy, xx = _grid(h, SHIELD_R)
    sol = solver.Solver(1e5, np.ones((n, n)), np.zeros((n, n)), h, n_pml=0, shield_radius_m=SHIELD_R)
    b = sol.fields(coils.birdcage(COIL_R, RUNGS))
    ref = analytic.quasi_static_b1(n_rungs=RUNGS, source_radius_m=COIL_R, shield_radius_m=SHIELD_R)
    sel = np.hypot(yy, xx) < 0.04
    rows.append(_row("S6", "solver", "quasi-static limit: empty shielded birdcage of 16 rungs at 100 kHz, 2 mm grid: |B1+| within 4 cm of the axis, "
                     "and |B1-| there", "magnetostatics: uniform, mu0 N I / (4 pi R) (1 - (R / S)^2); no B1-",
                     max(np.abs(np.abs(b["b1p"][0][sel]) - ref).max(), np.abs(b["b1m"][0][sel]).max()) / ref, 1e-4, unit="relative"))
    errs = []
    for hh in (0.004, 0.002, 0.001):
        n, _, yy, xx = _grid(hh, SHIELD_R)
        sol = solver.Solver(128e6, np.ones((n, n)), np.zeros((n, n)), hh, n_pml=0, shield_radius_m=SHIELD_R)
        b = sol.fields(coils.birdcage(COIL_R, RUNGS))["b1p"][0]
        sel = np.hypot(yy, xx) < 0.09
        cyl = analytic.LayeredCylinder(128e6, source_radius_m=COIL_R, shield_radius_m=SHIELD_R)
        want = cyl.fields(analytic.birdcage_modes(COIL_R, RUNGS, n_max=40), yy[sel], xx[sel])["b1p"]
        errs.append(np.abs(b[sel] - want).max() / np.abs(want).max())
    rows.append(_row("S6b", "solver", "the shield, a circle on a square grid: B1+ of the empty shielded birdcage at 128 MHz within 9 cm, from 4 to 2 "
                     "to 1 mm: order of convergence", "second order (the series with Ez = 0 on the shield)", _order(errs), 1.7, kind="min"))
    rows.append(_row("S6c", "solver", "the same: error on the 2 mm grid", "the series", errs[1], 5e-4, unit="relative"))

    # S7 power balance
    f = 298e6
    res = cylinder_case(f, bodies(f)["fat_muscle"], h, stride=4)
    sol, ez, cyl, modes = res["solver"], res["fields"]["ez"][0], res["cyl"], res["modes"]
    p_src = sol.source_power(res["fields"]["ez"], res["channels"])
    p_abs = sol.absorbed_power(ez, sol.box(0.16))
    rows.append(_row("S7", "solver", "power balance at 298 MHz (fat / muscle cylinder, open coil): delivered by the rungs - absorbed - flux through "
                     "the faces of a box of 32 cm", "zero: the discrete form of Poynting's theorem holds for the difference equations",
                     abs(p_src - p_abs - sol.outward_flux(ez, 0.16)) / p_src, 1e-9, unit="of the delivered power"))
    rows.append(_row("S7b", "solver", "the same flux through a box of 30 cm", "the same flux: nothing is absorbed or delivered between the two boxes",
                     abs(sol.outward_flux(ez, 0.15) / sol.outward_flux(ez, 0.16) - 1), 1e-9, unit="relative"))
    rows.append(_row("S7c", "solver", "radiated power: the flux through the box", "the series: far-field sum over the azimuthal orders", abs(sol.outward_flux(ez, 0.16) / cyl.radiated_power(modes) - 1), 1e-3, unit="relative"))
    res_s = cylinder_case(f, bodies(f)["fat_muscle"], h, SHIELD_R, stride=4)
    sol_s = res_s["solver"]
    rows.append(_row("S7d", "solver", "the same body in the shielded coil: delivered - absorbed", "zero: nothing leaves a shield",
                     abs(sol_s.source_power(res_s["fields"]["ez"], res_s["channels"]) / sol_s.absorbed_power(res_s["fields"]["ez"][0]) - 1), 1e-9,
                     unit="relative"))

    # S8 reciprocity on a heterogeneous body
    wm = brain_model(298e6)
    sol = wm.solver
    a = sol._matrix()
    asym = abs(a - a.T).max() / abs(a).max()
    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.12, 0.12, (12, 2))
    ez = sol.solve([[(float(p[0]), float(p[1]), 1.0)] for p in pts])
    g = np.array([[sol.sample(ez[i], float(pts[j][0]), float(pts[j][1])) for j in range(12)] for i in range(12)])
    rows.append(_row("S8", "solver", "reciprocity in the brain world at 298 MHz: Ez at point j from a line current at point i, 12 random points inside "
                     "and outside the head, and the asymmetry of the matrix", "Ez at i from the same current at j; a symmetric matrix",
                     max(np.abs(g - g.T).max() / np.abs(g).max(), asym), 1e-10, unit="relative"))

    # S9 quadrature against linear drive in the solver
    f = 128e6
    n, n_pml, yy, xx = _grid(h)
    cyl = analytic.LayeredCylinder(f, *bodies(f)["fat_muscle"], COIL_R)
    e_map, s_map = cyl.cell_maps(yy, xx, h)
    sol = solver.Solver(f, e_map, s_map, h, n_pml=n_pml)
    fq = sol.fields(coils.birdcage(COIL_R, RUNGS, "quadrature"))
    fl = sol.fields(coils.birdcage(COIL_R, RUNGS, "linear", current_a=2.0))
    ratio_b = abs(sol.sample(fl["b1p"][0], 0, 0) / sol.sample(fq["b1p"][0], 0, 0))
    ratio_p = sol.absorbed_power(fl["ez"][0]) / sol.absorbed_power(fq["ez"][0])
    rows.append(_row("S9", "solver", "linear drive with twice the rung current of the quadrature drive (fat / muscle cylinder, 128 MHz, 2 mm grid): B1+ "
                     "on the axis and absorbed power", "the same B1+ for twice the power: for a body with circular symmetry the orders +1 and -1 do not mix, at any frequency (Glover et al. "
                     "1985 report up to 50 % less power with circular polarisation)", max(abs(ratio_b - 1), abs(ratio_p - 2) / 2), 1e-5, unit="relative"))

    # S10 central brightening
    prof, cv = 0.0, {}
    for f in FREQS:
        radii, eps, sig = bodies(f)["homogeneous"]
        n, n_pml, yy, xx = _grid(h)
        cyl = analytic.LayeredCylinder(f, radii, eps, sig, COIL_R)
        e_map, s_map = cyl.cell_maps(yy, xx, h)
        sol = solver.Solver(f, e_map, s_map, h, n_pml=n_pml)
        b = np.abs(sol.fields(coils.birdcage(COIL_R, RUNGS))["b1p"][0])
        rho = np.hypot(yy, xx)
        sel = rho < radii[0] - 1.5 * h
        bn = b[sel] / sol.sample(b, 0, 0)
        prof = max(prof, np.abs(bn - np.abs(analytic.homogeneous_b1p_profile(f, eps[0], sig[0], rho[sel]))).max())
        cv[f] = float(bn.std() / bn.mean())
    rows.append(_row("S10", "solver", "central brightening: |B1+| relative to the axis over a head-sized brain-like cylinder (radius 9 cm) in the "
                     "16-rung birdcage, 2 mm grid, 64 / 128 / 298 MHz", "|J0(k rho)|: the pattern of the ideal quadrature sheet", prof, 5e-3))
    rows.append(_row("S10b", "solver", "its size at 298 MHz: coefficient of variation of |B1+| over the cylinder, as a factor from the published value "
                     "(the larger of ratio and inverse ratio)", f"published: relative standard deviation {PUBLISHED_CV_300MHZ:g} over an axial slice of a "
                     "three-dimensional head model at 300 MHz, 16 elements driven as a birdcage (Mao et al. 2006, Magn Reson Med 56:918): the same "
                     "order of size, within a factor 2.5", max(cv[298e6] / PUBLISHED_CV_300MHZ, PUBLISHED_CV_300MHZ / cv[298e6]), 2.5))
    rows.append(_row("S10c", "solver", "coefficient of variation at 64 MHz over that at 298 MHz", "near absence at 1.5 T: below 0.4", cv[64e6] / cv[298e6], 0.4))

    # S11 wavelength: the first null of B1+ in a lossless water cylinder
    f, eps_w = 298e6, 78.0
    n, _, yy, xx = _grid(0.001, SHIELD_R)
    cyl = analytic.LayeredCylinder(f, [0.09], [eps_w], [0.0], COIL_R, SHIELD_R)
    e_map, s_map = cyl.cell_maps(yy, xx, 0.001)
    sol = solver.Solver(f, e_map, s_map, 0.001, n_pml=0, shield_radius_m=SHIELD_R)
    b = sol.fields(coils.birdcage(COIL_R, RUNGS))["b1p"][0]
    c = n // 2
    line = np.real(b[c, c:c + 80] / b[c, c])                               # along +x from the axis, 1 mm steps
    i = int(np.flatnonzero(line < 0)[0])
    null = (i - 1 + line[i - 1] / (line[i - 1] - line[i])) * 0.001
    lam = float(wavelength_m(f, eps_w, 0.0))
    rows.append(_row("S11", "solver", "wavelength in a lossless water cylinder (permittivity 78) at 298 MHz, 1 mm grid, shielded coil: radius of the "
                     "first null of B1+, as a wavelength 2 pi x radius / 2.4048", "lambda = c / (f sqrt(eps)) = 113.9 mm (B1+ is J0(k rho), first zero "
                     "at k rho = 2.4048)", abs(2 * math.pi * null / special.jn_zeros(0, 1)[0] / lam - 1), 2e-3, unit="relative"))

    # S12 SAR and the square of the frequency, in the solver
    p = {}
    for f in (1e6, 2e6):
        n, _, yy, xx = _grid(h, SHIELD_R)
        cyl = analytic.LayeredCylinder(f, [0.09], [60.0], [0.6], COIL_R, SHIELD_R)
        e_map, s_map = cyl.cell_maps(yy, xx, h)
        sol = solver.Solver(f, e_map, s_map, h, n_pml=0, shield_radius_m=SHIELD_R)
        fld = sol.fields(coils.birdcage(COIL_R, RUNGS))
        p[f] = sol.absorbed_power(fld["ez"][0]) / abs(sol.sample(fld["b1p"][0], 0, 0)) ** 2
    closed = math.pi * 0.6 * (2 * math.pi * 1e6) ** 2 * 0.09 ** 4 / 4
    rows.append(_row("S12", "solver", "absorbed power per unit B1+^2 on the axis at 1 and 2 MHz (shielded coil, 2 mm grid): the ratio of the two, and "
                     "the value at 1 MHz", "4 (the square of the frequency); pi sigma w^2 a^4 / 4", max(abs(p[2e6] / p[1e6] / 4 - 1), abs(p[1e6] / closed - 1)),
                     2e-3, unit="relative"))

    # S13 SAR maps
    wm = brain_model(128e6)
    sol = wm.solver
    fld = sol.fields(coils.birdcage(COIL_R, RUNGS))
    ez = fld["ez"][0]
    avg = sol.averaged_sar(ez, wm.density)
    kernel = sol.averaging_kernel()
    w2, m = np.outer(kernel, kernel), len(kernel) // 2
    pd = 0.5 * sol.sigma * np.abs(ez) ** 2
    rng = np.random.default_rng(5)
    body = np.argwhere(wm.masks["body"])
    err = 0.0
    for i, j in body[rng.choice(len(body), 200, replace=False)]:
        sl = (slice(i - m, i + m + 1), slice(j - m, j + m + 1))
        mass = np.sum(w2 * wm.density[sl])
        want = np.sum(w2 * pd[sl]) / mass if mass / w2.sum() >= 500.0 else 0.0
        err = max(err, abs(avg[i, j] - want) / max(avg.max(), 1e-30))
    point = sol.point_sar(ez, wm.density)
    tissue = wm.density > 0
    err = max(err, abs(np.sum(point * wm.density) * sol.h ** 2 / sol.absorbed_power(ez) - 1), float(np.abs(point[~tissue]).max(initial=0)))
    rows.append(_row("S13", "solver", f"SAR in the brain world at 128 MHz: the average over the square of {kernel.sum() * sol.h * 1e3:.1f} mm "
                     f"({len(kernel)} pixels across, the two at the ends counted with {kernel[0]:.3f}) at 200 random pixels of the head; the point "
                     "SAR integrated over the mass", "a loop over the pixels of each square: weighted power in the square / weighted mass in the "
                     "square; the absorbed power", err, 1e-12, unit="relative"))
    uni = solver.Solver(64e6, np.full((61, 61), 50.0), np.full((61, 61), 0.5), 0.002, n_pml=0)
    flat = uni.averaged_sar(np.ones((61, 61)), np.full((61, 61), 1000.0))
    rows.append(_row("S13b", "solver", "the averaged SAR of a uniform field in a uniform medium, away from the edge of the grid",
                     "sigma |E|^2 / (2 rho)", np.abs(flat[10:-10, 10:-10] - 0.5 * 0.5 / 1000.0).max() / (0.25 / 1000.0), 1e-12, unit="relative"))
    sides = []
    for factor in (4, 5, 8):
        wm_f = brain_model(128e6, factor)
        e_f = wm_f.solver.fields(coils.birdcage(COIL_R, RUNGS))["ez"][0]
        sides.append(wm_f.solver.averaged_sar(e_f, wm_f.density).max() / (wm_f.solver.absorbed_power(e_f) / (wm_f.density.sum() * wm_f.solver.h ** 2)))
    rows.append(_row("S13c", "solver", "peak averaged SAR over mean SAR of the brain world at 128 MHz on grids of 1.6, 2 and 3.2 mm: largest "
                     "difference from the 1.6 mm value", "the definition does not depend on the pixel: within 3 %",
                     max(abs(v / sides[0] - 1) for v in sides), 0.03, unit="relative"))

    # S14 time
    t = timing()
    rows.append(_row("S14", "solver", f"one factorisation and 8 channel solves at 298 MHz on the brain world ({t['unknowns']:,} unknowns, "
                     f"{t['h_mm']:g} mm; best of 3, one thread)", "requirement", t["seconds"], 2.0, unit="s"))
    return rows


# Published size of the non-uniformity at 7 T used by row S10b (fetched and read): Mao, Smith and Collins 2006, relative
# standard deviation of the flip angle over an axial slice of the head at 300 MHz before RF shimming.
PUBLISHED_CV_300MHZ = 0.138
# Peak 10 g SAR over head-average SAR in a three-dimensional head model in a volume coil (fetched and read):
# Wang et al. 2007, at 64, 128 and 300 MHz.
PUBLISHED_10G_OVER_MEAN = {64: 3.4, 128: 3.2, 300: 4.1}


def timing(repeat=3):
    """Time of one factorisation and 8 channel solves on the brain world at 298 MHz (best of ``repeat``)."""
    from . import properties as props
    w = brain_world()
    wm = brain_model(298e6)
    sol = wm.solver
    channels = coils.array(COIL_R, 8, "loop")
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        s2 = solver.Solver(298e6, sol.eps_r, sol.sigma, sol.h, sol.origin, n_pml=sol.n_pml)
        s2.fields(channels)
        best = min(best, time.perf_counter() - t0)
    t0 = time.perf_counter()
    props.world_maps(w, 298e6)
    return dict(seconds=best, unknowns=sol.n ** 2, n=sol.n, h_mm=sol.h * 1e3, maps_seconds=time.perf_counter() - t0,
                master_n=int(w["n"]), master_mm=float(w["extent_mm"]) / int(w["n"]))


# ---------------------------------------------------------------------------------------------- coils and reception
def check_coils():
    rows = []
    f = 128e6
    wm = brain_model(f)
    sol = wm.solver

    # C1 the birdcage as two ports
    ports = sol.fields(coils.birdcage_ports(COIL_R, RUNGS))
    quad = sol.fields(coils.birdcage(COIL_R, RUNGS, "quadrature"))
    rev = sol.fields(coils.birdcage(COIL_R, RUNGS, "reverse"))
    comb_q, comb_r = solver.combine(ports, [1, -1j]), solver.combine(ports, [1, 1j])
    err = max(np.abs(comb_q[k] - quad[k][0]).max() / np.abs(quad[k][0]).max() for k in ("ez", "b1p", "b1m"))
    err = max(err, max(np.abs(comb_r[k] - rev[k][0]).max() / np.abs(rev[k][0]).max() for k in ("ez", "b1p", "b1m")))
    rows.append(_row("C1", "coils", "the two linear ports of a birdcage combined with weights (1, -i) and (1, +i) in the brain world at 128 MHz: Ez, B1+, B1-",
                     "the quadrature and the reverse mode solved directly (B1- of a combination is the conjugate of the combined conj(B1-))", err, 1e-10,
                     unit="relative"))
    organ = wm.masks["organ"]
    b_q, b_r = np.abs(quad["b1p"][0][organ]).mean(), np.abs(rev["b1p"][0][organ]).mean()
    rows.append(_row("C1b", "coils", "mean |B1+| over the brain: reverse mode over quadrature mode", "the reverse mode turns against the magnetisation: "
                     "below 0.5 at 3 T", b_r / b_q, 0.5))

    # C2 mode weights of an array
    ratio = 0.0
    n, _, yy, xx = _grid(0.004, SHIELD_R)
    empty = solver.Solver(64e6, np.ones((n, n)), np.zeros((n, n)), 0.004, n_pml=0, shield_radius_m=SHIELD_R)
    for element in ("line", "loop"):
        arr = empty.fields(coils.array(COIL_R, 8, element))
        cp = solver.combine(arr, coils.mode_weights(8))
        anti = solver.combine(arr, coils.mode_weights(8, sense=-1))
        ratio = max(ratio, abs(empty.sample(anti["b1p"], 0, 0)) / abs(empty.sample(cp["b1p"], 0, 0)),
                    abs(empty.sample(cp["b1m"], 0, 0)) / abs(empty.sample(cp["b1p"], 0, 0)))
    rows.append(_row("C2", "coils", "8 line elements and 8 loop elements in an empty shielded coil at 64 MHz, with the weights of the birdcage-like "
                     "mode and with the opposite sense: B1+ at the isocentre of the opposite sense, and B1- of the birdcage-like mode, over B1+ of "
                     "the birdcage-like mode", "zero: on the axis of an empty coil each sense makes one rotating component only", ratio, 1e-9))

    # C3 loss matrix and absorbed power
    arr = sol.fields(coils.array(COIL_R, 8, "loop"))
    r = sol.power_matrix(arr["ez"])
    rng = np.random.default_rng(3)
    err = 0.0
    for _ in range(5):
        w = rng.normal(size=8) + 1j * rng.normal(size=8)
        err = max(err, abs(shim.power(r, w) / sol.absorbed_power(solver.combine(arr, w)["ez"]) - 1))
    psi = coils.noise_resistance(sol, arr["ez"])
    rows.append(_row("C3", "coils", "8 loops in the brain world at 128 MHz: Re(w^H R w) / 2 for 5 random weight vectors",
                     "the power absorbed by the field of the combined currents", err, 1e-10, unit="relative"))

    # C4 the best combination does not depend on how the channels are mixed
    rx = coils.receive_maps(arr)
    t = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    mixed_channels = [[(y, x, c * t[i, j]) for j, ch in enumerate(arr["channels"]) for (y, x, c) in ch] for i in range(8)]
    mix = sol.fields(mixed_channels)
    snr_a = coils.snr_optimal(rx[:, organ], psi)
    snr_b = coils.snr_optimal(coils.receive_maps(mix)[:, organ], coils.noise_resistance(sol, mix["ez"]))
    rows.append(_row("C4", "coils", "signal-to-noise ratio of the best combination over the brain, for the 8 loops and for 8 channels that are random "
                     "complex mixtures of them, solved again from their own currents", "equal: the optimum does not depend on the choice of channels "
                     "(this fails if the noise matrix is conjugated)", np.abs(snr_b / snr_a - 1).max(), 1e-8, unit="relative"))
    wrong = coils.snr_optimal(coils.receive_maps(mix)[:, organ], coils.noise_resistance(sol, mix["ez"]).T)
    rows.append(_row("C4b", "coils", "the same with the transposed (conjugated) noise matrix of the mixed channels", "not equal: a change above 1 % shows "
                     "that row C4 can tell the two apart", np.abs(wrong / snr_a - 1).max(), 0.01, kind="min", unit="relative"))

    # C5 a fixed combination against the optimum
    worst, best = 0.0, 0.0
    idx = np.flatnonzero(organ.ravel())[::97][:40]
    s_all = rx.reshape(8, -1)[:, idx]
    for k in range(s_all.shape[1]):
        s = s_all[:, k]
        opt = float(coils.snr_optimal(s[:, None], psi)[0])
        matched = np.linalg.solve(psi, s)
        best = max(best, abs(float(coils.snr_combined(s[:, None], psi, matched)[0]) / opt - 1))
        for _ in range(5):
            w = rng.normal(size=8) + 1j * rng.normal(size=8)
            worst = max(worst, float(coils.snr_combined(s[:, None], psi, w)[0]) / opt - 1)
    rows.append(_row("C5", "coils", "40 pixels of the brain: the combination with weights psi^-1 s against the optimum; 5 random combinations each, "
                     "excess over the optimum", "equal; never above (Roemer et al. 1990)", max(best, worst), 1e-10, unit="relative"))

    # C6 g-factor
    err = 0.0
    for _ in range(50):
        s = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        cosang = abs(np.vdot(s[:, 0], s[:, 1])) / (np.linalg.norm(s[:, 0]) * np.linalg.norm(s[:, 1]))
        err = max(err, np.abs(coils.g_factor(s, np.eye(2)) * math.sqrt(1 - cosang ** 2) - 1).max())
        s3 = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
        p = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
        p = p @ p.conj().T + np.eye(6)
        white = np.linalg.inv(np.linalg.cholesky(p))                       # whitened channels have unit noise
        err = max(err, np.abs(coils.g_factor(s3, p) / coils.g_factor(white @ s3, np.eye(6)) - 1).max())
        err = max(err, abs(float(coils.g_factor(s3[:, :1], p)[0]) - 1))
    rows.append(_row("C6", "coils", "g-factor: two channels and two folded positions with unit noise (50 random cases); six channels with a random "
                     "noise matrix; one position", "1 / sin(angle between the two sensitivity vectors); the same after whitening the channels; 1",
                     err, 1e-9, unit="relative"))
    lo = sol.n_pml
    rows_used = (sol.n - 2 * lo) // 6 * 6                                    # a number of rows that both 2 and 3 divide
    rx_c, organ_c = rx[:, lo:lo + rows_used], organ[lo:lo + rows_used]
    psi_inv = np.linalg.inv(psi)
    err, low = 0.0, np.inf
    for accel in (2, 3):
        g = coils.g_map(rx_c, psi, accel, organ_c)
        step = rows_used // accel
        for i in range(0, step, 3):
            fold = i + step * np.arange(accel)
            for j in np.flatnonzero(organ_c[fold].any(axis=0))[::5]:
                sel = fold[organ_c[fold, j]]
                sm = rx_c[:, sel, j]
                unfold = np.linalg.inv(sm.conj().T @ psi_inv @ sm) @ sm.conj().T @ psi_inv        # the unfolding matrix
                var_unfolded = np.real(np.diag(unfold @ psi @ unfold.conj().T))                    # noise of each unfolded pixel
                var_full = 1.0 / np.real(np.einsum("ik,ij,jk->k", sm.conj(), psi_inv, sm))        # noise of the best combination without folding
                explicit = np.sqrt(var_unfolded / var_full)
                err = max(err, np.abs(g[sel, j] / explicit - 1).max())
                low = min(low, explicit.min())
    rows.append(_row("C6b", "coils", "g-factor map of the 8 loops over the brain for twofold and threefold undersampling along y",
                     "explicit unfolding: noise of each unfolded pixel through the unfolding matrix, over the noise of the best combination of "
                     "the fully sampled pixel", err, 1e-8, unit="relative"))
    rows.append(_row("C6c", "coils", "smallest g-factor of that explicit computation, minus 1", "unfolding never lowers the noise: not below zero",
                     low - 1.0, -1e-10, kind="min"))
    return rows


# ---------------------------------------------------------------------------------------------- shimming
def shim_setup(f_hz=298e6, n_channels=8):
    """The shimming problem of the brain world: 8 line elements in a shielded coil (the shield carries their return
    currents). dict(a, r, q_point, q_avg, support, cp, target, model, fields); the SAR matrices cover every pixel at
    which the SAR map has a value (``Solver.sar_support``)."""
    key = ("shim", f_hz, n_channels)
    if key not in _world_cache:
        wm = brain_model(f_hz, shield=SHIELD_R)
        sol = wm.solver
        fld = sol.fields(coils.array(COIL_R, n_channels, "line"))
        a = shim.system_matrix(fld["b1p"], wm.masks["organ"])
        support = sol.sar_support(wm.density, solver.TEN_GRAM_SIDE_M)
        _world_cache[key] = dict(a=a, r=sol.power_matrix(fld["ez"]), q_point=sol.sar_matrices(fld["ez"], wm.density, wm.masks["tissue"]),
                                 q_avg=sol.sar_matrices(fld["ez"], wm.density, support, solver.TEN_GRAM_SIDE_M), support=support,
                                 cp=coils.mode_weights(n_channels), target=1e-6, model=wm, fields=fld)
    return _world_cache[key]


def check_shim():
    rows = []
    su = shim_setup()
    a, r, q, cp, t = su["a"], su["r"], su["q_avg"], su["cp"], su["target"]
    wm, fld = su["model"], su["fields"]
    sol = wm.solver
    rng = np.random.default_rng(9)

    # H1 the quadratic forms are the maps
    err = 0.0
    tissue = wm.masks["tissue"]
    for _ in range(3):
        w = rng.normal(size=8) + 1j * rng.normal(size=8)
        ez = solver.combine(fld, w)["ez"]
        point, avg = sol.point_sar(ez, wm.density), sol.averaged_sar(ez, wm.density)
        support = su["support"]
        err = max(err, np.abs(shim.quad(su["q_point"], w) - point[tissue]).max() / point.max(), np.abs(shim.quad(q, w) - avg[support]).max() / avg.max(),
                  abs(shim.power(r, w) / sol.absorbed_power(ez) - 1), float(np.abs(point[~tissue]).max(initial=0.0) + np.abs(avg[~support]).max(initial=0.0)))
    rows.append(_row("H1", "shim", "brain world at 298 MHz, 8 line elements in a shielded coil, 3 random weight vectors: Re(w^H Q_p w) for point "
                     "SAR at every pixel that holds tissue and for the averaged SAR at every pixel where it has a value; Re(w^H R w) / 2; the "
                     "two maps outside those pixels", "the SAR maps and the absorbed power of the combined field; zero outside", err, 1e-10,
                     unit="relative"))

    # H2 gradients
    w = rng.normal(size=8) + 1j * rng.normal(size=8)
    w = shim.scaled_reference(a, w, t)
    _, g = shim._f_grad(a, w, t)
    num = np.zeros(8, complex)
    step = 1e-6 * np.abs(w).max()
    for i in range(8):
        for d in (step, 1j * step):
            wp, wn = w.copy(), w.copy()
            wp[i] += d
            wn[i] -= d
            slope = (shim.objective(a, wp, t) - shim.objective(a, wn, t)) / (2 * step)
            num[i] += slope if d == step else 1j * slope
    rows.append(_row("H2", "shim", "gradient of the objective with respect to the real and imaginary parts of the weights", "centred differences",
                     np.abs(num - 2 * g).max() / np.abs(g).max(), 1e-6, unit="relative"))

    # H3 never worse than the birdcage-like mode; limits kept
    ref = shim.scaled_reference(a, cp, t)
    p_ref, s_ref = shim.power(r, ref), shim.peak_sar(q, ref)[0]
    cases = [("full", {}), ("phase", {}), ("full", dict(r=r, p_max=0.7 * p_ref)), ("phase", dict(r=r, p_max=0.7 * p_ref)),
             ("full", dict(q=q, sar_max=0.6 * s_ref)), ("phase", dict(q=q, sar_max=0.6 * s_ref)),
             ("full", dict(r=r, p_max=0.8 * p_ref, q=q, sar_max=0.5 * s_ref)), ("full", dict(q=q, sar_max=0.25 * s_ref))]
    worse, over, fallbacks, kkt, results = -np.inf, 0.0, 0, 0.0, []
    for kind, lim in cases:
        res = shim.solve(a, t, cp, kind=kind, **lim)
        start = shim.objective(a, shim.scaled_reference(a, cp, t, lim.get("r"), lim.get("p_max"), lim.get("q"), lim.get("sar_max")), t)
        worse = max(worse, res["raw_f"] - start)
        fallbacks += res["fallback"]
        over = max(over, res["raw_excess"])
        kkt = max(kkt, res["kkt"])
        results.append(res)
    rows.append(_row("H3", "shim", "8 problems (magnitude-and-phase and phase-only; no limit, a power limit, a limit on the averaged local SAR, both): "
                     "objective of what the optimiser delivers, before the final scaling and the fallback, minus the objective of the "
                     "birdcage-like mode at its best feasible scale; the largest of the eight", "below zero: shimming is at least as good as the "
                     "mode it starts from, on its own objective (the returned result cannot be worse by construction: it falls back on the start)",
                     worse, 0.0))
    rows.append(_row("H3b", "shim", "the same problems: how often the result had to fall back on the start", "never", fallbacks, 0))
    rows.append(_row("H3c", "shim", "the same problems: largest excess over a limit (power; averaged local SAR at every pixel where it has a value) in what the "
                     "optimiser delivers, before the final scaling removes it", "none beyond the tolerance of the optimiser", over, 1e-6, unit="relative"))
    rows.append(_row("H3d", "shim", "the same problems: first-order optimality at the result (what is left of the gradient of the objective after the "
                     "active limits, for a step of relative size 1; the objective itself is 0.01 to 0.2 there)", "zero at a local optimum", kkt, 1e-5))
    cv0 = shim.uniformity(a, ref)["cv"]
    multi = shim.solve(a, t, cp, starts=shim.random_starts(cp, 8))
    rows.append(_row("H3e", "shim", "coefficient of variation of |B1+| over the brain at 298 MHz without limits: magnitude-and-phase shim (best of "
                     "9 starts) over birdcage-like mode", "smaller: below 0.7", multi["cv"] / cv0, 0.7))

    # H4 a small problem with a known optimum: two channels, phase only, a dense scan
    a2 = a[:, [0, 4]]
    start = np.array([1.0, 1.0], complex)
    res = shim.solve(a2, t, start, kind="phase")
    best = np.inf
    for ph in np.linspace(-math.pi, math.pi, 7201):
        w2 = shim.scaled_reference(a2, np.array([1.0, np.exp(1j * ph)]), t)
        best = min(best, shim.objective(a2, w2, t))
    lim1 = 0.3 * shim.peak_sar(q[:, :1, :1], shim.scaled_reference(a[:, :1], [1.0], t))[0]
    one = shim.solve(a[:, :1], t, np.array([1.0 + 0j]), q=q[:, :1, :1], sar_max=lim1)
    one_ref = shim.scaled_reference(a[:, :1], [1.0], t, q=q[:, :1, :1], sar_max=lim1)
    rows.append(_row("H4", "shim", "two channels, phase only: objective of the result over the smallest objective on a scan of 7,201 relative "
                     "phases, each at its best scale, minus 1; one channel under a SAR limit: the result against the scaled start",
                     "the scan (best of the two starts 0 and pi); with one channel only the scale is free", max(
                         min(res["f"], shim.solve(a2, t, np.array([1.0, -1.0], complex), kind="phase")["f"]) / best - 1,
                         abs(one["f"] / shim.objective(a[:, :1], one_ref, t) - 1)), 1e-5, unit="relative"))

    # H5 trade-off curve
    limits = s_ref * np.array([0.2, 0.35, 0.5, 0.75, 1.0, 1.5])
    curve = shim.tradeoff(a, t, cp, q, limits)
    fs = [c["f"] for c in curve]
    rows.append(_row("H5", "shim", "trade-off of uniformity against the limit on averaged local SAR (6 limits from 0.2 to 1.5 of the SAR of the "
                     "birdcage-like mode): largest rise of the objective when the limit is relaxed", "none, by construction: each limit also starts "
                     "from the solution of the tighter one, which stays allowed", max(b - a_ for a_, b in zip(fs, fs[1:])), 0.0))
    rows.append(_row("H5b", "shim", "the same curve: objective at the tightest limit over objective at the loosest", "a real trade-off: above 1.5",
                     fs[0] / fs[-1], 1.5, kind="min"))
    free_sar = shim.peak_sar(q, results[0]["w"])[0]                           # local SAR of the shim without limits
    binding = [c["peak_sar"] / c["sar_max"] for c in curve if c["sar_max"] < 0.9 * free_sar]
    rows.append(_row("H5c", "shim", f"the {len(binding)} limits of the curve that lie below the local SAR of the shim without limits: peak SAR of the "
                     "result over the limit, smallest", "the limit is reached: a result that stays below it gives uniformity away", min(binding), 0.999,
                     kind="min"))
    return rows


# ---------------------------------------------------------------------------------------------- EPT
def check_ept():
    rows = []
    h = 0.002
    err_a = err_s = 0.0
    band = np.inf
    for f in FREQS:
        radii, eps, sig = bodies(f)["homogeneous"]
        res = cylinder_case(f, (radii, eps, sig), h, stride=8)
        yy, xx = res["yy"], res["xx"]
        inside = np.hypot(yy, xx) < radii[0]
        valid = ept.valid_region(inside, 2) & (np.hypot(yy, xx) < radii[0] - 0.012)
        exact = res["cyl"].fields(analytic.sheet_modes("quadrature"), yy, xx)["b1p"]
        got = ept.helmholtz(exact, f, h, 2)
        err_a = max(err_a, np.abs(got["eps_r"][valid] / eps[0] - 1).max(), np.abs(got["sigma"][valid] / sig[0] - 1).max())
        got = ept.helmholtz(res["fields"]["b1p"][0], f, h, 2)
        err_s = max(err_s, np.abs(got["eps_r"][valid] / eps[0] - 1).max(), np.abs(got["sigma"][valid] / sig[0] - 1).max())
        edge = np.abs(np.hypot(yy, xx) - radii[0]) < h
        band = min(band, np.median(np.abs(got["sigma"][edge] / sig[0] - 1)))
    rows.append(_row("E1", "ept", "permittivity and conductivity of the homogeneous cylinder from the B1+ of the Bessel series (2 mm grid, 5 x 5 "
                     "window), more than 12 mm inside the surface, 64 / 128 / 298 MHz", "the values the field was computed with", err_a, 0.02, unit="relative"))
    rows.append(_row("E1b", "ept", "the same from the B1+ of the finite-difference solver", "the same", err_s, 0.02, unit="relative"))
    rows.append(_row("E2", "ept", "where it fails: median error of the conductivity in the pixels at the surface of the cylinder, smallest over the "
                     "three frequencies", "large: the Helmholtz relation does not hold across a boundary (above 50 %)", band, 0.5, kind="min",
                     unit="relative"))

    # E3 the approximations
    f = 298e6
    radii, eps, sig = bodies(f)["homogeneous"]
    cyl = analytic.LayeredCylinder(f, radii, eps, sig, COIL_R)
    n, n_pml, yy, xx = _grid(h)
    valid = np.hypot(yy, xx) < radii[0] - 0.012
    tx = cyl.fields(analytic.sheet_modes("quadrature"), yy, xx)
    rx_rev = np.conj(cyl.fields(analytic.sheet_modes("reverse"), yy, xx)["b1m"])
    spread = np.abs(np.angle(tx["b1p"][valid] * np.conj(rx_rev[valid]) * np.conj(tx["b1p"][n // 2, n // 2]) * rx_rev[n // 2, n // 2])).max()
    rows.append(_row("E3", "ept", "transceive-phase assumption on the centred cylinder under the ideal current sheet at 298 MHz (transmit in "
                     "quadrature, receive in the reverse mode): difference between the transmit phase and the receive phase over the cylinder",
                     "zero: the two fields are mirror images, so half the transceive phase is the transmit phase", spread, 1e-9, unit="rad"))
    e_map, s_map = cyl.cell_maps(yy, xx, h)
    sol = solver.Solver(f, e_map, s_map, h, n_pml=n_pml)
    by_rungs = {}
    for rungs in (16, 4):
        b_tx = sol.fields(coils.birdcage(COIL_R, rungs, "quadrature"))["b1p"][0]
        rx = coils.receive_maps(sol.fields(coils.birdcage(COIL_R, rungs, "reverse")))[0]
        got = ept.from_transceive(np.abs(b_tx), ept.transceive_phase(b_tx, rx), f, h, 2)
        by_rungs[rungs] = max(np.abs(got["eps_r"][valid] / eps[0] - 1).max(), np.abs(got["sigma"][valid] / sig[0] - 1).max())
    rows.append(_row("E3a", "ept", "the same with the solver and a birdcage of 16 rungs, B1+ taken as |B1+| exp(i transceive phase / 2): "
                     "permittivity and conductivity", "the values of the cylinder (16 rungs are close to the sheet)", by_rungs[16], 0.02, unit="relative"))
    rows.append(_row("E3a2", "ept", "the same with a birdcage of 4 rungs", "the assumption fails with few rungs: above 10 %", by_rungs[4], 0.1,
                     kind="min", unit="relative"))
    lin = cyl.fields(analytic.sheet_modes("linear"), yy, xx)
    got = ept.from_transceive(np.abs(lin["b1p"]), ept.transceive_phase(lin["b1p"], np.conj(lin["b1m"])), f, h, 2)
    rows.append(_row("E3b", "ept", "the same assumption with a linearly driven sheet used for both: median error of the conductivity",
                     "the assumption fails (transmit and receive phase differ): above 10 %", np.median(np.abs(got["sigma"][valid] / sig[0] - 1)), 0.1,
                     kind="min", unit="relative"))
    phase_only = ept.sigma_from_phase(np.angle(tx["b1p"]), f, h, 2)
    rows.append(_row("E3c", "ept", "phase-only conductivity laplace(phi) / (w mu0) at 298 MHz on the same cylinder: median error", "an approximation "
                     "that drops the term 2 grad(ln |B1+|) . grad(phi): above 10 % here", np.median(np.abs(phase_only[valid] / sig[0] - 1)), 0.1,
                     kind="min", unit="relative"))

    # E4 the derivative kernel
    v, u = np.meshgrid(np.arange(-20, 21) * 0.5, np.arange(-20, 21) * 0.5, indexing="ij")
    poly = 1.5 + 0.3 * v - 0.2 * u + 0.7 * v * v + 0.4 * v * u - 0.9 * u * u + 0.05 * v ** 3 - 0.02 * u * v * v
    exact = 2 * 0.7 - 2 * 0.9 + 0.3 * v - 0.04 * u
    err = 0.0
    for m in (1, 2, 3):
        err = max(err, np.abs(ept.laplacian(poly, 0.5, m) - exact)[m:-m, m:-m].max())
    rows.append(_row("E4", "ept", "Laplacian of a cubic polynomial with the 3 x 3, 5 x 5 and 7 x 7 least-squares kernels", "exact", err, 1e-10))
    return rows


# ---------------------------------------------------------------------------------------------- tables for the document
def demonstration():
    """|B1+| of a 16-rung quadrature birdcage (radius 14 cm, open) in the brain world at 64, 128 and 298 MHz, and the
    engine's declared pattern at 3 T. Returns a list of records."""
    wm_world = brain_world()
    out = []
    for f in FREQS:
        wm = brain_model(f)
        sol = wm.solver
        fld = sol.fields(coils.birdcage(COIL_R, RUNGS))
        lin = [sol.fields(coils.birdcage(COIL_R, RUNGS, "linear", current_a=2.0, phi0=p)) for p in (0.0, math.pi / 2)]
        organ, body = wm.masks["organ"], wm.masks["body"]
        b = np.abs(fld["b1p"][0])
        centre = float(sol.sample(b, 0, 0))
        bn = b / centre
        rim = organ & ~_erode(organ, 3)
        ez = fld["ez"][0]
        point = sol.point_sar(ez, wm.density)
        avg = sol.averaged_sar(ez, wm.density)
        mass = wm.density.sum() * sol.h ** 2
        mean_sar = sol.absorbed_power(ez) / mass
        per_ut2 = mean_sar * (1e-6 / centre) ** 2
        out.append(dict(
            label=f"computed, {f / 1e6:.0f} MHz", cv=float(bn[organ].std() / bn[organ].mean()), centre_edge=float(1.0 / bn[rim].mean()),
            min=float(bn[organ].min()), max=float(bn[organ].max()), b1m=float(np.abs(fld["b1m"][0][organ]).mean() / b[organ].mean()),
            peak_point=float(point[body].max() / mean_sar), peak_avg=float(avg.max() / mean_sar), sar_per_ut2=float(per_ut2),
            b1rms_at_head_limit=float(math.sqrt(3.2 / per_ut2)),
            linear_power=[float(sol.absorbed_power(q["ez"][0]) / sol.absorbed_power(ez) * abs(centre / sol.sample(q["b1p"][0], 0, 0)) ** 2)
                          for q in lin],
            grid=sol.n, h_mm=sol.h * 1e3))
    declared = np.asarray(wm_world["fields"]["b1"], float)
    organ = np.asarray(wm_world["masks"]["brain"], bool)
    n = wm_world["n"]
    centre = float(declared[n // 2, n // 2])
    dn = declared / centre
    rim = organ & ~_erode(organ, 15)
    out.append(dict(label="declared pattern of the engine, 3 T (`world.B1_SHAPE`)", cv=float(dn[organ].std() / dn[organ].mean()),
                    centre_edge=float(1.0 / dn[rim].mean()), min=float(dn[organ].min()), max=float(dn[organ].max()), b1m=None, peak_point=None,
                    peak_avg=None, sar_per_ut2=None, b1rms_at_head_limit=None, linear_power=None, grid=n, h_mm=wm_world["extent_mm"] / n))
    # how the two patterns differ at 3 T, on the master grid
    wm = brain_model(128e6)
    fld = wm.solver.fields(coils.birdcage(COIL_R, RUNGS))
    comp = np.abs(wm.to_world(fld["b1p"][0], order=3))
    comp = comp / comp[n // 2, n // 2]
    diff = comp[organ] / dn[organ]
    out.append(dict(label="computed at 128 MHz over declared, pixel by pixel in the brain", cv=float(diff.std() / diff.mean()), centre_edge=None,
                    min=float(diff.min()), max=float(diff.max()), b1m=None, peak_point=None, peak_avg=None, sar_per_ut2=None,
                    b1rms_at_head_limit=None, linear_power=None, grid=n, h_mm=wm_world["extent_mm"] / n,
                    correlation=float(np.corrcoef(comp[organ], dn[organ])[0, 1])))
    return out


def _erode(mask, cells):
    import scipy.ndimage
    return scipy.ndimage.binary_erosion(mask, iterations=cells)


def _f(x, digits=3):
    return "–" if x is None else f"{x:.{digits}g}"


def demonstration_markdown(recs=None):
    recs = demonstration() if recs is None else recs
    out = ["| \\|B1+\\| over the brain, relative to the isocentre | Coefficient of variation | Centre over rim | Smallest | Largest | "
           "Mean \\|B1-\\| over mean \\|B1+\\| | Peak point SAR over head mean | Peak averaged SAR over head mean | Head-mean SAR at 1 µT, W/kg | "
           "B1+ at which the head mean is 3.2 W/kg, µT | Power of linear over quadrature drive, field along y, along x |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        out.append(f"| {r['label']} | {_f(r['cv'])} | {_f(r['centre_edge'])} | {_f(r['min'])} | {_f(r['max'])} | {_f(r['b1m'])} | {_f(r['peak_point'])} | "
                   f"{_f(r['peak_avg'])} | {_f(r['sar_per_ut2'])} | {_f(r['b1rms_at_head_limit'])} | "
                   f"{'–' if r['linear_power'] is None else ', '.join(_f(v) for v in r['linear_power'])} |")
    from ..scanner import world
    by = {r["label"]: r for r in recs}
    c64, c128, c298 = (by[f"computed, {m} MHz"] for m in (64, 128, 298))
    declared, last = recs[-2], recs[-1]
    out += ["", "Derived from the table (and from the published numbers named in section 7):", "",
            f"- Coefficient of variation, declared over computed at 128 MHz: {declared['cv'] / c128['cv']:.2f}. Correlation of the two patterns "
            f"over the brain: {last['correlation']:.2f}.",
            f"- Mean |B1-| over mean |B1+| in the brain at 64, 128, 298 MHz: {c64['b1m']:.2f}, {c128['b1m']:.2f}, {c298['b1m']:.2f}.",
            f"- Coefficient of variation at 298 MHz over the published {PUBLISHED_CV_300MHZ:g} of a three-dimensional head model: "
            f"{c298['cv'] / PUBLISHED_CV_300MHZ:.2f}.",
            "- Peak averaged SAR over mean SAR, computed over published (" + ", ".join(f"{v:g}" for v in PUBLISHED_10G_OVER_MEAN.values())
            + f"): {c64['peak_avg'] / PUBLISHED_10G_OVER_MEAN[64]:.2f}, {c128['peak_avg'] / PUBLISHED_10G_OVER_MEAN[128]:.2f}, "
            f"{c298['peak_avg'] / PUBLISHED_10G_OVER_MEAN[300]:.2f} at 64, 128 and 298 (published: 300) MHz.",
            f"- B1+ at which the mean SAR of the slice is 3.2 W/kg, over the engine's public B1rms limit: "
            f"{c64['b1rms_at_head_limit'] / world.B1RMS_LIMIT_UT[1.5]:.2f} at 1.5 T ({world.B1RMS_LIMIT_UT[1.5]:g} µT), "
            f"{c128['b1rms_at_head_limit'] / world.B1RMS_LIMIT_UT[3.0]:.2f} at 3 T ({world.B1RMS_LIMIT_UT[3.0]:g} µT).",
            f"- Mean SAR for a fixed B1+ at the isocentre, 128 over 64 MHz: {c128['sar_per_ut2'] / c64['sar_per_ut2']:.2f} (the square of the "
            f"frequency ratio: 4.00); 298 over 128 MHz: {c298['sar_per_ut2'] / c128['sar_per_ut2']:.2f} (the square: {(298 / 128) ** 2:.2f}).",
            "- Power of linear drive over quadrature drive, mean of the two field directions: "
            + ", ".join(f"{np.mean(r['linear_power']):.2f}" for r in (c64, c128, c298)) + " (a circular body: 2)."]
    return "\n".join(out)


def tissue_markdown():
    """The tissue table of docs/physics/em.md: every engine class at the three frequencies, with its row and tag."""
    out = ["| Engine class | Row of the dispersion model | Tag | 64 MHz: permittivity, S/m | 128 MHz | 298 MHz | Wavelength at 298 MHz, mm | "
           "Density, kg/m³ | Note |", "|---|---|---|---|---|---|---|---|---|"]
    for name, (row, density, tag, note) in properties.CLASS_ROWS.items():
        cells = []
        for f in FREQS:
            e, s = properties.cole_cole(row, f)
            cells.append(f"{e:.1f}, {s:.3f}")
        e, s = properties.cole_cole(row, 298e6)
        out.append(f"| `{name}` | {row} | {tag} | {cells[0]} | {cells[1]} | {cells[2]} | {wavelength_m(298e6, e, s) * 1e3:.0f} | {density:g} | {note} |")
    for key, (e, s, density) in properties.PHANTOM.items():
        out.append(f"| `{key}` (test objects) | declared | C | {e:.1f}, {s:.3f} | {e:.1f}, {s:.3f} | {e:.1f}, {s:.3f} | "
                   f"{wavelength_m(298e6, e, s) * 1e3:.0f} | {density:g} | see `properties.PHANTOM` |")
    return "\n".join(out)


def cost_3d(repeat=10):
    """What the next step would cost: the time of one step of a three-dimensional finite-difference time-domain
    (Yee) update in numpy, measured here on one thread, times the number of steps to a steady state.

    Assumptions, not measurements: a box of 0.4 x 0.4 x 0.5 m around a head coil; the Courant step h / (c sqrt(3));
    15 RF periods to reach the steady state of a loaded coil driven by ideal currents; single precision; no
    absorbing layer, no sub-cell materials. Returns a list of records, one per (cell size, frequency)."""
    from .units import C0
    out = []
    for h in (0.005, 0.008):
        shape = tuple(int(round(v / h)) for v in (0.4, 0.4, 0.5))
        f32 = np.float32
        ex, ey, ez, hx, hy, hz = (np.zeros(shape, f32) for _ in range(6))
        ca, cb, db = np.full(shape, 0.99, f32), np.full(shape, 0.5, f32), f32(0.5)
        ez[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1.0

        def step(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, ca=ca, cb=cb, db=db):
            hx[:, :-1, :-1] -= db * ((ez[:, 1:, :-1] - ez[:, :-1, :-1]) - (ey[:, :-1, 1:] - ey[:, :-1, :-1]))
            hy[:-1, :, :-1] -= db * ((ex[:-1, :, 1:] - ex[:-1, :, :-1]) - (ez[1:, :, :-1] - ez[:-1, :, :-1]))
            hz[:-1, :-1, :] -= db * ((ey[1:, :-1, :] - ey[:-1, :-1, :]) - (ex[:-1, 1:, :] - ex[:-1, :-1, :]))
            ex[:, 1:, 1:] = ca[:, 1:, 1:] * ex[:, 1:, 1:] + cb[:, 1:, 1:] * ((hz[:, 1:, 1:] - hz[:, :-1, 1:]) - (hy[:, 1:, 1:] - hy[:, 1:, :-1]))
            ey[1:, :, 1:] = ca[1:, :, 1:] * ey[1:, :, 1:] + cb[1:, :, 1:] * ((hx[1:, :, 1:] - hx[1:, :, :-1]) - (hz[1:, :, 1:] - hz[:-1, :, 1:]))
            ez[1:, 1:, :] = ca[1:, 1:, :] * ez[1:, 1:, :] + cb[1:, 1:, :] * ((hy[1:, 1:, :] - hy[:-1, 1:, :]) - (hx[1:, 1:, :] - hx[1:, :-1, :]))
        step()
        t0 = time.perf_counter()
        for _ in range(repeat):
            step()
        per_step = (time.perf_counter() - t0) / repeat
        cells = int(np.prod(shape))
        for f in FREQS:
            steps = int(math.ceil(15.0 / f / (h / (C0 * math.sqrt(3.0)))))
            out.append(dict(h_mm=h * 1e3, cells=cells, mhz=f / 1e6, step_s=per_step, steps=steps, seconds=steps * per_step,
                            megabytes=cells * 12 * 4 / 1e6))
    return out


def cost_3d_text(recs=None):
    recs = cost_3d() if recs is None else recs
    lines = ["cell   cells      MHz   steps to 15 periods   s per step   s per channel   MB (6 fields + 6 coefficients, single precision)"]
    for r in recs:
        lines.append(f"{r['h_mm']:.0f} mm  {r['cells']:>8,}  {r['mhz']:>5.0f}   {r['steps']:>19,}   {r['step_s']:>10.4f}   {r['seconds']:>13.0f}   {r['megabytes']:.0f}")
    return "\n".join(lines)


BLOCKS = ("validation table", "tissue table", "demonstration", "cost")


def fill_document(text, rows=None):
    """The page with its four generated blocks refreshed (``<!-- name: begin -->`` ... ``<!-- name: end -->``)."""
    rows = run()[0] if rows is None else rows
    body = {"validation table": markdown_table(rows), "tissue table": tissue_markdown(), "demonstration": demonstration_markdown(),
            "cost": "```text\n" + cost_3d_text() + "\n```"}
    for name in BLOCKS:
        head, rest = text.split(f"<!-- {name}: begin -->")
        _, tail = rest.split(f"<!-- {name}: end -->")
        text = f"{head}<!-- {name}: begin -->\n{body[name].strip()}\n<!-- {name}: end -->{tail}"
    return text


# ---------------------------------------------------------------------------------------------- run and print
CHECKS = dict(properties=check_properties, analytic=check_analytic, solver=check_solver, coils=check_coils, shim=check_shim, ept=check_ept)


def run(groups=None):
    """Run every check, or those of the named groups. Returns (rows, seconds)."""
    t0 = time.perf_counter()
    rows = []
    for name in (groups or CHECKS):
        rows += CHECKS[name]()
    return rows, time.perf_counter() - t0


def _number(x):
    if x == 0:
        return "0"
    if abs(x) >= 1e-3 and abs(x) < 1e4:
        return f"{x:.4g}"
    return f"{x:.1e}"


def markdown_table(rows):
    """The validation table of docs/physics/em.md. Errors are printed with one significant digit below 1e-3: they
    are rounding noise or depend on the machine."""
    out = []
    for group, title in GROUPS.items():
        part = [r for r in rows if r["group"] == group]
        if not part:
            continue
        out += [f"### {title}", "", "| # | Quantity | Closed form, independent method or source | Largest error | Tolerance | |", "|---|---|---|---|---|---|"]
        for r in part:
            sign = "≤" if r["kind"] == "max" else "≥"
            unit = f" {r['unit']}" if r["unit"] else ""
            quantity, reference, unit = (v.replace("|", "\\|") for v in (r["quantity"], r["reference"], unit))   # a bar inside a cell is escaped
            out.append(f"| {r['id']} | {quantity} | {reference} | {_number(r['measured'])}{unit} | {sign} {_number(r['tolerance'])} | "
                       f"{'pass' if r['ok'] else 'FAIL'} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--tissues" in argv:
        print(tissue_markdown())
        return 0
    if "--demo" in argv:
        print(demonstration_markdown())
        return 0
    if "--cost3d" in argv:
        print(cost_3d_text())
        return 0
    if "--write-doc" in argv:
        path = argv[argv.index("--write-doc") + 1]
        with open(path, encoding="utf-8") as f:
            text = f.read()
        rows, _ = run()
        with open(path, "w", encoding="utf-8") as f:
            f.write(fill_document(text, rows))
        print(f"{path}: {sum(r['ok'] for r in rows)} of {len(rows)} rows pass")
        return 0 if all(r["ok"] for r in rows) else 1
    if "--timing" in argv:
        t = timing()
        print(f"master grid {t['master_n']} x {t['master_n']} at {t['master_mm']:g} mm; solver grid {t['n']} x {t['n']} at {t['h_mm']:g} mm = "
              f"{t['unknowns']:,} unknowns; one factorisation and 8 channel solves: {t['seconds']:.3f} s; property maps of the world: "
              f"{t['maps_seconds']:.3f} s")
        return 0
    rows, seconds = run([a for a in argv if a in CHECKS] or None)
    if "--markdown" in argv:
        print(markdown_table(rows))
    else:
        for r in rows:
            sign = "<=" if r["kind"] == "max" else ">="
            print(f"{r['id']:5s} {'pass' if r['ok'] else 'FAIL'}  {_number(r['measured']):>9s} {sign} {_number(r['tolerance']):<8s} {r['unit']:22s} {r['quantity'][:110]}")
        print(f"{sum(r['ok'] for r in rows)} of {len(rows)} rows pass; {seconds:.1f} s")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
