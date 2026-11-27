"""Validation of the spin simulator: every function against a closed form or an independent computation.

    python -m mrgym.spins.validate                 the table as text; exit status 1 if a row fails
    python -m mrgym.spins.validate --markdown      the table of docs/physics/spins.md

"Independent" means a computation that shares no code with the function under test: Pauli matrices and
``scipy.linalg.expm`` in the product basis (``_Brute``) against the eigenbasis propagation, an explicit ensemble of
gradient phases against the projection on coherence orders, the product-operator rules written out by hand, real
3 x 3 Bloch rotations with relaxation (``_bloch``) against the relaxation factors, the engine's fat spectrum against
the basis entry. Rows E2 and S6 are of another kind: their reference is ``mrgym.design.rotation.rotate``, which the
model "product" itself calls; they show that the spin layer and the design layer agree (and, for the model "full",
that a separate integration reproduces the design layer), not that either is right. Row E3 is the independent
check of a shaped pulse.
The references were written by the same authors as the code: the rows show that the code computes the stated
equations. Rows that quote a published number say which one and where it was read.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import scipy.linalg

from ..design import pulses, rotation
from ..design.units import GAMMA_HZ_PER_T
from . import evolve, metabolites, operators, quantify, sequences

GROUPS = dict(operators="Operators and Hamiltonian", evolution="Propagation and detection", sequences="Sequences",
              basis="Metabolite basis", quantify="Quantification")
SEED = 20261126


def _row(id, group, quantity, reference, measured, tolerance, kind="max", unit=""):
    """kind "max": passes when measured <= tolerance; "min": when measured >= tolerance."""
    measured = float(measured)
    ok = bool(np.isfinite(measured) and (measured <= tolerance if kind == "max" else measured >= tolerance))
    return dict(id=id, group=group, quantity=quantity, reference=reference, measured=measured, tolerance=float(tolerance), kind=kind,
                unit=unit, ok=ok)


# ---------------------------------------------------------------------------------------------- independent code
class _Brute:
    """n protons in the product basis with Pauli matrices and ``scipy.linalg.expm``: no code of the package."""

    def __init__(self, freqs_hz, couplings, strong=True):
        n = len(freqs_hz)
        sx, sy, sz = np.array([[0, 0.5], [0.5, 0]]), np.array([[0, -0.5j], [0.5j, 0]]), np.array([[0.5, 0], [0, -0.5]])

        def embed(m, k):
            out = np.eye(1)
            for q in range(n):
                out = np.kron(out, m if q == k else np.eye(2))
            return out

        self.n = n
        self.ix, self.iy, self.iz = ([embed(m, k) for k in range(n)] for m in (sx, sy, sz))
        self.fx, self.fy, self.fz = sum(self.ix), sum(self.iy), sum(self.iz)
        h = sum(f * z for f, z in zip(freqs_hz, self.iz)).astype(complex)
        for i, j, value in couplings:
            h = h + value * self.iz[i] @ self.iz[j]
            if strong:
                h = h + value * (self.ix[i] @ self.ix[j] + self.iy[i] @ self.iy[j])
        self.h = h

    def free(self, ms):
        return scipy.linalg.expm(-2e-3j * np.pi * self.h * ms)

    def pulse(self, flip_rad, phase_rad=0.0, spins=None):
        fx = self.fx if spins is None else sum(self.ix[k] for k in spins)
        fy = self.fy if spins is None else sum(self.iy[k] for k in spins)
        return scipy.linalg.expm(-1j * flip_rad * (fx * math.cos(phase_rad) + fy * math.sin(phase_rad)))

    def z(self, angle):
        return scipy.linalg.expm(-1j * angle * self.fz)

    def waveform(self, b1_ut, dt_ms, offset_hz=0.0):
        """Exact propagator of a piecewise-constant B1 under the whole Hamiltonian (plus a common offset)."""
        u = np.eye(2 ** self.n, dtype=complex)
        for b in np.asarray(b1_ut, complex):
            h = self.h + offset_hz * self.fz + 1e-6 * GAMMA_HZ_PER_T * (b.real * self.fx + b.imag * self.fy)
            u = scipy.linalg.expm(-2e-3j * np.pi * h * dt_ms) @ u
        return u

    def signal(self, rho):
        return 1j * np.trace(rho @ (self.fx + 1j * self.fy)) / 2 ** (self.n - 2)


def _bloch(events, tr_ms, t1_ms, t2_ms, df_hz=37.0, reps=40):
    """Signal i Mxy at the event "acquire" in the steady state of a repetition, for one isochromat, with real 3 x 3
    rotations and relaxation. Events: ("rf", axis, angle), ("free", ms, extra phase in rad), ("spoil",), ("acquire",).
    "spoil" removes the transverse magnetisation; it is also removed at the end of every repetition."""
    def rot(axis, angle):
        n = np.asarray(axis, float)
        k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)

    m, out = np.array([0.0, 0.0, 1.0]), 0.0
    for _ in range(reps):
        used = 0.0
        for ev in events + (("free", None, 0.0),):
            if ev[0] == "rf":
                m = rot(ev[1], ev[2]) @ m
            elif ev[0] == "acquire":
                out = 1j * (m[0] + 1j * m[1])
            elif ev[0] == "spoil":
                m = np.array([0.0, 0.0, m[2]])
            else:
                ms = tr_ms - used if ev[1] is None else ev[1]
                used += ms
                m = rot((0, 0, 1), 2e-3 * np.pi * df_hz * ms + ev[2]) @ m
                e1, e2 = math.exp(-ms / t1_ms), math.exp(-ms / t2_ms)
                m = np.array([m[0] * e2, m[1] * e2, 1 + (m[2] - 1) * e1])
        m = np.array([0.0, 0.0, m[2]])
    return out


def _apply(u, rho):
    return u @ rho @ u.conj().T


def _order(errors):
    return float(np.mean([math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]))


def _sorted_lines(lines):
    k = np.argsort(lines.freq_hz)
    return lines.freq_hz[k], lines.amp[k]


# ---------------------------------------------------------------------------------------------- operators
def check_operators():
    rows, rng = [], np.random.default_rng(SEED)

    # O1 spin matrices
    err = 0.0
    for d in range(1, 11):
        j = 0.5 * d
        jx, jy, jz, jp, jm = operators.spin_matrices(j)
        err = max(err, np.abs(jx @ jy - jy @ jx - 1j * jz).max(), np.abs(jy @ jz - jz @ jy - 1j * jx).max(),
                  np.abs(jz @ jx - jx @ jz - 1j * jy).max(), np.abs(jx @ jx + jy @ jy + jz @ jz - j * (j + 1) * np.eye(d + 1)).max())
    rows.append(_row("O1", "operators", "spin matrices for j = 1/2 to 5", "[Jx, Jy] = i Jz and cyclic; Jx^2 + Jy^2 + Jz^2 = j (j + 1)", err, 1e-12))

    # O2 multiplicities of equivalent groups
    err = max(abs(sum(g * (2 * j + 1) for j, g in operators.multiplicities(n)) - 2 ** n) for n in range(1, 13))
    rows.append(_row("O2", "operators", "total-spin decomposition of 1 to 12 equivalent protons", "sum over j of g(n, j) (2j + 1) = 2^n", err, 0))

    # O3 Wigner matrices against the rotation of the symmetric product state space
    err = 0.0
    for _ in range(40):
        axis, angle = rng.standard_normal(3), rng.uniform(0, 2 * np.pi)
        a, b = rotation.axis_angle(axis, angle)
        n_hat = axis / np.linalg.norm(axis)
        for d in (2, 3, 4, 5, 8, 10):
            jx, jy, jz, _, _ = operators.spin_matrices(0.5 * (d - 1))
            ref = scipy.linalg.expm(-1j * angle * (n_hat[0] * jx + n_hat[1] * jy + n_hat[2] * jz))
            err = max(err, np.abs(operators.wigner(0.5 * (d - 1), a, b) - ref).max())
    rows.append(_row("O3", "operators", "rotation matrices of a group of total spin 1/2 to 9/2 for 40 random rotations given as Cayley-Klein parameters of `mrgym.design.rotation`",
                     "`scipy.linalg.expm` of -i angle n.J", err, 1e-12))

    # O4 the Hamiltonian commutes with Fz; eigenvectors have a definite M
    err = 0.0
    for n in (2, 4, 6):
        s = operators.SpinSystem("r", tuple(rng.uniform(0.5, 4.5, n)), tuple((i, j, rng.uniform(-15, 15)) for i in range(n) for j in range(i + 1, n)))
        c = operators.compile(s, 3.0, 4.7)
        sec = c.sectors[0]
        br = _Brute(c.freq_hz, s.j_hz)
        h = sec.vec @ np.diag(sec.energy_hz) @ sec.vec.T
        fz = sec.vec @ np.diag(sec.m) @ sec.vec.T
        err = max(err, np.abs(h - br.h).max() / np.abs(br.h).max(), np.abs(fz - br.fz).max(), np.abs(sec.vec.T @ sec.vec - np.eye(sec.dim)).max())
    rows.append(_row("O4", "operators", "Hamiltonian of 2, 4 and 6 random strongly coupled protons rebuilt from its eigenvalues and eigenvectors; Fz of the eigenvectors",
                     "sum nu Iz + sum J I.I from Pauli matrices; Fz diagonal with a definite M per eigenvector", err, 1e-12))

    # O5 AX and AB: positions and intensities
    def two_spin(delta_hz, jj, coupling):
        s = operators.SpinSystem("ab", (4.7 + delta_hz / operators.larmor_hz_per_ppm(3.0), 4.7), ((0, 1, jj),))
        return _sorted_lines(evolve.run(operators.compile(s, 3.0, 4.7, coupling), sequences.pulse_acquire().blocks)[0])

    f, a = two_spin(300.0, 7.0, "weak")
    err = max(np.abs(f - np.array([-3.5, 3.5, 296.5, 303.5])).max(), np.abs(a - 0.5).max())
    rows.append(_row("O5", "operators", "AX spectrum (weak coupling, 300 Hz apart, J = 7 Hz): four line positions (Hz) and amplitudes",
                     "nu_A +- J/2, nu_X +- J/2; amplitude 1/2 each", err, 1e-10))
    err = 0.0
    for delta, jj in ((300.0, 7.0), (40.0, 7.0), (12.0, 10.0), (3.0, 15.0), (25.0, -12.0)):
        f, a = two_spin(delta, jj, "strong")
        c, centre = math.hypot(delta, jj), delta / 2
        pos = np.array([centre - (c + abs(jj)) / 2, centre - (c - abs(jj)) / 2, centre + (c - abs(jj)) / 2, centre + (c + abs(jj)) / 2])
        amp = 0.5 * np.array([1 - abs(jj) / c, 1 + abs(jj) / c, 1 + abs(jj) / c, 1 - abs(jj) / c])
        err = max(err, np.abs(f - pos).max(), np.abs(a - amp).max())
    rows.append(_row("O5b", "operators", "AB spectrum (strong coupling; shift difference d from 300 to 3 Hz, J from 7 to 15 Hz and -12 Hz): positions and amplitudes",
                     "centre +- (C +- |J|)/2, C = sqrt(d^2 + J^2); inner lines (1 + |J|/C)/2, outer (1 - |J|/C)/2: the roof effect", err, 1e-10))

    # O6 A3X in weak coupling: doublet and quartet
    s = operators.SpinSystem("a3x", (1.3142, 4.0974), ((0, 1, 6.933),), counts=(3, 1))
    c = operators.compile(s, 3.0, 4.7, "weak")
    f, a = _sorted_lines(evolve.run(c, sequences.pulse_acquire().blocks)[0])
    fa, fx = c.freq_hz
    pos = np.array([fa - 6.933 / 2, fa + 6.933 / 2, fx - 1.5 * 6.933, fx - 0.5 * 6.933, fx + 0.5 * 6.933, fx + 1.5 * 6.933])
    amp = np.array([1.5, 1.5, 1 / 8, 3 / 8, 3 / 8, 1 / 8])
    rows.append(_row("O6", "operators", "A3X spectrum (lactate shifts and coupling, weak coupling) at 3 T: positions (Hz) and amplitudes of the six lines",
                     "doublet nu_A +- J/2 of 3/2 each; quartet nu_X + (-3, -1, 1, 3) J/2 of 1/8, 3/8, 3/8, 1/8", max(np.abs(f - pos).max(), np.abs(a - amp).max()), 1e-10))
    lines = evolve.run(operators.compile(s, 3.0, 4.7, "strong"), sequences.pulse_acquire().blocks)[0]
    rows.append(_row("O6b", "operators", "the same with strong coupling: total amplitude of the lines within 20 Hz of the doublet and of the quartet",
                     "3 and 1 protons to within 1e-6: at 3 T (J / shift difference = 0.02) the coupling moves intensity inside each multiplet, and between the two only in a high order",
                     max(abs(lines.window(fa - 20, fa + 20).total() - 3), abs(lines.window(fx - 20, fx + 20).total() - 1)), 1e-6))

    # O7 equivalent groups: total-spin sectors against every proton as its own site
    err, dims = 0.0, []
    for counts in ((3, 1), (3, 2, 1), (2, 2, 2), (3, 2, 2)):
        shifts = tuple(rng.uniform(0.9, 4.3, len(counts)))
        js = tuple((i, i + 1, rng.uniform(5, 9)) for i in range(len(counts) - 1))
        s = operators.SpinSystem("eq", shifts, js, counts=counts)
        seq = sequences.press(40.0, te1_ms=14.0).blocks[:-1] + (evolve.rf(37.0, 20.0), evolve.delay(3.0), evolve.acquire())
        la = evolve.run(operators.compile(s, 3.0, 4.7), seq)[0]
        lb = evolve.run(operators.compile(s.expand(intra_j_hz=-12.4), 3.0, 4.7), seq)[0]
        err = max(err, np.abs(la.fid(64, 0.7) - lb.fid(64, 0.7)).max())
        dims.append((max(x.dim for x in operators.compile(s).sectors), 2 ** sum(counts)))
    rows.append(_row("O7", "operators", "equivalent groups as total spins (A3X, A3M2X, A2M2X2, A3M2X2, random shifts, strong coupling), free induction decay after "
                     f"PRESS and a further pulse; largest matrix {dims[-1][0]} instead of {dims[-1][1]}",
                     "every proton as its own site, with a coupling of -12.4 Hz inside the groups (it must not matter)", err, 1e-11))

    # O8 field strength: shift dispersion in Hz scales, J does not
    err = 0.0
    for b0 in (1.5, 3.0, 7.0):
        s = operators.SpinSystem("ab", (2.0, 3.5), ((0, 1, 9.0),))
        f, _ = _sorted_lines(evolve.run(operators.compile(s, b0, 4.7), sequences.pulse_acquire().blocks)[0])
        d = 1.5 * 1e-6 * GAMMA_HZ_PER_T * b0
        err = max(err, abs((f[1] - f[0]) - 9.0), abs((f[3] - f[2]) - 9.0), abs(0.5 * (f[2] + f[3] - f[0] - f[1]) - math.hypot(d, 9.0)))
    rows.append(_row("O8", "operators", "two protons 1.5 ppm apart with J = 9 Hz at 1.5, 3 and 7 T: the two splittings and the distance of the doublet centres",
                     "splittings J at every field; distance sqrt(d^2 + J^2) with d = 1.5 ppm x gamma B0 (95.8, 191.6, 447.1 Hz)", err, 1e-10, unit="Hz"))
    return rows


# ---------------------------------------------------------------------------------------------- evolution
def check_evolution():
    rows, rng = [], np.random.default_rng(SEED + 1)

    def random_system(n, lo=5.0, hi=9.0):
        return operators.SpinSystem("r", tuple(rng.uniform(1.0, 4.4, n)), tuple((i, j, rng.uniform(lo, hi)) for i in range(n) for j in range(i + 1, n)))

    # E1 unitarity and conserved quantities
    err = 0.0
    p = pulses.sinc(180.0, 3.0, 4.0, thickness_mm=10.0, n=150)
    for n in (2, 3, 5):
        c = operators.compile(random_system(n), 3.0, 4.7)
        sec = c.sectors[0]
        us = [evolve.hard_propagator(sec, c, 1.1, 0.4), evolve.pulse_propagator(sec, c, p.b1_ut, p.dt_ms, p.g_mt_m, np.array([-3.0, 2.0]), 55.0, 0.3, "product"),
              evolve.pulse_propagator(sec, c, p.b1_ut, p.dt_ms, p.g_mt_m, np.array([-3.0, 2.0]), 55.0, 0.3, "full")]
        for u in us:
            err = max(err, np.abs(u @ np.conj(np.swapaxes(u, -1, -2)) - np.eye(sec.dim)).max())
        rho = sec.fz() + 0.3 * sec.site_operator(0, "x")
        out = evolve.free(sec, evolve.sandwich(us[2][0], evolve.free(sec, evolve.sandwich(us[0], rho), 13.0)), 7.0)
        err = max(err, abs(np.trace(out) - np.trace(rho)), abs(np.trace(out @ out) - np.trace(rho @ rho)) / abs(np.trace(rho @ rho)),
                  np.abs(out - out.conj().T).max())
    rows.append(_row("E1", "evolution", "propagators of a hard pulse and of a slice-selective sinc (both models) for 2, 3 and 5 coupled protons; a state carried "
                     "through pulse, delay, pulse, delay", "U U^+ = 1; trace, trace of rho^2 and Hermiticity conserved", err, 1e-11))

    # E2 uncoupled spins: the density matrix against the Bloch rotation of mrgym.design
    err = 0.0
    shapes = [pulses.sinc(90.0, 2.0, 4.0, thickness_mm=5.0, n=128), pulses.slr(180.0, 4.0, 5.0, thickness_mm=8.0, kind="refocusing", n=160),
              pulses.sech(18.0, 8.0, 1.3, 5.0, thickness_mm=10.0, n=400)]
    pos = np.linspace(-9.0, 9.0, 13)
    s3 = operators.SpinSystem("u", (1.3, 3.0, 4.7), ())
    c = operators.compile(s3, 3.0, 4.7)
    sec = c.sectors[0]
    start = evolve.sandwich(evolve.hard_propagator(sec, c, 0.7, 1.9), sec.fz())           # a tilted start
    mxy0, mz0 = rotation.apply(*rotation.axis_angle((math.cos(1.9), math.sin(1.9), 0), 0.7))
    for pl in shapes:
        for model in evolve.MODELS:
            u = evolve.pulse_propagator(sec, c, pl.b1_ut, pl.dt_ms, pl.g_mt_m, pos, 0.0, 0.0, model)
            rho = evolve.sandwich(u, start)
            for site, f in enumerate(c.freq_hz):
                a, b = rotation.rotate(pl.b1_ut, pl.dt_ms, pl.g_mt_m, pos, f)
                mxy, mz = rotation.apply(a, b, mxy0, mz0)
                for k in range(pos.size):
                    got = evolve.magnetisation(sec, c, rho[k], site)
                    err = max(err, abs(got[0] - mxy[k]), abs(got[1] - mz[k]))
    rows.append(_row("E2", "evolution", "three uncoupled protons (1.3, 3.0, 4.7 ppm) from a tilted start through a slice-selective sinc, a Shinnar-Le Roux "
                     "refocusing pulse and a hyperbolic secant at 13 positions, both pulse models: Mxy and Mz of every site",
                     "`mrgym.design.rotation.rotate` and `apply` at the site's frequency (the Bloch rotation)", err, 1e-11))

    # E3 the model "full" against exact piecewise-constant propagation under the whole Hamiltonian
    s2 = operators.SpinSystem("ab", (2.0, 2.25), ((0, 1, 11.0),))
    c = operators.compile(s2, 3.0, 4.7)
    sec = c.sectors[0]
    br = _Brute(c.freq_hz, s2.j_hz)
    errs = []
    for n in (50, 100, 200, 400):
        g = sequences.gaussian(180.0, 12.0, n=n)
        b1 = g.b1_ut * np.exp(0.6j)
        ref = br.waveform(b1, 12.0 / n, offset_hz=300.0)                           # in the frame of a carrier at -300 Hz
        zc = br.z(np.pi * 1e-3 * (-300.0) * 12.0)
        ref = zc @ ref @ zc
        u = sec.to_product(evolve.pulse_propagator(sec, c, g.b1_ut, g.dt_ms, None, None, -300.0, 0.6, "full"))
        errs.append(np.abs(u - ref).max())
    rows.append(_row("E3", "evolution", "a 12 ms Gaussian 180 near two strongly coupled protons (32 Hz apart, J = 11 Hz), carrier 300 Hz below the reference, "
                     "model full, 400 samples: the propagator", "`scipy.linalg.expm` of the whole Hamiltonian for every sample, Pauli matrices in the product basis", errs[-1], 1e-4))
    rows.append(_row("E3o", "evolution", "the same: order of convergence from 50 to 400 samples", "second order (hard-pulse splitting)", abs(_order(errs) - 2.0), 0.05, unit="in the order"))
    g = sequences.gaussian(180.0, 12.0, n=400)
    up = sec.to_product(evolve.pulse_propagator(sec, c, g.b1_ut, g.dt_ms, None, None, -300.0, 0.6, "product"))
    c0 = operators.compile(operators.SpinSystem("ab", (2.0, 2.25), ()), 3.0, 4.7)
    u0 = [evolve.pulse_propagator(c0.sectors[0], c0, g.b1_ut, g.dt_ms, None, None, -300.0, 0.6, m) for m in evolve.MODELS]
    rows.append(_row("E3b", "evolution", "the two pulse models for that pulse without coupling", "equal: the model product is exact for uncoupled spins", np.abs(u0[0] - u0[1]).max(), 1e-11))
    rows.append(_row("E3c", "evolution", "the model product for that pulse with the coupling (J x T = 0.13)", "the model full; the difference is the coupling evolution that the pulse interrupts, a fraction of J x T",
                     np.abs(up - ref).max(), 0.02))

    # E4 detection: lines against a time-stepped trace
    s = random_system(4)
    c = operators.compile(s, 3.0, 4.7)
    blocks = (evolve.rf(90.0), evolve.delay(21.0), evolve.rf(180.0, 90.0), evolve.delay(9.0), evolve.rf(53.0, 40.0), evolve.delay(4.0))
    lines = evolve.run(c, blocks + (evolve.acquire(),))[0]
    br = _Brute(c.freq_hz, s.j_hz)
    rho = br.fz.astype(complex)
    for u in (br.pulse(np.pi / 2), br.free(21.0), br.pulse(np.pi, np.pi / 2), br.free(9.0), br.pulse(math.radians(53.0), math.radians(40.0)), br.free(4.0)):
        rho = _apply(u, rho)
    step = br.free(0.37)
    ref = []
    for _ in range(24):
        ref.append(br.signal(rho))
        rho = _apply(step, rho)
    rows.append(_row("E4", "evolution", "free induction decay (24 samples) of four strongly coupled protons after three pulses, from the line list",
                     "i Tr(rho(t) F+) / 2^(n-2), the state stepped with `scipy.linalg.expm` in the product basis", np.abs(lines.fid(24, 0.37) - np.array(ref)).max(), 1e-11))

    # E5 the spectrum of one line
    t2, amp, dwell, n = 80.0, 1.7, 0.25, 16384
    df = 1e3 / (n * dwell)
    f0 = 152 * df
    one = evolve.Lines(np.array([f0]), np.array([amp + 0j]), np.array([0]), ("g",))
    f, spec = evolve.spectrum(one.fid(n, dwell, t2), dwell)
    k = int(np.argmax(spec.real))
    err = max(abs(spec.real[k] / (amp * 1e-3 * t2) - 1), abs(quantify.linewidth_hz(f, spec.real, (0, 80)) * np.pi * 1e-3 * t2 - 1),
              abs(spec.real.sum() * df / (amp / 2) - 1), abs(f[k] - f0) / df)
    rows.append(_row("E5", "evolution", "spectrum of one decaying line (T2 = 80 ms, on a grid point, 16,384 samples of 0.25 ms): height, full width at half maximum, area of the real part, position",
                     "height A T2, width 1 / (pi T2), area A / 2 (relative errors); position (in points)", err, 2e-3))
    z = 1e-3 * dwell * (1e3 / t2 + 2j * np.pi * (f - f0))
    exact = amp * 1e-3 * dwell * 0.5 / np.tanh(z / 2)
    rows.append(_row("E5b", "evolution", "the sampled spectrum at every frequency, relative to the peak", "the sum of the sampled decay in closed form: A dwell (1/2) coth(z / 2), z = dwell (1 / T2 + 2 pi i (f - f0))",
                     np.abs(spec - exact).max() / np.abs(exact).max(), 1e-10))
    cont = evolve.lorentzian_spectrum(one, f, t2)
    rows.append(_row("E5c", "evolution", "the sampled spectrum against the continuous one, relative to the peak", "A / (1 / T2 + 2 pi i (f - f0)); the difference is the sampling, A dwell z / 12 to first order (1.0e-03 at the edge of the band)",
                     np.abs(spec - cont).max() / np.abs(cont).max(), 2e-3))

    # E6 spin-echo J modulation, weak coupling
    err = 0.0
    ax = operators.SpinSystem("ax", (1.3, 4.1), ((0, 1, 7.3),), groups=("a", "x"))
    for te in (10.0, 35.0, 68.5, 100.0, 137.0, 200.0):
        both = sequences.simulate(ax, sequences.spin_echo(te), coupling="weak").total("a")
        sel = sequences.simulate(ax, sequences.Sequence("se", (evolve.rf(90.0), evolve.delay(te / 2), evolve.rf(180.0, 90.0, (1.0, 2.0), "refocus"),
                                                              evolve.delay(te / 2), evolve.acquire())), coupling="weak").total("a")
        err = max(err, abs(both - math.cos(np.pi * 7.3e-3 * te)), abs(sel - 1.0))
    rows.append(_row("E6", "evolution", "spin echo of an AX pair (weak coupling, J = 7.3 Hz), echo times 10 to 200 ms: signal of A with a 180 on both spins and with a 180 on A only",
                     "cos(pi J TE) (the chemical shift is refocused, the coupling is not); 1 (the coupling is refocused too)", err, 1e-11))

    # E7 INEPT-like transfer, product operators
    err = 0.0
    c = operators.compile(ax, 3.0, 4.7, "weak")
    sec = c.sectors[0]
    two_iz_sy = 2 * sec.site_operator(0, "z") @ sec.site_operator(1, "y")
    two_ix_sz = 2 * sec.site_operator(0, "x") @ sec.site_operator(1, "z")
    for tau in (5.0, 17.0, 1e3 / (4 * 7.3), 50.0):
        head = (evolve.rf(90.0, 0.0, (1.0, 2.0)), evolve.delay(tau), evolve.rf(180.0, 0.0), evolve.delay(tau))
        mid = evolve.final_state(c, head)
        end = evolve.final_state(c, head + (evolve.rf(90.0, 90.0, (1.0, 2.0)), evolve.rf(90.0, 0.0, (4.0, 4.2))))
        cs, sn = math.cos(2 * np.pi * 7.3e-3 * tau), math.sin(2 * np.pi * 7.3e-3 * tau)

        def ex(rho, o):
            return (np.trace(rho @ o) / np.trace(o @ o)).real

        iy, sz = sec.site_operator(0, "y"), sec.site_operator(1, "z")
        err = max(err, abs(ex(mid, iy) - cs), abs(ex(mid, two_ix_sz) + sn), abs(ex(mid, sz) + 1), abs(ex(end, iy) - cs), abs(ex(end, two_iz_sy) + sn))
    rows.append(_row("E7", "evolution", "INEPT on an AX pair (weak coupling): 90x(I) - tau - 180x(I, S) - tau, then 90y(I) 90x(S); coefficients of Iy, 2 Ix Sz, Sz and 2 Iz Sy for four tau, one of them 1 / (4J)",
                     "product operators (Sorensen et al. 1983): Iz -> Iy cos(2 pi J tau) - 2 Ix Sz sin(2 pi J tau), Sz -> -Sz; then 2 Ix Sz -> 2 Iz Sy: complete transfer to antiphase S at tau = 1 / (4J)", err, 1e-11))

    # E8 gradients: projection on coherence orders against an explicit ensemble
    s = random_system(3)
    c = operators.compile(s, 3.0, 4.7)
    sel = (evolve.rf(90.0), evolve.delay(11.0), evolve.rf(140.0, 90.0, pathway="refocus"), evolve.delay(17.0), evolve.rf(160.0, 30.0, (0.0, 3.0), "keep"),
           evolve.delay(6.0), evolve.acquire())
    exp = (evolve.rf(90.0), evolve.delay(11.0), evolve.gradient(1, "a", 16), evolve.rf(140.0, 90.0), evolve.gradient(1, "a", 16), evolve.average("a"), evolve.delay(17.0),
           evolve.gradient(2, "b", 16), evolve.rf(160.0, 30.0, (0.0, 3.0)), evolve.gradient(-2, "b", 16), evolve.delay(6.0), evolve.acquire())
    rows.append(_row("E8", "evolution", "three strongly coupled protons: a 140 degree pulse between equal crushers and a selective 160 degree pulse between opposite crushers, "
                     "as projections on coherence orders (p -> -p, p -> p)", "explicit ensembles of 16 gradient phases, summed at the end",
                     np.abs(evolve.run(c, sel)[0].fid(32, 0.9) - evolve.run(c, exp)[0].fid(32, 0.9)).max(), 1e-11))
    br = _Brute(c.freq_hz, s.j_hz)
    acc = 0.0
    ref = np.zeros(8, complex)
    step = br.free(0.9)
    for ka in range(16):
        for kb in range(16):
            za, zb = br.z(2 * np.pi * ka / 16), br.z(2 * np.pi * 2 * kb / 16)
            rho = br.fz.astype(complex)
            for u in (br.pulse(np.pi / 2), br.free(11.0), za, br.pulse(math.radians(140.0), np.pi / 2), za, br.free(17.0), zb,
                      br.pulse(math.radians(160.0), math.radians(30.0), [k for k, x in enumerate(s.shifts_ppm) if x <= 3.0]), zb.conj().T, br.free(6.0)):
                rho = _apply(u, rho)
            for k in range(8):
                ref[k] += br.signal(rho) / 256
                rho = _apply(step, rho)
    acc = np.abs(evolve.run(c, sel)[0].fid(8, 0.9) - ref).max()
    rows.append(_row("E8b", "evolution", "the same sequence", "256 isochromats with z rotations for the gradients, Pauli matrices and `scipy.linalg.expm`", acc, 1e-11))

    # E9 recovery between repetitions
    err = 0.0
    for tr, t1, te in ((1500.0, 1200.0, 30.0), (3000.0, 1400.0, 144.0), (700.0, 900.0, 20.0)):
        e1, eh = math.exp(-tr / t1), math.exp(-(tr - te / 2) / t1)
        err = max(err, abs(sequences.recovery(tr, t1) - (1 - e1)), abs(sequences.recovery(tr, t1, (te / 2,)) - (1 - 2 * eh + e1)),
                  abs(sequences.recovery(tr, t1, (), 30.0) - (1 - e1) / (1 - e1 * math.cos(math.radians(30.0)))))
    rows.append(_row("E9", "evolution", "longitudinal magnetisation before the excitation in the steady state: 90 degree pulse; spin echo; 30 degree pulse",
                     "1 - E1; 1 - 2 exp(-(TR - TE/2) / T1) + E1; (1 - E1) / (1 - E1 cos 30)", err, 1e-13))

    # E9b, E9c relaxation factors of the sequences against the Bloch equations
    one, t1, t2 = operators.SpinSystem("w", (4.99,), ()), 1350.0, 110.0
    x, y, half, full = (1, 0, 0), (0, 1, 0), np.pi / 2, np.pi

    def echo(times, te):
        ev, now = [("rf", x, half)], 0.0
        for t in times:
            ev += [("free", t - now, 0.0), ("rf", y, full)]
            now = t
        return tuple(ev) + (("free", te - now, 0.0), ("acquire",))

    err = 0.0
    for seq, events, tr in ((sequences.spin_echo(40.0), echo((20.0,), 40.0), 900.0), (sequences.press(te1_ms=20.0, te2_ms=40.0), echo((10.0, 40.0), 60.0), 1200.0),
                            (sequences.slaser(64.0), echo((8.0, 24.0, 40.0, 56.0), 64.0), 1500.0), (sequences.mega_press(68.0, None), echo((8.5, 42.5), 68.0), 1100.0)):
        got = sequences.simulate(one, seq, relaxation={"w": (t1, t2)}, tr_ms=tr).total()
        err = max(err, abs(got - _bloch(events, tr, t1, t2, c_df := float(operators.frequencies_hz(one.shifts_ppm, 3.0, 4.7)[0]))))
    rows.append(_row("E9b", "evolution", "an uncoupled proton with T1 = 1350 ms, T2 = 110 ms in the steady state of spin echo (TE 40, TR 900 ms), PRESS (20 + 40, 1200), semi-LASER (64, 1500) "
                     "and MEGA-PRESS without editing (68, 1100): signal from the relaxation factors of the sequence", "the Bloch equations stepped over 40 repetitions with real 3 x 3 rotations, refocusing pulses at times written by hand",
                     err, 1e-12))
    err, err_coherent = 0.0, 0.0
    for te, tm, tr in ((20.0, 30.0, 2000.0), (30.0, 100.0, 1000.0)):
        got = sequences.simulate(one, sequences.steam(te, tm), relaxation={"w": (t1, t2)}, tr_ms=tr).total()
        for spoil in (True, False):
            ref = np.mean([_bloch((("rf", x, half), ("free", te / 2, 2 * np.pi * ka / 8), ("rf", x, half)) + ((("spoil",),) if spoil else ())
                                  + (("free", tm, 2 * np.pi * kb / 8), ("rf", x, -half), ("free", te / 2, 2 * np.pi * ka / 8), ("acquire",)), tr, t1, t2, c_df)
                           for ka in range(8) for kb in range(8)])
            if spoil:
                err = max(err, abs(got - ref))
            else:
                err_coherent = max(err_coherent, abs(got - ref) / abs(ref))
    rows.append(_row("E9c", "evolution", "the same proton in the steady state of STEAM (TE 20, TM 30, TR 2000 ms and TE 30, TM 100, TR 1000 ms)",
                     "the Bloch equations for 8 crusher phases of TE, transverse magnetisation removed in TM: the third pulse leaves no z magnetisation, so the recovery runs for TR - TE/2 - TM",
                     err, 1e-12))
    rows.append(_row("E9d", "evolution", "the same against 64 isochromats whose TM crusher is a phase that repeats in every repetition, relative",
                     "not equal: what is transverse in TM returns to z at the third pulse and comes back as signal one repetition later. The ideal spoiler of this package leaves that pathway out",
                     err_coherent, 0.01))
    return rows


# ---------------------------------------------------------------------------------------------- sequences
def _four_compartments(f, j_hz, te1_ms, te2_ms):
    """Signal of the A doublet of an AX pair after PRESS with two refocusing pulses of rectangular profile, per unit
    of the ideal signal: X is not refocused in the fraction f of each slice (Yablonskiy et al. 1998; Lange et al. 2006)."""
    c12, c1, c2 = (math.cos(np.pi * j_hz * 1e-3 * t) for t in (te1_ms + te2_ms, te1_ms, te2_ms))
    return (1 - f) ** 2 * c12 + f * (1 - f) * (c1 + c2) + f * f


def check_sequences():
    rows, rng = [], np.random.default_rng(SEED + 2)
    lac = metabolites.get("lactate").systems[0]
    lac = operators.SpinSystem("lactate", lac.shifts_ppm, lac.j_hz, lac.counts, lac.labels, ("ch3", "ch"))
    j = lac.j_hz[0][2]
    hz_ppm = operators.larmor_hz_per_ppm(3.0)
    d_hz = (lac.shifts_ppm[1] - lac.shifts_ppm[0]) * hz_ppm

    # S1 lactate at TE = 1 / J with ideal pulses
    te = 1e3 / j
    err = max(abs(sequences.simulate(lac, seq, coupling="weak").total("ch3") + 3) for seq in (sequences.spin_echo(te), sequences.press(te), sequences.press(te1_ms=30.0, te2_ms=te - 30.0)))
    rows.append(_row("S1", "sequences", f"lactate methyl doublet at TE = 1 / J = {te:.1f} ms, ideal pulses, weak coupling: spin echo, symmetric and asymmetric PRESS",
                     "-3: the doublet is inverted (cos(pi J TE) = -1)", err, 1e-10))
    strong = sequences.simulate(lac, sequences.press(te)).total("ch3")
    rows.append(_row("S1b", "sequences", "the same with strong coupling at 3 T (symmetric PRESS)", "-3 to within 1 % (J / shift difference = 0.02)", abs(strong + 3) / 3, 0.01))

    # S2 one slice-selective refocusing pulse of 874 Hz: the loss reported by Lange et al. 2006
    f = d_hz / 874.0
    sharp = pulses.slr(180.0, 16.0 / 0.874, 16.0, thickness_mm=20.0, kind="refocusing", n=732, d1=0.005, d2=0.005)
    ideal = sequences.simulate(lac, sequences.spin_echo(144.0)).total("ch3")
    got = {m: sequences.simulate(lac, sequences.spin_echo(144.0, refocus=sharp, carrier_ppm=lac.shifts_ppm[0], n_pos=64, model=m)).total("ch3") for m in evolve.MODELS}
    loss = 1 - (got["full"] / ideal).real
    rows.append(_row("S2", "sequences", f"lactate at 3 T, spin echo of 144 ms with one slice-selective refocusing pulse of 874 Hz (time-bandwidth 16, model full): relative loss of the doublet, {loss:.3f}",
                     f"2 x shift difference / bandwidth = 2 x {d_hz:.0f} / 874 = {2 * f:.3f} for a rectangular profile; Lange et al. 2006 (AJNR 27:895, appendix) state 81.2 % for 355 Hz and 874 Hz and measured 72.2 %",
                     abs(loss - 2 * f), 0.03))
    rows.append(_row("S2b", "sequences", "the same: model product against model full, relative to the ideal signal", "equal up to the coupling evolution during the 18 ms pulse",
                     abs(got["product"] - got["full"]) / abs(ideal), 0.03))
    soft = pulses.slr(180.0, 4.0 / 0.874, 4.0, thickness_mm=20.0, kind="refocusing", n=228)
    loss_soft = 1 - (sequences.simulate(lac, sequences.spin_echo(144.0, refocus=soft, carrier_ppm=lac.shifts_ppm[0], n_pos=64, model="full")).total("ch3") / ideal).real
    rows.append(_row("S2c", "sequences", f"the same loss with a 4.6 ms pulse of the same bandwidth (time-bandwidth 4: wide transition bands), {loss_soft:.3f}",
                     "the same 2 x shift difference / bandwidth: transition bands that are symmetric about the slice edge do not change the loss to first order. "
                     "The 72.2 % that Lange et al. measured is 9 points below their formula and below this simulation; the width of the transition bands does not explain it",
                     abs(loss_soft - 2 * f), 0.03))

    # S3 PRESS: the four compartments
    err = 0.0
    for tbw, bw, te1, te2 in ((12.0, 1200.0, 72.0, 72.0), (12.0, 1200.0, 30.0, 114.0), (12.0, 2000.0, 144.0, 144.0), (12.0, 1000.0, 40.0, 60.0)):
        p = pulses.slr(180.0, 1e3 * tbw / bw, tbw, thickness_mm=20.0, kind="refocusing", n=300, d1=0.005, d2=0.005)
        sim = sequences.simulate(lac, sequences.press(te1_ms=te1, te2_ms=te2, refocus=p, carrier_ppm=lac.shifts_ppm[0], n_pos=48, model="full")).total("ch3")
        err = max(err, abs(sim / 3 - _four_compartments(d_hz / bw, j, te1, te2)))
    rows.append(_row("S3", "sequences", "lactate at 3 T, PRESS with two slice-selective Shinnar-Le Roux refocusing pulses (time-bandwidth 12; 1.0, 1.2, 2.0 kHz; TE1 / TE2 = 72 / 72, 30 / 114, 144 / 144, 40 / 60 ms), "
                     "model full: methyl signal per proton", "four compartments with rectangular profiles: (1-f)^2 cos(pi J TE) + f (1-f) (cos(pi J TE1) + cos(pi J TE2)) + f^2, f = shift difference / bandwidth",
                     err, 0.015))

    # S4 chemical-shift displacement of the refocused slice
    err = 0.0
    p = pulses.slr(180.0, 6.0, 6.0, thickness_mm=10.0, kind="refocusing", n=300)
    pos = np.linspace(-15.0, 15.0, 1201)
    for off_ppm in (-3.0, -1.0, 2.0):
        one = operators.compile(operators.SpinSystem("s", (3.0 + off_ppm,), ()), 3.0, 3.0)
        u = evolve.pulse_propagator(one.sectors[0], one, p.b1_ut, p.dt_ms, p.g_mt_m, pos, 0.0, 0.0, "full")
        w = np.abs(u[:, 1, 0]) ** 2
        k = np.nonzero(w > 0.5)[0]
        edges = [pos[i] + (0.5 - w[i]) / (w[j] - w[i]) * (pos[j] - pos[i]) for i, j in ((k[0] - 1, k[0]), (k[-1] + 1, k[-1]))]
        err = max(err, abs(0.5 * sum(edges) / 10.0 + off_ppm * hz_ppm / p.bandwidth_hz))
    rows.append(_row("S4", "sequences", "centre of the refocused slice (midpoint of the half-height edges of |b|^2) of a site 3 ppm below, 1 ppm below and 2 ppm above the carrier, 1 kHz pulse at 3 T, in slice thicknesses",
                     "-(frequency offset) / bandwidth: a site at a higher frequency is selected at a more negative position (0.38 of the slice for 3 ppm at 127.7 MHz and 1 kHz; Oz et al. 2021 give 35 % per 3 ppm for 1.06 kHz, which is 123 MHz)", err, 1e-3))

    # S5 STEAM
    one = operators.SpinSystem("w", (4.2,), ())
    relax = {"w": (1200.0, 90.0)}
    got = sequences.simulate(one, sequences.steam(20.0, 30.0), relaxation=relax).total()
    rows.append(_row("S5", "sequences", "STEAM (TE 20 ms, TM 30 ms) of an uncoupled proton 64 Hz off resonance with T1 = 1200 ms, T2 = 90 ms", "(1/2) exp(-TE / T2) exp(-TM / T1): half of a spin echo",
                     abs(got - 0.5 * math.exp(-20.0 / 90.0) * math.exp(-30.0 / 1200.0)), 1e-12))
    ax = operators.SpinSystem("ax", (1.3, 4.1), ((0, 1, 7.0),), groups=("a", "x"))
    tms = np.linspace(5.0, 25.0, 41)
    sig = np.array([sequences.simulate(ax, sequences.steam(40.0, tm), coupling="weak").total("a") for tm in tms])
    zq = (1.3 - 4.1) * hz_ppm
    design = np.stack([np.ones_like(tms), np.cos(2e-3 * np.pi * zq * tms), np.sin(2e-3 * np.pi * zq * tms)], axis=1)
    coef = np.linalg.lstsq(design, sig, rcond=None)[0]
    rows.append(_row("S5b", "sequences", "STEAM (TE 40 ms) of an AX pair (weak coupling, J = 7 Hz, 2.8 ppm apart at 3 T), 41 mixing times from 5 to 25 ms: signal of A",
                     f"a constant plus one oscillation at the zero-quantum frequency nu_A - nu_X = {zq:.1f} Hz (residual of that fit)", np.abs(design @ coef - sig).max(), 1e-10))
    eps = 2e-3 * np.pi * zq * 20.0
    c0 = 0.5 * math.cos(np.pi * 7e-3 * 20.0) ** 2
    c1 = 0.25 * math.sin(np.pi * 7e-3 * 20.0) ** 2 * (-(1 - math.cos(eps)) + 1j * math.sin(eps))
    rows.append(_row("S5c", "sequences", "the same: the constant and the coefficients of the cosine and the sine",
                     "(1/2) cos^2(t) and (1/4) sin^2(t) (-(1 - cos e) + i sin e), 0, with t = pi J TE/2 and e = 2 pi (nu_A - nu_X) TE/2 (product operators; derivation in docs/physics/spins.md)",
                     max(abs(coef[0] - c0), abs(coef[1] - c1), abs(coef[2])), 1e-10))
    rows.append(_row("S5d", "sequences", "the same: size of the oscillation relative to the constant at TE = 40 ms", "not zero: the mixing time changes the signal of a coupled spin",
                     abs(coef[1] + 1j * coef[2]) / abs(coef[0]), 0.05, kind="min"))
    s3 = operators.SpinSystem("abc", (2.1, 2.35, 3.75), ((0, 1, -14.8), (0, 2, 7.3), (1, 2, 4.7)))
    c = operators.compile(s3, 3.0, 4.7)
    br = _Brute(c.freq_hz, s3.j_hz)
    ref = np.zeros(6, complex)
    for ka in range(8):
        for kb in range(8):
            za, zb = br.z(2 * np.pi * ka / 8), br.z(2 * np.pi * 3 * kb / 8)
            rho = br.fz.astype(complex)
            for u in (br.pulse(np.pi / 2), br.free(15.0), za, br.pulse(np.pi / 2), zb, br.free(13.0), br.pulse(np.pi / 2, np.pi), za, br.free(15.0)):
                rho = _apply(u, rho)
            for k in range(6):
                ref[k] += br.signal(rho) / 64
                rho = _apply(br.free(1.1), rho)
    rows.append(_row("S5e", "sequences", "STEAM (TE 30 ms, TM 13 ms) of three strongly coupled protons: free induction decay",
                     "64 isochromats with z rotations for the crushers of TE and TM, Pauli matrices and `scipy.linalg.expm`",
                     np.abs(sequences.simulate(s3, sequences.steam(30.0, 13.0)).fid(6, 1.1) - ref).max(), 1e-11))

    # S6 semi-LASER with hyperbolic-secant pairs
    afp = pulses.sech(25.0, 8.0, 1.3, 4.83, thickness_mm=20.0, n=400)                # 2.0 kHz; gamma B1 / beta = 5.1 > mu
    one = operators.SpinSystem("s", (lac.shifts_ppm[0],), ())
    seq = sequences.slaser(144.0, afp=afp, carrier_ppm=lac.shifts_ppm[0], n_pos=64, extent=2.0)
    s1 = sequences.simulate(one, seq).total()
    fine = np.linspace(-30.0, 30.0, 6001)
    b = rotation.rotate(afp.b1_ut, afp.dt_ms, afp.g_mt_m, fine)[1]
    pair = (np.abs(b) ** 4).sum() * (fine[1] - fine[0]) / 20.0
    rows.append(_row("S6", "sequences", f"semi-LASER (TE 144 ms) with two pairs of hyperbolic-secant pulses ({afp.bandwidth_hz:.0f} Hz, 8 ms, 25 uT), 64 positions per axis, on a singlet at the carrier: signal per proton, {s1.real:.3f}",
                     "(integral of |b|^4 over the slice / thickness)^2 with b from `mrgym.design.rotation.rotate` on 6,001 positions: the phase that one adiabatic pulse leaves is removed by the second; "
                     "the signal is below 1 because the profile of this pulse has wide transitions", abs(s1 - pair ** 2), 2e-3))
    sl = {m: sequences.simulate(lac, sequences.slaser(144.0, afp=afp, carrier_ppm=lac.shifts_ppm[0], n_pos=64, extent=2.0, model=m)).total("ch3") / 3 for m in evolve.MODELS}
    rows.append(_row("S6b", "sequences", "lactate methyl under that sequence: model product against model full, per proton", "equal up to the coupling evolution during the four 8 ms pulses", abs(sl["product"] - sl["full"]), 0.03))
    f_afp = d_hz / afp.bandwidth_hz
    kept = (sl["full"] / s1).real
    rows.append(_row("S6c", "sequences", f"the same (model full) relative to the singlet: {kept:.3f}; PRESS with 1.2 kHz pulses keeps {_four_compartments(d_hz / 1200.0, j, 72.0, 72.0):.3f}",
                     f"compartments with rectangular profiles, two pairs: -(1 - 2f) = {-(1 - 2 * f_afp):.3f} with f = shift difference / bandwidth = {f_afp:.3f}; the profile of the pulse is not rectangular, hence the tolerance",
                     abs(kept + (1 - 2 * f_afp)), 0.06))

    # S7 MEGA editing
    te = 1e3 / (2 * 7.35)
    toy = operators.SpinSystem("a2m2", (3.01, 1.89), ((0, 1, 7.35),), counts=(2, 2), groups=("h4", "h3"))
    on = sequences.simulate(toy, sequences.mega_press(te, 1.9), coupling="weak").total("h4")
    off = sequences.simulate(toy, sequences.mega_press(te, None), coupling="weak").total("h4")
    rows.append(_row("S7", "sequences", "MEGA-PRESS of an A2M2 system (weak coupling, J = 7.35 Hz) at TE = 1 / (2J) = 68 ms, ideal inversion of M: signal of A with and without the editing pulses, and (on - off) / 2 per proton",
                     "on: 2 (triplet refocused); off: 0 (outer lines inverted, centre line unchanged); editing efficiency 1/2, the value assumed for GABA (Mullins et al. 2014; Harris et al. 2015)",
                     max(abs(on - 2), abs(off), abs((on - off) / 4 - 0.5)), 1e-10))
    gaba = metabolites.get("gaba").systems[0]
    lo, hi = (3.0128 - 0.35) * hz_ppm - 4.7 * hz_ppm, (3.0128 + 0.35) * hz_ppm - 4.7 * hz_ppm
    edit = sequences.gaussian(180.0, 14.0)
    res = {}
    for label, args in (("ideal", {}), ("gauss", dict(edit=edit))):
        on = sequences.simulate(gaba, sequences.mega_press(68.0, 1.9, **args)).window(lo, hi)
        off = sequences.simulate(gaba, sequences.mega_press(68.0, 7.46 if args else None, **args)).window(lo, hi)
        res[label] = (on, off, (on.total() - off.total()).real / 4)
    rows.append(_row("S7b", "sequences", f"GABA (the basis entry) at 3 T, MEGA-PRESS TE 68 ms, editing at 1.9 ppm: (on - off) / 2 of the 3.01 ppm multiplet per proton, with an ideal inversion ({res['ideal'][2]:.3f}) "
                     f"and with a 14 ms Gaussian of {edit.bandwidth_hz:.0f} Hz, off at 7.46 ppm ({res['gauss'][2]:.3f})",
                     "0.5 (the efficiency assumed in quantification: Mullins et al. 2014, eq. 1; Harris et al. 2015); the multiplet is not an ideal triplet, so the value is near but not at 0.5",
                     max(abs(res["ideal"][2] - 0.5), abs(res["gauss"][2] - 0.5)), 0.03))
    on, off, _ = res["ideal"]
    fgrid = np.linspace(lo, hi, 2001)
    diff = (evolve.lorentzian_spectrum(on, fgrid, lorentz_hz=2.0) - evolve.lorentzian_spectrum(off, fgrid, lorentz_hz=2.0)).real
    centre = (3.0128 - 4.7) * hz_ppm
    left, right = fgrid < centre, fgrid >= centre
    k1, k2 = np.argmax(np.where(left, diff, -np.inf)), np.argmax(np.where(right, diff, -np.inf))
    dip = diff[np.argmin(np.abs(fgrid - centre))] / min(diff[k1], diff[k2])
    rows.append(_row("S7c", "sequences", f"shape of that difference spectrum (ideal inversion, 2 Hz lines): two maxima {fgrid[k2] - fgrid[k1]:.1f} Hz apart, the value at the centre relative to them",
                     "a pseudo-doublet: the outer lines of the multiplet are edited, the centre is low but not absent (Near et al. 2013); distance of the maxima between 12 and 18 Hz (the sums of the two vicinal couplings are 12.5 and 17.6 Hz)",
                     max(dip - 0.6, 12.0 - (fgrid[k2] - fgrid[k1]), (fgrid[k2] - fgrid[k1]) - 18.0), 0.0))
    on = sequences.simulate(lac, sequences.mega_press(1e3 / j, 4.1, te1_ms=30.0), coupling="weak").total("ch3")
    off = sequences.simulate(lac, sequences.mega_press(1e3 / j, None, te1_ms=30.0), coupling="weak").total("ch3")
    rows.append(_row("S7d", "sequences", "lactate, MEGA-PRESS at TE = 1 / J with an ideal inversion of the methine proton at 4.1 ppm (weak coupling): methyl signal on and off",
                     "+3 and -3: the whole doublet is in the difference (efficiency 1)", max(abs(on - 3), abs(off + 3)), 1e-10))

    # S8 two-dimensional experiments
    ax = operators.SpinSystem("ax", (1.3, 4.1), ((0, 1, 7.0),))
    n1, d1, n2, d2 = 16, 8.0, 32, 0.6
    mat = sequences.simulate_2d(ax, sequences.jres(20.0, n1, d1), n2, d2, coupling="weak")
    t1 = 20.0 + d1 * np.arange(n1)[:, None]
    t2 = d2 * np.arange(n2)[None, :]
    ref = sum(0.5 * np.exp(2e-3j * np.pi * ((nu + s * 3.5) * t2 + s * 3.5 * t1)) for nu in operators.frequencies_hz(ax.shifts_ppm, 3.0, 4.7) for s in (-1, 1))
    rows.append(_row("S8", "sequences", "2D J-resolved signal of an AX pair (weak coupling), 16 echo times x 32 samples", "sum over the four lines of (1/2) exp(2 pi i (nu +- J/2) t2) exp(+- i pi J t1): a line at nu +- J/2 in F2 sits at +- J/2 in F1",
                     np.abs(mat - ref).max(), 1e-11))
    s2 = operators.SpinSystem("ab", (2.0, 3.2), ((0, 1, 9.0),))
    c = operators.compile(s2, 3.0, 4.7)
    br = _Brute(c.freq_hz, s2.j_hz)
    n1, d1 = 6, 1.7
    mat = sequences.simulate_2d(s2, sequences.cosy(n1, d1), 8, 0.9)
    ref = np.zeros((n1, 8), complex)
    for k1 in range(n1):
        for ka in range(8):
            za = br.z(2 * np.pi * ka / 8)
            rho = br.fz.astype(complex)
            for u in (br.pulse(np.pi / 2), br.free(k1 * d1), za, br.pulse(np.pi / 2), za):
                rho = _apply(u, rho)
            for k in range(8):
                ref[k1, k] += br.signal(rho) / 8
                rho = _apply(br.free(0.9), rho)
    rows.append(_row("S8b", "sequences", "gradient-selected COSY of two strongly coupled protons, 6 x 8 samples", "8 isochromats with equal crushers around the second pulse, Pauli matrices and `scipy.linalg.expm`",
                     np.abs(mat - ref).max(), 1e-11))
    s0 = operators.SpinSystem("u", (2.0, 3.2), ())
    t1, t2 = 1.7 * np.arange(6)[:, None], 0.9 * np.arange(8)[None, :]
    ref = sum(-0.5 * np.exp(2e-3j * np.pi * nu * (t2 - t1)) for nu in operators.frequencies_hz(s0.shifts_ppm, 3.0, 4.7))
    rows.append(_row("S8c", "sequences", "the same experiment on two uncoupled protons", "-(1/2) sum over the sites of exp(2 pi i nu (t2 - t1)): diagonal peaks only, no cross peak without coupling",
                     np.abs(sequences.simulate_2d(s0, sequences.cosy(6, 1.7), 8, 0.9) - ref).max(), 1e-12))
    cross = []
    for system in (s2, s0):
        fa, fb = operators.frequencies_hz(system.shifts_ppm, 3.0, 4.7)
        m = sequences.simulate_2d(system, sequences.cosy(64, 2.0), 64, 2.0, t2_ms=60.0) * np.exp(-2.0 * np.arange(64)[:, None] / 60.0)
        spec = np.abs(np.fft.fft2(m))
        f = np.fft.fftfreq(64, 2e-3)
        ka, kb, kd = (int(np.argmin(np.abs((f - x + 250.0) % 500.0 - 250.0))) for x in (-fa, fb, fa))    # aliased bins
        cross.append(spec[max(ka - 2, 0):ka + 3, max(kb - 2, 0):kb + 3].max() / spec[max(ka - 2, 0):ka + 3, max(kd - 2, 0):kd + 3].max())
    rows.append(_row("S8d", "sequences", f"magnitude of the 2D spectrum (64 x 64, 60 ms decay) at (F1 of A, F2 of B) relative to the diagonal peak of A: {cross[0]:.2f} with J = 9 Hz, {cross[1]:.2f} without (leakage of the diagonal); the difference",
                     "a cross peak appears with the coupling: at least 0.3", cross[0] - cross[1], 0.3, kind="min"))
    _ = rng
    return rows


# ---------------------------------------------------------------------------------------------- basis
def _timed(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def press_basis(refocus=None, n=2048, dwell_ms=0.5, te_ms=30.0, b0_t=3.0, names=None, model="product", n_pos=32):
    """Free induction decays [M, n] of the default basis under PRESS; the acceptance timing uses this function."""
    seq = sequences.press(te_ms, refocus=refocus, carrier_ppm=3.0 if refocus is not None else None, n_pos=n_pos, model=model)
    return np.array([sequences.simulate(m, seq, b0_t=b0_t).fid(n, dwell_ms, 200.0) for m in metabolites.basis(names)])


def check_basis():
    from ..scanner import console, signal, tissue
    rows = []
    rows.append(_row("M1", "basis", "the six peaks of the entry `triglyceride` and the gyromagnetic ratio", "`mrgym.scanner.tissue.FAT_PEAKS`, `console.GAMMA_HZ_PER_T`",
                     float(metabolites.ENGINE_FAT_PEAKS != tissue.FAT_PEAKS) + abs(GAMMA_HZ_PER_T - console.GAMMA_HZ_PER_T), 0))
    err = 0.0
    t_ms = 0.37 * np.arange(64)
    for b0 in (1.5, 3.0):
        fid = sequences.simulate(metabolites.get("triglyceride"), sequences.pulse_acquire(), b0_t=b0, ref_ppm=metabolites.WATER_PPM).fid(64, 0.37)
        err = max(err, np.abs(fid - signal.fat_spectrum(1e-3 * t_ms, b0, tissue.FAT_PEAKS)).max())
    rows.append(_row("M1b", "basis", "free induction decay of `triglyceride` after a 90 degree pulse at 1.5 and 3 T, reference on water", "the engine's `scanner/signal.py::fat_spectrum`", err, 1e-12))
    err_n, err_c = 0.0, 0.0
    for m in metabolites.BASIS:
        lines = sequences.simulate(m, sequences.pulse_acquire())
        err_n = max(err_n, abs(lines.total() - m.protons))
        centre = sum(s.scale * c * f for s in m.systems for c, f in zip(s.counts, operators.frequencies_hz(s.shifts_ppm, 3.0, 4.7)))
        err_c = max(err_c, abs((lines.amp * lines.freq_hz).sum() - centre))
    rows.append(_row("M2", "basis", f"all {len(metabolites.BASIS)} entries after a 90 degree pulse at 3 T: sum of the line amplitudes", "the number of protons of the entry", err_n, 1e-10))
    rows.append(_row("M2b", "basis", "the same: first moment of the spectrum, sum of amplitude x frequency", "sum of the site frequencies (a coupling moves no centre of gravity: Tr(F- [H_J, F+]) = 0)", err_c, 1e-8, unit="Hz"))
    err = 0.0
    for name in ("lactate", "alanine", "lipid_pair"):
        m = metabolites.get(name)
        seq = sequences.press(35.0, te1_ms=12.0)
        a = sequences.simulate(m, seq).fid(128, 0.8)
        b = sequences.simulate(m, seq, composite=False).fid(128, 0.8)
        err = max(err, np.abs(a - b).max())
    rows.append(_row("M3", "basis", "lactate, alanine and the lipid pair under PRESS: methyl and methylene groups as total spins", "every proton as its own site", err, 1e-11))
    t_ideal = _timed(lambda: press_basis())
    rows.append(_row("M4", "basis", f"time of the PRESS basis (TE 30 ms, {len(metabolites.DEFAULT)} entries, 2,048 points, 3 T), ideal refocusing pulses, one core (best of 3)", "requirement", t_ideal, 2.0, unit="s"))
    p = pulses.slr(180.0, 5.0, 6.0, thickness_mm=20.0, kind="refocusing", n=250)
    t_shaped = _timed(lambda: press_basis(p))
    rows.append(_row("M4b", "basis", "the same with slice-selective refocusing pulses (1.2 kHz, 32 positions per pulse, model product)", "requirement", t_shaped, 2.0, unit="s"))
    rng = np.random.default_rng(SEED + 4)
    seven = operators.SpinSystem("seven", tuple(rng.uniform(1.0, 4.4, 7)), tuple((i, k, rng.uniform(2.0, 9.0)) for i in range(7) for k in range(i + 1, 7)))
    for label, seq, limit, rid in (("ideal refocusing pulses", sequences.press(30.0), 2.0, "M4c"),
                                   ("the slice-selective pulses of M4b, model product", sequences.press(30.0, refocus=p, carrier_ppm=3.0), 10.0, "M4d")):
        t = _timed(lambda seq=seq: sequences.simulate(seven, seq).fid(2048, 0.5, 200.0), repeat=1)
        rows.append(_row(rid, "basis", f"time of one network of seven strongly coupled protons (matrices of 128, about 3,000 lines) under PRESS, 2,048 points: {label}", "requirement", t, limit, unit="s"))
    return rows


# ---------------------------------------------------------------------------------------------- quantify
def check_quantify():
    rows, rng = [], np.random.default_rng(SEED + 3)
    names = ("naa", "creatine", "choline", "glutamate", "myo_inositol", "lactate")
    n, dwell = 1024, 0.5
    basis = press_basis(None, n, dwell, names=names)
    truth = np.array([12.0, 8.0, 2.0, 9.0, 6.0, 1.5])
    clean = quantify.model(basis, dwell, truth, 2.3, 1.7, 0.4)

    # Q1 no noise
    fit = quantify.fit(clean, basis, dwell)
    err = max(np.abs(fit.amplitudes - truth).max(), abs(fit.shift_hz - 2.3), abs(fit.width_hz - 1.7), abs(fit.phase_rad - 0.4))
    rows.append(_row("Q1", "quantify", "fit of a noise-free mixture of six basis spectra (PRESS, 3 T) with a shift of 2.3 Hz, 1.7 Hz of extra width and a phase of 0.4 rad", "the amplitudes, shift, width and phase that were put in", err, 1e-6))
    fb = quantify.fit(clean, basis, dwell, band_hz=(-500.0, -60.0))
    rows.append(_row("Q1b", "quantify", "the same, fitted in the frequency domain between 0.8 and 4.2 ppm", "the amplitudes that were put in", np.abs(fb.amplitudes - truth).max(), 1e-6))

    # Q2 Cramer-Rao bounds against a numerical Fisher matrix
    sigma = 0.6
    bound = quantify.crb(basis, dwell, truth, sigma, 2.3, 1.7, 0.4)
    p0 = np.concatenate([truth, [2.3, 1.7, 0.4]])
    jac = []
    for k in range(p0.size):
        h = 1e-5 * max(1.0, abs(p0[k]))
        hi, lo = p0.copy(), p0.copy()
        hi[k] += h
        lo[k] -= h
        jac.append((quantify.model(basis, dwell, hi[:6], *hi[6:]) - quantify.model(basis, dwell, lo[:6], *lo[6:])) / (2 * h))
    jac = np.array(jac)
    num = np.sqrt(np.diag(np.linalg.inv((jac.conj() @ jac.T).real / sigma ** 2)))
    ana = np.concatenate([bound["amplitudes"], [bound["shift_hz"], bound["width_hz"], bound["phase_rad"]]])
    rows.append(_row("Q2", "quantify", "Cramer-Rao bounds of the nine parameters of that fit", "inverse Fisher matrix from central differences of the model", np.abs(ana / num - 1).max(), 1e-6, unit="relative"))

    # Q3 Monte Carlo
    trials = 300
    est = np.array([quantify.fit(quantify.add_noise(clean, sigma, rng), basis, dwell, start=(2.0, 1.5, 0.3)).amplitudes for _ in range(trials)])
    sd, bias = est.std(axis=0, ddof=1), est.mean(axis=0) - truth
    snr_naa = truth[0] * np.abs(basis[0][0]) / sigma
    rows.append(_row("Q3", "quantify", f"{trials} noisy copies of that mixture (seeded; noise {sigma} per sample, first sample of NAA / noise = {snr_naa:.0f}): standard deviation of every fitted amplitude relative to its "
                     f"Cramer-Rao bound (the bounds are {100 * (bound['amplitudes'] / truth).min():.1f} to {100 * (bound['amplitudes'] / truth).max():.1f} % of the amplitudes)",
                     "1 (a least-squares fit attains the bound at this noise); the tolerance is 3.7 standard errors of a standard deviation from 300 trials", np.abs(sd / bound["amplitudes"] - 1).max(), 0.15))
    rows.append(_row("Q3b", "quantify", "the same: bias of every amplitude in units of its bound", "0 within 3.5 standard errors of the mean (0.2 bounds)", np.abs(bias / bound["amplitudes"]).max(), 0.2))

    # Q4 water-referenced concentration
    comp = {k: dict(volume=v, **metabolites.WATER_3T[k]) for k, v in (("gm", 0.55), ("wm", 0.35), ("csf", 0.10))}
    te, tr, conc, t1m, t2m = 30.0, 2000.0, 10.3, 1350.0, 160.0
    cr = metabolites.get("creatine")
    seq = sequences.press(te)
    water_g = {k: c["volume"] * c["water"] for k, c in comp.items()}                 # kg of water per litre of voxel, by compartment
    rm = math.exp(-te / t2m) * (1 - math.exp(-tr / t1m))
    s_met = 3 * conc * (water_g["gm"] + water_g["wm"]) * rm * sequences.simulate(cr, seq).total("cr_ch3").real / 3
    s_wat = sum(2 * 55510.0 * water_g[k] * math.exp(-te / c["t2_ms"]) * (1 - math.exp(-tr / c["t1_ms"])) for k, c in comp.items())
    s_wat *= sequences.simulate(metabolites.get("water"), seq).total().real / 2
    back = quantify.water_referenced(s_met, s_wat, 3, te, tr, t1m, t2m, comp)
    rows.append(_row("Q4", "quantify", "water-referenced concentration of creatine (methyl singlet, PRESS TE 30 ms, TR 2 s) in a voxel of 55 % grey matter, 35 % white matter, 10 % CSF. The signals are counted "
                     "by hand: protons x mmol per kg of water x kg of water in the compartment x exp(-TE/T2) (1 - exp(-TR/T1)), water at 55,510 mmol/kg in all three compartments, creatine in the tissue only",
                     "the molal concentration that was put in, 10.3 mmol per kg of tissue water (formula of Near et al. 2021, eqs. 4 and 5; water content and relaxation times as quoted by Harris et al. 2015)",
                     abs(back / conc - 1), 1e-12, unit="relative"))
    s_met2 = s_met / rm * math.exp(-te / t2m) * sequences.recovery(tr, t1m, seq.inversions_ms)
    s_wat2 = sum(2 * 55510.0 * water_g[k] * math.exp(-te / c["t2_ms"]) * sequences.recovery(tr, c["t1_ms"], seq.inversions_ms) for k, c in comp.items())
    rows.append(_row("Q4b", "quantify", "the same when the signals are built with the recovery of a PRESS repetition (two inversions after the excitation) and the formula corrects with 1 - exp(-TR / T1)",
                     "an error of the order of TE / TR, stated here so that a task can account for it", abs(quantify.water_referenced(s_met2, s_wat2, 3, te, tr, t1m, t2m, comp) / conc - 1), 0.01, unit="relative"))

    # Q5 line width and signal-to-noise ratio as the consensus defines them
    t2, amp, dwell, n, sigma = 60.0, 3.0, 0.5, 4096, 0.02
    df = 1e3 / (n * dwell)
    one = evolve.Lines(np.array([-80 * df]), np.array([amp + 0j]), np.array([0]), ("g",))
    f, spec = evolve.spectrum(quantify.add_noise(one.fid(n, dwell, t2), sigma, rng), dwell)
    predicted = amp * 1e-3 * t2 / quantify.spectral_noise_sd(sigma, n, dwell)
    got = quantify.snr(f, spec.real, (-80 * df - 20, -80 * df + 20), (300.0, 900.0))
    rows.append(_row("Q5", "quantify", f"signal-to-noise ratio of a singlet in the frequency domain (peak height / standard deviation of the noise, Kreis et al. 2020), measured {got:.0f}",
                     f"A T2 / (sigma x dwell x sqrt(n - 3/4)) = {predicted:.0f}; the tolerance covers the noise on the peak and on the estimate of the noise", abs(got / predicted - 1), 0.05, unit="relative"))
    f, spec = evolve.spectrum(one.fid(n, dwell, t2, lorentz_hz=3.0), dwell, zero_fill=4)
    rows.append(_row("Q5b", "quantify", "full width at half maximum of that singlet with 3 Hz of extra Lorentzian width, measured on the real spectrum (4 x zero filling)", "1 / (pi T2) + 3 Hz = 8.31 Hz",
                     abs(quantify.linewidth_hz(f, spec.real, (-80 * df - 20, -80 * df + 20)) - (1e3 / (np.pi * t2) + 3.0)), 0.02, unit="Hz"))
    return rows


CHECKS = dict(operators=check_operators, evolution=check_evolution, sequences=check_sequences, basis=check_basis, quantify=check_quantify)


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
    """The validation table of docs/physics/spins.md. Errors are printed with one significant digit below 1e-3:
    they are rounding noise and differ between machines."""
    out = []
    for group, title in GROUPS.items():
        part = [r for r in rows if r["group"] == group]
        if not part:
            continue
        out += [f"### {title}", "", "| # | Quantity | Closed form or independent method | Largest error | Tolerance | |", "|---|---|---|---|---|---|"]
        for r in part:
            sign = "≤" if r["kind"] == "max" else "≥"
            unit = f" {r['unit']}" if r["unit"] else ""
            out.append(f"| {r['id']} | {r['quantity']} | {r['reference']} | {_number(r['measured'])}{unit} | {sign} {_number(r['tolerance'])} | "
                       f"{'pass' if r['ok'] else 'FAIL'} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rows, seconds = run([a for a in argv if a in CHECKS] or None)
    if "--markdown" in argv:
        print(markdown_table(rows))
    else:
        for r in rows:
            sign = "<=" if r["kind"] == "max" else ">="
            print(f"{r['id']:5s} {'pass' if r['ok'] else 'FAIL'}  {_number(r['measured']):>9s} {sign} {_number(r['tolerance']):<8s} {r['unit']:14s} {r['quantity'][:120]}")
        print(f"{sum(r['ok'] for r in rows)} of {len(rows)} rows pass; {seconds:.1f} s")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
