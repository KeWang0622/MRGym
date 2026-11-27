"""Validation of the design layer: every function against a closed form or an independent computation.

    python -m mrgym.design.validate                 the table as text; exit status 1 if a row fails
    python -m mrgym.design.validate --markdown      the table of docs/physics/design.md

A row has a quantity, the reference it is compared with, the largest error found and a tolerance. "Independent"
means a computation that shares no code with the function under test: real 3 x 3 rotation matrices against the spinor
code, ``scipy.integrate.solve_ivp`` against the operator splitting, an isochromat ensemble against the phase graph,
numerical quadrature against the exact segment integrals, the engine (``mrgym.scanner``) against this layer. The
references were written by the same authors: the rows show that the code computes the stated equations, not that
the equations describe a scanner.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import scipy.integrate

from . import gradients, profile, pulses, rotation, sequence
from .units import (
    B_PER_MT_M_MS_SQUARED_MS,
    CYCLES_PER_MM_PER_MT_M_MS,
    GAMMA_HZ_PER_T,
    RAD_PER_HZ_MS,
    RAD_PER_MT_M_MM_MS,
    RAD_PER_UT_MS,
)

GROUPS = dict(rotation="RF rotation", pulses="Pulse generators", profile="Profile figures and bounds",
              sequence="Sequences (extended phase graph)", gradients="Gradient programs and limits")


def _row(id, group, quantity, reference, measured, tolerance, kind="max", unit=""):
    """kind "max": passes when measured <= tolerance; "min": when measured >= tolerance."""
    measured = float(measured)
    ok = bool(np.isfinite(measured) and (measured <= tolerance if kind == "max" else measured >= tolerance))
    return dict(id=id, group=group, quantity=quantity, reference=reference, measured=measured, tolerance=float(tolerance), kind=kind,
                unit=unit, ok=ok)


def _rodrigues(axis, angle):
    """Right-handed rotation matrix about ``axis`` by ``angle``: written without the spinor code."""
    n = np.asarray(axis, float)
    n = n / np.linalg.norm(n)
    k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _order(errors):
    """Convergence order from errors at successively halved steps: the mean of log2 of the ratios."""
    return float(np.mean([math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]))


# ---------------------------------------------------------------------------------------------- rotation
def check_rotation():
    from ..scanner import console, rf
    rows, rng = [], np.random.default_rng(20261126)

    # D1 one hard sub-pulse between two halves of the precession
    th, ph, df = rng.uniform(0, 2 * np.pi, 300), rng.uniform(-np.pi, np.pi, 300), rng.uniform(-2000, 2000, 300)
    dt = 0.1
    err = 0.0
    for t, p, f in zip(th, ph, df):
        a, b = rotation.rotate([t / (RAD_PER_UT_MS * dt) * np.exp(1j * p)], dt, None, None, f)
        psi = RAD_PER_HZ_MS * f * dt
        err = max(err, abs(a - math.cos(t / 2) * np.exp(-0.5j * psi)), abs(b + 1j * np.exp(1j * p) * math.sin(t / 2)))
        mxy, mz = rotation.apply(a, b)
        err = max(err, abs(mxy + 1j * np.exp(1j * (p + psi / 2)) * math.sin(t)), abs(mz - math.cos(t)))
    rows.append(_row("D1", "rotation", "hard pulse: (a, b), Mxy, Mz for 300 random flip angles (0 to 360 deg), phases and off-resonances",
                     "a = cos(t/2) exp(-i psi/2), b = -i exp(i p) sin(t/2); Mxy = -i exp(i (p + psi/2)) sin t, Mz = cos t", err, 1e-13))

    # D2 rectangular pulse off resonance: rotation about the effective field
    err, n, dur = 0.0, 100, 1.0
    cases = [(rng.uniform(0.5, 25.0), rng.uniform(-np.pi, np.pi), rng.uniform(-3000, 3000), rng.uniform(0.5, 1.5)) for _ in range(200)]
    for amp, p, f, sc in cases:
        a, b = rotation.rotate(np.full(n, amp * np.exp(1j * p)), dur / n, None, None, f, sc, model="piecewise")
        w1, dw = RAD_PER_UT_MS * amp * sc * dur, RAD_PER_HZ_MS * f * dur
        w = math.hypot(w1, dw)
        a0, b0 = math.cos(w / 2) - 1j * dw / w * math.sin(w / 2), -1j * np.exp(1j * p) * w1 / w * math.sin(w / 2)
        err = max(err, abs(a - a0), abs(b - b0), abs((abs(a) ** 2 - abs(b) ** 2) - (1 - 2 * (w1 / w) ** 2 * math.sin(w / 2) ** 2)))
    rows.append(_row("D2", "rotation", "rectangular pulse off resonance (model piecewise), 200 random amplitudes, offsets and B1 scales",
                     "one rotation by |w_eff| T about the effective field (w1 cos p, w1 sin p, dw); Mz = 1 - 2 (w1/w_eff)^2 sin^2(w_eff T/2)", err, 1e-12))

    # D3 the split model converges to it as dt^2
    amp, p, f = 14.0, 0.7, 900.0
    w1, dw = RAD_PER_UT_MS * amp * dur, RAD_PER_HZ_MS * f * dur
    w = math.hypot(w1, dw)
    a0, b0 = math.cos(w / 2) - 1j * dw / w * math.sin(w / 2), -1j * np.exp(1j * p) * w1 / w * math.sin(w / 2)
    errs = []
    for n in (50, 100, 200, 400, 800):
        a, b = rotation.rotate(np.full(n, amp * np.exp(1j * p)), dur / n, None, None, f)
        errs.append(max(abs(a - a0), abs(b - b0)))
    rows.append(_row("D3", "rotation", "the same pulse in the model split (hard sub-pulses): order of convergence with the raster, 50 to 800 samples",
                     "second order: the error falls by 4 when the raster is halved", abs(_order(errs) - 2.0), 0.02, unit="in the order"))
    rows.append(_row("D3b", "rotation", "error of the model split for that pulse at 100 samples (10 us raster)", "the effective-field closed form",
                     errs[1], 1e-3))

    # D4 the engine's cayley_klein for every console pulse
    err = 0.0
    for dur, tbw in rf.PULSES.values():
        for window in rf.WINDOWS:
            for alpha_deg in (5.0, 30.0, 90.0, 137.0, 180.0):
                for rephase, width in ((0.5, 1.0), (0.0, rf.REFOCUS_WIDTH)):
                    zeta = np.linspace(-4, 4, 321)
                    ae, be = rf.cayley_klein(math.radians(alpha_deg), tbw, zeta / width, rephase=rephase, window=window)
                    thick = 5.0 * width
                    pulse = pulses.sinc(alpha_deg, dur * width, tbw, thick, window, "excitation" if rephase else "refocusing", n=rf.N_SAMPLES)
                    a, b = pulse.rotate(zeta * 5.0)
                    err = max(err, np.abs(a - ae).max(), np.abs(b - be).max())
    rows.append(_row("D4", "rotation", "agreement with the engine: (a, b) of `scanner/rf.py::cayley_klein` for the pulses fast / normal / sharp, Hann and "
                     "Hamming, 5 to 180 deg, as excitation (with the rephasing lobe) and as the refocusing pulse on the 1.5 x slice",
                     "the engine", err, 1e-12))

    # D5 what the console computes from the pulse: peak B1, energy, slice gradient, RF power
    err = 0.0
    for dur, tbw in rf.PULSES.values():
        for flip in (10.0, 90.0, 180.0):
            pulse = pulses.sinc(flip, dur, tbw, 5.0, n=rf.N_SAMPLES)
            hard = console._b1_ut(flip, dur)
            err = max(err, abs(pulse.peak_b1_ut / (rf.peak_factor(tbw) * hard) - 1), abs(pulse.energy_ut2_ms / (hard ** 2 * dur * rf.energy_factor(tbw)) - 1),
                      abs(pulse.g_mt_m[0] / (1e3 * tbw / (dur * 1e-3 * console.GAMMA_HZ_PER_T * 5e-3)) - 1))
    rows.append(_row("D5", "rotation", "peak B1, energy and slice gradient of the three console pulses", "`rf.peak_factor`, `rf.energy_factor`, "
                     "`console._b1_ut`, G = tbw / (duration gamma thickness)", err, 1e-12, unit="relative"))
    plan = console.plan(dict(sequence="spgr", flip_deg=30.0, tr_ms=20.0, slice_mm=5.0))
    dur, tbw = plan["realised"]["rf_ms"], rf.PULSES[plan["realised"]["rf_pulse"]][1]
    pulse = pulses.sinc(30.0, dur, tbw, 5.0, n=rf.N_SAMPLES)
    power = 100.0 * pulse.b1rms_ut(plan["timing"]["cycle_ms"]) ** 2 / 3.0 ** 2
    rows.append(_row("D5b", "rotation", "RF power figure and slice gradient of a console protocol (spgr, 30 deg, TR 20 ms, 5 mm)",
                     "`console.plan`: 100 B1rms^2 / (3 uT)^2 printed to 0.01 %, gradient printed to 0.001 mT/m",
                     max(abs(power - plan["limits"]["rf_power_percent"]) / 0.005, abs(pulse.g_mt_m[0] - plan["timing"]["slice_gradient_mt_m"]) / 0.0005),
                     1.0 + 1e-6, unit="of the printed rounding"))
    rows.append(_row("D5c", "rotation", "gyromagnetic ratio of this layer", "`console.GAMMA_HZ_PER_T`", abs(GAMMA_HZ_PER_T - console.GAMMA_HZ_PER_T), 0.0))

    # D6 small-tip Fourier relation
    err = 0.0
    for tbw, window in ((2.0, "hann"), (4.0, "hamming"), (8.0, "blackman")):
        alpha, n = math.radians(1.0), 256
        pulse = pulses.sinc(1.0, 2.56, tbw, 5.0, window, n=n)
        z = np.linspace(-15, 15, 601)
        a, b = pulse.rotate(z, rephase=False)
        psi = RAD_PER_MT_M_MM_MS * pulse.g_mt_m[0] * pulse.dt_ms[0] * z
        theta = RAD_PER_UT_MS * pulse.b1_ut.real * pulse.dt_ms
        fourier = -1j * (theta[None, :] * np.exp(1j * psi[:, None] * (n - np.arange(n)[None, :] - 0.5))).sum(1)
        err = max(err, np.abs(2 * np.conj(a) * b - fourier).max() / alpha)
    rows.append(_row("D6", "rotation", "slice profile of a 1 degree windowed sinc (time-bandwidth 2, 4, 8; Hann, Hamming, Blackman)",
                     "small-tip approximation: Mxy = -i x the Fourier sum of the sub-pulse angles with the precession that follows each", err, 1e-4,
                     unit="of the flip angle"))

    # D7 real 3 x 3 cascade, and the refocusing relations between crushers
    pulse = pulses.sinc(180.0, 3.0, 4.0, 7.5, kind="refocusing", n=128)
    z = np.linspace(-9, 9, 37)
    a, b = pulse.rotate(z)
    mats = rotation.matrix(a, b)
    theta = RAD_PER_UT_MS * pulse.b1_ut.real * pulse.dt_ms
    err_m, err_e = 0.0, 0.0
    m0 = np.array([0.31, -0.52, 0.44])
    for i, zi in enumerate(z):
        half = _rodrigues([0, 0, 1], 0.5 * RAD_PER_MT_M_MM_MS * pulse.g_mt_m[0] * pulse.dt_ms[0] * zi)
        total = np.eye(3)
        for t in theta:
            total = half @ _rodrigues([1, 0, 0], t) @ half @ total
        err_m = max(err_m, np.abs(mats[i] - total).max())
        acc = np.zeros(3)
        for c in 2 * np.pi * np.arange(32) / 32:                 # crusher of c before and after the pulse
            acc += _rodrigues([0, 0, 1], c) @ total @ _rodrigues([0, 0, 1], c) @ m0
        acc /= 32
        mxy0 = m0[0] + 1j * m0[1]
        err_e = max(err_e, abs(acc[0] + 1j * acc[1] + b[i] ** 2 * np.conj(mxy0)), abs(acc[2] - (1 - 2 * abs(b[i]) ** 2) * m0[2]))
    rows.append(_row("D7", "rotation", "rotation matrix of a 180 degree sinc at 37 positions", "separately written cascade of real 3 x 3 rotations "
                     "(Rodrigues formula), sample by sample", err_m, 1e-12))
    rows.append(_row("D7b", "rotation", "refocusing between equal crushers: echo and longitudinal magnetisation, averaged over 32 crusher phases of "
                     "that cascade", "echo = -b^2 conj(Mxy), Mz' = (1 - 2 |b|^2) Mz", err_e, 1e-12))

    # D8 unitarity
    b1 = (rng.normal(size=512) + 1j * rng.normal(size=512)) * 0.5
    z = np.linspace(-10, 10, 2000)
    err = 0.0
    for model in rotation.MODELS:
        a, b = rotation.rotate(b1, 0.01, 10.0, z, model=model, normalise=False)
        err = max(err, np.abs(np.abs(a) ** 2 + np.abs(b) ** 2 - 1).max())
    g3, pos3 = rng.normal(size=(512, 3)) * 10, rng.normal(size=(2000, 3)) * 5
    a, b = rotation.rotate(b1, 0.01, g3, pos3, rng.uniform(-300, 300, 2000), rng.uniform(0.6, 1.4, 2000), normalise=False)
    err = max(err, np.abs(np.abs(a) ** 2 + np.abs(b) ** 2 - 1).max())
    rows.append(_row("D8", "rotation", "unitarity before the final normalisation: 512 random samples, 2,000 isochromats, both models, one and "
                     "three gradient axes", "|a|^2 + |b|^2 = 1", err, 1e-12))
    a, b = rotation.rotate((rng.normal(size=4096) + 1j * rng.normal(size=4096)) * 0.1, 0.01, 10.0, z)
    rows.append(_row("D8b", "rotation", "unitarity of the returned (normalised) parameters, 4,096 samples", "|a|^2 + |b|^2 = 1",
                     np.abs(np.abs(a) ** 2 + np.abs(b) ** 2 - 1).max(), 1e-14))

    # D9 time reversal
    n = 200
    b1 = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 2.0
    g2, pos2 = rng.normal(size=(n, 2)) * 8, rng.normal(size=(50, 2)) * 6
    dts, df, sc = rng.uniform(0.005, 0.02, n), rng.uniform(-400, 400, 50), rng.uniform(0.7, 1.3, 50)
    err = 0.0
    for model in rotation.MODELS:
        a, b = rotation.rotate(b1, dts, g2, pos2, df, sc, model=model)
        ar, br = rotation.rotate(np.conj(b1[::-1]), dts[::-1], g2[::-1], pos2, df, sc, model=model)
        err = max(err, np.abs(ar - a).max(), np.abs(br + np.conj(b)).max())
        au, bu = rotation.rotate(-b1[::-1], dts[::-1], -g2[::-1], pos2, -df, sc, model=model)
        ai, bi = rotation.compose(a, b, au, bu)
        err = max(err, np.abs(ai - 1).max(), np.abs(bi).max())
    rows.append(_row("D9", "rotation", "time reversal, 200 random samples of unequal duration under two gradient axes, 50 isochromats, both models",
                     "the reversed, conjugated pulse gives (a, -conj(b)); the reversed pulse with B1, gradient and offset negated undoes the rotation",
                     err, 1e-12))

    # D10 relaxation against an independent integration of the Bloch equations
    dur, grad = 4.0, 8.0

    def b1f(t):
        u = t / dur - 0.5
        return 9.0 * np.sinc(6 * u) * 0.5 * (1 + np.cos(2 * np.pi * u)) * np.exp(1j * (3.0 * np.pi * u + 3.0 * u * u))

    iso = [(0.0, 0.0, 1.0, 800.0, 60.0), (1.5, 40.0, 0.9, 1200.0, 8.0), (-3.0, -120.0, 1.15, 300.0, 25.0), (4.0, 250.0, 1.0, 2000.0, 4.0)]
    start = np.array([0.2, -0.1, 0.9])
    ref = []
    for zi, dfi, sci, t1, t2 in iso:
        def rhs(t, m, zi=zi, dfi=dfi, sci=sci, t1=t1, t2=t2):
            bb = b1f(t) * sci
            w = np.array([RAD_PER_UT_MS * bb.real, RAD_PER_UT_MS * bb.imag, RAD_PER_MT_M_MM_MS * grad * zi + RAD_PER_HZ_MS * dfi])
            d = np.cross(w, m)
            return [d[0] - m[0] / t2, d[1] - m[1] / t2, d[2] - (m[2] - 1.0) / t1]
        ref.append(scipy.integrate.solve_ivp(rhs, (0.0, dur), start, method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1])
    ref = np.array(ref)
    zi, dfi, sci, t1, t2 = (np.array(x) for x in zip(*iso))
    for model, rid in (("split", "D10"), ("piecewise", "D10b")):
        errs = []
        for n in (200, 400, 800, 1600):
            t = (np.arange(n) + 0.5) * dur / n
            mxy, mz = rotation.bloch(b1f(t), dur / n, grad, zi, dfi, sci, t1, t2, mxy0=start[0] + 1j * start[1], mz0=start[2], model=model)
            errs.append(max(np.abs(mxy.real - ref[:, 0]).max(), np.abs(mxy.imag - ref[:, 1]).max(), np.abs(mz - ref[:, 2]).max()))
        rows.append(_row(rid, "rotation", f"relaxation during a 4 ms complex pulse (model {model}): 4 isochromats, T2 4 to 60 ms, T1 300 to 2000 ms, "
                         "400 samples (10 us raster)", "`scipy.integrate.solve_ivp` (DOP853, rtol 1e-12) on the Bloch equations with the continuous "
                         "waveform", errs[1], 1e-4))
        rows.append(_row(rid + "o", "rotation", f"the same (model {model}): order of convergence with the raster, 200 to 1,600 samples",
                         "second order (symmetric splitting, samples at interval centres)", abs(_order(errs) - 2.0), 0.02, unit="in the order"))

    # D11 no relaxation: bloch = apply(rotate);  D12 three axes = projection on one
    b1 = (rng.normal(size=128) + 1j * rng.normal(size=128)) * 3.0
    z = np.linspace(-6, 6, 41)
    err = 0.0
    for model in rotation.MODELS:
        a, b = rotation.rotate(b1, 0.01, 12.0, z, 55.0, 0.93, model=model)
        m1, z1 = rotation.apply(a, b, 0.3 - 0.2j, 0.7)
        m2, z2 = rotation.bloch(b1, 0.01, 12.0, z, 55.0, 0.93, mxy0=0.3 - 0.2j, mz0=0.7, model=model)
        err = max(err, np.abs(m1 - m2).max(), np.abs(z1 - z2).max())
    rows.append(_row("D11", "rotation", "`bloch` without relaxation, from a tilted start", "`apply(rotate(...))`", err, 1e-12))
    u = np.array([0.36, -0.48, 0.8])
    shape = rng.normal(size=128) * 10
    pos = rng.normal(size=(60, 3)) * 5
    a3, b3 = rotation.rotate(b1, 0.01, shape[:, None] * u[None, :], pos)
    a1, bb1 = rotation.rotate(b1, 0.01, shape, pos @ u)
    rows.append(_row("D12", "rotation", "a time-varying gradient along an oblique direction on three axes", "the same waveform on one axis at the "
                     "projected positions", max(np.abs(a3 - a1).max(), np.abs(b3 - bb1).max()), 1e-12))

    # D13 speed
    b1 = (rng.normal(size=512) + 1j * rng.normal(size=512)) * 0.5
    z = np.linspace(-10, 10, 2000)
    best = min(_timed(lambda: rotation.rotate(b1, 0.01, 10.0, z)) for _ in range(3))
    rows.append(_row("D13", "rotation", "time of one rotation: 512 samples x 2,000 positions (best of 3)", "requirement", best, 0.1, unit="s"))
    return rows


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------- pulses
SLR_CASES = (("excitation", 90.0), ("excitation", 30.0), ("refocusing", 180.0), ("refocusing", 150.0), ("inversion", 180.0), ("saturation", 90.0))


def _slr_target(kind, flip):
    half = math.sin(math.radians(flip) / 2)
    return dict(excitation=math.sin(math.radians(flip)), refocusing=half ** 2, inversion=half ** 2, saturation=half ** 2)[kind]


def check_pulses():
    from ..scanner import rf
    rows, rng = [], np.random.default_rng(7)

    # P1 the console's pulse shape
    err = 0.0
    for dur, tbw in rf.PULSES.values():
        for window in rf.WINDOWS:
            pulse = pulses.sinc(77.0, dur, tbw, window=window, n=rf.N_SAMPLES)
            err = max(err, np.abs(RAD_PER_UT_MS * pulse.b1_ut * pulse.dt_ms / math.radians(77.0) - rf.sinc_pulse(tbw, rf.N_SAMPLES, window)).max(),
                      abs(pulse.flip_deg() - 77.0) / 77.0)
    rows.append(_row("P1", "pulses", "`sinc` with 128 samples: sub-pulse weights and flip angle on resonance", "`scanner/rf.py::sinc_pulse`", err, 1e-12))

    # P2 inverse Shinnar-Le Roux transform: forward simulation against the designed polynomial
    theta = (rng.normal(size=96) + 1j * rng.normal(size=96)) * 0.15
    a_poly, b_poly = pulses.forward_slr(theta)
    psi = np.linspace(-np.pi, np.pi, 721)
    dt = 0.01
    a, b = rotation.rotate(theta / (RAD_PER_UT_MS * dt), dt, None, None, psi / (RAD_PER_HZ_MS * dt))
    x = np.exp(1j * psi)
    err = max(np.abs(a - np.exp(-0.5j * 96 * psi) * np.polyval(a_poly[::-1], x)).max(),
              np.abs(b - np.exp(-0.5j * 95 * psi) * np.polyval(b_poly[::-1], x)).max())
    rows.append(_row("P2", "pulses", "the polynomials A, B of `forward_slr` for 96 random complex sub-pulses", "`rotation.rotate` on 721 precession "
                     "angles per sample: a = exp(-i n psi/2) A, b = exp(-i (n-1) psi/2) B", err, 1e-12))
    rows.append(_row("P2b", "pulses", "`inverse_slr` of those polynomials", "the sub-pulses they were made from",
                     np.abs(pulses.inverse_slr(b_poly, a_poly) - theta).max(), 1e-12))
    err, worst_ripple, jensen = 0.0, 0.0, 0.0
    for kind, flip in SLR_CASES:
        for filt, n, tbw in (("pm", 256, 6.0), ("ls", 255, 6.0), ("pm", 150, 4.0)):
            b_design, rec = pulses.slr_filter(n, tbw, kind, flip, filt)
            pulse = pulses.slr(flip, 0.01 * n, tbw, 5.0, kind, filt, n=n)
            a, b = rotation.rotate(pulse.b1_ut, pulse.dt_ms, None, None, psi / (RAD_PER_HZ_MS * pulse.dt_ms[0]))
            err = max(err, np.abs(b - np.exp(-0.5j * (n - 1) * psi) * np.polyval(b_design[::-1], x)).max())
            th = RAD_PER_UT_MS * pulse.b1_ut * pulse.dt_ms
            energy_sum = -np.sum(np.log(np.cos(np.abs(th) / 2)))
            jensen = max(jensen, abs(energy_sum / profile.jensen_integral(pulse) - 1))
            if filt == "pm":                                    # the profile ripples that were asked for
                z = profile.sample_positions(5.0)
                aa, bb = pulse.rotate(z)
                r = np.abs(profile.response(kind, aa, bb))
                mult = 2.0 if kind in ("inversion", "saturation") else 1.0      # the specification is on Mz = 1 - 2 |b|^2
                w = rec["transition_fraction"]
                worst_ripple = max(worst_ripple, mult * profile.passband_ripple(z, r, 5.0, w, _slr_target(kind, flip)) / 0.01,
                                   mult * profile.stopband_ripple(z, r, 5.0, w) / 0.01)
    rows.append(_row("P3", "pulses", "inverse transform of the designs: b of the forward-simulated `slr` pulse (excitation 90 / 30, refocusing "
                     "180 / 150, inversion, saturation; equiripple and least squares; 150 to 256 samples)",
                     "the designed polynomial B on 721 points of the unit circle", err, 1e-9))
    rows.append(_row("P3b", "pulses", "the minimum-phase A of those designs: -sum ln cos(theta_j / 2) of the pulse", "Jensen's formula: "
                     "-(1/4 pi) x integral of ln(1 - |B|^2), equality for a minimum-phase A", jensen, 1e-6, unit="relative"))
    rows.append(_row("P4", "pulses", "ripples of the simulated profiles of the equiripple designs, in the slice and outside, relative to the "
                     "ripples asked for (1 % of M0 each)", "the specification: |Mxy| (excitation), Mz (inversion, saturation), |b|^2 (refocusing)",
                     worst_ripple, 1.02, unit="of the requested ripple"))
    d = 0.01
    table = dict(excitation=(90.0, math.sqrt(d / 2), d / math.sqrt(2)), saturation=(90.0, d / 2, math.sqrt(d)),
                 refocusing=(180.0, d / 4, math.sqrt(d)), inversion=(180.0, d / 8, math.sqrt(d / 2)))
    err = 0.0
    for kind, (flip, p1, p2) in table.items():
        scale, q1, q2, _ = pulses.polynomial_ripples(kind, flip, d, d)
        err = max(err, abs(q1 / p1 - 1), abs(q2 * scale / (p2 * (1.0 if flip == 180.0 else math.sqrt(0.5))) - 1))
    rows.append(_row("P5", "pulses", "filter ripples for profile ripples of 1 %: `polynomial_ripples` (solved numerically)", "the relations of Pauly "
                     "et al. 1991, which are first order in the ripples", err, 0.04, unit="relative"))

    # P6 hyperbolic secant against the exact solution
    beta, mu, dur = 1.2, 3.0, 25.0
    df = np.linspace(-1500, 1500, 61)
    errs = []
    for n in (1000, 2000, 4000):
        e = 0.0
        for peak in (2.0, 5.0, 10.0, 14.0):
            a, b = pulses.sech(peak, dur, beta, mu, n=n).rotate(None, df)
            e = max(e, np.abs(np.abs(a) ** 2 - np.abs(b) ** 2 - pulses.sech_mz(peak, beta, mu, df)).max())
        errs.append(e)
    rows.append(_row("P6", "pulses", "Mz after a hyperbolic-secant pulse (mu 3, truncated at 6e-7 of the peak, 4,000 samples), 4 amplitudes x 61 "
                     "offsets", "exact solution for the untruncated pulse (Silver, Joseph, Hoult 1985)", errs[2], 5e-5))
    rows.append(_row("P6o", "pulses", "the same: order of convergence from 1,000 to 4,000 samples", "second order", abs(_order(errs) - 2.0), 0.3,
                     unit="in the order"))
    # adiabatic threshold: above it the inversion no longer depends on B1; a hard pulse does
    beta, mu, dur, peak = 1.06, 5.0, 10.0, 12.0
    pulse = pulses.sech(peak, dur, beta, mu)
    x0 = RAD_PER_UT_MS * peak / beta
    need = math.sqrt(mu ** 2 - (mu - math.log(2 / 0.01) / math.pi) ** 2) / x0       # scale at which the exact Mz is -0.99 (large mu)
    scales = np.linspace(need, 2.5 * need, 60)
    a, b = pulse.rotate(None, 0.0, scales)
    mz = np.abs(a) ** 2 - np.abs(b) ** 2
    rows.append(_row("P7", "pulses", "adiabatic threshold: 10 ms hyperbolic secant (mu 5, truncated at 1 %), B1 from the threshold of the exact "
                     "solution for Mz = -0.99 to 2.5 x that", "Mz stays below -0.98 (exact solution of the untruncated pulse: below -0.99)",
                     mz.max(), -0.98))
    below = np.linspace(0.2, 1.0, 9) * need
    a, b = pulse.rotate(None, 0.0, below)
    rows.append(_row("P7b", "pulses", "the same pulse below the threshold (0.2 to 1 x): Mz against B1 scale", "exact solution of the untruncated "
                     "pulse; the truncation at 1 % of the peak is the difference", np.abs(np.abs(a) ** 2 - np.abs(b) ** 2
                                                                                        - pulses.sech_mz(peak, beta, mu, 0.0, below)).max(), 6e-3))
    hard = pulses.control_points([math.pi / (RAD_PER_UT_MS * 1.0)], 1.0)
    a, b = hard.rotate(None, 0.0, np.array([1.0, 2.0]))
    rows.append(_row("P7c", "pulses", "a hard 180 degree pulse over the same factor 2 in B1", "Mz = cos(pi x scale): -1 and +1",
                     max(abs(1 - 2 * abs(b[0]) ** 2 + 1), abs(1 - 2 * abs(b[1]) ** 2 - 1)), 1e-12))

    # P8 control points
    pts = np.array([1.0 + 2.0j, 4.0 - 1.0j])
    cp = pulses.control_points(pts, 2.0, n=200)
    t = (np.arange(200) + 0.5) * 0.01
    err = np.abs(cp.b1_ut - (pts[0] + (pts[1] - pts[0]) * t / 2.0)).max()
    one = pulses.control_points([6.0], 0.5)
    err = max(err, abs(one.flip_deg() - math.degrees(RAD_PER_UT_MS * 6.0 * 0.5)) / 100, abs(one.energy_ut2_ms - 18.0), abs(one.b1rms_ut(50.0) - 0.6))
    knots = np.linspace(0, 1.0, 9)
    spline = pulses.control_points(np.sin(3 * knots) + 1j * knots ** 2, 1.0, interp="cubic", n=100)
    lin = pulses.control_points(np.sin(3 * knots) + 1j * knots ** 2, 1.0, interp="linear", n=100)
    tt = (np.arange(100) + 0.5) * 0.01
    truth = np.sin(3 * tt) + 1j * tt ** 2
    rows.append(_row("P8", "pulses", "control points: two points (linear rule at the sample centres); one point (hard pulse): flip, energy, B1rms",
                     "the straight line; flip = 2 pi gamma B1 T, energy = B1^2 T, B1rms = B1 sqrt(T / TR)", err, 1e-12))
    rows.append(_row("P8b", "pulses", "nine control points on a smooth curve: error of the cubic rule relative to the error of the linear rule",
                     "a spline through the points is closer to the curve than straight lines", np.abs(spline.b1_ut - truth).max()
                     / np.abs(lin.b1_ut - truth).max(), 0.5))

    # P9 variable rate
    base = pulses.sinc(90.0, 2.0, 4.0, 5.0)
    z = np.linspace(-12, 12, 481)
    a0, b0 = base.rotate(z)
    err, off, lim = 0.0, 1.0, 0.0
    for slew in (None, 120.0):
        fast = pulses.verse(base, 6.0, 20.0, slew)
        a1, b1 = fast.rotate(z)
        err = max(err, np.abs(a1 - a0).max(), np.abs(b1 - b0).max())
        af, bf = base.rotate(z, 200.0)
        a2, b2 = fast.rotate(z, 200.0)
        off = min(off, np.abs(2 * np.conj(af) * bf - 2 * np.conj(a2) * b2).max())
        chk = gradients.check_pulse_gradient(fast, gradients.Hardware(g_max_mt_m=20.0, slew_t_m_s=120.0 if slew else 1e9, b1_peak_ut=6.0))
        lim = max(lim, fast.peak_b1_ut / 6.0, chk["peak_mt_m"] / 20.0, (chk["slew_t_m_s"] / 120.0) if slew else 0.0, 0.0 if chk["ok"] else 2.0)
    rows.append(_row("P9", "pulses", "variable-rate rescaling of a 90 degree sinc to a peak of 6 uT (gradient up to 20 mT/m, with and without a "
                     "slew limit of 120 T/m/s): (a, b) on resonance at 481 positions", "the pulse before rescaling", err, 1e-12))
    rows.append(_row("P9b", "pulses", "the same pulses 200 Hz off resonance: largest change of Mxy", "off resonance the rotation is NOT preserved "
                     "(a change above 0.05 of M0 is expected)", off, 0.05, kind="min"))
    rows.append(_row("P9c", "pulses", "peak B1, gradient and slew of the rescaled pulses relative to the limits given", "at most 1", lim, 1 + 1e-9))
    full = pulses.verse(base, 6.0, 1e9)
    rows.append(_row("P9d", "pulses", "duration of the rescaled pulse when only the peak B1 binds", "sum |theta_j| / (2 pi gamma B1max) "
                     "(`profile.variable_rate_floor_ms`)", abs(full.duration_ms / profile.variable_rate_floor_ms(base, 6.0) - 1), 1e-12, unit="relative"))
    return rows


# ---------------------------------------------------------------------------------------------- profile
def generator_outputs():
    """A spread of generator outputs on a constant raster and gradient, for the bound checks."""
    out = []
    for tbw, dur in ((2.0, 1.0), (4.0, 2.0), (8.0, 4.0)):
        for flip in (10.0, 60.0, 90.0, 180.0):
            for window in ("hann", "hamming", "rect"):
                out.append((f"sinc {window} tbw {tbw:g} {flip:g} deg", pulses.sinc(flip, dur, tbw, 5.0, window, "refocusing" if flip == 180 else "excitation")))
    for kind, flip in SLR_CASES:
        for filt, n, tbw in (("pm", 256, 6.0), ("ls", 255, 6.0), ("pm", 150, 4.0), ("pm", 400, 10.0)):
            out.append((f"slr {kind} {flip:g} {filt} tbw {tbw:g}", pulses.slr(flip, 0.01 * n, tbw, 5.0, kind, filt, n=n)))
    for peak, beta, mu, dur in ((12.0, 1.06, 5.0, 10.0), (16.0, 2.0, 3.0, 6.0), (8.0, 0.7, 8.0, 14.0)):
        out.append((f"sech {peak:g} uT mu {mu:g}", pulses.sech(peak, dur, beta, mu, 5.0)))
    rng = np.random.default_rng(3)
    for k in (4, 16, 64):
        out.append((f"control points {k}", pulses.control_points((rng.normal(size=k) + 1j * rng.normal(size=k)) * 4.0, 2.0, 3000.0, 5.0)))
    return out


def check_profile():
    rows = []
    # F1 figures of synthetic profiles with known answers
    z = np.linspace(-30, 30, 24001)
    sigma = 2.3
    gauss = np.exp(-0.5 * (z / sigma) ** 2)
    err = abs(profile.fwhm(z, gauss) / (2 * math.sqrt(2 * math.log(2)) * sigma) - 1)
    lo, hi = sigma * math.sqrt(-2 * math.log(0.9)), sigma * math.sqrt(-2 * math.log(0.1))
    err = max(err, abs(profile.transition_width(z, gauss) / (hi - lo) - 1))
    d, w = 6.0, 0.25                                            # trapezoid profile: flat to (1 - w) d / 2, zero from (1 + w) d / 2
    trap = np.clip(((1 + w) * d / 2 - np.abs(z)) / (w * d), 0, 1) * 0.8
    err = max(err, abs(profile.fwhm(z, trap) / d - 1), abs(profile.transition_width(z, trap) / (0.8 * w * d) - 1),
              abs(profile.out_of_slice_fraction(z, trap, d) / (w / 4) - 1))
    wob = trap * (1 + 0.03 * np.cos(7 * z)) + 0.012 * (np.abs(z) >= (1 + w) * d / 2) * np.abs(np.sin(5 * z))
    err = max(err, abs(profile.passband_ripple(z, wob, d, w) / (0.8 * 0.03) - 1), abs(profile.stopband_ripple(z, wob, d, w) / 0.012 - 1))
    rows.append(_row("F1", "profile", "width at half maximum, 10-90 % transition width, out-of-slice fraction, pass-band and stop-band ripple of "
                     "synthetic profiles (Gaussian, trapezoid with added ripples) on 24,001 positions", "their analytic values", err, 1e-3, unit="relative"))
    slope = 0.21
    box = np.exp(1j * slope * z) * (np.abs(z) <= d / 2 + 1e-9)
    ph = profile.phase_figures(z, box, d)
    arg = slope * d / 2
    err = max(abs(ph["slope_rad_per_mm"] / slope - 1), abs(ph["refocused_fraction"] / (math.sin(arg) / arg) - 1), abs(ph["linear_rms_rad"]),
              abs(ph["rms_rad"] / (slope * d / math.sqrt(12)) - 1))
    rows.append(_row("F2", "profile", "phase figures of a uniform slice with a linear phase: slope, rms phase, residual after the line, refocused "
                     "fraction", "slope; slope d / sqrt(12); 0; sin(slope d / 2) / (slope d / 2)", err, 1e-3, unit="relative"))
    mask = profile.standard_mask(d, w, 0.8, 0.03, 0.02)
    good = profile.check_mask(z, trap, mask)
    bad_stop = profile.check_mask(z, trap + 0.05 * (z > 10), mask)
    bad_pass = profile.check_mask(z, trap * (1 - 0.2 * (np.abs(z) < 0.5)) + 0.05 * (z > 10), mask)
    ok = (good["passed"] and good["first_violation"] is None and not bad_stop["passed"] and bad_stop["first_violation"] == "stopband_high"
          and bad_pass["first_violation"] == "passband" and abs(bad_stop["margins"]["stopband_high"] + 0.03) < 1e-12
          and not profile.check_mask(z[np.abs(z) < 2], trap[np.abs(z) < 2], mask)["passed"])
    rows.append(_row("F3", "profile", "profile mask: a profile inside the mask passes; a planted stop-band or pass-band excess fails with that band "
                     "named first and the margin -0.03; a band without samples fails", "by construction", 0.0 if ok else 1.0, 0.0))

    # F4, F5 the bounds on every generator output
    outs = generator_outputs()
    e_ratio, t_ratio, t_name = math.inf, math.inf, ""
    for name, pulse in outs:
        e_ratio = min(e_ratio, pulse.energy_ut2_ms / profile.pulse_energy_floor_ut2_ms(pulse))
        d_mm = pulse.thickness_mm
        zz = profile.sample_positions(d_mm, extent=1.0, per_thickness=400)
        _, b = pulse.rotate(zz)
        flips = np.degrees(rotation.flip_angle(b))
        for frac in (0.2, 0.5, 0.8):                             # the specification "at least this flip over this band"
            inside = np.abs(zz) <= frac * d_mm / 2
            bound = profile.duration_bound_ms(max(flips[inside].min(), 1e-6), frac * pulse.bandwidth_hz, pulse.peak_b1_ut, float(pulse.dt_ms[0]))
            if pulse.duration_ms / bound["bound_ms"] < t_ratio:
                t_ratio, t_name = pulse.duration_ms / bound["bound_ms"], name
    rows.append(_row("F4", "profile", f"RF energy of {len(outs)} generator outputs (sinc, Shinnar-Le Roux, hyperbolic secant, control points) "
                     "relative to the energy floor of their own |B|: smallest ratio, minus 1", "no pulse is below the floor (Jensen's formula); "
                     "the ratio is 1 for a minimum-phase A and small sub-pulse angles", e_ratio - 1.0, 0.0, kind="min"))
    rows.append(_row("F5", "profile", "duration of the same outputs relative to `duration_bound_ms` at their own specification (smallest flip angle "
                     "over 20, 50 and 80 % of the bandwidth, their own peak B1): smallest ratio", f"no design is shorter than the bound (closest: {t_name})",
                     t_ratio, 1.0, kind="min"))
    # a pulse with the same |B| and a non-minimum-phase A needs more energy
    b_design, _ = pulses.slr_filter(128, 4.0, "excitation", 90.0)
    a_min = pulses.min_phase_a(b_design)
    a_max = np.conj(a_min[::-1])
    a_max = a_max * np.exp(-1j * np.angle(a_max[0]))
    th_min, th_max = pulses.inverse_slr(b_design, a_min), pulses.inverse_slr(b_design * np.exp(-1j * np.angle(np.conj(a_min[-1]))), a_max)
    p_min = pulses.Pulse(th_min / (RAD_PER_UT_MS * 0.01), 0.01, 5.0)
    p_max = pulses.Pulse(th_max / (RAD_PER_UT_MS * 0.01), 0.01, 5.0)
    zz = np.linspace(-20, 20, 801)
    same = np.abs(np.abs(p_min.rotate(zz)[1]) - np.abs(p_max.rotate(zz)[1])).max()
    rows.append(_row("F6", "profile", "a second pulse with the same |b| at 801 positions but the maximum-phase A: difference of |b|", "equal by "
                     "construction (the floor depends on |B| only)", same, 1e-9))
    rows.append(_row("F6b", "profile", "its energy relative to the floor (the minimum-phase pulse: "
                     f"{p_min.energy_ut2_ms / profile.pulse_energy_floor_ut2_ms(p_min):.6f})", "above 1: the floor separates the two",
                     p_max.energy_ut2_ms / profile.pulse_energy_floor_ut2_ms(p_max), 1.05, kind="min"))
    hard = pulses.control_points([15.0], 0.5)
    bound = profile.duration_bound_ms(hard.flip_deg(), 0.0, 15.0, 0.01)
    rows.append(_row("F7", "profile", "a hard pulse at the peak limit against the bound for its flip angle", "the hard-pulse floor is attained: ratio 1",
                     abs(hard.duration_ms / bound["bound_ms"] - 1), 1e-9, unit="relative"))
    return rows


# ---------------------------------------------------------------------------------------------- sequence
def _isochromats(steps, t1, t2, n_iso=64):
    """Independent reference for the phase graph: n_iso isochromats with dephasing angles 2 pi j / n_iso per cycle,
    real 3 x 3 rotations, relaxation as a diagonal decay plus regrowth. ``steps``: ("rf", flip_deg, phase_deg),
    ("grad", cycles), ("delay", ms), ("read",). Returns the mean transverse magnetisation at every read."""
    m = np.tile(np.array([0.0, 0.0, 1.0]), (n_iso, 1))
    phases = 2 * np.pi * np.arange(n_iso) / n_iso
    out = []
    for step in steps:
        if step[0] == "rf":
            p = math.radians(step[2])
            m = m @ _rodrigues([math.cos(p), math.sin(p), 0.0], math.radians(step[1])).T
        elif step[0] == "grad":
            c, s = np.cos(step[1] * phases), np.sin(step[1] * phases)
            m = np.stack([c * m[:, 0] - s * m[:, 1], s * m[:, 0] + c * m[:, 1], m[:, 2]], 1)
        elif step[0] == "delay":
            e1, e2 = math.exp(-step[1] / t1), math.exp(-step[1] / t2)
            m = np.stack([m[:, 0] * e2, m[:, 1] * e2, 1 + (m[:, 2] - 1) * e1], 1)
        else:
            out.append(np.mean(m[:, 0] + 1j * m[:, 1]))
    return np.array(out)


def bssfp_closed_form(a, theta, tr_ms, t1_ms, t2_ms):
    """Balanced SSFP right after the pulse (Freeman & Hill 1971): the periodic state of rotation about x by ``a``
    and precession ``theta`` with relaxation over TR, solved as a 3 x 3 linear system. Returns Mxy (complex)."""
    e1, e2 = math.exp(-tr_ms / t1_ms), math.exp(-tr_ms / t2_ms)
    free = np.diag([e2, e2, e1]) @ _rodrigues([0, 0, 1], theta)
    rx = _rodrigues([1, 0, 0], a)
    m = np.linalg.solve(np.eye(3) - rx @ free, rx @ np.array([0.0, 0.0, 1 - e1]))
    return m[0] + 1j * m[1]


def check_sequence():
    from ..scanner import epg, kspace, rf
    from ..scanner import signal as engine_signal
    S = sequence
    rows = []
    t1, t2 = 1000.0, 80.0

    # E1 spin echo, E2 stimulated echo
    err = 0.0
    for te in (10.0, 30.0, 90.0):
        for crush in (0, 1, 3):
            r = S.run([S.rf(90, 90), S.delay(te / 2), S.gradient(crush), S.rf(180, 0), S.gradient(crush), S.delay(te / 2), S.readout()], t1, t2,
                      df_hz=37.0)
            err = max(err, abs(r.signal[0] - 1j * math.exp(-te / t2)))
    rows.append(_row("E1", "sequence", "spin echo (90 about y, 180 about x; echo times 10 to 90 ms; with and without crushers; 37 Hz off resonance)",
                     "exp(-TE / T2), no off-resonance phase at the echo", err, 1e-13))
    err = 0.0
    for a1, a2, a3, tau, tm in ((90, 90, 90, 6.0, 30.0), (70, 50, 110, 8.0, 200.0), (20, 160, 45, 3.0, 900.0)):
        r = S.run([S.rf(a1), S.delay(tau), S.gradient(1), S.rf(a2), S.delay(tm), S.gradient(3), S.rf(a3), S.gradient(1), S.delay(tau),
                   S.readout()], t1, t2)
        ref = 0.5 * math.sin(math.radians(a1)) * math.sin(math.radians(a2)) * math.sin(math.radians(a3)) * math.exp(-2 * tau / t2) * math.exp(-tm / t1)
        err = max(err, abs(abs(r.signal[0]) - ref))
    rows.append(_row("E2", "sequence", "stimulated echo of three pulses with a mixing-time crusher", "(1/2) sin a1 sin a2 sin a3 exp(-2 tau / T2) "
                     "exp(-TM / T1)", err, 1e-13))

    # E3 CPMG with imperfect refocusing: closed forms of the first two echoes, then an isochromat ensemble
    err, esp = 0.0, 10.0
    for al in (180.0, 150.0, 120.0, 60.0):
        blocks = [S.rf(90, 90)]
        for i in range(2):
            blocks += [S.delay(esp / 2), S.gradient(1), S.rf(al, 0), S.gradient(1), S.delay(esp / 2), S.readout(i)]
        r = S.run(blocks, t1, t2)
        s2 = math.sin(math.radians(al) / 2) ** 2
        e2, e1 = math.exp(-esp / t2), math.exp(-esp / t1)
        err = max(err, abs(abs(r.signal[0]) - s2 * e2), abs(abs(r.signal[1]) - (s2 * s2 * e2 * e2 + 0.5 * math.sin(math.radians(al)) ** 2 * e2 * e1)))
    rows.append(_row("E3", "sequence", "Carr-Purcell-Meiboom-Gill train with refocusing angles 180 to 60 deg: echoes 1 and 2",
                     "sin^2(a/2) E2 and sin^4(a/2) E2^2 + (1/2) sin^2(a) E2 E1 (direct and stimulated pathway; E = exp(-spacing / T))", err, 1e-13))
    err = 0.0
    for al, ph_ref, ph_exc in ((120.0, 0.0, 90.0), (120.0, 0.0, 0.0), (75.0, 40.0, 90.0)):       # CPMG, non-CPMG, a phase error
        blocks, steps = [S.rf(90, ph_exc)], [("rf", 90, ph_exc)]
        for i in range(14):
            blocks += [S.delay(esp / 2), S.gradient(1), S.rf(al, ph_ref), S.gradient(1), S.delay(esp / 2), S.readout(i)]
            steps += [("delay", esp / 2), ("grad", 1), ("rf", al, ph_ref), ("grad", 1), ("delay", esp / 2), ("read",)]
        r = S.run(blocks, t1, t2, demodulation=1.0)
        err = max(err, np.abs(r.signal - _isochromats(steps, t1, t2)).max())
    rows.append(_row("E4", "sequence", "14 echoes of trains with 120 and 75 deg refocusing: the CPMG phase, the non-CPMG phase, a 40 deg phase error",
                     "separately written ensemble of 64 isochromats with real 3 x 3 rotations (the recursion the phase graph replaces)", err, 1e-12))
    # long trains without relaxation: the pseudo steady state, with and without the 90 + a/2 first pulse
    err_q, err_t = 0.0, 0.0
    for al in (60.0, 120.0, 150.0):
        s_half = math.sin(math.radians(al) / 2)
        for first, closed in ((al, s_half), (90 + al / 2, 2 * s_half / (1 + s_half))):
            acc = 0.0
            for phi in 2 * np.pi * (np.arange(720) + 0.5) / 720:       # every isochromat ends on its own rotation axis
                rz = _rodrigues([0, 0, 1], phi)
                period = rz @ _rodrigues([1, 0, 0], math.radians(al)) @ rz
                axis = np.array([period[2, 1] - period[1, 2], period[0, 2] - period[2, 0], period[1, 0] - period[0, 1]])
                axis /= np.linalg.norm(axis)
                m1 = rz @ _rodrigues([1, 0, 0], math.radians(first)) @ rz @ np.array([1.0, 0.0, 0.0])
                acc += (m1 @ axis) * axis[0] / 720
            err_q = max(err_q, abs(acc - closed))
            blocks = [S.rf(90, 90), S.delay(esp / 2), S.gradient(1), S.rf(first, 0), S.gradient(1), S.delay(esp / 2), S.readout()]
            for _ in range(400):
                blocks += [S.delay(esp / 2), S.gradient(1), S.rf(al, 0), S.gradient(1), S.delay(esp / 2), S.readout()]
            err_t = max(err_t, abs(np.abs(S.run(blocks, 1e12, 1e12).signal[-100:]).mean() - closed))
    rows.append(_row("E4b", "sequence", "trains of 400 refocusing pulses of 60, 120, 150 deg without relaxation, mean of the last 100 echoes: all "
                     "pulses equal, and with a first pulse of 90 + a/2 deg", "pseudo steady state: sin(a / 2) without and 2 sin(a / 2) / (1 + sin(a / 2)) "
                     "with the first pulse", err_t, 5e-4))
    rows.append(_row("E4c", "sequence", "those two closed forms", "separately written isochromat picture: every isochromat's magnetisation at the "
                     "first echo projected on the rotation axis of its echo period, averaged over 720 dephasing angles", err_q, 1e-9))

    # E5 spoiled gradient echo: Ernst formula by iteration
    err = 0.0
    for a, tr in ((5.0, 5.0), (15.0, 8.0), (60.0, 30.0), (90.0, 500.0)):
        r = S.run(S.repeat([S.rf(a), S.readout(), S.delay(tr), S.spoil()], int(12 * t1 / tr) + 5), t1, t2)
        e1 = math.exp(-tr / t1)
        err = max(err, abs(r.signal[-1] - math.sin(math.radians(a)) * (1 - e1) / (1 - e1 * math.cos(math.radians(a)))))
    rows.append(_row("E5", "sequence", "spoiled gradient echo with an ideal spoiler, iterated for 12 T1: 4 flip angles and repetition times",
                     "Ernst formula sin a (1 - E1) / (1 - E1 cos a)", err, 1e-9))

    # E6 balanced SSFP on and off resonance
    err, err_engine = 0.0, 0.0
    for a, tr, df in ((40.0, 5.0, 0.0), (40.0, 5.0, 20.0), (40.0, 5.0, 61.0), (70.0, 4.0, 100.0), (25.0, 8.0, -33.0)):
        n = int(25 * t1 / tr)
        blocks = []
        for j in range(n):
            blocks += [S.rf(a, 180.0 * j), S.readout(j, 180.0 * j), S.delay(tr)]
        r = S.run(blocks, t1, t2, df_hz=df, demodulation=1.0)
        theta = 2 * math.pi * df * tr * 1e-3 + math.pi
        err = max(err, abs(r.signal[-1] - bssfp_closed_form(math.radians(a), theta, tr, t1, t2)))
        err_engine = max(err_engine, abs(1j * r.signal[-1] - engine_signal.bssfp(math.radians(a), theta, tr, t1, t2)))
    rows.append(_row("E6", "sequence", "balanced SSFP with 180 deg phase cycling, iterated for 25 T1: on resonance, in the pass band, near a null",
                     "periodic state of the 3 x 3 Bloch matrices (Freeman & Hill 1971), solved as a linear system", err, 1e-9))
    rows.append(_row("E6b", "sequence", "the same states", "the engine's closed form `scanner/signal.py::bssfp`", err_engine, 1e-9))

    # E7 RF-spoiled gradient echo, 117 degrees; E8 with diffusion
    a, tr = 15.0, 8.0
    ernst = math.sin(math.radians(a)) * (1 - math.exp(-tr / t1)) / (1 - math.exp(-tr / t1) * math.cos(math.radians(a)))
    for bd, rid, what in ((0.0, "E7", "RF-spoiled gradient echo (117 deg increment, one cycle of dephasing per TR), 1,500 repetitions"),
                          (0.002, "E8", "the same with diffusion (bd = 0.002 per repetition)")):
        blocks = []
        for j in range(1500):
            ph = 117.0 * j * (j + 1) / 2
            blocks += [S.rf(a, ph), S.readout(j, ph), S.gradient(1, ms=tr, bd=bd)]
        r = S.run(blocks, t1, t2, kmax=200)
        rows.append(_row(rid, "sequence", what, "the engine's steady state `scanner/epg.py::steady_state` (a banded linear solve, no time stepping)",
                         abs(r.signal[-1] - epg.steady_state(math.radians(a), tr, t1, t2, bd=bd)[0]), 1e-9))
        if bd == 0.0:
            rows.append(_row("E7b", "sequence", "its distance from the Ernst value (15 deg, TR 8 ms, T1 1000 ms, T2 80 ms)", "RF spoiling approaches "
                             "ideal spoiling: within 5 %", abs(abs(r.signal[-1]) / ernst - 1), 0.05, unit="relative"))

    # E9 inversion recovery
    err = 0.0
    for ti in (50.0, 300.0, 693.147, 2000.0):
        r = S.run([S.rf(180), S.spoil(), S.delay(ti), S.rf(90), S.readout()], t1, t2)
        err = max(err, abs(r.signal[0] - (1 - 2 * math.exp(-ti / t1))))
    tr, ti, a = 1500.0, 400.0, 30.0
    r = S.run(S.repeat([S.rf(180), S.spoil(), S.delay(ti), S.rf(a), S.readout(), S.spoil(), S.delay(tr - ti)], 30), t1, t2)
    err = max(err, abs(r.signal[-1] - engine_signal.ir_gre(math.radians(a), ti, tr, t1)))
    rows.append(_row("E9", "sequence", "inversion recovery from equilibrium (4 inversion times, one at the null) and in the steady state with a "
                     "30 deg readout", "1 - 2 exp(-TI / T1); the engine's `signal.ir_gre`", err, 1e-12))

    # E10 diffusion: a pulsed-gradient spin echo
    adc, cyc, delta, big = 1.1, 4, 12.0, 40.0                 # um^2/ms, cycles per voxel of 2 mm, ms, ms
    voxel = 2.0
    kappa = 2 * math.pi / voxel                                # rad/mm per cycle
    unit = adc * 1e-3 * kappa ** 2 * 1e-3                      # D kappa^2 per ms  (D in mm^2/s, time in s)
    r = S.run([S.rf(90, 90), S.gradient(cyc, ms=delta, bd=unit * delta), S.delay(big - delta, bd=unit * (big - delta)), S.rf(180, 0),
               S.gradient(cyc, ms=delta, bd=unit * delta), S.readout()], 1e12, 1e12)
    g_amp = cyc / (voxel * CYCLES_PER_MM_PER_MT_M_MS * delta)  # mT/m of the rectangular lobe that gives `cyc` cycles over the voxel
    eps = 1e-7
    t = np.array([0, eps, delta, delta + eps, big, big + eps, big + delta, big + delta + eps])
    b = gradients.b_value(t, np.array([0, g_amp, g_amp, 0, 0, g_amp, g_amp, 0.0]), refocus_ms=(0.5 * (delta + big),))
    rows.append(_row("E10", "sequence", "pulsed-gradient spin echo in the phase graph (4 cycles per 2 mm voxel, 12 ms lobes 40 ms apart)",
                     "exp(-b D) with b from `gradients.b_value` of the same lobes, and b = q^2 (Delta - delta / 3) (Stejskal & Tanner 1965)",
                     max(abs(abs(r.signal[0]) - math.exp(-b * adc * 1e-3)),
                         abs(b / ((kappa * cyc) ** 2 * (big - delta / 3) * 1e-3) - 1)), 1e-6))
    rows.append(_row("E10b", "sequence", "`bd_of_b` for that lobe", "bd = D kappa^2 t",
                     abs(S.bd_of_b((kappa * cyc) ** 2 * delta * 1e-3 / 3, adc, cyc) / (unit * delta) - 1), 1e-12, unit="relative"))

    # E11 a designed pulse through the slice: one column per position, against the engine's slice integral
    err = 0.0
    for name, flip, tr in (("normal", 25.0, 30.0), ("sharp", 60.0, 12.0), ("fast", 12.0, 8.0)):
        dur, tbw = rf.PULSES[name]
        zeta, wq = rf.quadrature(tbw)
        pos = np.concatenate([-zeta[:0:-1], zeta]) * 5.0
        weights = np.concatenate([wq[:0:-1], wq]) / 2
        weights[zeta.size - 1] = wq[0]
        pulse = pulses.sinc(flip, dur, tbw, 5.0, n=rf.N_SAMPLES)
        cycle = [S.rf_pulse(pulse, pos, weights=weights), S.readout(), S.delay(tr), S.spoil()]
        r = S.run(S.repeat(cycle, int(25 * t1 / tr) + 5), t1, t2)
        ref = kspace.slice_signal(dict(sequence="spgr", tr_ms=tr), dict(t1_ms=t1), math.radians(flip), tbw, tier=1)[0][0]
        err = max(err, abs(r.signal[-1] - ref) / abs(ref))
    rows.append(_row("E11", "sequence", "spoiled gradient echo of a slice-selective pulse (fast, normal, sharp): one column per position of the "
                     "engine's quadrature, each with its own (a, b), summed", "the engine's slice integral `scanner/kspace.py::slice_signal` "
                     "(ideal spoiling)", err, 1e-9, unit="relative"))

    # E12 transients are conserved: the state as isochromats; and truncation
    blocks = [S.rf(35, 20), S.gradient(2), S.delay(7.0), S.rf(110, -40), S.gradient(-1), S.delay(3.0), S.rf(64, 170), S.gradient(3), S.delay(11.0)]
    steps = [("rf", 35, 20), ("grad", 2), ("delay", 7.0), ("rf", 110, -40), ("grad", -1), ("delay", 3.0), ("rf", 64, 170), ("grad", 3), ("delay", 11.0)]
    res = S.run(blocks, t1, t2)
    m = np.tile(np.array([0.0, 0.0, 1.0]), (64, 1))
    phases = 2 * np.pi * np.arange(64) / 64
    for step in steps:
        if step[0] == "rf":
            p = math.radians(step[2])
            m = m @ _rodrigues([math.cos(p), math.sin(p), 0.0], math.radians(step[1])).T
        elif step[0] == "grad":
            c, s = np.cos(step[1] * phases), np.sin(step[1] * phases)
            m = np.stack([c * m[:, 0] - s * m[:, 1], s * m[:, 0] + c * m[:, 1], m[:, 2]], 1)
        else:
            e1, e2 = math.exp(-step[1] / t1), math.exp(-step[1] / t2)
            m = np.stack([m[:, 0] * e2, m[:, 1] * e2, 1 + (m[:, 2] - 1) * e1], 1)
    err = 0.0
    for j in (0, 5, 17, 40):
        mxy, mz = S.magnetisation(res, j / 64)
        err = max(err, abs(mxy[0] - (m[j, 0] + 1j * m[j, 1])), abs(mz[0] - m[j, 2]))
    rows.append(_row("E12", "sequence", "the whole state after three pulses with positive and negative dephasing, read back as the magnetisation "
                     "of single isochromats", "the isochromat ensemble", err, 1e-12))

    # E13 speed
    blocks = [S.rf(90, 90)]
    for i in range(400):
        blocks += [S.delay(2.5), S.gradient(1), S.rf(150, 0), S.gradient(1), S.delay(2.5), S.readout(i)]
    best = min(_timed(lambda: S.run(blocks, t1, t2, kmax=64)) for _ in range(3))
    rows.append(_row("E13", "sequence", "time of a train of 400 refocusing pulses with 64 states (best of 3)", "requirement", best, 0.1, unit="s"))
    return rows


# ---------------------------------------------------------------------------------------------- gradients
def check_gradients():
    from ..scanner import console
    G = gradients
    rows, rng = [], np.random.default_rng(11)
    hw = G.Hardware.console()
    gamma = 2 * math.pi * GAMMA_HZ_PER_T

    # G1 trapezoid closed forms and the console's lobe
    err, sysd = 0.0, dict(console.SYSTEM)
    for area in (0.05, 0.5, 5.0, 50.0, 400.0):                  # mT/m ms
        tz = G.trapezoid(area, sysd["g_max_mt_m"] * sysd["derate"], sysd["slew_t_m_s"] * sysd["derate"])
        err = max(err, abs(tz["duration_ms"] / console._lobe_ms(area * 1e-6, sysd) - 1))
        t, g = np.array(tz["vertices"]).T
        err = max(err, abs(G.moments(t, g, 0.0, 0)[0, 0] / area - 1))
    rows.append(_row("G1", "gradients", "time-optimal triangle / trapezoid of 5 areas at the console's derated limits: duration and area",
                     "`scanner/console.py::_lobe_ms`; 2 sqrt(A / S) or A / G + G / S", err, 1e-12, unit="relative"))
    err, late = 0.0, 0.0
    for area in rng.uniform(0.05, 60.0, 40):
        tz = G.trapezoid(area, 40.0, 150.0, raster_ms=0.01)
        t, g = G.rasterise(G.program(x=tz["vertices"]))
        chk = G.check(t, g, G.Hardware(pns_limit_percent=1e9))
        err = max(err, abs(G.moments(t, g, 0.0, 0)[0, 0] / area - 1), 0.0 if chk["ok"] else 1.0)
        late = max(late, (tz["duration_ms"] - G.trapezoid(area, 40.0, 150.0)["duration_ms"]) / 0.01)
    rows.append(_row("G2", "gradients", "the trapezoid on the 10 us raster, 40 random areas: area, and amplitude and slew recomputed from the "
                     "rasterised waveform", "the requested area; within 40 mT/m and 150 T/m/s", err, 1e-12, unit="relative"))
    rows.append(_row("G2b", "gradients", "its duration beyond the closed form", "at most 3 raster steps, never shorter", late, 3.0, unit="raster steps"))

    # G3 moments of a trapezoid
    amp, ramp, flat, t0 = 23.0, 0.31, 1.27, 0.4
    t, g = G.rasterise(G.program(y=[(t0, 0), (t0 + ramp, amp), (t0 + ramp + flat, amp), (t0 + 2 * ramp + flat, 0)]))
    ref_t = 7.5
    centre = t0 + ramp + flat / 2
    m0 = amp * (flat + ramp)
    central = m0 * ((flat + ramp) ** 2 + ramp ** 2) / 12
    mom = G.moments(t, g, ref_t, 2)[:, 1]
    err = max(abs(mom[0] / m0 - 1), abs(mom[1] / (m0 * (centre - ref_t)) - 1), abs(mom[2] / (m0 * (centre - ref_t) ** 2 + central) - 1),
              abs(G.heating(t, g)[1] / (amp ** 2 * (flat + 2 * ramp / 3)) - 1),
              abs(G.k_trajectory(t, g)[1][-1, 1] / (CYCLES_PER_MM_PER_MT_M_MS * m0) - 1))
    rows.append(_row("G3", "gradients", "moments 0 to 2 of a trapezoid about a later time, its integral of G^2 and its final k",
                     "m0 = G (flat + ramp); m1 = m0 x distance; m2 = m0 x distance^2 + m0 ((flat + ramp)^2 + ramp^2) / 12; G^2 (flat + 2 ramp / 3); "
                     "gamma m0", err, 1e-12, unit="relative"))

    # G4 flow-compensated lobes
    def train(amps, ramp=0.2, flat=1.0):
        v, start = [(0.0, 0.0)], 0.0
        for a in amps:
            v += [(start + ramp, a), (start + ramp + flat, a), (start + 2 * ramp + flat, 0.0)]
            start += 2 * ramp + flat
        return v

    t, g = G.rasterise(G.program(z=train((10, -20, 10))))
    m121 = G.moments(t, g, t[-1] + 3.0, 2)[:, 2]
    t, g = G.rasterise(G.program(z=train((5, -15, 15, -5))))
    m1331 = G.moments(t, g, t[-1] + 3.0, 3)[:, 2]
    scale = 10 * 1.2
    rows.append(_row("G4", "gradients", "binomial lobe trains about an echo 3 ms later: 1:-2:1 (m0, m1) and 1:-3:3:-1 (m0, m1, m2)",
                     "zero (velocity and acceleration compensation); the next moment is not zero",
                     max(abs(m121[0]), abs(m121[1]), abs(m1331[0]), abs(m1331[1]), abs(m1331[2])) / scale, 1e-12, unit="of one lobe's area"))
    rows.append(_row("G4b", "gradients", "the first moment that is left: m2 of 1:-2:1 and m3 of 1:-3:3:-1", "not zero",
                     min(abs(m121[2]), abs(m1331[3])), 1.0, kind="min", unit="mT/m ms^(n+1)"))

    # G5 Stejskal-Tanner
    amp, delta, big, zeta = 40.0, 20.0, 40.0, 0.4               # delta = plateau + ramp, big = start to start
    closed = gamma ** 2 * (amp * 1e-3) ** 2 * ((delta * 1e-3) ** 2 * (big * 1e-3 - delta * 1e-3 / 3) + (zeta * 1e-3) ** 3 / 30
                                               - delta * 1e-3 * (zeta * 1e-3) ** 2 / 6) * 1e-6

    def lobe(start, sign):
        return [(start, 0.0), (start + zeta, sign * amp), (start + delta, sign * amp), (start + delta + zeta, 0.0)]

    t, g = G.rasterise(G.program(x=lobe(0.0, 1) + lobe(big, 1)))
    mono = G.b_value(t, g, refocus_ms=(30.0,))
    k_end = G.k_trajectory(t, g, refocus_ms=(30.0,))[1][-1, 0]
    t, g = G.rasterise(G.program(x=lobe(0.0, 1) + lobe(big, -1)))
    bip = G.b_value(t, g)
    rows.append(_row("G5", "gradients", "b-value of a trapezoid pair (40 mT/m, 20 ms, 40 ms apart, 0.4 ms ramps): two equal lobes around a "
                     "refocusing pulse, and two opposed lobes without one", "gamma^2 G^2 [delta^2 (Delta - delta/3) + ramp^3/30 - delta ramp^2/6] "
                     f"= {closed:.2f} s/mm^2 (Stejskal & Tanner 1965 with ramps)", max(abs(mono / closed - 1), abs(bip / closed - 1), abs(k_end)),
                     1e-12, unit="relative"))
    d, tiny = 10.0, 1e-9
    t = np.array([0, tiny, d, d + 2 * tiny, 2 * d, 2 * d + tiny])
    pair = np.array([0, amp, amp, -amp, -amp, 0.0])
    one = (2 / 3) * gamma ** 2 * (amp * 1e-3) ** 2 * (d * 1e-3) ** 3 * 1e-6
    t4 = np.concatenate([t, t[1:] + 2 * d + tiny])
    four = np.concatenate([pair, -pair[1:]])
    rows.append(_row("G5b", "gradients", "rectangular lobes (ramps of 1e-9 ms): an adjacent bipolar pair, and the velocity-compensated double pair "
                     "+ - - +", "(2/3) gamma^2 G^2 delta^3 and twice that; first moment of the double pair zero",
                     max(abs(G.b_value(t, pair) / one - 1), abs(G.b_value(t4, four) / (2 * one) - 1),
                         abs(G.moments(t4, four, 0.0, 1)[1, 0]) / (amp * d * d)), 1e-8, unit="relative"))

    # G6 b-tensor of a random three-axis program with two refocusing pulses, against numerical quadrature
    verts = {}
    for axis in "xyz":
        times = np.sort(rng.choice(np.arange(1, 6000), 14, replace=False)) * 0.01
        verts[axis] = [(0.0, 0.0)] + [(float(tt), float(rng.uniform(-35, 35))) for tt in times] + [(60.0, 0.0)]
    t, g = G.rasterise(G.program(**verts))
    refocus = (17.303, 41.0)
    tensor = G.b_tensor(t, g, refocus)
    fine = np.linspace(0, 60.0, 600001)
    cut = np.array(refocus)
    region = np.searchsorted(cut, fine, side="right")           # F = sign x (running integral of G) + a constant per region
    run_int = scipy.integrate.cumulative_trapezoid(np.stack([np.interp(fine, t, g[:, i]) for i in range(3)], 1), fine, axis=0, initial=0)
    at_cut = np.stack([np.interp(cut, fine, run_int[:, i]) for i in range(3)], 1)
    f = run_int.copy()
    f[region == 1] = 2 * at_cut[0] - run_int[region == 1]
    f[region == 2] = 2 * at_cut[0] - 2 * at_cut[1] + run_int[region == 2]
    quad = B_PER_MT_M_MS_SQUARED_MS * np.array([[scipy.integrate.simpson(f[:, i] * f[:, j], x=fine) for j in range(3)] for i in range(3)])
    rows.append(_row("G6", "gradients", "b-tensor of a random three-axis program (16 vertices per axis, 60 ms, two refocusing pulses, one of them "
                     "between raster times)", "numerical quadrature on 600,001 points (`scipy.integrate`: cumulative trapezoid, then Simpson)", np.abs(tensor - quad).max() / np.trace(quad),
                     1e-6, unit="of the trace"))
    u = np.array([0.6, 0.0, 0.8])
    t1d, g1d = G.rasterise(G.program(x=lobe(0.0, 1) + lobe(big, 1)))
    b1d = G.b_value(t1d, g1d, (30.0,))
    rows.append(_row("G6b", "gradients", "b-tensor of one waveform played along the direction (0.6, 0, 0.8)", "b u u^T",
                     np.abs(G.b_tensor(t1d, g1d[:, :1] * u[None, :], (30.0,)) - b1d * np.outer(u, u)).max() / b1d, 1e-12, unit="relative"))

    # G7 limits recomputed from the waveform
    t, g = G.rasterise(G.program(x=[(0, 0), (0.2, 30.0), (1.0, 30.0), (1.2, 0)], y=[(0.5, 0), (0.6, 28.0), (0.9, 28.0), (1.3, 0)]))
    axis_rule, vector_rule = G.check(t, g, G.Hardware(pns_limit_percent=1e9)), G.check(t, g, G.Hardware(rule="vector", pns_limit_percent=1e9))
    ok = (not axis_rule["ok"] and [v["code"] for v in axis_rule["violations"]] == ["SLEW_LIMIT"] and axis_rule["violations"][0]["axis"] == "y"
          and abs(axis_rule["violations"][0]["t_ms"] - 0.5) < 1e-9 and abs(axis_rule["slew_t_m_s"]["y"] - 280.0) < 1e-6
          and abs(axis_rule["slew_t_m_s"]["x"] - 150.0) < 1e-6 and abs(axis_rule["peak_mt_m"]["x"] - 30.0) < 1e-9
          and [v["code"] for v in vector_rule["violations"]] == ["GRADIENT_LIMIT", "SLEW_LIMIT"]
          and abs(vector_rule["peak_vector_mt_m"] - math.hypot(30.0, 28.0)) < 1e-6 and vector_rule["violations"][0]["axis"] == "vector")
    off = G.check(*G.rasterise(G.program(x=[(0.003, 0), (0.203, 20), (0.403, 0)])), G.Hardware(pns_limit_percent=1e9))
    step = G.check(*G.rasterise(G.program(x=[(0, 5.0), (0.5, 5.0)])), G.Hardware(pns_limit_percent=1e9))
    ok = ok and off["ok"] and not G.on_raster(G.program(x=[(0.003, 0), (0.203, 20)])) and [v["code"] for v in step["violations"]] == ["NOT_ZERO_AT_ENDS"]
    rows.append(_row("G7", "gradients", "limit check of a two-axis program: per-axis rule names the slew of y at 0.5 ms (280 T/m/s); the vector "
                     "rule also names the amplitude (41.04 mT/m); a program that does not return to zero is named", "by construction", 0.0 if ok else 1.0,
                     0.0))
    err = 0.0
    for amp, ramp in ((40.0, 40.0 / 150.0), (20.0, 0.4), (8.0, 0.05)):
        t, g = G.rasterise(G.program(x=[(0, 0), (ramp, amp), (ramp + 1.0, amp), (2 * ramp + 1.0, 0)]), raster_ms=ramp / 40)
        pct, _ = G.pns_percent(t, g, hw)
        ref = 100 * (amp / ramp) * hw.pns_radius_m / (hw.pns_rheobase_t_s * (1 + hw.pns_chronaxie_ms / ramp))
        err = max(err, abs(pct[0] / ref - 1))
    t, g = G.rasterise(G.program(x=[(0, 0), (0.2, 30.0), (0.6, -30.0), (0.8, 0)]))
    ref = 100 * 150.0 * hw.pns_radius_m / (hw.pns_rheobase_t_s * (1 + hw.pns_chronaxie_ms / 0.4))
    err = max(err, abs(G.pns_percent(t, g, hw)[0][0] / ref - 1))
    rows.append(_row("G8", "gradients", "stimulation figure (declared model) of trapezoid ramps and of a full reversal", "100 x slew x radius / "
                     "(rheobase (1 + chronaxie / ramp time)); a reversal is one ramp of twice the time", err, 1e-9, unit="relative"))

    # G9 the linear programme against the closed forms
    worst, below, resid, limits = 0.0, 0.0, 0.0, 0.0
    gm, sl, dt = 40.0, 150.0, 0.01
    for area in (0.5, 3.0, 5.0, 12.0, 30.0):
        res = G.optimal_lobe({0: area}, gm, sl)
        closed = G.trapezoid(area, gm, sl)["duration_ms"]
        worst = max(worst, abs(res["steps"] - math.ceil(closed / dt - 1e-9)))
        below = max(below, (closed - res["duration_ms"]) / dt)
        resid = max(resid, res["residual"] / area)
        limits = max(limits, 0.0 if G.check(res["t_ms"], res["g_mt_m"], G.Hardware(pns_limit_percent=1e9))["ok"] else 1.0)
    for m1 in (0.5, 2.0, 20.0, 60.0):
        a = (m1 * math.sqrt(sl) / 2) ** (2 / 3)                # two opposed triangles of area a, 2 sqrt(a / S) apart
        if math.sqrt(a * sl) <= gm:
            closed = 4 * math.sqrt(a / sl)
        else:                                                   # two opposed trapezoids at Gmax: a (a / G + G / S) = m1
            a = (-gm / sl + math.sqrt(gm ** 2 / sl ** 2 + 4 * m1 / gm)) * gm / 2
            closed = 2 * (a / gm + gm / sl)
        res = G.optimal_lobe({0: 0.0, 1: m1}, gm, sl)
        worst = max(worst, abs(res["steps"] - math.ceil(closed / dt - 1e-9)))
        below = max(below, (closed - res["duration_ms"]) / dt)
        resid = max(resid, abs(res["moments"][1] - m1) / m1, abs(res["moments"][0]) / (gm * res["duration_ms"]))
        limits = max(limits, 0.0 if G.check(res["t_ms"], res["g_mt_m"], G.Hardware(pns_limit_percent=1e9))["ok"] else 1.0)
    rows.append(_row("G9", "gradients", "shortest lobe by linear programming: area only (5 areas, triangle and trapezoid regime) and zero area "
                     "with a first moment (4 values, both regimes)", "closed forms rounded up to the raster: 2 sqrt(A / S), A / G + G / S; two opposed "
                     "triangles 4 sqrt(a / S) with 2 a^(3/2) / sqrt(S) = m1, or two opposed trapezoids at Gmax", worst, 0.0, unit="raster steps"))
    rows.append(_row("G9b", "gradients", "the same results: shorter than the closed form of the continuous problem", "never (a waveform on the "
                     "raster is a waveform)", below, 1e-9, unit="raster steps"))
    rows.append(_row("G9c", "gradients", "the same results: moments recomputed from the returned samples; amplitude and slew rechecked",
                     "the targets; the limits", max(resid, limits), 1e-9, unit="relative"))
    res = G.optimal_lobe({0: 5.0, 1: 0.0}, gm, sl, offset_ms=1.0)
    plain = G.optimal_lobe({0: 5.0}, gm, sl)
    shorter = G._lobe_lp(res["steps"] - 1, {0: 5.0, 1: 0.0}, gm, sl, dt, "end", 1.0)
    rows.append(_row("G9d", "gradients", "flow-compensated lobe (area 5 mT/m ms, first moment zero at an echo 1 ms after it): residual moments; one "
                     "raster step shorter is infeasible; it is longer than the lobe without compensation", "the targets; the certificate",
                     max(res["residual"] / 5.0, 0.0 if shorter is None else 1.0, 0.0 if res["steps"] > plain["steps"] else 1.0), 1e-9))
    return rows


# ---------------------------------------------------------------------------------------------- run and print
CHECKS = dict(rotation=check_rotation, pulses=check_pulses, profile=check_profile, sequence=check_sequence, gradients=check_gradients)


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
    """The validation table of docs/physics/design.md. Errors are printed with one significant digit below 1e-3:
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
            print(f"{r['id']:5s} {'pass' if r['ok'] else 'FAIL'}  {_number(r['measured']):>9s} {sign} {_number(r['tolerance']):<8s} {r['unit']:22s} {r['quantity'][:110]}")
        print(f"{sum(r['ok'] for r in rows)} of {len(rows)} rows pass; {seconds:.1f} s")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
