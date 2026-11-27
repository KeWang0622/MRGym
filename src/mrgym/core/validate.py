"""Validation of the signal core: every number against a closed form or an independent computation.

    python -m mrgym.core.validate                 the table as text; exit status 1 if a row fails
    python -m mrgym.core.validate --markdown      the table of docs/physics/core.md
    python -m mrgym.core.validate --speed         the speed table (not part of the rows; machine dependent)
    python -m mrgym.core.validate agreement       one group only (agreement, engine, sequences)

Groups
  agreement   the three evaluators against each other on randomised block lists (seeded), the conventions against
              ``mrgym.design`` and ``mrgym.scanner``, diffusion against a grid integration of the Bloch-Torrey equation
  engine      every sequence the scanner serves, written as a block list, against the engine's own numbers
              (``scanner/signal.py``, ``scanner/epg.py``, ``scanner/kspace.py``): the protocols and tissues of rows S1 to
              S15 and R1 to R12 of ``mrgym validate``. Where the engine itself approximates, the row says which side
              is exact. Rows Q16 to Q16d: the fat-saturation pulse of engine 0.3.0 as an event of the block list.
  sequences   the sequences the scanner does not serve, each against at least two independent checks: closed forms
              (their source is named, with "paper not opened" where it was not read for this work), the isochromat ensemble, the
              engine's or the design layer's separately written code

"Independent" means sharing no code with the function under test. The references were written by the same authors
as the code: the rows show that the code computes the stated equations, not that the equations describe a scanner.
"""
from __future__ import annotations

import itertools
import math
import sys
import time

import numpy as np
import scipy.linalg

from ..design import gradients as design_gradients
from ..design import rotation as design_rotation
from ..design import sequence as design_sequence
from ..design.units import GAMMA_HZ_PER_T
from . import bloch, epg, periodic
from . import blocks as B
from . import library as L
from . import operators as O

GROUPS = dict(agreement="The evaluators against each other, and the conventions", engine="The six served sequences against the engine",
              sequences="Sequences the scanner does not serve")
D = math.radians


def _row(id, group, quantity, reference, measured, tolerance, kind="max", unit=""):
    """kind "max": passes when measured <= tolerance; "min": when measured >= tolerance."""
    measured = float(measured)
    ok = bool(np.isfinite(measured) and (measured <= tolerance if kind == "max" else measured >= tolerance))
    return dict(id=id, group=group, quantity=quantity, reference=reference, measured=measured, tolerance=float(tolerance), kind=kind,
                unit=unit, ok=ok)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def _order(errors):
    return float(np.mean([math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]))


def _tissues(b0=3.0, names=("wm", "gm", "csf", "scalp_fat", "muscle", "blood")):
    from ..scanner import tissue
    tab = tissue.table(b0)
    return [dict(tab[n], name=n) for n in names]


# ---------------------------------------------------------------------------------------------- random block lists
def _random_spins(rng, n=5, relax=(300.0, 3000.0, 20.0, 300.0), df=40.0, **kw):
    t1 = rng.uniform(relax[0], relax[1], n)
    return O.spins(t1, np.minimum(rng.uniform(relax[2], relax[3], n), t1), df_hz=rng.uniform(-df, df, n), b1=rng.uniform(0.7, 1.3, n),
                   m0=rng.uniform(0.5, 1.5, n), **kw)


def _random_blocks(rng, n, cycles=(-2, -1, 1, 2, 3), axes=1, ideal=True, with_adc=True):
    """A random block list: hard pulses of any angle and phase, crushed pulses, delays, gradients with and without a
    duration, ideal spoilers and crushers, Mz preparations; an adc after every block."""
    out = []
    for _ in range(n):
        u = rng.uniform()
        if u < 0.36:
            out.append(B.rf(rng.uniform(5, 180), rng.uniform(-180, 180), crushed=bool(ideal and rng.uniform() < 0.12)))
        elif u < 0.58:
            out.append(B.delay(rng.uniform(0.5, 25.0)))
        elif u < 0.90:
            cyc = [0.0] * axes
            cyc[rng.integers(axes)] = float(rng.choice(cycles))
            out.append(B.gradient(tuple(cyc), ms=float(rng.choice([0.0, rng.uniform(0.5, 8.0)]))))
        elif not ideal:
            out.append(B.delay(rng.uniform(0.5, 5.0)))
        elif u < 0.94:
            out.append(B.spoil())
        elif u < 0.97:
            out.append(B.crush())
        else:
            out.append(B.mz(rng.uniform(-1, 1)))
        if with_adc:
            out.append(B.adc(len(out), phase_deg=rng.uniform(-180, 180)))
    return out


# ================================================================================================== agreement
def check_agreement():
    from ..scanner import rf as engine_rf
    rows = []
    rng = np.random.default_rng(20261126)

    # C1 the conventions: a hard pulse from equilibrium, in the three evaluators and in the design layer
    err = 0.0
    for _ in range(40):
        al, p, df, t = rng.uniform(0, 180), rng.uniform(-180, 180), rng.uniform(-200, 200), rng.uniform(0.1, 5)
        sp = O.spins(1e18, 1e18, df_hz=df)
        seq = [B.rf(al, p), B.adc("a", 0.0), B.delay(t), B.adc("b")]
        ref = -1j * np.exp(1j * D(p)) * math.sin(D(al)) * np.array([1.0, np.exp(2j * math.pi * df * t * 1e-3)])
        a, b = design_rotation.axis_angle([math.cos(D(p)), math.sin(D(p)), 0.0], D(al))
        op = B.compile_ops(seq).ops[0]
        err = max(err, abs(O.rotation(op, sp)[0][0] - a), abs(O.rotation(op, sp)[1][0] - b))
        for ev in (epg.run, bloch.run):
            err = max(err, np.abs(ev(seq, sp, demodulation=1.0).signal[:, 0] - ref).max())
        err = max(err, abs(epg.run(seq, sp).signal[0, 0] - 1j * ref[0]))
    rows.append(_row("C1", "agreement", "a hard pulse of flip angle a and phase p from equilibrium, then free precession at df (40 random cases; "
                     "both evaluators; the Cayley-Klein parameters against `design.rotation.axis_angle`)",
                     "Mxy = -i exp(i p) sin a, then x exp(+2 pi i df t); the default demodulation +i makes a pulse about x real and positive",
                     err, 1e-12))

    # C2 selective columns against the engine's spinor cascade, and the quadrature
    err = 0.0
    for _, tbw in engine_rf.PULSES.values():
        zeta, w = O.slice_columns(tbw, two_sided=False)
        zq, wq = engine_rf.quadrature(tbw)
        err = max(err, np.abs(zeta - zq).max(), np.abs(w - wq).max())
        z2, w2 = O.slice_columns(tbw)
        err = max(err, abs(w2.sum() - wq.sum()), np.abs(z2 + z2[::-1]).max())
        for window in ("hann", "hamming"):
            err = max(err, abs(O.slice_calibration(tbw, window) - engine_rf.fwhm_factor(tbw, window)))
            for deg in (5, 30, 90, 120, 180):
                a, b = O.selective_rotation(B.Selective(tbw, window), D(deg), z2)
                a0, b0 = engine_rf.cayley_klein(D(deg), tbw, z2, window=window)
                ar, br = O.selective_rotation(B.Selective(tbw, window, rephase=0.0, width=engine_rf.REFOCUS_WIDTH), D(deg), z2)
                ar0, br0 = engine_rf.cayley_klein(D(deg), tbw, z2 / engine_rf.REFOCUS_WIDTH, rephase=0.0, window=window)
                err = max(err, np.abs(a - a0).max(), np.abs(b - b0).max(), np.abs(ar - ar0).max(), np.abs(br - br0).max())
    rows.append(_row("C2", "agreement", "slice columns: the calibration of the slice gradient (nominal bandwidths across the width at half "
                     "maximum of the small-flip-angle profile, by bisection on the transform of the design layer's pulse), positions and "
                     "weights of the integral through the slice; the rotation of the windowed sinc at every column (fast, normal, sharp; Hann, "
                     "Hamming; 5 to 180 deg; excitation and the refocusing pulse on the 1.5 x slice), computed with `design.rotation.rotate`",
                     "`scanner/rf.py::fwhm_factor`, `quadrature` and `cayley_klein` (rows R1 to R3 and R13 of `mrgym validate` check that "
                     "cascade and that calibration)", err, 1e-12))

    # C3 the design layer's phase graph on its own block lists
    S, err = design_sequence, 0.0
    for _ in range(12):
        blocks = []
        for i in range(30):
            u = rng.uniform()
            if u < 0.4:
                blocks.append(S.rf(rng.uniform(5, 180), rng.uniform(-180, 180)))
            elif u < 0.6:
                blocks.append(S.delay(rng.uniform(0.5, 20)))
            elif u < 0.93:
                blocks.append(S.gradient(int(rng.choice([-2, -1, 1, 2])), ms=float(rng.choice([0.0, 3.0]))))
            else:
                blocks.append(S.spoil())
            blocks.append(S.readout(i, rng.uniform(-180, 180)))
        t1, t2, df, sc = 900.0, 70.0, 23.0, 0.93
        ref = S.run(blocks, t1, t2, df_hz=df, b1_scale=sc).signal
        err = max(err, np.abs(epg.run(B.from_design(blocks), O.spins(t1, t2, df_hz=df, b1=sc)).signal[:, 0] - ref).max())
    rows.append(_row("C3", "agreement", "12 random block lists of `mrgym.design.sequence` (30 blocks: pulses, delays, gradients, ideal spoilers), "
                     "converted with `blocks.from_design`", "`design.sequence.run`, the time-stepping phase graph of the design layer (separate code)",
                     err, 1e-12))

    # X1 Bloch ensemble against the phase graph: whole cycles
    err = err_state = 0.0
    n_lists = 40
    for _ in range(n_lists):
        sp = _random_spins(rng)
        seq = _random_blocks(rng, 24)
        e, r = epg.run(seq, sp), bloch.run(seq, sp)
        err = max(err, np.abs(e.signal - r.signal).max())
        mxy, mz = epg.magnetisation(e.state, r.state["x"])
        err_state = max(err_state, np.abs(mxy - (r.state["m"][0] + 1j * r.state["m"][1])[:, 0]).max(), np.abs(mz - r.state["m"][2][:, 0]).max())
    rows.append(_row("X1", "agreement", f"extended phase graph, {n_lists} random block lists of 24 blocks x 5 columns (hard and crushed pulses, "
                     "delays, gradients of -2 to 3 cycles, ideal spoilers and crushers, Mz preparations; random T1, T2, off-resonance, B1, M0): "
                     "every sample", "isochromat ensemble (`core.bloch`: real 3 x 3 rotations, as many isochromats as the gradients can reach "
                     "orders + 1)", err, 1e-10))
    rows.append(_row("X1b", "agreement", "the same lists: the final state read back as the magnetisation of every isochromat",
                     "the isochromats of the ensemble", err_state, 1e-10))

    # X2 fractional cycles on two axes: the state exactly, the voxel mean to the accuracy of the ensemble
    err_state, errs = 0.0, []
    for case in range(10):
        sp = _random_spins(rng, 3)
        seq = _random_blocks(rng, 14, cycles=(-0.5, 0.25, 0.75, 1.5), axes=2, ideal=False)
        e = epg.run(seq, sp)
        r = bloch.run(seq, sp, n_sub=(12, 12))
        mxy, mz = epg.magnetisation(e.state, r.state["x"])
        err_state = max(err_state, np.abs(mxy - (r.state["m"][0] + 1j * r.state["m"][1])[:, 0]).max(), np.abs(mz - r.state["m"][2][:, 0]).max())
        if case < 2:
            errs.append([np.abs(e.signal - bloch.run(seq, sp, n_sub=(n, n)).signal).max() for n in (24, 48, 96)])
    rows.append(_row("X2", "agreement", "fractional dephasing (quarter cycles) on two gradient axes, 10 random lists: the final state as the "
                     "magnetisation of 12 x 12 isochromats", "the isochromat ensemble", err_state, 1e-10))
    rows.append(_row("X2b", "agreement", "the same: the sample sum_k F_k sinc(k) against the mean over 24, 48, 96 isochromats per axis: order of "
                     "convergence", "second order in the spacing of the isochromats (midpoint rule)", abs(np.mean([_order(e) for e in errs]) - 2), 0.1,
                     unit="in the order"))
    rows.append(_row("X2c", "agreement", "its error at 96 isochromats per axis", "the mean over the ensemble", max(e[-1] for e in errs), 2e-3))

    # X3 laboratory frame against pulse frame
    err = err_state = 0.0
    for _ in range(20):
        sp = _random_spins(rng, 4)
        head = [B.rf(flip_table="a", phase_deg=rng.uniform(-180, 180)), B.adc("x"), B.delay(rng.uniform(0.5, 3)), B.rf(rng.uniform(5, 90), rng.uniform(-180, 180)),
                B.adc("y", rng.uniform(-180, 180))]
        net = int(rng.choice([-2, -1, 1, 2, 3]))
        tail = [B.gradient(net + 1, ms=rng.uniform(1, 5)), B.adc("w", rng.uniform(-180, 180)), B.delay(rng.uniform(0, 4)),
                B.gradient(-1, ms=float(rng.choice([0.0, 2.0])))]
        rep = B.repeat(head + tail, 25, phase_inc_deg=rng.uniform(-180, 180), spoil_inc_deg=float(rng.choice([117.0, 50.0, rng.uniform(1, 179)])),
                       tables=dict(a=tuple(rng.uniform(5, 120, int(rng.integers(1, 4))))))
        seq = [B.rf(rng.uniform(10, 170), rng.uniform(-180, 180)), B.gradient(1), rep, B.delay(3.0), B.rf(40, 10), B.adc("z")]
        lab, fr = epg.run(seq, sp), epg.run(seq, sp, frame=True)
        err = max(err, np.abs(lab.signal - fr.signal).max())
        err_state = max(err_state, np.abs(lab.state.f - fr.state.f).max(), np.abs(lab.state.z - fr.state.z).max())
    rows.append(_row("X3", "agreement", "RF phase increments in the pulse frame (state turned and translated after every repetition): 20 random "
                     "repeats of 25 repetitions with linear and quadratic increments, flip-angle tables, two pulses per repetition, gradients of "
                     "both signs and a sample away from zero dephasing, inside a longer list: every sample and the final state", "the same lists with the increments added to the pulse "
                     "and receiver phases (laboratory frame)", max(err, err_state), 1e-10))

    # X4 the periodic solve against stepping
    err, err_b, n_cases = 0.0, 0.0, 24
    for case in range(n_cases):
        sp = _random_spins(rng, 4, relax=(25.0, 60.0, 8.0, 25.0), df=60.0)
        tabs = dict(a=tuple(rng.uniform(5, 120, int(rng.integers(1, 4)))))
        if case % 3 == 0:                                   # balanced: no gradients, linear phase cycle
            body = [B.rf(flip_table="a"), B.delay(rng.uniform(1, 3)), B.adc("x"), B.delay(rng.uniform(1, 3)), B.rf(rng.uniform(5, 60), 30.0), B.delay(1.0),
                    B.adc("y")]
            rep = B.repeat(body, 1, phase_inc_deg=rng.uniform(-180, 180), tables=tabs)
        else:                                               # unbalanced with RF spoiling
            body = [B.rf(flip_table="a"), B.adc("x"), B.delay(rng.uniform(0.5, 2)), B.rf(rng.uniform(5, 60), 30.0), B.adc("y"),
                    B.gradient(int(rng.choice([2, 3])), ms=rng.uniform(1, 4)), B.delay(rng.uniform(0, 2)), B.gradient(-1, ms=1.0)]
            rep = B.repeat(body, 1, phase_inc_deg=rng.uniform(-180, 180), spoil_inc_deg=float(rng.choice([0.0, 117.0, rng.uniform(1, 179)])), tables=tabs)
        kmax = 14
        fixed = periodic.steady_state([rep], sp, kmax=kmax, method="matrix")
        reps = len(tabs["a"])
        n = fixed.signal.shape[0]
        stepped = epg.run([rep], sp, kmax=kmax, frame=True, repetitions=reps * 260)
        err = max(err, np.abs(stepped.signal[-n:] - fixed.signal).max())
        if case < 6:                                        # the laboratory frame and the ensemble reach the same samples
            lab = epg.run([B.repeat(rep.body, reps * 260, rep.phase_inc_deg, rep.spoil_inc_deg, dict(rep.tables))], sp, kmax=kmax)
            err_b = max(err_b, np.abs(lab.signal[-n:] - fixed.signal).max())
    rows.append(_row("X4", "agreement", f"the periodic state as a fixed point (sparse linear solve on the phase-graph states, all columns at once): "
                     f"{n_cases} random periods x 4 columns: balanced with a linear phase cycle, and unbalanced with quadratic RF spoiling; flip-angle "
                     "tables of 1 to 3 repetitions, two pulses per repetition, 14 orders kept", "the phase graph stepped through 260 periods from "
                     "equilibrium (T1 25 to 60 ms, so that it has converged), same truncation", max(err, err_b), 1e-10))

    # X5 reduced (Mz only) against the matrix method and against stepping
    err = 0.0
    for _ in range(12):
        sp = _random_spins(rng, 4, relax=(80.0, 400.0, 20.0, 80.0))
        body = _random_blocks(rng, 10, cycles=(-1, 1, 2), ideal=False) + [B.delay(rng.uniform(20, 200)), B.crush()]
        rep = [B.repeat(body, 1)]
        red = periodic.steady_state(rep, sp)
        mat = periodic.steady_state(rep, sp, kmax=B.compile_ops(rep).reach, method="matrix")
        half = periodic.steady_state(rep, sp, kmax=red.state.kmax, method="matrix")
        stepped = epg.run(rep, sp, repetitions=250)
        n = red.signal.shape[0]
        err = max(err, np.abs(red.signal - mat.signal).max(), np.abs(red.signal - stepped.signal[-n:]).max(), np.abs(red.state.z - half.state.z).max(),
                  np.abs(red.state.f - half.state.f).max())
    rows.append(_row("X5", "agreement", "a period that ends with an ideal crusher: the fixed point of Mz from two runs of the period that carry only "
                     "the orders up to half its reach (`reduced`), 12 random lists of 10 blocks: samples and the state at the start of the period",
                     "the matrix method on all states the gradients can reach, and the phase graph stepped through 250 periods", err, 1e-10))

    # X6 two pools
    err_mt = err_cx = 0.0
    for case in range(16):
        n = 3
        kw = dict(f_b=rng.uniform(0.05, 0.2, n), k_ab_per_s=rng.uniform(1.0, 6.0, n), t1b_ms=rng.uniform(300, 1200, n))
        if case % 2 == 0:
            sp = _random_spins(rng, n, pool="mt", lineshape="gaussian", t2b_ms=rng.uniform(0.008, 0.015, n), **kw)
            seq = []
            for blk in _random_blocks(rng, 16):
                seq.append(B.rf(blk.flip_deg, blk.phase_deg, crushed=blk.crushed, energy_ut2_ms=rng.uniform(0, 400)) if blk.kind == "rf" else blk)
                if rng.uniform() < 0.1:
                    seq.append(B.rf(energy_ut2_ms=rng.uniform(100, 3000), offset_hz=rng.uniform(500, 3000)))
        else:
            kw["k_ab_per_s"] = rng.uniform(20, 300, n)
            sp = _random_spins(rng, n, pool="cx", t2b_ms=rng.uniform(5, 40, n), shift_b_hz=rng.uniform(-300, 300, n), **kw)
            seq = _random_blocks(rng, 16)
        e, r = epg.run(seq, sp), bloch.run(seq, sp)
        d = np.abs(e.signal - r.signal).max()
        body = [b for b in seq if b.kind not in ("crush", "spoil", "mz")][:10] + [B.gradient(2, ms=3.0)]
        if not _reaches_zero(body):
            body = [B.rf(40, 0), B.adc("x"), B.delay(2.0), B.rf(30, 60), B.adc("y"), B.gradient(2, ms=3.0)]
        sp_fast = O.spins(**{**_as_kwargs(sp), "t1_ms": sp.t1_ms / 30, "t2_ms": sp.t2_ms / 4, "t1b_ms": sp.t1b_ms / 30})
        rep = [B.repeat(body, 1, spoil_inc_deg=117.0)]
        fixed = periodic.steady_state(rep, sp_fast, kmax=10, method="matrix")
        stepped = epg.run(rep, sp_fast, kmax=10, frame=True, repetitions=300).signal[-fixed.signal.shape[0]:]
        centred = np.array([op.at_zero for op in fixed.program.ops if op.kind == "adc"])[:, None]      # elsewhere only the magnitude is periodic
        d = max(d, np.where(centred, np.abs(stepped - fixed.signal), np.abs(np.abs(stepped) - np.abs(fixed.signal))).max())
        if case % 2 == 0:
            err_mt = max(err_mt, d)
        else:
            err_cx = max(err_cx, d)
    rows.append(_row("X6", "agreement", "two pools, magnetisation transfer (bound pool saturated by every pulse and by off-resonance pulses, "
                     "exchange during every delay): 8 random lists x 3 columns in the phase graph, and their periodic state by the matrix method",
                     "isochromat ensemble with `scipy.linalg.expm` of the exchange generator; the phase graph stepped through 300 periods",
                     err_mt, 1e-10))
    rows.append(_row("X6b", "agreement", "two pools, chemical exchange (both pools transverse, a frequency shift between them): 8 random lists",
                     "the same two references", err_cx, 1e-10))

    # X7 diffusion against a grid integration of the Bloch-Torrey equation
    seq = B.sequence([B.rf(70, 20), B.gradient(2, ms=6.0), B.delay(4.0), B.rf(150, 100), B.gradient(3, ms=5.0), B.rf(60, -40), B.gradient(-2, ms=7.0),
                      B.delay(2.0), B.rf(120, 10), B.gradient(1, ms=3.0), B.adc("a"), B.gradient(-2, ms=4.0), B.adc("b")], cycle_mm=0.06)
    sp = O.spins([900.0, 1500.0], [80.0, 200.0], adc_um2_ms=[1.0, 3.0], df_hz=[10.0, -25.0])
    e = epg.run(seq, sp)
    free = epg.run(B.sequence(seq.blocks), sp)
    errs = [np.abs(bloch.run(seq, sp, diffusion_substeps=m).signal - e.signal).max() for m in (8, 16, 32, 64)]
    rows.append(_row("X7", "agreement", "free diffusion in the phase graph (attenuation of every state by its own order during every block; "
                     "four pulses, gradients of both signs, D 1 and 3 um^2/ms, 60 um per cycle): order of convergence of the reference towards "
                     "it, 8 to 64 substeps per block", "Bloch-Torrey equation on the grid of isochromats: a Gaussian blur in the Fourier "
                     "domain of the grid, the gradient advanced between half steps (second order); no closed-form factor is used", abs(_order(errs) - 2), 0.1, unit="in the order"))
    rows.append(_row("X7b", "agreement", "its distance at 64 substeps", "the same reference", errs[-1], 5e-5))
    rows.append(_row("X7c", "agreement", "size of the effect in that test: largest change of a sample by diffusion", "it must be large for the test "
                     "to mean something", np.abs(e.signal - free.signal).max(), 0.02, kind="min"))

    # X8 T2' as an ensemble of static off-resonances
    t2p, te = 30.0, 12.0
    sp = O.spins(1000.0, 80.0, t2p_ms=t2p)
    gre = [B.rf(90), B.delay(te), B.adc("g", static_ms=te), B.delay(te), B.adc("g2", static_ms=2 * te)]
    spin = [B.rf(90, 0), B.delay(te), B.rf(180, 90), B.delay(te), B.adc("s", static_ms=0.0)]
    errs = [np.abs(bloch.run(gre, sp, spread=O.lorentzian_spread(n)).signal - epg.run(gre, sp).signal).max() for n in (500, 2000, 8000)]
    err_se = np.abs(bloch.run(spin, sp, spread=O.lorentzian_spread(501)).signal - epg.run(spin, sp).signal).max()
    rows.append(_row("X8", "agreement", "T2' of a gradient echo as the factor exp(-static time / T2') of the adc (T2' 30 ms, echoes at 12 and "
                     "24 ms)", "an ensemble of 8,000 static off-resonances with a Lorentzian distribution (`operators.lorentzian_spread`); "
                     "it converges as 1 / n", errs[-1], 2e-3))
    rows.append(_row("X8b", "agreement", "the same at a spin echo (static time 0)", "the ensemble of 501 off-resonances: every isochromat is "
                     "refocused", err_se, 1e-12))

    # X9 data: a sequence survives JSON, its digest is the digest of its content
    bad = 0
    for _, fn, args in _library_cases():
        seq = fn(*args)
        back = B.from_data(B.to_data(seq))
        bad += int(back != seq or B.digest(back) != B.digest(seq) or hash(back) != hash(seq))
    rows.append(_row("X9", "agreement", "every library sequence written to JSON-ready data and read back (17 sequences)",
                     "the same sequence: equal, same hash, same content digest", bad, 0))
    return rows


def _reaches_zero(body):
    """True when every rf of the body sits at zero accumulated dephasing and the body has a net dephasing."""
    acc, ok = 0.0, True
    for blk in body:
        if blk.kind == "gradient":
            acc += blk.cycles[0]
        elif blk.kind == "rf":
            ok = ok and abs(acc) < 1e-9
    return ok and abs(acc) > 1e-9


def _as_kwargs(sp):
    keys = ("t1_ms", "t2_ms", "t2p_ms", "df_hz", "b1", "m0", "adc_um2_ms", "zeta", "column", "pool", "f_b", "k_ab_per_s", "t1b_ms", "t2b_ms",
            "shift_b_hz", "lineshape")
    return {k: getattr(sp, k) for k in keys if getattr(sp, k) is not None}


def _library_cases():
    """(name, function, arguments) of every library sequence in a typical protocol (for X9 and the speed table)."""
    sel = B.Selective(4.0)
    return [
        ("spgr", L.spgr, (10.0, 20.0)), ("spgr, ideal spoiling", lambda: L.spgr(10.0, 20.0, spoiling="ideal", selective=sel), ()),
        ("afi", L.afi, (20.0, 5, 60.0)), ("epi", L.epi, (2000.0, 90.0)), ("se", lambda: L.se(500.0, 20.0, selective=sel), ()),
        ("ir", lambda: L.ir(2000.0, 20.0, 400.0, inv_eff=0.93, selective=sel), ()), ("bssfp", L.bssfp, (5.0, 40.0)),
        ("ir_gre", L.ir_gre, (1500.0, 400.0, 30.0)), ("fse", lambda: L.fse(3000.0, 8.0, 32, 150.0), ()),
        ("mese", lambda: L.mese(2000.0, 10.0, 16, 160.0), ()), ("megre", L.megre, (40.0, 20.0, (3.0, 8.0, 13.0, 18.0))),
        ("spgr_lobe", L.spgr_lobe, (10.0, 20.0, 3.0)), ("dess", L.dess, (15.0, 30.0)), ("mprage", L.mprage, (900.0, 128, 8.0, 9.0, 600.0)),
        ("t2prep", L.t2prep, (1000.0, 50.0)), ("satrec", L.satrec, (800.0, 300.0)), ("dwse", L.dwse, (3000.0, 80.0, 30.0, 20.0, 40.0, 1.0)),
        ("steam", L.steam, (3000.0, 40.0, 100.0, 30.0, 10.0, 1.0)), ("mt_spgr", L.mt_spgr, (30.0, 10.0, 2000.0, 2000.0)),
    ]


# ================================================================================================== the engine
def ssfp_fid(a, tr_ms, t1_ms, t2_ms):
    """Steady-state free induction decay right after the pulse with one cycle of dephasing per TR and no RF spoiling
    (Hanicke & Vogel 2003, Magn Reson Med 49:771):  tan(a / 2) (1 - (E1 - cos a) r)."""
    e1, e2 = np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms)
    r = np.sqrt((1 - e2 ** 2) / ((1 - e1 * np.cos(a)) ** 2 - e2 ** 2 * (e1 - np.cos(a)) ** 2))
    return np.tan(a / 2) * (1 - (e1 - np.cos(a)) * r)


def ssfp_echo(a, tr_ms, t1_ms, t2_ms):
    """The echo of the same steady state, tan(a / 2) (1 - (1 - E1 cos a) r) (attributed to Hanicke & Vogel 2003 and to
    the DESS literature, Bruder et al. 1988; the papers were not opened: row N7b checks it against the ensemble)."""
    e1, e2 = np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms)
    r = np.sqrt((1 - e2 ** 2) / ((1 - e1 * np.cos(a)) ** 2 - e2 ** 2 * (e1 - np.cos(a)) ** 2))
    return np.tan(a / 2) * (1 - (1 - e1 * np.cos(a)) * r)


def _slice_spins(cls, tbw, flip_deg, b1=1.0, two_sided=True, **kw):
    """Columns through the slice for one tissue: (spins, weights of the slice integral)."""
    zeta, w = O.slice_columns(tbw, two_sided=two_sided)
    return O.grid([cls], zeta, [b1], **kw), w


def _engine_world():
    from ..scanner import kspace
    from ..scanner import world as W
    w = W.make_world("5b" * 16, dict(kind="qa", b1_shape_amp=(0.2, 0.24)))
    return w, kspace.Engine(w)


def _served_amplitudes(eng, w, proto, per_class=3, model="flip"):
    """Largest relative difference between ``Engine.amplitudes`` and the core for a protocol, over ``per_class``
    voxels of every tissue class of the world; the engine's number is divided by weight x proton density."""
    from ..scanner import console
    pl = console.plan(proto, w["console"])
    rp = pl["realised"]
    tbw, window = eng.pulse(rp)
    worst = 0.0
    for c, name in enumerate(w["names"]):
        cls, sup = w["tissues"][name], eng.sup[c]
        idx = np.flatnonzero(w["weights"][c][sup] > 0.5)
        idx = idx[::max(1, idx.size // per_class)][:per_class]
        scale = w["weights"][c][sup][idx] * cls["pd"]
        served = [v[idx] / scale for v in eng.amplitudes(rp, c)]
        df = eng.df[sup][idx] if rp["sequence"] == "bssfp" else 0.0
        if tbw is None:
            seq, labels = L.served(rp, inv_eff=w["system"]["inv_eff"])
            sp = O.spins(cls["t1_ms"], cls["t2_ms"], b1=eng.b1[sup][idx], adc_um2_ms=cls["adc_um2_ms"], df_hz=df)
            got = periodic.steady_state(seq, sp)
            mine = [got.of(lab)[0] for lab in labels]
        else:
            sel = B.Selective(tbw, window, model=model)
            seq, labels = L.served(rp, selective=sel, inv_eff=w["system"]["inv_eff"])
            zeta, wz = O.slice_columns(tbw, two_sided=False, window=window)
            shp = (idx.size, zeta.size)
            sp = O.spins(cls["t1_ms"], cls["t2_ms"], b1=np.broadcast_to(eng.b1[sup][idx][:, None], shp), adc_um2_ms=cls["adc_um2_ms"],
                         zeta=np.broadcast_to(zeta[None, :], shp), df_hz=np.broadcast_to(np.asarray(df)[..., None], shp))
            got = periodic.steady_state(seq, sp)
            if model == "flip":                                         # the phase of the column is a weight; both sides of the slice
                wgt = (wz[None, :] * O.column_phase(sel, rp["flip_deg"], sp).real.reshape(shp))
                mine = [(got.of(lab)[0].reshape(shp) * wgt).sum(1) for lab in labels]
            else:                                                       # the mirror column gives the complex conjugate
                mine = [(got.of(lab)[0].reshape(shp).real * wz[None, :]).sum(1) for lab in labels]
        decay = np.exp(-rp["te_ms"] / cls["t2_ms"]) if rp["sequence"] in ("se", "ir", "bssfp") else 1.0     # the core's echo carries it
        for a, b in zip(mine, served):
            worst = max(worst, float(np.max(np.abs(a / decay - b)) / np.max(np.abs(b))))
    return worst, rp


def _served_saturated(proto, per_class=2):
    """Largest relative difference between the amplitude maps the engine serves under fat saturation, line by line
    (``Engine._species``), and the core, for a protocol on an abdomen world at 1.5 T (three fat tissues in a field that
    moves their lines across the band of the pulse); and the smallest and largest saturation factor met. The band
    of the pulse (which share of the saturation a line receives) is the engine's; the steady state is the core's."""
    from ..scanner import console, kspace, tissue
    from ..scanner import world as W
    w = W.make_world("5b" * 16, dict(kind="abdomen", b0_t=1.5, n=320))
    eng = kspace.Engine(w)
    rp = console.plan(proto, w["console"])["realised"]
    tbw, window = eng.pulse(rp)
    sy, f0 = w["system"], GAMMA_HZ_PER_T * 1e-6 * w["b0_t"]                # Hz per ppm
    centre = tissue.FAT_PPM * f0 + sy.get("fatsat_offset_hz", 0.0) + rp.get("fatsat_offset_hz", 0.0)
    worst, q_seen = 0.0, []
    for c, name in enumerate(w["names"]):
        cls, sup = w["tissues"][name], eng.sup[c]
        lines, served = eng._species(rp, c)
        if lines is None:
            continue
        idx = np.flatnonzero(w["weights"][c][sup] > 0.5)
        idx = idx[::max(1, idx.size // per_class)][:per_class]
        for j, (ppm, _) in enumerate(lines):
            for i in idx:
                q = 1 - sy["fatsat_eff"] * float(kspace.sat_band(eng.df[sup][i] + ppm * f0 - centre, sy["fatsat_bw_hz"]))
                q_seen.append(q)
                sat = (q, rp["fatsat_delay_ms"])
                want = served[0][j, i] / (w["weights"][c][sup][i] * cls["pd"])
                if tbw is None:
                    seq, labels = L.served(rp, inv_eff=sy["inv_eff"], sat=sat)
                    mine = periodic.steady_state(seq, O.spins(cls["t1_ms"], cls["t2_ms"], b1=eng.b1[sup][i], adc_um2_ms=cls["adc_um2_ms"])).of(labels[0])[0, 0]
                else:
                    sel = B.Selective(tbw, window, model="flip")
                    seq, labels = L.served(rp, selective=sel, inv_eff=sy["inv_eff"], sat=sat)
                    zeta, wz = O.slice_columns(tbw, two_sided=False, window=window)
                    sp = O.spins(cls["t1_ms"], cls["t2_ms"], b1=eng.b1[sup][i], adc_um2_ms=cls["adc_um2_ms"], zeta=zeta)
                    mine = np.sum(wz * O.column_phase(sel, rp["flip_deg"], sp).real * periodic.steady_state(seq, sp).of(labels[0])[0])
                decay = math.exp(-rp["te_ms"] / cls["t2_ms"]) if rp["sequence"] in ("se", "ir") else 1.0
                worst = max(worst, abs(mine / decay - want) / abs(want))
    return worst, min(q_seen), max(q_seen)


def check_engine():
    from ..scanner import bloch as engine_bloch
    from ..scanner import epg as engine_epg
    from ..scanner import kspace
    from ..scanner import rf as engine_rf
    from ..scanner import signal as S
    rows = []
    rng = np.random.default_rng(15)
    t1g = np.array([250.0, 400, 700, 1100, 1800, 2800])
    b1g = np.round(np.arange(0.6, 1.41, 0.1), 2)
    grid_sp = O.spins(np.repeat(t1g, b1g.size), 60.0, b1=np.tile(b1g, t1g.size))
    hard = dict(excitation="nonselective")
    w, eng = _engine_world()

    # ---------------- spgr
    worst, n = 0.0, 0
    for flip, tr in itertools.product([1, 3, 5, 10, 20, 40, 60, 90], [4, 8, 15, 30, 100, 500, 3000, 15000]):
        got = periodic.steady_state(L.spgr(tr, flip, spoiling="ideal"), grid_sp).signal[0]
        worst = max(worst, _rel(got, S.spgr(D(flip) * grid_sp.b1, tr, grid_sp.t1_ms)))
        n += got.size
    for tr, t1 in itertools.product([5.0, 15.0, 40.0], [300.0, 1000.0, 2500.0]):         # S4: the Ernst angle
        ernst = math.degrees(math.acos(math.exp(-tr / t1)))
        a = ernst + np.array([-0.5, 0.0, 0.5])
        s = [periodic.steady_state(L.spgr(tr, x, spoiling="ideal"), O.spins(t1, 60.0)).signal[0, 0].real for x in a]
        worst = max(worst, 0.0 if s[1] > s[0] and s[1] > s[2] else 1.0)
    rows.append(_row("Q1", "engine", f"spgr with ideal spoiling (the engine's tier 1 and `epi`), hard pulse: the grid of row S1 ({n:,} values: "
                     "1 to 90 deg, TR 4 to 15,000 ms, T1 250 to 2,800 ms, B1 0.6 to 1.4) and the Ernst angle of row S4", "`scanner/signal.py::spgr` "
                     "(the Ernst formula). Both sides are exact", worst, 1e-12, unit="relative"))

    cases = [(20, 10, 1000, 80, 117, 0.0), (45, 5, 1000, 80, 50, 0.0), (30, 8, 600, 40, 117, 0.0), (25, 12, 1500, 200, 150, 0.0), (12, 6, 300, 40, 117, 0.0),
             (30, 6, 1000, 80, 117, 0.003), (20, 20, 1500, 200, 117, 0.15), (15, 8, 1000, 80, 117, 0.002)]
    worst, n = 0.0, 0
    for flip, tr, t1, t2, phi, bd in cases:                             # rows S5, S6 (and E7, E8 of the design layer)
        cyc = 0.1
        adc = bd / (1e-6 * (2 * math.pi / cyc) ** 2 * tr)
        seq = B.sequence(L.spgr(tr, flip, phase_inc_deg=phi).blocks, cycle_mm=cyc)
        got = periodic.steady_state(seq, O.spins(t1, t2, adc_um2_ms=adc)).signal[0, 0]
        worst = max(worst, abs(got - engine_epg.steady_state(D(flip), tr, t1, t2, phi, (1,), bd)[0]) / abs(got))
        n += 1
    for flip, tr, (t1, t2) in itertools.product([5, 15, 30, 60, 90], [4.0, 10.0, 25.0, 60.0], [(300, 40), (800, 80), (1500, 200), (2500, 150), (4300, 2000)]):
        got = periodic.steady_state(L.spgr(tr, flip, phase_inc_deg=0.0), O.spins(t1, t2)).signal[0, 0]     # row S7
        worst = max(worst, abs(got - engine_epg.steady_state(D(flip), tr, t1, t2, 0.0)[0]) / abs(got), abs(got - ssfp_fid(D(flip), tr, t1, t2)) / abs(got))
        n += 1
    tis = _tissues()
    for flip, tr, phi, spoiler in itertools.product([5, 30, 60], [8.0, 30.0], [117.0, 50.0], [60.0, 300.0]):
        seq = L.spgr(tr, flip, phase_inc_deg=phi, spoiler_mt_ms_per_m=spoiler)
        got = periodic.steady_state(seq, O.grid(tis)).signal[0]
        ref = np.array([engine_epg.steady_state(D(flip), tr, c["t1_ms"], c["t2_ms"], phi, (1,), kspace.diffusion_bd(c["adc_um2_ms"], spoiler, tr))[0] for c in tis])
        worst = max(worst, _rel(got, ref))
        n += got.size
    rows.append(_row("Q2", "engine", f"spgr with RF spoiling and diffusion (tier 2), hard pulse, {n} steady states: the cases of rows S5 and S6, "
                     "the 100 cases of row S7 (increment 0) and six tissues at 3 T x 3 flip angles x TR 8 and 30 ms x increments 117 and 50 deg x "
                     "spoiler moments 60 and 300 mT ms/m", "`scanner/epg.py::steady_state` at the same truncation (and for increment 0 the closed "
                     "form of Hanicke & Vogel 2003). Both sides solve the same linear system", worst, 1e-9, unit="relative"))

    lim = max(max(_rel(periodic.steady_state(L.spgr(tr, f), O.spins(t1, 0.05)).signal[0], S.spgr(D(f), tr, t1))
                  for f, tr, t1 in itertools.product([5, 30, 80], [5, 20], [300, 2500])),
              max(_rel(periodic.steady_state(L.spgr(600, f), O.spins(t1, 40.0)).signal[0], S.spgr(D(f), 600, t1)) for f in (5, 30, 80) for t1 in (300, 2500)),
              max(_rel(periodic.steady_state(B.sequence(L.spgr(5, f).blocks, cycle_mm=1e-4), O.spins(1000.0, 80.0, adc_um2_ms=2.5)).signal[0],
                       S.spgr(D(f), 5, 1000)) for f in (5, 30, 80)))
    gap = abs(abs(periodic.steady_state(L.spgr(5, 30), O.spins(1500.0, 200.0)).signal[0, 0]) / S.spgr(D(30), 5, 1500) - 1)
    rows.append(_row("Q2b", "engine", "the limits of row S8: T2 << TR, TR >> T2, complete diffusion spoiling", "the Ernst formula", lim, 1e-6, unit="relative"))
    rows.append(_row("Q2c", "engine", "row S9: distance from the Ernst formula at TR 5 ms, 30 deg, T1 1500 ms, T2 200 ms", "incomplete spoiling: "
                     "at least 3 % (the engine's row gives the same number)", gap, 0.03, kind="min", unit="relative"))

    # the slice: flip model against the engine's direct sum (row R8), the served interpolation, the isochromat ensemble (R7)
    direct = interp = 0.0
    for flip, tbw, tr, t1, t2, adc, spoiler in ((20, 4, 20.0, 1000, 80, 1.0, 100.0), (60, 4, 8.0, 2500, 200, 2.0, 100.0), (90, 2, 10.0, 4300, 600, 3.0, 100.0),
                                                (10, 8, 5.0, 838, 75, 0.75, 60.0), (40, 8, 30.0, 300, 40, 2.2, 300.0)):
        cls = dict(t1_ms=float(t1), t2_ms=float(t2), adc_um2_ms=adc)
        sel = B.Selective(tbw, model="flip")
        sp, wz = _slice_spins(cls, tbw, flip, two_sided=False)
        got = periodic.steady_state(L.spgr(tr, flip, spoiler_mt_ms_per_m=spoiler, selective=sel), sp).signal[0]
        mine = np.sum(wz * O.column_phase(sel, flip, sp).real * got)
        th, ph = engine_rf.excitation(D(flip), tbw, sp.zeta)
        bd = kspace.diffusion_bd(adc, spoiler, tr)
        ref = np.sum(wz * ph.real * np.array([engine_epg.steady_state(float(t), tr, t1, t2, 117.0, (1,), bd)[0] if t > 1e-9 else 0j for t in th]))
        direct = max(direct, abs(mine - ref) / abs(ref))
        rp = dict(sequence="spgr", tr_ms=tr, rf_phase_increment_deg=117.0, spoiler_mt_ms_per_m=spoiler)
        interp = max(interp, abs(mine - kspace.slice_signal(rp, cls, D(flip), tbw)[0][0]) / abs(mine))
    rows.append(_row("Q3", "engine", "spgr through the slice (the cases of row R8, with diffusion): one hard pulse of the column's flip angle "
                     "per column, weighted with the transverse phase of the column (`Selective(model='flip')`, the engine's model)",
                     "the engine's sum over the same columns with `epg.steady_state` at every column. Both sides are exact for that model",
                     direct, 1e-9, unit="relative"))
    rows.append(_row("Q3b", "engine", "the same cases as the engine SERVES them (`kspace.slice_signal`)", "the engine interpolates the phase "
                     "graph over the flip angle on 16 Chebyshev nodes (its row R8 allows 5e-4); the core solves every column: the core is the "
                     "exact side", interp, 5e-4, unit="relative"))
    model = ens_rot = ens_flip = 0.0
    for flip, tbw, tr, t1, t2 in ((20, 4, 20.0, 600.0, 60.0), (40, 2, 15.0, 400.0, 40.0), (10, 8, 20.0, 1000.0, 60.0), (60, 4, 12.0, 800.0, 50.0)):
        cls = dict(t1_ms=t1, t2_ms=t2)
        sp, wz = _slice_spins(cls, tbw, flip)
        rot = np.sum(wz * periodic.steady_state(L.spgr(tr, flip, selective=B.Selective(tbw)), sp).signal[0])
        sel = B.Selective(tbw, model="flip")
        flp = np.sum(wz * O.column_phase(sel, flip, sp) * periodic.steady_state(L.spgr(tr, flip, selective=sel), sp).signal[0])
        zmax = 0.5 + engine_rf.WING / tbw
        ref = engine_bloch.slice_spgr_isochromats(D(flip), tbw, tr, t1, t2, n_z=int(round(2 * zmax * 12 * tbw)) + 1, z_max=zmax, n_iso=192)
        model, ens_rot, ens_flip = max(model, abs(rot - flp) / abs(rot)), max(ens_rot, abs(rot - ref) / abs(ref)), max(ens_flip, abs(flp - ref) / abs(ref))
    rows.append(_row("Q3c", "engine", "the cases of row R7 with the ROTATION of every column (Cayley-Klein parameters of the selective pulse, "
                     "`Selective(model='rotation')`)", "the engine's separately written isochromat ensemble through the slice "
                     "(`scanner/bloch.py::slice_spgr_isochromats`, 3 x 3 cascades). The rotation model is the exact one; the ensemble has 192 "
                     "spoiler phases and twice the columns", ens_rot, 2e-4, unit="relative"))
    rows.append(_row("Q3d", "engine", "the engine's flip model against the rotation model in those cases", "equal: the part of a column's "
                     "rotation that turns about z is the same in every repetition, and with a net spoiler moment per TR a constant turn about z "
                     f"is a translation along the spoiler axis. The flip model is exact for spgr and afi (it is {ens_flip:.0e} from the ensemble)",
                     model, 1e-12, unit="relative"))

    worst_h, rp = _served_amplitudes(eng, w, dict(sequence="spgr", tr_ms=8.0, flip_deg=30.0, spoiler_mt_ms_per_m=60.0, matrix=64, **hard))
    worst_s, rp = _served_amplitudes(eng, w, dict(sequence="spgr", tr_ms=12.0, flip_deg=30.0, matrix=64))
    rows.append(_row("Q4", "engine", "spgr as `Engine.amplitudes` serves it on a phantom world with a strong B1 pattern, 3 voxels of each of 7 "
                     "materials: hard pulse (the protocol of row S15) and the default selective pulse (row R10)", "the engine interpolates over "
                     "the flip angle (5 to 16 Chebyshev nodes; its rows allow 1e-4); the core solves every voxel and column: the core is the exact "
                     "side for the engine's model", max(worst_h, worst_s), 1e-4, unit="relative"))

    # ---------------- afi
    worst, n = 0.0, 0
    for flip, tr1, ratio in itertools.product([10, 20, 40, 60, 80, 90], [4, 10, 20, 50, 150], [2, 3, 5, 10]):
        got = periodic.steady_state(L.afi(tr1, ratio, flip, spoiling="ideal"), grid_sp)
        ref = S.afi(D(flip) * grid_sp.b1, tr1, ratio * tr1, grid_sp.t1_ms)
        worst = max(worst, _rel(got.of(("s", 0))[0], ref[0]), _rel(got.of(("s", 1))[0], ref[1]))
        n += 2 * grid_sp.n
    rows.append(_row("Q5", "engine", f"afi with ideal spoiling, hard pulse: the grid of row S2 ({n:,} values)", "`scanner/signal.py::afi` "
                     "(Yarnykh 2007). Both sides are exact", worst, 1e-12, unit="relative"))
    worst = 0.0
    for pattern, flip, tr, t1, t2, phi, bd in (((1, 1, 0, 0, 0, 0), 60, 20, 1000, 80, 129.3, 0.0), ((1, 1, 0, 0), 45, 10, 600, 60, 117, 0.0),
                                               ((1, 1, 0, 0, 0), 70, 15, 1500, 150, 39, 0.0), ((1, 1, 0, 0, 0, 0), 70, 30, 1200, 150, 45, 0.4)):
        cyc = 0.1
        adc = bd / (1e-6 * (2 * math.pi / cyc) ** 2 * tr)
        seq = B.sequence(L.afi(tr, len(pattern) - 1, flip, phase_inc_deg=phi).blocks, cycle_mm=cyc)
        got = periodic.steady_state(seq, O.spins(t1, t2, adc_um2_ms=adc))
        ref = engine_epg.steady_state(D(flip), tr, t1, t2, phi, pattern, bd)
        worst = max(worst, abs(got.of(("s", 0))[0, 0] - ref[0]) / abs(ref[0]), abs(got.of(("s", 1))[0, 0] - ref[1]) / abs(ref[1]))
    for flip, tr, ratio, phi, spoiler in itertools.product([30, 60], [10.0, 20.0], [2, 5], [129.3], [300.0]):
        got = periodic.steady_state(L.afi(tr, ratio, flip, phase_inc_deg=phi, spoiler_mt_ms_per_m=spoiler), O.grid(tis))
        pattern = (1, 1) + (0,) * (ratio - 1)
        ref = np.array([engine_epg.steady_state(D(flip), tr, c["t1_ms"], c["t2_ms"], phi, pattern, kspace.diffusion_bd(c["adc_um2_ms"], spoiler, tr)) for c in tis])
        worst = max(worst, _rel(got.of(("s", 0))[0], ref[:, 0]), _rel(got.of(("s", 1))[0], ref[:, 1]))
    worst_a, _ = _served_amplitudes(eng, w, dict(sequence="afi", tr_ms=20.0, flip_deg=60.0, matrix=64, **hard), per_class=2)
    rows.append(_row("Q5b", "engine", "afi with RF spoiling and diffusion: the dual-TR cases of rows S5 and S6 and six tissues x 2 flip angles x "
                     "TR1 10 and 20 ms x TR2 / TR1 2 and 5 (a table of flip angles on a raster of TR1: pulse, pulse, n - 1 empty positions)",
                     "`scanner/epg.py::steady_state` with its pulse pattern. Both sides are exact", worst, 1e-9, unit="relative"))
    rows.append(_row("Q5c", "engine", "afi as `Engine.amplitudes` serves it (the protocol of row S15, hard pulse)", "the engine interpolates over "
                     "the flip angle; the core is the exact side", worst_a, 1e-4, unit="relative"))

    # ---------------- epi
    worst = 0.0
    for flip, tr in itertools.product([30, 60, 90], [500.0, 2000.0, 6000.0]):
        worst = max(worst, _rel(periodic.steady_state(L.epi(tr, flip), grid_sp).signal[0], S.spgr(D(flip) * grid_sp.b1, tr, grid_sp.t1_ms)))
    worst_e, _ = _served_amplitudes(eng, w, dict(sequence="epi", tr_ms=2000.0, matrix=64, fat_sat=False, **hard))
    worst_es, _ = _served_amplitudes(eng, w, dict(sequence="epi", tr_ms=2000.0, matrix=64, fat_sat=False))
    rows.append(_row("Q6", "engine", "epi (one excitation per TR, ideal spoiler): 3 flip angles x 3 TR x the T1 and B1 grid; and as "
                     "`Engine.amplitudes` serves it, hard and selective, fat saturation off", "the Ernst formula; the engine. Exact on both sides "
                     "for the hard pulse; the selective pulse is interpolated by the engine over the B1 range", max(worst, worst_e, worst_es), 1e-6,
                     unit="relative"))

    # ---------------- se, ir
    v = 400
    t1, a = rng.uniform(200, 4500, v), rng.uniform(60, 120, v)
    sp = O.spins(t1, 80.0, b1=a / 90.0)
    worst = 0.0
    for tr, te in ((600.0, 20.0), (300.0, 15.0), (2000.0, 90.0), (6000.0, 120.0)):
        got = periodic.steady_state(L.se(tr, te), sp).signal[0]
        worst = max(worst, _rel(got, S.se(np.radians(a), tr, te, t1) * np.exp(-te / 80.0)))
    eff = 0.0
    for beta in (180.0, 149.0, 114.6, 60.0):                           # row S12 with real crushers
        got = epg.run([B.rf(90, 0), B.delay(5), B.gradient(1), B.rf(beta, 90), B.gradient(1), B.delay(5), B.adc()], O.spins(1e18, 1e18)).signal[0, 0]
        eff = max(eff, abs(got - math.sin(D(beta) / 2) ** 2))
    rows.append(_row("Q7", "engine", "se, hard pulse: 400 random (T1 200 to 4,500 ms, pulses at 60 to 120 % of 90 and 180 deg) x 4 (TR, TE) "
                     "(row S10); the refocusing efficiency with REAL crushers, one echo from fresh magnetisation (row S12)",
                     "`scanner/signal.py::se` x exp(-TE / T2) (the engine applies the decay outside the amplitude); sin^2(beta / 2). Both sides "
                     "are exact for the crushed model", max(worst, eff), 1e-12, unit="relative"))
    worst = 0.0
    for tr, te, ti, kap in ((2000.0, 20.0, 400.0, 0.94), (6000.0, 90.0, 2500.0, 1.0), (1500.0, 15.0, 150.0, 0.8), (600.0, 20.0, 150.0, 0.94)):
        got = periodic.steady_state(L.ir(tr, te, ti, inv_eff=kap), sp).signal[0]
        worst = max(worst, np.abs(got - S.ir_se(np.radians(a), tr, te, ti, t1, kap) * np.exp(-te / 80.0)).max())
    null = 0.0
    for t1n in (400.0, 1000.0, 2500.0):                                # row S13: the null of inversion recovery for TR >> T1
        ti = t1n * math.log(2)
        s = [periodic.steady_state(L.ir(60000.0, 10.0, ti * f), O.spins(t1n, 80.0)).signal[0, 0].real for f in (0.999, 1.0, 1.001)]
        null = max(null, abs(s[1]) / abs(s[2] - s[0]) * 0.002)          # distance of the zero crossing from T1 ln 2, relative
    rows.append(_row("Q8", "engine", "ir, hard pulse: the same 400 voxels x 4 (TR, TE, TI, inversion efficiency 0.8 to 1) (row S11; absolute, the "
                     "signal crosses zero)", "`scanner/signal.py::ir_se` x exp(-TE / T2). Both sides are exact", worst, 1e-12))
    rows.append(_row("Q8b", "engine", "the null of ir at TR 60 s for T1 400, 1000, 2500 ms (row S13): distance of the zero crossing from T1 ln 2",
                     "TI = T1 ln 2 for TR >> T1 (at T1 2500 ms, exp(-TR / T1) = 4e-11 is still there)", null, 1e-9, unit="relative"))
    worst = 0.0
    for scale, tbw, tr, te, t1v, ti, kap in ((1.0, 4, 500.0, 20.0, 800.0, None, 1.0), (0.8, 4, 300.0, 15.0, 1200.0, None, 1.0), (1.15, 2, 600.0, 20.0, 500.0, None, 1.0),
                                             (1.0, 8, 400.0, 20.0, 900.0, None, 1.0), (0.9, 4, 2000.0, 20.0, 1000.0, 400.0, 0.93)):
        cls = dict(t1_ms=t1v, t2_ms=80.0)
        sps, wz = _slice_spins(cls, tbw, 90.0, b1=scale)
        sel = B.Selective(tbw)
        seq = L.se(tr, te, selective=sel) if ti is None else L.ir(tr, te, ti, inv_eff=kap, selective=sel)
        mine = np.sum(wz * periodic.steady_state(seq, sps).signal[0]) / math.exp(-te / 80.0)
        rp = dict(sequence="se" if ti is None else "ir", tr_ms=tr, te_ms=te, ti_ms=ti)
        worst = max(worst, abs(mine - kspace.slice_signal(rp, cls, scale * math.pi / 2, tbw, inv_eff=kap)[0][0]) / abs(mine))
    worst_s, _ = _served_amplitudes(eng, w, dict(sequence="se", tr_ms=400.0, matrix=64), model="rotation")
    worst_i, _ = _served_amplitudes(eng, w, dict(sequence="ir", tr_ms=2000.0, ti_ms=400.0, matrix=64), model="rotation")
    worst_h, _ = _served_amplitudes(eng, w, dict(sequence="se", tr_ms=500.0, matrix=64, **hard))
    rows.append(_row("Q9", "engine", "se and ir through the slice with the rotation of every column, refocusing on the 1.5 x slice (the cases of "
                     "row R9, B1 80 to 115 %)", "`scanner/kspace.py::slice_signal`. Both sides are exact for the crushed model: without "
                     "transverse magnetisation before a pulse, flip model and rotation model coincide", worst, 1e-12, unit="relative"))
    rows.append(_row("Q9b", "engine", "se and ir as `Engine.amplitudes` serves them: selective (rows R10) and hard (the protocol of row R6)",
                     "the engine interpolates the slice integral over the B1 range (5 to 16 nodes); the core is the exact side",
                     max(worst_s, worst_i, worst_h), 1e-6, unit="relative"))

    # ---------------- bssfp
    worst = sign = 0.0
    for _ in range(200):                                               # row S14
        t1v, tr, al = rng.uniform(200, 4000), rng.uniform(3, 12), rng.uniform(5, 90)
        t2v, phi = rng.uniform(20, min(t1v, 2000)), rng.uniform(-np.pi, np.pi) * 3
        df = phi / (2 * math.pi * tr * 1e-3)
        got = periodic.steady_state(L.bssfp(tr, al), O.spins(t1v, t2v, df_hz=df)).signal[0, 0]
        ref = S.bssfp(D(al), phi + math.pi, tr, t1v, t2v) * np.exp(0.5j * phi) * math.exp(-0.5 * tr / t2v)
        worst = max(worst, abs(got - ref) / abs(ref))
    for cyc in (0.0, 90.0, 137.0, 270.0):                              # the sign of the phase cycle
        t1v, t2v, tr, al = 900.0, 90.0, 6.0, 45.0
        df = rng.uniform(-80, 80, 9)
        free = 2 * math.pi * df * tr * 1e-3
        ref = S.bssfp(D(al), free - D(cyc), tr, t1v, t2v) * np.exp(0.5j * free) * math.exp(-0.5 * tr / t2v)
        plus = periodic.steady_state(L.bssfp(tr, al, +cyc), O.spins(t1v, t2v, df_hz=df)).signal[0]
        worst = max(worst, _rel(plus, ref))
    for cyc in (90.0, 137.0, 270.0):                                   # the same sign in what the engine serves: its amplitude maps
        sign = max(sign, _served_amplitudes(eng, w, dict(sequence="bssfp", tr_ms=6.0, flip_deg=45.0, phase_cycle_deg=cyc, matrix=64,
                                                         excitation="nonselective"), per_class=6)[0])
    rows.append(_row("Q10", "engine", "bssfp, hard pulse: 200 random (T1, T2, TR, flip angle, off-resonance over three bands) of row S14 with the "
                     "180 deg cycle, and phase cycles of 0, 90, 137, 270 deg", "`scanner/signal.py::bssfp` at the precession per TR seen from the "
                     "pulse, 2 pi df TR - `phase_cycle_deg`, x exp(i pi df TR) x exp(-TR / 2 T2): the RF phase ADVANCES by `phase_cycle_deg` "
                     "per TR, as the engine counts it since 0.3.0. Both sides are exact", worst, 1e-10, unit="relative"))
    rows.append(_row("Q10b", "engine", "the sign of the phase cycle in what the engine serves: `Engine.amplitudes` of a nonselective bssfp with "
                     "phase cycles of 90, 137, 270 deg on a phantom world with a field (6 voxels of each of 7 materials)",
                     "`library.served` with the RF phase advancing by `phase_cycle_deg` per TR (conventions.md, section 5). Engine 0.2.0 put "
                     "the bands at the opposite off-resonance; this row found it, and with 0.3.0 the two agree at the same off-resonance",
                     sign, 1e-6, unit="relative"))
    direct = model = 0.0
    for flip, tbw, tr, df in ((50, 4, 8.0, 12.0), (30, 2, 5.0, -40.0), (70, 8, 6.0, 55.0)):
        cls = dict(t1_ms=900.0, t2_ms=90.0)
        sel = B.Selective(tbw, model="flip")
        sps, wz = _slice_spins(cls, tbw, flip, two_sided=False, df_hz=df)
        mine = np.sum(wz * O.column_phase(sel, flip, sps).real * periodic.steady_state(L.bssfp(tr, flip, selective=sel), sps).signal[0])
        th, ph = engine_rf.excitation(D(flip), tbw, sps.zeta)
        free = 2 * math.pi * df * tr * 1e-3
        ref = np.sum(wz * ph.real * S.bssfp(th, free + math.pi, tr, 900.0, 90.0)) * np.exp(0.5j * free) * math.exp(-0.5 * tr / 90.0)
        direct = max(direct, abs(mine - ref) / abs(ref))
        sp2, w2 = _slice_spins(cls, tbw, flip, df_hz=df)
        rot = np.sum(w2 * periodic.steady_state(L.bssfp(tr, flip, selective=B.Selective(tbw)), sp2).signal[0])
        model = max(model, abs(rot - mine) / abs(rot))
    worst_b, _ = _served_amplitudes(eng, w, dict(sequence="bssfp", tr_ms=8.0, flip_deg=50.0, matrix=64))
    rows.append(_row("Q11", "engine", "bssfp through the slice, flip model: 3 cases off resonance; and as `Engine.amplitudes` serves it (row R10)",
                     "the engine's sum of its closed form over the columns (exact on both sides); the served value interpolates sin and cos of the "
                     "flip angle over the B1 range (the core is the exact side)", max(direct, worst_b), 1e-6, unit="relative"))
    rows.append(_row("Q11b", "engine", "the engine's idealisation for bssfp in those 3 cases: flip model against the rotation of every column "
                     "with a balanced slice gradient (half the rewinding area before the pulse, half after)", "in a balanced sequence the "
                     "transverse magnetisation survives, and the rotation of an off-centre column has a component about z that acts as an "
                     "off-resonance of that column. Size of the difference in the slice integral (the rotation model is the exact one)",
                     model, 0.3, unit="relative"))

    # ---------------- ir_gre (row S3), the slice integral (R4, R5), through-slice dephasing (R11), the two-angle bias (R12)
    worst = 0.0
    for ti, tr, k in itertools.product([20, 50, 100, 300, 700, 1500, 3000, 10000], [500, 1500, 3000, 6000, 15000], [1.0, 0.95, 0.8]):
        if ti + 2 >= tr:
            continue
        got = periodic.steady_state(L.ir_gre(tr, ti, 90.0, k), grid_sp).signal[0]
        worst = max(worst, np.abs(got - S.ir_gre(grid_sp.b1 * math.pi / 2, ti, tr, grid_sp.t1_ms, k)).max())
    rows.append(_row("Q12", "engine", "inversion recovery with one gradient-echo pulse per inversion: the grid of row S3 (inversion efficiency "
                     "0.8 to 1)", "`scanner/signal.py::ir_gre`. Both sides are exact", worst, 1e-12))
    usual = extreme = 0.0
    fine_z = np.linspace(-12.0, 12.0, 8001)
    fine_w = np.full(fine_z.size, fine_z[1] - fine_z[0])
    fine_w[[0, -1]] /= 2
    for tbw, flip in itertools.product((2, 4, 8), (2, 10, 40, 90)):
        sel = B.Selective(tbw)
        zeta, wz = O.slice_columns(tbw)
        t1s = [300.0, 2500.0, 4300.0]
        coarse, fine = O.grid([dict(t1_ms=t, t2_ms=60.0) for t in t1s], zeta), O.grid([dict(t1_ms=t, t2_ms=60.0) for t in t1s], fine_z)
        for tr in (5.0, 20.0, 2000.0):
            seq = L.spgr(tr, flip, spoiling="ideal", selective=sel)
            a = (periodic.steady_state(seq, coarse).signal[0].reshape(3, -1) * wz).sum(1)
            b = (periodic.steady_state(seq, fine).signal[0].reshape(3, -1) * fine_w).sum(1)
            if tr >= 20:
                usual = max(usual, _rel(a, b))
            else:
                extreme = max(extreme, _rel(a, b))
    rows.append(_row("Q13", "engine", "the quadrature through the slice (6 x tbw points per thickness out to 0.5 + 12 / tbw thicknesses), ideal "
                     "spoiling, TR 20 and 2,000 ms: a subset of row R4 (3 pulses x 4 flip angles x 3 T1)", "the core on 8,001 columns over +- 12 "
                     "thicknesses (the bound of row R4)", usual, 2e-4, unit="relative"))
    rows.append(_row("Q13b", "engine", "the same at TR 5 ms (row R5)", "the same reference (the bound of row R5)", extreme, 1e-3, unit="relative"))
    deph = 0.0
    for tbw in (2, 4, 8):
        zeta = np.linspace(-12.0, 12.0, 8001)
        wz = np.full(zeta.size, zeta[1] - zeta[0])
        for cyc in (0.25, 0.5, 1.0, 1.7):                               # cycles of phase across one thickness at the echo
            te = 10.0
            sp0 = O.grid([dict(t1_ms=1000.0, t2_ms=80.0)], zeta)
            spd = O.spins(1000.0, 80.0, zeta=zeta, df_hz=cyc * zeta / (te * 1e-3))
            seq = L.spgr(100.0, 2.0, te_ms=te, spoiling="ideal", selective=B.Selective(tbw))
            ratio = abs(np.sum(wz * periodic.steady_state(seq, spd).signal[0])) / abs(np.sum(wz * periodic.steady_state(seq, sp0).signal[0]))
            deph = max(deph, abs(ratio - abs(float(engine_rf.dephasing_factor(cyc, tbw)))))
    rows.append(_row("Q14", "engine", "signal left by a field gradient through the slice (0.25 to 1.7 cycles across the thickness at the echo, "
                     "2 deg): the columns given their own off-resonance in the block list (row R11)", "`scanner/rf.py::dephasing_factor`: "
                     "sinc(cycles) x window(cycles / tbw), the small-tip form the engine applies (the bound of row R11)", deph, 2e-3))
    # ---------------- the saturation pulse of engine 0.3.0: an event of every repetition, Mz -> q Mz a delay before the excitation
    sat_sp = O.spins([260.0, 340.0, 1000.0, 2500.0], 60.0)
    worst = 0.0
    for flip, tr, q, d in itertools.product([10, 30, 60, 90], [30.0, 500.0, 4000.0], [0.0, 0.1, 0.5, 1.0], [4.5, 7.0, 12.0]):
        got = periodic.steady_state(L.spgr(tr, flip, spoiling="ideal", sat=(q, d)), sat_sp).signal[0]
        worst = max(worst, _rel(got, S.spgr(D(flip), tr, sat_sp.t1_ms, sat=(q, d))))
    for tr, te, q, d in itertools.product([300.0, 600.0, 4000.0], [15.0, 90.0], [0.0, 0.2, 1.0], [7.0, 12.0]):
        got = periodic.steady_state(L.se(tr, te, 80.0, sat=(q, d)), sat_sp).signal[0]
        worst = max(worst, _rel(got, S.se(D(80.0), tr, te, sat_sp.t1_ms, sat=(q, d)) * math.exp(-te / 60.0)))
    for tr, ti, q, d in itertools.product([1500.0, 4000.0], [150.0, 400.0], [0.0, 0.2, 1.0], [7.0, 12.0]):
        got = periodic.steady_state(L.ir(tr, 15.0, ti, 85.0, 0.93, sat=(q, d)), sat_sp).signal[0]
        worst = max(worst, float(np.abs(got - S.ir_se(D(85.0), tr, 15.0, ti, sat_sp.t1_ms, 0.93, sat=(q, d)) * math.exp(-15.0 / 60.0)).max()))
    ratio = []
    for tr, flip in ((30.0, 10.0), (30.0, 60.0), (500.0, 60.0), (4000.0, 90.0)):          # the cases of the audit: fat at 1.5 T, a perfect pulse on the line
        one = O.spins(343.0, 58.0)
        ratio.append(periodic.steady_state(L.spgr(tr, flip, spoiling="ideal", sat=(0.0, 7.0)), one).signal[0, 0].real
                     / periodic.steady_state(L.spgr(tr, flip, spoiling="ideal"), one).signal[0, 0].real)
    rows.append(_row("Q16", "engine", "a saturation pulse in every repetition (Mz x q, 4.5 to 12 ms before the excitation; q 0 to 1), hard pulse, "
                     "ideal spoiling: spgr and epi (144 cases x 4 T1), se (36 x 4), ir (24 x 4; absolute, the signal crosses zero)",
                     "`scanner/signal.py::spgr`, `se`, `ir_se` with `sat` (the closed form (beta0 + q beta1) / (1 - q alpha), engine 0.3.0). "
                     "Both sides are exact. A fully saturated line of T1 343 ms (fat at 1.5 T), the pulse 7 ms before the excitation, keeps " + ", ".join(f"{v:.3f}" for v in ratio)
                     + " of its signal at TR / flip angle 30 ms / 10 deg, 30 / 60, 500 / 60, 4000 / 90: not one factor", worst, 1e-12,
                     unit="relative"))
    worst = 0.0
    for flip, tr, t1v, t2v, q, d in ((30, 30.0, 340.0, 60.0, 0.0, 7.0), (60, 30.0, 260.0, 50.0, 0.2, 7.0), (20, 12.0, 340.0, 60.0, 0.5, 4.5),
                                     (60, 100.0, 340.0, 60.0, 0.05, 12.0), (10, 30.0, 1000.0, 80.0, 0.0, 7.0), (45, 20.0, 300.0, 40.0, 1.0, 7.0)):
        got = periodic.steady_state(L.spgr(tr, flip, sat=(q, d)), O.spins(t1v, t2v)).signal[0, 0]
        worst = max(worst, abs(got - engine_epg.spgr(D(flip), tr, t1v, t2v, 117.0, 0.0, sat=(q, d))) / abs(got))
    rows.append(_row("Q16b", "engine", "the same pulse in the RF-spoiled steady state of spgr, acting on the longitudinal states of every order "
                     "(6 cases, 0 to 100 % saturated, no diffusion)", "`scanner/epg.py::spgr` with `sat`, the engine's phase graph (separate "
                     "code; with diffusion the two models differ in where the spoiler plays and are not compared)", worst, 1e-10, unit="relative"))
    worst = 0.0
    for seq, tbw, flip, tr, te, ti, t1v, q, d in (("spgr", 4, 30.0, 30.0, None, None, 260.0, 0.05, 7.0), ("spgr", 2, 60.0, 30.0, None, None, 340.0, 0.3, 6.5),
                                                  ("epi", 4, 90.0, 2000.0, None, None, 340.0, 0.0, 7.0), ("se", 4, 90.0, 500.0, 20.0, None, 260.0, 0.1, 7.0),
                                                  ("se", 8, 90.0, 600.0, 15.0, None, 340.0, 0.6, 8.0), ("ir", 4, 90.0, 1500.0, 20.0, 150.0, 260.0, 0.05, 7.0)):
        cls = dict(t1_ms=t1v, t2_ms=80.0)
        sps, wz = _slice_spins(cls, tbw, flip)
        sel = B.Selective(tbw)
        seq_c = (L.spgr(tr, flip, spoiling="ideal", selective=sel, sat=(q, d)) if seq in ("spgr", "epi")
                 else L.se(tr, te, flip, selective=sel, sat=(q, d)) if seq == "se" else L.ir(tr, te, ti, flip, 0.93, selective=sel, sat=(q, d)))
        mine = np.sum(wz * periodic.steady_state(seq_c, sps).signal[0]) / (math.exp(-te / 80.0) if te else 1.0)
        rp = dict(sequence=seq, tr_ms=tr, te_ms=te, ti_ms=ti)
        ref = kspace.slice_signal(rp, cls, np.array([D(flip)]), tbw, tier=1, inv_eff=0.93, sat=(q, d))[0][0]
        worst = max(worst, abs(mine - ref) / abs(ref))
    rows.append(_row("Q16c", "engine", "the same through the slice with the rotation of every column: spgr and epi (ideal spoiling), se, ir; "
                     "fast, normal and sharp pulses (6 cases)", "`scanner/kspace.py::slice_signal` with `sat`. Both sides are exact", worst, 1e-12,
                     unit="relative"))
    worst, q_lo, q_hi = 0.0, 1.0, 0.0
    for proto in (dict(sequence="spgr", tr_ms=30.0, flip_deg=30.0, excitation="nonselective"), dict(sequence="spgr", tr_ms=30.0, flip_deg=30.0),
                  dict(sequence="se", tr_ms=600.0), dict(sequence="ir", tr_ms=1500.0, ti_ms=150.0), dict(sequence="epi", tr_ms=2000.0)):
        err, lo, hi = _served_saturated(dict(proto, matrix=64, fat_sat=True))
        worst, q_lo, q_hi = max(worst, err), min(q_lo, lo), max(q_hi, hi)
    rows.append(_row("Q16d", "engine", "fat saturation as the engine SERVES it, line by line (`Engine._species`): spgr (hard and selective, RF "
                     "spoiled), se, ir, epi on an abdomen world at 1.5 T, 2 voxels of each of 3 fat tissues x 6 lines; the lines keep "
                     f"{100 * q_lo:.0f} to {100 * q_hi:.0f} % of their longitudinal magnetisation at the pulse",
                     "`library.served` with the saturation of each line as an event (the share of the band a line receives is the engine's, "
                     "`kspace.sat_band`). The engine interpolates over the flip angle and over q; the core solves every line, voxel and "
                     "column. For the RF-spoiled spgr the core plays the spoiler before the saturation pulse and the engine spreads it "
                     "over the whole TR: with the diffusion of fat the two models differ by the size printed here; se, ir and epi agree to "
                     "the engine's interpolation (1e-8)", worst, 1e-4, unit="relative"))

    bias, same = [], 0.0
    for tbw in (2, 4):
        for t1v, (f1, f2) in itertools.product((300.0, 1000.0, 2500.0), ((4, 20), (3, 17), (6, 35))):
            cls = dict(t1_ms=t1v, t2_ms=60.0)
            sig = []
            for f in (f1, f2):
                sps, wz = _slice_spins(cls, tbw, f)
                sig.append(abs(np.sum(wz * periodic.steady_state(L.spgr(20.0, f, spoiling="ideal", selective=B.Selective(tbw)), sps).signal[0])))
            sig, ang = np.array(sig), D(1.0) * np.array([f1, f2])
            ref = np.abs(kspace.slice_signal(dict(sequence="spgr", tr_ms=20.0), dict(t1_ms=t1v), ang, tbw, tier=1)[0])
            same = max(same, _rel(sig, ref))
            y, x = sig / np.sin(ang), sig / np.tan(ang)
            bias.append(-20.0 / math.log((y[1] - y[0]) / (x[1] - x[0])) / t1v - 1)
    rows.append(_row("Q15", "engine", "two-angle variable-flip-angle T1 on a slice with the nominal angles in the fit (row R12): the two slice "
                     "signals", "`kspace.slice_signal` with ideal spoiling; the bias they give is "
                     f"{100 * max(bias):.0f} to {100 * min(bias):.0f} % (published: 40 to 80 % underestimation, Svedin & Parker 2017)", same, 1e-12,
                     unit="relative"))
    return rows


# ================================================================================================== new sequences
def check_sequences():
    from ..scanner import bloch as engine_bloch
    from ..scanner import epg as engine_epg
    from ..scanner import signal as S
    rows = []
    rng = np.random.default_rng(7)
    sp2 = O.spins([1000.0, 600.0, 2500.0], [80.0, 40.0, 300.0])
    t1, t2 = sp2.t1_ms, sp2.t2_ms

    # ---------------- fast spin echo / CPMG
    esp = 10.0
    r = epg.run(L.fse(4000.0, esp, 16, 180.0), sp2)
    err = np.abs(r.signal - np.exp(-esp * np.arange(1, 17)[:, None] / t2)).max()
    for al in (150.0, 120.0, 60.0):
        s = epg.run(L.fse(4000.0, esp, 2, al), sp2).signal
        s2, e2, e1 = math.sin(D(al) / 2) ** 2, np.exp(-esp / t2), np.exp(-esp / t1)
        err = max(err, np.abs(s[0] - s2 * e2).max(), np.abs(s[1] - (s2 * s2 * e2 * e2 + 0.5 * math.sin(D(al)) ** 2 * e2 * e1)).max())
    rows.append(_row("N1", "sequences", "fse / CPMG train: 16 echoes with 180 deg pulses; echoes 1 and 2 with 150, 120, 60 deg",
                     "exp(-n ESP / T2); sin^2(b/2) E2 and sin^4(b/2) E2^2 + (1/2) sin^2(b) E2 E1 (spin echo and stimulated echo; Hennig 1988)",
                     err, 1e-12))
    err = 0.0
    for case in range(6):
        flips = rng.uniform(40, 180, 14) if case < 4 else np.linspace(160, 60, 14)
        seq = L.fse(3000.0, rng.uniform(6, 14), 14, flips, excitation_deg=rng.uniform(70, 100), crusher_cycles=int(rng.integers(1, 3)))
        sp = _random_spins(rng, 4)
        err = max(err, np.abs(epg.run(seq, sp).signal - bloch.run(seq, sp).signal).max())
    seq = L.fse(900.0, 9.0, 8, np.linspace(170, 90, 8), excitation_deg=80.0)
    red = periodic.steady_state(seq, sp2)
    mat = periodic.steady_state(seq, sp2, kmax=B.compile_ops(seq).reach, method="matrix")
    stepped = epg.run(seq, sp2, repetitions=40).signal[-8:]
    iso = bloch.run(B.compile_ops(seq, repetitions=14), sp2).signal[-8:]
    rows.append(_row("N1b", "sequences", "14 echoes of 6 trains with variable refocusing angles (40 to 180 deg), imperfect excitation, random "
                     "T1, T2, off-resonance and B1", "isochromat ensemble (the recursion that the phase graph replaces)", err, 1e-10))
    rows.append(_row("N1c", "sequences", "fse with an 80 deg excitation and 8 refocusing angles from 170 to 90 deg, repeated every 900 ms (T1 600 "
                     "to 2,500 ms): the periodic state from two runs of one period", "the matrix method; the phase graph stepped through 40 TR; the "
                     "isochromat ensemble stepped through 14 TR",
                     max(np.abs(red.signal - mat.signal).max(), np.abs(red.signal - stepped).max(), np.abs(iso - red.signal).max()), 1e-10))
    err_t = err_q = 0.0
    for al in (60.0, 120.0, 150.0):
        s_half = math.sin(D(al) / 2)
        for first, closed in ((al, s_half), (90 + al / 2, 2 * s_half / (1 + s_half))):
            flips = np.r_[first, np.full(400, al)]
            sig = epg.run(L.fse(1e7, esp, 401, flips), O.spins(1e12, 1e12), kmax=120).signal[:, 0]
            err_t = max(err_t, abs(np.abs(sig[-100:]).mean() - closed))
            # every isochromat settles on the rotation axis of its echo period; the quaternion of Rz(p) Rx(a) Rz(p) is
            # (cos(a/2) cos p; sin(a/2), 0, cos(a/2) sin p), so the axis is along n = (sin(a/2), 0, cos(a/2) sin p)
            p = 2 * math.pi * (np.arange(4096) + 0.5) / 4096
            n = np.stack([np.full(p.size, s_half), np.zeros(p.size), math.cos(D(al) / 2) * np.sin(p)])
            n /= np.linalg.norm(n, axis=0)
            c1, s1 = math.cos(D(first)), math.sin(D(first))
            # magnetisation (1, 0, 0) after Rz(p) Rx(first) Rz(p): Rz(p) x = (cos p, sin p, 0); Rx: (cos p, c1 sin p, -s1 sin p)...
            m = np.stack([np.cos(p), c1 * np.sin(p), s1 * np.sin(p)])        # after Rz(p), then Rx(first)
            m = np.stack([np.cos(p) * m[0] - np.sin(p) * m[1], np.sin(p) * m[0] + np.cos(p) * m[1], m[2]])
            err_q = max(err_q, abs(np.mean(np.sum(m * n, 0) * n[0]) - closed))
    rows.append(_row("N2", "sequences", "pseudo steady state of long trains without relaxation (60, 120, 150 deg; 400 echoes, mean of the last "
                     "100; 120 orders kept): equal pulses, and a first pulse of 90 + a/2 deg", "sin(a/2), and 2 sin(a/2) / (1 + sin(a/2)) with "
                     "the first pulse (derived for the design layer, rows E4b, E4c; not taken from a paper)", err_t, 5e-4))
    rows.append(_row("N2b", "sequences", "those two closed forms", "re-derived here: each isochromat keeps the component of its first-echo "
                     "magnetisation along the rotation axis of its echo period, the unit vector along (sin(a/2), 0, cos(a/2) sin p) (from the quaternion of "
                     "Rz(p) Rx(a) Rz(p)); mean over 4,096 dephasing angles p", err_q, 1e-9))

    # ---------------- multi-echo spin echo: T2 with stimulated echoes
    err = 0.0
    n_e, al = 16, 150.0
    te = esp * np.arange(1, n_e + 1)
    ideal = epg.run(L.mese(4000.0, esp, n_e, al, ideal_crushers=True), sp2).signal.real
    err = max(err, np.abs(ideal - (math.sin(D(al) / 2) ** 2 * np.exp(-esp / t2)) ** np.arange(1, n_e + 1)[:, None]).max())
    fit_ideal = -1.0 / np.polyfit(te, np.log(ideal), 1)[0]
    err = max(err, np.abs(fit_ideal - 1.0 / (1.0 / t2 - math.log(math.sin(D(al) / 2) ** 2) / esp)).max() / 40)
    real = epg.run(L.mese(4000.0, esp, n_e, al), sp2).signal.real
    fit_all = -1.0 / np.polyfit(te, np.log(real), 1)[0]
    fit_even = -1.0 / np.polyfit(te[1:], np.log(real[1:]), 1)[0]
    exact = epg.run(L.mese(4000.0, esp, n_e, 180.0), sp2).signal.real
    err = max(err, np.abs(-1.0 / np.polyfit(te, np.log(exact), 1)[0] / t2 - 1).max())
    rows.append(_row("N3", "sequences", "mese, 16 echoes, 150 deg: with ideal crushers (spin-echo pathway only) the echoes and the fitted T2; "
                     "with 180 deg the fitted T2", "(sin^2(b/2) E2)^n, so 1 / T2fit = 1 / T2 - ln(sin^2(b/2)) / ESP; T2 itself", err, 1e-10))
    bias_all, bias_skip = float(np.max(np.abs(fit_all / t2 - 1))), float(np.max(np.abs(fit_even / t2 - 1)))
    rows.append(_row("N3b", "sequences", "mese with real crushers, 150 deg: bias of a mono-exponential T2 fit over all 16 echoes (T2 40, 80, 300 ms; "
                     f"without the first echo the bias is still {100 * bias_skip:.1f} %)", "stimulated echoes: the first echo is low and later "
                     "echoes are fed from Mz, which decays with T1, so T2 is overestimated by several per cent and dropping the first echo does not "
                     "remove it: the train has to be fitted with its own model. The echoes themselves equal the isochromat ensemble (row N1b)",
                     bias_all, 0.03, kind="min", unit="relative"))

    # ---------------- multi-echo gradient echo
    te_list = (3.0, 8.0, 13.0, 18.0)
    spg = O.spins(t1, t2, t2p_ms=[60.0, 30.0, 200.0], df_hz=[25.0, -40.0, 7.0])
    tr, fl = 40.0, 20.0
    fixed = periodic.steady_state(L.megre(tr, fl, te_list), spg).signal
    ernst = S.spgr(D(fl), tr, t1)
    ref = np.array([ernst * np.exp(-t * (1 / t2 + 1 / spg.t2p_ms)) * np.exp(2j * math.pi * spg.df_hz * t * 1e-3) for t in te_list])
    err = np.abs(fixed - ref).max()
    plain = O.spins(t1, t2, df_hz=spg.df_hz)
    nper = 12 * int(2500 / tr)
    err_b = np.abs(bloch.run(B.compile_ops(L.megre(tr, fl, te_list), repetitions=nper), plain).signal[-4:] - ref * np.exp(np.array(te_list)[:, None] / spg.t2p_ms)).max()
    rows.append(_row("N4", "sequences", "megre, 4 echoes: the periodic state with T2' and off-resonance", "Ernst formula x exp(-TE / T2*) x "
                     "exp(2 pi i df TE), 1 / T2* = 1 / T2 + 1 / T2' (the engine's `signal.spgr` and decay model)", err, 1e-12))
    rows.append(_row("N4b", "sequences", "megre without T2': the same echoes", f"isochromat ensemble stepped through {nper} TR from equilibrium",
                     err_b, 1e-10))

    # ---------------- RF-spoiled gradient echo with a real spoiler
    err = err_iso = err_fid = 0.0
    for fl, tr, te, t1v, t2v, phi in ((20, 10.0, 3.0, 1000.0, 80.0, 117.0), (45, 6.0, 2.0, 600.0, 40.0, 50.0), (12, 8.0, 3.5, 300.0, 40.0, 150.0)):
        sp = O.spins(t1v, t2v)
        got = periodic.steady_state(L.spgr_lobe(tr, fl, te, phi, spoiler_ms=2.0), sp).signal[0, 0]
        err = max(err, abs(got - engine_epg.steady_state(D(fl), tr, t1v, t2v, phi)[0] * math.exp(-te / t2v)))
        err_iso = max(err_iso, abs(got - engine_bloch.rf_spoiled_isochromats(D(fl), tr, t1v, t2v, phi, n_iso=1024)[0] * math.exp(-te / t2v)))
        fid = periodic.steady_state(L.spgr_lobe(tr, fl, te, 0.0, spoiler_cycles=2, spoiler_ms=1.0), sp)
        err_fid = max(err_fid, abs(fid.signal[0, 0] - ssfp_fid(D(fl), tr, t1v, t2v) * math.exp(-te / t2v)))
    rows.append(_row("N5", "sequences", "spgr_lobe (pulse, TE, sample, a spoiler lobe of 2 ms, rest of TR; no diffusion): 3 cases",
                     "`scanner/epg.py::steady_state` x exp(-TE / T2): without diffusion only the dephasing per TR matters, not when it is played",
                     err, 1e-9))
    rows.append(_row("N5b", "sequences", "the same 3 cases", "the engine's separately written isochromat ensemble "
                     "(`scanner/bloch.py::rf_spoiled_isochromats`, 1,024 spins, time stepping) x exp(-TE / T2)", err_iso, 1e-9))
    rows.append(_row("N5c", "sequences", "increment 0 with a spoiler of 2 cycles", "tan(a/2) (1 - (E1 - cos a) r) x exp(-TE / T2) (Hanicke & "
                     "Vogel 2003): the gradient-spoiled steady state does not depend on the number of whole cycles", err_fid, 1e-9))
    fl, tr, t1v, t2v = 30.0, 10.0, 1000.0, 80.0
    sp = O.spins(t1v, t2v)
    ernst = float(S.spgr(D(fl), tr, t1v))
    sweep = {phi: abs(periodic.steady_state(L.spgr_lobe(tr, fl, 0.0, phi, spoiler_ms=2.0), sp).signal[0, 0]) / ernst for phi in (0.0, 117.0, 180.0)}
    spd = O.spins(t1v, t2v, adc_um2_ms=2.5)
    strong = max(abs(abs(periodic.steady_state(L.spgr_lobe(tr, fl, 0.0, phi, spoiler_ms=8.0, cycle_mm=2e-4), spd, kmax=30).signal[0, 0]) / ernst - 1) for phi in (0.0, 117.0, 180.0))
    rows.append(_row("N6", "sequences", f"RF spoiling against the increment (30 deg, TR 10 ms, T1 1000 ms, T2 80 ms): signal / Ernst is "
                     f"{sweep[0.0]:.2f} at 0 deg, {sweep[117.0]:.3f} at 117 deg, {sweep[180.0]:.2f} at 180 deg; measured: the distance at 117 deg",
                     "117 deg is close to the Ernst signal (Zur 1991; Preibisch & Deichmann 2009)", abs(sweep[117.0] - 1), 0.05, unit="relative"))
    rows.append(_row("N6b", "sequences", "the same: the smaller distance from Ernst of the increments 0 and 180 deg", "increments of 0 and 180 deg "
                     "do not spoil: the signal is far from the Ernst value", min(abs(sweep[0.0] - 1), abs(sweep[180.0] - 1)), 0.2, kind="min", unit="relative"))
    rows.append(_row("N6c", "sequences", "the same three increments with strong diffusion across the spoiler (D 2.5 um^2/ms, 0.2 um per cycle)",
                     "diffusion removes every coherence between repetitions: the Ernst signal whatever the increment", strong, 1e-6, unit="relative"))

    # ---------------- DESS
    err = err_iso = 0.0
    for fl, tr, t1v, t2v in ((30.0, 15.0, 1000.0, 80.0), (60.0, 8.0, 600.0, 40.0), (15.0, 25.0, 2500.0, 300.0)):
        sp = O.spins(t1v, t2v)
        seq = L.dess(tr, fl)
        got = periodic.steady_state(seq, sp)
        fid, echo = got.of("fid")[0, 0], got.of("echo")[0, 0]
        err = max(err, abs(fid - ssfp_fid(D(fl), tr, t1v, t2v)), abs(echo + ssfp_echo(D(fl), tr, t1v, t2v)))
        nper = int(12 * t1v / tr)
        iso = bloch.run(B.compile_ops(seq, repetitions=nper), sp, n_sub=128)
        err_iso = max(err_iso, np.abs(iso.signal[-2:, 0] - got.signal[:, 0]).max())
    rows.append(_row("N7", "sequences", "dess, 3 cases: the free induction decay right after the pulse and the echo right before the next pulse",
                     "tan(a/2) (1 - (E1 - cos a) r) and -tan(a/2) (1 - (1 - E1 cos a) r), r = sqrt((1 - E2^2) / ((1 - E1 cos a)^2 - E2^2 (E1 - cos a)^2)) "
                     "(Hanicke & Vogel 2003; paper not opened for the second: this computation finds that it is the echo at the moment of "
                     "the next pulse, with the opposite sign of the free induction decay, and row N7b confirms the number)", err, 1e-10))
    rows.append(_row("N7b", "sequences", "the same two samples", "isochromat ensemble of 128 spins stepped through 12 T1 from equilibrium "
                     "(states above order 128 alias; they have decayed)", err_iso, 1e-6))

    # ---------------- magnetisation-prepared gradient echo
    err = err_one = err_iso = 0.0
    for ti, n, es, fl, td, kap in ((900.0, 128, 8.0, 9.0, 600.0, 1.0), (400.0, 32, 10.0, 12.0, 300.0, 0.9), (150.0, 8, 6.0, 25.0, 50.0, 0.95)):
        got = periodic.steady_state(L.mprage(ti, n, es, fl, td, kap), sp2).signal
        c, e, ei, ed = math.cos(D(fl)), np.exp(-es / t1), np.exp(-ti / t1), np.exp(-td / t1)
        mss = (1 - e) / (1 - c * e)
        big_a, big_b = mss + (1 - ei - mss) * (c * e) ** n, kap * ei * (c * e) ** n
        mc = (1 - ed + ed * big_a) / (1 + ed * big_b)                   # Mz after the crusher, periodic
        m0 = 1 - (1 + kap * mc) * ei
        ref = math.sin(D(fl)) * (mss + (m0 - mss) * (c * e) ** np.arange(n)[:, None])
        err = max(err, np.abs(got - ref).max())
    for ti, es, fl, td, kap in ((400.0, 300.0, 30.0, 800.0, 1.0), (100.0, 50.0, 90.0, 2000.0, 0.8)):
        got = periodic.steady_state(L.mprage(ti, 1, es, fl, td, kap), sp2).signal[0]
        err_one = max(err_one, np.abs(got - S.ir_gre(D(fl), ti, ti + es + td, t1, kap)).max())
    seq = L.mprage(150.0, 8, 6.0, 25.0, 50.0, 0.95)
    spf = O.spins([200.0, 120.0], [50.0, 30.0])
    err_iso = np.abs(bloch.run(B.compile_ops(seq, repetitions=60), spf).signal[-8:] - periodic.steady_state(seq, spf).signal).max()
    rows.append(_row("N8", "sequences", "mprage (inversion of efficiency 0.9 to 1, 8 to 128 pulses, recovery): every line of the train in the "
                     "periodic state", "Look-Locker recursion in closed form: Mz_i = Mss + (Mz_0 - Mss)(E cos a)^i, Mss = (1 - E) / (1 - E cos a), "
                     "closed over the cycle (derived here)", err, 1e-12))
    rows.append(_row("N8b", "sequences", "its one-pulse limit; and 8 pulses for T1 120 and 200 ms", "`scanner/signal.py::ir_gre`; isochromat "
                     "ensemble stepped through 60 cycles", max(err_one, err_iso), 1e-12))

    # ---------------- T2 preparation, saturation recovery
    err = err_iso = 0.0
    for tr, tp, nref in ((1000.0, 50.0, 1), (3000.0, 80.0, 2), (600.0, 20.0, 4)):
        got = periodic.steady_state(L.t2prep(tr, tp, 90.0, nref), sp2).signal[0]
        err = max(err, np.abs(got - (1 - np.exp(-(tr - tp) / t1)) * np.exp(-tp / t2)).max())
        sp = _random_spins(rng, 4, relax=(60.0, 200.0, 15.0, 60.0), df=30.0)
        seq = L.t2prep(tr / 10, tp / 4, 70.0, nref)
        err_iso = max(err_iso, np.abs(bloch.run(B.compile_ops(seq, repetitions=40), sp).signal[-1] - periodic.steady_state(seq, sp).signal[0]).max())
    rows.append(_row("N9", "sequences", "t2prep (90 - 180 x 1, 2, 4 - 90 back, exact pulses, 90 deg readout): the periodic signal",
                     "(1 - exp(-(TR - TEprep) / T1)) exp(-TEprep / T2): Mz that regrows during the preparation ends transverse and is spoiled",
                     err, 1e-12))
    rows.append(_row("N9b", "sequences", "t2prep with B1 at 70 to 130 %, off-resonance up to 30 Hz, a 70 deg readout", "isochromat ensemble stepped "
                     "through 40 TR", err_iso, 1e-10))
    err = err_iso = 0.0
    for tr, ts, fl, eta in ((800.0, 300.0, 90.0, 1.0), (500.0, 100.0, 30.0, 1.0), (1200.0, 500.0, 60.0, 0.85), (400.0, 150.0, 20.0, 0.6)):
        got = periodic.steady_state(L.satrec(tr, ts, fl, eta), sp2).signal[0]
        es, er, c = np.exp(-ts / t1), np.exp(-(tr - ts) / t1), math.cos(D(fl))
        # M (before the saturation) = 1 - (1 - c m) er,  m (before the pulse) = 1 - (1 - (1 - eta) M) es
        m = (1 - es + (1 - eta) * es * (1 - er)) / (1 - (1 - eta) * es * er * c)
        err = max(err, np.abs(got - math.sin(D(fl)) * m).max())
        if eta == 1.0:
            err = max(err, np.abs(got - math.sin(D(fl)) * (1 - es)).max())
        spf = O.spins(t1 / 8, t2 / 4)
        seq = L.satrec(tr / 8, ts / 8, fl, eta)
        err_iso = max(err_iso, np.abs(bloch.run(B.compile_ops(seq, repetitions=60), spf).signal[-1] - periodic.steady_state(seq, spf).signal[0]).max())
    rows.append(_row("N10", "sequences", "satrec, saturation efficiency 0.6 to 1: the periodic signal", "sin a (1 - exp(-TS / T1)) for a complete "
                     "saturation, whatever TR and the flip angle; the closed fixed point of the two-interval recursion otherwise (derived here)",
                     err, 1e-12))
    rows.append(_row("N10b", "sequences", "the same sequences at T1 / 8", "isochromat ensemble stepped through 60 TR", err_iso, 1e-12))

    # ---------------- diffusion: Stejskal-Tanner spin echo and stimulated echo
    err = err_b = 0.0
    b_all = []
    gam = 2 * math.pi * GAMMA_HZ_PER_T
    spd = O.spins([1000.0, 1500.0, 4000.0], [80.0, 100.0, 2000.0], adc_um2_ms=[0.75, 1.1, 3.0])
    for g, delta, big, ramp, te in ((30.0, 20.0, 40.0, 0.0, 80.0), (32.0, 15.0, 45.0, 1.0, 90.0), (20.0, 5.0, 60.0, 5.0, 100.0), (40.0, 25.0, 30.0, 0.3, 60.0)):
        seq = L.dwse(3000.0, te, g, delta, big, ramp, start_ms=2.0)
        t, wave, refocus = L.diffusion_waveform("se", te, g, delta, big, ramp_ms=ramp, start_ms=2.0)
        b = design_gradients.b_value(t, wave, refocus)
        got = epg.run(seq, spd).signal[0].real
        err = max(err, np.abs(got - np.exp(-te / spd.t2_ms) * np.exp(-b * spd.adc_um2_ms * 1e-3)).max())
        closed = gam ** 2 * (g * 1e-3) ** 2 * ((delta * 1e-3) ** 2 * (big - delta / 3) * 1e-3 + (ramp * 1e-3) ** 3 / 30 - delta * 1e-3 * (ramp * 1e-3) ** 2 / 6) * 1e-6
        k, mine = 0.0, 0.0
        for op in B.compile_ops(seq).ops:                               # follow the echo pathway: the 180 turns order k into -k
            if op.kind == "rf" and op.flip_deg == 180.0:
                k = -k
            elif op.kind == "free":
                mine += B.b_increment(k, op.n[0], op.ms, seq.cycle_mm[0], op.shape)
                k += op.n[0]
            if op.kind == "adc":
                break
        err_b = max(err_b, abs(b / closed - 1), abs(mine / closed - 1))
        b_all.append(b)
    seq = L.dwse(200.0, 80.0, 30.0, 20.0, 40.0, 0.0, start_ms=2.0)
    errs = [np.abs(bloch.run(seq, spd, diffusion_substeps=m).signal - epg.run(seq, spd).signal).max() for m in (16, 32, 64)]
    rows.append(_row("N11", "sequences", f"dwse, 4 gradient programs (rectangular, trapezoid, triangle; b {min(b_all):.0f} to {max(b_all):.0f} s/mm^2) x 3 "
                     "tissues: the echo",
                     "exp(-TE / T2) exp(-b D) with b from `mrgym.design.gradients.b_value` of the same program (Stejskal & Tanner 1965)", err, 1e-8))
    rows.append(_row("N11b", "sequences", "the b-value of those programs: `gradients.b_value`, and the sum of `blocks.b_increment` over the blocks",
                     "gamma^2 G^2 (delta^2 (Delta - delta / 3) + eps^3 / 30 - delta eps^2 / 6), the trapezoid form (textbook form, source not "
                     "opened; `gradients.b_value` is the exact integral of the waveform and agrees with it)", err_b, 1e-7, unit="relative"))
    rows.append(_row("N11c", "sequences", "the first program against the Bloch-Torrey grid integration at 64 substeps per block (order "
                     f"{_order(errs):.2f})", "no closed-form diffusion factor is used by the reference", errs[-1], 1e-5))
    err = err_iso = 0.0
    for fl, te, tm, g, delta, ramp in (((90, 90, 90), 40.0, 100.0, 30.0, 10.0, 0.0), ((70, 50, 110), 30.0, 500.0, 25.0, 8.0, 1.0), ((20, 160, 45), 50.0, 900.0, 35.0, 12.0, 0.5)):
        seq = L.steam(4000.0, te, tm, g, delta, ramp, fl)
        t, wave, refocus = L.diffusion_waveform("steam", te, g, delta, tm_ms=tm, ramp_ms=ramp)
        b = design_gradients.b_value(t, wave, refocus)
        got = epg.run(seq, spd).signal[0]
        ref = 0.5 * np.prod(np.sin(np.radians(fl))) * np.exp(-te / spd.t2_ms) * np.exp(-tm / spd.t1_ms) * np.exp(-b * spd.adc_um2_ms * 1e-3)
        err = max(err, np.abs(np.abs(got) - ref).max())
        plain = O.spins(spd.t1_ms, spd.t2_ms)
        err_iso = max(err_iso, np.abs(epg.run(seq, plain).signal - bloch.run(seq, plain).signal).max())
    rows.append(_row("N12", "sequences", "steam with diffusion lobes, 3 sets of flip angles, TM 100 to 900 ms, 3 tissues", "(1/2) sin a1 sin a2 sin a3 "
                     "exp(-TE / T2) exp(-TM / T1) exp(-b D), b from `gradients.b_value` (the grating is stored along z during TM and keeps "
                     "diffusing)", err, 1e-8))
    rows.append(_row("N12b", "sequences", "the same sequences without diffusion", "isochromat ensemble on two gradient axes", err_iso, 1e-12))

    # ---------------- spin echo with the grating kept (what the served `se` leaves out)
    from ..scanner import kspace
    change = []
    for scale, tbw, tr, te, t1v, ti, kap in ((1.0, 4.0, 500.0, 20.0, 800.0, None, 1.0), (0.8, 4.0, 300.0, 15.0, 1200.0, None, 1.0),
                                             (1.15, 2.0, 600.0, 20.0, 500.0, None, 1.0), (1.0, 8.0, 400.0, 20.0, 900.0, None, 1.0),
                                             (0.9, 4.0, 2000.0, 20.0, 1000.0, 400.0, 0.93)):
        zeta, wz = O.slice_columns(tbw)
        sps = O.grid([dict(t1_ms=t1v, t2_ms=1e15)], zeta, [scale])
        sel = B.Selective(tbw)
        head = [] if ti is None else [B.mz(-kap), B.delay(ti)]
        body = head + [B.rf(90, 0, selective=sel), B.delay(te / 2), B.gradient(1), B.rf(180, 90, selective=B.Selective(tbw, rephase=0.0, width=L.REFOCUS_WIDTH)),
                       B.gradient(1), B.delay(te / 2), B.adc("echo"), B.spoil(), B.delay(tr - te - (ti or 0.0))]
        got = np.sum(wz * periodic.steady_state([B.repeat(body, 1)], sps, kmax=8, method="matrix").signal[0]).real
        served = kspace.slice_signal(dict(sequence="se" if ti is None else "ir", tr_ms=tr, te_ms=te, ti_ms=ti), dict(t1_ms=t1v), scale * math.pi / 2, tbw, inv_eff=kap)[0][0].real
        change.append(100 * (got / served - 1))
    rows.append(_row("N13", "sequences", "se and ir through the slice with REAL crushers and an ideal spoiler after the echo, so that the grating "
                     "stored along z by an imperfect refocusing pulse comes back as a stimulated echo in the next TR (5 cases, B1 80 to 115 %): "
                     f"change against the served model, {min(change):+.2f} to {max(change):+.2f} %", "docs/physics/limits.md, block 8a: -2 to +39 %, "
                     "from a third, separately written integrator (`tools/external/independent_bloch.py`, row L1.4); measured: distance of the "
                     "rounded ends of the range", max(abs(round(min(change)) + 2), abs(round(max(change)) - 39)), 0, unit="points"))

    # ---------------- magnetisation transfer
    f_b, kab, t1a, t1b, t2b = 0.139, 3.2, 1000.0, 1000.0, 0.012          # white-matter-like sizes (not sourced)
    norm = 0.0
    for kind in O.LINESHAPES:                                           # the lines integrate to 1 over the angular frequency
        x = np.linspace(-1, 1, 40001)[1:-1]
        wgrid = np.tan(0.5 * math.pi * x) / (t2b * 1e-3)                # rad/s, the whole axis
        jac = 0.5 * math.pi / np.cos(0.5 * math.pi * x) ** 2 / (t2b * 1e-3)
        g = O.lineshape(kind, wgrid / (2 * math.pi), t2b) if kind != "superlorentzian" else O.lineshape(kind, np.where(wgrid == 0, 1e-9, wgrid) / (2 * math.pi), t2b)
        norm = max(norm, abs(np.sum(g * jac) * (x[1] - x[0]) - 1))
    rows.append(_row("M1", "sequences", "absorption lines of the bound pool (Lorentzian, Gaussian, super-Lorentzian): integral over the angular "
                     "frequency", "1 (the formulas are attributed to Morrison & Henkelman 1995, paper not opened; this checks their normalisation only)", norm, 2e-3))
    spm = O.spins(t1a, 80.0, pool="mt", f_b=f_b, k_ab_per_s=kab, t1b_ms=t1b, t2b_ms=t2b)
    err, raw, orders = 0.0, 0.0, []
    for offset, b1_ut in ((1000.0, 3.0), (2000.0, 5.0), (10000.0, 10.0)):
        g = float(O.lineshape("superlorentzian", offset, t2b))
        w_rf = math.pi * (gam * b1_ut * 1e-6) ** 2 * g                  # 1/s
        ra, rb, kf, kr = 1e3 / t1a, 1e3 / t1b, kab, kab / f_b
        closed = (rb * kf / ra + w_rf + rb + kr) / (kf / ra * (rb + w_rf) + w_rf + rb + kr)
        solved = np.linalg.solve(np.array([[-(ra + kf), kr], [kf, -(rb + kr + w_rf)]]), -np.array([ra, rb * f_b]))[0]
        vals = []
        for dt in (4.0, 2.0, 1.0):                                      # the pulse train tends to continuous irradiation
            half = B.rf(energy_ut2_ms=b1_ut ** 2 * dt / 2, offset_hz=offset)
            st = periodic.steady_state([B.repeat([half, B.delay(dt), half], 1)], spm, method="matrix").state
            vals.append(st.z[0, 0].real)
        errs = [abs(v - closed) for v in vals]
        limit = vals[2] + (vals[2] - vals[1]) / 3                       # Richardson: the error is second order in the spacing
        err, orders = max(err, abs(limit - closed), abs(solved - closed)), orders + [_order(errs)]
        raw = max(raw, errs[-1])
    rows.append(_row("M2", "sequences", "two pools under continuous off-resonance irradiation (1 to 10 kHz, 3 to 10 uT), as a train of saturation "
                     f"pulses every 4, 2, 1 ms extrapolated to zero spacing (at 1 ms the distance is {raw:.0e}): Mz of free water in the steady state", "(Rb kf / Ra + W + Rb + kr) / (kf / Ra (Rb + W) + W + Rb + kr), the "
                     "closed form of Henkelman et al. 1993 without its direct-saturation term (paper not opened; it equals the linear solve of "
                     "the two rate equations, which is in this row)", err, 1e-7))
    rows.append(_row("M2b", "sequences", "the same: order of convergence with the spacing of the pulses, 4 to 1 ms", "second order (the saturation "
                     "is split symmetrically around the delay)", abs(np.mean(orders) - 2), 0.1, unit="in the order"))
    err = err_iso = lim = 0.0
    deltas = []
    for fl in (6.0, 15.0, 30.0):
        tr, gap, energy, offset = 30.0, 2.0, 3000.0, 2000.0
        seq = L.mt_spgr(tr, fl, energy, offset, gap)
        got = periodic.steady_state(seq, spm).signal[0, 0].real
        sat = float(O.saturation(B.compile_ops(seq).ops[0], spm)[0])
        gen = np.array([[-(1 / t1a + kab * 1e-3), kab * 1e-3 / f_b], [kab * 1e-3, -(1 / t1b + kab * 1e-3 / f_b)]])
        m_inf = np.array([1.0, f_b])
        ops = [(np.diag([1.0, sat]), None), (scipy.linalg.expm(gen * gap), gap), (np.diag([math.cos(D(fl)), 1.0]), None), (scipy.linalg.expm(gen * (tr - gap)), tr - gap)]
        a_mat, c_vec = np.eye(2), np.zeros(2)
        for m, t in ops:
            a_mat, c_vec = m @ a_mat, m @ c_vec + (0 if t is None else m_inf - m @ m_inf)
        start = np.linalg.solve(np.eye(2) - a_mat, c_vec)
        before = ops[1][0] @ (ops[0][0] @ start) + m_inf - ops[1][0] @ m_inf
        err = max(err, abs(got - math.sin(D(fl)) * before[0]))
        fast = O.spins(t1a / 20, 80.0, pool="mt", f_b=f_b, k_ab_per_s=kab * 20, t1b_ms=t1b / 20, t2b_ms=t2b)
        err_iso = max(err_iso, abs(bloch.run(B.compile_ops(seq, repetitions=120), fast).signal[-1, 0] - periodic.steady_state(seq, fast).signal[0, 0]))
        tiny = O.spins(t1a, 80.0, pool="mt", f_b=1e-13, k_ab_per_s=kab * 1e-13 / f_b, t1b_ms=t1b, t2b_ms=t2b)
        lim = max(lim, abs(periodic.steady_state(seq, tiny).signal[0, 0].real / float(S.spgr(D(fl), tr, t1a)) - 1))
        s0 = periodic.steady_state(L.mt_spgr(tr, fl, 0.0, offset, gap), spm).signal[0, 0].real
        a = D(fl)
        r1tr = (a * a / 2) * s0 / (a - s0)                              # small-angle Ernst form, amplitude 1
        deltas.append(a * r1tr / got - a * a / 2 - r1tr)
    rows.append(_row("M3", "sequences", "mt_spgr (saturation pulse, 2 ms, excitation of 6, 15, 30 deg, TR 30 ms; bound fraction 13.9 %): the "
                     "periodic signal", "fixed point of the 2 x 2 chain [saturate bound pool, exchange and recover, excite free pool, exchange and "
                     "recover] with `scipy.linalg.expm` (the matrix form used for pulsed MT, e.g. Yarnykh 2012; written here independently)",
                     err, 1e-12))
    rows.append(_row("M3b", "sequences", "the same at T1 / 20; and with a vanishing bound pool", "isochromat ensemble stepped through 120 TR; "
                     "the Ernst signal", max(err_iso, lim), 1e-8))
    spread = (max(deltas) - min(deltas)) / np.mean(deltas)
    rows.append(_row("M4", "sequences", f"MT saturation delta from the signals with and without the pulse at 6, 15, 30 deg: "
                     f"{100 * min(deltas):.2f} to {100 * max(deltas):.2f} %; measured: its spread over the flip angles",
                     "delta = (A a / S - 1) R1 TR - a^2 / 2 is nearly independent of the flip angle (Helms et al. 2008, as restated by Tabelow "
                     "et al. 2019); an approximation, so a band, not an equality", spread, 0.2, unit="relative"))
    return rows


# ================================================================================================== speed
def speed_table(repeat=2):
    """Time of the amplitudes of 6 tissues x 85 slice columns (510 columns) for every library sequence: stepped
    from equilibrium against the periodic solve. One process, one thread; the best of ``repeat`` runs."""
    from ..scanner import kspace
    tis = _tissues()
    zeta, _ = O.slice_columns(4.0, two_sided=False)
    sp = O.grid(tis, zeta)
    flip, rot = B.Selective(4.0, model="flip"), B.Selective(4.0)

    def best(fn):
        out = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            val = fn()
            out.append(time.perf_counter() - t0)
        return min(out), val

    # (name, sequence, repetitions stepped, orders kept when stepping, orders kept in the periodic solve, columns)
    cases = [
        ("spgr, RF-spoiled, TR 10 ms; stepping keeps all 500 orders", L.spgr(10.0, 20.0, selective=flip), 500, None, None, sp),
        ("the same; stepping keeps 64 orders", L.spgr(10.0, 20.0, selective=flip), 500, 64, None, sp),
        ("spgr, ideal spoiling", L.spgr(10.0, 20.0, spoiling="ideal", selective=flip), 500, None, None, sp),
        ("afi, TR 20 + 100 ms", L.afi(20.0, 5, 60.0, selective=flip), 600, 64, None, sp),
        ("epi, TR 2000 ms", L.epi(2000.0, 90.0, selective=flip), 10, None, None, sp),
        ("se, TR 500 ms", L.se(500.0, 20.0, selective=rot), 10, None, None, sp),
        ("ir, TR 2000 ms", L.ir(2000.0, 20.0, 400.0, selective=rot), 10, None, None, sp),
        ("bssfp, TR 5 ms", L.bssfp(5.0, 40.0, selective=flip), 1000, None, None, sp),
        ("fse, 16 echoes", L.fse(3000.0, 8.0, 16, 150.0, selective=rot), 4, None, None, sp),
        ("fse, 32 echoes", L.fse(3000.0, 8.0, 32, 150.0, selective=rot), 4, None, None, sp),
        ("fse, 64 echoes", L.fse(3000.0, 8.0, 64, 150.0, selective=rot), 4, None, None, sp),
        ("fse, 128 echoes", L.fse(3000.0, 8.0, 128, 150.0, selective=rot), 4, None, None, sp),
        ("fse, 128 echoes, 32 orders kept", L.fse(3000.0, 8.0, 128, 150.0, selective=rot), 4, 32, 32, sp),
        ("mese, 16 echoes", L.mese(2000.0, 10.0, 16, 160.0, selective=rot), 4, None, None, sp),
        ("megre, 4 echoes", L.megre(40.0, 20.0, (3.0, 8.0, 13.0, 18.0), selective=flip), 200, None, None, sp),
        ("spgr_lobe, TR 10 ms", L.spgr_lobe(10.0, 20.0, 3.0, selective=flip), 500, 64, None, sp),
        ("dess, TR 15 ms", L.dess(15.0, 30.0, selective=flip), 400, 64, None, sp),
        ("mprage, 128 pulses", L.mprage(900.0, 128, 8.0, 9.0, 600.0), 4, None, None, sp),
        ("t2prep", L.t2prep(1000.0, 50.0), 10, None, None, sp),
        ("satrec", L.satrec(800.0, 300.0), 10, None, None, sp),
        ("dwse", L.dwse(3000.0, 80.0, 30.0, 20.0, 40.0, 1.0), 4, None, None, sp),
        ("steam", L.steam(3000.0, 40.0, 100.0, 30.0, 10.0, 1.0), 4, None, None, sp),
        ("mt_spgr, two pools", L.mt_spgr(30.0, 10.0, 2000.0, 2000.0), 300, None, None,
         O.grid([dict(t, f_b=0.1, k_ab_per_s=3.0, t1b_ms=1000.0, t2b_ms=0.012) for t in tis], zeta, pool="mt")),
    ]
    # the engine's shortcut for RF-spoiled sequences: 16 flip angles per tissue instead of 85 columns
    nodes = O.grid(tis, b1=0.5 * (1 + np.cos(math.pi * (np.arange(16) + 0.5) / 16)) * 1.02 + 1e-3)
    cases += [("spgr, RF-spoiled, on 16 flip angles per tissue (hard pulse)", L.spgr(10.0, 20.0), 500, 64, None, nodes),
              ("afi on 16 flip angles per tissue (hard pulse)", L.afi(20.0, 5, 60.0), 600, 64, None, nodes)]
    out = []
    for name, seq, reps, k_step, k_per, cols in cases:
        t_step, a = best(lambda: epg.run(seq, cols, kmax=k_step, frame=True, repetitions=reps, keep_state=False).signal)        # noqa: B023
        t_per, b = best(lambda: periodic.steady_state(seq, cols, kmax=k_per).signal)                                          # noqa: B023
        gap = float(np.max(np.abs(a[-b.shape[0]:] - b)) / np.max(np.abs(b)))
        out.append(dict(sequence=name, columns=cols.n, periods=reps, stepped_s=t_step, periodic_s=t_per, gap=gap))
    rp = dict(sequence="spgr", tr_ms=10.0, rf_phase_increment_deg=117.0, spoiler_mt_ms_per_m=100.0)
    kspace.epg._cached.cache_clear()
    t0 = time.perf_counter()
    for c in tis:
        kspace.slice_signal(rp, c, D(20.0), 4.0)
    out.append(dict(sequence="for scale, the engine today: `kspace.slice_signal`, spgr tier 2, the same 6 tissues (16 flip angles per tissue, "
                    "first call)", columns=96, periods=0, stepped_s=float("nan"), periodic_s=time.perf_counter() - t0, gap=float("nan")))
    return out


def truncation_table():
    """What dropping orders costs in an echo train: (echoes, refocusing angle, orders kept, largest change of a
    slice-integrated echo relative to the first echo), for 6 tissues x 85 columns."""
    tis = _tissues()
    zeta, wz = O.slice_columns(4.0, two_sided=False)
    sp = O.grid(tis, zeta)
    out = []
    for etl, flip in ((64, 150.0), (128, 150.0), (128, 120.0)):
        seq = L.fse(3000.0, 8.0, etl, flip, selective=B.Selective(4.0))
        full = (periodic.steady_state(seq, sp).signal.real.reshape(etl, len(tis), -1) * wz).sum(-1)
        for kmax in (16, 32, 64):
            part = (periodic.steady_state(seq, sp, kmax=kmax).signal.real.reshape(etl, len(tis), -1) * wz).sum(-1)
            out.append((etl, flip, kmax, float(np.abs(part - full).max() / np.abs(full).max())))
    return out


def speed_markdown(rows):
    out = ["| Sequence | Columns | Repetitions stepped | Stepped (s) | Periodic solve (s) | Stepped value against the periodic one |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        f = lambda v, fmt: "–" if not np.isfinite(v) else format(v, fmt)      # noqa: E731
        out.append(f"| {r['sequence']} | {r['columns']} | {r['periods'] or '–'} | {f(r['stepped_s'], '.3f')} | {f(r['periodic_s'], '.3f')} | {f(r['gap'], '.0e')} |")
    return "\n".join(out)


# ================================================================================================== running
CHECKS = dict(agreement=check_agreement, engine=check_engine, sequences=check_sequences)


def run(groups=None):
    t0 = time.time()
    rows = []
    for name, fn in CHECKS.items():
        if not groups or name in groups:
            rows += fn()
    return rows, time.time() - t0


def _number(x):
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.1e}"
    return f"{x:.4g}"


def markdown_table(rows):
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
                       f"{'pass' if r['ok'] else '**FAIL**'} |")
        out.append("")
    return "\n".join(out).rstrip()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--speed" in argv:
        print(speed_markdown(speed_table()))
        print("\n| Echoes | Refocusing angle (deg) | Orders kept | Change of the slice-integrated echoes, relative to the first echo |\n|---:|---:|---:|---:|")
        for etl, flip, kmax, err in truncation_table():
            print(f"| {etl} | {flip:.0f} | {kmax} | {err:.0e} |")
        return 0
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
