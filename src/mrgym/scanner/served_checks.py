"""Checks of the sequences a console serves from the block-list core (``mrgym validate``, group ``core``, rows K2 on).

Every row compares a number the scanner serves (``Engine.acquire`` of a realised protocol of ``fse``, ``mese``,
``megre``, ``dwse``, or a stage of it) with a closed form, or with a computation that shares no code with what is
served: the isochromat ensemble of the core, the phase graph of the design layer (``mrgym.design.sequence``), a
direct evaluation of the Fourier sum, the integral of a gradient waveform (``mrgym.design.gradients``), the engine's
own closed forms where the sequences overlap. The reference of each row is named in the row.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import bloch as core_bloch
from ..core import blocks as B
from ..core import operators as O
from ..design import gradients as DG
from ..design import pulses as design_pulses
from ..design import sequence as Q
from . import console, kspace, rf, served, tissue
from . import world as W

HARD = dict(excitation="nonselective")
TISSUES = dict(a=dict(pd=1.0, t1_ms=900.0, t2_ms=80.0, t2s_ms=60.0, t2p_ms=240.0, adc_um2_ms=0.8, fat=False),
               b=dict(pd=0.9, t1_ms=2500.0, t2_ms=400.0, t2s_ms=250.0, t2p_ms=1 / (1 / 250.0 - 1 / 400.0), adc_um2_ms=3.0, fat=False),
               c=dict(pd=0.8, t1_ms=380.0, t2_ms=50.0, t2s_ms=35.0, t2p_ms=1 / (1 / 35.0 - 1 / 50.0), adc_um2_ms=1.6, fat=False))


def _row(*args, **kw):
    from .. import validate
    return validate._row(*args, **kw)


def _disc(r, cy=0.0, cx=0.0):
    return lambda y, x: np.hypot(y - cy, x - cx) - r


def _strip(hy, hx):
    return lambda y, x: np.maximum(np.abs(y) - hy, np.abs(x) - hx)


def _world(shapes, n=320, tissues=None, b1=None, rx=None, b0=None, b0_t=3.0, **system):
    """A flat world around hand-made shapes whose console offers every sequence; optional smooth field patterns
    (functions of y and x in mm)."""
    ph = W.shapes_phantom(shapes, b0_t=b0_t, n=n, tissues=dict(TISSUES, **(tissues or {})))
    w = W.flat_world(ph, **system)
    y, x = np.meshgrid(*(2 * [(np.arange(n) - n // 2) * (ph["extent_mm"] / n)]), indexing="ij")
    for key, f, kind in (("b1", b1, float), ("rx", rx, complex), ("b0_hz", b0, float)):
        if f is not None:
            w["fields"][key] = np.asarray(f(y, x), kind)
    w["console"].update(sequences="all", min_pixel_mm=0.5)
    return w


def _plan(w, **protocol):
    pl = console.plan(protocol, w["console"])
    if not pl["accept"]:
        raise AssertionError("; ".join(v["message"] for v in pl["violations"]))
    return pl


def _region(img, rp, cy, cx, r):
    """Mean of the complex images [image, y, x] inside a circle (mm)."""
    yy = (np.arange(rp["matrix_pe"]) - rp["matrix_pe"] // 2) * rp["fov_pe_mm"] / rp["matrix_pe"]
    xx = (np.arange(rp["matrix"]) - rp["matrix"] // 2) * rp["fov_mm"] / rp["matrix"]
    mask = np.hypot(yy[:, None] - cy, xx[None, :] - cx) < r
    return img[:, mask].mean(1)


def _unit_amplitudes(eng, rp, name):
    """What ``Engine.amplitudes`` serves for the first voxel of a tissue, per unit of weight and proton density."""
    c = eng.w["names"].index(name)
    scale = (eng.w["weights"][c][eng.sup[c]] * eng.w["tissues"][name]["pd"])
    i = int(np.argmax(scale))
    return np.array([a[i] / scale[i] for a in eng.amplitudes(rp, c)])


# ================================================================================================== echo trains: amplitudes
def train_closed_form(n, esp, tr, t1, t2):
    """Echoes of a train of exact 180 degree pulses after an exact 90 degree pulse, in the periodic state: Mz before
    the excitation x exp(-k esp / T2). Mz follows one number through the repetition: zero after the excitation,
    recovery for esp / 2, inversion, (n - 1) times recovery for esp and inversion, recovery for the rest of TR."""
    z = -(1 - math.exp(-esp / (2 * t1)))
    for _ in range(n - 1):
        z = -(1 - (1 - z) * math.exp(-esp / t1))
    z = 1 - (1 - z) * math.exp(-(tr - (n - 0.5) * esp) / t1)
    return z * np.exp(-np.arange(1, n + 1) * esp / t2)


def echo_pathway_b(rp):
    """b-value (s/mm^2) that the gradients of the first echo of a realised protocol give the spin-echo pathway (order 0
    after the excitation, negated by the refocusing pulse): the sum of ``blocks.b_increment`` over the blocks of the
    registry's block list up to the first sample. For ``fse`` and ``mese``: one pair of crushers."""
    seq, _ = console.REGISTRY[rp["sequence"]].blocks(rp)
    prog = B.compile_ops(seq, frame=False)
    total, k, pulses = 0.0, 0.0, 0
    for op in prog.ops:
        if op.kind == "rf":
            pulses += 1
            k = -k if pulses == 2 else k
        elif op.kind == "adc":
            break
        elif op.kind == "free" and pulses:
            total += B.b_increment(k, op.n[0], op.ms, prog.cycle_mm[0], op.shape)
            k += op.n[0]
    return total


def check_train_amplitudes():
    w = _world([("a", _disc(60.0)), ("b", _disc(20.0, 0.0, 25.0)), ("c", _disc(15.0, 30.0, -20.0))], n=128)
    eng = kspace.Engine(w)
    worst, count, b_pair = 0.0, 0, 0.0
    for seq, n, tr, esp in (("fse", 8, 2500.0, 12.0), ("fse", 32, 4000.0, 9.0), ("mese", 16, 2600.0, 10.0), ("mese", 32, 12000.0, 15.0), ("mese", 32, 600.0, 15.0)):
        rp = _plan(w, sequence=seq, tr_ms=tr, echo_spacing_ms=esp, matrix=64, **{"etl" if seq == "fse" else "n_echoes": n}, **HARD)["realised"]
        b_pair = echo_pathway_b(rp)
        for name in ("a", "b", "c"):
            cls = w["tissues"][name]
            want = train_closed_form(n, rp["echo_spacing_ms"], rp["tr_ms"], cls["t1_ms"], cls["t2_ms"]) \
                * np.exp(-np.arange(1, n + 1) * b_pair * cls["adc_um2_ms"] * 1e-3)
            worst, count = max(worst, float(np.abs(_unit_amplitudes(eng, rp, name) - want).max() / want.max())), count + n
    rows = [_row("J7", "core", f"`fse` and `mese`, exact pulses (hard, 90 and 180 degrees, transmit scale 1): every echo of trains of 8 to 32 echoes at TR 0.6 "
                 f"to 12 s as `Engine.amplitudes` serves it, 3 materials ({count} echoes)", "Mz before the excitation x exp(-k ESP / T2) x exp(-k b D), with "
                 "Mz from the recursion of the longitudinal magnetisation through the repetition (recovery, inversion by every refocusing pulse; "
                 f"written here) and b = {b_pair:.4f} s/mm^2 the diffusion weighting of one pair of crusher lobes (`blocks.b_increment` along the echo "
                 "pathway). Exact pulses store no grating, and what is transverse at the end of the repetition is not carried", worst, 1e-12,
                 f"{worst:.1e} relative")]

    # ---- the reduction to the refocusing flip angle against the rotation of every slice column
    wb = _world([("a", _disc(60.0)), ("b", _disc(20.0, 0.0, 25.0)), ("c", _disc(15.0, 30.0, -20.0))], n=128,
                b1=lambda y, x: 0.78 + 0.37 * np.exp(-(y * y + x * x) / (2 * 55.0 ** 2)))
    engb = kspace.Engine(wb)
    worst, cases = 0.0, 0
    for seq, n, beta, pulse, more in (("fse", 8, 180.0, "normal", {}), ("mese", 16, 150.0, "fast", {}), ("fse", 32, 120.0, "sharp", {}),
                                      ("fse", 12, 70.0, "fast", dict(tx_scale=1.25)), ("mese", 10, 180.0, "normal", HARD)):
        rp = _plan(wb, sequence=seq, tr_ms=1500.0, refocus_deg=beta, rf_pulse=pulse, matrix=64, **{"etl" if seq == "fse" else "n_echoes": n}, **more)["realised"]
        fast = served.table(engb, rp, solver="train", nodes=4, columns=1, split=False)["a"]
        full = served.table(engb, rp, solver="columns", nodes=4)["a"]
        worst = max(worst, float(np.abs(fast - full).max() / np.abs(full).max()))
        cases += 1
    rows.append(_row("J8", "core", f"echo trains through the slice as they are served: the train stepped from every state that the last repetition can "
                     f"leave, on (echoes + 1) Chebyshev points of the cosine of the refocusing flip angle, from which the periodic state and every echo "
                     f"of every slice column follow by one small linear solve (`served.train_response`, `carry_columns`); {cases} cases (fast, normal, "
                     "sharp and hard pulses, 8 to 32 echoes, 70 to 180 degrees, transmit scale 0.78 to 1.44, TR 1500 ms), 4 transmit scales x every "
                     "slice column x 3 materials", "`periodic.steady_state` (method carry, the same box of states) of the registry's block list with the "
                     "Cayley-Klein rotation of the excitation and of the refocusing pulse at every column (no reduction: it shows that a refocusing "
                     "rotation acts through its flip angle, the phase of its axis and a translation along the crusher axis alone, and that the "
                     "response of the train is a polynomial in the cosine of that angle, times its sine where it connects a transverse with a "
                     "longitudinal state)", worst, 1e-10, f"{worst:.1e} relative"))

    # ---- the same amplitudes against the isochromat ensemble (no phase graph), stepped to the periodic state. Nothing
    # diffuses in the ensemble, so the materials have no diffusion here (rows X7, N11c check diffusion in the phase graph).
    quick = dict(a=dict(TISSUES["a"], t1_ms=160.0, t2_ms=45.0, adc_um2_ms=0.0), b=dict(TISSUES["b"], t1_ms=280.0, t2_ms=90.0, adc_um2_ms=0.0))
    wq = _world([("a", _disc(60.0)), ("b", _disc(20.0, 0.0, 25.0))], n=96, tissues=quick,
                b1=lambda y, x: 0.85 + 0.25 * np.exp(-(y * y + x * x) / (2 * 55.0 ** 2)))
    engq = kspace.Engine(wq)
    ens_err, box_err, effect, text = 0.0, 0.0, 0.0, []
    for tr, reps in ((1500.0, 6), (330.0, 22), (200.0, 30)):
        rp = _plan(wq, sequence="mese", tr_ms=tr, n_echoes=6, refocus_deg=140.0, matrix=64)["realised"]
        tab = served.table(engq, rp, solver="train", nodes=3, columns=1, carry=((45,), False), split=False)
        tbw, window = engq.pulse(rp)
        seq, labels = console.REGISTRY["mese"].blocks(rp, selective=B.Selective(tbw, window))
        zeta, wz = O.slice_columns(tbw, two_sided=False, window=window)
        sp = O.grid([served._tissue(quick[k]) for k in ("a", "b")], zeta, tab["gn"])
        ens = core_bloch.run(B.compile_ops(seq, frame=False, repetitions=reps), sp, n_sub=512).signal[-len(labels):]
        ens = np.moveaxis((ens.reshape(len(labels), 2, zeta.size, tab["gn"].size).real * wz[None, None, :, None]).sum(2), 0, -1)
        big = float(np.abs(ens).max())
        ens_err = max(ens_err, float(np.abs(ens - tab["a"]).max() / big))
        box_err = max(box_err, float(np.abs(served.table(engq, rp, solver="train", nodes=3, columns=1, split=False)["a"] - ens).max() / big))
        ideal = served.table(engq, rp, solver="train", nodes=3, columns=1, carry=((0,), False))["a"]
        effect = max(effect, float(np.abs(ideal - ens).max() / big))
        text.append(f"{100 * float(np.abs(ideal - ens).max() / big):.1f} %")
    rows.append(_row("J9", "core", "`mese` through the slice (6 echoes, 140 degrees, normal pulse, transmit scale 0.85 to 1.1, T1 160 and 280 ms, T2 45 and "
                     "90 ms) at TR 1500, 330 and 200 ms, where the gratings along z reach the next excitation: the table as it is solved, with 45 orders "
                     "carried", "isochromat ensemble of the core (`core.bloch`: real 3 x 3 rotations, no phase graph, 512 isochromats per crusher "
                     "cycle) on the registry's block list (real crushers and spoiler, the transverse magnetisation set to zero at the end of the "
                     "repetition), stepped through 6, 22 and 30 repetitions from equilibrium, same slice columns", ens_err, 1e-8,
                     f"{ens_err:.1e} relative"))
    rows.append(_row("J9b", "core", f"the same three scans as they are SERVED: the box of carried gratings cut at a factor of {served.CARRY_TOL:g} by relaxation "
                     f"and diffusion, at most {served.CARRY_CAP[0]} orders (no diffusion here, so the box is at its largest)", "the same ensemble: what "
                     "the cut of the box costs where it costs most (TR of one to two T1, no diffusion)", box_err, 1e-3, f"{box_err:.1e} relative"))
    rows.append(_row("J9c", "core", "what an ideal crusher at the end of the repetition would have left out in those scans: " + ", ".join(text) + " of the "
                     "largest echo at TR 1500, 330, 200 ms; measured: the largest", "the ensemble: the stimulated echoes of the last repetition are part "
                     "of the signal at a TR of a few T1 (engine before this row: the ideal crusher)", effect, 0.02, f"{effect:.3f} relative", kind="min"))

    # ---- what is NOT exact: interpolation over the transmit scale and the quadrature through the slice
    wide = _world([("a", _disc(70.0)), ("b", _disc(20.0, 0.0, 25.0)), ("c", _disc(15.0, 30.0, -20.0))], n=128,
                  b1=lambda y, x: 0.75 + 0.27 * np.exp(-(y * y + x * x) / (2 * 28.0 ** 2)))
    engw = kspace.Engine(wide)
    lo, hi = engw.b1_range
    fine = np.linspace(lo, hi, 41)
    classes = served._classes(engw)
    node_err, quad_err, counts = 0.0, 0.0, []
    for n, beta, pulse in ((8, 180.0, "normal"), (16, 180.0, "fast"), (32, 180.0, "normal"), (32, 150.0, "fast"), (32, 120.0, "sharp"), (32, 60.0, "fast"),
                           (64, 180.0, "normal"), (64, 140.0, "fast")):
        rp = _plan(wide, sequence="fse", tr_ms=3000.0, etl=min(n, 32), refocus_deg=beta, rf_pulse=pulse, matrix=64)["realised"]
        if n > 32:                                                      # beyond what the console accepts: a realised protocol written by hand
            rp = dict(rp, etl=n)
        ref = served._table_train(engw, rp, classes, fine, columns=2 * served.TRAIN_COLUMNS)
        tab = served.table(engw, rp)
        got = np.einsum("fj,cjs->cfs", kspace.bary_matrix(tab["gn"], tab["gw"], fine), tab["a"])
        node_err = max(node_err, float(np.abs(got - ref).max() / np.abs(ref).max()))
        quad_err = max(quad_err, float(np.abs(served._table_train(engw, rp, classes, fine) - ref).max() / np.abs(ref).max()))
        counts.append(tab["gn"].size)
    rows.append(_row("J10", "core", f"echo trains: interpolation of the table over the transmit scale ({min(counts)} to {max(counts)} Chebyshev nodes by "
                     f"`served.node_count`), 8 trains of 8 to 64 echoes (the console accepts 32) at 60 to 180 degrees, transmit scale {lo:.2f} to {hi:.2f}, 41 scales x 3 materials; "
                     "error relative to the largest echo (it includes the quadrature of the next row)", "the closed form of row J8 evaluated at "
                     "each of the 41 scales, on twice the slice columns", node_err, 2e-6, f"{node_err:.1e} relative"))
    rows.append(_row("J11", "core", f"echo trains: the integral through the slice on {served.TRAIN_COLUMNS} x the columns of the engine's quadrature (an echo "
                     "of a long train oscillates through the slice with the local refocusing angle; on the engine's own columns a train of 32 echoes "
                     "at 150 degrees with the fast pulse is off by 6e-3)", "the same on twice as many columns", quad_err, 1e-7, f"{quad_err:.1e} relative"))
    return rows


# ================================================================================================== image formation with line-dependent amplitudes
def direct_kspace(eng, rp, tab, plan):
    """The Fourier sum of the signal equation, line by line: for every sample of the cycle the amplitude map of every
    class is transformed with explicit DFT matrices (no chirp-z transform, no factorisation), multiplied by the
    readout evolution of the class, and every line is taken from the sample that fills it. No displacement by
    off-resonance (use an engine without the ``b0shift`` effect)."""
    w, n, d = eng.w, eng.n, eng.d
    ro_os = console.SYSTEM["ro_os"]
    k_ro, k_pe = ro_os * rp["matrix"], rp["lines_full"]
    fov_ro, fov_pe = ro_os * rp["fov_mm"], rp["fov_pe_mm"] * k_pe / rp["matrix_pe"]
    pos = (np.arange(n) - n // 2) * d
    cy, cx = rp.get("centre_y_mm", 0.0), rp.get("centre_x_mm", 0.0)
    p0, r0 = (cx, cy) if rp["pe_dir"] == "x" else (cy, cx)
    e_pe = np.exp(-2j * np.pi * np.outer(np.arange(k_pe) - k_pe // 2, pos - p0) / fov_pe)          # [line, master row]
    e_ro = np.exp(-2j * np.pi * np.outer(pos - r0, np.arange(k_ro) - k_ro // 2) / fov_ro)          # [master column, readout sample]
    tau = rp["ro_polarity"] * (np.arange(k_ro) - k_ro // 2) / (k_ro * rp["bandwidth_hz_per_px"])
    keep = np.abs(pos - r0) <= fov_ro / 2 if "aa_filter" in eng.effects else np.ones(n, bool)
    n_s = tab["a"].shape[2]
    per_sample = np.zeros((n_s, k_pe, k_ro), complex)
    for c in tab["classes"]:
        cls = w["tissues"][w["names"][c]]
        g = served.readout_function(eng, rp, cls, tau)
        maps = served.amplitudes(eng, rp, c, tab)
        for s in range(n_s):
            img = np.zeros((n, n), complex)
            img[eng.sup[c]] = maps[s]
            img *= w["fields"]["rx"]
            if rp["pe_dir"] == "x":
                img = img.T
            per_sample[s] += (e_pe @ (img * keep[None, :]) @ e_ro) * g[None, :]
    acquired = eng._acquired(rp)
    out = np.zeros((len(plan.images), k_pe, k_ro), complex)
    for i, (s, _) in enumerate(plan.images):
        which = np.full(k_pe, s) if plan.lines is None else np.asarray(plan.lines[i])
        for p in np.flatnonzero(acquired):
            out[i, p] = per_sample[which[p], p]
    return out * (d * d * rp["slice_mm"])


def check_line_formation():
    shapes = [("a", _disc(55.0)), ("b", _disc(18.0, 5.0, 22.0)), ("subcut_fat", _disc(14.0, -25.0, -18.0))]
    fields = dict(b1=lambda y, x: 0.82 + 0.3 * np.exp(-(y * y + x * x) / (2 * 60.0 ** 2)),
                  rx=lambda y, x: (1 + 0.15 * x / 100) * np.exp(0.4j * y / 100))
    w = _world(shapes, n=192, sigma0=0.02, **fields)
    still = kspace.Engine(w, effects=kspace.ALL - {"b0shift"})
    rows, direct_err, ratio, bound_sd, terms, actual_sd = [], 0.0, 0.0, 0.0, [], 0.0
    cases = (dict(sequence="fse", tr_ms=2500.0, etl=8, te_ms=40.0, echo_order="linear", matrix=48, refocus_deg=150.0),
             dict(sequence="fse", tr_ms=2500.0, etl=6, echo_order="centric", matrix=48, partial_fourier=0.75, pe_dir="x", centre_x_mm=10.0, centre_y_mm=-8.0),
             dict(sequence="mese", tr_ms=1200.0, n_echoes=5, matrix=32, phase_fov=0.75, refocus_deg=130.0, rf_pulse="fast"))
    for prot in cases:
        rp = _plan(w, **prot)["realised"]
        tab, plan = served.table(still, rp), console.sampling(rp)
        exact = served.form_lines(still, rp, tab, plan, tol=0.0)
        want = direct_kspace(still, rp, tab, plan)
        direct_err = max(direct_err, float(np.abs(exact - want).max() / np.abs(want).max()))
        det = {}
        cut = served.form_lines(still, rp, tab, plan, details=det)
        sd = served.noise_per_sample(still, rp, served.MAX_AVERAGES)
        err = float(np.abs(cut - exact).max())
        ratio, bound_sd, actual_sd = max(ratio, err / det["bound"]), max(bound_sd, det["bound"] / sd), max(actual_sd, err / sd)
        terms.append(det["terms"])
    rows.append(_row("J12", "core", "k-space of `fse` (linear and centric order, partial Fourier, phase encoding along x, an off-centre field of view) and "
                     "`mese` on a phantom of 3 materials (one of them fat) with a transmit pattern and a complex receive pattern, every term of the "
                     "line basis kept: every sample of every image", "the Fourier sum of the signal equation line by line with explicit DFT matrices: "
                     "for every echo of the train the amplitude map of every class, times the evolution of the class along the readout, and every line "
                     "taken from the echo that fills it (`served_checks.direct_kspace`; no chirp-z transform, no factorisation)", direct_err, 1e-9,
                     f"{direct_err:.1e} relative"))
    # ---- the same cut with the displacement by off-resonance (its gain kappa is in the bound), and what the cut does to the image:
    # worlds from nearly noiseless to useless, since the tolerance follows the noise
    pixel, k_err = 0.0, 0.0
    for sigma0 in (0.01, 1.0, 100.0, 1000.0):
        wn = _world(shapes + [("b", _disc(4.0, 20.0, -25.0))], n=160, sigma0=sigma0, b0=lambda y, x: 30.0 * x / 100 - 12.0, **fields)
        moved = kspace.Engine(wn)
        for prot in (cases[0], dict(sequence="mese", tr_ms=2500.0, n_echoes=16, matrix=64, refocus_deg=150.0)):
            rp = _plan(wn, **prot)["realised"]
            tab, plan = served.table(moved, rp), console.sampling(rp)
            exact = served.form_lines(moved, rp, tab, plan, tol=0.0)
            det = {}
            cut = served.form_lines(moved, rp, tab, plan, details=det)
            err = float(np.abs(cut - exact).max())
            ratio, k_err = max(ratio, err / det["bound"]), max(k_err, err / served.noise_per_sample(moved, rp, served.MAX_AVERAGES))
            bound_sd = max(bound_sd, det["bound"] / served.noise_per_sample(moved, rp, served.MAX_AVERAGES))
            terms.append(det["terms"])
            image = np.abs(np.array([moved.recon(a, rp) - moved.recon(b, rp) for a, b in zip(cut, exact)])).max()
            pixel = max(pixel, float(image / moved.noise_sd(dict(rp, averages=served.MAX_AVERAGES))))
    rows.append(_row("J13", "core", f"the same scans with the line basis cut as they are served, and 8 more with the displacement by an off-resonance "
                     f"field at receiver noise levels from 0.01 to 1000 ({min(terms)} to {max(terms)} terms): largest error of a k-space sample, as a "
                     "fraction of the bound the cut is made with", "the bound of `served.line_basis` (gain of the displacement operator x the sum over "
                     "the classes of the largest sample a class can make x the largest residual of its amplitude x the Lebesgue constant of the "
                     "interpolation); the exact side is the full basis", ratio, 1.0, f"{ratio:.2f} of the bound"))
    rows.append(_row("J14", "core", f"what the cut does in the image, same scans: largest error of a pixel in standard deviations of the noise of a pixel at "
                     f"{served.MAX_AVERAGES} averages (with one average {math.sqrt(served.MAX_AVERAGES):.0f} times less). The bound holds the error of a "
                     f"k-space sample below {served.LINE_NOISE_FRACTION:g} of the noise of a sample (reached: {bound_sd:.3f}; measured error {max(k_err, actual_sd):.4f}); "
                     "in the image the errors of the samples add coherently and the noise does not", "a measurement, not a bound: it must stay a "
                     "small fraction of the noise of a pixel", pixel, 0.1, f"{pixel:.3f} sd"))

    # ---- the same path against the engine's own formation where the amplitude does not change from line to line
    wf = _world(shapes, n=192, b0=lambda y, x: 25.0 * x / 100 - 10.0, **fields)
    eng = kspace.Engine(wf)
    se = _plan(wf, sequence="se", tr_ms=900.0, te_ms=30.0, matrix=64)["realised"]
    dw = _plan(wf, sequence="dwse", tr_ms=900.0, te_ms=30.0, matrix=64, b_value=0.0)["realised"]
    ideal = {k: v for k, v in dw.items() if k != "crusher_g_mt_m"}      # a protocol written by hand, without the console's lobes: perfect crushers
    a, b, c = eng.acquire(se)[0], eng.acquire(ideal)[0], eng.acquire(dw)[0]
    same, real = float(np.abs(a - b).max() / np.abs(a).max()), float(np.abs(a - c).max() / np.abs(a).max())
    rows.append(_row("J15", "core", "`dwse` with b = 0 and perfect crushers (a realised protocol without the console's lobes; served from the core through "
                     "the line path: slice columns by the core, the Fourier integrals along the phase-encode axis first, one factorisation) on a phantom "
                     "with a transmit pattern, a receive pattern, fat and an off-resonance field: the image", "`se` at the same TR and TE (the engine's "
                     "closed form through its own formation: readout first, its readout basis cut at `kspace.SVD_TOL` = 1e-4 of the largest singular "
                     "value, which is what the difference measures)", same, 1e-4, f"{same:.1e} relative"))
    rows.append(_row("J15b", "core", "`dwse` with b = 0 as the console plays it (crusher lobes of 1 ms, a spoiler, everything that survives the "
                     "repetition kept) against `se` in that scan (TR 900 ms): largest difference of a pixel relative to the largest pixel", "not an "
                     "equality: `se` and `ir` keep the perfect crushers of engine 0.3.0 (limits.md, block 8a); `dwse` has the stimulated echo of the "
                     "last repetition. Its size here", real, 1e-3, f"{real:.3f} relative", kind="min"))
    return rows


def _fwhm(profile, upsample=32):
    """Full width at half maximum, in samples, of the magnitude of a profile after Fourier interpolation."""
    n = profile.size
    k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(profile)))
    z = np.zeros(n * upsample, complex)
    z[(n * upsample) // 2 - n // 2:(n * upsample) // 2 - n // 2 + n] = k
    fine = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(z))))
    i = int(np.argmax(fine))
    half = fine[i] / 2
    lo = i
    while fine[lo] > half:
        lo -= 1
    hi = i
    while fine[hi] > half:
        hi += 1
    left = lo + (half - fine[lo]) / (fine[lo + 1] - fine[lo])
    right = hi - (half - fine[hi]) / (fine[hi - 1] - fine[hi])
    return (right - left) / upsample


def check_blur():
    rows, cf_err, ref_err, text = [], 0.0, 0.0, []
    for matrix, etl, t2, esp, order in ((32, 32, 80.0, 10.0, "centric"), (64, 16, 40.0, 10.0, "centric"), (64, 32, 60.0, 9.0, "centric")):
        px = 256.0 / matrix
        thin = 0.25 * px
        w = _world([("t", _strip(thin / 2, 70.0))], n=640, tissues=dict(t=dict(TISSUES["a"], t2_ms=t2, t1_ms=800.0)))
        eng = kspace.Engine(w)
        rp = _plan(w, sequence="fse", tr_ms=4000.0, etl=etl, echo_spacing_ms=esp, echo_order=order, matrix=matrix, fov_mm=256.0, **HARD)["realised"]
        img = eng.acquire(rp)[0]
        got = _fwhm(img[:, matrix // 2])
        closed = 2 * etl * rp["echo_spacing_ms"] / (math.pi * t2)
        # the same width from the modulation itself: exp(-TE(line) / T2) on the lines, times the transform of the strip
        lines = np.asarray(console.sampling(rp).lines[0])
        k = np.arange(matrix) - matrix // 2
        mod = np.exp(-(lines + 1) * rp["echo_spacing_ms"] / t2) * np.sinc(k * thin / 256.0)
        ref = _fwhm(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(mod))))
        cf_err, ref_err = max(cf_err, abs(got / closed - 1)), max(ref_err, abs(got / ref - 1))
        text.append(f"{got:.3f} / {closed:.3f} / {ref:.3f} px")
    rows.append(_row("J16", "core", "T2 blurring of `fse`: full width at half maximum, along the phase direction, of the image of a strip a quarter of a "
                     "pixel thin (centric order, exact pulses; 32 lines in one shot of 32 echoes, 64 lines in 4 shots of 16 and in 2 shots of 32; echo "
                     "train 4 to 4.8 T2 long); image / closed form / the modulation's own transform: " + "; ".join(text),
                     "the Lorentzian of an exponential modulation exp(-2 abs(k) ETL ESP / (lines T2)): FWHM = 2 ETL ESP / (pi T2) pixels (the continuous "
                     "form; the lines are filled in steps and the two sides of k = 0 one echo apart)", cf_err, 0.05, f"{100 * cf_err:.1f} %"))
    rows.append(_row("J17", "core", "the same widths", "the inverse DFT of exp(-TE(line) / T2) on the lines of the sampling plan x the transform of the strip, "
                     "computed here (no engine code)", ref_err, 0.01, f"{100 * ref_err:.2f} %"))

    # ---- the effective echo time: the contrast of a large region follows the echo that fills the centre
    t2 = 100.0
    w = _world([("t", _disc(70.0))], n=320, tissues=dict(t=dict(TISSUES["a"], t2_ms=t2)))
    eng = kspace.Engine(w)
    means = {}
    for te in (20.0, 60.0, 120.0):
        pl = _plan(w, sequence="fse", tr_ms=3000.0, etl=16, echo_spacing_ms=10.0, te_ms=te, echo_order="linear", matrix=64, **HARD)
        means[pl["realised"]["te_ms"]] = abs(_region(eng.acquire(pl["realised"]), pl["realised"], 0.0, 0.0, 30.0)[0])
    te = sorted(means)
    worst = max(abs(means[b] / means[a] / math.exp(-(b - a) / t2) - 1) for a, b in zip(te, te[1:]))
    rows.append(_row("J18", "core", f"effective echo time of `fse` (16 echoes of 10 ms, linear order, te_ms {', '.join(f'{t:g}' for t in te)} ms realised on "
                     "echoes 2, 6 and 12): ratio of the mean signal of the middle of a disc of T2 100 ms between two effective echo times",
                     "exp(-(TE2 - TE1) / T2): the centre of k-space sets the signal of a large uniform region (the other lines, filled at other "
                     "echo times, change its edges)", worst, 0.03, f"{100 * worst:.2f} %"))
    return rows


# ================================================================================================== multi-echo spin echo
def design_train(rp, cls, tbw, window, scale, crusher=True, repetitions=24):
    """Echoes of the train of a realised ``mese`` protocol for one material and one transmit scale, by the phase graph of the
    design layer (``mrgym.design.sequence``): the console's two pulses as designed pulses (windowed sinc of 128
    samples), their rotations at positions through the slice, a crusher of one cycle either side of every
    refocusing pulse, ``repetitions`` repetitions stepped from equilibrium. ``crusher``: between two repetitions only
    Mz survives (the ideal crusher that ends a repetition of the served sequence); False: the gratings that the train
    stored along z survive too, as they do behind a real spoiler. Separate code from ``mrgym.core``."""
    cal = rf.fwhm_factor(tbw, window)
    thk = rp["slice_mm"]
    zeta, wz = rf.quadrature(tbw, window)
    half = np.linspace(0.0, float(zeta.max()), 4 * (zeta.size - 1) + 1)                 # 4 x the engine's columns: an echo oscillates through the slice
    z = np.concatenate([-half[:0:-1], half]) * thk
    weights = np.full(z.size, (half[1] - half[0]))
    weights[[0, -1]] *= 0.5
    exc = design_pulses.sinc(rp["flip_deg"], 2.0, tbw, thickness_mm=thk / cal, window=window, kind="excitation", n=128)
    ref = design_pulses.sinc(rp["refocus_deg"], 3.0, tbw, thickness_mm=1.5 * thk / cal, window=window, kind="refocusing", n=128)
    n, esp = rp["n_echoes"], rp["echo_spacing_ms"]
    one = [Q.rf_pulse(exc, pos_mm=z, weights=weights, b1_scale=scale)]
    for k in range(n):
        one += [Q.gradient(1, ms=esp / 2), Q.rf_pulse(ref, pos_mm=z, phase_deg=90.0, b1_scale=scale), Q.gradient(1, ms=esp / 2), Q.readout(k + 1)]
    one += [Q.spoil(), Q.delay(rp["tr_ms"] - n * esp)]
    if not crusher:                                                     # orders above 2 n cannot return within a train
        return Q.run(one * repetitions, cls["t1_ms"], cls["t2_ms"], kmax=2 * n + 2).signal[-n:]
    state, res = None, None
    for _ in range(repetitions):
        res = Q.run(one, cls["t1_ms"], cls["t2_ms"], state=state)
        state = (np.zeros_like(res.fp[:1]), np.zeros_like(res.fm[:1]), res.z[:1])       # Mz of every column, nothing else
    return res.signal


def _t2_fit(te, s):
    """T2 of a straight-line fit to the logarithm of the echoes."""
    return -1.0 / np.polyfit(te, np.log(np.abs(s)), 1)[0]


def check_mese():
    rows = []
    worst = 0.0
    for name in ("a", "c"):                                             # one disc per image: no ringing of a neighbour in the region
        w = _world([(name, _disc(30.0, 5.0, -10.0))], n=192)
        rp = _plan(w, sequence="mese", tr_ms=1500.0, n_echoes=12, echo_spacing_ms=12.0, matrix=64, **HARD)["realised"]
        img = kspace.Engine(w).acquire(rp)
        want = 1.0 / (1.0 / TISSUES[name]["t2_ms"] + echo_pathway_b(rp) * TISSUES[name]["adc_um2_ms"] * 1e-3 / rp["echo_spacing_ms"])
        worst = max(worst, abs(_t2_fit(np.array(rp["echo_times_ms"]), _region(img, rp, 5.0, -10.0, 15.0)) / want - 1))
    rows.append(_row("J19", "core", "`mese` with exact pulses (hard, transmit scale 1), 12 echoes of 12 ms: T2 of a straight-line fit to the logarithm of the "
                     "region means of the 12 images, a disc of T2 80 ms and one of 50 ms", "1 / (1 / T2 + b D / ESP): without stimulated echoes the images "
                     "decay as exp(-TE / T2) x exp(-b D) per echo, b the weighting of one pair of crushers (0.15 s/mm^2: the fitted T2 is 0.1 % below "
                     "the material's for D = 0.8 um^2/ms)", worst, 1e-9, f"{worst:.1e} relative"))

    # ---- the stimulated echoes through the slice and across the repetition, against the phase graph of the design layer
    cls = dict(t1_ms=900.0, t2_ms=80.0, adc_um2_ms=0.0)
    wd = _world([("t", _disc(40.0))], n=96, tissues=dict(t=dict(TISSUES["a"], **cls)))
    engd = kspace.Engine(wd)
    classes = served._classes(engd)
    worst, bias, bias_err, cost, left_out, grid, box = 0.0, [], 0.0, 0.0, [], 0.0, 0.0
    moving = dict(cls, adc_um2_ms=0.8)                                  # the same material with the diffusion of tissue
    engm = kspace.Engine(_world([("t", _disc(40.0))], n=96, tissues=dict(t=dict(TISSUES["a"], **moving))))
    for beta, pulse, scale in ((180.0, "normal", 1.0), (150.0, "normal", 1.0), (120.0, "fast", 1.0), (180.0, "normal", 0.85), (160.0, "sharp", 1.1)):
        rp = _plan(wd, sequence="mese", tr_ms=1500.0, n_echoes=18, echo_spacing_ms=15.0 if pulse == "sharp" else 12.0, refocus_deg=beta, rf_pulse=pulse,
                   tx_scale=scale, slice_mm=3.0, matrix=64)["realised"]
        tbw, window = engd.pulse(rp)
        gn = np.array([scale * engd.b1_range[0]])
        got = served._table_train(engd, rp, classes, gn, carry=((38,), False), split=False)[0, 0]
        want = design_train(rp, cls, tbw, window, scale, crusher=False, repetitions=12)
        worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
        te = np.array(rp["echo_times_ms"])
        b_got, b_want = _t2_fit(te, got) / cls["t2_ms"] - 1, _t2_fit(te, want) / cls["t2_ms"] - 1
        bias.append(100 * b_got)
        bias_err = max(bias_err, 100 * abs(b_got - b_want))
        cost = max(cost, float(np.abs(_unit_amplitudes(engd, rp, "t") - want).max() / np.abs(want).max()))
        left_out.append(float(np.abs(design_train(rp, cls, tbw, window, scale) - want).max() / np.abs(want).max()))
        # the two cuts of the serving path apart, with diffusion: the box against 38 orders, the engine's slice columns against 4 x as many
        ref = served._table_train(engm, rp, classes, gn, carry=((38,), False), split=False)[0, 0]
        big = float(np.abs(ref).max())
        box = max(box, float(np.abs(served._table_train(engm, rp, classes, gn, split=False)[0, 0] - ref).max() / big))
        grid = max(grid, float(np.abs(served._table_train(engm, rp, classes, gn, carry=((38,), False))[0, 0] - ref).max() / big))
    rows.append(_row("J20", "core", "`mese` through the slice, 18 echoes of 12 ms on a 3 mm slice at TR 1500 ms: every echo of a material of T1 900 / T2 80 ms "
                     "without diffusion; refocusing 180, 150, 120 (fast pulse) degrees, 180 at transmit scale 0.85, 160 (sharp pulse, 15 ms) at 1.1; the "
                     "gratings of 38 orders carried into the next repetition, every slice column on the fine grid", "the time-stepping phase graph of the "
                     "design layer (`mrgym.design.sequence`, separate code) with the console's two pulses as designed pulses, their rotation at 4 x the "
                     "engine's slice positions, real crushers, 12 repetitions from equilibrium with the gratings along z kept between them "
                     "(`served_checks.design_train`; exp(-(TR - train) / T2) = 1e-7, so nothing transverse is left)",
                     worst, 1e-6, f"{worst:.1e} relative"))
    rows.append(_row("J21", "core", "what the stimulated echoes do to a straight-line T2 fit over all 18 echoes in those five cases: T2 is overestimated by "
                     + ", ".join(f"{b:+.1f}" for b in bias) + " %; measured: the smallest of them", "a refocusing pulse that is not 180 degrees "
                     "everywhere in the slice stores magnetisation along z, where it decays with T1 and returns in later echoes: the fit is biased "
                     "upwards even at a nominal 180 degrees (the size is compared with a measurement in docs/physics/measured.md, rung M3b)",
                     min(bias), 5.0, f"{min(bias):+.1f} %", kind="min"))
    rows.append(_row("J22", "core", "the same bias, against the design layer's phase graph", "equal", bias_err, 1e-3, f"{bias_err:.1e} points"))
    rows.append(_row("J23", "core", f"the same five scans as they are SERVED, for the material WITHOUT diffusion: the box of carried states is then cut by "
                     f"its limit of {served.CARRY_CAP[0]} orders, not by a tolerance: largest difference of an echo from the design layer's train, "
                     "relative to the largest echo", "the cost of the limit of the box where it is largest (no diffusion, TR 1500 ms, T1 900 ms; with "
                     "20 orders it is 4e-4 to 2e-4 still). For scale: an ideal crusher at the end of the repetition (engine before this row) left out "
                     + ", ".join(f"{v:.1e}" for v in left_out) + " in these cases", cost, 1e-3, f"{cost:.1e} relative"))
    rows.append(_row("J23b", "core", f"the same scans for the material with the diffusion of tissue (0.8 um^2/ms; the crushers dephase by one cycle per "
                     f"{1 / console.crusher_lobe(console.SYSTEM)['cycles_per_mm']:.3f} mm, so a grating of order 9 is gone after a second): the box that is "
                     f"served (a factor of {served.CARRY_TOL:g}) against 38 orders", "the cost of the box where diffusion sets it", box, 1e-6,
                     f"{box:.1e} relative"))
    rows.append(_row("J23c", "core", "the same: what survives the repetition integrated through the slice on the engine's own columns (the echoes behind an "
                     f"ideal crusher on {served.TRAIN_COLUMNS} x as many) against everything on the fine grid", "the cost of the coarser grid for the part that "
                     "is a few per cent of an echo", grid, 1e-5, f"{grid:.1e} relative"))
    # ---- per tissue and TR: what reaches the next excitation, as a share of the largest echo
    tab3, names = tissue.table(3.0), ("wm", "gm", "csf", "scalp_fat", "muscle", "blood", "liver")
    wt = _world([(nm, _disc(8.0, 0.0, 22.0 * (i - 3))) for i, nm in enumerate(names)], n=192, tissues={nm: tab3[nm] for nm in names})
    engt = kspace.Engine(wt)
    sizes, text = [], []
    for tr in (500.0, 1500.0, 4000.0):
        rp = _plan(wt, sequence="mese", tr_ms=tr, n_echoes=8, echo_spacing_ms=10.0, matrix=64)["realised"]
        full, ideal = served.table(engt, rp)["a"], served.table(engt, rp, carry=((0,), False))["a"]
        share = np.abs(full - ideal).max((1, 2)) / np.abs(full).max((1, 2))
        order = [wt["names"][c] for c in served._classes(engt)]
        sizes.append(share)
        text.append(f"TR {tr:.0f} ms: " + ", ".join(f"{nm} {100 * share[order.index(nm)]:.2f}" for nm in names))
    rows.append(_row("J24", "core", "what reaches the next excitation, per tissue of the table at 3 T (`mese`, 8 echoes of 10 ms, 180 degrees, normal pulse, "
                     "5 mm): largest change of an echo against the same train behind an ideal crusher, in per cent of the largest echo. " + "; ".join(text)
                     + "; measured: cerebrospinal fluid at TR 4000 ms", "the stimulated echoes of the last repetition and its transverse magnetisation: "
                     "large for a fluid at any TR a protocol uses, a per cent for tissue at TR 1500 ms, gone for fat. Engine before this row served the "
                     "ideal crusher", float(sizes[2][order.index("csf")]), 0.02, f"{100 * float(sizes[2][order.index('csf')]):.2f} %", kind="min"))
    cut, text = {}, []
    for tr, n in ((500.0, 16), (1500.0, 16), (4000.0, 16)):             # what the cut of the carried box costs for the same tissues
        rp = _plan(wt, sequence="mese", tr_ms=tr, n_echoes=n, echo_spacing_ms=10.0, matrix=64)["realised"]
        ref, got = served.table(engt, rp, nodes=3, carry=((2 * n + 8,), False))["a"], served.table(engt, rp, nodes=3)["a"]
        err = np.abs(got - ref).max((1, 2)) / np.abs(ref).max((1, 2))
        cut[tr] = float(err.max())
        text.append(f"TR {tr:.0f} ms: {cut[tr]:.0e} ({order[int(np.argmax(err))]})")
    rows.append(_row("J24b", "core", f"the same tissues in a train of 16 echoes: what the cut of the carried gratings costs (kept: a factor of {served.CARRY_TOL:g} "
                     f"by relaxation and diffusion, at most {served.CARRY_CAP[0]} orders), largest change of an echo against 40 orders, relative to the "
                     "largest echo. " + "; ".join(text) + "; measured: the largest", "a truncation, measured: largest where a train is repeated faster "
                     "than T1 and nothing diffuses (fat), small where diffusion erases the fine gratings (water-like tissue at a TR of seconds)",
                     max(cut.values()), 1e-3, f"{max(cut.values()):.1e} relative"))
    # ---- the cut against the time left after the train, hard pulses at 80 and 100 % B1, refocusing 180, 120, 60 degrees
    from ..core import periodic
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=1e3, b0_t=3.0, sequences="all")
    worst, text = {}, []
    for rest in (4.0, 100.0, 400.0, 1000.0):
        worst[rest] = 0.0
        for n in (8, 24):
            for beta in (180.0, 120.0, 60.0):
                rp = console.plan(dict(sequence="fse", tr_ms=n * 8.4 + rest, etl=n, refocus_deg=beta, matrix=64, **HARD), exam)["realised"]
                seq, _ = console.REGISTRY["fse"].blocks(rp)
                for nm in names:
                    tis = served._tissue(tab3[nm])
                    sp = O.grid([tis], (0.0,), (0.8, 1.0))
                    ref = periodic.steady_state(seq, sp, method="carry", carry=(90,), carry_transverse=False).signal
                    got = periodic.steady_state(seq, sp, method="carry", carry=served.carry_of(seq, tis)[0], carry_transverse=False).signal
                    worst[rest] = max(worst[rest], float(np.abs(got - ref).max() / np.abs(ref).max()))
        text.append(f"{rest:g} ms: {worst[rest]:.0e}")
    rows.append(_row("J24c", "core", "the cut against the time between the end of the train and the next excitation: `fse` with hard pulses, 8 and 24 "
                     "echoes, refocusing 180, 120 and 60 degrees, B1 80 and 100 %, the seven tissues; largest change of an echo against 90 orders. "
                     + "; ".join(text) + "; measured: the largest (a train repeated without a pause)", "the same truncation over the protocols the "
                     "console accepts: it costs most when nothing has relaxed before the next excitation and the refocusing angle is low", max(worst.values()),
                     2e-3, f"{max(worst.values()):.1e} relative"))
    # ---- what is NOT carried: the transverse magnetisation at the end of the repetition
    add, swing = {}, {}
    for nm in ("csf", "blood", "gm", "wm"):
        for tr in (500.0, 1500.0, 4000.0):
            rp = console.plan(dict(sequence="mese", tr_ms=tr, n_echoes=8, echo_spacing_ms=10.0, matrix=64, **HARD), exam)["realised"]
            seq, _ = console.REGISTRY["mese"].blocks(rp)
            body = seq.blocks[0].body
            assert body[-1].kind == "spoil"
            kept = B.sequence([B.repeat(body[:-1], 1)], cycle_mm=seq.cycle_mm)      # the same repetition without the ideal spoiler at its end
            tis = served._tissue(tab3[nm])
            base = periodic.steady_state(seq, O.grid([tis], (0.0,), (0.9,)), method="carry", carry=(30,), carry_transverse=False).signal[:, 0]
            period_hz = 1e3 / (tr - 8.5 * 10.0)                         # the echo is periodic in the off-resonance with 1 / (time of free precession)
            sig = np.array([periodic.steady_state(kept, O.grid([tis], (0.0,), (0.9,), df_hz=f), method="carry", carry=(30,)).signal[:, 0]
                            for f in np.arange(8) / 8 * period_hz])
            big = float(np.abs(base).max())
            add[nm, tr] = float(np.abs(sig[0] - base).max() / big)
            swing[nm, tr] = float(np.abs(sig - sig.mean(0)).max() / big)
    text = "; ".join(f"{nm}: " + ", ".join(f"{100 * add[nm, tr]:.1f} ({100 * swing[nm, tr]:.1f})" for tr in (500.0, 1500.0, 4000.0)) for nm in ("csf", "blood", "gm", "wm"))
    rows.append(_row("J24d", "core", "what is NOT served: the transverse magnetisation that is left when the repetition ends. For a voxel whose spins "
                     "stay in phase with each other until the next excitation it returns in the next train; `mese`, 8 echoes of 10 ms, hard pulses at 90 % "
                     "B1, TR 500, 1500, 4000 ms: largest change of an echo for spins exactly on resonance, and in brackets how far the echo swings "
                     "when the voxel is off resonance by a fraction of 1 / TR, in per cent of the largest echo. " + text + "; measured: cerebrospinal "
                     "fluid on resonance at TR 1500 ms", "a declared simplification (limits.md, block 8a): the contribution depends on the off-resonance "
                     "of the voxel to a fraction of a hertz and on the spread inside it (T2'), which the amplitude of these sequences does not hold; it "
                     "is taken as dephased. The core computes it (`periodic`, method carry with the transverse states, core rows N14c)",
                     add["csf", 1500.0], 0.02, f"{100 * add['csf', 1500.0]:.1f} %", kind="min"))
    return rows


# ================================================================================================== multi-echo gradient echo
def check_megre():
    from ..core import validate as CV
    rows = []
    w, eng = CV._engine_world()
    w["console"].update(sequences="all")
    worst = 0.0
    for more in (HARD, {}, dict(rf_pulse="fast"), dict(rf_pulse="sharp", flip_deg=60.0)):
        prot = dict(dict(tr_ms=40.0, flip_deg=25.0, matrix=64), **more)
        a = console.plan(dict(prot, sequence="megre", n_echoes=3), w["console"])["realised"]
        b = console.plan(dict(prot, sequence="spgr", n_echoes=3), w["console"])["realised"]
        for c in range(len(w["names"])):
            x, y = eng.amplitudes(a, c)[0], eng.amplitudes(b, c)[0]
            if y.size:
                worst = max(worst, float(np.abs(x - y).max() / np.abs(y).max()))
    rows.append(_row("J25", "core", "`megre`: the amplitude map of every material of a phantom world with a transmit pattern (hard, fast, normal and sharp "
                     "pulse; 25 and 60 degrees, TR 40 ms): the steady state of the registry's block list by the periodic solve of the core, ideal spoiling in "
                     "every slice column and the departure from it on 16 nodes of the flip angle", "`Engine.amplitudes` of `spgr` with the same TR, flip angle, "
                     "spoiler and phase increment (the engine's own phase graph, `scanner/epg.py`, through its own nodes)", worst, 1e-6, f"{worst:.1e} relative"))

    # ---- images: the two sequences where they overlap
    ww = _world([("a", _disc(55.0)), ("b", _disc(18.0, 5.0, 22.0)), ("subcut_fat", _disc(14.0, -25.0, -18.0))], n=256,
                b1=lambda y, x: 0.85 + 0.25 * np.exp(-(y * y + x * x) / (2 * 60.0 ** 2)), b0=lambda y, x: 20.0 * x / 100 + 8.0)
    engw = kspace.Engine(ww)
    prot = dict(tr_ms=60.0, flip_deg=20.0, matrix=64, n_echoes=2)
    a = engw.acquire(_plan(ww, sequence="megre", **prot)["realised"])
    b = engw.acquire(_plan(ww, sequence="spgr", **prot)["realised"])
    worst = float(np.abs(a - b).max() / np.abs(b).max())
    digits = -math.log10(max(worst, 1e-16))
    rows.append(_row("J26", "core", "`megre` with 2 monopolar echoes on a phantom with fat, a transmit pattern and an off-resonance field: both complex "
                     f"images (they agree to {digits:.1f} digits)", "`spgr` with n_echoes = 2 at the same echo times (bit-level agreement is not "
                     "expected: the amplitudes come from two solvers)", worst, 1e-6, f"{worst:.1e} relative"))

    # ---- T2* from the echoes of one disc, and the field from their phase
    df = 12.0
    wt = _world([("a", _disc(50.0))], n=256, cf_offset_hz=df)
    engt = kspace.Engine(wt)
    rp = _plan(wt, sequence="megre", tr_ms=120.0, flip_deg=25.0, matrix=64, n_echoes=12, echo_spacing_ms=6.0)["realised"]
    s = _region(engt.acquire(rp), rp, 0.0, 0.0, 25.0)
    te = np.array(rp["echo_times_ms"])
    t2s = _t2_fit(te, s)
    slope = np.polyfit(te * 1e-3, np.unwrap(np.angle(s)), 1)[0] / (2 * np.pi)
    e1, e2 = abs(t2s / wt["tissues"]["a"]["t2s_ms"] - 1), abs(slope - df)
    rows.append(_row("J27", "core", f"`megre`, 12 echoes of 6 ms: T2* of a straight-line fit to the logarithm of the region means of a disc ({t2s:.2f} ms)",
                     "T2* of the material (60 ms)", e1, 1e-3, f"{e1:.1e} relative"))
    rows.append(_row("J28", "core", f"the same echoes: frequency from the slope of the unwrapped phase of the region mean ({slope:.4f} Hz)",
                     f"the off-resonance of the world ({df:g} Hz): phase = 2 pi df TE", e2, 1e-3, f"{e2:.1e} Hz"))

    # ---- bipolar: every second echo is displaced the other way
    wf = _world([("subcut_fat", _disc(12.0))], n=512, b0_t=1.5)
    engf = kspace.Engine(wf, fat_model="single")
    rp = _plan(wf, sequence="megre", tr_ms=60.0, flip_deg=20.0, matrix=128, fov_mm=256.0, bandwidth_hz_per_px=100.0, n_echoes=4, readout="bipolar")["realised"]
    img = engf.acquire(rp)
    shift = tissue.FAT_PPM * 1e-6 * console.GAMMA_HZ_PER_T * 1.5 / rp["bandwidth_hz_per_px"]          # pixels, signed
    from .. import validate
    got = np.array([validate._centroid_px(im, 1) for im in img])
    want = -np.array(rp["echo_polarities"]) * shift
    worst = float(np.abs(got - want).max())
    rows.append(_row("J29", "core", f"`megre`, bipolar readout, a disc of single-line fat at 1.5 T and 100 Hz per pixel: centroid of the disc along the readout "
                     f"in the 4 echoes ({', '.join(f'{v:+.2f}' for v in got)} px)", f"-polarity of the echo x chemical shift / bandwidth per pixel "
                     f"({', '.join(f'{v:+.2f}' for v in want)} px): the displacement changes sign with the direction of the readout", worst, 0.05,
                     f"{worst:.3f} px"))
    mono = _plan(wf, sequence="megre", tr_ms=60.0, flip_deg=20.0, matrix=128, fov_mm=256.0, bandwidth_hz_per_px=100.0, n_echoes=4)
    bip = _plan(wf, sequence="megre", tr_ms=60.0, flip_deg=20.0, matrix=128, fov_mm=256.0, bandwidth_hz_per_px=100.0, n_echoes=4, readout="bipolar")
    gain = mono["realised"]["echo_spacing_ms"] - bip["realised"]["echo_spacing_ms"]
    flyback = console._lobe_ms(128 / (console.GAMMA_HZ_PER_T * 0.256), console.SYSTEM)
    rows.append(_row("J30", "core", f"`megre`: the shortest echo spacing, monopolar minus bipolar ({gain:.2f} ms)", "the flyback lobe of the full readout area at "
                     "the derated limits (the time-optimal trapezoid), on the raster", abs(gain - flyback), 0.011, f"{abs(gain - flyback):.3f} ms"))
    return rows


# ================================================================================================== diffusion-weighted spin echo
def check_dwse():
    rows = []
    w = _world([("a", _disc(28.0, 0.0, -45.0))], n=64)                 # its console card is what the plans of this check need
    b_err, sum_err, cf_err, fits, n_plans = 0.0, 0.0, 0.0, 0, 0
    for b in (1.0, 50.0, 400.0, 1000.0, 3000.0):
        for more in ({}, dict(matrix=128, bandwidth_hz_per_px=150.0, rf_pulse="sharp"), dict(te_ms=140.0, slice_mm=3.0)):
            pl = _plan(w, sequence="dwse", tr_ms=3000.0, b_value=b, matrix=more.get("matrix", 64), **{k: v for k, v in more.items() if k != "matrix"})
            rp = pl["realised"]
            t, g, flips = console.diffusion_waveform(rp)
            wave = DG.b_value(t, g, flips)
            b_err = max(b_err, abs(rp["b_value"] / b - 1), abs(wave / rp["b_value"] - 1))
            cf_err = max(cf_err, abs(console.diffusion_b(rp["diffusion_g_mt_m"], rp["diffusion_delta_ms"], rp["diffusion_separation_ms"],
                                                         rp["diffusion_ramp_ms"]) / wave - 1))
            seq, _ = console.REGISTRY["dwse"].blocks(rp)
            prog = B.compile_ops(seq, frame=False)
            total, k, pulses = 0.0, 0.0, 0                               # the echo pathway: order 0 after the excitation, negated by the refocusing pulse
            for op in prog.ops:
                if op.kind == "rf":
                    pulses += 1
                    k = -k if pulses == 2 else k
                elif op.kind == "adc":
                    break
                elif op.kind == "free" and pulses:
                    total += B.b_increment(k, op.n[0], op.ms, prog.cycle_mm[0], op.shape)
                    k += op.n[0]
            sum_err = max(sum_err, abs(total / wave - 1))
            # the lobes fit where the console puts them, inside the limits
            s = console.SYSTEM
            ok = (rp["diffusion_g_mt_m"] <= s["g_max_mt_m"] * s["derate"] * (1 + 1e-9)
                  and rp["diffusion_g_mt_m"] / rp["diffusion_ramp_ms"] <= s["slew_t_m_s"] * s["derate"] * (1 + 1e-9)
                  and rp["diffusion_start_ms"] >= rp["rf_ms"] / 2 - 1e-9
                  and rp["diffusion_start_ms"] + rp["diffusion_delta_ms"] + rp["diffusion_ramp_ms"] <= rp["te_ms"] / 2 - pl["timing"]["rf_180_ms"] / 2 + 1e-9
                  and rp["diffusion_start_ms"] + rp["diffusion_separation_ms"] >= rp["te_ms"] / 2 + pl["timing"]["rf_180_ms"] / 2 - 1e-9
                  and rp["diffusion_start_ms"] + rp["diffusion_separation_ms"] + rp["diffusion_delta_ms"] + rp["diffusion_ramp_ms"]
                  <= rp["te_ms"] - pl["timing"]["readout_ms"] / 2 + 1e-9)
            fits += not ok
            n_plans += 1
    rows.append(_row("J31", "core", f"`dwse`: the b-value the console states (`realised.b_value`) for requests of 1 to 3000 s/mm^2 in {n_plans} protocols "
                     "(three geometries; minimum and longer TE): against the request, and against the waveform it plays",
                     "`mrgym.design.gradients.b_value`: the exact integral of the piecewise-linear waveform of the two lobes with the sign change at "
                     "the refocusing pulse (Stejskal & Tanner 1965); the lobes are sized so that it equals the request", b_err, 1e-6, f"{b_err:.1e} relative"))
    rows.append(_row("J32", "core", "the same waveforms: the trapezoid closed form the console sizes the lobes with (`console.diffusion_b`)",
                     "gamma^2 G^2 (delta^2 (Delta - delta / 3) + eps^3 / 30 - delta eps^2 / 6) against the integral of the waveform", cf_err, 1e-9,
                     f"{cf_err:.1e} relative"))
    rows.append(_row("J33", "core", "the same protocols as block lists of the registry: the b-value the core attenuates with, summed over its gradient blocks "
                     "(`blocks.b_increment`, the shape moments of each trapezoid)", "the integral of the waveform", sum_err, 1e-9, f"{sum_err:.1e} relative"))
    rows.append(_row("J34", "core", "the same protocols: lobes that do not fit between the excitation and the refocusing pulse or between it and the readout, or "
                     "exceed the derated amplitude or slew limit", "none", fits, 0, f"{fits} of {n_plans}"))

    # ---- the signal: ADC from two images, and the price in echo time. One disc per world: next to another object the
    # region mean also holds that object's truncation ringing, which does not scale with exp(-b D) of this one (with
    # the three discs in one image the same reading is off by up to 3e-4).
    worst, short, text = 0.0, 0.0, []
    for tr in (20000.0, 3000.0):
        b_hi = _plan(w, sequence="dwse", tr_ms=tr, b_value=1000.0, matrix=64)["realised"]
        b_lo = _plan(w, sequence="dwse", tr_ms=tr, b_value=0.0, te_ms=b_hi["te_ms"], matrix=64)["realised"]
        weighting = echo_pathway_b(b_hi) - echo_pathway_b(b_lo)        # the crushers of the b = 0 scan weight it too
        for name in ("a", "b", "c"):
            one = kspace.Engine(_world([(name, _disc(28.0, 10.0, -15.0))], n=192))
            hi, lo = one.acquire(b_hi), one.acquire(b_lo)
            adc = math.log(abs(_region(lo, b_lo, 10.0, -15.0, 12.0)[0]) / abs(_region(hi, b_hi, 10.0, -15.0, 12.0)[0])) / weighting * 1e3      # um^2/ms
            if tr > 10000:
                worst = max(worst, abs(adc / TISSUES[name]["adc_um2_ms"] - 1))
                text.append(f"{adc:.6f}")
            else:
                short = max(short, abs(adc / TISSUES[name]["adc_um2_ms"] - 1))
    rows.append(_row("J35", "core", f"`dwse`: diffusion coefficient from the region means of two images, b = 0 and b = 1000 s/mm^2 at the same TE "
                     f"({b_hi['te_ms']:g} ms) and TR 20 s, a disc of each of three materials ({', '.join(text)} um^2/ms), with the difference of the two "
                     f"weightings in the denominator (the crushers of the b = 0 scan weight it by {echo_pathway_b(b_lo):.3f} s/mm^2)", "the coefficients of "
                     "the materials (0.8, 3.0, 1.6 um^2/ms): signal x exp(-b D), every other factor equal in the two scans once nothing of the last "
                     "repetition is left (T1 up to 2500 ms: exp(-TR / T1) = 3e-4, and the lobes of b = 1000 erase what the crushers of b = 0 keep)",
                     worst, 2e-5, f"{worst:.1e} relative"))
    rows.append(_row("J35b", "core", "the same reading at TR 3 s: largest error of the coefficient", "not an equality: the b = 0 scan carries the stimulated "
                     "echo of its last repetition and the b = 1000 scan does not (its lobes dephase the grating beyond recall), so the ratio of the two "
                     "images is not exp(-b D) alone. A two-point coefficient wants b > 0 in both scans or a TR of several T1; the size is printed",
                     short, 1e-4, f"{short:.1e} relative", kind="min"))
    te = [_plan(w, sequence="dwse", tr_ms=3000.0, b_value=b, matrix=64)["realised"]["te_ms"] for b in (0.0, 100.0, 500.0, 1000.0, 2000.0, 3000.0)]
    rising = min(b - a for a, b in zip(te, te[1:]))
    rows.append(_row("J36", "core", "`dwse`: the minimum echo time for b = 0, 100, 500, 1000, 2000, 3000 s/mm^2 at the default geometry: "
                     + ", ".join(f"{t:g}" for t in te) + " ms; measured: the smallest step", "the lobes lengthen the echo time: it rises with b",
                     rising, 0.5, f"{rising:.2f} ms", kind="min"))
    return rows


# ================================================================================================== prepared sequences
def check_prepared():
    rows = []
    mats = dict(a=dict(TISSUES["a"], adc_um2_ms=0.0), b=dict(TISSUES["b"], t2_ms=60.0, t2s_ms=40.0, t2p_ms=120.0, adc_um2_ms=0.0), c=dict(TISSUES["c"], adc_um2_ms=0.0))
    shapes = [("a", _disc(60.0)), ("b", _disc(20.0, 0.0, 25.0)), ("c", _disc(15.0, 30.0, -20.0))]
    # ---- saturation recovery: closed forms
    worst, count = 0.0, 0
    for eff in (1.0, 0.8):
        w = _world(shapes, n=128, tissues=mats, sat_eff=eff)
        eng = kspace.Engine(w)
        for ts, tr, flip in ((30.0, 80.0, 90.0), (300.0, 340.0, 90.0), (300.0, 2000.0, 20.0), (1500.0, 1600.0, 60.0), (5000.0, 9000.0, 35.0)):
            rp = _plan(w, sequence="satrec", tr_ms=tr, ts_ms=ts, flip_deg=flip, matrix=64, **HARD)["realised"]
            for name in ("a", "b", "c"):
                t1 = mats[name]["t1_ms"]
                es, er, ca = math.exp(-ts / t1), math.exp(-(tr - ts) / t1), math.cos(math.radians(flip))
                z1 = (1 - es + (1 - eff) * es * (1 - er)) / (1 - (1 - eff) * es * er * ca)
                worst, count = max(worst, abs(_unit_amplitudes(eng, rp, name)[0] - math.sin(math.radians(flip)) * z1)), count + 1
    rows.append(_row("J41", "core", f"`satrec`, hard pulse: the amplitude `Engine.amplitudes` serves, {count} cases (saturation time 30 to 5000 ms, TR 80 to "
                     "9000 ms, 20 to 90 degrees, 3 materials, a complete saturation and one of 80 %)", "sin a (1 - exp(-TS / T1)) for the complete "
                     "saturation, whatever TR and the flip angle (textbook); otherwise sin a times the fixed point of the two intervals, Mz before "
                     "the pulse = (1 - Es + (1 - e) Es (1 - Er)) / (1 - (1 - e) Es Er cos a), Es = exp(-TS / T1), Er = exp(-(TR - TS) / T1) (derived "
                     "here)", worst, 1e-12, f"{worst:.1e}"))
    # ---- through the slice, against the isochromat ensemble
    wq = _world(shapes[:2], n=96, tissues=mats, sat_eff=0.85, b1=lambda y, x: 0.85 + 0.25 * np.exp(-(y * y + x * x) / (2 * 55.0 ** 2)))
    engq = kspace.Engine(wq)
    worst = 0.0
    for pulse, flip in (("normal", 90.0), ("fast", 40.0)):
        rp = _plan(wq, sequence="satrec", tr_ms=420.0, ts_ms=250.0, flip_deg=flip, rf_pulse=pulse, matrix=64)["realised"]
        tab = served.table(engq, rp, nodes=3)
        tbw, window = engq.pulse(rp)
        seq, labels = console.REGISTRY["satrec"].blocks(rp, selective=B.Selective(tbw, window), sat_eff=0.85)
        zeta, wz = O.slice_columns(tbw, two_sided=False, window=window)
        sp = O.grid([served._tissue(mats[k]) for k in ("a", "b")], zeta, tab["gn"])
        ens = core_bloch.run(B.compile_ops(seq, frame=False, repetitions=14), sp).signal[-1]
        ens = (ens.reshape(2, zeta.size, tab["gn"].size).real * wz[None, :, None]).sum(1)
        worst = max(worst, float(np.abs(ens - tab["a"][:, :, 0]).max() / np.abs(ens).max()))
    rows.append(_row("J42", "core", "`satrec` through the slice (normal pulse at 90 degrees, fast pulse at 40; transmit scale 0.85 to 1.1; saturation of "
                     "85 %): the table that is served", "isochromat ensemble of the core on the registry's block list with the rotation of the pulse at "
                     "every slice column, stepped through 14 repetitions", worst, 1e-9, f"{worst:.1e} relative"))
    # ---- T1 from the images of a series, one disc per world
    ts = np.array([60.0, 200.0, 500.0, 1200.0, 3000.0])
    worst, text = 0.0, []
    for name in ("a", "c"):
        w1 = _world([(name, _disc(30.0, 5.0, -10.0))], n=192, tissues=mats)
        eng1 = kspace.Engine(w1)
        for flip in (90.0, 25.0):
            m = []
            for t in list(ts) + [15000.0]:
                rp = _plan(w1, sequence="satrec", tr_ms=t + 20.0, ts_ms=t, flip_deg=flip, matrix=64)["realised"]
                m.append(abs(_region(eng1.acquire(rp), rp, 5.0, -10.0, 15.0)[0]))
            ratio = np.array(m[:-1]) / m[-1] * (1 - math.exp(-15000.0 / mats[name]["t1_ms"]))
            t1 = -ts / np.log(1 - ratio)
            worst = max(worst, float(np.abs(t1 / mats[name]["t1_ms"] - 1).max()))
            text.append(f"{t1.mean():.4f}")
    rows.append(_row("J43", "core", "`satrec`: T1 from the region means of five images (saturation time 60 to 3000 ms; slice-selective, 90 and 25 degrees) "
                     "relative to a nearly recovered one, a disc of T1 900 ms and one of 380 ms (" + ", ".join(text) + " ms)", "T1 of the material: "
                     "S(TS) / S(15 s) = (1 - exp(-TS / T1)) / (1 - exp(-15 s / T1)), with no knowledge of the flip angle or of the slice profile",
                     worst, 1e-9, f"{worst:.1e} relative"))
    # ---- a preparation in front of the echo train
    w = _world(shapes, n=128, tissues=mats, inv_eff=0.94)
    eng = kspace.Engine(w)
    worst, null = 0.0, 0.0
    for prep, tr, ti, n, esp in (("inversion", 6000.0, 1700.0, 8, 10.0), ("inversion", 2500.0, 300.0, 16, 9.0), ("inversion", 900.0, 150.0, 4, 12.0),
                                 ("saturation", 700.0, 500.0, 4, 10.0), ("saturation", 5000.0, 1200.0, 12, 10.0)):
        rp = _plan(w, sequence="fse", tr_ms=tr, etl=n, echo_spacing_ms=esp, prep=prep, ti_ms=ti, matrix=64, **HARD)["realised"]
        for name in ("a", "b", "c"):
            t1, t2 = mats[name]["t1_ms"], mats[name]["t2_ms"]
            if math.exp(-(tr - ti - n * esp) / t2) > 1e-12:              # the spoiler after the preparation removes what is transverse; before it: negligible here
                continue
            u = 2 - math.exp(-esp / (2 * t1))                           # u = 1 - Mz: 1 after the 90, an inversion makes it 2 - u
            for _ in range(n - 1):
                u = 2 - u * math.exp(-esp / t1)
            u = u * math.exp(-(tr - ti - (n - 0.5) * esp) / t1)
            mz = 1 - (1 + 0.94 * (1 - u)) * math.exp(-ti / t1) if prep == "inversion" else 1 - math.exp(-ti / t1)
            worst = max(worst, float(np.abs(_unit_amplitudes(eng, rp, name) - mz * np.exp(-np.arange(1, n + 1) * esp / t2)).max()))
    rows.append(_row("J44", "core", "`fse` with a preparation (inversion of efficiency 0.94, or saturation) 150 to 1700 ms before the excitation, exact "
                     "pulses, trains of 4 to 16 echoes at TR 0.7 to 6 s, 3 materials: every echo", "Mz at the excitation x exp(-k ESP / T2), Mz = 1 - "
                     "(1 + q Mz') exp(-TI / T1) after an inversion of efficiency q, Mz' being what the train (every 180 inverts what has regrown) and "
                     "the rest of the repetition left; 1 - exp(-TI / T1) after a saturation, whatever the train left (written here)", worst, 1e-12,
                     f"{worst:.1e}"))
    # the null of a fluid: bisection on the served amplitude against the closed form of its null
    t1 = mats["b"]["t1_ms"]
    n, esp = 8, 10.0

    def served_mz(ti, tr):
        return _unit_amplitudes(eng, _plan(w, sequence="fse", tr_ms=tr, etl=n, echo_spacing_ms=esp, prep="inversion", ti_ms=ti, matrix=64, **HARD)["realised"], "b")[0].real

    def closed_mz(ti, tr):
        u = 2 - math.exp(-esp / (2 * t1))
        for _ in range(n - 1):
            u = 2 - u * math.exp(-esp / t1)
        u = u * math.exp(-(tr - ti - (n - 0.5) * esp) / t1)
        return 1 - (1 + 0.94 * (1 - u)) * math.exp(-ti / t1)
    for tr in (20000.0, 9000.0, 6000.0):
        zero = []
        for f in (served_mz, closed_mz):
            lo, hi = 200.0, 2400.0
            for _ in range(40 if f is closed_mz else 18):
                mid = round(0.5 * (lo + hi), 2) if f is served_mz else 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(mid, tr) < 0 else (lo, mid)
            zero.append(0.5 * (lo + hi))
        null = max(null, abs(zero[0] - zero[1]))
    rows.append(_row("J45", "core", "`fse` with an inversion: the inversion time that nulls a fluid of T1 2500 ms at TR 20, 9 and 6 s (the zero crossing of "
                     "the signed amplitude that is served, found by bisection on the console's raster of 0.01 ms)", "the zero of the closed form of row "
                     "J44; for TR >> T1 it is T1 ln(1 + q) (1657 ms for q = 0.94; T1 ln 2 for a complete inversion), and it falls with TR", null, 0.02,
                     f"{null:.3f} ms"))
    # ---- through the slice with its real crushers, against the isochromat ensemble
    quick = dict(a=dict(TISSUES["a"], t1_ms=160.0, t2_ms=45.0, adc_um2_ms=0.0), b=dict(TISSUES["b"], t1_ms=280.0, t2_ms=90.0, adc_um2_ms=0.0))
    wq = _world(shapes[:2], n=96, tissues=quick, inv_eff=0.94, b1=lambda y, x: 0.85 + 0.25 * np.exp(-(y * y + x * x) / (2 * 55.0 ** 2)))
    engq = kspace.Engine(wq)
    worst = 0.0
    for prep, ti, reps in (("inversion", 110.0, 24), ("saturation", 90.0, 8)):
        rp = _plan(wq, sequence="fse", tr_ms=330.0, etl=5, echo_spacing_ms=12.0, refocus_deg=140.0, prep=prep, ti_ms=ti, matrix=64)["realised"]
        tab = served.table(engq, rp, solver="train", nodes=3, columns=1, carry=((45,), False))
        tbw, window = engq.pulse(rp)
        seq, labels = console.REGISTRY["fse"].blocks(rp, selective=B.Selective(tbw, window), inv_eff=0.94)
        zeta, wz = O.slice_columns(tbw, two_sided=False, window=window)
        sp = O.grid([served._tissue(quick[k]) for k in ("a", "b")], zeta, tab["gn"])
        ens = core_bloch.run(B.compile_ops(seq, frame=False, repetitions=reps), sp, n_sub=512).signal[-len(labels):]
        ens = np.moveaxis((ens.reshape(len(labels), 2, zeta.size, tab["gn"].size).real * wz[None, None, :, None]).sum(2), 0, -1)
        worst = max(worst, float(np.abs(ens - tab["a"]).max() / np.abs(ens).max()))
    rows.append(_row("J46", "core", "`fse` with an inversion and with a saturation through the slice (5 echoes, 140 degrees, TR 330 ms, T1 160 and 280 ms): "
                     "the table, with the gratings that survive the repetition and are inverted or saturated with Mz (45 orders carried)", "isochromat "
                     "ensemble of the core on the registry's block list (real crushers and spoiler, the preparation as an event on Mz of every "
                     "isochromat and an ideal spoiler of what is transverse), stepped through 24 and 8 repetitions", worst, 1e-8, f"{worst:.1e} relative"))
    # ---- time and RF power
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0, sequences="all")
    sy = console.SYSTEM
    worst_t = worst_p = 0.0
    b1 = lambda deg, ms: math.radians(deg) / (2 * math.pi * console.GAMMA_HZ_PER_T * ms * 1e-3) * 1e6      # noqa: E731
    for prot in (dict(sequence="satrec", tr_ms=900.0, ts_ms=400.0, flip_deg=60.0, averages=2), dict(sequence="satrec", tr_ms=300.0, ts_ms=120.0, **HARD),
                 dict(sequence="fse", tr_ms=9000.0, etl=16, prep="inversion", ti_ms=2500.0, refocus_deg=150.0), dict(sequence="fse", tr_ms=3000.0, etl=8)):
        pl = console.plan(prot, exam)
        assert pl["accept"], pl["violations"]
        rp = pl["realised"]
        cycles = math.ceil(rp["lines"] / rp["etl"]) if rp["sequence"] == "fse" else rp["lines"]
        worst_t = max(worst_t, abs(pl["scan_s"] - (sy["overhead_s"] + rp["tr_ms"] / 1e3 + rp["tr_ms"] * cycles * rp["averages"] * rp["partitions"] / 1e3)))
        energy = 1.0 if rp["rf_tbw"] is None else rf.energy_factor(rp["rf_tbw"])
        power = b1(rp["flip_deg"], rp["rf_ms"]) ** 2 * rp["rf_ms"] * energy
        if rp["sequence"] == "fse":
            power += rp["etl"] * b1(rp["refocus_deg"], pl["timing"]["rf_180_ms"]) ** 2 * pl["timing"]["rf_180_ms"] * energy
        if rp["sequence"] == "satrec" or rp["prep"] != "none":
            power += sy["inv_b1_ut"] ** 2 * sy["inv_ms"]
        worst_p = max(worst_p, abs(pl["limits"]["rf_power_percent"] - round(100 * power / rp["tr_ms"] / exam["b1rms_limit_ut"] ** 2, 2)))
    rows.append(_row("J47", "core", "scan time and RF power of `satrec` and of `fse` with and without a preparation (4 protocols)", "overhead + one dummy "
                     "repetition + TR x cycles x averages x partitions; (B1^2 x duration of every pulse of the repetition, the saturation or inversion "
                     "pulse of 12 uT and 8 ms among them) / TR / limit^2, recomputed here; measured: seconds, then percentage points", max(worst_t, worst_p),
                     0.011, f"{worst_t:.1e} s, {worst_p:.3f} points"))
    return rows


# ================================================================================================== the console
def check_console():
    rows = []
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0, sequences="all")
    s = console.SYSTEM
    bad, n_cases = 0, 0
    worst_t, worst_p = 0.0, 0.0
    for seq, key, n, more in (("fse", "etl", 8, {}), ("fse", "etl", 24, dict(matrix=192, phase_fov=0.8, averages=2, refocus_deg=140.0)),
                              ("fse", "etl", 7, dict(partial_fourier=0.7, echo_order="centric", rf_pulse="sharp")),
                              ("mese", "n_echoes", 16, dict(phase_oversampling=0.5, tx_scale=0.9)), ("mese", "n_echoes", 32, dict(refocus_deg=120.0, **HARD))):
        pl = console.plan(dict(dict(sequence=seq, tr_ms=4000.0), **{key: n}, **more), exam)
        assert pl["accept"], pl["violations"]
        rp, n_cases = pl["realised"], n_cases + 1
        shots = math.ceil(rp["lines"] / n) if seq == "fse" else rp["lines"]
        time_s = s["overhead_s"] + s["prepared_dummy_cycles"] * rp["tr_ms"] / 1e3 + rp["tr_ms"] * shots * rp["averages"] * rp["partitions"] / 1e3
        worst_t = max(worst_t, abs(pl["scan_s"] - time_s))
        # RF power: the excitation and n refocusing pulses, each a hard pulse of its angle and duration x the energy factor of the shape
        tbw = rp["rf_tbw"]
        energy = 1.0 if tbw is None else rf.energy_factor(tbw)
        b1 = lambda deg, ms: math.radians(deg) / (2 * math.pi * console.GAMMA_HZ_PER_T * ms * 1e-3) * 1e6      # noqa: E731
        d90, d180 = rp["rf_ms"], pl["timing"]["rf_180_ms"]
        power = (b1(90.0 * rp["tx_scale"], d90) ** 2 * d90 + n * b1(rp["refocus_deg"] * rp["tx_scale"], d180) ** 2 * d180) * energy / rp["tr_ms"]
        worst_p = max(worst_p, abs(pl["limits"]["rf_power_percent"] - round(100 * power / exam["b1rms_limit_ut"] ** 2, 2)))
        # the sampling plan
        plan = console.sampling(rp)
        acquired = kspace.Engine.line_weights(rp) > 0
        bad += list(acquired) != console.acquired_lines(rp["lines_full"], rp["lines"])
        if seq == "fse":
            lines = np.asarray(plan.lines[0])
            used = np.bincount(lines[acquired], minlength=n)
            bad += (int(np.any(lines[~acquired] != -1)) + int(np.any(lines[acquired] < 0)) + int(used.max() > plan.shots) + int(used.sum() != rp["lines"])
                    + int(plan.shots != shots) + int(abs((lines[rp["lines_full"] // 2] + 1) * rp["echo_spacing_ms"] - rp["te_ms"]) > 1e-6))
        else:
            bad += int([i for i, _ in plan.images] != list(range(n))) + int([t for _, t in plan.images] != [round((i + 1) * rp["echo_spacing_ms"], 6) for i in range(n)])
    rows.append(_row("J37", "core", f"scan time of `fse` and `mese` in {n_cases} protocols (partial Fourier, phase oversampling, averages, 3D)",
                     "overhead + one dummy repetition + TR x cycles x averages x partitions, cycles = ceil(lines / etl) for fse and lines for mese",
                     worst_t, 1e-6, f"{worst_t:.1e} s"))
    rows.append(_row("J38", "core", "RF power of the same protocols, percent of the limit", "(B1^2 x duration of the excitation + echoes x B1^2 x duration of a "
                     "refocusing pulse at its nominal angle x tx_scale) x energy factor of the pulse shape / TR / limit^2, recomputed here",
                     worst_p, 0.011, f"{worst_p:.3f} points"))
    rows.append(_row("J39", "core", "sampling plans of the same protocols: an acquired line without an echo, an echo on a line that is not acquired, an echo "
                     "used more often than there are shots, the centre line not on the echo of te_ms, images of `mese` not on echoes 1 to n, the "
                     "acquired lines not those the reconstruction weights", "none", bad, 0, f"{bad} faults"))

    # ---- refusals: every one names a value that is accepted
    cases = (dict(sequence="fse", tr_ms=3000.0, etl=100), dict(sequence="fse", tr_ms=3000.0, etl=8, te_ms=500.0, echo_order="linear"), dict(sequence="fse", tr_ms=3000.0, etl=8, matrix=384),
             dict(sequence="fse", tr_ms=3000.0, etl=8, te_ms=5.0, echo_order="linear"),
             dict(sequence="fse", tr_ms=3000.0, etl=8, echo_order="centric", te_ms=40.0), dict(sequence="fse", tr_ms=3000.0, etl=8, echo_spacing_ms=3.0),
             dict(sequence="mese", tr_ms=2000.0, n_echoes=48), dict(sequence="megre", tr_ms=100.0, flip_deg=20.0, n_echoes=20),
             dict(sequence="megre", tr_ms=100.0, flip_deg=20.0, n_echoes=4, echo_spacing_ms=1.0), dict(sequence="dwse", tr_ms=3000.0, b_value=5000.0),
             dict(sequence="dwse", tr_ms=3000.0, b_value=1000.0, te_ms=20.0), dict(sequence="fse", tr_ms=60.0, etl=16, refocus_deg=60.0),
             dict(sequence="fse", tr_ms=140.0, etl=16))
    unrepaired = 0
    for prot in cases:
        pl = console.plan(prot, exam)
        v = pl["violations"][0] if not pl["accept"] else None
        fix = None if v is None else v.get("nearest_valid", v.get("minimum"))
        if v is not None and v["code"] == "RF_POWER_LIMIT":
            continue
        if v is None or fix is None or not console.plan(dict(prot, **{v["parameter"]: fix}), exam)["accept"]:
            unrepaired += 1
    classic = console.plan(dict(sequence="fse", tr_ms=3000.0, etl=8), dict(exam, sequences=None))
    unrepaired += int(classic["accept"] or classic["violations"][0]["code"] != "UNKNOWN_SEQUENCE" or "fse" in classic["violations"][0]["message"])
    rows.append(_row("J40", "core", f"{len(cases)} refused protocols of the four sequences (echo train too long, effective TE beyond the train or not on the first "
                     "echo of a centric order, matrix, echo spacing, echoes, b-value beyond the range, TE below what the lobes need, TR below the train): a "
                     "refusal whose stated nearest valid value is not accepted when written into the protocol (an effective TE below the first echo is refused as "
                     "in `se`, not raised; the RF power limit, which a train of 16 pulses of 180 "
                     "degrees every 140 ms exceeds, names no value: there the refusal itself is the check); and a console whose card does not list the "
                     "sequences accepting one", "none", unrepaired, 0, f"{unrepaired} faults"))
    return rows


CHECKS = (check_train_amplitudes, check_line_formation, check_blur, check_mese, check_megre, check_dwse, check_prepared, check_console)
