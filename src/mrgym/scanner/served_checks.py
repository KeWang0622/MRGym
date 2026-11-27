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


def check_train_amplitudes():
    w = _world([("a", _disc(60.0)), ("b", _disc(20.0, 0.0, 25.0)), ("c", _disc(15.0, 30.0, -20.0))], n=128)
    eng = kspace.Engine(w)
    worst, count = 0.0, 0
    for seq, n, tr, esp in (("fse", 8, 2500.0, 12.0), ("fse", 32, 4000.0, 9.0), ("mese", 16, 1200.0, 10.0), ("mese", 32, 600.0, 15.0)):
        rp = _plan(w, sequence=seq, tr_ms=tr, echo_spacing_ms=esp, matrix=64, **{"etl" if seq == "fse" else "n_echoes": n}, **HARD)["realised"]
        for name in ("a", "b", "c"):
            cls = w["tissues"][name]
            want = train_closed_form(n, rp["echo_spacing_ms"], rp["tr_ms"], cls["t1_ms"], cls["t2_ms"])
            worst = max(worst, float(np.abs(_unit_amplitudes(eng, rp, name) - want).max() / want.max()))
            count += n
    rows = [_row("J7", "core", f"`fse` and `mese`, exact pulses (hard, 90 and 180 degrees, transmit scale 1): every echo of trains of 8 to 32 echoes as "
                 f"`Engine.amplitudes` serves it, 3 materials ({count} echoes)", "Mz before the excitation x exp(-k ESP / T2), with Mz from the recursion of the "
                 "longitudinal magnetisation through the repetition (recovery, inversion by every refocusing pulse; written here)", worst, 1e-12,
                 f"{worst:.1e} relative")]

    # ---- the reduction to the refocusing flip angle against the rotation of every slice column
    wb = _world([("a", _disc(60.0)), ("b", _disc(20.0, 0.0, 25.0)), ("c", _disc(15.0, 30.0, -20.0))], n=128,
                b1=lambda y, x: 0.78 + 0.37 * np.exp(-(y * y + x * x) / (2 * 55.0 ** 2)))
    engb = kspace.Engine(wb)
    worst, cases = 0.0, 0
    for seq, n, beta, pulse, more in (("fse", 8, 180.0, "normal", {}), ("mese", 16, 150.0, "fast", {}), ("fse", 32, 120.0, "sharp", {}),
                                      ("fse", 12, 70.0, "fast", dict(tx_scale=1.25)), ("mese", 10, 180.0, "normal", HARD)):
        rp = _plan(wb, sequence=seq, tr_ms=1500.0, refocus_deg=beta, rf_pulse=pulse, matrix=64, **{"etl" if seq == "fse" else "n_echoes": n}, **more)["realised"]
        fast = served.table(engb, rp, solver="train", nodes=4, columns=1)["a"]
        full = served.table(engb, rp, solver="columns", nodes=4)["a"]
        worst = max(worst, float(np.abs(fast - full).max() / np.abs(full).max()))
        cases += 1
    rows.append(_row("J8", "core", f"echo trains through the slice as they are served: three single shots on (echoes + 3) Chebyshev points of the cosine of "
                     f"the refocusing flip angle, from which the periodic Mz and every echo of every slice column follow in closed form "
                     f"(`served.train_response`); {cases} cases (fast, normal, sharp and hard pulses, 8 to 32 echoes, 70 to 180 degrees, transmit scale 0.78 "
                     "to 1.44), 4 transmit scales x every slice column x 3 materials", "`periodic.steady_state` of the registry's block list with the "
                     "Cayley-Klein rotation of the excitation and of the refocusing pulse at every column (no reduction: it shows that a refocusing "
                     "rotation acts on the echoes through its flip angle and the phase of its axis alone, and that the echoes are polynomials in the "
                     "cosine of that angle)", worst, 1e-10, f"{worst:.1e} relative"))

    # ---- the same amplitudes against the isochromat ensemble (no phase graph), stepped to the periodic state
    quick = dict(a=dict(TISSUES["a"], t1_ms=160.0, t2_ms=45.0), b=dict(TISSUES["b"], t1_ms=280.0, t2_ms=90.0))
    wq = _world([("a", _disc(60.0)), ("b", _disc(20.0, 0.0, 25.0))], n=96, tissues=quick,
                b1=lambda y, x: 0.85 + 0.25 * np.exp(-(y * y + x * x) / (2 * 55.0 ** 2)))
    engq = kspace.Engine(wq)
    rp = _plan(wq, sequence="mese", tr_ms=1500.0, n_echoes=6, refocus_deg=140.0, matrix=64)["realised"]
    tab = served.table(engq, rp, solver="train", nodes=3, columns=1)
    tbw, window = engq.pulse(rp)
    seq, labels = console.REGISTRY["mese"].blocks(rp, selective=B.Selective(tbw, window))
    zeta, wz = O.slice_columns(tbw, two_sided=False, window=window)
    sp = O.grid([served._tissue(quick[k]) for k in ("a", "b")], zeta, tab["gn"])
    reps = 6
    ens = core_bloch.run(B.compile_ops(seq, frame=False, repetitions=reps), sp).signal[-len(labels):]
    ens = (ens.reshape(len(labels), 2, zeta.size, tab["gn"].size).real * wz[None, None, :, None]).sum(2)
    worst = float(np.abs(np.moveaxis(ens, 0, -1) - tab["a"]).max() / np.abs(tab["a"]).max())
    rows.append(_row("J9", "core", "`mese` through the slice (6 echoes, 140 degrees, normal pulse, transmit scale 0.85 to 1.1, T1 160 and 280 ms): the "
                     "table that is served", f"isochromat ensemble of the core (`core.bloch`: real 3 x 3 rotations, no phase graph) on the registry's "
                     f"block list, stepped through {reps} repetitions from equilibrium, same slice columns", worst, 1e-8, f"{worst:.1e} relative"))

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
    a, b = eng.acquire(se)[0], eng.acquire(dw)[0]
    same = float(np.abs(a - b).max() / np.abs(a).max())
    rows.append(_row("J15", "core", "`dwse` with b = 0 (the spin echo between perfect crushers, served from the core through the line path: slice columns "
                     "by the core, the Fourier integrals along the phase-encode axis first, one factorisation) on a phantom with a transmit pattern, a "
                     "receive pattern, fat and an off-resonance field: the image", "`se` at the same TR and TE (the engine's closed form through its "
                     "own formation: readout first, its readout basis cut at `kspace.SVD_TOL` = 1e-4 of the largest singular value, which is what the "
                     "difference measures)", same, 1e-4, f"{same:.1e} relative"))
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
        return Q.run(one * min(repetitions, 6), cls["t1_ms"], cls["t2_ms"], kmax=2 * n + 2).signal[-n:]
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
        worst = max(worst, abs(_t2_fit(np.array(rp["echo_times_ms"]), _region(img, rp, 5.0, -10.0, 15.0)) / TISSUES[name]["t2_ms"] - 1))
    rows.append(_row("J19", "core", "`mese` with exact pulses (hard, transmit scale 1), 12 echoes of 12 ms: T2 of a straight-line fit to the logarithm of the "
                     "region means of the 12 images, a disc of T2 80 ms and one of 50 ms", "T2 of the material: without stimulated echoes the images decay "
                     "as exp(-TE / T2)", worst, 1e-9, f"{worst:.1e} relative"))

    # ---- the stimulated echoes through the slice, against the phase graph of the design layer
    cls = dict(t1_ms=900.0, t2_ms=80.0, adc_um2_ms=0.0)
    wd = _world([("t", _disc(40.0))], n=96, tissues=dict(t=dict(TISSUES["a"], **cls)))
    engd = kspace.Engine(wd)
    worst, bias, bias_err, left_out = 0.0, [], 0.0, 0.0
    for beta, pulse, scale in ((180.0, "normal", 1.0), (150.0, "normal", 1.0), (120.0, "fast", 1.0), (180.0, "normal", 0.85), (160.0, "sharp", 1.1)):
        rp = _plan(wd, sequence="mese", tr_ms=1500.0, n_echoes=18, echo_spacing_ms=15.0 if pulse == "sharp" else 12.0, refocus_deg=beta, rf_pulse=pulse,
                   tx_scale=scale, slice_mm=3.0, matrix=64)["realised"]
        tbw, window = engd.pulse(rp)
        got = _unit_amplitudes(engd, rp, "t")
        want = design_train(rp, cls, tbw, window, scale)
        worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
        te = np.array(rp["echo_times_ms"])
        b_got, b_want = _t2_fit(te, got) / cls["t2_ms"] - 1, _t2_fit(te, want) / cls["t2_ms"] - 1
        bias.append(100 * b_got)
        bias_err = max(bias_err, 100 * abs(b_got - b_want))
        real = design_train(rp, cls, tbw, window, scale, crusher=False)
        left_out = max(left_out, float(np.abs(real - want).max() / np.abs(want).max()))
    rows.append(_row("J20", "core", "`mese` through the slice, 18 echoes of 12 ms on a 3 mm slice: every echo served for a material of T1 900 / T2 80 ms; "
                     "refocusing 180, 150, 120 (fast pulse) degrees, 180 at transmit scale 0.85, 160 (sharp pulse, 15 ms) at 1.1", "the time-stepping phase "
                     "graph of the design layer (`mrgym.design.sequence`, separate code) with the console's two pulses as designed pulses, their "
                     "rotation at 4 x the engine's slice positions, real crushers, 24 repetitions from equilibrium, only Mz kept between repetitions "
                     "(`served_checks.design_train`)",
                     worst, 1e-6, f"{worst:.1e} relative"))
    rows.append(_row("J21", "core", "what the stimulated echoes do to a straight-line T2 fit over all 18 echoes in those five cases: T2 is overestimated by "
                     + ", ".join(f"{b:+.1f}" for b in bias) + " %; measured: the smallest of them", "a refocusing pulse that is not 180 degrees "
                     "everywhere in the slice stores magnetisation along z, where it decays with T1 and returns in later echoes: the fit is biased "
                     "upwards even at a nominal 180 degrees (the size is compared with a measurement in docs/physics/measured.md, rung M3b)",
                     min(bias), 5.0, f"{min(bias):+.1f} %", kind="min"))
    rows.append(_row("J22", "core", "the same bias, served against the design layer's phase graph", "equal", bias_err, 1e-3, f"{bias_err:.1e} points"))
    rows.append(_row("J23", "core", "what the ideal crusher at the end of the repetition leaves out, in those five cases (TR 1500 ms, T1 900 ms): the design "
                     "layer's train without it (6 repetitions), where the gratings stored along z survive to the next repetition and return as "
                     "stimulated echoes; "
                     "largest change of an echo relative to the largest echo", "a declared simplification of `fse`, `mese` and `dwse` (limits.md, "
                     "block 8a): its size, not an equality. It grows with T1 / TR", left_out, 0.02, f"{left_out:.1e} relative"))
    fluid, sizes = dict(t1_ms=4000.0, t2_ms=2000.0, adc_um2_ms=0.0), []
    for tr in (1500.0, 3000.0, 6000.0):                                 # a fluid: the gratings live as long as T1
        rp = _plan(wd, sequence="mese", tr_ms=tr, n_echoes=18, echo_spacing_ms=12.0, refocus_deg=150.0, slice_mm=3.0, matrix=64)["realised"]
        tbw, window = engd.pulse(rp)
        with_crusher = design_train(rp, fluid, tbw, window, 1.0)
        sizes.append(float(np.abs(design_train(rp, fluid, tbw, window, 1.0, crusher=False, repetitions=6) - with_crusher).max() / np.abs(with_crusher).max()))
    rows.append(_row("J24", "core", "the same for a fluid (T1 4000 ms, T2 2000 ms; 18 echoes, 150 degrees): TR 1500, 3000, 6000 ms give "
                     + ", ".join(f"{v:.3f}" for v in sizes) + " of the largest echo; measured: the largest", "the same declared simplification where it is "
                     "largest: a train on a material whose T1 is long against TR is NOT what a scanner with a constant spoiler gives (the independent "
                     "review found this size with its own isochromat ensemble: 0.17 at TR 1500 ms)", max(sizes), 0.25, f"{max(sizes):.3f} relative"))
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
    worst, text = 0.0, []
    b_hi = _plan(w, sequence="dwse", tr_ms=3000.0, b_value=1000.0, matrix=64)["realised"]
    b_lo = _plan(w, sequence="dwse", tr_ms=3000.0, b_value=0.0, te_ms=b_hi["te_ms"], matrix=64)["realised"]
    for name in ("a", "b", "c"):
        one = kspace.Engine(_world([(name, _disc(28.0, 10.0, -15.0))], n=192))
        hi, lo = one.acquire(b_hi), one.acquire(b_lo)
        adc = math.log(abs(_region(lo, b_lo, 10.0, -15.0, 12.0)[0]) / abs(_region(hi, b_hi, 10.0, -15.0, 12.0)[0])) / b_hi["b_value"] * 1e3      # um^2/ms
        worst = max(worst, abs(adc / TISSUES[name]["adc_um2_ms"] - 1))
        text.append(f"{adc:.6f}")
    rows.append(_row("J35", "core", f"`dwse`: diffusion coefficient from the region means of two images, b = 0 and b = 1000 s/mm^2 at the same TE "
                     f"({b_hi['te_ms']:g} ms), a disc of each of three materials ({', '.join(text)} um^2/ms)", "the coefficients of the materials (0.8, 3.0, "
                     "1.6 um^2/ms): signal x exp(-b D), every other factor equal in the two scans", worst, 1e-9, f"{worst:.1e} relative"))
    te = [_plan(w, sequence="dwse", tr_ms=3000.0, b_value=b, matrix=64)["realised"]["te_ms"] for b in (0.0, 100.0, 500.0, 1000.0, 2000.0, 3000.0)]
    rising = min(b - a for a, b in zip(te, te[1:]))
    rows.append(_row("J36", "core", "`dwse`: the minimum echo time for b = 0, 100, 500, 1000, 2000, 3000 s/mm^2 at the default geometry: "
                     + ", ".join(f"{t:g}" for t in te) + " ms; measured: the smallest step", "the lobes lengthen the echo time: it rises with b",
                     rising, 0.5, f"{rising:.2f} ms", kind="min"))
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


CHECKS = (check_train_amplitudes, check_line_formation, check_blur, check_mese, check_megre, check_dwse, check_console)
