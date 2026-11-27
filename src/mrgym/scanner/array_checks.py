"""Validation of the receive array (``array``) and of the reconstruction mathematics (``recon``).

    python -m mrgym.scanner.array_checks                the table as text; exit status 1 if a row fails
    python -m mrgym.scanner.array_checks --markdown     the table of docs/physics/array.md
    python -c "from mrgym.scanner import recon; recon.validate()"      the same table

A row has a quantity, the closed form or independent computation it is compared with, the number found and a
tolerance. Rows marked "reported" have no tolerance: they state a size (a residual, a ratio) that a reader needs and
that no formula fixes. Everything runs on the engine's brain world of one seed, with two arrays: the declared one
(8 loops on a ring, ``array.make_array``) and the array of the design prototype handed in from outside
(``array.given``: the on-axis loop law with a linear phase, neighbour correlations 0.35 and 0.10), for which the
design recorded mean g-factors of 1.135 (R = 2) and 1.944 (R = 3) and a cost target of 1.5 single-channel scans.
Noise is measured by pseudo replicas (``recon.pseudo_replica``) through the reconstruction
under test. The references were written by the same author as the code: the rows show that the code computes the
stated equations, not that the equations describe a scanner.
"""
from __future__ import annotations

import resource
import sys
import time

import numpy as np

from . import array as A
from . import console, kspace, recon
from . import world as W

SEED = "0123456789abcdef0123456789abcdef"
PROTOCOL = dict(sequence="spgr", tr_ms=30, te_ms=6, flip_deg=20, fov_mm=240, slice_mm=5)
GROUPS = dict(formation="Formation of multi-channel data", noise="Noise", combination="Coil combination", sense="SENSE and the g-factor",
              cost="Cost")
REPLICAS = 300
_ctx: dict = {}


def _row(id, group, quantity, reference, measured, tolerance=None, kind="max", unit=""):
    """kind "max": passes when measured <= tolerance; "min": when measured >= tolerance; "info": reported, no verdict."""
    measured = float(measured)
    if kind == "info":
        return dict(id=id, group=group, quantity=quantity, reference=reference, measured=measured, tolerance=None, kind=kind, unit=unit, ok=True)
    ok = bool(np.isfinite(measured) and (measured <= tolerance if kind == "max" else measured >= tolerance))
    return dict(id=id, group=group, quantity=quantity, reference=reference, measured=measured, tolerance=float(tolerance), kind=kind,
                unit=unit, ok=ok)


# ---------------------------------------------------------------------------------------------- the two arrays
def memo_maps(y, x, z=0.0, nc=8, ring=135.0, a=95.0):
    """The maps of the design prototype: loops on a ring of 135 mm, the on-axis law of a loop of radius 95 mm for
    the magnitude, a linear phase."""
    y, x = np.broadcast_arrays(np.asarray(y, float), np.asarray(x, float))
    out = []
    for k in range(nc):
        ang = 2 * np.pi * (k + 0.25) / nc
        cy, cx = ring * np.sin(ang), ring * np.cos(ang)
        d2 = (y - cy) ** 2 + (x - cx) ** 2
        out.append((1 + d2 / a ** 2) ** -1.5 * np.exp(1j * (ang + 0.006 * ((y - cy) * np.cos(ang) - (x - cx) * np.sin(ang)))))
    return np.array(out)


def memo_psi(nc=8, r1=0.35, r2=0.10):
    psi = np.eye(nc, dtype=complex)
    for i in range(nc):
        j = (i + 1) % nc
        psi[i, j] = psi[j, i] = r1 * np.exp(0.4j) if i < j else r1 * np.exp(-0.4j)
        psi[i, (i + 2) % nc] = psi[(i + 2) % nc, i] = r2
    return 0.5 * (psi + psi.conj().T)


def world():
    if "world" not in _ctx:
        _ctx["world"] = W.make_world(SEED, dict(kind="brain", array=dict(n=8)))
    return _ctx["world"]


def engine(name="ring"):
    """ArrayEngine of the brain world with the declared ring ("ring"), the prototype array ("memo"), a row of four
    loops behind the head ("row"), the ring with one dead element ("dead")."""
    key = ("engine", name)
    if key not in _ctx:
        w = world()
        if name == "memo":
            arr = A.given(memo_maps, memo_psi())
        elif name == "row":
            arr = A.make_array(dict(n=4, layout="row", side="posterior"), w, W.rng_for(SEED, "fields", 2))
        elif name == "dead":
            arr = A.with_faults(w["array"], dead=[2])
        else:
            arr = w["array"]
        _ctx[key] = A.ArrayEngine(A.attach(w, arr))
    return _ctx[key]


def realised(matrix=96, **more):
    pl = console.plan(dict(PROTOCOL, matrix=matrix, **more), world()["console"])
    assert pl["accept"], pl
    return pl["realised"]


def _rms(a, b):
    return float(np.sqrt((np.abs(a - b) ** 2).sum() / (np.abs(b) ** 2).sum()))


def scan(name, matrix=96, **more):
    """Noise-free data of one protocol and what the rows share: k-space, maps on the pixels, the optimal full image,
    the object (pixels above 10 % of that image's maximum)."""
    key = ("scan", name, matrix, tuple(sorted(more.items())))
    if key not in _ctx:
        ae, rp = engine(name), realised(matrix, **more)
        ks = ae.clean_kspace(rp)[0]
        maps = ae.pixel_maps(rp)
        full = recon.roemer(ae.coil_images(ks, rp, full_pe=True), maps, ae.psi)
        _ctx[key] = dict(ae=ae, rp=rp, ks=ks, maps=maps, full=full, body=np.abs(full) > 0.1 * np.abs(full).max())
    return _ctx[key]


# ---------------------------------------------------------------------------------------------- formation
def check_formation():
    rows, w = [], world()
    # one uniform coil = the single-channel scanner (receive profile 1), noise included
    w1 = dict(w, fields=dict(w["fields"], rx=np.ones((w["n"], w["n"]), complex)))
    one = A.attach(w1, A.given(lambda y, x, z=0.0: np.ones((1,) + np.broadcast(y, x).shape, complex), np.eye(1)))
    rp = realised(64, n_echoes=2)
    ref = kspace.Engine(w1).acquire(rp, np.random.default_rng(7))
    ae = A.ArrayEngine(one)
    ks = ae.acquire_kspace(rp, np.random.default_rng(7), method="per_coil")
    got = np.array([recon.roemer(ae.coil_images(k, rp), np.ones((1,) + ref.shape[1:], complex)) for k in ks])
    rows.append(_row("F1", "formation", "one uniform coil, full sampling, formed per coil, with noise (two echoes): the combined image against the "
                     "engine's single-channel image of the same world and noise stream", "equal bit for bit", np.abs(got - ref).max(), 0.0))
    plan = recon.sense_plan(np.ones((1, rp["lines_full"], rp["matrix"]), complex), None, 1)
    got = np.array([ae.to_image(recon.sense_unfold(plan, ae.coil_images(k, rp, full_pe=True)), rp) for k in ks])
    rows.append(_row("F2", "formation", "the same through the SENSE unfolding at R = 1", "equal", np.abs(got - ref).max() / np.abs(ref).max(), 1e-14))
    ks = ae.acquire_kspace(rp, np.random.default_rng(7), method="fast")
    got = np.array([recon.roemer(ae.coil_images(k, rp), np.ones((1,) + ref.shape[1:], complex)) for k in ks])
    clean = kspace.Engine(w1).acquire(rp)
    rows.append(_row("F3", "formation", "the same by the fast formation: rms difference / rms of the noise-free image (a uniform map: what is left "
                     "is the engine's own compression of the time functions, SVD_TOL = 1e-4, which the split into water and fat changes)",
                     "the per-coil formation", np.sqrt((np.abs(got - ref) ** 2).sum() / (np.abs(clean) ** 2).sum()), 2e-4))
    # fast against per coil, the declared ring of 8 loops
    ae = engine("ring")
    for i, matrix in enumerate((96, 64)):
        rp = realised(matrix)
        rows.append(_row(f"F{4 + i}", "formation", f"fast against per-coil formation, 8 loops on a ring, spgr, matrix {matrix}, 300 Hz per pixel: rms "
                         "difference of the k-space / its rms", "per coil: the map in place of the receive profile on the master grid",
                         _rms(ae.fast(rp), ae.per_coil(rp)), 5e-3))
    # where the difference comes from: one loop, smaller and smaller, against the pixel
    w2, errs = world(), []
    rp = realised(64)
    for radius in (60.0, 30.0, 15.0):
        el = dict(y_mm=w2["size_mm"][0] / 2 + 20.0, x_mm=0.0, z_mm=0.0, normal_yx=(-1.0, 0.0), radius_mm=radius)
        a1 = A.ArrayEngine(A.attach(w2, A.given(A.LoopMaps([el]))))
        errs.append(_rms(a1.fast(rp), a1.per_coil(rp)))
    px = rp["fov_mm"] / rp["matrix"]
    for i, (radius, e) in enumerate(zip((60.0, 30.0, 15.0), errs)):
        rows.append(_row(f"F{6 + i}", "formation", f"the same for one loop of radius {radius:g} mm ({radius / px:.0f} pixels) 20 mm behind the head, "
                         "matrix 64: the difference grows as the map varies more inside the blur of a sample", "reported", e, kind="info"))
    rows.append(_row("F9", "formation", "the positions used for the maps on a turned and warped grid (rotation 30 degrees, gradient non-linearity "
                     "0.03) against the master grid of that world", "world.grid", _coords_error(), 1e-9, unit="mm"))
    return rows


def _coords_error():
    w = W.make_world(SEED, dict(kind="qa", n=96, gnl=0.03))
    t = W.turned(w, 30.0)
    c = (np.arange(t["n"]) - t["n"] // 2) * (t["extent_mm"] / t["n"])
    got = A.patient_coords(t, c[:, None], c[None, :])
    return max(float(np.abs(a - b).max()) for a, b in zip(got, W.grid(t)))


# ---------------------------------------------------------------------------------------------- noise
def check_noise():
    rows = []
    ae, rp, rng = engine("ring"), realised(32), np.random.default_rng(20261126)
    acc, n_scans = 0.0, 1000
    for _ in range(n_scans):
        nz = ae.noise(rp, rng).reshape(ae.n_coils, -1)
        acc = acc + nz @ nz.conj().T
    n = n_scans * nz.shape[1]
    est = acc / (2 * n * ae.sample_sd(rp) ** 2)
    se = np.sqrt(np.outer(np.diag(ae.psi).real, np.diag(ae.psi).real) / n)
    rows.append(_row("N1", "noise", f"covariance of the samples of {n_scans} noise scans (8 channels, {nz.shape[1]} samples each) / (2 sd^2), "
                     "sd from the single-channel noise law: largest deviation from psi in units of its sampling error",
                     "psi of the array; sampling error sqrt(psi_ii psi_jj / N)", (np.abs(est - ae.psi) / se).max(), 4.5, unit="standard errors"))
    rows.append(_row("N2", "noise", "the same: largest relative deviation of an element", "reported",
                     (np.abs(est - ae.psi) / np.sqrt(np.outer(np.diag(ae.psi).real, np.diag(ae.psi).real))).max(), kind="info"))
    rp = realised(96)
    rows.append(_row("N3", "noise", "noise of a coil image of a channel with psi_ii = 1 (ArrayEngine.image_sd)", "Engine.noise_sd of the same protocol",
                     abs(ae.image_sd(rp) / kspace.Engine(world()).noise_sd(rp) - 1), 1e-12))
    # background of the root sum of squares: chi statistics
    mean, sd = recon.chi_moments(ae.n_coils)
    for whiten in (True, False):
        vals = np.concatenate([recon.sos(ae.coil_images(ae.noise(rp, rng), rp, full_pe=True) / ae.image_sd(rp), ae.psi if whiten else None).ravel()
                               for _ in range(40)])
        if whiten:
            rows.append(_row("N4", "noise", f"background of the root sum of squares of 8 whitened channels ({vals.size} pixels of noise): mean",
                             f"chi distribution with 16 degrees of freedom: sqrt(2) Gamma(8.5) / Gamma(8) = {mean:.4f} sd", abs(vals.mean() / mean - 1), 3e-3))
            rows.append(_row("N5", "noise", "the same: standard deviation", f"sqrt(16 - mean^2) = {sd:.4f} sd", abs(vals.std() / sd - 1), 5e-3))
        else:
            scale = np.sqrt(np.trace(ae.psi).real / ae.n_coils)
            rows.append(_row("N6", "noise", "the same without whitening (channels correlated as psi says), mean / the chi mean for the mean channel "
                             "variance: the formula needs white channels", "reported", vals.mean() / (mean * scale), kind="info"))
    vals = np.concatenate([np.abs(ae.coil_images(ae.noise(rp, rng), rp, full_pe=True)[0]).ravel() for _ in range(40)]) / (ae.image_sd(rp) * np.sqrt(ae.psi[0, 0].real))
    mean, sd = recon.chi_moments(1)
    rows.append(_row("N7", "noise", "background magnitude of one channel: mean and standard deviation (the larger deviation)",
                     f"Rayleigh: {mean:.4f} sd, {sd:.4f} sd", max(abs(vals.mean() / mean - 1), abs(vals.std() / sd - 1)), 5e-3))
    return rows


# ---------------------------------------------------------------------------------------------- combination
def check_combination():
    rows, rng = [], np.random.default_rng(1990)
    s = scan("ring")
    ae, rp, maps, body, psi = s["ae"], s["rp"], s["maps"], s["body"], s["ae"].psi
    snr_r = 1.0 / recon.roemer_noise(maps, psi)
    snr_s = recon.sos_snr(maps, psi)
    best = (np.abs(maps) / np.sqrt(np.diag(psi).real)[:, None, None]).max(0)
    rows.append(_row("C1", "combination", "SNR of the optimal combination / SNR of the root sum of squares, smallest value over the object "
                     "(closed forms, 8 loops)", "at least 1 (Cauchy-Schwarz)", (snr_r / snr_s)[body].min(), 1.0, "min"))
    rows.append(_row("C2", "combination", "SNR of the root sum of squares / SNR of the optimal combination, mean over the object", "reported",
                     (snr_s / snr_r)[body].mean(), kind="info"))
    rows.append(_row("C3", "combination", "SNR of the optimal combination / SNR of the best single channel, smallest value over the object",
                     "at least 1 (the single channel is one of the linear combinations)", (snr_r / best)[body].min(), 1.0, "min"))
    rows.append(_row("C4", "combination", "share of the object where the root sum of squares has at least the SNR of the best single channel",
                     "reported (no theorem: see C5)", (snr_s >= best)[body].mean(), kind="info"))
    noisy = A.with_faults(ae.arr, noisy={0: 16.0})
    m2, p2 = noisy["maps"](*ae.pixel_coords(rp)), noisy["psi"]
    b2 = (np.abs(m2) / np.sqrt(np.diag(p2).real)[:, None, None]).max(0)
    rows.append(_row("C5", "combination", "the same share when one channel has 16 times the noise variance, without whitening", "reported",
                     (recon.sos_snr(m2, p2) >= b2)[body].mean(), kind="info"))
    rows.append(_row("C6", "combination", "root sum of squares of whitened channels: SNR / SNR of the optimal combination, largest deviation "
                     "(closed forms, noisy channel included)", "equal", np.abs(recon.sos_snr(recon.prewhiten(m2, p2)) * recon.roemer_noise(m2, p2) - 1)[body].max(), 1e-12))
    # measured by replicas
    shape = s["ks"].shape[1:]
    sd_k, sd_i = ae.sample_sd(rp), ae.image_sd(rp)
    sd_r = recon.pseudo_replica(lambda k: recon.roemer(ae.coil_images(k, rp, True), maps, psi), shape, psi, sd_k, REPLICAS, rng)
    rows.append(_row("C7", "combination", f"noise of the optimal combination by {REPLICAS} replicas / its closed form, mean over the object",
                     "sd of a coil image / sqrt(S^H psi^-1 S)", abs((sd_r / (sd_i * recon.roemer_noise(maps, psi)))[body].mean() - 1), 0.01))
    hi = s["ks"] * 20.0                                                 # a scan with 20 times the signal: the high-SNR regime of the formula
    sd_s = recon.pseudo_replica(lambda k: recon.sos(ae.coil_images(k, rp, True)), shape, psi, sd_k, REPLICAS, rng, data=hi)
    sig = recon.sos(ae.coil_images(hi, rp, True))
    ratio = (sig / sd_s) / (np.abs(s["full"]) * 20.0 / sd_i * snr_s)
    strong = body & (np.abs(s["full"]) > 0.3 * np.abs(s["full"]).max())
    rows.append(_row("C8", "combination", "SNR of the root sum of squares by replicas of a scan with 20 times the signal / its closed form, mean "
                     "over the bright half of the object", "S^H S / sqrt(S^H psi S) x magnetisation / sd", abs(ratio[strong].mean() - 1), 0.02))
    # two channels, closed forms
    worst = 0.0
    for rho in (0.0, 0.3, -0.5, 0.8):
        p = np.array([[1, rho], [rho, 1]], complex)
        worst = max(worst, abs(1 / recon.roemer_noise(np.ones((2, 1, 1), complex), p)[0, 0] - np.sqrt(2 / (1 + rho))))
    rows.append(_row("C9", "combination", "two channels with equal sensitivity and noise correlation rho (0, 0.3, -0.5, 0.8): gain of the optimal "
                     "combination over one channel", "sqrt(2 / (1 + rho))", worst, 1e-12))
    worst = 0.0
    for rho in (0.3, 0.6 * np.exp(1.1j), -0.9):
        p = np.array([[1, rho], [np.conj(rho), 1]], complex)
        worst = max(worst, abs(1 / recon.roemer_noise(np.array([1, 0], complex).reshape(2, 1, 1), p)[0, 0] - 1 / np.sqrt(1 - abs(rho) ** 2)))
    rows.append(_row("C10", "combination", "two channels, the second sees no signal but shares noise (rho = 0.3, 0.6 e^(1.1 i), -0.9): gain of "
                     "the optimal combination over the first channel alone", "1 / sqrt(1 - |rho|^2)", worst, 1e-12))
    # adaptive
    img, _ = recon.adaptive(ae.coil_images(s["ks"], rp, True), psi, block=5)
    ref = np.abs(recon.roemer(ae.coil_images(s["ks"], rp, True), maps, psi, snr_units=True))
    rows.append(_row("C11", "combination", "adaptive combination (Walsh, 5 x 5 pixels, no maps) of the noise-free scan: magnitude / the optimal "
                     "combination in the same units, median deviation over the object", "equal where the block sees one sensitivity direction",
                     np.median(np.abs(np.abs(img) / ref - 1)[body]), 0.02))
    noisy_ks = s["ks"] + recon.correlated_noise(shape, psi, sd_k, rng)
    sd_a = recon.pseudo_replica(lambda k: np.abs(recon.adaptive(ae.coil_images(k, rp, True), psi, 5)[0]), shape, psi, sd_k, 60, rng, data=noisy_ks)
    rows.append(_row("C12", "combination", "noise of the magnitude of the adaptive combination of a noisy scan by 60 replicas, in units of the "
                     "whitened channel noise, mean over the bright half of the object", "reported; 1 for fixed unit weights",
                     (sd_a / sd_i)[strong].mean(), kind="info"))
    return rows


# ---------------------------------------------------------------------------------------------- SENSE
def _discrete(rng, r, offset):
    """Round-off test of the unfolding: an object on the lattice, coil images = maps x object on the pixels."""
    s = scan("ring")
    maps = s["maps"][:, :, ::2]
    obj = (rng.standard_normal(maps.shape[1:]) + 1j * rng.standard_normal(maps.shape[1:])) * s["body"][:, ::2]
    ks = recon.fft2c(maps * obj)
    lines = recon.sampling_mask(maps.shape[1], r, 0, offset)
    plan = recon.sense_plan(maps, s["ae"].psi, r, offset)
    return np.abs(recon.sense(ks * lines[:, None], plan) - obj).max() / np.abs(obj).max(), ks, lines, maps, obj


def check_sense():
    rows, rng = [], np.random.default_rng(1999)

    def add(quantity, reference, measured, tolerance=None, kind="max", unit=""):
        rows.append(_row(f"P{len(rows) + 1}", "sense", quantity, reference, measured, tolerance, kind, unit))

    worst = max(_discrete(rng, r, off)[0] for r, off in ((2, 0), (3, 0), (4, 0), (2, 1), (3, 2), (4, 3)))
    add("residual of the unfolding when the data are the maps times an object on the pixels (R = 2, 3, 4, every offset of the lattice): largest "
        "error / largest value", "round-off", worst, 1e-10)
    _, ks, lines, maps, obj = _discrete(rng, 3, 0)
    lines2 = recon.sampling_mask(maps.shape[1], 3, 12)
    cg = recon.sense_cg(ks * lines2[:, None], lines2, maps, engine("ring").psi, n_iter=300, tol=1e-12)
    add("the same by conjugate gradients on R = 3 with 12 calibration lines", "the object", np.abs(cg - obj).max() / np.abs(obj).max(), 1e-6)
    # g for two aliased pixels: closed form
    c = 6
    a = rng.standard_normal((c, 2, 1)) + 1j * rng.standard_normal((c, 2, 1))
    q = rng.standard_normal((c, c)) + 1j * rng.standard_normal((c, c))
    psi = q @ q.conj().T + c * np.eye(c)
    pi = np.linalg.inv(psi)
    cos = (a[:, 0, 0].conj() @ pi @ a[:, 1, 0]) / np.sqrt((a[:, 0, 0].conj() @ pi @ a[:, 0, 0]).real * (a[:, 1, 0].conj() @ pi @ a[:, 1, 0]).real)
    add("g of two pixels folded onto each other, 6 random channels and a random psi", "1 / sqrt(1 - |c|^2), c = the cosine between the two pixels' "
        "sensitivity vectors in the metric psi^-1", abs(recon.sense_plan(a, psi, 2)["g"][0, 0] - 1 / np.sqrt(1 - abs(cos) ** 2)), 1e-10)
    worst = 0.0
    for rho in (0.3, 0.6 * np.exp(1.1j), -0.9):
        p = np.array([[1, rho], [np.conj(rho), 1]], complex)
        worst = max(worst, abs(recon.sense_plan(np.eye(2, dtype=complex).reshape(2, 2, 1), p, 2)["g"][0, 0] - 1 / np.sqrt(1 - abs(rho) ** 2)))
    add("two channels, each seeing one of the two folded pixels, noise correlation rho (0.3, 0.6 e^(1.1 i), -0.9): g", "1 / sqrt(1 - |rho|^2)", worst, 1e-12)
    # the engine's brain world: analytic g, replicas, the SNR law
    for name, label in (("memo", "the prototype array (given from outside)"), ("ring", "8 loops on a ring")):
        s = scan(name)
        ae, rp, maps, body, psi = s["ae"], s["rp"], s["maps"], s["body"], s["ae"].psi
        shape, sd_k = s["ks"].shape[1:], ae.sample_sd(rp)

        def full_image(k, ae=ae, rp=rp, maps=maps, psi=psi):
            return recon.roemer(ae.coil_images(k, rp, True), maps, psi)

        sd_full = recon.pseudo_replica(full_image, shape, psi, sd_k, REPLICAS, rng)
        for r in (2, 3):
            plan, lines = recon.sense_plan(maps, psi, r), recon.sampling_mask(rp["lines_full"], r)

            def unfold(k, plan=plan, ae=ae, rp=rp):
                return recon.sense_unfold(plan, ae.coil_images(k, rp, True))

            rec = unfold(s["ks"] * lines[:, None])
            sd_r = recon.pseudo_replica(unfold, shape, psi, sd_k, REPLICAS, rng, lines=lines)
            g = plan["g"]
            if name == "memo":
                add(f"{label}, matrix 96, R = {r}: mean g over the object, closed form", f"the value recorded for the prototype: {dict([(2, 1.135), (3, 1.944)])[r]}",
                    abs(g[body].mean() - {2: 1.135, 3: 1.944}[r]), 0.002)
            else:
                add(f"{label}, matrix 96, R = {r}: mean g over the object, closed form", "reported", g[body].mean(), kind="info")
            add(f"{label}, R = {r}: g measured by {REPLICAS} replicas (noise of SENSE / noise of the full optimal image / sqrt(R)), mean over the object, "
                "/ the mean closed form", "1", abs((sd_r / sd_full / np.sqrt(r))[body].mean() / g[body].mean() - 1), 0.01)
            law = (np.abs(s["full"]) / sd_full) / (np.abs(rec) / sd_r) / (g * np.sqrt(r))
            add(f"{label}, R = {r}: SNR_full / SNR_R / (g sqrt(R)), mean over the object", "1", abs(law[body].mean() - 1), 0.01)
            add(f"{label}, R = {r}: the verifier's noise map (ArrayEngine.noise_sd_map) / the noise by replicas, mean over the object",
                "sd_full(p) g(p) sqrt(R)", abs((ae.to_image(sd_r, rp) / ae.noise_sd_map(rp, r))[ae.to_image(body, rp)].mean() - 1), 0.01)
            res = (np.abs(rec - s["full"]) / np.abs(s["full"]))[body]
            add(f"{label}, R = {r}: residual aliasing of the engine's data with the true maps at the pixel centres, median of |SENSE - full| / |full| "
                "over the object (a map varies inside a pixel; the data know, the pixel value does not)", "reported", np.median(res), kind="info")
    _estimated_maps(add)
    return rows


def _estimated_maps(add):
    """Residual aliasing with maps the agent's side can have (from a prescan with the body coil, from calibration
    lines); the geometry of the array; a dead element; regularisation."""
    s = scan("ring")
    ae, rp, psi, body = s["ae"], s["rp"], s["ae"].psi, s["body"]
    shape = s["maps"].shape[1:]
    low = realised(64, bandwidth_hz_per_px=600)
    one = kspace.Engine(world())
    cal = recon.fft2c(ae.coil_images(ae.clean_kspace(low)[0], low, True))
    ref = recon.fft2c(ae.coil_images(one.clean_kspace(low)[0], low, True))
    est, _ = recon.estimate_maps(cal, ref, shape)
    full_imgs = ae.coil_images(s["ks"], rp, True)
    body_full = ae.coil_images(one.clean_kspace(rp)[0], rp, True)       # what the body coil shows at full sampling
    own = recon.roemer(full_imgs, est, psi)                             # the full scan combined with the estimated maps

    def med(a, b):
        return float(np.median((np.abs(a - b) / np.abs(b))[body]))

    add("8 loops on a ring, maps estimated from a prescan (matrix 64, 600 Hz per pixel) with the body coil as reference, full sampling: median of "
        "|combined image - body-coil image| / |body-coil image| over the object (the error of the maps alone)", "reported", med(own, body_full), kind="info")
    for r in (2, 3):
        lines = recon.sampling_mask(rp["lines_full"], r)
        rec = recon.sense_unfold(recon.sense_plan(est, psi, r), ae.coil_images(s["ks"] * lines[:, None], rp, True))
        add(f"the same maps, R = {r}: median of |SENSE - body-coil image| / |body-coil image| over the object", "reported", med(rec, body_full), kind="info")
        add(f"the same maps, R = {r}: the part that is residual aliasing: median of |SENSE - full scan combined with the same maps| / the latter",
            "reported", med(rec, own), kind="info")
    lines = recon.sampling_mask(rp["lines_full"], 2, 24)
    k_img = recon.fft2c(ae.coil_images(s["ks"] * lines[:, None], rp, True))
    est, _ = recon.estimate_maps(recon.calibration_block(k_img, lines), None, shape)
    add("8 loops on a ring, R = 2 with 24 calibration lines, maps estimated from those lines (no body coil): median of |SENSE - full scan combined "
        "with the same maps| / the latter over the object", "reported", med(recon.sense(k_img, recon.sense_plan(est, psi, 2)), recon.roemer(full_imgs, est, psi)), kind="info")
    # geometry: where the coils do not differ, g rises
    out = {}
    for name in ("ring", "row"):
        for pe in ("y", "x"):
            sc = scan(name, 96, pe_dir=pe)
            for r in (2, 3):
                out[name, pe, r] = recon.sense_plan(sc["maps"], sc["ae"].psi, r)["g"][sc["body"]].mean()
    add("a row of 4 loops behind the head (they differ along x only), R = 2: mean g over the object, phase direction along y", "reported", out["row", "y", 2], kind="info")
    add("the same, phase direction along x", "reported", out["row", "x", 2], kind="info")
    for r in (2, 3):
        add(f"the row of 4 loops, R = {r}: mean g with the phase direction along y / along x", "above 1: folding along the axis on which the loops do "
            "not differ costs more", out["row", "y", r] / out["row", "x", r], 1.1, "min")
    add("the ring of 8 loops, R = 3: the same ratio", "reported", out["ring", "y", 3] / out["ring", "x", 3], kind="info")
    # a dead element
    d, h = scan("dead"), scan("ring")
    pd, ph = recon.sense_plan(d["maps"], d["ae"].psi, 3), recon.sense_plan(h["maps"], h["ae"].psi, 3)
    ratio = np.where(body, pd["unfold_noise"] / ph["unfold_noise"], 1.0)
    add("one of the 8 loops dead, R = 3: noise of the unfolded image / noise with all loops, smallest value over the object",
        "at least 1 (a channel less cannot lower the noise)", ratio[body].min(), 1 - 1e-9, "min")
    gr = np.where(body, pd["g"] / ph["g"], 1.0)
    i = np.unravel_index(np.argmax(gr), gr.shape)
    y, x = d["ae"].pixel_coords(rp)
    ranks = []
    for row in pd["rows"][i[0] % pd["rows"].shape[0]]:                   # the pixels folded onto each other there
        dist = [np.hypot(y[row, i[1]] - e["y_mm"], x[row, i[1]] - e["x_mm"]) for e in h["ae"].arr["elements"]]
        ranks.append(1 + np.argsort(dist).tolist().index(2))
    add("the same: at the pixel where g rises most, one of the three pixels folded onto each other has the dead loop as its nearest: best rank of "
        "the dead loop by distance among them", "1: the loop that told that pixel from the others is gone", min(ranks), 1)
    add("the same: g at that pixel with all loops", "reported", ph["g"][i], kind="info")
    add("the same: g at that pixel with the dead loop", "reported", pd["g"][i], kind="info")
    add("the same: mean g over the object with the dead loop / with all loops", "reported", pd["g"][body].mean() / ph["g"][body].mean(), kind="info")
    # regularisation
    p0, p1 = recon.sense_plan(h["maps"], psi, 4), recon.sense_plan(h["maps"], psi, 4, reg=0.003)
    al = ae.coil_images(s["ks"] * recon.sampling_mask(rp["lines_full"], 4)[:, None], rp, True)
    add("R = 4: mean g over the object", "reported", p0["g"][body].mean(), kind="info")
    add("R = 4, Tikhonov regularisation 0.003: mean noise amplification over the object / unregularised", "reported: below 1, paid with bias",
        p1["g_reg"][body].mean() / p0["g"][body].mean(), kind="info")
    add("R = 4: median residual |SENSE - full| / |full| over the object, unregularised", "reported", med(recon.sense_unfold(p0, al), s["full"]), kind="info")
    add("the same with the regularisation", "reported", med(recon.sense_unfold(p1, al), s["full"]), kind="info")


# ---------------------------------------------------------------------------------------------- cost
def check_cost():
    rows = []
    ae, one, rng = engine("ring"), kspace.Engine(world()), np.random.default_rng(3)
    rp = realised(96)
    lines = recon.sampling_mask(rp["lines_full"], 2)
    ae.acquire_kspace(rp, rng, lines), one.acquire(rp, rng)             # the geometry has been seen; contrasts below are new
    t_a, t_1 = [], []
    for i in range(3):
        ra, r1 = realised(96, flip_deg=31.0 + 0.61 * i), realised(96, flip_deg=41.0 + 0.61 * i)
        t0 = time.perf_counter()
        ks = ae.acquire_kspace(ra, rng, lines, method="fast")[0]
        recon.sense_unfold(recon.sense_plan(ae.pixel_maps(ra), ae.psi, 2), ae.coil_images(ks, ra, True))
        t_a.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        one.acquire(r1, rng)
        t_1.append(time.perf_counter() - t0)
    rows.append(_row("T1", "cost", "whole scan of a new contrast, 8 channels, R = 2, matrix 96 (fast formation, noise, SENSE with its unfolding matrices "
                     "and g map) / the single-channel scan of a new contrast, medians of 3 on one core", "design target: 1.5",
                     np.median(t_a) / np.median(t_1), 1.5))
    rows.append(_row("T2", "cost", "the same: the 8-channel scan", "reported", np.median(t_a), kind="info", unit="s"))
    rows.append(_row("T3", "cost", "the same: the single-channel scan", "reported", np.median(t_1), kind="info", unit="s"))
    rows.append(_row("T4", "cost", "peak resident memory of this process so far (worlds on the 640 grid, per-coil references included)", "reported; limit 3000",
                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, kind="info", unit="MB"))
    return rows


# ---------------------------------------------------------------------------------------------- run and print
CHECKS = dict(formation=check_formation, noise=check_noise, combination=check_combination, sense=check_sense, cost=check_cost)


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
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.4g}"
    return f"{x:.1e}"


def markdown_table(rows):
    """The validation table of docs/physics/array.md. Measured numbers below 1e-3 are printed with one digit; they
    and the timings differ between machines."""
    out = []
    for group, title in GROUPS.items():
        part = [r for r in rows if r["group"] == group]
        if not part:
            continue
        out += [f"### {title}", "", "| # | Quantity | Closed form or independent method | Found | Tolerance | |", "|---|---|---|---|---|---|"]
        for r in part:
            unit = f" {r['unit']}" if r["unit"] else ""
            tol = "reported" if r["kind"] == "info" else f"{'≤' if r['kind'] == 'max' else '≥'} {_number(r['tolerance'])}"
            quantity, reference = (r[k].replace("|", "\\|") for k in ("quantity", "reference"))     # a bar inside a cell
            out.append(f"| {r['id']} | {quantity} | {reference} | {_number(r['measured'])}{unit} | {tol} | "
                       f"{'' if r['kind'] == 'info' else 'pass' if r['ok'] else 'FAIL'} |")
        out.append("")
    return "\n".join(out)


def main(argv=None, exit=True):
    argv = sys.argv[1:] if argv is None else list(argv)
    rows, seconds = run([a for a in argv if a in CHECKS] or None)
    if "--markdown" in argv:
        print(markdown_table(rows))
    else:
        for r in rows:
            tol = "reported" if r["kind"] == "info" else f"{'<=' if r['kind'] == 'max' else '>='} {_number(r['tolerance'])}"
            print(f"{r['id']:4s} {'    ' if r['kind'] == 'info' else 'pass' if r['ok'] else 'FAIL'}  {_number(r['measured']):>9s} {tol:<11s} {r['unit']:16s} {r['quantity']}")
        print(f"{sum(r['ok'] for r in rows)} of {len(rows)} rows pass ({sum(r['kind'] == 'info' for r in rows)} of them reported without a tolerance); {seconds:.1f} s")
    if exit:
        sys.exit(0 if all(r["ok"] for r in rows) else 1)
    return rows


if __name__ == "__main__":
    main()
