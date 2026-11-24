"""Physics validation: every behaviour of the scanner against an analytic formula or a separately written calculation.

    mrgym validate            prints the table and exits 1 if any check fails
    mrgym validate --json     the same rows as JSON

Three groups:
  signal    the per-tissue signal models against (a) published closed forms and (b) Bloch calculations in ``bloch.py``
            that share no code with them (matrix steady state, isochromat ensemble, time-stepping EPG)
  encoding  the k-space engine against a brute-force evaluation of the time-dependent signal equation, and each
            textbook image-formation effect against its formula
  console   the console's arithmetic

What this does and does not establish: the scanner computes the equations it states, to the tolerances listed. It is
not a comparison with a measurement or with a real scanner. Rows whose reference is written by the same authors say
so ("separately written"); rows whose reference is a published formula name the publication.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from .scanner import bloch, console, epg, kspace, tissue
from .scanner import signal as S
from .scanner import world as W

D = np.deg2rad
G = S.GAMMA_MHZ_PER_T


def _row(id, group, check, reference, value, limit, measured=None, kind="max"):
    """kind 'max': passes when value <= limit; 'min': passes when value >= limit."""
    ok = bool(value <= limit) if kind == "max" else bool(value >= limit)
    return dict(id=id, group=group, check=check, reference=reference, value=float(value), limit=float(limit),
                kind=kind, measured=measured or f"{value:.2g}", ok=ok)


def _disc(r, cy=0.0, cx=0.0):
    return lambda y, x: np.hypot(y - cy, x - cx) - r


def _rect(hy, hx):
    return lambda y, x: np.maximum(np.abs(y) - hy, np.abs(x) - hx)


def _centroid_px(img, axis):
    """Centroid of |img| (above 10 % of its maximum) along ``axis`` in pixels relative to the centre index n // 2."""
    a = np.abs(img)
    a = np.where(a > 0.1 * a.max(), a, 0.0)
    n = a.shape[axis]
    prof = a.sum(1 - axis)
    return float((prof * (np.arange(n) - n // 2)).sum() / prof.sum())


def _upsample(img, f=8):
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    ny, nx = k.shape
    z = np.zeros((ny * f, nx * f), complex)
    z[ny * f // 2 - ny // 2: ny * f // 2 - ny // 2 + ny, nx * f // 2 - nx // 2: nx * f // 2 - nx // 2 + nx] = k
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(z))) * f * f


def _engine(shapes, system=None, b0_t=3.0, tissues=None, edge=None, **kw):
    ph = W.shapes_phantom(shapes, b0_t=b0_t, tissues=tissues, **({} if edge is None else {"edge": edge}))
    return kspace.Engine(W.flat_world(ph, **(system or {})), **kw)


# ================================================================================================== signal models
def ssfp_fid(a, tr_ms, t1_ms, t2_ms):
    """Gradient-spoiled (no RF spoiling) steady-state FID signal right after the pulse, averaged over one cycle of
    dephasing per TR (Hänicke & Vogel 2003, Magn Reson Med 49:771):
        S = tan(a / 2) (1 - (E1 - cos a) r),   r = sqrt((1 - E2^2) / ((1 - E1 cos a)^2 - E2^2 (E1 - cos a)^2))
    """
    e1, e2 = np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms)
    r = np.sqrt((1 - e2 ** 2) / ((1 - e1 * np.cos(a)) ** 2 - e2 ** 2 * (e1 - np.cos(a)) ** 2))
    return np.tan(a / 2) * (1 - (e1 - np.cos(a)) * r)


def check_closed_forms():
    t1 = np.array([250.0, 400, 700, 1100, 1800, 2800])
    b1 = np.round(np.arange(0.6, 1.41, 0.1), 2)
    g = np.array(list(itertools.product(t1, b1)))
    t1, b1 = g[:, 0].copy(), g[:, 1].copy()
    w = dict(spgr=0.0, afi=0.0, ir=0.0, n=0)
    for flip, tr in itertools.product([1, 3, 5, 10, 20, 40, 60, 90], [4, 8, 15, 30, 100, 500, 3000, 15000]):
        a = b1 * D(flip)
        c = S.spgr(a, tr, t1)
        w["spgr"] = max(w["spgr"], np.abs(c - bloch.spgr_ideal(a, tr, t1)).max())
        w["n"] += c.size
    for flip, tr1, n in itertools.product([10, 20, 40, 60, 80, 90], [4, 10, 20, 50, 150], [2, 3, 5, 10]):
        a = b1 * D(flip)
        c = np.array(S.afi(a, tr1, n * tr1, t1))
        w["afi"] = max(w["afi"], np.abs(c - np.array(bloch.afi_ideal(a, tr1, n * tr1, t1))).max())
        w["n"] += c.size
    for ti, tr, k in itertools.product([20, 50, 100, 300, 700, 1500, 3000, 10000], [500, 1500, 3000, 6000, 15000],
                                       [1.0, 0.95, 0.8]):
        if ti + 2 >= tr:
            continue
        a = b1 * np.pi / 2
        c = S.ir_gre(a, ti, tr, t1, k)
        w["ir"] = max(w["ir"], np.abs(c - bloch.ir_ideal(a, ti, tr, t1, k)).max())
        w["n"] += c.size
    ref = "separately written matrix Bloch steady state (bloch.py); an implementation check of the closed forms"
    return [
        _row("S1", "signal", "Ernst formula, spoiled gradient echo", f"Ernst & Anderson 1966; {ref}", w["spgr"], 1e-12),
        _row("S2", "signal", "dual-TR (AFI) signals", f"Yarnykh 2007; {ref}", w["afi"], 1e-12),
        _row("S3", "signal", "inversion recovery with a gradient-echo readout, inversion efficiency 0.8 to 1", ref, w["ir"], 1e-12),
    ]


def check_ernst_angle():
    worst = 0.0
    for tr, t1 in itertools.product([5.0, 15.0, 40.0], [300.0, 1000.0, 2500.0]):
        a = np.linspace(0.2, 89.8, 44801)
        peak = a[np.argmax(S.spgr(D(a), tr, t1))]
        worst = max(worst, abs(peak - np.degrees(np.arccos(np.exp(-tr / t1)))))
    return [_row("S4", "signal", "flip angle of maximum spoiled-GRE signal", "Ernst angle arccos(exp(-TR / T1)); Ernst & Anderson 1966",
                 worst, 0.005, f"{worst:.4f} deg")]


def check_epg():
    cases = [((1,), 20, 10, 1000, 80, 117), ((1,), 45, 5, 1000, 80, 50), ((1,), 30, 8, 600, 40, 117),
             ((1,), 25, 12, 1500, 200, 150), ((1,), 12, 6, 300, 40, 117), ((1, 1, 0, 0, 0, 0), 60, 20, 1000, 80, 129.3),
             ((1, 1, 0, 0), 45, 10, 600, 60, 117), ((1, 1, 0, 0, 0), 70, 15, 1500, 150, 39)]
    step = iso = 0.0
    for pattern, flip, tr, t1, t2, phi in cases:
        s = epg.steady_state(D(flip), tr, t1, t2, phi, pattern)
        step = max(step, np.abs(s - bloch.rf_spoiled_epg_stepping(D(flip), tr, t1, t2, phi, pattern)).max())
        iso = max(iso, np.abs(s - bloch.rf_spoiled_isochromats(D(flip), tr, t1, t2, phi, pattern, n_iso=1024)).max())
    dif = 0.0
    for pattern, flip, tr, t1, t2, phi, bd in (((1,), 30, 6, 1000, 80, 117, 0.003), ((1,), 20, 20, 1500, 200, 117, 0.15),
                                               ((1, 1, 0, 0, 0, 0), 70, 30, 1200, 150, 45, 0.4)):
        s = epg.steady_state(D(flip), tr, t1, t2, phi, pattern, bd)
        dif = max(dif, np.abs(s - bloch.rf_spoiled_epg_stepping(D(flip), tr, t1, t2, phi, pattern, bd=bd)).max())
    fid = 0.0
    for flip, tr, (t1, t2) in itertools.product([5, 15, 30, 60, 90], [4.0, 10.0, 25.0, 60.0],
                                                [(300, 40), (800, 80), (1500, 200), (2500, 150), (4300, 2000)]):
        s = epg.steady_state(D(flip), tr, t1, t2, 0.0)[0]
        fid = max(fid, abs(s - ssfp_fid(D(flip), tr, t1, t2)))
    rel = lambda s, c: float(np.max(np.abs(np.asarray(s) - np.asarray(c)) / np.abs(c)))   # noqa: E731
    lim = max(max(rel(epg.steady_state(D(f), tr, t1, 0.05)[0], S.spgr(D(f), tr, t1))
                  for f, tr, t1 in itertools.product([5, 30, 80], [5, 20], [300, 2500])),
              max(rel(epg.steady_state(D(f), 600, t1, 40)[0], S.spgr(D(f), 600, t1)) for f in (5, 30, 80) for t1 in (300, 2500)),
              max(rel(epg.steady_state(D(f), 20, t1, 0.05, 129.3, (1, 1) + (0,) * (n - 1)), S.afi(D(f), 20, 20 * n, t1))
                  for f in (30, 60, 90) for t1 in (300, 2500) for n in (2, 5)),
              max(rel(epg.steady_state(D(f), 5, 1000, 80, bd=200.0)[0], S.spgr(D(f), 5, 1000)) for f in (5, 30, 80)))
    gap = abs(abs(epg.steady_state(D(30), 5, 1500, 200)[0]) / S.spgr(D(30), 5, 1500) - 1)
    return [
        _row("S5", "signal", "RF-spoiled steady state (EPG) of spgr and afi, 8 cases", "separately written isochromat-ensemble Bloch simulation (1,024 spins, time stepping)", iso, 1e-9),
        _row("S6", "signal", "the same, and 3 cases with diffusion", "separately written time-stepping EPG in the laboratory frame (same b-factor formula: checks the solver, not the diffusion model)", max(step, dif), 1e-9),
        _row("S7", "signal", "EPG with RF phase increment 0, 100 cases", "SSFP-FID closed form; Hänicke & Vogel 2003", fid, 1e-10),
        _row("S8", "signal", "EPG reduces to the ideal-spoiling formulas when T2 << TR, TR >> T2, or diffusion spoiling is complete", "Ernst formula; AFI formula", lim, 1e-6, f"{lim:.1e} relative"),
        _row("S9", "signal", "EPG differs from the Ernst formula where it should (TR 5 ms, 30 deg, T1 1500 ms, T2 200 ms)", "incomplete spoiling at short TR and long T2 (Zur 1991; Preibisch & Deichmann 2009)", gap, 0.03, f"{100 * gap:.1f} % apart", kind="min"),
    ]


def check_spin_echo():
    rng = np.random.default_rng(5)
    v = 2000
    t1, a, t2 = rng.uniform(200, 4500, v), D(rng.uniform(60, 120, v)), np.full(v, 80.0)
    tr, te, ti, kap = 600.0, 20.0, 150.0, 0.94
    cyc = [bloch.rf_x(a), bloch.spoil(v), bloch.free(te / 2, t1, t2), bloch.rf_x(2 * a), bloch.spoil(v), bloch.free(tr - te / 2, t1, t2)]
    se = np.abs(S.se(a, tr, te, t1) - bloch.periodic(cyc)[:, 2] * np.sin(a) * np.sin(a) ** 2).max()
    ideal = np.abs(S.se(np.pi / 2, tr, te, t1) - (1 - 2 * np.exp(-(tr - te / 2) / t1) + np.exp(-tr / t1))).max()
    cyc = [bloch.rf_x(a), bloch.spoil(v), bloch.free(te / 2, t1, t2), bloch.rf_x(2 * a), bloch.spoil(v),
           bloch.free(tr - ti - te / 2, t1, t2), bloch.invert(v, kap), bloch.free(ti, t1, t2)]
    ir = np.abs(S.ir_se(a, tr, te, ti, t1, kap) - bloch.periodic(cyc)[:, 2] * np.sin(a) * np.sin(a) ** 2).max()
    phi, eff = np.linspace(0, 2 * np.pi, 720, endpoint=False), 0.0
    for beta in (np.pi, 2.6, 2.0):
        m = np.exp(1j * phi)                                             # dephased by the first crusher
        echo = np.mean((m.real + 1j * m.imag * np.cos(beta)) * np.exp(1j * phi))   # rotation about x, second crusher
        eff = max(eff, abs(abs(echo) - np.sin(beta / 2) ** 2))
    # null point of inversion recovery with ideal pulses and TR >> T1: TI = T1 ln 2
    t1n = np.array([400.0, 1000.0, 2500.0])
    ti_grid = np.linspace(100, 3000, 290001)
    null = max(abs(ti_grid[np.argmin(np.abs(S.ir_se(np.pi / 2, 60000.0, 10.0, ti_grid, t)))] / (t * math.log(2)) - 1) for t in t1n)
    return [
        _row("S10", "signal", "spin echo with imperfect pulses (B1 60 to 120 % of nominal)", "separately written matrix Bloch chain; ideal pulses: 1 - 2 E(TR - TE/2) + E(TR)", max(se, ideal), 1e-12),
        _row("S11", "signal", "inversion-recovery spin echo, inversion efficiency 0.94", "separately written matrix Bloch chain", ir, 1e-12),
        _row("S12", "signal", "refocusing efficiency of a spin echo", "sin^2(beta / 2); Hennig 1988", eff, 1e-12),
        _row("S13", "signal", "inversion-recovery null point for TR >> T1", "TI_null = T1 ln 2", null, 1e-4, f"{null:.1e} relative"),
    ]


def check_bssfp():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        t1, tr, al = rng.uniform(200, 4000), rng.uniform(3, 12), D(rng.uniform(5, 90))
        t2, phi = rng.uniform(20, min(t1, 2000)), rng.uniform(-np.pi, np.pi) * 3
        e1, e2 = np.exp(-tr / t1), np.exp(-tr / t2)
        m, z, last = 0j, 1.0, 0j
        for _ in range(int(12 * t1 / tr)):
            my = m.imag * np.cos(al) + z * np.sin(al)
            z = -m.imag * np.sin(al) + z * np.cos(al)
            m = m.real + 1j * my                                         # RF about x
            last = m
            m, z = m * e2 * np.exp(1j * (phi + np.pi)), 1 + (z - 1) * e1  # free precession + 180 degree phase cycle
        worst = max(worst, abs(1j * S.bssfp(al, phi + np.pi, tr, t1, t2) - last))
    return [_row("S14", "signal", "balanced SSFP steady state, 200 random (T1, T2, TR, flip, off-resonance)",
                 "closed form (Freeman & Hill 1971; Scheffler & Lehnhardt 2003) against Bloch time stepping", worst, 1e-4)]


# ================================================================================================== encoding
def check_numerics():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 50)) + 1j * rng.standard_normal((3, 50))
    c, m, off = 0.0123, 20, 0.37
    q = np.arange(m) - m // 2 + off
    direct = np.array([[np.sum(row * np.exp(-2j * np.pi * qq * c * (np.arange(50) - 25))) for qq in q] for row in x])
    czt = np.abs(kspace.zoom_dft(x, 1, m, c, off) - direct).max() / np.abs(direct).max()

    n, ext = 96, 96.0
    ph = W.shapes_phantom([("wm", _disc(34)), ("scalp_fat", lambda y, x: np.abs(np.hypot(y, x) - 30) - 4),
                           ("csf", _disc(7, 8, -6)), ("gm", _disc(6, -12, 9))], n=n, extent_mm=ext)
    w = W.flat_world(ph, cf_offset_hz=17.0)
    y, x = np.meshgrid((np.arange(n) - n // 2) * (ext / n), (np.arange(n) - n // 2) * (ext / n), indexing="ij")
    w["fields"]["b0_hz"] = 60 * (x / 40) ** 2 - 45 * (y / 40) + 25 * np.sin(x / 9)
    w["fields"]["rx"] = (1 + 0.3 * x / 48) * np.exp(1j * (0.4 + y / 30))
    eng = kspace.Engine(w, tier=1, effects=kspace.ALL - {"aa_filter"})
    exam = dict(w["console"], b1rms_limit_ut=1e3, min_pixel_mm=0.5)

    def brute(rp, te):
        epi = rp["sequence"] == "epi"
        k_ro, k_pe = 2 * rp["matrix"], rp["lines_full"]
        fov_ro, fov_pe = 2 * rp["fov_mm"], rp["fov_pe_mm"] * k_pe / rp["matrix_pe"]
        yy, xx = (x, y) if rp["pe_dir"] == "x" else (y, x)              # yy = phase coordinate, xx = readout coordinate
        out = np.zeros((k_pe, k_ro), complex)
        amps = [eng.amplitudes(rp, c)[0] for c in range(len(w["names"]))]
        for ip in range(k_pe):
            for iq in range(k_ro):
                kp, kq = (ip - k_pe // 2) / fov_pe, (iq - k_ro // 2) / fov_ro
                tau = (rp["pe_polarity"] * (ip - k_pe // 2) * rp["echo_spacing_ms"] * 1e-3 if epi
                       else rp["ro_polarity"] * (iq - k_ro // 2) / (k_ro * rp["bandwidth_hz_per_px"]))
                m_ = np.zeros((n, n), complex)
                for c_, nm in enumerate(w["names"]):
                    g = S.time_function(w["tissues"][nm], np.array([tau]), np.array([tau]), te, "gre", w["b0_t"], tissue.FAT_PEAKS)[0]
                    m_[eng.sup[c_]] += amps[c_] * g
                m_ *= w["fields"]["rx"] * np.exp(2j * np.pi * eng.df * (te * 1e-3 + tau)) * np.exp(-2j * np.pi * (kp * yy + kq * xx))
                out[ip, iq] = m_.sum()
        return out * eng.d ** 2 * rp["slice_mm"]

    worst, cases = 0.0, (
        dict(sequence="spgr", tr_ms=40, flip_deg=20, fov_mm=96, matrix=32, bandwidth_hz_per_px=60, te_ms=12),
        dict(sequence="spgr", tr_ms=40, flip_deg=20, fov_mm=96, matrix=32, bandwidth_hz_per_px=60, te_ms=12, pe_dir="x", ro_polarity=-1),
        dict(sequence="epi", tr_ms=2000, fov_mm=96, matrix=32, bandwidth_hz_per_px=1500, echo_spacing_ms=0.8, fat_sat=False),
        dict(sequence="epi", tr_ms=2000, fov_mm=96, matrix=32, bandwidth_hz_per_px=1500, echo_spacing_ms=0.8, fat_sat=False,
             pe_polarity=-1, partial_fourier=0.75))
    for proto in cases:
        rp = console.plan(proto, exam)["realised"]
        fast = eng.clean_kspace(rp)[0]
        ref = brute(rp, rp["echo_times_ms"][0])
        acq = eng._acquired(rp)
        worst = max(worst, np.linalg.norm((fast - ref)[acq]) / np.linalg.norm(ref[acq]))
    return [
        _row("K1", "encoding", "chirp-z transform", "direct evaluation of the Fourier sum", czt, 1e-12),
        _row("K2", "encoding", "k-space of the engine: 2 gradient-echo and 2 EPI protocols, 4 tissues incl. six-peak fat, +-60 Hz field",
             "brute-force sum of the time-dependent signal equation over every spin and every sample (shares the per-tissue amplitudes with the engine: checks the encoding)",
             worst, 1e-3, f"{worst:.1e} relative L2"),
    ]


def check_fat_shift():
    worst = other = 0.0
    for b0 in (1.5, 3.0):
        eng = _engine([("scalp_fat", _disc(40))], b0_t=b0, fat_model="single")
        f_fat = tissue.FAT_PPM * G * b0
        for bw in (100, 200, 400, 800):
            for pol, pe in ((1, "y"), (-1, "y"), (1, "x")):
                im = eng.scan(dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=256, matrix=128, bandwidth_hz_per_px=bw,
                                   ro_polarity=pol, pe_dir=pe))["images"][0]
                worst = max(worst, abs(_centroid_px(im, 1 if pe == "y" else 0) - (-pol * f_fat / bw)))
                other = max(other, abs(_centroid_px(im, 0 if pe == "y" else 1)))
    cf = 0.0
    for off in (40.0, -75.0):
        eng = _engine([("wm", _disc(40))], dict(cf_offset_hz=off))
        for bw in (100, 400):
            im = eng.scan(dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=256, matrix=128, bandwidth_hz_per_px=bw))["images"][0]
            cf = max(cf, abs(_centroid_px(im, 1) - (-off / bw)))
    return [
        _row("K3", "encoding", "water-fat shift along the readout, 24 cases (1.5 and 3 T, 100 to 800 Hz/px, both polarities, both readout axes)",
             "shift = -polarity x 3.4 ppm x gamma B0 / bandwidth per pixel", worst, 0.03, f"max error {worst:.3f} px"),
        _row("K4", "encoding", "the same scans along the phase-encode axis", "no shift", other, 0.01, f"{other:.3f} px"),
        _row("K5", "encoding", "centre-frequency offset of +40 and -75 Hz", "whole object shifts by -polarity x offset / bandwidth per pixel", cf, 0.03, f"max error {cf:.3f} px"),
    ]


def check_wrap():
    eng = _engine([("wm", _disc(15, cy=100.0))])
    base = dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=160, matrix=80, bandwidth_hz_per_px=300)
    im = eng.scan(base)["images"][0]
    pos = _centroid_px(im, 0) * 2.0
    e0 = np.sum(np.abs(im) ** 2)
    e_os = np.sum(np.abs(eng.scan(dict(base, phase_oversampling=0.5))["images"][0]) ** 2) / e0
    e_ro = np.sum(np.abs(eng.scan(dict(base, pe_dir="x"))["images"][0]) ** 2) / e0
    return [
        _row("K6", "encoding", "object at y = +100 mm, phase field of view 160 mm", "appears at y - FOV = -60 mm (aliasing of the phase-encode direction)", abs(pos + 60.0), 0.05, f"seen at {pos:+.2f} mm"),
        _row("K7", "encoding", "the same with 50 % phase oversampling, and with the object outside the READOUT field of view", "no wrap-around (energy of the wrapped image removed)", max(e_os, e_ro), 1e-3, f"{max(e_os, e_ro):.1e} of the energy remains"),
    ]


def check_gibbs():
    eng = _engine([("wm", _rect(60, 60))], edge=0.05)
    im = np.abs(_upsample(eng.scan(dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=256, matrix=64, bandwidth_hz_per_px=300))["images"][0]))
    col = im[:, im.shape[1] // 2]                                        # profile along the phase axis: pure truncation
    over = 100 * (col.max() / col[len(col) // 2 - 8: len(col) // 2 + 8].mean() - 1)
    return [_row("K8", "encoding", "overshoot at a sharp edge (4 mm pixels)", "Gibbs phenomenon: 8.95 % of the step", abs(over - 8.95), 0.3, f"{over:.2f} %")]


def check_t2star_blur():
    tis = dict(spot=dict(pd=1.0, t1_ms=800.0, t2_ms=80.0, t2s_ms=10.0, t2p_ms=1 / (1 / 10 - 1 / 80), adc_um2_ms=1.0, fat=False))
    ph = W.shapes_phantom([("spot", _disc(1.0))], tissues=tis)
    a_eng = kspace.Engine(W.flat_world(ph), tier=1)
    b_eng = kspace.Engine(W.flat_world(ph), tier=1, effects=kspace.ALL - {"t2star"})
    werr = ferr = 0.0
    for bw in (50, 100, 400):
        rp = console.plan(dict(sequence="spgr", tr_ms=100, flip_deg=30, fov_mm=256, matrix=128, bandwidth_hz_per_px=bw), a_eng.w["console"])["realised"]
        ka, kb = a_eng.clean_kspace(rp)[0], b_eng.clean_kspace(rp)[0]
        k_ro = ka.shape[-1]
        row, row0 = ka[ka.shape[0] // 2] / kb[kb.shape[0] // 2], kb[kb.shape[0] // 2]
        pred = np.exp(-(np.arange(k_ro) - k_ro // 2) / (k_ro * bw) / 0.010)
        werr = max(werr, np.abs(np.abs(row) / pred - 1).max())

        def width(spec):
            win = np.zeros(len(spec) * 16, complex)
            win[:len(spec)] = spec
            psf = np.abs(np.fft.fft(win))
            return np.sum(psf > psf.max() / 2) / 16
        up = np.abs(_upsample(a_eng.recon(ka, rp), 16))
        prof = up[up.shape[0] // 2]
        ferr = max(ferr, abs(np.sum(prof > prof.max() / 2) / 16 - width(row0 * pred)))
    return [
        _row("K9", "encoding", "k-space weighting along the readout of a spot with T2* 10 ms", "exp(-tau / T2*), tau = time from the echo centre", werr, 1e-9),
        _row("K10", "encoding", "width of the resulting point spread (50, 100, 400 Hz/px)", "transform of the weighted spectrum", ferr, 0.07, f"max error {ferr:.3f} px"),
    ]


def check_bssfp_bands():
    ph = W.shapes_phantom([("wm", _rect(40, 110))])
    n, d = ph["n"], ph["extent_mm"] / ph["n"]
    x = np.broadcast_to((np.arange(n) - n // 2) * d, (n, n))
    worst, bw = 0.0, 1000
    for tr, cf, cyc in ((5.0, 0.0, 180.0), (5.0, 30.0, 180.0), (8.0, 0.0, 180.0)):
        w = W.flat_world(ph, cf_offset_hz=cf)
        w["fields"]["b0_hz"] = 2.0 * x                                    # 2 Hz / mm along x
        im = np.abs(kspace.Engine(w).scan(dict(sequence="bssfp", tr_ms=tr, flip_deg=40, fov_mm=256, matrix=256,
                                              bandwidth_hz_per_px=bw, phase_cycle_deg=cyc))["images"][0])
        prof = im[128 - 10:128 + 10].mean(0)
        xs = (np.arange(256) - 128) * 1.0
        inside = np.abs(xs) < 100
        mins = [xs[i] for i in range(1, 255) if inside[i] and prof[i] < prof[i - 1] and prof[i] < prof[i + 1]
                and prof[i] < 0.5 * np.median(prof[inside])]
        pred = []
        for mth in range(-6, 7):
            f = (mth + 0.5 - (cyc - 180.0) / 360.0) * 1e3 / tr           # total off-resonance of a null
            xx = (f - cf) / 2.0 - f / bw                                 # position, with the readout displacement (1 mm pixels)
            if abs(xx) < 100:
                pred.append(xx)
        if len(mins) != len(pred):
            return [_row("K11", "encoding", "bSSFP dark bands in a 2 Hz/mm field", "nulls at off-resonance (m + 1/2) / TR", 99, 1.0, f"{len(mins)} bands found, {len(pred)} predicted")]
        worst = max(worst, max(abs(a - b) for a, b in zip(sorted(mins), sorted(pred))))
    return [_row("K11", "encoding", "position of bSSFP dark bands in a 2 Hz/mm field (TR 5 and 8 ms, with a 30 Hz offset)",
                 "nulls at off-resonance (m + 1/2) / TR", worst, 1.0, f"max error {worst:.2f} mm (1 mm pixels)")]


def check_epi():
    shift = 0.0
    for esp in (0.7, 1.0, 1.4):
        for pol in (1, -1):
            eng = _engine([("wm", _disc(40))], dict(cf_offset_hz=25.0))
            im = eng.scan(dict(sequence="epi", tr_ms=3000, fov_mm=240, matrix=64, bandwidth_hz_per_px=2500, echo_spacing_ms=esp,
                               pe_polarity=pol, fat_sat=False))["images"][0]
            shift = max(shift, abs(_centroid_px(im, 0) - (-pol * 25 * esp * 1e-3 * 64)))
    ghost = 0.0
    for delay in (0.3, 1.0):
        eng = _engine([("wm", _disc(40))], dict(grad_delay_us=delay))
        im = np.abs(eng.scan(dict(sequence="epi", tr_ms=3000, fov_mm=240, matrix=64, bandwidth_hz_per_px=2500, echo_spacing_ms=1.0,
                                  fat_sat=False))["images"][0])
        for col in (36, 40, 24):
            xx = (col - 32) * 240 / 64
            meas = np.concatenate([im[:6, col], im[-6:, col]]).sum() / im[32 - 6:32 + 6, col].sum()
            ghost = max(ghost, abs(meas - abs(np.tan(2 * np.pi * delay * 1e-6 * 64 * 2500 / 240 * xx))))
    return [
        _row("K12", "encoding", "EPI displacement along phase for a +25 Hz offset (echo spacing 0.7 to 1.4 ms, both blip polarities)",
             "shift = -polarity x offset x echo spacing x lines; Jezzard & Balaban 1995", shift, 0.03, f"max error {shift:.3f} px"),
        _row("K13", "encoding", "Nyquist ghost from a gradient delay of 0.3 and 1.0 us", "ghost / main = tan(2 pi x delay x matrix x bandwidth / FOV x position)", ghost, 2e-3, f"max error {ghost:.1e}"),
    ]


def check_interference():
    worst = 0
    for bw, pol in ((100, 1), (200, 1), (400, 1), (200, -1)):
        eng = _engine([("wm", _disc(30))], dict(rfi=dict(freq_hz=4000.0, amp=3.0)))
        im = np.abs(eng.scan(dict(sequence="se", tr_ms=500, te_ms=30, fov_mm=256, matrix=128, bandwidth_hz_per_px=bw, ro_polarity=pol),
                             np.random.default_rng(1))["images"][0])
        prof = np.concatenate([im[:20], im[-20:]]).mean(0)
        worst = max(worst, abs(int(np.argmax(prof)) - 64 - (-pol * 4000 / bw)))
    return [_row("K14", "encoding", "narrow-band interference at 4 kHz", "a line at -polarity x frequency / bandwidth per pixel from the centre", worst, 1.0, f"max error {worst:.0f} px")]


def check_noise():
    eng = _engine([("wm", _disc(40))], dict(sigma0=0.03))
    base = dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=256, matrix=128, bandwidth_hz_per_px=200)
    rng = np.random.default_rng(5)
    sd = ray = snr = 0.0
    snrs = []
    for ch in ({}, dict(matrix=256), dict(bandwidth_hz_per_px=800), dict(averages=4), dict(phase_oversampling=0.5),
               dict(partial_fourier=0.75), dict(phase_fov=0.75), dict(slice_mm=2.5), dict(fov_mm=320)):
        r = eng.scan(dict(base, **ch), rng)
        rp, im = r["plan"]["realised"], r["images"][0]
        clean = eng.scan(dict(base, **ch))["images"][0]
        noise = (im - clean).real.std()
        dx, dy = rp["fov_mm"] / rp["matrix"], rp["fov_pe_mm"] / rp["matrix_pe"]
        pred = 0.03 * np.sqrt(rp["bandwidth_hz_per_px"] * rp["lines"] / rp["averages"]) / (dx * dy * rp["lines_full"])
        sd = max(sd, abs(noise / pred - 1))
        bg = np.abs(im)[np.abs(clean) < 1e-3 * np.abs(clean).max()]
        ray = max(ray, abs(bg.mean() / noise - 1.2533), abs(bg.std() / noise - 0.6551))
        yy, xx = np.meshgrid((np.arange(rp["matrix_pe"]) - rp["matrix_pe"] // 2) * dy, (np.arange(rp["matrix"]) - rp["matrix"] // 2) * dx, indexing="ij")
        snrs.append(np.abs(clean)[np.hypot(yy, xx) < 25].mean() / noise / r["plan"]["derived"]["relative_snr"])
    snr = max(snrs) / min(snrs) - 1
    return [
        _row("K15", "encoding", "noise SD of the image over 9 protocols (matrix, bandwidth, averages, phase oversampling, partial Fourier, rectangular FOV, slice, FOV)",
             "sigma = sigma0 sqrt(bandwidth x lines / averages) / (dx dy lines_full)", sd, 0.02, f"max deviation {100 * sd:.1f} %"),
        _row("K16", "encoding", "magnitude of the background", "Rayleigh: mean 1.2533 sigma, SD 0.6551 sigma; Gudbjartsson & Patz 1995", ray, 0.02, f"max deviation {ray:.3f} sigma"),
        _row("K17", "encoding", "measured SNR / relative SNR printed in the header, over the same 9 protocols", "constant (voxel volume x sqrt(sampling time))", snr, 0.03, f"spread {100 * snr:.1f} %"),
    ]


def check_in_opposed():
    worst = 0.0
    for b0 in (1.5, 3.0):
        ph = W.shapes_phantom([("liver", _disc(40)), ("subcut_fat", _disc(40))], b0_t=b0)
        ph["weights"][:] = 0.5 * ph["weights"].sum(0)[None]
        w = W.flat_world(ph)
        eng = kspace.Engine(w, tier=1, fat_model="single")
        f = tissue.FAT_PPM * G * b0
        tw, tf = w["tissues"]["liver"], w["tissues"]["subcut_fat"]
        yy, xx = np.meshgrid(np.arange(128) - 64, np.arange(128) - 64, indexing="ij")
        for k in (2, 3, 4, 5, 6):
            te = 0.5e3 / abs(f) * k
            im = np.abs(eng.scan(dict(sequence="spgr", tr_ms=150, flip_deg=60, te_ms=te, fov_mm=256, matrix=128, bandwidth_hz_per_px=1000))["images"][0])
            te = console.plan(dict(sequence="spgr", tr_ms=150, flip_deg=60, te_ms=te, fov_mm=256, matrix=128, bandwidth_hz_per_px=1000), w["console"])["realised"]["te_ms"]
            ww = 0.5 * tw["pd"] * S.spgr(D(60), 150, tw["t1_ms"]) * np.exp(-te / tw["t2s_ms"])
            ff = 0.5 * tf["pd"] * S.spgr(D(60), 150, tf["t1_ms"]) * np.exp(-te / tf["t2s_ms"])
            pred = abs(ww + ff * np.exp(2j * np.pi * f * te * 1e-3)) * 5.0
            worst = max(worst, abs(im[np.hypot(yy, xx) < 12].mean() / pred - 1))
    return [_row("K18", "encoding", "gradient echo of a 50 / 50 water-fat voxel at in- and opposed-phase echo times, 1.5 and 3 T",
                 "|W exp(-TE/T2*w) + F exp(-TE/T2*f) exp(2 pi i f TE)|, f = -3.4 ppm x gamma B0", worst, 0.01, f"max deviation {100 * worst:.2f} %")]


def check_positions():
    eng = _engine([("wm", _disc(12, cy=-37.0, cx=52.0))])
    worst = 0.0
    for prot in (dict(fov_mm=256, matrix=128), dict(fov_mm=200, matrix=128, phase_fov=0.75), dict(fov_mm=256, matrix=128, pe_dir="x")):
        r = eng.scan(dict(sequence="se", tr_ms=500, te_ms=20, bandwidth_hz_per_px=600, **prot))
        rp, im = r["plan"]["realised"], np.abs(r["images"][0])
        from .scanner.session import fov_yx, pixel_positions
        ypos, xpos = pixel_positions(im.shape, fov_yx(rp))
        m = np.where(im > 0.1 * im.max(), im, 0)
        worst = max(worst, abs((m.sum(1) * ypos).sum() / m.sum() + 37.0), abs((m.sum(0) * xpos).sum() / m.sum() - 52.0))
    return [_row("K19", "encoding", "centroid of an off-centre disc for square, rectangular and swapped geometries",
                 "world coordinates (pixel i at (i - n // 2) x FOV / n)", worst, 0.15, f"max error {worst:.2f} mm")]


# ================================================================================================== console
def check_console():
    b1 = console._b1_ut(10.0, 1.0)
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
    p = console.plan(dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=96), exam)
    g = 300 * 96 / (console.GAMMA_HZ_PER_T * 0.24) * 1e3                 # mT/m
    sar = 100 * b1 ** 2 * 1.0 / 20.0 / 3.0 ** 2
    err = max(abs(b1 / 0.6522 - 1), abs(p["limits"]["gradient_percent"] / (100 * g / 40.0) - 1), abs(p["limits"]["rf_power_percent"] - round(sar, 2)))
    lines = p["realised"]["lines"]
    t = abs(p["scan_s"] - (3.0 + math.ceil(4000 / 20) * 0.02 + 0.02 * lines))
    wfs = abs(p["derived"]["water_fat_shift_px"] - 3.4e-6 * console.GAMMA_HZ_PER_T * 3.0 / 300)
    return [
        _row("C1", "console", "hard-pulse amplitude, readout gradient and RF power of a 10 degree, 1 ms pulse at TR 20 ms",
             "B1 = alpha / (2 pi gamma tau) = 0.652 uT; G = bandwidth x matrix / (gamma FOV); B1rms^2 = B1^2 tau / TR", err, 1e-3),
        _row("C2", "console", "scan time of the same protocol", "overhead + dummy cycles x TR + TR x lines x averages", t, 1e-9, f"{p['scan_s']:.2f} s"),
        _row("C3", "console", "water-fat shift printed in the header", "3.4 ppm x gamma B0 / bandwidth per pixel", wfs, 1e-3, f"{p['derived']['water_fat_shift_px']:.3f} px"),
    ]


CHECKS = (check_closed_forms, check_ernst_angle, check_epg, check_spin_echo, check_bssfp, check_numerics, check_fat_shift,
          check_wrap, check_gibbs, check_t2star_blur, check_bssfp_bands, check_epi, check_interference, check_noise,
          check_in_opposed, check_positions, check_console)


def run(groups=None):
    """Run every check. Returns (rows, seconds)."""
    t0 = time.time()
    rows = []
    for fn in CHECKS:
        rows += fn()
    if groups:
        rows = [r for r in rows if r["group"] in groups]
    return rows, time.time() - t0


def text_table(rows):
    out = [f"{'id':4s} {'':4s} {'measured':28s} {'bound':10s} check  |  reference"]
    for r in rows:
        bound = ("<= " if r["kind"] == "max" else ">= ") + f"{r['limit']:g}"
        out.append(f"{r['id']:4s} {'ok' if r['ok'] else 'FAIL':4s} {r['measured'][:28]:28s} {bound:10s} {r['check']}  |  {r['reference']}")
    n_ok = sum(r["ok"] for r in rows)
    out.append(f"{n_ok} / {len(rows)} checks passed")
    return "\n".join(out)


def markdown_table(rows):
    out = ["| # | Check | Reference | Measured | Bound | |", "|---|---|---|---|---|---|"]
    for r in rows:
        bound = ("≤ " if r["kind"] == "max" else "≥ ") + f"{r['limit']:g}"
        out.append(f"| {r['id']} | {r['check']} | {r['reference']} | {r['measured']} | {bound} | {'pass' if r['ok'] else '**FAIL**'} |")
    return "\n".join(out)
