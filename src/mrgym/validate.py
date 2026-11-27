"""Physics validation: every behaviour of the scanner against an analytic formula or a separately written calculation.

    mrgym validate               prints the table and exits 1 if any check fails
    mrgym validate --group rf    one group only
    mrgym validate --json        the same rows as JSON
    mrgym validate --perturb ID  re-scores an environment's reference policy under physics perturbations (``perturb.py``)

Four groups:
  signal    the per-tissue signal models against (a) published closed forms and (b) Bloch calculations in ``bloch.py``
            that share no code with them (matrix steady state, isochromat ensemble, time-stepping EPG)
  rf        slice-selective excitation: the pulse's rotation against a separately written 3 x 3 cascade and the
            small-tip Fourier relation, the integral through the slice against fine quadrature and against an
            isochromat ensemble, the size of the published variable-flip-angle bias
  encoding  the k-space engine against a brute-force evaluation of the time-dependent signal equation, each
            textbook image-formation effect against its formula, the movable and turned field of view, the
            periodic-motion ghost
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

from .scanner import anatomy, bloch, console, epg, kspace, rf, tissue
from .scanner import signal as S
from .scanner import world as W

D = np.deg2rad
G = S.GAMMA_MHZ_PER_T
GROUPS = dict(signal="Signal models", rf="Slice-selective excitation", encoding="Image formation in k-space", console="Console arithmetic")
HARD = dict(excitation="nonselective")      # checks of a formula for one flip angle per voxel ask for the hard pulse


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


def check_bssfp_phase_cycle():
    """The RF phase advance of ``bssfp`` against a pulse-by-pulse simulation in which the phase turns the axis of every
    pulse (``bloch.slice_bssfp_isochromats``): the sign of the phase cycle, and what the served slice integral leaves
    out (the balanced pulse of a column away from the slice centre is not a rotation about the pulse axis)."""
    rng = np.random.default_rng(3)
    psi = D(np.array([-170.0, -120.0, -75.0, -30.0, 0.0, 20.0, 65.0, 110.0, 155.0]))
    worst = 0.0
    for cyc in (0.0, 37.0, 90.0, 180.0, 270.0, 301.0):
        t1, tr, al = rng.uniform(300, 2500), rng.uniform(3, 10), D(rng.uniform(15, 80))
        t2 = rng.uniform(30, min(t1, 400))
        ref = bloch.slice_bssfp_isochromats(al, None, tr, t1, t2, psi, cyc)
        worst = max(worst, float(np.abs(S.bssfp(al, psi - D(cyc), tr, t1, t2) - ref).max()))
    rows = [_row("S18", "signal", "balanced SSFP with an RF phase advance of 0, 37, 90, 180, 270 and 301 degrees per TR, 9 off-resonances each",
                 "pulse-by-pulse Bloch time stepping, the phase turning the axis of every pulse: the spin is ahead of the pulse by "
                 "2 pi df TR - phase cycle", worst, 1e-6)]
    ph = W.shapes_phantom([("wm", _rect(40, 110))])
    n, d = ph["n"], ph["extent_mm"] / ph["n"]
    w = W.flat_world(ph)
    w["fields"]["b0_hz"] = 2.0 * np.broadcast_to((np.arange(n) - n // 2) * d, (n, n))
    eng = kspace.Engine(w)
    cls, sup = w["tissues"]["wm"], eng.sup[0]
    row = np.flatnonzero((w["weights"][0][sup] > 0.999) & (np.abs(eng.b1[sup] - 1) < 1e-12))
    inside, band = 0.0, 0.0
    for pulse, flip, tr, cyc in (("normal", 40.0, 6.0, 180.0), ("fast", 60.0, 5.0, 90.0), ("sharp", 50.0, 8.0, 300.0)):
        rp = console.plan(dict(sequence="bssfp", tr_ms=tr, flip_deg=flip, matrix=64, rf_pulse=pulse, phase_cycle_deg=cyc), w["console"])["realised"]
        free = 2 * np.pi * eng.df[sup][row] * rp["tr_ms"] * 1e-3
        _, pick = np.unique(np.round(np.rad2deg(free - D(cyc)) % 360.0 / 15.0), return_index=True)     # one voxel per 15 degrees of the band
        idx = row[pick]
        served = eng.amplitudes(rp, 0)[0][idx] / (w["weights"][0][sup][idx] * cls["pd"])
        ref = bloch.slice_bssfp_isochromats(D(flip), rp["rf_tbw"], rp["tr_ms"], cls["t1_ms"], cls["t2_ms"], free[pick], cyc) * np.exp(0.5j * free[pick])
        from_null = np.abs((np.rad2deg(free[pick]) - cyc + 180.0) % 360.0 - 180.0)                     # degrees per TR from the middle of the dark band
        top = np.abs(ref).max()
        inside = max(inside, float((np.abs(served - ref) / top)[from_null >= 90.0].max()))
        band = max(band, float((np.abs(served - ref) / top)[from_null < 90.0].max()))
    rows.append(_row("R15", "rf", "`bssfp` of a slice (normal 40 deg, fast 60 deg, sharp 50 deg; phase cycles 180, 90, 300 degrees), in the half of the "
                     "period around the middle of the pass band, relative to the largest signal",
                     "isochromats through the slice, every column turned by its whole balanced pulse (prephasing lobe, 128 sub-pulses, rephasing "
                     "lobe), pulse by pulse", inside, 0.01,
                     f"{100 * inside:.2f} % / in the band {100 * band:.1f} %"))
    return rows


def check_node_interpolation():
    """The engine evaluates the EPG on a few Chebyshev nodes of the flip angle per tissue and interpolates."""
    w = W.make_world("5b" * 16, dict(kind="qa", b1_shape_amp=(0.2, 0.24)))
    eng = kspace.Engine(w)
    worst = 0.0
    for prot in (dict(sequence="spgr", tr_ms=8.0, flip_deg=30.0, spoiler_mt_ms_per_m=60.0, matrix=64),
                 dict(sequence="afi", tr_ms=20.0, flip_deg=60.0, matrix=64)):
        rp = console.plan(dict(prot, **HARD), w["console"])["realised"]
        for c, name in enumerate(w["names"]):
            cls, sup = w["tissues"][name], eng.sup[c]
            idx = np.flatnonzero(w["weights"][c][sup] > 0.5)[::997][:12]
            a = np.deg2rad(rp["flip_deg"]) * eng.b1[sup][idx]
            bd = kspace.diffusion_bd(cls["adc_um2_ms"], rp["spoiler_mt_ms_per_m"], rp["tr_ms"])
            pattern = (1,) if rp["sequence"] == "spgr" else (1, 1) + (0,) * (rp["tr_ratio"] - 1)
            direct = np.array([epg.steady_state(float(x), rp["tr_ms"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"], pattern, bd) for x in a]).T
            fast = [v[idx] / (w["weights"][c][sup][idx] * cls["pd"]) for v in eng.amplitudes(rp, c)]
            worst = max(worst, max(float(np.max(np.abs(f - d)) / np.max(np.abs(d))) for f, d in zip(fast, direct)))
    return [_row("S15", "signal", "per-tissue EPG interpolated over the flip angle (5 to 10 Chebyshev nodes), strong B1 pattern, hard pulse",
                 "the EPG solved at the flip angle of every sampled voxel", worst, 1e-4, f"{worst:.1e} relative")]


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
    for tr, cf, cyc in ((5.0, 0.0, 180.0), (5.0, 30.0, 180.0), (8.0, 0.0, 180.0), (5.0, 0.0, 90.0), (8.0, 0.0, 300.0)):
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
            f = (mth + cyc / 360.0) * 1e3 / tr                           # total off-resonance of a null: the spin keeps step with the pulse phase
            xx = (f - cf) / 2.0 - f / bw                                 # position, with the readout displacement (1 mm pixels)
            if abs(xx) < 100:
                pred.append(xx)
        if len(mins) != len(pred):
            return [_row("K11", "encoding", "bSSFP dark bands in a 2 Hz/mm field, phase cycles 180, 90 and 300 degrees",
                         "nulls at off-resonance (m + phase cycle / 360 degrees) / TR", 99, 1.0, f"{len(mins)} bands found, {len(pred)} predicted")]
        worst = max(worst, max(abs(a - b) for a, b in zip(sorted(mins), sorted(pred))))
    return [_row("K11", "encoding", "position of bSSFP dark bands in a 2 Hz/mm field (TR 5 and 8 ms, with a 30 Hz offset; phase cycles 180, "
                 "90 and 300 degrees)", "nulls at off-resonance (m + phase cycle / 360 degrees) / TR", worst, 1.0,
                 f"max error {worst:.2f} mm (1 mm pixels)")]


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
        dx, dy = rp["fov_mm"] / rp["matrix"], rp["fov_pe_mm"] / rp["matrix_pe"]
        yy, xx = np.meshgrid((np.arange(rp["matrix_pe"]) - rp["matrix_pe"] // 2) * dy, (np.arange(rp["matrix"]) - rp["matrix"] // 2) * dx, indexing="ij")
        partial = rp["lines"] < rp["lines_full"]
        if partial:
            # homodyne detection keeps the component in phase with the signal: its noise is read inside the object (real, positive),
            # and the lines that stand in for their conjugates weigh 2: 2 n + 1 lines of weight 1, lines_full / 2 - 1 - n of weight 2
            n = rp["lines"] - rp["lines_full"] // 2
            lines = (2 * n + 1) + 4 * (rp["lines_full"] // 2 - 1 - n)
            inside = np.hypot(yy, xx) < 32
            noise = float(np.sqrt(np.mean([(eng.scan(dict(base, **ch), rng)["images"][0] - clean).real[inside].var() for _ in range(8)])))
        else:
            lines, noise = rp["lines"], (im - clean).real.std()
        pred = 0.03 * np.sqrt(rp["bandwidth_hz_per_px"] * lines / rp["averages"]) / (dx * dy * rp["lines_full"])
        sd = max(sd, abs(noise / pred - 1))
        if not partial:                                                  # the background of a homodyne image is not Rayleigh distributed
            bg = np.abs(im)[np.abs(clean) < 1e-3 * np.abs(clean).max()]
            ray = max(ray, abs(bg.mean() / noise - 1.2533), abs(bg.std() / noise - 0.6551))
        snrs.append(np.abs(clean)[np.hypot(yy, xx) < 25].mean() / noise / r["plan"]["derived"]["relative_snr"])
    snr = max(snrs) / min(snrs) - 1
    # partial Fourier: the image of a real object is the fully sampled one; the header's relative SNR falls
    full = eng.scan(dict(base, matrix=64))
    pf, hdr = 0.0, []
    for frac in (0.55, 0.65, 0.75, 0.9):
        part = eng.scan(dict(base, matrix=64, partial_fourier=frac))
        pf = max(pf, float(np.abs(part["images"][0] - full["images"][0]).max() / np.abs(full["images"][0]).max()))
        n = part["plan"]["realised"]["lines"] - 32
        hdr.append(abs(part["plan"]["derived"]["relative_snr"] / full["plan"]["derived"]["relative_snr"] / math.sqrt(64 / ((2 * n + 1) + 4 * (31 - n))) - 1))
    # the background of a homodyne image: one Gaussian component, not two (noise 10 x higher, so that nothing of the object reaches it)
    loud, half = _engine([("wm", _disc(40))], dict(sigma0=0.3)), []
    for matrix, frac in itertools.product((64, 128), (0.55, 0.65, 0.75)):
        rp = console.plan(dict(base, matrix=matrix, partial_fourier=frac), loud.w["console"])["realised"]
        d = rp["fov_mm"] / matrix
        yy, xx = np.meshgrid((np.arange(matrix) - matrix // 2) * d, (np.arange(matrix) - matrix // 2) * d, indexing="ij")
        far = np.hypot(yy, xx) >= 75
        half.append(float(np.mean([np.abs(loud.acquire(rp, rng)[0])[far].mean() for _ in range(8)]) / loud.noise_sd(rp)))
    one = max(abs(v / math.sqrt(2 / math.pi) - 1) for v in half)
    return [
        _row("K37", "encoding", "mean magnitude of the background of a partial Fourier image over the noise SD of a pixel, fractions 0.55 to 0.75, 64 and 128 lines",
             "sqrt(2 / pi) = 0.798, the mean of the magnitude of one Gaussian component (the reconstruction keeps the component in phase with a "
             "low-resolution image); a fully sampled image has the Rayleigh mean 1.2533 (K16)", one, 0.05, f"{min(half):.3f} to {max(half):.3f}"),
        _row("K15", "encoding", "noise SD of the image over 9 protocols (matrix, bandwidth, averages, phase oversampling, partial Fourier, rectangular FOV, slice, FOV)",
             "sigma = sigma0 sqrt(bandwidth x lines / averages) / (dx dy lines_full); partial Fourier: lines = the sum of the squared homodyne weights, "
             "noise of the component in phase with the signal", sd, 0.02, f"max deviation {100 * sd:.1f} %"),
        _row("K16", "encoding", "magnitude of the background (the 8 fully sampled protocols)", "Rayleigh: mean 1.2533 sigma, SD 0.6551 sigma; Gudbjartsson & Patz 1995", ray, 0.02,
             f"max deviation {ray:.3f} sigma"),
        _row("K17", "encoding", "measured SNR / relative SNR printed in the header, over the same 9 protocols", "constant (voxel volume x sqrt(sampling time))", snr, 0.03, f"spread {100 * snr:.1f} %"),
        _row("K29", "encoding", "partial Fourier 0.55 to 0.9 of a real object, reconstructed by homodyne detection",
             "the fully sampled image: conjugate synthesis is exact for a real object (one unpaired line at the edge of k-space is not acquired)",
             pf, 0.03, f"differs by at most {100 * pf:.1f} % of the maximum"),
        _row("K30", "encoding", "relative SNR in the header of the same four partial Fourier scans over that of the full scan",
             "sqrt(lines_full / sum of the squared line weights), weights 1 on the symmetric lines and 2 on the others: Noll, Nishimura & Macovski 1991",
             max(hdr), 1e-3),
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
            prot = dict(sequence="spgr", tr_ms=150, flip_deg=60, te_ms=te, fov_mm=256, matrix=128, bandwidth_hz_per_px=1000, **HARD)
            im = np.abs(eng.scan(prot)["images"][0])
            te = console.plan(prot, w["console"])["realised"]["te_ms"]
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


def check_field_map():
    ph = W.shapes_phantom([("wm", _disc(60))])
    w = W.flat_world(ph, cf_offset_hz=12.0)
    n, d = ph["n"], ph["extent_mm"] / ph["n"]
    x = np.broadcast_to((np.arange(n) - n // 2) * d, (n, n))
    w["fields"]["b0_hz"] = 0.4 * x                                        # Hz, rising with x
    r = kspace.Engine(w, tier=1).scan(dict(sequence="spgr", tr_ms=60.0, flip_deg=20.0, n_echoes=2, echo_spacing_ms=5.0, matrix=128,
                                           fov_mm=256.0, bandwidth_hz_per_px=600.0))
    te = r["plan"]["realised"]["echo_times_ms"]
    fmap = np.angle(r["images"][1] * np.conj(r["images"][0])) / (2 * np.pi * (te[1] - te[0]) * 1e-3)
    xs = (np.arange(128) - 64) * 2.0
    sel = np.abs(xs) < 40
    err = np.abs(fmap[64, sel] - (12.0 + 0.4 * xs[sel])).max()
    return [_row("K20", "encoding", "field map from the phase of two gradient echoes (12 Hz offset plus 0.4 Hz/mm)",
                 "phase = +2 pi x off-resonance x TE: faster spins gain positive phase", err, 0.5, f"max error {err:.2f} Hz")]


# ================================================================================================== console
def check_console():
    b1 = console._b1_ut(10.0, 1.0)
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
    p = console.plan(dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=96, **HARD), exam)
    g = 300 * 96 / (console.GAMMA_HZ_PER_T * 0.24) * 1e3                 # mT/m
    sar = 100 * b1 ** 2 * 1.0 / 20.0 / 3.0 ** 2
    err = max(abs(b1 / 0.6522 - 1), abs(p["limits"]["gradient_percent"] / (100 * g / 40.0) - 1), abs(p["limits"]["rf_power_percent"] - round(sar, 2)))
    p = console.plan(dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=96), exam)
    lines = p["realised"]["lines"]
    t = abs(p["scan_s"] - (3.0 + math.ceil(4000 / 20) * 0.02 + 0.02 * lines))
    wfs = abs(p["derived"]["water_fat_shift_px"] - 3.4e-6 * console.GAMMA_HZ_PER_T * 3.0 / 300)
    return [
        _row("C1", "console", "hard-pulse amplitude, readout gradient and RF power of a 10 degree, 1 ms hard pulse (nonselective) at TR 20 ms",
             "B1 = alpha / (2 pi gamma tau) = 0.652 uT; G = bandwidth x matrix / (gamma FOV); B1rms^2 = B1^2 tau / TR", err, 1e-3),
        _row("C2", "console", "scan time of the same protocol", "overhead + dummy cycles x TR + TR x lines x averages", t, 1e-9, f"{p['scan_s']:.2f} s"),
        _row("C3", "console", "water-fat shift printed in the header", "3.4 ppm x gamma B0 / bandwidth per pixel", wfs, 1e-3, f"{p['derived']['water_fat_shift_px']:.3f} px"),
    ]


# ================================================================================================== slice selection
def _fine_profile(alpha, tbw, n=16001, zmax=12.0, window="hann"):
    z = np.linspace(-zmax, zmax, n)
    w = np.full(n, 2 * zmax / (n - 1))
    w[[0, -1]] /= 2
    theta, ph = rf.excitation(alpha, tbw, z, window)
    return theta, ph, w


def check_rf_pulse():
    z = np.linspace(-2.5, 2.5, 1001)
    rot = tip = 0.0
    for tbw, window in ((2, "hann"), (4, "hann"), (8, "hann"), (4, "hamming")):
        for deg in (10, 30, 60, 90, 120, 180):
            a, b = rf.cayley_klein(D(deg), tbw, z, window=window)
            th3, mxy3 = bloch.slice_profile(D(deg), tbw, z, window=window)
            # the cascade takes Mz to +My at the centre, the spinor convention to -i: opposite sign of Mxy
            rot = max(rot, np.abs(2 * np.arcsin(np.clip(np.abs(b), 0, 1)) - th3).max(), np.abs(2 * np.conj(a) * b + mxy3).max())
        t = (np.arange(rf.N_SAMPLES) + 0.5) / rf.N_SAMPLES - 0.5
        ft = np.abs(np.exp(-2j * np.pi * tbw * bloch.calibration(tbw, window=window) * np.outer(z, t)) @ rf.sinc_pulse(tbw, window=window))
        tip = max(tip, np.abs(rf.excitation(D(1.0), tbw, z, window)[0] / D(1.0) - ft).max())
    # the calibration: full width at half maximum of the 1 degree profile, read from the profile itself
    wide, cal, zz = 0.0, 0.0, np.linspace(0.0, 1.0, 20001)
    for tbw, window in ((2, "hann"), (4, "hann"), (8, "hann"), (2, "hamming"), (4, "hamming"), (8, "hamming")):
        m = np.sin(rf.excitation(D(1.0), tbw, zz, window)[0])
        i = int(np.argmax(m < 0.5 * m[0]))
        half = zz[i - 1] + (m[i - 1] - 0.5 * m[0]) / (m[i - 1] - m[i]) * (zz[i] - zz[i - 1])
        wide, cal = max(wide, abs(2 * half - 1)), max(cal, abs(rf.fwhm_factor(tbw, window) / bloch.calibration(tbw, window=window) - 1))
    sym = 0.0
    for tbw in (2, 4, 8):
        zz = np.linspace(-3, 3, 601)
        th, ph = rf.excitation(D(90.0), tbw, zz)
        b = rf.refocusing(D(180.0), tbw, zz)
        sym = max(sym, np.abs(th - th[::-1]).max(), np.abs((ph - np.conj(ph[::-1]))[th > 1e-6]).max(), np.abs((b * b).imag).max())
    return [
        _row("R1", "rf", "rotation of the windowed-sinc pulse (spinor cascade), 10 to 180 degrees, time-bandwidth 2 / 4 / 8, Hann and Hamming",
             "separately written cascade of real 3 x 3 rotations (bloch.py): flip angle (rad) and transverse magnetisation", rot, 1e-7),
        _row("R2", "rf", "slice profile of a 1 degree pulse", "small-tip approximation: flip angle = alpha x abs(Fourier transform of the pulse)",
             tip, 1e-4, f"{tip:.1e} of alpha"),
        _row("R3", "rf", "symmetry used by the slice integral: flip angle even in z, transverse phase conjugate, echo factor -b^2 real",
             "time-symmetric pulse under a constant gradient", sym, 1e-9),
        _row("R13", "rf", "full width at half maximum of the 1 degree profile, time-bandwidth 2 / 4 / 8, Hann and Hamming: the slice gradient is calibrated per pulse",
             "the prescribed slice thickness; the calibration factor against the one found on the separately written cascade (bloch.py)",
             max(wide, cal), 1e-4, f"width off by {wide:.1e} thicknesses; factors agree to {cal:.1e}"),
    ]


def check_slice_integral():
    """The quadrature the engine serves (``rf.quadrature``) against 16,001 positions on +- 12 slice thicknesses."""
    usual = extreme = 0.0
    for tbw in (2, 4, 8):
        zeta, wz = rf.quadrature(tbw)
        for flip in (2, 5, 10, 20, 40, 60, 90):
            th_f, ph_f, w_f = _fine_profile(D(flip), tbw)
            th, ph = rf.excitation(D(flip), tbw, zeta)
            for tr, t1 in itertools.product((5, 20, 100, 2000), (300, 1000, 2500, 4300)):
                ref = np.sum(w_f * ph_f * S.spgr(th_f, tr, t1))
                err = abs(np.sum(wz * ph.real * S.spgr(th, tr, t1)) - ref) / abs(ref)
                if tr >= 20:
                    usual = max(usual, err)
                else:
                    extreme = max(extreme, err)
    return [
        _row("R4", "rf", "integral of the spoiled-GRE signal through the slice as served (6 x tbw points per nominal bandwidth of the slice gradient, to 0.5 + 12 / tbw of them), "
             "252 cases: TR 20 to 2000 ms, T1 300 to 4300 ms, 2 to 90 degrees", "the same integral on 16,001 positions over +- 12 thicknesses",
             usual, 2e-4, f"{usual:.1e} relative"),
        _row("R5", "rf", "the same at TR 5 ms (84 cases), where the wings of the profile pass the Ernst angle and outweigh the slice centre",
             "the same reference", extreme, 1e-3, f"{extreme:.1e} relative"),
    ]


def check_slice_models():
    # R6: the hard pulse of the 3D mode is the engine of version 0.1.0 (numbers recorded on mrgym-engine-0.1.0)
    w = W.make_world("000102030405060708090a0b0c0d0e0f", dict(kind="qa"))
    eng = kspace.Engine(w)
    m = np.abs(eng.scan(dict(sequence="se", tr_ms=500, matrix=64, **HARD))["images"])
    old = max(abs(float(m.mean()) / 0.7259219837862718 - 1), abs(float(m.max()) / 2.778571927919207 - 1), abs(float(m[0, 32, 32]) / 2.3098207026200184 - 1))
    # R7: RF-spoiled gradient echo of a slice against the isochromat ensemble
    iso = 0.0
    for flip, tbw, tr, t1, t2 in ((20, 4, 20.0, 600.0, 60.0), (40, 2, 15.0, 400.0, 40.0), (10, 8, 20.0, 1000.0, 60.0), (60, 4, 12.0, 800.0, 50.0)):
        rp = dict(sequence="spgr", tr_ms=tr, rf_phase_increment_deg=117.0, spoiler_mt_ms_per_m=100.0)
        got = kspace.slice_signal(rp, dict(t1_ms=t1, t2_ms=t2, adc_um2_ms=0.0), D(flip), tbw)[0][0]
        zmax = 0.5 + rf.WING / tbw
        ref = bloch.slice_spgr_isochromats(D(flip), tbw, tr, t1, t2, n_z=int(round(2 * zmax * 12 * tbw)) + 1, z_max=zmax, n_iso=192)
        iso = max(iso, abs(got - ref) / abs(ref))
    # R8: the EPG through the profile is interpolated (difference to the ideal-spoiling form on 16 nodes)
    itp = 0.0
    for flip, tbw, tr, t1, t2, adc, sp in ((20, 4, 20.0, 1000, 80, 1.0, 100.0), (60, 4, 8.0, 2500, 200, 2.0, 100.0), (90, 2, 10.0, 4300, 600, 3.0, 100.0),
                                           (10, 8, 5.0, 838, 75, 0.75, 60.0), (40, 8, 30.0, 300, 40, 2.2, 300.0)):
        rp = dict(sequence="spgr", tr_ms=tr, rf_phase_increment_deg=117.0, spoiler_mt_ms_per_m=sp)
        cls = dict(t1_ms=t1, t2_ms=t2, adc_um2_ms=adc)
        zeta, wz = rf.quadrature(tbw)
        th, ph = rf.excitation(D(flip), tbw, zeta)
        bd = kspace.diffusion_bd(adc, sp, tr)
        direct = np.sum(wz * ph.real * np.array([epg.steady_state(float(t), tr, t1, t2, 117.0, (1,), bd)[0] if t > 1e-9 else 0j for t in th]))
        itp = max(itp, abs(kspace.slice_signal(rp, cls, D(flip), tbw)[0][0] - direct) / abs(direct))
    # R9: spin echo and inversion recovery of a slice against isochromats over (position, crusher phase)
    se = 0.0
    for scale, tbw, tr, te, t1, ti, kap in ((1.0, 4, 500.0, 20.0, 800.0, None, 1.0), (0.8, 4, 300.0, 15.0, 1200.0, None, 1.0), (1.15, 2, 600.0, 20.0, 500.0, None, 1.0),
                                            (1.0, 8, 400.0, 20.0, 900.0, None, 1.0), (0.9, 4, 2000.0, 20.0, 1000.0, 400.0, 0.93)):
        rp = dict(sequence="se" if ti is None else "ir", tr_ms=tr, te_ms=te, ti_ms=ti)
        got = kspace.slice_signal(rp, dict(t1_ms=t1), scale * np.pi / 2, tbw, inv_eff=kap)[0][0]
        ref = bloch.slice_se_isochromats(scale * np.pi / 2, tbw, tr, te, t1, ti_ms=ti, inv_eff=kap)
        se = max(se, abs(got - ref) / abs(ref))
    # R14: the same two with a saturation pulse before every excitation
    sat = 0.0
    for flip, tbw, tr, t1, t2, q, delay in ((20, 4, 20.0, 380.0, 60.0, 0.1, 7.0), (60, 2, 15.0, 400.0, 40.0, 0.5, 6.5), (10, 8, 30.0, 300.0, 60.0, 0.03, 8.0)):
        rp = dict(sequence="spgr", tr_ms=tr, rf_phase_increment_deg=117.0, spoiler_mt_ms_per_m=100.0)
        got = kspace.slice_signal(rp, dict(t1_ms=t1, t2_ms=t2, adc_um2_ms=0.0), D(flip), tbw, sat=(q, delay))[0][0]
        zmax = 0.5 + rf.WING / tbw
        ref = bloch.slice_spgr_isochromats(D(flip), tbw, tr, t1, t2, n_z=int(round(2 * zmax * 12 * tbw)) + 1, z_max=zmax, n_iso=192, sat=(q, delay))
        sat = max(sat, abs(got - ref) / abs(ref))
    for scale, tbw, tr, te, t1, ti, kap, q, delay in ((1.0, 4, 500.0, 20.0, 380.0, None, 1.0, 0.05, 7.0), (0.85, 2, 2000.0, 15.0, 300.0, None, 1.0, 0.5, 6.5),
                                                      (1.1, 4, 2000.0, 20.0, 400.0, 250.0, 0.93, 0.1, 7.0)):
        rp = dict(sequence="se" if ti is None else "ir", tr_ms=tr, te_ms=te, ti_ms=ti)
        got = kspace.slice_signal(rp, dict(t1_ms=t1), scale * np.pi / 2, tbw, inv_eff=kap, sat=(q, delay))[0][0]
        ref = bloch.slice_se_isochromats(scale * np.pi / 2, tbw, tr, te, t1, ti_ms=ti, inv_eff=kap, sat=(q, delay))
        sat = max(sat, abs(got - ref) / abs(ref))
    return [
        _row("R14", "rf", "the slice integrals of R7 and R9 with a saturation pulse before every excitation (3 gradient-echo, 2 spin-echo, 1 inversion-recovery case; "
             "9 to 97 % saturated)", "the same isochromat ensembles with Mz multiplied by q at that time in every repetition", sat, 1e-3, f"{sat:.1e} relative"),
        _row("R6", "rf", "excitation = nonselective (hard pulse, one flip angle per voxel): spin-echo image of the reference world",
             "mean, maximum and centre pixel recorded on mrgym-engine-0.1.0", old, 1e-9, f"{old:.1e} relative"),
        _row("R7", "rf", "RF-spoiled gradient echo integrated over the slice (EPG per column), 4 cases, time-bandwidth 2 / 4 / 8",
             "separately written isochromat ensemble: positions through the slice x 192 spoiler phases, each turned by the 3 x 3 pulse "
             "cascade of its position, time stepping from equilibrium", iso, 1e-3, f"{iso:.1e} relative"),
        _row("R8", "rf", "the EPG through the profile, interpolated over the flip angle (difference to the ideal-spoiling form on 16 Chebyshev nodes), 5 cases with diffusion",
             "the EPG solved at the flip angle of every position", itp, 5e-4, f"{itp:.1e} relative"),
        _row("R9", "rf", "spin echo and inversion recovery integrated over the slice (refocusing slice 1.5 x the excitation slice), B1 80 to 115 % of nominal",
             "separately written isochromat ensemble: positions x crusher phases, 3 x 3 cascades of both pulses, time stepping", se, 1e-3, f"{se:.1e} relative"),
    ]


def check_slice_engine():
    """The engine's use of the profile: interpolation over the object's B1 range, through-slice dephasing, the bias."""
    w = W.make_world("5b" * 16, dict(kind="qa", b1_shape_amp=(0.2, 0.24)))
    eng = kspace.Engine(w)
    worst = 0.0
    for prot in (dict(sequence="spgr", tr_ms=12.0, flip_deg=30.0, matrix=64), dict(sequence="se", tr_ms=400.0, matrix=64),
                 dict(sequence="bssfp", tr_ms=8.0, flip_deg=50.0, matrix=64)):
        rp = console.plan(prot, w["console"])["realised"]
        tbw = rp["rf_tbw"]
        zeta, wz = rf.quadrature(tbw)
        for c, name in enumerate(w["names"]):
            cls, sup = w["tissues"][name], eng.sup[c]
            idx = np.flatnonzero(w["weights"][c][sup] > 0.5)[::1499][:8]
            a = D(rp["flip_deg"]) * eng.b1[sup][idx]
            fast = eng.amplitudes(rp, c)[0][idx] / (w["weights"][c][sup][idx] * cls["pd"])
            if rp["sequence"] == "bssfp":
                free = 2 * np.pi * eng.df[sup][idx] * rp["tr_ms"] * 1e-3
                direct = []
                for ai, fi in zip(a, free):
                    th, ph = rf.excitation(ai, tbw, zeta)
                    direct.append(np.sum(wz * ph.real * S.bssfp(th, fi - D(rp["phase_cycle_deg"]), rp["tr_ms"], cls["t1_ms"], cls["t2_ms"])) * np.exp(0.5j * fi))
                direct = np.array(direct)
            else:
                direct = kspace.slice_signal(rp, cls, a, tbw, inv_eff=w["system"]["inv_eff"])[0]
            worst = max(worst, float(np.max(np.abs(fast - direct)) / np.max(np.abs(direct))))
    # through-slice dephasing: numeric integral of a small-tip profile against the closed form, and the engine's image
    deph = 0.0
    for tbw in (2, 4, 8):
        th, ph, wf = _fine_profile(D(2.0), tbw)
        z = np.linspace(-12.0, 12.0, 16001)
        for cyc in (0.25, 0.5, 1.0, 1.7):
            num = abs(np.sum(wf * th * ph * np.exp(2j * np.pi * cyc * z))) / abs(np.sum(wf * th * ph))
            deph = max(deph, abs(num - abs(float(rf.dephasing_factor(cyc, tbw)))))
    ph_ = W.shapes_phantom([("wm", _disc(40))])
    img = []
    for dbdz in (0.0, 5.0):                                              # Hz / mm; 5 mm slice, TE 20 ms: 0.5 cycles
        wd = W.flat_world(ph_)
        wd["fields"]["dbdz_hz_per_mm"] = np.full((ph_["n"], ph_["n"]), dbdz)
        img.append(np.abs(kspace.Engine(wd).scan(dict(sequence="spgr", tr_ms=100, flip_deg=10, te_ms=20, matrix=64, fov_mm=256))["images"][0])[32, 32])
    drop = abs(img[1] / img[0] - float(rf.dephasing_factor(0.5, 4.0)))
    # two-angle variable-flip-angle T1 with the nominal angles: the published 2D bias
    bias = []
    for tbw in (2, 4):
        for t1, (f1, f2) in itertools.product((300.0, 1000.0, 2500.0), ((4, 20), (3, 17), (6, 35))):
            a = D(np.array([f1, f2]))
            sig = np.abs(kspace.slice_signal(dict(sequence="spgr", tr_ms=20.0), dict(t1_ms=t1), a, tbw, tier=1)[0])
            y, x = sig / np.sin(a), sig / np.tan(a)
            bias.append(-20.0 / np.log((y[1] - y[0]) / (x[1] - x[0])) / t1 - 1)
    lo, hi = min(bias), max(bias)
    outside = max(0.0, -0.80 - lo, hi + 0.35)
    return [
        _row("R10", "rf", "slice-integrated signal interpolated over the flip angles of the object (5 to 16 Chebyshev nodes), strong B1 pattern: spgr, se, bssfp",
             "the slice integral at the flip angle and off-resonance of every sampled voxel", worst, 1e-4, f"{worst:.1e} relative"),
        _row("R11", "rf", "signal left by a field gradient through the slice (0.25 to 1.7 cycles across the thickness), small flip angle",
             "numeric integral of profile x phase ramp against sinc(c) x window(c / tbw), c = cycles / calibration factor (cycles across one nominal "
             "bandwidth of the slice gradient): the profile is the transform of the pulse",
             max(deph, drop), 2e-3, f"{max(deph, drop):.1e}; image at 0.5 cycles: {img[1] / img[0]:.3f}"),
        _row("R12", "rf", "two-angle variable-flip-angle T1 on a 2D slice, nominal angles in the fit (TR 20 ms, T1 300 to 2500 ms, time-bandwidth 2 and 4, ideal spoiling)",
             "published underestimation without profile correction: 40 to 80 % (Svedin & Parker 2017), -63 % (Dieringer 2014); band -35 to -80 %",
             outside, 0.0, f"{100 * hi:.0f} to {100 * lo:.0f} %"),
    ]


# ================================================================================================== field of view, motion
def check_field_of_view():
    eng = _engine([("wm", _disc(15, cy=30.0, cx=-40.0))])
    base = dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=160, matrix=80, bandwidth_hz_per_px=300)
    from .scanner.session import fov_yx, pixel_grid, roi_readouts, rotation_of
    im = eng.scan(dict(base, centre_y_mm=30.0, centre_x_mm=-40.0))["images"][0]
    off = max(abs(_centroid_px(im, 0)), abs(_centroid_px(im, 1))) * 2.0
    # a field of view centred 100 mm away along phase: the disc at 70 mm beyond the edge wraps to ((y - y0 + FOV/2) mod FOV) - FOV/2
    im = eng.scan(dict(base, centre_y_mm=-70.0, centre_x_mm=-40.0))["images"][0]
    wrap = abs(_centroid_px(im, 0) * 2.0 - (((30.0 + 70.0 + 80.0) % 160.0) - 80.0))
    # rotation: an ellipse turned by hand and imaged upright against the upright description imaged with a turned field of view
    turn, cy, cx = 30.0, 12.0, -18.0
    c, s = np.cos(D(turn)), np.sin(D(turn))
    upright = kspace.Engine(W.flat_world(W.shapes_phantom([("wm", lambda y, x: anatomy.ell(y, x, cy, cx, 28.0, 52.0, 10.0)), ("csf", _disc(9, -45.0, 30.0))])))
    by_hand = kspace.Engine(W.flat_world(W.shapes_phantom([
        ("wm", lambda y, x: anatomy.ell(y, x, cy * c + cx * s, -cy * s + cx * c, 28.0, 52.0, 10.0 + turn)),
        ("csf", _disc(9, -45.0 * c + 30.0 * s, 45.0 * s + 30.0 * c))])))
    prot = dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=220, matrix=96, bandwidth_hz_per_px=300, phase_fov=0.75)
    a = upright.scan(dict(prot, rotation_deg=turn, centre_y_mm=5.0, centre_x_mm=-7.0))
    b = by_hand.scan(dict(prot, centre_y_mm=5.0 * c - 7.0 * s, centre_x_mm=-5.0 * s - 7.0 * c))["images"][0]
    rot = float(np.linalg.norm(a["images"][0] - b) / np.linalg.norm(b))
    # pixel coordinates of the turned image: the centroid of the small disc, read in patient coordinates
    rp = a["plan"]["realised"]
    yy, xx = pixel_grid(a["images"].shape, fov_yx(rp), (rp["centre_y_mm"], rp["centre_x_mm"]), rotation_of(rp))
    reg = roi_readouts(a["images"], [dict(id="d", y_mm=-45.0, x_mm=30.0, radius_mm=6.0), dict(id="o", y_mm=-45.0, x_mm=-30.0, radius_mm=6.0)],
                       fov_yx(rp), (rp["centre_y_mm"], rp["centre_x_mm"]), rotation_of(rp))
    csf = np.abs(a["images"][0]) * (np.hypot(yy + 45.0, xx - 30.0) < 16)
    pos = max(abs((csf * yy).sum() / csf.sum() + 45.0), abs((csf * xx).sum() / csf.sum() - 30.0))
    return [
        _row("K21", "encoding", "disc at (30, -40) mm with the field of view centred on it", "centroid at the image centre", off, 0.05, f"{off:.3f} mm"),
        _row("K22", "encoding", "the same disc with the field of view centred 100 mm away along phase encoding (160 mm)",
             "wraps to ((y - centre + FOV / 2) mod FOV) - FOV / 2", wrap, 0.05, f"error {wrap:.3f} mm"),
        _row("K23", "encoding", "field of view turned by 30 degrees (off-centre, rectangular): an ellipse and a disc",
             "the object turned by hand and imaged upright: the turned world is re-evaluated, not interpolated", rot, 1e-6, f"{rot:.1e} relative L2"),
        _row("K24", "encoding", "pixel coordinates and region read-out of the turned image: centroid of the disc, and a region placed on it",
             "patient coordinates of the disc; the region off the object reads no signal", max(pos, 0.15 * (reg[1]["mean_magnitude"][0] / reg[0]["mean_magnitude"][0] > 0.02)),
             0.15, f"centroid error {pos:.2f} mm"),
    ]


def check_motion():
    """Periodic modulation of the k-space lines of one class: ghosts along phase encoding."""
    ph = W.shapes_phantom([("wm", _disc(60)), ("blood", _disc(4, cy=0.0, cx=20.0))])
    static = kspace.Engine(W.flat_world(ph))
    pos = amp = 0.0
    n = 128
    cases = ((0.25, 400.0, 1, 0.3), (1.1, 400.0, 1, 0.3), (1.1, 400.0, 2, 0.3), (0.3, 250.0, 1, 0.5), (0.9, 600.0, 3, 0.2), (1.4, 350.0, 4, 0.4))
    alone = kspace.Engine(W.flat_world(W.shapes_phantom([("blood", _disc(4, cy=0.0, cx=20.0))])))
    col = slice(n // 2 + 10 - 4, n // 2 + 10 + 5)                        # the vessel sits 20 mm = 10 px right of the centre
    for f, tr, nex, depth in cases:
        w = W.flat_world(ph)
        w["motion"] = [dict(cls="blood", kind="pulse", f_hz=f, depth=depth, phase_key=7)]
        prot = dict(sequence="se", tr_ms=tr, te_ms=20, fov_mm=256, matrix=n, bandwidth_hz_per_px=300, averages=nex)
        prof = np.abs(kspace.Engine(w).scan(prot)["images"][0] - static.scan(prot)["images"][0])[:, col].sum(1)
        cyc = f * tr * 1e-3 * nex
        want = (n // 2 + (cyc - np.round(cyc)) * n) % n                 # row of the copy displaced towards +phase
        k = int(np.round(want)) % n
        rows = np.arange(k - 4, k + 5)
        pos = max(pos, abs((prof[rows % n] * rows).sum() / prof[rows % n].sum() - want))
        vessel = np.abs(alone.scan(dict(prot, averages=1))["images"][0])[n // 2 - 4:n // 2 + 5, col].sum()
        factor = depth / 2 * abs(np.sin(np.pi * f * tr * 1e-3 * nex) / (nex * np.sin(np.pi * f * tr * 1e-3)))
        amp = max(amp, abs(prof[rows % n].sum() / vessel / factor - 1))
    # a mover frozen in time (f = 0) is a displaced copy: sign and unit of the shift model
    w = W.flat_world(W.shapes_phantom([("blood", _disc(8, cy=-20.0, cx=10.0))]))
    w["motion"] = [dict(cls="blood", kind="shift", f_hz=0.0, amp_mm=6.0, dir="y", phase_key=3)]
    eng = kspace.Engine(w)
    prot = dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=256, matrix=128, bandwidth_hz_per_px=300)
    rp = console.plan(prot, w["console"])["realised"]
    moved = eng.scan(prot)["images"][0]
    still = kspace.Engine(w, effects=kspace.ALL - {"motion"}).scan(prot)["images"][0]
    dy = (_centroid_px(moved, 0) - _centroid_px(still, 0)) * 2.0
    dx = (_centroid_px(moved, 1) - _centroid_px(still, 1)) * 2.0
    import hashlib
    h = hashlib.sha256(repr((3, kspace.physics_key(rp, averages=True))).encode()).digest()
    want = 6.0 * np.cos(2 * np.pi * int.from_bytes(h[:4], "big") / 2 ** 32)
    return [
        _row("K25", "encoding", "ghosts of a vessel whose signal pulses at 0.25 to 1.4 Hz (TR 250 to 600 ms, 1 to 4 averages): position along phase encoding",
             "copies at +- lines x frac(f x TR x averages) pixels (Zaitsev 2015)", pos, 0.5, f"max error {pos:.2f} px"),
        _row("K26", "encoding", "the same six scans: summed ghost amplitude over the vessel's own signal",
             "(depth / 2) x abs(sin(pi f TR averages) / (averages x sin(pi f TR))): averaging the lines of a pulsing source is a Dirichlet kernel, not 1 / sqrt(averages)",
             amp, 0.05, f"max deviation {100 * amp:.1f} %"),
        _row("K27", "encoding", "a class displaced by a constant 6 mm x cos(phase) along y (motion frozen: f = 0)",
             "the image of that class moves by the displacement along y and not along x", max(abs(dy - want), abs(dx)), 0.05, f"{dy:+.2f} mm (expected {want:+.2f})"),
    ]


def _se_saturated(q, tr, te, t1, delay):
    """Mz at the excitation of a spin echo with exact 90 and 180 degree pulses and a saturation Mz -> q Mz ``delay``
    before every excitation (written out here: the 90 degree pulse leaves no Mz, so every repetition starts from 0)."""
    e = lambda t: np.exp(-t / t1)                                               # noqa: E731
    return q * (1 - (2 - e(te / 2)) * e(tr - te / 2 - delay)) * e(delay) + 1 - e(delay)


def check_fat_saturation_offset():
    worst = 0.0
    f_fat = tissue.FAT_PPM * G * 3.0
    plain = dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=256, matrix=64, bandwidth_hz_per_px=800, **HARD)
    delay = 8.0 / 2 + 2.0 + 2.0 / 2                                              # half the saturation pulse, the gap, half the 90 degree pulse
    for sys_off, prot_off in ((0.0, 0.0), (60.0, 0.0), (0.0, 110.0), (-100.0, 0.0), (140.0, 0.0), (80.0, -80.0), (-f_fat, 0.0), (-f_fat + 200.0, -200.0)):
        for name in ("subcut_fat", "wm"):
            eng = _engine([(name, _disc(40))], dict(fatsat_eff=1.0, fatsat_bw_hz=200.0, fatsat_offset_hz=sys_off), fat_model="single")
            on = np.abs(eng.scan(dict(plain, fat_sat=True, fatsat_offset_hz=prot_off))["images"][0])[32, 32]
            off = np.abs(eng.scan(plain)["images"][0])[32, 32]
            spin = (0.0 if name == "subcut_fat" else -f_fat) - sys_off - prot_off      # the spin's frequency seen from the pulse centre
            q = 1 - 0.5 * (1 - np.tanh((abs(spin) - 100.0) / 8.0))
            t1 = eng.w["tissues"][name]["t1_ms"]
            pred = _se_saturated(q, 500.0, 20.0, t1, delay) / _se_saturated(1.0, 500.0, 20.0, t1, delay)
            worst = max(worst, abs(on / off - pred))
    # the six lines of fat, each saturated by its own place in the band (the lines are in phase at the centre of a spin echo)
    lines = 0.0
    for b0_t, shift in ((3.0, 0.0), (3.0, 60.0), (1.5, 0.0), (1.5, 3.4 * G * 1.5)):
        f0, bw = G * b0_t, 1.3 * G * b0_t
        eng = _engine([("subcut_fat", _disc(40))], dict(fatsat_eff=0.97, fatsat_bw_hz=bw), b0_t=b0_t)
        p = dict(plain, tr_ms=1500)
        mid = lambda img: abs(img[28:37, 28:37].mean())                         # noqa: E731  the lines ring differently at a single pixel
        got = mid(eng.scan(dict(p, fat_sat=True, fatsat_offset_hz=shift))["images"][0]) / mid(eng.scan(p)["images"][0])
        t1, total = eng.w["tissues"]["subcut_fat"]["t1_ms"], sum(a for _, a in tissue.FAT_PEAKS)
        pred = sum(a / total * _se_saturated(1 - 0.97 * 0.5 * (1 - np.tanh((abs((ppm - tissue.FAT_PPM) * f0 - shift) - bw / 2) / 8.0)), 1500.0, 20.0, t1, delay)
                   for ppm, a in tissue.FAT_PEAKS) / _se_saturated(1.0, 1500.0, 20.0, t1, delay)
        lines = max(lines, abs(got - pred))
    return [_row("K28", "encoding", "fat saturation whose pulse misses the fat line (hidden offset of the system, shift requested in the protocol, both): fat and water discs, "
                 "one line each, hard pulses",
                 "signal left = steady state of the spin echo with Mz x q before every excitation, q = 1 - efficiency x band(abs(spin frequency - pulse centre)); "
                 "a requested shift of the opposite size restores the suppression; a pulse moved onto the water line saturates water", worst, 1e-6),
            _row("K31", "encoding", "fat saturation of the six-line fat spectrum, 1.5 and 3 T, pulse on the main line, 60 Hz off it, and on the water line",
                 "sum over the lines of share x steady state with that line's q: a band of 1.3 ppm leaves the lines outside it (21.9 % of the signal; "
                 "the two next to water go only with water)", lines, 3e-4, f"{lines:.1e} of the unsaturated signal")]


def check_saturation():
    """A saturation pulse before every excitation: the closed forms, the RF-spoiled steady state, the engine's maps."""
    t1 = np.array([250.0, 380.0, 700.0, 1100.0, 1800.0, 4300.0])
    b1 = np.round(np.arange(0.6, 1.41, 0.2), 2)
    g = np.array(list(itertools.product(t1, b1)))
    t1, b1 = g[:, 0].copy(), g[:, 1].copy()
    closed = 0.0
    for q, delay in itertools.product((0.0, 0.03, 0.3, 0.9, 1.0), (4.5, 7.0, 12.0)):
        for flip, tr in itertools.product((3, 10, 30, 60, 90), (13, 20, 40, 150, 2000)):
            a = b1 * D(flip)
            closed = max(closed, np.abs(S.spgr(a, tr, t1, (q, delay)) - bloch.spgr_ideal(a, tr, t1, sat=(q, delay))).max())
        for tr, te in itertools.product((60, 300, 1500, 6000), (10.0, 30.0)):
            a = b1 * np.pi / 2
            closed = max(closed, np.abs(S.se(a, tr, te, t1, (q, delay)) - bloch.se_ideal(a, tr, te, t1, sat=(q, delay))).max())
            if tr >= 300:
                for ti in (30.0, 180.0):
                    closed = max(closed, np.abs(S.ir_se(a, tr, te, ti, t1, 0.9, (q, delay)) - bloch.se_ideal(a, tr, te, t1, ti=ti, inv_eff=0.9, sat=(q, delay))).max())
    iso = 0.0
    for flip, tr, t1_, t2_, q, delay in ((10, 30.0, 382.0, 68.0, 0.1, 6.5), (60, 30.0, 382.0, 68.0, 0.1, 6.5), (25, 15.0, 343.0, 58.0, 0.0, 7.0), (40, 13.0, 300.0, 40.0, 0.5, 8.0),
                                         (5, 13.0, 1200.0, 100.0, 0.03, 7.0), (60, 500.0, 382.0, 68.0, 0.25, 7.0)):
        s = epg.steady_state(D(flip), tr, t1_, t2_, sat=(q, delay))[0]
        ref = bloch.rf_spoiled_isochromats(D(flip), tr, t1_, t2_, n_iso=1024, sat=(q, delay))[0]
        iso = max(iso, abs(s - ref) / abs(ref))
    # the engine's amplitude maps: q differs from voxel to voxel and from line to line
    w = W.make_world("5b" * 16, dict(kind="qa", qa=dict(n_oil=4), b0_hz=400.0, cf_offset_hz=100.0, b1_shape_amp=(0.2, 0.24), fatsat_eff=(0.93, 0.93)))
    eng = kspace.Engine(w)
    sy, f0 = w["system"], G * w["b0_t"]
    maps = 0.0
    for prot in (dict(sequence="spgr", tr_ms=24.0, flip_deg=25.0), dict(sequence="spgr", tr_ms=24.0, flip_deg=25.0, **HARD), dict(sequence="se", tr_ms=300.0),
                 dict(sequence="ir", tr_ms=1500.0, ti_ms=180.0), dict(sequence="epi", tr_ms=300.0, flip_deg=70.0)):
        rp = console.plan(dict(prot, matrix=64, fat_sat=True, fatsat_offset_hz=-40.0), w["console"])["realised"]
        tbw, delay = rp["rf_tbw"], rp["fatsat_delay_ms"]
        for c, name in enumerate(w["names"]):
            cls, sup = w["tissues"][name], eng.sup[c]
            if not cls["fat"]:
                continue
            lines, amps = eng._species(rp, c)
            idx = np.flatnonzero(w["weights"][c][sup] > 0.5)[::397][:6]
            for (ppm, _), amp in zip(lines, amps[0]):
                off = eng.df[sup][idx] + (ppm - tissue.FAT_PPM) * f0 - sy["fatsat_offset_hz"] + 40.0
                q = 1 - sy["fatsat_eff"] * 0.5 * (1 - np.tanh((np.abs(off) - sy["fatsat_bw_hz"] / 2) / 8.0))
                for i, qi in zip(idx, q):
                    a = D(rp["flip_deg"]) * eng.b1[sup][i]
                    if tbw is not None:
                        direct = kspace.slice_signal(rp, cls, a, tbw, inv_eff=sy["inv_eff"], sat=(qi, delay))[0][0]
                    elif rp["sequence"] == "spgr":
                        direct = epg.spgr(a, rp["tr_ms"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"],
                                          kspace.diffusion_bd(cls["adc_um2_ms"], rp["spoiler_mt_ms_per_m"], rp["tr_ms"]), (qi, delay))
                    else:
                        direct = S.se(a, rp["tr_ms"], rp["te_ms"], cls["t1_ms"], (qi, delay))
                    got = amp[i] / (w["weights"][c][sup][i] * cls["pd"])
                    maps = max(maps, abs(got - direct) / abs(direct))
    return [
        _row("S16", "signal", "longitudinal steady state with a saturation pulse (Mz x q, 4.5 to 12 ms before every excitation): spoiled gradient echo, "
             "spin echo, inversion recovery; q 0 to 1, B1 60 to 140 %", "separately written matrix Bloch steady state with a saturation operator (bloch.py)", closed, 1e-12),
        _row("S17", "signal", "RF-spoiled steady state with the saturation acting on the longitudinal states of every order (6 cases, 50 to 100 % saturated)",
             "isochromat ensemble (1024 spins) by time stepping, Mz x q at that time in every repetition", iso, 1e-6, f"{iso:.1e} relative"),
        _row("K32", "encoding", "amplitude maps under fat saturation in a field that moves the lines across the band (spgr 2D and 3D, se, ir, epi; four oil vials, "
             "six lines each): cubic interpolation of the slice integral over q, the spoiling term on 13 nodes of q",
             "the steady state solved at the flip angle and the saturation of every sampled voxel and line", maps, 2e-4, f"{maps:.1e} relative"),
    ]


def check_adjustments():
    """cf_shift_hz, shim_y / shim_x_hz_per_mm, tx_scale: the adjusted system serves the data of the system that needs none."""
    shapes = [("wm", _disc(60)), ("subcut_fat", _disc(12, 20.0, -15.0))]
    prots = [dict(sequence="spgr", tr_ms=30, flip_deg=20, n_echoes=2, fat_sat=True), dict(sequence="se", tr_ms=400, fat_sat=True), dict(sequence="ir", tr_ms=1500, ti_ms=200),
             dict(sequence="bssfp", tr_ms=8, flip_deg=40), dict(sequence="epi", tr_ms=500), dict(sequence="afi", tr_ms=30, flip_deg=50)]
    ph = W.shapes_phantom(shapes)
    y, x = W.grid(W.flat_world(ph))
    ref = kspace.Engine(W.flat_world(ph, fatsat_eff=0.95))
    off = kspace.Engine(W.flat_world(ph, fatsat_eff=0.95, cf_offset_hz=-90.0))
    tilted = W.flat_world(ph, fatsat_eff=0.95)
    tilted["fields"]["b0_hz"] = 25.0 + 0.6 * y - 0.35 * x
    tilted = kspace.Engine(tilted)
    weak = kspace.Engine(W.flat_world(ph, fatsat_eff=0.95, tx_gain=0.8))
    cf = shim = tx = moved = 0.0
    for p in prots:
        p = dict(p, matrix=64, fov_mm=256)
        want = ref.scan(p)["images"]
        top = np.abs(want).max()
        cf = max(cf, np.abs(off.scan(dict(p, cf_shift_hz=-90.0))["images"] - want).max() / top)
        shim = max(shim, np.abs(tilted.scan(dict(p, cf_shift_hz=25.0, shim_y_hz_per_mm=-0.6, shim_x_hz_per_mm=0.35))["images"] - want).max() / top)
        tx = max(tx, np.abs(weak.scan(dict(p, tx_scale=1.25))["images"] - want).max() / top)
        moved = max(moved, min(np.abs(off.scan(p)["images"] - want).max(), np.abs(tilted.scan(p)["images"] - want).max(), np.abs(weak.scan(p)["images"] - want).max()) / top)
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
    one, more = (console.plan(dict(sequence="bssfp", tr_ms=6.0, flip_deg=40.0, matrix=64, tx_scale=v), exam) for v in (1.0, 1.4))
    free = max(abs(more["limits"]["rf_power_percent"] / one["limits"]["rf_power_percent"] / 1.4 ** 2 - 1), abs(more["scan_s"] - one["scan_s"]),
               abs(console.plan(dict(sequence="se", tr_ms=500, cf_shift_hz=120.0, shim_x_hz_per_mm=1.0), exam)["scan_s"] - console.plan(dict(sequence="se", tr_ms=500), exam)["scan_s"]))
    return [
        _row("K33", "encoding", "a system 90 Hz off resonance scanned with cf_shift_hz = -90 (six sequences, fat saturation on where it exists)",
             "the image of the system on resonance: displacement, echo phase, bSSFP bands and the saturation band move together", cf, 1e-9, f"{cf:.1e} of the maximum"),
        _row("K34", "encoding", "a field of 25 Hz + 0.6 Hz/mm x y - 0.35 Hz/mm x x scanned with the opposite shim terms and cf_shift_hz = 25",
             "the image of the homogeneous system", shim, 1e-9, f"{shim:.1e} of the maximum"),
        _row("K35", "encoding", "a transmitter at 80 % scanned with tx_scale = 1.25 (excitation and refocusing angles; the adiabatic inversion is not scaled)",
             "the image of the calibrated system", tx, 1e-9, f"{tx:.1e} of the maximum"),
        _row("K36", "encoding", "the same three systems scanned without the adjustment", "an image that differs (the smallest of the largest differences over the six sequences)",
             moved, 0.01, f"{100 * moved:.1f} % of the maximum", kind="min"),
        _row("C9", "console", "price of the adjustments: RF power with tx_scale = 1.4, scan time with any of them", "RF power x tx_scale^2; the scan time does not change", free, 2e-3),
    ]


# ================================================================================================== console, continued
def check_console_slice():
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
    gamma = console.GAMMA_HZ_PER_T
    # energy and peak of the shape, from a fine sampling of the continuous pulse
    t = (np.arange(8192) + 0.5) / 8192 - 0.5
    shape = max(max(abs(rf.energy_factor(tbw) / (np.mean(b * b) / np.mean(b) ** 2) - 1), abs(rf.peak_factor(tbw) / (b.max() / np.mean(b)) - 1))
                for tbw in (2.0, 4.0, 8.0) for b in [np.sinc(tbw * t) * 0.5 * (1 + np.cos(2 * np.pi * t))])
    err = 0.0
    for name, (dur, tbw) in rf.PULSES.items():
        p = console.plan(dict(sequence="spgr", tr_ms=25, flip_deg=30, matrix=96, rf_pulse=name, slice_mm=4.0), exam)
        g = bloch.calibration(tbw) * tbw / (dur * 1e-3 * gamma * 4e-3) * 1e3
        power = 100 * console._b1_ut(30.0, dur) ** 2 * dur * rf.energy_factor(tbw) / 25.0 / 3.0 ** 2
        err = max(err, abs(p["timing"]["slice_gradient_mt_m"] / g - 1), abs(p["limits"]["rf_power_percent"] - round(power, 2)),
                  abs(p["realised"]["rf_ms"] - dur), abs(p["realised"]["rf_tbw"] - tbw))
    thin = bloch.calibration(4.0) * 4.0 / (2e-3 * gamma * 0.8 * 40e-3) * 1e3
    lim = [console.plan(dict(sequence="spgr", tr_ms=25, flip_deg=30, matrix=96, slice_mm=v), exam) for v in (round(thin - 0.02, 2), round(thin + 0.02, 2))]
    refused = lim[0]["accept"] or lim[1]["accept"] is not True or lim[0]["violations"][0]["code"] != "SLICE_TOO_THIN"
    # the 3D mode: time x partitions, noise / sqrt(partitions), the header says so
    two, three = (console.plan(dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=64, slice_mm=5.0, **kw), exam) for kw in ({}, HARD))
    parts = math.ceil(160.0 / 5.0)
    price = max(abs(three["scan_s_parts"]["acquisition_s"] / two["scan_s_parts"]["acquisition_s"] - parts), abs(three["realised"]["partitions"] - parts),
                abs(three["derived"]["relative_snr"] / two["derived"]["relative_snr"] / math.sqrt(parts) - 1))
    eng = _engine([("wm", _disc(40))], dict(sigma0=0.03))
    prot = dict(sequence="se", tr_ms=500, te_ms=20, fov_mm=256, matrix=128, bandwidth_hz_per_px=200)
    sd = []
    for kw in ({}, HARD):
        r = eng.scan(dict(prot, **kw), np.random.default_rng(3))
        sd.append((r["images"][0] - eng.scan(dict(prot, **kw))["images"][0]).real.std() / eng.noise_sd(r["plan"]["realised"]))
    noise = max(abs(sd[0] - 1), abs(sd[1] - 1), abs(eng.noise_sd(console.plan(dict(prot, **HARD), exam)["realised"])
                                                     / eng.noise_sd(console.plan(prot, exam)["realised"]) * math.sqrt(parts) - 1))
    return [
        _row("C4", "console", "energy and peak factors of the windowed-sinc pulses (time-bandwidth 2 / 4 / 8)",
             "mean(b^2) / mean(b)^2 and max(b) / mean(b) of the continuous shape (8,192 samples; the console plays 128 sub-pulses)", shape, 3e-3),
        _row("C5", "console", "slice gradient and RF power of the three selective pulses (30 degrees, TR 25 ms, 4 mm slice)",
             "G = k x tbw / (duration x gamma x thickness), k = width at half maximum of the small-angle profile of the separately written cascade "
             "in nominal bandwidths; B1rms^2 = B1(hard pulse of that duration)^2 x duration x energy factor / TR", err, 1e-3),
        _row("C6", "console", f"thinnest slice of the normal pulse: {thin:.2f} mm (80 % of 40 mT/m)", "refused with SLICE_TOO_THIN below it, accepted above",
             float(refused), 0.0, "refused / accepted" if not refused else "wrong verdict"),
        _row("C7", "console", "excitation = nonselective (3D): price and header", "acquisition time x ceil(160 mm / slice), relative SNR x sqrt(partitions)", price, 1e-3,
             f"{parts} partitions"),
        _row("C8", "console", "noise of a selective and of a nonselective scan", "Engine.noise_sd of the realised protocol; the 3D scan has 1 / sqrt(partitions) of the 2D noise",
             noise, 0.03, f"max deviation {100 * noise:.1f} %"),
    ]


CHECKS = (check_closed_forms, check_ernst_angle, check_epg, check_spin_echo, check_bssfp, check_bssfp_phase_cycle, check_node_interpolation,
          check_rf_pulse, check_slice_integral, check_slice_models, check_slice_engine,
          check_numerics, check_fat_shift, check_wrap, check_gibbs, check_t2star_blur, check_bssfp_bands, check_epi,
          check_interference, check_noise, check_in_opposed, check_positions, check_field_map, check_field_of_view, check_motion,
          check_fat_saturation_offset, check_saturation, check_adjustments, check_console, check_console_slice)
GROUP_OF = {check_bssfp_phase_cycle: ("signal", "rf"), check_saturation: ("signal", "encoding"), check_adjustments: ("encoding", "console"), check_closed_forms: "signal", check_ernst_angle: "signal", check_epg: "signal", check_spin_echo: "signal", check_bssfp: "signal",
            check_node_interpolation: "signal", check_rf_pulse: "rf", check_slice_integral: "rf", check_slice_models: "rf", check_slice_engine: "rf",
            check_console: "console", check_console_slice: "console"}


def run(groups=None):
    """Run every check, or those of the named groups (``GROUPS``). Returns (rows, seconds)."""
    t0 = time.time()
    rows = []
    for fn in CHECKS:
        own = GROUP_OF.get(fn, "encoding")
        if not groups or set((own,) if isinstance(own, str) else own) & set(groups):
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
