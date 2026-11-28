"""``rfscan``: a spoiled gradient echo whose excitation is a stored design. How a designed pulse meets the hidden scanner.

A console offers it when its card lists it (``console_sequences`` of the world spec) and holds a table of designs
(``exam["designs"]``, filled by ``ScannerSession.design``). The six sequences every console offers, and the four served
from the block-list core, are not touched: their parameter tables, plans and images are what they were.

What is played. The stored waveform, as designed, through the transmit chain of this scanner (``design.hardware``):
the amplifier's compression, then the transmit scale at the voxel, the slice-select gradient late by the scanner's
RF-to-gradient delay, the voxel's off-resonance. The console's own pulses do not pass through that amplifier law and
that delay: the manufacturer has linearised and timed them. A designed waveform is played as it is.

Where the slice-select gradient lies (``rf_axis``).
  ``readout``   on the readout axis (x when pe_dir = y). The pulse then selects a stripe |position along the readout|
                <~ thickness / 2 through the isocentre instead of a slice through the plane, and the image shows the
                profile of the pulse across that axis: every pixel column holds the spins the pulse turned by the
                flip angle of that position. This is how a slice profile is measured on a scanner. Nothing selects
                through the plane: the signal is that of the whole object along the slice axis, ``2 x OBJECT_HALF_MM``.
  ``slice``     through the plane, as in every other 2D scan: the image is the object, each voxel weighted by the
                integral of the signal over the profile, from -OBJECT_HALF_MM to +OBJECT_HALF_MM along the slice axis.
The stripe and the slice go through the isocentre: a design carries no frequency offset.

The signal of a spin turned by the Cayley-Klein parameters (a, b): the flip angle 2 asin|b| enters the steady state
of ``spgr`` (the engine's own, at the scan's TR, spoiler and RF phase increment) and the transverse phase is that of
2 conj(a) b after the rephasing lobe of the design. For ideal spoiling that is exact; with the RF-spoiled phase graph
it is the treatment the engine gives its own slice-selective pulses (independent columns, each with its flip angle
and phase). At a repetition time of five T1 the signal is sin(flip) to within 1 % (the factor
(1 - E1) / (1 - E1 cos(flip)) is 0.998 at 40 degrees and 0.993 at 90).

Exactness. The profile is evaluated at the master-grid points along the readout axis (0.4 mm apart on the QA
phantom), as every field of the world is, and at every voxel's own transmit scale and off-resonance by
interpolation between Chebyshev nodes that span the object (one node when the field is uniform). Through the plane
the integral is a sum over positions 1 / ``SLICE_POINTS`` of the thickness apart.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import console

NAME = "rfscan"
MAX_DESIGNS = 32                                 # stored designs per session (``mrgym.design.tool.MAX_DESIGNS``)
OBJECT_HALF_MM = 40.0                            # the test object reaches this far along the slice axis, either side of the isocentre
SLICE_POINTS = 100                               # positions per thickness of the through-plane integral
NODES = 5                                        # Chebyshev nodes over the transmit scale and over the off-resonance of the object

PARAMETERS = {
    **console._GEOM, **console._TE,
    # what a design fixes or what does not apply to it: one value each, so that the console refuses anything else.
    # slice_mm is the extent of the object along the slice axis: what the signal of a voxel is proportional to with
    # rf_axis readout (a slice of thickness d through the plane gives d / 80 of it, and of the relative SNR a plan prints)
    "slice_mm": (2 * OBJECT_HALF_MM, 2 * OBJECT_HALF_MM, "f", 2 * OBJECT_HALF_MM), "rf_pulse": (("normal",), None, "c", "normal"), "excitation": (("selective",), None, "c", "selective"),
    "tx_scale": (1.0, 1.0, "f", 1.0), "rotation_deg": (0.0, 0.0, "f", 0.0),
    "tr_ms": (3.0, 20000.0, "f", None), "spoiler_mt_ms_per_m": (10.0, 600.0, "f", 100.0), "rf_phase_increment_deg": (0.0, 180.0, "f", 117.0),
    "rf_design": (1, MAX_DESIGNS, "i", None), "rf_axis": (("readout", "slice"), None, "c", "readout"),
}
console.PARAMETER_UNITS.update(
    rf_design="number of a stored design (the design tool): the excitation pulse of the scan",
    rf_axis="where the slice-select gradient of the design lies: readout = the image shows the pulse's profile across the readout axis; "
            "slice = through the plane, as in an ordinary 2D scan")
TITLE = "gradient echo with a designed excitation pulse"
DESCRIPTION = ("spoiled gradient echo whose excitation is a stored design (rf_design), played as designed through this scanner's transmit "
               "chain. rf_axis readout: the slice-select gradient lies on the readout axis and the image shows the flip-angle profile of "
               "the pulse across it; rf_axis slice: an ordinary 2D image through the designed slice. The steady state is that of spgr; at "
               "tr_ms of five T1 the signal is sin(flip angle) to within 1 %")


def _time(c):
    """Timing rule: that of ``spgr`` with one echo, with the duration, energy and gradient of the design."""
    p, s, r, timing, bad, ras = c.p, c.s, c.r, c.timing, c.bad, c.ras
    k, stored = p["rf_design"], c.exam.get("designs") or {}
    d = stored.get(k)
    if d is None:
        bad.append(console._violation("UNKNOWN_DESIGN", "rf_design", f"no stored design {k}" + (
            f"; stored: {', '.join(str(i) for i in sorted(stored))}" if stored else "; make one with the design tool first"),
            requested=k, allowed=sorted(stored)))
        f = dict(duration_ms=1.0, energy_ut2_ms=0.0, gradient_mt_m=0.0, rephase_mt_ms_m=0.0, flip_deg=0.0, limits=dict(ok=True, violations=[]))
    else:
        f = d["facts"]
        # what the hardware cannot play. The RF power of a design is stated for the repetition time of its task; a scan
        # has its own, and the console checks that one below (``_conclude``), as for every sequence
        hard = [v for v in f["limits"]["violations"] if v["code"] != "RF_POWER_LIMIT"]
        if hard:
            v = hard[0]
            bad.append(console._violation("DESIGN_LIMIT", "rf_design", f"design {k} violates a limit ({v['code']}: {v['value']:g} against "
                                          f"{v['limit']:g} {v['unit']}); a design that violates a hardware limit is not played", requested=k))
    rf_ms = console._up(f["duration_ms"], ras)
    ramp_ss = 1e3 * f["gradient_mt_m"] * 1e-3 / c.slew                   # the slice gradient falls to zero after the pulse
    # the rephasing lobe also takes back what the slice gradient adds while it falls (as for the console's own pulses)
    rephase = console._lobe_ms((abs(f["rephase_mt_ms_m"]) + 0.5 * f["gradient_mt_m"] * ramp_ss) * 1e-6, s)
    pre_ro = console._lobe_ms(c.n / (2 * console.GAMMA_HZ_PER_T * c.fov_m), s)
    pre = pre_ro + rephase if p["rf_axis"] == "readout" else max(pre_ro, rephase)   # on one axis the two lobes follow each other
    te_min = console._up(rf_ms / 2 + ramp_ss + pre + c.ramp + c.t_ro / 2 + s["dead_ms"], ras)
    te = te_min if p["te_ms"] == "min" else console._up(p["te_ms"], ras)
    area = p["spoiler_mt_ms_per_m"] * 1e-6
    cyc_px = area * console.GAMMA_HZ_PER_T * c.px * 1e-3
    if cyc_px < 1 - 1e-9:
        need = 1 / (console.GAMMA_HZ_PER_T * c.px * 1e-3) * 1e6
        bad.append(console._violation("SPOILER_BELOW_MINIMUM", "spoiler_mt_ms_per_m", "the spoiler must dephase at least one cycle across a "
                                      f"pixel: >= {need:.1f} mT*ms/m at this pixel size", minimum=round(need, 1)))
    spoil = console._lobe_ms(area, s)
    tr_min = console._up(ramp_ss + rf_ms / 2 + te + c.t_ro / 2 + c.ramp + spoil + s["dead_ms"], ras)
    tr = console._up(p["tr_ms"], ras)
    r.update(tr_ms=tr, te_ms=te, flip_deg=float(f["flip_deg"]), spoiler_mt_ms_per_m=p["spoiler_mt_ms_per_m"],
             rf_phase_increment_deg=p["rf_phase_increment_deg"], rf_design=k, rf_axis=p["rf_axis"], rf_ms=round(rf_ms, 6), rf_tbw=None,
             rf_call=None if d is None else d["call_json"])
    timing.update(spoiler_ms=round(spoil, 3), spoiler_cycles_per_px=round(cyc_px, 2), rf_ms=round(rf_ms, 6),
                  slice_gradient_mt_m=round(f["gradient_mt_m"], 3))
    b1 = math.sqrt(f["energy_ut2_ms"] / rf_ms)                           # the amplitude of a hard pulse of the same duration and energy
    c.__dict__.update(te=te, te_min=te_min, tr=tr, tr_min=tr_min, pulses=[(b1, rf_ms)], cycle=tr, dummies=math.ceil(s["dummy_ms"] / tr),
                      esp=None, n_echo=1)


console.register(console.Sequence(
    name=NAME, parameters=PARAMETERS, title=TITLE, description=DESCRIPTION, timing=_time, sampling=console._sample_steady, serves="design",
    header=lambda r: f" design {r['rf_design']} on {r['rf_axis']}"))


# ------------------------------------------------------------------ image formation
def pulse_of(rp):
    """The pulse a realised ``rfscan`` protocol plays: rebuilt from the canonical call it carries."""
    from ..design import tool
    return tool.build(json.loads(rp["rf_call"]), tool.Limits(g_max_mt_m=math.inf, slew_t_m_s=math.inf))[0]


def chain_of(world, tx_scale, df_hz):
    """The transmit chain of ``world`` at a place with the given transmit scale and off-resonance."""
    from ..design import hardware
    sy = world["system"]
    return hardware.Chain(tx_scale=float(tx_scale), amp_sat_ut=sy.get("rf_amp_sat_ut"), grad_delay_us=float(sy.get("rf_grad_delay_us", 0.0)),
                          df_hz=float(df_hz))


def _nodes(values, kspace):
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9 * max(1.0, abs(lo)):
        return np.array([lo]), np.array([1.0])
    pad = 0.01 * (hi - lo)
    return kspace.cheb_nodes(lo - pad, hi + pad, NODES)


def excitation(eng, rp):
    """Per class, the complex amplitude of ``rp`` over the class's support: the steady state at the flip angle the
    design gives each voxel, with its transverse phase, times what lies along the slice axis."""
    from ..design import hardware
    from . import kspace
    from . import world as W
    w = eng.w
    pulse = pulse_of(rp)
    body = eng.body
    df_all = eng._df(rp)
    sn, sw = _nodes(eng.b1[body], kspace) if body.any() else (np.array([1.0]), np.array([1.0]))
    dn, dw = _nodes(df_all[body], kspace) if body.any() else (np.array([0.0]), np.array([1.0]))
    gre = dict(rp, sequence="spgr", n_echoes=1)
    stripe = rp["rf_axis"] == "readout"
    if stripe:                                                           # position of every master-grid point along the readout axis
        y, x = W.grid(w)
        u = (x if rp["pe_dir"] == "y" else y)
        pos, index = np.unique(u, return_inverse=True)
        index = index.reshape(u.shape)
    else:
        step = (pulse.thickness_mm or 10.0) / SLICE_POINTS
        m = int(math.ceil(OBJECT_HALF_MM / step))
        pos = np.arange(-m, m + 1) * step
        pos, dz = pos * (OBJECT_HALF_MM / (m * step)), OBJECT_HALF_MM / m
    flips, phases = np.empty((sn.size, dn.size, pos.size)), np.empty((sn.size, dn.size, pos.size), complex)
    for i, sc in enumerate(sn):
        for j, df in enumerate(dn):
            a, b = hardware.rotate(pulse, chain_of(w, sc, df), pos)
            mxy = 2 * np.conj(a) * b
            flips[i, j] = 2 * np.arcsin(np.clip(np.abs(b), 0.0, 1.0))
            mag = np.abs(mxy)
            # the rotation runs to the END of the pulse and the echo time is counted from its MIDDLE: the precession of the
            # half in between is in the rotation already, so it is taken out here (the engine adds 2 pi df TE)
            phases[i, j] = np.where(mag > 1e-12, mxy / np.where(mag > 1e-12, mag, 1.0), 1.0) * np.exp(-1j * np.pi * df * pulse.duration_ms * 1e-3)
    out = []
    for c, name in enumerate(w["names"]):
        cls, sup = w["tissues"][name], eng.sup[c]
        table = kspace.gre_steady(gre, cls, flips, eng.tier, kspace.PROFILE_NODES)[0] * phases        # [scale node, df node, position]
        ls = kspace.bary_matrix(sn, sw, eng.b1[sup]) if sn.size > 1 else np.ones((int(sup.sum()), 1))
        ld = kspace.bary_matrix(dn, dw, df_all[sup]) if dn.size > 1 else np.ones((int(sup.sum()), 1))
        if stripe:
            col = index[sup]
            amp = np.zeros(col.size, complex)
            for i in range(sn.size):
                for j in range(dn.size):
                    amp += ls[:, i] * ld[:, j] * table[i, j, col]
        else:
            wz = np.full(pos.size, dz)
            wz[[0, -1]] = dz / 2                                         # trapezoid rule over the object along the slice axis
            integral = (table * wz).sum(-1) / rp["slice_mm"]              # [scale node, df node]
            amp = np.einsum("vi,vj,ij->v", ls, ld, integral)
        out.append(amp * (w["weights"][c][sup] * cls["pd"]))
    return out


def form(eng, rp, key, moving):
    """Image formation (``kspace.FORMATION["design"]``): one amplitude map per class, the same on every line."""
    amps = excitation(eng, rp)
    ks = []
    for k, te in console.sampling(rp).images:
        ks.append(eng.kspace(rp, te, amps))
        if moving:
            ks[-1] = eng._motion(ks[-1], rp, te, k, amps, key)
    return ks
