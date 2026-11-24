"""``Scanner/NoWrap``: instances, scoring and the verifier.

The job is the first decision of any examination: look at a localiser, then prescribe the geometry of the real scan.
The real scan is a spin echo with a fixed repetition time, so its scan time is proportional to the number of
phase-encode lines. The object's size, orientation and position are hidden, and they decide which axis should be
phase-encoded and how large the phase field of view must be.

Deliverable (geometry of the target scan):

    {"pe_dir": "y" | "x", "fov_mm": 260.0, "phase_fov": 0.75, "matrix": 130, "phase_oversampling": 0.0}

Reward: hard gates, then a continuous score of the scan time of the delivered protocol.
    gates   the scanner log replays and stays within budget; at least one accepted scan; the deliverable is a valid
            geometry that the console accepts; the pixel is not larger than required; the whole object lies inside
            the image; a noise-free re-scan of the delivered protocol on the hidden world shows no wrap-around
    time    T = scan time of the delivered protocol on the console
    score   clip( ln(T_safe / T) / ln(T_safe / (TOL * T_opt)), 0, 1 )
            T_safe = scan time of the geometry that is safe for any object (the largest field of view both ways),
            T_opt  = scan time of the fastest geometry that passes the gates on this world (exhaustive, by the verifier)
    reward  = score if all gates pass else 0.   pass := gates and T <= TOL * T_opt.
"""
from __future__ import annotations

import math
import os

import numpy as np

from ...scanner import console, kspace
from ...scanner import session as core
from ...scanner import world as W

FAMILY = "nowrap"
TOL = 1.10                                  # a delivered scan may take 10 % longer than the fastest valid one
TARGET = dict(sequence="se", tr_ms=500.0, bandwidth_hz_per_px=300.0)
SAFE_FOV_MM = 260.0                         # covers the whole region an object can occupy
MARGIN_MM = 1.0                             # the verifier's fastest geometry keeps this distance from the object
COVER_SLACK_MM = 0.5
WRAP_LIMIT = 2e-3                           # energy of (delivered - unwrapped reference) / energy of the reference
FIELDS = ("pe_dir", "fov_mm", "phase_fov", "matrix", "phase_oversampling")
BASE = dict(kind="qa", b0_t=1.5, tx_gain_dev=(0.0, 0.05))
LEVELS = {
    1: dict(world_spec=dict(BASE, qa=dict(body_radius_mm=(50.0, 100.0), offset_mm=3.0, body_rotation_deg=(0.0, 90.0),
                                          ring_radius_mm=(26.0, 30.0), vial_radius_mm=(11.0, 13.0), n_ring=(3, 4))),
            budget_s=(30.0, 30.0), pixel_mm=2.0,
            summary="object centred, long axis along x or y (size and orientation hidden)"),
    2: dict(world_spec=dict(BASE, qa=dict(body_radius_mm=(50.0, 100.0), offset_mm=20.0,
                                          ring_radius_mm=(26.0, 30.0), vial_radius_mm=(11.0, 13.0), n_ring=(3, 4))),
            budget_s=(30.0, 30.0), pixel_mm=2.0,
            summary="object off-centre by up to 20 mm and rotated by any angle (the field of view stays centred)"),
}


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    lo, hi = lv["budget_s"]
    budget = float(np.round(W.rng_for(seed, "task").uniform(lo, hi)))
    task = dict(family=FAMILY, target_scan=dict(TARGET, te_ms="console minimum", averages=1, slice_mm=5.0),
                max_pixel_mm=lv["pixel_mm"], regions=[],
                deliverable={"pe_dir": "y or x", "fov_mm": "<number>", "phase_fov": "<number, 0.25 to 1>",
                             "matrix": "<even integer>", "phase_oversampling": "<number, 0 to 1; optional>"})
    return dict(task_id=task_id or "nowrap-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed,
                world_spec=lv["world_spec"], budget_s=budget, max_requests=32, task=task)


def statement(cfg: dict) -> str:
    t = cfg["task"]
    return f"""# Prescribe the fastest scan that shows the whole object without wrap-around

You operate a virtual MR scanner through tool calls. An object of unknown size, orientation and position lies near
the isocentre. The examination needs one spin-echo image of the whole object:

    sequence se, tr_ms {TARGET['tr_ms']:g}, te_ms at the console minimum, bandwidth_hz_per_px {TARGET['bandwidth_hz_per_px']:g}, 1 average, slice 5 mm,
    in-plane pixel size at most {t['max_pixel_mm']:g} mm along both axes.

Deliver the geometry of that scan with `submit`: phase-encode axis (`pe_dir`), readout field of view (`fov_mm`),
`phase_fov` (phase field of view as a fraction of `fov_mm`), `matrix`, and optionally `phase_oversampling`.
The field of view is always centred on the isocentre.

Requirements: the whole object lies inside the image; no wrap-around; the pixel is not larger than required.
Among the geometries that meet them, the scan time decides: the target scan itself is not run in this episode, the
scanner evaluates the geometry you deliver. Use the scan-time budget for whatever you need to see first.

Scoring: 0 for an invalid geometry, a cut-off object or wrap-around; otherwise higher for a shorter scan, full marks
within {round(100 * (TOL - 1))} % of the fastest valid geometry. The scanner's log must show that you measured."""


def protocol_of(geometry: dict) -> dict:
    return dict(TARGET, **{k: geometry[k] for k in FIELDS if k in geometry})


def check_deliverable(answer, exam: dict):
    """Returns (plan or None, diagnostic)."""
    if not isinstance(answer, dict) or not {"pe_dir", "fov_mm", "phase_fov", "matrix"} <= set(answer) or set(answer) - set(FIELDS):
        return None, 'the answer must be {"pe_dir", "fov_mm", "phase_fov", "matrix"} and optionally "phase_oversampling"'
    pl = console.plan(protocol_of(answer), exam)
    if not pl["accept"]:
        hint = " (a field of view of this size looks like centimetres)" if isinstance(answer.get("fov_mm"), (int, float)) \
            and not isinstance(answer.get("fov_mm"), bool) and 0 < answer["fov_mm"] < 80 else ""
        return None, "the console rejects this geometry: " + "; ".join(v["message"] for v in pl["violations"]) + hint
    return pl, "ok"


def half_extents(world: dict) -> tuple[float, float]:
    """Largest distance of the object from the isocentre along y and along x, in mm (verifier side)."""
    e = W.extent_mm(world)
    return max(abs(e["y"][0]), abs(e["y"][1])), max(abs(e["x"][0]), abs(e["x"][1]))


def fastest(world: dict, pixel_mm: float) -> dict:
    """The fastest geometry that covers the object with MARGIN_MM to spare at the largest allowed pixel."""
    hy, hx = half_extents(world)
    best = None
    for pe_dir, h_pe, h_ro in (("y", hy, hx), ("x", hx, hy)):
        n_pe = 2 * math.ceil((h_pe + MARGIN_MM) / pixel_mm)
        n_ro = max(2 * math.ceil((h_ro + MARGIN_MM) / pixel_mm), n_pe)
        n_ro = max(n_ro, 32)
        geo = dict(pe_dir=pe_dir, fov_mm=round(n_ro * pixel_mm, 6), phase_fov=n_pe / n_ro, matrix=n_ro, phase_oversampling=0.0)
        pl = console.plan(protocol_of(geo), world["console"])
        if pl["accept"] and (best is None or pl["scan_s"] < best["scan_s"]):
            best = dict(geometry=geo, scan_s=pl["scan_s"], lines=pl["realised"]["lines"])
    return best


def safe(world: dict, pixel_mm: float) -> dict:
    n = 2 * math.ceil(SAFE_FOV_MM / 2 / pixel_mm)
    geo = dict(pe_dir="y", fov_mm=round(n * pixel_mm, 6), phase_fov=1.0, matrix=n, phase_oversampling=0.0)
    return dict(geometry=geo, scan_s=console.plan(protocol_of(geo), world["console"])["scan_s"])


def wrap_fraction(engine: kspace.Engine, realised: dict) -> float:
    """Energy of the wrap-around in a noise-free scan of ``realised``, relative to the image without it."""
    img = engine.recon(engine.clean_kspace(realised)[0], realised)
    need = engine.w["extent_mm"] * 1.05 / realised["fov_pe_mm"] - 1.0
    if need <= realised["phase_oversampling"]:
        return 0.0                                                        # the encoded field of view covers every spin
    ref_p = console.plan(dict(console.request_of(realised), phase_oversampling=min(1.0, need)), engine.w["console"])["realised"]
    ref = engine.recon(engine.clean_kspace(ref_p)[0], ref_p)
    return float(np.sum(np.abs(img - ref) ** 2) / max(np.sum(np.abs(ref) ** 2), 1e-30))


def score(cfg: dict, answer, measured: int, log_ok: bool = True, world: dict | None = None, engine: kspace.Engine | None = None) -> dict:
    """Reward and details for one episode. ``measured``: number of accepted scans in the scanner's log."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    px = cfg["task"]["max_pixel_mm"]
    opt, sf = fastest(world, px), safe(world, px)
    pl, why = check_deliverable(answer, world["console"])
    out = dict(reward=0.0, passed=False, gate_log=bool(log_ok), gate_measured=bool(measured > 0), gate_deliverable=pl is not None,
               gate_resolution=False, gate_coverage=False, gate_no_wrap=False, diagnostic=why, scan_s=None,
               scan_s_fastest=opt["scan_s"], scan_s_safe=sf["scan_s"], tolerance=TOL, error=None)
    if pl is None:
        return out
    rp = pl["realised"]
    fy, fx = core.fov_yx(rp)
    hy, hx = half_extents(world)
    wrap = wrap_fraction(engine, rp)
    out.update(scan_s=pl["scan_s"], lines=rp["lines"], pixel_mm=round(rp["fov_mm"] / rp["matrix"], 4), wrap_fraction=wrap,
               gate_resolution=bool(rp["fov_mm"] / rp["matrix"] <= px + 1e-6),
               gate_coverage=bool(fy / 2 >= hy - COVER_SLACK_MM and fx / 2 >= hx - COVER_SLACK_MM),
               gate_no_wrap=bool(wrap <= WRAP_LIMIT))
    out["error"] = pl["scan_s"] / opt["scan_s"] - 1.0                     # relative excess scan time
    gates = all(out[k] for k in ("gate_log", "gate_measured", "gate_resolution", "gate_coverage", "gate_no_wrap"))
    t, t_pass = pl["scan_s"], TOL * opt["scan_s"]
    if t <= t_pass:
        s = 1.0
    elif t_pass >= sf["scan_s"] or t >= sf["scan_s"]:
        s = 0.0
    else:
        s = float(np.clip(math.log(sf["scan_s"] / t) / math.log(sf["scan_s"] / t_pass), 0.0, 1.0))
    out.update(score=s, passed=bool(gates and t <= t_pass), reward=s if gates else 0.0)
    if not gates:
        out["diagnostic"] = "; ".join(m for k, m in (("gate_resolution", "the pixel is larger than required"),
                                                    ("gate_coverage", "part of the object lies outside the image"),
                                                    ("gate_no_wrap", "the image shows wrap-around"),
                                                    ("gate_measured", "no scan in the log"), ("gate_log", "the log does not replay"))
                                      if not out[k])
    return out


def verify(cfg: dict, log_dir: str, answer) -> dict:
    """Verifier entry for an episode played against the HTTP service: replays the scanner log, then scores."""
    rep = core.replay(cfg, core.read_log(os.path.join(log_dir, core.LOG_NAME)))
    out = score(cfg, answer, rep["scans"], rep["ok"] and not rep["over_budget"])
    out["replay"] = rep
    return out
