"""Control policies of ``Scanner/NoWrap`` and the per-instance admission rule.

Every policy plays through the agent's interface only (``info`` / ``plan`` / ``scan`` / ``view`` / ``budget``) and
returns the deliverable. None reads the hidden world.

Admission controls (each must fail on an admitted instance)
  prior            no scan; the geometry that is safe for any object            (defines the zero of the score)
  console_default  one token scan, then the console's default geometry (field of view 240 mm, phase along y)
  fixed_recipe     one constant geometry, tuned on other worlds, applied blindly
  brute_force      shrinks a square field of view scan by scan until the object touches the border; never thinks
                   about which axis to phase-encode
  wrong_units      the reference's geometry with the field of view in centimetres
  seed_attacker    tries to rebuild the world from the public generator with guessable seeds; failing that, replays
                   the best geometry of another instance
Reported, not part of the admission rule
  no_swap          measures like the reference but always phase-encodes along y
  size_only        measures the object's size but ignores that it is off-centre (the field of view is centred)
  safe_max         one scan, then the safe geometry
The reference: one localiser, reads the extent of the object, phase-encodes the shorter centred extent.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from ...scanner import kspace
from ...scanner import session as core
from ...scanner import world as W
from . import task

LOCALISER = dict(sequence="spgr", tr_ms=10.0, flip_deg=10.0, fov_mm=288.0, matrix=160, averages=2)
MARGIN_MM = 1.0


def _px(card):
    return card["task"]["max_pixel_mm"]


def _geometry(h_pe, h_ro, pe_dir, px):
    """Smallest centred geometry that reaches ``h_pe`` / ``h_ro`` mm from the isocentre at pixel ``px``."""
    n_pe = 2 * math.ceil(h_pe / px)
    n_ro = max(2 * math.ceil(h_ro / px), n_pe, 32)
    return dict(pe_dir=pe_dir, fov_mm=round(n_ro * px, 6), phase_fov=round(n_pe / n_ro, 6), matrix=n_ro, phase_oversampling=0.0)


def _safe(card):
    n = 2 * math.ceil(task.SAFE_FOV_MM / 2 / _px(card))
    return dict(pe_dir="y", fov_mm=round(n * _px(card), 6), phase_fov=1.0, matrix=n, phase_oversampling=0.0)


def _localise(sc, protocol=LOCALISER):
    """One localiser and the extent of the object on it: (y_lo, y_hi, x_lo, x_hi) in mm, or None."""
    r = sc.scan(protocol)
    box = sc.view(r["scan_index"])["bounding_box"]
    if not box or box["touches_edge"]:
        return None
    return (*box["y_mm"], *box["x_mm"])


def reference(sc):
    card = sc.info()
    box = _localise(sc)
    if box is None:
        return _safe(card)
    hy, hx = max(abs(box[0]), abs(box[1])) + MARGIN_MM, max(abs(box[2]), abs(box[3])) + MARGIN_MM
    return _geometry(hy, hx, "y", _px(card)) if hy <= hx else _geometry(hx, hy, "x", _px(card))


def prior(sc):
    return _safe(sc.info())


def safe_max(sc):
    card = sc.info()
    sc.scan(LOCALISER)
    return _safe(card)


def console_default(sc):
    card = sc.info()
    sc.scan(LOCALISER)
    n = 2 * math.ceil(card["default_fov_mm"] / 2 / _px(card))
    return dict(pe_dir="y", fov_mm=round(n * _px(card), 6), phase_fov=1.0, matrix=n, phase_oversampling=0.0)


def recipe_candidates():
    return [dict(pe_dir=d, h_pe=float(h)) for d in ("y", "x") for h in range(52, 132, 2)]


def fixed_recipe(recipe):
    def policy(sc):
        card = sc.info()
        sc.scan(LOCALISER)
        return _geometry(recipe["h_pe"], task.SAFE_FOV_MM / 2, recipe["pe_dir"], _px(card))
    return policy


DEFAULT_RECIPE = dict(pe_dir="y", h_pe=100.0)


def brute_force(sc):
    """Square fields of view from large to small with a cheap scan each; keeps the last one the object did not touch."""
    card = sc.info()
    px, best = _px(card), task.SAFE_FOV_MM
    for fov in (240.0, 216.0, 192.0, 168.0, 144.0, 120.0):
        prot = dict(sequence="spgr", tr_ms=10.0, flip_deg=15.0, fov_mm=fov, matrix=64)
        p = sc.plan(prot)
        if not p["accept"] or p["scan_s"] > sc.budget()["remaining_s"]:
            break
        box = sc.view(sc.scan(prot)["scan_index"])["bounding_box"]
        if not box or box["touches_edge"]:
            break
        best = fov
    return _geometry(best / 2, best / 2, "y", px)


def wrong_units(sc):
    a = reference(sc)
    return dict(a, fov_mm=a["fov_mm"] / 10.0)


def no_swap(sc):
    card = sc.info()
    box = _localise(sc)
    if box is None:
        return _safe(card)
    hy, hx = max(abs(box[0]), abs(box[1])) + MARGIN_MM, max(abs(box[2]), abs(box[3])) + MARGIN_MM
    if hx < hy:                                                          # phase stays along y; the readout must still cover x
        return dict(_geometry(hy, hy, "y", _px(card)))
    return _geometry(hy, hx, "y", _px(card))


def size_only(sc):
    card = sc.info()
    box = _localise(sc)
    if box is None:
        return _safe(card)
    hy, hx = (box[1] - box[0]) / 2 + MARGIN_MM, (box[3] - box[2]) / 2 + MARGIN_MM
    return _geometry(hy, hx, "y", _px(card)) if hy <= hx else _geometry(hx, hy, "x", _px(card))


def guessable_seeds(task_id, n_small=200):
    out = [f"{i:032x}" for i in range(n_small)]
    for s in (task_id, task.FAMILY, task_id + task.FAMILY, "0", "seed", "mrgym"):
        out += [hashlib.sha256(s.encode()).hexdigest()[:32], hashlib.md5(s.encode()).hexdigest()]
    return out


def _layout_half_extents(layout):
    """Half extents of a rotated ellipse about the isocentre, from the public generator's layout (no rasterising)."""
    b = layout["body"]
    c, s = math.cos(math.radians(b["rot"])), math.sin(math.radians(b["rot"]))
    ey, ex = math.hypot(b["a"] * c, b["b"] * s), math.hypot(b["a"] * s, b["b"] * c)
    return abs(b["y"]) + ey, abs(b["x"]) + ex, (b["y"] - ey, b["y"] + ey, b["x"] - ex, b["x"] + ex)


def seed_attacker(sc, candidates=None, diagnostics=None):
    """One localiser (so the log is not empty), then: (1) rebuild candidate worlds with the PUBLIC generator from
    guessable seeds and accept one whose outline matches the localiser; (2) failing that, deliver the fastest
    geometry of another instance."""
    card = sc.info()
    box = _localise(sc)
    px, other = _px(card), None
    for seed in candidates if candidates is not None else guessable_seeds(card["task_id"]):
        for level in task.LEVELS.values():
            hy, hx, edges = _layout_half_extents(W.make_world(seed, level["world_spec"], layout_only=True))
            geo = _geometry(hy + MARGIN_MM, hx + MARGIN_MM, "y", px) if hy <= hx else _geometry(hx + MARGIN_MM, hy + MARGIN_MM, "x", px)
            if box is not None and max(abs(a - b) for a, b in zip(edges, box)) < 0.6:
                if diagnostics is not None:
                    diagnostics["world_recovered"] = True
                return geo
            other = other or geo
    if diagnostics is not None:
        diagnostics["world_recovered"] = False
    return other or _safe(card)


# ------------------------------------------------------------------ running and admission
def run_policy(cfg, policy, noise_draw=0, engine=None, budget_scale=1.0):
    """Play one episode in process. Returns (answer or None, session, error text or None)."""
    session = core.ScannerSession(dict(cfg, noise_draw=noise_draw, budget_s=cfg["budget_s"] * budget_scale), engine=engine)
    try:
        return policy(core.LocalScanner(session)), session, None
    except Exception as e:  # noqa: BLE001  a crashing policy is a failed episode
        return None, session, f"{type(e).__name__}: {e}"


ADMISSION_CONTROLS = ("prior", "console_default", "fixed_recipe", "brute_force", "wrong_units", "seed_attacker")
REPORTED_CONTROLS = ("no_swap", "size_only", "safe_max")
DESCRIPTIONS = dict(
    reference="one localiser, reads the object's extent, phase-encodes the shorter centred extent",
    prior="no scan; the geometry that is safe for any object (defines reward 0)",
    console_default="one token scan, then the console's default geometry",
    fixed_recipe="one constant geometry tuned on other worlds, applied blindly",
    brute_force="shrinks a square field of view scan by scan until the object touches the border",
    wrong_units="the reference's geometry with the field of view in centimetres",
    seed_attacker="rebuilds worlds from guessable seeds with the public generator; failing that, replays the fastest geometry of another instance",
    no_swap="measures like the reference but always phase-encodes along y",
    size_only="uses the object's size and ignores that it is off-centre",
    safe_max="one scan, then the safe geometry",
)


def policies(recipe=None):
    recipe = recipe or DEFAULT_RECIPE
    return dict(reference=(reference, 1.0), prior=(prior, 1.0), console_default=(console_default, 1.0),
                fixed_recipe=(fixed_recipe(recipe), 1.0), brute_force=(brute_force, 1.0), wrong_units=(wrong_units, 1.0),
                seed_attacker=(seed_attacker, 1.0), no_swap=(no_swap, 1.0), size_only=(size_only, 1.0), safe_max=(safe_max, 1.0))


def tune_recipe(cfgs, candidates=None, draws=(0,)):
    """"Tuned on other worlds": the constant geometry that passes the most tuning instances (ties: the highest mean
    reward). This is the strongest constant a shortcut could use."""
    worlds = [W.make_world(c["seed"], c["world_spec"]) for c in cfgs]
    engines = [kspace.Engine(w) for w in worlds]
    best = None
    for cand in candidates or recipe_candidates():
        scores = []
        for cfg, w, eng in zip(cfgs, worlds, engines):
            for d in draws:
                answer, session, _ = run_policy(cfg, fixed_recipe(cand), d, eng)
                scores.append(task.score(cfg, answer, session.n_scans, world=w, engine=eng))
        key = (float(np.mean([s["passed"] for s in scores])), float(np.mean([s["reward"] for s in scores])))
        if best is None or key > best[0]:
            best = (key, cand)
    return dict(best[1], tuned_pass_rate=best[0][0], tuned_mean_reward=best[0][1])


def admit(cfg, recipe=None, n_cal=0, n_test=8, n_ctrl=4):
    """Per-instance admission: the reference passes on ``n_test`` noise draws; every admission control fails on every
    one of its ``n_ctrl`` draws. (``n_cal`` is unused: the pass rule of this environment needs no calibration.)"""
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    eng = kspace.Engine(world, cache=64)
    pol = policies(recipe)
    opt, sf = task.fastest(world, cfg["task"]["max_pixel_mm"]), task.safe(world, cfg["task"]["max_pixel_mm"])
    rep = dict(task_id=cfg["task_id"], level=cfg["level"], budget_s=cfg["budget_s"], reasons=[], controls={},
               scan_s_fastest=opt["scan_s"], scan_s_safe=sf["scan_s"], fastest_pe_dir=opt["geometry"]["pe_dir"],
               tolerance=task.TOL)

    def evaluate(name, draws):
        fn, scale = pol[name]
        res, used, errors = [], [], []
        for d in draws:
            answer, session, err = run_policy(cfg, fn, d, eng, scale)
            res.append(task.score(cfg, answer, session.n_scans, world=world, engine=eng))
            used.append(session.used)
            if err:
                errors.append(err)
        errs = [s["error"] for s in res if s["error"] is not None]
        return dict(pass_rate=float(np.mean([s["passed"] for s in res])), reward_mean=float(np.mean([s["reward"] for s in res])),
                    error_median=float(np.median(errs)) if errs else None, scan_s_median=float(np.median(used)),
                    gate_fail=sorted({s["diagnostic"] for s in res if s["diagnostic"] != "ok"})[:2], policy_errors=errors[:1])

    ref = evaluate("reference", range(n_test))
    rep["controls"]["reference"] = ref
    if task.TOL * opt["scan_s"] >= sf["scan_s"]:
        rep["reasons"].append("the fastest geometry is not well below the safe geometry")
    if ref["pass_rate"] < 0.9:
        rep["reasons"].append("reference does not pass")
    for name in ADMISSION_CONTROLS + REPORTED_CONTROLS:
        rep["controls"][name] = evaluate(name, range(1 if name in ("prior",) else n_ctrl))
        if name in ADMISSION_CONTROLS and rep["controls"][name]["pass_rate"] > 0:
            rep["reasons"].append(f"{name} passes")
    answer, session, _ = run_policy(cfg, safe_max, 0, eng)
    rep["reward_prior_after_one_scan"] = task.score(cfg, answer, session.n_scans, world=world, engine=eng)["reward"]
    rep["admitted"] = not rep["reasons"]
    return rep
