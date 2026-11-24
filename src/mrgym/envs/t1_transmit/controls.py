"""Control policies of ``Scanner/T1Transmit`` and the per-instance admission rule.

Every policy plays an episode through the agent's interface only (``info`` / ``plan`` / ``scan`` / ``budget`` of a
``scanner.session.LocalScanner``, of the HTTP ``client.Scanner``, or of ``mrgym.policies.ToolScanner`` on top of the
tools) and returns the deliverable. None reads the hidden world. What a policy may use: the task card, the instrument
datasheet, the public generator's priors, and its own physics.

The physics here is the AGENT-SIDE helper physics: ideal-spoiling closed forms, written again in this file on purpose
(the scanner serves the EPG steady state; sharing its code would let errors cancel). The protocol constants of the
reference (spoiler moments, repetition times, RF phase increment of the dual-TR scan) come from an offline study of
the documented spoiling mechanism over the public priors.

Policies
  reference            maps B1 (dual-TR, signed with the phase images), sets the gradient-echo flip angles from the
                       measured B1, fills the budget with averages, joint fit of (M0, T1, B1) per region
  prior                zero scans, the prior's geometric mean                      (defines the zero of the score)
  blind_vfa            textbook two-angle variable flip angle with NOMINAL angles  (confounder-blind)
  fixed_recipe         one constant protocol + the same fit, tuned on other worlds, applied blindly
  brute_force          uniform sweep over sequences and flip angles at console defaults + the same generic fit
  ir_only              inversion recovery only, as much as fits
  wrong_units          the reference's numbers in seconds
  seed_attacker        tries to rebuild the world from the public generator with guessable seeds, replays another
                       instance's answer, and checks whether a repeated scan returns the same noise
Reported, not part of the admission rule:
  open_loop            the fixed recipe with averages scaled to the instance's budget (no use of any measurement)
  short_tr_ideal       B1-corrected pipeline at the shortest repetition times and default spoilers (trusts ideal spoiling)
  ir_unpriced          ir_only with 40x the budget (shows whether only the price closes that route)
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from ...scanner import kspace
from ...scanner import session as core
from ...scanner import world as W
from . import task

D2R = math.pi / 180.0


# ------------------------------------------------------------------ agent-side physics (ideal spoiling)
def spgr_ideal(a, tr, t1):
    e1 = np.exp(-tr / t1)
    return np.sin(a) * (1 - e1) / (1 - e1 * np.cos(a))


def afi_ideal(a, tr1, tr2, t1):
    e1, e2, c = np.exp(-tr1 / t1), np.exp(-tr2 / t1), np.cos(a)
    den = 1 - e1 * e2 * c * c
    return np.sin(a) * (1 - e2 + (1 - e1) * e2 * c) / den, np.sin(a) * (1 - e1 + (1 - e2) * e1 * c) / den


def _model(meas, t1, b1):
    rows = []
    for m in meas:
        a = b1 * m["flip"]
        if m["kind"] == "spgr":
            rows.append(spgr_ideal(a, m["tr"], t1))
        else:
            rows.extend(afi_ideal(a, m["tr"], m["tr2"], t1))
    return np.stack(np.broadcast_arrays(*rows))


def fit_region(meas, y, w, b1_fixed=None):
    """Weighted least squares of (M0, T1, B1) to the values ``y`` of one region; M0 is linear, (T1, B1) by a zooming
    grid. Returns (t1_ms, b1, m0)."""
    t1g = np.geomspace(150.0, 5000.0, 72)
    b1g = np.array([b1_fixed]) if b1_fixed is not None else np.linspace(0.45, 1.75, 53)
    w2 = w * w
    for _ in range(5):
        tt, bb = np.meshgrid(t1g, b1g, indexing="ij")
        m = _model(meas, tt, bb)
        num = np.einsum("n,nij->ij", w2 * y, m)
        den = np.einsum("n,nij->ij", w2, m * m)
        sse = -num * num / np.maximum(den, 1e-300)
        i, j = np.unravel_index(np.argmin(sse), sse.shape)
        best = (float(t1g[i]), float(b1g[j]), float(num[i, j] / max(den[i, j], 1e-300)))
        t1g = np.geomspace(t1g[max(i - 1, 0)], t1g[min(i + 1, len(t1g) - 1)], 13)
        if b1_fixed is None:
            b1g = np.linspace(b1g[max(j - 1, 0)], b1g[min(j + 1, len(b1g) - 1)], 13)
    return best


# ------------------------------------------------------------------ helpers on the agent's interface
def _rois(card):
    return [dict(id=r["id"], y_mm=r["y_mm"], x_mm=r["x_mm"], radius_mm=r["radius_mm"]) for r in card["task"]["regions"]]


def _values(resp):
    """Region signals [n_images, n_regions] from the complex-mean read-out (no noise floor)."""
    return np.array([r["complex_mean_magnitude"] for r in resp["roi_readouts"]], float).T


def _afi_signed(resp):
    """Dual-TR signals [2, n_regions] with the sign of image 2 relative to image 1 taken from the phase images: above
    about 100 degrees actual flip the second signal is negative, and magnitudes alone would fold the B1 estimate."""
    v = _values(resp)
    ph = np.array([r["complex_mean_phase_rad"] for r in resp["roi_readouts"]], float).T
    return np.stack([v[0], v[1] * np.sign(np.cos(ph[1] - ph[0]))])


def _answer(ids, t1):
    return {"units": "ms", "t1_ms": {k: float(np.clip(v, 31.0, 19999.0)) for k, v in zip(ids, t1)}}


def _prior_answer(card):
    return _answer([r["id"] for r in card["task"]["regions"]], [task.T1_PRIOR_MS] * len(card["task"]["regions"]))


def _cost(sc, protocol):
    """(fixed seconds, seconds per average) of a protocol, or None if the console rejects it."""
    p = sc.plan(dict(protocol, averages=1))
    if not p["accept"]:
        return None
    return p["scan_s"] - p["scan_s_parts"]["acquisition_s"], p["scan_s_parts"]["acquisition_s"]


COMMON = dict(matrix=64, bandwidth_hz_per_px=300.0, fov_mm=240.0)
# From the offline spoiling study (strongest spoiler that fits the repetition time; RF phase increment scanned):
# dual-TR B1 error <= 0.2 % and two-angle T1 error <= 1.5 % over the public priors of T1, T2 and diffusion.
AFI = dict(COMMON, sequence="afi", tr_ms=30.0, tr_ratio=5, spoiler_mt_ms_per_m=585.0, rf_phase_increment_deg=45.0)
SPGR = dict(COMMON, sequence="spgr", tr_ms=20.0, spoiler_mt_ms_per_m=340.0)
AFI_FLIP, TARGETS = 70.0, (4.0, 20.0)      # degrees; AFI_FLIP nominal, TARGETS are angles the spins should see
AFI_SHARE = 0.6                             # of the acquisition time: the B1 map limits the precision of T1


def _allocate(budget, cost_afi, cost_spgr, n_spgr, share=AFI_SHARE):
    """Averages (afi, spgr each) that fill ``budget`` with ``share`` of the acquisition time on the B1 map."""
    acq = budget - cost_afi[0] - n_spgr * cost_spgr[0] - 0.05
    if acq < cost_afi[1] + n_spgr * cost_spgr[1]:
        return None
    n_a = int(np.clip(math.floor(share * acq / cost_afi[1]), 1, 64))
    n_s = int(np.clip(math.floor((acq - n_a * cost_afi[1]) / (n_spgr * cost_spgr[1])), 1, 64))
    return n_a, n_s


def _pipeline(sc, afi_flip, flips, n_afi, n_spgr, afi=AFI, spgr=SPGR, flips_are_actual=False, b1_fixed=None):
    """B1 map + variable flip angle + joint fit. ``flips_are_actual``: the flip angles are targets for the angle the
    spins see, so the nominal angles are set from the B1 just measured (rounded to 0.5 degrees)."""
    card = sc.info()
    rois, ids = _rois(card), [r["id"] for r in card["task"]["regions"]]
    meas, ys, ws = [], [], []
    b1_med = 1.0
    if n_afi:
        r = sc.scan(dict(afi, flip_deg=afi_flip, averages=n_afi), rois)
        v, n = _afi_signed(r), afi["tr_ratio"]
        meas.append(dict(kind="afi", flip=afi_flip * D2R, tr=r["realised"]["tr_ms"], tr2=r["realised"]["tr2_ms"]))
        ys += [v[0], v[1]]
        ws += [math.sqrt(n_afi)] * 2
        ratio = np.clip(v[1] / np.maximum(v[0], 1e-12), -0.99, 1.0)
        b1_first = np.degrees(np.arccos(np.clip((ratio * n - 1) / (n - ratio), -1, 1))) / afi_flip
        b1_med = float(np.clip(np.median(b1_first), 0.5, 1.7))
    nominal = [float(np.clip(round(2 * f / b1_med) / 2 if flips_are_actual else f, 1.0, 90.0)) for f in flips]
    for f in nominal:
        r = sc.scan(dict(spgr, flip_deg=f, averages=n_spgr), rois)
        meas.append(dict(kind="spgr", flip=f * D2R, tr=r["realised"]["tr_ms"]))
        ys.append(_values(r)[0])
        ws.append(math.sqrt(n_spgr))
    y, w = np.array(ys), np.array(ws)
    return _answer(ids, [fit_region(meas, y[:, i], w, b1_fixed)[0] for i in range(len(ids))])


# ------------------------------------------------------------------ the policies
def reference(sc):
    """Map B1 first, set the gradient-echo flip angles from it, spend what is left on averages, fit jointly."""
    ca, cs = _cost(sc, dict(AFI, flip_deg=AFI_FLIP)), _cost(sc, dict(SPGR, flip_deg=10.0))
    n_a, n_s = _allocate(sc.budget()["remaining_s"], ca, cs, len(TARGETS))
    return _pipeline(sc, AFI_FLIP, TARGETS, n_a, n_s, flips_are_actual=True)


def prior(sc):
    return _prior_answer(sc.info())


def blind_vfa(sc):
    cs = _cost(sc, dict(SPGR, flip_deg=10.0))
    n = int(np.clip(math.floor((sc.budget()["remaining_s"] - 2 * cs[0] - 0.05) / (2 * cs[1])), 1, 64))
    return _pipeline(sc, None, TARGETS, 0, n, b1_fixed=1.0)


def recipe_candidates(min_budget):
    """Constant protocols that fit the smallest budget of a level (the recipe cannot look at the instance)."""
    out = []
    for afi_flip in (60.0, 70.0, 80.0):
        for flips in ((4.0, 20.0), (3.0, 15.0), (5.0, 25.0), (3.0, 22.0), (3.0, 10.0, 25.0), (4.0, 12.0, 30.0)):
            for share in (0.45, 0.6, 0.75):
                out.append(dict(afi_flip=afi_flip, flips=flips, share=share, min_budget=min_budget))
    return out


def fixed_recipe(recipe, scale_to_budget=False):
    def policy(sc):
        ca, cs = _cost(sc, dict(AFI, flip_deg=recipe["afi_flip"])), _cost(sc, dict(SPGR, flip_deg=10.0))
        budget = sc.budget()["remaining_s"] if scale_to_budget else recipe["min_budget"]
        alloc = _allocate(budget, ca, cs, len(recipe["flips"]), recipe["share"])
        if alloc is None:
            return _prior_answer(sc.info())
        return _pipeline(sc, recipe["afi_flip"], recipe["flips"], alloc[0], alloc[1])
    return policy


DEFAULT_RECIPE = dict(afi_flip=70.0, flips=(4.0, 20.0), share=0.6, min_budget=45.0)


def brute_force(sc):
    """Uniform sweep at console defaults: every third scan a B1 scan, flip angles spread evenly, one average each."""
    card = sc.info()
    rois, ids = _rois(card), [r["id"] for r in card["task"]["regions"]]
    afi, spgr = dict(sequence="afi", tr_ms=20.0), dict(sequence="spgr", tr_ms=15.0)
    ca, cs = sc.plan(dict(afi, flip_deg=60.0))["scan_s"], sc.plan(dict(spgr, flip_deg=10.0))["scan_s"]
    budget, kinds = sc.budget()["remaining_s"], []
    while budget >= (ca if len(kinds) % 3 == 0 else cs):
        kinds.append("afi" if len(kinds) % 3 == 0 else "spgr")
        budget -= ca if kinds[-1] == "afi" else cs
    n_a, n_s = kinds.count("afi"), kinds.count("spgr")
    fa = iter(np.linspace(30.0, 90.0, n_a + 2)[1:-1])
    fs = iter(np.linspace(2.0, 90.0, n_s + 2)[1:-1])
    meas, ys, ws = [], [], []
    for k in kinds:
        if k == "afi":
            f = float(next(fa))
            r = sc.scan(dict(afi, flip_deg=f), rois)
            meas.append(dict(kind="afi", flip=f * D2R, tr=r["realised"]["tr_ms"], tr2=r["realised"]["tr2_ms"]))
            ys += list(_afi_signed(r))
            ws += [1.0, 1.0]
        else:
            f = float(next(fs))
            r = sc.scan(dict(spgr, flip_deg=f), rois)
            meas.append(dict(kind="spgr", flip=f * D2R, tr=r["realised"]["tr_ms"]))
            ys.append(_values(r)[0])
            ws.append(1.0)
    if not meas:
        return _prior_answer(card)
    y, w = np.array(ys), np.array(ws)
    return _answer(ids, [fit_region(meas, y[:, i], w)[0] for i in range(len(ids))])


def fit_ir(ti, y):
    """Three-parameter fit  s = a - b exp(-TI / T1)  to magnitude data with unknown polarity (every sign pattern that
    switches once is tried; a and b are linear)."""
    ti, y = np.asarray(ti, float), np.asarray(y, float)
    order = np.argsort(ti)
    ti, y = ti[order], y[order]
    t1g = np.geomspace(100.0, 6000.0, 600)
    e = np.exp(-ti[None, :] / t1g[:, None])                 # [grid, n]
    n, se, see = len(ti), e.sum(1), (e * e).sum(1)
    det = n * see - se * se
    best = (np.inf, task.T1_PRIOR_MS)
    for k in range(len(ti) + 1):                           # the first k points are negative
        s = np.where(np.arange(len(ti)) < k, -y, y)
        ss, ses = s.sum(), e @ s
        a = (see * ss - se * ses) / det
        b = (se * ss - n * ses) / det
        sse = ((a[:, None] - b[:, None] * e - s[None, :]) ** 2).sum(1)
        sse = np.where((a > 0) & (b > a), sse, np.inf)
        i = int(np.argmin(sse))
        if sse[i] < best[0]:
            best = (float(sse[i]), float(t1g[i]))
    return best[1]


def ir_only(sc):
    """Inversion recovery at the cheapest useful settings; needs at least three inversion times for its fit."""
    card = sc.info()
    rois, ids = _rois(card), [r["id"] for r in card["task"]["regions"]]
    budget = sc.budget()["remaining_s"]
    for matrix, tr, tis in ((64, 6000.0, (60.0, 250.0, 600.0, 1300.0, 2800.0, 5500.0)), (64, 3000.0, (60.0, 300.0, 800.0, 1800.0)),
                            (48, 2500.0, (60.0, 500.0, 1500.0)), (48, 1000.0, (40.0, 300.0, 900.0))):
        p = sc.plan(dict(sequence="ir", matrix=matrix, tr_ms=tr, ti_ms=tis[0]))
        if p["accept"] and p["scan_s"] * len(tis) <= budget:
            data = [_values(sc.scan(dict(sequence="ir", matrix=matrix, tr_ms=tr, ti_ms=t), rois))[0] for t in tis]
            y = np.array(data)
            return _answer(ids, [fit_ir(tis, y[:, i]) for i in range(len(ids))])
    p = sc.plan(dict(sequence="ir", matrix=48, tr_ms=1000.0, ti_ms=300.0))
    if p["accept"] and p["scan_s"] <= budget:              # one inversion time fits: it is in the log, and useless
        sc.scan(dict(sequence="ir", matrix=48, tr_ms=1000.0, ti_ms=300.0), rois)
    return _prior_answer(card)


def wrong_units(sc):
    a = reference(sc)
    return {"units": "ms", "t1_ms": {k: v / 1000.0 for k, v in a["t1_ms"].items()}}


def short_tr_ideal(sc):
    """The same B1-corrected pipeline run "as fast as possible": console-default spoilers and RF phase increments,
    repetition times at the console minimum, textbook dual-TR ratio. Trusts the ideal-spoiling formulas."""
    afi = dict(COMMON, sequence="afi", tr_ratio=5)
    spgr = dict(COMMON, sequence="spgr")
    afi["tr_ms"] = sc.plan(dict(afi, tr_ms=200.0, flip_deg=60.0))["timing"]["tr_min_ms"]
    spgr["tr_ms"] = sc.plan(dict(spgr, tr_ms=200.0, flip_deg=10.0))["timing"]["tr_min_ms"]
    ca, cs = _cost(sc, dict(afi, flip_deg=AFI_FLIP)), _cost(sc, dict(spgr, flip_deg=10.0))
    n_a, n_s = _allocate(sc.budget()["remaining_s"], ca, cs, 2)
    return _pipeline(sc, 60.0, TARGETS, n_a, n_s, afi=afi, spgr=spgr, flips_are_actual=True)


def guessable_seeds(task_id, n_small=300):
    """What an attacker without the secret can try: small integers and hashes of public strings."""
    out = [f"{i:032x}" for i in range(n_small)]
    for s in (task_id, task.FAMILY, task_id + task.FAMILY, "0", "seed", "mrgym"):
        out += [hashlib.sha256(s.encode()).hexdigest()[:32], hashlib.md5(s.encode()).hexdigest()]
    return out


def seed_attacker(sc, candidates=None, diagnostics=None):
    """One cheap scan (so the log is not empty), then: (1) rebuild candidate worlds with the PUBLIC generator from
    guessable seeds and accept one whose region layout equals the task card's; (2) failing that, replay the answer of
    another instance; (3) repeat one scan and record whether the noise repeats."""
    card = sc.info()
    regions, ids = card["task"]["regions"], [r["id"] for r in card["task"]["regions"]]
    prot = dict(sequence="spgr", matrix=48, tr_ms=10.0, flip_deg=8.0)
    a = sc.scan(prot, _rois(card))
    if diagnostics is not None and sc.budget()["remaining_s"] >= a["cost_s"]:
        b = sc.scan(prot, _rois(card))
        diagnostics["repeat_scan_identical"] = bool(np.allclose(_values(a), _values(b), rtol=0, atol=0))
    key = sorted((r["y_mm"], r["x_mm"]) for r in regions)
    other = None
    for seed in candidates if candidates is not None else guessable_seeds(card["task_id"]):
        for level in task.LEVELS.values():
            w = W.make_world(seed, level["world_spec"], layout_only=True)
            pub = W.regions_public(w)
            if sorted((r["y_mm"], r["x_mm"]) for r in pub) == key:
                truth, at = W.truth(w), {(r["y_mm"], r["x_mm"]): r["id"] for r in pub}
                if diagnostics is not None:
                    diagnostics["world_recovered"] = True
                return _answer(ids, [truth[at[(r["y_mm"], r["x_mm"])]]["t1_ms"] for r in regions])
            if other is None and len(pub) == len(regions):
                other = [W.truth(w)[r["id"]]["t1_ms"] for r in pub]
    if diagnostics is not None:
        diagnostics["world_recovered"] = False
    return _answer(ids, other) if other else _prior_answer(card)


def tune_recipe(cfgs, candidates, draws=(0, 1)):
    """"Tuned on other worlds": the candidate with the smallest median error over the tuning instances."""
    worlds = [W.make_world(c["seed"], c["world_spec"]) for c in cfgs]
    engines = [kspace.Engine(w) for w in worlds]
    best = None
    for cand in candidates:
        errs = []
        for cfg, w, eng in zip(cfgs, worlds, engines):
            for d in draws:
                answer, session, _ = run_policy(cfg, fixed_recipe(cand), d, eng)
                e = task.score(cfg, answer, 1.0, session.n_scans, world=w)["error"]
                errs.append(e if e is not None and session.n_scans else 10.0)
        m = float(np.median(errs))
        if best is None or m < best[0]:
            best = (m, cand)
    return dict(best[1], tuned_median_error=best[0])


# ------------------------------------------------------------------ running and admission
def run_policy(cfg, policy, noise_draw=0, engine=None, budget_scale=1.0):
    """Play one episode in process. Returns (answer or None, session, error text or None)."""
    session = core.ScannerSession(dict(cfg, noise_draw=noise_draw, budget_s=cfg["budget_s"] * budget_scale), engine=engine)
    try:
        return policy(core.LocalScanner(session)), session, None
    except Exception as e:  # noqa: BLE001  a crashing policy is a failed episode
        return None, session, f"{type(e).__name__}: {e}"


def tolerance_from(errors):
    """Pass tolerance from the reference's error distribution over noise draws: twice its median, and at least 1.25
    times its 95th percentile (so the reference passes with margin both when bias and when noise dominates)."""
    e = np.asarray(errors, float)
    return float(max(2.0 * np.median(e), 1.25 * np.quantile(e, 0.95)))


ADMISSION_CONTROLS = ("prior", "blind_vfa", "fixed_recipe", "brute_force", "ir_only", "wrong_units", "seed_attacker")
REPORTED_CONTROLS = ("open_loop", "short_tr_ideal", "ir_unpriced")


def policies(recipe=None):
    recipe = recipe or DEFAULT_RECIPE
    return dict(reference=(reference, 1.0), prior=(prior, 1.0), blind_vfa=(blind_vfa, 1.0),
                fixed_recipe=(fixed_recipe(recipe), 1.0), brute_force=(brute_force, 1.0), ir_only=(ir_only, 1.0),
                wrong_units=(wrong_units, 1.0), seed_attacker=(seed_attacker, 1.0),
                open_loop=(fixed_recipe(recipe, scale_to_budget=True), 1.0), short_tr_ideal=(short_tr_ideal, 1.0),
                ir_unpriced=(ir_only, 40.0))


def admit(cfg, recipe=None, n_cal=32, n_test=32, n_ctrl=8):
    """Per-instance admission. Reference: tolerance from ``n_cal`` noise draws, then it must pass on fresh draws with
    margin. Every admission control must fail on every one of its ``n_ctrl`` draws.

    Returns a JSON-serialisable report; ``admitted`` and ``reasons`` summarise it.
    """
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    eng = kspace.Engine(world, cache=256)
    pol = policies(recipe)

    def play(name, draw):
        fn, scale = pol[name]
        answer, session, err = run_policy(cfg, fn, draw, eng, scale)
        return answer, session, err

    def errors(name, draws):
        out = []
        for d in draws:
            answer, session, err = play(name, d)
            out.append(task.score(cfg, answer, 1.0, session.n_scans, world=world)["error"])
        return out

    cal = errors("reference", range(n_cal))
    rep = dict(task_id=cfg["task_id"], level=cfg["level"], budget_s=cfg["budget_s"], regions=len(cfg["task"]["regions"]),
               reasons=[], controls={})
    if any(e is None for e in cal):
        rep.update(admitted=False, reasons=["reference failed to deliver"])
        return rep
    tol = tolerance_from(cal)
    e_prior = task.score(cfg, _prior_answer(dict(task=cfg["task"])), tol, 1, world=world)["e_prior"]
    rep.update(tolerance=tol, e_prior=e_prior, reference_cal_median=float(np.median(cal)),
               reference_cal_q95=float(np.quantile(cal, 0.95)))

    def evaluate(name, draws):
        res, used = [], []
        for d in draws:
            answer, session, err = play(name, d)
            s = task.score(cfg, answer, tol, session.n_scans, world=world)
            res.append(s)
            used.append(session.used)
        errs = [s["error"] for s in res if s["error"] is not None]
        return dict(pass_rate=float(np.mean([s["passed"] for s in res])), reward_mean=float(np.mean([s["reward"] for s in res])),
                    error_median=float(np.median(errs)) if errs else None, scan_s_median=float(np.median(used)),
                    gate_fail=[s["diagnostic"] for s in res if not s["gate_deliverable"]][:1]
                    + (["no scan in the log"] if not all(s["gate_measured"] for s in res) else []))

    ref = evaluate("reference", range(n_cal, n_cal + n_test))
    rep["controls"]["reference"] = ref
    if tol >= 0.5 * e_prior:
        rep["reasons"].append("tolerance is not well below the prior policy's error")
    if ref["pass_rate"] < 0.9 or ref["error_median"] > 0.6 * tol:
        rep["reasons"].append("reference does not pass fresh noise draws with margin")
    for name in ADMISSION_CONTROLS + REPORTED_CONTROLS:
        deterministic = name in ("prior",)
        rep["controls"][name] = evaluate(name, range(1 if deterministic else n_ctrl))
        if name in ADMISSION_CONTROLS and rep["controls"][name]["pass_rate"] > 0:
            rep["reasons"].append(f"{name} passes")
    # the zero of the score: the prior answer delivered after one token scan earns exactly 0
    answer, session, _ = run_policy(cfg, lambda sc: (sc.scan(dict(sequence="spgr", matrix=48, tr_ms=10.0, flip_deg=8.0)),
                                                     _prior_answer(sc.info()))[1], 0, eng)
    rep["reward_prior_after_one_scan"] = task.score(cfg, answer, tol, session.n_scans, world=world)["reward"]
    others = [n for n in ADMISSION_CONTROLS if n != "fixed_recipe" and rep["controls"][n]["pass_rate"] > 0]
    rep["admitted_without_fixed_recipe_rule"] = not others and not [x for x in rep["reasons"] if not x.endswith(" passes")]
    rep["adaptivity_gap"] = dict(
        fixed_over_reference=rep["controls"]["fixed_recipe"]["error_median"] / ref["error_median"],
        open_loop_over_reference=rep["controls"]["open_loop"]["error_median"] / ref["error_median"])
    rep["admitted"] = not rep["reasons"]
    return rep
