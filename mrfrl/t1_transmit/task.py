"""F1 ``t1_transmit``: instances, the agent-facing instruction, and the verifier.

An instance is an episode config for ``mrfrl.scanner`` (hidden: seed, world spec) plus a public task card (regions,
budget, deliverable). The agent delivers ``answer.json``:

    {"units": "ms", "t1_ms": {"R1": 812.0, "R2": 1530.0, ...}}

Reward (memo 17 / 19): hard gates, then a continuous score of the final relative T1 error.
    gates   log replays from the seed and stays within budget; at least one accepted scan; deliverable valid (JSON
            object, units "ms", exactly the region ids, finite values inside the physical range)
    error   E = sqrt(mean over regions of (T1_delivered / T1_true - 1)^2)
    score   clip( ln(E_prior / E) / ln(E_prior / tol), 0, 1 )
            E_prior = E of the zero-measurement answer (the prior's geometric mean in every region) on this instance,
            tol     = tolerance measured at admission from the reference policy's error distribution over noise draws
    reward  = score if all gates pass else 0.   pass := E <= tol (score = 1).
Process quantities (scans, seconds used, refusals) are reported, never paid.
"""
import json
import math
import os

import numpy as np

from ..phys import world as W
from ..scanner import core

FAMILY = "t1_transmit"
T1_RANGE_MS = (300.0, 2500.0)               # told to the agent
T1_PRIOR_MS = math.sqrt(T1_RANGE_MS[0] * T1_RANGE_MS[1])
T1_VALID_MS = (30.0, 20000.0)               # anything outside is not a T1 in milliseconds
STRONG = dict(b1_shape_amp=(0.14, 0.24))
LEVELS = {
    1: dict(world_spec=dict(b1_shape_amp=(0.01, 0.04)), budget_s=(60.0, 100.0),
            summary="mild B1 non-uniformity (one global transmit-gain error dominates)"),
    2: dict(world_spec=dict(STRONG), budget_s=(60.0, 100.0),
            summary="strong B1 non-uniformity (every region has its own flip-angle error)"),
    3: dict(world_spec=dict(STRONG, insert_t2_ms=(80.0, 250.0), insert_adc_um2_ms=(0.6, 1.2)), budget_s=(45.0, 60.0),
            summary="strong B1 non-uniformity, tight budget, long T2 and slow diffusion (spoiling is hard)"),
}


def make_instance(seed, level, task_id=None):
    """Episode config. The task id is an independent random token; nothing public is derived from the seed except
    through the world it generates (region positions) and the budget."""
    spec = LEVELS[level]["world_spec"]
    world = W.make_world(seed, spec)
    lo, hi = LEVELS[level]["budget_s"]
    budget = float(np.round(W.rng_for(seed, "task").uniform(lo, hi)))
    task = dict(family=FAMILY, regions=W.regions_public(world), t1_range_ms=list(T1_RANGE_MS),
                deliverable={"file": "answer.json", "format": {"units": "ms", "t1_ms": {"<region id>": "<number>"}}})
    return dict(task_id=task_id or "t1tx-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=spec,
                budget_s=budget, max_requests=48, task=task)


def instruction(cfg):
    regions = "\n".join(f"| {r['id']} | {r['y_mm']:.2f} | {r['x_mm']:.2f} | {r['radius_mm']:.2f} |" for r in cfg["task"]["regions"])
    return f"""# Measure T1 in marked regions with an uncalibrated transmit field

You operate a virtual MR scanner through `mrscan.py` (see `scanner.md`, the instrument datasheet). An object of unknown
composition lies in the field of view. The transmit field is not calibrated: the flip angle a spin experiences is the
nominal flip angle times an unknown, position-dependent factor.

**Deliver** the longitudinal relaxation time T1 of each marked region, in milliseconds, as `answer.json`:

    {{"units": "ms", "t1_ms": {{"{cfg['task']['regions'][0]['id']}": 0.0, ...}}}}

**Regions** (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):

| id | y_mm | x_mm | radius_mm |
|---|---|---|---|
{regions}

**Budget:** {cfg['budget_s']:.0f} s of scan time (the scanner counts it; `plan` is free and shows what a scan would cost).
At most {cfg['max_requests']} requests in total (plans and scans).

**Sequences:** `spgr` (RF-spoiled gradient echo), `afi` (dual-TR actual flip-angle imaging), `ir` (inversion recovery).
T1 values lie between {T1_RANGE_MS[0]:.0f} and {T1_RANGE_MS[1]:.0f} ms. Nothing else about the object or the system is known.

**Scoring:** your answer is compared with the true T1 of every region (root-mean-square relative error). The scan log
kept by the scanner must show that you measured. An answer in the wrong unit is wrong.
"""


def error_metric(t1_est, t1_true):
    e = np.array([t1_est[k] / t1_true[k] - 1.0 for k in t1_true])
    return float(np.sqrt(np.mean(e * e)))


def check_deliverable(answer, region_ids):
    """Returns (t1 dict or None, diagnostic)."""
    if not isinstance(answer, dict) or set(answer) != {"units", "t1_ms"}:
        return None, 'answer must be {"units": "ms", "t1_ms": {...}}'
    if answer["units"] != "ms":
        return None, f"units must be \"ms\" (got {answer['units']!r}); deliver T1 in milliseconds"
    t1 = answer["t1_ms"]
    if not isinstance(t1, dict) or set(t1) != set(region_ids):
        return None, f"t1_ms must hold exactly the regions {sorted(region_ids)}"
    for k, v in t1.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return None, f"t1_ms[{k}] is not a finite number"
        if not T1_VALID_MS[0] <= v <= T1_VALID_MS[1]:
            hint = " (values of this size look like seconds)" if 0 < v < T1_VALID_MS[0] else ""
            return None, f"t1_ms[{k}] = {v} is outside {T1_VALID_MS} ms{hint}"
    return {k: float(v) for k, v in t1.items()}, "ok"


def score(cfg, answer, tolerance, measured, log_ok=True, world=None):
    """Reward and details for one episode. ``measured``: number of accepted scans in the service's log."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    truth = {k: v["t1_ms"] for k, v in W.truth(world).items()}
    e_prior = error_metric({k: T1_PRIOR_MS for k in truth}, truth)
    t1, why = check_deliverable(answer, truth)
    out = dict(reward=0.0, passed=False, gate_log=bool(log_ok), gate_measured=bool(measured > 0),
               gate_deliverable=t1 is not None, diagnostic=why, error=None, e_prior=e_prior, tolerance=tolerance)
    if t1 is None:
        return out
    e = error_metric(t1, truth)
    if e <= tolerance:
        s = 1.0
    elif tolerance >= e_prior or e >= e_prior:
        s = 0.0
    else:
        s = float(np.clip(math.log(e_prior / e) / math.log(e_prior / tolerance), 0.0, 1.0))
    out.update(error=e, score=s, passed=bool(e <= tolerance and out["gate_log"] and out["gate_measured"]),
               per_region={k: round(t1[k] / truth[k] - 1.0, 5) for k in truth})
    if out["gate_log"] and out["gate_measured"]:
        out["reward"] = s
    return out


def verify(cfg, log_dir, answer_path, tolerance):
    """Verifier entry for a real episode: replays the scanner log, then scores the deliverable."""
    events = core.read_log(os.path.join(log_dir, core.LOG_NAME))
    rep = core.replay(cfg, events)
    try:
        with open(answer_path) as f:
            answer = json.load(f)
    except Exception as e:  # noqa: BLE001
        answer = None
        rep["answer_error"] = f"{type(e).__name__}: {e}"
    out = score(cfg, answer, tolerance, rep["scans"], rep["ok"] and not rep["over_budget"])
    out["replay"] = rep
    return out
