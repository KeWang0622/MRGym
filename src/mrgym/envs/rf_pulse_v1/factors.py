"""The duration factors x_k of ``Scanner/RFPulse-v1``, and the best constant recipe, on worlds of the split ``tune``.

    python -m mrgym.envs.rf_pulse_v1.factors --write [--n 200] [--workers 4]

The rule, fixed before any gate was run on a factor: on ``n`` worlds of the split ``tune`` (seeds ``factor:<i>``) the
reference policy plays the level; among the episodes whose design meets every hard requirement (limits, mask,
refocused signal) the largest duration / bound is taken, 5 % are added, and the result is rounded up to 0.05:

    x_k = ceil(1.05 x max(duration / bound) / 0.05) x 0.05

The gate measures on the split ``gate``; nothing of it enters here. The stored file also holds, per level, the share
of those worlds on which the best constant recipe of ``controls.recipes`` passes with that factor (chosen with
hindsight on the same worlds: an upper estimate of what a constant can do).
"""
from __future__ import annotations

import argparse
import datetime
import importlib.resources
import json
import math
import multiprocessing
import os

import numpy as np

ENV_ID = "Scanner/RFPulse-v1"
MARGIN, STEP = 1.05, 0.05
HARD = ("deliverable_valid", "limits", "in_slice", "out_of_slice", "phase", "not_below_bound", "log_replays_within_budget", "measured")


def factor_of(ratios) -> float:
    return round(math.ceil(MARGIN * max(ratios) / STEP - 1e-9) * STEP, 2)


def _reference(args):
    import mrgym
    from mrgym import policies
    level, i = args
    r = policies.run(mrgym.make(ENV_ID, level=level, seed=f"factor:{i}", split="tune"), "reference")
    s = r.info.get("score", {})
    checks = s.get("checks", {})
    return dict(hard=bool(checks) and all(checks.get(k, False) for k in HARD), ratio=s.get("duration_over_bound"), outcome=s.get("outcome"),
                limited_by=(s.get("bound") or {}).get("limited_by"), policy_error=r.info.get("policy_error"))


def _constants(args):
    """Per recipe, whether its design passes on world ``i`` with the factor ``x`` (the verifier's rule, no episode)."""
    import mrgym
    from mrgym import gate, policies  # noqa: F401
    from mrgym.scanner import world as W

    from . import controls, task
    level, i, x = args
    cfg = gate._instance(mrgym.spec(ENV_ID), level, f"factor:{i}", "tune")
    cfg = dict(cfg, task=dict(cfg["task"], duration_factor=x))
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    lv = task.LEVELS[level]
    out = []
    for recipe in controls.recipes(len(lv["hidden"]) > 1, lv["power"] == "binding"):
        try:
            out.append(controls.passes(cfg, world, controls.recipe_call(cfg["task"], recipe)))
        except Exception:  # noqa: BLE001  a recipe that cannot be built does not pass
            out.append(False)
    return out


def run(n: int = 200, workers: int = 4, levels=(1, 2, 3)) -> dict:
    import mrgym

    from . import controls, task
    out = dict(env_id=ENV_ID, engine=mrgym.ENGINE_ID, date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
               command=f"python -m mrgym.envs.rf_pulse_v1.factors --write --n {n}", split="tune", seeds=f"factor:0 .. factor:{n - 1}",
               rule=f"x = ceil({MARGIN:g} x max(duration / bound of the reference's designs that meet every hard requirement) / {STEP:g}) x {STEP:g}",
               levels={})
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for lv in levels:
            rows = pool.map(_reference, [(lv, i) for i in range(n)], chunksize=1)
            ratios = np.array([r["ratio"] for r in rows if r["hard"]])
            x = factor_of(ratios)
            table = np.array(pool.map(_constants, [(lv, i, x) for i in range(n)], chunksize=1))
            recipes = controls.recipes(len(task.LEVELS[lv]["hidden"]) > 1, task.LEVELS[lv]["power"] == "binding")
            best = int(np.argmax(table.sum(0)))
            out["levels"][str(lv)] = dict(
                factor=x, n=n, hard_met=int(ratios.size), ratio_min=round(float(ratios.min()), 4), ratio_median=round(float(np.median(ratios)), 4),
                ratio_q95=round(float(np.quantile(ratios, 0.95)), 4), ratio_max=round(float(ratios.max()), 4),
                rf_limited=int(sum(r["limited_by"] == "rf" for r in rows)), outcomes_not_hard=sorted({str(r["outcome"]) for r in rows if not r["hard"]}),
                policy_errors=sorted({str(r["policy_error"]) for r in rows if r["policy_error"]})[:3],
                constant=dict(recipes=len(recipes), best=recipes[best], passed=int(table[:, best].sum()), rate=round(float(table[:, best].mean()), 4),
                              any_recipe_passed_on=int(table.any(1).sum())))
    return out


def path() -> str:
    return os.path.join(str(importlib.resources.files("mrgym.envs").joinpath("results")), "Scanner-RFPulse-v1.factors.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    out = run(a.n, a.workers)
    for lv, s in out["levels"].items():
        c = s["constant"]
        print(f"level {lv}: factor {s['factor']:g}  (duration / bound of the reference: min {s['ratio_min']:g}, median {s['ratio_median']:g}, q95 {s['ratio_q95']:g}, "
              f"max {s['ratio_max']:g}; {s['hard_met']} of {s['n']} designs meet every hard requirement; RF-limited bound on {s['rf_limited']})")
        print(f"         best constant of {c['recipes']} recipes: {c['passed']} of {s['n']} ({100 * c['rate']:.1f} %): {c['best']}")
    if a.write:
        with open(path(), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print("wrote", path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
