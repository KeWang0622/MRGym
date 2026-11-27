"""The admission gate: one verdict per (environment, level), measured on fresh instances.

A level is ADMITTED when, on ``n`` instances that nothing was tuned on (the public split ``gate``, seeds ``0 .. n-1``),

  solvable     the reference policy passes: one-sided 95 % lower bound of its pass rate >= ``REF_MIN`` (0.90)
  not blind    every policy that ignores the object fails: one-sided 95 % upper bound of the pass rate of each
               must-fail policy <= ``BLIND_MAX`` (0.10). The must-fail list of a family holds the no-scan answer, the
               wrong unit, a sweep without a model, and the best answer that uses only what the task states, tuned on
               ``TUNING_SETS`` disjoint sets of other worlds (the public split ``tune``, seeds ``<set>:<i>``)
  the claim    every ablation the family lists for this level (the reference with one decision removed) loses at
               least ``SEP_MIN`` (0.25) of pass rate against the reference, as the lower bound of the paired difference

Bounds are Clopper-Pearson (exact binomial). With n = 300 a must-fail policy is cleared when it passes at most 21
instances and the reference needs at least 279. If a single condition is truly violated the level is admitted with
probability at most 5 %. There is no per-instance verdict: every instance of an admitted level is served.

What the gate shows: the listed policies behave as required on this engine. It does not prove that no shortcut exists.

    mrgym gate Scanner/NullTissue-v0 --n 300 --write
"""
from __future__ import annotations

import datetime
import importlib
import importlib.resources
import json
import multiprocessing
import os
import time

import numpy as np

from .stats.gate import gap_lower_bound, lower_bound, upper_bound  # the bound functions live in mrgym.stats

SCHEMA = "mrgym-gate/2"
ALPHA, REF_MIN, BLIND_MAX, SEP_MIN = 0.05, 0.90, 0.10, 0.25
TUNING_SETS, TUNING_WORLDS = 3, 16


# ------------------------------------------------------------------------------------------------ storage
def _path(env_id: str, directory: str | None = None) -> str:
    d = directory or str(importlib.resources.files("mrgym.envs").joinpath("results"))
    return os.path.join(d, env_id.replace("/", "-") + ".gate.json")


def load(env_id: str) -> dict | None:
    try:
        with open(_path(env_id), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError):
        return None


def write(result: dict, directory: str | None = None) -> str:
    path = _path(result["env_id"], directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    return path


def level_status(env_id: str, level: int) -> dict | None:
    """What an episode's ``info["gate"]`` carries: the stored verdict of its level (None when nothing is stored)."""
    g = load(env_id)
    lv = (g or {}).get("levels", {}).get(str(level))
    if not lv:
        return None
    return dict(verdict=lv["verdict"], reference_lower95=lv["reference"]["lower95"], blind_upper95=lv["blind_ceiling"]["upper95"],
                engine=g["engine"], n=lv["n"])


def status_line(spec, g: dict | None) -> str:
    """One line on the claim of an id and its evidence. A stored result counts only on the engine it was measured on:
    a result from another engine is named as that and its verdict is not repeated. Where no result is stored on this
    engine and the id is known not to work as stated on it (``spec.not_ready``), the line says so and why."""
    import mrgym
    if spec.kind == "sandbox":
        return "SANDBOX: no reward"
    known = f". NOT READY on this engine: {spec.not_ready}" if getattr(spec, "not_ready", "") else ""
    if not g:
        return f"{spec.kind.upper()}: the gate has not been run (no stored result)" + known
    if g.get("engine") != mrgym.ENGINE_ID:
        return (f"{spec.kind.upper()}: the gate has not been run on this engine ({mrgym.ENGINE_ID}); the stored result is from "
                f"another engine ({g.get('engine')}) and says nothing about this one") + known
    lv = g["levels"]
    adm = [k for k, v in lv.items() if v["verdict"] == "admitted"]
    if len(adm) == len(lv):
        ref = min(v["reference"]["lower95"] for v in lv.values())
        blind = max(v["blind_ceiling"]["upper95"] for v in lv.values())
        return (f"ADMITTED: on {min(v['n'] for v in lv.values())} fresh instances per level the reference passes >= {100 * ref:.0f} %, "
                f"no policy that ignores the object passes more than {100 * blind:.0f} % (95 % bounds)")
    first = next(v for v in lv.values() if v["verdict"] != "admitted")
    return f"PRACTICE: level(s) {', '.join(adm) or 'none'} admitted; " + first["reasons"][0]


# ------------------------------------------------------------------------------------------------ running
def _instance(spec, level, seed, split="tune"):
    from . import keys
    from .env import derive_seed
    task = importlib.import_module(spec.family + ".task")
    seed_hex = derive_seed(spec.id, level, split, seed, environ={})
    return keys.make_instance(task, seed_hex, level, keys.task_id_for(spec, seed_hex, split), split)


def _tune(args):
    import mrgym
    env_id, level, k, n_worlds = args
    spec = mrgym.spec(env_id)
    controls = importlib.import_module(spec.family + ".controls")
    return controls.tune([_instance(spec, level, f"{k}:{i}", "tune") for i in range(n_worlds)])


def _play(args):
    """One instance, every policy: {policy: dict(passed, error, used_s, policy_error)}."""
    import mrgym

    from . import policies
    env_id, level, seed, recipes = args
    spec = mrgym.spec(env_id)
    controls = importlib.import_module(spec.family + ".controls")
    names = [controls.REFERENCE, *[n for n in controls.MUST_FAIL if n != "fixed_recipe"], *controls.ABLATIONS, *controls.REPORTED]
    jobs = [(n, n, None) for n in names] + [(f"fixed_recipe#{k}", "fixed_recipe", r) for k, r in enumerate(recipes)]
    out = {}
    for label, name, recipe in jobs:
        t0 = time.time()
        r = policies.run(mrgym.make(env_id, level=level, seed=seed, split="gate"), name, recipe=recipe)
        s = r.info.get("score", {})
        out[label] = dict(passed=bool(r.reward >= spec.reward_threshold), error=s.get("error"),
                          used_s=r.info.get("budget", {}).get("used_s"), policy_error=r.info.get("policy_error"),
                          seconds=round(time.time() - t0, 3))
    return out


def decide(passes: dict, controls, level: int, n: int) -> dict:
    """The verdict of one level from the instance x policy pass matrix (``passes``: {policy label: bool array})."""
    ref = passes[controls.REFERENCE]
    k_ref = int(ref.sum())
    out = dict(n=n, reference=dict(passed=k_ref, rate=round(k_ref / n, 4), lower95=round(lower_bound(k_ref, n), 4)),
               must_fail=[], ablations=[], reported=[], reasons=[])
    if out["reference"]["lower95"] < REF_MIN:
        out["reasons"].append(f"the reference passes {k_ref} of {n} (lower bound {100 * out['reference']['lower95']:.1f} % < {100 * REF_MIN:.0f} %)")
    for label in sorted(k for k in passes if k.split("#")[0] in controls.MUST_FAIL):
        k = int(passes[label].sum())
        row = dict(policy=label, passed=k, rate=round(k / n, 4), upper95=round(upper_bound(k, n), 4))
        row["cleared"] = row["upper95"] <= BLIND_MAX
        out["must_fail"].append(row)
        if not row["cleared"]:
            out["reasons"].append(f"{label} passes {k} of {n} (upper bound {100 * row['upper95']:.1f} % > {100 * BLIND_MAX:.0f} %)")
    worst = max(out["must_fail"], key=lambda r: r["upper95"])
    j = len(out["must_fail"])
    out["blind_ceiling"] = dict(policy=worst["policy"], rate=worst["rate"], upper95=worst["upper95"],
                                upper95_simultaneous=round(upper_bound(worst["passed"], n, ALPHA / j), 4))
    for name, levels in controls.ABLATIONS.items():
        k = int(passes[name].sum())
        gap, lo = gap_lower_bound(ref, passes[name])
        row = dict(policy=name, passed=k, rate=round(k / n, 4), gap=round(gap, 4), gap_lower95=round(lo, 4),
                   required=level in levels)
        row["cleared"] = (not row["required"]) or lo >= SEP_MIN
        out["ablations"].append(row)
        if not row["cleared"]:
            out["reasons"].append(f"{name} loses only {100 * gap:.1f} points against the reference (lower bound {100 * lo:.1f} < {100 * SEP_MIN:.0f})")
    for name in controls.REPORTED:
        k = int(passes[name].sum())
        out["reported"].append(dict(policy=name, passed=k, rate=round(k / n, 4)))
    out["verdict"] = "admitted" if not out["reasons"] else "practice"
    return out


def claim_levels(spec) -> list:
    """The levels an environment's claim covers: all registered levels except those its family marks as practice
    (``task.LEVELS[lv]["kind"] == "practice"``: playable on-ramps that state part of the method). The gate, the suite
    manifests and published tables use these levels."""
    if not spec.family:
        return list(spec.levels)
    table = getattr(importlib.import_module(spec.family + ".task"), "LEVELS", {})
    return [lv for lv in spec.levels if table.get(lv, {}).get("kind") != "practice"]


def run(env_id: str, n: int = 300, levels=None, workers: int = 2, blocks: int = 5) -> dict:
    """Run the gate of ``env_id``. Returns the result dict (see ``write``)."""
    import mrgym
    spec = mrgym.spec(env_id)
    if not spec.family:
        raise mrgym.Error(f"{env_id} is a sandbox: it has no gate")
    controls = importlib.import_module(spec.family + ".controls")
    levels = list(levels or claim_levels(spec))                          # a practice level is outside the admitted claim
    workers = max(1, min(int(workers), 8))
    t0 = time.time()
    from . import keys
    result = dict(schema=SCHEMA, canary=keys.CANARY, env_id=env_id, engine=mrgym.ENGINE_ID, package_version=mrgym.__version__,
                  date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                  command=f"mrgym gate {env_id} --n {n} --write",
                  seeds=dict(gate=f"split gate, seeds 0..{n - 1}", tune=f"split tune, seeds <set 0..{TUNING_SETS - 1}>:<0..{TUNING_WORLDS - 1}>"),
                  thresholds=dict(alpha=ALPHA, reference_lower_min=REF_MIN, blind_upper_max=BLIND_MAX, ablation_gap_lower_min=SEP_MIN),
                  descriptions=dict(controls.DESCRIPTIONS), levels={})
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for lv in levels:
            t1 = time.time()
            recipes = pool.map(_tune, [(env_id, lv, k, TUNING_WORLDS) for k in range(TUNING_SETS)])
            rows = pool.map(_play, [(env_id, lv, i, recipes) for i in range(n)], chunksize=1)
            passes = {k: np.array([r[k]["passed"] for r in rows]) for k in rows[0]}
            out = decide(passes, controls, lv, n)
            size = n // blocks
            verdicts = [decide({k: np.delete(v, np.s_[b * size:(b + 1) * size]) for k, v in passes.items()}, controls, lv, n - size)["verdict"]
                        for b in range(blocks)] if blocks > 1 and n >= 10 * blocks else []   # one block left out leaves nothing
            out["stability"] = dict(leave_one_block_out=verdicts)
            if verdicts and len(set(verdicts + [out["verdict"]])) > 1:
                out["reasons"].append("borderline: the verdict changes when one fifth of the instances is left out")
                out["verdict"] = "practice"
            errs = [r[controls.REFERENCE]["error"] for r in rows if r[controls.REFERENCE]["error"] is not None]
            out["reference"].update(error_median=round(float(np.median(errs)), 5) if errs else None,
                                    error_q95=round(float(np.quantile(errs, 0.95)), 5) if errs else None,
                                    scan_s_median=round(float(np.median([r[controls.REFERENCE]["used_s"] or 0.0 for r in rows])), 1),
                                    policy_errors=sorted({r[controls.REFERENCE]["policy_error"] for r in rows if r[controls.REFERENCE]["policy_error"]})[:3])
            out["fixed_recipes"] = recipes
            out["wall_seconds"] = round(time.time() - t1, 1)
            result["levels"][str(lv)] = out
    result["wall_seconds"] = round(time.time() - t0, 1)
    return result


def text_table(result: dict) -> str:
    out = []
    for lv, s in result["levels"].items():
        out.append(f"== {result['env_id']} level {lv}: {s['verdict'].upper()} on {s['n']} fresh instances ({s['wall_seconds']} s)")
        r = s["reference"]
        out.append(f"   {'reference':<18}{r['passed']:>5}/{s['n']}  {100 * r['rate']:5.1f} %   lower bound {100 * r['lower95']:5.1f} %   "
                   f"error median {r['error_median']} q95 {r['error_q95']}   scan {r['scan_s_median']} s")
        for row in s["must_fail"]:
            out.append(f"   {row['policy']:<18}{row['passed']:>5}/{s['n']}  {100 * row['rate']:5.1f} %   upper bound {100 * row['upper95']:5.1f} %   "
                       f"{'cleared' if row['cleared'] else 'NOT CLEARED'}")
        for row in s["ablations"]:
            need = "required" if row["required"] else "reported"
            out.append(f"   {row['policy']:<18}{row['passed']:>5}/{s['n']}  {100 * row['rate']:5.1f} %   gap {100 * row['gap']:5.1f} points, lower bound "
                       f"{100 * row['gap_lower95']:5.1f}   ({need}{'' if row['cleared'] else ', NOT CLEARED'})")
        for row in s["reported"]:
            out.append(f"   {row['policy']:<18}{row['passed']:>5}/{s['n']}  {100 * row['rate']:5.1f} %   (reported)")
        for reason in s["reasons"]:
            out.append(f"   reason: {reason}")
    out.append(f"wall {result['wall_seconds']} s")
    return "\n".join(out)
