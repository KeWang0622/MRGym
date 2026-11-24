"""Admission by control policies: the gate for calling an environment admitted.

An instance (environment, level, seed) is admitted when a reference policy passes it and every shortcut policy fails
it: the no-scan prior answer, the best constant recipe tuned on other worlds, a blind sweep, the right numbers in the
wrong unit, an attempt to rebuild the world from guessable seeds, and the environment's own confounder-blind
policies. The environment's ``controls`` module defines the policies and the per-instance rule (``admit``); this
module runs it over a list of public training seeds, summarises it and stores the result next to the package, where
the environment card and ``mrgym list`` read it. Nothing here is typed by hand.

    mrgym check Scanner/NoWrap-v0 --admission --n 24 --write

What the gate does and does not show: the listed shortcut policies fail. It does not prove that no shortcut exists.
"""
from __future__ import annotations

import collections
import datetime
import importlib
import importlib.resources
import json
import multiprocessing
import os
import time

import numpy as np

SCHEMA = "mrgym-admission/1"


def _file(env_id: str) -> str:
    return env_id.replace("/", "-") + ".json"


def results_dir() -> str:
    return str(importlib.resources.files("mrgym.envs").joinpath("results"))


def load(env_id: str) -> dict | None:
    """The stored control-policy results of an environment id, or None."""
    try:
        text = importlib.resources.files("mrgym.envs").joinpath("results", _file(env_id)).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    return json.loads(text)


def stored_instance(env_id: str, task_id: str) -> dict | None:
    m = load(env_id)
    if not m:
        return None
    for lv in m["levels"].values():
        for inst in lv["instance_list"]:
            if inst["task_id"] == task_id:
                return inst
    return None


def admitted_seeds(env_id: str, level: int) -> list[int]:
    """Public training seeds of ``level`` that the stored control-policy run admits (empty when none is stored)."""
    m = load(env_id)
    if not m or str(level) not in m["levels"]:
        return []
    return [i["seed"] for i in m["levels"][str(level)]["instance_list"] if i["admitted"]]


def counts(measured: dict | None) -> tuple[int, int]:
    """(admitted instances, instances) over all levels of a stored result."""
    if not measured:
        return 0, 0
    return (sum(lv["admitted"] for lv in measured["levels"].values()),
            sum(lv["instances"] for lv in measured["levels"].values()))


def status_line(spec, measured: dict | None) -> str:
    k, n = counts(measured)
    if spec.kind == "sandbox":
        return "SANDBOX: no reward"
    if not measured:
        return f"{spec.kind.upper()}: control policies not run (no stored result)"
    if spec.kind == "admitted":
        return f"ADMITTED: {k} of {n} instances pass the gate" if k else f"NOT ADMITTED: 0 of {n} instances pass the gate"
    reasons = collections.Counter()
    for lv in measured["levels"].values():
        reasons.update(lv["rejection_reasons"])
    top = reasons.most_common(1)[0][0] if reasons else "no reason recorded"
    return f"PRACTICE: its own gate rejects it ({k} of {n} instances admitted; most frequent reason: {top})"


# ------------------------------------------------------------------ running
def task_id_for(spec, seed_hex: str) -> str:
    """The public task id of an episode: a one-way function of the seed (the id does not give the seed)."""
    from .scanner import session
    return spec.name.lower() + "-" + session._sha(("task-id:" + seed_hex).encode())[:12]


def _instance(spec, level, seed):
    from .env import derive_seed
    family = importlib.import_module(spec.family + ".task")
    seed_hex = derive_seed(spec.id, level, "train", seed, environ={})
    return family.make_instance(seed_hex, level, task_id=task_id_for(spec, seed_hex))


def _admit_one(args):
    import mrgym
    env_id, level, seed, recipe, n_cal, n_test, n_ctrl = args
    spec = mrgym.spec(env_id)
    controls = importlib.import_module(spec.family + ".controls")
    t0 = time.time()
    rep = controls.admit(_instance(spec, level, seed), recipe, n_cal, n_test, n_ctrl)
    rep.update(seconds=time.time() - t0, seed=seed)
    return rep


def _tune(args):
    import mrgym
    env_id, level, n_tune = args
    spec = mrgym.spec(env_id)
    controls = importlib.import_module(spec.family + ".controls")
    return controls.tune_recipe([_instance(spec, level, f"tune:{i}") for i in range(n_tune)])


def summarise(reports: list, names: tuple) -> dict:
    out = dict(instances=len(reports), admitted=int(sum(r["admitted"] for r in reports)),
               rejection_reasons=dict(collections.Counter(x for r in reports for x in r["reasons"]).most_common()),
               seconds_per_instance=round(float(np.mean([r["seconds"] for r in reports])), 2), controls={})
    tol = [r["tolerance"] for r in reports if r.get("tolerance") is not None]
    if tol:
        out["tolerance_median"] = round(float(np.median(tol)), 4)
    ep = [r["e_prior"] for r in reports if r.get("e_prior") is not None]
    if ep:
        out["e_prior_median"] = round(float(np.median(ep)), 3)
    out["reward_prior_after_one_scan_max"] = float(max(abs(r.get("reward_prior_after_one_scan", 0.0)) for r in reports))
    for n in names:
        rows = [r["controls"][n] for r in reports if n in r["controls"]]
        if not rows:
            continue
        errs = [c["error_median"] for c in rows if c["error_median"] is not None]
        out["controls"][n] = dict(
            pass_rate=round(float(np.mean([c["pass_rate"] for c in rows])), 3),
            instances_passed_at_least_once=int(sum(c["pass_rate"] > 0 for c in rows)),
            reward_mean=round(float(np.mean([c["reward_mean"] for c in rows])), 3),
            error_median=round(float(np.median(errs)), 4) if errs else None,
            scan_s_median=round(float(np.median([c["scan_s_median"] for c in rows])), 1))
    out["instance_list"] = [dict(seed=r["seed"], task_id=r["task_id"], admitted=bool(r["admitted"]), reasons=r["reasons"],
                                 tolerance=r.get("tolerance")) for r in reports]
    return out


def run(env_id: str, n: int = 8, levels=None, workers: int = 4, cal: int = 32, test: int = 32, ctrl: int = 8, tune: int = 12,
        first_seed: int = 0) -> dict:
    """Run the control policies of ``env_id`` on the public training seeds ``first_seed .. first_seed + n - 1`` of
    every level. Returns the result dict (see ``write``)."""
    import mrgym
    spec = mrgym.spec(env_id)
    if not spec.family:
        raise mrgym.Error(f"{env_id} is a sandbox: it has no control policies")
    controls = importlib.import_module(spec.family + ".controls")
    names = ("reference",) + tuple(controls.ADMISSION_CONTROLS) + tuple(controls.REPORTED_CONTROLS)
    levels = list(levels or spec.levels)
    workers = max(1, min(int(workers), 8))
    t0 = time.time()
    result = dict(schema=SCHEMA, env_id=env_id, engine=mrgym.ENGINE_ID, package_version=mrgym.__version__,
                  date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                  command=f"mrgym check {env_id} --admission --n {n} --write",
                  seeds=f"train split, public key, seeds {first_seed}..{first_seed + n - 1} per level",
                  noise_draws=dict(calibration=cal, held_out=test, per_control=ctrl), tuning_instances=tune, levels={})
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        recipes = dict(zip(levels, pool.map(_tune, [(env_id, lv, tune) for lv in levels])))
        for lv in levels:
            jobs = [(env_id, lv, first_seed + i, recipes[lv], cal, test, ctrl) for i in range(n)]
            reports = pool.map(_admit_one, jobs, chunksize=1)
            result["levels"][str(lv)] = dict(summarise(reports, names), fixed_recipe=recipes[lv])
    result["wall_seconds"] = round(time.time() - t0, 1)
    return result


def write(env_id: str, result: dict, directory: str | None = None) -> str:
    path = os.path.join(directory or results_dir(), _file(env_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    return path


def text_table(result: dict) -> str:
    out = []
    for lv, s in result["levels"].items():
        out.append(f"== {result['env_id']} level {lv}: admitted {s['admitted']} of {s['instances']} instances "
                   f"({s['seconds_per_instance']} s per instance)")
        out.append(f"   {'policy':<17}{'pass rate':>10}{'instances passed':>18}{'mean reward':>13}{'median error':>14}{'scan s':>8}")
        for n, c in s["controls"].items():
            err = "-" if c["error_median"] is None else f"{100 * c['error_median']:.1f} %"
            out.append(f"   {n:<17}{c['pass_rate']:>10.3f}{c['instances_passed_at_least_once']:>18d}{c['reward_mean']:>13.3f}{err:>14}{c['scan_s_median']:>8.1f}")
        out.append(f"   rejection reasons: {json.dumps(s['rejection_reasons'])}")
        out.append(f"   fixed recipe tuned on {result['tuning_instances']} other worlds: {json.dumps(s['fixed_recipe'])}")
    out.append(f"wall {result['wall_seconds']} s")
    return "\n".join(out)
