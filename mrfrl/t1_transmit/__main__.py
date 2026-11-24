"""Command line of family F1 (``t1_transmit``).

    python -m mrfrl.t1_transmit selfcheck --n 50          admission statistics and control pass rates per level
    python -m mrfrl.t1_transmit make --level 2 --out DIR  one task: DIR/agent (what the agent sees), DIR/hidden (config)
    python -m mrfrl.t1_transmit verify --hidden DIR/hidden --log-dir LOG --answer answer.json

``selfcheck`` draws 128-bit instance seeds as HMAC-SHA256(master, label); the master comes from the operating system
unless --master-file names a file holding one (for a repeatable run). Seeds are never printed.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # before numpy loads: one thread per worker
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import collections
import hashlib
import hmac
import json
import multiprocessing
import shutil
import sys
import time

import numpy as np


def _seed(master, label):
    return hmac.new(master.encode(), label.encode(), hashlib.sha256).hexdigest()[:32]


def _admit_one(args):
    from .. import controls
    from . import task
    seed, level, recipe, n_cal, n_test, n_ctrl = args
    t0 = time.time()
    rep = controls.admit(task.make_instance(seed, level), recipe, n_cal, n_test, n_ctrl)
    rep["seconds"] = time.time() - t0
    return rep


def _tune(args):
    from .. import controls
    from . import task
    master, level, n_tune = args
    cfgs = [task.make_instance(_seed(master, f"tune:{level}:{i}"), level) for i in range(n_tune)]
    return controls.tune_recipe(cfgs, controls.recipe_candidates(task.LEVELS[level]["budget_s"][0]))


def summarise(reports):
    from .. import controls
    names = ("reference",) + controls.ADMISSION_CONTROLS + controls.REPORTED_CONTROLS
    ok = [r for r in reports if "tolerance" in r]
    out = dict(instances=len(reports), admitted=sum(r["admitted"] for r in reports),
               admission_rate=round(float(np.mean([r["admitted"] for r in reports])), 3),
               admission_rate_without_fixed_recipe_rule=round(float(np.mean(
                   [r.get("admitted_without_fixed_recipe_rule", False) for r in reports])), 3),
               rejection_reasons=dict(collections.Counter(x for r in reports for x in r["reasons"]).most_common()),
               seconds_per_instance=round(float(np.mean([r["seconds"] for r in reports])), 2),
               tolerance_median=round(float(np.median([r["tolerance"] for r in ok])), 4),
               tolerance_range=[round(float(min(r["tolerance"] for r in ok)), 4), round(float(max(r["tolerance"] for r in ok)), 4)],
               e_prior_median=round(float(np.median([r["e_prior"] for r in ok])), 3),
               reward_at_prior_policy_max=float(max(abs(r["reward_prior_after_one_scan"]) for r in ok)),
               adaptivity_gap_fixed_over_reference_median=round(float(np.median(
                   [r["adaptivity_gap"]["fixed_over_reference"] for r in ok])), 2),
               adaptivity_gap_open_loop_over_reference_median=round(float(np.median(
                   [r["adaptivity_gap"]["open_loop_over_reference"] for r in ok])), 2),
               controls={})
    for n in names:
        rows = [r["controls"][n] for r in ok]
        errs = [c["error_median"] for c in rows if c["error_median"] is not None]
        out["controls"][n] = dict(
            pass_rate=round(float(np.mean([c["pass_rate"] for c in rows])), 3),
            instances_passed_at_least_once=round(float(np.mean([c["pass_rate"] > 0 for c in rows])), 3),
            reward_mean=round(float(np.mean([c["reward_mean"] for c in rows])), 3),
            error_median=round(float(np.median(errs)), 4) if errs else None,
            scan_s_median=round(float(np.median([c["scan_s_median"] for c in rows])), 1))
    adm = [r for r in ok if r["admitted"]]
    if adm:
        out["on_admitted"] = {n: round(float(np.mean([r["controls"][n]["pass_rate"] for r in adm])), 3) for n in names}
    return out


def cmd_selfcheck(a):
    if a.master_file:
        with open(a.master_file) as f:
            master = f.read().strip()
    else:
        master = os.urandom(16).hex()
    levels = [int(x) for x in a.levels.split(",")]
    workers = max(1, min(a.workers, 8))
    t0 = time.time()
    result = dict(n=a.n, noise_draws=dict(calibration=a.cal, held_out=a.test, per_control=a.ctrl),
                  master_commitment=hashlib.sha256(master.encode()).hexdigest()[:16], levels={})
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        recipes = dict(zip(levels, pool.map(_tune, [(master, lv, a.tune) for lv in levels])))
        for lv in levels:
            jobs = [(_seed(master, f"inst:{lv}:{i}"), lv, recipes[lv], a.cal, a.test, a.ctrl) for i in range(a.n)]
            reports = pool.map(_admit_one, jobs, chunksize=1)
            result["levels"][lv] = dict(summarise(reports), fixed_recipe=recipes[lv])
            if a.dump:
                result["levels"][lv]["reports"] = reports
    result["wall_seconds"] = round(time.time() - t0, 1)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    for lv in levels:
        s = result["levels"][lv]
        print(f"\n== level {lv}: {s['instances']} instances, admitted {s['admitted']} ({100 * s['admission_rate']:.0f} %), "
              f"{s['seconds_per_instance']} s per instance; without the fixed-recipe rule {100 * s['admission_rate_without_fixed_recipe_rule']:.0f} %")
        print(f"   tolerance median {100 * s['tolerance_median']:.1f} % (range {100 * s['tolerance_range'][0]:.1f}-"
              f"{100 * s['tolerance_range'][1]:.1f} %), prior-policy error median {100 * s['e_prior_median']:.0f} %, "
              f"reward at the prior policy {s['reward_at_prior_policy_max']}")
        print(f"   adaptivity gap (median error ratio to the reference): fixed recipe {s['adaptivity_gap_fixed_over_reference_median']}"
              f", open loop {s['adaptivity_gap_open_loop_over_reference_median']}")
        print(f"   fixed recipe tuned on {a.tune} other worlds: {json.dumps({k: v for k, v in s['fixed_recipe'].items()})}")
        print("   control            pass rate   instances passed >= once   mean reward   median error   median scan s")
        for n, c in s["controls"].items():
            err = "-" if c["error_median"] is None else f"{100 * c['error_median']:.1f} %"
            print(f"   {n:<18} {c['pass_rate']:>8.3f}   {c['instances_passed_at_least_once']:>22.3f}   {c['reward_mean']:>11.3f}"
                  f"   {err:>12}   {c['scan_s_median']:>13.1f}")
        print(f"   rejection reasons: {json.dumps(s['rejection_reasons'])}")
    print(f"\nwall {result['wall_seconds']} s on {workers} workers")


def cmd_make(a):
    from .. import controls
    from ..scanner import core
    from . import task
    cfg = task.make_instance(core.new_seed(), a.level)
    hidden = dict(cfg)
    if not a.no_admission:
        rep = controls.admit(cfg)
        hidden["admission"] = rep
        if not rep["admitted"]:
            print(json.dumps(dict(admitted=False, reasons=rep["reasons"])))
            sys.exit(2)
    root = os.path.join(a.out, cfg["task_id"])
    agent, hid = os.path.join(root, "agent"), os.path.join(root, "hidden")
    os.makedirs(agent), os.makedirs(hid)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(agent, "instruction.md"), "w") as f:
        f.write(task.instruction(cfg))
    with open(os.path.join(agent, "regions.json"), "w") as f:
        json.dump(cfg["task"]["regions"], f, indent=1)
    shutil.copy(os.path.join(here, "..", "scanner", "client.py"), os.path.join(agent, "mrscan.py"))
    shutil.copy(os.path.join(here, "..", "..", "docs", "scanner.md"), os.path.join(agent, "scanner.md"))
    with open(os.path.join(hid, "episode.json"), "w") as f:
        json.dump(hidden, f, indent=1)
    print(json.dumps(dict(task=root, admitted=None if a.no_admission else True)))


def cmd_verify(a):
    from ..scanner import core
    from . import task
    cfg = core.load_config(os.path.join(a.hidden, "episode.json"))
    tol = a.tolerance if a.tolerance is not None else cfg["admission"]["tolerance"]
    print(json.dumps(task.verify(cfg, a.log_dir, a.answer, tol), indent=1))


def main():
    ap = argparse.ArgumentParser(prog="python -m mrfrl.t1_transmit", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selfcheck")
    p.add_argument("--n", type=int, default=50, help="candidate instances per level")
    p.add_argument("--levels", default="1,2,3")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cal", type=int, default=32, help="noise draws that set the tolerance")
    p.add_argument("--test", type=int, default=32, help="fresh noise draws the reference must pass")
    p.add_argument("--ctrl", type=int, default=8, help="noise draws per control")
    p.add_argument("--tune", type=int, default=12, help="other worlds on which the fixed recipe is tuned")
    p.add_argument("--master-file", default=None)
    p.add_argument("--out", default=None, help="write the full JSON report here")
    p.add_argument("--dump", action="store_true", help="include per-instance reports in --out")
    p = sub.add_parser("make")
    p.add_argument("--level", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--out", required=True)
    p.add_argument("--no-admission", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("--hidden", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--answer", required=True)
    p.add_argument("--tolerance", type=float, default=None)
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
