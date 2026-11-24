"""Command line of the recon-repair task set.

    python -m recon_repair make  --seed 0 --level 1 --split train --out tasks/        write one task directory
    python -m recon_repair batch --seeds 0:50 --level 1 --split train --out tasks/    write many, print admission statistics
    python -m recon_repair workspace --task tasks/<name> --to /tmp/ws                 copy what the agent sees (local runs)
    python -m recon_repair verify --task tasks/<name> --submission /tmp/ws            grade a workspace against hidden scans
    python -m recon_repair selfcheck --n 12                                           build instances and run the controls
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from . import cartesian as cart
from . import task


def _verify(task_dir, submission):
    out = tempfile.mkdtemp(prefix="recon_repair_verify_")
    try:
        subprocess.run([sys.executable, os.path.join(task_dir, "tests", "verify.py"), "--submission", submission,
                        "--hidden", os.path.join(task_dir, "tests", "hidden"), "--out", os.path.join(out, "reward.json"),
                        "--details", os.path.join(out, "details.json")], check=True, capture_output=True)
        return json.load(open(os.path.join(out, "details.json")))
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _with_recon(source_text):
    d = tempfile.mkdtemp(prefix="recon_repair_sub_")
    with open(os.path.join(d, "recon.py"), "w") as fh:
        fh.write(source_text)
    return d


def cmd_make(a):
    inst = task.build(a.seed, a.level, a.split)
    print(task.write(inst, a.out))


def cmd_batch(a):
    lo, hi = (int(x) for x in a.seeds.split(":"))
    used, rejected, t0 = collections.Counter(), collections.Counter(), time.time()
    attempts = []
    for seed in range(lo, hi):
        inst = task.build(seed, a.level, a.split)
        task.write(inst, a.out)
        used.update(inst["variants"])
        attempts.append(inst["attempt"] + 1)
        for r in inst["rejected"]:
            rejected[r["reason"].split(" alone")[0]] += 1
    n = hi - lo
    print(json.dumps(dict(instances=n, seconds_per_instance=round((time.time() - t0) / n, 2),
                          draws_per_instance=round(sum(attempts) / n, 2), variants_used=dict(sorted(used.items())),
                          rejections=dict(rejected)), indent=1))


def cmd_workspace(a):
    shutil.copytree(os.path.join(a.task, "environment", "app"), a.to)
    shutil.copy(os.path.join(a.task, "instruction.md"), os.path.join(a.to, "TASK.md"))
    print(a.to)


def cmd_verify(a):
    print(json.dumps(_verify(a.task, a.submission), indent=1))


def cmd_selfcheck(a):
    """End-to-end controls through the real verifier subprocess, on freshly built instances."""
    root = tempfile.mkdtemp(prefix="recon_repair_selfcheck_")
    totals = collections.Counter()
    try:
        for seed in range(a.n):
            level, split = 1 + seed % 3, ("train", "heldout_mechanism")[seed % 2]
            inst = task.build(seed, level, split)
            tdir = task.write(inst, root)
            d = inst["dialect"]
            controls = {"reference": cart.recon_source(d), "as_given": cart.recon_source(d, inst["variants"]),
                        "writes_zeros": "import sys, numpy as np\nnp.save(sys.argv[2], np.zeros((8, 8)))\n",
                        "crashes": "raise SystemExit(3)\n"}
            for v in inst["variants"] if len(inst["variants"]) > 1 else []:
                controls["one_fault_left:" + v] = cart.recon_source(d, [v])
            for name, text in controls.items():
                sub = _with_recon(text)
                reward = _verify(tdir, sub)["reward"]["reward"]
                shutil.rmtree(sub)
                key = name.split(":")[0]
                totals[key + "_runs"] += 1
                totals[key + "_reward_1"] += int(reward == 1.0)
            agent_files = set(os.listdir(os.path.join(tdir, "environment", "app")))
            totals["leaks"] += int(bool(agent_files - {"recon.py", "FORMAT.md", "data"}))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(json.dumps(dict(instances=a.n, **totals), indent=1))
    ok = (totals["reference_reward_1"] == totals["reference_runs"] and totals["leaks"] == 0 and
          not any(totals[k + "_reward_1"] for k in ("as_given", "writes_zeros", "crashes", "one_fault_left")))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(prog="recon_repair", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make", "batch"):
        p = sub.add_parser(name)
        p.add_argument("--level", type=int, default=1, choices=sorted(task.LEVEL_FAULTS))
        p.add_argument("--split", default="train", choices=sorted(task.SPLITS))
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=0) if name == "make" else p.add_argument("--seeds", required=True)
    p = sub.add_parser("workspace"); p.add_argument("--task", required=True); p.add_argument("--to", required=True)
    p = sub.add_parser("verify"); p.add_argument("--task", required=True); p.add_argument("--submission", required=True)
    p = sub.add_parser("selfcheck"); p.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
