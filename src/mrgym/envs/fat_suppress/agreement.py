"""Oracle of ``Scanner/FatSuppress`` against the verifier and against the must-pass policy, on fresh instances.

    python -m mrgym.envs.fat_suppress.agreement --level 2 --n 60 [--seed-prefix fresh]

prints per level: how often the oracle's own answer passes the verifier; how often the oracle's closed form and the
verifier's scans agree on the method of a region; the pass rate of the reference policy and the distance of its
inversion times from the oracle's nulls. (``oracle.py`` itself never imports the policies; this module plays them.)
"""
from __future__ import annotations

import numpy as np

import mrgym
from mrgym import policies

from . import controls, oracle, task


def compare(level: int, n: int, seed_prefix: str = "oracle") -> dict:
    """Oracle against verifier and reference policy on ``n`` fresh instances of ``level`` (see the module docstring)."""
    switch = 0.5 * (task.FAT_MAX + task.FAT_SAT_SUFFICIENT)
    rows, same, total, off = [], 0, 0, []
    for i in range(n):
        env = mrgym.make("Scanner/FatSuppress-v0", level=level, seed=f"{seed_prefix}:{i}")
        env.reset()
        answer = controls.reference(policies.ToolScanner(env))
        step = env.step({"name": "submit", "arguments": answer})
        cfg, world, engine = env.episode_config(), env.hidden_world(), env._session.engine
        tr = oracle.truth(cfg, world)
        s = task.score(cfg, oracle.answer(cfg, world, tr), world, engine)
        for k, row in s["residual"].items():                             # the method of a region by the oracle and by the verifier's scans
            total += 1
            same += (tr[k]["fat_sat_left"] < switch) == (row["fat_sat"] < switch)
        off += [abs(e["ti_ms"] - tr[k]["null_ms"]) for k, e in answer.items() if e["method"] == "stir"]
        rows.append(dict(oracle_passes=bool(s["passed"]), reference_passes=bool(step.reward >= 1.0)))
    k = lambda key: int(sum(x[key] for x in rows))                                       # noqa: E731
    return dict(level=level, n=n, seeds=f"split train, seeds {seed_prefix}:0..{n - 1}", oracle_passes=k("oracle_passes"), reference_passes=k("reference_passes"),
                pass_regret=round((k("oracle_passes") - k("reference_passes")) / n, 4), same_method=f"{same}/{total}",
                reference_ti_off_oracle_null_ms=dict(median=round(float(np.median(off)), 2), q95=round(float(np.quantile(off, 0.95)), 2)) if off else None)


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed-prefix", default="oracle")
    a = ap.parse_args()
    for lv in a.level:
        print(json.dumps(compare(lv, a.n, a.seed_prefix)), flush=True)
