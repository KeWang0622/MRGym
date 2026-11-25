"""Under which pricing of a scan is ``Scanner/ProtocolToSpec`` a decision task? The same level played on three consoles.

    adjustments   the console of the task: 30 s of adjustments before a scan, 3 s before one that reuses them
    flat          30 s before every scan, whatever it repeats (the first version of this family); budget for two scans
    default       the console's default overhead of 3 s before every scan; the budget of the task

For each console the reference and every fixed list of scans (``task.OPEN_LOOP``, and ``characterise_all``) are played
on the same seeds; the open-loop control is the best of the lists, and the gap comes with the lower bound of the paired
difference (``mrgym.stats.gate.gap_lower_bound``).

    python -m mrgym.envs.protocol_to_spec.report --n 30 --level 3
"""
from __future__ import annotations

import argparse
import json

import numpy as np

CONSOLES = dict(adjustments=(dict(overhead_s=30.0, repeat_overhead_s=3.0), 63.0), flat=(dict(overhead_s=30.0), 90.4), default=({}, 63.0))


def pricing_study(level: int = 3, n: int = 30, consoles=tuple(CONSOLES)) -> dict:
    import mrgym
    from mrgym import policies
    from mrgym.stats.gate import gap_lower_bound

    from . import controls, task
    names = [controls.REFERENCE, *task.OPEN_LOOP, "characterise_all"]
    saved, budget = dict(task.CONSOLE), task.LEVELS[level]["budget_s"]
    out = {}
    try:
        for label in consoles:
            system, task.LEVELS[level]["budget_s"] = CONSOLES[label]
            task.CONSOLE.clear()
            task.CONSOLE.update(system)
            task._instance.cache_clear()
            controls.card_table.cache_clear()
            passes = {k: [] for k in names}
            for seed in range(n):
                for k in names:
                    env = mrgym.make("Scanner/ProtocolToSpec-v0", level=level, seed=f"pricing:{seed}")
                    passes[k].append(bool(policies.run(env, k).reward >= 1.0))
                    env.close()
            ref = np.array(passes[controls.REFERENCE])
            best = max(names[1:], key=lambda k: sum(passes[k]))
            gap, lower = gap_lower_bound(ref, np.array(passes[best]))
            out[label] = dict(console=dict(system), budget_s=task.LEVELS[level]["budget_s"], n=n, passed={k: int(sum(v)) for k, v in passes.items()},
                              best_open_loop=best, gap_points=round(100 * gap, 1), gap_lower95_points=round(100 * lower, 1))
    finally:
        task.CONSOLE.clear()
        task.CONSOLE.update(saved)
        task.LEVELS[level]["budget_s"] = budget
        task._instance.cache_clear()
        controls.card_table.cache_clear()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--consoles", nargs="*", default=list(CONSOLES))
    a = ap.parse_args()
    print(json.dumps(pricing_study(a.level, a.n, a.consoles), indent=1))
