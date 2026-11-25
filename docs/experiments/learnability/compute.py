"""The compute table of PREREG-learnability.md, computed from plan.json. Nothing is launched.

    python docs/experiments/learnability/compute.py            prints the table
    python docs/experiments/learnability/compute.py --check    exits 1 when the table in the plan file differs
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
START, END = "<!-- compute-table: python docs/experiments/learnability/compute.py -->", "<!-- /compute-table -->"


def rows(plan: dict) -> list[dict]:
    a, e, c = plan["algorithm"], plan["evaluation"], plan["compute_inputs"]
    per_ep, gpus = c["gpu_s_per_episode"]["value"], c["gpus_per_node"]
    per_step = a["group_size"] * a["groups_per_step"]

    def rl(arms):
        steps = sum(x["steps"] for x in arms)
        return dict(episodes=steps * per_step, low=steps * c["seconds_per_step"]["low"] * gpus / 3600.0,
                    high=steps * c["seconds_per_step"]["high"] * gpus / 3600.0)

    def evals(arms):
        trained = [x for x in arms if x["steps"]]
        dev = len(trained) * (a["steps"] // e["dev"]["every_steps"]) * e["dev"]["instances"] * e["dev"]["episodes"] * len(e["dev"]["levels"])
        test = len(arms) * len(e["test"]["cells"]) * e["test"]["instances"] * e["test"]["episodes"]
        return dict(episodes=dev + test, low=(dev + test) * per_ep / 3600.0, high=(dev + test) * per_ep / 3600.0)

    cal = e["calibration"]["instances"] * e["calibration"]["episodes"] * len(e["calibration"]["levels"])
    out = []
    for label, arms in (("Full design", plan["arms"]), ("Core only", [x for x in plan["arms"] if x["core"]])):
        names = "+".join(x["name"] for x in arms)
        parts = [dict(part="Calibration", episodes=cal, low=cal * per_ep / 3600.0, high=cal * per_ep / 3600.0),
                 dict(part="SFT arm", episodes=0, low=c["sft_gpu_h"]["value"], high=c["sft_gpu_h"]["value"]),
                 dict(part="RL arms", **rl(arms)), dict(part="Dev and test evaluations", **evals(arms))]
        total = dict(part="Total", episodes=sum(p["episodes"] for p in parts), low=sum(p["low"] for p in parts), high=sum(p["high"] for p in parts))
        over = 1.0 + c["preemption_overhead"]
        out.append(dict(design=label, arms=names, parts=parts, total=total,
                        node_h=(total["low"] / gpus, total["high"] / gpus), node_h_preempted=(total["low"] / gpus * over, total["high"] / gpus * over)))
    return out


def table(plan: dict) -> str:
    def span(lo, hi):
        return f"{lo:.0f}" if round(lo) == round(hi) else f"{lo:.0f} - {hi:.0f}"
    out = ["| Design | Part | Episodes | GPU-hours |", "|---|---|---:|---:|"]
    for d in rows(plan):
        for p in d["parts"] + [d["total"]]:
            out.append(f"| {d['design']} ({d['arms']}) | {p['part']} | {p['episodes']:,} | {span(p['low'], p['high'])} |")
        out.append(f"| {d['design']} | Node-hours on {plan['compute_inputs']['gpus_per_node']}-GPU nodes; with "
                   f"{100 * plan['compute_inputs']['preemption_overhead']:.0f} % for preemption | | "
                   f"{span(*d['node_h'])}; {span(*d['node_h_preempted'])} |")
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(os.path.join(HERE, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    text = table(plan)
    if "--check" not in argv:
        print(text)
        return 0
    with open(os.path.join(HERE, "..", "PREREG-learnability.md"), encoding="utf-8") as f:
        doc = f.read()
    stored = doc[doc.index(START) + len(START):doc.index(END)].strip()
    if stored != text:
        print("the compute table of PREREG-learnability.md is stale; paste the output of this script between the markers")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
