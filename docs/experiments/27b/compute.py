"""The memory, room and compute tables of PREREG-27b.md, computed from plan.json and token_table.json. Nothing is launched.

    python docs/experiments/27b/compute.py            prints both tables
    python docs/experiments/27b/compute.py --check    exits 1 when a table in PREREG-27b.md differs

Every number is arithmetic on the inputs of plan.json (``hardware``, ``compute_inputs``, ``algorithm``,
``evaluation``), and each input names its source there. Measured inputs: the weight size and dimensions of the model,
the generation time per episode (on the v0 environments), and token_table.json (token lengths of the parts of an
episode that do not depend on the model, measured with the model's tokenizer on CPU; the file names its command).
Everything about the actor update is an estimate.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKS = {"memory": ("<!-- memory-table: python docs/experiments/27b/compute.py -->", "<!-- /memory-table -->"),
         "room": ("<!-- room-table: python docs/experiments/27b/compute.py -->", "<!-- /room-table -->"),
         "compute": ("<!-- compute-table: python docs/experiments/27b/compute.py -->", "<!-- /compute-table -->")}
GB = 1e9


def memory(plan: dict) -> list[dict]:
    """Per card: what one GPU holds while the replicas generate, and how many tokens of one micro-batch fit the actor."""
    h, a = plan["hardware"], plan["algorithm"]
    d, r, act = h["model_dims"], h["rollout"], h["actor"]
    weights = d["weights_gb_bf16"]
    layer_inputs = d["layers"] * d["hidden"] * 2
    one_layer = 6 * d["mlp"] * 2
    low, high = layer_inputs + one_layer + d["vocab"] * 2, layer_inputs + one_layer + 3 * d["vocab"] * 2
    cap = a["max_prompt_len"] + a["max_response_len"]
    out = []
    for card in h["cards"]:
        budget = r["gpu_memory_utilization"] * card["gb"]
        kv_gb = budget - weights / r["tensor_parallel"]
        kv_tokens = kv_gb * GB * r["tensor_parallel"] / r["kv_bytes_per_token"]
        free = card["gb"] - act["reserve_gb"] - weights / act["fsdp_size"]
        fit_low, fit_high = free * GB / high, free * GB / low
        verdict = "fits on paper" if cap <= fit_low else "does not fit on paper" if cap > fit_high else "between the two estimates: the smoke decides"
        out.append(dict(card=card["name"], gb=card["gb"], rollout_weights=weights / r["tensor_parallel"], rollout_budget=budget, kv_gb=kv_gb,
                        kv_tokens_per_replica=kv_tokens, episodes_in_cache=kv_tokens / cap, actor_shard=weights / act["fsdp_size"], actor_free=free,
                        tokens_fit=(fit_low, fit_high), mb_per_token=(low / 1e6, high / 1e6), cap=cap, verdict=verdict))
    return out


def memory_table(plan: dict) -> str:
    r, a = plan["hardware"]["rollout"], plan["hardware"]["actor"]
    out = [f"| Card | While generating: weights of a replica per GPU (TP {r['tensor_parallel']}) | Rollout budget per GPU ({r['gpu_memory_utilization']:.2f}) | "
           f"KV cache per replica | Longest episodes ({memory(plan)[0]['cap']:,} tokens) held in cache per replica | While training: actor shard per GPU | "
           f"Left for activations (reserve {a['reserve_gb']} GB) | Tokens of one micro-batch that fit | The plan's {memory(plan)[0]['cap']:,}-token episode |",
           "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for m in memory(plan):
        out.append(f"| {m['card']} {m['gb']} GB | {m['rollout_weights']:.1f} GB | {m['rollout_budget']:.0f} GB | {m['kv_gb'] * r['tensor_parallel']:.0f} GB = "
                   f"{m['kv_tokens_per_replica'] / 1e6:.2f} M tokens | {m['episodes_in_cache']:.0f} | {m['actor_shard']:.1f} GB | {m['actor_free']:.0f} GB | "
                   f"{m['tokens_fit'][0] / 1000:.0f}k - {m['tokens_fit'][1] / 1000:.0f}k ({m['mb_per_token'][1]:.2f} - {m['mb_per_token'][0]:.2f} MB per token) | {m['verdict']} |")
    return "\n".join(out)


def room(plan: dict, tokens: dict) -> list[dict]:
    """Per cell: what is left for the model's own text, per tool call of the reference procedure, at the plan's episode
    length and at the lower estimate of what each larger card fits (the calls of the procedure and their observations
    are in the stream whatever the model writes around them)."""
    a = plan["algorithm"]
    caps = [("plan", a["max_prompt_len"] + a["max_response_len"])] + [(m["card"], int(m["tokens_fit"][0])) for m in memory(plan) if m["card"] != "H100"]
    out = []
    for cell, t in tokens["cells"].items():
        fixed, calls = t["reference_episode_without_model_text_max"], t["reference_calls_max"]
        out.append(dict(cell=cell, first_prompt=t["first_prompt_max"], fixed=fixed, calls=calls, limit=t["tool_call_limit"],
                        every_call=t["observations_if_every_call_used"], per_call={name: (cap - fixed) / calls for name, cap in caps}, caps=dict(caps)))
    return out


def room_table(plan: dict, tokens: dict) -> str:
    rows = room(plan, tokens)
    caps = rows[0]["caps"]
    head = " | ".join(f"at {cap:,} tokens ({'the plan' if name == 'plan' else name + ', lower estimate'})" for name, cap in caps.items())
    out = [f"| Cell | First prompt | Reference procedure: calls | Its stream without any model text | Model text per call: {head} | "
           "Stream without model text if every allowed call is used |", "|---|---:|---:|---:|" + "---:|" * len(caps) + "---:|"]
    for r in rows:
        per = " | ".join(f"{v:,.0f}" if v > 0 else "does not fit" for v in r["per_call"].values())
        out.append(f"| {r['cell']} | {r['first_prompt']:,} | {r['calls']} | {r['fixed']:,} | {per} | {r['every_call']:,} (limit {r['limit']}) |")
    return "\n".join(out)


def compute(plan: dict) -> dict:
    a, e, c, gpus = plan["algorithm"], plan["evaluation"], plan["compute_inputs"], plan["hardware"]["gpus_per_node"]
    per_ep = c["gpu_s_per_episode_generation"]["value"]
    slow = c["lora_serving_slowdown"]
    per_step = a["group_size"] * a["groups_per_step"]
    gen = (per_step * per_ep * slow["low"] / 3600.0, per_step * per_ep * slow["high"] / 3600.0)
    tokens = per_step * c["tokens_per_episode_trained"]["value"]
    upd = (tokens / c["trained_tokens_per_s_per_node"]["high"] * gpus / 3600.0, tokens / c["trained_tokens_per_s_per_node"]["low"] * gpus / 3600.0)
    step = (gen[0] + upd[0], gen[1] + upd[1])
    launch = c["eval_overhead_s_per_launch"]["value"] * gpus / 3600.0
    rows = []

    def add(part, episodes, lo, hi, note=""):
        rows.append(dict(part=part, episodes=episodes, low=lo, high=hi, note=note))

    cal = e["calibration"]["instances"] * e["calibration"]["episodes"] * e["calibration"]["levels"]
    add("Calibration (served base model, no adapter)", cal, cal * per_ep / 3600.0 + launch, cal * per_ep / 3600.0 + launch, f"{e['calibration']['levels']} levels x {e['calibration']['instances']} x {e['calibration']['episodes']}")
    sm = e["smoke"]["episodes"]
    add("Training smoke", sm, sm * per_ep * slow["low"] / 3600.0 + 3 * launch, sm * per_ep * slow["high"] / 3600.0 + 3 * launch, "3 steps at a tiny batch, two trainer starts, the CPU gate")
    out = dict(per_step=dict(generation=gen, update=upd, total=step, seconds=(step[0] * 3600 / gpus, step[1] * 3600 / gpus), episodes=per_step, tokens=tokens), designs=[])
    for label, arms in (("Core (R1 + B)", [x for x in plan["arms"] if x["core"]]), ("With the second seed (R1 + R2 + B)", plan["arms"])):
        part = list(rows)
        trained = [x for x in arms if x["steps"]]
        steps = sum(x["steps"] for x in trained)
        part.append(dict(part="RL steps", episodes=steps * per_step, low=steps * step[0], high=steps * step[1], note=f"{steps} steps of {per_step} episodes"))
        dev_eps = len(trained) * (a["steps"] // e["dev"]["every_steps"] + 1) * e["dev"]["instances"] * e["dev"]["episodes"] * e["dev"]["cells_assumed"]
        part.append(dict(part="Dev evaluations (in training)", episodes=dev_eps, low=dev_eps * per_ep * slow["low"] / 3600.0, high=dev_eps * per_ep * slow["high"] / 3600.0,
                         note=f"{a['steps'] // e['dev']['every_steps'] + 1} evaluations per arm, {e['dev']['cells_assumed']} cells assumed"))
        test_eps = len(arms) * e["test"]["instances"] * e["test"]["episodes"] * e["test"]["cells_assumed"]
        part.append(dict(part="Held-out readout", episodes=test_eps, low=test_eps * per_ep * slow["low"] / 3600.0 + len(arms) * launch,
                         high=test_eps * per_ep * slow["high"] / 3600.0 + len(arms) * launch, note=f"{len(arms)} models x {e['test']['cells_assumed']} cells assumed x {e['test']['instances']} x {e['test']['episodes']}"))
        total = dict(episodes=sum(p["episodes"] for p in part), low=sum(p["low"] for p in part), high=sum(p["high"] for p in part))
        over = 1.0 + c["preemption_overhead"]
        out["designs"].append(dict(design=label, parts=part, total=total, node_h=(total["low"] / gpus, total["high"] / gpus),
                                   node_h_preempted=(total["low"] / gpus * over, total["high"] / gpus * over)))
    return out


def span(lo, hi, digits=0):
    return f"{lo:.{digits}f}" if round(lo, digits) == round(hi, digits) else f"{lo:.{digits}f} - {hi:.{digits}f}"


def compute_table(plan: dict) -> str:
    c, gpus = compute(plan), plan["hardware"]["gpus_per_node"]
    s = c["per_step"]
    out = [f"One step = {s['episodes']} episodes, {s['tokens'] / 1e6:.1f} M trained tokens: generation {span(*s['generation'], digits=1)} GPU-h, update "
           f"{span(*s['update'], digits=1)} GPU-h, total {span(*s['total'], digits=1)} GPU-h = {span(*s['seconds'])} s on one {gpus}-GPU node.", "",
           "| Design | Part | Episodes | GPU-hours | Basis |", "|---|---|---:|---:|---|"]
    for d in c["designs"]:
        for p in d["parts"]:
            out.append(f"| {d['design']} | {p['part']} | {p['episodes']:,} | {span(p['low'], p['high'])} | {p['note']} |")
        out.append(f"| {d['design']} | Total | {d['total']['episodes']:,} | {span(d['total']['low'], d['total']['high'])} | node-hours on {gpus}-GPU nodes: "
                   f"{span(*d['node_h'])}; with {100 * plan['compute_inputs']['preemption_overhead']:.0f} % for preemption: {span(*d['node_h_preempted'])} |")
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    with open(os.path.join(HERE, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    with open(os.path.join(HERE, "token_table.json"), encoding="utf-8") as f:
        tokens = json.load(f)
    tables = dict(memory=memory_table(plan), room=room_table(plan, tokens), compute=compute_table(plan))
    if "--check" not in argv:
        print(tables["memory"] + "\n\n" + tables["room"] + "\n\n" + tables["compute"])
        return 0
    with open(os.path.join(HERE, "..", "PREREG-27b.md"), encoding="utf-8") as f:
        doc = f.read()
    bad = [k for k, (start, end) in MARKS.items() if doc[doc.index(start) + len(start):doc.index(end)].strip() != tables[k]]
    if bad:
        print(f"stale in PREREG-27b.md: {', '.join(bad)} table; paste the output of this script between the markers")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
