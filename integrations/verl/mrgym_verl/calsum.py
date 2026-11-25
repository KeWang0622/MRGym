"""Summary of a calibration run: which (environment, level) cells a model can start training on, and what its
episodes cost and how long they are.

    python -m mrgym_verl.calsum --out RUN/cal --model NAME --plan "Scanner/NoWrap-v1:0,1,2,3 Scanner/SNRSpec-v0:1,2,3" \
        --n 32 --episodes 8 --first-seed 9000 --gpus 8 --concurrency 32 --write RUN/summary

reads the episode records ``mrgym calibrate`` wrote under ``--out`` (nothing is played here) and writes

    calibration_summary.json   per cell: the calibration row of ``mrgym.calibrate.level_row`` (pass rate, interval over
                               instances, mixed-group share, intraclass correlation, outcome counts, band) plus what the
                               launch gate of the plan asks about the harness: the share of calls taken as written, the
                               submit rate, the share of passes that used the last tool call, tokens and seconds
    calibration_summary.md     the same as a table
    train_arms.json            the cells that meet the rule (pass rate in [0.20, 0.60] and at least 80 % of the groups
                               mixed), as the ``TRAIN_ARMS`` string of the training scripts, with their mean pass rate

Token columns. ``completion`` is what the model wrote per episode, reasoning included. ``stream_est`` estimates the
length of the same episode as one token stream in the trainer, where earlier reasoning stays in the context: the
prompt of the last request plus all completion tokens. It over-counts the visible text of earlier replies once; it
is the number to size ``MAX_RESPONSE_LEN`` with, and the training smoke measures the real one.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mrgym import calibrate, learn
from mrgym import record as mrecord

from . import instances


def _pct(values, q):
    return None if not len(values) else int(np.percentile(values, q))


def cell_stats(records: list[dict], out: str, model: str, env_id: str, level: int) -> dict:
    from mrgym.agents import runner
    recs = [r for r in records if r["level"] == level]
    if not recs:
        return {}
    calls = sum((r.get("interface") or {}).get("calls", 0) for r in recs)
    tax = sum((r.get("interface") or {}).get("shape", 0) + (r.get("interface") or {}).get("vocabulary", 0) for r in recs)
    submitted = [bool((r.get("ladder") or {}).get("submitted", r.get("stop_reason") == "submitted")) for r in recs]
    passes = [r for r in recs if r.get("passed")]
    last_call = [r for r in passes if r.get("max_steps") and (r.get("env_tool_calls") or 0) >= r["max_steps"]]
    completion, stream, wall = [], [], []
    for r in recs:
        completion.append(r.get("tokens_out") or 0)
        wall.append(r.get("wall_s") or 0.0)
        d = os.path.join(out, "episodes", runner.slug(model), runner.env_slug(env_id), f"L{level}", f"s{r['seed']}", f"e{r['episode']}")
        try:
            with open(os.path.join(d, mrecord.TRANSCRIPT_FILE), encoding="utf-8") as f:
                turns = json.load(f).get("turns") or []
            last = next((t["usage"].get("prompt_tokens") for t in reversed(turns) if (t.get("usage") or {}).get("prompt_tokens")), None)
            if last is not None:
                stream.append(int(last) + int(r.get("tokens_out") or 0))
        except (OSError, ValueError):
            pass
    return dict(calls=calls, calls_taken_as_written=None if not calls else round(1.0 - tax / calls, 4),
                submit_rate=round(float(np.mean(submitted)), 4), cut_reply_share=round(float(np.mean([bool((r.get("interface") or {}).get("cut_replies")) for r in recs])), 4),
                passes_on_last_call=None if not passes else round(len(last_call) / len(passes), 4),
                tool_calls_mean=round(float(np.mean([r.get("env_tool_calls") or 0 for r in recs])), 2),
                tokens_estimated=any(r.get("tokens_estimated") for r in recs),
                completion_tokens=dict(p50=_pct(completion, 50), p90=_pct(completion, 90), p99=_pct(completion, 99), max=_pct(completion, 100)),
                stream_tokens_est=dict(p50=_pct(stream, 50), p90=_pct(stream, 90), p99=_pct(stream, 99), max=_pct(stream, 100), episodes=len(stream)),
                wall_s=dict(p50=round(float(np.median(wall)), 1), mean=round(float(np.mean(wall)), 1), max=round(float(np.max(wall)), 1), sum=round(float(np.sum(wall)), 1)))


def summarize(out: str, model: str, plan, n: int, episodes: int, first_seed: int, gpus: int = 8, concurrency: int | None = None) -> dict:
    cells, complete = [], True
    want = set(range(first_seed, first_seed + n))
    by_env: dict = {}
    for env_id, level in plan:
        by_env.setdefault(env_id, []).append(level)
    for env_id, levels in by_env.items():
        records = [r for r in calibrate.read_records(out, env_id, model) if r["seed"] in want and r["episode"] < episodes]
        rows = [calibrate.level_row(records, lv) for lv in levels]
        verdict = calibrate.decide(rows)
        for row in rows:
            lv = row["level"]
            done = row.get("n_episodes", 0)
            ok = bool(done and learn.ENTRY_BAND[0] <= row["pass"] <= learn.ENTRY_BAND[1] and (row["mixed_group_share"] or 0.0) >= learn.MIXED_MIN)
            complete = complete and done == n * episodes
            cells.append(dict(env_id=env_id, **row, expected_episodes=n * episodes, in_rule=ok, entry_rung=lv == verdict["entry_rung"],
                              **cell_stats(records, out, model, env_id, lv)))
    chosen = [(c["env_id"], c["level"]) for c in cells if c["in_rule"]]
    total = sum(c.get("n_episodes", 0) for c in cells)
    busy = sum((c.get("wall_s") or {}).get("sum", 0.0) for c in cells)       # seconds of episodes, summed over the parallel slots
    return dict(schema="mrgym-verl-calibration/1", model=model, complete=complete, episodes=total, n=n, k=episodes, seeds=f"{first_seed}-{first_seed + n - 1}",
                rule=f"pass rate in [{learn.ENTRY_BAND[0]}, {learn.ENTRY_BAND[1]}] and at least {learn.MIXED_MIN:.0%} of the groups of {episodes} mixed",
                cells=cells, train_arms=instances.arms_text(chosen) if chosen else "",
                base_rate=round(float(np.mean([c["pass"] for c in cells if c["in_rule"]])), 4) if chosen else None,
                episode_seconds=round(busy, 1), gpus=gpus, concurrency=concurrency,
                gpu_s_per_episode=None if not concurrency or not total else round(gpus * busy / concurrency / total, 1))


def markdown(s: dict) -> str:
    out = [f"Calibration of `{s['model']}`: {s['episodes']} episodes, seeds {s['seeds']} x {s['k']} per instance"
           + ("" if s["complete"] else " (INCOMPLETE: some cells have fewer episodes than planned)"), "",
           "| Environment | Level | Episodes | Pass | 95 % interval | Mixed groups | Band | In rule | Calls as written | Submit | Completion tokens p50 / p99 | Stream estimate p50 / p99 / max |",
           "|---|---:|---:|---:|---|---:|---|---|---:|---:|---|---|"]
    for c in s["cells"]:
        if not c.get("n_episodes"):
            out.append(f"| {c['env_id']} | {c['level']} | 0 | | | | | | | | | |")
            continue
        ct, st = c["completion_tokens"], c["stream_tokens_est"]
        mixed = "-" if c["mixed_group_share"] is None else f"{100 * c['mixed_group_share']:.0f} %"
        taken = "-" if c["calls_taken_as_written"] is None else f"{100 * c['calls_taken_as_written']:.1f} %"
        out.append(f"| {c['env_id']} | {c['level']} | {c['n_episodes']} | {100 * c['pass']:.1f} % | {100 * c['ci95'][0]:.1f} .. {100 * c['ci95'][1]:.1f} | {mixed} | "
                   f"{c['band']} | {'yes' if c['in_rule'] else 'no'} | {taken} | {100 * c['submit_rate']:.0f} % | {ct['p50']} / {ct['p99']} | "
                   f"{st['p50']} / {st['p99']} / {st['max']} |")
    out += ["", f"Rule: {s['rule']}.", f"Cells that meet it (`TRAIN_ARMS`): `{s['train_arms'] or 'none'}`"
            + ("" if s["base_rate"] is None else f"; their mean pass rate (`BASE_RATE`): {s['base_rate']}"),
            "" if s["gpu_s_per_episode"] is None else f"{s['episode_seconds']:.0f} episode-seconds in {s['concurrency']} parallel slots on {s['gpus']} GPUs: "
                                                      f"{s['gpu_s_per_episode']} GPU-seconds per episode (slots assumed busy; idle tails make it an under-estimate)."]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=learn.GROUP_SIZE)
    ap.add_argument("--first-seed", type=int, default=calibrate.FIRST_SEED)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=None, help="episodes in flight (processes x workers), for GPU-seconds per episode")
    ap.add_argument("--write", required=True, help="folder for the three output files")
    a = ap.parse_args(argv)
    s = summarize(a.out, a.model, instances.parse_arms(a.plan), a.n, a.episodes, a.first_seed, a.gpus, a.concurrency)
    os.makedirs(a.write, exist_ok=True)
    with open(os.path.join(a.write, "calibration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    with open(os.path.join(a.write, "calibration_summary.md"), "w", encoding="utf-8") as f:
        f.write(markdown(s))
    with open(os.path.join(a.write, "train_arms.json"), "w", encoding="utf-8") as f:
        json.dump(dict(model=s["model"], rule=s["rule"], train_arms=s["train_arms"], base_rate=s["base_rate"], complete=s["complete"],
                       cells=[dict(env_id=c["env_id"], level=c["level"], **{"pass": c.get("pass")}, mixed_group_share=c.get("mixed_group_share"),
                                   band=c.get("band"), in_rule=c["in_rule"]) for c in s["cells"]]), f, indent=1)
    print(markdown(s))
    return 0 if s["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
