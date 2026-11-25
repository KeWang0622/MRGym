"""The interface tax: the share of an agent's calls, and of its turns, that the interface refused for how they were
written rather than for what they asked.

    mrgym iface-tax --replay tests/data/u03_interface_calls.jsonl --max 0.03     recorded calls, replayed on this interface
    mrgym iface-tax --replay tests/data/c03_interface_calls.jsonl --max 0.05     a corpus of seven environments, each episode on its own instance
    mrgym iface-tax --dir runs/first                                             the episodes of a run folder

Definitions (per corpus or per agent; calls cluster in episodes, so intervals are an episode bootstrap):

    call tax   calls answered with an error of class ``shape`` or ``vocabulary`` (``mrgym.interface``) / calls issued
    turn tax   assistant turns with at least one call in which every call was such an error / turns

Reported next to them and never inside: refusals of the instrument (class ``task``: console limits, budget, a scan
index that does not exist), calls the canonicaliser rewrote (``coerced``), replies cut at the reply limit.

``--replay`` reads a corpus of recorded calls (JSON lines; the first line describes it, every other line is one call:
``{"episode", "turn", "name", "arguments", "recorded", "lost"}`` with the arguments as the model wrote them and
``lost`` = it was an interface loss when it was recorded; a call may also be written as the list of these six values
in this order). A corpus of one environment names it in its header (``env_id``) and is replayed on one instance of it
(``--level``, ``--seed``); a corpus of several lists ``instances``, one ``[env_id, level, seed]`` per episode, and every
episode is replayed on its own instance, so that a delivery meets the deliverable it was written for. Every call that
was lost is sent to ``env.step`` on a fresh episode and classed by the answer; a call that was not lost keeps its class, as its outcome depended on the state of
its own episode (the scans stored, the budget left). In addition every call that was not lost goes through the front
half of ``step`` (``env.admit``: parse, canonicalise, validate, nothing executed), which is all that can produce a
``shape`` or ``vocabulary`` error; a recorded call that was accepted then and is refused now is counted and listed.

What the number is and is not. A replayed call meets a fresh episode, not the state it was written in, so the split
between "accepted" and "refused by the instrument" is approximate (a read-out of a scan that the fresh episode does not
hold, a name that an earlier ``calc`` call assigned); the tax itself does not depend on state. And a
corpus that the interface was designed against is a regression test of that interface, not an estimate for a new
model: that number comes from ``--dir`` on a run that no rule was written from.
"""
from __future__ import annotations

import collections
import glob
import json
import os

import numpy as np

DEFAULT_MAX = 0.03
DRAWS = 10_000
CALL_KEYS = ("episode", "turn", "name", "arguments", "recorded", "lost")


def read_corpus(path: str) -> tuple[dict, list]:
    """(header, calls) of a corpus file."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or not isinstance(rows[0], dict) or "turns" not in rows[0]:
        raise ValueError(f"{path}: the first line must describe the corpus ({{\"episodes\", \"turns\": [turns per episode], ...}})")
    calls = [dict(zip(CALL_KEYS, r)) if isinstance(r, list) else r for r in rows[1:]]
    return rows[0], calls


def _bootstrap(episodes: list, draws: int, seed: int = 0) -> dict:
    """95 % interval of the call tax and the turn tax over episodes: ``{"calls": (lo, hi), "turns": (lo, hi)}``.
    Episodes are drawn with replacement (a fixed generator, so the interval is the same on every run)."""
    table = np.array([[e["lost_calls"], e["calls"], e["lost_turns"], e["turns"]] for e in episodes], float)
    pick = np.random.default_rng(seed).integers(0, len(episodes), size=(draws, len(episodes)))
    sums = table[pick].sum(axis=1)                                       # [draws, 4]
    out = {}
    for name, (k, n) in (("calls", (0, 1)), ("turns", (2, 3))):
        lo, hi = np.percentile(sums[:, k] / np.maximum(sums[:, n], 1.0), [2.5, 97.5])
        out[name] = (float(lo), float(hi))
    return out


def _summary(episodes: list, draws: int) -> dict:
    calls, turns = sum(e["calls"] for e in episodes), sum(e["turns"] for e in episodes)
    lost_calls, lost_turns = sum(e["lost_calls"] for e in episodes), sum(e["lost_turns"] for e in episodes)
    out = dict(episodes=len(episodes), calls=calls, turns=turns, lost_calls=lost_calls, lost_turns=lost_turns,
               call_tax=lost_calls / max(1, calls), turn_tax=lost_turns / max(1, turns),
               episodes_with_a_lost_turn=sum(e["lost_turns"] > 0 for e in episodes))
    if draws and episodes:
        out["interval95"] = _bootstrap(episodes, draws)
    return out


def _per_episode(header: dict, calls: list, lost) -> list:
    """Per-episode counts for a predicate ``lost(call index) -> bool``."""
    by_turn = collections.defaultdict(list)
    for i, c in enumerate(calls):
        by_turn[(c["episode"], c["turn"])].append(lost(i))
    out = [dict(calls=0, lost_calls=0, turns=int(n), lost_turns=0) for n in header["turns"]]
    for (e, _), flags in by_turn.items():
        out[e]["calls"] += len(flags)
        out[e]["lost_calls"] += sum(flags)
        out[e]["lost_turns"] += all(flags)
    return out


def replay(path: str, env_id: str | None = None, level: int = 1, seed: int = 0, draws: int = DRAWS) -> dict:
    """Replay a corpus of recorded calls on the current interface. Returns the result dict that ``report`` prints."""
    import mrgym

    from .. import interface as I
    from ..registry import retired
    header, calls = read_corpus(path)
    recorded_on = header.get("env_id")
    instances = header.get("instances")
    if instances:                                                        # every episode on the instance it was played on
        where = [tuple(x) for x in instances]
        env_id = level = seed = None
    else:
        env_id = env_id or recorded_on
        if env_id in retired:                                            # the id the corpus was recorded on is no longer served:
            env_id = retired[env_id][1] or env_id                        # its successor has the same scanner tools
        if level not in mrgym.spec(env_id).levels:
            level = mrgym.spec(env_id).default_level
        where = [(env_id, level, seed)] * len(header["turns"])
    terminal_then = header.get("terminal_tool", "submit")
    outcome, regressions = [None] * len(calls), []
    by_instance = collections.defaultdict(list)
    for i, c in enumerate(calls):
        by_instance[where[c["episode"]]].append(i)
    for (eid, lv, sd), indices in by_instance.items():                   # one environment at a time: a world is not small
        env = mrgym.make(eid, level=lv, seed=sd, max_steps=10 ** 9, max_rejected_calls=10 ** 9)   # no limit ends a replay
        cache = {}

        def fresh(env=env):
            """An episode in which nothing was scanned or delivered beyond what its reset holds: what a call meets on a
            fresh episode. (A plan or a refused call leaves it fresh, so the episode is only started again after a scan or
            a submission.)"""
            if env._episode is None or env._episode["done"] or env.state()["scans"] != fresh.scans:
                env.reset()
                fresh.scans = env.state()["scans"]
            getattr(env, "_names", {}).clear()                           # and nothing an earlier calc call assigned
        fresh.scans = None
        for i in indices:
            c = calls[i]
            key = (c["name"], json.dumps(c["arguments"], sort_keys=True))
            call = {"name": c["name"], "arguments": c["arguments"]}
            if not c["lost"]:                                            # front half only: the state-free part of the verdict
                if key not in cache:
                    fresh()
                    problems = env.admit(call)[3]
                    cls = min((p["class"] for p in problems), key=I.CLASSES.index) if problems else None
                    cache[key] = dict(cls=cls, type=next((p["type"] for p in sorted(problems, key=I._rank)), None),
                                      parameter=next((p["parameter"] for p in sorted(problems, key=I._rank)), None))
                if cache[key]["cls"] in I.INTERFACE_CLASSES and c["recorded"].startswith("ok"):
                    regressions.append(dict(index=i, name=c["name"], arguments=c["arguments"], **cache[key]))
                continue
            if ("step",) + key not in cache:
                fresh()                                                  # every lost call meets a fresh episode
                obs = env.step(call).observation
                err = obs.data.get("error") or {}
                cache[("step",) + key] = dict(cls=I.error_class(obs.data) if obs.is_error else None, type=err.get("type"),
                                              parameter=err.get("parameter"), code=err.get("code"), message=(err.get("message") or "")[:160])
            outcome[i] = cache[("step",) + key]
        env.close()
    now_lost = lambda i: bool(outcome[i]) and outcome[i]["cls"] in I.INTERFACE_CLASSES      # noqa: E731
    became, residual = collections.Counter(), collections.Counter()
    for i, c in enumerate(calls):
        if not c["lost"]:
            continue
        o = outcome[i]
        what = "accepted" if o["cls"] is None else f"refused by the instrument ({o['type']})" if o["cls"] not in I.INTERFACE_CLASSES \
            else f"still lost ({o['type']})"
        became[(c["recorded"], what)] += 1
        if now_lost(i):
            residual[(o["type"], str(o["parameter"]), c["name"], o["message"])] += 1
    return dict(
        corpus=dict(path=path, **{k: v for k, v in header.items() if k not in ("turns", "instances")}), env_id=env_id, level=level, seed=seed,
        recorded_on=recorded_on, instances=len(by_instance) if instances else None,
        environments=sorted({w[0] for w in where}),
        # calls to the delivering tool were written for the deliverable of the recorded id: on another id they are answered
        # against another schema, so their outcome says nothing about the interface the model met
        not_comparable=sum(c["name"] == terminal_then for c in calls) if not instances and env_id != recorded_on else 0,
        recorded=_summary(_per_episode(header, calls, lambda i: bool(calls[i]["lost"])), draws),
        now=_summary(_per_episode(header, calls, now_lost), draws),
        became=[dict(recorded=k[0], now=k[1], calls=n) for k, n in sorted(became.items(), key=lambda kv: (-kv[1], kv[0]))],
        residual=[dict(type=k[0], parameter=k[1], tool=k[2], message=k[3], calls=n) for k, n in sorted(residual.items(), key=lambda kv: (-kv[1], kv[0]))],
        accepted_then_refused_now=dict(checked=sum(not c["lost"] and c["recorded"].startswith("ok") for c in calls), refused=len(regressions),
                                       examples=regressions[:10]))


def from_records(folder: str, draws: int = DRAWS) -> dict:
    """The tax of every agent of a run folder, from the ``interface`` block of its ``record.json`` files (written
    by the runner for every model episode): ``{agent: summary}``."""
    by_agent = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(folder, "episodes", "*", "*", "*", "*", "*", "record.json"))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        x = rec.get("interface")
        if rec.get("status") != "completed" or not isinstance(x, dict) or rec.get("turns") is None:
            continue
        by_agent[rec["agent"]].append(dict(calls=x["calls"], lost_calls=x["shape"] + x["vocabulary"], turns=rec["turns"],
                                           lost_turns=x.get("lost_turns", 0), task_refusals=x["task_refusals"], coerced=x["coerced"],
                                           cut_replies=x.get("cut_replies", 0), idle_turns=x.get("idle_turns", 0)))
    out = {}
    for agent, eps in by_agent.items():
        out[agent] = dict(_summary(eps, draws), **{k: sum(e[k] for e in eps) for k in ("task_refusals", "coerced", "cut_replies", "idle_turns")})
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.2f} %"


def _line(label: str, s: dict) -> str:
    text = (f"  {label:<16} calls {s['lost_calls']:>4} of {s['calls']} = {_pct(s['call_tax']):>8}   turns {s['lost_turns']:>4} of {s['turns']} = "
            f"{_pct(s['turn_tax']):>8}   episodes with a lost turn {s['episodes_with_a_lost_turn']} of {s['episodes']}")
    if "interval95" in s:
        c, t = s["interval95"]["calls"], s["interval95"]["turns"]
        text += f"\n  {'':<16} 95 % interval (episode bootstrap): calls {_pct(c[0])} to {_pct(c[1])}, turns {_pct(t[0])} to {_pct(t[1])}"
    return text


def report(result: dict) -> str:
    """The text ``mrgym iface-tax --replay`` prints."""
    c, x = result["corpus"], result["accepted_then_refused_now"]
    out = [f"interface tax of the recorded calls in {c['path']}",
           f"  {c.get('description', 'recorded tool calls')}",
           (f"  replayed on the instance of each episode ({result['instances']} instances of {len(result['environments'])} environments): "
            if result.get("instances") else f"  replayed on {result['env_id']}, level {result['level']}, seed {result['seed']}: ")
           + "every call that was lost when recorded meets a fresh episode"
           + (f"\n  recorded on {result['recorded_on']}, which is no longer served; the scanner tools are the same, the deliverable is not: "
              f"{result['not_comparable']} of {result['now']['calls']} calls (those to the delivering tool) are not comparable"
              if not result.get("instances") and result.get("recorded_on") != result["env_id"] else ""),
           _line("as recorded", result["recorded"]), _line("this interface", result["now"]),
           "  what the recorded losses became"]
    out += [f"    {b['calls']:>4}  {b['recorded']:<32} -> {b['now']}" for b in result["became"]]
    if result["residual"]:
        out.append("  calls still lost to the interface")
        out += [f"    {r['calls']:>4}  {r['tool']} · {r['type']} · {r['parameter']}: {r['message']}" for r in result["residual"][:40]]
        if len(result["residual"]) > 40:
            out.append(f"    ... and {len(result['residual']) - 40} more kinds ({sum(r['calls'] for r in result['residual'][40:])} calls)")
    out.append(f"  accepted when recorded, refused now for shape or vocabulary: {x['refused']} of {x['checked']} (front half of step only; nothing executed)")
    out += [f"    {r['name']} · {r['type']} · {r['parameter']}: {json.dumps(r['arguments'])[:140]}" for r in x["examples"]]
    out.append("  a corpus the interface was designed against is a regression test, not an estimate for a new model")
    return "\n".join(out)


def cmd(a) -> int:
    if bool(a.replay) == bool(a.dir):
        raise SystemExit("mrgym iface-tax: give --replay FILE or --dir RUN_FOLDER")
    if a.dir:
        table = from_records(a.dir, a.draws)
        if a.json:
            print(json.dumps(table, indent=1))
        elif not table:
            print(f"no model episode with an interface block under {a.dir}")
        for agent, s in ([] if a.json else table.items()):
            print(f"{agent}\n{_line('this interface', s)}\n  {'':<16} refused by the instrument {s['task_refusals']} · rewritten by the canonicaliser "
                  f"{s['coerced']} · replies cut {s['cut_replies']} · replies without a call {s['idle_turns']}"
                  + ("\n  interface-limited: more than 10 % of the calls were lost to the interface" if s["call_tax"] > 0.10 else ""))
        worst = max((max(s["call_tax"], s["turn_tax"]) for s in table.values()), default=0.0)
    else:
        try:
            result = replay(a.replay, a.env, a.level, a.seed, a.draws)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(f"mrgym iface-tax: {type(e).__name__}: {e}") from None
        print(json.dumps(result, indent=1) if a.json else report(result))
        worst = max(result["now"]["call_tax"], result["now"]["turn_tax"])
        upper = max(result["now"].get("interval95", {}).get(k, (0, 0))[1] for k in ("calls", "turns"))
        if a.max_upper is not None and upper > a.max_upper:
            print(f"FAIL: the upper bound of the 95 % interval, {_pct(upper)}, is above --max-upper {_pct(a.max_upper)}")
            return 1
    if a.max is not None:
        ok = worst <= a.max
        print(f"{'PASS' if ok else 'FAIL'}: call tax and turn tax {'are' if ok else 'are not both'} at most --max {_pct(a.max)} (larger of the two: {_pct(worst)})")
        return 0 if ok else 1
    return 0


def add_parser(sub) -> None:
    """Register ``mrgym iface-tax``."""
    import argparse
    p = sub.add_parser("iface-tax", help="interface tax: share of calls and turns lost to how a call was written",
                       description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--replay", metavar="FILE.jsonl", help="corpus of recorded calls to replay on the current interface")
    p.add_argument("--dir", metavar="RUN", help="run folder of mrgym run-agent: the tax of every model in it, from its records")
    p.add_argument("--max", type=float, default=None, metavar="X", help="exit 1 if the call tax or the turn tax is above X (for example 0.03)")
    p.add_argument("--max-upper", type=float, default=None, metavar="X", help="--replay: also exit 1 if the upper bound of the 95 %% interval is above X")
    p.add_argument("--env", default=None, metavar="ID", help="--replay: environment to replay on (default: the one the corpus names)")
    p.add_argument("--level", type=int, default=1, help="--replay: level (default 1)")
    p.add_argument("--seed", type=int, default=0, help="--replay: seed of the public train split (default 0)")
    p.add_argument("--draws", type=int, default=DRAWS, help=f"bootstrap draws over episodes (default {DRAWS}; 0: no interval)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.set_defaults(fn=cmd)
