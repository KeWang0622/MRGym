"""Play the Inspect task with a scripted mock model, one process, no network, no model call.

    python integrations/inspect/mock_run.py [--seeds 3] [--id ID ...] [--json]

For every registered id (found through ``mrgym.list_envs()``) and ``--seeds`` seeds:
    reference   the mock model replays the tool calls of the id's scripted reference policy. Expected: the same reward,
                observation digest and scanner-log head as the native API (reward 1.0 where the reference passes natively;
                an id whose reference does not pass natively is compared the same way and listed).
    no-op       the mock model never calls a tool. Expected: the floor of the id (0.0).
An id without control policies (a sandbox) is played with the probe policy and has no reward to reach.
One eval per seed, because the task statement can be identical across instances and cannot key the script.
Exit 0 when every expectation holds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402
from mrgym_inspect import mrgym_scanner  # noqa: E402

import mrgym  # noqa: E402
from mrgym import compat  # noqa: E402

KEYS = ("reward", "done", "observations", "text_sha256", "log_head")


def play(env_id, level, seed, calls, log_dir):
    """One eval of one sample with a mock model that emits ``calls`` (none = a model that never uses a tool)."""
    def outputs(input, tools, tool_choice, config):
        n = sum(1 for m in input if m.role == "assistant")
        if n >= len(calls):
            return ModelOutput.from_content("mockllm/model", "I have nothing further to do.")
        return ModelOutput.for_tool_call("mockllm/model", calls[n]["name"], calls[n].get("arguments", {}))

    log = inspect_eval(mrgym_scanner(env_id, level, f"{seed}-{seed}"), model=get_model("mockllm/model", custom_outputs=outputs),
                       log_dir=log_dir, display="none")[0]
    if log.status != "success" or not log.samples or not log.samples[0].scores:
        return dict(status=log.status, error=str(log.error)[:300] if log.error else None)
    score = log.samples[0].scores["mrgym_reward"]
    return dict(status=log.status, passed=score.value["passed"], **{k: score.metadata.get(k) for k in KEYS},
                undescribed_parameters=score.metadata.get("undescribed_parameters"))


def schemas(env_id, level, seed) -> dict:
    env = mrgym.make(env_id, level=level, seed=seed)
    env.reset()
    out = {t["name"]: t["parameters"] for t in env.tools("neutral")}
    env.close()
    return out


def _valid(call, by_name) -> bool:
    from mrgym import tools as T
    if not isinstance(call, dict) or call.get("name") not in by_name or not isinstance(call.get("arguments", {}), dict):
        return False
    return not T.validate(call.get("arguments", {}), by_name[call["name"]])


def run(ids=None, seeds=3, verbose=True):
    rows = []
    with tempfile.TemporaryDirectory() as log_dir:
        for env_id in ids or mrgym.list_envs():
            spec = mrgym.spec(env_id)
            for seed in range(seeds):
                scripted = compat.scripted_episode(env_id, spec.default_level, seed)
                # Inspect validates the arguments of a tool call against the schema itself and answers a violation with its
                # own message, so only schema-valid calls reach the environment: those are the ones compared.
                calls = [c for c in scripted["calls"] if _valid(c, schemas(env_id, spec.default_level, seed))]
                native = dict(compat.replay(env_id, spec.default_level, seed, calls), policy=scripted["policy"])
                ref = play(env_id, spec.default_level, seed, calls, log_dir)
                noop = play(env_id, spec.default_level, seed, [], log_dir)
                floor = float(spec.reward_range[0])
                ok = (ref.get("status") == "success" and all(ref.get(k) == native[k] for k in KEYS)
                      and noop.get("status") == "success" and noop.get("reward") == floor and not noop.get("done"))
                # An id whose scripted reference does not pass natively cannot pass here either: it is compared like every
                # other id (equal reward, digest and log head) and reported, not counted as a failure of the adapter.
                native_passes = native["policy"] != "probe" and native["done"] and native["reward"] >= spec.reward_threshold
                rows.append(dict(id=env_id, level=spec.default_level, seed=seed, policy=native["policy"], ok=bool(ok), native_passes=native_passes,
                                 native={k: native[k] for k in KEYS}, reference=ref, noop=noop))
                if verbose:
                    print(f"{env_id} L{spec.default_level} seed {seed}: native {native['policy']} reward {native['reward']:.3g} "
                          f"({len(calls)} calls) | inspect {native['policy']} reward {ref.get('reward')} status {ref.get('status')} "
                          f"same digest and log head: {all(ref.get(k) == native[k] for k in KEYS)} | inspect no-op reward {noop.get('reward')} "
                          f"-> {'OK' if ok else 'MISMATCH'}", flush=True)
    return rows


def main(argv=None) -> int:
    import inspect_ai
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--id", action="append")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = run(a.id, a.seeds, verbose=not a.json)
    ok = sum(r["ok"] for r in rows)
    passing = [r for r in rows if r["native_passes"]]
    not_passing = sorted({r["id"] for r in rows if r["policy"] != "probe" and not r["native_passes"]})
    summary = dict(inspect_ai=inspect_ai.__version__, mrgym=mrgym.__version__, engine=mrgym.ENGINE_ID, episodes=len(rows), ok=ok,
                   reference_reward_1=sum(r["reference"].get("reward") == 1.0 for r in passing), scored_episodes=len(passing),
                   native_reference_does_not_pass=not_passing,
                   noop_reward_0=sum(r["noop"].get("reward") == 0.0 for r in rows),
                   undescribed_parameters=max((r["reference"].get("undescribed_parameters") or 0 for r in rows), default=0))
    if a.json:
        print(json.dumps(dict(summary, rows=rows), indent=1))
    else:
        print(f"inspect-ai {summary['inspect_ai']} mock model: {ok} of {len(rows)} episodes match the native API; reference reward 1.0 on "
              f"{summary['reference_reward_1']} of {len(passing)} episodes whose native reference passes; no-op reward 0.0 on "
              f"{summary['noop_reward_0']} of {len(rows)}; tool parameters without a description (filled with the parameter name): "
              f"{summary['undescribed_parameters']}" + (f"; ids whose scripted reference does not pass natively on this engine: {not_passing}" if not_passing else ""))
    return 0 if rows and ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
