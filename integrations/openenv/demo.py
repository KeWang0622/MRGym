"""Every registered MR Gym id behind OpenEnv: in process, then over HTTP + WebSocket with OpenEnv's generic client.

Needs ``pip install openenv uvicorn`` next to ``mrgym`` (run with openenv 0.14). From anywhere:

    python integrations/openenv/demo.py [--id ID ...] [--seed 0]

For each id it replays the tool calls of the id's scripted reference policy (the probe policy for an id without
control policies) through the OpenEnv ``Environment`` class and through a served app, and compares reward,
observation digest and scanner-log head with the native API. Exit 0 when all three agree for every id.
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrgym_scanner_env"))

import uvicorn  # noqa: E402
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction  # noqa: E402
from openenv.core.generic_client import GenericEnvClient  # noqa: E402
from server.mrgym_environment import make_app, make_environment_class  # noqa: E402

import mrgym  # noqa: E402
from mrgym import compat  # noqa: E402

PORT = 18923
KEYS = ("reward", "done", "observations", "text_sha256")


def text_of(result) -> str:
    return result["content"][0]["text"] if "content" in result else result.get("text", "")


def in_process(env_id, level, seed, calls) -> dict:
    env = make_environment_class(env_id)()
    o = env.reset(seed=seed, level=level)
    names = [t.name for t in env.step(ListToolsAction()).tools]
    texts, reward, done = [o.metadata["text"]], 0.0, False
    for call in calls:
        o = env.step(CallToolAction(tool_name=call["name"], arguments=call.get("arguments") or {}))
        texts.append(text_of(o.result))
        reward, done = o.reward, o.done
        if done:
            break
    state = env.state.model_dump()
    return dict(compat.trace(reward, done, texts, env.mrgym_env), tools=names, state_keys=sorted(state))


def over_the_wire(client, env_id, level, seed, calls) -> dict:
    r = client.reset(seed=seed, env_id=env_id, level=level)
    texts, reward, done = [r.observation["metadata"]["text"]], 0.0, False
    for call in calls:
        r = client.step({"type": "call_tool", "tool_name": call["name"], "arguments": call.get("arguments") or {}})
        texts.append(text_of(r.observation["result"]))
        reward, done = r.reward, r.done
        if done:
            break
    return dict(reward=float(reward or 0.0), done=bool(done), observations=len(texts), text_sha256=compat.text_digest(texts))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--id", action="append")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ids = a.id or mrgym.list_envs()
    server = uvicorn.Server(uvicorn.Config(make_app(ids[0]), host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    ok = True
    try:
        with GenericEnvClient(base_url=f"http://127.0.0.1:{PORT}").sync() as client:
            for env_id in ids:
                level = mrgym.spec(env_id).default_level
                native = compat.scripted_episode(env_id, level, a.seed)
                calls = [c for c in native["calls"] if isinstance(c, dict) and isinstance(c.get("name"), str)]
                native = dict(compat.replay(env_id, level, a.seed, calls), policy=native["policy"])
                local = in_process(env_id, level, a.seed, calls)
                wire = over_the_wire(client, env_id, level, a.seed, calls)
                same_local = all(local[k] == native[k] for k in KEYS) and local["log_head"] == native["log_head"]
                same_wire = all(wire[k] == native[k] for k in KEYS)
                ok = ok and same_local and same_wire
                print(f"{env_id} L{level} seed {a.seed}: native {native['policy']} reward {native['reward']:g} ({len(calls)} calls) | "
                      f"in process reward {local['reward']:g}, same digest and log head: {same_local} | "
                      f"over HTTP + WebSocket reward {wire['reward']:g}, same digest: {same_wire}")
    finally:
        server.should_exit = True
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
