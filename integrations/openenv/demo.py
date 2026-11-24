"""``Scanner/NoWrap-v0`` behind OpenEnv: in process, then over HTTP + WebSocket with OpenEnv's GenericEnvClient.

Needs ``pip install openenv uvicorn`` next to ``mrgym`` (written against openenv 0.14.0). Run from anywhere:

    python integrations/openenv/demo.py

It plays three things: a few tool calls in process, the same calls over the wire, and the environment's reference
control policy (``mrgym.policies``) over the wire, unchanged.
"""
import threading
import time
from types import SimpleNamespace

import uvicorn
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.generic_client import GenericEnvClient
from openenv_server import make_app, make_environment_class

import mrgym
from mrgym.policies import ToolScanner, controls_of

ID = "Scanner/NoWrap-v0"
LEVEL, SEED, PORT = 1, 0, 18923
LOCALISER = {"sequence": "spgr", "tr_ms": 10.0, "flip_deg": 10.0, "fov_mm": 288.0, "matrix": 160, "averages": 2}
SAFE = {"pe_dir": "y", "fov_mm": 260.0, "phase_fov": 1.0, "matrix": 130}      # valid for any object, so it earns nothing


def first_line(result):
    return result["content"][0]["text"].splitlines()[0]


def in_process():
    env = make_environment_class(ID)()
    o = env.reset(seed=SEED, level=LEVEL)
    print("[in-process] reset:", o.metadata["text"].splitlines()[0], "| budget", o.metadata["task"]["budget_s"], "s")
    print("[in-process] tools:", [t.name for t in env.step(ListToolsAction()).tools])
    o = env.step(CallToolAction(tool_name="scan", arguments={"protocol": LOCALISER}))
    print("[in-process] scan ->", first_line(o.result), "| reward", o.reward, "| done", o.done)
    o = env.step(CallToolAction(tool_name="scan", arguments={"protocol": dict(LOCALISER, tr_ms=1.0)}))
    print("[in-process] refused scan -> isError", o.result["isError"], "|", o.result["structuredContent"]["error"]["type"])
    o = env.step(CallToolAction(tool_name="submit", arguments=SAFE))
    print("[in-process] submit the safe geometry -> reward", round(o.reward, 4), "| done", o.done)
    print("[in-process] state():", env.state.model_dump())


class Remote:
    """The two members ``ToolScanner`` uses (``step``, ``state``), served by an OpenEnv client."""

    def __init__(self, client):
        self.client = client

    def step(self, call):
        r = self.client.step({"type": "call_tool", "tool_name": call["name"], "arguments": call["arguments"]})
        res = r.observation["result"]
        obs = SimpleNamespace(is_error=res["isError"], data=res["structuredContent"], text=res["content"][0]["text"])
        return obs, r.reward, r.done, False, {}

    def state(self):
        st = self.client.state()
        return st["public"] if isinstance(st, dict) else st.public


def over_the_wire():
    server = uvicorn.Server(uvicorn.Config(make_app(ID), host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        with GenericEnvClient(base_url=f"http://127.0.0.1:{PORT}").sync() as client:
            r = client.reset(seed=SEED, level=LEVEL)
            print("[http/ws] reset:", r.observation["metadata"]["text"].splitlines()[0])
            r = client.step({"type": "list_tools"})
            print("[http/ws] tools:", [t["name"] for t in r.observation["tools"]])
            r = client.step({"type": "call_tool", "tool_name": "scan", "arguments": {"protocol": LOCALISER}})
            print("[http/ws] scan ->", first_line(r.observation["result"]), "| reward", r.reward, "| done", r.done)
            r = client.step({"type": "call_tool", "tool_name": "submit", "arguments": SAFE})
            print("[http/ws] submit the safe geometry -> reward", round(r.reward, 4), "| done", r.done)

            # a second episode on the same connection: the reference control policy, unchanged, playing over the wire
            client.reset(seed=SEED, level=LEVEL)
            remote = Remote(client)
            answer = controls_of(mrgym.spec(ID)).reference(ToolScanner(remote))
            r = client.step({"type": "call_tool", "tool_name": "submit", "arguments": answer})
            st = remote.state()
            print("[http/ws] reference policy over the wire: reward", r.reward, "| done", r.done, "| scans", st["scans"],
                  "| scan time used", st["scan_time_used_s"], "s")
    finally:
        server.should_exit = True


if __name__ == "__main__":
    in_process()
    over_the_wire()
