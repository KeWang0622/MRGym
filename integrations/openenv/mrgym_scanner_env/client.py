"""Client of the MR Gym OpenEnv server: OpenEnv's generic client plus two helpers that know nothing about any tool.

    with MRGymScannerEnv(base_url="http://localhost:8000").sync() as client:
        r = client.reset(seed=3, env_id="Scanner/NoWrap-v1", level=1)
        tools = client.step(list_tools()).observation["tools"]
        r = client.step(call_tool("scanner_info", {}))        # r.observation["result"]["content"][0]["text"], r.reward, r.done
"""
from openenv.core.generic_client import GenericEnvClient


class MRGymScannerEnv(GenericEnvClient):
    """One WebSocket session = one episode at a time on the server."""


def list_tools() -> dict:
    return {"type": "list_tools"}


def call_tool(name: str, arguments: dict | None = None) -> dict:
    return {"type": "call_tool", "tool_name": name, "arguments": arguments or {}}
