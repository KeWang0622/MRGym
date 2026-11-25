"""MCP face of an environment: a transport-free JSON-RPC handler.

``handle(env, request)`` answers ``initialize``, ``tools/list`` and ``tools/call`` for one live episode. A stdio or
streamable-HTTP server is this function inside a read loop; an OpenEnv ``MCPEnvironment`` registers the same tools
on a FastMCP server and routes ``CallToolAction`` to ``env.step``. Episode control (reset, reward, state) is not an
MCP tool: it stays with the process that owns the episode.
"""
import mrgym

PROTOCOL_VERSION = "2025-06-18"


def handle(env, request):
    if not isinstance(request, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "a request is a JSON object"}}
    rid, method, params = request.get("id"), request.get("method"), request.get("params")
    params = params if isinstance(params, dict) else {}             # tools/call without a name is answered by the environment

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    if method == "initialize":
        return ok({"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "scanner", "title": env.spec.id, "version": mrgym.__version__}})
    if method == "tools/list":
        return ok({"tools": env.tools("mcp")})
    if method == "tools/call":
        try:
            step = env.step({"name": params.get("name"), "arguments": params.get("arguments")})
        except mrgym.ResetNeeded as e:                      # the episode is over: a tool error, not a protocol error
            return ok({"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        return ok(step.observation.to_mcp())
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}
