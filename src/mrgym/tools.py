"""Tool definitions: one neutral form, three wire formats, and a small JSON-Schema checker.

A tool is ``ToolSpec(name, description, parameters, cost, counted, terminal)``. ``parameters`` is a JSON Schema object
restricted to the subset every function-calling API accepts (type, properties, required, enum, minimum / maximum,
items, maxItems, anyOf, additionalProperties: false). The same object is emitted as

    openai      {"type": "function", "function": {"name", "description", "parameters"}}      Chat Completions tools
    mcp         {"name", "description", "inputSchema", "annotations"}                        MCP tools/list entry
    anthropic   {"name", "description", "input_schema"}                                      Messages API tools

``cost`` / ``counted`` / ``terminal`` are environment facts the agent is told in the description and the trainer can
read from the spec: what a call is charged, whether it counts against the request limit, whether it ends the episode.
"""
from __future__ import annotations

import dataclasses
import json
import math

FORMATS = ("openai", "mcp", "anthropic", "neutral")
RESERVED = frozenset({"reset", "step", "state", "close"})       # the infrastructure boundary (same list as OpenEnv)


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    cost: str = "free"            # "free" | "scan_time"
    counted: bool = False         # counts against the scanner's request limit
    terminal: bool = False        # ends the episode

    def __post_init__(self):
        if self.name in RESERVED:
            raise ValueError(f"tool name {self.name!r} is reserved for the infrastructure boundary")

    def to(self, fmt: str = "openai") -> dict:
        if fmt == "openai":
            return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                     "parameters": self.parameters}}
        if fmt == "mcp":
            return {"name": self.name, "description": self.description, "inputSchema": self.parameters,
                    "annotations": {"readOnlyHint": self.cost == "free" and not self.terminal,
                                    "idempotentHint": self.cost == "free" and not self.terminal,
                                    "destructiveHint": False, "openWorldHint": False}}
        if fmt == "anthropic":
            return {"name": self.name, "description": self.description, "input_schema": self.parameters}
        if fmt == "neutral":
            return dataclasses.asdict(self)
        raise ValueError(f"format must be one of {FORMATS}")


def parse_tool_call(call) -> tuple[str, dict, str | None]:
    """Accept the three shapes a harness produces and return (name, arguments dict, call id or None).

        {"name": "scan", "arguments": {...}}                                              neutral / Anthropic tool_use
        {"id": "...", "type": "function", "function": {"name": .., "arguments": "<json>"}}  OpenAI tool call
        {"method": "tools/call", "params": {"name": .., "arguments": {...}}, "id": ..}      MCP JSON-RPC request
    Raises ValueError with a message that is safe to show to the model.
    """
    if not isinstance(call, dict):
        raise ValueError("a tool call is a JSON object {\"name\": ..., \"arguments\": {...}}")
    call_id = call.get("id")
    if call.get("method") == "tools/call":
        call = call.get("params") or {}
    elif isinstance(call.get("function"), dict):
        call = call["function"]
    elif call.get("type") == "tool_use":
        call = {"name": call.get("name"), "arguments": call.get("input")}
    name, args = call.get("name"), call.get("arguments", {})
    if not isinstance(name, str):
        raise ValueError("tool call has no \"name\"")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"arguments of {name} are not valid JSON: {e.msg} at character {e.pos}") from None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError(f"arguments of {name} must be a JSON object")
    return name, args, call_id


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def unsafe_values(value, schema: dict | None = None, path: str = "arguments") -> list[str]:
    """Problems that would make a tool call unusable before any tool sees it: numbers that are not finite or absurdly
    large (JSON text may carry NaN, Infinity or a 300-digit integer), and a list or an object where the schema declares
    a single value. An environment must answer these with an error, never with an exception."""
    bad: list[str] = []
    t = (schema or {}).get("type")
    props = (schema or {}).get("properties", {})
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return bad
    if isinstance(value, float):
        if not math.isfinite(value):
            bad.append(f"{path}: not a finite number")
    elif isinstance(value, int):
        if abs(value) > 10 ** 12:
            bad.append(f"{path}: number out of range")
    elif isinstance(value, dict):
        if t in ("number", "integer", "string", "boolean"):
            return [f"{path}: expected a single {t}, not an object"]
        for k, v in value.items():
            bad += unsafe_values(v, props.get(k), f"{path}.{k}")
    elif isinstance(value, list):
        if t in ("number", "integer", "string", "boolean"):
            return [f"{path}: expected a single {t}, not a list"]
        for i, v in enumerate(value):
            bad += unsafe_values(v, (schema or {}).get("items"), f"{path}[{i}]")
    else:
        bad.append(f"{path}: not a JSON value")
    return bad


def normalise(value, schema: dict):
    """Undo the string encoding some tool-call formats apply to typed arguments, where the schema says what was meant.

    Models whose native tool-call format is XML-like write every parameter as text, so a nested object arrives as a JSON
    string and a number as "10". Only strict JSON of the declared type is accepted; anything else is returned unchanged and
    then fails validation as before. No field is added, renamed or defaulted here.
    """
    if "anyOf" in schema:
        for s in schema["anyOf"]:
            v = normalise(value, s)
            if not validate(v, s):
                return v
        return value
    t = schema.get("type")
    if isinstance(value, str) and t in ("object", "array", "number", "integer", "boolean"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        ok = {"object": isinstance(parsed, dict), "array": isinstance(parsed, list), "boolean": isinstance(parsed, bool),
              "number": isinstance(parsed, (int, float)) and not isinstance(parsed, bool),
              "integer": isinstance(parsed, (int, float)) and not isinstance(parsed, bool)}[t]
        if not ok:
            return value
        value = parsed
    if t == "integer" and isinstance(value, float) and math.isfinite(value) and value == int(value):
        return int(value)                                   # 32.0 where an integer is declared: JSON has one number type
    if t == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        return {k: normalise(v, props[k]) if k in props else (normalise(v, extra) if isinstance(extra, dict) else v)
                for k, v in value.items()}
    if t == "array" and isinstance(value, list) and "items" in schema:
        return [normalise(v, schema["items"]) for v in value]
    return value


def validate(value, schema: dict, path: str = "arguments") -> list[str]:
    """Return a list of human-readable problems (empty = valid). Covers exactly the subset used by the tool schemas."""
    bad = []
    if "anyOf" in schema:
        subs = [validate(value, s, path) for s in schema["anyOf"]]
        return [] if any(not s for s in subs) else [f"{path}: matches none of the allowed forms ({'; or '.join(s[0] for s in subs)})"]
    t = schema.get("type")
    if t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return [f"{path}: expected a finite {t}"]
        if t == "integer" and int(value) != value:
            return [f"{path}: expected an integer"]
        if "minimum" in schema and value < schema["minimum"]:
            bad.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            bad.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    elif t in _TYPES:
        if not isinstance(value, _TYPES[t]) or (t != "boolean" and isinstance(value, bool)):
            return [f"{path}: expected {t}"]
    if "enum" in schema and value not in schema["enum"]:
        bad.append(f"{path}: must be one of {schema['enum']}")
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                bad.append(f"{path}.{k}: required")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                bad += validate(v, props[k], f"{path}.{k}")
            elif extra is False:
                bad.append(f"{path}.{k}: unknown field; allowed: {sorted(props)}")
            elif isinstance(extra, dict):
                bad += validate(v, extra, f"{path}.{k}")
    if t == "array":
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            bad.append(f"{path}: at most {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                bad += validate(v, schema["items"], f"{path}[{i}]")
    return bad


_ALLOWED_KEYS = {"type", "properties", "required", "enum", "minimum", "maximum", "items", "maxItems", "anyOf",
                 "additionalProperties", "description"}


def check_schema(schema: dict, path: str = "parameters") -> list[str]:
    """Lint a tool schema against the portable subset. Returns a list of problems."""
    bad = [f"{path}: keyword {k!r} is outside the portable subset" for k in schema if k not in _ALLOWED_KEYS]
    if "anyOf" in schema:
        for i, s in enumerate(schema["anyOf"]):
            bad += check_schema(s, f"{path}.anyOf[{i}]")
        return bad
    if schema.get("type") not in ("object", "array", "string", "number", "integer", "boolean"):
        bad.append(f"{path}: missing or unknown type")
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        bad += [f"{path}: required key {k!r} has no property" for k in schema.get("required", []) if k not in props]
        for k, s in props.items():
            bad += check_schema(s, f"{path}.{k}")
        if isinstance(schema.get("additionalProperties"), dict):
            bad += check_schema(schema["additionalProperties"], f"{path}.*")
    if schema.get("type") == "array" and "items" in schema:
        bad += check_schema(schema["items"], f"{path}[]")
    return bad
