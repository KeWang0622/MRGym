"""Tool definitions: one neutral form, three wire formats, and a small JSON-Schema checker.

A tool is ``ToolSpec(name, description, parameters, cost, counted, terminal, examples, aliases, protocol, lines, see)``.
``parameters`` is a JSON Schema object restricted to the subset every function-calling API accepts (type, properties,
required, enum, minimum / maximum, items, maxItems, anyOf, additionalProperties: false). The same object is emitted as

    openai      {"type": "function", "function": {"name", "description", "parameters"}}      Chat Completions tools
    mcp         {"name", "description", "inputSchema", "annotations"}                        MCP tools/list entry
    anthropic   {"name", "description", "input_schema"}                                      Messages API tools

``cost`` / ``counted`` / ``terminal`` are environment facts the agent is told in the description and the trainer can
read from the spec: what a call is charged, whether it reaches the scanner's log, whether it ends the episode.
``examples`` are valid argument objects; the first one is printed at the end of the description in every wire format
(a schema cannot say when an optional parameter is written, an example can).

Limits on what a call may carry (``MAX_*``): anything deeper, larger or longer is answered with an error before any
tool sees it, so no input can exhaust the stack or the memory of the process that runs the episode.
"""
from __future__ import annotations

import dataclasses
import json
import math

FORMATS = ("openai", "mcp", "anthropic", "neutral")
RESERVED = frozenset({"reset", "step", "state", "close"})       # the infrastructure boundary (same list as OpenEnv)
MAX_DEPTH = 16                  # nesting levels of the arguments
MAX_NODES = 2000                # values (objects, arrays, scalars) in the arguments
MAX_STRING = 20_000             # characters of one string value
MAX_ARGUMENT_CHARS = 65_536     # characters of arguments that arrive as JSON text
MAX_NUMBER = 10 ** 12           # absolute value of an integer


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    cost: str = "free"            # "free" | "scan_time"
    counted: bool = False         # reaches the scanner's log (plan, scan)
    terminal: bool = False        # ends the episode
    examples: tuple = ()          # valid argument objects; the first is shown in the description
    aliases: dict = dataclasses.field(default_factory=dict)   # {accepted spelling: parameter name}
    protocol: bool = False        # the console's protocol parameters are top-level arguments of this tool
    lines: tuple = ()             # string parameters that also accept a list of strings (joined by newlines)
    see: str = ""                 # the tool that documents the parameters this one lists without a description

    def __post_init__(self):
        if self.name in RESERVED:
            raise ValueError(f"tool name {self.name!r} is reserved for the infrastructure boundary")

    def text(self) -> str:
        """The description with one example call, as every wire format carries it."""
        if not self.examples:
            return self.description
        return f"{self.description} Example: {self.name} {json.dumps(self.examples[0], separators=(', ', ': '))}"

    def to(self, fmt: str = "openai") -> dict:
        if fmt == "openai":
            return {"type": "function", "function": {"name": self.name, "description": self.text(),
                                                     "parameters": self.parameters}}
        if fmt == "mcp":
            return {"name": self.name, "description": self.text(), "inputSchema": self.parameters,
                    "annotations": {"readOnlyHint": self.cost == "free" and not self.terminal,
                                    "idempotentHint": self.cost == "free" and not self.terminal,
                                    "destructiveHint": False, "openWorldHint": False}}
        if fmt == "anthropic":
            return {"name": self.name, "description": self.text(), "input_schema": self.parameters}
        if fmt == "neutral":
            return dict(name=self.name, description=self.description, parameters=self.parameters, cost=self.cost,
                        counted=self.counted, terminal=self.terminal, examples=list(self.examples))
        raise ValueError(f"format must be one of {FORMATS}")


def _no_constant(name):
    raise ValueError(f"{name} is not a JSON number")


def parse_tool_call(call) -> tuple[str, dict, str | None]:
    """Accept the shapes a harness produces and return (name, arguments dict, call id or None).

        {"name": "scan", "arguments": {...}}                                              neutral
        {"id": "...", "type": "function", "function": {"name": .., "arguments": "<json>"}}  OpenAI tool call
        {"method": "tools/call", "params": {"name": .., "arguments": {...}}, "id": ..}      MCP JSON-RPC request
        {"type": "tool_use", "id": .., "name": .., "input": {...}}                          Anthropic tool_use
    Raises ValueError with a message that is safe to show to the model. Never raises anything else: every unwrapping
    step checks the type of what it unwraps.
    """
    shape = "a tool call is a JSON object {\"name\": ..., \"arguments\": {...}}"
    if not isinstance(call, dict):
        raise ValueError(shape)
    call_id = call.get("id")
    if not isinstance(call_id, (str, int)) or isinstance(call_id, bool):
        call_id = None
    if call.get("method") == "tools/call":
        call = call.get("params")
    elif "function" in call and not isinstance(call.get("name"), str):
        call = call.get("function")
    elif call.get("type") == "tool_use":
        call = {"name": call.get("name"), "arguments": call.get("input")}
    if not isinstance(call, dict):
        raise ValueError(shape)
    name, args = call.get("name"), call.get("arguments", {})
    if not isinstance(name, str) or not name or len(name) > 64:
        raise ValueError("tool call has no \"name\"")
    if isinstance(args, str):
        if len(args) > MAX_ARGUMENT_CHARS:
            raise ValueError(f"arguments of {name} are longer than {MAX_ARGUMENT_CHARS} characters")
        try:
            args = json.loads(args, parse_constant=_no_constant) if args.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"arguments of {name} are not valid JSON: {e.msg} at character {e.pos}") from None
        except RecursionError:
            raise ValueError(f"arguments of {name} are nested too deeply") from None
        except ValueError as e:                             # NaN, Infinity, a number too long to convert
            raise ValueError(f"arguments of {name} are not valid JSON: {str(e)[:80]}") from None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError(f"arguments of {name} must be a JSON object")
    return name, args, call_id


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


def unsafe_values(value, schema: dict | None = None, path: str = "arguments") -> list[str]:
    """Problems that make a tool call unusable before any tool sees it: nesting deeper than ``MAX_DEPTH``, more than
    ``MAX_NODES`` values, a string longer than ``MAX_STRING``, a number that is not finite or absurdly large (JSON
    text may carry NaN, Infinity or a 300-digit integer), a key that is not a string, a value that is not JSON.

    The walk uses its own stack, so the depth of the input can never become the depth of the interpreter's stack.
    ``schema`` is accepted for compatibility and not used: types are the business of ``validate``.
    """
    bad: list[str] = []
    stack, nodes = [(value, path, 0)], 0
    while stack and len(bad) < 8:
        v, p, depth = stack.pop()
        nodes += 1
        if nodes > MAX_NODES:
            return [f"{path}: too large (more than {MAX_NODES} values)"]
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, str):
            if len(v) > MAX_STRING:
                bad.append(f"{p}: text longer than {MAX_STRING} characters")
        elif isinstance(v, float):
            if not math.isfinite(v):
                bad.append(f"{p}: not a finite number")
        elif isinstance(v, int):
            if abs(v) > MAX_NUMBER:
                bad.append(f"{p}: number out of range")
        elif isinstance(v, (dict, list)):
            if depth >= MAX_DEPTH:
                return [f"{path}: nested too deeply (more than {MAX_DEPTH} levels)"]
            if nodes + len(v) > MAX_NODES:                               # before anything is built for its items
                return [f"{path}: too large (more than {MAX_NODES} values)"]
            if isinstance(v, dict):
                for k, x in v.items():
                    if not isinstance(k, str) or len(k) > 128:
                        bad.append(f"{p}: a key is not a short string")
                        break
                    stack.append((x, f"{p}.{k}", depth + 1))
            else:
                stack.extend((x, f"{p}[{i}]", depth + 1) for i, x in enumerate(v))
        else:
            bad.append(f"{p}: not a JSON value")
    return bad


def _show(value, limit: int = 60) -> str:
    """A value as short ASCII text for a message (never the raw bytes of what the agent sent)."""
    try:
        text = json.dumps(value, default=lambda o: type(o).__name__)
    except (TypeError, ValueError, RecursionError):
        text = type(value).__name__
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _join(path: str, key) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


def _is_number(value) -> bool:
    """A finite JSON number. An integer beyond ``MAX_NUMBER`` is not one (it cannot even be compared as a float)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= MAX_NUMBER
    return isinstance(value, float) and math.isfinite(value)


def problems(value, schema: dict, path: str = "") -> list[dict]:
    """Structured problems of ``value`` against ``schema`` (empty = valid). Covers exactly the subset used by the
    tool schemas. Each problem is ``{"parameter", "kind", "message"}`` plus what a repair needs:

        kind      type | integer | enum | range | required | unknown | items
        requested the offending value (scalars only)        allowed   {"min", "max"} / {"values"} / {"type"}
        nearest_valid  the closest value the schema accepts, where one exists
    """
    if "anyOf" in schema:
        subs = [problems(value, s, path) for s in schema["anyOf"]]
        if any(not s for s in subs):
            return []
        for s, sub in zip(schema["anyOf"], subs):             # the branch of the value's own type says most
            if (s.get("type") in ("number", "integer") and _is_number(value)) or (s.get("type") == "string" and isinstance(value, str)):
                if s.get("type") == "string" and "enum" in s:
                    num = next((x for x in schema["anyOf"] if x.get("type") in ("number", "integer")), None)
                    words = " or ".join(json.dumps(e) for e in s["enum"])
                    return [dict(parameter=path, kind="enum", requested=value, allowed={"values": list(s["enum"]), "type": num["type"] if num else None},
                                 message=f"{_show(value)} is neither a number nor {words}", nearest_valid=s["enum"][0])]
                return sub
        forms = " or ".join(json.dumps(e) for s in schema["anyOf"] for e in s.get("enum", [])) or "the allowed forms"
        return [dict(parameter=path, kind="type", message=f"expected a number or {forms}, got {_show(value)}")]
    out, t = [], schema.get("type")
    if t in ("number", "integer"):
        if not _is_number(value):
            return [dict(parameter=path, kind="type", allowed={"type": t}, message=f"expected {'an integer' if t == 'integer' else 'a number'}, got {_show(value)}")]
        if t == "integer" and int(value) != value:
            lo, hi = math.floor(value), math.ceil(value)
            return [dict(parameter=path, kind="integer", requested=value, allowed={"type": "integer"}, nearest_valid=[lo, hi],
                         message=f"{value} is not an integer (nearest: {lo}, {hi})")]
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            near = lo if lo is not None and value < lo else hi
            out.append(dict(parameter=path, kind="range", requested=value, allowed={"min": lo, "max": hi}, nearest_valid=near,
                            message=f"{value} is {'below the minimum' if near == lo and value < near else 'above the maximum'} {near}"))
    elif t in _TYPES:
        if not isinstance(value, _TYPES[t]) or (t != "boolean" and isinstance(value, bool)):
            return [dict(parameter=path, kind="type", allowed={"type": t}, message=f"expected {'an ' if t in ('object', 'array') else 'a '}{t}, got {_show(value)}")]
    if "enum" in schema and not any(value == e and type(value) is type(e) for e in schema["enum"]):
        out.append(dict(parameter=path, kind="enum", requested=value if isinstance(value, (str, int, float)) else None,
                        allowed={"values": list(schema["enum"])}, message=f"{_show(value)} is not one of {json.dumps(schema['enum'])}"))
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                out.append(dict(parameter=_join(path, k), kind="required", message="required"))
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                out += problems(v, props[k], _join(path, k))
            elif extra is False:
                out.append(dict(parameter=_join(path, k), kind="unknown", allowed={"values": list(props)},
                                message=f"unknown parameter; allowed: {', '.join(props)}"))
            elif isinstance(extra, dict):
                out += problems(v, extra, _join(path, k))
    if t == "array":
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(dict(parameter=path, kind="items", allowed={"max_items": schema["maxItems"]},
                            message=f"{len(value)} items; at most {schema['maxItems']}"))
        if "items" in schema:
            for i, v in enumerate(value[:schema.get("maxItems", MAX_NODES)]):
                out += problems(v, schema["items"], _join(path, i))
    return out


def validate(value, schema: dict, path: str = "arguments") -> list[str]:
    """Human-readable problems (empty = valid): the messages of ``problems``, each led by its path."""
    return [f"{_join(path, p['parameter']) if p['parameter'] else path}: {p['message']}" for p in problems(value, schema)]


_ALLOWED_KEYS = {"type", "properties", "required", "enum", "minimum", "maximum", "items", "maxItems", "anyOf",
                 "additionalProperties", "description"}
MAX_DESCRIPTION = 1024


def check_schema(schema: dict, path: str = "parameters", _top: bool = True, described: bool = True) -> list[str]:
    """Lint a tool schema against the portable subset. Returns a list of problems.

    Besides the keywords: the arguments are one object, and every parameter of it carries a description (a model
    reads nothing else about it). ``described=False`` lifts that for a tool whose parameters are documented by
    another tool (``ToolSpec.see``): the same names, written once.
    """
    bad = [f"{path}: keyword {k!r} is outside the portable subset" for k in schema if k not in _ALLOWED_KEYS]
    if _top and schema.get("type") != "object":
        return bad + [f"{path}: the arguments of a tool are one object"]
    if "anyOf" in schema:
        for i, s in enumerate(schema["anyOf"]):
            bad += check_schema(s, f"{path}.anyOf[{i}]", False)
        return bad
    if schema.get("type") not in ("object", "array", "string", "number", "integer", "boolean"):
        bad.append(f"{path}: missing or unknown type")
    if len(schema.get("description", "")) > MAX_DESCRIPTION:
        bad.append(f"{path}: description longer than {MAX_DESCRIPTION} characters")
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        bad += [f"{path}: required key {k!r} has no property" for k in schema.get("required", []) if k not in props]
        for k, s in props.items():
            if _top and described and not str(s.get("description", "")).strip():
                bad.append(f"{path}.{k}: parameter without a description")
            bad += check_schema(s, f"{path}.{k}", False)
        if isinstance(schema.get("additionalProperties"), dict):
            bad += check_schema(schema["additionalProperties"], f"{path}.*", False)
    if schema.get("type") == "array" and "items" in schema:
        bad += check_schema(schema["items"], f"{path}[]", False)
    return bad


def check_tool(spec: ToolSpec) -> list[str]:
    """Lint one tool: its schema (``check_schema``), its description, and that every example is a valid call."""
    bad = check_schema(spec.parameters, spec.name, described=not spec.see)
    if not spec.description.strip() or len(spec.text()) > MAX_DESCRIPTION:
        bad.append(f"{spec.name}: the description is empty or longer than {MAX_DESCRIPTION} characters with its example")
    for i, ex in enumerate(spec.examples):
        bad += [f"{spec.name}: example {i}: {m}" for m in validate(ex, spec.parameters)]
    if spec.protocol:
        for k, s in spec.parameters.get("properties", {}).items():
            if s.get("type") == "object":
                bad.append(f"{spec.name}.{k}: a protocol tool takes flat arguments, not an object")
    return bad
