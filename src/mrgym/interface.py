"""The agent-facing contract: how a written call becomes the call that runs, and how a refusal is said.

Two things live here and nowhere else.

**The canonicaliser** (``canonicalise``). A model writes the same call in many spellings: every value as text, the
protocol nested or flat, ``"minimum"`` for ``"min"``, ``True`` for ``true``, ``"240 mm"`` for ``240``. A spelling is
accepted when it has exactly one reading under the tool's schema and that reading is not a physical conversion; the
call is then rewritten to its canonical form (one byte sequence per meaning) and every rewrite is logged as
``{"rule", "parameter", "from", "to"}``. The rules:

    C1   JSON text where an object, array, number, integer or boolean is declared; 32.0 where an integer is declared
    C2   True / False / None; a single-quoted object or array (Python literal)
    C3   null, "", "default", "auto" on an optional parameter                               -> omitted
    C4   "minimum", "shortest", "console minimum" where "min" may be written                -> "min"
    C5   a number followed by the unit of the parameter's own name ("240 mm", "500ms")      -> the number
    C6   an enum value that differs in case, surrounding spaces or quotes                   -> the enum value
    C7   {"protocol": {...}} (object or text) on a protocol tool; one wrapper key ``arguments`` / ``parameters`` /
         ``input``                                                                          -> flat arguments
    C8   regions "task" / "all" / ["task"]  -> ["all"];  circles written under ``regions``  -> ``circles``;
         one circle object where a list is declared                                         -> a one-element list
    C9   a tool whose only parameter is an enum, called with the value as key or under another key
    C10  an alias of a parameter name (``ToolSpec.aliases``)                                -> the parameter name
    C11  a parameter the console fixes for the sequence, sent with the fixed value          -> dropped
    C12  a list of strings where the tool takes lines of text (``ToolSpec.lines``)          -> joined by newlines
    C13  a value cut by an unclosed ``<parameter=name>`` tag of an XML-style tool-call format -> split at the tag
    C14  ``regions`` / ``circles`` on a protocol tool that reads nothing out (``plan``)       -> dropped
    C15  the fields of a nested object written next to it (a deliverable ``{"measured": {...}}`` sent flat), when
         each name belongs to exactly one object parameter                                  -> moved into it

Never applied, always answered with an error that names the repair: a misspelt name, another unit (``"24 cm"``,
``"0.5 s"``), percent for a fraction, rounding a non-integer or an odd matrix, clipping to a range, dropping an
unknown parameter, a parameter of another sequence with another value than the fixed one, synonyms of an enum value.
The scan that runs is the scan that was asked for, or none.

**The error contract** (``problem``, ``error_observation``). One closed vocabulary of error types. Every refusal is
an observation that names the parameter, what is allowed, the nearest valid value, a ``fix`` (a patch which, applied
to the call, gives a call that is not refused for the same reason) and what was charged. No refusal ends the episode.

    type                 class        counts as a tool call
    malformed_call       shape        no     not JSON, no tool name, nested too deeply, too large
    unknown_tool         vocabulary   no
    unknown_parameter    vocabulary   no     a name the tool does not have (``did_you_mean``, never applied); with
                                             ``code`` NOT_SIMULATED a parameter of real consoles this scanner does
                                             not model, with ``code`` VENDOR_TERM a vendor's word for a parameter
                                             in another unit (the conversion is stated, never applied)
    not_applicable       vocabulary   no     a parameter of another sequence
    invalid_value        per problem  no     wrong type (shape), not in the enum (vocabulary), out of a fixed range,
                                             odd matrix, missing required, another unit (task)
    console_limit        task         yes    the console cannot realise the protocol (``code`` = the console's code)
    budget               task         yes    scan time, request limit
    not_found            task         yes    no stored scan / image of that index, unknown region id
    invalid_deliverable  task         no     a submission the console cannot evaluate (at most 3 per episode)
    episode_over         task         no
    internal_error       internal     no     a fault of the environment, never of the agent; counted for the trainer

The class says who lost the turn. ``shape`` and ``vocabulary`` are the interface tax: the model knew what it wanted
and the interface did not take it. ``task`` is the instrument saying no, which is part of the job.
"""
from __future__ import annotations

import ast
import difflib
import json
import math
import re

from . import tools as T
from .scanner import console

# type: (class, or None when each problem carries its own; whether the call counts as an executed tool call)
ERRORS = {
    "malformed_call": ("shape", False),
    "unknown_tool": ("vocabulary", False),
    "unknown_parameter": ("vocabulary", False),
    "not_applicable": ("vocabulary", False),
    "invalid_value": (None, False),
    "console_limit": ("task", True),
    "budget": ("task", True),
    "not_found": ("task", True),
    "invalid_deliverable": ("task", False),
    "episode_over": ("task", False),
    "internal_error": ("internal", False),
}
ERROR_TYPES = tuple(ERRORS)
CLASSES = ("shape", "vocabulary", "task", "internal")
INTERFACE_CLASSES = ("shape", "vocabulary")                 # what the interface-tax metric counts
MAX_PROBLEMS = 8                                            # listed per refusal; the rest is counted in ``more``
MAX_REJECTED_CALLS = 8                                      # per episode
MAX_INVALID_DELIVERABLES = 3                                # per episode, within the rejected calls

MIN_WORDS = frozenset({"min", "minimum", "shortest", "console minimum"})
OMIT_WORDS = frozenset({"", "default", "auto", "null", "none"})
WRAPPERS = ("arguments", "parameters", "input")
ALL_REGIONS = "all"
REGION_WORDS = frozenset({"task", "all"})
# suffix of a parameter name: (spellings of its own unit, {another unit: (factor to the own unit, name of the own unit)})
UNITS = {
    "_ms": (("ms", "msec", "millisecond", "milliseconds"), {"s": 1e3, "sec": 1e3, "seconds": 1e3, "us": 1e-3, "µs": 1e-3}, "milliseconds"),
    "_mm": (("mm", "millimetre", "millimetres", "millimeter", "millimeters"), {"cm": 10.0, "m": 1e3}, "millimetres"),
    "_deg": (("deg", "degree", "degrees", "°"), {"rad": 180.0 / math.pi, "radians": 180.0 / math.pi}, "degrees"),
    "_hz_per_px": (("hz/px", "hz per px", "hz/pixel", "hz per pixel", "hz"), {"khz": 1e3, "khz/px": 1e3}, "Hz per pixel"),
    "_hz": (("hz",), {"khz": 1e3}, "Hz"),
}
FRACTIONS = ("phase_fov", "phase_oversampling", "partial_fourier")       # and every name that ends in _frac
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_WITH_UNIT = re.compile(rf"^\s*({_NUMBER})\s*([a-zA-Zµ°%/ ]+?)\s*$")
_XML_TAG = re.compile(r"\n?<parameter=([A-Za-z_][A-Za-z0-9_]*)>\n?")
_show = T._show


# ------------------------------------------------------------------------------------------------ problems
def problem(type: str, parameter, message: str, cls: str | None = None, **extra) -> dict:
    """One reason a call is refused. ``extra``: code, requested, allowed, nearest_valid, fix, did_you_mean."""
    if type not in ERRORS:
        raise ValueError(f"error type {type!r} is not in the closed vocabulary")
    return {"type": type, "class": cls or ERRORS[type][0] or "task", "parameter": parameter, "message": message,
            **{k: v for k, v in extra.items() if v is not None}}


def _rank(p: dict):
    return CLASSES.index(p["class"]), ERROR_TYPES.index(p["type"])


def merged_fix(problems: list) -> dict | None:
    """The patch that repairs every listed problem: ``{parameter: new value}``; ``null`` removes the parameter. Where
    two problems move the same number the same way (a repetition time that is both below its minimum and too short
    for the RF power limit), the patch carries the value that satisfies both."""
    fix = {}
    for p in problems:
        for k, v in (p.get("fix") or {}).items():
            was, asked = fix.get(k), p.get("requested")
            if k in fix and all(T._is_number(x) for x in (v, was, asked)) and (v > asked) == (was > asked):
                fix[k] = max(was, v) if v > asked else min(was, v)
            else:
                fix.setdefault(k, v)
    return fix or None


def apply_fix(arguments: dict, fix: dict | None) -> dict:
    """The call with the patch applied (a new dict). A ``None`` value removes the parameter."""
    out = dict(arguments)
    for k, v in (fix or {}).items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    return out


def error_data(problems: list) -> dict:
    """``Observation.data`` of a refusal: the first problem (the most basic one) at the top, all of them listed."""
    ordered = sorted(problems, key=_rank)
    first = ordered[0]
    error = {k: first.get(k) for k in ("type", "class", "code", "parameter", "message", "requested", "allowed", "nearest_valid", "did_you_mean")
             if k in ("type", "class", "parameter", "message") or first.get(k) is not None}
    error["fix"] = merged_fix(ordered[:MAX_PROBLEMS])
    error["problems"] = [{k: v for k, v in p.items() if k != "class"} for p in ordered[:MAX_PROBLEMS]]
    error["more"] = max(0, len(ordered) - MAX_PROBLEMS)
    return {"error": error}


def error_text(data: dict) -> str:
    """What the model reads: the error type, then the patch that repairs the call (``fix``, where there is one), then
    one line per problem with the parameter first."""
    e = data["error"]
    lines = []
    for i, p in enumerate(e["problems"]):
        head = f"error {p['type']}" if i == 0 else f"also {p['type']}"
        if i == 0 and e.get("fix"):
            head += " · fix: " + json.dumps(e["fix"], separators=(", ", ": ")) + (" (null: leave it out)" if None in e["fix"].values() else "")
        lines.append(f"{head} · " + (f"{p['parameter']}: " if p.get("parameter") else "") + p["message"])
    if e["more"]:
        lines.append(f"... and {e['more']} more problem(s)")
    return "\n".join(lines)


def error_class(data: dict) -> str | None:
    """Class of an error observation's data (``shape`` | ``vocabulary`` | ``task`` | ``internal``), None if it is no error."""
    e = (data or {}).get("error")
    return e.get("class") if isinstance(e, dict) else None


def counts_as_tool_call(data: dict) -> bool:
    e = (data or {}).get("error")
    return ERRORS.get(e.get("type"), (None, False))[1] if isinstance(e, dict) else True


def did_you_mean(word, candidates):
    if not isinstance(word, str):
        return None
    hit = difflib.get_close_matches(word.lower(), [c for c in candidates if isinstance(c, str)], n=1, cutoff=0.6)
    return hit[0] if hit else None


# ------------------------------------------------------------------------------------------------ units
def _own_units(name: str):
    for suffix, entry in UNITS.items():
        if name.endswith(suffix):
            return entry
    return None


def is_fraction(name: str) -> bool:
    return name in FRACTIONS or name.endswith("_frac")


def unit_slip(name: str, value, lo=None, hi=None):
    """A number that is out of range in the parameter's unit and in range when read as centimetres (a length) or as
    percent (a fraction): ``(hint text, converted value)`` or None. Named in the refusal, never applied. Times are
    left alone: a short time is as likely a real request as a value in seconds."""
    if not T._is_number(value) or value <= 0:
        return None
    ok = lambda v: (lo is None or v >= lo) and (hi is None or v <= hi)    # noqa: E731
    if ok(value):
        return None
    if is_fraction(name) and 1 < value <= 100 and ok(value / 100):
        return f"{name} is a fraction; {value:g} % = {value / 100:g}", round(value / 100, 6)
    if name.endswith("_mm") and ok(value * 10):
        return f"{name} expects millimetres; {value:g} cm = {value * 10:g} mm", round(value * 10, 6)
    return None


def _number_with_unit(name: str, text: str):
    """``("own", number)`` for a number with the parameter's own unit, ``("other", number, unit, converted, unit name)``
    for a neighbouring unit or percent, else None."""
    m = _WITH_UNIT.match(text)
    if not m:
        return None
    number, unit = float(m.group(1)), m.group(2).strip().lower()
    if unit == "%":
        return ("other", number, "%", round(number / 100, 9), "a fraction") if is_fraction(name) else None
    entry = _own_units(name)
    if not entry:
        return None
    own, others, unit_name = entry
    if unit in own:
        return ("own", number)
    if unit in others:
        return ("other", number, unit, round(number * others[unit], 9), unit_name)
    return None


# ------------------------------------------------------------------------------------------------ canonical form
def _decode(text: str):
    """Strict JSON, then the Python spellings (literals, single-quoted containers). Returns (value, rule) or (text, None)."""
    s = text.strip()
    if not s or len(s) > T.MAX_STRING:
        return text, None
    try:
        return json.loads(s, parse_constant=T._no_constant), "C1"
    except (ValueError, RecursionError):
        pass
    if s in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[s], "C2"
    if s[0] in "[{(" and s[-1] in "]})":
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError, RecursionError, MemoryError, TypeError):
            return text, None
        if isinstance(v, tuple):
            v = list(v)
        if isinstance(v, (dict, list)) and not T.unsafe_values(v):
            return v, "C2"
    return text, None


def _branches(schema: dict) -> list:
    return list(schema["anyOf"]) if "anyOf" in schema else [schema]


def _accepts_min(schema: dict) -> bool:
    return any(b.get("type") == "string" and console.MIN in b.get("enum", ()) for b in _branches(schema))


def _is_free_string(schema: dict) -> bool:
    return "anyOf" not in schema and schema.get("type") == "string" and "enum" not in schema


class _Run:
    """One canonicalisation: the coercion log and the problems found on the way."""

    def __init__(self):
        self.log, self.problems = [], []

    def note(self, rule, parameter, old, new):
        self.log.append({"rule": rule, "parameter": parameter, "from": _show(old, 80), "to": _show(new, 80)})


def _scalar(run: _Run, name: str, path: str, value, schema: dict, required: bool):
    """Canonical value of one argument, or the marker ``_OMIT``."""
    types = {b.get("type") for b in _branches(schema)}
    number = next((b for b in _branches(schema) if b.get("type") in ("number", "integer")), None)
    if value is None:
        if not required:
            run.note("C3", path, None, "omitted")
            return _OMIT
        return value
    if isinstance(value, str) and not _is_free_string(schema):
        s = value.strip()
        low = s.strip("\"'").strip().lower()
        if _accepts_min(schema) and low in MIN_WORDS:
            if value != console.MIN:
                run.note("C4", path, value, console.MIN)
            return console.MIN
        if not required and low in OMIT_WORDS:
            run.note("C3", path, value, "omitted")
            return _OMIT
        enum = [e for b in _branches(schema) if b.get("type") == "string" for e in b.get("enum", ())]
        if enum:
            hits = [e for e in enum if e.lower() == low]
            if len(hits) == 1:
                if hits[0] != value:
                    run.note("C6", path, value, hits[0])
                return hits[0]
        if types & {"number", "integer", "boolean", "object", "array"}:
            parsed, rule = _decode(value)
            if rule and parsed is None and not required:
                run.note(rule, path, value, "omitted")
                return _OMIT
            if rule and _fits(parsed, types):
                run.note(rule, path, value, parsed)
                value = parsed
            elif number is not None:
                unit = _number_with_unit(name, s)
                if unit and unit[0] == "own":
                    run.note("C5", path, value, unit[1])
                    value = unit[1]
                elif unit:
                    _, n, written, converted, unit_name = unit
                    fixed = int(converted) if number.get("type") == "integer" and converted == int(converted) else converted
                    what = (f"{name} is a fraction; {n:g} % = {fixed:g}" if written == "%" else
                            f"{name} expects {unit_name}; {n:g} {written} = {fixed:g} {_own_units(name)[0][0]}")
                    run.problems.append(problem("invalid_value", path, f"{what} (a unit is never converted for you)", "task",
                                                requested=value, nearest_valid=fixed, fix=_fix_at(path, fixed)))
                    return _BAD
    if number is not None and T._is_number(value):
        if number["type"] == "integer":
            if float(value) == int(value):
                return int(value)
        else:
            return float(value)
    if isinstance(value, bool) and "boolean" not in types and not _is_free_string(schema):
        return value                                                    # a boolean is never read as a number: validation refuses it
    return value


def _fits(value, types) -> bool:
    if isinstance(value, bool):
        return "boolean" in types
    if isinstance(value, (int, float)):
        return bool(types & {"number", "integer"}) and T._is_number(value)
    if isinstance(value, dict):
        return "object" in types
    if isinstance(value, list):
        return "array" in types
    return False


_OMIT, _BAD = object(), object()


def _fix_at(path: str, value) -> dict | None:
    """A patch for a top-level parameter; a nested value has no one-key patch (the message carries the repair)."""
    return {path: value} if path and "." not in path and "[" not in path else None


def _value(run: _Run, name: str, path: str, value, schema: dict, required: bool, depth: int = 0):
    """Canonical value of an argument of any declared type (recursive over objects and arrays)."""
    value = _scalar(run, name, path, value, schema, required)
    if value is _OMIT or value is _BAD or depth > T.MAX_DEPTH:
        return value
    t = schema.get("type")
    if t == "array" and isinstance(value, dict) and (schema.get("items") or {}).get("type") == "object":
        run.note("C8", path, value, [value])
        value = [value]
    if t == "object" and isinstance(value, dict):
        props, extra, need = schema.get("properties", {}), schema.get("additionalProperties", True), schema.get("required", [])
        out = {}
        for k, v in value.items():
            sub = props.get(k, extra if isinstance(extra, dict) else None)
            c = _value(run, k, T._join(path, k), v, sub, k in need, depth + 1) if sub else v
            if c is _BAD:
                return _BAD
            if c is not _OMIT:
                out[k] = c
        return {k: out[k] for k in list(props) + sorted(set(out) - set(props)) if k in out}
    if t == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        out = []
        for i, v in enumerate(value[:T.MAX_NODES]):
            c = _value(run, name, T._join(path, i), v, schema["items"], True, depth + 1)
            if c is _BAD:
                return _BAD
            out.append(c)
        return out
    return value


def _split_xml_tail(run: _Run, raw: dict, props: dict) -> dict:
    """C13: a text value that runs into ``<parameter=name>`` (the closing tag of the parameter before it was never
    written): cut there and read the rest as the parameters it names. A parameter that is free text is left alone."""
    out = {}
    for key, value in raw.items():
        if not isinstance(value, str) or "<parameter=" not in value or (key in props and _is_free_string(props[key])):
            out.setdefault(key, value)
            continue
        parts = _XML_TAG.split(value)
        out.setdefault(key, parts[0].split("</parameter>")[0].strip())
        for name, tail in zip(parts[1::2], parts[2::2]):
            out.setdefault(name, tail.split("</parameter>")[0].strip())
        run.note("C13", key, value, {k: out[k] for k in [key] + parts[1::2]})
    return out


def _nest(run: _Run, args: dict, props: dict):
    """C15: names that are no parameter of the tool and belong to exactly one of its object parameters are moved into
    that object (a nested deliverable written flat). Nothing moves when a name fits two objects or is already inside."""
    objects = {k: v.get("properties", {}) for k, v in props.items() if v.get("type") == "object" and v.get("properties")}
    if not objects:
        return
    for key in [k for k in args if k not in props]:
        owners = [o for o, inner in objects.items() if key in inner]
        if len(owners) != 1:
            continue
        home = args.get(owners[0])
        if isinstance(home, str):
            home, _ = _decode(home)
        if home is None:
            home = {}
        if not isinstance(home, dict) or key in home:
            continue
        home[key] = args.pop(key)
        args[owners[0]] = home
        run.note("C15", key, key, f"{owners[0]}.{key}")


def _regions(run: _Run, args: dict, props: dict):
    """C8: the spellings of ``regions`` (ids of the task's marked regions) and ``circles`` (free circles)."""
    for key in ("regions", "circles"):
        if isinstance(args.get(key), str):
            parsed, rule = _decode(args[key])
            if rule and isinstance(parsed, (list, dict)):
                run.note(rule, key, args[key], parsed)
                args[key] = parsed
    if "regions" in args:
        r, before = args["regions"], args["regions"]
        if isinstance(r, str) and r.strip().strip("\"'").lower() in REGION_WORDS:
            r = [ALL_REGIONS]
        elif isinstance(r, dict):
            r = [r]
        if isinstance(r, list):
            circles = [x for x in r if isinstance(x, dict)]
            ids = [x for x in r if not isinstance(x, dict)]
            if circles:                                                 # circles written under ``regions``: these and no others
                if isinstance(args.get("circles"), dict):
                    args["circles"] = [args["circles"]]
                if args.get("circles") is None or isinstance(args["circles"], list):
                    args["circles"] = circles + list(args.get("circles") or [])
                    r = ids
            if any(isinstance(x, str) and x.strip().lower() in REGION_WORDS for x in r):
                r = [ALL_REGIONS]
        if r is None or (r in ([ALL_REGIONS], []) and "regions" not in props):
            del args["regions"]                                         # "all of none": this task marks no regions
            run.note("C8" if r is not None else "C3", "regions", before, "omitted")
        else:
            if r != before:
                run.note("C8", "regions", before, r)
            args["regions"] = r


def canonicalise(spec: T.ToolSpec, raw: dict) -> tuple[dict, list, list]:
    """``(canonical arguments, coercions, problems)`` of a call to ``spec``. See the module docstring for the rules.

    The canonical arguments carry the schema's own names in the schema's order, numbers as numbers (a declared
    number as a float, a declared integer as an int), and nothing that was not written. What a tool answers is a
    function of the canonical call and the state alone: the coercions are reported in ``Observation.data["coercions"]``
    and never change the text, so a transcript replayed from its canonical calls reads the same. ``problems`` holds what the
    canonicaliser itself refuses (another unit, percent, a clash between two spellings); anything else that is wrong
    is left in place for the schema validation to name.
    """
    run = _Run()
    props = spec.parameters.get("properties", {})
    required = spec.parameters.get("required", [])
    args = dict(raw)
    if len(args) == 1:                                                   # C7: a wrapper around the arguments
        (k, v), = args.items()
        if k in WRAPPERS and k not in props:
            inner, _ = _decode(v) if isinstance(v, str) else (v, None)
            if isinstance(inner, dict):
                run.note("C7", k, v if isinstance(v, str) else f"{{{k}: ...}}", "flat arguments")
                args = dict(inner)
    args = _split_xml_tail(run, args, props)
    if spec.protocol and "protocol" in args and "protocol" not in props:    # C7: the nested protocol object
        inner = args.pop("protocol")
        if isinstance(inner, str):
            inner, _ = _decode(inner)
        if isinstance(inner, dict):
            clash = sorted(k for k in inner if k in args and args[k] != inner[k])
            if clash:
                merged = {**inner, **args}
                run.problems.append(problem("invalid_value", clash[0], f"{', '.join(clash)} written twice with different values: inside "
                                            "protocol and next to it", "shape", fix={"protocol": None, **{k: merged[k] for k in clash}}))
            args = {**inner, **args}
            run.note("C7", "protocol", "{protocol: {...}}", "flat arguments")
        elif inner is not None:
            run.problems.append(problem("invalid_value", "protocol", "the protocol parameters are written as top-level arguments "
                                        f"(sequence, tr_ms, ...); protocol={_show(inner, 40)} is not an object", "shape", fix={"protocol": None}))
    if spec.protocol and "circles" not in props:                         # C14: a dry run reads nothing out
        for key in ("regions", "circles"):
            if key in args:
                run.note("C14", key, args.pop(key), "omitted")
    _nest(run, args, props)                                              # C15
    for alias, name in spec.aliases.items():                             # C10
        if alias in args and alias not in props and name in props:
            if name in args and args[name] != args[alias]:
                run.problems.append(problem("invalid_value", name, f"{alias} and {name} are the same parameter, written with different values",
                                            "shape", fix={alias: None}))
            args.setdefault(name, args[alias])
            del args[alias]
            run.note("C10", alias, alias, name)
    enums = [k for k, v in props.items() if "enum" in v]
    if len(enums) == 1 and len(props) <= 2 and not any(k in props for k in args):   # C9: the one enum parameter under another key
        only, schema = enums[0], props[enums[0]]
        if only not in args and args:
            values = [e for e in schema["enum"] if isinstance(e, str)]
            words = [w for kv in args.items() for w in kv if isinstance(w, str)]
            hits = {e for e in values for w in words if w.strip().strip("\"'").lower() == e.lower()}
            if len(hits) == 1:
                run.note("C9", only, args, {only: next(iter(hits))})
                args = {only: next(iter(hits))}
            elif not hits and all(v in ("", None) for v in args.values()) and only not in required:
                run.note("C9", only, args, {})
                args = {}
    if "circles" in props:
        _regions(run, args, props)
    for name in spec.lines:                                              # C12
        v = args.get(name)
        if isinstance(v, str) and v.strip().startswith("["):
            parsed, rule = _decode(v)
            if rule and isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
                v = parsed
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            run.note("C12", name, args[name], "lines")
            args[name] = "\n".join(v)
    out = {}
    for key, v in args.items():
        if key not in props:
            out[key] = v
            continue
        c = _value(run, key, key, v, props[key], key in required)
        if c is _OMIT:
            continue
        out[key] = v if c is _BAD else c
    if spec.protocol:
        _sequence_rules(run, out, props)
    ordered = {k: out[k] for k in list(props) + sorted((k for k in out if k not in props), key=str) if k in out}
    return ordered, run.log, run.problems


def _sequence_rules(run: _Run, args: dict, props: dict):
    """C11 and the two vocabulary refusals of a protocol: a parameter of another sequence, an unknown name."""
    seq = args.get("sequence")
    if not isinstance(seq, str) or seq not in console.SEQUENCES:
        return
    own = console.SEQUENCES[seq]
    for key in [k for k in args if k in props and k != "sequence" and k not in own and any(k in s for s in console.SEQUENCES.values())]:
        fixed = console.FIXED.get(seq, {}).get(key)
        if fixed is not None and T._is_number(args[key]) and float(args[key]) == fixed:
            run.note("C11", key, args[key], "omitted")
            del args[key]
            continue
        used = sorted(q for q, s in console.SEQUENCES.items() if key in s)
        why = f"{key} is fixed at {fixed:g} for {seq}" if fixed is not None else f"{seq} has no {key} (it belongs to {', '.join(used)})"
        run.problems.append(problem("not_applicable", key, why,
                                    requested=args[key] if isinstance(args[key], (str, int, float, bool)) else None,
                                    allowed={"values": [k for k in console.parameter_names() if k in own]}, fix={key: None}))


def schema_problems(spec: T.ToolSpec, args: dict) -> list:
    """The problems of canonical arguments against the tool's schema, in the error vocabulary."""
    props = spec.parameters.get("properties", {})
    seq = args.get("sequence") if spec.protocol else None
    seq = seq if isinstance(seq, str) and seq in console.SEQUENCES else None
    own = [k for k in props if seq is None or k in console.SEQUENCES[seq] or k in ("sequence", "regions", "circles", "verbose")]
    out = []
    found = T.problems(args, spec.parameters)
    if spec.protocol:
        found += _range_problems(args, "")
    for key, schema in props.items():                                    # a protocol delivered as one argument
        if schema.get("type") == "object" and {"sequence", "tr_ms"} <= set(schema.get("properties", {})) and isinstance(args.get(key), dict):
            found += _range_problems(args[key], key)
    named = set()
    for p in found:
        if (p["parameter"], p["kind"]) in named:
            continue
        named.add((p["parameter"], p["kind"]))
        name, kind = p["parameter"], p["kind"]
        top = name.split(".")[0].split("[")[0]
        if kind == "unknown" and spec.protocol and name == top and isinstance(top, str) and top in console.NOT_SIMULATED:
            out.append(problem("unknown_parameter", name, console.not_simulated_message(top), code="NOT_SIMULATED",
                               allowed={"values": own}, fix={top: None}))
        elif kind == "unknown" and spec.protocol and name == top and isinstance(top, str) and top in console.VENDOR_TERMS:
            text, target, value = console.vendor_term(top, args.get(top), args)
            out.append(problem("unknown_parameter", name, text, code="VENDOR_TERM", did_you_mean=target, allowed={"values": own},
                               fix={top: None, target: value} if value is not None and target not in args else {top: None}))
        elif kind == "unknown":
            guess = did_you_mean(top, own) or did_you_mean(top, list(spec.aliases))
            guess = spec.aliases.get(guess, guess)
            nested = name != top
            fix = None if nested else ({top: None, guess: args[top]} if guess and guess not in args else {top: None})
            listed = "" if nested or guess else (f'; its parameters: scanner_info(topic="{seq}")' if seq else f"; parameters: {', '.join(own)}"
                                                 if len(own) <= 12 else "")
            out.append(problem("unknown_parameter", name, f"{spec.name} has no parameter {_show(top, 40)}"
                               + (f"; did you mean {guess}?" if guess else "") + listed,
                               did_you_mean=guess, allowed={"values": own} if not nested else p.get("allowed"), fix=fix))
        elif kind == "required":
            what = (props.get(top) or {}).get("description", "").split(";")[0] if "." not in name else ""
            out.append(problem("invalid_value", name, "required" + (f" ({what})" if what else ""),
                               "task", allowed=_allowed(props.get(top, {}))))
        elif kind == "type":
            out.append(problem("invalid_value", name, p["message"], "shape", allowed=p.get("allowed")))
        elif kind == "enum":
            values = (p.get("allowed") or {}).get("values") or []
            guess = did_you_mean(p.get("requested"), [v for v in values if isinstance(v, str)])
            out.append(problem("invalid_value", name, p["message"] + (f"; did you mean {json.dumps(guess)}?" if guess else ""), "vocabulary",
                               requested=p.get("requested"), allowed=p.get("allowed"), did_you_mean=guess,
                               nearest_valid=guess, fix=_fix_at(name, guess) if guess else None))
        else:                                                           # integer, range, items
            near = p.get("nearest_valid")
            slip = unit_slip(name.split(".")[-1], p.get("requested"), *[(p.get("allowed") or {}).get(k) for k in ("min", "max")]) if kind == "range" else None
            value = slip[1] if slip else near[0] if isinstance(near, list) and near else near
            out.append(problem("invalid_value", name, p["message"] + (f"; {slip[0]}" if slip else ""), "task", requested=p.get("requested"),
                               allowed=p.get("allowed"), nearest_valid=near, fix=_fix_at(name, value) if value is not None else None))
    return out


def _range_problems(protocol: dict, path: str) -> list:
    """The fixed ranges of the console's parameters, as problems of the form ``tools.problems`` gives. The schema a
    model reads states these ranges in words (one line per parameter); they are checked here, for ``plan``, ``scan``
    and a delivered protocol alike. A range that depends on the rest of the protocol is the console's to check."""
    out = []
    for name, value in protocol.items():
        specs = [s[name] for s in console.SEQUENCES.values() if name in s]
        if not specs or specs[0][2] not in "fi" or name in console.PROTOCOL_DEPENDENT or not T._is_number(value):
            continue
        lo, hi = min(s[0] for s in specs), max(s[1] for s in specs)
        if value < lo or value > hi:
            near = lo if value < lo else hi
            out.append(dict(parameter=T._join(path, name), kind="range", requested=value, allowed={"min": lo, "max": hi}, nearest_valid=near,
                            message=f"{value:g} is {'below the minimum' if value < lo else 'above the maximum'} {near:g}"))
    return out


def _allowed(schema: dict):
    for b in _branches(schema):
        if "enum" in b and b.get("type") == "string" and len(_branches(schema)) == 1:
            return {"values": list(b["enum"])}
        if "minimum" in b or "maximum" in b:
            return {"min": b.get("minimum"), "max": b.get("maximum")}
    return None


# ------------------------------------------------------------------------------------------------ console refusals
_LIMIT_CODES = ("TE_BELOW_MINIMUM", "TR_BELOW_MINIMUM", "TI_NOT_REALISABLE", "ECHO_SPACING_BELOW_MINIMUM", "RESOLUTION_LIMIT",
                "SPOILER_BELOW_MINIMUM", "RF_POWER_LIMIT", "GRADIENT_LIMIT", "SLICE_TOO_THIN")
CONSOLE_CODES = ("INVALID_REQUEST", "UNKNOWN_SEQUENCE", "UNKNOWN_PARAMETER", "NOT_SIMULATED", "MISSING_PARAMETER", "OUT_OF_RANGE") + _LIMIT_CODES


def console_problems(violations: list, protocol: dict | None = None, exam: dict | None = None) -> list:
    """The console's typed violations (``console.validate`` / ``console.plan``) in the error vocabulary, each with the
    nearest valid value and a patch where the console's numbers give one. ``protocol``: the request, for the patches
    that need another of its values (the field of view for a resolution limit, the repetition time for RF power)."""
    protocol, exam = protocol if isinstance(protocol, dict) else {}, exam or {}
    seq = protocol.get("sequence")
    spec = console.SEQUENCES.get(seq, {}) if isinstance(seq, str) else {}
    out = []
    for v in violations:
        code, name, msg = v.get("code"), v.get("parameter"), v.get("message", "")
        value = protocol.get(name) if name else None
        if code == "INVALID_REQUEST":
            out.append(problem("malformed_call", None, msg, code=code))
        elif code == "UNKNOWN_SEQUENCE":
            guess = did_you_mean(protocol.get("sequence"), v.get("allowed") or [])
            out.append(problem("invalid_value", "sequence", msg, "vocabulary" if "sequence" in protocol else "task", code=code,
                               allowed={"values": v.get("allowed")}, did_you_mean=guess, fix={"sequence": guess} if guess else None))
        elif code == "UNKNOWN_PARAMETER":
            if v.get("applies_to"):
                out.append(problem("not_applicable", name, msg, code=code, allowed={"values": v.get("allowed")}, fix={name: None} if name else None))
            else:
                guess = did_you_mean(name, v.get("allowed") or [])
                out.append(problem("unknown_parameter", name, msg + (f"; did you mean {guess}?" if guess else ""), code=code, did_you_mean=guess,
                                   allowed={"values": v.get("allowed")},
                                   fix=({name: None, guess: value} if guess and guess not in protocol else {name: None}) if name else None))
        elif code == "NOT_SIMULATED":                                   # a real console word this instrument lacks
            out.append(problem("unknown_parameter", name, msg, code=code, allowed={"values": v.get("allowed")}, fix={name: None} if name else None))
        elif code == "MISSING_PARAMETER":
            lo, hi = (spec.get(name) or (None, None))[:2]
            out.append(problem("invalid_value", name, msg, "task", code=code,
                               allowed={"min": lo, "max": hi} if T._is_number(lo) else None))
        elif code == "OUT_OF_RANGE":
            near = v.get("nearest_valid")
            if "low" in v:
                slip = unit_slip(name, v.get("requested"), v["low"], v["high"])
                if T._is_number(v.get("requested")):                    # the fact in a few words (the console adds the parameter's meaning)
                    msg = f"{v['requested']:g} is outside {v['low']:g} to {v['high']:g}" + (f" for {seq}" if isinstance(seq, str) else "")
                out.append(problem("invalid_value", name, msg + (f"; {slip[0]}" if slip else ""), "task", code=code, requested=v.get("requested"),
                                   allowed={"min": v["low"], "max": v["high"]}, nearest_valid=near, fix={name: slip[1] if slip else near}))
            elif isinstance(near, list) and near:
                out.append(problem("invalid_value", name, f"{msg} (nearest: {', '.join(str(n) for n in near)})", "task", code=code,
                                   requested=v.get("requested"), nearest_valid=near, fix={name: near[0]}))
            else:
                allowed = v.get("allowed")
                guess = did_you_mean(value, allowed or []) if isinstance(value, str) else None
                out.append(problem("invalid_value", name, msg, "vocabulary" if allowed and isinstance(value, (str, int)) and not isinstance(value, bool) else "shape",
                                   code=code, allowed={"values": allowed} if allowed else None, did_you_mean=guess,
                                   fix={name: guess} if guess else None))
        elif code in _LIMIT_CODES:
            out.append(_limit_problem(v, protocol, exam, spec))
        else:
            out.append(problem("console_limit", name, msg, code=code))
    return out


def _limit_problem(v: dict, protocol: dict, exam: dict, spec: dict) -> dict:
    code, name, msg = v["code"], v.get("parameter"), v.get("message", "")
    lo, hi = (spec.get(name) or (None, None))[:2]
    in_range = lambda x: T._is_number(x) and (not T._is_number(lo) or lo <= x <= hi)      # noqa: E731
    near = fix = None
    if code in ("TE_BELOW_MINIMUM", "TR_BELOW_MINIMUM", "TI_NOT_REALISABLE", "ECHO_SPACING_BELOW_MINIMUM"):
        near = v.get("minimum")
    elif code == "SPOILER_BELOW_MINIMUM" and T._is_number(v.get("minimum")):
        near = round(v["minimum"] + 0.1, 1)
    elif code == "SLICE_TOO_THIN" and T._is_number(v.get("minimum_mm")):         # the thinnest slice of this pulse, rounded up
        near = math.ceil(v["minimum_mm"] * 100 + 1e-6) / 100
    elif code == "RESOLUTION_LIMIT" and T._is_number(v.get("minimum_mm")):
        fov = protocol.get("fov_mm", exam.get("fov_mm", 240.0))
        if T._is_number(fov):
            near = int(2 * math.floor(fov / v["minimum_mm"] / 2 + 1e-9))
    elif code == "RF_POWER_LIMIT" and T._is_number(v.get("percent_of_limit")) and T._is_number(protocol.get("tr_ms")):
        near = math.ceil(protocol["tr_ms"] * v["percent_of_limit"] / 100.0 * 1.01 * 100) / 100
    elif code == "GRADIENT_LIMIT" and T._is_number(v.get("percent_of_limit")):
        bw = protocol.get("bandwidth_hz_per_px", (spec.get("bandwidth_hz_per_px") or (0, 0, "f", 300.0))[3])
        if T._is_number(bw):
            near = math.floor(bw * 100.0 / v["percent_of_limit"] * 0.99)
    if near is not None and in_range(near):
        fix = {name: near}
    else:
        near = None
    allowed = {"min": near if code.endswith(("MINIMUM", "REALISABLE", "THIN")) else lo, "max": hi} if near is not None and T._is_number(hi) else None
    return problem("console_limit", name, msg, code=code, requested=v.get("requested", protocol.get(name) if T._is_number(protocol.get(name)) else None),
                   allowed=allowed, nearest_valid=near, fix=fix)
