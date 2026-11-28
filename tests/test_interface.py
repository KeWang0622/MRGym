"""The agent-facing interface: schemas, the canonical form of a call, the error contract, the limits, what a tool reveals.

    schema    flat plan / scan arguments, one type per parameter, a description on every parameter, valid examples
    te_ms     "min" is a value; a number below the console minimum is raised to it and reported; nothing else is adjusted
    errors    every refusal is typed, names the parameter and a fix, charges no scan time and never ends the episode
    limits    the task text, the status line, the status tool and info state the same numbers, and they are the ones that apply
    reveal    view shows an image and measures nothing in it; profile gives the numbers along a line
    total     step never raises, returns copies, and answers its own faults as internal_error
"""
import inspect
import json
import math
import os
import re

import numpy as np
import pytest

import mrgym
from mrgym import interface as I
from mrgym import policies
from mrgym import tools as T
from mrgym.agents import runner
from mrgym.envs import scanner_env
from mrgym.scanner import console, session

NOWRAP, T1, EXPLORE, NULL = "Scanner/NoWrap-v1", "Scanner/T1Transmit-v1", "Scanner/Explore-v0", "Scanner/NullTissue-v0"
EVERY = [(i, lv) for i in mrgym.list_envs() for lv in mrgym.spec(i).levels]         # every registered id, every level it registers
LOCALISER = {"sequence": "spgr", "tr_ms": 12, "flip_deg": 10, "fov_mm": 288, "matrix": 96}       # the minimum TR of the default pulse is about 10.5 ms
CHEAP = {"sequence": "spgr", "tr_ms": 12, "flip_deg": 10, "matrix": 48}
CENTRED = {"centre_y_mm": 0, "centre_x_mm": 0, "rotation_deg": 0}                       # the rest of a Scanner/NoWrap-v1 geometry
SE = {"sequence": "se", "tr_ms": 500, "bandwidth_hz_per_px": 300}


def make(env_id=NOWRAP, level=1, seed=0, **kw):
    env = mrgym.make(env_id, level=level, seed=seed, **kw)
    env.reset()
    return env


def call(env, name, **arguments):
    return env.step({"name": name, "arguments": arguments})


def walk(value, path=""):
    """Every key and every string of a JSON value."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from walk(v, f"{path}.{k}")
    elif isinstance(value, list):
        for v in value:
            yield from walk(v, path)
    elif isinstance(value, str):
        yield value


# ------------------------------------------------------------------------------------------------ schema
@pytest.mark.parametrize("env_id, level", EVERY)
def test_schema_is_flat_typed_described_and_its_examples_run(env_id, level):
    env = make(env_id, level)
    specs = {t.name: t for t in env._tool_specs()}
    assert not [p for t in specs.values() for p in T.check_tool(t)]              # portable subset, descriptions, valid examples
    assert list(specs)[:8] == ["scanner_info", "status", "plan", "scan", "read_regions", "view", "profile", "calc"]
    for name in ("plan", "scan"):
        props = specs[name].parameters["properties"]
        assert specs[name].parameters["required"] == ["sequence", "tr_ms"] and specs[name].protocol
        assert [k for k, s in props.items() if "anyOf" in s] == ["te_ms", "echo_spacing_ms"]      # the only unions: number or "min"
        assert all(s["anyOf"] == [{"type": "number"}, {"type": "string", "enum": ["min"]}] for s in props.values() if "anyOf" in s)
        assert not [k for k, s in props.items() if s.get("type") == "object"]
        assert not [k for k in console.PROTOCOL_DEPENDENT for b in props[k].get("anyOf", [props[k]]) if "minimum" in b]   # none a model could copy
    assert set(specs["scan"].parameters["properties"]) - set(specs["plan"].parameters["properties"]) <= {"regions", "circles"}
    assert ("regions" in specs["scan"].parameters["properties"]) == bool(env._regions)            # offered only when the task marks regions
    text = json.dumps(env.tools("openai"))
    assert not [w for w in ("console minimum", "exam default", "sequence default") if w in text]
    assert f"default {env._session.exam['fov_mm']:g}" in specs["plan"].parameters["properties"]["fov_mm"]["description"]
    # every wire format carries the same schema and one example call
    for fmt, key in (("openai", None), ("mcp", "inputSchema"), ("anthropic", "input_schema")):
        for t, spec in zip(env.tools(fmt), specs.values()):
            t = t["function"] if fmt == "openai" else t
            assert t[key or "parameters"] == spec.parameters and ("Example: " in t["description"]) == bool(spec.examples)
    # the examples are calls the environment executes
    assert not call(env, "scan", **CHEAP).observation.is_error
    for spec in specs.values():
        if spec.terminal:
            continue
        for example in spec.examples:
            if spec.cost == "scan_time":                                # a fresh budget: a console that adjusts itself before a scan
                env = make(env_id, level)                               # (Scanner/ProtocolToSpec-v0) has no room for two scans
            step = env.step({"name": spec.name, "arguments": example})
            assert not step.observation.is_error and "coercions" not in step.observation.data, (spec.name, step.observation.text)
    for seq, example in scanner_env.SEQUENCE_EXAMPLES.items():
        assert not env.step({"name": "plan", "arguments": example}).observation.is_error, seq
        assert json.dumps(example, separators=(", ", ": ")) in call(env, "scanner_info", topic=seq).observation.text
    assert set(scanner_env.SEQUENCE_EXAMPLES) == set(console.SEQUENCES)
    target = env.episode_config().get("task", {}).get("target_scan")    # a target scan printed on a task card is a call plan takes
    if target and all(k in target for k in console.SEQUENCES[target["sequence"]] if console.SEQUENCES[target["sequence"]][k][3] is None):
        assert not env.step({"name": "plan", "arguments": target}).observation.is_error


def test_schema_flat_nested_and_text_spellings_are_one_call():
    env = make()
    spellings = [dict(LOCALISER), {"protocol": LOCALISER}, {"protocol": json.dumps(LOCALISER)}, {k: str(v) for k, v in LOCALISER.items()},
                 {"arguments": LOCALISER}, {"protocol": str(LOCALISER)}]
    canon = {json.dumps(env.admit({"name": "plan", "arguments": a})[1]) for a in spellings}
    texts = {env.step({"name": "plan", "arguments": a}).observation.text.split("\n[")[0] for a in spellings}
    assert len(canon) == 1 and len(texts) == 1 and next(iter(texts)).startswith("plan ok")
    assert json.loads(next(iter(canon))) == {"sequence": "spgr", "tr_ms": 12.0, "flip_deg": 10.0, "fov_mm": 288.0, "matrix": 96}


# ------------------------------------------------------------------------------------------------ te_ms and "min"
def test_schema_te_below_the_minimum_is_raised_and_reported_and_nothing_else_is():
    env = make()
    p = call(env, "plan", sequence="se", tr_ms=500, te_ms=0.5).observation
    assert not p.is_error and p.data["adjusted"] == [dict(parameter="te_ms", requested=0.5, realised=p.data["timing"]["te_min_ms"],
                                                         reason="console minimum for this matrix, field of view and bandwidth")]
    assert p.data["realised"]["te_ms"] == p.data["timing"]["te_min_ms"] and p.text.splitlines()[1].startswith("note: te_ms 0.5 -> ")
    for te in ("min", "minimum", "console minimum", None, 0, 0.0):
        same = call(env, "plan", sequence="se", tr_ms=500, te_ms=te).observation
        assert not same.is_error and same.data["realised"] == p.data["realised"], te
    assert "adjusted" not in call(env, "plan", sequence="se", tr_ms=500, te_ms="min").observation.data
    assert "adjusted" not in call(env, "plan", sequence="se", tr_ms=500, te_ms=20).observation.data        # a realisable value is kept
    assert call(env, "plan", sequence="se", tr_ms=500, te_ms=20).observation.data["realised"]["te_ms"] == 20.0
    # a scan with a short echo time runs at the minimum, costs what the plan said, and the scanner's log replays
    s = call(env, "scan", sequence="se", tr_ms=60, te_ms=0.5, matrix=32, phase_fov=0.25).observation
    assert not s.is_error and s.data["adjusted"][0]["realised"] == s.data["realised"]["te_ms"] and "note: te_ms 0.5 -> " in s.text
    assert session.replay(env.episode_config(), env.scanner_log(), head=env._session.receipt())["ok"]
    # nothing else is adjusted: a short repetition time, inversion time or echo spacing is refused with the nearest valid value
    for args, name in ((dict(sequence="se", tr_ms=50, te_ms=80), "tr_ms"), (dict(sequence="ir", tr_ms=3000, ti_ms=5.5, fat_sat=True), "ti_ms"),
                       (dict(sequence="epi", tr_ms=2000, echo_spacing_ms=0.21), "echo_spacing_ms"), (dict(sequence="se", tr_ms=500, te_ms=-1), "te_ms")):
        e = call(env, "plan", **args).observation
        assert e.is_error and e.data["error"]["parameter"] == name and "adjusted" not in e.data, args
    assert not call(env, "plan", sequence="epi", tr_ms=2000, echo_spacing_ms="min").observation.is_error
    assert console.plan(dict(sequence="se", tr_ms=500, te_ms="min"))["realised"] == console.plan(dict(sequence="se", tr_ms=500))["realised"]
    assert not console.plan(dict(sequence="se", tr_ms="min"))["accept"]                                 # only where the console default is its minimum


# ------------------------------------------------------------------------------------------------ canonical form
RULES = [   # (rule, tool, as written, canonical)
    ("C1", "plan", dict(sequence="spgr", tr_ms="10", flip_deg="1e1", matrix=64.0, fat_sat="true"), dict(sequence="spgr", tr_ms=10.0, flip_deg=10.0, matrix=64, fat_sat=True)),
    ("C2", "plan", dict(sequence="spgr", tr_ms=10, flip_deg=10, fat_sat="True"), dict(sequence="spgr", tr_ms=10.0, flip_deg=10.0, fat_sat=True)),
    ("C2", "read_regions", dict(scan_index=0, circles="[{'y_mm': 0, 'x_mm': 0, 'radius_mm': 5}]"), dict(scan_index=0, circles=[dict(y_mm=0.0, x_mm=0.0, radius_mm=5.0)])),
    ("C3", "plan", dict(sequence="se", tr_ms=500, fov_mm=None, matrix="", pe_dir="default", averages="auto"), dict(sequence="se", tr_ms=500.0)),
    ("C4", "plan", dict(sequence="se", tr_ms=500, te_ms="Minimum"), dict(sequence="se", tr_ms=500.0, te_ms="min")),
    ("C4", "plan", dict(sequence="epi", tr_ms=2000, echo_spacing_ms="shortest"), dict(sequence="epi", tr_ms=2000.0, echo_spacing_ms="min")),
    ("C5", "plan", dict(sequence="se", tr_ms="500 ms", fov_mm="240mm", bandwidth_hz_per_px="300 Hz/px"), dict(sequence="se", tr_ms=500.0, fov_mm=240.0, bandwidth_hz_per_px=300.0)),
    ("C5", "plan", dict(sequence="spgr", tr_ms=10, flip_deg="10 deg"), dict(sequence="spgr", tr_ms=10.0, flip_deg=10.0)),
    ("C6", "plan", dict(sequence=" SE ", tr_ms=500, pe_dir='"X"'), dict(sequence="se", tr_ms=500.0, pe_dir="x")),
    ("C7", "plan", dict(protocol=dict(sequence="se", tr_ms=500)), dict(sequence="se", tr_ms=500.0)),
    ("C7", "plan", dict(parameters=dict(sequence="se", tr_ms=500)), dict(sequence="se", tr_ms=500.0)),
    ("C8", "scan", dict(sequence="se", tr_ms=500, regions="task"), dict(sequence="se", tr_ms=500.0)),
    ("C8", "scan", dict(sequence="se", tr_ms=500, regions=[dict(y_mm=0, x_mm=0, radius_mm=5)]), dict(sequence="se", tr_ms=500.0, circles=[dict(y_mm=0.0, x_mm=0.0, radius_mm=5.0)])),
    ("C8", "read_regions", dict(scan_index=0, circles=dict(y_mm=0, x_mm=0, radius_mm=5)), dict(scan_index=0, circles=[dict(y_mm=0.0, x_mm=0.0, radius_mm=5.0)])),
    ("C9", "scanner_info", dict(task="task"), dict(topic="task")),
    ("C9", "scanner_info", dict(section="Physics"), dict(topic="physics")),
    ("C10", "plan", dict(sequence="spgr", tr_ms=10, flip_angle_deg=10, slice_thickness_mm=5, phase_fov_frac=0.5), dict(sequence="spgr", tr_ms=10.0, flip_deg=10.0, phase_fov=0.5, slice_mm=5.0)),
    ("C10", "view", dict(scan=0, image=0), dict(scan_index=0, image_index=0)),
    ("C11", "plan", dict(sequence="se", tr_ms=500, flip_deg=90), dict(sequence="se", tr_ms=500.0)),
    ("C12", "calc", dict(expressions=["a = 2", "a * 3"]), dict(expressions="a = 2\na * 3")),
    ("C12", "calc", dict(expressions='["a = 2", "a * 3"]'), dict(expressions="a = 2\na * 3")),
    ("C13", "scan", dict(protocol='{"sequence": "se", "tr_ms": 500}\n<parameter=regions>\n["task"]'), dict(sequence="se", tr_ms=500.0)),
    # the parameters of the slice-selective engine follow the same rules
    ("C6", "plan", dict(sequence="se", tr_ms=500, rf_pulse="SHARP", excitation=" Selective "), dict(sequence="se", tr_ms=500.0, rf_pulse="sharp", excitation="selective")),
    ("C5", "plan", dict(sequence="se", tr_ms=500, fat_sat=True, fatsat_offset_hz="-40 Hz", rotation_deg="30 deg"),
     dict(sequence="se", tr_ms=500.0, fat_sat=True, fatsat_offset_hz=-40.0, rotation_deg=30.0)),
    ("C10", "plan", dict(sequence="se", tr_ms=500, fov_rotation_deg=30, fat_sat_offset_hz=20), dict(sequence="se", tr_ms=500.0, fatsat_offset_hz=20.0, rotation_deg=30.0)),
    ("C3", "plan", dict(sequence="se", tr_ms=500, rf_pulse="default", rotation_deg=None, excitation=""), dict(sequence="se", tr_ms=500.0)),
    ("C1", "plan", dict(sequence="se", tr_ms=500, rotation_deg="-45", fatsat_offset_hz="1e1"), dict(sequence="se", tr_ms=500.0, fatsat_offset_hz=10.0, rotation_deg=-45.0)),
]


@pytest.mark.parametrize("rule, tool, written, canonical", RULES, ids=[f"{r[0]}-{i}" for i, r in enumerate(RULES)])
def test_canonical_form_of_every_safe_spelling(rule, tool, written, canonical):
    env = make()
    name, args, coercions, problems = env.admit({"name": tool, "arguments": written})
    assert not problems and json.dumps(args) == json.dumps(canonical), (args, problems)      # byte-equal, key order included
    assert rule in {c["rule"] for c in coercions} and all(set(c) == {"rule", "parameter", "from", "to"} for c in coercions)
    again = env.admit({"name": tool, "arguments": args})                                         # the canonical form is a fixed point
    assert json.dumps(again[1]) == json.dumps(canonical) and not again[2] and not again[3]


def test_canonical_form_is_reported_in_data_and_never_changes_the_text():
    a, b = make(), make()
    plain = call(a, "plan", sequence="se", tr_ms=500.0, flip_deg=90.0)                       # C11 only
    spelt = call(b, "plan", protocol=json.dumps(dict(sequence="SE", tr_ms="500 ms", flip_angle_deg=90)))
    assert plain.observation.text == spelt.observation.text and a._session.chain == b._session.chain
    assert {c["rule"] for c in spelt.observation.data["coercions"]} == {"C7", "C6", "C5", "C10", "C11"}
    assert a.transcript()[-1]["call"] == b.transcript()[-1]["call"] == {"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500.0}}
    assert b.interface_counts()["coerced"] == 1 and b.interface_counts()["shape"] == 0
    regions = make(T1, 1)                                                                     # a task that marks regions
    ids = [r["id"] for r in regions._regions]
    for spelling in ("task", "all", ["task"], '["task"]', ["all"]):
        assert regions.admit({"name": "scan", "arguments": dict(CHEAP, regions=spelling)})[1]["regions"] == ["all"]
    got = call(regions, "scan", **CHEAP, regions=ids[:2], circles=[dict(id="bg", y_mm=-100, x_mm=0, radius_mm=8)]).observation
    assert [r["id"] for r in got.data["roi_readouts"]] == ids[:2] + ["bg"]
    assert [r["id"] for r in call(regions, "read_regions", scan_index=0).observation.data["roi_readouts"]] == ids
    assert call(regions, "scan", **CHEAP, regions=[]).observation.data["roi_readouts"] == []


NEVER = [   # (what is never done for the agent, tool, arguments, parameter named in the refusal)
    ("a misspelt name is not guessed", "plan", dict(sequence="se", tr_ms=500, te_mm=10), "te_mm"),
    ("a misspelt name is not guessed", "plan", dict(sequence="se", tr_ms=500, averges=2), "averges"),
    ("another unit is not converted", "plan", dict(sequence="se", tr_ms=500, fov_mm="24 cm"), "fov_mm"),
    ("another unit is not converted", "plan", dict(sequence="se", tr_ms="0.5 s"), "tr_ms"),
    ("a length in centimetres is not converted", "plan", dict(sequence="se", tr_ms=500, fov_mm=24), "fov_mm"),
    ("percent is not read as a fraction", "plan", dict(sequence="se", tr_ms=500, phase_fov="75%"), "phase_fov"),
    ("percent is not read as a fraction", "plan", dict(sequence="se", tr_ms=500, phase_fov=75), "phase_fov"),
    ("a non-integer is not rounded", "plan", dict(sequence="se", tr_ms=500, matrix=64.5), "matrix"),
    ("an odd matrix is not rounded", "plan", dict(sequence="se", tr_ms=500, matrix=65), "matrix"),
    ("a value is not clipped to its range", "plan", dict(sequence="se", tr_ms=500, fov_mm=600), "fov_mm"),
    ("an unknown parameter is not dropped", "plan", dict(sequence="se", tr_ms=500, shim="auto"), "shim"),
    ("a parameter of another sequence is not dropped", "plan", dict(sequence="se", tr_ms=500, ti_ms=300), "ti_ms"),
    ("a fixed parameter with another value is not dropped", "plan", dict(sequence="se", tr_ms=500, flip_deg=60), "flip_deg"),
    ("a synonym of an enum value is not mapped", "plan", dict(sequence="se", tr_ms=500, pe_dir="ROW"), "pe_dir"),
    ("a synonym of an enum value is not mapped", "plan", dict(sequence="se", tr_ms=500, pe_dir="A>>P"), "pe_dir"),
    ("a synonym of an enum value is not mapped", "submit", dict(pe_dir="i", fov_mm=240, phase_fov=1, matrix=120, **CENTRED), "pe_dir"),
    ("a misspelt enum value is not corrected", "plan", dict(sequence="spgre", tr_ms=10, flip_deg=10), "sequence"),
    ("a boolean is not read as a number", "plan", dict(sequence="se", tr_ms=500, averages=True), "averages"),
    ("a synonym of an enum value is not mapped", "plan", dict(sequence="se", tr_ms=500, rf_pulse="sinc"), "rf_pulse"),
    ("a synonym of an enum value is not mapped", "plan", dict(sequence="se", tr_ms=500, excitation="3d"), "excitation"),
    ("another unit is not converted", "plan", dict(sequence="se", tr_ms=500, rotation_deg="0.5 rad"), "rotation_deg"),
    ("another unit is not converted", "plan", dict(sequence="se", tr_ms=500, fatsat_offset_hz="0.1 kHz"), "fatsat_offset_hz"),
    ("a value is not clipped to its range", "plan", dict(sequence="se", tr_ms=500, rotation_deg=180), "rotation_deg"),
    ("a parameter of another sequence is not dropped", "plan", dict(sequence="afi", tr_ms=20, flip_deg=60, fatsat_offset_hz=50), "fatsat_offset_hz"),
    ("a misspelt name is not guessed", "plan", dict(sequence="se", tr_ms=500, rotation=30), "rotation"),
]


NEVER_AS_TOOL_CALL = {"another unit is not converted", "a length in centimetres is not converted", "percent is not read as a fraction",
                      "a non-integer is not rounded", "an odd matrix is not rounded", "a value is not clipped to its range"}


@pytest.mark.parametrize("what, tool, arguments, parameter", NEVER, ids=[f"{n[1]}-{n[3]}-{i}" for i, n in enumerate(NEVER)])
def test_canonical_form_never_guesses(what, tool, arguments, parameter):
    env = make()
    step = call(env, tool, **arguments)
    err = step.observation.data["error"]
    assert step.observation.is_error and not step.done and err["parameter"] == parameter, what
    assert err.get("fix") or err.get("did_you_mean") or err.get("allowed"), what
    # nothing ran. A value the instrument refuses (class task: another unit, percent, a non-integer, an odd matrix, out of
    # range) is a tool call; a name or a word the interface does not have (vocabulary), a wrong type (shape) is a rejected call
    as_tool_call = what in NEVER_AS_TOOL_CALL
    assert (err["class"] == "task") == as_tool_call and env._session.events == [], what
    assert (env.state()["tool_calls"], env.state()["rejected_calls"]) == ((1, 0) if as_tool_call else (0, 1)), what
    if err.get("fix"):                                                   # the fix repairs it: not refused for the same reason again
        fixed = call(env, tool, **I.apply_fix(env.admit({"name": tool, "arguments": arguments})[1], err["fix"])).observation
        again = fixed.data.get("error", {})
        assert not fixed.is_error or (again.get("type"), again.get("parameter")) != (err["type"], parameter), fixed.text


# ------------------------------------------------------------------------------------------------ errors
def _probes():
    """The calls a model actually gets wrong (the probe list of the interface review), each with the error type it must be
    answered with, or None when the interface now takes it. ``fresh``: on an episode without a stored scan."""
    p = dict(SE)
    small = dict(SE, matrix=32, phase_fov=0.25, tr_ms=60)
    circle = dict(y_mm=0, x_mm=0, radius_mm=10)
    nowrap = [
        ("unknown tool", {"name": "budget", "arguments": {}}, "unknown_tool"),
        ("arguments not JSON", {"id": "x", "type": "function", "function": {"name": "scan", "arguments": "{protocol: se}"}}, "malformed_call"),
        ("flat protocol", {"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500}}, None),
        ("protocol as prose", {"name": "plan", "arguments": {"protocol": "se tr 500"}}, "invalid_value"),
        ("protocol as JSON text", {"name": "plan", "arguments": {"protocol": json.dumps(p)}}, None),
        ("unknown protocol field", {"name": "plan", "arguments": {"protocol": dict(p, fov=200)}}, "unknown_parameter"),
        ("te_ms = min", {"name": "plan", "arguments": {"protocol": dict(p, te_ms="min")}}, None),
        ("te_ms copied from the task card", {"name": "plan", "arguments": {"protocol": dict(p, te_ms="console minimum")}}, None),
        ("te_ms = 0", {"name": "plan", "arguments": {"protocol": dict(p, te_ms=0)}}, None),
        ("te_ms = null", {"name": "plan", "arguments": {"protocol": dict(p, te_ms=None)}}, None),
        ("odd matrix", {"name": "plan", "arguments": {"protocol": dict(p, matrix=65, fov_mm=130)}}, "invalid_value"),
        ("field of view in cm", {"name": "plan", "arguments": {"protocol": dict(p, fov_mm=24)}}, "invalid_value"),
        ("missing tr_ms", {"name": "plan", "arguments": {"protocol": {"sequence": "se"}}}, "invalid_value"),
        ("missing sequence", {"name": "plan", "arguments": {"protocol": {"tr_ms": 500}}}, "invalid_value"),
        ("sequence in capitals", {"name": "plan", "arguments": {"protocol": {"sequence": "SE", "tr_ms": 500}}}, None),
        ("regions on plan", {"name": "plan", "arguments": {"protocol": p, "regions": "task"}}, None),       # a dry run reads nothing out: dropped
        ("regions = all, no marked regions", {"name": "scan", "arguments": {"protocol": small, "regions": "all"}}, None),
        ("regions as a JSON-text list of circles", {"name": "scan", "arguments": {"protocol": small, "regions": json.dumps([circle])}}, None),
        ("circle with other keys", {"name": "scan", "arguments": {"protocol": small, "regions": [{"center": [0, 0], "radius": 10}]}}, "unknown_parameter"),
        ("circle with x, y", {"name": "read_regions", "arguments": {"scan": 0, "regions": [{"x": 0, "y": 0, "radius_mm": 10}]}}, "unknown_parameter"),
        ("read_regions of no regions", {"name": "read_regions", "arguments": {"scan": 0, "regions": "task"}}, "invalid_value"),
        ("scan not stored", {"name": "view", "arguments": {"scan": 7}}, "not_found"),
        ("image out of range", {"name": "view", "arguments": {"scan": 0, "image": 3}}, "not_found"),
        ("scan index as text", {"name": "view", "arguments": {"scan": "0"}}, None),
        ("scan index as a word", {"name": "view", "arguments": {"scan": "latest"}}, "invalid_value"),
        ("scan over budget", {"name": "scan", "arguments": {"protocol": dict(p, matrix=128)}}, "budget"),
        ("TR below minimum", {"name": "plan", "arguments": {"protocol": {"sequence": "se", "tr_ms": 50, "te_ms": 80}}}, "console_limit"),
        ("spgr without flip angle", {"name": "plan", "arguments": {"protocol": {"sequence": "spgr", "tr_ms": 10}}}, "invalid_value"),
        ("spoiler too weak for the pixel", {"name": "plan", "arguments": {"protocol": {"sequence": "spgr", "tr_ms": 12, "flip_deg": 10, "matrix": 256, "spoiler_mt_ms_per_m": 10}}}, "console_limit"),
        ("unknown topic", {"name": "scanner_info", "arguments": {"topic": "budget"}}, "invalid_value"),
        # the deliverable of Scanner/NoWrap-v1: the geometry with its centre and rotation; every field has the console's range,
        # so a value outside it is refused as a value, and what only the console can see (an odd matrix) as a deliverable
        ("submit: odd matrix", {"name": "submit", "arguments": dict(pe_dir="y", fov_mm=130, phase_fov=0.5, matrix=65, **CENTRED)}, "invalid_deliverable"),
        ("submit: missing field", {"name": "submit", "arguments": dict(pe_dir="y", fov_mm=240, matrix=120, **CENTRED)}, "invalid_value"),
        ("submit: the deliverable of the retired id", {"name": "submit", "arguments": dict(pe_dir="y", fov_mm=240, phase_fov=1, matrix=120)}, "invalid_value"),
        ("submit: extra field", {"name": "submit", "arguments": dict(pe_dir="y", fov_mm=240, phase_fov=1, matrix=120, tr_ms=500, **CENTRED)}, "unknown_parameter"),
        ("submit: header vocabulary", {"name": "submit", "arguments": dict(pe_dir="A>>P", fov_mm=240, phase_fov=1, matrix=120, **CENTRED)}, "invalid_value"),
        ("submit: phase_fov above 1", {"name": "submit", "arguments": dict(pe_dir="y", fov_mm=240, phase_fov=1.07, matrix=120, **CENTRED)}, "invalid_value"),
    ]
    fresh = [("submit: field of view in cm", {"name": "submit", "arguments": dict(pe_dir="y", fov_mm=24, phase_fov=0.5, matrix=120, **CENTRED)}, "invalid_value"),
             ("submit: an odd matrix before any scan", {"name": "submit", "arguments": dict(pe_dir="y", fov_mm=260, phase_fov=0.5, matrix=131, **CENTRED)}, "invalid_deliverable")]
    t1 = [
        ("T1 submit: a list", {"name": "submit", "arguments": {"t1_ms": [800, 900]}}, "invalid_value"),
        ("T1 submit: a units field", {"name": "submit", "arguments": {"units": "s", "t1_ms": {"R1": 1}}}, "unknown_parameter"),
        ("T1 submit: regions missing", {"name": "submit", "arguments": {"t1_ms": {"R1": 800}}}, "invalid_value"),
        ("T1 submit: flat", {"name": "submit", "arguments": {"R1": 800}}, "invalid_value"),          # moved into t1_ms; the other regions are missing
        ("T1 scan with the marked regions", {"name": "scan", "arguments": {"protocol": {"sequence": "afi", "tr_ms": 20, "flip_deg": 60, "matrix": 64}}}, None),
        ("T1 view", {"name": "view", "arguments": {"scan": 0}}, None),
        ("T1 submit: values that look like seconds", {"name": "submit", "arguments": {"t1_ms": {"R1": 1.2, "R2": 0.9}}}, "invalid_value"),
    ]
    return [(NOWRAP, "scan", x) for x in nowrap] + [(NOWRAP, "fresh", x) for x in fresh] + [(T1, "fresh", x) for x in t1]


PROBES = _probes()


# the ``invalid_value`` probes that are rejected calls (class shape or vocabulary); every other one is of class task: a tool call
REJECTED_VALUE_PROBES = {"protocol as prose", "scan index as a word", "unknown topic", "submit: header vocabulary", "T1 submit: a list"}


def test_errors_every_probe_is_typed_charged_nothing_and_never_ends_the_episode():
    envs, seen = {}, set()
    for env_id, state, (label, tool_call, expected) in PROBES:
        key = (env_id, state)
        if key not in envs:
            envs[key] = make(env_id, 1, max_rejected_calls=1000)
            if state == "scan":
                assert not call(envs[key], "scan", **dict(SE, matrix=32, phase_fov=0.25, tr_ms=60)).observation.is_error
        env = envs[key]
        used = env.state()["scan_time_used_s"]
        calls, rejected = env.state()["tool_calls"], env.state()["rejected_calls"]
        step = env.step(tool_call)
        obs = step.observation
        assert not step.done and step.reward == 0.0 and set(step.info) <= {"step"}, label      # no score before the end (training hooks add "step")
        if expected is None:
            assert not obs.is_error, (label, obs.text)
            continue
        err = obs.data["error"]
        assert obs.is_error and err["type"] == expected and err["type"] in I.ERROR_TYPES and err["class"] in I.CLASSES, (label, obs.text)
        assert obs.data["charged"]["scan_time_s"] == 0 and env.state()["scan_time_used_s"] == used and obs.data["episode_continues"], label
        # what the refusal is charged as: a rejected call for class shape / vocabulary and for a refused deliverable, a
        # tool call for class task (``I.counts_as_tool_call``); the invalid_value probes of each side are listed below
        counted = expected not in I.REJECTED_TYPES and err["class"] == "task"
        assert counted == (I.ERRORS[expected][1] if I.ERRORS[expected][1] is not None else label not in REJECTED_VALUE_PROBES), (label, err["class"])
        assert (env.state()["tool_calls"] - calls, env.state()["rejected_calls"] - rejected) == (int(counted), int(not counted)), label
        assert obs.data["charged"] == dict(scan_time_s=0.0, tool_calls=int(counted), rejected_calls=int(not counted)), label
        assert obs.text.startswith(f"error {expected} · ") and "charged: " not in obs.text and len(obs.text) < 700, (label, obs.text)
        assert ("(nothing charged)" in obs.text.splitlines()[-1]) == (not counted), (label, obs.text)          # the status line says what a refusal cost
        assert {"type", "parameter", "message", "fix", "problems", "more"} <= set(err) and 1 <= len(err["problems"]) <= I.MAX_PROBLEMS, label
        assert ("fix: " in obs.text) == bool(err["fix"]), label
        seen.add(expected)
    assert seen == {"malformed_call", "unknown_tool", "unknown_parameter", "invalid_value", "console_limit", "budget", "not_found", "invalid_deliverable"}
    assert len(PROBES) == 45


_SHARED = {}
FIXABLE = [   # refusals whose fix is a patch: applied to the call it gives a call that is not refused for the same reason
    ("plan", dict(sequence="se", tr_ms=50, te_ms=80), "console_limit", "TR_BELOW_MINIMUM", NOWRAP),
    ("plan", dict(sequence="ir", tr_ms=3000, ti_ms=5.5, fat_sat=True), "console_limit", "TI_NOT_REALISABLE", NOWRAP),
    ("plan", dict(sequence="epi", tr_ms=2000, echo_spacing_ms=0.21), "console_limit", "ECHO_SPACING_BELOW_MINIMUM", NOWRAP),
    ("plan", dict(sequence="spgr", tr_ms=12, flip_deg=10, matrix=512), "console_limit", "RESOLUTION_LIMIT", NOWRAP),
    ("plan", dict(sequence="spgr", tr_ms=12, flip_deg=10, matrix=256, spoiler_mt_ms_per_m=10), "console_limit", "SPOILER_BELOW_MINIMUM", NOWRAP),
    ("plan", dict(sequence="bssfp", tr_ms=4.2, flip_deg=90, matrix=64, bandwidth_hz_per_px=1500), "console_limit", "RF_POWER_LIMIT", EXPLORE),
    ("plan", dict(sequence="bssfp", tr_ms=3.5, flip_deg=90, matrix=64, bandwidth_hz_per_px=1500), "console_limit", "TR_BELOW_MINIMUM", EXPLORE),   # and RF power
    ("plan", dict(sequence="se", tr_ms=500, slice_mm=1, rf_pulse="sharp"), "console_limit", "SLICE_TOO_THIN", NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, rotation_deg=120), "invalid_value", None, NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, fat_sat=True, fatsat_offset_hz="0.1 kHz"), "invalid_value", None, NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, rotation_deg="0.5 rad"), "invalid_value", None, NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, rf_pulse="sharpe"), "invalid_value", None, NOWRAP),
    ("plan", dict(sequence="bssfp", tr_ms=8, flip_deg=40, fatsat_offset_hz=40), "not_applicable", None, NOWRAP),
    ("plan", dict(sequence="spgr", tr_ms=12, flip_deg=10, matrix=256, bandwidth_hz_per_px=3000), "console_limit", "GRADIENT_LIMIT", NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, matrix=65), "invalid_value", "OUT_OF_RANGE", NOWRAP),
    ("plan", dict(sequence="epi", tr_ms=2000, matrix=256), "invalid_value", "OUT_OF_RANGE", NOWRAP),
    ("plan", dict(sequence="se", tr_ms=20), "invalid_value", "OUT_OF_RANGE", NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, fov_mm=24), "invalid_value", None, NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, fov_mm="24 cm"), "invalid_value", None, NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, phase_fov="75 %"), "invalid_value", None, NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, matrix=64.5), "invalid_value", None, NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, te_mm=10), "unknown_parameter", None, NOWRAP),
    ("plan", dict(sequence="se", tr_ms=500, ti_ms=300, phase_cycle_deg=0), "not_applicable", None, NOWRAP),
    ("plan", dict(sequence="spgre", tr_ms=12, flip_deg=10), "invalid_value", None, NOWRAP),
    ("plan", dict(protocol=dict(sequence="se", tr_ms=500), tr_ms=600), "invalid_value", None, NOWRAP),
    ("view", dict(scan_index=5), "not_found", None, NOWRAP),
    ("view", dict(scan_index=0, image_index=4), "not_found", None, NOWRAP),
    ("view", dict(scan_index=0, size=100), "invalid_value", None, NOWRAP),
    ("profile", dict(scan_index=0, axis="y", position_mm=900), "invalid_value", None, NOWRAP),
    ("profile", dict(scan_index=0, axis="x", start_mm=900), "invalid_value", None, NOWRAP),
    ("read_regions", dict(scan_index=0), "invalid_value", None, NOWRAP),
    ("scan", dict(CHEAP, regions=["R9"]), "unknown_parameter", None, NOWRAP),
    ("scan", dict(CHEAP, regions=["R1", "R99"]), "not_found", None, T1),
    ("submit", dict(pe_dir="y", fov_mm=130, phase_fov=0.5, matrix=65, **CENTRED), "invalid_deliverable", "OUT_OF_RANGE", NOWRAP),
    ("submit", dict(pe_dir="y", fov_mm=240, phase_fov=1.07, matrix=120, **CENTRED), "invalid_value", None, NOWRAP),
    ("submit", dict(pe_dir="y", fov_mm=24, phase_fov=1, matrix=120, **CENTRED), "invalid_value", None, NOWRAP),
    ("submit", dict(pe_dir="y", fov_mm=240, phase_fov=1, matrix=120, centre_y_mm="1.2 cm", centre_x_mm=0, rotation_deg=0), "invalid_value", None, NOWRAP),
]


@pytest.mark.parametrize("tool, arguments, kind, code, env_id", FIXABLE, ids=[f"{f[0]}-{f[3] or f[2]}-{i}" for i, f in enumerate(FIXABLE)])
def test_errors_every_fix_repairs_its_error(tool, arguments, kind, code, env_id):
    if tool == "submit" or env_id not in _SHARED:                       # one episode with a stored scan serves every case but a delivery
        env = make(env_id, 1, max_steps=10 ** 6, max_rejected_calls=10 ** 6)
        assert not call(env, "scan", **CHEAP).observation.is_error
        if tool != "submit":
            _SHARED[env_id] = env
    env = env if tool == "submit" else _SHARED[env_id]
    first = call(env, tool, **arguments)
    err = first.observation.data["error"]
    assert first.observation.is_error and not first.done and err["type"] == kind and err.get("code") == code, first.observation.text
    assert err["fix"] and all(p.get("parameter") for p in err["problems"]), first.observation.text
    canonical = env.admit({"name": tool, "arguments": arguments})[1]
    step = call(env, tool, **I.apply_fix(canonical, err["fix"]))
    again = step.observation.data.get("error") or {}
    same = {(p["type"], p.get("code"), p["parameter"]) for p in err["problems"]} & {(p["type"], p.get("code"), p["parameter"]) for p in again.get("problems", [])}
    assert not same, step.observation.text
    if tool == "submit":                                                # the repaired deliverable is delivered
        assert step.terminated and step.info["submitted"]
    else:
        assert not step.observation.is_error, step.observation.text


def test_errors_state_facts_of_the_console_only_and_list_every_problem():
    env = make(NOWRAP, 2, seed=3)
    call(env, "scan", **CHEAP)
    many = call(env, "plan", sequence="se", tr_ms=500, fov_mm=24, matrix=65, phase_fov=2, shim=1, ti_ms=5).observation
    err = many.data["error"]
    assert {p["parameter"] for p in err["problems"]} == {"shim", "ti_ms", "fov_mm", "phase_fov"} and err["type"] == "unknown_parameter"
    assert set(err["fix"]) == {"shim", "ti_ms", "fov_mm", "phase_fov"} and many.text.count("\n") <= 8
    after = call(env, "plan", **I.apply_fix(env.admit({"name": "plan", "arguments": dict(sequence="se", tr_ms=500, fov_mm=24, matrix=65, phase_fov=2, shim=1, ti_ms=5)})[1], err["fix"]))
    assert after.observation.data["error"]["parameter"] == "matrix"                     # what the first refusal could not see yet
    world = env.hidden_world()
    hidden = [f"{v:.4g}" for v in (world["system"]["tx_gain"], world["system"]["sigma0"])] + ["the object", "wrap-around", "outside the image", "cut off"]
    probe = make(NOWRAP, 1, max_rejected_calls=99)
    for _, _, (label, tool_call, expected) in PROBES[:30]:
        text = probe.step(tool_call).observation.text
        assert not expected or not [h for h in hidden if h in text.split("\n[")[0]], label


def test_errors_a_refused_deliverable_leaves_the_episode_running_until_the_third():
    env = make(NOWRAP, 1)
    call(env, "scan", **LOCALISER)
    bad = dict(pe_dir="y", fov_mm=130, phase_fov=0.5, matrix=65, **CENTRED)
    for k in (1, 2):
        step = call(env, "submit", **bad)
        assert not step.done and step.observation.data["error"]["type"] == "invalid_deliverable" and env.state()["rejected_calls"] == k
    ok = call(env, "submit", **I.apply_fix(bad, step.observation.data["error"]["fix"]))
    assert ok.terminated and ok.info["submitted"] and ok.info["score"]["checks"]["deliverable_valid"]
    env = make(NOWRAP, 1)
    steps = [call(env, "submit", **bad) for _ in range(3)]
    assert [s.done for s in steps] == [False, False, True] and steps[-1].truncated and steps[-1].reward == 0.0
    assert steps[-1].info["reason"] == "rejected_call_limit" and not steps[-1].observation.data["episode_continues"]
    # the families' own check goes through the same contract
    null = make(NULL, 1)
    ids = [r["id"] for r in null._regions]
    refused = call(null, "submit", ti_ms={i: 6000.0 for i in ids})
    assert not refused.done and refused.observation.data["error"]["type"] == "invalid_deliverable" and "nothing was delivered" in refused.observation.text
    run = policies.run(mrgym.make(NOWRAP, level=1, seed=0), "wrong_units")                   # a scripted policy whose deliverable is refused
    assert run.reward == 0.0 and run.truncated and not run.info["submitted"] and not run.info["score"]["checks"]["deliverable_valid"]


# ------------------------------------------------------------------------------------------------ limits
def _numbers(text):
    """The limits as the task text states them."""
    block = text[text.index("Limits. "):]
    return dict(scan_time_budget_s=float(re.search(r"Scan time: ([\d.]+) s", block)[1]), max_tool_calls=int(re.search(r"Tool calls: (\d+) ", block)[1]),
                max_rejected_calls=int(re.search(r"Rejected calls: (\d+) ", block)[1]),
                max_invalid_deliverables=int(re.search(r"of them at most (\d+) refused", block)[1]))


def _status(line):
    """The status line: the tool calls always; the scan time and the rejected calls when they changed (None otherwise)."""
    m = re.fullmatch(r"\[(?:scan time left ([\d.]+) of ([\d.]+) s · )?calls (\d+)/(\d+)(?: · rejected (\d+)/(\d+)(?: \(nothing charged\))?)?"
                     r"(?: · \w+ now or the episode scores 0)?\]", line)
    assert m, line
    number = lambda x, kind: None if x is None else kind(x)              # noqa: E731
    out = dict(left_s=number(m[1], float), scan_time_budget_s=number(m[2], float), tool_calls=int(m[3]), max_tool_calls=int(m[4]),
               left=int(m[4]) - int(m[3]), rejected=number(m[5], int), max_rejected_calls=number(m[6], int))
    out["used_s"] = None if out["left_s"] is None else round(out["scan_time_budget_s"] - out["left_s"], 2)
    return out


@pytest.mark.parametrize("env_id, level", EVERY)
@pytest.mark.parametrize("max_steps", [8, 24, 40])
def test_limits_are_one_set_of_numbers_and_the_ones_that_apply(env_id, level, max_steps):
    env = mrgym.make(env_id, level=level, seed=1, max_steps=max_steps)
    obs, info = env.reset()
    told, line = _numbers(obs.text), _status(obs.text.splitlines()[-1])
    assert told == info["limits"] == obs.data["limits"] == env.limits() and told["max_tool_calls"] == max_steps
    assert {k: line[k] for k in ("scan_time_budget_s", "max_tool_calls", "max_rejected_calls")} == {k: told[k] for k in ("scan_time_budget_s", "max_tool_calls", "max_rejected_calls")}
    assert "requests" not in obs.text and env._session.max_requests >= max_steps          # the scanner's own counter never binds first
    # what is used at reset: nothing, except on a practice level that acquires a measurement at reset. That scan is in
    # the log and charged, and the reset status, the `status` tool and the state give the same numbers for it
    log = env.scanner_log() if hasattr(env, "scanner_log") else []
    practice = getattr(getattr(env, "task", None), "LEVELS", {}).get(level, {}).get("kind") == "practice"
    acquired = [e for e in log if e["type"] == "scan"]
    assert practice or not log
    assert (line["used_s"], line["tool_calls"], line["rejected"]) == (round(sum(e["cost_s"] for e in acquired), 2), 0, 0)
    assert env.state()["scans"] == len(acquired)
    status = call(env, "status").observation
    assert status.data["limits"] == told and f"tool calls: 1 of {max_steps} used, {max_steps - 1} left" in status.text
    assert f"scan time: {line['used_s']:.2f} of {told['scan_time_budget_s']:g} s used, {line['left_s']:.2f} s left" in status.text
    assert f"stored scans: {len(acquired)}" in status.text
    after = _status(status.text.splitlines()[-1])                       # under the status tool the line is the full one again
    assert (after["tool_calls"], after["used_s"], after["rejected"], after["max_rejected_calls"]) == (1, line["used_s"], 0, told["max_rejected_calls"])
    assert len(status.data.get("scans", acquired)) == len(acquired) and env.state()["scan_time_used_s"] == pytest.approx(line["used_s"], abs=0.006)
    for k in range(3):                                                   # a refused call is not a tool call; the line says so when it happens
        refused = _status(call(env, "nope").observation.text.splitlines()[-1])
        assert (refused["tool_calls"], refused["rejected"], refused["max_rejected_calls"], refused["left_s"]) == (1, k + 1, told["max_rejected_calls"], None)
    assert env.state()["tool_calls"] == 1 and env.state()["rejected_calls"] == 3
    step = None
    for k in range(2, max_steps + 1):                                    # the episode ends at exactly the call the text names
        assert not env.state()["done"]
        step = call(env, "plan", **CHEAP) if k % 2 else call(env, "calc", expressions="1 + 1")
        last = _status(step.observation.text.splitlines()[-1])
        # nothing but the call count changed: the line holds the call count and the limit, and nothing else
        assert (last["tool_calls"], last["left"], last["rejected"], last["left_s"]) == (k, max_steps - k, None, None)
        assert ("now or the episode scores 0" in step.observation.text) == (0 < max_steps - k <= 3)
        assert step.observation.status["rejected_calls"] == 3 and step.observation.status["max_rejected_calls"] == told["max_rejected_calls"]
    assert step.truncated and not step.terminated and step.reward == 0.0 and step.info["reason"] == "max_steps"
    assert step.info["limits"] == told and _status(step.observation.text.splitlines()[-1])["max_tool_calls"] == max_steps
    assert step.info["interface"] == dict(calls=max_steps + 3, executed=max_steps, rejected=3, shape=0, vocabulary=3, task_refusals=0,
                                          coerced=0, internal_errors=0)


def test_limits_rejected_calls_end_the_episode_at_the_number_told():
    env = mrgym.make(NOWRAP, level=1, seed=0)
    obs, info = env.reset(options={"harness": {"max_reply_tokens": 4096}})
    assert "Reply length: 4096 tokens." in obs.text and info["limits"]["harness"] == {"max_reply_tokens": 4096}
    steps = [call(env, "plan", sequence="se", tr_ms="soon") for _ in range(info["limits"]["max_rejected_calls"])]      # a wrong type: shape
    assert [s.done for s in steps] == [False] * 7 + [True] and steps[-1].truncated and steps[-1].info["reason"] == "rejected_call_limit"
    assert "rejected 8/8 (nothing charged)" in steps[-1].observation.text and env.state()["tool_calls"] == 0
    with pytest.raises(mrgym.ResetNeeded):
        call(env, "status")
    for bad in ({"seed": 1}, {"harness": {"max_turns": 3}}, {"harness": {"max_reply_tokens": 0}}):
        with pytest.raises(mrgym.Error):
            env.reset(options=bad)
    for bad in (dict(max_steps=0), dict(max_steps=2.5), dict(max_rejected_calls=0), dict(max_steps=True)):
        with pytest.raises(mrgym.Error):
            mrgym.make(NOWRAP, **bad)


def test_limits_the_runner_has_no_cap_of_its_own():
    source = inspect.getsource(runner.play_model)
    assert "max_turns" not in source and "while out[\"final\"] is None" in source           # the loop ends with the environment or on idle replies
    comparisons = re.findall(r"^.*(?:max_turns\s*[<>]=?|[<>]=?\s*max_turns).*$", inspect.getsource(runner), re.M)
    assert len(comparisons) == 1 and "raise mrgym.Error" in inspect.getsource(runner).split(comparisons[0])[1][:120]      # input validation only
    item, = runner.dry_run([NOWRAP], levels=[1], seeds=[0], max_turns=17, max_tokens=4096)
    task = item["request"]["messages"][1]["content"]
    assert item["limits"]["max_tool_calls"] == 17 and _numbers(task)["max_tool_calls"] == 17 and "Reply length: 4096 tokens" in task
    assert not re.search(r"\d", item["request"]["messages"][0]["content"].replace(str(runner.MAX_CALLS_PER_TURN), ""))     # the note states no limit
    assert runner.make_env(NOWRAP, 1, 0, runner.model_settings(17, False, None, None)).max_steps == 17
    assert runner.make_env(NOWRAP, 1, 0, runner.model_settings(None, True, None, None)).max_steps == mrgym.spec(NOWRAP).max_steps


# ------------------------------------------------------------------------------------------------ what a tool reveals
REVEAL = re.compile(r"extent|bounding|touches|centroid|mask", re.I)
REVEAL_SEEDS = int(os.environ.get("MRGYM_REVEAL_SEEDS", "3"))                      # 50 for the acceptance run


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("obs_mode", ["text", "vision"])
def test_reveal_no_tool_measures_the_object(level, obs_mode):
    for seed in range(REVEAL_SEEDS):
        env = make(NOWRAP, level, seed=seed, obs=obs_mode)
        steps = [call(env, "plan", **LOCALISER), call(env, "scan", **LOCALISER), call(env, "view", scan_index=0),
                 call(env, "view", scan_index=0, size=48), call(env, "profile", scan_index=0, axis="y"), call(env, "status"),
                 call(env, "read_regions", scan_index=0, circles=[dict(y_mm=0, x_mm=0, radius_mm=20)]), call(env, "scanner_info")]
        for step in steps:
            obs = step.observation
            assert not obs.is_error and not REVEAL.search(obs.text), (obs.tool, REVEAL.search(obs.text))
            assert not [w for w in walk(obs.data) if REVEAL.search(w) and obs.tool != "scanner_info"], obs.tool
        view = steps[2].observation
        assert set(view.data) == {"scan_index", "image_index", "kind", "shape", "fov_yx_mm", "centre_yx_mm", "pixel_yx_mm", "sequence"} \
            | ({"png_bytes", "display_range"} if obs_mode == "vision" else {"digit_map"})
        assert "array" not in view.data and all(not isinstance(v, np.ndarray) for v in view.data.values())
    assert "profile" in " ".join(t["name"] for t in env.tools("neutral"))
    assert not REVEAL.search(json.dumps([t for t in env.tools("openai") if t["function"]["name"] in ("view", "scan", "plan", "read_regions")]))


def test_reveal_the_ruler_of_scripted_policies_is_on_the_trainer_side():
    env = make(NOWRAP, 2, seed=4)
    sc = policies.ToolScanner(env)
    k = sc.scan(LOCALISER)["scan_index"]
    seen = sc.view(k)                                                    # one tool call, plus the stored array read in process
    assert env.state()["tool_calls"] == 2 and isinstance(seen["array"], np.ndarray) and seen["scan"] == k
    box = seen["bounding_box"]
    assert box == session.image_extent(env.stored_scans()[k]) == session.LocalScanner(env._session).__class__.view(env._scanner, k)["bounding_box"]
    assert not box["touches_edge"] and box["y_mm"][0] < 0 < box["y_mm"][1]
    last = env.transcript()[-1]                                          # what the tool itself returned: no ruler, no array
    assert last["call"]["name"] == "view" and not REVEAL.search(last["text"]) and "bounding_box" not in env._log[-1]["observation"].data
    assert not hasattr(env, "_tool_extent") and "bounding" not in inspect.getsource(scanner_env)


def _edges(profile, fraction=0.2):
    top = max(profile["values"])
    hit = [y for y, v in zip(profile["positions_mm"], profile["values"]) if v > fraction * top]
    return min(hit) - profile["sample_mm"] / 2, max(hit) + profile["sample_mm"] / 2


def profile_policy(sc):
    """What a text-only agent can do with the numbers of ``profile``: one localiser, a maximum-intensity projection
    along each axis, the outermost samples above 20 % of the largest; the field of view centred on what they bound,
    the shorter side phase-encoded. (It does not turn the field of view: enough where the object lies along x or y.)"""
    px = sc.info()["task"]["max_pixel_mm"]
    k = sc.scan(dict(sequence="spgr", tr_ms=12.0, flip_deg=10.0, fov_mm=288.0, matrix=128, averages=2))["scan_index"]
    size, centre = {}, {}
    for axis in ("y", "x"):
        lo, hi = _edges(sc.profile(k, axis, width_mm=288.0, reduce="max"))
        # pixel i of an n-point axis is centred at centre + (i - n // 2) * pixel (datasheet): the image sits half a pixel
        # towards the negative side of its centre, so the centre goes half a pixel beyond the middle of the object
        size[axis], centre[axis] = hi - lo + 3.0, round((lo + hi) / 2 + px / 2, 2)             # 1.5 mm to spare on each side
    pe, ro = ("y", "x") if size["y"] <= size["x"] else ("x", "y")
    n_pe = 2 * math.ceil(size[pe] / 2 / px)
    n_ro = max(2 * math.ceil(size[ro] / 2 / px), n_pe, 32)
    return dict(pe_dir=pe, fov_mm=round(n_ro * px, 6), phase_fov=round(n_pe / n_ro, 6), matrix=n_ro, centre_y_mm=centre["y"],
                centre_x_mm=centre["x"], rotation_deg=0.0, phase_oversampling=0.0)


def test_reveal_the_numbers_of_profile_suffice_for_the_task():
    """The ruler is gone from ``view``; the task is still solvable from numbers alone, by an agent that measures."""
    runs = [policies.run(mrgym.make(NOWRAP, level=2, seed=s), profile_policy) for s in range(6)]
    assert [r.info["policy_error"] for r in runs] == [None] * 6 and sum(r.reward == 1.0 for r in runs) >= 5
    assert all(r.info["budget"]["scans"] == 1 for r in runs)


def test_profile_gives_the_numbers_along_a_line():
    env = make(NOWRAP, 1, seed=2)
    call(env, "scan", sequence="spgr", tr_ms=12, flip_deg=10, fov_mm=288, matrix=96, phase_fov=0.75)       # 72 x 96, pixel 3 mm
    img = env.stored_scans()[0]["magnitude"][0]
    ypos, xpos = session.pixel_positions(img.shape, (216.0, 288.0))
    line = call(env, "profile", scan_index=0, axis="y", position_mm=6).observation
    d = line.data
    assert d["positions_mm"] == [round(float(v), 3) for v in ypos] and d["width_px"] == 1 and d["pixels_per_sample"] == 1 and d["pixel_mm"] == 3.0
    assert np.allclose(d["values"], img[:, list(xpos).index(6.0)], rtol=1e-5) and (d["start_mm"], d["stop_mm"]) == (-109.5, 106.5)
    rows = line.text.split("\n[")[0].splitlines()
    assert "rows: y_mm value (centre of the sample; image units)" in rows[1] and len(rows) == 2 + 72 and float(rows[2].split()[0]) == ypos[0]
    assert float(rows[9].split()[1]) == pytest.approx(d["values"][7], abs=1e-3 * max(d["values"]))    # the numbers are in the text
    across = call(env, "profile", scan_index=0, axis="x", position_mm=0, width_mm=9, start_mm=-30, stop_mm=30).observation.data
    cols = [i for i, x in enumerate(xpos) if -31.5 <= x <= 31.5]
    assert across["width_px"] == 3 and across["positions_mm"] == [round(float(xpos[i]), 3) for i in cols]
    assert np.allclose(across["values"], img[35:38][:, cols].mean(0), rtol=1e-5)
    mip = call(env, "profile", scan_index=0, axis="x", width_mm=600, reduce="max", samples=24).observation.data
    assert len(mip["values"]) == 24 and mip["pixels_per_sample"] == 4 and mip["sample_mm"] == 12.0
    assert np.allclose(mip["values"], img.max(0).reshape(24, 4).max(1), rtol=1e-5)
    big = make(EXPLORE, 1)
    call(big, "scan", sequence="spgr", tr_ms=12, flip_deg=10, matrix=256)
    capped = call(big, "profile", scan_index=0, axis="y").observation
    assert len(capped.data["values"]) == session.MAX_PROFILE_SAMPLES == 128 and capped.data["pixels_per_sample"] == 2
    assert len(capped.text) < 4000 and "samples" in capped.text
    full = call(big, "profile", scan_index=0, axis="y", start_mm=-40, stop_mm=40).observation.data        # narrowed: one sample per pixel
    assert full["pixels_per_sample"] == 1 and len(full["values"]) < 128
    phase = call(env, "profile", scan_index=0, axis="y", kind="phase", width_mm=9).observation
    assert all(-math.pi <= v <= math.pi for v in phase.data["values"]) and "rad" in phase.text
    outside = call(env, "profile", scan_index=0, axis="y", position_mm=500).observation
    assert outside.is_error and outside.data["error"]["parameter"] == "position_mm" and outside.data["error"]["fix"] == {"position_mm": 142.5}
    direct = session.LocalScanner(env._session)                          # the same call on the scanner interface and through the tools
    direct.scans = env._scanner.scans
    assert policies.ToolScanner(env).profile(0, "y", position_mm=6)["values"] == direct.profile(0, "y", position_mm=6)["values"] == d["values"]


def test_console_every_violation_code_has_a_typed_error_with_a_nearest_valid_value():
    """Every code ``console.validate`` / ``console.plan`` can emit is mapped, and every limit of the console comes with
    the nearest valid value and a fix that the console then accepts."""
    source = inspect.getsource(console)
    emitted = set(re.findall(r'_violation\("([A-Z_]+)"', source))
    assert emitted == set(I.CONSOLE_CODES), emitted ^ set(I.CONSOLE_CODES)
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
    limits = dict(
        TE_BELOW_MINIMUM=dict(sequence="spgr", tr_ms=20, flip_deg=10, te_ms=0.6),
        TR_BELOW_MINIMUM=dict(sequence="spgr", tr_ms=10, flip_deg=10),                      # 10 ms: below the minimum of the default pulse
        TI_NOT_REALISABLE=dict(sequence="ir", tr_ms=3000, ti_ms=5.5, fat_sat=True),
        ECHO_SPACING_BELOW_MINIMUM=dict(sequence="epi", tr_ms=2000, echo_spacing_ms=0.21),
        RESOLUTION_LIMIT=dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=512),
        SPOILER_BELOW_MINIMUM=dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=256, spoiler_mt_ms_per_m=10),
        RF_POWER_LIMIT=dict(sequence="bssfp", tr_ms=4.2, flip_deg=90, matrix=64, bandwidth_hz_per_px=1500),
        GRADIENT_LIMIT=dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=256, bandwidth_hz_per_px=3000),
        SLICE_TOO_THIN=dict(sequence="se", tr_ms=500, slice_mm=1, rf_pulse="sharp"))
    assert set(limits) == set(I._LIMIT_CODES)
    for code, protocol in limits.items():
        pl = console.plan(protocol, exam)
        assert not pl["accept"] and code in [v["code"] for v in pl["violations"]], (code, pl.get("violations"))
        problems = I.console_problems(pl["violations"], protocol, exam)
        p = next(x for x in problems if x.get("code") == code)
        assert p["type"] == "console_limit" and p["class"] == "task" and p["parameter"] and p["nearest_valid"] is not None, p
        assert p["fix"] == {p["parameter"]: p["nearest_valid"]} and p["allowed"], p
        again = console.plan(I.apply_fix(protocol, I.merged_fix(problems)), exam)
        assert code not in [v["code"] for v in again.get("violations", [])], (code, again.get("violations"))
    for protocol, code, kind in ((dict(sequence="nope"), "UNKNOWN_SEQUENCE", "invalid_value"), (dict(sequence="se", tr_ms=500, matrx=64), "UNKNOWN_PARAMETER", "unknown_parameter"),
                                 (dict(sequence="se", tr_ms=500, ti_ms=5), "UNKNOWN_PARAMETER", "not_applicable"), (dict(sequence="spgr", tr_ms=12), "MISSING_PARAMETER", "invalid_value"),
                                 (dict(sequence="se", tr_ms=500, rotation_deg=95), "OUT_OF_RANGE", "invalid_value"), (dict(sequence="se", tr_ms=500, rf_pulse="hard"), "OUT_OF_RANGE", "invalid_value"),
                                 (dict(sequence="se", tr_ms=500, matrix=65), "OUT_OF_RANGE", "invalid_value"), ("se", "INVALID_REQUEST", "malformed_call")):
        problems = I.console_problems(console.validate(protocol, exam)[2], protocol if isinstance(protocol, dict) else None, exam)
        assert [(p["code"], p["type"]) for p in problems] == [(code, kind)], (protocol, problems)
        assert all(p["type"] in I.ERROR_TYPES for p in problems)
    assert I.console_problems(console.validate(dict(sequence="se", tr_ms=500, rotation_deg=95), exam)[2], {}, exam)[0]["nearest_valid"] == 90.0
    assert I.console_problems(console.validate(dict(sequence="se", tr_ms=500, rf_pulse="hard"), exam)[2], {}, exam)[0]["allowed"] == {"values": ["fast", "normal", "sharp"]}


def test_schema_carries_the_parameters_of_the_slice_selective_engine():
    env = make(T1, 1)
    props = {t.name: t.parameters["properties"] for t in env._tool_specs()}
    for name in ("rotation_deg", "rf_pulse", "excitation", "fatsat_offset_hz"):
        assert name in props["plan"] and name in props["scan"] and name in console.PARAMETER_MEANING and name in console.PARAMETER_ORDER
        assert len(props["plan"][name]["description"]) < 120 and "default" in props["plan"][name]["description"] and "description" not in props["scan"][name]
    assert props["plan"]["rf_pulse"]["enum"] == ["fast", "normal", "sharp"] and "+-90" in props["plan"]["rotation_deg"]["description"]
    assert set(console.parameter_names()) == {k for q in console.SEQUENCES.values() for k in q} == set(props["plan"]) - {"sequence", "verbose"}
    assert set(console.ALIASES.values()) <= set(console.parameter_names()) and console.ALIASES["fat_sat_offset_hz"] == "fatsat_offset_hz"
    p = call(env, "plan", sequence="spgr", tr_ms=12, flip_deg=10, rf_pulse="fast", fov_rotation_deg="30 deg").observation
    assert not p.is_error and p.data["realised"]["rf_pulse"] == "fast" and p.data["realised"]["rotation_deg"] == 30.0 and "rot +30" in p.data["header"]
    assert "rot +30" in call(env, "plan", sequence="spgr", tr_ms=12, flip_deg=10, rotation_deg=30, verbose=True).observation.text      # the header line: verbose
    text = call(env, "scanner_info", topic="se").observation.text
    assert all(name in text for name in ("rotation_deg 0", "rf_pulse normal", "excitation selective", "fatsat_offset_hz: -300 to 300; default 0"))
    shared = call(env, "scanner_info", topic="geometry").observation.text
    assert "rotation_deg, -90 to 90, 0: " in shared and "rf_pulse, fast / normal / sharp, normal: " in shared
    thin = call(env, "plan", sequence="se", tr_ms=500, slice_mm=1, rf_pulse="sharp").observation
    err = thin.data["error"]
    assert (err["type"], err["code"], err["parameter"]) == ("console_limit", "SLICE_TOO_THIN", "slice_mm") and err["fix"] == {"slice_mm": err["nearest_valid"]}
    assert not call(env, "plan", sequence="se", tr_ms=500, rf_pulse="sharp", **err["fix"]).observation.is_error


def test_profile_and_view_on_a_turned_field_of_view():
    """With ``rotation_deg`` the image axes are not the patient's: positions are along the image rows and columns, and
    every profile sample carries its patient position, the same one the scanner uses for a region read-out."""
    env = make(EXPLORE, 1, seed=3, phantom="qa")
    prot = dict(sequence="spgr", tr_ms=12, flip_deg=10, fov_mm=288, matrix=96, phase_fov=0.75)
    call(env, "scan", **prot)
    turned = call(env, "scan", **prot, rotation_deg=30, centre_y_mm=12, centre_x_mm=-9).observation
    assert not turned.is_error and turned.data["realised"]["rotation_deg"] == 30.0
    stored = env.stored_scans()[1]
    img = stored["magnitude"][0]
    ygrid, xgrid = session.pixel_grid(img.shape, stored["fov_yx_mm"], stored["centre_yx_mm"], 30.0)      # the engine's own map
    for axis, col in (("y", 40), ("x", 30)):
        frame = session.frame_centre(stored["centre_yx_mm"], 30.0)
        v, u = session.pixel_positions(img.shape, stored["fov_yx_mm"], frame)
        obs = call(env, "profile", scan_index=1, axis=axis, position_mm=float(u[col] if axis == "y" else v[col])).observation
        d = obs.data
        line = img[:, col] if axis == "y" else img[col, :]
        assert d["rotation_deg"] == 30.0 and np.allclose(d["values"], line, rtol=1e-5) and d["width_px"] == 1
        patient = np.array(d["patient_yx_mm"])
        want = np.stack([ygrid[:, col], xgrid[:, col]] if axis == "y" else [ygrid[col, :], xgrid[col, :]], axis=1)
        assert np.allclose(patient, want, atol=2e-3)                    # every sample where the engine puts that pixel
        rows = obs.text.split("\n[")[0].splitlines()
        assert f"rows: {axis}_mm value patient_y_mm patient_x_mm (" in rows[1] and "turned by r = +30 degrees" in rows[2] and len(rows[3].split()) == 4
        k = int(np.argmax(line))                                         # a one-pixel circle at that patient position reads the same pixel
        got = call(env, "read_regions", scan_index=1, regions=[], circles=[dict(y_mm=float(patient[k, 0]), x_mm=float(patient[k, 1]), radius_mm=1.0)]).observation.data
        assert got["roi_readouts"][0]["n_pixels"] == 1 and got["roi_readouts"][0]["mean_magnitude"][0] == pytest.approx(d["values"][k], rel=1e-5)
    # the object is where it was: its outline along the turned rows, mapped to the patient, lies on the outline of the upright scan
    mip = call(env, "profile", scan_index=1, axis="y", width_mm=400, reduce="max").observation.data
    lo, hi = _edges(mip)
    box = session.image_extent(stored)                                   # the trainer's ruler measures in the same frame
    assert abs(lo - box["y_mm"][0]) <= 2 * mip["sample_mm"] and abs(hi - box["y_mm"][1]) <= 2 * mip["sample_mm"]
    view = call(env, "view", scan_index=1).observation
    assert "turned by +30 degrees" in view.text and "patient position (y cos r - x sin r, y sin r + x cos r)" in view.text
    assert view.data["rotation_deg"] == 30.0 and view.data["frame_centre_yx_mm"] == [round(c, 3) for c in session.frame_centre((12.0, -9.0), 30.0)]
    assert f"top = {session.frame_centre((12.0, -9.0), 30.0)[0] - 108:.0f} mm" in view.text
    upright = call(env, "view", scan_index=0).observation
    assert "turned" not in upright.text and "rotation_deg" not in upright.data and "patient_yx_mm" not in call(env, "profile", scan_index=0, axis="y").observation.data
    sc = policies.ToolScanner(env)                                       # the scanner interface of the policies agrees
    assert sc.view(1)["rotation_deg"] == 30.0 and sc.view(1)["bounding_box"] == box and sc.view(0)["rotation_deg"] == 0.0
    vision = make(T1, 1, obs="vision")
    call(vision, "scan", **prot, rotation_deg=-20)
    assert not call(vision, "view", scan_index=0).observation.is_error and "turned by -20 degrees" in call(vision, "view", scan_index=0).observation.text


# ------------------------------------------------------------------------------------------------ step is total
def test_total_a_fault_of_the_environment_is_answered_counted_and_survived(monkeypatch):
    env = make()
    monkeypatch.setattr(env, "_tool_plan", lambda a: (_ for _ in ()).throw(RuntimeError("secret path /x/y")))
    step = call(env, "plan", **CHEAP)
    obs = step.observation
    assert obs.is_error and not step.done and obs.data["error"]["type"] == "internal_error" and obs.data["error"]["class"] == "internal"
    assert "secret" not in obs.text and "secret" not in json.dumps(obs.data) and env.faults() == ["RuntimeError: secret path /x/y"]
    assert env.state()["internal_errors"] == 1 and env.state()["tool_calls"] == 0 and env.interface_counts()["internal_errors"] == 1
    assert not call(env, "scan", **CHEAP).observation.is_error                              # the episode is still usable
    monkeypatch.setattr(env, "_tool_view", lambda a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):                                                  # never BaseException
        call(env, "view", scan_index=0)


def test_total_nothing_returned_shares_an_object_with_the_episode():
    env = make()
    written = dict(CHEAP, circles=[dict(y_mm=0, x_mm=0, radius_mm=9)])
    step = env.step({"name": "scan", "arguments": written})
    written["circles"][0]["radius_mm"] = 99                               # the caller's object changes afterwards
    step.observation.data["realised"]["matrix"] = 1
    step.observation.data["roi_readouts"].clear()
    assert env.scanner_log()[-1]["realised"]["matrix"] == 48 and env.transcript()[-1]["call"]["arguments"]["circles"][0]["radius_mm"] == 9.0
    assert env._log[-1]["observation"].data["realised"]["matrix"] == 1                   # the observation is the caller's own copy ...
    assert env._scanner.scans[0]["realised"]["matrix"] == 48                             # ... and not the scanner's record
    log = env.scanner_log()
    log[-1]["cost_s"] = 0.0
    cfg = env.episode_config()
    cfg["task"]["regions"] = "changed"
    cfg["budget_s"] = 1e9
    assert env.scanner_log()[-1]["cost_s"] > 0 and env.episode_config()["budget_s"] == 30.0 and env.episode_config()["task"]["regions"] == []
    assert session.replay(env.episode_config(), env.scanner_log(), head=env._session.receipt())["ok"]
    end = call(env, "submit", pe_dir="y", fov_mm=260, phase_fov=1, matrix=130, **CENTRED)
    end.info["score"]["reward"] = 5
    assert env._episode["info"]["score"]["reward"] != 5


@pytest.mark.parametrize("hostile, kind", [
    ({"method": "tools/call", "params": [1, 2], "id": 1}, "malformed_call"), ({"method": "tools/call", "params": "x"}, "malformed_call"),
    ({"name": ["scan"], "arguments": {}}, "malformed_call"), ([1], "malformed_call"), (None, "malformed_call"),
    ({"type": "function", "function": "plan"}, "malformed_call"), ({"name": "plan", "arguments": {1: 2}}, "malformed_call"),
    ({"name": "plan", "arguments": {"protocol": {"sequence": b"spgr"}}}, "malformed_call"),
    ({"name": "plan", "arguments": {"protocol": {"sequence": {"spgr"}}}}, "malformed_call"),
    ({"name": "plan", "arguments": {"sequence": "s" * 1_000_000}}, "malformed_call"),
    ({"name": "plan", "arguments": '{"sequence": "spgr", "tr_ms": NaN}'}, "malformed_call"),
    ({"name": "plan", "arguments": '{"sequence": "spgr", "tr_ms": 1' + "0" * 400 + "}"}, "malformed_call"),
    ({"name": "plan", "arguments": "[" * 5000 + "]" * 5000}, "malformed_call"),
    ({"name": "plan", "arguments": "x" * 70000}, "malformed_call"),
    ({"name": "plan", "arguments": {"sequence": "\ud800", "tr_ms": 10}}, "invalid_value"),
    ({"name": "x" * 100, "arguments": {}}, "malformed_call"),
    ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": "1" + "0" * 400}}, "invalid_value"),
    ({"name": "plan", "arguments": {"sequence": "se", "tr_ms": 500, "matrix": "[" + "-" * 19000 + "1]"}}, "invalid_value"),
    ({"name": "scan", "arguments": {"sequence": "se", "tr_ms": 500, "circles": "(" * 9000 + ")" * 9000}}, "invalid_value"),
])
def test_total_hostile_calls_are_malformed_calls(hostile, kind):
    env = make()
    step = env.step(hostile)
    assert step.observation.is_error and not step.done and step.observation.data["error"]["type"] == kind
    step.observation.text.encode("utf-8")                                # what goes to a model is always encodable
    json.dumps(env.transcript())                                         # and the transcript stays serialisable
    deep = {}
    node = deep
    for _ in range(3000):
        node["a"] = {}
        node = node["a"]
    for tool_call in ({"name": "plan", "arguments": {"protocol": deep}}, {"name": "scan", "arguments": dict(CHEAP, circles=deep)},
                      {"name": "plan", "arguments": {"a": list(range(5000))}}):
        obs = env.step(tool_call).observation
        assert obs.data["error"]["type"] == "malformed_call" and ("nested too deeply" in obs.text or "too large" in obs.text)
    json.dumps(env.transcript())
    assert not call(env, "plan", **CHEAP).observation.is_error


@pytest.mark.parametrize("env_id", mrgym.list_envs())
def test_total_sizes_of_the_tool_schemas(env_id):
    """Characters (about 3.3 per token) of what a model is sent before its first call. The parameters are documented
    once, in ``plan``, in one short line each; ``scan`` repeats names and types; the tables are behind ``scanner_info``."""
    env = mrgym.make(env_id, seed=0)
    obs, _ = env.reset()
    size = {t["function"]["name"]: len(json.dumps(t)) for t in env.tools("openai")}
    terminal = next(t.name for t in env._tool_specs() if t.terminal)
    # engine 0.3.0: + 550 in all, + 380 in plan, + 150 in scan for the four adjustment parameters (cf_shift_hz, tx_scale, shim_y / shim_x_hz_per_mm)
    # a design family: + 4,300 for its two tools (design, simulate) and + 350 in plan for the scan that plays a design
    design = {"design", "simulate"} <= set(size)
    assert design == (env_id == "Scanner/RFPulse-v0")
    assert sum(v for k, v in size.items() if k != terminal) < 11950 + (4300 if design else 0), size   # the scanner tools; the terminal tool is the family's
    assert size["plan"] < 4380 + (350 if design else 0) and size["scan"] < 2600 and size["profile"] < 1500
    assert len(obs.text[obs.text.index("Limits. "):].splitlines()[0]) < 360 and "Tools:" not in obs.text
    specs = {t.name: t for t in env._tool_specs()}
    props = {name: t.parameters["properties"] for name, t in specs.items()}
    assert max(len(p["description"]) for p in props["plan"].values()) < 160
    # scan lists the parameters of plan by name and type; plan documents them (ToolSpec.see), ranges included, in words
    assert specs["scan"].see == "plan" and all("description" not in p and k in props["plan"] for k, p in props["scan"].items() if k not in ("regions", "circles"))
    assert not [k for name in ("plan", "scan") for k, p in props[name].items() if "minimum" in p]
    assert "80-500" in props["plan"]["fov_mm"]["description"] and "32-512" in props["plan"]["matrix"]["description"]
    for name in ("plan", "scan"):                                                    # the range is checked where the call arrives, for both
        low = env.step({"name": name, "arguments": {"sequence": "se", "tr_ms": 500, "fov_mm": 24}}).observation
        assert low.is_error and low.data["error"]["fix"] == {"fov_mm": 240.0} and "24 cm = 240 mm" in low.text
    refused = env.step({"name": "scan", "arguments": {"sequence": "se", "tr_ms": 500, "averages": 100}}).observation
    assert refused.is_error and refused.data["error"]["parameter"] == "averages" and refused.data["error"]["fix"] == {"averages": 64}


def test_total_sizes_of_what_the_agent_reads():
    """Budgets in characters of every result."""
    env = make(NOWRAP, 1)
    sizes = dict(plan=call(env, "plan", **LOCALISER), scan=call(env, "scan", **LOCALISER), view=call(env, "view", scan_index=0),
                 view32=call(env, "view", scan_index=0, size=32), profile=call(env, "profile", scan_index=0, axis="y", samples=32),
                 status=call(env, "status"), error=call(env, "plan", sequence="se", tr_ms=5), se=call(env, "scanner_info", topic="se"),
                 overview=call(env, "scanner_info"), physics=call(env, "scanner_info", topic="physics"), verbose=call(env, "plan", verbose=True, **LOCALISER))
    # view 700 and view32 1550 (600 and 1400 before): the y of every row and an x ruler, 150 and 225 characters (test_interface_clarity.py)
    # physics 760 (700 before engine 0.3.0): one more line, the topic "adjustments"; verbose 1600 (1500): four more realised parameters
    limit = dict(plan=450, scan=400, view=700, view32=1550, profile=700, status=800, error=200, se=1150, overview=1800, physics=760, verbose=1600)
    assert {k: len(v.observation.text) for k, v in sizes.items() if len(v.observation.text) > limit[k]} == {}
    # what the short forms leave out is one call away: every realised parameter and the timing of a plan, the image at a finer grid
    assert "realised: " in sizes["verbose"].observation.text and "timing: " in sizes["verbose"].observation.text and "realised: " not in sizes["plan"].observation.text
    assert sizes["plan"].observation.data["realised"] == sizes["verbose"].observation.data["realised"]      # data: everything, always
    assert len(sizes["view"].observation.data["digit_map"]) == scanner_env.VIEW_SIZE == 16 and len(sizes["view32"].observation.data["digit_map"]) == 32
