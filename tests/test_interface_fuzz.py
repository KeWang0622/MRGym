"""Seeded fuzz of the tool interface (standard library only; ``random.Random(0)``, so every run sends the same calls).

    never raises     mutations of valid calls for every tool on two environments: ``env.step`` returns the five-tuple,
                     an error has a type of the closed vocabulary, charges no scan time and does not end the episode
    metamorphic      a valid call written in any safe spelling has the same canonical arguments, byte for byte, the
                     same observation text and the same head of the scanner's log
    wire formats     the neutral, OpenAI, MCP and Anthropic shapes of a call are one call

``MRGYM_FUZZ_N`` sets the number of mutations (default 10000).
"""
import json
import os
import random
import time

import pytest

import mrgym
from mrgym import interface as I
from mrgym.scanner import console

NOWRAP, T1 = "Scanner/NoWrap-v1", "Scanner/T1Transmit-v1"
N_MUTATIONS = int(os.environ.get("MRGYM_FUZZ_N", "10000"))
BIG = 10 ** 9


def fresh(env_id):
    env = mrgym.make(env_id, level=1, seed=0, max_steps=BIG, max_rejected_calls=BIG)
    env.reset()
    assert not env.step({"name": "scan", "arguments": {"sequence": "spgr", "tr_ms": 12, "flip_deg": 10, "matrix": 32}}).observation.is_error
    return env


# ------------------------------------------------------------------------------------------------ valid calls
def protocol(rng):
    """A protocol the schema accepts (the console may still refuse it: that is a refusal of the instrument)."""
    seq = rng.choice(sorted(console.SEQUENCES))
    p = {"sequence": seq, "tr_ms": {"spgr": 12, "afi": 20, "se": 500, "ir": 2000, "bssfp": 6, "epi": 2000}[seq]}
    if seq in ("spgr", "afi", "bssfp"):
        p["flip_deg"] = rng.choice([5, 10, 30, 60])
    if seq == "ir":
        p["ti_ms"] = rng.choice([100, 400, 800])
    for name, values in (("fov_mm", [200, 240, 288]), ("matrix", [32, 48, 64, 96]), ("phase_fov", [0.5, 0.75, 1]), ("averages", [1, 2]),
                         ("pe_dir", ["y", "x"]), ("slice_mm", [3, 5]), ("bandwidth_hz_per_px", [600, 1500]), ("phase_oversampling", [0, 0.5])):
        if rng.random() < 0.4:
            p[name] = rng.choice(values)
    if "te_ms" in console.SEQUENCES[seq] and rng.random() < 0.5:
        p["te_ms"] = rng.choice(["min", 20, 0.5])
    if "fat_sat" in console.SEQUENCES[seq] and rng.random() < 0.3:
        p["fat_sat"] = rng.choice([True, False])
        if rng.random() < 0.5:
            p["fatsat_offset_hz"] = rng.choice([-40, 0, 25.5])
    for name, values in (("rf_pulse", ["fast", "normal", "sharp"]), ("rotation_deg", [-30, 0, 45]), ("excitation", ["selective", "nonselective"])):
        if rng.random() < 0.2:
            p[name] = rng.choice(values)
    return p


def priced_out(rng):
    """A valid protocol that no budget of these environments can pay for."""
    p = protocol(rng)
    return dict(p, averages=64, matrix=96, phase_fov=1, tr_ms=min(5000, console.SEQUENCES[p["sequence"]]["tr_ms"][1]))


def valid_call(rng, env):
    """(tool name, arguments) of a call that fits its schema."""
    ids = [r["id"] for r in env._regions]
    circle = lambda: {"y_mm": rng.choice([-40, 0, 25.5]), "x_mm": rng.choice([-30, 0, 12]), "radius_mm": rng.choice([5, 8, 12.5])}   # noqa: E731
    kind = rng.choice(["plan", "plan", "plan", "scan", "read_regions", "view", "profile", "scanner_info", "status", "calc", "submit"])
    if kind == "plan":
        return kind, protocol(rng)
    if kind == "scan":
        a = priced_out(rng) if rng.random() < 0.97 else protocol(rng)   # mostly priced out: the scan time stays for the other calls
        if rng.random() < 0.5:
            a["circles"] = [circle() for _ in range(rng.randint(1, 3))]
        if ids and rng.random() < 0.5:
            a["regions"] = rng.choice([["all"], ids[:2], []])
        return kind, a
    if kind == "read_regions":
        return kind, {"scan_index": 0, "circles": [dict(circle(), id=f"c{i}") for i in range(rng.randint(1, 3))]}
    if kind == "view":
        return kind, dict({"scan_index": 0}, **rng.choice([{}, {"image_index": 0}, {"kind": "phase"}, {"size": 16}]))
    if kind == "profile":
        return kind, dict({"scan_index": 0, "axis": rng.choice(["y", "x"])}, **rng.choice([{}, {"position_mm": 10}, {"width_mm": 20, "reduce": "max"}, {"samples": 16}]))
    if kind == "scanner_info":
        return kind, rng.choice([{}, {"topic": "task"}, {"topic": "se"}, {"topic": "physics"}])
    if kind == "status":
        return kind, {}
    if kind == "calc":
        return kind, {"expressions": rng.choice(["1 + 1", "a = [1, 2, 3]\nmean(a)", "sqrt(2) * 3"])}
    if ids:
        return kind, {"t1_ms": {i: rng.choice([800.0, 1200.0]) for i in ids}}
    return kind, {"pe_dir": rng.choice(["y", "x"]), "fov_mm": 260, "phase_fov": 1, "matrix": 130, "centre_y_mm": rng.choice([0, 12.5]),
                  "centre_x_mm": 0, "rotation_deg": rng.choice([0, 30])}


# ------------------------------------------------------------------------------------------------ mutations
def _deep(n):
    out = node = {}
    for _ in range(n):
        node["a"] = {}
        node = node["a"]
    return out


JUNK = [None, True, False, 0, -1, 1.5, 1e308, float("nan"), float("inf"), -float("inf"), 10 ** 300, "", " ", "min", "NaN", "null", "0x10", "1e400",
        "é\u0000\ud800", b"bytes", [], {}, [[]], [{}], {"a": {"b": {"c": 1}}}, "{", "[1, 2", '{"a": 1}', "'x'", "True", "240 mm", "24 cm", "75%",
        "a" * 30, {1: 2}, {"": ""}, (1, 2), {1, 2}, object, range(3), "1" + "0" * 400, "[" + "-" * 3000 + "1]", "(" * 300 + ")" * 300, "[9**9**9]"]


def mutate(rng, name, args):
    """One hostile or broken variant of a valid call, as the object a harness would hand to ``env.step``."""
    args = json.loads(json.dumps(args))
    keys = list(args)
    op = rng.randrange(28)
    if op == 0 and keys:                                                 # a value of another type
        args[rng.choice(keys)] = rng.choice(JUNK)
    elif op == 1 and keys:                                               # every value as text
        args = {k: json.dumps(v) if rng.random() < 0.8 else str(v) for k, v in args.items()}
    elif op == 2 and keys:                                               # a key removed
        del args[rng.choice(keys)]
    elif op == 3 and keys:                                               # a key misspelt
        k = rng.choice(keys)
        args[k[:-1] + rng.choice("xyz_") if rng.random() < 0.5 else k.upper()] = args.pop(k)
    elif op == 4:                                                        # an unknown key
        args[rng.choice(["seed", "world", "x", "protocol", "arguments", "", "regions", "scan", "__class__", "a" * 200])] = rng.choice(JUNK)
    elif op == 5 and keys:                                               # a number out of every range
        args[rng.choice(keys)] = rng.choice([-1e30, 1e30, 10 ** 15, -(10 ** 15), 1e-300, 2 ** 63])
    elif op == 6:                                                        # the arguments as JSON text with constants no JSON has
        return {"name": name, "arguments": json.dumps(args)[:-1] + rng.choice([', "x": NaN}', ', "x": Infinity}', ', "x": 1' + "0" * 400 + "}", ', "x": 1e400}', ""])}
    elif op == 7:                                                        # a megabyte of text
        args[rng.choice(keys) if keys else "x"] = "s" * 1_000_000
    elif op == 8:                                                        # nesting far beyond any schema
        args[rng.choice(keys) if keys else "x"] = _deep(rng.choice([20, 50, 2000]))
    elif op == 9:                                                        # nesting as text
        return {"name": name, "arguments": "[" * 50 + "]" * 50 if rng.random() < 0.5 else '{"a": ' * 3000 + "1" + "}" * 3000}
    elif op == 10:                                                       # many values
        args["circles" if "circles" in args else "x"] = [dict(y_mm=0, x_mm=0, radius_mm=5)] * rng.choice([17, 500, 5000])
    elif op == 11:                                                       # the wrong envelope
        return rng.choice([args, [name, args], {"tool": name, "args": args}, {"name": name}, {"name": None, "arguments": args},
                           {"name": [name], "arguments": args}, {"name": name, "arguments": [args]}, {"name": name, "arguments": 7},
                           {"function": name}, {"function": {"name": name, "arguments": "{cut"}}, {"type": "function", "function": [name]},
                           {"method": "tools/call", "params": [name, args]}, {"method": "tools/call", "params": name},
                           {"method": "tools/call"}, {"type": "tool_use", "name": name, "input": "x"}, {"type": "tool_use", "input": args},
                           name, None, 0, b"scan", {"name": name * 40, "arguments": args}, {"id": {"a": 1}, "name": name, "arguments": args}])
    elif op == 12:                                                       # a tool that does not exist
        return {"name": rng.choice(["reset", "step", "state", "close", "eval", "run_python", name + "s", name.upper(), "", " "]), "arguments": args}
    elif op == 13 and keys:                                              # a duplicated key in JSON text (the last one wins in JSON)
        k = rng.choice(keys)
        return {"name": name, "arguments": json.dumps(args)[:-1] + f', {json.dumps(k)}: {json.dumps(rng.choice(["x", 1, None]))}' + "}"}
    elif op == 14:                                                       # bytes that are not UTF-8, lone surrogates
        args[rng.choice(keys) if keys else "x"] = rng.choice([b"\xff\xfe", "\udcff", "\ud83d", "\x00\x01"])
    elif op == 15 and keys:                                              # a unit, a percent sign, a word
        k = rng.choice(keys)
        args[k] = f"{args[k]} {rng.choice(['mm', 'cm', 'ms', 's', '%', 'deg', 'px', 'Hz'])}"
    elif op == 16:                                                       # the protocol nested, once or twice
        args = {"protocol": args if rng.random() < 0.5 else {"protocol": args}}
    elif op == 17 and keys:                                              # a list or an object where one value belongs
        k = rng.choice(keys)
        args[k] = rng.choice([[args[k]], {"value": args[k]}, [args[k], args[k]]])
    elif op == 18:                                                       # an XML tail inside a value
        args[rng.choice(keys) if keys else "x"] = '{"a": 1}\n<parameter=regions>\n["task"]\n</parameter>\n<parameter=' + rng.choice(["x>", "sequence>\nse", ">"])
    elif op == 19 and keys:                                              # boundary values
        args[rng.choice(keys)] = rng.choice([0, 1, -0.0, 0.5, 31, 32, 33, 512, 513, 0.249999, 1.0000001, 64.5, 2 ** 31, 10 ** 12, 10 ** 12 + 1])
    elif op == 20:                                                       # everything null
        args = dict.fromkeys(keys or ["x"])
    elif op == 21:                                                       # the arguments as Python text
        return {"name": name, "arguments": str(args)}
    elif op == 22:                                                       # empty or blank arguments
        return {"name": name, "arguments": rng.choice(["", " ", "{}", "null", "[]", "\n"])}
    elif op == 23 and keys:                                              # aliases next to their parameters
        args.update(scan=rng.choice(JUNK), image=rng.choice(JUNK), flip_angle_deg=rng.choice(JUNK))
    elif op == 24:                                                       # an OpenAI call whose id and type are junk
        return {"id": rng.choice(JUNK[:20]), "type": rng.choice(["function", "tool", 3]), "function": {"name": name, "arguments": json.dumps(args)}}
    elif op == 26 and any(isinstance(v, int) and not isinstance(v, bool) for v in args.values()):   # an even integer made odd (a matrix)
        k = rng.choice([k for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool)])
        args[k] += 1
    elif op == 25 and "sequence" in args:                               # a parameter of another sequence
        other = sorted({k for q in console.SEQUENCES.values() for k in q} - set(console.SEQUENCES.get(args["sequence"], {})))
        if other:
            args[rng.choice(other)] = rng.choice([1, 5, 90, 300, "min", True])
    else:                                                                # the valid call, untouched
        pass
    return {"name": name, "arguments": args}


def would_be_scored(env, tool_call) -> bool:
    """Whether a call to the terminal tool would end the episode with a score (the front half of step, and the
    environment's own check of the deliverable; nothing is executed)."""
    name, args, _, problems = env.admit(tool_call)
    return not problems and env.check_deliverable(args) is None


# ------------------------------------------------------------------------------------------------ (a) never raises
def test_fuzz_step_never_raises_and_every_error_is_typed():
    rng = random.Random(0)
    envs = {i: fresh(i) for i in (NOWRAP, T1)}
    seen, slow, n_errors, by_tool, times = set(), [], 0, set(), []
    for k in range(N_MUTATIONS):
        env_id = (NOWRAP, T1)[k % 2]
        env = envs[env_id]
        name, args = valid_call(rng, env)
        tool_call = mutate(rng, name, args)
        if name == "submit" and would_be_scored(env, tool_call) and rng.random() < 0.95:
            continue                                                     # a deliverable that would be scored: only one in twenty is sent (scoring is slow)
        used, scans = env.state()["scan_time_used_s"], env.state()["scans"]
        t0 = time.perf_counter()
        result = env.step(tool_call)                                     # must not raise, whatever was sent
        dt = time.perf_counter() - t0
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, mrgym.Observation) and isinstance(reward, float) and reward == reward and isinstance(info, dict)
        assert type(terminated) is bool and type(truncated) is bool and not (terminated and truncated)
        assert isinstance(obs.text, str) and obs.text and isinstance(obs.data, dict) and obs.status["tool_calls"] >= 0
        by_tool.add(name)
        if obs.is_error:
            err = obs.data["error"]
            n_errors += 1
            seen.add(err["type"])
            times.append(dt)
            assert err["type"] in I.ERROR_TYPES and err["class"] in I.CLASSES and err["type"] != "internal_error", (tool_call, obs.text)
            assert obs.data["charged"]["scan_time_s"] == 0 and env.state()["scan_time_used_s"] == used and env.state()["scans"] == scans
            assert obs.data["episode_continues"] == (not result.done)
            assert not result.done or info["reason"] == "rejected_call_limit", obs.text      # only the third refused deliverable ends an episode
            if dt > 0.05:
                slow.append((round(dt, 3), name, str(tool_call)[:80]))
        else:
            assert reward == 0.0 or terminated
        if result.done:                                                  # a delivered answer, or three refused ones: play on
            assert terminated or info["reason"] == "rejected_call_limit"
            envs[env_id] = fresh(env_id)
    assert envs[NOWRAP].state()["internal_errors"] == 0 and envs[T1].state()["internal_errors"] == 0
    assert {"malformed_call", "unknown_tool", "unknown_parameter", "not_applicable", "invalid_value", "console_limit", "budget", "not_found",
            "invalid_deliverable"} <= seen
    assert by_tool == {"plan", "scan", "read_regions", "view", "profile", "scanner_info", "status", "calc", "submit"}
    assert n_errors > 0.5 * N_MUTATIONS
    times.sort()
    assert times[int(0.99 * len(times))] < 0.05 and len(slow) < 0.005 * N_MUTATIONS, slow[:5]    # a refusal is cheap


# ------------------------------------------------------------------------------------------------ (b) metamorphic
def _text(v):
    return json.dumps(v)


def spell(rng, name, args, env):
    """The same call in another safe spelling, and the rule it uses."""
    a = dict(args)
    rules = ["C1", "C1i", "C2", "C3", "C6", "C7", "C7t", "C7w"]
    if name in ("plan", "scan"):
        rules += ["C5", "C10"]
        if a.get("te_ms") == "min":
            rules.append("C4")
        if a["sequence"] in console.FIXED and "flip_deg" not in a:
            rules.append("C11")
        if name == "scan" and a.get("regions") == ["all"]:
            rules.append("C8")
        if name == "scan" and len(a.get("circles", [])) == 1:
            rules.append("C8c")
    if name in ("view", "read_regions", "profile"):
        rules.append("C10s")
    if name == "scanner_info" and "topic" in a:
        rules.append("C9")
    if name == "calc":
        rules.append("C12")
    if name not in ("plan", "scan"):
        rules = [r for r in rules if r not in ("C7", "C7t")]
    rule = rng.choice(rules)
    if rule == "C1":                                                    # every typed value as JSON text (a free text stays as it is)
        a = {k: v if k == "expressions" else _text(v) for k, v in a.items()}
    elif rule == "C1i":
        a = {k: float(v) if isinstance(v, int) and not isinstance(v, bool) else v for k, v in a.items()}
    elif rule == "C2":
        a = {k: str(v) if isinstance(v, (bool, list, dict)) else v for k, v in a.items()}
    elif rule == "C3":
        spec = next(t for t in env._tool_specs() if t.name == name)
        optional = [k for k, s in spec.parameters["properties"].items() if k not in a and k not in spec.parameters.get("required", [])
                    and not (s.get("type") == "string" and "enum" not in s)
                    and (name not in ("plan", "scan") or k in console.SEQUENCES[a["sequence"]] or k in ("regions", "circles"))]
        for k in rng.sample(optional, min(len(optional), 3)):
            a[k] = rng.choice([None, "", "default", "auto"])
    elif rule == "C4":
        a["te_ms"] = rng.choice(["minimum", "Shortest", " console minimum ", "MIN"])
    elif rule == "C5":
        for k, unit in (("tr_ms", "ms"), ("fov_mm", " mm"), ("flip_deg", " deg"), ("bandwidth_hz_per_px", " Hz/px"), ("ti_ms", "ms"), ("slice_mm", "mm"),
                        ("rotation_deg", " deg"), ("fatsat_offset_hz", " Hz")):
            if k in a:
                a[k] = f"{a[k]}{unit}"
    elif rule == "C6":
        a = {k: rng.choice([v.upper(), f" {v} ", f'"{v}"']) if isinstance(v, str) and k in ("sequence", "pe_dir", "kind", "axis", "topic", "reduce", "units",
                                                                                                "rf_pulse", "excitation") else v
             for k, v in a.items()}
    elif rule == "C7":
        a = {"protocol": {k: v for k, v in a.items() if k not in ("regions", "circles")}, **{k: v for k, v in a.items() if k in ("regions", "circles")}}
    elif rule == "C7t":
        a = {"protocol": _text(a)} if name == "plan" else {"protocol": _text({k: v for k, v in a.items() if k not in ("regions", "circles")}),
                                                           **{k: v for k, v in a.items() if k in ("regions", "circles")}}
    elif rule == "C7w":
        a = {rng.choice(I.WRAPPERS): a}
    elif rule == "C8":
        a["regions"] = rng.choice(["task", "all", ["task"], '["task"]', "ALL"])
    elif rule == "C8c":
        a["circles"] = a["circles"][0]
    elif rule == "C9":
        a = {rng.choice(["section", a["topic"], "name"]): a["topic"]}
    elif rule == "C10":
        for alias, k in console.ALIASES.items():
            if k in a:
                a[alias] = a.pop(k)
    elif rule == "C10s":
        a = {{"scan_index": "scan", "image_index": "image"}.get(k, k): v for k, v in a.items()}
    elif rule == "C11":
        a["flip_deg"] = rng.choice([90, 90.0, "90"])
    elif rule == "C12":
        a["expressions"] = a["expressions"].split("\n")
    return a, rule


def test_fuzz_every_safe_spelling_is_the_same_call():
    rng = random.Random(0)
    used = set()
    for env_id in (NOWRAP, T1):
        plain, spelt = fresh(env_id), fresh(env_id)
        for _ in range(250):
            name, args = valid_call(rng, plain)
            if name in ("submit", "scan"):                               # scans are compared in their canonical form only (below)
                name, args = "plan", protocol(rng)
            other, rule = spell(rng, name, args, plain)
            used.add(rule)
            a, b = plain.admit({"name": name, "arguments": args}), spelt.admit({"name": name, "arguments": other})
            assert json.dumps(a[1]) == json.dumps(b[1]), (rule, args, other)                   # byte-equal canonical arguments
            assert [p["type"] for p in a[3]] == [p["type"] for p in b[3]] == [], (rule, other, b[3])
            x, y = plain.step({"name": name, "arguments": args}), spelt.step({"name": name, "arguments": other})
            assert x.observation.text == y.observation.text and x.observation.is_error == y.observation.is_error, (rule, other)
            assert plain._session.chain == spelt._session.chain, rule                          # the same head of the scanner's log
            assert plain.transcript()[-1]["call"] == spelt.transcript()[-1]["call"]
        for _ in range(60):                                              # scans: regions and circles in every spelling
            a = priced_out(rng)
            if plain._regions:
                a["regions"] = ["all"]
            if rng.random() < 0.5:
                a["circles"] = [{"y_mm": 0, "x_mm": 10, "radius_mm": 6}]
            other, rule = spell(rng, "scan", a, plain)
            used.add(rule)
            assert json.dumps(plain.admit({"name": "scan", "arguments": a})[1]) == json.dumps(spelt.admit({"name": "scan", "arguments": other})[1]), (rule, other)
    assert used >= {"C1", "C1i", "C2", "C3", "C4", "C5", "C6", "C7", "C7t", "C7w", "C8", "C8c", "C9", "C10", "C10s", "C11", "C12"}, used


# ------------------------------------------------------------------------------------------------ (e) wire formats
def test_fuzz_the_wire_formats_carry_one_call():
    rng = random.Random(1)
    env = fresh(T1)
    for k in range(200):
        name, args = valid_call(rng, env)
        shapes = [{"name": name, "arguments": args},
                  {"id": f"call_{k}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}},
                  {"jsonrpc": "2.0", "id": k, "method": "tools/call", "params": {"name": name, "arguments": args}},
                  {"type": "tool_use", "id": f"toolu_{k}", "name": name, "input": args}]
        seen = {json.dumps(env.admit(s)[:2]) for s in shapes}
        assert len(seen) == 1 and not env.admit(shapes[1])[3], (name, args)


@pytest.mark.parametrize("env_id", [NOWRAP, T1])
def test_fuzz_a_replayed_transcript_reads_the_same(env_id):
    """Whatever was sent, the calls the transcript kept reproduce the episode: text for text, and the same log head."""
    rng = random.Random(2)
    env = fresh(env_id)
    for _ in range(300):
        if env.step(mutate(rng, *valid_call(rng, env))).done:
            break
    again = fresh(env_id)
    for e in env.transcript()[2:]:
        again.step(e["call"] if "raw" not in e["call"] else e["call"]["raw"])
    skip = lambda t: "<omitted" in json.dumps(t["call"])                 # noqa: E731  a call too large to keep is kept as a note
    assert [e["text"] for e in env.transcript() if not skip(e)] == [e["text"] for e in again.transcript() if not skip(e)]
    assert env._session.chain == again._session.chain
