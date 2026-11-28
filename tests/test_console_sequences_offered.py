"""The tool layer on a console whose card lists more sequences (world spec ``console_sequences``).

A family says which of ``fse``, ``mese``, ``megre``, ``dwse`` its console offers. The schemas of ``plan`` and ``scan``,
the topics of ``scanner_info``, the refusals and the answers then know those sequences with their parameters, units
and ranges, and nothing else changes: a console without the key has the tool layer of before (the id fingerprints,
the golden snapshots and ``tests/test_compat_2d_unchanged.py`` hold that byte for byte; here the functions that were
given a table are held to the table of the six).
"""
import json

import pytest

import mrgym
from mrgym import interface as I
from mrgym import tools as T
from mrgym.envs import scanner_env
from mrgym.envs.t2_echoes import task as t2_echoes
from mrgym.scanner import console

MESE, CLASSIC = "Scanner/T2Echoes-v0", "Scanner/T2Decay-v0"
CARD = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5)


def make(env_id, level=2, seed=0, **kw):
    env = mrgym.make(env_id, level=level, seed=seed, **kw)
    env.reset()
    return env


def call(env, name, **arguments):
    return env.step({"name": name, "arguments": arguments}).observation


def body(obs):
    return "\n".join(obs.text.split("\n")[:-1])


def _listing(monkeypatch, sequences, **spec):
    """An environment like level 2 of T2Echoes whose console lists ``sequences`` (the one shipped family lists `mese`)."""
    world = {k: v for k, v in dict(t2_echoes.LEVELS[2]["world_spec"], console_sequences=sequences, **spec).items() if v is not None}
    monkeypatch.setitem(t2_echoes.LEVELS, 2, dict(t2_echoes.LEVELS[2], world_spec=world))
    return make(MESE, max_rejected_calls=100)


@pytest.fixture()
def every_sequence(monkeypatch):
    env = _listing(monkeypatch, "all")
    yield env
    env.close()


@pytest.fixture()
def diffusion_console(monkeypatch):
    env = _listing(monkeypatch, ["dwse"], console_system=None)           # the default console: 3 s of adjustments
    yield env
    env.close()


# ------------------------------------------------------------------------------------------------ a console of before
def test_a_card_without_the_key_has_the_tables_of_the_six():
    assert console.offered(CARD) is console.SEQUENCES and scanner_env.topics(CARD) == scanner_env.TOPICS
    assert scanner_env.sequence_topics(None) == scanner_env.SEQUENCE_TOPICS == tuple(console.SEQUENCES)
    assert set(scanner_env.SEQUENCE_EXAMPLES) == set(console.SEQUENCES) and set(scanner_env.SEQUENCE_EXAMPLES_EXTENDED) == set(console.EXTENDED)
    props = scanner_env.protocol_properties(CARD)
    assert list(props) == ["sequence"] + console.parameter_names() and props["sequence"]["enum"] == sorted(console.SEQUENCES)
    assert I._table(props) is console.SEQUENCES and I._table({}) is console.SEQUENCES
    assert not [k for k in props if k in ("etl", "echo_order", "refocus_deg", "readout", "b_value", "diffusion_dir")]
    env = make(CLASSIC)
    tools = {t["name"]: t for t in env.tools("neutral")}
    assert tools["scanner_info"]["parameters"]["properties"]["topic"]["enum"] == list(scanner_env.TOPICS)
    assert "afi: 2, spgr: one per echo" in tools["view"]["parameters"]["properties"]["image_index"]["description"]
    err = call(env, "plan", sequence="mese", tr_ms=1000).data["error"]   # not on this console: said with what it has
    assert (err["type"], err["parameter"]) == ("invalid_value", "sequence") and err["allowed"]["values"] == sorted(console.SEQUENCES)
    err = call(env, "plan", sequence="se", tr_ms=1000, b_value=500).data["error"]
    assert (err["type"], err["code"]) == ("unknown_parameter", "NOT_SIMULATED") and "no diffusion-weighted sequence" in err["message"]
    assert "mese" not in body(call(env, "scanner_info")) and "dwse" not in body(call(env, "scanner_info", topic="cost"))


# ------------------------------------------------------------------------------------------------ a console that lists one
def test_the_tools_of_a_console_that_lists_mese():
    env = make(MESE, max_rejected_calls=100)
    specs = {t.name: t for t in env._tool_specs()}
    assert not [m for t in specs.values() for m in T.check_tool(t)]
    plan, scan = specs["plan"].parameters["properties"], specs["scan"].parameters["properties"]
    assert plan["sequence"]["enum"] == sorted([*console.SEQUENCES, "mese"]) and "mese = multi-echo spin echo" in plan["sequence"]["description"]
    assert set(scan) - {"regions", "circles"} == set(plan) - {"verbose"}
    assert '"fse"' not in json.dumps(env.tools("openai")) and "b_value" not in plan and "etl" not in plan     # only what the card lists
    assert plan["refocus_deg"]["description"] == "nominal refocusing flip angle, 60-180; only mese; default 180"
    assert plan["n_echoes"]["description"] == "2-32 (spgr: 1-8); only mese, spgr; default 8 (spgr: 1)"
    assert "spgr, mese: one per echo" in specs["view"].parameters["properties"]["image_index"]["description"]
    topics = specs["scanner_info"].parameters["properties"]["topic"]["enum"]
    assert topics == list(scanner_env.topics(env._session.exam)) and "mese" in topics and "fse" not in topics and "dwse" not in topics
    # the manual: the paragraph of the supplement, the parameters with ranges and defaults, a call that the console accepts
    text = body(call(env, "scanner_info", topic="mese"))
    assert text.startswith("`mese`: multi-echo spin echo. The train of `fse`: a 90 degree excitation and `n_echoes` refocusing pulses")
    assert "described above" not in text and "A single exponential\nfitted to all echoes then overestimates T2" in text    # it stands on its own
    assert "  n_echoes: 2 to 32; default 8" in text and "  refocus_deg: 60 to 180; default 180" in text and '  echo_spacing_ms: 2 to 100; default "min"' in text
    example = scanner_env.SEQUENCE_EXAMPLES_EXTENDED["mese"]
    assert f"example: plan {json.dumps(example, separators=(', ', ': '))}" in text and not call(env, "plan", **example).is_error
    overview = body(call(env, "scanner_info"))
    assert "spgr | afi | se | ir | bssfp | epi | mese " in overview and "1 cycle (se, ir, mese)" in overview
    cost = body(call(env, "scanner_info", topic="cost"))
    assert "- `mese`, `dwse`, `satrec`: overhead + one dummy repetition + `tr_ms` x lines x averages." in cost and "- RF power: every pulse of the repetition counts." in cost
    assert "- `dwse`: the diffusion gradients" not in cost               # a bullet is served when it names a listed sequence
    # this console charges its own adjustment time: the cost topic says so next to the datasheet's default, as the card does
    assert "- Overhead of this console: 40 s per scan (the adjustments before every scan); the 3 s stated above are those of the default console." in cost
    assert "overhead 40 s" in overview and env._session.info()["cost"]["overhead_s_per_scan"] == 40.0
    hidden = body(call(env, "scanner_info", topic="hidden"))
    assert "Not simulated in the further sequences of this console (mese): refocusing angles that change along the train" in hidden
    assert "The transverse magnetisation that is left at the end of a repetition does not return" in hidden
    assert "The repetition does not start from uniform longitudinal magnetisation." in text    # what the trains share is said under mese too
    one = body(call(env, "scanner_info", parameter="n_echoes"))
    assert "  spgr: range 1 to 8, default 1" in one and "  mese: range 2 to 32, default 8" in one
    assert body(call(env, "scanner_info", parameter="refocus_deg")).startswith("refocus_deg: degrees (nominal), flip angle of every refocusing pulse")


def test_plan_scan_and_refusals_of_mese():
    env = make(MESE, max_rejected_calls=100)
    text = body(call(env, "plan", sequence="mese", tr_ms=1000, n_echoes=32, matrix=48))
    assert "te_ms 8.36, 16.72, 25.08, ... 267.52 (32 echoes, 8.36 apart)" in text and "minimums: echo_spacing_ms 8.36, te_ms 8.36, tr_ms 271.24" in text
    short = body(call(env, "plan", sequence="mese", tr_ms=1000, n_echoes=4, matrix=48))
    assert "te_ms 8.36, 16.72, 25.08, 33.44 " in short                   # up to eight echoes: every one
    for arguments, parameter, fix, words in (
            (dict(n_echoes=40), "n_echoes", {"n_echoes": 32}, "40 is above the maximum 32 (a limit of the simulation, not of a scanner)"),
            (dict(matrix=300), "matrix", {"matrix": 256}, "300 is outside 32 to 256 for mese (a limit of the simulation, not of a scanner)"),
            (dict(refocus_deg=40), "refocus_deg", {"refocus_deg": 60.0}, "40 is below the minimum 60"),
            (dict(echo_spacing_ms=3), "echo_spacing_ms", {"echo_spacing_ms": 8.4}, "minimum echo spacing is 8.4 ms"),
            (dict(n_echoes=32, tr_ms=200), "tr_ms", {"tr_ms": 272.54}, "below the minimum 272.54 ms")):
        for tool in ("plan", "scan"):
            err = call(env, tool, **dict(dict(sequence="mese", tr_ms=1000), **arguments)).data["error"]
            assert (err["parameter"], err["fix"]) == (parameter, fix) and words in err["message"], (tool, arguments, err)
            assert err["class"] == "task"                                # the instrument saying no: a tool call, no scan time
    err = call(env, "plan", sequence="mese", tr_ms=1000, etl=8).data["error"]         # a word of a sequence this console does not list
    assert (err["type"], err["class"], err["fix"]) == ("unknown_parameter", "vocabulary", {"etl": None})
    err = call(env, "plan", sequence="mese", tr_ms=1000, flip_deg=90).data["error"]
    assert (err["type"], err["fix"]) == ("not_applicable", {"flip_deg": None}) and "mese has no flip_deg (it belongs to afi, bssfp, epi, spgr)" in err["message"]
    err = call(env, "plan", sequence="se", tr_ms=1000, n_echoes=4).data["error"]
    assert err["type"] == "not_applicable" and "se has no n_echoes (it belongs to mese, spgr)" in err["message"]
    assert env.state()["scan_time_used_s"] == 0.0
    scan = call(env, "scan", sequence="mese", tr_ms=300, n_echoes=3, matrix=32)
    assert not scan.is_error and "3 images 32x32 (y, x) stored" in scan.text and "te_ms 8.35, 16.7, 25.05" in scan.text
    assert len(scan.data["roi_readouts"][0]["complex_mean_magnitude"]) == 3 and "MESE 1.5T" in body(call(env, "status"))


def test_the_tools_of_a_console_that_lists_dwse(diffusion_console):
    env = diffusion_console
    plan = {t.name: t for t in env._tool_specs()}["plan"].parameters["properties"]
    assert plan["b_value"]["description"] == "diffusion weighting, s/mm^2, 0-3000; only dwse" and "mese" not in plan["sequence"]["enum"]
    assert plan["diffusion_dir"]["enum"] == ["x", "y", "z"]
    text = body(call(env, "scanner_info", topic="dwse"))
    assert text.startswith("`dwse`: diffusion-weighted spin echo.") and "at the SAME echo time" in text and "  b_value: 0 to 3000; required" in text
    cost = body(call(env, "scanner_info", topic="cost"))
    assert "- `dwse`: the diffusion gradients are two equal trapezoids" in cost and "Overhead of this console" not in cost    # the default console
    weighted = call(env, "plan", sequence="dwse", tr_ms=800, matrix=64, b_value=1000)
    assert "te_ms 58.56" in weighted.text and "realised" not in weighted.text and weighted.data["realised"]["b_value"] == pytest.approx(1000.0, rel=1e-6)
    raised = call(env, "plan", sequence="dwse", tr_ms=800, matrix=64, b_value=1000, te_ms=20)
    assert "note: te_ms 20 -> 58.56 ms" in raised.text                   # the one adjustment the interface makes, here too
    same = call(env, "plan", sequence="dwse", tr_ms=800, matrix=64, b_value=0, te_ms=58.56)
    assert "te_ms 58.56" in same.text and "note" not in same.text
    err = call(env, "plan", sequence="dwse", tr_ms=800, b_value=4000).data["error"]
    assert (err["parameter"], err["fix"]) == ("b_value", {"b_value": 3000.0})
    err = call(env, "plan", sequence="dwse", tr_ms=800).data["error"]
    assert (err["type"], err["parameter"]) == ("invalid_value", "b_value") and "dwse needs b_value [s/mm^2" in err["message"]


# ------------------------------------------------------------------------------------------------ all four
def test_every_paragraph_of_the_supplement_is_behind_a_topic(every_sequence):
    env, md, parts = every_sequence, scanner_env.datasheet_core(), scanner_env.datasheet_core_parts()
    assert set(parts) == {"cost", "hidden", *console.EXTENDED}
    cut = lambda a, b: md[md.index(a):md.index(b) if b else len(md)].split("\n", 1)[1]              # noqa: E731
    served = "\n\n".join([*(parts[q] for q in console.EXTENDED), "\n".join(f"- {b}" for b in parts["cost"]), parts["hidden"]])
    for a, b in (("## 2.", "## 3."), ("## 3.", "## 4."), ("## 4.", None)):                           # sections 2 to 4: nothing in no piece
        for para in [p.strip() for p in cut(a, b).split("\n\n") if p.strip()]:
            assert para.replace("**", "") in served, para[:80]
    topics = scanner_env.topics(env._session.exam)
    assert set(topics[4:4 + len(console.SEQUENCES) + len(console.EXTENDED)]) == {*console.SEQUENCES, *console.EXTENDED} and topics[4:10] == tuple(console.SEQUENCES)
    texts = {t: call(env, "scanner_info", topic=t) for t in topics}
    assert not [t for t, o in texts.items() if o.is_error or len(body(o)) < 100]
    for seq in console.EXTENDED:
        text = body(texts[seq])
        assert parts[seq] in text and texts[seq].data == {seq: env._session.info()["sequences"][seq]}
        example = scanner_env.SEQUENCE_EXAMPLES_EXTENDED[seq]
        assert json.dumps(example, separators=(", ", ": ")) in text and not call(env, "plan", **example).is_error, seq
        for name in console.EXTENDED[seq]:                                                          # every parameter of the sequence: in its own list or in the shared one
            assert f"  {name}: " in text or f"{name} " in text.split("taken by every sequence (default): ")[1], (seq, name)
    cost = body(texts["cost"])
    assert all(f"- {b}" in cost for b in parts["cost"]) and "megre" in body(texts["overview"])
    geometry = body(texts["geometry"])
    assert "(epi: 32 to 192, default 96; fse: 32 to 256, default 128; mese: 32 to 256, default 128)" in geometry


def test_the_answers_of_fse_say_what_the_request_does_not(every_sequence):
    env = every_sequence
    text = body(call(env, "plan", sequence="fse", tr_ms=3000, etl=16, te_ms=80, echo_order="linear", matrix=128))
    assert "realised (on the console's raster): te_ms 80 -> 84" in text                             # realised on an echo
    assert "te_ms 84 (effective: echo 10 of 16, echo_spacing_ms 8.4), 8 repetitions per average" in text
    err = call(env, "plan", sequence="fse", tr_ms=3000, etl=16, te_ms=80, matrix=128).data["error"]
    assert err["fix"] == {"te_ms": 8.4} and "with echo_order=centric the first echo fills the centre of k-space" in err["message"]
    err = call(env, "plan", sequence="fse", tr_ms=3000, etl=40).data["error"]
    assert err["fix"] == {"etl": 32} and "a limit of the simulation" in err["message"]
    err = call(env, "plan", sequence="fse", tr_ms=100, etl=32, matrix=128).data["error"]
    assert (err["type"], err["code"], err["parameter"]) == ("console_limit", "TR_BELOW_MINIMUM", "tr_ms")
    hot = call(env, "plan", sequence="fse", tr_ms=328, etl=32, matrix=128, tx_scale=2.0).data["error"]
    assert hot["code"] == "RF_POWER_LIMIT" and hot["fix"]["tr_ms"] > 328                             # the train's RF power, refused at the limit
    cool = call(env, "plan", sequence="fse", tr_ms=328, etl=32, matrix=128, tx_scale=2.0, refocus_deg=150)
    assert not cool.is_error and cool.data["limits"]["rf_power_percent"] < 75.0                       # lower refocusing angles: the square of the angle
    megre = call(env, "plan", sequence="megre", tr_ms=60, flip_deg=20, n_echoes=6, matrix=96, readout="bipolar")
    assert not megre.is_error and megre.text.count("te_ms") >= 2
    err = call(env, "plan", sequence="megre", tr_ms=60, flip_deg=20, n_echoes=20).data["error"]
    assert err["fix"] == {"n_echoes": 16}
