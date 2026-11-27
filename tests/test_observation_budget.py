"""What an episode costs in text: the tool definitions, the task text and every answer say what a decision needs and
nothing twice. ``Observation.data`` keeps everything at full precision; only ``Observation.text`` is short.

    datasheet   served in pieces by ``scanner_info``; no paragraph of it is out of reach
    plan, scan  what the request does not say; the rest behind ``verbose`` and ``status``
    status      one short line: the call count, and what changed
    read-outs   the latest scan by default; a circle with a marked region's id; a profile up to the edge of the image
    canonical   read-outs on ``plan``, a nested deliverable written flat
    sizes       ceilings per environment id, and the accounting script on a recorded conversation
"""
import importlib.util
import json
import os

import pytest

import mrgym
from mrgym import interface as I
from mrgym.agents import runner
from mrgym.envs import scanner_env
from mrgym.scanner import console

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOWRAP, T1, NULL, QA, SNR, CLINIC = ("Scanner/NoWrap-v1", "Scanner/T1Transmit-v1", "Scanner/NullTissue-v0", "Scanner/PhantomQA-v0",
                                    "Scanner/SNRSpec-v0", "Scanner/ArtefactClinic-v0")
LOCALISER = {"sequence": "spgr", "tr_ms": 12, "flip_deg": 10, "fov_mm": 288, "matrix": 96}
IR = {"sequence": "ir", "tr_ms": 800, "ti_ms": 100, "matrix": 64, "partial_fourier": 0.6}


def make(env_id=NOWRAP, level=1, seed=0, **kw):
    env = mrgym.make(env_id, level=level, seed=seed, **kw)
    env.reset()
    return env


def call(env, name, **arguments):
    return env.step({"name": name, "arguments": arguments}).observation


def body(obs):
    """The answer without its status line."""
    lines = obs.text.split("\n")
    assert lines[-1].startswith("[") and lines[-1].endswith("]"), obs.text
    return "\n".join(lines[:-1])


# ------------------------------------------------------------------------------------------------ the datasheet, in pieces
def test_datasheet_every_paragraph_is_behind_a_topic():
    md, parts = scanner_env.datasheet(), scanner_env.datasheet_parts()
    assert set(parts) == {"cost", "outputs", "hidden", "notation", "encoding", "slice_profile", "gradient_echo", "spin_echo", "fat", "motion"} | set(console.SEQUENCES)
    served = "\n\n".join(parts.values())
    for a, b in (("## 4.", "## 5."), ("## 5.", "## 6."), ("## 6.", "## 7."), ("## 7.", None)):          # sections 4 to 7: nothing in no piece
        section = md[md.index(a):md.index(b) if b else len(md)].split("\n", 1)[1]
        for para in [p.strip() for p in section.split("\n\n") if p.strip()]:
            assert para.replace("**", "") in served, para[:80]
    env = make(T1)
    texts = {topic: call(env, "scanner_info", topic=topic) for topic in scanner_env.TOPICS}
    assert not [t for t, o in texts.items() if o.is_error or len(body(o)) < 100]
    index = body(texts["physics"])
    assert len(index) < 700 and all(t in index for t in scanner_env.PHYSICS_TOPICS + scanner_env.SEQUENCE_TOPICS) and parts["notation"] in index
    assert all(t in body(texts["overview"]) for t in ("geometry", "cost", "outputs", "physics", "task", "presets", "parameter="))
    for topic in scanner_env.PHYSICS_TOPICS + ("cost", "outputs"):
        assert body(texts[topic]) == parts[topic] and texts[topic].data == {"markdown": parts[topic]}
    for seq in console.SEQUENCES:                                         # a sequence: its signal model, the echo it reads, its parameters, a call
        text = body(texts[seq])
        assert parts[seq] in text and (seq == "bssfp" or parts["spin_echo" if seq in ("se", "ir") else "gradient_echo"] in text)
        assert json.dumps(scanner_env.SEQUENCE_EXAMPLES[seq], separators=(", ", ": ")) in text and len(text) < 2400
        assert "tr_ms: " in text and "; required" in text and "taken by every sequence (default): fov_mm " in text
        assert texts[seq].data == {seq: env._session.info()["sequences"][seq]}
    assert set(texts["overview"].data) >= {"task", "cost", "sequences", "default_fov_mm", "hardware"}       # the whole card: what a scripted policy reads
    # the six tables at once are gone; ``sequences`` names them
    assert len(body(texts["sequences"])) < 1100 and all(f"\n{s}: " in "\n" + body(texts["sequences"]) for s in console.SEQUENCES)


def test_datasheet_one_parameter_at_a_time():
    env = make(NULL)
    matrix = call(env, "scanner_info", parameter="matrix")
    assert body(matrix).split("\n") == ["matrix: pixels along the readout (even); phase-encode lines follow from phase_fov",
                                        "  spgr, afi, se, ir, bssfp: range 32 to 512, default 128", "  epi: range 32 to 192, default 96"]
    assert set(matrix.data["parameter"]["matrix"]) == set(console.SEQUENCES)
    flip = body(call(env, "scanner_info", parameter="flip_angle"))       # an alias of the name is read as the name
    assert flip.startswith("flip_deg: ") and "se: fixed at 90; ir: fixed at 90" in flip and "default required" in flip
    both = body(call(env, "scanner_info", topic="ir", parameter="ti_ms"))
    assert both.startswith("ti_ms: ") and "not a parameter of spgr, afi, se, bssfp, epi" in both and "inversion recovery" in both
    wrong = call(env, "scanner_info", parameter="phase_fow")
    err = wrong.data["error"]
    assert wrong.is_error and (err["type"], err["class"], err["did_you_mean"], err["fix"]) == ("invalid_value", "vocabulary", "phase_fov", {"parameter": "phase_fov"})
    assert not call(env, "scanner_info", **err["fix"]).is_error
    # the one enum parameter under another key is still read (a tool with a second, free parameter)
    assert env.admit({"name": "scanner_info", "arguments": {"task": "task"}})[1] == {"topic": "task"}


# ------------------------------------------------------------------------------------------------ plan, scan, status
def test_plan_and_scan_say_what_the_request_does_not():
    env = make(NULL)
    plan = call(env, "plan", **IR)
    lines = body(plan).split("\n")
    # the price is that of scanning the protocol now, with what is left after it (never "the budget": a plan is also how
    # a deliverable is looked at); the image as (y, x) with the phase-encode axis named; the minimums the console enforces
    # (ir refuses a tr_ms below 200, whatever the timing would allow): tests/test_interface_clarity.py
    assert lines[0] == f"plan ok · as a scan now: 35.00 s, leaves {env.state()['scan_time_left_s'] - 35:.2f} s" and len(lines) == 2
    assert lines[1].startswith("64x64 px (y, x), FOV 240 x 240 mm, phase-encode y: 39 lines · te_ms 8.37 · minimums: ti_ms 6.04, te_ms 8.37, tr_ms 200 · SAR ")
    assert " · WFS 0.7 px (readout) · rel. SNR 3.18" in lines[1] and "gradient" not in lines[1]            # a limit is named when it comes close
    assert not [w for w in ("realised:", "timing:", "IR 1.5T", "tr_ms 800", "ti_ms 100") if w in body(plan)]   # nothing the request said
    verbose = call(env, "plan", verbose=True, **IR)
    assert body(verbose).startswith(body(plan)) and plan.data == verbose.data                           # data: everything, always
    extra = body(verbose)[len(body(plan)):]
    assert plan.data["header"] in extra and "\nrealised: sequence=ir " in extra and "\ntiming: " in extra and "cost: overhead 3 + " in extra
    assert body(call(env, "plan", **dict(IR, tr_ms=8000))).startswith(f"plan ok · as a scan now: needs 323.00 s, {env.state()['scan_time_left_s']:.2f} s left\n")
    near = body(call(env, "plan", sequence="spgr", tr_ms=12, flip_deg=10, matrix=256, bandwidth_hz_per_px=1300))
    assert "gradient at " in near and "% of its limit" in near
    # what the console changes against the request is said: the raster, and the one adjustment
    moved = body(call(env, "plan", **dict(IR, tr_ms=800.004, te_ms=2)))
    assert "note: te_ms 2 -> 8.37 ms (console minimum" in moved and "realised (on the console's raster): tr_ms 800.004 -> 800.01" in moved
    scan = call(env, "scan", **IR)
    head, table = body(scan).split("\n", 1)
    assert head == "scan_index 0 · 1 image 64x64 (y, x) stored · cost 35.00 s · te_ms 8.37" and scan.data["header"] not in scan.text
    assert table.split("\n")[0] == "region n_px |complex_mean| phase_rad mean|S| sd|S|" and len(table.split("\n")) == 1 + len(env._regions)
    row, data = table.split("\n")[1].split(" "), scan.data["roi_readouts"][0]
    assert row[0] == data["id"] and int(row[1]) == data["n_pixels"] and float(row[2]) == pytest.approx(data["complex_mean_magnitude"][0], rel=1e-4)
    assert float(row[5]) == pytest.approx(data["sd_magnitude"][0], rel=5e-3) and data["complex_mean_magnitude"][0] != float(row[2])     # data: full precision
    assert scan.data["header"] in body(call(env, "status"))                                             # the header line of a stored scan: status
    echoes = make(T1)                                                     # several images: the image column and every echo time
    multi = call(echoes, "scan", sequence="spgr", tr_ms=30, flip_deg=10, matrix=64, n_echoes=3)
    assert body(multi).split("\n")[0].endswith(" · te_ms " + ", ".join(f"{t:g}" for t in multi.data["realised"]["echo_times_ms"]))
    assert body(multi).split("\n")[1] == "region n_px image |complex_mean| phase_rad mean|S| sd|S|"


def test_status_line_is_the_call_count_and_what_changed():
    env = mrgym.make(NULL, level=1, seed=0)
    obs, info = env.reset()
    budget = info["limits"]["scan_time_budget_s"]
    assert obs.text.split("\n")[-1] == f"[scan time left {budget:.2f} of {budget:g} s · calls 0/48 · rejected 0/8]"
    line = lambda o: o.text.split("\n")[-1]                               # noqa: E731
    assert line(call(env, "plan", **IR)) == "[calls 1/48]" and line(call(env, "calc", expressions="1 + 1")) == "[calls 2/48]"
    scan = call(env, "scan", **IR)
    assert line(scan) == f"[scan time left {budget - 35:.2f} of {budget:g} s · calls 3/48]"
    refused = call(env, "plan", sequence="ir", tr_ms=800, ti_ms=100, flipp=3)
    assert line(refused) == "[calls 3/48 · rejected 1/8 (nothing charged)]" and "charged:" not in refused.text
    assert refused.data["charged"] == dict(scan_time_s=0.0, tool_calls=0, rejected_calls=1) and refused.data["episode_continues"]
    over = call(env, "scan", **dict(IR, tr_ms=8000))                      # a refusal of the instrument counts as a call and costs no scan time
    assert over.data["error"]["type"] == "budget" and line(over) == "[calls 4/48]" and over.data["charged"]["scan_time_s"] == 0
    assert line(call(env, "view")) == "[calls 5/48]"
    full = call(env, "status")                                            # asked for: everything, and the line under it too
    assert line(full) == f"[scan time left {budget - 35:.2f} of {budget:g} s · calls 6/48 · rejected 1/8]"
    assert "rejected calls: 1 of 8; refused submit calls: 0 of 3" in full.text and "stored scans: 1" in full.text
    for o in (scan, refused, over, full):                                 # the status of an observation holds every number, whatever the line prints
        assert {"scan_time_left_s", "scan_time_budget_s", "tool_calls", "max_tool_calls", "rejected_calls", "max_rejected_calls"} <= set(o.status)
    short = mrgym.make(NULL, level=1, seed=0, max_steps=5)
    short.reset()
    lines = [line(call(short, "calc", expressions="1")) for _ in range(4)]
    assert lines == ["[calls 1/5]", "[calls 2/5 · submit now or the episode scores 0]", "[calls 3/5 · submit now or the episode scores 0]",
                     "[calls 4/5 · submit now or the episode scores 0]"]


# ------------------------------------------------------------------------------------------------ read-outs
def test_readouts_the_latest_scan_a_circle_with_a_region_id_and_a_full_call():
    env = make(NULL)
    ids = [r["id"] for r in env._regions]
    assert call(env, "view").data["error"]["type"] == "not_found"         # nothing stored yet
    call(env, "scan", **IR)
    call(env, "scan", **dict(IR, ti_ms=300))
    assert call(env, "view").data["scan_index"] == 1 and call(env, "profile", axis="y").data["scan_index"] == 1
    assert call(env, "read_regions").data["scan_index"] == 1 and call(env, "read_regions", scan_index=0).data["scan_index"] == 0
    # a circle that carries the id of a marked region is read in its place, when nothing was said about the regions
    own = dict(id=ids[0], y_mm=0, x_mm=0, radius_mm=8)
    got = call(env, "read_regions", circles=[own])
    assert [r["id"] for r in got.data["roi_readouts"]] == ids[1:] + [ids[0]]
    marked = next(r for r in call(env, "read_regions").data["roi_readouts"] if r["id"] == ids[0])
    assert got.data["roi_readouts"][-1]["n_pixels"] != marked["n_pixels"]                               # the circle of the call, not the marked one
    clash = call(env, "read_regions", regions=[ids[0]], circles=[own])    # the region and a circle of its name, both asked for
    assert clash.is_error and "two read-outs carry the id" in clash.text
    # circles that fill the call alone: the marked regions give way; more circles than fit are refused
    many = [dict(id=f"c{k}", y_mm=-60 + 8 * k, x_mm=0, radius_mm=3) for k in range(scanner_env.session.MAX_ROIS)]
    assert [r["id"] for r in call(env, "read_regions", circles=many).data["roi_readouts"]] == [c["id"] for c in many]
    assert [r["id"] for r in call(env, "read_regions", circles=many[:4]).data["roi_readouts"]] == ids + [c["id"] for c in many[:4]]
    assert call(env, "read_regions", circles=many + [dict(y_mm=0, x_mm=0, radius_mm=3)]).is_error
    assert call(env, "read_regions", regions=["all"], circles=many).is_error


def test_profile_reads_up_to_the_edge_and_takes_the_ends_in_either_order():
    env = make(NOWRAP)
    call(env, "scan", **LOCALISER)
    whole = call(env, "profile", axis="x", samples=16)
    lo, hi = whole.data["start_mm"], whole.data["stop_mm"]
    beyond = call(env, "profile", axis="x", start_mm=-900, stop_mm=900, samples=16)
    assert not beyond.is_error and beyond.data["values"] == whole.data["values"] and (beyond.data["start_mm"], beyond.data["stop_mm"]) == (lo, hi)
    assert [x["parameter"] for x in beyond.data["adjusted"]] == ["start_mm", "stop_mm"] and "adjusted" not in whole.data
    assert f"note: start_mm -900 -> {lo:g} mm (the edge of the image)" in beyond.text and f"note: stop_mm 900 -> {hi:g} mm" in beyond.text
    part = call(env, "profile", axis="x", start_mm=20, stop_mm=900)
    assert part.data["start_mm"] == pytest.approx(20, abs=3) and part.data["stop_mm"] == hi and len(part.data["adjusted"]) == 1
    swapped = call(env, "profile", axis="y", start_mm=30, stop_mm=-30)
    assert not swapped.is_error and swapped.data["values"] == call(env, "profile", axis="y", start_mm=-30, stop_mm=30).data["values"]
    for outside in (dict(start_mm=900), dict(stop_mm=-900), dict(start_mm=400, stop_mm=900), dict(position_mm=900)):     # nothing of the image in it
        refused = call(env, "profile", axis="x", **outside)
        assert refused.is_error and refused.data["error"]["type"] == "invalid_value" and refused.data["error"]["class"] == "task", outside
    # the table: one row per sample, position and value, one resolution for all rows; data keeps every digit
    rows = [r.split(" ") for r in body(whole).split("\n")[2:]]
    assert "; rows: x_mm value (centre of the sample; image units)" in body(whole).split("\n")[1] and len(rows) == 16 and all(len(r) == 2 for r in rows)
    assert [float(r[0]) for r in rows] == pytest.approx(whole.data["positions_mm"], abs=0.051)       # a tenth of a millimetre for samples of 18 mm
    assert len({len(r[1].split(".")[1]) for r in rows}) == 1 and [float(r[1]) for r in rows] == pytest.approx(whole.data["values"], abs=1e-3 * max(whole.data["values"]))
    view = call(env, "view")                                              # the picture: 16 blocks unless more are asked for
    # 700 (600 before): the y of every row and an x ruler, 150 characters on 16 rows (tests/test_interface_clarity.py)
    assert len(view.data["digit_map"]) == 16 and len(call(env, "view", size=40).data["digit_map"]) == 40 and len(view.text) < 700


# ------------------------------------------------------------------------------------------------ the canonical form
def test_canonical_readouts_on_plan_and_a_nested_deliverable_written_flat():
    env = make(NULL)
    name, args, coercions, problems = env.admit({"name": "plan", "arguments": dict(IR, regions='["R1", "R2"]', circles=[])})
    assert not problems and args == env.admit({"name": "plan", "arguments": IR})[1]
    assert [c["rule"] for c in coercions] == ["C14", "C14"] and not call(env, "plan", **dict(IR, regions="all")).is_error
    ids = [r["id"] for r in env._regions]
    flat = env.admit({"name": "submit", "arguments": {i: 400 + k for k, i in enumerate(ids)}})
    assert not flat[3] and flat[1] == {"ti_ms": {i: 400.0 + k for k, i in enumerate(ids)}} and {c["rule"] for c in flat[2]} == {"C15"}
    qa = make(QA)
    answer = dict(diameter_y_mm="189.5", diameter_x_mm="189", piu_percent="76", ghosting_ratio="0.0003", failed_tests=["intensity_uniformity"], faults=["b1_transmit"])
    got = qa.admit({"name": "submit", "arguments": answer})
    assert not got[3] and got[1]["measured"] == dict(diameter_y_mm=189.5, diameter_x_mm=189.0, piu_percent=76.0, ghosting_ratio=0.0003)
    mixed = qa.admit({"name": "submit", "arguments": dict(measured=dict(diameter_y_mm=190), diameter_x_mm=189, piu_percent=90, ghosting_ratio=0.01, failed_tests=[], faults=[])})
    assert not mixed[3] and set(mixed[1]["measured"]) == {"diameter_y_mm", "diameter_x_mm", "piu_percent", "ghosting_ratio"}
    twice = qa.admit({"name": "submit", "arguments": dict(measured=dict(diameter_y_mm=190, diameter_x_mm=189, piu_percent=90, ghosting_ratio=0.01),
                                                          diameter_y_mm=191, failed_tests=[], faults=[])})
    assert [p["type"] for p in twice[3]] == ["unknown_parameter"]         # a name that is inside already: not moved, not guessed
    snr = make(SNR)                                                       # a name that fits two objects is no one's
    refused = snr.admit({"name": "submit", "arguments": dict(tr_ms=500, averages=1, bandwidth_hz_per_px=300)})
    assert {p["type"] for p in refused[3]} >= {"unknown_parameter"} and not refused[2]


def test_ranges_are_stated_in_words_and_checked_for_plan_scan_and_a_delivered_protocol():
    env = make(NULL, max_rejected_calls=100)
    props = {t.name: t.parameters["properties"] for t in env._tool_specs()}
    for name, spec in console.SEQUENCES["spgr"].items():                  # every fixed range of the console is in the one line plan gives the parameter
        if spec[2] in "fi" and name not in console.PROTOCOL_DEPENDENT:
            lo, hi = min(s[name][0] for s in console.SEQUENCES.values() if name in s), max(s[name][1] for s in console.SEQUENCES.values() if name in s)
            assert (f"+-{hi:g}" if lo == -hi else f"{lo:g}-{hi:g}") in props["plan"][name]["description"], name
    for tool in ("plan", "scan"):
        for arguments, parameter, fix in ((dict(IR, averages=100), "averages", {"averages": 64}), (dict(IR, phase_fov=75), "phase_fov", {"phase_fov": 0.75}),
                                          (dict(IR, matrix=600), "matrix", {"matrix": 512}), (dict(IR, rotation_deg=-120), "rotation_deg", {"rotation_deg": -90.0})):
            err = call(env, tool, **arguments).data["error"]
            assert (err["type"], err["class"], err["parameter"], err["fix"]) == ("invalid_value", "task", parameter, fix), (tool, arguments)
    assert "75 % = 0.75" in call(env, "plan", **dict(IR, phase_fov=75)).text
    narrow = call(env, "plan", sequence="epi", tr_ms=2000, matrix=256).data["error"]                    # a range of one sequence: the console's
    assert (narrow["code"], narrow["fix"]) == ("OUT_OF_RANGE", {"matrix": 192}) and narrow["message"] == "256 is outside 32 to 192 for epi"
    clinic = make(CLINIC)
    task = clinic.episode_config()["task"]
    schema = clinic.tools("neutral")[-1]["parameters"]["properties"]["protocol"]
    assert set(schema["properties"]) == set(props["scan"]) - {"regions", "circles"} and not [k for k, p in schema["properties"].items() if "description" in p]
    answer = dict(causes={f["id"]: [] for f in task["findings"]}, protocol=dict(task["current_protocol"], fov_mm=24))
    err = call(clinic, "submit", **answer).data["error"]
    assert (err["type"], err["class"], err["parameter"]) == ("invalid_value", "task", "protocol.fov_mm") and "24 cm = 240 mm" in err["message"]


# ------------------------------------------------------------------------------------------------ sizes
# Characters of the tool definitions (OpenAI wire format) plus the task text, the largest level of each id. A ceiling
# is a ratchet: it is lowered when the text gets shorter and raised only with a reason in the changelog.
CEILING = {"Scanner/ArtefactClinic-v0": 17800, "Scanner/BandFree-v0": 16100, "Scanner/Explore-v0": 12000, "Scanner/FatSuppress-v0": 16100,
           "Scanner/FrequencyAdjust-v0": 14300, "Scanner/TransmitAdjust-v0": 14800, "Scanner/NoWrap-v1": 15500, "Scanner/NullTissue-v0": 14900, "Scanner/PhantomQA-v0": 16500, "Scanner/ProtocolToSpec-v0": 15900,
           "Scanner/SNRSpec-v0": 15200, "Scanner/T1Transmit-v1": 13900, "Scanner/T2Decay-v0": 14100, "Scanner/FatFraction-v0": 14900}


def _budget_tool():
    spec = importlib.util.spec_from_file_location("obs_budget", os.path.join(ROOT, "tools", "obs_budget.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("env_id", mrgym.list_envs())
def test_sizes_of_tools_and_statement_stay_under_their_ceiling(env_id):
    assert set(CEILING) == set(mrgym.list_envs())
    worst = 0
    for level in mrgym.spec(env_id).levels:
        env = mrgym.make(env_id, level=level, seed=9000)
        obs, _ = env.reset()
        worst = max(worst, len(json.dumps(env.tools("openai"))) + len(obs.text))
    assert 0.9 * CEILING[env_id] < worst <= CEILING[env_id], worst


def test_sizes_the_accounting_script_replays_a_recorded_conversation(tmp_path):
    """``tools/obs_budget.py`` on a conversation recorded with this tree: every call is answered with the recorded
    text, so the two columns of its table are equal; and its pieces add up to the conversation."""
    if not os.path.exists(os.path.join(ROOT, "tools", "obs_budget.py")):
        pytest.skip("the tests are not running from a checkout of the repository")
    budget = _budget_tool()
    env = mrgym.make(NULL, level=1, seed=3)
    settings = runner.model_settings(None, True, 1.0, 2048)
    obs, _ = env.reset(options=runner.reset_options(settings))
    messages = runner.first_messages(env, obs.text)
    turns = [[("scanner_info", {"topic": "ir"}), ("plan", IR)], [("scan", IR)], [("calc", {"expressions": "a = (1"}), ("view", {})],
             [("profile", {"axis": "x", "start_mm": -900}), ("plan", dict(IR, flipp=1))], [("submit", {r["id"]: 400 for r in env._regions})]]
    n = 0
    for k, calls in enumerate(turns):
        made = [dict(id=f"c{n + j}", type="function", function=dict(name=name, arguments=json.dumps(args))) for j, (name, args) in enumerate(calls)]
        messages.append(dict(role="assistant", content=f"turn {k}", tool_calls=made, turn=k + 1))
        for c in made:
            messages += env.step(c).observation.to_openai(c["id"])
        n += len(calls)
    folder = tmp_path / "episodes" / "stub" / "Scanner-NullTissue-v0" / "L1" / "s3" / "e0"
    folder.mkdir(parents=True)
    usage = dict(prompt_tokens=5000, completion_tokens=10)
    (folder / "record.json").write_text(json.dumps(dict(status="completed", agent="stub", env_id=NULL, level=1, seed=3, split="train", episode=0,
                                                        settings=settings, max_steps=env.max_steps, tokens_out=50, passed=False)))
    (folder / "messages.json").write_text(json.dumps(dict(tools=env.tools("openai"), messages=messages, turns=[dict(usage=usage)] * len(turns))))
    count = budget.Counter(None)
    assert not count.exact and count("x" * 33) == 10                     # no tokenizer: characters / 3.3, and it says so
    row = budget.replay_episode(str(folder), count)
    assert len(row["calls"]) == n == 8 and all(c["same"] for c in row["calls"]) and row["tokens_before"] == row["tokens_after"]
    assert [bool(c["is_error"]) for c in row["calls"]] == [False, False, False, True, False, False, True, False]
    assert [c["cls"] for c in row["calls"] if c["is_error"]] == ["shape", "vocabulary"] and row["calls"][5]["coerced"] == []
    pieces = row["cats"]
    assert {"statement", "limits", "status", "system", "assistant", "tools:plan", "result:scanner_info:ir", "result:plan", "result:scan", "error:calc",
            "result:view", "result:profile", "error:plan", "result:submit"} <= set(pieces)
    assert pieces["status"][0] == 9 and "charged" not in pieces           # one status line per answer and one under the task text
    whole = sum(len(m.get("content") or "") for m in messages) + sum(len(json.dumps(t, ensure_ascii=False)) for t in env.tools("openai"))
    assert row["chars_before"] >= whole and sum(v[1] for v in pieces.values()) == row["chars_before"]
    summary = budget.summarise([row], count.exact)
    text = budget.report(summary)
    assert "characters / 3.3; no tokenizer" in text and "== NullTissue: 1 episodes, 8 calls (8 answered with the recorded text" in text
    assert summary["families"]["NullTissue"]["cut_median"] == 0.0
    corpus = tmp_path / "calls.jsonl"                                     # the same calls as a replay corpus of the interface-tax metric
    header = budget.write_corpus(str(corpus), [row], "unit")
    assert (header["calls"], header["lost_when_recorded"], header["instances"], header["turns"]) == (8, 2, [[NULL, 1, 3]], [5])
    from mrgym.agents import iface_tax
    result = iface_tax.replay(str(corpus), draws=0)
    assert result["now"]["lost_calls"] == 2 and result["instances"] == 1 and result["accepted_then_refused_now"]["refused"] == 0
    assert I.INTERFACE_CLASSES == ("shape", "vocabulary") and budget.sizes_report([dict(env_id=NULL, level=1, tools=1, statement=2, total=3, per_tool={})])
