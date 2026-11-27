"""The interface and the task texts do not make a reader believe something false, and do not hide what a scan would
pay for. Each test pins one finding from the read of stored model conversations (run c04, thinking off): the sentence
or clause is there where it is true, absent where it is not, and what it says is checked against the code that scores.

    rules of the game   own scans are free in sequence and parameters; the delivered scan is not charged (7 families)
    plan                (y, x) order with the phase axis; the price as a scan now and what it leaves; regions the image
                        misses; the realised phase side; the minimum the console enforces; the price of a scan of its own
    profile             a width cut at the edge is said; samples above the maximum are cut, not refused
    view                the y of every row and an x ruler; the map itself unchanged
    phase               what a phase value is and is not (no sign without a reference)
    texts               slice_profile, PhantomQA (one fault per failed test, tangential, entirely inside), ArtefactClinic
                        (fields left out), the budget refusal
"""
import inspect
import json
import re

import numpy as np
import pytest

import mrgym
from mrgym import policies, qa
from mrgym import render as R
from mrgym.envs import family as F
from mrgym.envs import scanner_env
from mrgym.scanner import console, session

NOWRAP, T1, NULL, QA, SNR, CLINIC, FAT, BAND, SPEC, EXPLORE = (
    "Scanner/NoWrap-v1", "Scanner/T1Transmit-v1", "Scanner/NullTissue-v0", "Scanner/PhantomQA-v0", "Scanner/SNRSpec-v0",
    "Scanner/ArtefactClinic-v0", "Scanner/FatSuppress-v0", "Scanner/BandFree-v0", "Scanner/ProtocolToSpec-v0", "Scanner/Explore-v0")
TARGET_SCAN = (NOWRAP, NULL, FAT, BAND, SNR, SPEC, CLINIC)                # the deliverable is evaluated on a scan the task names
FREQ, TRANSMIT = "Scanner/FrequencyAdjust-v0", "Scanner/TransmitAdjust-v0"
OWN_SCAN = (QA, T1, EXPLORE, FREQ, TRANSMIT)                              # the deliverable is a measurement on the agent's own scans
IR = {"sequence": "ir", "tr_ms": 800, "ti_ms": 100, "matrix": 64, "partial_fourier": 0.6}
LOCALISER = {"sequence": "spgr", "tr_ms": 12, "flip_deg": 10, "fov_mm": 288, "matrix": 96}
CLAUSE_MAX = 160                                                          # characters of a clause that is printed only when its condition holds


def make(env_id=NULL, level=1, seed=0, **kw):
    env = mrgym.make(env_id, level=level, seed=seed, **kw)
    obs, _ = env.reset()
    return env, obs


def call(env, name, **arguments):
    return env.step({"name": name, "arguments": arguments}).observation


def body(obs):
    lines = obs.text.split("\n")
    assert lines[-1].startswith("[") and lines[-1].endswith("]"), obs.text
    return lines[:-1]


# ------------------------------------------------------------------------------------------------ 1. the rules of the game
def test_own_scans_sentence_is_the_same_wherever_a_target_scan_is_delivered_and_nowhere_else():
    assert set(TARGET_SCAN) | set(OWN_SCAN) == set(mrgym.list_envs())
    both = F.OWN_SCANS + " " + F.NOT_CHARGED
    for env_id in mrgym.list_envs():
        levels = mrgym.spec(env_id).levels                                # every hint tier where a level adds text; else one level (the text is one template)
        for level in levels if env_id in (NOWRAP, NULL) else levels[:1]:
            text = " ".join(make(env_id, level, seed=9000)[1].text.split())
            assert (both in text) == (env_id in TARGET_SCAN), (env_id, level)
            assert "Use the scan-time budget for whatever" not in text and text.count(F.OWN_SCANS) <= 1
    # it names no sequence and no value, and it is one sentence
    words = set(re.findall(r"[a-z_]+", both.lower()))
    assert not words & (set(console.SEQUENCES) | set(console.parameter_names())) and not re.search(r"\d", both)
    assert F.OWN_SCANS.count(".") == 1 and F.OWN_SCANS.endswith(".") and len(both) <= 215


@pytest.mark.parametrize("env_id", TARGET_SCAN)
def test_the_delivered_scan_is_not_charged_score_never_sees_the_budget(env_id):
    """``NOT_CHARGED`` is a statement about the verifier: ``score`` is handed the instance, the answer and the hidden
    world, never the session, and the budget gate is the replay of the agent's own log (``FamilyEnv.finish``)."""
    task = __import__(mrgym.spec(env_id).family + ".task", fromlist=["task"])
    assert list(inspect.signature(task.score).parameters) == ["cfg", "answer", "world", "engine"]
    source = inspect.getsource(task.score)
    assert "budget" not in source and "remaining" not in source and "session" not in source
    finish = inspect.getsource(F.FamilyEnv.finish)
    assert 'log_replays_within_budget=bool(rep["ok"] and not rep["over_budget"])' in finish


def test_the_delivered_scan_is_not_limited_by_the_budget():
    """NoWrap, where stored conversations believed the delivered scan had to fit the 30 s. The reference's delivered
    scan is longer than the scan time it has left, and passes; a delivered scan longer than the whole budget is scored
    by the pass rule alone (the budget gate holds)."""
    for seed in range(3):
        env = mrgym.make(NOWRAP, level=2, seed=seed)
        res = policies.run(env, "reference")
        score, budget = res.info["score"], res.info["budget"]
        assert score["passed"] and score["scan_s"] > budget["remaining_s"] and budget["used_s"] <= budget["budget_s"]
        assert sum(e["cost_s"] for e in env.scanner_log() if e.get("type") == "scan") == pytest.approx(budget["used_s"])   # only its own scans
        env.close()
    env, _ = make(NOWRAP, 2)
    call(env, "scan", **LOCALISER)
    res = env.step({"name": "submit", "arguments": dict(pe_dir="y", fov_mm=400, phase_fov=1.0, matrix=200, centre_y_mm=0, centre_x_mm=0, rotation_deg=0)})
    score = res.info["score"]
    assert score["scan_s"] > 3 * res.info["budget"]["budget_s"] and score["checks"]["log_replays_within_budget"] and score["checks"]["deliverable_valid"]
    assert [k for k, ok in score["checks"].items() if not ok] == ["scan_time_within_factor"]         # too slow by the pass rule; not over a budget


# ------------------------------------------------------------------------------------------------ 2. plan
def test_plan_prints_the_image_as_y_x_with_the_phase_axis_named():
    env, _ = make(NOWRAP, 2)
    geometry = dict(sequence="se", tr_ms=500, fov_mm=300, matrix=48, phase_fov=0.75)
    for pe, shape, fov in (("y", (36, 48), (225, 300)), ("x", (48, 36), (300, 225))):
        line = body(call(env, "plan", pe_dir=pe, **geometry))[-1]
        assert line.startswith(f"{shape[0]}x{shape[1]} px (y, x), FOV {fov[0]} x {fov[1]} mm, phase-encode {pe}: 36 lines · te_ms "), line
    scan = call(env, "scan", pe_dir="x", fov_mm=300, matrix=48, phase_fov=0.75, sequence="spgr", tr_ms=12, flip_deg=10)
    assert " 48x36 (y, x) stored" in scan.text and scan.data["shape"][1:] == [48, 36]            # the order scan prints, and view
    assert "48x36 (y, x), FOV 300 x 225 mm" in call(env, "view").text


def test_plan_prices_a_scan_now_and_says_what_it_leaves():
    env, _ = make(NULL)
    left = env.state()["scan_time_left_s"]
    fits = call(env, "plan", **IR)
    assert body(fits)[0] == f"plan ok · as a scan now: 35.00 s, leaves {left - 35:.2f} s" and fits.data["fits_budget"]
    over = call(env, "plan", **dict(IR, tr_ms=8000))
    assert body(over)[0] == f"plan ok · as a scan now: needs 323.00 s, {left:.2f} s left" and not over.data["fits_budget"]
    for obs in (fits, over):                                              # a plan is also how a deliverable is looked at: no verdict on "the budget"
        assert "budget" not in obs.text and "NOT" not in obs.text and "of its own" not in obs.text
    call(env, "scan", **IR)
    assert body(call(env, "plan", **IR))[0] == f"plan ok · as a scan now: 35.00 s, leaves {left - 70:.2f} s"


def test_plan_lists_the_marked_regions_the_image_would_miss():
    env, _ = make(NULL)
    ids = [r["id"] for r in env._regions]
    inside = call(env, "plan", **IR)
    assert len(body(inside)) == 2 and "regions_without_pixels" not in inside.data
    seen = {}
    for phase_fov, fov in ((0.25, 100), (0.3, 240), (0.5, 240), (1.0, 120), (0.25, 240)):
        protocol = dict(IR, phase_fov=phase_fov, fov_mm=fov)
        plan = call(env, "plan", **protocol)
        none, cut = plan.data.get("regions_without_pixels", []), plan.data.get("regions_partly_outside", [])
        last = body(plan)[-1]
        if none or cut:
            want = "; ".join(x for x in ("no pixel of this image in " + ", ".join(none) if none else "",
                                         ("partly outside it: " if none else "partly outside this image: ") + ", ".join(cut) if cut else "") if x)
            assert last == want and len(last) <= CLAUSE_MAX
        else:
            assert "pixel of this image" not in plan.text and "partly outside" not in plan.text
        seen[(phase_fov, fov)] = (none, cut)
        # the read-out of a scan of that protocol agrees: no pixels exactly in the regions the plan named
        rows = session.roi_readouts(np.zeros((1,) + tuple(plan.data["realised"][k] for k in ("matrix_pe", "matrix")), complex), env._regions,
                                    session.fov_yx(plan.data["realised"]), session.centre_yx(plan.data["realised"]))
        assert [r["id"] for r in rows if r["n_pixels"] == 0] == none and not set(none) & set(cut)
    assert any(none for none, _ in seen.values()) and any(cut for _, cut in seen.values()), seen
    # a scan that was paid for and holds no pixel of a region: the plan before it said so
    protocol = dict(IR, phase_fov=0.25, fov_mm=100)
    said = call(env, "plan", **protocol).data["regions_without_pixels"]
    scan = call(env, "scan", **protocol)
    assert said and [r["id"] for r in scan.data["roi_readouts"] if r["n_pixels"] == 0] == said
    # the longest clause: every region of a task with many regions
    many, _ = make(T1, 3)
    line = body(call(many, "plan", sequence="spgr", tr_ms=12, flip_deg=10, fov_mm=80, matrix=64, phase_fov=0.25, centre_y_mm=150, centre_x_mm=150))[-1]
    assert line == "no pixel of this image in " + ", ".join(r["id"] for r in many._regions) and len(line) <= CLAUSE_MAX and len(many._regions) >= 4
    assert ids and not scanner_env.ScannerEnv._regions_missed(make(NOWRAP, 2)[0], inside.data["realised"], (64, 64))[0]     # no marked regions: nothing listed


def test_plan_says_what_the_phase_side_was_realised_as():
    env, _ = make(NULL)
    odd = call(env, "plan", sequence="se", tr_ms=500, matrix=88, phase_fov=45 / 88)                 # 45 lines asked for
    lines = body(odd)
    assert lines[1] == "realised (on the console's raster): phase_fov 0.511364 -> 0.5 (44 px, an even number)" and len(lines[1]) <= CLAUSE_MAX
    assert lines[2].startswith("44x88 px (y, x), FOV 120 x 240 mm, phase-encode y: 44 lines") and odd.data["realised"]["matrix_pe"] == 44
    assert "realised" not in call(env, "plan", sequence="se", tr_ms=500, matrix=88, phase_fov=0.5).text       # an even count: nothing to say
    assert "realised" not in call(env, "plan", sequence="se", tr_ms=500, matrix=88).text


@pytest.mark.parametrize("sequence", sorted(console.SEQUENCES))
def test_plan_shows_the_minimum_the_console_enforces(sequence):
    env, _ = make(NULL, max_rejected_calls=100)
    protocol = dict(scanner_env.SEQUENCE_EXAMPLES[sequence])
    plan = call(env, "plan", **protocol)
    shown = dict(re.findall(r"(\w+_ms) ([\d.]+)", plan.text.split("minimums: ")[1].split(" · ")[0]))
    low = console.SEQUENCES[sequence]
    assert "tr_ms" in shown and all(float(v) >= low[k][0] for k, v in shown.items() if k in low), shown
    tr = float(shown["tr_ms"])
    codes = lambda value: {v["code"] for v in console.plan(dict(protocol, tr_ms=value), env._session.exam)["violations"]   # noqa: E731
                           if v["parameter"] == "tr_ms" and v["code"] in ("OUT_OF_RANGE", "TR_BELOW_MINIMUM")}
    assert not codes(tr) and codes(round(tr - 0.02, 6)), (tr, codes(tr))                              # accepted at the minimum shown, refused below it
    if sequence == "se":                                                  # the stored case: 12.14 was shown, below 50 is refused
        assert tr == 50 and plan.data["timing"]["tr_min_ms"] < 50
        assert call(env, "plan", **dict(protocol, tr_ms=plan.data["timing"]["tr_min_ms"])).is_error


def test_plan_prices_a_scan_of_its_own_when_the_console_reuses_adjustments():
    """ProtocolToSpec: a scan that repeats the geometry of the one before is charged the short adjustments; a
    delivered image is charged the full ones (``task.measure``: the console's plan for it as a scan of its own)."""
    from mrgym.envs.protocol_to_spec import task as T
    env, _ = make(SPEC, 1, seed=9000)
    cfg = env.episode_config()
    t = cfg["task"]
    image = dict(t["scan"], **t["standard_protocol"])
    probe = dict(image, matrix=32)
    before = call(env, "plan", **image)
    assert "of its own" not in before.text and "scan_s_own_scan" not in before.data                 # nothing to reuse yet: one price
    call(env, "scan", **probe)
    after = call(env, "plan", **image)
    verifier = T.measure(cfg, "A", t["standard_protocol"], env.hidden_world(), env._session.engine)["scan_s"]
    first = body(after)[0]
    clause = f" (reuses the adjustments of the scan before) · as a scan of its own: {verifier:.2f} s"
    assert first.endswith(clause) and len(clause) <= CLAUSE_MAX and after.data["scan_s_own_scan"] == verifier == before.data["scan_s"]
    assert after.data["scan_s"] == pytest.approx(verifier - (T.CONSOLE["overhead_s"] - T.CONSOLE["repeat_overhead_s"]))
    assert first.startswith(f"plan ok · as a scan now: needs {after.data['scan_s']:.2f} s, ")
    moved = call(env, "plan", **dict(probe, fov_mm=300))                  # another field of view: the adjustments are repeated, one price again
    assert "of its own" not in moved.text
    assert "of its own" not in call(make(NULL)[0], "plan", **IR).text     # a console without repeat adjustments never prints it


# ------------------------------------------------------------------------------------------------ 3. profile
def test_profile_says_when_the_width_was_cut_at_the_edge():
    env, _ = make(NOWRAP, 1)
    call(env, "scan", **LOCALISER)                                        # 96 x 96 pixels of 3 mm: the image reaches from -145.5 to 142.5 mm
    cut = call(env, "profile", axis="x", position_mm=130, width_mm=63)
    note = next(line for line in body(cut) if line.startswith("note: width_mm"))
    assert note == f"note: width_mm 63 -> {cut.data['width_mm']:g} mm (cut at the edge of the image; the line is centred at y = {cut.data['position_mm']:g} mm)"
    assert cut.data["width_mm"] < 63 and cut.data["position_mm"] != 130 and len(note) <= CLAUSE_MAX
    assert [x["parameter"] for x in cut.data["adjusted"]] == ["width_mm"]
    whole = call(env, "profile", axis="y", width_mm=1000, reduce="max")   # a projection asked for with a width beyond the image
    assert "note: width_mm 1000 -> 288 mm (cut at the edge of the image; " in whole.text
    for inside in (dict(position_mm=0, width_mm=63), dict(position_mm=100, width_mm=10), dict(width_mm=288), dict()):
        obs = call(env, "profile", axis="x", **inside)
        assert "note:" not in obs.text and "adjusted" not in obs.data, inside


def test_profile_samples_text_and_behaviour_agree():
    top = session.MAX_PROFILE_SAMPLES
    env, _ = make(NOWRAP, 1)
    described = next(t for t in env.tools("neutral") if t["name"] == "profile")["parameters"]["properties"]["samples"]["description"]
    assert described == f"default one per pixel up to {top}, the most per call"
    call(env, "scan", sequence="spgr", tr_ms=12, flip_deg=10, fov_mm=288, matrix=256, phase_fov=0.25)        # a line of 256 pixels
    default = call(env, "profile", axis="x")
    assert len(default.data["values"]) == top and default.data["pixels_per_sample"] == 2 and "note:" not in default.text
    short = call(env, "profile", axis="y")                                # fewer pixels than the maximum: one sample per pixel
    assert len(short.data["values"]) == 64 and short.data["pixels_per_sample"] == 1
    more = call(env, "profile", axis="x", samples=256)                    # more than a call returns: cut and said, not refused
    assert not more.is_error and more.data["values"] == default.data["values"]
    assert f"note: samples 256 -> {top} samples (the most a call returns)" in more.text
    assert more.data["adjusted"] == [dict(parameter="samples", requested=256, realised=top, reason="the most a call returns")]
    assert call(env, "profile", axis="x", samples=0).is_error and "note:" not in call(env, "profile", axis="x", samples=top).text


# ------------------------------------------------------------------------------------------------ 4. view
@pytest.mark.parametrize("size, protocol", [(None, LOCALISER), (24, dict(LOCALISER, fov_mm=300, matrix=48, phase_fov=0.75)),
                                            (48, dict(LOCALISER, centre_y_mm=-20, centre_x_mm=15, matrix=128)), (16, dict(LOCALISER, rotation_deg=30))])
def test_view_gives_the_y_of_every_row_and_an_x_ruler(size, protocol):
    env, _ = make(NOWRAP, 2)
    call(env, "scan", **protocol)
    view = call(env, "view", **({"size": size} if size else {}))
    stored = env.stored_scans()[0]
    img, fov = stored["magnitude"][0], stored["fov_yx_mm"]
    frame = session.frame_centre(stored["centre_yx_mm"], stored["rotation_deg"])
    grid, legend = R.digit_map(img, "magnitude", size or scanner_env.VIEW_SIZE, fov, frame)
    rows = grid.splitlines()
    lines = body(view)
    shown, ruler = lines[-len(rows) - 1:-1], lines[-1]
    assert view.data["digit_map"] == rows and [s.split(" ")[0] for s in shown] == rows               # the map itself: unchanged
    assert lines[1] == legend + "; block centres: y_mm after each row, x_mm below, where a number starts"
    # the numbers are the centres of the blocks, from the pixel positions of the image
    ypos, xpos = session.pixel_positions(img.shape, fov, frame)
    centres = lambda pos, n: [float(pos[a:b].mean()) for a, b in zip(np.linspace(0, len(pos), n + 1).astype(int)[:-1],   # noqa: E731
                                                                     np.linspace(0, len(pos), n + 1).astype(int)[1:])]
    ys, xs = centres(ypos, len(rows)), centres(xpos, len(rows[0]))
    assert [float(s.split(" ")[1]) for s in shown] == pytest.approx(ys, abs=0.51) and len(set(s.split(" ")[1] for s in shown)) == len(rows)
    marks = [(m.start(), float(m.group())) for m in re.finditer(r"-?\d+(?:\.\d+)?", ruler)]          # a number starts in the column it belongs to
    assert len(marks) >= 3 and marks[0][0] == 0 and all(value == pytest.approx(xs[col], abs=0.51) for col, value in marks)
    assert all(b[0] - a[0] == marks[1][0] for a, b in zip(marks, marks[1:])) and re.fullmatch(r"[-\d. ]+", ruler)
    # what it costs: a few characters per row, one line and one clause
    added = len("\n".join(lines)) - len("\n".join([lines[0], legend] + lines[2:-len(rows) - 1] + rows))
    assert added <= 7 * len(rows) + len(rows[0]) + 80, added


def test_view_of_a_png_carries_no_ruler():
    env, _ = make(NOWRAP, 2, obs="vision")
    call(env, "scan", **LOCALISER)
    view = call(env, "view")
    assert view.images and "block centres" not in view.text and len(body(view)) == 1


# ------------------------------------------------------------------------------------------------ 5. the phase of a read-out
def test_phase_text_says_what_a_phase_value_is_and_is_not():
    env, _ = make(NULL)
    scan = next(t for t in env.tools("neutral") if t["name"] == "scan")["description"]
    assert "(no noise floor; a phase alone gives no sign: scanner_info outputs)" in scan
    ir, outputs, sequences = (call(env, "scanner_info", topic=t).text for t in ("ir", "outputs", "sequences"))
    for text in (ir, outputs, sequences, json.dumps(env._session.info())):
        assert "the sign is in the phase" not in text
    assert "a negative signal has the phase of a positive signal of\nthe same region plus π. A phase value alone gives no sign" in ir
    assert "the number and its own sign say\n  nothing about the sign of the signal" in outputs and "read-out of the same region whose sign is known" in outputs
    assert "a phase value alone gives no sign" in sequences
    added = outputs[outputs.index("The phase\n  (`phase_rad`"):outputs.index("- `view` shows")]
    assert not re.search(r"\bTI\b|ti_ms|longest|inversion time|\d\d", ir.split("The signal is signed")[1].split("For TR much longer")[0] + added)   # no method


def test_phase_of_a_signed_signal_is_what_the_text_says():
    """Signals of opposite sign differ by pi; the phase of the positive one is another number in another session."""
    positive = []
    for seed in (0, 1):
        env, _ = make(NULL, seed=seed)
        phase = {}
        for ti in (60, 700):                                              # before and after the null of every material of the level
            rows = call(env, "scan", **dict(IR, ti_ms=ti)).data["roi_readouts"]
            phase[ti] = np.array([r["complex_mean_phase_rad"][0] for r in rows])
        assert np.abs(np.angle(np.exp(1j * (phase[60] - phase[700])))) == pytest.approx(np.pi, abs=0.05)
        positive.append(phase[700][0])
        assert {np.sign(p) for ti in phase for p in phase[ti]} == {-1.0, 1.0}                       # the number's own sign: both, in one session
    assert abs(positive[0] - positive[1]) > 1.0                           # no fixed phase means "positive"


# ------------------------------------------------------------------------------------------------ 6 to 9. texts
def test_slice_profile_says_which_flip_angle_a_hard_pulse_gives():
    env, _ = make(T1)
    text = call(env, "scanner_info", topic="slice_profile").text
    assert ("With `excitation: nonselective`\nevery voxel sees the same actual flip angle a (the nominal angle times the local transmit factor), "
            "without a slice\nprofile, and the formulas below are what you receive.") in text
    assert "every voxel sees the flip angle a and" not in scanner_env.datasheet()


def test_phantom_qa_statement_one_fault_per_failed_test_tangential_entirely_inside():
    from mrgym.envs.phantom_qa import task as T
    text = " ".join(make(QA, 1, seed=9000)[1].text.split())
    for level in mrgym.spec(QA).levels:
        assert "more than one fault moves each test. A failed test has exactly one fault. The scan-time budget" in text
        assert "the 63 mm side tangential (top, bottom: along x; left, right: along y)" in text and "long side along the edge" not in text
        assert "of a 1 cm^2 circle (radius 5.64 mm) lying entirely inside the 200 cm^2 circle" in text and "anywhere inside" not in text
        for k in range(150):                                              # by construction: the faults of a draw move different tests
            faults = T._draw(f"{k:032x}", level)["faults"]
            assert len({T.FAULTS[f]["test"] for f in faults}) == len(faults) <= T.LEVELS[level]["n_faults"]
    rect = qa.background_rois((250.0, 250.0))                             # (y0, y1, x0, x1): the 63 mm side runs along the edge of the phantom
    size = {k: (round(v[1] - v[0], 6), round(v[3] - v[2], 6)) for k, v in rect.items()}
    assert size["top"] == size["bottom"] == (16.0, 63.0) and size["left"] == size["right"] == (63.0, 16.0)
    assert "ok = dist <= big - small" in inspect.getsource(qa.uniformity)  # the whole small circle inside the large one


def test_artefact_clinic_says_what_a_field_left_out_becomes():
    from mrgym.envs.artefact_clinic import task as T
    env, obs = make(CLINIC, 1, seed=9000)
    assert "A field left out takes the console default, not the current protocol's value." in " ".join(obs.text.split())
    cfg = env.episode_config()
    current = cfg["task"]["current_protocol"]
    name = next(k for k in ("averages", "slice_mm", "bandwidth_hz_per_px") if k in current
                and console.plan(current, env._session.exam)["realised"][k] != console.SEQUENCES[current["sequence"]][k][3])
    without = {k: v for k, v in current.items() if k != name}
    assert T.check(cfg, dict(causes={f["id"]: [] for f in cfg["task"]["findings"]}, protocol=without), env._session.exam) is None
    assert console.plan(without, env._session.exam)["realised"][name] == console.SEQUENCES[current["sequence"]][name][3]    # what score plans
    assert 'console.plan(answer["protocol"], world["console"])' in inspect.getsource(T.score)


def test_budget_refusal_states_the_numbers_and_steers_nowhere():
    env, _ = make(NOWRAP, 2)
    refused = call(env, "scan", sequence="se", tr_ms=500, fov_mm=300, matrix=60)
    assert body(refused) == ["error budget · this scan needs 33.50 s and 30.00 s are left; any accepted protocol that fits can be scanned"]
    err = refused.data["error"]
    assert (err["type"], err["code"], err["requested"], err["allowed"]) == ("budget", "SCAN_TIME", 33.5, {"max": 30.0})
    words = set(re.findall(r"[a-z_]+", refused.text.lower()))
    assert not words & (set(console.SEQUENCES) | set(console.parameter_names()) | {"fewer", "shorter", "lines", "averages", "lower", "reduce"})
    assert env.state()["scan_time_left_s"] == 30.0 and not call(env, "scan", **LOCALISER).is_error   # nothing charged; a protocol that fits is scanned
