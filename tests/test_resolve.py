"""``Scanner/Resolve-v0``: the physics behind the pass rule checked against closed forms, the verifier, the oracle, the
instances and the control policies. Expected numbers are worked out here (sinc arithmetic, the console's cost rule,
the outline of the hidden object), not taken from ``task.score``."""
import concurrent.futures
import json
import math

import numpy as np
import pytest

import mrgym
from mrgym import coverage, keys, policies
from mrgym.env import derive_seed
from mrgym.envs import family as F
from mrgym.envs import oracle as O
from mrgym.envs.nowrap import task as N
from mrgym.envs.resolve import controls, oracle, task
from mrgym.scanner import console, kspace
from mrgym.scanner import world as W

ENV_ID = "Scanner/Resolve-v0"


def _cfg(level, seed, split="train"):
    return keys.make_instance(task, derive_seed(ENV_ID, level, split, seed, environ={}), level, "t", split)


@pytest.fixture(scope="module")
def instance():
    """One level-3 instance with its hidden world, engine and optima (verifier side)."""
    cfg = _cfg(3, 0)
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    engine = kspace.Engine(world)
    return dict(cfg=cfg, world=world, engine=engine, outline=N.outline(world), solution=oracle.solve(cfg, world, engine))


# ---------------------------------------------------------------------------------------------- physics, checked independently
def _two_points(spacing_px, at):
    """Image of two unit points ``spacing_px`` pixels apart behind an ideal low-pass, at positions ``at`` (pixels)."""
    return np.abs(np.sinc(at) + np.sinc(at - spacing_px))


def test_a_pixel_as_large_as_the_spacing_does_not_separate_two_points():
    """Textbook arithmetic, no package code: the image of a point is sinc(x / pixel). One pixel apart, the midpoint
    holds 2 sinc(1 / 2) = 4 / pi and each point holds 1: one maximum. Two pixels apart: the midpoint is 0."""
    assert _two_points(1.0, np.array([0.5]))[0] == pytest.approx(4 / math.pi) and _two_points(1.0, np.array([0.0]))[0] == pytest.approx(1.0)
    assert _two_points(2.0, np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-12)
    # ... and the measure of the task module says the same, and where the dip reaches one half
    assert task.point_pair_dip(1.0) == 0.0 and task.point_pair_dip(2.0) == pytest.approx(1.0, abs=1e-6)
    x = np.linspace(1.4, 1.9, 501)
    half = x[np.argmax([task.point_pair_dip(v) >= 0.5 for v in x])]
    assert half == pytest.approx(1.633, abs=0.005)
    assert 1 / half == pytest.approx(0.612, abs=0.003)                  # pixel / spacing at a dip of one half, for two points
    # every finer pixel separates as well: from two pixels on the image falls to zero between the two points (the side
    # lobe half-way between them, up to 2 |sinc(1.43)| = 0.43, lies between two zeros)
    assert all(task.point_pair_dip(v) >= task.MIN_DIP for v in np.linspace(half, 12.0, 400))
    assert min(task.point_pair_dip(v) for v in np.linspace(2.0, 12.0, 300)) > 0.99
    assert max(_two_points(v, np.array([v / 2]))[0] for v in np.linspace(2.0, 4.0, 201)) == pytest.approx(0.434, abs=0.002)


def _flat_pair(spacing_mm, radius_mm=0.6):
    """Two small bright discs on nothing, ``spacing_mm`` apart along x, in a world with ideal fields."""
    ph = W.shapes_phantom([("csf", lambda y, x: np.hypot(y, x + spacing_mm / 2) - radius_mm), ("csf", lambda y, x: np.hypot(y, x - spacing_mm / 2) - radius_mm)],
                          b0_t=1.5, edge=0.2)
    world = W.flat_world(ph)
    return world, kspace.Engine(world)


@pytest.mark.parametrize("spacing_px", [1.0, 1.5, 1.8, 2.0, 2.6])
def test_the_engine_gives_the_dip_of_the_closed_form_for_two_small_features(spacing_px):
    """The central formula against the engine: two discs much smaller than the pixel are two points, and the dip the
    verifier reads on a noise-free scan is the dip of two sinc functions."""
    pixel, spacing = 4.0, 4.0 * spacing_px
    world, engine = _flat_pair(spacing)
    rp = console.plan(dict(task.TARGET, pe_dir="y", fov_mm=pixel * 40, matrix=40, phase_fov=1.0), world["console"])["realised"]
    got = task.separation(engine, rp, (0.0, -spacing / 2), (0.0, spacing / 2))
    assert got == pytest.approx(task.point_pair_dip(spacing_px), abs=0.03)
    # along the phase-encode axis the point-spread function is the same sinc
    rp = console.plan(dict(task.TARGET, pe_dir="x", fov_mm=pixel * 40, matrix=40, phase_fov=1.0), world["console"])["realised"]
    assert task.separation(engine, rp, (0.0, -spacing / 2), (0.0, spacing / 2)) == pytest.approx(task.point_pair_dip(spacing_px), abs=0.03)


def test_the_image_between_the_pixels_is_the_console_image_at_the_pixel_centres(instance):
    """``image_at`` (the Fourier sum of the acquired k-space at any point) against the console reconstruction, for both
    phase-encode axes and a rectangular field of view."""
    engine, exam = instance["engine"], instance["world"]["console"]
    for pe_dir in ("y", "x"):
        rp = console.plan(dict(task.TARGET, pe_dir=pe_dir, fov_mm=240.0, matrix=60, phase_fov=0.7), exam)["realised"]
        img = engine.acquire(rp)[0]
        ny, nx = img.shape
        fy, fx = (rp["fov_pe_mm"], rp["fov_mm"]) if pe_dir == "y" else (rp["fov_mm"], rp["fov_pe_mm"])
        pts = np.array([[(i - ny // 2) * fy / ny, (j - nx // 2) * fx / nx] for i, j in ((3, 5), (ny // 2, nx // 2), (ny - 2, nx - 7), (ny // 2 + 4, 2))])
        want = np.array([img[3, 5], img[ny // 2, nx // 2], img[ny - 2, nx - 7], img[ny // 2 + 4, 2]])
        assert np.abs(task.image_at(engine, rp, pts) - want).max() <= 1e-9 * np.abs(img).max()


def test_scan_time_grows_with_the_phase_encode_lines_only(instance):
    """The console's cost rule for the target scan: 3 s overhead, one dummy cycle, one line per repetition."""
    exam = instance["world"]["console"]
    for matrix, phase_fov, lines in ((60, 0.5, 30), (120, 0.5, 60), (64, 1.0, 64)):
        pl = console.plan(task.protocol_of(dict(pe_dir="y", fov_mm=240.0, phase_fov=phase_fov, matrix=matrix)), exam)
        assert pl["realised"]["lines"] == lines and pl["scan_s"] == pytest.approx(3.0 + 0.5 * (1 + lines))
    a = console.plan(task.protocol_of(dict(pe_dir="y", fov_mm=240.0, phase_fov=0.5, matrix=80)), exam)["scan_s"]
    assert a == console.plan(task.protocol_of(dict(pe_dir="x", fov_mm=400.0, phase_fov=0.25, matrix=160)), exam)["scan_s"]   # 40 lines each


# ---------------------------------------------------------------------------------------------- the verifier
def test_the_pass_rule_requirement_by_requirement(instance):
    cfg, world, engine, best = instance["cfg"], instance["world"], instance["engine"], instance["solution"]["pairs"]
    good = instance["solution"]["argmin"]
    s = task.score(cfg, good, world=world, engine=engine)
    assert s["passed"] and s["error"] == 0.0 and s["diagnostic"] == ""
    assert list(s["checks"]) == ["deliverable_valid"] + [f"{r}_{i}" for r in task.REQUIREMENTS for i in task.PAIRS]
    g = good["A"]
    pixel = g["fov_mm"] / g["matrix"]

    def with_a(**change):
        return task.score(cfg, dict(good, A=dict(g, **change)), world=world, engine=engine)

    def failing(score):
        return [k for k, ok in score["checks"].items() if not ok]
    # a coarser pixel with the same lines: nothing is cut off, the pair merges (the pixel was the largest that separates... within one line pair)
    coarse = with_a(fov_mm=round(1.3 * g["fov_mm"], 4))
    assert failing(coarse) == ["features_separated_A"] and task.outcome(coarse) == "not_separated"
    assert coarse["pairs"]["A"]["dip"] < task.MIN_DIP <= s["pairs"]["A"]["dip"]
    # two lines fewer at the same pixel: the object leaves the image
    lines = best["A"]["lines"]
    cut = with_a(phase_fov=round((lines - 4) / g["matrix"], 6))
    assert "whole_object_inside_A" in failing(cut) and task.outcome(cut) == "object_cut_off"
    # the same pixel with half as many lines again: valid and too slow
    more = 2 * math.ceil(0.75 * lines)
    slow = with_a(matrix=max(g["matrix"], more), fov_mm=round(pixel * max(g["matrix"], more), 6), phase_fov=round(more / max(g["matrix"], more), 6))
    assert failing(slow) == ["scan_time_within_factor_A"] and task.outcome(slow) == "too_slow" and slow["pairs"]["A"]["lines"] == more
    assert slow["error"] == pytest.approx((3.5 + 0.5 * more) / best["A"]["scan_s"] - 1.0)
    # the finest pixel of the console: over the time cap and slow; under the SNR floor where the SNR says so
    fine = with_a(fov_mm=240.0, matrix=300, phase_fov=0.8)
    assert {"within_time_cap_A", "scan_time_within_factor_A"} <= set(failing(fine)) and fine["pairs"]["A"]["scan_s"] == 123.5 > task.MAX_SCAN_S
    assert ("snr_floor_A" in failing(fine)) == (fine["pairs"]["A"]["snr"] < task.MIN_SNR) and fine["pairs"]["A"]["snr"] < 0.25 * s["pairs"]["A"]["snr"]
    assert task.outcome(fine) == ("snr_too_low" if "snr_floor_A" in failing(fine) else "over_time_cap")


def test_the_snr_of_the_reference_region_follows_the_voxel_and_the_lines(instance):
    """The law the SNR floor rests on, on the engine: signal per pixel grows with the pixel area, the noise of a pixel
    falls with the square root of the acquired samples per unit of k-space area, so SNR ~ pixel^2 sqrt(lines x matrix) / sqrt(matrix)
    = pixel^2 sqrt(lines) at a fixed bandwidth per pixel."""
    cfg, world, engine = instance["cfg"], instance["world"], instance["engine"]
    ref = next(r for r in cfg["task"]["regions"] if r["id"] == task.REFERENCE_REGION)

    def snr(pixel, matrix, lines):
        rp = console.plan(task.protocol_of(dict(pe_dir="y", fov_mm=pixel * matrix, phase_fov=lines / matrix, matrix=matrix)), world["console"])["realised"]
        assert rp["lines"] == lines
        return task.reference_snr(engine, rp, ref)
    base = snr(2.0, 128, 128)
    assert snr(4.0, 64, 64) / base == pytest.approx(4.0 * math.sqrt(0.5), rel=0.06)
    assert snr(2.0, 128, 96) / base == pytest.approx(math.sqrt(0.75), rel=0.02)
    assert snr(3.0, 100, 80) / base == pytest.approx(2.25 * math.sqrt(80 / 128), rel=0.06)
    assert task.reference_snr(engine, console.plan(task.protocol_of(dict(pe_dir="y", fov_mm=80.0, phase_fov=0.25, matrix=40)), world["console"])["realised"],
                              dict(ref, y_mm=140.0)) == 0.0             # a region outside the image has no signal-to-noise ratio


def test_the_time_factor_is_inclusive_and_measured_against_the_optimum_of_each_pair(instance, monkeypatch):
    cfg, world, engine = instance["cfg"], instance["world"], instance["engine"]
    good = instance["solution"]["argmin"]
    plain = task.optimum(cfg, world, engine)
    scan_s = {i: console.plan(task.protocol_of(good[i]), world["console"])["scan_s"] for i in task.PAIRS}
    monkeypatch.setattr(task, "optimum", lambda c, w, e: {i: dict(plain[i], scan_s=scan_s[i] / task.TOL) for i in task.PAIRS})
    assert task.score(cfg, good, world=world, engine=engine)["passed"]
    monkeypatch.setattr(task, "optimum", lambda c, w, e: {i: dict(plain[i], scan_s=scan_s[i] / task.TOL * (1 - 1e-6 if i == "B" else 1)) for i in task.PAIRS})
    s = task.score(cfg, good, world=world, engine=engine)
    assert not s["passed"] and [k for k, ok in s["checks"].items() if not ok] == ["scan_time_within_factor_B"]
    assert s["diagnostic"] == "pair B: the scan is slower than allowed"


def test_refused_deliverables(instance):
    cfg, exam, good = instance["cfg"], instance["world"]["console"], instance["solution"]["argmin"]
    assert task.check(cfg, good, exam) is None
    shape = "the deliverable is {" + ", ".join(f'"{i}": {{"pe_dir", "fov_mm", "phase_fov", "matrix"}}' for i in "ABC") + "}"
    for bad in (None, [], dict(A=good["A"], B=good["B"]), dict(good, D=good["A"]), dict(good, A=dict(good["A"], averages=2)),
                dict(good, A={k: v for k, v in good["A"].items() if k != "matrix"}), dict(good, B=7)):
        assert task.check(cfg, bad, exam) == shape
    cm = task.check(cfg, dict(good, B=dict(good["B"], fov_mm=good["B"]["fov_mm"] / 10)), exam)
    assert cm.startswith("the console rejects the geometry of pair B: ") and cm.endswith("(a field of view of this size looks like centimetres)")
    assert "matrix must be even" in task.check(cfg, dict(good, A=dict(good["A"], matrix=61)), exam)
    assert "pe_dir must be one of" in task.check(cfg, dict(good, A=dict(good["A"], pe_dir="z")), exam)
    s = task.score(cfg, dict(A=good["A"]), world=instance["world"], engine=instance["engine"])
    assert not s["passed"] and s["checks"] == dict(deliverable_valid=False) and s["error"] is None


# ---------------------------------------------------------------------------------------------- the oracle
def test_the_oracle_is_independent_of_the_control_policies():
    assert O.independence_problems("mrgym.envs.resolve") == []


def test_the_oracle_optimum_passes_and_everything_cheaper_in_its_search_fails(instance):
    cfg, world, engine, sol = instance["cfg"], instance["world"], instance["engine"], instance["solution"]
    assert sol["value"] == 0.0 and sol["method"] == "exhaustive" and set(sol["pairs"]) == set(task.PAIRS) and sol["seconds"] >= 0
    assert task.score(cfg, sol["argmin"], world=world, engine=engine)["passed"]
    for i in task.PAIRS:
        p = sol["pairs"][i]
        assert p["scan_s"] == pytest.approx(3.0 + 0.5 * (1 + p["lines"])) and p["pixel_mm"] == pytest.approx(p["geometry"]["fov_mm"] / p["geometry"]["matrix"], abs=1e-4)
    cert = sol["certificate"]
    assert cert and all(c["cost"] <= sol["pairs"][c["pair"]]["scan_s"] for c in cert)
    assert {c["fails"] for c in cert} <= {"features_separated", "whole_object_inside"}
    for c in cert[:3] + cert[-3:]:                                       # each one fails the verifier's own check, on the engine
        assert not task.pair_checks(cfg, c["pair"], c["deliverable"], world, engine, instance["outline"])["checks"][c["fails"]]


def test_the_oracle_geometry_is_the_smallest_pixel_that_holds_the_object(instance):
    """Worked out by hand from the outline: n pixels of size p about the isocentre reach from -(n + 1) p / 2 to (n - 1) p / 2."""
    world, outline = instance["world"], instance["outline"]
    box = oracle.extent(outline, world["extent_mm"] / world["n"])
    grid = world["extent_mm"] / world["n"]
    for axis, col in (("y", 0), ("x", 1)):
        assert box[axis] == pytest.approx((outline[:, col].min() - grid / 2 + 0.5, outline[:, col].max() + grid / 2 - 0.5))
    for pe_dir in ("y", "x"):
        lo, hi = box[pe_dir]
        for n in (30, 44):
            g = oracle.geometry(box, pe_dir, n)
            p = g["fov_mm"] / g["matrix"]
            assert p == pytest.approx(max(2 * hi / (n - 1), -2 * lo / (n + 1)), rel=1e-5)
            rp = console.plan(task.protocol_of(g), world["console"])["realised"]
            assert rp["lines"] == n and N.overhang_mm(outline, rp, grid) <= task.COVER_SLACK_MM
            tighter = dict(g, fov_mm=round(g["fov_mm"] * 0.995, 6))
            assert N.overhang_mm(outline, console.plan(task.protocol_of(tighter), world["console"])["realised"], grid) > task.COVER_SLACK_MM


def test_the_search_starts_where_no_pixel_separates(instance):
    """``oracle.START_PIXEL``: a pixel of 0.7 x spacing and coarser shows no dip of one half, on the engine and for two points."""
    cfg, world, engine = instance["cfg"], instance["world"], instance["engine"]
    assert task.point_pair_dip(1 / oracle.START_PIXEL) < 0.2
    pe_dir = "y" if cfg["hidden"]["long_axis"] == "x" else "x"
    for i in task.PAIRS:
        h = cfg["hidden"]["pairs"][i]
        for share in (0.7, 0.85, 1.0):
            p = share * h["spacing_mm"]
            m = 2 * int(math.ceil(125 / p))
            rp = console.plan(task.protocol_of(dict(pe_dir=pe_dir, fov_mm=p * m, phase_fov=1.0, matrix=m)), world["console"])["realised"]
            assert task.separation(engine, rp, h["a_yx_mm"], h["b_yx_mm"]) < 0.25


# ---------------------------------------------------------------------------------------------- instances
def test_instances_are_reproducible_also_from_threads_and_hide_the_world():
    serial = {(lv, s): json.dumps(_cfg(lv, s), sort_keys=True) for lv in (1, 2, 3) for s in range(3)}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = dict(zip(serial, pool.map(lambda k: json.dumps(_cfg(*k), sort_keys=True), list(serial) * 2)))
    assert threaded == serial and len(set(serial.values())) == len(serial)
    for (lv, _), text in serial.items():
        cfg = json.loads(text)
        statement, public = task.statement(cfg), json.dumps(cfg["task"])
        for i in task.PAIRS:
            h = cfg["hidden"]["pairs"][i]
            assert task.SPACING_MM[0] <= h["spacing_mm"] <= task.SPACING_MM[1]
            for hidden in (f"{h['spacing_mm']:.2f}", f"{h['a_yx_mm'][0]:.2f}", f"{h['a_yx_mm'][1]:.2f}", f"{h['b_yx_mm'][1]:.2f}"):
                assert hidden not in statement and hidden not in public
        body = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)["body"]
        assert f"{body['a']:.1f}" not in statement and f"{body['b']:.1f}" not in statement
        assert ("long_axis" in cfg["task"]) == (lv != 3) and (cfg["task"].get("hints") == ["rule"]) == (lv == 1)
        assert ("How to do it" in statement) == (lv == 1)
        assert F.OWN_SCANS in statement and F.NOT_CHARGED in statement
        assert "seed" not in public and "hidden" not in public and "world_spec" not in public


def test_the_object_holds_what_the_task_says(instance):
    """Inserts inside the object, a pair inside its insert and its marked circle, the reference region in gel."""
    cfg, world = instance["cfg"], instance["world"]
    regions = {r["id"]: r for r in cfg["task"]["regions"]}
    assert list(regions) == ["A", "B", "C", "REF"] and world["names"] == ["body"] + [f"insert{k}" for k in range(9)]
    y, x = W.grid(world)
    signal = np.tensordot(np.array([world["tissues"][k]["pd"] for k in world["names"]], np.float32), world["weights"], 1)
    for i in task.PAIRS:
        h, r = cfg["hidden"]["pairs"][i], regions[i]
        for p in (h["a_yx_mm"], h["b_yx_mm"]):
            assert math.hypot(p[0] - r["y_mm"], p[1] - r["x_mm"]) + task.FEATURE_RADIUS * h["spacing_mm"] < r["radius_mm"]
            assert signal[(y - p[0]) ** 2 + (x - p[1]) ** 2 < 0.2].min() > 0.9                         # a feature: bright
        mid = [(h["a_yx_mm"][k] + h["b_yx_mm"][k]) / 2 for k in (0, 1)]
        assert signal[(y - mid[0]) ** 2 + (x - mid[1]) ** 2 < 0.2].max() < 0.02                         # plastic between the two
        ring = (np.hypot(y - r["y_mm"], x - r["x_mm"]) > 12) & (np.hypot(y - r["y_mm"], x - r["x_mm"]) < 18)
        assert signal[ring].max() < 0.02                                                               # ... and around them
    ref = regions["REF"]
    assert signal[(y - ref["y_mm"]) ** 2 + (x - ref["x_mm"]) ** 2 <= ref["radius_mm"] ** 2].min() > 0.6   # gel
    long_axis = cfg["hidden"]["long_axis"]
    box = W.extent_mm(world)
    assert (box["x"][1] - box["x"][0] > 1.2 * (box["y"][1] - box["y"][0])) == (long_axis == "x")
    for i in task.PAIRS:
        a, b = cfg["hidden"]["pairs"][i]["a_yx_mm"], cfg["hidden"]["pairs"][i]["b_yx_mm"]
        assert (a[0] == b[0]) == (long_axis == "x") and (a[1] == b[1]) == (long_axis == "y")


# ---------------------------------------------------------------------------------------------- control policies
def test_reference_passes_and_each_shortcut_fails():
    assert controls.REFERENCE == "reference" and set(controls.MUST_FAIL) == {"prior", "console_default", "fixed_recipe", "sweep", "wrong_units"}
    assert controls.ABLATIONS == {"pixel_is_spacing": (1, 2, 3), "fixed_axis": (3,)} and controls.REPORTED == ("half_spacing", "open_loop")
    assert set(controls.DESCRIPTIONS) == set(controls.policies())
    level, seed = 3, 0
    results = {name: policies.run(mrgym.make(ENV_ID, level=level, seed=seed), name) for name in controls.policies() if name != "open_loop"}
    ref = results["reference"]
    assert ref.reward == 1.0 and ref.info["policy_error"] is None and ref.info["score"]["error"] <= task.TOL - 1.0
    assert ref.info["budget"]["used_s"] <= task.BUDGET_S and ref.info["budget"]["scans"] == 1 and controls.open_loop is controls.reference
    for name in controls.MUST_FAIL + ("pixel_is_spacing",):
        assert results[name].reward == 0.0, name
    assert results["prior"].info["score"]["outcome"] == "no_measurement"
    assert results["wrong_units"].info["reason"] == "deliverable_refused"
    assert results["pixel_is_spacing"].info["score"]["outcome"] == "not_separated"
    assert results["console_default"].info["score"]["outcome"] == "over_time_cap"
    assert results["sweep"].info["budget"]["scans"] == 1                 # the budget pays one probe of its list
    # level 1: the reference follows the stated method (the same policy; the long axis is told there)
    assert policies.run(mrgym.make(ENV_ID, level=1, seed=1), "reference").reward == 1.0


def test_the_wrong_phase_encode_axis_is_too_slow_at_level_3():
    """Level 3: an object that lies along y needs the phase-encode axis along x."""
    seed = next(s for s in range(20) if _cfg(3, s)["hidden"]["long_axis"] == "y")
    r = policies.run(mrgym.make(ENV_ID, level=3, seed=seed), "fixed_axis")
    assert r.reward == 0.0 and r.info["score"]["outcome"] in ("too_slow", "over_time_cap")
    assert policies.run(mrgym.make(ENV_ID, level=3, seed=seed), "reference").reward == 1.0


def test_the_constant_recipe_is_tuned_by_the_verifiers_rule_and_plays_without_looking():
    cfgs = [_cfg(1, f"0:{i}", "tune") for i in range(2)]
    recipe = controls.tune(cfgs)
    assert set(recipe) == {"pe_dir", "pixel_mm", "lines", "tuned_pass_rate", "tuned_pair_pass_rate"} and 0 <= recipe["tuned_pass_rate"] <= 1
    assert recipe in [dict(c, tuned_pass_rate=recipe["tuned_pass_rate"], tuned_pair_pass_rate=recipe["tuned_pair_pass_rate"]) for c in controls.recipe_candidates()]
    env = mrgym.make(ENV_ID, level=1, seed=0)
    r = policies.run(env, "fixed_recipe", recipe=recipe)
    assert r.info["budget"]["scans"] == 1 and r.info["policy_error"] is None and r.info["score"]["checks"]["deliverable_valid"]


def test_the_card_is_complete():
    card = task.card()
    assert card["id"] == ENV_ID and set(card["levels"]) == {1, 2, 3} and card["practice_levels"] == []
    assert [c["name"] for c in card["controls"]] == ["reference", *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert set(task.COMPETENCIES) <= {r[0] for r in coverage._ROWS} and card["competencies"] == list(task.COMPETENCIES)
    assert mrgym.spec(ENV_ID).kind in ("candidate", "admitted") and task.OUTCOMES == ("object_cut_off", "not_separated", "snr_too_low", "over_time_cap", "too_slow")
