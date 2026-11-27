"""``Scanner/FoldAway-v0``: the fold rule checked against a brute-force overlap count and against the engine, the twin
without wrap-around, the verifier, the oracle, the instances and the control policies. Expected numbers are worked
out here (interval arithmetic, the periodic sum of an image, the console's cost rule), not taken from ``task.score``."""
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
from mrgym.envs.fold_away import controls, oracle, task
from mrgym.envs.nowrap import task as N
from mrgym.scanner import console, kspace
from mrgym.scanner import session as core
from mrgym.scanner import world as W

ENV_ID = "Scanner/FoldAway-v0"


def _cfg(level, seed, split="train"):
    return keys.make_instance(task, derive_seed(ENV_ID, level, split, seed, environ={}), level, "t", split)


def _instance(level, seed=0):
    cfg = _cfg(level, seed)
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    engine = kspace.Engine(world)
    return dict(cfg=cfg, world=world, engine=engine, support=N.outline(world, N.SUPPORT_THRESHOLD), outline=N.outline(world),
                solution=oracle.solve(cfg, world, engine))


@pytest.fixture(scope="module")
def upright():
    return _instance(2)


@pytest.fixture(scope="module")
def turned():
    return _instance(3)


# ---------------------------------------------------------------------------------------------- physics, checked independently
def test_the_fold_rule_against_a_count_of_overlapping_copies():
    """No package code on the right-hand side: copies of the interval (o_lo, o_hi) at every non-zero multiple of L,
    tested for overlap with (r_lo, r_hi) one by one."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        o_lo, o_hi = sorted(rng.uniform(-120, 120, 2))
        r_lo = rng.uniform(o_lo, o_hi)
        r_hi = rng.uniform(r_lo, o_hi)                                   # a region inside the object
        rule = task.fold_free_extent(o_lo, o_hi, r_lo, r_hi)
        assert rule == max(o_hi - r_lo, r_hi - o_lo) and rule <= o_hi - o_lo
        for L, free in ((rule + 1e-6, True), (rule - 1e-6, False), (rule + 30.0, True), (0.6 * rule, False)):
            overlap = any(o_lo + k * L < r_hi and o_hi + k * L > r_lo for k in range(-8, 9) if k)
            assert overlap is (not free)


def _probe_world():
    """The world of the design probe: a 180 x 100 mm ellipse (long axis along y) with a 24 mm disc at y = 30 mm."""
    ph = W.shapes_phantom([("gm", lambda y, x: (np.hypot(y / 90.0, x / 50.0) - 1) * 50.0), ("csf", lambda y, x: np.hypot(y - 30.0, x) - 12.0)], b0_t=1.5)
    world = W.flat_world(ph)
    return world, kspace.Engine(world)


def test_the_design_probe_20_5_s_against_29_5_s():
    """The probe the family was designed from: phase encoding along x at a pixel of 2 mm. Holding the whole object takes
    52 lines (29.5 s); the fold rule plus three pixels takes 34 (20.5 s) and leaves the region clean; 10 mm less folds
    onto it."""
    world, engine = _probe_world()
    region = dict(id="R", y_mm=30.0, x_mm=0.0, radius_mm=10.0)
    rule = task.fold_free_extent(-50.0, 50.0, -12.0, 12.0)              # the object and the disc along x
    assert rule == 62.0

    def run(L):
        lines = 2 * math.ceil(L / 4.0)
        prot = dict(sequence="se", tr_ms=500.0, bandwidth_hz_per_px=300.0, pe_dir="x", fov_mm=184.0, matrix=92, phase_fov=lines / 92)
        pl = console.plan(prot, world["console"])
        return pl["realised"]["lines"], pl["scan_s"], task.aliased_share(engine, pl["realised"], region)
    whole, fold, short = run(104.0), run(rule + 6.0), run(rule - 10.0)
    assert whole[:2] == (52, 29.5) and fold[:2] == (34, 20.5) and short[:2] == (26, 16.5)
    assert whole[2] < 0.02 and fold[2] < task.WRAP_MAX and short[2] > 1.0
    assert fold[2] == pytest.approx(0.0385, abs=0.01)                    # the probe: 3.85 % (against a twin with twice the lines)


def test_the_delivered_image_is_the_twin_added_to_itself_at_shifts_of_the_phase_field_of_view(upright):
    """Aliasing as a second method: sample k-space n times as densely and reconstruct the whole encoded field of view;
    the delivered image is that image summed over shifts of L. The twin of the verifier is the centre of it."""
    world, engine, support = upright["world"], upright["engine"], upright["support"]
    region = upright["cfg"]["task"]["regions"][0]
    for pe_dir in ("y", "x"):
        g = dict(pe_dir=pe_dir, phase_fov=0.375, centre_y_mm=region["y_mm"], centre_x_mm=region["x_mm"])
        rp = console.plan(task.protocol_of(g), world["console"])["realised"]
        twin = task.unwrapped(rp, world, support)
        factor = twin["lines_full"] // rp["lines_full"]
        assert factor >= 2 and twin["lines_full"] == factor * rp["lines_full"] and twin["matrix_pe"] == rp["matrix_pe"]
        delivered, clean = engine.acquire(rp)[0], engine.acquire(twin)[0]
        whole = engine.acquire(dict(twin, matrix_pe=twin["lines_full"], fov_pe_mm=rp["fov_pe_mm"] * factor))[0]   # nothing cropped
        whole = whole if pe_dir == "y" else whole.T
        n = rp["matrix_pe"]
        lo = whole.shape[0] // 2 - n // 2
        np.testing.assert_allclose(whole[lo:lo + n], clean if pe_dir == "y" else clean.T, rtol=0, atol=1e-9 * np.abs(whole).max())
        folded = sum(np.roll(whole, k * n, axis=0) for k in range(factor))[lo:lo + n]
        np.testing.assert_allclose(folded, delivered if pe_dir == "y" else delivered.T, rtol=0, atol=1e-9 * np.abs(whole).max())
        # the encoded field of view of the twin holds every spin from the centre of the field of view
        v, u = N.in_frame(support, core.centre_yx(rp), 0.0)
        assert np.abs(v if pe_dir == "y" else u).max() < factor * rp["fov_pe_mm"] / 2


def test_scan_time_grows_with_the_phase_encode_lines_only(upright):
    exam, region = upright["world"]["console"], upright["cfg"]["task"]["regions"][0]
    assert task.PIXEL_MM == 1.0 and oracle.MIN_LINES == 48 == controls.MIN_LINES
    for phase_fov, lines in ((0.25, 48), (0.5, 96), (0.4375, 84), (1.0, 192)):
        for pe_dir in ("y", "x"):
            pl = console.plan(task.protocol_of(dict(pe_dir=pe_dir, phase_fov=phase_fov, centre_y_mm=region["y_mm"], centre_x_mm=region["x_mm"])), exam)
            assert pl["realised"]["lines"] == lines and pl["scan_s"] == pytest.approx(3.0 + 0.5 * (1 + lines))
            assert pl["realised"]["fov_pe_mm"] == pytest.approx(lines * 1.0)


# ---------------------------------------------------------------------------------------------- the verifier
def test_inside_means_inside_the_pixels_of_the_image():
    region = dict(id="R", y_mm=10.0, x_mm=-20.0, radius_mm=5.0)

    def over(**g):
        rp = console.plan(task.protocol_of(dict(dict(pe_dir="y", phase_fov=0.25, centre_y_mm=10.0, centre_x_mm=-20.0), **g)), None)["realised"]
        return task.region_overhang_mm(region, rp)
    # 48 pixels of 1 mm about the centre reach from -24.5 to +23.5 mm; the circle needs +-5 mm
    assert over() == pytest.approx(5.0 - 23.5) and over(centre_y_mm=10.0 - 18.5) == pytest.approx(0.0)
    assert over(centre_y_mm=10.0 - 18.51) > 0 and over(centre_y_mm=10.0 + 19.5) == pytest.approx(0.0) and over(centre_y_mm=10.0 + 19.51) > 0
    assert over(centre_x_mm=-20.0 + 91.5) == pytest.approx(0.0) and over(centre_x_mm=-20.0 - 90.5) == pytest.approx(0.0)   # the readout: 192 mm
    assert over(pe_dir="x", centre_x_mm=-20.0 - 18.51) > 0 and over(pe_dir="x", centre_y_mm=10.0 - 80.0) < 0
    turned = console.plan(task.protocol_of(dict(pe_dir="y", phase_fov=0.25, centre_y_mm=10.0, centre_x_mm=-20.0, rotation_deg=90.0)), None)["realised"]
    moved = dict(turned, centre_x_mm=-20.0 - 18.51)                      # rows of a field of view turned by 90 degrees run along +x
    assert task.region_overhang_mm(region, turned) == pytest.approx(5.0 - 23.5) and task.region_overhang_mm(region, moved) > 0


def test_the_pass_rule_requirement_by_requirement(upright):
    cfg, world, engine, sol = upright["cfg"], upright["world"], upright["engine"], upright["solution"]
    good, best = sol["argmin"], sol["regions"]
    s = task.score(cfg, good, world=world, engine=engine)
    assert s["passed"] and s["error"] == 0.0 and s["diagnostic"] == ""
    assert list(s["checks"]) == ["deliverable_valid"] + [f"{r}_{i}" for r in task.REQUIREMENTS for i in task.REGIONS]
    assert all(v["aliased"] <= task.WRAP_MAX for v in s["regions"].values())
    g = good["B"]

    def with_b(**change):
        return task.score(cfg, dict(good, B=dict(g, **change)), world=world, engine=engine)

    def failing(score):
        return [k for k, ok in score["checks"].items() if not ok]
    region = next(r for r in cfg["task"]["regions"] if r["id"] == "B")
    full = with_b(phase_fov=1.0)                                         # nothing folds at all: valid and slow
    assert failing(full) == ["scan_time_within_factor_B"] and task.outcome(full) == "too_slow" and full["regions"]["B"]["aliased"] <= task.WRAP_MAX
    assert full["error"] == pytest.approx(99.5 / best["B"]["scan_s"] - 1.0)
    small = with_b(phase_fov=0.25)                                       # the field of view holds the region only
    assert failing(small) == ["no_wrap_on_region_B"] and task.outcome(small) == "wrap_on_region" and small["regions"]["B"]["aliased"] > 0.5
    ro = "centre_x_mm" if g["pe_dir"] == "y" else "centre_y_mm"
    away = with_b(**{ro: g[ro] - math.copysign(100.0, g[ro])})          # moved 100 mm along the readout: the region leaves the image
    assert "region_inside_image_B" in failing(away) and task.outcome(away) == "region_cut_off"
    # the aliasing depends on the phase field of view alone: the same lines, the field of view moved along the phase
    # axis by whole pixels until the region lies at the rim of the image
    pe = "centre_y_mm" if g["pe_dir"] == "y" else "centre_x_mm"
    lines = best["B"]["lines"]
    pixels = math.floor((0.5 * lines * task.PIXEL_MM - region["radius_mm"] - 3.0) / task.PIXEL_MM)
    shifted = with_b(**{pe: g[pe] - math.copysign(pixels * task.PIXEL_MM, g[pe])})
    assert shifted["passed"] and shifted["regions"]["B"]["aliased"] == pytest.approx(s["regions"]["B"]["aliased"], abs=2e-3)   # (the twin differs in its far tails)
    assert shifted["regions"]["B"]["overhang_mm"] > -5.0 > s["regions"]["B"]["overhang_mm"]


def test_the_time_factor_is_inclusive_and_measured_against_the_optimum_of_each_region(upright, monkeypatch):
    cfg, world, engine, good = upright["cfg"], upright["world"], upright["engine"], upright["solution"]["argmin"]
    plain = task.optimum(cfg, world, engine)
    scan_s = {i: console.plan(task.protocol_of(good[i]), world["console"])["scan_s"] for i in task.REGIONS}
    monkeypatch.setattr(task, "optimum", lambda c, w, e: {i: dict(plain[i], scan_s=scan_s[i] / task.TOL) for i in task.REGIONS})
    assert task.score(cfg, good, world=world, engine=engine)["passed"]
    monkeypatch.setattr(task, "optimum", lambda c, w, e: {i: dict(plain[i], scan_s=scan_s[i] / task.TOL * (1 - 1e-6 if i == "C" else 1)) for i in task.REGIONS})
    s = task.score(cfg, good, world=world, engine=engine)
    assert not s["passed"] and [k for k, ok in s["checks"].items() if not ok] == ["scan_time_within_factor_C"]
    assert s["diagnostic"] == "region C: the scan is slower than allowed"


def test_refused_deliverables(upright, turned):
    cfg, exam, good = upright["cfg"], upright["world"]["console"], upright["solution"]["argmin"]
    assert task.check(cfg, good, exam) is None and task.fields(cfg) == ("pe_dir", "phase_fov", "centre_y_mm", "centre_x_mm")
    one = '{"pe_dir", "phase_fov", "centre_y_mm", "centre_x_mm"}'
    shape = "the deliverable is {" + ", ".join(f'"{i}": {one}' for i in task.REGIONS) + "}"
    for bad in (None, [], {k: v for k, v in good.items() if k != "C"}, dict(good, E=good["A"]), dict(good, A=dict(good["A"], rotation_deg=10.0)),
                dict(good, A={k: v for k, v in good["A"].items() if k != "pe_dir"}), dict(good, B=0.4)):
        assert task.check(cfg, bad, exam) == shape
    mm = task.check(cfg, dict(good, B=dict(good["B"], phase_fov=84.0)), exam)
    assert mm.startswith("the console rejects the geometry of region B: ") and mm.endswith("(phase_fov is a fraction of fov_mm, not millimetres or percent)")
    assert "centre_y_mm" in task.check(cfg, dict(good, A=dict(good["A"], centre_y_mm=400.0)), exam)
    s = task.score(cfg, dict(A=good["A"]), world=upright["world"], engine=upright["engine"])
    assert not s["passed"] and s["checks"] == dict(deliverable_valid=False) and s["error"] is None
    # level 3 asks for the rotation as well
    cfg3, good3 = turned["cfg"], turned["solution"]["argmin"]
    assert task.fields(cfg3)[-1] == "rotation_deg" and task.check(cfg3, good3, exam) is None
    assert task.check(cfg3, {i: {k: v for k, v in g.items() if k != "rotation_deg"} for i, g in good3.items()}, exam).startswith("the deliverable is")
    assert set(task.submit_schema(cfg3)["properties"]["A"]["properties"]) == set(task.fields(cfg3))
    assert set(task.submit_schema(cfg)["properties"]["A"]["properties"]) == set(task.fields(cfg))


# ---------------------------------------------------------------------------------------------- the oracle
def test_the_oracle_is_independent_of_the_control_policies():
    assert O.independence_problems("mrgym.envs.fold_away") == []


@pytest.mark.parametrize("which", ["upright", "turned"])
def test_the_oracle_optimum_passes_and_everything_cheaper_in_its_search_fails(which, request):
    inst = request.getfixturevalue(which)
    cfg, world, engine, sol = inst["cfg"], inst["world"], inst["engine"], inst["solution"]
    assert sol["value"] == 0.0 and sol["method"] == "exhaustive" and set(sol["regions"]) == set(task.REGIONS)
    assert task.score(cfg, sol["argmin"], world=world, engine=engine)["passed"]
    regions = {r["id"]: r for r in cfg["task"]["regions"]}
    for i, p in sol["regions"].items():
        assert p["scan_s"] == pytest.approx(3.0 + 0.5 * (1 + p["lines"]))
        assert (p["geometry"]["centre_y_mm"], p["geometry"]["centre_x_mm"]) == (regions[i]["y_mm"], regions[i]["x_mm"])
        assert ("rotation_deg" in p["geometry"]) == (which == "turned")
    cert = sol["certificate"]
    assert cert and all(c["cost"] <= sol["regions"][c["region"]]["scan_s"] and c["fails"] == "no_wrap_on_region" for c in cert)
    for c in cert[:2] + cert[-2:]:
        assert task.aliased_share(engine, console.plan(task.protocol_of(c["deliverable"]), world["console"])["realised"],
                                  regions[c["region"]], inst["support"]) > task.WRAP_MAX
    # the wrong upright axis is not the optimum of every region: the direction is a decision
    if which == "upright":
        assert {p["geometry"]["pe_dir"] for p in sol["regions"].values()} <= {"y", "x"}
    else:
        assert any(abs(p["geometry"]["rotation_deg"]) not in (0.0, 90.0) for p in sol["regions"].values())


def test_the_edges_of_the_search(upright):
    """The two assumptions of ``oracle.candidates``, on the engine: ``START_MARGIN_PX`` below the line-by-line fold-free
    extent a geometry fails (a copy lies two pixels deep in the region); ``PASS_MARGIN_PX`` above it, it passes."""
    cfg, world, engine, support = upright["cfg"], upright["world"], upright["engine"], upright["support"]
    grid = world["extent_mm"] / world["n"]
    assert (oracle.START_MARGIN_PX, oracle.PASS_MARGIN_PX, oracle.TURNED_WITHIN_PX) == (-2.0, 2.0, 4.0)
    for region in cfg["task"]["regions"]:
        for pe_dir in ("y", "x"):
            e = oracle.extent(upright["outline"], region, pe_dir, 0.0, grid)
            below = 2 * math.floor((e + oracle.START_MARGIN_PX * task.PIXEL_MM) / task.PIXEL_MM / 2)
            above = 2 * math.ceil((e + oracle.PASS_MARGIN_PX * task.PIXEL_MM) / task.PIXEL_MM / 2)
            for lines, passes in ((below, False), (above, True), (above + 2, True), (above + 6, True)):
                if not oracle.MIN_LINES <= lines <= task.TARGET["matrix"]:
                    continue
                rp = console.plan(task.protocol_of(oracle.geometry(cfg, region, pe_dir, 0.0, lines)), world["console"])["realised"]
                assert rp["lines"] == lines and (task.aliased_share(engine, rp, region, support) <= task.WRAP_MAX) is passes


def test_the_line_by_line_extent_of_a_circle_under_a_straight_edge():
    """Worked out by hand: an edge at v = 40 + k c (v along the phase axis, c across it) above a circle of radius R at
    the origin asks for max over c of 40 + k c + sqrt(R^2 - c^2) = 40 + R sqrt(1 + k^2); the rule with the highest
    point of the edge over the whole width of the circle asks for 40 + k R + R, which is more."""
    R, k = 6.0, 0.4
    c = np.linspace(-20.0, 20.0, 4001)
    pts = np.concatenate([np.stack([40.0 + k * c, c], 1), np.stack([-30.0 + 0 * c, c], 1)])
    region = dict(id="R", y_mm=0.0, x_mm=0.0, radius_mm=R)
    got = oracle.extent(pts, region, "y", 0.0, 0.0)
    assert got == pytest.approx(40.0 + R * math.sqrt(1 + k * k), abs=1e-3)
    assert task.fold_free_extent(-30.0, 40.0 + k * R, -R, R) == pytest.approx(40.0 + k * R + R) and got < 40.0 + k * R + R
    assert oracle.extent(pts[:, ::-1], region, "x", 0.0, 0.0) == pytest.approx(got)                    # the same along x
    turned = oracle.extent(pts @ np.array([[0.0, 1.0], [-1.0, 0.0]]), region, "y", 90.0, 0.0)      # the object turned by 90 degrees, and the field of view
    assert turned == pytest.approx(got, abs=1e-3)


# ---------------------------------------------------------------------------------------------- instances
def test_instances_are_reproducible_also_from_threads_and_hide_the_world():
    serial = {(lv, s): json.dumps(_cfg(lv, s), sort_keys=True) for lv in (1, 2, 3) for s in range(3)}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = dict(zip(serial, pool.map(lambda k: json.dumps(_cfg(*k), sort_keys=True), list(serial) * 2)))
    assert threaded == serial and len(set(serial.values())) == len(serial)
    for (lv, _), text in serial.items():
        cfg = json.loads(text)
        statement, public = task.statement(cfg), json.dumps(cfg["task"])
        body = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)["body"]
        for hidden in (f"{body['a']:.2f}", f"{body['b']:.2f}", f"{body['y']:.2f}", f"{body['x']:.2f}", f"{body['rot']:.2f}"):
            assert hidden not in statement and hidden not in public
        assert cfg["task"]["rotation"] is (lv == 3) and (cfg["task"].get("hints") == ["rule"]) == (lv == 1)
        assert ("How to do it" in statement) == (lv == 1) and ("rotation_deg" in statement) == (lv == 3)
        assert F.OWN_SCANS in statement and F.NOT_CHARGED in statement
        assert "seed" not in public and "world_spec" not in public and "hidden" not in cfg
        # every region lies inside the object, REGION_INSET_MM from its edge, and the regions are apart
        c, s = math.cos(math.radians(body["rot"])), math.sin(math.radians(body["rot"]))
        centres = [(r["y_mm"], r["x_mm"]) for r in cfg["task"]["regions"]]
        assert [r["id"] for r in cfg["task"]["regions"]] == list(task.REGIONS)
        for y, x in centres:
            u, v = (y - body["y"]) * c - (x - body["x"]) * s, (y - body["y"]) * s + (x - body["x"]) * c      # along the long axis, across it
            assert abs(u) <= task.REGION_ALONG * body["a"] + 0.1
            ring = np.linspace(0, 2 * np.pi, 64)
            reach = task.REGION_RADIUS_MM + task.REGION_INSET_MM - 0.2
            assert np.all(((u + reach * np.cos(ring)) / body["a"]) ** 2 + ((v + reach * np.sin(ring)) / body["b"]) ** 2 < 1)
        assert min(math.dist(p, q) for i, p in enumerate(centres) for q in centres[i + 1:]) >= task.REGION_APART_MM - 0.2
        if lv != 3:
            assert body["rot"] in (0.0, 90.0)


def test_the_region_is_signal_inside_a_larger_object(upright):
    world = upright["world"]
    assert world["names"] == ["body"]
    box = W.extent_mm(world)
    assert min(box["y"][1] - box["y"][0], box["x"][1] - box["x"][0]) > 90.0      # larger than the fastest phase fields of view
    assert all(p["lines"] * task.PIXEL_MM < min(box["y"][1] - box["y"][0], box["x"][1] - box["x"][0]) for p in upright["solution"]["regions"].values())


# ---------------------------------------------------------------------------------------------- control policies
def test_reference_passes_and_each_shortcut_fails():
    assert controls.REFERENCE == "reference" and set(controls.MUST_FAIL) == {"prior", "fixed_recipe", "whole_object", "region_only", "sweep", "wrong_units"}
    assert controls.ABLATIONS == {"fixed_axis": (1, 2, 3), "whole_extent": (1, 2, 3), "no_turn": (3,), "no_margin": (2, 3)}
    assert controls.REPORTED == ("stated_method", "open_loop")
    assert set(controls.DESCRIPTIONS) == set(controls.policies())
    oblique = next(k for k in range(40) if 30.0 <= W.make_world(_cfg(3, k)["seed"], _cfg(3, k)["world_spec"], layout_only=True)["body"]["rot"] % 90.0 <= 60.0)
    results = {name: policies.run(mrgym.make(ENV_ID, level=3, seed=oblique), name) for name in controls.policies() if name != "open_loop"}
    ref = results["reference"]
    assert ref.reward == 1.0 and ref.info["policy_error"] is None and ref.info["score"]["error"] <= task.TOL - 1.0
    assert ref.info["budget"]["used_s"] <= task.LEVELS[3]["budget_s"] and ref.info["budget"]["scans"] == 1 and controls.open_loop is controls.reference
    for name in controls.MUST_FAIL:
        assert results[name].reward == 0.0, name
    assert results["prior"].info["score"]["outcome"] == "no_measurement"
    assert results["wrong_units"].info["reason"] == "deliverable_refused"
    # the two shortcuts the family is built against: the NoWrap answer is clean and slow, the region-only answer is fast and folded
    whole, only = results["whole_object"].info["score"], results["region_only"].info["score"]
    assert whole["outcome"] == "too_slow" and all(whole["checks"][f"no_wrap_on_region_{i}"] for i in task.REGIONS)
    assert only["outcome"] == "wrap_on_region" and all(only["checks"][f"scan_time_within_factor_{i}"] for i in task.REGIONS)
    assert results["no_turn"].reward == 0.0 and results["fixed_axis"].reward == 0.0       # an oblique object: the field of view has to be turned
    # level 1: the reference follows the stated method, and the same two shortcuts fail the same way
    level1 = {name: policies.run(mrgym.make(ENV_ID, level=1, seed=0), name) for name in ("reference", "whole_object", "region_only")}
    assert level1["reference"].reward == 1.0 and level1["reference"].info["score"]["error"] <= task.TOL - 1.0
    assert level1["whole_object"].info["score"]["outcome"] == "too_slow" and level1["region_only"].info["score"]["outcome"] == "wrap_on_region"


def test_the_recipe_from_the_task_text_is_tuned_by_the_verifiers_rule_and_plays_without_looking():
    cfgs = [_cfg(1, f"0:{i}", "tune") for i in range(2)]
    recipe = controls.tune(cfgs)
    assert set(recipe) == {"direction", "lines", "per_mm", "tuned_pass_rate", "tuned_region_pass_rate"} and 0 <= recipe["tuned_pass_rate"] <= 1
    assert {k: recipe[k] for k in ("direction", "lines", "per_mm")} in controls.recipe_candidates(False)
    r = policies.run(mrgym.make(ENV_ID, level=1, seed=0), "fixed_recipe", recipe=recipe)
    assert r.info["budget"]["scans"] == 1 and r.info["policy_error"] is None and r.info["score"]["checks"]["deliverable_valid"]
    t = cfgs[0]["task"]
    g = controls.recipe_geometry(dict(direction="across", lines=60, per_mm=0.0), t, t["regions"][0])
    assert g["phase_fov"] == pytest.approx(60 / 192, abs=1e-6) and (g["centre_y_mm"], g["centre_x_mm"]) == (t["regions"][0]["y_mm"], t["regions"][0]["x_mm"])
    assert "rotation_deg" not in g and {"rotation_deg"} <= set(controls.recipe_geometry(dict(direction=30.0, lines=60), _cfg(3, 0)["task"], _cfg(3, 0)["task"]["regions"][0]))


def test_the_card_is_complete():
    card = task.card()
    assert card["id"] == ENV_ID and set(card["levels"]) == {1, 2, 3} and card["practice_levels"] == []
    assert [c["name"] for c in card["controls"]] == ["reference", *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert set(task.COMPETENCIES) <= {r[0] for r in coverage._ROWS} and card["competencies"] == list(task.COMPETENCIES)
    assert mrgym.spec(ENV_ID).kind in ("candidate", "admitted") and task.OUTCOMES == ("region_cut_off", "wrap_on_region", "too_slow")
