"""``Scanner/NoWrap-v1``: what the scoring table (``tests/scoring/nowrap.py``) does not cover.

  * the instances of the table keep the margins its arithmetic needs
  * the oracle finds the optimum worked out by hand, its answer passes with no excess, and nothing cheaper passes:
    not on the raster of the search, not at other rotations, not with smaller pixels
  * the re-scan check: no scan is made when the encoded field of view holds every spin, and a cut object wraps
  * the reference passes through the tools, and each ablation fails on the check its missing decision would have met
  * levels: 0 states the procedure and has the localiser in the log; 1 states it; 2 and 3 state nothing
  * the constant recipe is tuned by the verifier's rule and plays without looking
"""
import math
import re

import numpy as np
import pytest

import mrgym
import scoring
from mrgym import gate, policies
from mrgym.envs.nowrap import controls, oracle, task
from mrgym.scanner import console, session
from scoring import nowrap as table

ENV_ID = "Scanner/NoWrap-v1"


@pytest.fixture(scope="module")
def l3():
    case = scoring.case(table, "L3")
    return case, oracle.solve(case.cfg, case.world, case.engine)


def test_the_instances_of_the_table_keep_their_margins():
    """The verifier's outline lives on 0.4 mm cells; the hand values of the table must not depend on that."""
    for key in table.INSTANCES:
        case = scoring.case(table, key)
        b = table.body(case)
        x = (2 * b["short"] - 2 * table.SLACK) / table.PIXEL / 2             # line pairs the object needs, before rounding up
        assert 0.15 < x - math.floor(x) < 0.85, (key, x)                      # 0.3 mm either way moves no line pair
        assert table.fewest_lines(case) >= 42, key                           # the oversampled row is slow only above 40 lines
        assert b["long"] / b["short"] > 1.5
        pts = task.outline(case.world)
        v, u = task.in_frame(pts, (b["y"], b["x"]), b["turn"])
        for reach, semi, cells in ((v.max(), b["short"], 0.55), (-v.min(), b["short"], 0.55), (u.max(), b["long"], 0.9), (-u.min(), b["long"], 0.9)):
            assert -cells < reach - semi <= 0.05      # the verifier's outline: grid points inside the scaled ellipse (its tip is narrow)
    b = table.body(scoring.case(table, "L2"))
    c, s = math.cos(math.radians(b["turn"])), math.sin(math.radians(b["turn"]))
    assert abs((b["y"] - 1.0) * c + (b["x"] - 1.0) * s) > 8.0                # off the isocentre along the short axis: the centred row wraps
    b = table.body(scoring.case(table, "L3"))
    assert 15.0 < abs(b["turn"]) < 75.0                                       # turned: neither image axis is the short axis


def test_the_oracle_finds_the_optimum_worked_out_by_hand(l3):
    case, sol = l3
    b = table.body(case)
    assert sol["method"] == "exhaustive" and sol["value"] == 0.0 and oracle.REGRET_KIND == "difference"
    assert sol["lines"] == table.fewest_lines(case) and sol["scan_s"] == table.scan_s(sol["lines"])
    assert abs((sol["argmin"]["rotation_deg"] - b["turn"] + 90.0) % 180.0 - 90.0) < 15.0    # the phase axis lies across the short axis
    score = case.task.score(case.cfg, sol["argmin"], world=case.world, engine=case.engine)
    assert score["passed"] and score["error"] == 0.0 and score["lines"] == sol["lines"] and score["wrap_fraction"] <= task.WRAP_LIMIT
    assert sol["grid"]["size"] > 4 * 180 > sol["grid"]["evaluated"] >= 1 and sol["directions"] > 180 and sol["seconds"] > 0
    assert all(c["fails"] == "whole_object_inside" and c["lines"] <= sol["lines"] for c in sol["certificate"])
    assert sol["lines"] - 2 in {c["lines"] for c in sol["certificate"]}       # two lines fewer cut the object at the best rotation
    assert task.optimum(case.cfg, case.world, case.engine) == dict(scan_s=sol["scan_s"], lines=sol["lines"], geometry=sol["argmin"])


def test_nothing_cheaper_than_the_optimum_passes(l3):
    """Minimality on a grid wider than the search: other rotations, smaller pixels, phase oversampling, and the
    centre moved. Every deliverable with a shorter scan fails a check (the cheap checks decide: no re-scan needed)."""
    case, sol = l3
    world, engine = case.world, case.engine
    shapes = dict(object=task.outline(world), support=task.outline(world, task.SUPPORT_THRESHOLD))
    vertices, grid_mm, tried = oracle.hull(shapes["object"]), world["extent_mm"] / world["n"], 0
    for rotation in np.arange(-87.5, 90.0, 2.5):
        for pixel in (2.0, 1.9, 1.8):
            for extra in (-6, -4, -2, 0, 2):
                g = oracle.geometry(vertices, float(rotation), pixel, grid_mm, extra)
                for change in ({}, dict(phase_oversampling=0.1), dict(centre_y_mm=g["centre_y_mm"] + 0.7)):
                    d = dict(g, **change)
                    pl = console.plan(task.protocol_of(d), world["console"])
                    if pl["accept"] and pl["scan_s"] < sol["scan_s"] - 1e-9:
                        tried += 1
                        assert oracle.first_failure(case.cfg, d, world, engine, shapes) is not None, d
    assert tried > 100


def test_the_search_tries_the_degree_raster_and_the_direction_across_every_hull_edge():
    square = np.array([[-10.0, -20.0], [-10.0, 20.0], [10.0, 20.0], [10.0, -20.0], [0.0, 0.0]])       # (y, x): 20 mm along y, 40 mm along x
    vertices = oracle.hull(square)
    assert len(vertices) == 4
    angles = oracle.directions(vertices)
    assert {0.0, 90.0, 45.0, -89.0} <= set(angles) and min(angles) > -90.0 and max(angles) == 90.0 and len(angles) == 180
    g = oracle.geometry(vertices, 0.0, 2.0, 0.0)                         # rows along y: 20 mm - 2 x 0.5 mm slack -> 10 lines
    assert round(g["phase_fov"] * g["matrix"]) == 10 and g["matrix"] == 40 and (g["centre_y_mm"], g["centre_x_mm"]) == (1.0, 1.0)
    turned = oracle.geometry(vertices, 90.0, 2.0, 0.0)                   # rows along x: 40 mm -> 20 lines
    assert round(turned["phase_fov"] * turned["matrix"]) == 20
    tilted = np.array([[0.0, 0.0], [30.0, 40.0], [24.0, 48.0], [-6.0, 8.0]])                     # a 50 x 10 mm rectangle turned by atan(4 / 3)
    assert any(abs(a - (-math.degrees(math.atan2(3, 4)))) < 1e-3 for a in oracle.directions(oracle.hull(tilted)))


def test_the_rescan_is_skipped_when_every_spin_is_encoded_and_a_cut_object_wraps(l3, monkeypatch):
    case, sol = l3
    safe = console.plan(task.protocol_of(dict(pe_dir="y", fov_mm=260.0, phase_fov=1.0, matrix=130, centre_y_mm=0.0, centre_x_mm=0.0,
                                              rotation_deg=0.0)), case.world["console"])["realised"]
    cut = console.plan(task.protocol_of(table.field(-10)(case)), case.world["console"])["realised"]
    wrapped = task.wrap_fraction(case.engine, cut)
    assert wrapped > 10 * task.WRAP_LIMIT
    over = console.plan(task.protocol_of(table.field(-10, oversampling=0.6)(case)), case.world["console"])["realised"]
    monkeypatch.setattr(case.engine, "clean_kspace", lambda rp: pytest.fail("a re-scan was made"))
    assert task.wrap_fraction(case.engine, safe) == 0.0 and task.wrap_fraction(case.engine, over) == 0.0


def test_inside_means_inside_the_pixels_of_the_image():
    """Pixel i of n sits at centre + (i - n / 2) x pixel: the image reaches from -FOV / 2 - pixel / 2 to FOV / 2 - pixel / 2."""
    rp = console.plan(task.protocol_of(dict(pe_dir="y", fov_mm=200.0, phase_fov=0.5, matrix=100, centre_y_mm=10.0, centre_x_mm=-20.0,
                                            rotation_deg=0.0)), dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5))["realised"]
    at = lambda y, x: task.overhang_mm(np.array([[y, x]]), rp, 0.0)         # noqa: E731
    assert at(10.0, -20.0) == pytest.approx(-49.0)                         # the centre: 49 mm from the nearest edge (high side of the rows)
    assert at(10.0 + 49.0, -20.0) == pytest.approx(0.0) and at(10.0 - 51.0, -20.0) == pytest.approx(0.0)       # phase: 100 mm
    assert at(10.0, -20.0 + 99.0) == pytest.approx(0.0) and at(10.0, -20.0 - 101.0) == pytest.approx(0.0)      # readout: 200 mm
    assert at(10.0 + 52.0, -20.0) == pytest.approx(3.0)
    assert task.overhang_mm(np.array([[10.0 + 52.0, -20.0]]), rp, 0.4) == pytest.approx(3.2)                    # a point stands for a grid cell
    turned = dict(rp, rotation_deg=90.0)                                   # rows now run along +x, columns along -y
    assert task.overhang_mm(np.array([[10.0, -20.0 + 49.0]]), turned, 0.0) == pytest.approx(0.0)
    assert task.overhang_mm(np.array([[10.0 + 101.0, -20.0]]), turned, 0.0) == pytest.approx(0.0)
    assert task.overhang_mm(np.array([[10.0, -20.0 + 300.0]]), rp, 0.0, phase_only=True) == pytest.approx(-49.0)


# ---------------------------------------------------------------------------------------------- policies through the tools
def _play(name, level, seed):
    env = mrgym.make(ENV_ID, level=level, seed=seed)
    r = policies.run(env, name)
    assert r.info["policy_error"] is None and r.info["replay"]["ok"]
    return r.info["score"], env


def test_reference_passes_and_each_ablation_fails_where_its_decision_is_missing():
    seed = table.INSTANCES["L3"]["seed"]
    score, env = _play("reference", 3, seed)
    assert score["passed"] and 0.0 <= score["error"] <= task.TOL - 1.0 and env.state()["scans"] == 1
    failed = lambda s: [k for k, ok in s["checks"].items() if not ok]       # noqa: E731
    assert failed(_play("size_only", 3, seed)[0])[0] == "whole_object_inside"            # the right size on the isocentre: cut off
    assert failed(_play("centred_only", 3, seed)[0]) == ["scan_time_within_factor"]      # holds the object from the isocentre: slow
    assert failed(_play("no_turn", 3, seed)[0]) == ["scan_time_within_factor"]           # upright box around a turned object: slow
    assert failed(_play("safe_max", 3, seed)[0]) == ["scan_time_within_factor"]
    assert failed(_play("prior", 3, seed)[0]) == ["scan_time_within_factor", "measured"]
    same, _ = _play("open_loop", 3, seed)
    assert same["passed"] and same["scan_s"] == score["scan_s"] and controls.open_loop is controls.reference


def test_the_family_lists_its_policies_and_where_each_ablation_must_lose():
    assert (controls.REFERENCE, controls.MUST_FAIL) == ("reference", ("prior", "console_default", "fixed_recipe", "brute_force", "wrong_units"))
    assert controls.ABLATIONS == {"size_only": (2, 3), "centred_only": (2, 3), "no_swap": (2, 3), "no_turn": (3,)}
    assert controls.REPORTED == ("open_loop", "safe_max")
    names = [controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert list(controls.policies()) == names and set(controls.DESCRIPTIONS) == set(names)
    assert gate.claim_levels(mrgym.spec(ENV_ID)) == [2, 3] and all(set(v) <= {2, 3} for v in controls.ABLATIONS.values())


# ---------------------------------------------------------------------------------------------- levels
def test_level_0_states_the_procedure_and_has_the_localiser_in_the_log():
    env = mrgym.make(ENV_ID, level=0, seed=4)
    obs, info = env.reset()
    assert env.task.LEVELS[0]["kind"] == env.task.LEVELS[1]["kind"] == "practice" and "kind" not in env.task.LEVELS[2]
    state = env.state()
    assert state["scans"] == 1 and state["scan_time_used_s"] == pytest.approx(10.524, abs=1e-3) and state["scan_time_left_s"] == pytest.approx(41.0 - 10.524, abs=1e-3)
    box = session.image_extent(env.stored_scans()[0])
    m = re.search(r"reach from y (-?[\d.]+) to (-?[\d.]+) mm and from x (-?[\d.]+) to (-?[\d.]+) mm", obs.text)
    y_lo, y_hi, x_lo, x_hi = map(float, m.groups())
    assert [y_lo, y_hi] == [round(v, 1) for v in box["y_mm"]] and [x_lo, x_hi] == [round(v, 1) for v in box["x_mm"]]
    assert "How to do it (this level states the procedure" in obs.text and "A localiser is already acquired: scan 0" in obs.text
    # the stated procedure, by hand
    p = 2.0
    e_y, e_x = y_hi - y_lo + 4.0, x_hi - x_lo + 4.0
    pe = "y" if e_y <= e_x else "x"
    e_pe, e_ro = (e_y, e_x) if pe == "y" else (e_x, e_y)
    lines = 2 * math.ceil(e_pe / (2 * p))
    matrix = max(2 * math.ceil(e_ro / (2 * p)), lines, 40)
    answer = dict(pe_dir=pe, fov_mm=p * matrix, phase_fov=lines / matrix, matrix=matrix, centre_y_mm=(y_lo + y_hi) / 2 + p / 2,
                  centre_x_mm=(x_lo + x_hi) / 2 + p / 2, rotation_deg=0)
    end = env.step({"name": "submit", "arguments": answer})
    assert end.terminated and end.reward == 1.0 and end.info["replay"]["ok"] and end.info["replay"]["scans"] == 1


def test_hints_by_level():
    texts = {}
    for level in (0, 1, 2, 3):
        env = mrgym.make(ENV_ID, level=level, seed=1)
        texts[level] = env.reset()[0].text
        assert "at most 10 % longer than the fastest geometry" in texts[level] and "`rotation_deg`" in texts[level]
        assert env.state()["scans"] == (1 if level == 0 else 0)
        assert env.episode_config()["task"].get("hints") == {0: ["rule", "first_measurement"], 1: ["rule"]}.get(level)
    assert "How to do it" in texts[1] and "Take a localiser that shows the whole object" in texts[1] and "already acquired" not in texts[1]
    assert all("How to do it" not in texts[lv] and "already acquired" not in texts[lv] for lv in (2, 3))
    assert all("the field of view is always centred" not in t.lower() for t in texts.values())      # the sentence of v0 is gone
    assert task.prefill(mrgym.make(ENV_ID, level=2, seed=1).task.make_instance("ab" * 16, 2), None) is None


def test_worlds_by_level():
    from mrgym.scanner import world as W
    seen = {lv: [] for lv in task.LEVELS}
    for lv in task.LEVELS:
        for i in range(12):
            cfg = task.make_instance(f"{lv:02x}{i:030x}", lv)
            seen[lv].append(W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)["body"])
    off = {lv: max(max(abs(b["y"]), abs(b["x"])) for b in bodies) for lv, bodies in seen.items()}
    assert off[0] <= 3.0 and off[1] <= 3.0 and 15.0 < off[2] <= 25.0 and 15.0 < off[3] <= 25.0
    assert all(b["rot"] in (0.0, 90.0) for lv in (0, 1, 2) for b in seen[lv]) and len({b["rot"] for b in seen[3]}) == 12
    for bodies in seen.values():
        assert all(74.0 <= b["a"] <= 90.0 and 1.6 <= b["a"] / b["b"] <= 2.0 for b in bodies)
        assert all(math.hypot(b["y"], b["x"]) + b["a"] < 126.0 for b in bodies)        # inside the master grid at every rotation
    assert task.LEVELS[0]["world_spec"] == task.LEVELS[1]["world_spec"]               # level 0 is the world of level 1


# ---------------------------------------------------------------------------------------------- the blind recipe
def test_the_constant_recipe_is_tuned_by_the_verifiers_rule_and_plays_without_looking():
    spec = mrgym.spec(ENV_ID)
    cfgs = [gate._instance(spec, 2, f"t:{i}", "tune") for i in range(4)]
    recipe = controls.tune(cfgs)
    assert set(recipe) == {"pe_dir", "lines", "rotation_deg", "tuned_pass_rate"} and 0.0 <= recipe["tuned_pass_rate"] <= 1.0
    assert {k: recipe[k] for k in ("pe_dir", "lines", "rotation_deg")} in controls.recipe_candidates()
    assert len(controls.recipe_candidates()) == 2 * 6 * 43
    sess = session.ScannerSession(cfgs[0])
    answer = controls.fixed_recipe(recipe)(session.LocalScanner(sess))
    assert sess.n_scans == 1 and (answer["centre_y_mm"], answer["centre_x_mm"]) == (0.0, 0.0)
    assert round(answer["phase_fov"] * answer["matrix"]) == recipe["lines"] and answer["pe_dir"] == recipe["pe_dir"]
    assert task.check(cfgs[0], answer, sess.exam) is None
