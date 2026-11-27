"""``Scanner/ShiftFree-v0`` beyond its scoring table (``tests/scoring/shift_free.py``): the two laws the family rests on,
checked against closed forms; instances; the task text; the control policies on one instance; the search of the verifier.

The scored episodes use the instances of the scoring table, so the verifier's search for the best setting (about ten
seconds per instance) is made once per process.
"""
import concurrent.futures
import json
import math

import numpy as np
import pytest

import mrgym
import scoring
from mrgym import policies
from mrgym.env import derive_seed
from mrgym.envs.shift_free import controls, task
from mrgym.scanner import console, kspace, tissue
from mrgym.scanner import world as W
from scoring import shift_free as table

ENV_ID = "Scanner/ShiftFree-v0"
CASE = lambda key="L2": scoring.case(table, key)       # noqa: E731


# ---------------------------------------------------------------------------------------------- the physics, independently
def _centroid(image, axis):
    """Centre of mass of a complex image along ``axis`` in pixels: of the signed signal (its phase taken out), so that
    the truncation ripples beside an edge cancel instead of adding up as magnitudes."""
    w = (image * np.exp(-1j * np.angle(image.sum()))).real.sum(axis=1 - axis)
    return float((w * np.arange(len(w))).sum() / w.sum())


@pytest.mark.parametrize("field_t", [1.5, 3.0])
def test_fat_is_displaced_by_frequency_over_bandwidth_pixels_along_the_readout(field_t):
    """The central law against its closed form. A fat disc alone in a perfect field: its centre of mass moves by
    (mean frequency of the six-peak fat spectrum) / (bandwidth per pixel) pixels along the readout axis, towards
    positive positions with ro_polarity 1, by nothing along the phase axis, and by nothing without the chemical shift.
    The mean frequency is computed here from the published table (Hamilton 2011, in ``tissue.FAT_PEAKS``)."""
    disc = lambda y, x: np.hypot(y - 6.0, x + 9.0) - 14.0                                 # noqa: E731
    world = W.flat_world(W.shapes_phantom([("subcut_fat", disc)], b0_t=field_t, n=320))
    shifted, still = kspace.Engine(world), kspace.Engine(world, effects=kspace.ALL - {"chemshift"})
    mean_hz = -sum(ppm * amp for ppm, amp in tissue.FAT_PEAKS) / sum(amp for _, amp in tissue.FAT_PEAKS) * 42.577478518 * field_t
    assert mean_hz == pytest.approx({1.5: 192.7, 3.0: 385.4}[field_t], abs=0.5)           # about 3.0 ppm, below the main line's 3.4
    for pe_dir, polarity, bandwidth in (("y", 1, 150.0), ("y", -1, 150.0), ("x", 1, 150.0), ("x", -1, 400.0), ("y", 1, 400.0)):
        protocol = dict(sequence="se", tr_ms=500.0, te_ms=30.0, fov_mm=240.0, matrix=96, bandwidth_hz_per_px=bandwidth,
                        pe_dir=pe_dir, ro_polarity=polarity)
        a, b = shifted.scan(protocol)["images"][0], still.scan(protocol)["images"][0]
        readout = 1 if pe_dir == "y" else 0                                               # image axis of the readout: x with pe_dir y
        assert _centroid(a, readout) - _centroid(b, readout) == pytest.approx(polarity * mean_hz / bandwidth, abs=0.003)
        assert _centroid(a, 1 - readout) - _centroid(b, 1 - readout) == pytest.approx(0.0, abs=0.001)
        assert task.approaching(pe_dir, polarity) == ("x" if pe_dir == "y" else "y") + ("-" if polarity == 1 else "+")
    # the figure every plan prints is the main line (3.4 ppm), a function of the field strength and the bandwidth only
    printed = console.plan(dict(sequence="se", tr_ms=500.0, bandwidth_hz_per_px=150.0), world["console"])["derived"]["water_fat_shift_px"]
    assert printed == pytest.approx(3.4e-6 * 42.577478518e6 * field_t / 150.0, abs=5e-4)


def test_noise_grows_with_the_square_root_of_the_bandwidth():
    """The other half of the trade: the standard deviation of a pixel of pure noise, measured on images of an empty
    scanner, is proportional to sqrt(bandwidth) and equals ``Engine.noise_sd`` (what the verifier divides by)."""
    world = W.flat_world(W.shapes_phantom([("wm", lambda y, x: np.hypot(y, x) - 1e-3)], n=64), sigma0=0.02)
    engine = kspace.Engine(world)
    sd = {}
    for bandwidth in (100.0, 400.0):
        rp = console.plan(dict(sequence="se", tr_ms=500.0, te_ms=30.0, fov_mm=240.0, matrix=96, bandwidth_hz_per_px=bandwidth),
                          world["console"])["realised"]
        image = engine.acquire(rp, np.random.default_rng(5))[0]
        sd[bandwidth] = float(np.concatenate([image.real[:30].ravel(), image.imag[:30].ravel()]).std())      # away from the speck of signal
        assert sd[bandwidth] == pytest.approx(engine.noise_sd(rp), rel=0.04)
    assert sd[400.0] / sd[100.0] == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------------------------- instances
def _instance(level, seed):
    return task.make_instance(derive_seed(ENV_ID, level, "train", seed, environ={}), level, task_id="t")


def test_an_instance_is_a_function_of_the_seed_also_under_threads():
    serial = {(lv, s): json.dumps(_instance(lv, s), sort_keys=True) for lv, s in ((1, 0), (2, 0), (3, 0), (3, 1))}
    assert len(set(serial.values())) == len(serial)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = dict(zip(serial, pool.map(lambda k: json.dumps(_instance(*k), sort_keys=True), list(serial) * 2)))
    assert threaded == serial


@pytest.mark.parametrize("level", [1, 2, 3])
def test_the_instance_follows_the_declared_geometry(level):
    """Gaps, sides and what the level leaves free are the ones the module documents; levels 1 and 2 share their worlds."""
    for seed in range(3):
        cfg = _instance(level, seed)
        t, sites = cfg["task"], cfg["truth"]["sites"]
        assert [r["id"] for r in t["regions"]] == ["R1", "R2", "R3"] == [s["id"] for s in sites]
        assert t["free"] == (["bandwidth_hz_per_px", "ro_polarity", "pe_dir"] if level == 3 else ["bandwidth_hz_per_px", "ro_polarity"])
        assert ("pe_dir" in t["target_scan"]) is (level != 3)
        assert len(cfg["world_spec"]["qa"]["inserts"]) == sum(len(s["gap_mm"]) for s in sites) == (12 if level == 3 else 6)
        for s in sites:
            gaps = sorted(s["gap_mm"].values())
            assert set(s["gap_mm"]) == set(task.LEVELS[level]["sides"])
            assert task.ROOM_MM[0] <= gaps[-1] <= task.ROOM_MM[1] and task.OTHER_MM <= gaps[0]
            assert gaps[-2] <= max(task.OTHER_SHARE * gaps[-1], task.OTHER_MM + 0.05) + 0.005      # the widest gap stands out
            assert all(v > 0 for v in t["snr_floor"].values())
            assert set(s["fat_t1_ms"]) == set(s["gap_mm"]) and all(382.0 * 0.85 <= v <= 382.0 * 1.15 for v in s["fat_t1_ms"].values())
        fats = [i["tissue"] for i in cfg["world_spec"]["qa"]["inserts"]]
        assert all(f["fat"] is True and (f["pd"], f["t2_ms"]) == (0.95, 68.0) for f in fats) and len({f["t1_ms"] for f in fats}) == len(fats)
        if level == 2:
            one = task.make_instance(cfg["seed"], 1, task_id="t")           # the same seed at level 1: the same world, with the method
            assert one["world_spec"] == cfg["world_spec"] and one["truth"] == cfg["truth"] and one["task"]["hints"] == ["method"]
            assert {k: v for k, v in one["task"].items() if k != "hints"} == t


# ---------------------------------------------------------------------------------------------- the task text
@pytest.mark.parametrize("level", [1, 2, 3])
def test_the_statement_is_made_of_the_public_task_and_names_no_hidden_value(level):
    cfg = _instance(level, 0)
    text = task.statement(cfg)
    assert text == task.statement(dict(task=cfg["task"]))                # nothing but the public task goes into it
    public = json.dumps(cfg["task"])
    assert "gap" not in public and "truth" not in public and "insert" not in public and "t1" not in public
    numbers = set(text.replace(",", " ").replace(";", " ").split())
    for s in cfg["truth"]["sites"]:
        for gap in s["gap_mm"].values():
            assert f"{gap:g}" not in numbers and f"{gap:.1f}" not in numbers, (gap, "is printed")
    assert ("How to do it" in text) is (level == 1)
    assert ("pe_dir y;" in text) is (level != 3) and ("ro_polarity and pe_dir" in text) is (level == 3)
    assert mrgym.envs.family.OWN_SCANS in text and mrgym.envs.family.NOT_CHARGED in text
    assert all(len(line) <= 170 for line in text.splitlines())


def test_the_stated_floor_is_the_ratio_at_600_hz_per_pixel_and_says_nothing_of_the_gaps():
    case = CASE()
    for region in case.cfg["task"]["regions"]:
        m = task.measure(case.cfg, region, dict(bandwidth_hz_per_px=600.0, ro_polarity=1), case.world, case.engine)
        assert 0.0 <= m["snr"] - case.cfg["task"]["snr_floor"][region["id"]] < 0.1 + 1e-9
        assert m["cover"] < 0.01                                         # at 600 Hz per pixel the fat has moved 0.9 mm: off every region


# ---------------------------------------------------------------------------------------------- refused deliverables
@pytest.mark.parametrize("level", [2, 3])
def test_a_deliverable_the_console_cannot_evaluate_is_refused_with_a_reason(level):
    cfg = _instance(level, 0)
    exam = dict(fov_mm=240.0, min_pixel_mm=1.0, b1rms_limit_ut=3.0, b0_t=3.0)
    one = dict(bandwidth_hz_per_px=150.0, ro_polarity=-1, **(dict(pe_dir="x") if level == 3 else {}))
    good = dict(scans={i: dict(one) for i in ("R1", "R2", "R3")})
    assert task.check(cfg, good, exam) is None
    assert task.target_protocol(cfg["task"], one) == dict(task.TARGET, pe_dir="x" if level == 3 else "y", bandwidth_hz_per_px=150.0, ro_polarity=-1)
    change = lambda **scan: dict(scans=dict(good["scans"], R2=dict(one, **scan)))        # noqa: E731
    for bad, why in ((None, "the deliverable is"), (dict(scans=[one] * 3), "the deliverable is"), (dict(good, note="x"), "the deliverable is"),
                     (dict(scans={k: v for k, v in good["scans"].items() if k != "R3"}), "the deliverable is"),
                     (dict(scans=dict(good["scans"], R4=one)), "the deliverable is"),
                     (dict(scans=dict(good["scans"], R2=150.0)), "the scan of R2 needs exactly"),
                     (change(averages=2), "the scan of R2 needs exactly"),
                     (change(bandwidth_hz_per_px=0.15), "the console rejects the scan of R2: bandwidth_hz_per_px=0.15 outside"),
                     (change(bandwidth_hz_per_px="150"), "the console rejects the scan of R2: bandwidth_hz_per_px must be a finite number"),
                     (change(bandwidth_hz_per_px=2500.0), "the console rejects the scan of R2: readout gradient is"),
                     (change(ro_polarity=2), "the console rejects the scan of R2: ro_polarity must be one of [1, -1]")):
        assert why in task.check(cfg, bad, exam), bad
    schema = task.submit_schema(cfg)["properties"]["scans"]
    assert schema["required"] == ["R1", "R2", "R3"] and schema["properties"]["R1"]["required"] == cfg["task"]["free"]
    assert set(schema["properties"]["R1"]["properties"]) == set(cfg["task"]["free"])


# ---------------------------------------------------------------------------------------------- the verifier's pieces
def test_the_fat_share_of_a_region_falls_as_the_bandwidth_rises():
    """What the bisection of ``lowest_clean`` relies on, on a coarse grid across the threshold, for the direction that
    moves the fat of the widest gap and for the opposite one."""
    case = CASE()
    region, site = case.cfg["task"]["regions"][0], case.cfg["truth"]["sites"][0]
    pe, pol = table.setting_for(table.widest(site))
    for polarity in (pol, -pol):
        cover = [task.measure(case.cfg, region, dict(bandwidth_hz_per_px=bw, ro_polarity=polarity), case.world, case.engine)["cover"]
                 for bw in task.BW_GRID[10:120:22]]
        assert all(a > b for a, b in zip(cover, cover[1:]) if a > 0.01), cover          # strictly falling until the fat has left
        assert cover[0] > task.COVER_MAX > cover[-1]


def _stand_in(needed):
    """In place of ``task.measure``: the fat is off a region from the bandwidth ``needed[approaching side]`` on, and the
    signal-to-noise ratio is 1000 / sqrt(bandwidth). Counts its calls."""
    calls = []

    def measure(cfg, region, scan, world, engine):
        bw = scan["bandwidth_hz_per_px"]
        calls.append(bw)
        side = task.approaching(scan.get("pe_dir") or "y", scan["ro_polarity"])
        return dict(cover=0.0 if bw >= needed[side] else 1.0, snr=1000.0 / math.sqrt(bw), realised=dict(bandwidth_hz_per_px=bw))
    measure.calls = calls
    return measure


def _own_instance(case, tag):
    """The instance of ``case`` under another seed label: its own entry in what the verifier keeps per instance."""
    return dict(case.cfg, seed=tag * 32)


def test_the_search_bisects_the_grid_and_the_best_setting_is_the_lowest_bandwidth_over_the_directions(monkeypatch):
    case = CASE()
    cfg, region = _own_instance(case, "a"), case.cfg["task"]["regions"][0]
    fake = _stand_in({"x-": 120.0, "x+": 333.0})
    monkeypatch.setattr(task, "measure", fake)
    k = task.lowest_clean(cfg, region, "y", 1, case.world, case.engine)
    assert task.BW_GRID[k] >= 120.0 > task.BW_GRID[k - 1] and len(fake.calls) <= 12          # a bisection, not a sweep of 187 points
    assert task.lowest_clean(cfg, region, "y", 1, case.world, case.engine, top=k) == k
    assert task.lowest_clean(cfg, region, "y", 1, case.world, case.engine, top=k - 1) is None  # nothing below it
    j = task.lowest_clean(cfg, region, "y", -1, case.world, case.engine)
    assert task.BW_GRID[j] >= 333.0 > task.BW_GRID[j - 1]
    best = task.best(cfg, case.world, case.engine)
    assert set(best) == {"R1", "R2", "R3"}
    for b in best.values():
        assert b == dict(bandwidth_hz_per_px=task.BW_GRID[k], ro_polarity=1, pe_dir="y", snr=1000.0 / math.sqrt(task.BW_GRID[k]))
    n = len(fake.calls)
    assert task.best(cfg, case.world, case.engine) is best and len(fake.calls) == n          # kept per instance
    # the score: 2 % above the best bandwidth is within 88 % of its signal-to-noise ratio, 40 % above is not
    right = {i: dict(bandwidth_hz_per_px=task.BW_GRID[k + 1], ro_polarity=1) for i in best}
    assert task.score(cfg, dict(scans=right), case.world, case.engine)["passed"] is True
    wide = dict(right, R2=dict(bandwidth_hz_per_px=task.BW_GRID[k + 17], ro_polarity=1))
    s = task.score(cfg, dict(scans=wide), case.world, case.engine)
    assert scoring.failing(s) == ["near_best_R2"] and task.outcome(s) == "bandwidth_too_high" and "R2: signal-to-noise ratio" in s["diagnostic"]
    assert s["regions"]["R2"]["snr"] / s["regions"]["R2"]["snr_best"] == pytest.approx(1.02 ** -8.5, rel=1e-3)
    assert s["distance"] == dict(regions=3, passed=2)


def test_the_search_at_the_ends_of_the_grid(monkeypatch):
    case = CASE()
    cfg, region = _own_instance(case, "b"), case.cfg["task"]["regions"][0]
    monkeypatch.setattr(task, "measure", _stand_in({"x-": 40.0, "x+": 5000.0}))
    assert task.lowest_clean(cfg, region, "y", 1, case.world, case.engine) == 0              # off the region at the console minimum already
    assert task.lowest_clean(cfg, region, "y", -1, case.world, case.engine) is None          # on the region at every bandwidth of the grid
    assert all(b["bandwidth_hz_per_px"] == 50.0 and b["ro_polarity"] == 1 for b in task.best(cfg, case.world, case.engine).values())
    # no direction keeps the fat off: no best setting, and no delivered scan can pass
    cfg = _own_instance(case, "c")
    monkeypatch.setattr(task, "measure", _stand_in({"x-": 5000.0, "x+": 5000.0}))
    assert task.best(cfg, case.world, case.engine) == dict(R1=None, R2=None, R3=None)
    s = task.score(cfg, dict(scans={i: dict(bandwidth_hz_per_px=300.0, ro_polarity=1) for i in ("R1", "R2", "R3")}), case.world, case.engine)
    assert s["passed"] is False and scoring.failing(s) == ["fat_off_R1", "fat_off_R2", "fat_off_R3"] and s["regions"]["R1"]["best"] is None


def test_what_is_kept_per_instance_is_bounded(monkeypatch):
    case = CASE()
    monkeypatch.setattr(task, "KEPT", 2)
    monkeypatch.setattr(task, "MEASURED_KEPT", 2)
    stores = [task._instance_store(_own_instance(case, tag)) for tag in "def"]
    assert len(task._kept) == 2 and task._instance_store(_own_instance(case, "f")) is stores[2]
    cfg, region = _own_instance(case, "f"), case.cfg["task"]["regions"][1]
    first = task.measure(cfg, region, dict(bandwidth_hz_per_px=900.0, ro_polarity=1), case.world, case.engine)
    again = task.measure(cfg, region, dict(bandwidth_hz_per_px=900.0, ro_polarity=1), case.world, case.engine)
    assert again == first and len(stores[2]["measured"]) == 1                                # the same question is answered from the store
    for bw in (1000.0, 1100.0):
        task.measure(cfg, region, dict(bandwidth_hz_per_px=bw, ro_polarity=1), case.world, case.engine)
    assert len(stores[2]["measured"]) == 2 and ("R2", "y", 1, 900.0) not in stores[2]["measured"]


def test_the_region_of_the_verifier_is_the_circle_not_the_pixels_whose_centres_fall_inside():
    rp = console.plan(dict(task.TARGET, pe_dir="y", bandwidth_hz_per_px=200.0, ro_polarity=1))["realised"]
    for centre in (0.0, 0.3, 0.6, 0.9):                                  # the same circle at four places on the pixel grid
        m = task.area_weights(rp, dict(y_mm=-10.0, x_mm=centre, radius_mm=5.0))
        assert m.sum() * task.PIXEL_MM ** 2 == pytest.approx(math.pi * 25.0, rel=0.01)
        assert m.max() == 1.0 and 0.0 < m[m > 0].min() < 0.2


def test_the_grid_of_the_search_runs_from_the_console_minimum_in_two_percent_steps():
    assert task.BW_GRID[0] == 50.0 and task.BW_GRID[-1] <= 2000.0 < task.BW_GRID[-1] * 1.02
    assert all(b / a == pytest.approx(1.02, abs=2e-4) for a, b in zip(task.BW_GRID, task.BW_GRID[1:]))
    assert console.plan(dict(task.TARGET, bandwidth_hz_per_px=task.BW_GRID[-1]))["accept"]          # the gradients can play all of it
    assert console.plan(dict(task.TARGET, bandwidth_hz_per_px=task.BW_GRID[0]))["accept"]           # ... and the echo time has room


def test_outcome_names_the_first_failed_rule():
    ok = dict(deliverable_valid=True, fat_off_R1=True, snr_floor_R1=True, near_best_R1=True)
    assert task.outcome(dict(checks=dict(ok, fat_off_R2=False, near_best_R2=False))) == "fat_on_region"
    assert task.outcome(dict(checks=dict(ok, snr_floor_R2=False, near_best_R2=False))) == "snr_below_floor"
    assert task.outcome(dict(checks=dict(ok, near_best_R2=False))) == "bandwidth_too_high"
    assert task.OUTCOMES == ("fat_on_region", "snr_below_floor", "bandwidth_too_high")
    assert (task.SUBJECT, task.KIND) == ("B", "set") and "bandwidth" in task.ISOLATES


# ---------------------------------------------------------------------------------------------- control policies
def _play(name, level=2, seed=0, recipe=None):
    env = mrgym.make(ENV_ID, level=level, seed=seed)
    result = policies.run(env, name, recipe=recipe)
    env.close()
    return result


def test_the_reference_passes_within_its_budget_and_plays_like_the_open_loop_control():
    r = _play("reference")
    assert r.reward == 1.0 and r.info["policy_error"] is None, r.info["score"].get("diagnostic")
    assert r.info["budget"]["used_s"] < 0.6 * r.info["budget"]["budget_s"]
    for row in r.info["score"]["regions"].values():                      # inside the window of rule 3, on both sides
        assert row["cover"] <= task.COVER_MAX and task.NEAR_BEST + 0.02 < row["snr"] / row["snr_best"] <= 1.0 + 1e-9
    assert controls.policies()["open_loop"] is controls.reference


@pytest.mark.parametrize("name", controls.MUST_FAIL)
def test_every_shortcut_fails(name):
    r = _play(name)
    assert r.reward == 0.0
    if name == "prior":
        assert r.info["score"]["outcome"] == "no_measurement"
    if name == "wrong_units":
        assert "bandwidth_hz_per_px" in r.info["refused"] and "below the minimum 50" in r.info["refused"]    # refused: no deliverable was scored
        assert r.info["submitted"] is False
    if name == "highest_bandwidth":
        assert r.info["score"]["outcome"] == "snr_below_floor"


def test_the_tuned_recipe_is_one_constant_setting():
    recipe = controls.tune([_instance(2, f"tune:{k}") for k in range(6)])
    assert set(recipe) == {"bandwidth_hz_per_px", "ro_polarity", "pe_dir", "tuned_region_pass_rate"}
    assert 50.0 <= recipe["bandwidth_hz_per_px"] <= 400.0 and recipe["tuned_region_pass_rate"] < 0.35
    assert _play("fixed_recipe", recipe=recipe).reward == 0.0


def test_the_reference_passes_with_the_axis_free():
    r = _play("reference", level=3)
    assert r.reward == 1.0, r.info["score"].get("diagnostic")
    assert {s["pe_dir"] for s in r.info["score"]["regions"].values() for s in [s["best"]]} <= {"y", "x"}
