"""``Scanner/GhostAside-v0`` beyond its scoring table (``tests/scoring/ghost_aside.py``): the ghost law, derived with a
discrete Fourier transform and checked on the engine; instances; the task text; the control policies on one instance;
the search of the verifier.

The scored episodes use the instances of the scoring table, so the verifier's search for the fastest scan (about ten
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
from mrgym.envs.ghost_aside import controls, task
from mrgym.scanner import console, kspace
from mrgym.scanner import world as W
from scoring import ghost_aside as table

ENV_ID = "Scanner/GhostAside-v0"
CASE = lambda key="L2": scoring.case(table, key)       # noqa: E731


# ---------------------------------------------------------------------------------------------- the physics, independently
def _dist(u):
    return abs(u - round(u))


@pytest.mark.parametrize("rate_hz,tr_s,averages", [(0.3, 0.4, 1), (1.3, 0.5, 1), (0.3, 0.3, 2), (0.9, 0.6, 3), (0.25, 1.0, 1)])
def test_the_ghost_law_from_the_shift_theorem(rate_hz, tr_s, averages):
    """The law, without the engine. The phase-encode lines k = -lines / 2 ... lines / 2 - 1 are acquired in that order,
    averages x TR apart. Multiplying line k of the transform of an object by 1 + m cos(2 pi f t(k) + phase) adds two
    copies of the object, each m / 2 strong, displaced by +- lines x dist(f x TR x averages) pixels (the shift theorem
    of the discrete Fourier transform)."""
    lines, depth = 128, 0.6
    obj = np.zeros(lines)
    obj[60:66] = 1.0
    order = np.fft.fftfreq(lines) * lines + lines // 2                   # position of every line of the transform in the acquisition
    modulated = np.fft.ifft(np.fft.fft(obj) * (1.0 + depth * np.cos(2 * np.pi * rate_hz * order * averages * tr_s + 0.7)))
    copies = modulated - obj
    ghost = np.abs(copies)
    want = lines * _dist(rate_hz * tr_s * averages)
    centre = 62.5
    for sign in (1, -1):
        where = (centre + sign * want) % lines
        near = np.arange(int(round(where)) - 6, int(round(where)) + 7)
        got = float((ghost[near % lines] * near).sum() / ghost[near % lines].sum())
        assert got == pytest.approx(where, abs=0.5)                      # the copy lies where the law puts it
        assert abs(copies[near % lines].sum()) == pytest.approx(depth / 2 * obj.sum(), rel=0.1)         # ... and holds m / 2 of the object
    assert task.ghost_distance_mm(rate_hz, 1e3 * tr_s, averages, fov_pe_mm=float(lines)) == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("rate_hz,tr_ms,averages,pe_dir,phase_fov", [
    (0.30, 400.0, 1, "y", 1.0), (0.30, 400.0, 1, "x", 1.0), (1.30, 500.0, 1, "x", 1.0), (0.30, 300.0, 2, "y", 1.0), (0.40, 350.0, 1, "y", 0.75)])
def test_the_engine_puts_the_ghost_where_the_law_says(rate_hz, tr_ms, averages, pe_dir, phase_fov):
    """The engine against the law: a small pulsating disc in a still one. The difference to the scan at rest holds two
    copies along the phase-encode axis only, phase field of view x dist(rate x TR x averages) from the disc."""
    spot = (10.0, -16.0)
    ph = W.shapes_phantom([("wm", lambda y, x: np.hypot(y, x) - 80.0), ("blood", lambda y, x: np.hypot(y - spot[0], x - spot[1]) - 5.0)], n=320)
    world = W.flat_world(ph)
    world["motion"] = [dict(cls="blood", kind="pulse", f_hz=rate_hz, depth=0.6, phase_key=11)]
    protocol = dict(sequence="se", tr_ms=tr_ms, te_ms=15.0, fov_mm=240.0, matrix=128, pe_dir=pe_dir, averages=averages, phase_fov=phase_fov)
    plan = console.plan(protocol, world["console"])
    moving = kspace.Engine(world).acquire(plan["realised"], None)[0]
    rest = kspace.Engine(world, effects=kspace.ALL - {"motion"}).acquire(plan["realised"], None)[0]
    ghost = np.abs(moving - rest)
    axis = 0 if pe_dir == "y" else 1                                     # image axis of the phase encoding
    fov_pe = plan["realised"]["fov_pe_mm"]
    assert fov_pe == pytest.approx(240.0 * phase_fov)
    along = ghost.sum(axis=1 - axis)
    pos = (np.arange(len(along)) - len(along) // 2) * fov_pe / len(along)
    want = fov_pe * _dist(rate_hz * tr_ms * 1e-3 * averages)
    assert task.ghost_distance_mm(rate_hz, tr_ms, averages, fov_pe) == pytest.approx(want, abs=1e-9)
    for sign in (1, -1):
        where = (spot[axis] + sign * want + fov_pe / 2) % fov_pe - fov_pe / 2
        near = np.abs((pos - where + fov_pe / 2) % fov_pe - fov_pe / 2) <= 9.0
        offset = (pos[near] - where + fov_pe / 2) % fov_pe - fov_pe / 2
        assert float((along[near] * offset).sum() / along[near].sum()) == pytest.approx(0.0, abs=0.6)   # within a third of a pixel
        assert along[near].sum() > 0.4 * along.sum()                     # the two copies hold the difference between them
    across = (ghost ** 2).sum(axis=axis)                                 # energy: the copies, not the ripples beside them
    other = (np.arange(len(across)) - len(across) // 2) * 240.0 / len(across)
    assert float((across * other).sum() / across.sum()) == pytest.approx(spot[1 - axis], abs=0.5)        # nothing moves along the readout


# ---------------------------------------------------------------------------------------------- instances
def _instance(level, seed):
    return task.make_instance(derive_seed(ENV_ID, level, "train", seed, environ={}), level, task_id="t")


def test_an_instance_is_a_function_of_the_seed_also_under_threads():
    serial = {(lv, s): json.dumps(_instance(lv, s), sort_keys=True) for lv in (1, 2, 3) for s in range(4)}
    assert len(set(serial.values())) == len(serial)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = dict(zip(serial, pool.map(lambda k: json.dumps(_instance(*k), sort_keys=True), list(serial) * 2)))
    assert threaded == serial


@pytest.mark.parametrize("level", [1, 2, 3])
def test_the_instance_follows_the_declared_geometry(level):
    """By the law alone: at the shortest repetition time the ghost of a slow tube lies well inside the reach of both of
    its regions; a fast tube's lies far beyond them; level 3 has exactly one tube that is not slow."""
    kinds = []
    for seed in range(40):
        cfg = _instance(level, seed)
        tubes = cfg["truth"]["tubes"]
        assert [u["id"] for u in tubes] == ["T1", "T2", "T3"] == [u["id"] for u in cfg["task"]["tubes"]]
        assert sum(u["kind"] != "slow" for u in tubes) == (1 if level == 3 else 0)
        assert cfg["task"]["complaint"] is (level != 3)
        assert len(cfg["world_spec"]["motion"]) == sum(u["kind"] != "none" for u in tubes)
        for k, u in enumerate(tubes):
            kinds.append(u["kind"])
            d = [table.distance(u, pe) for pe in ("y", "x")]
            assert task.NEAR_MM[0] - 0.1 <= min(d) <= task.NEAR_MM[1] + 0.1 and 5.9 <= max(d) - min(d) <= 7.1
            inside = 93.0 - 2.0                                          # the smallest body, at its largest offset
            assert math.hypot(u["y_mm"], u["x_mm"]) + task.TUBE_MM < inside
            assert all(math.hypot(r["y_mm"], r["x_mm"]) + task.REGION_MM < inside for r in u["regions"])
            if u["kind"] == "none":
                assert u["rate_hz"] is None
                continue
            g0 = table.ghost_mm(u["rate_hz"], task.TR_WINDOW_MS[0])
            if u["kind"] == "slow":
                assert all(-9.1 <= g0 - x <= 10.1 for x in d), (g0, d)   # on both regions, 5 mm or more inside their reach
                assert u["rate_hz"] * task.TR_WINDOW_MS[1] * 1e-3 < 0.5  # the ghost only moves outward inside the window
            else:
                assert g0 >= 75.0 and task.FAST_HZ[0] <= u["rate_hz"] <= task.FAST_HZ[1]
            # no other tube's ghosts can reach this tube's regions: its column and its row are its own
            for j, v in enumerate(tubes):
                if j != k:
                    assert all(abs(v["x_mm"] - r["x_mm"]) > task.TUBE_MM + task.REGION_MM + 1.0 for r in u["regions"])
                    assert all(abs(v["y_mm"] - r["y_mm"]) > task.TUBE_MM + task.REGION_MM + 1.0 for r in u["regions"])
        if level == 2:
            one = task.make_instance(cfg["seed"], 1, task_id="t")           # the same seed at level 1: the same world, with the method
            assert one["world_spec"] == cfg["world_spec"] and one["truth"] == cfg["truth"] and one["task"]["hints"] == ["method"]
    if level == 3:
        assert {"none", "fast"} <= set(kinds)


# ---------------------------------------------------------------------------------------------- the task text
@pytest.mark.parametrize("level", [1, 2, 3])
def test_the_statement_is_made_of_the_public_task_and_names_no_hidden_value(level):
    cfg = _instance(level, 0)
    text = task.statement(cfg)
    assert text == task.statement(dict(task=cfg["task"]))                # nothing but the public task goes into it
    public = json.dumps(cfg["task"])
    assert "rate" not in public and "kind" not in public and "depth" not in public
    for u in cfg["truth"]["tubes"]:
        if u["rate_hz"] is not None:
            assert f"{u['rate_hz']:g}" not in text and f"{u['rate_hz']:.2f}" not in text
    assert ("How to do it" in text) is (level == 1)
    assert ("Every tube pulsates" in text and "a ghost of every tube lies on one of its regions" in text) is (level != 3)
    assert ("One may pulsate faster (up to 1.5 Hz) or not at all" in text) is (level == 3)
    assert mrgym.envs.family.OWN_SCANS in text and mrgym.envs.family.NOT_CHARGED in text
    assert all(len(line) <= 170 for line in text.splitlines())


def test_the_method_of_level_1_states_the_law_and_the_numbers_of_this_scan():
    text = task.statement(_instance(1, 0))
    assert "g = (phase field of view) x dist(rate x TR x averages)" in text
    assert "rate = g / (72 mm s)" in text and f"D + {task.CLEAR_MM:g} mm" in text


# ---------------------------------------------------------------------------------------------- refused deliverables
def test_a_deliverable_the_console_cannot_evaluate_is_refused_with_a_reason():
    cfg = _instance(2, 0)
    exam = dict(fov_mm=240.0, min_pixel_mm=1.333, b1rms_limit_ut=6.0, b0_t=1.5)
    one = dict(pe_dir="x", tr_ms=480.0)
    good = dict(scans={i: dict(one) for i in ("T1", "T2", "T3")})
    assert task.check(cfg, good, exam) is None
    assert task.target_protocol(one) == dict(task.TARGET, pe_dir="x", tr_ms=480.0)
    change = lambda **scan: dict(scans=dict(good["scans"], T2=dict(one, **scan)))        # noqa: E731
    for bad, why in ((None, "the deliverable is"), (dict(scans=[one] * 3), "the deliverable is"), (dict(good, note="x"), "the deliverable is"),
                     (dict(scans={k: v for k, v in good["scans"].items() if k != "T3"}), "the deliverable is"),
                     (dict(scans=dict(good["scans"], T2=480.0)), "the scan of T2 needs exactly pe_dir and tr_ms"),
                     (change(averages=2), "the scan of T2 needs exactly pe_dir and tr_ms"),
                     (change(tr_ms=0.48), "the console rejects the scan of T2: tr_ms=0.48 outside"),
                     (change(tr_ms="480"), "the console rejects the scan of T2: tr_ms must be a finite number"),
                     (change(tr_ms=299.0), "tr_ms of T2 is 299: the examination allows 300 to 1000 ms"),
                     (change(tr_ms=1000.5), "tr_ms of T2 is 1000.5: the examination allows 300 to 1000 ms"),
                     (change(pe_dir="z"), "the console rejects the scan of T2: pe_dir must be one of ['y', 'x']")):
        assert why in task.check(cfg, bad, exam), bad
    assert task.check(cfg, change(tr_ms=300), exam) is None and task.check(cfg, change(tr_ms=1000.0), exam) is None      # the ends of the window
    schema = task.submit_schema(cfg)["properties"]["scans"]
    assert schema["required"] == ["T1", "T2", "T3"] and schema["properties"]["T1"]["required"] == ["pe_dir", "tr_ms"]
    assert "minimum" not in schema["properties"]["T1"]["properties"]["tr_ms"]            # the window is the task's rule: ``check`` states it


# ---------------------------------------------------------------------------------------------- the verifier's pieces
def test_at_the_shortest_repetition_time_the_ghost_lies_on_a_region_with_either_axis():
    """What the statement of levels 1 and 2 says, on the engine: region A is reached with pe_dir y, region B with x,
    and each ghost only reaches the region of its own axis."""
    case = CASE()
    for tube in case.cfg["task"]["tubes"]:
        for pe, own, other in (("y", 0, 1), ("x", 1, 0)):
            m = task.measure(case.cfg, tube, dict(pe_dir=pe, tr_ms=300.0), case.world, case.engine)
            assert m["ghost"][tube["regions"][own]["id"]] > 4 * task.GHOST_MAX
            assert m["ghost"][tube["regions"][other]["id"]] < 0.25 * task.GHOST_MAX


def test_along_one_axis_the_ghost_leaves_a_region_once():
    """What the bisection of ``first_clean`` relies on, on a coarse grid over the whole window."""
    case = CASE()
    tube = case.cfg["task"]["tubes"][0]
    worst = [task.measure(case.cfg, tube, dict(pe_dir="y", tr_ms=tr), case.world, case.engine)["worst"] for tr in range(300, 1001, 70)]
    clean = [w <= task.GHOST_MAX for w in worst]
    assert not clean[0] and clean[-1] and clean == sorted(clean)


def _stand_in(first_clear_ms):
    """In place of ``task.measure``: the ghost is off the tube's regions from the repetition time
    ``first_clear_ms[pe_dir]`` on; the scan time is the console's. Counts its calls."""
    calls = []

    def measure(cfg, tube, scan, world, engine):
        calls.append(scan["tr_ms"])
        worst = 0.0 if scan["tr_ms"] >= first_clear_ms[scan["pe_dir"]] else 0.5
        return dict(ghost={r["id"]: worst for r in tube["regions"]}, worst=worst, scan_s=table.scan_s(scan["tr_ms"]), realised={})
    measure.calls = calls
    return measure


def _own_instance(case, tag):
    """The instance of ``case`` under another seed label: its own entry in what the verifier keeps per instance."""
    return dict(case.cfg, seed=tag * 32)


def test_the_search_bisects_the_window_and_the_fastest_scan_is_the_earlier_axis(monkeypatch):
    case = CASE()
    cfg, tube = _own_instance(case, "a"), case.cfg["task"]["tubes"][0]
    fake = _stand_in(dict(y=611.0, x=455.0))
    monkeypatch.setattr(task, "measure", fake)
    grid = task.tr_grid()
    k = task.first_clean(cfg, tube, "y", case.world, case.engine)
    assert grid[k] == 612.0 and len(fake.calls) <= 13                    # the first point of the 2 ms grid at or after 611; a bisection
    assert task.first_clean(cfg, tube, "y", case.world, case.engine, top=k) == k
    assert task.first_clean(cfg, tube, "y", case.world, case.engine, top=k - 1) is None      # nothing earlier
    fast = task.fastest(cfg, case.world, case.engine)
    assert fast == {u: dict(pe_dir="x", tr_ms=456.0, scan_s=table.scan_s(456.0)) for u in ("T1", "T2", "T3")}
    n = len(fake.calls)
    assert task.fastest(cfg, case.world, case.engine) is fast and len(fake.calls) == n       # kept per instance
    # the score: 9 % more scan time passes, 11 % more does not, an earlier time leaves the ghost on
    scans = lambda tr: dict(scans=dict({u: dict(pe_dir="x", tr_ms=456.0) for u in ("T1", "T3")}, T2=dict(pe_dir="x", tr_ms=tr)))   # noqa: E731
    slower = lambda share: (table.scan_s(456.0) * share - 3.0) / 0.129   # noqa: E731  the repetition time of a scan that takes ``share`` as long
    assert task.score(cfg, scans(round(slower(1.09), 1)), case.world, case.engine)["passed"] is True
    s = task.score(cfg, scans(round(slower(1.11), 1)), case.world, case.engine)
    assert scoring.failing(s) == ["fastest_T2"] and task.outcome(s) == "too_slow" and "T2: " in s["diagnostic"] and "11 % above" in s["diagnostic"]
    assert s["sites"]["T2"]["time_ratio"] == pytest.approx(1.11, abs=2e-3) and s["distance"] == dict(tubes=3, passed=2)
    s = task.score(cfg, scans(440.0), case.world, case.engine)
    assert scoring.failing(s) == ["ghost_off_T2"] and task.outcome(s) == "ghost_on_region"


def test_the_search_at_the_ends_of_the_window(monkeypatch):
    case = CASE()
    cfg, tube = _own_instance(case, "b"), case.cfg["task"]["tubes"][0]
    monkeypatch.setattr(task, "measure", _stand_in(dict(y=200.0, x=5000.0)))
    assert task.first_clean(cfg, tube, "y", case.world, case.engine) == 0                    # clear at the shortest repetition time
    assert task.first_clean(cfg, tube, "x", case.world, case.engine) is None                 # never clear inside the window
    assert task.fastest(cfg, case.world, case.engine)["T1"] == dict(pe_dir="y", tr_ms=300.0, scan_s=table.scan_s(300.0))
    # no axis clears a tube inside the window: there is no fastest scan, and no delivered scan can pass
    cfg = _own_instance(case, "c")
    monkeypatch.setattr(task, "measure", _stand_in(dict(y=5000.0, x=5000.0)))
    assert task.fastest(cfg, case.world, case.engine) == dict(T1=None, T2=None, T3=None)
    s = task.score(cfg, dict(scans={u: dict(pe_dir="y", tr_ms=900.0) for u in ("T1", "T2", "T3")}), case.world, case.engine)
    assert s["passed"] is False and s["sites"]["T1"]["fastest"] is None and "T1: the ghost in" in s["diagnostic"]
    monkeypatch.setattr(task, "measure", lambda cfg, tube, scan, world, engine: dict(
        ghost={r["id"]: 0.0 for r in tube["regions"]}, worst=0.0, scan_s=table.scan_s(scan["tr_ms"]), realised={}))
    s = task.score(cfg, dict(scans={u: dict(pe_dir="y", tr_ms=900.0) for u in ("T1", "T2", "T3")}), case.world, case.engine)
    assert scoring.failing(s) == ["fastest_T1", "fastest_T2", "fastest_T3"] and "T1: no scan of the window keeps both regions free" in s["diagnostic"]


def test_what_is_kept_per_instance_is_bounded(monkeypatch):
    case = CASE()
    monkeypatch.setattr(task, "KEPT", 2)
    monkeypatch.setattr(task, "MEASURED_KEPT", 2)
    stores = [task._instance_store(_own_instance(case, tag), case.world) for tag in "def"]
    assert len(task._kept) == 2 and task._instance_store(_own_instance(case, "f"), case.world) is stores[2]
    cfg, tube = _own_instance(case, "f"), case.cfg["task"]["tubes"][1]
    first = task.measure(cfg, tube, dict(pe_dir="y", tr_ms=900.0), case.world, case.engine)
    again = task.measure(cfg, tube, dict(pe_dir="y", tr_ms=900.0), case.world, case.engine)
    assert again == first and len(stores[2]["measured"]) == 1            # the same question is answered from the store
    for tr in (910.0, 920.0):
        task.measure(cfg, tube, dict(pe_dir="y", tr_ms=tr), case.world, case.engine)
    assert len(stores[2]["measured"]) == 2 and ("T2", "y", 900.0) not in stores[2]["measured"]


def test_scan_time_and_grid_of_the_search():
    grid = task.tr_grid()
    assert grid[0] == 300.0 and grid[-1] == 1000.0 and all(b - a == 2.0 for a, b in zip(grid, grid[1:]))
    for tr in (300.0, 640.0, 1000.0):
        pl = console.plan(task.target_protocol(dict(pe_dir="x", tr_ms=tr)), dict(b0_t=1.5, b1rms_limit_ut=6.0))
        assert pl["accept"] and pl["scan_s"] == pytest.approx(table.scan_s(tr), abs=1e-6) and pl["realised"]["averages"] == 1


def test_outcome_names_the_first_failed_rule():
    ok = dict(deliverable_valid=True, ghost_off_T1=True, fastest_T1=True)
    assert task.outcome(dict(checks=dict(ok, ghost_off_T2=False, fastest_T2=True))) == "ghost_on_region"
    assert task.outcome(dict(checks=dict(ok, ghost_off_T2=True, fastest_T2=False))) == "too_slow"
    assert task.OUTCOMES == ("ghost_on_region", "too_slow")
    assert (task.SUBJECT, task.KIND) == ("B", "set") and "phase-encode" in task.ISOLATES


# ---------------------------------------------------------------------------------------------- control policies
def _play(name, level=2, seed=0, recipe=None):
    env = mrgym.make(ENV_ID, level=level, seed=seed)
    result = policies.run(env, name, recipe=recipe)
    env.close()
    return result


def test_the_reference_passes_within_its_budget_and_reads_the_rates():
    r = _play("reference")
    assert r.reward == 1.0 and r.info["policy_error"] is None, r.info["score"].get("diagnostic")
    assert r.info["budget"]["used_s"] == pytest.approx(2 * table.scan_s(300.0), abs=0.01) and r.info["budget"]["used_s"] < 0.7 * r.info["budget"]["budget_s"]
    for row in r.info["score"]["sites"].values():                        # inside the window of rule 2, on both sides
        assert row["ghost"] <= task.GHOST_MAX and 1.0 <= row["time_ratio"] < task.TIME_TOL - 0.02
    assert controls.policies()["open_loop"] is controls.reference


@pytest.mark.parametrize("name", controls.MUST_FAIL)
def test_every_shortcut_fails(name):
    r = _play(name)
    assert r.reward == 0.0
    if name == "prior":
        assert r.info["score"]["outcome"] == "no_measurement"
    if name == "shortest_tr":
        assert r.info["score"]["outcome"] == "ghost_on_region"
    if name == "wrong_units":
        assert "the console rejects the scan of T1: tr_ms=" in r.info["refused"]                     # refused: no deliverable was scored
        assert r.info["submitted"] is False


def test_the_tuned_recipe_ignores_the_object():
    recipe = controls.tune([_instance(2, f"tune:{k}") for k in range(16)])
    assert ("rate_hz" in recipe or "ghost_share" in recipe or {"tr_ms", "pe_dir"} <= set(recipe)) and recipe["tuned_tube_pass_rate"] < 0.4
    # the three kinds of recipe are answers made of the task card alone
    t = _instance(2, 0)["task"]
    for r in (dict(rate_hz=0.3), dict(ghost_share=0.5), dict(tr_ms=450.0, pe_dir="x")):
        answer = controls._recipe_answer(t, r)
        assert task.check(_instance(2, 0), answer, dict(b0_t=1.5, b1rms_limit_ut=6.0)) is None
    near = min(table.distance(t["tubes"][0], pe) for pe in ("y", "x"))
    assert controls._by_rate(t, 0.3)["scans"]["T1"]["tr_ms"] == pytest.approx(1e3 * (near + 15.0) / 72.0 * 1.05, abs=0.06)
    assert _play("fixed_recipe", recipe=recipe).reward == 0.0


def test_the_reference_passes_when_a_tube_is_still_or_fast():
    r = _play("reference", level=3)
    assert r.reward == 1.0, r.info["score"].get("diagnostic")
    other = table.other_tube(CASE("L3"))
    assert r.info["score"]["sites"][other]["fastest"]["tr_ms"] == 300.0 and r.info["score"]["sites"][other]["scan_s"] == pytest.approx(table.scan_s(300.0))
