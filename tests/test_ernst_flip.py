"""``Scanner/ErnstFlip-v0`` beyond its scoring table (``tests/scoring/ernst_flip.py``): the physics the family rests on,
checked three ways; the verifier's shortcut around images; the budget claim of the statement; the stated method of
level 1; instances, statements and the control policies.

Three implementations of the 2D signal curve meet here and none imports another:
    engine   ``mrgym.scanner``: EPG steady state, Cayley-Klein slice profile (what serves and scores)
    model    ``mrgym.envs.ernst_flip.model``: ideal spoiling, Cayley-Klein profile (what level 1 prints, what the reference fits)
    chain    ``tests/scoring/ernst_flip.py``: ideal spoiling, a cascade of 3 x 3 rotations (the scoring table)
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
from mrgym.envs import family as F
from mrgym.envs.ernst_flip import controls, model, task
from mrgym.scanner import console, kspace
from scoring import ernst_flip as table

ENV = "Scanner/ErnstFlip-v0"
CASE = lambda key="L1": scoring.case(table, key)       # noqa: E731


def _ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


# ---------------------------------------------------------------------------------------------- the physics, three ways
def test_one_flip_angle_per_spin_the_maximum_is_the_ernst_angle():
    """Closed form: d/da [sin a (1 - E) / (1 - E cos a)] = 0 at cos a = E. The model's formula and the engine's
    ideal-spoiling steady state (tier 1, hard pulse) both peak there."""
    flips = np.arange(1.0, 90.0, 0.01)
    rp = console.plan(dict(sequence="spgr", tr_ms=20.0, flip_deg=10.0, excitation="nonselective"))["realised"]
    for t1 in (150.0, 700.0, 3000.0):
        want = math.degrees(math.acos(math.exp(-20.0 / t1)))
        assert model.ernst_angle_deg(20.0, t1) == pytest.approx(want, abs=1e-12)
        assert flips[int(np.argmax(model.ernst(np.deg2rad(flips), 20.0, t1)))] == pytest.approx(want, abs=0.011)
        engine = kspace.gre_steady(rp, dict(t1_ms=t1, t2_ms=80.0, adc_um2_ms=1.5), np.deg2rad(flips), tier=1)[0]
        assert flips[int(np.argmax(np.abs(engine)))] == pytest.approx(want, abs=0.011)


def test_in_a_2d_scan_the_best_nominal_angle_is_well_above_the_ernst_angle_in_all_three_implementations():
    """The module docstring of the task: about 1.5 times the Ernst angle, and 92 to 96 % of the maximum at the Ernst
    angle itself. Engine (EPG), public model and the table's rotation chain agree on the angle within 6 %."""
    for key in ("L1", "L2"):
        case = CASE(key)
        tr = case.cfg["task"]["target_scan"]["tr_ms"]
        got = task.fractions(case.cfg, {i: model.ernst_angle_deg(tr, case.truth[i]["t1_ms"]) for i in _ids(case)}, case.world, case.engine)
        for i in _ids(case):
            t1 = case.truth[i]["t1_ms"]
            engine, public, chain = got[i]["best_flip_deg"], model.optimum_deg(tr, t1), table.best_deg(case, i)
            assert abs(math.log(engine / public)) < 0.06 and abs(math.log(engine / chain)) < 0.06 and abs(math.log(public / chain)) < 0.02
            assert 1.3 < engine / model.ernst_angle_deg(tr, t1) < 1.65
            assert 0.92 < got[i]["fraction"] < 0.96                      # what the textbook angle of the TRUE T1 gives


def test_a_transmit_scale_moves_the_nominal_maximum_by_its_inverse():
    case = CASE("L3")
    tr = case.cfg["task"]["target_scan"]["tr_ms"]
    assert 0.2 <= abs(case.world["system"]["tx_gain"] - 1.0) <= 0.3      # level 3: 20 to 30 % off
    best = {i: v["best_flip_deg"] for i, v in task.fractions(case.cfg, {i: 20.0 for i in _ids(case)}, case.world, case.engine).items()}
    for i in _ids(case):
        b1, t1 = case.truth[i]["b1"], case.truth[i]["t1_ms"]
        assert abs(b1 - 1.0) > 0.15
        assert abs(math.log(best[i] / model.optimum_deg(tr, t1, scale=b1))) < 0.06     # the public model with the true factor
        assert abs(math.log(best[i] * b1 / model.optimum_deg(tr, t1))) < 0.08          # = the calibrated optimum / factor
        assert abs(math.log(best[i] / model.optimum_deg(tr, t1))) > 0.12               # the calibrated optimum itself is off


def test_the_material_signal_of_the_verifier_is_what_noise_free_images_show():
    """The verdict is computed on the steady-state signal of the region's material, not on images (task docstring).
    The two fractions differ by what neighbouring materials leak into a region: below 5e-4 here."""
    for key, factor in (("L1", 0.86), ("L3", 1.15)):
        case = CASE(key)
        best = {i: v["best_flip_deg"] for i, v in task.fractions(case.cfg, {i: 20.0 for i in _ids(case)}, case.world, case.engine).items()}
        flips = {i: round(min(best[i] * factor, 90.0), 2) for i in _ids(case)[:2]}
        material, image = task.fractions(case.cfg, flips, case.world, case.engine), task.image_fractions(case.cfg, flips, case.world, case.engine)
        for i in flips:
            assert 0.98 < material[i]["fraction"] < 0.9995
            assert image[i] == pytest.approx(material[i]["fraction"], abs=5e-4)


def test_the_best_angle_is_a_maximum_of_the_image_signal():
    """A sweep of noise-free images around the located angle: no scanned angle gives the region more signal (to 1e-4)."""
    case = CASE("L2")
    i = _ids(case)[0]
    best = task.fractions(case.cfg, {i: 20.0}, case.world, case.engine)[i]["best_flip_deg"]
    at_best = task.image_fractions(case.cfg, {i: best}, case.world, case.engine)[i]
    assert at_best == pytest.approx(1.0, abs=1e-12)
    for factor in (0.9, 0.96, 1.04, 1.1):
        assert task.image_fractions(case.cfg, {i: round(best * factor, 2)}, case.world, case.engine)[i] < 1.0 + 1e-4


def test_the_target_scan_is_the_stated_one_and_plays_the_pulse_the_models_assume():
    case = CASE()
    t = case.cfg["task"]
    assert task.target_protocol(t, 17.5) == dict(sequence="spgr", matrix=64, tr_ms=t["target_scan"]["tr_ms"], flip_deg=17.5)
    rp = console.plan(task.target_protocol(t, 17.5), case.exam)["realised"]
    assert (rp["excitation"], rp["rf_pulse"], rp["rf_tbw"], rp["slice_mm"], rp["averages"]) == ("selective", "normal", table.TBW, 5.0, 1)
    assert (model.TBW, model.PULSE_SAMPLES) == (table.TBW, table.SUB_PULSES) == (4.0, 128)
    assert (task.FLIP_MIN_DEG, task.FLIP_MAX_DEG, task.PASS_FRACTION) == (1.0, 90.0, 0.995)
    assert set(task.submit_schema(case.cfg)["properties"]) == {"flip_deg"}


# ---------------------------------------------------------------------------------------------- the claims of the statement
def test_the_budget_pays_for_two_gradient_echo_scans_and_not_for_three():
    """Statement: "each costs at least 7 s". The cheapest spgr scan the console sells at any repetition time of the
    family, three times, is over the budget; the two scans of the reference fit with room for averages."""
    for level in task.LEVELS.values():
        for tr in (task.TR_RANGE_MS[0], task.TR_RANGE_MS[1]):
            cheapest = console.plan(dict(sequence="spgr", matrix=32, phase_fov=0.25, partial_fourier=0.55, tr_ms=tr, flip_deg=10.0))
            assert cheapest["accept"] and cheapest["scan_s"] >= 7.0 and 3 * cheapest["scan_s"] > level["budget_s"]
            probe = console.plan(dict(controls.PROBE, tr_ms=tr, flip_deg=40.0))
            assert probe["accept"] and 2 * probe["scan_s"] < level["budget_s"]
    for kind in ("afi", "bssfp"):                                        # the other steady-state sequences pay the same 7 s
        p = console.plan(dict(sequence=kind, matrix=32, phase_fov=0.25, partial_fourier=0.55, tr_ms=12.0, flip_deg=10.0))
        assert (not p["accept"]) or p["scan_s"] >= 7.0
    # single-shot scans are cheap, and cannot be run at the target's repetition time
    assert not console.plan(dict(sequence="epi", matrix=32, tr_ms=task.TR_RANGE_MS[1], flip_deg=20.0))["accept"]
    assert console.plan(dict(controls.EPI, flip_deg=20.0))["accept"]


def test_about_twelve_percent_in_angle_hold_the_pass_fraction():
    """Statement: "99.5 % of it is held within about 12 % of the best angle". On the engine's curves of two instances."""
    for key in ("L1", "L3"):
        case = CASE(key)
        best = {i: v["best_flip_deg"] for i, v in task.fractions(case.cfg, {i: 20.0 for i in _ids(case)}, case.world, case.engine).items()}
        for factor, inside in ((0.92, True), (1.09, True), (0.84, False), (1.2, False)):
            got = task.fractions(case.cfg, {i: min(best[i] * factor, 90.0) for i in best}, case.world, case.engine)
            assert all((v["fraction"] >= task.PASS_FRACTION) == inside for v in got.values()), (key, factor, got)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_statement_states_the_scan_and_no_hidden_value(level):
    seed_hex = derive_seed(ENV, level, "train", 4, environ={})
    cfg = task.make_instance(seed_hex, level, task_id="t")
    text = task.statement(cfg)
    world = scoring.W.make_world(cfg["seed"], cfg["world_spec"])
    for r in world["regions"]:
        cls = world["tissues"][r["cls"]]
        for hidden in (cls["t1_ms"], cls["t2_ms"], cls["t2s_ms"]):
            assert f"{hidden:.2f}" not in text and repr(hidden) not in text
    assert cfg["seed"] not in text and f"{world['system']['tx_gain']:.3f}" not in text
    assert f"tr_ms {cfg['task']['target_scan']['tr_ms']:g}," in text and "matrix 64" in text and "99.5 %" in text
    assert F.OWN_SCANS in text and F.NOT_CHARGED in text
    assert ("How to do it" in text) == (level == 1) == ("method" in cfg["task"])
    assert ("The transmit calibration is correct" in text) == (level < 3) == (world["system"]["tx_gain"] == 1.0)
    assert ("not calibrated" in text) == (level == 3)
    if level < 3:
        assert float(world["fields"]["b1"].min()) == float(world["fields"]["b1"].max()) == 1.0     # the sentence is true everywhere
    assert len(cfg["task"]["regions"]) == 5 and cfg["budget_s"] == 20.0
    assert task.TR_RANGE_MS[0] <= cfg["task"]["target_scan"]["tr_ms"] <= task.TR_RANGE_MS[1]


def test_instances_are_functions_of_the_seed_also_under_threads():
    jobs = [(lv, s) for lv in (1, 2, 3) for s in (0, 1, 2)]

    def build(job):
        cfg = task.make_instance(derive_seed(ENV, job[0], "train", job[1], environ={}), job[0], task_id="t")
        return json.dumps(dict(cfg, statement=task.statement(cfg)), sort_keys=True)
    serial = [build(j) for j in jobs]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        assert list(pool.map(build, jobs * 2)) == serial * 2
    assert len(set(serial)) == len(serial)


# ---------------------------------------------------------------------------------------------- the stated method of level 1
def test_the_table_of_level_one_is_the_public_model_and_leads_to_the_best_angle():
    case = CASE("L1")
    t = case.cfg["task"]
    tr, (lo, hi), rows = t["target_scan"]["tr_ms"], t["method"]["flips_deg"], t["method"]["table"]
    assert [lo, hi] == task.probe_flips(tr) and lo == round(lo) and hi == round(hi) and 2 < hi / lo < 6
    r, a = np.array(rows).T
    assert len(rows) == len(task.TABLE_T1_MS) and np.all(np.diff(r) > 0) and np.all(np.diff(a) > 0)     # one T1 per row: longer T1, smaller ratio and angle
    for t1 in (150.0, 420.0, 1100.0, 3000.0):                           # between the rows: linear interpolation, as the text says
        s_lo, s_hi = model.curve(np.deg2rad([lo, hi]), tr, t1)[0]
        assert float(np.interp(s_hi / s_lo, r, a)) == pytest.approx(model.optimum_deg(tr, t1), rel=0.02)
    text = task.statement(case.cfg)
    assert all(f"{row[0]:.4f}    {row[1]:5.1f}" in text for row in rows) and f"flip_deg {lo:g} and flip_deg {hi:g}" in text
    # ... and followed on the engine's noise-free read-outs it passes (the reference does this with noise)
    signals = {}
    for flip in (lo, hi):
        rp = console.plan(dict(controls.PROBE, tr_ms=tr, flip_deg=flip), case.exam)["realised"]
        for i, v in F.region_complex(case.engine, rp, t["regions"]).items():
            signals.setdefault(i, []).append(abs(v))
    answer = {"flip_deg": dict(zip(signals, controls._by_table(t, signals)))}
    assert task.score(case.cfg, answer, case.world, case.engine)["passed"] is True


def test_the_fit_recovers_t1_and_the_maximum_from_two_points_of_the_public_model():
    for tr, t1 in ((12.0, 200.0), (20.0, 900.0), (30.0, 2800.0)):
        flips = task.probe_flips(tr)
        s = 3.7 * model.curve(np.deg2rad(flips), tr, t1)[0]
        got_t1, got_m = model.fit(flips, s, tr)
        assert got_t1 == pytest.approx(t1, rel=2e-3) and got_m == pytest.approx(3.7, rel=2e-3)
        hard = model.fit(flips, s, tr, profile=False)[0]                # the one-angle formula reads a 2D curve as a shorter T1
        assert hard < 0.75 * t1


# ---------------------------------------------------------------------------------------------- control policies
@pytest.mark.parametrize("level,seed", [(1, 0), (2, 1), (3, 0), (3, 2)])
def test_reference_passes_within_two_scans(level, seed):
    r = policies.run(mrgym.make(ENV, level=level, seed=seed), "reference")
    assert r.reward == 1.0 and r.info["policy_error"] is None
    assert r.info["budget"]["scans"] == 2 and r.info["budget"]["used_s"] <= 20.0
    assert r.info["score"]["error"] < 1.0 - task.PASS_FRACTION and r.info["score"]["outcome"] == "pass"


@pytest.mark.parametrize("name", controls.MUST_FAIL + ("textbook",))        # ``ignores_profile`` passes one instance in ten: the gate counts it
def test_shortcuts_and_ablations_fail(name):
    level = 3 if name in ("epi_sweep", "brute_force") else 1
    r = policies.run(mrgym.make(ENV, level=level, seed=1), name)
    assert r.reward == 0.0 and r.info["policy_error"] is None
    if name == "prior":
        assert r.info["score"]["outcome"] == "no_measurement"
    elif name == "wrong_units":
        assert r.info["score"]["checks"]["deliverable_valid"] is False or r.info["score"]["outcome"] in task.OUTCOMES
    else:
        assert r.info["score"]["outcome"] in task.OUTCOMES and r.info["budget"]["scans"] >= 1


def test_tune_returns_a_recipe_the_fixed_policy_plays():
    cfgs = [task.make_instance(derive_seed(ENV, 1, "tune", f"0:{i}", environ={}), 1, task_id="t") for i in range(3)]
    recipe = controls.tune(cfgs)
    assert ("flip_deg" in recipe) != ("t1_ms" in recipe) and 0.0 <= recipe["tuned_pass_rate"] <= recipe["tuned_region_pass_rate"] <= 1.0
    assert controls.recipe_flip(dict(flip_deg=31.0), 20.0) == 31.0
    assert controls.recipe_flip(dict(t1_ms=700.0), 20.0) == model.optimum_deg(20.0, 700.0)
    r = policies.run(mrgym.make(ENV, level=1, seed=2), "fixed_recipe", recipe=recipe)
    assert r.reward == 0.0 and r.info["budget"]["scans"] == 1 and len(set(r.info["score"]["fraction"])) == 5


def test_outcome_and_the_declared_names():
    assert task.outcome(dict(checks=dict(deliverable_valid=True, near_maximum_R1=True, near_maximum_R2=False))) == "some_regions_near_maximum"
    assert task.outcome(dict(checks=dict(deliverable_valid=True, near_maximum_R1=False, measured=True))) == "no_region_near_maximum"
    assert task.OUTCOMES == ("no_region_near_maximum", "some_regions_near_maximum")
    assert controls.REFERENCE == "reference" and set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and "open_loop" in controls.REPORTED
    assert all("\n" not in d and len(d) < 130 for d in controls.DESCRIPTIONS.values())
    assert task.card()["id"] == ENV and mrgym.spec(ENV).kind == "candidate" and task.COMPETENCIES == ("parameter.flip_angle", "qc.transmit_gain")
