"""``Scanner/T1Transmit-v1``: what the scoring table (``tests/scoring/t1_transmit.py``) does not cover.

  * the agent-side slice profile of the policies, written from the datasheet, is the profile the scanner plays
  * the closed forms summed over that profile recover T1 from noise-free 2D scans; the one-angle forms do not
  * the reference chooses by the budget: non-selective 3D at level 1, 2D with the profile in the model at levels 2, 3
  * the fixed protocol (``open_loop``) plays the same scan list on instances of different levels
  * the oracle: the true T1 values, with the ends of each pass window
  * the blind recipe is one constant T1; the tolerance is a constant of the level
"""
import math

import numpy as np
import pytest

import mrgym
import scoring
from mrgym import gate, policies
from mrgym.envs.t1_transmit import controls, oracle, task
from mrgym.scanner import console, rf, session
from scoring import t1_transmit as table

ENV_ID = "Scanner/T1Transmit-v1"


def test_the_tolerance_is_a_constant_of_the_level_in_physical_units():
    assert task.TOLERANCE == 0.10 and all(lv["tolerance"] == 0.10 for lv in task.LEVELS.values())
    for level in task.LEVELS:
        cfg = task.make_instance("0f" * 16, level)
        assert cfg["task"]["tolerance"] == 0.10 and "within a factor 1.1 of the true T1 (between T1 / 1.1 and 1.1 x T1)" in task.statement(cfg)
        assert cfg["task"]["deliverable"] == {"t1_ms": {"<region id>": "<number, milliseconds>"}} and 4 <= len(cfg["task"]["regions"]) <= 7
    assert not hasattr(controls, "tolerance_from") and not hasattr(task, "verify")          # nothing is calibrated on the reference
    lo, hi = zip(*(lv["budget_s"] for lv in task.LEVELS.values()))
    assert lo == (260.0, 60.0, 45.0) and hi == (320.0, 100.0, 60.0)


def test_the_agent_side_slice_profile_is_the_one_the_scanner_plays():
    """Two implementations of the pulse in the datasheet: the policies' own table and ``mrgym.scanner.rf``."""
    for name, tbw in controls.PULSE_TBW.items():
        assert rf.PULSES[name][1] == tbw
        theta, weight = controls.slice_profile(tbw)
        zeta, wz = rf.quadrature(tbw)
        assert theta.shape == (len(controls.ALPHA_GRID), len(zeta))
        for k in (20, 140, 300):                                             # 10, 70 and 150 degrees at the slice centre
            want_theta, phase = rf.excitation(controls.ALPHA_GRID[k], tbw, zeta)
            assert np.allclose(theta[k], want_theta, atol=1e-9) and np.allclose(weight[k], wz * phase.real, atol=1e-9)
        assert theta[140, 0] == pytest.approx(controls.ALPHA_GRID[140])      # the slice centre sees the nominal angle
    th, w = controls._through_slice(np.array([0.3, 1.0]), None)             # a hard pulse: one angle, weight 1
    assert th.shape == (2, 1) and np.array_equal(th[:, 0], [0.3, 1.0]) and np.array_equal(w, np.ones((2, 1)))


@pytest.fixture(scope="module")
def noise_free():
    """Noise-free 2D read-outs of the reference's scan list on one level 2 world, straight from the engine."""
    case = scoring.case(table, "L2")
    regions, exam = case.cfg["task"]["regions"], case.exam
    meas, ys = [], []
    for protocol in (dict(controls.AFI, **controls.MODE_2D, flip_deg=controls.AFI_FLIP),
                     *(dict(controls.SPGR, **controls.MODE_2D, flip_deg=f) for f in controls.FLIPS)):
        rp = console.plan(protocol, exam)["realised"]
        img = case.engine.acquire(rp, None)
        rows = session.roi_readouts(img, regions, session.fov_yx(rp))
        mag = np.array([r["complex_mean_magnitude"] for r in rows]).T
        if rp["sequence"] == "afi":
            ph = np.array([r["complex_mean_phase_rad"] for r in rows]).T
            ys += [mag[0], mag[1] * np.sign(np.cos(ph[1] - ph[0]))]
            meas.append(dict(kind="afi", flip=math.radians(rp["flip_deg"]), tr=rp["tr_ms"], tr2=rp["tr2_ms"]))
        else:
            ys.append(mag[0])
            meas.append(dict(kind="spgr", flip=math.radians(rp["flip_deg"]), tr=rp["tr_ms"]))
    return case, meas, np.array(ys)


def test_the_profile_model_recovers_t1_from_noise_free_2d_scans_and_the_one_angle_model_does_not(noise_free):
    case, meas, y = noise_free
    ids = [r["id"] for r in case.cfg["task"]["regions"]]
    w = np.ones(len(y))
    with_profile = [controls.fit_region([dict(m, tbw=4.0) for m in meas], y[:, i], w) for i in range(len(ids))]
    one_angle = [controls.fit_region([dict(m, tbw=None) for m in meas], y[:, i], w) for i in range(len(ids))]
    err = lambda fits: np.array([abs(math.log(f[0] / case.truth[i]["t1_ms"])) for f, i in zip(fits, ids)])       # noqa: E731
    assert err(with_profile).max() < 0.02                                    # well inside ln(1.1) = 0.095
    assert np.abs([f[1] / case.truth[i]["b1"] - 1 for f, i in zip(with_profile, ids)]).max() < 0.02     # and it finds the transmit factor
    assert np.median(err(one_angle)) > 2 * np.median(err(with_profile)) and err(one_angle).max() > 0.04
    assert np.median([f[0] / case.truth[i]["t1_ms"] for f, i in zip(one_angle, ids)]) < 1.0      # the profile makes T1 read short


def _scans(env):
    return [e["realised"] for e in env.scanner_log() if e["type"] == "scan"]


def test_the_reference_chooses_the_excitation_by_the_budget():
    used = {}
    for level in (1, 2):
        env = mrgym.make(ENV_ID, level=level, seed=0)
        r = policies.run(env, "reference")
        assert r.info["policy_error"] is None and r.reward == 1.0, r.info["score"]["diagnostic"]
        scans = _scans(env)
        assert [s["sequence"] for s in scans] == ["afi", "spgr", "spgr"] and [s["flip_deg"] for s in scans] == [70.0, 4.0, 20.0]
        assert all(s["slice_mm"] == 10.0 and s["bandwidth_hz_per_px"] == 200.0 for s in scans)
        used[level] = (scans[0]["excitation"], scans[0]["partitions"], r.info["budget"]["used_s"], r.info["budget"]["budget_s"])
    assert used[1][:2] == ("nonselective", 16) and used[2][:2] == ("selective", 1)
    assert 245.0 < used[1][2] <= used[1][3] and used[2][3] <= 100.0 and 0.8 * used[2][3] < used[2][2] <= used[2][3]
    env = mrgym.make(ENV_ID, level=2, seed=0)                                # 3D does not fit level 2: a token scan and the prior answer
    r = policies.run(env, "nonselective_3d")
    assert r.reward == 0.0 and len(_scans(env)) == 1 and r.info["policy_error"] is None


def test_the_fixed_protocol_plays_one_scan_list_on_every_instance():
    lists = []
    for level, seed in ((1, 3), (3, 4)):
        env = mrgym.make(ENV_ID, level=level, seed=seed)
        r = policies.run(env, "open_loop")
        assert r.info["policy_error"] is None
        lists.append([{k: s[k] for k in ("sequence", "flip_deg", "tr_ms", "averages", "matrix", "excitation", "rf_pulse", "slice_mm")} for s in _scans(env)])
        assert r.info["budget"]["used_s"] <= controls.OPEN_LOOP_BUDGET_S
    assert lists[0] == lists[1] and len(lists[0]) == 3 and all(s["excitation"] == "selective" for s in lists[0])
    assert min(lv["budget_s"][0] for lv in task.LEVELS.values()) == controls.OPEN_LOOP_BUDGET_S


def test_the_family_lists_its_policies_and_where_each_ablation_must_lose():
    assert (controls.REFERENCE, controls.MUST_FAIL) == ("reference", ("prior", "fixed_recipe", "wrong_units"))
    assert controls.ABLATIONS == {"ignores_profile": (2, 3), "blind_vfa": (1, 2, 3)}
    assert controls.REPORTED == ("open_loop", "profile_2d", "nonselective_3d", "ir_only")
    names = [controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert set(controls.policies()) == set(names) == set(controls.DESCRIPTIONS) and all(callable(f) for f in controls.policies().values())
    spec = mrgym.spec(ENV_ID)
    assert spec.kind == "practice" and gate.claim_levels(spec) == [1, 2, 3]
    card = mrgym.card(ENV_ID)
    assert card["task_kind"] == "measurement" and "a fixed protocol passes" in card["task_kind_why"] and card["why_practice"]


def test_the_blind_recipe_is_one_constant_t1():
    spec = mrgym.spec(ENV_ID)
    cfgs = [gate._instance(spec, 2, f"t:{i}", "tune") for i in range(16)]
    recipe = controls.tune(cfgs)
    assert set(recipe) == {"t1_ms", "tuned_region_pass_rate", "tuned_pass_rate"} and 300.0 <= recipe["t1_ms"] <= 2500.0
    truths = np.array([v for c in cfgs for v in task.true_t1_ms(c).values()])
    rate = lambda c: float(np.mean(np.abs(np.log(c / truths)) <= math.log(1.1)))       # noqa: E731
    assert recipe["tuned_region_pass_rate"] == round(rate(recipe["t1_ms"]), 3) >= max(rate(c) for c in (400.0, 866.0, 1500.0)) - 0.05
    assert recipe["tuned_pass_rate"] <= 1 / 16                                  # four to seven regions within 10 % of one number
    sess = session.ScannerSession(cfgs[0])
    answer = controls.fixed_recipe(recipe)(session.LocalScanner(sess))
    assert sess.n_scans == 1 and set(answer["t1_ms"].values()) == {recipe["t1_ms"]}
    assert controls.prior(session.LocalScanner(session.ScannerSession(cfgs[0]))) == {"t1_ms": dict.fromkeys(answer["t1_ms"], math.sqrt(300.0 * 2500.0))}


def test_the_oracle_is_the_truth_with_the_ends_of_each_pass_window():
    case = scoring.case(table, "L3")
    sol = oracle.solve(case.cfg)
    truth = {i: case.truth[i]["t1_ms"] for i in sol["argmin"]["t1_ms"]}
    assert sol["value"] == 0.0 and sol["method"] == "exact" and sol["argmin"] == {"t1_ms": truth} and oracle.REGRET_KIND == "difference"
    assert sol["window_ms"] == {i: [t / 1.1, t * 1.1] for i, t in truth.items()} and len(sol["certificate"]) == 2 * len(truth)
    score = lambda change: case.task.score(case.cfg, {"t1_ms": dict(truth, **change)}, case.world)       # noqa: E731
    for i, (lo, hi) in sol["window_ms"].items():
        assert score({i: lo * 1.0001})["passed"] and score({i: hi / 1.0001})["passed"]
    for c in sol["certificate"]:
        s = score(c["deliverable"]["t1_ms"])
        assert not s["passed"] and [k for k, ok in s["checks"].items() if not ok] == [c["fails"]]
    miss = oracle.miss(case.cfg, {"t1_ms": {i: t * 1.1 for i, t in truth.items()}}, sol)
    assert all(v == pytest.approx(1.0) for v in miss.values()) and oracle.miss(case.cfg, {"t1_ms": {}}, sol) == dict.fromkeys(truth)
