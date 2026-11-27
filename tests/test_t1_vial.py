"""``Scanner/T1Vial-v0`` beyond its scoring table (``tests/scoring/t1_vial.py``): the one formula the family rests on,
checked against an independent pulse chain and against the engine; what the statement says about the system; the
scan lists and budgets; instances and the control policies.

The formula: at a fixed repetition time the inversion-recovery signal of a material is  a - b exp(-TI / T1)  with its
true T1, for any repetition time, inversion efficiency and slice profile. The independent calculation is the pulse
chain of ``tests/scoring/null_tissue.py`` (a cascade of 3 x 3 rotations for the selective pulses, the longitudinal
magnetisation walked through excitation, refocusing, recovery, inversion and inversion time until it repeats, summed
over the slice). It shares no code with the engine or with this family.
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
from mrgym.envs.t1_vial import controls, task
from mrgym.scanner import console, kspace
from scoring import null_tissue as chain
from scoring import t1_vial as table

ENV = "Scanner/T1Vial-v0"
CASE = lambda key="L1": scoring.case(table, key)       # noqa: E731


def _ids(case):
    return [r["id"] for r in case.cfg["task"]["regions"]]


# ---------------------------------------------------------------------------------------------- the formula
@pytest.mark.parametrize("tr,t1,kappa,flip", [(900.0, 1500.0, 0.93, 90.0), (1500.0, 300.0, 0.97, 90.0), (3500.0, 2000.0, 0.92, 90.0),
                                             (700.0, 60.0, 0.95, 90.0), (900.0, 800.0, 0.95, 70.0)])
def test_the_recovery_curve_of_the_pulse_chain_is_three_parameter_with_the_true_t1(tr, t1, kappa, flip):
    """The chain at five inversion times, a repetition time down to 0.6 T1, an incomplete inversion, the slice profile
    of excitation and refocusing (and, last case, a transmit error, which the task rules out): a - b exp(-TI / T1)
    through three of the points gives the true T1 to 0.1 % and predicts the other two to 1e-5 of the signal."""
    tis = [15.0, 0.25 * tr, 0.5 * tr, 0.75 * tr, tr - chain.TE_MS - 15.0]
    s = np.array([chain.slice_signal(math.radians(flip), tr, chain.TE_MS, t1, ti, kappa) for ti in tis])
    got = controls.solve_three([tis[0], tis[2], tis[4]], s[[0, 2, 4]])
    assert got == pytest.approx(t1, rel=1e-3)
    e = np.exp(-np.array(tis) / t1)
    b = (s[4] - s[0]) / (e[0] - e[4])
    a = s[0] + b * e[0]
    assert np.abs(a - b * e - s).max() < 1e-5 * abs(s[4])
    assert 1.2 < b / a <= 2.0                                           # 2 only for full recovery and a complete inversion
    assert controls.fit_three_parameters(tis, s) == pytest.approx(t1, rel=0.01)       # the grid of the least-squares fit
    # the textbook two-parameter curve on the same points is wrong unless the repetition time is many T1
    textbook = controls.fit_full_recovery(tis, s)
    assert (abs(math.log(textbook / t1)) > math.log(1.1)) == (tr < 4.0 * t1)


def test_the_engine_serves_that_curve():
    """The engine's own slice-integrated inversion-recovery signal of one material (no image, no neighbour): the
    stated equation returns the T1 of the material to 0.1 %, at the repetition time of level 2 and of level 3."""
    for tr, tis, t1, kappa in ((900.0, (15.0, 400.0, 875.0), 1400.0, 0.93), (3500.0, (10.0, 900.0, 3480.0), 1950.0, 0.96),
                              (1500.0, (30.0, 400.0, 1450.0), 320.0, 0.95)):
        cls = dict(t1_ms=t1, t2_ms=80.0, adc_um2_ms=1.5)
        s = []
        for ti in tis:
            rp = console.plan(dict(controls.PROBE, tr_ms=tr, ti_ms=ti))["realised"]
            s.append(float(kspace.slice_signal(rp, cls, np.array([math.pi / 2]), rp["rf_tbw"], inv_eff=kappa)[0][0].real))
        assert s[0] < 0 < s[2]                                           # signed: before and after the null
        assert controls.solve_three(tis, s) == pytest.approx(t1, rel=1e-3)


def test_the_stated_method_on_noise_free_images_and_what_partial_fourier_does_to_it():
    """Region means of noise-free images, signed with the phase, through the stated equation: within 6 % of the true
    T1 (what is left is the gel around a vial leaking into the region mean at matrix 32). The same scans with partial
    Fourier 0.55 are cheaper and wrong: zero-filled lines give that leak a complex weight (task module, ``PROBE``)."""
    case = CASE("L1")
    t = case.cfg["task"]
    tr, tis = t["method"]["protocol"]["tr_ms"], t["method"]["ti_ms"]

    def measured(**more):
        reads = [F.region_complex(case.engine, console.plan(dict(controls.PROBE, tr_ms=tr, ti_ms=ti, **more), case.exam)["realised"], t["regions"])
                 for ti in tis]
        out = {}
        for i in _ids(case):
            signed = [float((r[i] * np.conj(reads[-1][i] / abs(reads[-1][i]))).real) for r in reads]
            out[i] = abs(math.log(controls.solve_three(tis, signed) / case.truth[i]["t1_ms"]))
        return out
    full, partial = measured(), measured(partial_fourier=0.55)
    assert max(full.values()) < 0.06
    assert max(partial.values()) > 2 * max(full.values())


# ---------------------------------------------------------------------------------------------- what the statement says
@pytest.mark.parametrize("level", [1, 2, 3])
def test_statement_is_true_of_the_world_and_states_no_hidden_value(level):
    seed_hex = derive_seed(ENV, level, "train", 3, environ={})
    cfg = task.make_instance(seed_hex, level, task_id="t")
    text = task.statement(cfg)
    world = scoring.W.make_world(cfg["seed"], cfg["world_spec"])
    # "The transmit calibration is correct ... everywhere in the object"
    assert world["system"]["tx_gain"] == 1.0 and float(world["fields"]["b1"].min()) == float(world["fields"]["b1"].max()) == 1.0
    assert "The transmit calibration is correct" in text and "the noise level is not given" in text
    truth = task.true_t1_ms(cfg, world)
    lo, hi = cfg["task"]["t1_range_ms"]
    assert len(truth) == 3 and all(lo <= v <= hi for v in truth.values()) and f"between {lo:.0f} and {hi:.0f} ms" in text
    for r in world["regions"]:
        cls = world["tissues"][r["cls"]]
        for hidden in (cls["t1_ms"], cls["t2_ms"]):
            assert f"{hidden:.2f}" not in text and repr(hidden) not in text
    assert cfg["seed"] not in text and f"{world['system']['sigma0']:.3f}" not in text and f"{world['system']['inv_eff']:.3f}" not in text
    assert ("How to do it" in text) == (level == 1) == ("method" in cfg["task"])
    assert ("at least 10 times the shortest" in text) == (level == 3)
    if level == 3:
        assert max(truth.values()) / min(truth.values()) >= task.MIN_SPREAD == 10.0
    assert "they are charged to the scan-time budget" in text and F.NOT_CHARGED not in text      # the agent's scans ARE the measurement
    assert "with partial Fourier the zero-filled lines mix the\nmaterial around a vial into it" in text
    assert "within a factor 1.1 of the true T1" in text and "wrong unit" in text


def test_level_three_passes_over_worlds_whose_vials_do_not_differ_enough():
    """``family.reseed``: the world of an instance is the first attempt of its seed with the told spread."""
    attempts = []
    for s in range(12):
        seed_hex = derive_seed(ENV, 3, "train", s, environ={})
        cfg = task.make_instance(seed_hex, 3, task_id="t")
        k = next(a for a in range(64) if F.reseed(seed_hex, a) == cfg["seed"])
        attempts.append(k)
        for a in range(k):                                               # every attempt before it was too narrow
            layout = scoring.W.make_world(F.reseed(seed_hex, a), cfg["world_spec"], layout_only=True)
            t1 = [layout["tissues"][r["cls"]]["t1_ms"] for r in layout["regions"] if int(r["id"][1:]) <= 3]
            assert max(t1) / min(t1) < task.MIN_SPREAD
        t1 = list(task.true_t1_ms(cfg).values())
        assert max(t1) / min(t1) >= task.MIN_SPREAD
    assert max(attempts) >= 1                                            # the rule was exercised
    assert all(task.make_instance(derive_seed(ENV, lv, "train", 0, environ={}), lv, task_id="t")["seed"] == derive_seed(ENV, lv, "train", 0, environ={})
               for lv in (1, 2))                                        # levels 1 and 2 take the first world


def test_instances_are_functions_of_the_seed_also_under_threads():
    jobs = [(lv, s) for lv in (1, 2, 3) for s in (0, 1, 2)]

    def build(job):
        cfg = task.make_instance(derive_seed(ENV, job[0], "train", job[1], environ={}), job[0], task_id="t")
        return json.dumps(dict(cfg, statement=task.statement(cfg)), sort_keys=True)
    serial = [build(j) for j in jobs]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        assert list(pool.map(build, jobs * 2)) == serial * 2
    assert len(set(serial)) == len(serial)


# ---------------------------------------------------------------------------------------------- scan lists and budgets
def test_every_scan_list_is_accepted_and_fits_the_budget_of_its_level_only():
    cost = {}
    for lv, (tr, tis, matrix) in controls.DESIGN.items():
        plans = [console.plan(dict(controls.PROBE, matrix=matrix, tr_ms=tr, ti_ms=ti)) for ti in tis]
        assert all(p["accept"] for p in plans), [p.get("violations") for p in plans]
        assert all(p["realised"]["partial_fourier"] == 1.0 and p["realised"]["lines"] == matrix for p in plans)
        cost[lv] = sum(p["scan_s"] for p in plans)
        assert cost[lv] <= task.LEVELS[lv]["budget_s"] < cost[lv] + plans[0]["scan_s"]      # no room for one more scan of the list
    assert controls.DESIGN[1] == (task.STATED["tr_ms"], task.STATED["ti_ms"], 32) and len(controls.DESIGN[3][1]) == 4
    assert cost[1] > task.LEVELS[2]["budget_s"]                          # level 2 cannot play the scans level 1 states
    assert task.method_text(dict(method=dict(protocol=dict(task.PROBE, tr_ms=1500.0), ti_ms=[30.0, 400.0, 1450.0]))).count("52.5 s") == 1
    assert plans[0]["scan_s"] == pytest.approx(3.0 + 49 * 3.5)           # level 3: overhead + (one dummy + 48 lines) x TR


def test_the_bisection_of_the_stated_method():
    tis = (30.0, 400.0, 1450.0)
    for t1 in (60.0, 300.0, 905.5, 1500.0, 4000.0):
        s = 2.0 - 3.7 * np.exp(-np.array(tis) / t1)
        assert controls.solve_three(tis, s) == pytest.approx(t1, rel=1e-9)
    assert controls.solve_three(tis, [1.0, 1.0, 1.0]) == 5000.0         # no recovery seen: the upper end
    assert controls.solve_three(tis, [-1.0, 1.0, 1.0]) == 50.0          # recovered before the second point: the lower end
    assert controls.fit_three_parameters(tis, 2.0 - 3.7 * np.exp(-np.array(tis) / 700.0)) == pytest.approx(700.0, rel=0.006)


# ---------------------------------------------------------------------------------------------- control policies
@pytest.mark.parametrize("level,seed", [(1, 0), (2, 0), (3, 0), (3, 1)])
def test_reference_passes_within_the_budget(level, seed):
    r = policies.run(mrgym.make(ENV, level=level, seed=seed), "reference")
    assert r.reward == 1.0 and r.info["policy_error"] is None and r.info["score"]["outcome"] == "pass"
    assert r.info["budget"]["scans"] == len(controls.DESIGN[level][1]) and r.info["budget"]["used_s"] <= task.LEVELS[level]["budget_s"]
    assert r.info["score"]["error"] < math.log(1.1)


@pytest.mark.parametrize("name,level", [("prior", 1), ("fixed_recipe", 1), ("brute_force", 2), ("wrong_units", 1), ("unsigned", 1),
                                        ("full_recovery", 2), ("level1_scans", 2)])
def test_shortcuts_and_ablations_fail(name, level):
    r = policies.run(mrgym.make(ENV, level=level, seed=2), name)
    assert r.reward == 0.0
    if name == "prior":
        assert r.info["score"]["outcome"] == "no_measurement"
    elif name == "wrong_units":
        assert r.info["score"]["checks"]["deliverable_valid"] is False and "below the minimum 10" in r.info["refused"]     # the schema's bound
    elif (name, level) == ("level1_scans", 2):
        assert "BudgetExhausted" in r.info["policy_error"]              # the scans of level 1 are over the budget of level 2
    else:
        assert r.info["policy_error"] is None and r.info["score"]["outcome"] in task.OUTCOMES


def test_tune_outcome_and_the_declared_names():
    cfgs = [task.make_instance(derive_seed(ENV, 1, "tune", f"0:{i}", environ={}), 1, task_id="t") for i in range(8)]
    recipe = controls.tune(cfgs)
    assert task.NARROW_MS[0] <= recipe["t1_ms"] <= task.NARROW_MS[1] and 0.0 <= recipe["tuned_pass_rate"] <= recipe["tuned_region_pass_rate"] <= 1.0
    assert task.outcome(dict(checks=dict(deliverable_valid=True, t1_within_tolerance_R1=True, t1_within_tolerance_R2=False))) == "some_regions_within"
    assert task.outcome(dict(checks=dict(deliverable_valid=True, t1_within_tolerance_R1=False, measured=True))) == "no_region_within"
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and all("\n" not in d and len(d) < 130 for d in controls.DESCRIPTIONS.values())
    assert task.card()["id"] == ENV and mrgym.spec(ENV).kind == "candidate" and task.COMPETENCIES == ("parameter.ti", "parameter.tr")
    assert [controls._level(task.make_instance(derive_seed(ENV, lv, "train", 0, environ={}), lv, task_id="t")["task"]) for lv in (1, 2, 3)] == [1, 2, 3]
