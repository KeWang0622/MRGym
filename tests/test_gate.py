"""The family-level gate (``mrgym/gate.py``): the verdict of a level, the run, storage and the texts, and the bounds it
imports from ``mrgym.stats.gate``.

Expected numbers are computed here from the definitions, not by the module: Clopper-Pearson bounds from binomial tail
sums and bisection (the module uses the beta quantile), the paired gap from a tabulated Student quantile. The run is
played on ``tests/stub_family`` (what each policy passes is a known function of the seed) and, for n = 300, on pass
matrices written by hand.
"""
import inspect
import json
import math
import types

import numpy as np
import pytest

import mrgym
from conftest import FakeClock, FakeDatetime
from mrgym import gate, keys, policies
from mrgym.env import derive_seed
from stub_family import controls as stub_controls
from stub_family import task as stub_task

T95 = {1: 6.313752, 4: 2.131847, 29: 1.699127, 299: 1.649966}        # one-sided 95 % Student quantiles (standard table)


def _cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def cp_upper(k, n, alpha=0.05):
    """p with P(X <= k | n, p) = alpha."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _cdf(k, n, mid) > alpha else (lo, mid)
    return 0.5 * (lo + hi)


def cp_lower(k, n, alpha=0.05):
    """p with P(X >= k | n, p) = alpha."""
    if k <= 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if 1 - _cdf(k - 1, n, mid) < alpha else (lo, mid)
    return 0.5 * (lo + hi)


def first(n, k):
    """n instances, the first k pass."""
    return np.arange(n) < k


CONTROLS = types.SimpleNamespace(REFERENCE="reference", MUST_FAIL=("prior", "fixed_recipe", "wrong_units"),
                                 ABLATIONS={"textbook": (2, 3)}, REPORTED=("slow",))


def matrix(n=300, reference=300, prior=0, recipes=(0, 0), wrong_units=0, textbook=0, slow=None):
    m = dict(reference=first(n, reference), prior=first(n, prior), wrong_units=first(n, wrong_units), textbook=first(n, textbook),
             slow=first(n, n if slow is None else slow))
    m.update({f"fixed_recipe#{k}": first(n, v) for k, v in enumerate(recipes)})
    return m


# ---------------------------------------------------------------------------------------------- constants and bounds
def test_thresholds_are_the_documented_ones():
    assert (gate.ALPHA, gate.REF_MIN, gate.BLIND_MAX, gate.SEP_MIN) == (0.05, 0.90, 0.10, 0.25)
    assert (gate.TUNING_SETS, gate.TUNING_WORLDS) == (3, 16) and gate.SCHEMA == "mrgym-gate/2"


@pytest.mark.parametrize("k,n", [(0, 300), (1, 30), (21, 300), (22, 300), (5, 60), (299, 300), (150, 300)])
def test_upper_bound_is_clopper_pearson(k, n):
    assert gate.upper_bound(k, n) == pytest.approx(cp_upper(k, n), abs=1e-9)
    assert gate.upper_bound(k, n, 0.01) == pytest.approx(cp_upper(k, n, 0.01), abs=1e-9)
    assert type(gate.upper_bound(k, n)) is float


@pytest.mark.parametrize("k,n", [(300, 300), (29, 30), (279, 300), (278, 300), (55, 60), (1, 300), (150, 300)])
def test_lower_bound_is_clopper_pearson(k, n):
    assert gate.lower_bound(k, n) == pytest.approx(cp_lower(k, n), abs=1e-9)
    assert gate.lower_bound(k, n, 0.01) == pytest.approx(cp_lower(k, n, 0.01), abs=1e-9)
    assert type(gate.lower_bound(k, n)) is float


def test_bounds_at_the_ends_and_the_two_numbers_the_gate_documents():
    assert gate.upper_bound(0, 30) == pytest.approx(1 - 0.05 ** (1 / 30), abs=1e-12)        # closed form at k = 0
    assert gate.lower_bound(30, 30) == pytest.approx(0.05 ** (1 / 30), abs=1e-12)           # closed form at k = n
    assert gate.upper_bound(30, 30) == 1.0 and gate.upper_bound(31, 30) == 1.0
    assert gate.lower_bound(0, 30) == 0.0 and gate.lower_bound(-1, 30) == 0.0
    # "with n = 300 a must-fail policy is cleared when it passes at most 21 instances and the reference needs at least 279"
    assert gate.upper_bound(21, 300) <= 0.10 < gate.upper_bound(22, 300)
    assert gate.lower_bound(278, 300) < 0.90 <= gate.lower_bound(279, 300)


def test_gap_lower_bound_by_hand():
    ref, abl = [1, 1, 1, 1, 0], [0, 1, 0, 0, 0]                          # d = 1 0 1 1 0: mean 0.6, sd = sqrt(0.3)
    mean, lo = gate.gap_lower_bound(ref, abl)
    assert mean == pytest.approx(0.6, abs=1e-12)
    assert lo == pytest.approx(0.6 - T95[4] * math.sqrt(0.3) / math.sqrt(5), abs=1e-6)
    assert gate.gap_lower_bound(ref, abl, 0.5)[1] == pytest.approx(0.6, abs=1e-9)           # the median of t is 0
    assert gate.gap_lower_bound([True], [False]) == (1.0, 1.0)                              # one instance: no interval
    assert gate.gap_lower_bound([1, 1], [0, 0]) == (1.0, 1.0)                               # no spread: the mean
    mean, lo = gate.gap_lower_bound([1, 1], [0, 1])                                         # two instances: d = 1 0, sd = sqrt(0.5)
    assert mean == 0.5 and lo == pytest.approx(0.5 - T95[1] * math.sqrt(0.5) / math.sqrt(2), abs=1e-5)
    assert all(type(v) is float for v in gate.gap_lower_bound(np.array(ref, bool), np.array(abl, bool)))


# ---------------------------------------------------------------------------------------------- the verdict of a level
def test_a_level_that_meets_every_condition_at_the_edge_is_admitted():
    out = gate.decide(matrix(reference=279, prior=21, recipes=(0, 21), textbook=150, slow=299), CONTROLS, 2, 300)
    assert out["verdict"] == "admitted" and out["reasons"] == [] and out["n"] == 300
    assert out["reference"] == dict(passed=279, rate=0.93, lower95=round(cp_lower(279, 300), 4))
    assert [r["policy"] for r in out["must_fail"]] == ["fixed_recipe#0", "fixed_recipe#1", "prior", "wrong_units"]
    assert out["must_fail"][1] == dict(policy="fixed_recipe#1", passed=21, rate=0.07, upper95=round(cp_upper(21, 300), 4), cleared=True)
    assert out["must_fail"][0] == dict(policy="fixed_recipe#0", passed=0, rate=0.0, upper95=round(1 - 0.05 ** (1 / 300), 4), cleared=True)
    # the ceiling is the worst must-fail policy (the first of equals), with a bound that is simultaneous over the four
    assert out["blind_ceiling"] == dict(policy="fixed_recipe#1", rate=0.07, upper95=round(cp_upper(21, 300), 4),
                                        upper95_simultaneous=round(cp_upper(21, 300, 0.05 / 4), 4))
    d = np.r_[np.zeros(150), np.ones(129), np.zeros(21)]                 # reference minus ablation, instance by instance
    lo = d.mean() - T95[299] * d.std(ddof=1) / math.sqrt(300)
    assert out["ablations"] == [dict(policy="textbook", passed=150, rate=0.5, gap=0.43, gap_lower95=round(lo, 4), required=True,
                                     cleared=True)]
    assert out["reported"] == [dict(policy="slow", passed=299, rate=0.9967)]
    assert all(type(r["cleared"]) is bool for r in out["must_fail"] + out["ablations"])


def test_each_condition_failing_alone_makes_the_level_practice_with_its_reason():
    out = gate.decide(matrix(reference=278), CONTROLS, 2, 300)
    assert out["verdict"] == "practice" and out["reference"]["passed"] == 278
    assert out["reasons"] == [f"the reference passes 278 of 300 (lower bound {100 * round(cp_lower(278, 300), 4):.1f} % < 90 %)"]
    out = gate.decide(matrix(prior=22), CONTROLS, 2, 300)
    assert out["verdict"] == "practice" and [r["cleared"] for r in out["must_fail"]] == [True, True, False, True]
    assert out["reasons"] == [f"prior passes 22 of 300 (upper bound {100 * round(cp_upper(22, 300), 4):.1f} % > 10 %)"]
    assert out["blind_ceiling"]["policy"] == "prior" and out["blind_ceiling"]["rate"] == 0.0733
    out = gate.decide(matrix(recipes=(0, 0, 40)), CONTROLS, 2, 300)                        # any tuning set counts
    assert out["verdict"] == "practice" and out["reasons"][0].startswith("fixed_recipe#2 passes 40 of 300 (upper bound ")
    out = gate.decide(matrix(textbook=240), CONTROLS, 2, 300)                              # the ablation loses only 20 points
    d = np.r_[np.zeros(240), np.ones(60)]
    lo = d.mean() - T95[299] * d.std(ddof=1) / math.sqrt(300)
    assert out["verdict"] == "practice" and out["ablations"][0]["cleared"] is False and out["ablations"][0]["gap"] == 0.2
    assert out["reasons"] == [f"textbook loses only 20.0 points against the reference (lower bound {100 * lo:.1f} < 25)"]
    both = gate.decide(matrix(reference=200, prior=100, textbook=200), CONTROLS, 3, 300)
    assert len(both["reasons"]) == 3 and both["reasons"][0].startswith("the reference") and both["reasons"][1].startswith("prior")
    assert both["reasons"][2].startswith("textbook loses only 0.0 points")


def test_rates_and_gaps_are_rounded_to_four_decimals():
    out = gate.decide(matrix(n=299, reference=299, textbook=100, slow=100), CONTROLS, 2, 299)
    row = out["ablations"][0]
    assert (row["rate"], row["gap"]) == (0.3344, 0.6656) and (round(100 / 299, 5), round(199 / 299, 5)) == (0.33445, 0.66555)
    assert out["reported"][0]["rate"] == 0.3344 and out["reference"]["rate"] == 1.0


def test_an_ablation_counts_only_at_the_levels_it_names():
    out = gate.decide(matrix(textbook=300), CONTROLS, 1, 300)                              # level 1 is not in (2, 3)
    assert out["verdict"] == "admitted"
    assert out["ablations"] == [dict(policy="textbook", passed=300, rate=1.0, gap=0.0, gap_lower95=0.0, required=False, cleared=True)]
    assert gate.decide(matrix(textbook=300), CONTROLS, 3, 300)["verdict"] == "practice"


def test_the_three_thresholds_are_inclusive(monkeypatch):
    m = dict(matrix(n=40, reference=40), textbook=np.full(40, 0.75))     # every paired difference is exactly 0.25
    out = gate.decide(m, CONTROLS, 2, 40)
    assert out["ablations"][0]["gap_lower95"] == 0.25 and out["ablations"][0]["cleared"] is True
    monkeypatch.setattr(gate, "upper_bound", lambda k, n, alpha=0.05: 0.10)
    monkeypatch.setattr(gate, "lower_bound", lambda k, n, alpha=0.05: 0.90)
    out = gate.decide(matrix(n=40, reference=40), CONTROLS, 1, 40)
    assert out["verdict"] == "admitted" and all(r["cleared"] for r in out["must_fail"])
    monkeypatch.setattr(gate, "upper_bound", lambda k, n, alpha=0.05: 0.1001)
    assert gate.decide(matrix(n=40, reference=40), CONTROLS, 1, 40)["verdict"] == "practice"
    monkeypatch.setattr(gate, "upper_bound", lambda k, n, alpha=0.05: 0.10)
    monkeypatch.setattr(gate, "lower_bound", lambda k, n, alpha=0.05: 0.8999)
    assert gate.decide(matrix(n=40, reference=40), CONTROLS, 1, 40)["verdict"] == "practice"


# ---------------------------------------------------------------------------------------------- one instance, the tuning
def _number(level, seed, split="gate"):
    """The number of the stub instance the gate plays for ``seed``: public split ``gate`` (``tune`` for the tuning worlds)."""
    return stub_task.number(derive_seed("Test/Stub-v0", level, split, seed, environ={}), level)


def _recipes(level):
    out = []
    for k in range(3):
        numbers = [_number(level, f"{k}:{i}", "tune") for i in range(16)]
        out.append(dict(value=min(set(numbers), key=lambda v: (-numbers.count(v), v)), tuned_on=16))
    return out


def test_instances_and_tuning_worlds_come_from_the_public_gate_and_tune_splits(stub_env):
    spec = mrgym.spec(stub_env)
    cfg = gate._instance(spec, 2, "1:7")                                 # the split of the tuning worlds by default
    assert cfg["seed"] == derive_seed(stub_env, 2, "tune", "1:7", environ={}) and cfg["level"] == 2
    assert cfg["task"]["number"] == _number(2, "1:7", "tune")
    played = gate._instance(spec, 2, 7, "gate")
    assert played["seed"] == derive_seed(stub_env, 2, "gate", 7, environ={}) != derive_seed(stub_env, 2, "train", 7, environ={})
    # the label is the trainer's (never the agent's) and its format is not part of any contract: only that there is one
    assert isinstance(cfg["task_id"], str) and cfg["task_id"] and cfg["task_id"] != played["task_id"]
    for level in (1, 2):
        assert [gate._tune((stub_env, level, k, 16)) for k in range(3)] == _recipes(level)
    assert gate._tune((stub_env, 1, 1, 5))["tuned_on"] == 5


def test_one_instance_plays_every_policy_once_and_each_tuned_recipe(stub_env):
    recipes = [dict(value=_number(1, 0)), dict(value=1001)]               # the first recipe is this instance's number
    out = gate._play((stub_env, 1, 0, recipes))
    assert list(out) == ["reference", "prior", "brute_force", "wrong_units", "even_only", "late", "fixed_recipe#0", "fixed_recipe#1"]
    even = _number(1, 0) % 2 == 0
    assert {k: v["passed"] for k, v in out.items()} == dict(reference=True, prior=False, brute_force=False, wrong_units=False,
                                                           even_only=even, late=True, **{"fixed_recipe#0": True, "fixed_recipe#1": False})
    assert all(type(v["passed"]) is bool for v in out.values())
    assert out["reference"]["error"] == 0.0 and out["fixed_recipe#1"]["error"] == float(1001 - _number(1, 0))
    assert out["brute_force"]["error"] is None                            # a refused deliverable has no error
    scan_s = 3.0 + 3 * 0.050                                              # overhead + (1 average + 2 dummy shots) x TR
    assert out["reference"]["used_s"] == pytest.approx(scan_s) and out["late"]["used_s"] == pytest.approx(3 * scan_s)
    assert out["prior"]["used_s"] == 0.0
    assert all(v["policy_error"] is None and 0 <= v["seconds"] < 30 for v in out.values())
    assert set(out["reference"]) == {"passed", "error", "used_s", "policy_error", "seconds"}


def test_a_policy_that_raises_is_a_failed_instance_with_its_error_and_the_seconds_are_wall_time(stub_env, monkeypatch):
    def boom(sc):
        raise ZeroDivisionError("division by zero")
    monkeypatch.setattr(stub_controls, "late", boom)
    monkeypatch.setattr(gate, "time", FakeClock(1.23456))
    out = gate._play((stub_env, 2, 1, []))
    assert out["late"]["passed"] is False and out["late"]["policy_error"] == "ZeroDivisionError: division by zero"
    assert out["reference"]["passed"] is True and out["reference"]["policy_error"] is None
    assert {v["seconds"] for v in out.values()} == {1.235} and all(type(v["seconds"]) is float for v in out.values())


def test_a_result_without_a_score_or_a_budget_counts_as_a_fail(stub_env, monkeypatch):
    monkeypatch.setattr(policies, "run", lambda env, name, recipe=None: mrgym.StepResult((None, 0.0, False, True, {})))
    out = gate._play((stub_env, 1, 0, []))
    assert out["reference"] == dict(passed=False, error=None, used_s=None, policy_error=None, seconds=out["reference"]["seconds"])


def test_instances_do_not_depend_on_a_secret_in_the_environment(stub_env, monkeypatch):
    spec = mrgym.spec(stub_env)
    public = gate._instance(spec, 1, 3, "gate")
    monkeypatch.setenv("MRGYM_SECRET", "ab" * 32)
    monkeypatch.setenv("MRGYM_EPOCH", "7")
    monkeypatch.setenv("MRGYM_SALT", "cd" * 16)                          # a trainer's salt moves public worlds, on purpose ...
    assert derive_seed(stub_env, 1, "gate", 3) != public["seed"]
    assert gate._instance(spec, 1, 3, "gate") == public                  # ... but never the gate's: it is the same for everyone


def test_a_family_that_takes_the_split_gets_gate_and_tune(stub_env, monkeypatch):
    plain = stub_task.make_instance

    def make_instance(seed, level, task_id=None, split="train"):
        cfg = plain(seed, level, task_id)
        return dict(cfg, task=dict(cfg["task"], split=split))
    monkeypatch.setattr(stub_task, "make_instance", make_instance)
    spec = mrgym.spec(stub_env)
    for split in ("gate", "tune"):
        cfg = gate._instance(spec, 2, 7, split)
        assert cfg["task"]["split"] == split
        assert cfg["task_id"] == keys.instance_label("stub", derive_seed(stub_env, 2, split, 7, environ={}), split)
    assert gate._instance(spec, 2, "0:1")["task"]["split"] == "tune"


# ---------------------------------------------------------------------------------------------- the run
def _fake_play(passes_of):
    """A stand-in for ``gate._play``: {policy: bool} per (level, instance index) from ``passes_of``; records its calls."""
    calls = []

    def play(args):
        env_id, level, seed, recipes = args
        calls.append(args)
        i = int(seed)
        row = lambda ok, err, used, perr=None: dict(passed=ok, error=err, used_s=used, policy_error=perr, seconds=0.0)   # noqa: E731
        out = {k: row(bool(v), None if k == "brute_force" else 0.01 * (i % 5), 3.15) for k, v in passes_of(level, i).items()}
        out["reference"].update(error=0.0012347 * i, used_s=float(i), policy_error="Boom: x" if i in (3, 4) else None)
        if i == 0:
            out["reference"].update(error=None, used_s=None)
        return out
    return play, calls


def _run(monkeypatch, pool, passes_of, n, **kw):
    play, calls = _fake_play(passes_of)
    tunes = []
    monkeypatch.setattr(gate, "multiprocessing", pool)
    monkeypatch.setattr(gate, "time", FakeClock())
    monkeypatch.setattr(gate, "datetime", FakeDatetime)
    monkeypatch.setattr(gate, "_play", play)
    monkeypatch.setattr(gate, "_tune", lambda a: tunes.append(a) or dict(value=a[2], level=a[1]))
    return (gate.run("Test/Stub-v0", **kw) if n is None else gate.run("Test/Stub-v0", n=n, **kw)), calls, tunes


def _stub_row(ref=True, recipe=False, even=False):
    return dict(reference=ref, prior=False, brute_force=False, wrong_units=False, even_only=even, late=True,
                **{"fixed_recipe#0": recipe, "fixed_recipe#1": False, "fixed_recipe#2": False})


def test_run_assembles_the_stored_result(stub_env, in_process_pool, monkeypatch):
    # level 1: the reference fails 2 instances of every block of 60, the ablation passes every second instance;
    # level 2: the first tuned recipe passes every second instance
    def passes_of(level, i):
        return _stub_row(ref=i % 30 != 0, even=i % 2 == 0) if level == 1 else _stub_row(recipe=i % 2 == 0)
    res, calls, tunes = _run(monkeypatch, in_process_pool, passes_of, 300, workers=99)
    assert (in_process_pool.method, in_process_pool.workers, in_process_pool.maps) == ("fork", 8, 4)
    assert list(res) == ["schema", "canary", "env_id", "engine", "package_version", "date_utc", "command", "seeds", "thresholds",
                         "descriptions", "levels", "wall_seconds"]
    assert res["schema"] == "mrgym-gate/2" and res["env_id"] == stub_env and res["engine"] == mrgym.ENGINE_ID
    assert res["canary"] == keys.CANARY and "CANARY" in res["canary"]
    assert res["package_version"] == mrgym.__version__
    assert res["date_utc"] == "2026-11-25"                               # the UTC date, not the local one
    assert res["command"] == "mrgym gate Test/Stub-v0 --n 300 --write"
    assert res["seeds"] == dict(gate="split gate, seeds 0..299", tune="split tune, seeds <set 0..2>:<0..15>")
    assert res["thresholds"] == dict(alpha=0.05, reference_lower_min=0.90, blind_upper_max=0.10, ablation_gap_lower_min=0.25)
    assert res["descriptions"] == stub_controls.DESCRIPTIONS and res["descriptions"] is not stub_controls.DESCRIPTIONS
    assert list(res["levels"]) == ["1", "2"]
    # the clock is read at the start, at the start and the end of each level, and at the end: 1.234 s apart
    assert (res["wall_seconds"], res["levels"]["1"]["wall_seconds"], res["levels"]["2"]["wall_seconds"]) == (6.2, 1.2, 1.2)
    assert all(type(v) is float for v in (res["wall_seconds"], res["levels"]["1"]["wall_seconds"]))
    # what the workers were asked: three tuning sets of 16 worlds per level, then n fresh instances with those recipes
    assert tunes == [(stub_env, lv, k, 16) for lv in (1, 2) for k in range(3)]
    recipes = {lv: [dict(value=k, level=lv) for k in range(3)] for lv in (1, 2)}
    assert calls == [(stub_env, lv, i, recipes[lv]) for lv in (1, 2) for i in range(300)]

    one = res["levels"]["1"]
    assert one["verdict"] == "admitted" and one["reasons"] == [] and one["n"] == 300 and one["fixed_recipes"] == recipes[1]
    assert one["stability"] == dict(leave_one_block_out=["admitted"] * 5)
    errs = [0.0012347 * i for i in range(1, 300)]                      # instance 0 reported no error
    assert round(float(np.quantile(errs, 0.95)), 5) != round(float(np.quantile(errs, 0.95)), 6)      # the rounding is visible
    assert one["reference"] == dict(passed=290, rate=0.9667, lower95=round(cp_lower(290, 300), 4),
                                    error_median=round(float(np.median(errs)), 5), error_q95=round(float(np.quantile(errs, 0.95)), 5),
                                    scan_s_median=149.5, policy_errors=["Boom: x"])
    assert one["ablations"][0]["policy"] == "even_only" and one["ablations"][0]["passed"] == 150 and one["ablations"][0]["cleared"]
    assert one["ablations"][0]["required"] is True                       # the stub lists the ablation for level 1
    assert [r["policy"] for r in one["reported"]] == ["late"]

    two = res["levels"]["2"]
    k_up = round(cp_upper(150, 300), 4)
    assert two["verdict"] == "practice" and two["stability"] == dict(leave_one_block_out=["practice"] * 5)
    assert two["reasons"] == [f"fixed_recipe#0 passes 150 of 300 (upper bound {100 * k_up:.1f} % > 10 %)"]
    assert two["ablations"][0]["required"] is False and two["fixed_recipes"] == recipes[2]
    json.dumps(res)                                                      # the result is stored as JSON


def test_a_verdict_that_depends_on_one_fifth_of_the_instances_is_practice(stub_env, in_process_pool, monkeypatch):
    # 279 of 300 pass, all 21 failures in the first block: without that block 240 of 240, without another 219 of 240
    res, _, _ = _run(monkeypatch, in_process_pool, lambda level, i: _stub_row(ref=i >= 21, even=i % 2 == 0), 300, levels=[1], workers=0)
    assert (in_process_pool.workers, list(res["levels"])) == (1, ["1"])
    lv = res["levels"]["1"]
    assert cp_lower(279, 300) >= 0.90 > cp_lower(219, 240)
    assert lv["stability"] == dict(leave_one_block_out=["admitted", "practice", "practice", "practice", "practice"])
    assert lv["verdict"] == "practice"
    assert lv["reasons"] == ["borderline: the verdict changes when one fifth of the instances is left out"]


def test_the_left_out_blocks_are_judged_at_the_same_level(stub_env, in_process_pool, monkeypatch):
    # the ablation does not separate at level 1, where it is required: practice with that reason, and stably so
    res, _, _ = _run(monkeypatch, in_process_pool, lambda level, i: _stub_row(even=True), 300, levels=[1])
    lv = res["levels"]["1"]
    assert lv["verdict"] == "practice" and lv["stability"] == dict(leave_one_block_out=["practice"] * 5)
    assert lv["reasons"] == ["even_only loses only 0.0 points against the reference (lower bound 0.0 < 25)"]


def test_one_instance_is_a_valid_run_and_the_defaults_are_the_documented_ones(stub_env, in_process_pool, monkeypatch):
    res, calls, _ = _run(monkeypatch, in_process_pool, lambda level, i: _stub_row(), 1, levels=[2])
    assert len(calls) == 1 and res["levels"]["2"]["n"] == 1 and res["levels"]["2"]["stability"] == dict(leave_one_block_out=[])
    defaults = {k: v.default for k, v in inspect.signature(gate.run).parameters.items() if v.default is not inspect.Parameter.empty}
    assert defaults == dict(n=300, levels=None, workers=2, blocks=5)
    res, calls, _ = _run(monkeypatch, in_process_pool, lambda level, i: _stub_row(), None)             # a call without arguments
    assert len(calls) == 2 * 300 and in_process_pool.workers == 2 and res["command"] == "mrgym gate Test/Stub-v0 --n 300 --write"
    assert [len(lv["stability"]["leave_one_block_out"]) for lv in res["levels"].values()] == [5, 5]


@pytest.mark.parametrize("n,blocks,want", [(50, 5, 5), (49, 5, 0), (60, 2, 2), (30, 3, 3), (29, 3, 0), (40, 1, 0), (10, 1, 0)])
def test_stability_needs_ten_instances_per_block(stub_env, in_process_pool, monkeypatch, n, blocks, want):
    res, _, _ = _run(monkeypatch, in_process_pool, lambda level, i: _stub_row(even=i % 2 == 0), n, levels=(1,), blocks=blocks)
    lv = res["levels"]["1"]
    assert len(lv["stability"]["leave_one_block_out"]) == want
    left = max(n - n // blocks, 1)                                      # instances that remain when one block is left out
    stable = "admitted" if 0.05 ** (1 / left) >= 0.90 and 1 - 0.05 ** (1 / left) <= 0.10 else "practice"
    full = "admitted" if 0.05 ** (1 / n) >= 0.90 and 1 - 0.05 ** (1 / n) <= 0.10 else "practice"
    assert lv["stability"]["leave_one_block_out"] == [stable] * want
    assert lv["verdict"] == (full if not want or stable == full else "practice")
    assert lv["reference"]["policy_errors"] == ["Boom: x"] and lv["reference"]["scan_s_median"] == float(np.median([0.0, *range(1, n)]))


def test_a_reference_without_any_error_value_is_reported_as_none(stub_env, in_process_pool, monkeypatch):
    play, _ = _fake_play(lambda level, i: _stub_row())
    monkeypatch.setattr(gate, "multiprocessing", in_process_pool)
    monkeypatch.setattr(gate, "_tune", lambda a: dict(value=0))
    monkeypatch.setattr(gate, "_play", lambda a: {k: dict(v, error=None, used_s=None, policy_error=f"E{a[2]}") for k, v in play(a).items()})
    ref = gate.run("Test/Stub-v0", n=12, levels=[2])["levels"]["2"]["reference"]
    assert ref["error_median"] is None and ref["error_q95"] is None and ref["scan_s_median"] == 0.0
    assert ref["policy_errors"] == ["E0", "E1", "E10"]                    # at most three, sorted


def test_the_run_on_the_stub_family_gives_what_the_seeds_say(stub_env, in_process_pool, monkeypatch):
    monkeypatch.setattr(gate, "multiprocessing", in_process_pool)
    n = 12
    res = gate.run(stub_env, n=n, levels=[2])
    lv = res["levels"]["2"]
    numbers = [_number(2, i) for i in range(n)]
    recipes = _recipes(2)
    assert lv["fixed_recipes"] == recipes and lv["stability"] == dict(leave_one_block_out=[])
    want = {f"fixed_recipe#{k}": sum(v == r["value"] for v in numbers) for k, r in enumerate(recipes)}
    want.update(prior=0, brute_force=0, wrong_units=0)
    assert {r["policy"]: r["passed"] for r in lv["must_fail"]} == want
    assert lv["reference"]["passed"] == n and lv["reference"]["lower95"] == round(0.05 ** (1 / n), 4)
    assert lv["reference"]["error_median"] == 0.0 and lv["reference"]["scan_s_median"] == round(3.15, 1)
    assert lv["reference"]["policy_errors"] == []
    assert lv["ablations"] == [dict(policy="even_only", passed=sum(v % 2 == 0 for v in numbers), rate=round(numbers.count(0) / n, 4),
                                    gap=round(numbers.count(1) / n, 4), gap_lower95=lv["ablations"][0]["gap_lower95"],
                                    required=False, cleared=True)]
    assert lv["reported"] == [dict(policy="late", passed=n, rate=1.0)]
    assert lv["verdict"] == "practice" and lv["reasons"][0].startswith(f"the reference passes {n} of {n} (lower bound ")


def test_a_practice_level_is_outside_the_claim_and_the_default_run(stub_env, in_process_pool, monkeypatch):
    spec = mrgym.spec(stub_env)
    assert gate.claim_levels(spec) == [1, 2] and gate.claim_levels(mrgym.spec("Scanner/Explore-v0")) == [1]
    with monkeypatch.context() as m:
        m.setitem(stub_task.LEVELS, 1, dict(stub_task.LEVELS[1], kind="practice"))
        m.delitem(stub_task.LEVELS, 2)                                   # a level the table does not describe is claimed
        assert gate.claim_levels(spec) == [2]
        m.delattr(stub_task, "LEVELS")                                   # and so is every level of a family without a table
        assert gate.claim_levels(spec) == [1, 2]
    monkeypatch.setitem(stub_task.LEVELS, 1, dict(stub_task.LEVELS[1], kind="practice"))
    assert gate.claim_levels(spec) == [2]
    res, calls, _ = _run(monkeypatch, in_process_pool, lambda level, i: _stub_row(), 12)
    assert list(res["levels"]) == ["2"] and {c[1] for c in calls} == {2}
    res, _, _ = _run(monkeypatch, in_process_pool, lambda level, i: _stub_row(), 12, levels=[1])       # still playable on request
    assert list(res["levels"]) == ["1"]


def test_a_sandbox_has_no_gate():
    with pytest.raises(mrgym.Error, match="Scanner/Explore-v0 is a sandbox: it has no gate"):
        gate.run("Scanner/Explore-v0", n=2)


# ---------------------------------------------------------------------------------------------- storage and texts
def _stored(verdicts=("admitted", "admitted")):
    levels = {}
    for i, v in enumerate(verdicts, 1):
        levels[str(i)] = dict(
            verdict=v, n=300 - i, wall_seconds=12.5,
            reference=dict(passed=290, rate=0.9667, lower95=0.94 + 0.01 * i, error_median=0.012, error_q95=0.031, scan_s_median=88.2),
            must_fail=[dict(policy="prior", passed=0, rate=0.0, upper95=0.0099, cleared=True),
                       dict(policy="fixed_recipe#0", passed=30, rate=0.1, upper95=0.1334, cleared=v == "admitted")],
            blind_ceiling=dict(policy="fixed_recipe#0", rate=0.1, upper95=0.0545 + 0.01 * i),
            ablations=[dict(policy="textbook", passed=120, rate=0.4, gap=0.5667, gap_lower95=0.5123, required=i > 1, cleared=v == "admitted")],
            reported=[dict(policy="slow", passed=299, rate=0.9967)],
            reasons=[] if v == "admitted" else [f"reason {i}a", f"reason {i}b"])
    return dict(schema=gate.SCHEMA, env_id="Test/Stub-v0", engine="mrgym-engine-9.9.9", levels=levels, wall_seconds=25.0)


def test_write_load_and_the_file_name(tmp_path, monkeypatch):
    res = _stored()
    gate.write(dict(res, env_id="Other/First-v0"), str(tmp_path / "deep" / "er"))        # the directory may exist already
    path = gate.write(res, str(tmp_path / "deep" / "er"))
    assert path == str(tmp_path / "deep" / "er" / "Test-Stub-v0.gate.json") == gate._path("Test/Stub-v0", str(tmp_path / "deep" / "er"))
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert text == json.dumps(res, indent=1) + "\n"
    default = gate._path("Scanner/NullTissue-v0")
    assert default.replace("\\", "/").endswith("mrgym/envs/results/Scanner-NullTissue-v0.gate.json")
    assert gate.load("Scanner/NoSuchFamily-v0") is None
    monkeypatch.setattr(gate, "_path", lambda env_id, directory=None: path if env_id == "Test/Stub-v0" else str(tmp_path / "none.json"))
    assert gate.load("Test/Stub-v0") == res and gate.load("Other/Id-v0") is None
    assert gate.level_status("Test/Stub-v0", 2) == dict(verdict="admitted", reference_lower95=0.96, blind_upper95=0.0745,
                                                       engine="mrgym-engine-9.9.9", n=298)
    assert gate.level_status("Test/Stub-v0", 3) is None and gate.level_status("Other/Id-v0", 1) is None


def test_status_line(stub_env):
    spec = mrgym.spec(stub_env)
    here = lambda *a: dict(_stored(*a), engine=mrgym.ENGINE_ID)             # noqa: E731  a result measured on the engine that is installed
    assert gate.status_line(mrgym.spec("Scanner/Explore-v0"), here()) == "SANDBOX: no reward"
    assert gate.status_line(spec, None) == "CANDIDATE: the gate has not been run (no stored result)"
    assert gate.status_line(spec, {}) == "CANDIDATE: the gate has not been run (no stored result)"
    assert gate.status_line(spec, here()) == ("ADMITTED: on 298 fresh instances per level the reference passes >= 95 %, "
                                              "no policy that ignores the object passes more than 7 % (95 % bounds)")
    assert gate.status_line(spec, here(("admitted", "practice"))) == "PRACTICE: level(s) 1 admitted; reason 2a"
    assert gate.status_line(spec, here(("practice", "admitted", "admitted"))) == "PRACTICE: level(s) 2, 3 admitted; reason 1a"
    assert gate.status_line(spec, here(("practice", "practice"))) == "PRACTICE: level(s) none admitted; reason 1a"
    # a result from another engine is not repeated as a verdict: it says nothing about the engine that is installed
    stale = gate.status_line(spec, _stored())
    assert stale == (f"CANDIDATE: the gate has not been run on this engine ({mrgym.ENGINE_ID}); the stored result is from another engine "
                     "(mrgym-engine-9.9.9) and says nothing about this one") and "ADMITTED" not in stale


def test_text_table():
    lines = gate.text_table(_stored(("admitted", "practice"))).splitlines()
    assert lines == [
        "== Test/Stub-v0 level 1: ADMITTED on 299 fresh instances (12.5 s)",
        "   reference           290/299   96.7 %   lower bound  95.0 %   error median 0.012 q95 0.031   scan 88.2 s",
        "   prior                 0/299    0.0 %   upper bound   1.0 %   cleared",
        "   fixed_recipe#0       30/299   10.0 %   upper bound  13.3 %   cleared",
        "   textbook            120/299   40.0 %   gap  56.7 points, lower bound  51.2   (reported)",
        "   slow                299/299   99.7 %   (reported)",
        "== Test/Stub-v0 level 2: PRACTICE on 298 fresh instances (12.5 s)",
        "   reference           290/298   96.7 %   lower bound  96.0 %   error median 0.012 q95 0.031   scan 88.2 s",
        "   prior                 0/298    0.0 %   upper bound   1.0 %   cleared",
        "   fixed_recipe#0       30/298   10.0 %   upper bound  13.3 %   NOT CLEARED",
        "   textbook            120/298   40.0 %   gap  56.7 points, lower bound  51.2   (required, NOT CLEARED)",
        "   slow                299/298   99.7 %   (reported)",
        "   reason: reason 2a",
        "   reason: reason 2b",
        "wall 25.0 s",
    ]
    ok = gate.text_table(_stored(("admitted", "admitted"))).splitlines()
    assert ok[10] == "   textbook            120/298   40.0 %   gap  56.7 points, lower bound  51.2   (required)"
