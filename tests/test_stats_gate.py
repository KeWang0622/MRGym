"""The statistics of the admission gate (``mrgym.stats.gate``) and its operating characteristic.

Known answers on synthetic pass matrices: a family that violates one condition is admitted in at most 5 % of the
repetitions, a family that meets all of them comfortably in at least 95 %. The gate's own module keeps its rule and
its file: it imports the three bound functions from ``mrgym.stats``.
"""
import types

import numpy as np
import pytest
from scipy import stats as st

from mrgym import cli, gate, stats
from mrgym.stats import gate as G
from mrgym.stats import power

N, REPS = 300, 2000


def test_the_gate_module_uses_the_bounds_of_mrgym_stats():
    assert gate.upper_bound is G.upper_bound and gate.lower_bound is G.lower_bound and gate.gap_lower_bound is G.gap_lower_bound
    assert (gate.ALPHA, gate.REF_MIN, gate.BLIND_MAX, gate.SEP_MIN) == (G.ALPHA, G.REF_MIN, G.S_MAX, G.SEP_MIN) == (0.05, 0.90, 0.10, 0.25)
    assert gate.SCHEMA == "mrgym-gate/2"
    assert stats.upper_bound(21, 300) == pytest.approx(st.beta.ppf(0.95, 22, 279)) and stats.upper_bound(300, 300) == 1.0
    assert stats.lower_bound(279, 300) == pytest.approx(st.beta.ppf(0.05, 279, 22)) and stats.lower_bound(0, 300) == 0.0
    assert stats.upper_bound(21, 300) == stats.clopper_pearson(21, 300, 0.95, "upper")[1]
    ref, abl = np.r_[np.ones(280), np.zeros(20)], np.r_[np.ones(150), np.zeros(150)]
    mean, low = stats.gap_lower_bound(ref, abl)
    d = ref - abl
    assert mean == pytest.approx(d.mean()) and low == pytest.approx(d.mean() - st.t.ppf(0.95, 299) * d.std(ddof=1) / np.sqrt(300))


def test_thresholds_in_pass_counts():
    """With 300 instances a must-fail policy is cleared at 21 passes and not at 22; the reference needs 279."""
    assert stats.upper_bound(21, N) <= 0.10 < stats.upper_bound(22, N)
    assert stats.lower_bound(279, N) >= 0.90 > stats.lower_bound(278, N)
    assert (power.cleared_at_most(100), power.cleared_at_most(300), power.cleared_at_most(1000)) == (4, 21, 84)
    assert power.reference_at_least(300) == 279


def test_one_blind_policy_at_the_limit_is_admitted_in_at_most_five_percent():
    rng = np.random.default_rng(20261125)
    admitted = 0
    for _ in range(REPS):
        out = stats.family_gate({"constant": rng.random(N) < 0.10}, rng.random(N) < 0.99, {"ablation": rng.random(N) < 0.50})
        admitted += out["verdict"] == "admitted"
    assert admitted / REPS <= 0.05, admitted / REPS


def test_a_family_that_meets_every_condition_is_admitted_in_at_least_ninety_five_percent():
    rng = np.random.default_rng(20261126)
    admitted = 0
    for _ in range(REPS):
        tier1 = {f"constant#{j}": rng.random(N) < 0.03 for j in range(84)}
        out = stats.family_gate(tier1, rng.random(N) < 0.99, {"ablation": rng.random(N) < 0.50})
        admitted += out["verdict"] == "admitted"
    assert admitted / REPS >= 0.95, admitted / REPS


@pytest.mark.parametrize("violation", ["reference", "ablation"])
def test_the_other_two_conditions_have_the_same_error_rate(violation):
    rng = np.random.default_rng(5)
    admitted = 0
    for _ in range(REPS):
        ref = rng.random(N) < (0.90 if violation == "reference" else 0.99)
        abl = ref & (rng.random(N) < 0.5) if violation == "reference" else np.where(rng.random(N) < 0.25, False, ref)   # gap exactly 25 points of the reference's passes
        admitted += stats.family_gate({"prior": np.zeros(N, bool)}, ref, {"ablation": abl})["verdict"] == "admitted"
    assert admitted / REPS <= 0.05, admitted / REPS


def test_verdict_rows_and_reasons():
    ref = np.r_[np.ones(295, bool), np.zeros(5, bool)]
    good = stats.family_gate({"prior": np.zeros(N, bool), "constant": np.arange(N) < 9}, ref, {"textbook": np.arange(N) < 120})
    assert good["verdict"] == "admitted" and good["reasons"] == [] and good["n"] == N
    assert good["reference"] == dict(passed=295, rate=295 / 300, lower95=stats.lower_bound(295, 300), cleared=True)
    assert [r["member"] for r in good["tier1"]] == ["constant", "prior"] and all(r["cleared"] for r in good["tier1"])
    assert good["tier2"][0]["gap"] == pytest.approx((295 - 120) / 300) and good["tier2"][0]["cleared"]
    ceiling = good["shortcut_ceiling"]
    assert ceiling["member"] == "constant" and ceiling["members"] == 2
    assert ceiling["upper95"] == stats.upper_bound(9, 300) < ceiling["upper95_simultaneous"] == stats.upper_bound(9, 300, 0.025)
    bad = stats.family_gate({"constant": np.arange(N) < 30}, np.arange(N) < 260, {"textbook": np.arange(N) < 230})
    assert bad["verdict"] == "practice" and len(bad["reasons"]) == 3
    assert "the reference passes 260 of 300" in bad["reasons"][0] and "constant passes 30 of 300" in bad["reasons"][1]
    with pytest.raises(ValueError):
        stats.family_gate({"short": np.zeros(10, bool)}, ref, {})


def test_stability_under_leave_one_block_out():
    ref = np.ones(N, bool)
    steady = stats.stability({"constant": np.arange(N) % 30 == 0}, ref, {"textbook": np.arange(N) % 2 == 0})
    assert steady == dict(block_verdicts=["admitted"] * 5, verdict="admitted", stable=True)
    clustered = np.zeros(N, bool)
    clustered[:21] = True                                     # 21 passes, all in the first block: cleared on 300, and on 240 without that block,
    edge = stats.stability({"constant": clustered}, ref, {})  # but not on the 240 that keep it (21 of 240 has an upper bound above 10 %)
    assert edge["block_verdicts"] == ["admitted"] + ["practice"] * 4 and edge["verdict"] == "borderline" and edge["stable"] is False
    assert stats.stability({"constant": np.zeros(20, bool)}, np.ones(20, bool), {})["stable"] is None      # too few instances for blocks
    spread = G.tuned_constant([0.20, 0.25, 0.15, 0.20])
    assert spread["sets"] == 4 and spread["mean"] == pytest.approx(0.20) and (spread["min"], spread["max"]) == (0.15, 0.25)


def test_family_gate_agrees_with_the_gate_module_on_one_matrix():
    """``mrgym.gate.decide`` (the lead's rule and file schema) and ``mrgym.stats.family_gate`` give the same verdict
    and the same bounds on the same pass matrix."""
    rng = np.random.default_rng(1)
    controls = types.SimpleNamespace(REFERENCE="reference", MUST_FAIL=("prior", "fixed_recipe"), ABLATIONS={"textbook": (2,)}, REPORTED=())
    for p_const in (0.02, 0.12):
        passes = {"reference": rng.random(N) < 0.99, "prior": np.zeros(N, bool), "fixed_recipe#0": rng.random(N) < p_const,
                  "fixed_recipe#1": rng.random(N) < 0.02, "textbook": rng.random(N) < 0.4}
        old = gate.decide(passes, controls, 2, N)
        new = stats.family_gate({k: v for k, v in passes.items() if k.split("#")[0] in controls.MUST_FAIL}, passes["reference"],
                                {"textbook": passes["textbook"]})
        assert old["verdict"] == new["verdict"]
        assert old["reference"]["lower95"] == round(new["reference"]["lower95"], 4)
        assert [r["upper95"] for r in old["must_fail"]] == [round(r["upper95"], 4) for r in new["tier1"]]
        assert old["ablations"][0]["gap_lower95"] == round(new["tier2"][0]["gap_lower95"], 4)
        assert old["blind_ceiling"]["upper95_simultaneous"] == round(new["shortcut_ceiling"]["upper95_simultaneous"], 4)
        assert set(old) == {"n", "reference", "must_fail", "ablations", "reported", "reasons", "blind_ceiling", "verdict"}   # the file schema is unchanged


def test_operating_characteristic_table(capsys):
    oc = {n: stats.gate_power(n) for n in (100, 300, 1000)}
    assert [oc[n]["cleared_at_most"] for n in (100, 300, 1000)] == [4, 21, 84]
    assert [round(oc[100]["p_cleared"][p], 2) for p in (0.02, 0.03, 0.05, 0.07, 0.10)] == [0.95, 0.82, 0.44, 0.16, 0.02]
    assert [round(oc[300]["p_cleared"][p], 3) for p in (0.05, 0.07, 0.10, 0.12)] == [0.951, 0.558, 0.046, 0.003]
    assert [round(oc[1000]["p_cleared"][p], 3) for p in (0.07, 0.10)] == [0.961, 0.049]
    assert max(g["p_cleared"][0.10] for g in oc.values()) <= 0.05                       # the stated error rate, at the limit
    assert oc[300]["reference_at_least"] == 279
    assert round(oc[300]["p_reference"][0.95], 3) == 0.951 and round(oc[300]["p_reference"][0.90], 3) == 0.046
    assert power.ablation_gap_bound(300, 0.98, 0.60) > 0.31 and power.ablation_gap_bound(300, 0.98, 0.70) < 0.25
    assert cli.main(["stats", "power", "--gate"]) == 0
    out = capsys.readouterr().out
    assert "<= 21" in out and ">= 279" in out and "0.046" in out and "P(a must-fail policy is cleared" in out
