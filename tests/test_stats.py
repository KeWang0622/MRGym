"""``mrgym.stats``: the interval, the paired test, the power of a design, reliability and the sequential test.

The numbers pinned here are the acceptance table of the statistics specification. The pilot rows come from
``tests/data/s01_nowrap_pilot.json``: the pass bits of three open models on 96 instances x 8 episodes of
``Scanner/NoWrap-v0`` (the s01 pilot run), which is where the between-instance share of variance was measured.
"""
import json
import os
import re

import numpy as np
import pytest
from scipy import stats as st

from mrgym import cli, stats
from mrgym.stats import power, reliability, sequential

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def pilot():
    with open(os.path.join(ROOT, "tests", "data", "s01_nowrap_pilot.json"), encoding="utf-8") as f:
        return json.load(f)


def episodes(pilot, agent):
    """(pass per episode, instance key per episode) of one model."""
    y, key = [], []
    for k, bits in sorted(pilot["agents"][agent].items()):
        y += [float(b) for b in bits]
        key += [k] * len(bits)
    return y, key


def means(pilot, agent):
    return {k: sum(map(int, bits)) / len(bits) for k, bits in pilot["agents"][agent].items()}


# ------------------------------------------------------------------ the acceptance table
def test_clustered_rate_of_the_27b_pilot(pilot):
    y, key = episodes(pilot, "qwen3.8-27b")
    r = stats.cluster_rate(y, key)
    assert (r["episodes"], r["instances"], int(sum(y))) == (768, 96, 285) and r["method"] == "wilson-kg-t"
    assert round(r["rate"], 3) == 0.371 and (round(r["lo"], 3), round(r["hi"], 3)) == (0.330, 0.414)
    assert round(r["n_eff"]) == 518 and round(r["deff"], 2) == 1.48 and r["one_sided_bound"] is None
    assert round(stats.icc_oneway(y, key), 3) == 0.068
    naive = stats.wilson(sum(y), len(y))                       # episodes as units: too narrow
    assert naive[1] - naive[0] < r["hi"] - r["lo"]


def test_zero_passes_fall_back_to_instances_not_episodes(pilot):
    y, key = episodes(pilot, "qwen3.5-9b")
    r = stats.cluster_rate(y, key)
    assert sum(y) == 0 and (r["episodes"], r["instances"]) == (768, 96)
    assert r["lo"] == 0.0 and round(100 * r["hi"], 1) == 3.9 and round(100 * r["one_sided_bound"], 1) == 3.1
    assert r["n_eff"] == 96 and r["hi"] > 0.03                  # never the 0-0.5 % that 768 independent episodes would give
    assert stats.wilson(0, 768)[1] < 0.006


def test_paired_comparison_27b_against_4b(pilot):
    c = stats.paired_compare(means(pilot, "qwen3.8-27b"), means(pilot, "qwen3.5-4b"))
    assert c["instances"] == 96 and c["test"] == "paired-t"
    assert round(100 * c["diff"], 1) == 36.5 and (round(100 * c["lo"], 1), round(100 * c["hi"], 1)) == (32.2, 40.7)
    assert round(c["corr"], 2) == 0.02 and (c["wins"], c["losses"], c["ties"]) == (90, 1, 5) and c["p"] < 1e-20


def test_clopper_pearson_upper_bounds():
    assert round(stats.clopper_pearson(0, 96, side="upper")[1], 4) == 0.0307
    assert round(stats.clopper_pearson(0, 300, side="upper")[1], 4) == 0.0099
    assert stats.clopper_pearson(0, 96, side="upper")[0] == 0.0 and stats.clopper_pearson(96, 96, side="lower") == (pytest.approx(0.05 ** (1 / 96)), 1.0)
    lo, hi = stats.clopper_pearson(30, 120)
    assert lo == pytest.approx(st.beta.ppf(0.025, 30, 91)) and hi == pytest.approx(st.beta.ppf(0.975, 31, 90))


@pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
def test_coverage_of_the_clustered_interval(p):
    """2,000 synthetic cells, 100 instances x 2 episodes, Beta-binomial with a between-instance share of 0.3."""
    rng = np.random.default_rng(20261125)
    rho = 0.3
    theta = rng.beta(p * (1 - rho) / rho, (1 - p) * (1 - rho) / rho, size=(2000, 100))
    y = (rng.random((2000, 100, 2)) < theta[:, :, None]).astype(float)
    key = np.repeat(np.arange(100), 2)
    hit = 0
    for cell in y:
        r = stats.cluster_rate(cell.ravel(), key)
        hit += r["lo"] <= p <= r["hi"]
    assert 0.93 <= hit / 2000 <= 0.975, hit / 2000


def test_summarize_has_no_interval_of_its_own():
    with open(os.path.join(ROOT, "src", "mrgym", "agents", "summarize.py"), encoding="utf-8") as f:
        assert not re.search(r"^def wilson", f.read(), re.M)


# ------------------------------------------------------------------ intervals, in general
def test_wilson_matches_the_textbook_and_one_episode_per_instance_is_wilson_with_t_on_m_minus_1():
    lo, hi = stats.wilson(7, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.3968, 0.8922)                 # Wilson 1927, z = 1.96
    assert stats.wilson(0, 0) == (0.0, 1.0)
    y = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    r = stats.cluster_rate(y, list(range(12)))
    p = sum(y) / 12                                             # the sample variance divides by M - 1, so n_eff = M - 1
    assert r["n_eff"] == pytest.approx(11) and (r["lo"], r["hi"]) == pytest.approx(stats.wilson(p * 11, 11, crit=st.t.ppf(0.975, 11)))
    assert r["deff"] == pytest.approx(12 / 11)


def test_strata_weigh_levels_equally_and_use_m_minus_h_degrees_of_freedom():
    y = [1] * 30 + [0] * 10 + [1] * 2 + [0] * 8                              # level 1: 40 instances, level 2: 10
    key = [f"a{i}" for i in range(40)] + [f"b{i}" for i in range(10)]
    level = [1] * 40 + [2] * 10
    r = stats.cluster_rate(y, key, level)
    assert r["rate"] == pytest.approx((0.75 + 0.2) / 2)
    v = (0.25 * np.var([1] * 30 + [0] * 10, ddof=1) / 40 + 0.25 * np.var([1] * 2 + [0] * 8, ddof=1) / 10)
    assert r["se"] == pytest.approx(np.sqrt(v))
    m = stats.cluster_mean(y, key, level)
    assert m["mean"] == pytest.approx(r["rate"]) and (m["hi"] - m["lo"]) / 2 == pytest.approx(st.t.ppf(0.975, 48) * np.sqrt(v))
    with pytest.raises(ValueError):
        stats.cluster_rate([1, 0], ["a", "a"], [1, 2])


def test_beta_binomial_interval_is_the_beta_posterior_for_one_episode_per_instance():
    lo, hi = stats.beta_binomial_interval([1] * 6 + [0] * 18, 1)             # K = 1: the concentration drops out
    assert lo == pytest.approx(st.beta.ppf(0.025, 7, 19), abs=0.004) and hi == pytest.approx(st.beta.ppf(0.975, 7, 19), abs=0.004)
    wide = stats.beta_binomial_interval([8] * 3 + [0] * 9, 8)                # all-or-nothing instances: wider than 96 episodes
    flat = stats.beta_binomial_interval([2] * 12, 8)
    naive = stats.wilson(24, 96)
    assert wide[1] - wide[0] > 1.5 * (flat[1] - flat[0]) and wide[1] - wide[0] > 2 * (naive[1] - naive[0])
    assert stats.beta_binomial_interval([0] * 24, 1)[0] < 0.002


def test_mcnemar_holm_and_the_paired_test_of_single_episodes():
    assert stats.mcnemar_exact(0, 0) == 1.0 and stats.mcnemar_exact(5, 5) == 1.0
    assert stats.mcnemar_exact(9, 1) == pytest.approx(2 * st.binom.cdf(1, 10, 0.5))
    assert stats.mcnemar_exact(9, 1, "greater") == pytest.approx(st.binom.sf(8, 10, 0.5))
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([]) == []
    a = {i: float(i < 60) for i in range(100)}
    b = {i: float(i < 45) for i in range(100)}
    c = stats.paired_compare(a, b)
    assert c["test"] == "mcnemar-exact" and (c["wins"], c["losses"], c["ties"]) == (15, 0, 85) and c["p"] == pytest.approx(2 * 0.5 ** 15)
    assert c["diff"] == pytest.approx(0.15)
    one = stats.paired_compare(a, b, alternative="greater")
    assert one["p"] == pytest.approx(0.5 ** 15) and (one["lo"], one["hi"]) == (c["lo"], c["hi"])
    assert stats.paired_compare({1: 1.0}, {2: 0.0})["instances"] == 0


def test_unpaired_difference_of_two_levels(pilot):
    y, key = episodes(pilot, "qwen3.8-27b")
    one = [(v, k) for v, k in zip(y, key) if k.startswith("L1")]
    two = [(v, k) for v, k in zip(y, key) if k.startswith("L2")]
    d = stats.unpaired_cluster_diff([v for v, _ in two], [k for _, k in two], [v for v, _ in one], [k for _, k in one])
    assert round(100 * d["diff"], 1) == -1.3 and round(100 * d["se"], 1) == 4.3 and d["lo"] < 0 < d["hi"]     # the levels are not a ladder


# ------------------------------------------------------------------ power
def test_power_of_the_published_design(capsys):
    assert round(100 * power.halfwidth(300, 2), 1) == 4.2
    assert round(100 * stats.mde_paired(300, 2), 1) == 8.4
    assert round(100 * stats.mde_paired(300, 2, comparisons=10), 1) == 10.9
    assert round(100 * stats.mde_paired(300, 2, rho=0.3, r=0.5), 1) == 8.1
    assert [stats.instances_needed(d, 2) for d in (0.05, 0.08, 0.10, 0.15)] == [840, 329, 210, 94]
    same_budget = [round(100 * stats.mde_paired(m, k), 1) for m, k in ((600, 1), (200, 3), (75, 8))]
    assert same_budget == [8.1, 8.6, 9.9]                                    # more episodes per instance detect less
    assert [round(100 * stats.mde_paired(m, k, rho=0.3), 1) for m, k in ((600, 1), (200, 3), (75, 8))] == [8.1, 10.2, 14.2]
    assert round(power.icc_se(300, 2), 3) == 0.058 and power.icc_se(300, 1) is None
    assert cli.main(["stats", "power", "--instances", "300", "--episodes", "2"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"half-width of one pass rate\s+4\.2 points", out) and re.search(r"detectable paired difference\s+8\.4 points", out)
    assert re.search(r"Holm over 10 comparisons\s+10\.9 points", out) and "meets the published minimum" in out
    assert cli.main(["stats", "power", "--instances", "300", "--episodes", "2", "--rho", "0.3", "--r", "0.5"]) == 0
    assert re.search(r"detectable paired difference\s+8\.1 points", capsys.readouterr().out)
    assert cli.main(["stats", "power", "--instances", "24", "--episodes", "1"]) == 0
    assert "tier: pilot" in capsys.readouterr().out


def test_power_table_rows():
    table = power.power_markdown()
    for row in ("| 10 x 1 | 35.8 | 62.6 | 81.6 | 57.8 | none |", "| 24 x 1 | 21.1 | 40.4 | 52.7 | 37.3 | none |",
                "| 96 x 8 | 4.4 | 8.7 | 11.4 | 9.9 | pilot for rho |", "| 300 x 2 | 4.2 | 8.4 | 10.9 | 8.1 | published minimum |",
                "| 1500 x 2 | 1.9 | 3.7 | 4.9 | 3.6 | suite score |"):
        assert row in table


# ------------------------------------------------------------------ reliability
def test_icc_and_retest_on_known_data():
    y = [1, 1, 0, 0, 1, 1, 0, 0]
    assert stats.icc_oneway(y, [0, 0, 1, 1, 2, 2, 3, 3]) == pytest.approx(1.0)                # episodes of an instance always agree
    assert stats.icc_oneway([1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 2, 2, 3, 3]) == pytest.approx(-1.0)
    assert stats.icc_oneway([1, 0, 1], [0, 1, 2]) is None                                     # one episode per instance
    rt = stats.retest_agreement([1, 1, 0, 0, 1], [1, 0, 0, 0, 1])
    assert rt["agreement"] == pytest.approx(0.8) and rt["pass_both"] == pytest.approx(0.4) and rt["pass_any"] == pytest.approx(0.6)
    assert rt["phi"] == pytest.approx(np.corrcoef([1, 1, 0, 0, 1], [1, 0, 0, 0, 1])[0, 1])


def test_reliability_of_the_pilot(pilot):
    recs = []
    for agent, table in pilot["agents"].items():
        for key, bits in table.items():
            level, seed = key[1:].split(":")
            for e, b in enumerate(bits):
                rec = dict(env_id=pilot["env_id"], agent=agent, agent_kind="model", level=int(level), seed=int(seed), episode=e,
                           status="completed", passed=bool(int(b)))
                if agent == "qwen3.8-27b":
                    rec.update(error=pilot["error"][key][e], tolerance=pilot["tolerance"], gates={"all": bool(int(pilot["gates_ok"][key][e]))})
                recs.append(rec)
    rep = reliability.report(recs)
    big = next(c for c in rep["cells"] if c["agent"] == "qwen3.8-27b")
    assert round(big["icc"], 3) == 0.068 and big["icc_ci95"][0] < 0.068 < big["icc_ci95"][1]
    assert (big["all_fail_instances"], big["all_pass_instances"], big["instances"]) == (6, 1, 96)
    assert [round(100 * x, 1) for x in big["threshold_profile"]["pass_share"]] == [22.7, 37.1, 45.1, 48.7]   # tolerance 5, 10, 15, 20 %
    env = rep["environments"][0]
    assert not env["separation"]["estimable"] and env["cronbach_alpha"] is None
    text = reliability.text(rep)
    assert "ICC 0.068" in text and "separation: not estimable (< 5 agents with a non-zero rate)" in text
    assert "not estimable" in reliability.markdown(rep)


def test_alpha_and_dead_instances_on_a_synthetic_panel():
    """Eight agents of known ability on 310 instances x 2 episodes, ten of them planted dead (five that nobody passes,
    five that everybody passes): alpha is recovered within 0.03 of its known value and the dead ones are flagged."""
    rng = np.random.default_rng(7)
    ability = np.linspace(-1.5, 1.5, 8)
    difficulty = rng.normal(0, 1, 310)
    p = 1 / (1 + np.exp(-(ability[:, None] - difficulty[None, :])))
    p[:, :5], p[:, 5:10] = 0.0, 1.0
    y = (rng.random((8, 310, 2)) < p[:, :, None]).mean(2)
    items = stats.item_table(y, 16)
    dead = np.flatnonzero(items["dead"])
    assert set(range(10)) <= set(dead.tolist()) and len(dead) <= 14            # the planted ten, and any instance that came out all-0 / all-1
    assert all(y[:, j].min() == y[:, j].max() and y[0, j] in (0.0, 1.0) for j in dead)
    true = p.sum(1)                                                        # known: variance of the true scores over that plus the error variance
    known = true.var(ddof=1) / (true.var(ddof=1) + (p * (1 - p) / 2).sum(1).mean())
    assert abs(stats.cronbach_alpha(y) - known) < 0.03, (stats.cronbach_alpha(y), known)
    assert stats.split_half(y)["spearman_brown"] > 0.9
    live = [d for j, d in enumerate(items["discrimination"]) if d is not None and j not in dead]
    assert np.median(live) > 0.2 and np.mean(np.array(live) < 0) < 0.10       # eight agents: a few instances come out negative by chance
    rates, ses = y.mean(1), np.sqrt(y.mean(1) * (1 - y.mean(1)) / 310)
    sep = stats.separation(rates, ses)
    assert sep["estimable"] and sep["reliability"] > 0.9
    assert not stats.separation(rates[:3], ses[:3])["estimable"]


def test_difficulty_drivers_recover_a_planted_effect():
    rng = np.random.default_rng(3)
    level = np.repeat([0, 1], 400)
    inst = np.repeat(np.arange(400), 2)
    y = (rng.random(800) < 1 / (1 + np.exp(-(0.4 - 1.2 * level)))).astype(float)
    fit = stats.difficulty_drivers(level, y, inst, names=["level 2"])
    assert fit["names"] == ["intercept", "level 2"] and fit["clusters"] == 400
    assert fit["lo"][1] < -1.2 < fit["hi"][1] and fit["hi"][1] < 0                      # the level is harder: interval excludes 0
    assert stats.difficulty_drivers(level, np.zeros(800), inst) is None                 # nobody passes: not identified


# ------------------------------------------------------------------ sequential
def test_betting_test_keeps_its_error_rate_and_stops_early_on_a_real_difference():
    rng = np.random.default_rng(11)
    null = alt = 0
    for _ in range(2000):
        a, b = rng.binomial(2, 0.5, 600) / 2, rng.binomial(2, 0.5, 600) / 2
        null += stats.betting_test((a - b + 1) / 2)["reject"]
        a, b = rng.binomial(2, 0.55, 600) / 2, rng.binomial(2, 0.45, 600) / 2          # +10 points
        r = stats.betting_test((a - b + 1) / 2)
        alt += r["reject"] and r["stopped_at"] < 600
    assert null / 2000 <= 0.05 and alt / 2000 >= 0.80, (null / 2000, alt / 2000)
    with pytest.raises(ValueError):
        stats.betting_test([1.2])


def test_alpha_spending_over_looks():
    assert stats.alpha_spend(0.05, 5) == [0.01] * 5
    d = np.r_[np.full(150, 0.5), np.full(50, -0.5)]
    look = sequential.checkpoint_look(d, alpha=0.05, looks=5, mde=0.05)
    assert look["efficacy"] and not look["futility"] and look["level"] == pytest.approx(0.99)
    flat = sequential.checkpoint_look(np.r_[np.full(100, 0.02), np.full(100, -0.02)], looks=5, mde=0.05)
    assert flat["futility"] and not flat["efficacy"]
    small = sequential.checkpoint_look(np.r_[np.ones(14), -np.ones(10)], looks=5, mde=0.05)   # 24 instances support neither rule
    assert not small["efficacy"] and not small["futility"]


def test_paired_compare_with_equal_differences_uses_the_sign_test_not_a_zero_p():
    from mrgym.stats import compare
    two = compare.paired_compare({"a": 1.0, "b": 0.5}, {"a": 0.5, "b": 0.0})      # two instances, both +0.5: no variance
    assert two["diff"] == 0.5 and two["test"] == "sign-exact" and two["p"] == 0.5
    none = compare.paired_compare({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5})
    assert none["diff"] == 0.0 and none["p"] == 1.0
    ten = compare.paired_compare({i: 0.75 for i in range(10)}, {i: 0.25 for i in range(10)}, alternative="greater")
    assert ten["test"] == "sign-exact" and abs(ten["p"] - 0.5 ** 10) < 1e-12
