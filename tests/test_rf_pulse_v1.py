"""``Scanner/RFPulse-v1``: a design task whose pass rule compares the duration with a proven bound.

    physics     the rotation of the verifier's model (hard sub-pulses) against real 3 x 3 rotation matrices; the three
                corrections (amplitude, frequency, rephasing lobe) undo the three hidden quantities exactly
    scanner     the calibration scans give the transmit scale, the off-resonance and the gradient delay of the hidden
                scanner
    family      instances are functions of (seed, level); the statement holds no hidden value; the reference passes,
                every shortcut fails, each failure has its outcome; a design below the bound cannot pass; sizes
"""
import concurrent.futures
import json
import math

import numpy as np
import pytest

import mrgym
from mrgym import policies
from mrgym.design import bound, hardware, tool
from mrgym.design.units import RAD_PER_HZ_MS, RAD_PER_MT_M_MM_MS, RAD_PER_UT_MS
from mrgym.envs.rf_pulse_v1 import controls, model, task

ENV = "Scanner/RFPulse-v1"
SPEC = dict(thickness_mm=5.0, flip_deg=30.0, tol_in_deg=1.2, max_out_deg=0.9, transition=0.3, extent_mm="repeat", refocused_min=0.95)
LIMITS = tool.Limits(b1_peak_ut=40.0, g_max_mt_m=32.0, slew_t_m_s=120.0, b1rms_ut=6.0, tr_ms=10.0)


def make(level=2, seed=0, **kw):
    env = mrgym.make(ENV, level=level, seed=seed, **kw)
    env.reset()
    return env


def call(env, name, **arguments):
    return env.step({"name": name, "arguments": arguments}).observation


def run(name, level, seed):
    return policies.run(mrgym.make(ENV, level=level, seed=seed), name)


# ------------------------------------------------------------------------------------------------ physics
def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hard_train(b1, dt, g, z, df):
    """Magnetisation from equilibrium after a train of hard sub-pulses, with real rotation matrices: half the precession
    of a sample, the sub-pulse about (cos p, sin p, 0), the other half. Shares nothing with the spinor code but the units."""
    m = np.array([0.0, 0.0, 1.0])
    for b, d, gr in zip(b1, dt, g):
        half = _rz(0.5 * (RAD_PER_MT_M_MM_MS * gr * z + RAD_PER_HZ_MS * df) * d)
        angle, p = RAD_PER_UT_MS * abs(b) * d, np.angle(b)
        k = np.array([math.cos(p), math.sin(p), 0.0])
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rot = np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx
        m = half @ rot @ half @ m
    return m


@pytest.mark.parametrize("chain", [hardware.Chain(model="split", guard=True), hardware.Chain(0.7, None, 90.0, 310.0, "split", True),
                                   hardware.Chain(1.5, None, -120.0, -240.0, "split", True)])
def test_the_rotation_of_the_verifier_is_a_train_of_hard_pulses(chain):
    pulse, _ = tool.build(model.call_of(SPEC, 64, 30.0, (1.0, 1.0), 1.2, 180.0), LIMITS)
    b1, dt, g = hardware.played(pulse, chain)
    assert np.array_equal(g, pulse.g_mt_m)                               # the plateau outlasts the pulse: no delay reaches the samples
    for z in (0.0, 1.1, -2.9, 9.0):
        a, b = hardware.rotate(pulse, chain, np.array([z]), rephase=False)
        m = _hard_train(b1, dt, g, z, chain.df_hz)
        assert abs(2 * np.conj(a[0]) * b[0] - (m[0] + 1j * m[1])) < 1e-10 and abs(abs(a[0]) ** 2 - abs(b[0]) ** 2 - m[2]) < 1e-10
        a, b = hardware.rotate(pulse, chain, np.array([z]))              # then the rephasing lobe and what a late plateau leaves
        after = _rz(RAD_PER_MT_M_MM_MS * (pulse.rephase_mt_ms_m + g[-1] * chain.grad_delay_us * 1e-3) * z) @ m
        assert abs(2 * np.conj(a[0]) * b[0] - (after[0] + 1j * after[1])) < 1e-10


def test_three_corrections_undo_three_hidden_quantities():
    pulse, call_ = model.design_at(SPEC, LIMITS, 32.0)
    ideal = task.profile(pulse, hardware.Chain(**task.CHAIN), SPEC)
    assert all(ideal["checks"].values())
    chain = hardware.Chain(0.8, None, 70.0, 300.0, **task.CHAIN)
    assert not any(task.profile(pulse, chain, SPEC)["checks"].values())  # the uncorrected design misses all three requirements
    for wrong in (dict(scale=1.0), dict(df_hz=0.0), dict(delay_us=0.0), dict(df_hz=-300.0), dict(delay_us=-70.0)):
        arguments = dict(dict(scale=0.8, df_hz=300.0, delay_us=70.0), **wrong)
        assert not all(task.profile(model.design_at(SPEC, LIMITS, 32.0, **arguments)[0], chain, SPEC)["checks"].values()), wrong
    fixed = task.profile(model.design_at(SPEC, LIMITS, 32.0, 0.8, 300.0, 70.0)[0], chain, SPEC)
    for key in ("in_min_deg", "in_max_deg", "out_max_deg", "refocused"):
        assert abs(fixed[key] - ideal[key]) < 2e-3, key                  # amplitude and rephasing fraction are rounded to 4 decimals


def test_an_early_gradient_may_need_a_rephasing_lobe_of_the_gradients_own_sign():
    """Found on the split tune: 9 of 200 level-2 worlds had a delay so far negative that best + delay / duration is
    below zero. With the fraction cut at zero the reference failed the phase requirement on exactly those."""
    spec = dict(SPEC, thickness_mm=6.0, transition=0.4)
    chain = hardware.Chain(1.0, None, -149.0, 0.0, **task.CHAIN)
    pulse, call_ = model.design_at(spec, LIMITS, 32.0, 1.0, 0.0, -149.0)
    best = pulse.call["design"]["rephase_fraction"]
    assert call_["rephase_fraction"] == pytest.approx(best - 0.149 / pulse.duration_ms, abs=1e-4) and call_["rephase_fraction"] < 0
    assert task.profile(pulse, chain, spec)["checks"]["phase"]
    cut, _ = tool.build(dict(call_, rephase_fraction=0.0), LIMITS)
    assert not task.profile(cut, chain, spec)["checks"]["phase"]
    assert tool.range_of("minphase", "rephase_fraction") == (-1.0, 1.0) and tool.range_of("points", "rephase_fraction") == (0.0, 1.0)
    with pytest.raises(tool.Refusal):
        tool.build(dict(call_, rephase_fraction=-1.2), LIMITS)


def test_the_design_rule_lowers_the_gradient_when_the_rf_power_binds():
    free, _ = model.design_rule(SPEC, LIMITS)
    assert free.g_mt_m[0] == pytest.approx(32.0, rel=1e-6) and free.n == model.shortest(SPEC, 32.0)
    tight = tool.Limits(b1_peak_ut=40.0, g_max_mt_m=32.0, slew_t_m_s=120.0, b1rms_ut=round(math.sqrt(free.energy_ut2_ms / 10.0), 3), tr_ms=10.0)
    for scale, lower in ((1.2, False), (0.9, True), (0.6, True)):
        pulse, call_ = model.design_rule(SPEC, tight, scale)
        assert tool.facts(pulse, tight)["limits"]["ok"] and (pulse.g_mt_m[0] < 31.0) == lower
        if lower:                                                        # energy is proportional to the gradient: the rule ends near the limit
            assert 0.93 < pulse.energy_ut2_ms / model.energy_limit(tight) <= 1.0 and pulse.n > free.n
    assert model.tbw_of(32.0, 5.0, 0.6) <= 42.577478518e-3 * 32.0 * 5.0 * 0.6


# ------------------------------------------------------------------------------------------------ scanner
@pytest.mark.parametrize("seed", [0, 3, 8])
def test_the_calibration_scans_measure_the_hidden_chain(seed):
    env = make(2, seed)
    chain = task.chain_of(env._session.world)
    assert chain.model == "split" and chain.guard and chain.amp_sat_ut is None
    assert task.TX_RANGE[0] <= chain.tx_scale <= task.TX_RANGE[1] and abs(chain.df_hz) <= task.OFFSET_HZ + 5
    assert task.DELAY_US[0] <= abs(chain.grad_delay_us) <= task.DELAY_US[1]
    got = controls.measure(policies.ToolScanner(env))
    assert abs(got["scale"] / chain.tx_scale - 1) < 0.012 and abs(got["df_hz"] - chain.df_hz) < 2.0
    assert abs(got["delay_us"] - chain.grad_delay_us) < 0.12 * abs(chain.grad_delay_us)
    assert env._session.budget()["used_s"] == pytest.approx(3 * task.SCAN_S, abs=1.0)


def test_level_one_hides_the_transmit_scale_only():
    env = make(1, 4)
    chain = task.chain_of(env._session.world)
    assert chain.grad_delay_us == 0.0 and abs(chain.df_hz) < 6.0 and chain.model == "split"
    assert abs(controls.measure_scale(policies.ToolScanner(env)) / chain.tx_scale - 1) < 0.012


# ------------------------------------------------------------------------------------------------ family
def test_instances_are_functions_of_seed_and_level_also_under_threads():
    seeds = [format(i, "032x") for i in range(1, 9)]
    plain = [json.dumps(task.make_instance(s, 3, "t"), sort_keys=True) for s in seeds]
    task._instance.cache_clear()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(lambda s: json.dumps(task.make_instance(s, 3, "t"), sort_keys=True), seeds * 2))
    assert threaded == plain * 2 and len(set(plain)) == len(seeds)
    for lv in task.LEVELS:
        cfg = task.make_instance(seeds[0], lv, "t")
        spec, lim = cfg["task"]["design"]["specification"], cfg["task"]["design"]["limits"]
        assert spec["transition"] in task.LEVELS[lv]["transitions"] and 20 <= spec["flip_deg"] <= 40 and lim["g_max_mt_m"] == 32.0
        assert (lim["b1rms_ut"] < task.B1RMS_LOOSE_UT) == (lv == 3) and ("method" in cfg["task"]) == (lv == 1)
        assert cfg["task"]["duration_factor"] == task.FACTORS[lv] and cfg["budget_s"] == task.LEVELS[lv]["scans"] * task.SCAN_S + 20


@pytest.mark.parametrize("level", [1, 2, 3])
def test_the_statement_holds_no_hidden_value_and_states_the_class(level):
    env = mrgym.make(ENV, level=level, seed=11)
    obs, _ = env.reset()
    chain = task.chain_of(env._session.world)
    text = obs.text + json.dumps(env.tools("openai")) + call(env, "scanner_info", topic="design").text
    for hidden in (chain.tx_scale, chain.df_hz, chain.grad_delay_us):
        if hidden:
            assert f"{abs(hidden):.5g}" not in text and f"{abs(hidden):.4f}" not in text
    assert "hard sub-pulses under one constant slice gradient" in obs.text and "between the two: free" in obs.text
    assert f"at most {task.FACTORS[level]:g} times the bound" in obs.text and ("How to do it" in obs.text) == (level == 1)
    assert ("off-resonance" in obs.text) == (level > 1) and "verse" not in text and "slr" not in json.dumps(env.tools("openai"))


def test_sizes_of_the_task_and_of_the_answers():
    worst = 0
    for level in (1, 2, 3):
        env = mrgym.make(ENV, level=level, seed=9000)
        obs, _ = env.reset()
        worst = max(worst, len(json.dumps(env.tools("openai"))) + len(obs.text))
    assert worst <= 18500                                                # the largest ceiling of any other family (tests/test_observation_budget.py)
    env = make(2, 1)
    spec = env.episode_config()["task"]["design"]["specification"]
    stored = call(env, "design", **model.call_of(spec, 70, 31.0, (0.8, 0.8), 1.31, -212.5, 0.25))
    assert len(stored.text) < 720 and "largest deviation" in stored.text and "rephasing 0.25" in stored.text
    ideal = call(env, "simulate", design=1)
    worst_case = call(env, "simulate", design=1, b1_scale=0.77, grad_delay_us=-113.5, off_resonance_hz=187.5)
    assert len(ideal.text) < len(worst_case.text) <= 400                 # the longest answer measured: 372 characters, 216 tokens of the trained model's tokenizer
    assert "to -z" in worst_case.text and "NOT met" in worst_case.text and len(worst_case.data["profile"]["flip_deg"]) == 25


@pytest.mark.parametrize("level,seed", [(1, 0), (2, 0), (2, 5), (3, 1), (3, 2)])
def test_the_reference_passes_and_stays_above_the_bound(level, seed):
    r = run("reference", level, seed)
    s = r.info["score"]
    assert r.reward == 1.0 and s["outcome"] == "pass" and r.info.get("policy_error") is None
    assert 1.0 <= s["duration_over_bound"] <= s["duration_factor"] and s["bound"]["kind"] == "theorem" and s["quarantine"] is None
    assert s["record"]["specification_class"].startswith(f"level {level} | thickness")


@pytest.mark.parametrize("name,level,outcomes", [
    ("prior", 2, ("no_measurement",)), ("ideal_tuned", 2, ("mask_in_slice",)), ("default_pulse", 2, ("mask_in_slice",)),
    ("fixed_recipe", 2, ("mask_in_slice",)), ("random_search", 2, ("mask_in_slice", "mask_out_of_slice", "too_long")),
    ("wrong_units", 2, ("mask_in_slice",)), ("ignores_transmit", 1, ("mask_in_slice",)), ("ignores_offset", 2, ("mask_in_slice", "mask_out_of_slice")),
    ("ignores_delay", 2, ("mask_phase",)), ("v0_reference", 3, ("mask_in_slice", "mask_out_of_slice", "mask_phase"))])
def test_every_shortcut_fails_with_a_named_outcome(name, level, outcomes):
    r = run(name, level, 0)
    assert r.reward == 0.0 and r.info["score"]["outcome"] in outcomes, r.info["score"]["outcome"]
    assert set(outcomes) <= set(task.OUTCOMES) | {"no_measurement"}


def test_the_v0_reference_is_the_reference_where_nothing_new_is_hidden():
    assert run("v0_reference", 1, 0).reward == 1.0 and controls.ABLATIONS["v0_reference"] == (2, 3)
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert set(controls.DESCRIPTIONS) == set(controls.policies())


def test_a_refused_deliverable_leaves_the_episode_open():
    env = make(3, 1)
    assert "no stored design 4" in call(env, "submit", design=4).text
    spec = env.episode_config()["task"]["design"]["specification"]
    heavy = call(env, "design", **model.call_of(spec, model.shortest(spec, 32.0), 32.0, amplitude=3.0))
    assert "VIOLATED" in heavy.text and "B1rms" in heavy.text
    refused = call(env, "submit", design=1)
    assert refused.is_error and "violates a limit" in refused.text
    assert not call(env, "design", **model.call_of(spec, model.shortest(spec, 32.0), 32.0)).is_error      # the episode goes on
    assert '"slr" is not one of ["sinc", "minphase"]' in call(env, "design", kind="slr", flip_deg=30, duration_ms=1, tbw=4, thickness_mm=5,
                                                              tol_in_deg=1, max_out_deg=1).text


def test_a_design_shorter_than_the_bound_cannot_pass_and_would_be_quarantined(monkeypatch):
    env = make(2, 0)
    cfg, world = env.episode_config(), env._session.world
    chain = task.chain_of(world)
    need = task.bound_of(cfg["task"], chain.tx_scale)
    spec, limits = model.spec_of(cfg["task"]), model.limits_of(cfg["task"])
    for samples in (need["samples"] - 1, need["samples"] - 5):           # the best pulse of fewer samples, with the right corrections
        base, _ = tool.build(model.call_of(spec, samples, 32.0, (1.0, 1.0), 1 / chain.tx_scale, chain.df_hz), limits)
        fraction = base.call["design"]["rephase_fraction"] + chain.grad_delay_us * 1e-3 / base.duration_ms
        short = model.call_of(spec, samples, 32.0, (1.0, 1.0), 1 / chain.tx_scale, chain.df_hz, fraction)
        assert not controls.passes(cfg, world, short)
        s = task.score(cfg, dict(design=1), world)
        assert task.outcome(s) in ("mask_in_slice", "mask_out_of_slice") and s["checks"]["not_below_bound"] and s["quarantine"] is None
    at = model.call_of(spec, need["samples"], 32.0, (1.0, 1.0), 1 / chain.tx_scale, chain.df_hz)
    base, _ = tool.build(at, limits)
    at = dict(at, rephase_fraction=round(base.call["design"]["rephase_fraction"] + chain.grad_delay_us * 1e-3 / base.duration_ms, 4))
    assert controls.passes(cfg, world, at) and task.score(cfg, dict(design=1), world)["duration_over_bound"] == 1.0
    # a bound that claimed more than is true would be caught as a defect, not rewarded
    wrong = dict(need, samples=need["samples"] + 3, duration_ms=round((need["samples"] + 3) * 0.01, 6), cells=[])
    monkeypatch.setattr(task, "bound_of", lambda t, s: wrong)
    s = task.score(cfg, dict(design=1), world)
    assert not s["passed"] and task.outcome(s) == "below_bound" and s["quarantine"] == "beats_bound" and "defect" in s["diagnostic"]


def test_too_long_is_an_outcome_and_the_factor_is_the_rule(monkeypatch):
    env = make(2, 0)
    cfg, world = env.episode_config(), env._session.world
    chain = task.chain_of(world)
    spec, limits = model.spec_of(cfg["task"]), model.limits_of(cfg["task"])
    need = task.bound_of(cfg["task"], chain.tx_scale)
    samples = int(math.floor(task.FACTORS[2] * need["samples"])) + 2
    long_, _ = tool.build(model.call_of(spec, samples, 32.0, model.SHARE, 1 / chain.tx_scale, chain.df_hz), limits)
    fraction = long_.call["design"]["rephase_fraction"] + chain.grad_delay_us * 1e-3 / long_.duration_ms
    call_ = model.call_of(spec, samples, 32.0, model.SHARE, 1 / chain.tx_scale, chain.df_hz, fraction)
    assert not controls.passes(cfg, world, call_)
    s = task.score(cfg, dict(design=1), world)
    assert task.outcome(s) == "too_long" and [k for k, v in s["checks"].items() if not v] == ["short_enough"]


def test_the_factors_are_those_of_the_stored_derivation():
    from mrgym.envs.rf_pulse_v1 import factors
    with open(factors.path(), encoding="utf-8") as f:
        stored = json.load(f)
    assert stored["split"] == "tune" and stored["env_id"] == ENV
    for lv, row in stored["levels"].items():
        assert task.FACTORS[int(lv)] == row["factor"] == factors.factor_of([row["ratio_max"]]) and row["hard_met"] >= 0.95 * row["n"]
        assert row["ratio_min"] >= 1.0 and (row["constant"]["rate"] < 0.02 or int(lv) == 1)
    assert factors.factor_of([1.0]) == 1.05 and factors.factor_of([1.4301]) == 1.55


def test_the_card_and_the_registration():
    card = mrgym.spec(ENV).card() if callable(getattr(mrgym.spec(ENV), "card", None)) else task.card()
    assert card["id"] == ENV and card["practice_levels"] == [1] and "theorem" in card["summary"]
    assert mrgym.spec("Scanner/RFPulse-v0").family == "mrgym.envs.rf_pulse" and mrgym.spec(ENV).family == "mrgym.envs.rf_pulse_v1"
    assert task.in_class(tool.build(model.call_of(SPEC, 60, 30.0), LIMITS)[0])
    assert bound.Mask.of(SPEC).z_in == 1.75


def test_golden_design_section():
    """The additive section ``design`` of tests/golden/engine_digests.json (``golden_cases.design_cases``): the scan
    ``rfscan`` on the chain of ``-v0`` and on the chain of ``-v1``, bit for bit on the recorded stack, fingerprints
    within the replay tolerance elsewhere."""
    import golden_cases as G
    from mrgym.scanner import session
    golden = G.load(G.ENGINE_FILE)
    assert golden and "design" in golden, "run `python tools/audit/golden.py --write`"
    mode = G.mode(golden["stack"])[0]
    got, want = G.design_cases(), golden["design"]
    assert sorted(got) == sorted(want) and len(got) == 10
    for key in got:
        if key.endswith("chain_head"):
            assert mode != "bit-exact" or got[key] == want[key], key
            continue
        assert got[key]["shape"] == want[key]["shape"], key
        if mode == "bit-exact":
            assert got[key]["sha256"] == want[key]["sha256"], key
        else:
            assert session.fingerprints_close(got[key]["fingerprint"], want[key]["fingerprint"]), key
