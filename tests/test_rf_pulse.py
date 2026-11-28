"""``Scanner/RFPulse-v0``, the design tools and the scan that plays a design.

    physics     the rotation the verifier and the scanner use against an independent integration of the Bloch
                equations; the amplifier law and the delayed gradient against closed forms; the flip-angle generator;
                the floor of the pass rule
    scanner     the stripe of ``rfscan`` is sin(flip angle); the log holds the designs and replays on a fresh world
    tools       refusals name the parameter and the nearest valid value; answers stay small; other families are untouched
    family      instances are functions of (seed, level); the statement holds no hidden value; the reference passes,
                every shortcut fails, a refused deliverable leaves the episode open, each failure has its outcome
"""
import concurrent.futures
import json
import math

import numpy as np
import pytest
import scipy.integrate

import mrgym
from mrgym import policies
from mrgym.design import hardware, pulses, tool
from mrgym.design.units import RAD_PER_HZ_MS, RAD_PER_MT_M_MM_MS, RAD_PER_UT_MS
from mrgym.envs.rf_pulse import controls, model, task
from mrgym.scanner import rfscan, session

ENV = "Scanner/RFPulse-v0"
SPEC = dict(thickness_mm=5.0, flip_deg=30.0, tol_in_deg=1.2, max_out_deg=0.9, transition=0.4, extent_mm=40.0, refocused_min=0.95)
LIMITS = tool.Limits(b1_peak_ut=20.0, g_max_mt_m=32.0, slew_t_m_s=120.0, b1rms_ut=3.0, tr_ms=30.0)


def make(level=1, seed=0, **kw):
    env = mrgym.make(ENV, level=level, seed=seed, **kw)
    env.reset()
    return env


def call(env, name, **arguments):
    return env.step({"name": name, "arguments": arguments}).observation


# ------------------------------------------------------------------------------------------------ physics
def _bloch_ode(b1, dt, g, z_mm, df_hz):
    """Magnetisation after the pulse from equilibrium by integrating dM/dt = w x M with a general-purpose solver,
    sample by sample (the field is constant during a sample). Shares nothing with ``rotation.rotate`` but the three
    unit factors. w is the right-handed rotation vector of the stated convention: its z component is the precession
    (a positive frequency gains positive phase, dMxy/dt = +i w_z Mxy), and B1 along +x turns Mz towards -y."""
    m = np.array([0.0, 0.0, 1.0])
    for b, d, gr in zip(b1, dt, g):
        w = np.array([RAD_PER_UT_MS * b.real, RAD_PER_UT_MS * b.imag, RAD_PER_MT_M_MM_MS * gr * z_mm + RAD_PER_HZ_MS * df_hz])
        m = scipy.integrate.solve_ivp(lambda t, y: np.cross(w, y), (0.0, d), m, rtol=1e-11, atol=1e-13).y[:, -1]
    return m[0] + 1j * m[1], m[2]


@pytest.mark.parametrize("chain", [hardware.IDEAL, hardware.Chain(0.8, 14.0, 0.0, 12.0), hardware.Chain(1.3, 11.0, 15.0, -30.0)])
def test_the_rotation_of_the_verifier_is_the_bloch_equation(chain):
    slr = pulses.slr_flip(60.0, 1.2, 5.0, 5.0, 2.0, 1.5)
    variable = pulses.verse(pulses.slr_flip(60.0, 0.6, 4.0, 5.0, 2.0, 1.5), 14.0, 32.0, 120.0)
    for pulse in (slr, variable):
        b1, dt, g = hardware.played(pulse, chain)
        for z in (0.0, 1.3, -2.6, 7.0):
            a, b = hardware.rotate(pulse, chain, np.array([z]), rephase=False)
            mxy, mz = _bloch_ode(b1, dt, g, z, chain.df_hz)
            assert abs(2 * np.conj(a[0]) * b[0] - mxy) < 2e-7 and abs(abs(a[0]) ** 2 - abs(b[0]) ** 2 - mz) < 2e-7
            assert abs(float(hardware.flip_deg(pulse, chain, z)) - math.degrees(math.acos(np.clip(mz, -1, 1)))) < 2e-5


def test_the_amplifier_law_and_the_delayed_gradient():
    assert hardware.amplifier(7.0, None) == 7.0
    for a, ratio in ((0.5, 0.99742), (0.8, 0.96194), (1.0, 2 ** (-1 / 6))):
        assert abs(float(hardware.amplifier(a * 15.0, 15.0)) / (a * 15.0) - ratio) < 1e-5
    x = np.linspace(0, 60, 200)
    assert np.all(np.diff(hardware.amplifier(x, 15.0)) > 0) and np.all(hardware.amplifier(x, 15.0) <= np.minimum(x, 15.0) + 1e-12)
    g, dt = np.array([1.0, 1.0, 5.0, 5.0, 2.0]), np.full(5, 0.01)
    assert np.array_equal(hardware.delayed_gradient(np.full(5, 3.0), dt, 40.0), np.full(5, 3.0))          # a constant gradient feels no delay
    assert np.allclose(hardware.delayed_gradient(g, dt, 10.0), [1.0, 1.0, 1.0, 5.0, 5.0])                # one whole sample late
    assert np.allclose(hardware.delayed_gradient(g, dt, 5.0), [1.0, 1.0, 3.0, 5.0, 3.5])                 # half a sample: the mean over each sample
    assert np.allclose(hardware.delayed_gradient(g, dt, -10.0), [1.0, 5.0, 5.0, 2.0, 2.0])               # early: the last value is held


@pytest.mark.parametrize("flip", [25.0, 60.0, 90.0])
def test_the_flip_angle_generator_keeps_its_two_tolerances(flip):
    p = pulses.slr_flip(flip, 1.5, 6.0, 5.0, tol_in_deg=0.03 * flip, max_out_deg=0.02 * flip)
    w = p.call["design"]["transition_fraction"]
    f = tool.figures(p, hardware.IDEAL, dict(thickness_mm=5.0, transition=w, extent_mm=25.0))
    assert flip * 0.97 - 0.004 * flip <= f["in_min_deg"] and f["in_max_deg"] <= flip * 1.03 + 0.004 * flip      # piecewise against hard pulses: 0.4 %
    assert f["out_max_deg"] <= 0.02 * flip * 1.02 and f["refocused"] > 0.99
    strong = tool.build(dict(kind="slr", flip_deg=flip / 2, duration_ms=1.5, tbw=6, thickness_mm=5, tol_in_deg=1, max_out_deg=1, amplitude=2.0))[0]
    weak = tool.build(dict(kind="slr", flip_deg=flip / 2, duration_ms=1.5, tbw=6, thickness_mm=5, tol_in_deg=1, max_out_deg=1))[0]
    assert abs(strong.flip_deg() - 2 * weak.flip_deg()) < 1e-9          # on resonance at the centre the angle is the amplitude


def test_no_pulse_turns_a_spin_further_than_the_floor_allows():
    rng = np.random.default_rng(5)
    z = np.linspace(-20, 20, 801)
    for k in range(30):
        n = int(rng.integers(20, 200))
        pts = rng.uniform(-20, 20, int(rng.integers(2, 20))) * np.exp(1j * rng.uniform(0, 2 * np.pi, 1))
        p = pulses.control_points(pts, n * 0.01, 4000.0, 5.0, interp="cubic")
        chain = hardware.Chain(float(rng.uniform(0.6, 1.6)), float(rng.uniform(8, 30)) if k % 2 else None, 0.0, float(rng.uniform(-50, 50)))
        most = float(np.radians(hardware.flip_deg(p, chain, z)).max())
        assert most <= RAD_PER_UT_MS * chain.tx_scale * p.peak_b1_ut * p.duration_ms * (1 + 1e-9)
    assert abs(model.floor_ms(SPEC, LIMITS, 0.8) - math.radians(28.8) / (RAD_PER_UT_MS * 0.8 * 20.0)) < 1e-12


def test_the_flip_angle_of_a_train_of_hard_pulses_changes_no_faster_than_its_length_allows():
    """docs/physics/design.md, "What was tried for a tighter bound": for n hard sub-pulses with the precession psi
    between them, |d theta / d psi| <= n - 1 (Bernstein-Szego on -cos theta). Checked on random trains; the bound
    it gives on the duration is looser than the hard-pulse floor, so no code uses it."""
    from mrgym.design import rotation
    from mrgym.design.units import HZ_PER_MT_M_MM
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(40):
        n = int(rng.integers(3, 40))
        b1 = rng.uniform(0, 400, n) * np.exp(1j * rng.uniform(0, 2 * np.pi, n))         # sub-pulses of up to 61 degrees, any phase
        g, dt = 10.0, 0.01
        psi = np.linspace(0, 2 * np.pi, 20001)                                          # one turn of the precession per sample
        z = psi / (2 * np.pi * HZ_PER_MT_M_MM * g * dt * 1e-3)
        theta = rotation.flip_angle(rotation.rotate(b1, dt, g, z, model="split")[1])
        slope = np.abs(np.diff(theta) / np.diff(psi)).max()
        worst = max(worst, slope / (n - 1))
        assert slope <= (n - 1) * (1 + 1e-6), (n, slope)
    assert 0.2 < worst <= 1.0
    # it is attained: two 90 degree sub-pulses give cos(theta) = -cos(psi), so theta = pi - psi and the slope is n - 1 = 1
    quarter = (np.pi / 2) / (RAD_PER_UT_MS * 0.01)
    psi = np.linspace(0.2, 2.9, 2001)
    theta = rotation.flip_angle(rotation.rotate(np.array([quarter, quarter]), 0.01, 10.0, psi / (2 * np.pi * HZ_PER_MT_M_MM * 10.0 * 0.01 * 1e-3), model="split")[1])
    assert np.allclose(theta, np.pi - psi, atol=1e-9)


def test_the_reference_search_finds_a_valid_design_and_the_amplifier_makes_it_longer():
    ideal = model.search(SPEC, LIMITS, hardware.IDEAL)
    hard = model.search(SPEC, LIMITS, hardware.Chain(0.7, 9.0))
    for got, chain in ((ideal, hardware.IDEAL), (hard, hardware.Chain(0.7, 9.0))):
        assert tool.facts(got["pulse"], LIMITS)["limits"]["ok"] and model.meets(got["pulse"], SPEC, [chain])
        assert got["duration_ms"] > 3 * model.floor_ms(SPEC, LIMITS, chain.tx_scale)
    assert ideal["stretch"] == 1.0 and hard["stretch"] > 1.2 and hard["duration_ms"] > 1.3 * ideal["duration_ms"]
    assert not model.meets(ideal["pulse"], SPEC, [hardware.Chain(0.7, 9.0)])                               # tuned to the ideal model


# ------------------------------------------------------------------------------------------------ the scanner
def test_the_stripe_of_rfscan_is_the_sine_of_the_flip_angle_and_the_log_replays():
    env = make(level=2, seed=1)
    cfg, sess = env.episode_config(), env._session
    sc = session.LocalScanner(sess)
    chain = task.chain_of(sess.world)
    numbers = [sc.design(dict(task.CAL, amplitude=a))["design"] for a in (1.0, 1.7)]
    assert sc.design(dict(task.CAL))["design"] == numbers[0] and sess.n_designs == 2                       # the same call is the same design
    signal = []
    for k in numbers:
        rp = sess.plan({"protocol": dict(task.CAL_SCAN, rf_design=k)})["realised"]
        img = sess.engine.acquire(rp, None)                                                              # noise-free
        rows = session.roi_readouts(img, [task.CAL_CIRCLE], session.fov_yx(rp), session.centre_yx(rp), session.rotation_of(rp))
        signal.append(rows[0]["complex_mean_magnitude"][0])
    flips = [float(hardware.flip_deg(rfscan.pulse_of(dict(rf_call=sess.exam["designs"][k]["call_json"])), chain, 0.0)) for k in numbers]
    assert abs(signal[1] / signal[0] / (math.sin(math.radians(flips[1])) / math.sin(math.radians(flips[0]))) - 1) < 4e-3
    for k in numbers:
        sc.scan(dict(task.CAL_SCAN, rf_design=k), [task.CAL_CIRCLE])
    sess.seal("submit")
    assert [e["type"] for e in sess.events] == ["design", "design", "plan", "plan", "scan", "scan", "seal"]
    rep = session.replay(cfg, sess.events, head=sess.receipt(with_key=True))                              # a fresh world, rebuilt from the seed
    assert rep["ok"] and rep["mode"] == "bit-exact" and rep["scans"] == 2
    recorded = session.RecordedSession(cfg, sess.events, sess.receipt(with_key=True))
    assert sorted(recorded.exam["designs"]) == numbers and recorded.exam["designs"][2]["call"]["amplitude"] == 1.7
    tampered = [dict(e) for e in sess.events]
    tampered[1] = dict(tampered[1], call=dict(tampered[1]["call"], amplitude=1.8))
    assert not session.replay(cfg, tampered, head=sess.receipt(with_key=True))["ok"]


def test_a_design_that_breaks_a_hardware_limit_is_stored_and_not_played():
    env = make(level=3, seed=0)
    hot = call(env, "design", kind="slr", flip_deg=90, duration_ms=0.4, tbw=4, thickness_mm=5, tol_in_deg=3, max_out_deg=3)
    assert not hot.is_error and "VIOLATED, first peak B1" in hot.text and "not scanned or delivered" in hot.text
    refused = call(env, "scan", sequence="rfscan", rf_design=hot.data["design"], tr_ms=500)
    assert refused.is_error and "violates a limit" in refused.text and env.state()["scan_time_used_s"] == 0
    power = call(env, "design", **dict(task.CAL, duration_ms=1.2, amplitude=1.3))                         # only the task's RF power limit
    assert [v["code"] for v in power.data["facts"]["limits"]["violations"]] == ["RF_POWER_LIMIT"] and "scanned, not delivered" in power.text
    assert not call(env, "scan", **dict(task.CAL_SCAN, rf_design=power.data["design"])).is_error
    assert call(env, "submit", design=power.data["design"]).is_error and not env.state()["done"]
    assert call(env, "scan", sequence="rfscan", rf_design=9, tr_ms=500).is_error


# ------------------------------------------------------------------------------------------------ the tools
def test_design_refuses_with_the_parameter_and_the_nearest_valid_value():
    env = make()
    for args, name, near in ((dict(kind="slr", flip_deg=30, duration_ms=1.005, tbw=4, thickness_mm=5, tol_in_deg=1, max_out_deg=1), "duration_ms", 1.0),
                             (dict(kind="sinc", flip_deg=30, duration_ms=1.0, tbw=40, thickness_mm=5), "tbw", 24.0),
                             (dict(kind="slr", flip_deg=30, duration_ms=1.0, tbw=4, thickness_mm=5, tol_in_deg=40, max_out_deg=1), "tol_in_deg", 30.0)):
        obs = call(env, "design", **args)
        err = obs.data["error"]
        assert obs.is_error and err["problems"][0]["parameter"] == name and err["problems"][0].get("nearest_valid") == near, obs.text
    assert call(env, "design", kind="verse", of=1, b1_peak_ut=10).is_error                                # not a generator of this family
    assert call(env, "design", kind="slr", flip_deg=30, duration_ms=1.0, tbw=4, thickness_mm=5, tol_in_deg=1, max_out_deg=1, samples=[1, 2]).is_error
    assert call(env, "simulate", design=4).is_error and env.state()["scan_time_used_s"] == 0
    names = [t["function"]["name"] for t in env.tools("openai")]
    assert names[-3:] == ["design", "simulate", "submit"]
    other = mrgym.make("Scanner/ErnstFlip-v0", level=1, seed=0)
    other.reset()
    assert not {"design", "simulate"} & {t["function"]["name"] for t in other.tools("openai")}
    assert "rfscan" not in json.dumps(other.tools("openai")) and call(other, "scanner_info", topic="design").is_error


def test_answers_of_the_design_tools_are_small_and_say_what_the_model_is():
    env = make(level=2)
    made = call(env, "design", kind="slr", flip_deg=60, duration_ms=1.4, tbw=5, thickness_mm=5, tol_in_deg=1.2, max_out_deg=1.2, amplitude=1.1)
    sim = call(env, "simulate", design=1, b1_scale=0.9, amp_sat_ut=14)
    assert len(made.text) < 800 and len(sim.text) < 1100                                                  # about 200 and 275 tokens at 4 characters
    assert "b1_scale 0.9, amplifier saturating at 14 uT" in sim.text and "specification on this model" in sim.text
    assert len(sim.data["profile"]["flip_deg"]) == tool.PROFILE_BINS + 1 and "b1_ut" not in json.dumps(sim.data)
    assert "designed pulses" in call(env, "scanner_info").text and "Amplifier." in call(env, "scanner_info", topic="design").text
    assert "stored designs: 1" in call(env, "status").text
    kinds = tool.KINDS
    assert tool.MAX_DESIGNS == rfscan.MAX_DESIGNS and set(task.DESIGN_TOOLS["kinds"]) < set(kinds)
    every = tool.build(dict(kind="verse", of=dict(kind="sinc", flip_deg=90, duration_ms=0.6, tbw=4, thickness_mm=5), b1_peak_ut=12.0))[0]
    assert tool.facts(every, LIMITS)["limits"]["ok"] and tool.build(dict(kind="sech", b1_peak_ut=12, duration_ms=8, beta_rad_per_ms=1.2, mu=5))[0].kind == "inversion"


# ------------------------------------------------------------------------------------------------ the family
def test_an_instance_is_a_function_of_seed_and_level_also_under_threads():
    seeds = ["%032x" % (7 + i) for i in range(3)]
    one = [task.make_instance(s, lv, "t") for s in seeds for lv in (1, 2)]
    task._instance.cache_clear()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        two = list(pool.map(lambda a: task.make_instance(a[0], a[1], "t"), [(s, lv) for s in seeds for lv in (1, 2)]))
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    assert len({json.dumps(c["task"]["design"]["specification"], sort_keys=True) for c in one}) >= 3


@pytest.mark.parametrize("level", [1, 2, 3])
def test_the_statement_holds_no_hidden_value_and_states_every_requirement(level):
    env = mrgym.make(ENV, level=level, seed=3)
    obs, _ = env.reset()
    sy = env.hidden_world()["system"]
    for hidden in (sy["tx_gain"], sy.get("rf_amp_sat_ut"), sy["cf_offset_hz"]):
        assert hidden is None or all(f"{hidden:.{d}f}" not in obs.text for d in (2, 3))
    spec = env.episode_config()["task"]["design"]["specification"]
    for word in (f"{spec['flip_deg']:g} +- {spec['tol_in_deg']:g}", f"at most {spec['max_out_deg']:g} degrees", "peak B1 20 uT", "B1rms", "Duration: at most",
                 "between 0.6 and 1.6"):
        assert word in obs.text, word
    assert ("How to do it" in obs.text) == (level == 1) and ("amplifier" in obs.text) == (level > 1)
    for banned in ("safe", "compliant", "patient", "SAR"):
        assert banned not in obs.text


@pytest.mark.parametrize("seed", [0, 4])
def test_level_1_reference_passes_and_the_shortcuts_fail(seed):
    for name, passes in (("reference", True), ("ideal_tuned", False), ("default_pulse", False), ("ignores_transmit", False), ("wrong_units", False)):
        r = policies.run(mrgym.make(ENV, level=1, seed=seed), name)
        score = r.info["score"]
        assert bool(r.reward) == passes and r.info["policy_error"] is None, (name, score)
        if passes:
            assert score["floor_ms"] < score["duration_ms"] <= 1.5 * score["reference_ms"] and score["merit_floor"] < 0.5
            assert score["bound"]["kind"] == "theorem" and score["record"]["specification_class"].startswith("excitation|d")
        else:
            assert score["outcome"] in task.OUTCOMES


def test_a_refused_deliverable_leaves_the_episode_open_and_every_failure_has_its_outcome():
    env = make(level=1, seed=2)
    for bad in ({"design": 3}, {"design": "1"}, {"id": 1}):
        obs = call(env, "submit", **bad)
        assert obs.is_error and obs.data["error"]["type"] in ("invalid_deliverable", "invalid_value", "unknown_parameter", "malformed_call"), obs.text
        assert not env.state()["done"]
    cfg, world = env.episode_config(), env.hidden_world()
    spec, limits = model.spec_of(cfg["task"]), model.limits_of(cfg["task"])
    chain = task.chain_of(world)
    best = model.search(spec, limits, chain)
    cases = dict(
        pass_=best["call"],
        too_long=model.matched(spec, limits, best["tbw"], best["share"], 1.9, chain)[1],
        mask_in_slice=dict(best["call"], amplitude=round(best["call"]["amplitude"] * 1.08, 4), duration_ms=round(best["call"]["duration_ms"] * 1.2, 2)),
        mask_out_of_slice=dict(best["call"], max_out_deg=round(3 * spec["max_out_deg"], 3)),
    )
    seen = {}
    for label, c in cases.items():
        pulse, canon = tool.build(c, limits)
        world["console"]["designs"] = {1: dict(index=1, call=canon, call_json="", facts=tool.facts(pulse, limits))}
        s = task.score(cfg, dict(design=1), world)
        seen[label] = "pass" if s["passed"] else task.outcome(s)
    assert seen == dict(pass_="pass", too_long="too_long", mask_in_slice="mask_in_slice", mask_out_of_slice="mask_out_of_slice"), seen
    hot = tool.build(dict(best["call"], amplitude=4.0), limits)
    world["console"]["designs"] = {1: dict(index=1, call=hot[1], call_json="", facts=tool.facts(hot[0], limits))}
    assert not task.score(cfg, dict(design=1), world)["checks"]["deliverable_valid"]                    # a limit is refused before it is scored
    assert task.outcome(dict(checks=dict(limits=False))) == "limit_violated" and task.outcome(dict(checks=dict(not_below_floor=False))) == "below_floor"
    assert controls.REFERENCE == "reference" and set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
