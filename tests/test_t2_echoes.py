"""``Scanner/T2Echoes-v0``: the physics behind the task, written a second time; the numbers its texts state; the
instances; the verifier; the controls.

The physics check does not use the engine's formulas, the signal core or the family's model. ``ensemble`` below is a
multi-echo spin echo computed with 3 x 3 rotation matrices on isochromats: hard pulses, a crusher either side of every
refocusing pulse as a phase that differs from isochromat to isochromat, relaxation between. Three things must agree
with it: the recursion the family fits with (``model.train``, also as the text level 1 prints for ``calc``), the
closed forms of the first two echoes, and what the scanner serves for hard pulses (``excitation: nonselective``):
there the same isochromats are played through many repetitions, since the scanner serves the periodic state.
The numbers the task texts state about the 2D slice (how long a straight line reads T2, how close the one-angle fit
comes) are then measured on the engine and held equal to the table ``task.BIAS``.
"""
import concurrent.futures
import json
import math

import numpy as np
import pytest

import mrgym
from mrgym import calc, coverage, policies
from mrgym.env import derive_seed
from mrgym.envs.t2_echoes import controls, model, task
from mrgym.scanner import console, kspace
from mrgym.scanner import world as W

ENV_ID = "Scanner/T2Echoes-v0"


# ---------------------------------------------------------------------------------------------- the second calculation
def _rot(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    if axis == "y":
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    raise ValueError(axis)


def ensemble(n, esp, t1, t2, theta_deg, alpha_deg=90.0, spins=720):
    """Echoes 1 .. n of a train of hard pulses, by rotating isochromats: excitation about x from equilibrium, refocusing
    about y, a crusher of one cycle either side of every refocusing pulse (isochromat j gains the phase 2 pi j / spins),
    relaxation over every half echo spacing. The echo is the mean of My (the excitation about x puts M along y)."""
    phase = 2 * math.pi * np.arange(spins) / spins
    m = np.tile(_rot("x", math.radians(alpha_deg)) @ np.array([0.0, 0.0, 1.0]), (spins, 1))         # [spin, (x, y, z)]: M along +y
    e1, e2 = math.exp(-esp / (2 * t1)), math.exp(-esp / (2 * t2))
    pulse = _rot("y", math.radians(theta_deg))

    def half(m):                                                         # relaxation, then the crusher
        m = np.column_stack([m[:, 0] * e2, m[:, 1] * e2, 1.0 + (m[:, 2] - 1.0) * e1])
        c, s = np.cos(phase), np.sin(phase)
        return np.column_stack([c * m[:, 0] - s * m[:, 1], s * m[:, 0] + c * m[:, 1], m[:, 2]])

    out = []
    for _ in range(n):
        m = half(m) @ pulse.T
        m = half(m)
        out.append(m[:, 1].mean())
    return np.array(out)


def test_the_recursion_is_the_isochromat_ensemble_and_the_two_echo_closed_forms():
    worst = 0.0
    for t1, t2, theta, esp in ((900.0, 80.0, 180.0, 10.0), (400.0, 40.0, 150.0, 13.74), (2000.0, 200.0, 120.0, 8.0), (600.0, 150.0, 75.0, 12.0)):
        ref = ensemble(12, esp, t1, t2, theta)
        got = model.train(12, esp, t2, theta, t1)
        worst = max(worst, float(np.abs(got - ref).max()))
        b, e2, e1 = math.radians(theta), math.exp(-esp / t2), math.exp(-esp / t1)
        assert got[0] == pytest.approx(math.sin(b / 2) ** 2 * e2, abs=1e-12)                        # the datasheet's S_1
        assert got[1] == pytest.approx(math.sin(b / 2) ** 4 * e2 ** 2 + 0.5 * math.sin(b) ** 2 * e2 * e1, abs=1e-12)    # and S_2
    assert worst < 1e-10
    assert np.allclose(model.train(16, 10.0, 70.0, 180.0), np.exp(-10.0 * np.arange(1, 17) / 70.0), atol=1e-13)   # exact pulses: exp(-TE / T2)
    low = model.train(16, 10.0, 70.0, 140.0)
    decay = math.exp(-10.0 / 70.0)
    assert low[0] < 0.9 * decay and low[1] / low[0] > 1.1 * decay         # the first echo is low; the second is fed from z: it hardly falls


def test_the_text_for_calc_is_the_same_recursion_and_the_stated_fit_runs_in_calc():
    names = {}
    calc.calc(model.CALC_TEXT, names)
    rows = calc.calc("e = train(16, 13.74, 72.0, 133.0)\ne", names)
    assert np.allclose(rows[-1][1], model.train(16, 13.74, 72.0, 133.0), rtol=1e-6)
    echoes = 2.5 * model.train(12, 13.74, 72.0, 133.0)
    text = f"esp = 13.74\nechoes = array({[float(v) for v in echoes]})\n{task.fit_text()}\np"
    p = calc.calc(text, names)[-1][1]
    assert p[1] == pytest.approx(72.0, rel=1e-4) and p[2] == pytest.approx(133.0, rel=1e-3)
    assert controls.fit_model(13.74 * np.arange(1, 13), echoes) == pytest.approx(72.0, rel=1e-5)   # the reference's fit: the same answer
    assert model.CALC_TEXT in "\n".join(line[5:] for line in task.method_text().split("\n")) and task.fit_text() in task.method_text()


def _disc(r, cy=0.0, cx=0.0):
    return lambda y, x: np.hypot(y - cy, x - cx) - r


MATERIALS = [(400.0, 40.0), (400.0, 90.0), (400.0, 160.0), (900.0, 40.0), (900.0, 90.0), (900.0, 200.0), (2000.0, 40.0), (2000.0, 90.0), (2000.0, 200.0)]


def _world(tx, adc=2.0):
    """A hand-made world: nine discs, T1 400 to 2000 ms x T2 40 to 200 ms, uniform fields, transmit scale ``tx``."""
    tissues = {f"v{i}": dict(pd=1.0, t1_ms=t1, t2_ms=t2, t2p_ms=100.0, adc_um2_ms=adc, fat=False) for i, (t1, t2) in enumerate(MATERIALS)}
    k = len(MATERIALS)
    shapes = [(f"v{i}", _disc(10.0, 45 * math.sin(2 * math.pi * i / k), 45 * math.cos(2 * math.pi * i / k))) for i in range(k)]
    w = W.flat_world(W.shapes_phantom(shapes, b0_t=1.5, n=96, tissues=tissues), tx_gain=tx)
    w["console"].update(sequences=["mese"], b1rms_limit_ut=6.0, min_pixel_mm=0.5)
    return w


def _echoes(w, **change):
    """(echo times, per material the echoes the engine serves for a voxel of that material) of the train of the method."""
    pl = console.plan(dict(task.METHOD["scan"], **change), w["console"])
    assert pl["accept"], pl.get("violations")
    rp, eng = pl["realised"], kspace.Engine(w)
    out = []
    for i in range(len(MATERIALS)):
        c = eng.w["names"].index(f"v{i}")
        weight = eng.w["weights"][c][eng.sup[c]]
        j = int(np.argmax(weight))                                       # a voxel inside the disc: weight 1, proton density 1
        out.append(np.abs(np.array([a[j] / weight[j] for a in eng.amplitudes(rp, c)])))
    return np.array(rp["echo_times_ms"]), out


def periodic(n, esp, tr, t1, t2, theta_deg, alpha_deg, spins=720, repetitions=12):
    """The train of ``ensemble`` repeated every ``tr``: at the end of a repetition what is transverse is dephased (set to
    zero) and every isochromat keeps its own Mz, which recovers until the next excitation. The echoes of the last one."""
    phase = 2 * math.pi * np.arange(spins) / spins
    e1, e2 = math.exp(-esp / (2 * t1)), math.exp(-esp / (2 * t2))
    excite, pulse = _rot("x", math.radians(alpha_deg)), _rot("y", math.radians(theta_deg))
    c, s = np.cos(phase), np.sin(phase)

    def half(m):
        m = np.column_stack([m[:, 0] * e2, m[:, 1] * e2, 1.0 + (m[:, 2] - 1.0) * e1])
        return np.column_stack([c * m[:, 0] - s * m[:, 1], s * m[:, 0] + c * m[:, 1], m[:, 2]])

    mz = np.ones(spins)
    for _ in range(repetitions):
        m = np.column_stack([np.zeros(spins), np.zeros(spins), mz]) @ excite.T
        out = []
        for _ in range(n):
            m = half(half(m) @ pulse.T)
            out.append(m[:, 1].mean())
        mz = 1.0 + (m[:, 2] - 1.0) * math.exp(-(tr - n * esp) / t1)
    return np.array(out)


def test_the_scanner_serves_the_periodic_ensemble_for_hard_pulses():
    """Hard pulses (``excitation: nonselective``), materials that do not diffuse: the echoes the scanner serves are those
    of isochromats through many repetitions, with the gratings along z that one repetition leaves to the next. With
    the diffusion of water the crushers cost 0.2 to 0.9 % of the first echo more; one train from rest is 0.2 to 0.6 % off."""
    worst = dict(one_train=0.0, diffusion=0.0)
    for tx in (0.8, 1.0, 1.2):
        for tr in (3000.0, 1150.0):
            te, echoes = _echoes(_world(tx, adc=1e-9), excitation="nonselective", echo_spacing_ms=10.0, tr_ms=tr)
            _, wet = _echoes(_world(tx), excitation="nonselective", echo_spacing_ms=10.0, tr_ms=tr)
            for (t1, t2), s, d in zip(MATERIALS[::4], echoes[::4], wet[::4]):
                ref = periodic(len(te), 10.0, tr, t1, t2, 180.0 * tx, 90.0 * tx)
                one = ensemble(len(te), 10.0, t1, t2, 180.0 * tx, 90.0 * tx)
                err = float(np.abs(s / s[0] - ref / ref[0]).max())
                assert err < (1e-9 if tx == 1.0 else 2e-4), (tx, tr, err)    # exact pulses store nothing; else the engine carries 16 orders
                if tx != 1.0:
                    worst["one_train"] = max(worst["one_train"], float(np.abs(s / s[0] - one / one[0]).max()))
                worst["diffusion"] = max(worst["diffusion"], float(np.abs(d / d[0] - s / s[0]).max()))
    assert 1e-3 < worst["one_train"] < 7e-3 and 1e-3 < worst["diffusion"] < 9e-3, worst


def _bias(pulse, scales):
    """Bias of T2 in percent over the materials and transmit scales: {estimator: (lowest, highest)}."""
    rows = dict(line=[], line_without_first=[], model=[])
    for tx in scales:
        te, echoes = _echoes(_world(tx), rf_pulse=pulse)
        for (_, t2), s in zip(MATERIALS, echoes):
            rows["line"].append(100 * (controls.fit_line(te, s) / t2 - 1))
            rows["line_without_first"].append(100 * (controls.fit_line(te, s, slice(1, None)) / t2 - 1))
            rows["model"].append(100 * (controls.fit_model(te, s) / t2 - 1))
    return {k: (min(v), max(v)) for k, v in rows.items()}


BANDS = {"8-22": (0.78, 0.85, 0.92, 1.08, 1.15, 1.22), "25-45": (0.55, 0.65, 0.75, 1.25, 1.35, 1.45)}


def test_every_bias_the_texts_state_is_what_the_engine_gives_on_the_slice():
    """``task.BIAS`` against the engine: the slice profile of both pulses and the transmit scale, T1 400 to 2000 ms."""
    seen = {(pulse, band): _bias(pulse, BANDS[band]) for pulse, band in (("sharp", "8-22"), ("normal", "8-22"), ("sharp", "25-45"))}
    for estimator, table in task.BIAS.items():
        for key, (lo, hi) in table.items():
            got = seen[key][estimator]
            assert abs(got[0] - lo) < 0.8 and abs(got[1] - hi) < 0.8, (estimator, key, got)
    # what the levels rest on: within the tolerance a line fails and the model passes ...
    assert task.BIAS["line"][("normal", "8-22")][0] >= 100 * task.TOLERANCE and task.BIAS["model"][("sharp", "8-22")][1] < 100 * task.TOLERANCE
    # ... and the texts are written from the table
    assert set(task.TX_BANDS.values()) <= set(BANDS) and [task.TX_BANDS[lv] for lv in sorted(task.LEVELS)] == ["8-22", "8-22", "25-45"]
    for lv, d in task.LEVELS.items():
        lo, hi = d["world_spec"]["tx_gain_dev"]
        assert f"{round(100 * lo)}-{round(100 * hi)}" == task.TX_BANDS[lv]
    assert "the line reads T2 4 to 24 % long" in task.method_text() and "leaves T2 up to 10 % long" in task.method_text()
    assert "4 to 35 % long" in task.LEVELS[2]["summary"] and "5 to 76 % long" in task.LEVELS[3]["summary"]


# ---------------------------------------------------------------------------------------------- instances
def _cfg(level, seed):
    return task.make_instance(derive_seed(ENV_ID, level, "train", seed, environ={}), level, task_id="t")


def test_registration_and_the_constants_of_the_map():
    spec = mrgym.spec(ENV_ID)
    assert (spec.kind, spec.family, tuple(spec.levels), spec.default_level) == ("candidate", "mrgym.envs.t2_echoes", (1, 2, 3), 2)
    assert (task.SUBJECT, task.KIND) == ("A", "measure") and "stimulated echoes" in task.ISOLATES
    assert set(task.COMPETENCIES) <= set(coverage.ids())
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and {"prior", "fixed_recipe", "wrong_units", "mono_exp", "single_echoes"} <= set(controls.MUST_FAIL)
    assert task.card()["id"] == ENV_ID and [c["name"] for c in task.card()["controls"]][0] == "reference"
    assert all(d["world_spec"]["console_sequences"] == ["mese"] for d in task.LEVELS.values())     # the console offers what the task needs, no more


def test_an_instance_is_a_function_of_seed_and_level_also_under_threads():
    jobs = [(lv, s) for lv in task.LEVELS for s in range(4)]
    serial = {j: json.dumps(_cfg(*j), sort_keys=True) for j in jobs}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for _ in range(2):
            threaded = dict(zip(jobs, pool.map(lambda j: json.dumps(_cfg(*j), sort_keys=True), jobs)))
            assert threaded == serial
    assert len(set(serial.values())) == len(jobs)                        # another seed or level is another instance


def test_three_marked_vials_inside_the_told_ranges():
    for level, lv in task.LEVELS.items():
        seen, gains = [], []
        for seed in range(12):
            cfg = _cfg(level, seed)
            t = cfg["task"]
            layout = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
            truth = W.truth(layout)
            assert [r["id"] for r in t["regions"]] == ["R1", "R2", "R3"] and t["t2_range_ms"] == list(lv["t2_range_ms"])
            assert (cfg["budget_s"], t["tolerance"], t["t1_range_ms"]) == (lv["budget_s"], 0.10, [400.0, 2000.0])
            for r in t["regions"]:
                assert set(r) == {"id", "y_mm", "x_mm", "radius_mm"}     # a circle, no material
                assert t["t2_range_ms"][0] <= truth[r["id"]]["t2_ms"] <= t["t2_range_ms"][1]
                assert 400.0 <= truth[r["id"]]["t1_ms"] <= 2000.0 and truth[r["id"]]["t2_ms"] <= 0.4 * truth[r["id"]]["t1_ms"] + 1e-9
                seen.append(truth[r["id"]]["t2_ms"])
            assert task.true_t2_ms(cfg) == {r["id"]: truth[r["id"]]["t2_ms"] for r in t["regions"]}
            off = 100 * abs(layout["system"]["tx_gain"] - 1.0)
            assert t["transmit_off_percent"][0] <= off <= t["transmit_off_percent"][1]             # the told band holds
            gains.append(layout["system"]["tx_gain"])
        assert max(seen) / min(seen) > 3.0 and min(gains) < 1.0 < max(gains)                       # the range is used; off either way


def test_the_statement_is_written_from_the_public_task_alone():
    for level in task.LEVELS:
        cfg = _cfg(level, 1)
        text = task.statement(cfg)
        assert task.statement(dict(task=cfg["task"])) == text            # nothing hidden is an input of the text
        layout = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
        prose = "\n".join(line for line in text.splitlines() if not line.startswith("  R"))    # without the table of circles
        for v in W.truth(layout).values():
            for key in ("t2_ms", "t1_ms"):
                assert f"{v[key]:.2f}" not in prose and f"{v[key]:.1f}" not in prose
            assert f"{v['pd']:.3f}" not in prose
        assert f"{layout['system']['tx_gain']:.3f}" not in prose
        assert ("How to do it" in text) == (level == 1) and ("hints" in cfg["task"]) == (level == 1)
        lo, hi = cfg["task"]["transmit_off_percent"]
        assert "T2 values lie between 40 and 200 ms, T1 values between 400 and 2000 ms." in text and f"is off by {lo} to {hi} %, up or down" in text
        assert "within a factor 1.1 of the true T2 (between T2 / 1.1 and 1.1 x T2)" in text and "An answer in the wrong unit is wrong." in text
        assert "`mese`" in text and "Every scan of this console starts with 40 s of adjustments, which are part of its scan time: the scan time pays one scan of useful length." in text
    method = task.method_text()
    assert 'rf_pulse "sharp", n_echoes 16, tr_ms 1150, matrix 48, partial_fourier 0.75' in method
    assert "at least 8 % of the largest (at least 4)" in method and "T1 is set to 1000 ms" in method


# ---------------------------------------------------------------------------------------------- the verifier
def test_the_pass_window_is_a_factor_of_the_true_t2():
    cfg = _cfg(2, 0)
    truth = task.true_t2_ms(cfg)
    at = lambda f=1.0, **one: {"t2_ms": {i: v * one.get(i, f) for i, v in truth.items()}}      # noqa: E731
    for answer in (at(), at(1.099), at(1 / 1.099)):
        s = task.score(cfg, answer)
        assert s["passed"] and s["diagnostic"] == "" and set(s["checks"]) == {"deliverable_valid", *(f"t2_within_tolerance_{i}" for i in truth)}
    s = task.score(cfg, at(R2=1.101))
    assert not s["passed"] and [k for k, v in s["checks"].items() if not v] == ["t2_within_tolerance_R2"]
    assert s["diagnostic"] == "R2 is off by +10.1 %" and task.outcome(s) == "some_regions_within"
    s = task.score(cfg, at(0.9))                                         # 10 % short is outside: the window ends at 1 / 1.1
    assert not s["passed"] and task.outcome(s) == "no_region_within" and s["error"] == pytest.approx(-math.log(0.9))
    assert s["distance"] == dict(regions=3, within=0, worst_log_ratio=s["error"]) and task.OUTCOMES == ("no_region_within", "some_regions_within")


def test_deliverables_the_console_cannot_evaluate_are_refused():
    cfg = _cfg(1, 0)
    good = {"t2_ms": {"R1": 60.0, "R2": 80.0, "R3": 100.0}}
    assert task.check(cfg, good, {}) is None
    shape = 'the deliverable is {"t2_ms": {"R1": <number>, "R2": <number>, "R3": <number>}}'
    for bad in (None, [], {"t2": good["t2_ms"]}, {"t2_ms": {"R1": 60.0, "R2": 80.0}}, {"t2_ms": dict(good["t2_ms"], R4=1.0)}, {"t2_ms": 80.0}):
        assert task.check(cfg, bad, {}) == shape
    with_r1 = lambda v: {"t2_ms": dict(good["t2_ms"], R1=v)}             # noqa: E731
    assert task.check(cfg, with_r1("60"), {}) == task.check(cfg, with_r1(True), {}) == task.check(cfg, with_r1(float("nan")), {}) \
        == "the T2 of R1 is not a finite number"
    assert task.check(cfg, with_r1(0.06), {}) == ("the T2 of R1 is 0.06: not a T2 in milliseconds, which lies between 2 and 5000 "
                                                  "(a value of this size looks like seconds)")
    s = task.score(cfg, with_r1(0.06))
    assert s["checks"] == dict(deliverable_valid=False) and not s["passed"] and s["error"] is None
    env = mrgym.make(ENV_ID, level=1, seed=0)
    env.reset()
    step = env.step({"name": "submit", "arguments": with_r1(0.06)})
    assert step.observation.is_error and not step.done                   # refused; the episode continues
    env.close()


# ---------------------------------------------------------------------------------------------- the controls
def test_which_echoes_enter_a_fit():
    s = np.array([1.0, 1.1, 0.8, 0.5, 0.2, 0.09, 0.085, 0.03, 0.2])
    assert controls.kept(s) == 6                                         # down to the last above 8 % of the largest, the first below ends it
    assert controls.kept(np.array([1.0, 0.05, 0.01, 0.001, 0.0005])) == 4   # at least four
    assert controls.kept(np.array([1.0, 0.9, 0.8])) == 3
    te = 10.0 * np.arange(1, 9)
    assert controls.fit_line(te, np.exp(-te / 50.0)) == pytest.approx(50.0, rel=1e-9)
    assert controls.fit_line(te, np.ones(8)) == task.T2_VALID_MS[1]      # no decay seen: the top of the valid range


@pytest.mark.parametrize("level", sorted(task.LEVELS))
def test_the_reference_passes_and_every_shortcut_fails(level):
    for seed in ("test:0", "test:1"):
        r = policies.run(mrgym.make(ENV_ID, level=level, seed=seed), controls.REFERENCE)
        assert r.reward == 1.0 and r.info["policy_error"] is None and r.info["budget"]["used_s"] <= task.LEVELS[level]["budget_s"]
    for name in controls.MUST_FAIL:
        r = policies.run(mrgym.make(ENV_ID, level=level, seed="test:0"), name)
        assert r.reward == 0.0, name
        if name == "single_echoes":                                      # the series of spin echoes does not fit the scan time: one scan does
            assert "BudgetExhausted" in r.info["policy_error"] and r.info["budget"]["scans"] == 1
        else:
            assert r.info["policy_error"] is None, name
    r = policies.run(mrgym.make(ENV_ID, level=level, seed="test:0"), "line_fit")
    assert r.reward == 0.0 and min(r.info["score"]["per_region"].values()) > 0.0     # the same train, a line: every region long


def test_the_reference_plays_the_stated_method():
    env = mrgym.make(ENV_ID, level=1, seed="test:0")
    env.reset()
    sc = policies.ToolScanner(env)
    scans, scan = [], sc.scan
    sc.scan = lambda protocol, rois=None: scans.append(dict(protocol)) or scan(protocol, rois)
    controls.reference(sc)
    env.close()
    assert scans == [dict(task.METHOD["scan"])]
    cost = task.METHOD["scan"]["tr_ms"] * (48 * 0.75 + 1) / 1000 + task.ADJUST_S                     # adjustments + one dummy repetition + 36 lines
    assert cost < task.BUDGET_S < cost + task.ADJUST_S                   # the budget pays the train, and after it no second scan of any kind
    assert task.BUDGET_S - 2 * task.ADJUST_S == 10.0                     # two scans of any kind: ten seconds of scanning between them
    assert all(d["world_spec"]["console_system"] == dict(overhead_s=40.0) and d["budget_s"] == 90.0 for d in task.LEVELS.values())


def test_the_blind_recipe_is_one_constant_inside_the_told_range():
    cfgs = [_cfg(2, s) for s in range(16)]
    recipe = controls.tune(cfgs)
    assert set(recipe) == {"t2_ms", "tuned_region_pass_rate"} and 40.0 <= recipe["t2_ms"] <= 200.0
    assert recipe["tuned_region_pass_rate"] < 0.35                       # one region in a few; three at once far below 5 %
