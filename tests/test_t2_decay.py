"""``Scanner/T2Decay-v0``: the physics behind the task, written a second time; the instances; the verifier; the controls.

The physics check does not use the engine's formulas. ``spin_echo`` below is the 2D spin echo of the datasheet computed
with 3 x 3 rotation matrices: the windowed-sinc excitation and the refocusing pulse (1.5 times the slice) as cascades
of small rotations at every height through the slice, the crushers as an average over their phase, and the
longitudinal steady state by playing the cycle until it repeats. One slice thickness is the width at half maximum of
the small-flip-angle profile, as the datasheet states since engine 0.3.0 (``_half_width`` finds it on the transform
of the pulse; on 0.2.0 it was one nominal bandwidth of the pulse, 0.6 % less for this pulse). The engine serves Cayley-Klein parameters and a
closed-form periodic state (``kspace.slice_table``). The two must agree, and the consequences the task rests on must
follow from the independent one: at one repetition time T2 reads short, and holding ``tr - 0.75 te`` fixed removes it.
"""
import concurrent.futures
import json
import math

import numpy as np
import pytest

import mrgym
from mrgym import coverage, policies
from mrgym.env import derive_seed
from mrgym.envs import family as F
from mrgym.envs.t2_decay import controls, task
from mrgym.scanner import console, kspace, signal
from mrgym.scanner import world as W

ENV_ID = "Scanner/T2Decay-v0"
N_SUB, TBW = 128, 4.0                       # the `normal` pulse of the datasheet: 128 samples, time-bandwidth product 4


# ---------------------------------------------------------------------------------------------- the second calculation
def _rot_x(a):
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, s], -1), np.stack([z, -s, c], -1)], -2)


def _rot_z(a):
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, s, z], -1), np.stack([-s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _shape():
    t = (np.arange(N_SUB) + 0.5) / N_SUB - 0.5
    amp = np.sinc(TBW * t) * 0.5 * (1 + np.cos(2 * np.pi * t))
    return t, amp / amp.sum()


def _half_width():
    """Full width at half maximum of the small-flip-angle profile (the Fourier transform of the pulse) in nominal
    bandwidths of the slice gradient: the console sets the gradient so that this width is the prescribed thickness."""
    t, amp = _shape()
    lo, hi = 0.0, 1.0                                                    # the transform is 1 at the centre and past its first zero at 1
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if abs(np.sum(amp * np.exp(-2j * np.pi * TBW * mid * t))) > 0.5 else (lo, mid)
    return lo + hi


def _pulse(flip, zeta, rephase):
    """Rotation matrices [len(zeta), 3, 3] of the Hann-windowed sinc at the heights ``zeta`` (nominal bandwidths of the
    slice gradient): small rotations about x between half-steps of precession about z, then a rephasing lobe of
    ``rephase`` x the selection area."""
    _, amp = _shape()
    step = 2 * np.pi * TBW * zeta / N_SUB
    half, r = _rot_z(step / 2), np.broadcast_to(np.eye(3), (len(zeta), 3, 3))
    for w in amp:
        r = half @ _rot_x(np.full(len(zeta), w * flip)) @ half @ r
    return _rot_z(-rephase * step * N_SUB) @ r


def spin_echo(tr_ms, te_ms, t1_ms, b1=1.0, cycles=60):
    """Echo amplitude per unit M0 of the 2D spin echo (without the T2 decay): |sum over the slice|."""
    h = 1.0 / (6 * TBW)
    reach = 0.5 + 12.0 / TBW
    zeta = np.arange(-reach, reach + 1e-9, h)                            # both sides of the slice, explicitly, in nominal bandwidths
    weight = np.full(len(zeta), h) / _half_width()                       # the integral is per slice thickness
    weight[[0, -1]] /= 2
    exc, ref = _pulse(b1 * np.pi / 2, zeta, 0.5), _pulse(b1 * np.pi, zeta / 1.5, 0.0)
    after = exc[:, :, 2]                                                 # magnetisation after the excitation from Mz = 1
    transverse = np.stack([after[:, 0], after[:, 1], np.zeros(len(zeta))], -1)[:, :, None]
    crusher = [_rot_z(np.full(len(zeta), p)) for p in 2 * np.pi * np.arange(8) / 8]
    echo = np.mean([(c @ ref @ c @ transverse)[:, :, 0] for c in crusher], axis=0)   # what survives a crusher pair
    cos_exc, inv = exc[:, 2, 2], ref[:, 2, 2]                            # Mz after each pulse per Mz before
    mz, e_half, e_rest = np.ones(len(zeta)), math.exp(-te_ms / 2 / t1_ms), math.exp(-(tr_ms - te_ms / 2) / t1_ms)
    for _ in range(cycles):                                              # excite, wait TE / 2, refocus, wait TR - TE / 2
        mz = 1 + (inv * (1 + (cos_exc * mz - 1) * e_half) - 1) * e_rest
    return float(abs(np.sum(weight * mz * (echo[:, 0] + 1j * echo[:, 1]))))


def _fit_error(te_ms, tr_of, t1_ms, t2_ms):
    """Relative error of T2 from a line through ln(signal) of the second calculation, scans at tr_of(te)."""
    s = [spin_echo(tr_of(te), te, t1_ms) * math.exp(-te / t2_ms) for te in te_ms]
    return -1.0 / np.polyfit(te_ms, np.log(s), 1)[0] / t2_ms - 1.0


def test_the_engine_serves_the_spin_echo_of_the_second_calculation():
    for t1, b1, tr, te in ((400.0, 0.88, 600.0, 12.0), (1200.0, 1.0, 600.0, 110.0), (2400.0, 1.12, 1582.5, 110.0), (2400.0, 1.0, 3000.0, 300.0)):
        served = kspace.slice_signal(dict(sequence="se", tr_ms=tr, te_ms=te), dict(t1_ms=t1), np.pi / 2 * b1, TBW)[0][0]
        assert abs(served.imag) < 1e-12 and spin_echo(tr, te, t1, b1) == pytest.approx(served.real, rel=1e-9)
    # ideal hard pulses (what a nonselective scan serves): the textbook factor 1 - 2 E(TR - TE / 2) + E(TR), written out,
    # which is 1 - E(TR - TE) to first order in TE / T1: the recovery runs from the refocusing pulse
    ideal = 1 - 2 * math.exp(-(900.0 - 40.0) / 700.0) + math.exp(-900.0 / 700.0)
    assert float(signal.se(np.pi / 2, 900.0, 80.0, 700.0)) == pytest.approx(ideal, rel=1e-12)
    assert ideal == pytest.approx(1 - math.exp(-(900.0 - 80.0) / 700.0), rel=5e-3)


def test_one_repetition_time_reads_t2_short_and_holding_tr_minus_three_quarters_te_fixed_does_not():
    """The trap and its cure, from the second calculation alone (no engine): three echo times, TR0 = 600 ms."""
    te = np.array(controls.NARROW_TE_MS)
    assert task.RECOVERY_SHARE == 0.75
    for t1, t2 in ((500.0, 60.0), (1000.0, 150.0), (2500.0, 200.0)):
        fixed = _fit_error(te, lambda t: 600.0, t1, t2)
        held = _fit_error(te, lambda t: 600.0 + task.RECOVERY_SHARE * t, t1, t2)
        ideal_rule = _fit_error(te, lambda t: 600.0 + t, t1, t2)
        half = _fit_error(te, lambda t: 600.0 + 0.5 * t, t1, t2)
        assert fixed < -0.04 and abs(held) < 0.02                         # 4 to 21 % short; within 2 % with the recovery held
        assert half < held < ideal_rule and ideal_rule > 0.01             # the zero lies between 0.5 and 1: the pulse inverts in part
    assert _fit_error(te, lambda t: 600.0, 2500.0, 200.0) < -0.20          # past the tolerance of 10 % where T1 and T2 / T1 are large
    # the method of level 1 (TR0 1500 ms): the error left is far inside the tolerance
    m = task.METHOD
    assert abs(_fit_error(np.array(m["te_ms"]), lambda t: m["tr0_ms"] + task.RECOVERY_SHARE * t, 2500.0, 200.0)) < 0.01


@pytest.fixture(scope="module")
def scanned():
    """One level-1 world, scanned noise-free as the stated method scans it, and with one repetition time."""
    cfg = task.make_instance(derive_seed(ENV_ID, 1, "train", 3, environ={}), 1, task_id="t")
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    engine, m = kspace.Engine(world), task.METHOD
    regions = cfg["task"]["regions"] + [m["background"]]

    def read(te, tr):
        rp = console.plan(dict(controls.FINE, te_ms=te, tr_ms=tr), world["console"])["realised"]
        rows = F.noise_free(engine, rp, regions)[1]
        return {k: r["mean_magnitude"][0] for k, r in rows.items()}
    held = [read(te, m["tr0_ms"] + task.RECOVERY_SHARE * te) for te in m["te_ms"]]
    short = [read(te, 600.0) for te in controls.NARROW_TE_MS]
    return cfg, held, short, world


def test_the_stated_method_recovers_t2_on_noise_free_scans_of_the_engine(scanned):
    cfg, held, short, _ = scanned
    truth, m = task.true_t2_ms(cfg), task.METHOD
    for i, t2 in truth.items():
        got = controls.fit_t2(m["te_ms"], [s[i] for s in held], held[0]["background"])
        assert got == pytest.approx(t2, rel=0.02)
    low = [controls.fit_t2(controls.NARROW_TE_MS, [s[i] for s in short], short[0]["background"]) / t2 - 1 for i, t2 in truth.items()]
    assert max(low) < -0.02 and min(low) < -0.05                          # one repetition time of 600 ms: every region short


def test_the_background_circle_of_the_method_holds_no_object(scanned):
    cfg, held, _, world = scanned
    bg, n, d = task.METHOD["background"], world["n"], world["extent_mm"] / world["n"]
    y, x = np.meshgrid((np.arange(n) - n // 2) * d, (np.arange(n) - n // 2) * d, indexing="ij")
    inside = np.hypot(y - bg["y_mm"], x - bg["x_mm"]) <= bg["radius_mm"] + 5.0
    assert inside.sum() > 1000 and float(world["weights"].sum(0)[inside].max()) < 1e-6
    # whatever the draw: the body reaches at most 98 mm + 3.6 mm x sqrt(2) from the isocentre, the circle starts beyond that
    reach = 98.0 + 3.6 * math.sqrt(2.0)
    assert math.hypot(bg["y_mm"], bg["x_mm"]) - bg["radius_mm"] > reach + 25.0 and abs(bg["y_mm"]) + bg["radius_mm"] < world["console"]["fov_mm"] / 2
    # without noise what the truncated image leaks there is small against the echoes the fit keeps
    assert all(s["background"] < 0.005 * min(s[i] for i in task.true_t2_ms(cfg)) for s in held[:2])


def test_the_fit_leaves_out_echoes_at_the_floor():
    te, t2 = np.array([12.0, 40.0, 80.0, 160.0]), 15.0
    clean = 2.0 * np.exp(-te / t2)
    floor = 0.01
    seen = np.maximum(clean, 1.2 * floor)                                # the last two echoes sit on the floor
    assert controls.fit_t2(te, clean, floor) == pytest.approx(t2, rel=1e-9)
    assert controls.fit_t2(te, seen, floor) == pytest.approx(t2, rel=1e-9)             # dropped: below 5 floors
    assert controls.fit_t2(te, seen, floor, drop=False) > 1.2 * t2                     # kept: T2 reads long
    assert controls.fit_t2(te, np.array([1.0, 0.5, 0.2, 0.09]), 1e-6) == pytest.approx(
        -1 / np.polyfit(te[:3], np.log([1.0, 0.5, 0.2]), 1)[0])                         # below 10 % of the first echo: dropped
    assert controls.fit_t2(te[:2], np.array([1.0, 1.0]), floor) == task.T2_VALID_MS[1]  # no decay seen: the top of the valid range


# ---------------------------------------------------------------------------------------------- instances
def _cfg(level, seed):
    return task.make_instance(derive_seed(ENV_ID, level, "train", seed, environ={}), level, task_id="t")


def test_registration_and_the_constants_of_the_map():
    spec = mrgym.spec(ENV_ID)
    assert (spec.kind, spec.family, tuple(spec.levels), spec.default_level) == ("candidate", "mrgym.envs.t2_decay", (1, 2, 3), 2)
    assert (task.SUBJECT, task.KIND, task.ISOLATES) == ("A", "measure", "T2 decay of the spin echo")
    assert set(task.COMPETENCIES) <= set(coverage.ids())
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and {"prior", "fixed_recipe", "wrong_units"} <= set(controls.MUST_FAIL)
    assert task.card()["id"] == ENV_ID and [c["name"] for c in task.card()["controls"]][0] == "reference"


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
        seen = []
        for seed in range(12):
            cfg = _cfg(level, seed)
            t = cfg["task"]
            layout = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
            truth = W.truth(layout)
            assert [r["id"] for r in t["regions"]] == ["R1", "R2", "R3"] and t["t2_range_ms"] == list(lv["t2_range_ms"])
            assert (cfg["budget_s"], t["tolerance"], t["t1_range_ms"]) == (lv["budget_s"], 0.10, [300.0, 2500.0])
            for r in t["regions"]:
                assert set(r) == {"id", "y_mm", "x_mm", "radius_mm"}     # a circle, no material
                assert t["t2_range_ms"][0] <= truth[r["id"]]["t2_ms"] <= t["t2_range_ms"][1]
                assert 300.0 <= truth[r["id"]]["t1_ms"] <= 2500.0
                seen.append(truth[r["id"]]["t2_ms"])
            assert task.true_t2_ms(cfg) == {r["id"]: truth[r["id"]]["t2_ms"] for r in t["regions"]}
        assert max(seen) / min(seen) > (10.0 if level == 3 else 3.0)       # the range is used


def test_the_statement_is_written_from_the_public_task_alone():
    for level in task.LEVELS:
        cfg = _cfg(level, 1)
        text = task.statement(cfg)
        assert task.statement(dict(task=cfg["task"])) == text            # nothing hidden is an input of the text
        truth = W.truth(W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True))
        prose = "\n".join(line for line in text.splitlines() if not line.startswith("  R"))    # without the table of circles
        for v in truth.values():
            for key in ("t2_ms", "t1_ms", "pd"):
                assert f"{v[key]:.2f}" not in prose and f"{v[key]:.1f}" not in prose
        assert ("How to do it" in text) == (level == 1) and ("hints" in cfg["task"]) == (level == 1)
        lo, hi = cfg["task"]["t2_range_ms"]
        assert f"T2 values lie between {lo:g} and {hi:g} ms, T1 values between 300 and 2500 ms." in text
        assert "within a factor 1.1 of the true T2 (between T2 / 1.1 and 1.1 x T2)" in text and "An answer in the wrong unit is wrong." in text
    method = task.method_text()
    assert "te_ms 12, 40, 80 and 160" in method and "tr_ms = 1500 + 0.75 x te_ms (1509, 1530, 1560 and 1620)" in method
    assert "matrix 64 and partial_fourier 0.75" in method
    assert "y_mm -105, x_mm -105, radius_mm 10" in method and "at least 5 times" in method and "at least 10 %" in method


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
    assert "looks like seconds" not in task.check(cfg, with_r1(6000.0), {}) and "not a T2 in milliseconds" in task.check(cfg, with_r1(6000.0), {})
    s = task.score(cfg, with_r1(0.06))
    assert s["checks"] == dict(deliverable_valid=False) and not s["passed"] and s["error"] is None
    env = mrgym.make(ENV_ID, level=1, seed=0)
    env.reset()
    step = env.step({"name": "submit", "arguments": with_r1(0.06)})
    assert step.observation.is_error and not step.done                   # refused; the episode continues
    env.close()


# ---------------------------------------------------------------------------------------------- the controls
@pytest.mark.parametrize("level", sorted(task.LEVELS))
def test_the_reference_passes_and_every_shortcut_fails(level):
    for seed in ("test:0", "test:1"):
        r = policies.run(mrgym.make(ENV_ID, level=level, seed=seed), controls.REFERENCE)
        assert r.reward == 1.0 and r.info["policy_error"] is None and r.info["budget"]["used_s"] <= task.LEVELS[level]["budget_s"]
    for name in controls.MUST_FAIL:
        r = policies.run(mrgym.make(ENV_ID, level=level, seed="test:0"), name)
        assert r.reward == 0.0 and r.info["policy_error"] is None, name


def test_the_reference_plays_the_stated_method_at_level_1_and_three_echoes_on_a_tight_budget():
    seen = {}
    for level in (1, 2):
        env = mrgym.make(ENV_ID, level=level, seed="test:0")
        env.reset()
        sc = policies.ToolScanner(env)
        scans, scan = [], sc.scan
        sc.scan = lambda protocol, rois=None: scans.append(dict(protocol)) or scan(protocol, rois)
        controls.reference(sc)
        seen[level] = scans
        env.close()
    m = task.METHOD
    assert seen[1] == [dict(sequence="se", matrix=m["matrix"], partial_fourier=m["partial_fourier"], te_ms=te, tr_ms=m["tr0_ms"] + 0.75 * te)
                       for te in m["te_ms"]]
    assert [p["te_ms"] for p in seen[2]] == list(controls.NARROW_TE_MS) and all(p["matrix"] == 48 for p in seen[2])
    tr0 = {round(p["tr_ms"] - 0.75 * p["te_ms"], 1) for p in seen[2]}
    assert len(tr0) == 1 and 500.0 <= tr0.pop() <= 700.0                 # the recovery is held, at the TR0 the budget pays


def test_the_blind_recipe_is_one_constant_inside_the_told_range():
    cfgs = [_cfg(2, s) for s in range(16)]
    recipe = controls.tune(cfgs)
    assert set(recipe) == {"t2_ms", "tuned_region_pass_rate"} and 40.0 <= recipe["t2_ms"] <= 200.0
    assert recipe["tuned_region_pass_rate"] < 0.35                       # one region in a few; three at once far below 5 %
