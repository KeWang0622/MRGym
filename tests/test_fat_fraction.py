"""``Scanner/FatFraction-v0``: the physics behind the task, written a second time; the instances; the verifier; the controls.

The second calculation is ``mixture`` below: the echoes of a voxel that holds water and fat, from the six peaks typed
here, the Ernst formula and two exponentials. It shares no code with the engine (which serves an EPG steady state
integrated over the slice profile, a time function per tissue class compressed by an SVD, and an image) and none with
the task or the policies. Three things are checked against it: the engine's echoes, the three errors of the textbook
formula the task is about, and the estimators of the policies.
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
from mrgym.envs.fat_fraction import controls, task
from mrgym.scanner import console, kspace, tissue
from mrgym.scanner import world as W

ENV_ID = "Scanner/FatFraction-v0"
# The fat spectrum of the scanner: ppm relative to water, share. Typed here from the source the engine cites since 0.3.0
# (docs/physics/references.md: the six-peak model listed by Hernando 2017, without its 0.1 ppm room-temperature shift;
# engine 0.2.0 carried 0.693 for the main peak and named Hamilton 2011).
PEAKS = ((-3.80, 0.087), (-3.40, 0.694), (-2.60, 0.128), (-1.94, 0.004), (-0.39, 0.039), (0.60, 0.048))
HZ_PER_PPM = 42.577478518 * 1.5             # at 1.5 T


# ---------------------------------------------------------------------------------------------- the second calculation
def fat_vector(t_ms):
    total = sum(a for _, a in PEAKS)
    return sum(a * np.exp(2j * np.pi * ppm * HZ_PER_PPM * np.asarray(t_ms, float) * 1e-3) for ppm, a in PEAKS) / total


def ernst(flip_deg, tr_ms, t1_ms):
    e, a = math.exp(-tr_ms / t1_ms), math.radians(flip_deg)
    return math.sin(a) * (1 - e) / (1 - e * math.cos(a))


def mixture(t_ms, water, fat, flip_deg=3.0, tr_ms=60.0, psi_hz=0.0, phase=0.0):
    """Complex echoes of a voxel: ``water`` and ``fat`` are dicts of amount (share x proton density), t1_ms, t2s_ms."""
    t = np.asarray(t_ms, float)
    w = water["amount"] * ernst(flip_deg, tr_ms, water["t1_ms"]) * np.exp(-t / water["t2s_ms"])
    f = fat["amount"] * ernst(flip_deg, tr_ms, fat["t1_ms"]) * np.exp(-t / fat["t2s_ms"]) * fat_vector(t)
    return (w + f) * np.exp(1j * (phase + 2 * np.pi * psi_hz * t * 1e-3))


def _voxel(pdff, t1_water=900.0, t2s_water=30.0, t2s_fat=28.0):
    return dict(amount=1.0 - pdff, t1_ms=t1_water, t2s_ms=t2s_water), dict(amount=pdff, t1_ms=340.0, t2s_ms=t2s_fat)


TE = 2.3 + 2.3 * np.arange(6)               # the echo times of the stated scan


def test_the_fat_vector_of_the_statement_is_the_six_peaks():
    assert tuple(tissue.FAT_PEAKS) == PEAKS and task.B0_T == 1.5
    t = np.linspace(0.0, 20.0, 81)
    assert np.allclose(task.spectrum(t), fat_vector(t), atol=1e-12) and task.spectrum(0.0) == pytest.approx(1.0)
    c1, c2, c4 = (fat_vector(v) for v in (2.3, 4.6, 9.2))
    assert c1.real == pytest.approx(-0.793, abs=2e-3) and c2.real == pytest.approx(0.817, abs=2e-3) and c4.real == pytest.approx(0.556, abs=2e-3)
    assert abs(c1.imag) < 0.1 and abs(c2.imag) < 0.1                     # nearly opposed, nearly in phase; never -1 or +1
    text = task.method_text()
    assert f"c1 = {c1.real:+.3f} at t1, c2 = {c2.real:+.3f} at t2 and c4 = {c4.real:+.3f} at t4" in text
    assert 1.0 / (3.4 * HZ_PER_PPM) * 1e3 == pytest.approx(4.605, abs=2e-3)     # the main peak is back in phase after 4.6 ms


def test_the_textbook_formula_is_wrong_three_times_over():
    """On the second calculation alone, a disc of 30 % fat: each line of the task's docstring, with its sign."""
    two_point = lambda s: 100.0 * (abs(s[1]) - abs(s[0])) / (2.0 * abs(s[1]))      # noqa: E731
    water, fat = _voxel(0.30, t2s_water=1e9, t2s_fat=1e9)
    one_line = (1 - 0.30) + 0.30 * np.exp(-2j * np.pi * 3.4 * HZ_PER_PPM * TE[:2] * 1e-3)
    assert two_point(one_line) == pytest.approx(30.0, abs=0.05)          # one fat line, no decay, no T1: the formula is exact
    no_t1 = dict(flip_deg=0.01, tr_ms=5000.0)
    spectrum_only = two_point(mixture(TE[:2], water, fat, **no_t1))
    by_hand = 100.0 * 0.30 * (0.817 + 0.793) / (2.0 * (0.70 + 0.30 * 0.817))        # the fat vector is +0.817 and -0.793, not +1 and -1
    assert spectrum_only == pytest.approx(by_hand, abs=0.1) and by_hand == pytest.approx(30.0 - 4.4, abs=0.1)    # six peaks: low
    water, fat = _voxel(0.30)
    with_decay = two_point(mixture(TE[:2], water, fat, **no_t1))
    assert with_decay < spectrum_only - 1.5                              # T2* of 30 ms between the echoes: lower
    high_flip = two_point(mixture(TE[:2], water, fat, flip_deg=20.0, tr_ms=60.0))
    low_flip = two_point(mixture(TE[:2], water, fat, flip_deg=3.0, tr_ms=60.0))
    assert high_flip > with_decay + 4.0 and abs(low_flip - with_decay) < 0.6      # T1 at 20 degrees: high; at 3 degrees: out
    # the offset at 20 degrees depends on the hidden T1 of the water: no flip angle cancels the errors on every disc
    short, long = (two_point(mixture(TE[:2], *_voxel(0.30, t1_water=t1), flip_deg=20.0)) for t1 in (350.0, 2500.0))
    assert short < 30.0 - 3.0 and long > 30.0 + 3.0


def test_the_estimators_recover_the_fraction_of_the_second_calculation():
    worst = dict(stated=0.0, complex=0.0)
    for pdff in (0.05, 0.15, 0.30, 0.45):
        for t1, t2s in ((400.0, 12.0), (1200.0, 30.0), (2800.0, 60.0)):
            s = mixture(TE, *_voxel(pdff, t1, t2s, 0.9 * t2s), psi_hz=-47.0, phase=1.1)
            worst["stated"] = max(worst["stated"], abs(controls.stated_pdff(TE, np.abs(s)) - 100 * pdff))
            worst["complex"] = max(worst["complex"], abs(controls.fit_complex(TE, s, 100.0)["pdff"] - 100 * pdff))
    # inside the tolerance of 3 points; the largest part is the T1 weighting a flip angle of 3 degrees leaves for the
    # longest water T1 (the second calculation has one flip angle per voxel; the 2D slice of the engine weights less)
    assert worst["stated"] < 2.0 and worst["complex"] < 2.0
    for pdff in (0.60, 0.75, 0.92):                                      # fat is the larger part
        s = mixture(TE, *_voxel(pdff), psi_hz=35.0, phase=-2.0)
        fit = controls.fit_complex(TE, s, 100.0)
        assert fit["pdff"] == pytest.approx(100 * pdff, abs=1.5) and fit["psi_hz"] == pytest.approx(35.0, abs=3.0)
        assert controls.stated_pdff(TE, np.abs(s)) < 100 * pdff - 5.0    # magnitudes with fat taken as the smaller part: far too low
    s = mixture(TE, *_voxel(0.30), psi_hz=20.0)
    assert controls.two_point_pdff(np.abs(s)) < 30.0 - 4.0               # the textbook formula on the same echoes
    assert controls.fit_complex(TE, s, 100.0, single_peak=True)["pdff"] < 30.0 - 3.0


def _scan(protocol):
    """One level-3 world scanned noise-free: (cfg, echo times, {disc: complex echoes}, world, engine, realised protocol)."""
    cfg = task.make_instance(derive_seed(ENV_ID, 3, "train", 2, environ={}), 3, task_id="t")
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    engine = kspace.Engine(world)
    rp = console.plan(dict(protocol), world["console"])["realised"]
    rows = F.noise_free(engine, rp, cfg["task"]["regions"])[1]
    echoes = {k: np.array(r["complex_mean_magnitude"]) * np.exp(1j * np.array(r["complex_mean_phase_rad"])) for k, r in rows.items()}
    return cfg, np.array(rp["echo_times_ms"]), echoes, world, engine, rp


@pytest.fixture(scope="module")
def scanned():
    """The stated scan (``controls.SCAN``: partial Fourier 0.75): (cfg, echo times, {disc: complex echoes}, world, engine)."""
    return _scan(controls.SCAN)[:5]


def _voxels(cfg):
    """Per region the water and fat parts of its disc as ``mixture`` takes them."""
    inserts = cfg["world_spec"]["qa"]["inserts"]
    out = {}
    for k, r in enumerate(cfg["task"]["regions"]):
        parts = []
        for ins in inserts[2 * k:2 * k + 2]:
            m = ins["tissue"]
            parts.append(dict(amount=ins["weight"] * m["pd"], t1_ms=m["t1_ms"], t2s_ms=1.0 / (1.0 / m["t2_ms"] + 1.0 / m["t2p_ms"])))
        out[r["id"]] = parts
    return out


def test_the_engine_serves_the_echoes_of_the_second_calculation(scanned):
    """The physics of the echoes, on the stated scan with every line acquired: beat and decay of the six echoes."""
    cfg, te, echoes, world, _, rp = _scan(dict(controls.SCAN, partial_fourier=1.0))
    assert np.allclose(te, TE) and rp["fat_sat"] is False and rp["lines"] == rp["lines_full"]      # no saturation pulse, nothing synthesised
    for i, parts in _voxels(cfg).items():
        want, got = np.abs(mixture(te, *parts)), np.abs(echoes[i])
        # the beat and the decay of the six echoes, to 2 % of the first in-phase echo (an opposed-phase echo is the small
        # difference of water and fat, and the 2D slice of the engine weights T1 a little less than the one flip angle here)
        assert np.allclose(got / got[1], want / want[1], rtol=0.0, atol=0.02)
        # ... and the phase the fit has to see through: receive phase + off-resonance x echo time + the fat vector
        turn = np.angle(echoes[i][3] / echoes[i][1]) - np.angle(mixture(te, *parts)[3] / mixture(te, *parts)[1])
        psi = turn / (2 * np.pi * (te[3] - te[1]) * 1e-3)
        assert abs(psi) < task.OFF_RESONANCE_MAX_HZ


def test_what_partial_fourier_of_the_stated_scan_does_to_the_echoes(scanned):
    """The stated scan acquires 0.75 of the lines and the console synthesises the rest by homodyne detection (engine
    0.3.0; docs/physics/limits.md, block 5a): the image takes its phase from a low-resolution image. The in-phase
    echoes are those of the second calculation as before. An opposed-phase echo in which water and fat nearly cancel
    reads lower, because the low-resolution phase inside such a disc is partly that of the gel around it: up to 5 % of
    the in-phase echo in this world, whose discs D1 and D3 lie near the cancellation. The fit of the reference still
    finds every fraction (the next test, on these echoes); the tolerance of the task is not touched by this."""
    cfg, te, echoes, world, _ = scanned
    assert console.plan(dict(controls.SCAN), world["console"])["realised"]["lines"] < console.plan(dict(controls.SCAN), world["console"])["realised"]["lines_full"]
    lowest, gap = 1.0, 0.0
    for i, parts in _voxels(cfg).items():
        want, got = np.abs(mixture(te, *parts)), np.abs(echoes[i])
        want, got = want / want[1], got / got[1]
        assert np.allclose(got[1::2], want[1::2], rtol=0.0, atol=0.02)   # echoes 2, 4, 6: fat in phase with water
        assert np.allclose(got[0::2], want[0::2], rtol=0.0, atol=0.05)   # echoes 1, 3, 5: fat opposed
        lowest, gap = min(lowest, float(want[0::2].min())), max(gap, float(np.abs(got - want)[0::2].max()))
    assert lowest < 0.05 and 0.02 < gap                                   # the world has such an echo, and there homodyne detection shows


def test_the_fit_recovers_the_hidden_fractions_on_noise_free_scans_of_the_engine(scanned):
    cfg, te, echoes, _, _ = scanned
    truth = task.true_pdff_percent(cfg)
    assert max(truth.values()) > 50.0 > min(truth.values())             # this world has fat on both sides of one half
    for i, v in truth.items():
        assert controls.fit_complex(te, echoes[i], 100.0)["pdff"] == pytest.approx(v, abs=1.5)
    fat_larger = [i for i, v in truth.items() if v > 62.0]
    assert fat_larger and all(controls.stated_pdff(te, np.abs(echoes[i])) < truth[i] - 5.0 for i in fat_larger)


# ---------------------------------------------------------------------------------------------- instances
def _cfg(level, seed):
    return task.make_instance(derive_seed(ENV_ID, level, "train", seed, environ={}), level, task_id="t")


def test_registration_and_the_constants_of_the_map():
    spec = mrgym.spec(ENV_ID)
    assert (spec.kind, spec.family, tuple(spec.levels), spec.default_level) == ("candidate", "mrgym.envs.fat_fraction", (1, 2, 3), 2)
    assert (task.SUBJECT, task.KIND) == ("A", "measure") and task.ISOLATES.startswith("water-fat chemical shift")
    assert set(task.COMPETENCIES) <= set(coverage.ids())
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and {"prior", "fixed_recipe", "wrong_units", "two_point"} <= set(controls.MUST_FAIL)
    assert task.card()["id"] == ENV_ID and [c["name"] for c in task.card()["controls"]][0] == "reference"
    assert "fat_sat" not in controls.SCAN and controls.SCAN == task.METHOD      # the reference never saturates fat


def test_an_instance_is_a_function_of_seed_and_level_also_under_threads():
    jobs = [(lv, s) for lv in task.LEVELS for s in range(4)]
    serial = {j: json.dumps(_cfg(*j), sort_keys=True) for j in jobs}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for _ in range(2):
            assert dict(zip(jobs, pool.map(lambda j: json.dumps(_cfg(*j), sort_keys=True), jobs))) == serial
    assert len(set(serial.values())) == len(jobs)


def test_four_mixture_discs_inside_the_told_bounds():
    for level, lv in task.LEVELS.items():
        seen = []
        for seed in range(12):
            cfg = _cfg(level, seed)
            t, inserts = cfg["task"], cfg["world_spec"]["qa"]["inserts"]
            assert [r["id"] for r in t["regions"]] == ["D1", "D2", "D3", "D4"] and len(inserts) == 8
            assert t["pdff_range_percent"] == [100 * lv["pdff"][0], 100 * lv["pdff"][1]] and (cfg["budget_s"], t["tolerance_points"]) == (lv["budget_s"], 3.0)
            truth = task.true_pdff_percent(cfg)
            for k, r in enumerate(t["regions"]):
                water, fat = inserts[2 * k], inserts[2 * k + 1]
                assert set(r) == {"id", "y_mm", "x_mm", "radius_mm"} and r["radius_mm"] == task.ROI_MM < water["radius_mm"] == fat["radius_mm"]
                assert water["weight"] + fat["weight"] == pytest.approx(1.0, abs=1e-9) and fat["tissue"]["fat"] and not water["tissue"]["fat"]
                assert water["tissue"]["t2p_ms"] == fat["tissue"]["t2p_ms"]              # one field spread per voxel
                want = 100 * fat["weight"] * fat["tissue"]["pd"] / (fat["weight"] * fat["tissue"]["pd"] + water["weight"] * water["tissue"]["pd"])
                assert truth[r["id"]] == pytest.approx(want, abs=1e-9) and t["pdff_range_percent"][0] < want < t["pdff_range_percent"][1]
                others = [q for q in t["regions"] if q is not r]
                assert min(math.hypot(r["y_mm"] - q["y_mm"], r["x_mm"] - q["x_mm"]) for q in others) > 2 * task.DISC_MM + 10.0   # discs apart
                assert math.hypot(r["y_mm"], r["x_mm"]) + task.DISC_MM < 80.0            # inside the gel (radius 89 mm or more, off centre by at most 3.6 mm)
            seen += list(truth.values())
        assert (max(seen) > 50.0) == (level == 3) and max(seen) - min(seen) > 30.0


def test_the_world_is_the_mixture_and_the_field_stays_inside_the_told_bound(scanned):
    cfg, _, _, world, engine = scanned
    assert world["names"] == ["body"] + [f"insert{k}" for k in range(8)] and world["regions"] == []
    assert float(world["weights"].sum(0).max()) == pytest.approx(1.0, abs=1e-5)          # a mixture fills its disc once
    n, d = world["n"], world["extent_mm"] / world["n"]
    y, x = np.meshgrid((np.arange(n) - n // 2) * d, (np.arange(n) - n // 2) * d, indexing="ij")
    for r in cfg["task"]["regions"]:
        inside = np.hypot(y - r["y_mm"], x - r["x_mm"]) <= r["radius_mm"]
        assert float(np.abs(engine.df[inside]).max()) < task.OFF_RESONANCE_MAX_HZ
    # the largest field the level specs can draw at a disc: centre-frequency error + linear terms + saddle, in Hz
    reach = (task.RING_MM[1] + task.ROI_MM) / 240.0
    worst = task.BASE["cf_offset_hz"] * task.B0_T / 3.0 + task.BASE["b0_hz"] * (2 * reach + (reach + 0.1) ** 2 / 0.09)
    assert worst < task.OFF_RESONANCE_MAX_HZ


def test_the_statement_is_written_from_the_public_task_alone():
    for level in task.LEVELS:
        cfg = _cfg(level, 1)
        text = task.statement(cfg)
        assert task.statement(dict(task=cfg["task"])) == text            # nothing hidden is an input of the text
        prose = "\n".join(line for line in text.splitlines() if not line.startswith("  D"))    # without the table of circles
        for v in task.true_pdff_percent(cfg).values():
            assert f"{v:.1f}" not in prose and f"{v:.2f}" not in prose
        for ins in cfg["world_spec"]["qa"]["inserts"]:
            assert f"{ins['tissue']['t1_ms']:g}" not in prose and f"{ins['weight']:g}" not in prose
        assert ("How to do it" in text) == (level == 1) and ("hints" in cfg["task"]) == (level == 1)
        lo, hi = cfg["task"]["pdff_range_percent"]
        assert f"The fat fractions lie between {lo:g} and {hi:g} percent." in text and "within 3 percentage points" in text
        assert "-3.80 ppm 8.7 %, -3.40 ppm 69.4 %, -2.60 ppm 12.8 %, -1.94 ppm 0.4 %, -0.39 ppm 3.9 %, +0.60 ppm 4.8 %" in text      # the table of engine 0.3.0
        assert "At 1.5 T, 1 ppm is 63.87 Hz." in text and "below 80 Hz" in text and "An answer in the wrong unit is wrong." in text
    method = task.method_text()
    assert "flip_deg 3, tr_ms 60, n_echoes 6, te_ms 2.3, echo_spacing_ms 2.3, matrix 64" in method and "averages 8" in method


# ---------------------------------------------------------------------------------------------- the verifier
def test_the_pass_window_is_three_points_around_the_true_fraction():
    cfg = _cfg(2, 0)
    truth = task.true_pdff_percent(cfg)
    at = lambda d=0.0, **one: {"pdff_percent": {i: v + one.get(i, d) for i, v in truth.items()}}      # noqa: E731
    for answer in (at(), at(2.99), at(-2.99)):
        s = task.score(cfg, answer)
        assert s["passed"] and s["diagnostic"] == "" and set(s["checks"]) == {"deliverable_valid", *(f"pdff_within_tolerance_{i}" for i in truth)}
    s = task.score(cfg, at(D2=3.01))
    assert not s["passed"] and [k for k, v in s["checks"].items() if not v] == ["pdff_within_tolerance_D2"]
    assert s["diagnostic"] == "D2 is off by +3.0 points" and task.outcome(s) == "some_discs_within" and s["error"] == pytest.approx(3.01)
    s = task.score(cfg, at(-3.5))
    assert not s["passed"] and task.outcome(s) == "no_disc_within" and s["distance"] == dict(regions=4, within=0, worst_points=s["error"])
    as_fraction = {"pdff_percent": {i: v / 100.0 for i, v in truth.items()}}              # the right numbers in the wrong unit
    assert task.check(cfg, as_fraction, {}) is None and not task.score(cfg, as_fraction)["passed"]


def test_deliverables_the_console_cannot_evaluate_are_refused():
    cfg = _cfg(1, 0)
    good = {"pdff_percent": {"D1": 10.0, "D2": 20.0, "D3": 30.0, "D4": 40.0}}
    assert task.check(cfg, good, {}) is None
    shape = 'the deliverable is {"pdff_percent": {"D1": <number>, "D2": <number>, "D3": <number>, "D4": <number>}}'
    for bad in (None, [], {"pdff": good["pdff_percent"]}, {"pdff_percent": {"D1": 10.0}}, {"pdff_percent": dict(good["pdff_percent"], D5=1.0)}):
        assert task.check(cfg, bad, {}) == shape
    with_d1 = lambda v: {"pdff_percent": dict(good["pdff_percent"], D1=v)}                # noqa: E731
    assert task.check(cfg, with_d1("10"), {}) == task.check(cfg, with_d1(True), {}) == task.check(cfg, with_d1(float("inf")), {}) \
        == "the fat fraction of D1 is not a finite number"
    assert task.check(cfg, with_d1(-1.0), {}) == "the fat fraction of D1 is -1: a fraction in percent lies between 0 and 100"
    assert "between 0 and 100" in task.check(cfg, with_d1(120.0), {})
    s = task.score(cfg, with_d1(120.0))
    assert s["checks"] == dict(deliverable_valid=False) and not s["passed"] and s["error"] is None
    env = mrgym.make(ENV_ID, level=1, seed=0)
    env.reset()
    step = env.step({"name": "submit", "arguments": with_d1(120.0)})
    assert step.observation.is_error and not step.done                   # refused; the episode continues
    env.close()


# ---------------------------------------------------------------------------------------------- the controls
@pytest.mark.parametrize("level", sorted(task.LEVELS))
def test_the_reference_passes_and_every_shortcut_fails(level):
    seed = "test:0"
    r = policies.run(mrgym.make(ENV_ID, level=level, seed=seed), controls.REFERENCE)
    assert r.reward == 1.0 and r.info["policy_error"] is None and r.info["budget"]["used_s"] <= task.LEVELS[level]["budget_s"]
    for name in controls.MUST_FAIL:
        r = policies.run(mrgym.make(ENV_ID, level=level, seed=seed), name)
        assert r.reward == 0.0 and r.info["policy_error"] is None, name


def test_the_reference_plays_the_stated_scan_and_the_stated_arithmetic_only_where_it_is_stated():
    answers = {}
    for level in (1, 3):
        env = mrgym.make(ENV_ID, level=level, seed="test:1")
        env.reset()
        sc = policies.ToolScanner(env)
        scans, scan = [], sc.scan
        sc.scan = lambda protocol, rois=None: (scans.append((dict(protocol), scan(protocol, rois))), scans[-1][1])[1]
        answers[level] = controls.reference(sc)
        assert [p for p, _ in scans] == [task.METHOD]
        rows = {r["id"]: r for r in scans[0][1]["roi_readouts"]}
        te = scans[0][1]["realised"]["echo_times_ms"]
        by_hand = {i: round(controls.stated_pdff(te, rows[i]["mean_magnitude"]), 2) for i in rows}
        assert (answers[level]["pdff_percent"] == by_hand) == (level == 1)
        env.close()


def test_the_blind_recipe_is_one_constant_inside_the_told_range():
    recipe = controls.tune([_cfg(2, s) for s in range(16)])
    assert set(recipe) == {"pdff_percent", "tuned_region_pass_rate"} and 5.0 <= recipe["pdff_percent"] <= 45.0
    assert recipe["tuned_region_pass_rate"] < 0.35                       # one disc in a few; four at once far below 5 %
