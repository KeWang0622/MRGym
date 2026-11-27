"""``Scanner/NoiseSNR-v0`` beyond its scoring table (``tests/scoring/noise_snr.py``): the statistics the family rests on,
checked against closed forms and against repeated scans; the instances; the control policies through the tools."""
import concurrent.futures
import json
import math

import numpy as np
import pytest

import mrgym
from mrgym import cards, coverage, policies
from mrgym.env import derive_seed
from mrgym.envs import family as F
from mrgym.envs.noise_snr import controls, task
from mrgym.scanner import console, kspace
from mrgym.scanner import world as W

ENV_ID = "Scanner/NoiseSNR-v0"


def _cfg(level, seed):
    return task.make_instance(derive_seed(ENV_ID, level, "train", seed, environ={}), level, task_id="t")


@pytest.fixture(scope="module")
def clean():
    """A level-2 instance with its hidden world and engine."""
    cfg = _cfg(2, 0)
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    return cfg, world, kspace.Engine(world)


@pytest.fixture(scope="module")
def ghosted():
    cfg = _cfg(3, 0)
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    return cfg, world, kspace.Engine(world)


def _images(cfg, engine, *seeds):
    """(realised protocol, noise-free magnitude, noisy magnitudes for the given noise seeds, pixel positions y and x)."""
    rp = console.plan(dict(cfg["task"]["protocol"]), engine.w["console"])["realised"]
    clean = np.abs(engine.acquire(rp, None)[0])
    noisy = [np.abs(engine.acquire(rp, np.random.default_rng(s))[0]) for s in seeds]
    half = rp["fov_mm"] / 2.0                                            # phase_fov is 1: a square field of view
    pos = (np.arange(clean.shape[0]) - clean.shape[0] // 2) * rp["fov_mm"] / clean.shape[0]
    assert abs(pos[0] + half) < 1e-9
    return rp, clean, noisy, pos


# ---------------------------------------------------------------------------------------------- the physics, independently
def test_the_two_constants_of_the_background_method_are_the_rayleigh_moments():
    """Closed form: the magnitude of complex Gaussian noise of SD 1 per component has mean sqrt(pi / 2) and
    SD sqrt(2 - pi / 2); a difference of two independent Gaussian values has SD sqrt(2). Checked on drawn numbers."""
    rng = np.random.default_rng(7)
    z = np.abs(rng.standard_normal(400_000) + 1j * rng.standard_normal(400_000))
    assert math.sqrt(math.pi / 2) == pytest.approx(1.2533, abs=5e-5) and math.sqrt(2 - math.pi / 2) == pytest.approx(0.6551, abs=5e-5)
    assert z.mean() == pytest.approx(math.sqrt(math.pi / 2), rel=4e-3) and z.std() == pytest.approx(math.sqrt(2 - math.pi / 2), rel=4e-3)
    assert controls.RAYLEIGH_SD == math.sqrt(2 - math.pi / 2)
    assert z.mean() / z.std() == pytest.approx(1.913, abs=0.01)
    a, b = rng.standard_normal(400_000), rng.standard_normal(400_000)
    assert (a - b).std() == pytest.approx(math.sqrt(2.0), rel=4e-3)


def test_the_engine_noise_is_what_two_acquisitions_and_the_background_measure(clean):
    """The verifier's noise SD (``Engine.noise_sd``) against two methods that do not use it: the SD of the difference
    of two scans over the phantom / sqrt(2), and the Rayleigh moments of the corners of one scan."""
    cfg, world, engine = clean
    rp, _, (a, b), pos = _images(cfg, engine, 11, 12)
    sd = engine.noise_sd(rp)
    r = np.hypot(pos[:, None], pos[None, :])
    inside = r <= 80.0                                                   # 20 000 pixels or more: the SD is good to 0.5 %
    assert (a - b)[inside].std(ddof=1) / math.sqrt(2.0) == pytest.approx(sd, rel=0.02)
    corner = (np.abs(pos[:, None]) > 100.0) & (np.abs(pos[None, :]) > 100.0)
    assert a[corner].std(ddof=1) / math.sqrt(2 - math.pi / 2) == pytest.approx(sd, rel=0.05)
    assert a[corner].mean() / math.sqrt(math.pi / 2) == pytest.approx(sd, rel=0.05)
    assert a[corner].mean() / a[corner].std(ddof=1) == pytest.approx(1.913, abs=0.1)
    # and its laws: sqrt(bandwidth), 1 / sqrt(averages)
    twice = console.plan(dict(cfg["task"]["protocol"], bandwidth_hz_per_px=2 * rp["bandwidth_hz_per_px"], te_ms="min"), world["console"])["realised"]
    more = console.plan(dict(cfg["task"]["protocol"], averages=4 * rp["averages"]), world["console"])["realised"]
    assert engine.noise_sd(twice) == pytest.approx(math.sqrt(2.0) * sd, rel=1e-9) and engine.noise_sd(more) == pytest.approx(sd / 2.0, rel=1e-9)


def test_the_pixel_scatter_inside_a_region_is_shading_not_noise(clean):
    cfg, _, engine = clean
    rp, noise_free, (a,), pos = _images(cfg, engine, 13)
    sd = engine.noise_sd(rp)
    ratios = []
    for reg in cfg["task"]["regions"]:
        m = (pos[:, None] - reg["y_mm"]) ** 2 + (pos[None, :] - reg["x_mm"]) ** 2 <= reg["radius_mm"] ** 2
        ratios.append(a[m].std(ddof=1) / sd)
        assert noise_free[m].std() > 0.5 * sd or ratios[-1] < 1.2        # the scatter of the noise-free region is the excess
    assert max(ratios) > 1.5                                             # in at least one region the "noise" is half shading


def test_at_level_3_the_ghost_fills_the_corners_and_cancels_in_the_difference(ghosted):
    cfg, _, engine = ghosted
    rp, _, (a, b), pos = _images(cfg, engine, 21, 22)
    sd = engine.noise_sd(rp)
    assert rp["fov_mm"] == 200.0 and cfg["world_spec"]["motion"][0]["kind"] == "pulse"
    c = rp["fov_mm"] / 2 - controls.CORNER_INSET_MM
    for sy in (-1, 1):
        for sx in (-1, 1):
            m = (pos[:, None] - sy * c) ** 2 + (pos[None, :] - sx * c) ** 2 <= controls.CORNER_RADIUS_MM ** 2
            assert a[m].std(ddof=1) / controls.RAYLEIGH_SD > 1.5 * sd     # no corner is clean air
    inside = np.hypot(pos[:, None], pos[None, :]) <= 75.0
    assert (a - b)[inside].std(ddof=1) / math.sqrt(2.0) == pytest.approx(sd, rel=0.02)


# ---------------------------------------------------------------------------------------------- instances
def test_the_family_is_registered_and_says_what_it_isolates():
    spec = mrgym.spec(ENV_ID)
    assert (spec.kind, spec.levels, spec.default_level, spec.family) == ("candidate", (1, 2, 3), 2, "mrgym.envs.noise_snr")
    assert (task.SUBJECT, task.KIND) == ("C", "measure") and isinstance(task.ISOLATES, str) and task.ISOLATES
    assert set(task.COMPETENCIES) <= set(coverage.ids()) and "qc.snr" in task.COMPETENCIES
    card = dict(task.card(), status="x", gate=None)
    assert cards.lint(card) == [] and card["kind"] == "candidate" and set(card["levels"]) == {1, 2, 3}
    assert [c["name"] for c in card["controls"]] == [controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and all("\n" not in d for d in controls.DESCRIPTIONS.values())
    assert F.outcomes(task)[-3:] == ("no_region_within", "some_regions_within", "pass")


def test_an_instance_is_a_function_of_the_seed_also_under_threads():
    jobs = [(lv, s) for lv in (1, 2, 3) for s in ("a", "b")]
    canon = lambda cfg: json.dumps(cfg, sort_keys=True)                  # noqa: E731
    task._instance.cache_clear()
    serial = [canon(_cfg(*j)) for j in jobs]
    task._instance.cache_clear()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(lambda j: canon(_cfg(*j)), jobs[::-1]))[::-1]
    assert threaded == serial and len(set(serial)) == len(serial)
    cached = _cfg(1, "a")
    cached["world_spec"]["acr_fields"]["rx_fault"] = 0.0                 # a caller that edits its copy ...
    cached["task"]["regions"][0]["radius_mm"] = 1.0
    assert canon(_cfg(1, "a")) == serial[0]                             # ... does not edit the next one


@pytest.mark.parametrize("level", [1, 2, 3])
def test_the_statement_states_the_protocol_and_hides_the_system(level):
    cfg = _cfg(level, 1)
    text, t = task.statement(cfg), cfg["task"]
    public = text + json.dumps(t) + json.dumps(task.submit_schema(cfg))
    assert "hidden" not in t and cfg["seed"] not in public
    spec = cfg["world_spec"]
    for secret in (spec["sigma0"], spec["acr_fields"]["rx_fault"], *cfg["hidden"]["snr"].values()):
        assert f"{secret:.4g}" not in public and f"{secret:.3g}" not in public
    for word in ("ghost", "instab", "artefact", "artifact", "Rayleigh", "0.655"):
        assert word not in text
    p = t["protocol"]
    for k in ("tr_ms", "te_ms", "fov_mm", "matrix", "bandwidth_hz_per_px", "averages"):
        assert f"{k} {p[k]:g}" in text
    assert f"pe_dir {p['pe_dir']}" in text and "within 10 % of" in text
    # the agent measures on its own scans of the protocol, and they are charged: the sentence about an uncharged scan is not here
    assert F.OWN_SCANS in text and F.NOT_CHARGED not in text and "a scan of this protocol too" in text
    assert ("How to do it" in text) is (level == 1) and ("sqrt(2)" in text) is (level == 1)
    plan = console.plan(dict(p), dict(b0_t=t["field_strength_t"], min_pixel_mm=0.64, b1rms_limit_ut=W.B1RMS_LIMIT_UT[t["field_strength_t"]]))
    assert plan["accept"] and plan["scan_s"] == t["protocol_scan_s"]
    assert cfg["budget_s"] == round(task.LEVELS[level]["scans"] * plan["scan_s"] + 1.0, 1)
    lo, hi = task.budget_range(level)
    assert lo <= cfg["budget_s"] <= hi
    assert (p["fov_mm"], "motion" in spec) == ((200.0, True) if level == 3 else (250.0, False))


def test_the_hidden_snr_is_the_verifiers_and_no_single_number_fits_three_regions():
    for level, seed in ((1, 0), (2, 1), (2, 2), (3, 0), (3, 3)):
        cfg = _cfg(level, seed)
        world = W.make_world(cfg["seed"], cfg["world_spec"])
        want = task.truth(kspace.Engine(world), task.protocol_of(cfg["task"]), cfg["task"]["regions"])["snr"]
        got = cfg["hidden"]["snr"]
        assert got == pytest.approx(want, rel=1e-4)
        values = sorted(got.values())
        assert task.SNR_RANGE[0] <= values[1] <= task.SNR_RANGE[1]
        assert values[2] / values[0] > (1 + task.TOLERANCE) / (1 - task.TOLERANCE)      # one number cannot pass all three
        for reg in cfg["task"]["regions"]:
            assert math.hypot(reg["y_mm"], reg["x_mm"]) + reg["radius_mm"] < task.DIAMETER_MM / 2 - 15.0


# ---------------------------------------------------------------------------------------------- the controls, through the tools
def _play(level, seed, name, recipe=None):
    r = policies.run(mrgym.make(ENV_ID, level=level, seed=seed), name, recipe=recipe)
    assert r.info["policy_error"] is None, r.info["policy_error"]
    return r


def test_the_reference_passes_and_every_shortcut_fails_at_level_2():
    ref = _play(2, 0, "reference")
    assert ref.reward == 1.0 and ref.info["score"]["error"] < task.TOLERANCE and ref.info["replay"]["ok"] and ref.info["replay"]["scans"] == 2
    for name in controls.MUST_FAIL:
        r = _play(2, 0, name)
        assert r.reward == 0.0, name
        if name != "prior":
            assert r.info["score"]["checks"]["measured"] is True and r.info["score"]["outcome"] in task.OUTCOMES, name
    assert _play(2, 0, "prior").info["score"]["outcome"] == "no_measurement"
    assert _play(2, 0, "background").reward == 1.0 and _play(2, 0, "open_loop").reward == 1.0
    lost = _play(2, 0, "no_sqrt2").info["score"]
    assert lost["passed"] is False and lost["error"] == pytest.approx(1.0 - 1.0 / math.sqrt(2.0), abs=0.05)      # 29 % low in every region
    assert _play(2, 0, "neighbours").reward in (0.0, 1.0)               # reported only: it plays through the tools


def test_at_level_3_two_acquisitions_pass_within_the_budget_and_the_background_does_not():
    ref = _play(3, 0, "reference")
    cfg_budget = ref.info["budget"]
    assert ref.reward == 1.0 and cfg_budget["scans"] == 2 and cfg_budget["remaining_s"] == pytest.approx(1.0, abs=0.05)
    for name in ("background", "no_rayleigh"):
        r = _play(3, 0, name)
        assert r.reward == 0.0 and r.info["score"]["outcome"] == "no_region_within", name


def test_the_reference_passes_at_level_1_and_tuning_gives_a_deliverable():
    assert _play(1, 1, "reference").reward == 1.0
    cfgs = [_cfg(2, f"tune{i}") for i in range(4)]
    recipe = controls.tune(cfgs)
    assert recipe["kind"] in ("snr", "snr_over_relative_snr") and len(recipe["values"]) == 3 and 0.0 <= recipe["tuned_pass_rate"] <= 1.0
    r = _play(2, 0, "fixed_recipe", recipe)
    assert r.reward == 0.0 and r.info["submitted"] is True


def test_a_deliverable_in_another_shape_is_refused_and_the_episode_continues():
    env = mrgym.make(ENV_ID, level=2, seed=0)
    env.reset()
    for bad, why in ((dict(snr=dict(R1=100.0, R2=100.0)), "snr.R3: required"), (dict(snr=dict(R1=100.0, R2=100.0, R3=-3.0)), "snr.R3 must be a number from 0.01")):
        obs, reward, terminated, truncated, _ = env.step({"name": "submit", "arguments": bad})
        assert obs.is_error and not terminated and not truncated and reward == 0.0 and why in obs.text
    obs, reward, terminated, _, info = env.step({"name": "submit", "arguments": dict(snr=dict(R1=100.0, R2=100.0, R3=100.0))})
    assert terminated and reward == 0.0 and info["score"]["outcome"] == "no_measurement"
    env.close()
