"""``Scanner/EPISetup-v0``: the EPI physics the verifier relies on against closed forms and second methods, the measures
of the verifier, the oracle, the pass rule, the instances and the controls."""
import concurrent.futures
import json
import math

import numpy as np
import pytest

import mrgym
from mrgym import gate, policies
from mrgym.envs import family as F
from mrgym.envs import oracle as O
from mrgym.envs.epi_setup import controls, oracle, task
from mrgym.envs.epi_setup import metrics as M
from mrgym.scanner import console, kspace, tissue
from mrgym.scanner import session as core
from mrgym.scanner import world as W

ENV = "Scanner/EPISetup-v0"
EPI = dict(oracle.SCAN, matrix=64, bandwidth_hz_per_px=1000.0)


def _disc(cy, cx, r):
    return lambda y, x: np.hypot(y - cy, x - cx) - r


def _flat(shapes, **system):
    """A hand-made world: discs of tissue in perfect fields, with the given system state."""
    return W.flat_world(W.shapes_phantom(shapes, b0_t=1.5), **system)


def _plan(world, **fields):
    pl = console.plan(dict(EPI, **fields), world["console"])
    assert pl["accept"], pl["violations"]
    return pl["realised"]


def _centroid(img, rp):
    """Centre of the power of an image along the phase-encode axis, mm (the power: the tails of the point spread, which
    reach across the field of view, weigh nothing)."""
    power = np.abs(img) ** 2
    y, x = core.pixel_positions(img.shape, core.fov_yx(rp), core.centre_yx(rp))
    return float((power.sum(1) * y).sum() / power.sum()) if rp["pe_dir"] == "y" else float((power.sum(0) * x).sum() / power.sum())


# ------------------------------------------------------------------ physics, checked independently
@pytest.mark.parametrize("fields", [dict(), dict(pe_dir="x", pe_polarity=-1, bandwidth_hz_per_px=1800.0, phase_fov=0.75)], ids=["y+", "x-"])
def test_off_resonance_displaces_a_disc_by_frequency_times_echo_spacing_times_phase_fov(fields):
    """Textbook: an EPI displaces a spin df off resonance by df x echo spacing x phase field of view along the
    phase-encode axis (df / bandwidth per pixel in the phase direction, in pixels). Second method: the centre of the
    image of a uniformly off-resonant disc against the image of the engine without the displacement."""
    df = 18.0
    world = _flat([("gm", _disc(-20.0, 30.0, 15.0))], cf_offset_hz=df)
    rp = _plan(world, **fields)
    scans = M.Scans(world)
    moved = _centroid(scans.image(rp, "ghost"), rp) - _centroid(scans.image(rp, "ghost", "b0shift"), rp)
    law = -rp["pe_polarity"] * df * rp["echo_spacing_ms"] * 1e-3 * rp["fov_pe_mm"]
    assert abs(law) > 2.0 and moved == pytest.approx(law, abs=0.05)                         # 2 to 5 mm, to 0.05 mm
    assert M.displacement_mm(scans, rp, "gm") == pytest.approx(law, abs=0.03)               # the verifier's measure
    assert float(oracle.displacement_mm(df, rp)) == pytest.approx(law, rel=1e-12)           # the closed form of the search


def test_a_gradient_delay_makes_a_ghost_half_a_field_of_view_away_of_size_tan_phi():
    """Textbook: odd and even lines shifted against each other by a delay give a copy half a phase field of view away;
    with phi = 2 pi x bandwidth x matrix x delay x readout coordinate / fov the image holds cos(phi) of the object in
    place and sin(phi) of it in the copy. Second method: the two discs read off the magnitude image."""
    delay_us = 0.9
    world = _flat([("gm", _disc(-20.0, 40.0, 10.0))], grad_delay_us=delay_us)
    rp = _plan(world, bandwidth_hz_per_px=1500.0)
    scans = M.Scans(world)
    img = np.abs(scans.image(rp))
    parent = dict(id="p", y_mm=-20.0, x_mm=40.0, radius_mm=8.0)
    copy = dict(parent, id="g", y_mm=-20.0 + rp["fov_pe_mm"] / 2)
    ratio = img[M.mask(rp, copy)].sum() / img[M.mask(rp, parent)].sum()
    phi = 2 * np.pi * rp["bandwidth_hz_per_px"] * rp["matrix"] * delay_us * 1e-6 * 40.0 / rp["fov_mm"]
    assert 0.08 < math.tan(phi) < 0.2 and ratio == pytest.approx(math.tan(phi), rel=0.03)
    assert M.ghost(scans, rp, parent) == pytest.approx(math.sin(phi), rel=0.03)             # the verifier's measure: against the scan without delay
    assert oracle.ghost_of(parent, delay_us * 1e-6, rp) == pytest.approx(M.ghost(scans, rp, parent), rel=0.01)
    turned = _plan(world, bandwidth_hz_per_px=1500.0, pe_dir="x")                           # the readout now runs along y: the coordinate is -20 mm
    assert M.ghost(scans, turned, parent) == pytest.approx(math.sin(phi * 20.0 / 40.0), rel=0.04)


def test_fat_lands_where_its_main_line_times_echo_spacing_times_phase_fov_says():
    """Textbook: fat (3.4 ppm below water, 217 Hz at 1.5 T) is displaced along the phase-encode axis like any
    off-resonant spin; the blip polarity mirrors it. Second method: the brightest row of the image of a fat disc."""
    world = _flat([("scalp_fat", _disc(0.0, 0.0, 10.0))])
    hz = tissue.FAT_PPM * kspace.GAMMA_MHZ * 1.5
    assert hz == pytest.approx(-217.1, abs=0.1)
    for pol in (1, -1):
        rp = _plan(world, pe_polarity=pol)
        img = np.abs(M.Scans(world).image(rp))
        y, _ = core.pixel_positions(img.shape, core.fov_yx(rp))
        law = float(oracle.displacement_mm(hz, rp))
        assert abs(law) > 35.0 and np.sign(law) == pol
        assert y[int(np.argmax(img.sum(1)))] == pytest.approx(law, abs=rp["fov_pe_mm"] / rp["matrix_pe"])      # to one pixel
    f = dict(fat=[dict(y_mm=0.0, x_mm=0.0, radius_mm=10.0)], fat_hz=[hz], regions=[dict(y_mm=law, x_mm=0.0, radius_mm=6.0)])
    assert oracle.fat_clearance_mm(f, rp) == pytest.approx(-16.0, abs=1e-6)                 # the displaced disc sits on a region placed there
    assert oracle.fat_clearance_mm(dict(f, regions=[dict(y_mm=-law, x_mm=0.0, radius_mm=6.0)]), rp) > 50.0


def test_wrap_is_zero_for_an_object_inside_the_phase_field_of_view_and_large_beyond_it():
    world = _flat([("gm", _disc(0.0, 0.0, 70.0))], cf_offset_hz=10.0, grad_delay_us=0.5)
    scans = M.Scans(world)
    assert M.wrap(scans, _plan(world)) < 1e-4 and M.wrap(scans, _plan(world, phase_fov=0.75)) < 1e-4       # 240 and 180 mm hold 140 mm
    assert M.wrap(scans, _plan(world, phase_fov=0.5)) > 10 * oracle.WRAP_LIMIT                              # 120 mm do not
    twin = M.wrap_twin(_plan(world, phase_fov=0.5))
    assert twin["lines_full"] * 3.75 >= M.TWIN_FOV_MM and twin["lines_full"] * twin["echo_spacing_ms"] == pytest.approx(32 * _plan(world, phase_fov=0.5)["echo_spacing_ms"])


def test_the_noise_of_an_epi_pixel_is_the_analytic_one_and_follows_the_law_of_the_policies():
    world = _flat([("gm", _disc(0.0, 0.0, 40.0))], sigma0=0.2)
    engine = kspace.Engine(world)
    rp = _plan(world, phase_fov=0.75, averages=2)
    rng = np.random.default_rng(5)
    scans = np.array([engine.acquire(rp, rng)[0] for _ in range(16)])
    sd = float(np.sqrt(0.5 * (scans.real.var(0, ddof=1).mean() + scans.imag.var(0, ddof=1).mean())))
    assert sd == pytest.approx(engine.noise_sd(rp), rel=0.03)
    assert 0.2 * controls.noise_factor(rp) == pytest.approx(engine.noise_sd(rp), rel=1e-9)


# ------------------------------------------------------------------ one instance: measures, closed forms, oracle
@pytest.fixture(scope="module")
def instance():
    """One level-3 instance with its hidden world, the twins and the optimum (verifier side)."""
    env = mrgym.make(ENV, level=3, seed="test:3")
    env.reset()
    cfg, world = env.episode_config(), env.hidden_world()
    scans = M.Scans(world, env._session.engine)
    return cfg, world, scans, oracle.least(cfg, world, scans)


def test_the_closed_forms_of_the_search_agree_with_the_measures_of_the_verifier(instance):
    cfg, world, scans, best = instance
    t, f, m = cfg["task"], oracle.facts(world, cfg["task"]), best["measured"]
    rp = m["realised"]
    for r in f["regions"]:
        assert abs(float(oracle.displacement_mm(r["df_hz"], rp))) == pytest.approx(m["displacement_mm"][r["id"]], rel=0.02, abs=0.02)
        assert oracle.ghost_of(r, f["delay_s"], rp) == pytest.approx(m["ghost"][r["id"]], rel=0.05, abs=0.001)
    assert len(f["fat"]) == 1 and oracle.fat_clearance_mm(f, rp) > 0 and max(m["fat"].values()) <= oracle.FAT_LIMIT
    assert world["names"][-1] == "insert0" and f["fat"][0]["radius_mm"] == pytest.approx(cfg["world_spec"]["qa"]["inserts"][0]["radius_mm"], abs=0.3)
    assert oracle.wrap_clearance_mm(f, rp) >= 0 and m["wrap"] <= oracle.WRAP_LIMIT
    assert oracle.centre_pe(rp) == best["fields"]["centre_pe_mm"] == oracle.best_centre_mm(f, rp)          # centred on the displaced object
    assert oracle.matrix_of(t["pixel_max_mm"]) == rp["matrix"] and [oracle.matrix_of(p) for p in task.PIXELS_MM] == [80, 72, 64]


def test_the_oracle_answer_passes_and_one_average_less_does_not(instance):
    cfg, world, scans, best = instance
    assert O.independence_problems("mrgym.envs.epi_setup") == [] and oracle.REGRET_KIND == "difference"
    s = task.score(cfg, best["fields"], world, scans.engine())
    json.dumps(s)
    assert s["passed"] and s["error"] == 0.0 and s["fewest_averages"] == best["averages"] and all(s["checks"].values())
    assert set(s["checks"]) == {"deliverable_valid", "echo_time", "pixel", "wrap", "time"} | {f"{k}_{r['id']}" for r in cfg["task"]["regions"]
                                                                                              for k in ("displacement", "ghost", "fat", "snr")}
    more = task.score(cfg, dict(best["fields"], averages=best["averages"] + 1), world, scans.engine())
    assert not more["passed"] and [k for k, ok in more["checks"].items() if not ok] == ["time"] and task.outcome(more) == "too_many_averages"
    if best["averages"] > 1:
        fewer = task.score(cfg, dict(best["fields"], averages=best["averages"] - 1), world, scans.engine())
        assert not fewer["passed"] and all(k.startswith("snr_") for k, ok in fewer["checks"].items() if not ok) and task.outcome(fewer) == "snr_too_low"


def test_the_instance_is_what_the_generator_promises(instance):
    """Strict and stated limits need the same averages; few geometries meet displacement and ghost; the field decides."""
    cfg, world, scans, best = instance
    t = cfg["task"]
    strict, loose = oracle.search(world, t, scans, strict=True), oracle.search(world, t, scans)
    assert oracle.averages_for(strict["snr1"], t["snr_min"]) == oracle.averages_for(loose["snr1"], t["snr_min"]) == best["averages"] in task.AVERAGES_TARGET
    assert task.FEW[0] - 1 <= len(strict["candidates"]) <= 2 * task.FEW[1] + 1         # counted on the coarse grid of the generator; two polarities
    top = strict["candidates"][0]["realised"]
    f = oracle.facts(world, t)
    lowest = oracle.ladder(world["console"], top, t["te_window_ms"])[0]
    assert max(abs(float(oracle.displacement_mm(r["df_hz"], lowest))) for r in f["regions"]) > 1.15 * t["displacement_max_mm"]


def test_each_requirement_has_a_check_that_names_it(instance):
    cfg, world, scans, best = instance
    t, a = cfg["task"], best["fields"]
    failing = lambda fields: [k for k, ok in task.score(cfg, fields, world, scans.engine())["checks"].items() if not ok]      # noqa: E731
    low = dict(a, bandwidth_hz_per_px=500.0, te_ms=t["te_window_ms"][1] + 30.0)
    assert {"echo_time"} <= set(failing(low)) and any(k.startswith("displacement_") for k in failing(low))
    assert "pixel" in failing(dict(a, matrix=32, te_ms=t["te_window_ms"][0]))
    assert "wrap" in failing(dict(a, phase_fov=0.5, bandwidth_hz_per_px=3000.0, te_ms=t["te_window_ms"][0]))
    off = failing(dict(a, centre_pe_mm=a["centre_pe_mm"] + 30.0))                             # the same field of view, 30 mm off the object
    assert "wrap" in off and not any(k.startswith(("displacement_", "ghost_")) for k in off)
    finer = dict(t, pixel_max_mm=t["scan"]["fov_mm"] / (a["matrix"] + 8))                    # the search with a matrix above the smallest:
    assert oracle.search(world, finer, scans)["snr1"] < 0.9 * oracle.search(world, t, scans)["snr1"]      # less SNR, so never fewer averages
    assert task.outcome(dict(checks=dict(deliverable_valid=True, echo_time=True, pixel=True, displacement_R1=True, ghost_R1=False, wrap=False))) == "ghost"
    assert task.OUTCOMES == ("te_outside_window", "pixel_too_large", "displaced", "ghost", "wrap_around", "fat_on_region", "snr_too_low", "too_many_averages")


def test_deliverables_the_console_cannot_evaluate_are_refused(instance):
    cfg, world, _, best = instance
    exam, a = world["console"], best["fields"]
    assert task.check(cfg, a, exam) is None
    for bad, why in (({k: v for k, v in a.items() if k != "pe_polarity"}, "exactly the fields"), (dict(a, fat_sat=True), "exactly the fields"),
                     (dict(a, partial_fourier=0.6), "exactly the fields"), (dict(a, matrix=63), "the console rejects"),
                     (dict(a, bandwidth_hz_per_px=a["bandwidth_hz_per_px"] / 1000.0), "the console rejects"), (dict(a, te_ms="min"), "te_ms is a number"),
                     (dict(a, te_ms=5.0), "the console rejects"), ([a], "exactly the fields")):
        problem = task.check(cfg, bad, exam)
        assert isinstance(problem, str) and why in problem, bad
        s = task.score(cfg, bad, world)
        assert s["passed"] is False and s["checks"] == dict(deliverable_valid=False) and s["error"] is None
    want = dict(oracle.SCAN, **{k: v for k, v in a.items() if k != "centre_pe_mm"}, **{f"centre_{a['pe_dir']}_mm": a["centre_pe_mm"]})
    assert task.protocol_of(a) == want and oracle.SCAN["fat_sat"] is False and "partial_fourier" not in oracle.FIELDS
    assert set(task.submit_schema(cfg)["properties"]) == set(oracle.FIELDS)


# ------------------------------------------------------------------ instances
def _config(level, seed):
    return json.dumps(gate._instance(mrgym.spec(ENV), level, seed, "train"), sort_keys=True, default=float)


def test_an_instance_is_a_function_of_the_seed_also_next_to_other_threads():
    jobs = [(1, "test:3"), (1, "test:7"), (2, "test:1"), (2, "test:8"), (3, "test:3"), (3, "test:4")]
    task._instance.cache_clear()
    serial = {j: _config(*j) for j in jobs}
    task._instance.cache_clear()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = dict(zip(jobs, pool.map(lambda j: _config(*j), jobs)))
    assert threaded == serial and _config(2, "test:1") != _config(2, "test:8")


def test_the_statement_states_every_limit_and_no_hidden_value(instance):
    cfg, world, _, best = instance
    t, text = cfg["task"], task.statement(cfg)
    for part in (f"between {t['te_window_ms'][0]:g} and {t['te_window_ms'][1]:g} ms", f"at most {t['pixel_max_mm']:g} mm", "at most 3 mm", "at most 5 %",
                 f"at least {t['snr_min']:g}", "disc of oil", "no fat saturation", "full k-space", "centre_pe_mm", F.OWN_SCANS, F.NOT_CHARGED):
        assert part in text, part
    level1, level2 = (gate._instance(mrgym.spec(ENV), lv, seed, "train") for lv, seed in ((1, "test:3"), (2, "test:1")))
    assert "How to do it" in task.statement(level1) and "How to do it" not in task.statement(level2) and "oil" not in task.statement(level2)
    assert "inserts" not in json.dumps(t) and "world_spec" not in json.dumps(t)
    f = oracle.facts(world, t)
    hidden = [r["df_hz"] for r in f["regions"]] + [world["system"]["grad_delay_us"], world["system"]["sigma0"], world["system"]["cf_offset_hz"]]
    hidden += [v for cls in world["tissues"].values() for v in (cls["t1_ms"], cls["t2s_ms"])] + [f["fat"][0]["y_mm"], f["fat"][0]["x_mm"], best["fields"]["centre_pe_mm"]]
    for v in hidden:
        for form in (f"{abs(v):.3f}", f"{abs(v):.2f}"):
            assert form not in text, (v, form)
    assert "averages" not in {k for k in t if k != "deliverable"}                             # the fewest averages are not part of the task


# ------------------------------------------------------------------ the controls
def _play(name, level, seed, recipe=None):
    env = mrgym.make(ENV, level=level, seed=seed, split="gate")
    try:
        return policies.run(env, name, recipe=recipe)
    finally:
        env.close()


def test_the_reference_passes_through_the_tools_within_the_budget():
    for level, seed in ((1, 0), (3, 0)):
        r = _play("reference", level, seed)
        assert r.reward == 1.0 and r.info["policy_error"] is None, (level, seed, r.info["score"]["diagnostic"])
        assert r.info["budget"]["used_s"] <= task.LEVELS[level]["budget_s"] and r.info["score"]["error"] == 0.0


def test_every_listed_shortcut_fails_on_an_instance():
    r = _play("prior", 2, 0)
    assert r.reward == 0.0 and r.info["score"]["checks"]["measured"] is False
    for name in ("default_epi", "fastest_epi", "fixed_recipe", "wrong_units", "sweep"):
        r = _play(name, 2, 0)
        assert r.reward == 0.0 and r.info["policy_error"] is None and r.info["score"]["checks"]["measured"], name
        assert r.info["score"]["outcome"] in task.OUTCOMES, name


def test_the_controls_are_listed_and_described():
    names = [controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert set(names) == set(controls.policies()) == set(controls.DESCRIPTIONS)
    assert {"prior", "default_epi", "fastest_epi", "fixed_recipe", "wrong_units", "sweep"} == set(controls.MUST_FAIL) and controls.REPORTED == ("open_loop",)
    t = dict(regions=[dict(id="R1", y_mm=30.0, x_mm=5.0, radius_mm=6.0), dict(id="R2", y_mm=-35.0, x_mm=-8.0, radius_mm=6.0)], pixel_max_mm=3.0, snr_min=30.0,
             te_window_ms=[30.0, 45.0])
    assert controls.recipe_axis(dict(pe_dir="ghost_free"), t) == "y" and controls.recipe_axis(dict(pe_dir="x"), t) == "x"      # readout along x: the regions lie near x = 0
    assert controls.recipe_averages(dict(averages=2, snr_ref=None), t) == 2 and controls.recipe_averages(dict(averages=None, snr_ref=27.0), t) == 3
    assert controls.recipe_centre(dict(pe_dir="y", centre="regions"), t) == -2.5 and controls.recipe_centre(dict(pe_dir="y", centre=0.0), t) == 0.0
    assert isinstance(task.NEEDS, tuple) and "off-resonance displacement" in task.NEEDS and "phase-encode direction" in task.NEEDS
    assert all(isinstance(n, str) for n in task.NEEDS)
