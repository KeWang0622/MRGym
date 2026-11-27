"""``Scanner/GradientCal-v0`` beyond its scoring table (``tests/scoring/gradient_cal.py``): the three laws the family
rests on, checked on noise-free images against their closed forms with an edge finder written here; the instances; the
control policies through the tools."""
import concurrent.futures
import json

import numpy as np
import pytest

import mrgym
from mrgym import cards, coverage, policies
from mrgym.env import derive_seed
from mrgym.envs import family as F
from mrgym.envs.gradient_cal import controls, task
from mrgym.scanner import console, kspace
from mrgym.scanner import world as W

ENV_ID = "Scanner/GradientCal-v0"
SCAN = dict(sequence="se", tr_ms=60.0, fov_mm=250.0, matrix=384, bandwidth_hz_per_px=300.0)
PIXEL_MM = 250.0 / 384
GAMMA_HZ_PER_T = 42.577478518e6


def _cfg(level, seed):
    return task.make_instance(derive_seed(ENV_ID, level, "train", seed, environ={}), level, task_id="t")


def _world(level, seed=0):
    cfg = _cfg(level, seed)
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    return cfg, world, kspace.Engine(world)


def _image(engine, **more):
    rp = console.plan(dict(SCAN, **more), engine.w["console"])["realised"]
    img = np.abs(engine.acquire(rp, None)[0])
    return img, (np.arange(img.shape[0]) - img.shape[0] // 2) * PIXEL_MM


def _edges(line, pos):
    """(low, high) edge of the disc on one line of pixels: where it crosses half of the mean of the samples 4 to 12 mm
    inside the edge, by linear interpolation. Written here; the control policies have their own."""
    out = []
    for v, p in ((line, pos), (line[::-1], pos[::-1])):                  # from outside the object inwards
        i = int(np.argmax(v >= 0.5 * v.max()))
        far = np.abs(p - p[i])
        level = 0.5 * v[(far >= 4.0) & (far <= 12.0) & (np.arange(len(p)) > i)].mean()
        j = int(np.argmax(v >= level))
        out.append(p[j - 1] + (level - v[j - 1]) / (v[j] - v[j - 1]) * (p[j] - p[j - 1]))
    return out[0], out[1]


def _diameters(img, pos) -> dict:
    """Imaged diameter along y and along x through the centre of the disc (five pixel lines averaged)."""
    mid = len(pos) // 2
    x0, x1 = _edges(img[mid - 2:mid + 3].mean(0), pos)
    col = int(round(0.5 * (x0 + x1) / PIXEL_MM)) + mid
    y0, y1 = _edges(img[:, col - 2:col + 3].mean(1), pos)
    row = int(round(0.5 * (y0 + y1) / PIXEL_MM)) + mid
    x0, x1 = _edges(img[row - 2:row + 3].mean(0), pos)
    return dict(y=y1 - y0, x=x1 - x0)


# ---------------------------------------------------------------------------------------------- the physics, independently
def test_a_gradient_that_is_s_times_too_strong_draws_lengths_s_times_too_long():
    """Uniform field, linear gradients (level 0): imaged diameter / 190 mm is the hidden scale of that axis."""
    cfg, world, engine = _world(0)
    sy, sx = world["system"]["grad_scale"]
    assert cfg["hidden"]["scale"] == dict(y=round(sy, 6), x=round(sx, 6)) and task.true_scale(world) == dict(y=sy, x=sx)
    d = _diameters(*_image(engine))
    assert d["y"] / 190.0 == pytest.approx(sy, abs=1.5e-3) and d["x"] / 190.0 == pytest.approx(sx, abs=1.5e-3)
    assert abs(sy - 1) >= 0.01 and abs(sx - 1) >= 0.01                  # 1.000 is at least 3 tolerances away


def test_a_field_gradient_scales_the_readout_axis_by_gradient_x_pixel_over_bandwidth_and_not_the_phase_axis():
    """Closed form: a spin off resonance by df is displaced by -polarity x df / bandwidth pixels along the readout. A
    field gradient g (Hz/mm of object length) therefore changes the imaged readout diameter by -polarity x g x pixel /
    bandwidth x 190 mm. The phase-encode axis keeps its length, and so does the mean of the two polarities."""
    cfg, world, engine = _world(1)
    sy, sx = world["system"]["grad_scale"]
    gx = cfg["world_spec"]["acr_fields"]["shim_hz_per_mm"][1]
    assert task.FIELD_GRADIENT_HZ_PER_MM[0] <= abs(gx) <= task.FIELD_GRADIENT_HZ_PER_MM[1]
    eps = gx * PIXEL_MM / SCAN["bandwidth_hz_per_px"]
    assert abs(eps) > 4 * task.TOLERANCE
    plus, minus = (_diameters(*_image(engine, pe_dir="y", ro_polarity=pol)) for pol in (1, -1))
    assert plus["x"] / 190.0 == pytest.approx(sx - eps, abs=1.2e-3) and minus["x"] / 190.0 == pytest.approx(sx + eps, abs=1.2e-3)
    assert 0.5 * (plus["x"] + minus["x"]) / 190.0 == pytest.approx(sx, abs=1.2e-3)
    assert plus["y"] / 190.0 == pytest.approx(sy, abs=1.2e-3) and minus["y"] / 190.0 == pytest.approx(sy, abs=1.2e-3)
    swapped = _diameters(*_image(engine, pe_dir="x"))                    # the readout now runs along y
    assert swapped["x"] / 190.0 == pytest.approx(sx, abs=1.2e-3) and abs(swapped["y"] / 190.0 - sy) > 4 * task.TOLERANCE


def test_no_readout_gradient_of_the_console_makes_the_field_gradient_small_enough():
    """The readout gradient in Hz/mm is bandwidth per pixel / pixel size; the console refuses more than 40 mT/m."""
    limit_hz_per_mm = GAMMA_HZ_PER_T * console.SYSTEM["g_max_mt_m"] * 1e-3 * 1e-3
    assert limit_hz_per_mm == pytest.approx(1703.1, abs=0.1)
    assert task.FIELD_GRADIENT_HZ_PER_MM[0] / limit_hz_per_mm > task.TOLERANCE
    exam = dict(b0_t=1.5, min_pixel_mm=0.5, b1rms_limit_ut=6.0)
    below, above = (console.plan(dict(SCAN, bandwidth_hz_per_px=f * limit_hz_per_mm * PIXEL_MM), exam) for f in (0.99, 1.01))
    assert below["accept"] and [v["code"] for v in above["violations"]] == ["GRADIENT_LIMIT"]


def test_under_non_linear_gradients_the_edge_of_the_disc_is_not_at_the_scale_of_the_isocentre():
    """Declared model (``world._warp``): the image point r shows the object point r (1 - g r^2 / (250 mm)^2) / s. The
    imaged half diameter p along an axis therefore obeys p (1 - g p^2 / 250^2) = 95 s."""
    cfg, world, engine = _world(3)
    g, (sy, sx) = cfg["world_spec"]["gnl"], world["system"]["grad_scale"]
    assert task.NONLINEARITY[0] <= abs(g) <= task.NONLINEARITY[1]
    d = dict(y=_diameters(*_image(engine, pe_dir="y"))["y"], x=_diameters(*_image(engine, pe_dir="x"))["x"])
    for axis, s in (("y", sy), ("x", sx)):
        p = d[axis] / 2.0
        assert p * (1.0 - g * p * p / 250.0 ** 2) == pytest.approx(95.0 * s, abs=0.12)
        assert abs(d[axis] / 190.0 / s - 1.0) > 2.5 * task.TOLERANCE     # the diameter alone is outside the pass window


# ---------------------------------------------------------------------------------------------- instances
def test_the_family_is_registered_and_says_what_it_isolates():
    spec = mrgym.spec(ENV_ID)
    assert spec.kind in ("candidate", "admitted") and (spec.levels, spec.default_level, spec.family) == ((0, 1, 2, 3), 2, "mrgym.envs.gradient_cal")
    assert (task.SUBJECT, task.KIND) == ("C", "measure") and isinstance(task.ISOLATES, str) and task.ISOLATES
    assert set(task.COMPETENCIES) <= set(coverage.ids()) and "qc.geometric_accuracy" in task.COMPETENCIES
    assert (task.DIAMETER_MM, task.BAR_MM, task.BARS_MM) == (190.0, 1.2, (7.5, 22.5, 37.5, 52.5))
    card = dict(task.card(), status="x", gate=None)
    assert cards.lint(card) == [] and card["practice_levels"] == [0] and cards.claim_levels(card) == [1, 2, 3]
    assert [c["name"] for c in card["controls"]] == [controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED]
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and all("\n" not in d for d in controls.DESCRIPTIONS.values())
    assert F.outcomes(task)[-4:] == ("both_scales_off", "scale_y_off", "scale_x_off", "pass")
    for checks, label in ((dict(scale_y=False, scale_x=False), "both_scales_off"), (dict(scale_y=False, scale_x=True), "scale_y_off"),
                          (dict(scale_y=True, scale_x=False), "scale_x_off")):
        assert task.outcome(dict(checks=dict(deliverable_valid=True, **checks))) == label


def test_an_instance_is_a_function_of_the_seed_also_under_threads():
    jobs = [(lv, s) for lv in (0, 1, 2, 3) for s in ("a", "b", "c")]
    canon = lambda cfg: json.dumps(cfg, sort_keys=True)                  # noqa: E731
    serial = [canon(_cfg(*j)) for j in jobs]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        threaded = list(pool.map(lambda j: canon(_cfg(*j)), jobs[::-1]))[::-1]
    assert threaded == serial and len(set(serial)) == len(serial)
    by_level = {lv: _cfg(lv, "a") for lv in (0, 1, 2, 3)}
    assert len({json.dumps(c["hidden"]) for c in by_level.values()}) == 4          # the seed of a level is its own
    assert by_level[1]["task"]["told"] == by_level[2]["task"]["told"] == dict(main_field_uniform=False, gradients_linear=True)
    assert "acr_fields" not in by_level[0]["world_spec"] and "gnl" not in by_level[2]["world_spec"] and "gnl" in by_level[3]["world_spec"]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_the_statement_states_the_phantom_and_hides_the_system(level):
    cfg = _cfg(level, 1)
    text, t, spec = task.statement(cfg), cfg["task"], cfg["world_spec"]
    public = text + json.dumps(t) + json.dumps(task.submit_schema(cfg))
    assert "hidden" not in t and cfg["seed"] not in public
    secrets = [*cfg["hidden"]["scale"].values(), *(v - 1.0 for v in cfg["hidden"]["scale"].values()), *spec.get("acr_fields", {}).get("shim_hz_per_mm", []),
               *([spec["gnl"]] if "gnl" in spec else [])]
    for secret in secrets:
        assert f"{secret:.4f}" not in public and f"{abs(secret):.4g}" not in public and f"{100 * abs(secret):.3g} %" not in public
    assert "190.0 mm inner diameter" in text and "7.5, 22.5, 37.5, 52.5 mm" in text and "within 0.3 % of the true scale" in text and "(0.57 mm" in text
    assert ("How to do it" in text) is (level in (0, 1)) and ("pe_dir" in text) is (level == 1)
    assert ("The main field is uniform" in text) is (level == 0)
    assert ("are not linear" in text) is (level == 3) and ("The gradients are linear over the phantom." in text) is (level != 3)
    for word in ("shim", "Hz/mm", "polarity"):
        assert word not in text
    assert cfg["budget_s"] == (45.0 if level == 0 else 75.0)
    two = 2 * console.plan(dict(controls.SCAN, pe_dir="y"), dict(b0_t=1.5, min_pixel_mm=0.5, b1rms_limit_ut=6.0))["scan_s"]
    assert two == pytest.approx(52.2, abs=0.05) and two < 75.0 < 1.5 * two            # two reference scans, not three


# ---------------------------------------------------------------------------------------------- the controls, through the tools
def _play(level, seed, name, recipe=None):
    r = policies.run(mrgym.make(ENV_ID, level=level, seed=seed), name, recipe=recipe)
    assert r.info["policy_error"] is None, r.info["policy_error"]
    return r


@pytest.mark.parametrize("level,scans", [(0, 1), (1, 2), (2, 2), (3, 2)])
def test_the_reference_passes_through_the_tools(level, scans):
    r = _play(level, 0, "reference")
    assert r.reward == 1.0 and r.info["score"]["error"] < task.TOLERANCE
    assert r.info["replay"]["ok"] and r.info["replay"]["scans"] == scans and r.info["budget"]["used_s"] == pytest.approx(26.1 * scans, abs=0.05)


def test_every_shortcut_fails_at_level_2_and_the_diameter_alone_at_level_3():
    for name in controls.MUST_FAIL:
        r = _play(2, 0, name)
        assert r.reward == 0.0, name
        if name == "wrong_units":                                        # a percentage is not a ratio near 1: refused, nothing delivered
            assert r.info["submitted"] is False and "must be a number from 0.5 to 2" in r.info["refused"]
        elif name != "prior":
            assert r.info["score"]["outcome"] in task.OUTCOMES, name
    single = _play(2, 0, "single_scan").info["score"]["checks"]
    assert single["scale_y"] is True and single["scale_x"] is False     # the phase-encode axis of that scan is right, the readout axis is not
    assert _play(2, 0, "diameter_only").reward == 1.0
    lost = _play(3, 0, "diameter_only")
    assert lost.reward == 0.0 and lost.info["score"]["outcome"] == "both_scales_off"


def test_tuning_gives_a_deliverable_and_a_refused_deliverable_does_not_end_the_episode():
    recipe = controls.tune([_cfg(2, f"tune{i}") for i in range(6)])
    assert set(recipe) == {"scale_y", "scale_x", "tuned_pass_rate"} and 0.9 < recipe["scale_y"] < 1.1 and 0.9 < recipe["scale_x"] < 1.1
    env = mrgym.make(ENV_ID, level=2, seed=0)
    env.reset()
    for bad, why in ((dict(scale_y=1.02), "scale_x"), (dict(scale_y=2.4, scale_x=-1.1), "scale_y must be a number from 0.5 to 2")):
        obs, reward, terminated, truncated, _ = env.step({"name": "submit", "arguments": bad})
        assert obs.is_error and not terminated and not truncated and reward == 0.0 and why in obs.text
    obs, reward, terminated, _, info = env.step({"name": "submit", "arguments": dict(scale_y=1.0, scale_x=1.0)})
    assert terminated and reward == 0.0 and info["score"]["outcome"] == "no_measurement"
    env.close()
