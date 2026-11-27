"""``Scanner/BandFree-v0``: the physics the task rests on (the band of a 2D slice), the band model of the reference
policy against the engine, the verifier's table against scans, scoring mutants, the instance generator, the
independent oracle and the controls on a few instances. The scoring table itself is ``tests/scoring/band_free.py``."""
import numpy as np
import pytest

import mrgym
import scoring
from mrgym import coverage, policies
from mrgym.env import derive_seed
from mrgym.envs import family as F
from mrgym.envs.band_free import controls, oracle, task
from mrgym.scanner import console, kspace
from mrgym.scanner import session as core
from mrgym.scanner import world as W
from scoring import band_free as table

ENV = "Scanner/BandFree-v0"


def _share(amplitude, keep=0.8):
    a = np.abs(amplitude)
    return float((a >= keep * a.max()).mean())


def _ramp_world(t1_ms, t2_ms, tr_ms, n=48):
    """One material in a field that runs through one turn of the phase per TR across the grid, perfect transmit."""
    ph = W.shapes_phantom([("gm", lambda y, x: np.maximum(np.abs(y), np.abs(x)) - 500.0)], n=n,
                          tissues=dict(gm=dict(pd=1.0, t1_ms=t1_ms, t2_ms=t2_ms, t2s_ms=t2_ms / 2, t2p_ms=t2_ms, adc_um2_ms=1.0, fat=False)))
    world = W.flat_world(ph)
    theta = 360.0 * np.arange(n * n) / (n * n)
    world["fields"]["b0_hz"] = (theta / (360.0 * tr_ms * 1e-3)).reshape(n, n)
    return world, theta


def _engine_band(t1_ms, t2_ms, bandwidth, excitation="selective"):
    """|echo| of one material at every whole degree of the phase per TR, as the engine serves it for the target scan."""
    exam = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=1e3, b0_t=3.0)
    tr = task.tr_min_ms(bandwidth, exam)
    realised = console.plan(dict(task.TARGET, bandwidth_hz_per_px=bandwidth, tr_ms=tr, phase_cycle_deg=0.0, excitation=excitation), exam)["realised"]
    world, theta = _ramp_world(t1_ms, t2_ms, realised["tr_ms"])
    amp = np.abs(np.asarray(kspace.Engine(world).amplitudes(realised, 0)[0]))
    return np.interp(np.arange(360.0), theta, amp, period=360.0), realised["tr_ms"]


# ------------------------------------------------------------------------------------------------ the physics
MATERIALS = ((2000.0, 50.0), (1000.0, 100.0), (600.0, 200.0), (300.0, 150.0))        # T1, T2 in ms: T1 / T2 from 40 to 2


def test_the_band_of_a_2d_slice_is_narrower_than_the_band_of_a_hard_pulse():
    """Why the band model carries the slice profile (module docstring of the task): with the selective pulse of the
    target scan a material keeps 80 % of its best signal over a larger share of the band spacing than with a hard
    pulse at the same nominal angle, for every material."""
    selective = [_share(_engine_band(t1, t2, 300.0)[0]) for t1, t2 in MATERIALS]
    hard = [_share(_engine_band(t1, t2, 300.0, "nonselective")[0]) for t1, t2 in MATERIALS]
    assert all(s > h + 0.12 for s, h in zip(selective, hard)), (selective, hard)
    assert 0.55 < min(selective) < 0.70 and 0.80 < max(selective) < 0.95, selective
    assert 0.38 < min(hard) < 0.50 and 0.65 < max(hard) < 0.80, hard
    assert selective == sorted(selective) and hard == sorted(hard)        # the longer T2 against T1, the narrower the band


@pytest.mark.parametrize("t1_ms, t2_ms", MATERIALS)
def test_the_band_model_of_the_reference_is_the_band_the_engine_serves(t1_ms, t2_ms):
    """The policy's own slice profile (a cascade of rotations from the datasheet) and steady state against the engine's
    (Cayley-Klein tables, barycentric interpolation): the same band at both ends of the allowed bandwidths."""
    for bandwidth in (80.0, 600.0):
        engine, tr = _engine_band(t1_ms, t2_ms, bandwidth)
        model = controls.region_keep(t1_ms / t2_ms, tr, [0.0], t2_ms=t2_ms)
        assert np.abs(model - engine / engine.max()).max() <= 0.01, bandwidth
        assert abs(_share(model) - _share(engine)) <= 0.006
    hard = controls.region_keep(t1_ms / t2_ms, 6.0, [0.0], hard=True, t2_ms=t2_ms)
    assert _share(hard) < _share(controls.region_keep(t1_ms / t2_ms, 6.0, [0.0], t2_ms=t2_ms)) - 0.1


def test_fit_band_recovers_the_material_and_the_position_from_the_pixels_of_a_vial():
    """The field changes across a vial, so the pixels of a scan sample the band at many phases per TR: synthetic
    read-outs of a material (a receive factor, noise, a field map that is off by a few degrees) give back its T1 / T2,
    its T2 and the error of the map; and what counts, the share of the band spacing over which a region keeps 80 %."""
    rng = np.random.default_rng(3)
    phases = [0.0, 90.0, 180.0, 270.0]
    for ratio, t2, shift, span in ((8.0, 92.0, 6.0, 70.0), (2.0, 160.0, -4.5, 40.0), (30.0, 45.0, 0.0, 110.0)):
        theta = 100.0 + span * rng.uniform(-1, 1, 60)                    # phase per TR of 60 pixels as the map has it
        table = controls.band_table(ratio, 9.0, t2)
        z = np.array([[table[int(round(th + shift - ph)) % 720] * np.exp(0.5j * np.deg2rad(ph)) for ph in phases] for th in theta])
        z = (0.7 - 0.4j) * z + 0.01 * np.abs(table).max() * (rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape))
        got, got_t2, moved, misfit = controls.fit_band(theta, phases, z, 9.0, np.arange(-9.0, 9.1, 1.5))
        assert 0.7 < got * got_t2 / (ratio * t2) < 1.4 and abs(moved - shift) <= 1.5 and misfit < 0.1, (ratio, t2, got, got_t2, moved, misfit)
        assert abs(_share(controls.region_keep(got, 9.0, [0.0], t2_ms=got_t2)) - _share(controls.region_keep(ratio, 9.0, [0.0], t2_ms=t2))) <= 0.02
    fixed = controls.fit_band(theta, phases, z, 9.0, [0.0], ratios=[controls.UNIVERSAL_RATIO], t2s=[controls.UNIVERSAL_T2_MS])
    assert fixed[:2] == (controls.UNIVERSAL_RATIO, controls.UNIVERSAL_T2_MS)


def test_scout_phases_put_a_scan_on_a_flank_of_every_band():
    for freqs in ((12.0, 40.0, -31.0, 77.0, 3.0), (-120.0, 95.0, 20.0, 21.0, -55.0)):
        phases = controls.scout_phases([dict(coef=np.array([f, 0, 0, 0, 0, 0.0])) for f in freqs], 7.3)
        assert len(phases) == 4
        for f in freqs:                                                   # distance of the nearest scan from the middle of the band
            delta = np.abs((360.0 * f * 7.3e-3 - np.array(phases) + 180.0) % 360.0 - 180.0)
            assert np.abs(delta - controls.FLANK_DEG).min() <= 25.0, (f, phases)


def test_the_field_seen_by_a_pixel_is_the_field_where_its_spins_are():
    coef = np.array([100.0, 0.5, 2.0, 0.0, 0.0, 0.0])                    # 100 Hz at the centre, 2 Hz / mm along the readout
    assert controls.field_at(coef, 3.0, -2.0) == pytest.approx(100.0 + 1.5 - 4.0)
    seen = controls.seen_field(coef, 0.0, 0.0, 200.0, 2.0)               # displaced by f / 200 pixels of 2 mm: f = 100 + 2 x f / 100
    assert seen == pytest.approx(100.0 / (1.0 - 2.0 * 2.0 / 200.0), rel=1e-4)
    iy, ix, dy, dx = controls.pixels_near(dict(y_mm=10.3, x_mm=-20.1, radius_mm=6.0), 240.0, 120, 6.0)
    pos = controls.pixel_positions(240.0, 120)
    assert pos[60] == 0.0 and pos[1] - pos[0] == 2.0 and len(iy) in range(24, 33)
    assert np.allclose(pos[iy] - 10.3, dy) and np.allclose(pos[ix] + 20.1, dx) and (dy ** 2 + dx ** 2).max() <= 36.0


def _t(n=3):
    return dict(regions=[dict(id=f"R{i + 1}", radius_mm=8.0) for i in range(n)], keep_min=0.8, drift_hz=20.0, reference_drift_hz=26.0,
                bandwidth_hz_per_px=[60.0, 600.0], single_up_to=400.0)


def test_decide_takes_one_acquisition_when_the_spread_fits_and_two_when_it_cannot():
    tr_of = lambda bw: 3.4 + 1000.0 / bw                                 # noqa: E731
    grid = np.exp(np.arange(np.log(60.0), np.log(600.0), 0.03))
    one_pixel = (np.zeros(1), np.zeros(1))

    def decide(freqs, mode="auto"):
        n = len(freqs)
        vials = [dict(coef=np.array([f, 0, 0, 0, 0, 0.0]), ratio=10.0, t2=70.0, roi=one_pixel) for f in freqs]
        return controls.decide(controls.margins(vials, tr_of, grid, _t(n)), list(range(n)), _t(n), mode)
    near, wide = decide([40.0, 52.0, 61.0]), decide([40.0, 75.0, 105.0])
    assert len(near["phases"]) == 1 and len(wide["phases"]) == 1 and near["bandwidth"] < wide["bandwidth"]
    apart = [0.0, 60.0, 120.0]                                           # at no allowed repetition time does one pass band hold all three
    assert len(decide(apart)["phases"]) == 2 and len(decide(apart, "single")["phases"]) == 1
    assert len(decide([40.0, 52.0, 61.0], "pair")["phases"]) == 2
    # the pass band is centred on the regions: phase per TR - phase cycle = 180 degrees at the middle frequency
    tr = tr_of(near["bandwidth"])
    assert abs((360.0 * 50.5 * tr * 1e-3 - near["phases"][0]) % 360.0 - 180.0) <= 12.0


# ------------------------------------------------------------------------------------------------ the verifier
@pytest.fixture(scope="module")
def oil_case():
    """The level-3 instance of the scoring table: one region of image B holds oil."""
    return scoring.case(table, "L3")


def test_verifier_table_agrees_with_scans_for_water_and_for_oil(oil_case):
    """Rules 2 and 3 and the drift part of rule 1 use the amplitudes on the regions' pixels; the present frequency
    is scanned. The two see the same bands, in the oil region too (chemical shift and its displacement)."""
    cfg, world, engine = oil_case.cfg, oil_case.world, oil_case.engine
    fat = [world["tissues"][r["cls"]]["fat"] for r in world["regions"] if r["id"] in {x["id"] for x in cfg["task"]["regions"]}]
    assert sum(fat) == 1
    ev = task.Evaluator(world, cfg["task"]["regions"])
    for bw in (120.0, 300.0):
        keep, _ = ev.keep(bw)
        phases = [0, 60, 120, 180, 240, 300]
        sig = np.array([[abs(v) for v in F.region_complex(engine, task.target_plan(bw, float(p), world["console"])["realised"], cfg["task"]["regions"]).values()]
                        for p in phases]).T
        tab = keep[:, phases]
        scale = (sig * tab).sum(1) / (tab * tab).sum(1)                  # the table is relative: one factor per region
        assert np.abs(sig - tab * scale[:, None]).max() <= 0.04 * sig.max(), bw


def test_window_min_and_robust_mean_every_drift_not_three_of_them():
    """A band narrower than the drift cannot hide between the present frequency and the two ends of the drift."""
    keep = np.ones((1, 360))
    keep[0, 100:112] = 0.3                                               # a band 12 degrees wide
    tr = 10.0                                                            # 20 Hz of drift = 72 degrees
    assert task.drift_deg(20.0, tr) == pytest.approx(72.0)
    assert task.window_min(keep[0], 60.0, 72.0) == pytest.approx(0.3) and task.window_min(keep[0], 250.0, 72.0) == 1.0
    assert task.window_min(keep[0], 27.0, 72.0) == 1.0 and task.window_min(keep[0], 27.5, 72.0) == pytest.approx(0.65)   # the end of the window, between two degrees
    ok = task.robust(keep, tr, 20.0)[0]
    assert not ok[40:172].any() and ok[20] and ok[190]                   # every phase cycle within 72 degrees of the band is out
    assert not task.robust(keep, 30.0, 20.0).any()                       # the drift spans more than a turn: nothing is robust


def _mutants():
    """Each mutant is the scorer with one defect: (what is wrong, patch(monkeypatch), the table rows that expose it)."""
    def drift_not_checked(mp):                                            # only the present centre frequency counts
        mp.setattr(task, "window_min", lambda row, phi, half: float(row[int(round(phi)) % 360]))

    def best_region_instead_of_every(mp):
        real = task.kept

        def kept(*a, **k):
            out = real(*a, **k)
            top = max(out.values(), key=lambda v: min(v["now"], v["drift"]))
            return {i: dict(top) for i in out}
        mp.setattr(task, "kept", kept)

    def pair_always_allowed(mp):
        real = task.reference_protocols
        mp.setattr(task, "reference_protocols", lambda cfg, world: {n: dict(v, single_bandwidth=None) for n, v in real(cfg, world).items()})

    def bandwidth_rule_dropped(mp):
        real = task.reference_protocols
        mp.setattr(task, "reference_protocols", lambda cfg, world: {n: dict(v, single_bandwidth=None if v["single_bandwidth"] is None else 800.0)
                                                                    for n, v in real(cfg, world).items()})

    def first_image_only(mp):
        real = task.F.result
        mp.setattr(task.F, "result", lambda checks, *a, **k: real({c: v for c, v in checks.items() if not c.endswith("_B")}, *a, **k))

    def threshold_sign(mp):                                               # a band counts as kept signal
        real = task.kept
        mp.setattr(task, "kept", lambda *a, **k: {i: dict(v, now=1.6 - v["now"], drift=1.6 - v["drift"]) for i, v in real(*a, **k).items()})

    return [("drift not checked", drift_not_checked, ("image A at the edge of a band: fine now, lost within the drift",)),
            ("best region instead of every", best_region_instead_of_every, ("image A at the edge of a band: fine now, lost within the drift",)),
            ("pair always allowed", pair_always_allowed, ("image B with two acquisitions where one is band-free",)),
            ("bandwidth rule dropped", bandwidth_rule_dropped, ("image A band-free at 2.2 times the lowest bandwidth",)),
            ("first image only", first_image_only, ("image B half a turn away",)),
            ("threshold sign", threshold_sign, ("the oracle's protocols", "image B half a turn away"))]


def test_scoring_mutants_are_killed(monkeypatch):
    case = scoring.case(table, "L2")
    rows = {r.name: r for r in table.ROWS if r.instance == "L2" and not r.refused}
    verdict = lambda name: task.score(case.cfg, case.answer(rows[name]), case.world, case.engine)["passed"]   # noqa: E731
    exposing = sorted({e for _, _, names in _mutants() for e in names})
    truth = {name: verdict(name) for name in exposing}
    assert truth == {name: rows[name].passed for name in exposing}
    killed = []
    for name, patch, names in _mutants():
        with monkeypatch.context() as mp:
            patch(mp)
            if any(verdict(e) != truth[e] for e in names):
                killed.append(name)
    assert killed == [m[0] for m in _mutants()], killed


# ------------------------------------------------------------------------------------------------ instances and the card
def test_statement_prints_every_number_of_the_pass_rule_and_hides_the_world():
    env = mrgym.make(ENV, level=3, seed=1)
    text = env.reset()[0].text
    cfg = env.episode_config()
    t = cfg["task"]
    for number in (t["keep_min"] * 100, t["drift_hz"], t["reference_drift_hz"], t["bandwidth_slack"], t["single_up_to"], *t["bandwidth_hz_per_px"],
                   t["target_scan"]["flip_deg"], t["target_scan"]["matrix"]):
        assert f"{number:g}" in text, number
    assert all(r["id"] in text for r in t["regions"]) and [i for ids in t["images"].values() for i in ids] == [r["id"] for r in t["regions"]]
    public = (text + str(t) + str(task.submit_schema(cfg))).lower()
    assert not any(word in public for word in ("b0_hz", "seed", "attempt", "oil", "fat", "n_oil")), "the oil region is not told"
    assert mrgym.spec(ENV).kind in ("candidate", "admitted") and set(mrgym.card(ENV)) >= {"task_kind", "task_kind_why", "controls", "levels", "reward"}
    assert set(task.COMPETENCIES) <= set(coverage.ids()) and mrgym.card(ENV)["competencies"] == list(task.COMPETENCIES)


def test_level_3_marks_exactly_one_oil_region_and_the_ids_do_not_say_which():
    """The world of an instance is that of ``family.reseed(seed, attempt)``: the first attempt on which the oil vial is
    one of the marked regions R1 ... R5. Lower levels have no oil and take the first world."""
    where, attempts = [], []
    for i in range(12):
        seed_hex = derive_seed(ENV, 3, "train", f"oil:{i}", environ={})
        cfg = task.make_instance(seed_hex, 3, "x")
        layout = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
        marked = {r["id"] for r in cfg["task"]["regions"]}
        oil = [r["id"] for r in layout["regions"] if layout["tissues"][r["cls"]]["fat"]]
        assert len(oil) == 1 and oil[0] in marked and marked == {"R1", "R2", "R3", "R4", "R5"}
        assert cfg["seed"] == F.reseed(seed_hex, cfg["attempt"]) and 0 <= cfg["attempt"] < task.MAX_ATTEMPTS
        lo, hi = task.LEVELS[3]["b0_hz"]
        assert lo <= cfg["world_spec"]["b0_hz"] <= hi
        where.append(oil[0]), attempts.append(cfg["attempt"])
    assert len(set(where)) >= 3 and max(attempts) > 0 and min(attempts) == 0     # any of the five, and some worlds were re-drawn
    for level in (1, 2):
        cfg = task.make_instance(derive_seed(ENV, level, "train", "oil:0", environ={}), level, "x")
        layout = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
        assert cfg["attempt"] == 0 and not any(c["fat"] for c in layout["tissues"].values())


def test_task_kind_follows_the_stored_gate(monkeypatch):
    """Measurement or decision task, per level: read from the stored gate (open_loop against the reference), never typed."""
    from mrgym import gate
    level = lambda ref, open_loop: dict(n=60, reference=dict(rate=ref), reported=[dict(policy="open_loop", rate=open_loop)], ablations=[])   # noqa: E731
    fake = dict(engine=mrgym.ENGINE_ID, levels={"1": level(0.97, 0.95), "2": level(0.96, 0.40)})
    kind = task.task_kind(fake)
    assert kind["levels"] == {"1": "measurement", "2": "decision"} and kind["kind"] == "mixed"
    assert "level 1 is a measurement task, level 2 is a decision task" in kind["sentence"] and "2 at level 1, 56 at level 2" in kind["evidence"]
    assert task.task_kind(dict(fake, levels={"1": level(0.97, 0.95)}))["kind"] == "measurement"
    monkeypatch.setattr(gate, "load", lambda env_id: None)
    card = task.card()
    assert card["task_kind"] == "undetermined" and "not determined" in card["summary"] and task.task_kind()["levels"] == {}
    monkeypatch.setattr(gate, "load", lambda env_id: fake)
    assert task.card()["task_kind"] == "mixed" and "level 2 is a decision task" in task.card()["summary"]


# ------------------------------------------------------------------------------------------------ oracle and controls
def test_the_field_map_finds_the_oil_and_puts_it_on_the_fat_line(oil_case):
    env = mrgym.make(ENV, **table.INSTANCES["L3"])
    env.reset()
    sc = policies.ToolScanner(env)
    t = sc.info()["task"]
    vials = controls._field_map(sc, t)
    world = env.hidden_world()
    n, d = world["n"], world["extent_mm"] / world["n"]
    df = world["fields"]["b0_hz"] + world["system"]["cf_offset_hz"]
    for r, v in zip(t["regions"], vials):
        fat = world["tissues"][next(x["cls"] for x in world["regions"] if x["id"] == r["id"])]["fat"]
        i, j = int(round(r["y_mm"] / d)) + n // 2, int(round(r["x_mm"] / d)) + n // 2
        true = df[i, j] - (434.3 if fat else 0.0)
        assert v["oil"] is fat and abs(v["coef"][0] - true) <= (controls.OIL_MAP_ERROR_HZ if fat else controls.MAP_ERROR_HZ), (r["id"], v["coef"][0], true)
        k = int(round(6.0 / d))                                           # the gradient of the surface against the true field
        gy, gx = (df[i + k, j] - df[i - k, j]) / (2 * k * d), (df[i, j + k] - df[i, j - k]) / (2 * k * d)
        assert abs(v["coef"][1] - gy) <= 0.35 and abs(v["coef"][2] - gx) <= 0.35, (r["id"], v["coef"][1:3], gy, gx)
        assert len(v["roi"][0]) == oil_case.task.Evaluator(oil_case.world, [r]).points[0]["y"].size     # the pixels the console reads
    plain = controls._field_map(sc, t, find_oil=False)                    # the ablation: the oil stays where the map wraps it,
    assert not any(v["oil"] for v in plain)                               # which is the fat line or one echo-spacing period (901 Hz) off it
    moved = [abs(a["coef"][0] - b["coef"][0]) for a, b in zip(plain, vials)]
    assert all(m < 12.0 or abs(m - 900.9) < 12.0 for m in moved)          # (two scans: the water regions differ by their noise)
    env.close()


def test_reference_passes_replays_and_the_oracle_agrees():
    """One level-3 episode: the reference passes; the log replays against its head on a world rebuilt from the episode
    config; the deliverable scores the same from the config alone; the oracle's own answer passes too and the two
    agree on the lowest band-free bandwidths."""
    env = mrgym.make(ENV, level=3, seed="t:2")
    r = policies.run(env, "reference")
    assert r.info["policy_error"] is None and r.reward == 1.0, r.info["score"]
    cfg, world, engine = env.episode_config(), env.hidden_world(), env._session.engine
    answer = env.transcript()[-1]["call"]["arguments"]
    rep = core.replay(cfg, env.scanner_log(), head=env._session.receipt(with_key=True))
    assert rep["ok"] and rep["sealed"] and not rep["over_budget"] and rep["scans"] == 5 and env.state()["scan_time_used_s"] <= 43.0
    alone = task.score(cfg, answer)                                       # the verifier's side: world and engine from the config
    assert alone["passed"] and alone["checks"] == {k: v for k, v in r.info["score"]["checks"].items() if k not in F.GATES}
    assert alone["images"] == r.info["score"]["images"]
    o, v = oracle.solve(cfg, world), task.reference_protocols(cfg, world)
    for name, img in o["images"].items():
        one = v[name]["single_bandwidth"] is not None and v[name]["single_bandwidth"] <= cfg["task"]["single_up_to"]
        assert img["needs_two"] != one
        lowest, mine = (v[name]["single_bandwidth"], img["single_bandwidth"]) if one else (v[name]["pair_bandwidth"], img["pair_bandwidth"])
        assert 0.94 < mine / lowest < 1.06, (name, mine, lowest)
    assert task.score(cfg, o["answer"], world, engine)["passed"]
    env.close()


def test_shortcuts_fail_and_the_controls_are_the_documented_ones():
    env = mrgym.make(ENV, level=2, seed=5)
    assert policies.run(env, "prior").reward == 0.0 and not env._session.n_scans
    cfg, exam = env.episode_config(), env.hidden_world()["console"]
    good = {name: dict(bandwidth_hz_per_px=240.0, phase_cycle_deg=[112.0]) for name in cfg["task"]["images"]}
    assert task.check(cfg, good, exam) is None
    in_khz_and_radians = {name: dict(bandwidth_hz_per_px=0.24, phase_cycle_deg=[1.95]) for name in good}
    assert "outside the allowed range" in task.check(cfg, in_khz_and_radians, exam)
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED, *controls.FURTHER}
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and set(controls.LADDER.values()) <= set(controls.ABLATIONS)
    blind = controls.blind_answers(cfg["task"])
    assert len(blind) == 24 and all(task.check(cfg, a, exam) is None for a in blind.values())
    env.close()


def test_tuning_returns_deliverable_recipes():
    """``tune`` (the gate's fixed recipe) and ``tune_sweep`` (the bandwidth of ``brute_force``) on two worlds."""
    from mrgym import gate
    spec = mrgym.spec(ENV)
    cfgs = [gate._instance(spec, 3, f"tune-test:{i}") for i in range(2)]
    bws, worlds = controls._tables(cfgs, every=4)
    recipe = controls._recipe(bws, worlds)
    answer = controls.recipe_answer(recipe, cfgs[0]["task"])
    assert task.check(cfgs[0], answer, dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)) is None
    assert 0.0 <= recipe["tuned_pass_rate"] <= 1.0 and set(answer) == {"A", "B"}
    sweep = controls._sweep(bws, worlds, controls.SWEEP_SCANS)
    lo, hi = task.BANDWIDTH
    assert lo <= sweep["bandwidth_hz_per_px"] <= hi and 0.0 <= sweep["tuned_pass_rate"] <= 1.0
    assert controls.sweep_answer(np.array([[1.0, 0.2], [0.3, 1.0], [0.9, 0.85]]), [0.0, 120.0, 240.0], 0.8) == [240.0]
    assert controls.sweep_answer(np.array([[1.0, 0.2], [0.3, 1.0], [0.6, 0.7]]), [0.0, 120.0, 240.0], 0.8) == [0.0, 120.0]
