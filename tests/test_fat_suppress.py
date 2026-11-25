"""``Scanner/FatSuppress-v0``: the verifier's rules on a table of deliverables, scoring mutants, the hidden world
(everything comes out of ``make_world``; the keys the family uses, with a physics check), the independent oracle, and the controls on a few instances."""
import copy

import numpy as np
import pytest

import mrgym
from mrgym import coverage, policies
from mrgym.envs import family as F
from mrgym.envs.fat_suppress import controls, oracle, task
from mrgym.scanner import kspace, tissue
from mrgym.scanner import session as core
from mrgym.scanner import world as W

ENV = "Scanner/FatSuppress-v0"


def _instance(level, seed):
    """An instance on a coarser master grid (4x faster scans; the rules do not depend on the grid)."""
    env = mrgym.make(ENV, level=level, seed=seed)
    env.reset()
    cfg = copy.deepcopy(env.episode_config())
    cfg["world_spec"] = dict(cfg["world_spec"], n=320)
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    return cfg, world, kspace.Engine(world)


@pytest.fixture(scope="module")
def mixed():
    """A level-1 instance with a region where fat saturation clearly works and one where it clearly does not."""
    for seed in range(60):
        cfg, world, engine = _instance(1, f"table:{seed}")
        tr = oracle.truth(cfg, world)
        works = [i for i, r in tr.items() if r["fat_sat_left"] < 0.02]
        fails = [i for i, r in tr.items() if r["fat_sat_left"] > 0.30]
        if works and fails:
            return dict(cfg=cfg, world=world, engine=engine, truth=tr, works=works[0], fails=fails[0])
    raise AssertionError("no instance for the table")


def _rows(s):
    """Deliverables and what each region's two checks must say. The base is the oracle's answer."""
    tr, a, b = s["truth"], s["works"], s["fails"]
    base = oracle.answer(s["cfg"], s["world"])
    assert base[a]["method"] == "fat_sat" and base[b]["method"] == "stir"
    with_ = lambda **kw: {**base, **kw}                                    # noqa: E731
    return [
        ("oracle", base, {}),
        ("stir where fat saturation works", with_(**{a: dict(method="stir", ti_ms=tr[a]["null_ms"])}), {f"method_{a}": False, f"suppressed_{a}": True}),
        ("fat saturation where it fails", with_(**{b: dict(method="fat_sat")}), {f"suppressed_{b}": False, f"method_{b}": True}),
        ("null missed by a quarter", with_(**{b: dict(method="stir", ti_ms=round(1.25 * tr[b]["null_ms"], 1))}), {f"suppressed_{b}": False}),
        ("null of another region", with_(**{b: dict(method="stir", ti_ms=round(tr[b]["null_ms"] * 0.8, 1))}), {f"suppressed_{b}": False}),
    ]


def _verdicts(s, rows):
    out = {}
    for name, answer, _ in rows:
        r = task.score(s["cfg"], answer, s["world"], s["engine"])
        out[name] = dict(r["checks"], passed=r["passed"])
    return out


def test_scoring_table(mixed):
    rows = _rows(mixed)
    got = _verdicts(mixed, rows)
    for name, _, want in rows:
        assert {k: got[name][k] for k in want} == want, (name, got[name])
        assert got[name]["passed"] == (name == "oracle"), (name, got[name])


def _mutants(s):
    a, b = s["works"], s["fails"]

    def method_rule_dropped(mp):                                          # stir passes wherever it nulls
        mp.setattr(task, "FAT_SAT_SUFFICIENT", -1.0)
        s["cfg"]["task"]["fat_sat_sufficient"] = -1.0

    def threshold_inverted(mp):                                           # "at least" instead of "at most"
        real = task.residuals
        mp.setattr(task, "residuals", lambda *x, **k: {i: dict(r, delivered=1.0 - r["delivered"]) for i, r in real(*x, **k).items()})

    def first_region_only(mp):
        real = task.residuals
        first = sorted([a, b], key=lambda i: int(i[1:]))[0]
        mp.setattr(task, "residuals", lambda *x, **k: {i: r for i, r in real(*x, **k).items() if i == first})

    def best_region_instead_of_every(mp):
        real = task.F.result
        mp.setattr(task.F, "result", lambda checks, *x, **k: real(
            dict(deliverable_valid=checks["deliverable_valid"], any_region=any(v for n, v in checks.items() if n.startswith("suppressed_"))), *x, **k))

    def fat_sat_not_scanned(mp):                                          # fat saturation is taken to work everywhere
        real = task.residuals
        mp.setattr(task, "residuals", lambda cfg, answer, *x, **k: {
            i: dict(r, delivered=0.0 if answer[i]["method"] == "fat_sat" else r["delivered"]) for i, r in real(cfg, answer, *x, **k).items()})

    def unsuppressed_reference_dropped(mp):                               # compares with the suppressed scan itself
        real = task.residuals
        mp.setattr(task, "residuals", lambda *x, **k: {i: dict(r, delivered=min(r["delivered"], 0.01)) for i, r in real(*x, **k).items()})

    return [("method rule dropped", method_rule_dropped, ("stir where fat saturation works",)),
            ("threshold inverted", threshold_inverted, ("oracle", "null missed by a quarter")),
            ("first region only", first_region_only, ("stir where fat saturation works", "fat saturation where it fails", "null missed by a quarter")),
            ("best region instead of every", best_region_instead_of_every, ("fat saturation where it fails", "null missed by a quarter")),
            ("fat saturation not scanned", fat_sat_not_scanned, ("fat saturation where it fails",)),
            ("unsuppressed reference dropped", unsuppressed_reference_dropped, ("null missed by a quarter", "fat saturation where it fails"))]


def test_scoring_mutants_are_killed(mixed, monkeypatch):
    rows = {r[0]: r for r in _rows(mixed)}
    truth = _verdicts(mixed, list(rows.values()))
    killed = []
    for name, patch, exposing in _mutants(mixed):
        keep = mixed["cfg"]["task"]["fat_sat_sufficient"]
        with monkeypatch.context() as mp:
            patch(mp)
            got = _verdicts(mixed, [rows[e] for e in exposing])
        mixed["cfg"]["task"]["fat_sat_sufficient"] = keep
        if any(got[e]["passed"] != truth[e]["passed"] for e in exposing):
            killed.append(name)
    assert len(killed) >= 5, killed
    assert {"method rule dropped", "threshold inverted", "fat saturation not scanned", "best region instead of every"} <= set(killed)


def test_deliverables_the_console_cannot_evaluate_are_refused_and_the_episode_goes_on():
    env = mrgym.make(ENV, level=1, seed=3)
    env.reset()
    ids = [r["id"] for r in env.episode_config()["task"]["regions"]]
    good = {i: dict(method="fat_sat") for i in ids}
    for bad in ({**good, ids[0]: dict(method="stir")}, {**good, ids[0]: dict(method="stir", ti_ms=0.2)}, {**good, ids[0]: dict(method="dixon")},
                {**good, ids[0]: dict(method="fat_sat", ti_ms=200.0)}, {i: good[i] for i in ids[1:]}, {**good, "R99": dict(method="fat_sat")},
                {**good, ids[0]: dict(method="stir", ti_ms=1e5)}, {**good, ids[0]: "fat_sat"}):
        step = env.step({"name": "submit", "arguments": bad})
        assert step.observation.is_error and not step.terminated, bad
    assert task.check(env.episode_config(), {**good, ids[0]: dict(method="stir", ti_ms=200.0)}, env.hidden_world()["console"]) is None


def test_statement_prints_every_number_and_hides_the_world():
    env = mrgym.make(ENV, level=2, seed=1)
    text = env.reset()[0].text
    t = env.episode_config()["task"]
    for number in (100 * t["fat_max"], 100 * t["fat_sat_sufficient"], t["target_scan"]["tr_ms"], t["target_scan"]["matrix"]):
        assert f"{number:g}" in text, number
    assert len(t["regions"]) == task.QA["n_oil"] and all(r["id"] in text for r in t["regions"])
    assert not {"seed", "b0_hz", "oil_t1_jitter", "cls"} & set(str(t).replace("'", " ").split())
    assert mrgym.spec(ENV).kind == "admitted" and set(mrgym.card(ENV)) >= {"task_kind", "task_kind_why", "controls", "levels", "reward"}
    assert set(task.COMPETENCIES) <= set(coverage.ids()) and mrgym.card(ENV)["competencies"] == list(task.COMPETENCIES)


# ------------------------------------------------------------------------------------------------ the hidden world
def test_the_hidden_world_is_make_world_of_the_episode_config_and_nothing_else():
    """Everything hidden comes out of ``W.make_world(seed, world_spec)``: the world the episode is played on is the
    world a verifier rebuilds from the episode config (no hook completes it afterwards)."""
    assert not hasattr(task, "prepare_world") and not hasattr(task, "make_world")
    for level in (1, 2, 3):
        env = mrgym.make(ENV, level=level, seed=7)
        env.reset()
        cfg, live = env.episode_config(), env.hidden_world()
        again = W.make_world(cfg["seed"], cfg["world_spec"])
        assert again["system"] == live["system"] and again["tissues"] == live["tissues"]
        assert all(np.array_equal(again["fields"][k], live["fields"][k]) for k in ("b0_hz", "b1", "rx"))
        env.close()


def test_every_fat_vial_has_its_own_t1_and_the_pulse_band_is_drawn_where_the_level_says():
    table = tissue.table(3.0)["subcut_fat"]["t1_ms"]
    nominal = 1.3e-6 * W.GAMMA_HZ_PER_T * 3.0
    seen = {}
    for level in (1, 2):
        for seed in (7, 8, 9):
            env = mrgym.make(ENV, level=level, seed=seed)
            env.reset()
            cfg, world = env.episode_config(), env.hidden_world()
            j = cfg["world_spec"]["qa"]["oil_t1_jitter"]
            t1 = sorted(c["t1_ms"] for c in world["tissues"].values() if c["fat"])
            assert len(t1) == task.QA["n_oil"] and len(set(t1)) == len(t1) and all(table * (1 - j) <= v <= table * (1 + j) for v in t1)
            seen[level, seed] = (t1, world["system"]["fatsat_bw_hz"])
            env.close()
    assert seen[1, 7][0] != seen[1, 8][0]                                 # another world, other fats
    assert all(seen[1, s][1] == pytest.approx(nominal) for s in (7, 8, 9))            # level 1: the nominal band (key absent)
    lo, hi = task.LEVELS[2]["world_spec"]["fatsat_bw_hz"]
    bands = [seen[2, s][1] for s in (7, 8, 9)]
    assert all(lo <= b <= hi for b in bands) and len(set(bands)) == 3 and all(abs(b - nominal) > 1e-6 for b in bands)


def test_world_key_fatsat_bw_hz_is_additive_and_scales_with_the_field():
    """``fatsat_bw_hz`` in a world spec: absent, the world is the one built without the key (bit for bit); a number is
    the band at 3 T; a range is drawn from a sub-stream of its own, so nothing else in the world moves."""
    seed = "0123456789abcdef0123456789abcdef"
    base = dict(kind="qa", b0_t=3.0, qa=dict(n_oil=2), n=160)
    plain, ranged, fixed = (W.make_world(seed, dict(base, **kw)) for kw in ({}, dict(fatsat_bw_hz=(60.0, 330.0)), dict(fatsat_bw_hz=120.0)))
    assert plain["system"]["fatsat_bw_hz"] == pytest.approx(1.3e-6 * W.GAMMA_HZ_PER_T * 3.0)
    assert fixed["system"]["fatsat_bw_hz"] == 120.0 and 60.0 <= ranged["system"]["fatsat_bw_hz"] <= 330.0
    for other in (ranged, fixed):
        assert {k: v for k, v in other["system"].items() if k != "fatsat_bw_hz"} == {k: v for k, v in plain["system"].items() if k != "fatsat_bw_hz"}
        assert other["tissues"] == plain["tissues"] and np.array_equal(other["fields"]["b0_hz"], plain["fields"]["b0_hz"])
        assert np.array_equal(other["weights"], plain["weights"])
    low = W.make_world(seed, dict(base, b0_t=1.5, fatsat_bw_hz=120.0))
    assert low["system"]["fatsat_bw_hz"] == 60.0                          # the chemical shift it covers halves with the field
    assert W.make_world(seed, dict(base, fatsat_bw_hz=(60.0, 330.0)))["system"]["fatsat_bw_hz"] == ranged["system"]["fatsat_bw_hz"]
    layout = W.make_world(seed, dict(base, fatsat_bw_hz=(60.0, 330.0)), layout_only=True)
    assert layout["system"] == ranged["system"]


def test_physics_the_saturated_band_has_the_width_of_the_hidden_pulse():
    """Physics check of the key: fat in a uniform field, the saturation pulse swept across the fat line with the
    console's ``fatsat_offset_hz``. The fat is saturated while the pulse covers its line; the width of that range at
    half depth is the hidden bandwidth of the pulse, and the depth is the hidden efficiency."""
    from mrgym.scanner import console
    disc = lambda y, x: np.hypot(y, x) - 60.0                             # noqa: E731
    for band in (90.0, 166.0, 280.0):
        world = W.flat_world(W.shapes_phantom([("subcut_fat", disc)], n=96), fatsat_bw_hz=band, fatsat_eff=0.9)
        engine = kspace.Engine(world)
        c = world["names"].index("subcut_fat")
        centre = np.argmax(world["weights"][c][engine.sup[c]])
        amp = lambda **kw: abs(engine.amplitudes(console.plan(dict(sequence="se", tr_ms=500.0, matrix=32, **kw), world["console"])["realised"], c)[0][centre])   # noqa: E731
        plain = amp()
        offsets = np.arange(-250.0, 250.5, 2.0)
        left = np.array([amp(fat_sat=True, fatsat_offset_hz=float(o)) for o in offsets]) / plain
        assert left.min() == pytest.approx(1.0 - 0.9, abs=2e-3) and left[[0, -1]].min() > (0.95 if band < 200.0 else 0.6)
        inside = offsets[left < 1.0 - 0.45]                               # half the depth of the saturation
        assert inside.max() - inside.min() == pytest.approx(band, abs=4.0), band
        assert abs(inside.max() + inside.min()) <= 4.0                    # centred on the fat line


# ------------------------------------------------------------------------------------------------ the policy's model
def test_null_model_carries_a_fit_over_to_another_repetition_time():
    for t1, k in ((300.0, 0.95), (480.0, 0.82)):
        flip, te, tr_p, tr_t = np.pi / 2, 14.0, 700.0, 2400.0
        tis = [60.0, 130.0, 280.0, 600.0]
        vals = [oracle.inversion_recovery(flip, tr_p, te, ti, t1, k) for ti in tis]
        a, b, fit_t1 = controls._fit(tis, vals)
        got = controls._null_at(a, b, fit_t1, tr_p, tr_t, te)
        lo, hi = 10.0, 1500.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if oracle.inversion_recovery(flip, tr_t, te, mid, t1, k) < 0 else (lo, mid)
        assert abs(fit_t1 / t1 - 1) < 0.03 and abs(got - lo) < 4.0, (t1, k, got, lo)


def test_oracle_closed_form_agrees_with_the_verifier_scans(mixed):
    answer = oracle.answer(mixed["cfg"], mixed["world"])
    res = task.residuals(mixed["cfg"], answer, mixed["world"], mixed["engine"])
    for i, row in mixed["truth"].items():
        assert abs(res[i]["fat_sat"] - row["fat_sat_left"]) <= 0.03 + 0.15 * row["fat_sat_left"], (i, res[i], row)
        assert res[i]["delivered"] <= task.FAT_MAX


# ------------------------------------------------------------------------------------------------ controls
@pytest.mark.parametrize("level, seed", [(1, 0), (3, 1)])
def test_reference_passes_replays_and_the_oracle_answer_passes(level, seed):
    """The reference passes; the log replays against its head on a world rebuilt from the episode config; the
    deliverable scores the same from the config alone (what ``mrgym verify``, the service and an exported task do);
    the oracle's own answer passes too."""
    env = mrgym.make(ENV, level=level, seed=f"t:{seed}")
    r = policies.run(env, "reference")
    assert r.info["policy_error"] is None and r.reward == 1.0, r.info["score"]
    cfg, world, engine = env.episode_config(), env.hidden_world(), env._session.engine
    answer = env.transcript()[-1]["call"]["arguments"]
    rep = core.replay(cfg, env.scanner_log(), head=env._session.receipt(with_key=True))
    assert rep["ok"] and rep["sealed"] and not rep["over_budget"] and rep["scans"] == env._session.n_scans
    alone = task.score(cfg, answer)                                       # world and engine from the config alone
    assert alone["passed"] and alone["residual"] == r.info["score"]["residual"]
    assert alone["checks"] == {k: v for k, v in r.info["score"]["checks"].items() if k not in F.GATES}
    assert task.score(cfg, oracle.answer(cfg, world), world, engine)["passed"]
    env.close()


def test_task_kind_follows_the_stored_gate(monkeypatch):
    """Measurement or decision task, per level: read from the stored gate (open_loop against the reference), never typed."""
    from mrgym import gate
    level = lambda ref, open_loop: dict(n=60, reference=dict(rate=ref), reported=[dict(policy="open_loop", rate=open_loop)], ablations=[])   # noqa: E731
    fake = dict(engine=mrgym.ENGINE_ID, levels={"1": level(1.0, 1.0), "2": level(1.0, 0.98), "3": level(0.98, 0.70)})
    kind = task.task_kind(fake)
    assert kind["levels"] == {"1": "measurement", "2": "measurement", "3": "decision"} and kind["kind"] == "mixed"
    assert "0 at level 1, 2 at level 2, 28 at level 3" in kind["evidence"] and "n = 60 per level" in kind["evidence"]
    monkeypatch.setattr(gate, "load", lambda env_id: None)
    card = task.card()
    assert card["task_kind"] == "undetermined" and "not determined" in card["summary"]
    monkeypatch.setattr(gate, "load", lambda env_id: dict(fake, levels={k: level(1.0, 1.0) for k in "123"}))
    card = task.card()
    assert card["task_kind"] == "measurement" and "level 3 is a measurement task" in card["summary"]


def test_shortcuts_fail_and_same_seed_same_episode():
    env = mrgym.make(ENV, level=2, seed=5)
    assert policies.run(env, "prior").reward == 0.0 and not env._session.n_scans
    runs = [policies.run(mrgym.make(ENV, level=2, seed=5), "fat_sat_only") for _ in range(2)]
    assert runs[0].info["score"]["residual"] == runs[1].info["score"]["residual"]
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert set(controls.DESCRIPTIONS) == set(controls.policies())
    cfg, exam = env.episode_config(), env.hidden_world()["console"]
    blind = controls.blind_answers(cfg["task"])
    assert len(blind) == 11 and all(task.check(cfg, a, exam) is None for a in blind.values())


def test_tune_returns_a_deliverable_recipe():
    from mrgym import gate
    spec = mrgym.spec(ENV)
    cfgs = [gate._instance(spec, 2, f"tune-test:{i}") for i in range(4)]
    recipe = controls.tune(cfgs)
    answer = controls.recipe_answer(recipe, cfgs[0]["task"])
    assert task.check(cfgs[0], answer, dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)) is None
    assert 0.0 <= recipe["tuned_pass_rate"] <= 1.0
