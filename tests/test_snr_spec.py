"""``Scanner/SNRSpec-v0``: the noise formula, the oracle, the scoring rule and its mutants, the instances, the controls."""
import ast
import inspect
import json

import numpy as np
import pytest

import mrgym
from mrgym import policies
from mrgym.envs import family as F
from mrgym.envs import oracle as O
from mrgym.envs.snr_spec import controls, oracle, report, task
from mrgym.scanner import console, kspace
from mrgym.scanner import world as W

ENV = "Scanner/SNRSpec-v0"


@pytest.fixture(scope="module")
def instance():
    """One level-3 instance with its hidden world, engine and optima (verifier side)."""
    env = mrgym.make(ENV, level=3, seed="test:1")
    env.reset()
    cfg = env.episode_config()
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    engine = kspace.Engine(world)
    return cfg, world, engine, task.optima(cfg, world, engine)


def _fields(best):
    return {k: {f: v[f] for f in task.FIELDS} for k, v in best.items()}


# ------------------------------------------------------------------ noise
@pytest.mark.parametrize("protocol", report.NOISE_PROTOCOLS, ids=lambda p: f"{p['sequence']}{p['matrix']}")
def test_the_noise_formula_matches_repeated_scans(qa_world, qa_engine, protocol):
    """``Engine.noise_sd`` (what the verifier divides by) against the scatter of 64 repeated scans, within 3 %; the
    law the control policies scale noise with is the same function of the realised protocol."""
    r = report.noise_check(qa_world, qa_engine, protocol)
    assert r["measured"] == pytest.approx(r["formula"], rel=0.03)
    assert r["policy_law"] == pytest.approx(r["formula"], rel=1e-9)


def test_the_background_of_a_magnitude_image_gives_the_noise(qa_world, qa_engine):
    """What the reference policy relies on: mean of the magnitude in the corners of its noise scan = 1.2533 SD."""
    from mrgym.scanner import session
    rp = console.plan(controls.NOISE_SCAN, qa_world["console"])["realised"]
    img = qa_engine.acquire(rp, np.random.default_rng(3))
    rows = session.roi_readouts(img, controls._corners(controls.NOISE_SCAN["fov_mm"], controls.NOISE_DISC_MM), session.fov_yx(rp))
    assert min(r["n_pixels"] for r in rows) > 8000
    sd = np.mean([r["mean_magnitude"][0] for r in rows]) / controls.RAYLEIGH_MEAN
    assert sd == pytest.approx(qa_engine.noise_sd(rp), rel=0.01)
    wrong = np.mean([r["sd_magnitude"][0] for r in rows])                # the unit error of ``wrong_units``
    assert wrong == pytest.approx(controls.RAYLEIGH_SD * qa_engine.noise_sd(rp), rel=0.02)


# ------------------------------------------------------------------ oracle
def test_the_oracle_is_independent_of_the_control_policies():
    assert O.independence_problems("mrgym.envs.snr_spec") == []
    for module in (oracle, task):
        tree = ast.parse(inspect.getsource(module))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name for n in top for a in n.names} | {n.module or "" for n in top if isinstance(n, ast.ImportFrom)}
        assert not any("controls" in n for n in names), module.__name__
    assert "oracle" not in "".join(inspect.getsource(f) for f in (controls._measure, controls._solve, controls.reference))


def test_the_oracle_meets_the_shared_contract(instance):
    """``mrgym.envs.oracle``: ``solve(cfg, world)`` returns T*, a deliverable the verifier passes at regret 0, and the
    nearest failing answers; the reference's regret against it is measured through the tools."""
    cfg, world, engine, best = instance
    sol = oracle.solve(cfg, world)
    assert oracle.REGRET_KIND == "ratio" and sol["method"] == "exhaustive" and sol["seconds"] >= 0
    assert sol["value"] == pytest.approx(sum(v["scan_s"] for v in best.values())) and sol["argmin"] == _fields(best)
    s = task.score(cfg, sol["argmin"], world=world)
    assert s["passed"] and s["error"] == pytest.approx(sol["value"]) and s["regret"] == pytest.approx(0.0, abs=1e-9)
    assert sorted(c["fails"] for c in sol["certificate"]) == ["snr_A", "snr_B"]
    for c in sol["certificate"]:                                         # each certificate fails the check it names, and only that one
        answer = dict(sol["argmin"], **c["deliverable"])
        checks = task.score(cfg, answer, world, engine)["checks"]
        assert [k for k, ok in checks.items() if not ok] == [c["fails"]]
    rep = O.policy_regret(ENV, 3, ["test:3"])
    assert rep["policy"] == "reference" and rep["kind"] == "ratio" and rep["oracle_passes"] == 1 and rep["passed"] == 1
    assert rep["regret"]["negative_passing"] == 0 and 0.0 <= rep["regret"]["max"] <= task.TIME_FACTOR - 1.0


def test_the_oracle_optimum_passes_the_verifier_and_is_tight(instance):
    cfg, world, engine, best = instance
    good = _fields(best)
    s = task.score(cfg, good, world, engine)
    assert s["passed"] and all(d["time_ratio"] == pytest.approx(1.0) for d in s["images"].values())
    for name, d in s["images"].items():                                 # the model of the search agrees with the scan
        assert 0.0 <= d["snr_margin"] <= 2.5 * oracle.MARGIN
        shorter = dict(good, **{name: dict(good[name], tr_ms=round(good[name]["tr_ms"] * 0.985, 2))})
        s2 = task.score(cfg, shorter, world, engine)
        assert not s2["checks"][f"snr_{name}"] and s2["checks"][f"time_{name}"]


def test_no_neighbouring_protocol_beats_the_oracle_on_the_engine(instance):
    """Independent of the oracle's model: for other numbers of averages and other bandwidths, bisect the repetition
    time on noise-free engine scans until the requirement is just met; none is faster than the oracle's optimum."""
    cfg, world, engine, best = instance
    name = "A"
    opt, snr_min = best[name], cfg["task"]["images"][name]["snr_min"]

    def shortest(averages, bw):
        ok = lambda tr: min(task.measure(cfg, name, dict(tr_ms=tr, averages=averages, bandwidth_hz_per_px=bw), world, engine)["snr"].values()) >= snr_min  # noqa: E731
        lo, hi = 50.0, 12000.0
        if not ok(hi):
            return float("inf")
        for _ in range(9):
            mid = float(np.sqrt(lo * hi))
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        return task.measure(cfg, name, dict(tr_ms=hi, averages=averages, bandwidth_hz_per_px=bw), world, engine)["scan_s"]

    n, bw = opt["averages"], opt["bandwidth_hz_per_px"]
    for averages, band in ((n, bw), (n + 1, bw), (max(1, n - 1), bw), (n, bw * 1.3), (n, max(50.0, bw / 1.3))):
        assert shortest(averages, band) >= opt["scan_s"] * 0.995, (averages, band)


def test_the_search_is_exhaustive_on_its_own_table(instance):
    cfg, world, engine, best = instance
    rng = np.random.default_rng(0)
    for name in task.IMAGES:
        img = task.image_of(cfg["task"], name)
        table = oracle.snr_table(world, img, engine)
        opt = oracle.optimum(table, img)
        assert opt["scan_s"] == best[name]["scan_s"]
        for _ in range(4000):
            n, bw = int(rng.integers(1, 20)), float(rng.choice(oracle.BANDWIDTHS))
            tr = float(np.exp(rng.uniform(np.log(60.0), np.log(8000.0))))
            scan_s = 3.0 + tr * oracle.cycles(table, n) / 1000.0
            if scan_s < opt["scan_s"] * 0.999:
                assert oracle.snr_of(table, tr, n, bw).min() < img["snr_min"] * (1 + oracle.MARGIN)
        one = oracle.optimum(table, img, averages=[1])
        narrow = oracle.optimum(table, img, bandwidths=[oracle.BANDWIDTHS[0]])
        assert (one is None or one["scan_s"] >= opt["scan_s"]) and narrow["scan_s"] >= opt["scan_s"]


# ------------------------------------------------------------------ scoring and its mutants
def _verdicts(cfg, world, engine, best):
    good = _fields(best)
    a, b = good["A"], good["B"]
    cases = dict(
        optimum=good,
        too_little_signal=dict(good, A=dict(a, tr_ms=round(a["tr_ms"] * 0.8, 1))),
        too_slow=dict(good, B=dict(b, averages=b["averages"] * 2)),
        slightly_slow=dict(good, B=dict(b, tr_ms=round(b["tr_ms"] * 1.10, 1))),
        much_too_slow=dict(good, B=dict(b, tr_ms=round(b["tr_ms"] * 1.25, 1))),
    )
    return {k: task.score(cfg, v, world, engine)["passed"] for k, v in cases.items()}


EXPECTED = dict(optimum=True, too_little_signal=False, too_slow=False, slightly_slow=True, much_too_slow=False)


def test_the_pass_rule(instance):
    assert _verdicts(*instance) == EXPECTED


def _swap_regions(t, name):
    other = [k for k in t["images"] if k != name][0]
    return [r for r in t["regions"] if r["id"] in t["images"][other]["regions"]]


MUTANTS = dict(
    requirement_sign=("meets", lambda snr, snr_min: snr <= snr_min),
    time_ratio_inverted=("in_time", lambda scan_s, shortest_s, factor: shortest_s <= factor * scan_s),
    time_rule_dropped=("in_time", lambda scan_s, shortest_s, factor: True),
    wrong_region=("regions_scored", _swap_regions),
)


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_a_scoring_mutant_is_killed(instance, monkeypatch, mutant):
    """Each mutation of the verifier changes at least one verdict that ``test_the_pass_rule`` pins."""
    name, fn = MUTANTS[mutant]
    monkeypatch.setattr(task, name, fn)
    assert _verdicts(*instance) != EXPECTED


def test_the_measurement_gate_and_its_mutant(instance, monkeypatch):
    """The optimum delivered without a scan earns nothing; with the gate dropped (the mutant) it would."""
    cfg, world, engine, best = instance

    def play():
        env = mrgym.make(ENV, level=3, seed="test:1", reveal_score=True)
        env.reset()
        return env.step({"name": "submit", "arguments": _fields(best)})
    r = play()
    assert r.terminated and r.reward == 0.0 and not r.info["score"]["checks"]["measured"]
    assert all(v for k, v in r.info["score"]["checks"].items() if k != "measured")
    monkeypatch.setattr(F, "GATES", ())
    assert play().reward == 1.0


# ------------------------------------------------------------------ instances and the interface
def test_instances_are_reproducible_and_hide_the_world():
    a, b = (task.make_instance("0f" * 16, 2, task_id="x") for _ in range(2))
    assert a == b and a != task.make_instance("1f" * 16, 2, task_id="x")
    text = json.dumps(a["task"]) + task.statement(a)
    assert not any(k in text for k in ("t1_ms", "t2_ms", "sigma0", "tx_gain", "cls", a["seed"]))
    for img in a["task"]["images"].values():
        assert f"SNR at least {img['snr_min']:g}" in task.statement(a) and f"matrix {img['scan']['matrix']}" in task.statement(a)
    assert f"{task.TIME_FACTOR:g} times the shortest" in task.statement(a)
    regions = [i for img in a["task"]["images"].values() for i in img["regions"]]
    assert sorted(regions) == sorted(r["id"] for r in a["task"]["regions"]) and len(set(regions)) == 4


def test_the_levels_are_different_regimes():
    """On the material values: level 1 needs one average, level 2 several, level 3 a bandwidth above the minimum."""
    stats = {}
    for lv in task.LEVELS:
        rows = []
        for k in range(16):
            cfg = task.make_instance(f"{k:02x}" * 16, lv)
            world = W.make_world(cfg["seed"], cfg["world_spec"], layout_only=True)
            rows += [oracle.shortest(world, task.image_of(cfg["task"], name)) for name in task.IMAGES]
        stats[lv] = (np.array([r["averages"] for r in rows]), np.array([r["bandwidth_hz_per_px"] for r in rows]),
                     np.array([r["scan_s"] for r in rows]))
    assert (stats[1][0] == 1).all() and (stats[2][0] >= 2).mean() > 0.9 and (stats[3][1] > 60.0).mean() > 0.4
    assert (stats[1][1] < 60.0).all() and (stats[2][1] < 60.0).all()
    for lv in task.LEVELS:
        assert task.SCAN_S_RANGE[0] <= stats[lv][2].min() and stats[lv][2].max() <= task.SCAN_S_RANGE[1]


def test_a_deliverable_the_console_cannot_run_is_refused_and_the_episode_continues():
    env = mrgym.make(ENV, level=1, seed="test:2")
    env.reset()
    ok = dict(tr_ms=500.0, averages=1, bandwidth_hz_per_px=50.0)
    for bad in (dict(A=ok), dict(A=ok, B=dict(ok, tr_ms=0.5)), dict(A=ok, B=dict(ok, averages=0)), dict(A=ok, B=dict(ok, flip_deg=90.0)),
                dict(A=ok, B=dict(ok, bandwidth_hz_per_px=5.0))):
        r = env.step({"name": "submit", "arguments": bad})
        assert r.observation.is_error and not r.terminated and r.reward == 0.0
    cfg = env.episode_config()
    assert "console rejects the scan of image B" in task.check(cfg, dict(A=ok, B=dict(ok, tr_ms=20.0)), env.hidden_world()["console"])
    assert task.check(cfg, dict(A=ok, B=ok), env.hidden_world()["console"]) is None
    r = env.step({"name": "submit", "arguments": dict(A=ok, B=ok)})
    assert r.terminated and r.reward == 0.0


# ------------------------------------------------------------------ the control policies
def test_the_reference_passes_with_small_regret_and_the_shortcuts_fail():
    r = policies.run(mrgym.make(ENV, level=3, seed="test:3"), "reference")
    s = r.info["score"]
    assert r.info["policy_error"] is None and r.reward == 1.0 and r.info["replay"]["ok"]
    assert -0.002 <= s["regret"] <= task.TIME_FACTOR - 1.0
    assert s["error"] == pytest.approx(sum(d["scan_s"] for d in s["images"].values()))
    assert r.info["budget"]["used_s"] <= r.info["budget"]["budget_s"]
    floor = policies.run(mrgym.make(ENV, level=3, seed="test:3"), "prior")
    assert floor.reward == 0.0 and not floor.info["score"]["checks"]["measured"]
    unit = policies.run(mrgym.make(ENV, level=3, seed="test:3"), "wrong_units")
    assert unit.info["policy_error"] is None and unit.reward == 0.0
    assert not all(unit.info["score"]["checks"][f"snr_{k}"] for k in task.IMAGES)
    const = policies.run(mrgym.make(ENV, level=3, seed="test:3"), "fixed_recipe")
    assert const.info["policy_error"] is None and const.reward == 0.0 and const.info["budget"]["scans"] == 1


def test_the_tuned_blind_policies_deliver_runnable_protocols(monkeypatch):
    monkeypatch.setattr(controls, "TUNE_EXTRA", 1)
    cfgs = [task.make_instance(f"{k:02x}" * 16, 2) for k in range(6)]
    recipe = controls.tune(cfgs)
    assert recipe["tuned_on_worlds"] == 12 and 0.0 <= recipe["tuned_joint_pass_rate"] <= 1.0
    table = controls.spec_table(2, 24)
    exam = dict(b0_t=1.5)
    for name in task.IMAGES:
        assert set(table["protocols"][name]) == {f"{m}:{b}" for m in task.MATRICES[name] for b in range(controls.SNR_BINS)}
        for fields in [recipe[name], *table["protocols"][name].values()]:
            assert console.plan(dict(task.SCAN, matrix=task.MATRICES[name][-1], **{k: fields[k] for k in task.FIELDS}), exam)["accept"]


def test_the_card_is_complete_and_claims_nothing_unmeasured():
    card = mrgym.card(ENV)
    assert card["kind"] == mrgym.spec(ENV).kind and {c["name"] for c in card["controls"]} == set(controls.policies())   # the kind is the registered one
    assert [c["name"] for c in card["controls"] if c["must"] == "pass"] == [controls.REFERENCE]
    assert set(controls.DESCRIPTIONS) == set(controls.policies()) and set(controls.LADDER) == set(task.LEVELS)
    if card["gate"] is None:
        assert card["task_kind"]["kind"] == "undetermined" and "not determined" in card["summary"]
    fake = dict(engine="x", levels={"1": dict(n=10, reference=dict(rate=1.0), reported=[dict(policy="open_loop", rate=0.9)]),
                                    "2": dict(n=10, reference=dict(rate=1.0), reported=[dict(policy="open_loop", rate=0.6)])})
    assert task.task_kind(fake)["kind"] == "decision" and "decision task" in task.task_kind(fake)["sentence"]
    fake["levels"]["2"]["reported"][0]["rate"] = 0.8
    assert task.task_kind(fake)["kind"] == "measurement" and "20 at level 2" in task.task_kind(fake)["sentence"]
