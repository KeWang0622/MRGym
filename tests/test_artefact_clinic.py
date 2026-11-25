"""``Scanner/ArtefactClinic-v0``: the twins, the generator's guarantees, the pass rule (every component is killed by a
mutant of the scoring code), the control policies on one instance, the bank and the sheet-classifier check."""
import copy
import importlib.util
import inspect
import os
import types

import numpy as np
import pytest

import mrgym
from mrgym import policies
from mrgym.env import derive_seed
from mrgym.envs.artefact_clinic import bank, controls, metrics, task
from mrgym.envs.artefact_clinic import catalogue as C
from mrgym.envs.artefact_clinic import generator as G
from mrgym.scanner import console, kspace
from mrgym.scanner import world as W

ID = "Scanner/ArtefactClinic-v0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "mrgym.envs.artefact_clinic"


def _cfg(level, seed):
    return task.make_instance(derive_seed(ID, level, "train", seed, environ={}), level, task_id="t")


@pytest.fixture(scope="module")
def case():
    """A level-2 instance with a cause the twins switch (not truncation alone), its world and its verified remedy."""
    for seed in range(12):
        cfg = _cfg(2, seed)
        causes = {c for v in cfg["hidden"]["causes"].values() for c in v}
        if causes - {"truncation_ringing"} and cfg["hidden"]["remedy"]["averages"] >= 3:
            world = W.make_world(cfg["seed"], cfg["world_spec"])
            good = dict(causes=copy.deepcopy(cfg["hidden"]["causes"]), protocol=dict(cfg["hidden"]["remedy"]))
            return dict(cfg=cfg, world=world, good=good, seed=seed)
    raise AssertionError("no suitable instance among the first training seeds")


# ------------------------------------------------------------------------------------------------ catalogue and twins
def test_every_cause_names_a_switch_the_twins_know_and_controls_are_described():
    assert set(C.MECHANISMS) <= set(kspace.ALL) | set(metrics.OWN)
    assert all(c.mechanism in kspace.ALL and c.label not in C.CATALOGUE for c in C.DISABLED.values())
    for c in C.CATALOGUE.values():
        assert c.absent < c.present and c.clean <= c.present and c.remedy and c.remedies and c.formula and c.experiment
    names = {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert names == set(controls.policies()) == set(controls.DESCRIPTIONS)
    assert all(set(v) <= set(mrgym.spec(ID).levels) for v in controls.ABLATIONS.values())


def test_twins_switch_one_mechanism_each_on_a_hand_made_object():
    """Oil disc in gel at 1.5 T on a coarse master grid: each mechanism is seen by its own twin and by no other."""
    disc = lambda cy, cx, r: (lambda y, x: np.hypot(y - cy, x - cx) - r)    # noqa: E731
    ph = W.shapes_phantom([("muscle", disc(0, 0, 95)), ("subcut_fat", disc(0, 40, 16))], b0_t=1.5, n=320)
    world = W.flat_world(ph)
    tw = metrics.Twins(world, C.MECHANISMS)
    base = dict(sequence="spgr", fov_mm=240.0, matrix=96, tr_ms=150.0, flip_deg=70.0)
    border = dict(y_mm=0.0, x_mm=56.0, radius_mm=3.3)                       # across the +x border of the oil
    arrival = dict(y_mm=0.0, x_mm=61.5, radius_mm=3.3)                      # gel just outside it, where shifted fat lands
    quiet = dict(y_mm=-45.0, x_mm=-30.0, radius_mm=3.3)                     # gel far from everything
    fold = dict(y_mm=-70.0, x_mm=-10.0, radius_mm=3.3)
    period = C.fat_period_ms(1.5)
    shifted = metrics.realise(dict(base, bandwidth_hz_per_px=90.0, te_ms=round(2 * period, 2), ro_polarity=1), world)
    s = tw.solo(shifted, arrival)
    assert s["chemshift"] >= 0.10 and s["fatphase"] < 0.03 and s["wrap"] < 1e-6
    opposed = metrics.realise(dict(base, bandwidth_hz_per_px=1100.0, te_ms=round(1.5 * period, 2)), world)
    s = tw.solo(opposed, border)
    assert s["fatphase"] >= 0.10 and s["wrap"] < 1e-6
    in_phase = metrics.realise(dict(base, bandwidth_hz_per_px=1100.0, te_ms=round(2 * period, 2)), world)
    assert tw.solo(in_phase, border)["fatphase"] < 0.01
    wrapped = metrics.realise(dict(base, bandwidth_hz_per_px=1100.0, te_ms=round(2 * period, 2), phase_fov=0.65), world)
    s = tw.solo(wrapped, fold)
    assert s["wrap"] >= 0.10 and s["chemshift"] < 0.03 and s["fatphase"] < 0.03
    assert metrics.without_wrap(in_phase, world) is in_phase and metrics.without_wrap(wrapped, world)["lines_full"] > wrapped["lines_full"]
    for rp in (shifted, opposed, in_phase, wrapped):                        # a clean place is clean under every protocol
        e = tw.errors(rp, quiet)
        assert e["mech"] < 0.03 and e["trunc"] < 0.03, (rp["bandwidth_hz_per_px"], e)


def test_truth_image_has_the_scale_of_a_scan_and_block_average_keeps_a_flat_image():
    ph = W.shapes_phantom([("muscle", lambda y, x: np.hypot(y, x) - 90)], b0_t=1.5, n=320)
    world = W.flat_world(ph)
    tw = metrics.Twins(world, C.MECHANISMS)
    rp = metrics.realise(dict(sequence="spgr", fov_mm=240.0, matrix=64, tr_ms=150.0, flip_deg=70.0), world)
    m = metrics.area_mask(rp, dict(y_mm=10.0, x_mm=-20.0, radius_mm=12.0))
    assert abs(np.sum(m * tw.image(rp)) / np.sum(m * tw.truth(rp)) - 1.0) < 0.005
    assert abs(np.sum(m) * (240.0 / 64) ** 2 - np.pi * 144.0) < 0.02 * np.pi * 144.0     # area of the circle, in pixels
    fine = metrics.realise(dict(sequence="spgr", fov_mm=240.0, matrix=128, tr_ms=150.0, flip_deg=70.0), world)
    avg = metrics.block_average(tw.image(fine), fine, rp)
    assert avg.shape == tw.image(rp).shape and metrics.change(tw.image(rp), avg, m) < 0.01
    assert metrics.change(2.0 * avg, avg, m, shape_only=True) < 1e-9 and abs(metrics.change(2.0 * avg, avg, m) - 1.0) < 1e-6


# ------------------------------------------------------------------------------------------------ instances
def test_an_instance_is_deterministic_keeps_its_promises_and_hides_the_diagnosis(case):
    cfg, world = case["cfg"], case["world"]
    assert _cfg(2, case["seed"]) == cfg
    from mrgym.envs import family
    assert cfg["seed"] == family.reseed(derive_seed(ID, 2, "train", case["seed"], environ={}), cfg["attempt"]) and "attempt" not in cfg["task"]
    t, h = cfg["task"], cfg["hidden"]
    tw = metrics.Twins(world, h["mechanisms"])
    rp0, rp1 = metrics.realise(t["current_protocol"], world), metrics.realise(h["remedy"], world)
    for f in t["findings"]:
        solo, causes = tw.solo(rp0, f), h["causes"][f["id"]]
        assert G.cause_set(solo) == tuple(sorted(causes))                   # exactly these causes, nothing in between
        assert G.is_clean(tw.errors(rp1, f), G.VERIFY)                      # the remedy cleans it with the margin
    plan = console.plan(h["remedy"], world["console"])
    assert plan["scan_s"] <= t["requirements"]["max_scan_s"] and h["snr_remedy"] >= task.min_snr(cfg) == t["requirements"]["min_snr_reference"]
    assert abs(metrics.snr(tw.engine(), rp1, t["reference_region"]) - h["snr_remedy"]) < 0.2
    assert cfg["budget_s"] < t["current_scan_s"]                            # no full-length repeat of the complaint
    env = mrgym.make(ID, level=2, seed=case["seed"])
    obs, _ = env.reset()
    card = env.step({"name": "scanner_info", "arguments": {"topic": "task"}}).observation
    public = obs.text + card.text + repr(obs.data) + repr(card.data)
    assert "hidden" not in card.data and "solo" not in public and "remedy" not in card.data
    lines = [ln for ln in obs.text.splitlines() if not any(lab in ln for lab in C.LABELS)]
    for word in ("fold", "ghost", "gibbs", "ringing", "cancel", "chemical", "opposed", "displace"):
        assert word not in "\n".join(lines).lower(), word                  # appearance only: no mechanism named outside the vocabulary
    assert not [p for p in mrgym.tools.check_schema(task.submit_schema(cfg), "submit")]


def test_the_wanted_causes_are_drawn_before_any_world_and_uniformly():
    lv = task.LEVELS[2]
    draws = [G.draw_wanted(derive_seed(ID, 2, "train", i, environ={}), lv) for i in range(400)]
    flat = ["+".join(w) or "none" for d in draws for w in d["wanted"]]
    share = {k: flat.count(k) / len(flat) for k in set(flat)}
    assert len(share) == 5 and max(share[k] for k in ("none", "truncation_ringing", "wrap_around")) < 0.30
    fat = share["chemical_shift_displacement"] + share["opposed_phase_cancellation"]
    assert 0.20 < fat < 0.30 and all(len(d["armed"]) == 3 for d in draws)
    assert all(len({c for w in d["wanted"] for c in w} & {"chemical_shift_displacement", "opposed_phase_cancellation"}) <= 1 for d in draws)


def test_a_banked_instance_is_the_generated_instance(monkeypatch):
    seed_hex = derive_seed(ID, 1, "train", 1, environ={})
    if bank.get(seed_hex, 1) is None:
        pytest.skip("no bank line for this seed on this engine")
    stored = copy.deepcopy(bank.get(seed_hex, 1))
    monkeypatch.setattr(bank, "get", lambda seed, level: None)
    task._instance.cache_clear()
    fresh = task._instance(seed_hex, 1)
    task._instance.cache_clear()
    assert fresh == stored


def test_a_stale_bank_line_is_ignored(monkeypatch):
    line = bank.line("ab" * 16, 2, dict(x=1))
    monkeypatch.setattr(bank, "GENERATOR_VERSION", bank.GENERATOR_VERSION + 1)
    import json
    assert json.loads(line)["stamp"] != bank.stamp()


# ------------------------------------------------------------------------------------------------ the pass rule
def _mutant(module, old, new, **globals_):
    """The module with one piece of source replaced, executed under another name."""
    src = inspect.getsource(module)
    assert src.count(old) >= 1, old
    mod = types.ModuleType(module.__name__ + "_mutant")
    mod.__package__ = PKG
    exec(compile(src.replace(old, new, 1), module.__file__, "exec"), mod.__dict__)
    for k, v in globals_.items():
        setattr(mod, k, v)
    return mod


def _wrong(case):
    """Deliverables that each break exactly one component of the pass rule: {failed check: answer}."""
    cfg, good = case["cfg"], case["good"]
    t, h = cfg["task"], cfg["hidden"]
    p, p0 = good["protocol"], t["current_protocol"]
    caused = next(i for i, v in h["causes"].items() if set(v) - {"truncation_ringing"})
    other = next(lab for lab in C.LABELS if lab not in h["causes"][caused])
    period = C.fat_period_ms(t["exam"]["field_t"])
    out = {
        f"diagnosis_{caused}": dict(good, causes=dict(good["causes"], **{caused: [other]})),
        "scan_time": dict(good, protocol=dict(p, averages=p["averages"] + 4)),
        "pixel_size": dict(good, protocol=dict(p, matrix=p["matrix"] - 2)),
        "locked_parameters": dict(good, protocol=dict(p, flip_deg=p["flip_deg"] - 1.0)),
        "te_window": dict(good, protocol=dict(p, te_ms=round(p["te_ms"] + 2 * period, 2))),
        "snr_reference": dict(good, protocol=dict(p, averages=1)),         # at least three averages in the remedy: SNR x 0.58 or less
        f"clean_{caused}": dict(good, protocol=dict(p0)),
    }
    superset = dict(good, causes=dict(good["causes"], **{caused: [*h["causes"][caused], other]}))
    return out, superset, caused


MUTANTS = [   # (what is removed, module, old source, new source, the check that should have failed)
    ("the diagnosis is not compared", "task", "return sorted(set(delivered)) == sorted(truth)", "return True", "diagnosis"),
    ("a superset of the causes is accepted", "task", "return sorted(set(delivered)) == sorted(truth)", "return set(truth) <= set(delivered)", "superset"),
    ("the scan-time limit is dropped", "task", "return scan_s <= max_scan_s + 1e-9", "return True", "scan_time"),
    ("the pixel limit is dropped", "task", "return rp[\"fov_mm\"] / rp[\"matrix\"] <= max_pixel_mm + 1e-9", "return True", "pixel_size"),
    ("locked parameters are not compared", "task", "return all(rp.get(k) == rp0.get(k) for k in locked)", "return True", "locked_parameters"),
    ("the TE window is dropped", "task", "return window[0] - 1e-9 <= value <= window[1] + 1e-9", "return True", "te_window"),
    ("the SNR floor is dropped", "task", "return snr >= min_snr", "return True", "snr_reference"),
    ("the SNR comparison is reversed", "task", "return snr >= min_snr", "return snr <= min_snr", "good"),
    ("the comparison with the mechanism-free twin is dropped", "generator", "return err[\"mech\"] <= margin[0] * CLEAN_MECH and", "return", "clean"),
]


def test_each_wrong_deliverable_fails_exactly_its_own_check(case):
    cfg, world, good = case["cfg"], case["world"], case["good"]
    s = task.score(cfg, good, world=world)
    assert s["passed"] and all(s["checks"].values()) and s["error"] <= max(G.VERIFY) + 1e-9
    wrong, superset, caused = _wrong(case)
    for name, answer in wrong.items():
        s = task.score(cfg, answer, world=world)
        failed = [k for k, ok in s["checks"].items() if not ok]
        assert not s["passed"] and name in failed and {k.split("_")[0] for k in failed} == {name.split("_")[0]}, (name, failed, s["diagnostic"])
    assert not task.score(cfg, superset, world=world)["checks"][f"diagnosis_{caused}"]
    assert task.check(cfg, dict(good, causes={}), world["console"]) and task.check(cfg, dict(good, protocol=dict(good["protocol"], te_ms=0.004)), world["console"])
    assert not task.score(cfg, dict(good, protocol=dict(good["protocol"], te_ms=0.004)), world=world)["passed"]


@pytest.mark.parametrize("what,module,old,new,target", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_every_scoring_mutant_is_killed(case, what, module, old, new, target):
    """Each mutant removes one decision of the verifier; a deliverable that the true rule fails (or passes) must be
    judged otherwise by the mutant. A mutant that survives means that component is not tested."""
    cfg, world, good = case["cfg"], case["world"], case["good"]
    wrong, superset, caused = _wrong(case)
    answer = {"good": good, "superset": superset, "diagnosis": wrong[f"diagnosis_{caused}"], "clean": wrong[f"clean_{caused}"]}.get(target) or wrong[target]
    mod = _mutant(task, old, new) if module == "task" else _mutant(task, "FAMILY = ", "FAMILY = ", G=_mutant(G, old, new))
    truth = task.score(cfg, answer, world=world)["passed"]
    assert truth == (target == "good")
    assert mod.score(cfg, answer, world=world)["passed"] != truth, f"mutant survived: {what}"


# ------------------------------------------------------------------------------------------------ policies
def test_reference_passes_and_the_floor_policies_fail_through_the_tools(case):
    ref = policies.run(mrgym.make(ID, level=2, seed=case["seed"]), "reference")
    assert ref.reward == 1.0 and not ref.info["policy_error"] and ref.info["score"]["checks"]["measured"]
    assert ref.info["budget"]["used_s"] <= ref.info["budget"]["budget_s"]
    for name in ("prior", "wrong_units", "brute_force", "remedy_battery"):
        r = policies.run(mrgym.make(ID, level=2, seed=case["seed"]), name)
        assert r.reward == 0.0, name


def test_the_pixels_the_reference_reads_are_what_read_regions_returns(case):
    env = mrgym.make(ID, level=2, seed=case["seed"])
    env.reset()
    sc = policies.ToolScanner(env)
    t = sc.info()["task"]
    probe = G.experiments(t, lambda p: sc.plan(p))[0]["protocol"]
    r = sc.scan(probe, rois=[])
    img = controls._image(sc, r["scan_index"])
    y, x = metrics.pixel_axes(r["realised"])
    assert (y[1] - y[0]) > 1.0 and (x[1] - x[0]) > 1.0
    f = t["findings"][0]
    iy, ix = int(np.argmin(np.abs(y - f["y_mm"]))), int(np.argmin(np.abs(x - f["x_mm"])))
    circles = [dict(id=f"p{k}", y_mm=float(y[iy + dy]), x_mm=float(x[ix + dx]), radius_mm=1.0) for k, (dy, dx) in enumerate([(0, 0), (1, 0), (0, 1), (-1, -1)])]
    rows = sc.read_regions(r["scan_index"], circles)
    for row, (dy, dx) in zip(rows, [(0, 0), (1, 0), (0, 1), (-1, -1)]):
        assert row["n_pixels"] == 1 and abs(row["mean_magnitude"][0] - img[iy + dy, ix + dx]) <= 1e-6 * img.max()


# ------------------------------------------------------------------------------------------------ the bank tool
def _tool():
    spec = importlib.util.spec_from_file_location("build_clinic_bank", os.path.join(ROOT, "tools", "build_clinic_bank.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_sheet_classifier_finds_a_leak_and_does_not_invent_one():
    tool = _tool()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 600)
    groups = np.arange(600) // 3
    noise = rng.normal(size=(600, 6))
    leaky = np.c_[noise, y + 0.3 * rng.normal(size=600)]
    assert tool.classifier_accuracy(noise, y, groups) < 0.25 + 0.08
    assert tool.classifier_accuracy(leaky, y, groups) > 0.60


def test_the_report_checks_what_the_generator_promises(case):
    tool = _tool()
    inst = dict(seed=case["cfg"]["seed"], task=case["cfg"]["task"], hidden=case["cfg"]["hidden"], rejected=[])
    rep = tool.report([dict(instance=inst, seconds=1.0, from_bank=False, seed=i) for i in range(10)], 2)
    assert rep["checks"]["complaint_present"] and rep["checks"]["benign_absent"] and rep["checks"]["remedy_clean"]
    assert len(tool.features(inst, True)[0]) == len(tool.features(inst, False)[0]) + 4


# ------------------------------------------------------------------------------------------------ level 3 and the battery
def _told_apart(a, b, experiments) -> bool:
    """Two cause sets are told apart by an experiment whose effect is determined for both and differs."""
    return any(C.expected(a, e) is not None and C.expected(b, e) is not None and C.expected(a, e) != C.expected(b, e) for e in experiments)


def test_battery_is_minimal_sufficient():
    """The battery of a level is the set of experiments that diagnoses every served cause set without looking: with
    it any two cause sets change a finding under different experiments, without any one of its experiments two of them
    do not, and an experiment that only confirms (the reversed readout) is not in it. An experiment whose effect on a
    cause is not determined (a ghost copy under an experiment that does not move it) tells nothing apart."""
    for level in (1, 2, 3):
        lv = task.LEVELS[level]
        draw = G.draw_wanted(derive_seed(ID, level, "train", 0, environ={}), lv)
        options = [tuple(o) for o in draw.get("level_options", draw["options"])]
        vocabulary = [lab for lab in lv["causes"] if C.CATALOGUE[lab].mechanism not in lv.get("never", ())]
        battery = C.minimal_battery(vocabulary, options)
        pairs = [(a, b) for i, a in enumerate(options) for b in options[:i]]
        assert all(_told_apart(a, b, battery) for a, b in pairs), (level, battery)
        for e in battery:
            assert not all(_told_apart(a, b, [x for x in battery if x != e]) for a, b in pairs), (level, e)
        assert "polarity" not in battery and set(battery) <= set(C.mechanisms(vocabulary))
        assert all(C.decode({e for e in battery if C.expected(o, e)}, vocabulary) == sorted(o) for o in options)
    assert C.minimal_battery(C.LABELS) == tuple(m for m in C.MECHANISMS)
    assert C.minimal_battery(task.LEVELS[3]["causes"][1:2] + task.LEVELS[3]["causes"][3:]) == ("fatphase", "wrap", "motion", "dropout")


def test_the_adaptive_plan_runs_the_experiment_of_each_look_and_settles_every_finding():
    t = dict(exam=dict(field_t=1.5), causes_vocabulary=[lab for lab in C.LABELS if lab != "truncation_ringing"], diagnosis="by_look",
             current_protocol=dict(phase_fov=0.6, bandwidth_hz_per_px=90.0, te_ms=9.2))
    plans = G.class_plan(t)
    assert plans["edge"][0] == "chemshift" and plans["bright"][:2] == ("wrap", "motion") and plans["dark"] == ("dropout", None, False)
    classes = dict(F1="edge", F2="edge", F3="dark")
    assert G.next_experiments(plans, classes, {}) == ["chemshift", "dropout"]
    flags = dict(chemshift=dict(F1=True, F2=False, F3=True), dropout=dict(F1=True, F2=False, F3=True))
    assert G.next_experiments(plans, classes, flags) == []
    assert G.settle(plans, classes, flags, t["causes_vocabulary"]) == dict(
        F1=["chemical_shift_displacement"], F2=[], F3=["susceptibility_signal_loss"])   # only the experiment of its own look counts
    # a sheet that arms the cancellation (opposed echo, high bandwidth): a dark finding the slice does not change needs the echo
    t3 = dict(t, current_protocol=dict(phase_fov=0.5, bandwidth_hz_per_px=1100.0, te_ms=6.9))
    plans = G.class_plan(t3)
    assert plans["dark"] == ("dropout", "fatphase", False)
    dark = dict(F1="dark", F2="dark")
    assert G.next_experiments(plans, dark, dict(dropout=dict(F1=True, F2=False))) == ["fatphase"]
    done = dict(dropout=dict(F1=True, F2=False), fatphase=dict(F1=True, F2=True))
    assert G.next_experiments(plans, dark, done) == []
    assert G.settle(plans, dark, done, t["causes_vocabulary"]) == dict(F1=["susceptibility_signal_loss"], F2=["opposed_phase_cancellation"])
    bright = dict(F1="bright", F2="bright", F3="bright")
    done = dict(wrap=dict(F1=True, F2=True, F3=False), motion=dict(F1=True, F2=False, F3=False))
    assert G.next_experiments(plans, bright, dict(wrap=done["wrap"])) == ["motion"]
    assert G.settle(plans, bright, done, t["causes_vocabulary"]) == dict(F1=["periodic_motion_ghost"], F2=["wrap_around"], F3=[])


def test_signal_loss_of_the_qa_gradient_follows_the_through_slice_formula():
    """``world_spec["dbdz"]`` gives the QA phantom patches of through-slice gradient G. A gradient echo then keeps
    the engine's dephasing factor of G x slice x TE cycles: exp(-(pi G dz TE)^2 / (4 ln 2)) for a hard pulse, the
    factor of the slice profile (``rf.dephasing_factor``) for a selective one. Without the key nothing changes."""
    from mrgym.scanner import rf
    seed = "000102030405060708090a0b0c0d0e0f"
    base = dict(kind="qa", n=320, b0_t=1.5, b0_hz=0.0, cf_offset_hz=0.0)
    plain = W.make_world(seed, base)
    world = W.make_world(seed, dict(base, dbdz=dict(n=2, peak_hz_per_mm=(6.0, 8.0), sigma_mm=(14.0, 16.0))))
    assert not plain["fields"]["dbdz_hz_per_mm"].any() and np.array_equal(plain["fields"]["b0_hz"], world["fields"]["b0_hz"])
    g = world["fields"]["dbdz_hz_per_mm"]
    iy, ix = np.unravel_index(np.argmax(np.abs(g) * world["masks"]["body"]), g.shape)
    d = world["extent_mm"] / world["n"]
    spot = dict(y_mm=float((iy - world["n"] // 2) * d), x_mm=float((ix - world["n"] // 2) * d), radius_mm=2.0)
    on, off = kspace.Engine(world), kspace.Engine(world, effects=kspace.ALL - {"dropout"})
    for te, dz, excitation in ((8.0, 5.0, "nonselective"), (12.0, 5.0, "nonselective"), (8.0, 5.0, "selective"), (8.0, 2.5, "selective")):
        rp = metrics.realise(dict(sequence="spgr", fov_mm=240.0, matrix=96, tr_ms=40.0, flip_deg=30.0, te_ms=te, slice_mm=dz,
                                  bandwidth_hz_per_px=600.0, excitation=excitation), world)
        m = metrics.area_mask(rp, spot)
        ratio = float(np.sum(m * np.abs(on.acquire(rp, None)[0])) / np.sum(m * np.abs(off.acquire(rp, None)[0])))
        cycles = abs(g[iy, ix]) * dz * te * 1e-3
        tbw, window = on.pulse(rp)
        want = float(np.exp(-(np.pi * cycles) ** 2 / (4 * np.log(2)))) if tbw is None else float(abs(rf.dephasing_factor(np.array([cycles]), tbw, window)[0]))
        assert want < 0.99 and abs(ratio - want) < 0.02, (te, dz, excitation, ratio, want)


@pytest.fixture(scope="module")
def case3():
    cfg = _cfg(3, 0)
    return dict(cfg=cfg, world=W.make_world(cfg["seed"], cfg["world_spec"]))


def test_a_level_3_instance_looks_alike_fits_its_budget_and_the_reference_passes(case3):
    cfg, world = case3["cfg"], case3["world"]
    t, h = cfg["task"], cfg["hidden"]
    assert t["diagnosis"] == "by_look" and len(set(h["look"].values())) == 1 and len(t["findings"]) == 4
    tw = metrics.Twins(world, h["mechanisms"])
    rp0 = metrics.realise(t["current_protocol"], world)
    for f in t["findings"]:
        lk = G.look(tw.image(rp0), rp0, f, t["reference_region"])
        assert lk["cls"] == h["look"][f["id"]] and G.look_is_clear(lk)
        assert all(lk["cls"] in C.CATALOGUE[c].look for c in h["causes"][f["id"]])
    cost = h["experiment_s"]
    battery = cost["complaint"] + sum(cost[e] for e in C.minimal_battery(t["causes_vocabulary"]) if e in cost)
    assert abs(battery - h["battery_s"]) < 0.01 and abs(h["battery_over_budget"] - battery / cfg["budget_s"]) < 2e-3
    assert h["adaptive_s"] <= cfg["budget_s"] + 1e-9                      # the first look and the experiments of the look
    ref = policies.run(mrgym.make(ID, level=3, seed=0), "reference")
    assert ref.reward == 1.0 and not ref.info["policy_error"]
    for name in ("prior", "wrong_units", "remedy_battery"):
        assert policies.run(mrgym.make(ID, level=3, seed=0), name).reward == 0.0, name
