"""Level 3 of ``Scanner/ArtefactClinic-v0``: findings on constructed sites.

What is pinned here: the three world-spec keys the constructed sites need (placed inserts, placed patches of
through-slice gradient, the phase key of a pulsing structure), each with a physics check and each leaving a world
without the key as it was; the wrap twin of a scan in which something pulses; the doses (slice, averages) and the
tracking of a ghost copy; the draw of an episode and the estimator of the open loop; and what an instance of either
look guarantees (same look, exactly its causes, positions that do not depend on the causes, a budget below four
scans, the reference passes, a fixed dose does not).
"""
import itertools
import math

import numpy as np
import pytest

import mrgym
from mrgym import policies
from mrgym.env import derive_seed
from mrgym.envs.artefact_clinic import catalogue as C
from mrgym.envs.artefact_clinic import controls, metrics, task
from mrgym.envs.artefact_clinic import generator as G
from mrgym.scanner import console, kspace, rf
from mrgym.scanner import world as W

ID = "Scanner/ArtefactClinic-v0"
SEED = "000102030405060708090a0b0c0d0e0f"
QA = dict(n_oil=0, n_ring=(0, 0), p_centre=0.0, offset_mm=0.0, body_radius_mm=(56.0, 56.0))
BASE = dict(kind="qa", n=320, b0_t=1.5, b0_hz=0.0, cf_offset_hz=0.0, edge_mm=2.5, qa=QA)
WATER = dict(pd=0.8, t1_ms=900.0, t2_ms=80.0, t2p_ms=150.0)
PROTOCOL = dict(sequence="spgr", fov_mm=240.0, matrix=64, phase_fov=0.5, tr_ms=16.0, flip_deg=30.0, averages=6, slice_mm=8.0,
                bandwidth_hz_per_px=1100.0, te_ms=6.91)
LOSS, CANCEL, WRAP, GHOST = "susceptibility_signal_loss", "opposed_phase_cancellation", "wrap_around", "periodic_motion_ghost"


def _spot(y, x, radius=4.9):
    return dict(y_mm=float(y), x_mm=float(x), radius_mm=radius)


def _mean(image, rp, region):
    m = metrics.area_mask(rp, region)
    return np.sum(m * image) / np.sum(m)


# ------------------------------------------------------------------------------------------------ world-spec keys
def test_the_world_keys_of_constructed_sites_leave_a_world_without_them_unchanged():
    plain = W.make_world(SEED, BASE)
    empty = W.make_world(SEED, dict(BASE, qa=dict(QA, inserts=[]), dbdz=dict(n=0, patches=[])))
    assert plain["names"] == empty["names"] == ["body"] and np.array_equal(plain["weights"], empty["weights"])
    for k in ("b0_hz", "dbdz_hz_per_mm", "b1", "rx"):
        assert np.array_equal(plain["fields"][k], empty["fields"][k]), k
    assert plain["system"] == empty["system"] and plain["tissues"] == empty["tissues"]
    drawn = W.make_world(SEED, dict(BASE, motion=[dict(cls="*", kind="pulse", f_hz=(0.9, 1.5))]))
    keyed = W.make_world(SEED, dict(BASE, motion=[dict(cls="*", kind="pulse", f_hz=(0.9, 1.5), phase_key=5)]))
    assert keyed["motion"][0] == dict(drawn["motion"][0], phase_key=5) and drawn["system"] == keyed["system"]


def test_an_insert_is_a_disc_of_its_material_and_two_on_one_disc_share_every_voxel():
    """Partial volume is linear: a disc that holds 70 % water and 30 % oil gives 0.7 x the water disc + 0.3 x the oil
    disc, as complex signals, at every echo time; water and oil are near opposed at 1.5 fat periods and near in phase
    at one period, so the mixture is darker at the opposed echo."""
    disc = dict(y_mm=10.0, x_mm=-20.0, radius_mm=12.0)
    both = [dict(disc, tissue=WATER, weight=0.7), dict(disc, tissue="oil", weight=0.3)]
    worlds = {k: W.make_world(SEED, dict(BASE, qa=dict(QA, inserts=v)))
              for k, v in dict(water=[dict(disc, tissue=WATER)], oil=[dict(disc, tissue="oil")], mix=both).items()}
    mix = worlds["mix"]
    n, d = mix["n"], mix["extent_mm"] / mix["n"]
    iy, ix = int(round(10.0 / d)) + n // 2, int(round(-20.0 / d)) + n // 2
    assert mix["names"] == ["body", "insert0", "insert1"] and mix["tissues"]["insert1"]["fat"] and not mix["tissues"]["insert0"]["fat"]
    assert np.allclose(mix["weights"][:, iy, ix], [0.0, 0.7, 0.3], atol=1e-3) and not mix["regions"]
    assert abs(mix["weights"][0, n // 2 - 30, n // 2 + 30] - 1.0) < 1e-3     # plain gel away from the disc
    layout = W.make_world(SEED, dict(BASE, qa=dict(QA, inserts=both)), layout_only=True)
    assert layout["names"] == mix["names"] and layout["tissues"]["insert0"]["t1_ms"] == 900.0
    period, spot, level = C.fat_period_ms(1.5), _spot(10.0, -20.0), {}
    for name, te in (("opposed", 1.5 * period), ("in", period)):
        rp = metrics.realise(dict(PROTOCOL, te_ms=round(te, 2)), mix)
        got = {k: _mean(kspace.Engine(w).acquire(rp, None)[0], rp, spot) for k, w in worlds.items()}
        assert abs(got["mix"] - (0.7 * got["water"] + 0.3 * got["oil"])) < 0.01 * abs(got["water"]), name
        level[name] = abs(got["mix"]) / abs(got["water"])
        turn = abs(np.angle(got["oil"] / got["water"]))
        assert (turn > 2.4) if name == "opposed" else (turn < 0.7), (name, turn)
    assert level["opposed"] < 0.75 * level["in"]


def test_a_placed_patch_has_its_gradient_where_it_was_put_and_the_signal_loss_of_the_console_pulse():
    patch = dict(y_mm=-15.0, x_mm=25.0, peak_hz_per_mm=-9.0, sigma_mm=2.5, flat_mm=10.0)
    world = W.make_world(SEED, dict(BASE, dbdz=dict(n=0, patches=[patch])))
    g, (y, x) = world["fields"]["dbdz_hz_per_mm"], W.grid(world)
    r = np.hypot(y + 15.0, x - 25.0)
    assert np.allclose(g[r <= 10.0], -9.0) and abs(g[np.argmin(np.abs(r - 12.5), axis=None) // g.shape[1], np.argmin(np.abs(r - 12.5), axis=None) % g.shape[1]]
                                                    + 9.0 * math.exp(-0.5)) < 0.15
    assert np.all(np.abs(g[r > 25.0]) < 1e-6)
    on, off = kspace.Engine(world), kspace.Engine(world, effects=kspace.ALL - {"dropout"})
    for te, dz in ((6.91, 8.0), (4.6, 8.0), (6.91, 4.0)):
        rp = metrics.realise(dict(PROTOCOL, te_ms=te, slice_mm=dz), world)
        ratio = float(abs(_mean(np.abs(on.acquire(rp, None)[0]), rp, _spot(-15.0, 25.0))) / abs(_mean(np.abs(off.acquire(rp, None)[0]), rp, _spot(-15.0, 25.0))))
        want = float(rf.dephasing_factor(np.array([9.0 * dz * te * 1e-3]), rp["rf_tbw"])[0])
        assert want < 0.97 and abs(ratio - want) < 0.02, (te, dz, ratio, want)


@pytest.fixture(scope="module")
def pulsing():
    """A vial that pulses 0.05 cycles per TR in gel, and a tube beside the body one phase field of view from (-30, 30)."""
    f_hz = 0.05 / (PROTOCOL["tr_ms"] * 1e-3)
    inserts = [dict(y_mm=-10.0, x_mm=-20.0, radius_mm=12.0, tissue=dict(WATER, t1_ms=150.0)),
               dict(y_mm=90.0, x_mm=30.0, radius_mm=12.0, tissue=dict(WATER, pd=0.6))]

    def world(key):
        return W.make_world(SEED, dict(BASE, qa=dict(QA, inserts=inserts),
                                       motion=[dict(cls="insert0", kind="pulse", f_hz=f_hz, depth=0.8, phase_key=key)]))
    rp = metrics.realise(PROTOCOL, world(3))
    copies = C.ghost_copies(-10.0, 0.05, PROTOCOL["averages"], rp["fov_pe_mm"])
    return dict(world=world, rp=rp, copies=copies, f_hz=f_hz)


def test_a_ghost_copy_lies_where_the_formula_puts_it_with_the_phase_of_its_key(pulsing):
    """Copies at the source -+ frac(f TR averages) x the phase field of view, each depth / 2 x the Dirichlet share of
    the source; the phase key turns the two copies by the phase of the pulsation, in opposite directions."""
    rp, copies = pulsing["rp"], pulsing["copies"]
    assert sorted(round(c, 1) for c in copies) == [-46.0, 26.0]                # -10 -+ 0.3 x 120 mm
    assert C.ghost_copies(40.0, 0.05, 6, 120.0)[0] == pytest.approx(-44.0)     # 40 + 36 = 76 folds to -44
    ghost = {}
    for key in (3, 11):
        w = pulsing["world"](key)
        on, off = (kspace.Engine(w, effects=kspace.ALL - e).acquire(rp, None)[0] for e in (set(), {"motion"}))
        ghost[key] = [_mean(on - off, rp, _spot(c, -20.0)) for c in copies]
        source = abs(_mean(off, rp, _spot(-10.0, -20.0)))
        for g in ghost[key]:
            assert abs(abs(g) / source - 0.4 * G.ghost_share(0.05, 6)) < 0.03
        assert abs(_mean(on - off, rp, _spot(8.0, -20.0))) < 0.02 * source      # nothing between the copies
    turn = G.ghost_phase(11, rp) - G.ghost_phase(3, rp)
    got = sorted(float(np.angle(b / a * np.exp(-1j * s * turn))) for a, b in zip(ghost[3], ghost[11]) for s in (1.0,))
    assert min(abs(got[0]), abs(got[1])) < 0.05                               # one copy turns by +turn ...
    other = [float(np.angle(b / a * np.exp(1j * turn))) for a, b in zip(ghost[3], ghost[11])]
    assert min(abs(v) for v in other) < 0.05                                  # ... the other by -turn


def test_the_wrap_twin_keeps_the_ghost_copies_and_only_removes_the_fold(pulsing):
    world, rp = pulsing["world"](3), pulsing["rp"]
    tw = metrics.Twins(world, C.MECHANISMS)
    copy, fold, gel = _spot(pulsing["copies"][0], -20.0), _spot(-30.0, 30.0), _spot(20.0, 0.0)
    s = tw.solo(rp, copy)
    assert s["motion"] >= 0.10 and s["wrap"] < 0.03 and s["dropout"] < 0.01
    s = tw.solo(rp, fold)
    assert s["wrap"] >= 0.10 and s["motion"] < 0.03
    assert all(v < 0.03 for v in tw.solo(rp, gel).values())
    twin = metrics.without_wrap(rp, world)
    assert twin["lines_full"] > rp["lines_full"] and twin["twin_line_spacing"] == rp["lines_full"] / twin["lines_full"]


def test_the_source_of_a_ghost_copy_is_tracked_from_the_one_average_scan(pulsing):
    world, rp = pulsing["world"](3), pulsing["rp"]
    engine = kspace.Engine(world)
    before = engine.acquire(rp, None)[0]
    after = engine.acquire(metrics.realise(dict(PROTOCOL, averages=1), world), None)[0]
    finding = _spot(max(pulsing["copies"]), -20.0)                            # the copy at +26 mm
    track = G.ghost_track(before, after, rp, finding, 1)
    assert abs(track["pe_source_mm"] + 10.0) < 1.0 and abs(track["cycles_per_tr"] / 0.05 - 1.0) < 0.03
    assert G.ghost_track(before, before, rp, finding, 1) is None             # nothing moved
    assert G.ghost_track(before, after, rp, _spot(20.0, 0.0), 1) is None     # plain gel: no copy left this place


# ------------------------------------------------------------------------------------------------ doses
def test_the_slice_dose_leaves_the_residual_loss_at_the_echo_of_the_remedy():
    factor = lambda c: float(rf.dephasing_factor(np.array([c]), 4.0)[0])     # noqa: E731
    for turns, te_ratio in ((0.35, 1.0), (0.5, 0.667), (0.62, 0.8)):
        ratio = factor(0.5 * turns) / factor(turns)                          # what the half-slice experiment shows
        dose = C.slice_dose(8.0, ratio, te_ratio, 4.0)
        assert C.SLICE_MIN_MM < dose < 4.0                                   # thinner than the textbook half slice
        left = 1.0 - factor(turns * dose / 8.0 * te_ratio)
        more = 1.0 - factor(turns * (dose + 0.1) / 8.0 * te_ratio)
        assert left <= C.DROPOUT_RESIDUAL + 1e-6 < more, (turns, dose, left, more)
        assert 1.0 - factor(turns * 0.5 * te_ratio) > 0.05                   # half the slice is not clean
    assert C.slice_dose(8.0, 1.0001, 1.0, 4.0) == 8.0 and C.slice_dose(8.0, 50.0, 1.0, 4.0) == C.SLICE_MIN_MM
    gauss = C.slice_dose(5.0, math.exp(0.75 * 0.2))                           # hard pulse: exp(-x), x = 0.2 at the full slice
    assert gauss == math.floor(10 * 5.0 * math.sqrt(-math.log(1 - C.DROPOUT_RESIDUAL) / 0.2)) / 10


def test_the_averages_of_the_ghost_remedy_follow_from_the_measured_displacement():
    ghost = dict(pe_source_mm=-10.0, cycles_per_tr=0.05, column=[0])
    # finding at +26 mm (the copy of six averages on a 120 mm field of view); the remedy runs on 240 mm
    assert C.ghost_averages([ghost], [26.0, -40.0], 240.0, 3) == 1            # 3: copy at +26; 2: at +14; 1: at +2
    assert C.ghost_averages([ghost], [26.0, -40.0], 240.0, 6) == 6            # 6: copy at +62
    assert C.ghost_averages([dict(ghost, column=[0, 1])], [2.0, -22.0], 240.0, 1) is None
    assert C.ghost_averages([], [26.0], 240.0, 4) == 4


# ------------------------------------------------------------------------------------------------ the draw and the estimator
def test_an_episode_holds_both_causes_of_its_look_and_the_benign_share_is_a_quarter():
    lv = task.LEVELS[3]
    assert all(len(G.look_alike_sets(cls, 4)) == 50 for cls in G.LOOK_CAUSES)
    draws = [G.draw_wanted(derive_seed(ID, 3, "train", i, environ={}), lv) for i in range(600)]
    looks = [d["look"] for d in draws]
    assert 0.42 < looks.count("dark") / len(looks) < 0.58
    for d in draws:
        a, b = ((lab,) for lab in G.LOOK_CAUSES[d["look"]])
        assert a in d["wanted"] and b in d["wanted"] and all(w in (a, b, ()) for w in d["wanted"])
        assert d["armed"] == ["dropout", "fatphase", "motion", "wrap"] and not d["lock_bandwidth"]
    flat = [w for d in draws for w in d["wanted"]]
    assert 0.20 < flat.count(()) / len(flat) < 0.28                           # 12 of 50 sets hold a given finding benign
    by_field = {f: [w for d in draws if d["field_t"] == f for w in d["wanted"]] for f in (1.5, 3.0)}
    assert all(0.19 < v.count(()) / len(v) < 0.29 for v in by_field.values())


def test_the_estimator_names_the_only_set_that_agrees_when_both_experiments_of_the_look_ran():
    labels = [lab for lab in C.LABELS if C.CATALOGUE[lab].mechanism not in task.LEVELS[3]["never"]]
    ids = ["F1", "F2", "F3", "F4"]
    for cls, experiments in (("dark", ("dropout", "fatphase")), ("bright", ("wrap", "motion"))):
        classes = {i: cls for i in ids}
        for truth in G.look_alike_sets(cls, 4):
            hidden = dict(zip(ids, truth))
            flags = {e: {i: bool(C.expected(hidden[i], e)) for i in ids} for e in experiments}
            assert G.infer(classes, flags, labels) == {i: list(hidden[i]) for i in ids}
            assert G.count_consistent(cls, hidden, experiments) == 1 and G.count_consistent(cls, hidden, ()) == 50
            one = G.count_consistent(cls, hidden, experiments[:1])
            assert one >= 1 and (one > 1 or sum(1 for v in truth if C.expected(v, experiments[0]) is False) <= 1 or cls == "bright")
        assert G.infer(classes, {}, labels) == {i: list(c) for i, c in zip(ids, G.look_alike_sets(cls, 4)[0])}
    mixed = dict(F1="dark", F2="bright")                                     # a misread look: finding by finding
    assert G.infer(mixed, dict(dropout=dict(F1=True, F2=False), wrap=dict(F1=False, F2=True)), labels) == dict(F1=[LOSS], F2=[WRAP])


def test_the_open_loop_is_tuned_to_the_two_experiments_of_one_look_and_constants_may_use_positions():
    cost = dict(complaint=10.0, wrap=8.1, motion=7.6, fatphase=8.2, dropout=8.2)
    ids = ["F1", "F2", "F3", "F4"]
    worlds = []
    for k, (cls, n) in enumerate((("dark", 5), ("bright", 3))):
        for truth in G.look_alike_sets(cls, 4)[k::7][:n]:
            worlds.append(dict(budget_s=27.0, hidden=dict(experiment_s=cost, look={i: cls for i in ids}, causes=dict(zip(ids, map(list, truth))))))
    sheet = dict(sequence="spgr", fov_mm=240.0, matrix=64, phase_fov=0.5, tr_ms=15.0, flip_deg=30.0, averages=7, slice_mm=8.0,
                 bandwidth_hz_per_px=1100.0, te_ms=6.91)
    worlds = [dict(w, task=dict(current_protocol=sheet), world_spec=dict(kind="qa", n=320, b0_t=1.5)) for w in worlds]
    tuned = controls.tune_open_loop(worlds)
    assert set(tuned["order"][:2]) == {"dropout", "fatphase"} and tuned["worlds"] == 8      # the look most of these worlds have
    assert 5.0 <= tuned["expected_worlds_all_findings"] < 5.0 + 3 * 0.5
    assert tuned["several_worlds"] == 8 and tuned["list"] == "several"                      # 10.0 + 7.98 + 8.2 s fit 27 s
    assert controls.tune_open_loop([dict(w, budget_s=25.0) for w in worlds])["list"] == "single"
    t = dict(current_protocol=dict(pe_dir="x"), findings=[dict(id=f"F{k + 1}", y_mm=y, x_mm=x) for k, (y, x) in enumerate(((5, -30), (0, 40), (9, -12), (-7, 20)))])
    assert controls._ranked(t, "phase") == ["F1", "F3", "F4", "F2"] and controls._ranked(t, "phase_distance") == ["F3", "F4", "F1", "F2"]
    recipe = dict(rank="phase", assign=[[LOSS], [], [CANCEL], [LOSS]])
    assert controls._constant_causes(t, recipe) == dict(F1=[LOSS], F3=[], F4=[CANCEL], F2=[LOSS])


def test_the_fixed_doses_are_tuned_on_the_hidden_remedies_and_a_stated_dose_is_applied():
    def world(slice_mm, averages, causes, share=0.9):
        return dict(task=dict(current_protocol=dict(slice_mm=8.0), requirements=dict(snr_share=share)),
                    hidden=dict(remedy=dict(slice_mm=slice_mm, averages=averages), causes=dict(F1=causes, F2=[])))
    worlds = [world(2.0, 8, [LOSS]), world(2.1, 8, [LOSS]), world(3.4, 8, [LOSS]), world(8.0, 2, [GHOST]), world(8.0, 2, [GHOST]), world(8.0, 1, [GHOST])]
    tuned = controls.tune_doses(worlds)
    assert tuned["worlds_with_loss"] == 3 and tuned["slice_worlds"] == 2 and tuned["averages"] == 2 and tuned["averages_worlds"] == 2
    low, high = 0.9 * 2.1 / 8.0, 2.0 * 1.2 / 8.0                            # thick enough for the SNR of both, thin enough to be clean on both
    assert low <= tuned["slice_fraction"] <= high
    assert set(controls.TUNED_DOSE) == {"slice_fraction", "averages"} and "told_cause_tuned_dose" in controls.REPORTED
    # a stated dose replaces the textbook dose of a remedy; a remedy that is not needed is not touched
    p0 = dict(PROTOCOL, phase_oversampling=0.0, partial_fourier=1.0, pe_dir="y", ro_polarity=1, fat_sat=False, centre_y_mm=0.0, centre_x_mm=0.0, averages=8)
    card = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5)
    t = dict(exam=dict(field_t=1.5), current_protocol=p0, diagnosis="by_look", findings=[_spot(20.0, 0.0)],
             requirements=dict(max_scan_s=console.plan(p0, card)["scan_s"], te_ms=[2.0, 12.0], locked={}))
    plan = lambda p: console.plan(p, card)                                   # noqa: E731
    assert G.assemble(t, [LOSS], plan)["slice_mm"] == 4.0 and G.assemble(t, [LOSS], plan, dict(slice_mm=2.5))["slice_mm"] == 2.5
    assert G.assemble(t, [GHOST], plan)["averages"] == 4 and G.assemble(t, [GHOST], plan, dict(averages=2, slice_mm=2.5)) == dict(p0, averages=2)
    assert G.assemble(t, [CANCEL], plan, dict(averages=2, slice_mm=2.5)) == dict(p0, te_ms=round(C.fat_period_ms(1.5), 2))


# ------------------------------------------------------------------------------------------------ instances
def _cfg(seed):
    return task.make_instance(derive_seed(ID, 3, "train", seed, environ={}), 3, task_id="t")


@pytest.fixture(scope="module")
def cases():
    """The first training instance of either look, with its world: {look: dict(cfg, world, seed)}."""
    out = {}
    for seed in range(16):
        cfg = _cfg(seed)
        look = next(iter(cfg["hidden"]["look"].values()))
        out.setdefault(look, dict(cfg=cfg, seed=seed, world=W.make_world(cfg["seed"], cfg["world_spec"])))
        if len(out) == 2:
            return out
    raise AssertionError("one look is missing among the first training seeds")


@pytest.mark.parametrize("look", ["dark", "bright"])
def test_a_constructed_instance_keeps_its_promises(cases, look):
    cfg, world = cases[look]["cfg"], cases[look]["world"]
    t, h = cfg["task"], cfg["hidden"]
    p0 = t["current_protocol"]
    tw = metrics.Twins(world, h["mechanisms"])
    rp0, rp1 = metrics.realise(p0, world), metrics.realise(h["remedy"], world)
    causes = [tuple(h["causes"][f["id"]]) for f in t["findings"]]
    assert set(h["look"].values()) == {look} and all(((lab,) in causes) for lab in G.LOOK_CAUSES[look])
    for f in t["findings"]:
        assert G.cause_set(tw.solo(rp0, f)) == tuple(sorted(h["causes"][f["id"]]))   # exactly these causes
        lk = G.look(tw.image(rp0), rp0, f, t["reference_region"])
        assert lk["cls"] == look and G.look_is_clear(lk)
        assert G.is_clean(tw.errors(rp1, f), G.VERIFY_BUILT)
    # positions: one stripe per finding, outside the centre line, inside a body that does not fold
    pe, ro = ("y_mm", "x_mm") if p0["pe_dir"] == "y" else ("x_mm", "y_mm")
    stripes = sorted(f[ro] for f in t["findings"])
    assert min(np.diff(stripes)) >= G.COLUMN_GAP_MM - 0.11 and all(abs(f[pe]) >= G.PE_INNER_MM - 0.06 for f in t["findings"])
    extent = W.extent_mm(dict(world, weights=world["weights"][:1], names=world["names"][:1]))["y" if pe == "y_mm" else "x"]
    assert max(abs(v) for v in extent) < rp0["fov_pe_mm"] / 2
    # budget: the first look and two experiments; four scans of any protocol cost more
    cost = h["experiment_s"]
    assert G.SCAN_FLOOR_S == 7.0 and all(v >= G.SCAN_FLOOR_S for v in cost.values()) and cfg["budget_s"] <= 4 * G.SCAN_FLOOR_S
    assert h["adaptive_s"] <= cfg["budget_s"] and h["battery_over_budget"] >= 1.5
    assert sorted(cost) == ["complaint", "dropout", "fatphase", "motion", "wrap"]
    assert console.plan(h["remedy"], world["console"])["scan_s"] <= t["requirements"]["max_scan_s"]
    # the SNR floor is a share of the verified remedy's SNR: the share is on the card, the number is not
    share = t["requirements"]["snr_share"]
    assert "min_snr_reference" not in t["requirements"] and 0.85 <= share <= 0.93
    assert task.min_snr(cfg) == h["min_snr_reference"] and abs(h["min_snr_reference"] - share * h["snr_remedy"]) < 0.11   # both stored to 0.1
    assert f"at least {100 * share:g} % of that of the protocol" in task.statement(cfg) and f"{h['min_snr_reference']:g}" not in task.statement(cfg)
    # the dose is not the textbook's
    if look == "dark":
        assert h["remedy"]["slice_mm"] < 0.5 * p0["slice_mm"] and h["remedy"]["te_ms"] < p0["te_ms"]
    else:
        assert h["remedy"]["phase_fov"] == 1.0 and h["remedy"]["averages"] != max(1, p0["averages"] // C.MOTION_AVERAGES_DIVISOR)


@pytest.mark.parametrize("look", ["dark", "bright"])
def test_the_reference_passes_and_a_fixed_dose_or_no_experiment_does_not(cases, look):
    seed = cases[look]["seed"]
    ref = policies.run(mrgym.make(ID, level=3, seed=seed), "reference")
    assert ref.reward == 1.0 and not ref.info["policy_error"] and ref.info["budget"]["used_s"] <= ref.info["budget"]["budget_s"]
    assert ref.info["budget"]["scans"] == 3                                    # the first look and the two experiments of the look
    told = policies.run(mrgym.make(ID, level=3, seed=seed), "told_cause_fixed_dose")
    assert told.reward == 0.0 and told.info["score"]["diagnosis_correct"] == 4 and not told.info["policy_error"]
    for name in ("prior", "brute_force", "remedy_battery", "half_budget"):
        assert policies.run(mrgym.make(ID, level=3, seed=seed), name).reward == 0.0, name
    tuned = policies.run(mrgym.make(ID, level=3, seed=seed), "told_cause_tuned_dose")       # may pass: its doses are right on part of the worlds
    assert not tuned.info["policy_error"] and tuned.info["score"]["diagnosis_correct"] == 4 and tuned.info["budget"]["scans"] == 1
    # the fixed list that carries several changes per scan diagnoses either look; the single-change list only the look it starts with
    assert controls.OPEN_LOOP["list"] == "several"
    open_loop = policies.run(mrgym.make(ID, level=3, seed=seed), "open_loop")
    assert open_loop.reward == 1.0 and not open_loop.info["policy_error"] and open_loop.info["budget"]["scans"] == 3
    env = mrgym.make(ID, level=3, seed=seed)
    env.reset()
    sc = policies.ToolScanner(env)
    single = controls._answer(sc, *controls._by_look(sc, order=controls.OPEN_LOOP["order"]))
    covered = set(C.mechanisms(G.LOOK_CAUSES[look])) <= set(controls.OPEN_LOOP["order"][:2])
    truth = env.episode_config()["hidden"]["causes"]
    assert all(sorted(single["causes"][i]) == sorted(v) for i, v in truth.items()) == covered


def test_positions_sheet_and_ids_do_not_depend_on_the_causes(cases):
    """The same world seed with other causes, of either look: the same sheet, the same circles under the same ids."""
    lv, cfg = task.LEVELS[3], cases["dark"]["cfg"]
    base = G.draw_wanted(derive_seed(ID, 3, "train", cases["dark"]["seed"], environ={}), lv)
    seen = 0
    for look, k in itertools.product(("bright", "dark"), (3, 22, 41)):
        draw = dict(base, look=look, wanted=list(G.look_alike_sets(look, 4)[k]), options=[(lab,) for lab in G.LOOK_CAUSES[look]] + [()])
        try:
            other = G.build_constructed(cfg["seed"], lv, draw)
        except G.Reject:
            continue
        seen += 1
        assert other["task"]["current_protocol"] == cfg["task"]["current_protocol"]
        assert other["task"]["findings"] == cfg["task"]["findings"] and other["budget_s"] == cfg["budget_s"]
        assert other["task"]["requirements"]["locked"] == cfg["task"]["requirements"]["locked"]
    assert seen >= 3
