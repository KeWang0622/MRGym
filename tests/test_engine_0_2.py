"""Engine 0.2.0: slice-selective excitation and its price, the turned field of view, periodic motion, the hidden
state of fat saturation, oil T1 jitter. The physics itself is checked by ``mrgym validate`` (asserted in
``test_scanner.py``); this file pins the contracts around it."""
import hashlib
import json

import numpy as np
import pytest

from conftest import SEED
from mrgym.scanner import console, kspace, rf, session
from mrgym.scanner import world as W

EXAM = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
SPGR = dict(sequence="spgr", tr_ms=20.0, flip_deg=20.0, matrix=64)
# (weights, fields, system, tissues, regions) of six worlds, recorded on mrgym-engine-0.1.0
WORLDS_0_1_0 = {
    "qa": (dict(kind="qa"), "5dc716ce06a26e332da7b24d"),
    "qa_15t": (dict(kind="qa", b0_t=1.5, qa=dict(n_ring=(4, 5)), rfi="random"), "b90af50c267a9343544312b8"),
    "brain": (dict(kind="brain"), "f17f6d10327b737d581b2602"),
    "brain_ms": (dict(kind="brain", lesion="ms"), "ec3c43eb040bfe309917f873"),
    "brain_tumour": (dict(kind="brain", lesion="tumour", b0_t=1.5), "4df4a33bcf609b128cf67cb8"),
    "abdomen": (dict(kind="abdomen", lesion="liver"), "a25f457cb1a482e3e8aaf2cd"),
}


def world_digest(w):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(w["weights"]).tobytes())
    for k in ("b0_hz", "dbdz_hz_per_mm", "b1", "rx"):
        h.update(np.ascontiguousarray(w["fields"][k]).tobytes())
    h.update(json.dumps({k: v for k, v in w["system"].items() if k != "fatsat_offset_hz"}, sort_keys=True).encode())
    h.update(json.dumps(w["tissues"], sort_keys=True).encode())
    h.update(json.dumps(w.get("regions", []), sort_keys=True).encode())
    return h.hexdigest()[:24]


# ------------------------------------------------------------------ worlds
@pytest.mark.parametrize("name", ["qa", "qa_15t", "brain_ms", "abdomen"])
def test_a_world_without_the_new_features_is_the_world_of_engine_0_1_0(name):
    spec, digest = WORLDS_0_1_0[name]
    w = W.make_world(SEED, spec)
    assert world_digest(w) == digest
    assert w["system"]["fatsat_offset_hz"] == 0.0 and "motion" not in w and w["rot_deg"] == 0.0


def test_motion_fat_offset_edge_and_nonlinearity_draw_from_nothing_else(qa_world):
    more = W.make_world(SEED, dict(kind="qa", fatsat_offset_hz=80.0, motion=[dict(cls="R2", kind="pulse", f_hz="cardiac"),
                                                                           dict(cls="body", kind="shift", f_hz=(0.2, 0.3))]))
    assert world_digest(more) == world_digest(qa_world)
    assert 0 < abs(more["system"]["fatsat_offset_hz"]) <= 80.0
    a, b = more["motion"]
    assert a["cls"] == next(r["cls"] for r in more["regions"] if r["id"] == "R2") and 0.9 <= a["f_hz"] <= 1.5 and 0.1 <= a["depth"] <= 0.6
    assert b["cls"] == "body" and 0.2 <= b["f_hz"] <= 0.3 and 1.0 <= b["amp_mm"] <= 8.0 and b["dir"] in ("y", "x")
    assert more["motion"] == W.make_world(SEED, more["source"]["spec"])["motion"]
    soft = W.make_world(SEED, dict(kind="qa", edge_mm=2.0))
    assert soft["system"] == qa_world["system"] and soft["regions"] == qa_world["regions"] and not np.array_equal(soft["weights"], qa_world["weights"])
    with pytest.raises(ValueError):
        W.make_world(SEED, dict(kind="qa", motion=[dict(cls="no_such_class")]))


def test_oil_t1_differs_between_worlds_and_moves_no_other_draw():
    spec = dict(kind="qa", qa=dict(n_oil=2))
    t1, fixed = [], []
    for i in range(12):
        seed = f"{i:032x}"
        w = W.make_world(seed, spec, layout_only=True)
        still = W.make_world(seed, dict(kind="qa", qa=dict(n_oil=2, oil_t1_jitter=0.0)), layout_only=True)
        oil = [k for k, t in w["tissues"].items() if t["fat"]]
        assert len(oil) == 2
        t1 += [w["tissues"][k]["t1_ms"] for k in oil]
        fixed += [still["tissues"][k]["t1_ms"] for k in oil]
        assert all(w["tissues"][k] == still["tissues"][k] for k in w["tissues"] if k not in oil)
    assert len(set(fixed)) == 1 and min(t1) >= 0.8 * fixed[0] and max(t1) <= 1.2 * fixed[0] and np.std(t1) > 0.05 * fixed[0]


# ------------------------------------------------------------------ console
def test_selective_excitation_is_the_default_and_carries_its_price():
    p = console.plan(SPGR, EXAM)
    r = p["realised"]
    assert (r["excitation"], r["rf_pulse"], r["rf_ms"], r["rf_tbw"], r["partitions"]) == ("selective", "normal", 2.0, 4.0, 1)
    hard = console.plan(dict(SPGR, excitation="nonselective"), EXAM)
    assert hard["realised"]["rf_tbw"] is None and hard["realised"]["partitions"] == 32 and "3Dx32" in hard["header"]
    assert hard["scan_s_parts"]["acquisition_s"] == pytest.approx(32 * p["scan_s_parts"]["acquisition_s"])
    assert p["timing"]["te_min_ms"] > hard["timing"]["te_min_ms"] and p["timing"]["tr_min_ms"] > hard["timing"]["tr_min_ms"]
    assert p["limits"]["rf_power_percent"] > hard["limits"]["rf_power_percent"]
    for name, (dur, tbw) in rf.PULSES.items():
        q = console.plan(dict(SPGR, rf_pulse=name), EXAM)["realised"]
        assert (q["rf_ms"], q["rf_tbw"]) == (dur, tbw)
    assert [round(rf.energy_factor(t), 2) for t in (2.0, 4.0, 8.0)] == [1.80, 3.11, 7.16]


def test_a_slice_too_thin_for_the_pulse_is_refused_and_3d_is_the_way_out():
    thin = console.plan(dict(SPGR, slice_mm=1.2), EXAM)
    assert not thin["accept"] and [v["code"] for v in thin["violations"]] == ["SLICE_TOO_THIN"] and thin["violations"][0]["minimum_mm"] == pytest.approx(1.468, abs=2e-3)
    assert console.plan(dict(SPGR, slice_mm=1.5), EXAM)["accept"]
    assert console.plan(dict(SPGR, slice_mm=1.2, excitation="nonselective"), EXAM)["accept"]


def test_a_pulse_above_the_peak_b1_limit_is_lengthened():
    p = console.plan(dict(SPGR, flip_deg=90.0, tr_ms=200.0), EXAM, system=dict(b1_peak_ut=8.0))
    r = p["realised"]
    assert p["accept"] and r["rf_ms"] > 2.0
    assert rf.peak_factor(4.0) * console._b1_ut(90.0, r["rf_ms"]) <= 8.0 + 1e-6 < rf.peak_factor(4.0) * console._b1_ut(90.0, r["rf_ms"] - 0.02)


@pytest.mark.parametrize("protocol", [dict(SPGR, rotation_deg=-35.0, centre_x_mm=20.0, rf_pulse="sharp"), dict(sequence="se", tr_ms=500.0, fat_sat=True, fatsat_offset_hz=-40.0),
                                      dict(sequence="epi", tr_ms=2000.0, fatsat_offset_hz=25.0, excitation="nonselective")])
def test_new_parameters_survive_the_round_trip_of_a_realised_protocol(protocol):
    p = console.plan(protocol, EXAM)
    assert p["accept"], p["violations"]
    assert console.plan(console.request_of(p["realised"]), EXAM)["realised"] == p["realised"]
    assert all(p["realised"][k] == v for k, v in protocol.items())


# ------------------------------------------------------------------ engine
def test_the_served_signal_depends_on_the_pulse_and_the_hard_pulse_is_one_code_path(qa_world, qa_engine):
    rp = {k: console.plan(dict(SPGR, **kw), qa_world["console"])["realised"] for k, kw in
          (("normal", {}), ("fast", dict(rf_pulse="fast")), ("sharp", dict(rf_pulse="sharp")), ("hard", dict(excitation="nonselective")))}
    amp = {k: qa_engine.amplitudes(v, 1)[0] for k, v in rp.items()}
    assert not np.allclose(amp["normal"], amp["hard"], rtol=1e-3) and not np.allclose(amp["normal"], amp["fast"], rtol=1e-3)
    rect = kspace.Engine(qa_world, rf=dict(profile="rect"))
    assert np.array_equal(rect.amplitudes(rp["normal"], 1)[0], amp["hard"])
    wide = kspace.Engine(qa_world, rf=dict(tbw=2.0))
    assert np.allclose(wide.amplitudes(rp["normal"], 1)[0], amp["fast"], rtol=1e-12)
    assert qa_engine.noise_sd(rp["hard"]) == pytest.approx(qa_engine.noise_sd(rp["normal"]) / np.sqrt(32))


def test_a_turned_brain_is_the_same_brain(seed):
    w = W.make_world(seed, dict(kind="brain", lesion="ms"))
    t = W.turned(w, 90.0)                                              # 90 degrees: grid point (i, j) is the upright point (n - j, i)
    n, j = w["n"], np.arange(1, w["n"])
    assert np.array_equal(t["weights"][:, :, 1:], np.transpose(w["weights"][:, n - j, :], (0, 2, 1)))
    assert np.allclose(t["fields"]["b0_hz"][:, 1:], w["fields"]["b0_hz"][n - j, :].T, atol=1e-9)
    assert np.allclose(t["fields"]["b1"][:, 1:], w["fields"]["b1"][n - j, :].T, atol=1e-12)
    assert t["tissues"] is w["tissues"] and t["system"] is w["system"] and t["rot_deg"] == 90.0 and len(w["picks"]) >= 4
    with pytest.raises(ValueError):
        W.turned(dict(w, source=None), 10.0)


def test_region_readouts_follow_the_object_when_the_field_of_view_turns(qa_world, qa_engine):
    regions = [dict(id=r["id"], y_mm=r["y_mm"], x_mm=r["x_mm"], radius_mm=8.0) for r in qa_world["regions"]]
    sc = session.LocalScanner(session.ScannerSession(dict(task_id="t", seed=SEED, world_spec=dict(kind="qa"), budget_s=1e4, task={}), engine=qa_engine))
    base = dict(sequence="se", tr_ms=400.0, matrix=128)
    a = sc.scan(base, regions)
    b = sc.scan(dict(base, rotation_deg=37.0, centre_y_mm=6.0, centre_x_mm=-9.0), regions)
    assert b["rotation_deg"] == 37.0 and sc.view(1)["rotation_deg"] == 37.0 and "rot +37" in b["header"]
    for ra, rb in zip(a["roi_readouts"], b["roi_readouts"]):
        assert rb["mean_magnitude"][0] == pytest.approx(ra["mean_magnitude"][0], rel=0.03)
    again = sc.read_regions(1, regions)
    assert [r["n_pixels"] for r in again] == [r["n_pixels"] for r in b["roi_readouts"]]
    assert session.replay(sc.session.cfg, sc.session.events, engine=qa_engine, head=sc.session.receipt(with_key=True))["ok"]


def test_nothing_moves_unless_the_world_says_so(qa_world, qa_engine):
    rp = console.plan(dict(sequence="se", tr_ms=400.0, matrix=64, averages=2), qa_world["console"])["realised"]
    still = kspace.Engine(qa_world, effects=kspace.ALL - {"motion"})
    assert np.array_equal(qa_engine.clean_kspace(rp), still.clean_kspace(rp))
    one = dict(rp, averages=1)
    assert qa_engine.clean_kspace(one) is qa_engine.clean_kspace(rp)        # averages do not change noise-free data of a static world
    moving = dict(qa_world, motion=[dict(cls="vial0", kind="pulse", f_hz=0.3, depth=0.4, phase_key=11)])
    eng = kspace.Engine(moving)
    k2, k1 = eng.clean_kspace(rp), eng.clean_kspace(one)
    assert not np.allclose(k2, still.clean_kspace(rp)) and not np.allclose(k2, k1)
    assert np.array_equal(k2, kspace.Engine(moving).clean_kspace(rp))
    assert np.array_equal(kspace.Engine(moving, effects=kspace.ALL - {"motion"}).clean_kspace(rp), still.clean_kspace(rp))


def test_the_fat_saturation_pulse_can_miss_and_be_moved_back():
    world = W.make_world(SEED, dict(kind="qa", qa=dict(n_oil=2), b0_hz=0.0, cf_offset_hz=0.0))
    oil = [r for r in world["regions"] if world["tissues"][r["cls"]]["fat"]][0]
    roi = [dict(id="oil", y_mm=oil["y_mm"], x_mm=oil["x_mm"], radius_mm=8.0)]
    prot = dict(sequence="se", tr_ms=400.0, matrix=64, bandwidth_hz_per_px=800.0)

    def left(offset_system, **kw):
        world["system"]["fatsat_offset_hz"] = offset_system
        eng = kspace.Engine(world)
        rp = [console.plan(dict(prot, **k), world["console"])["realised"] for k in ({}, dict(fat_sat=True, **kw))]
        m = [session.roi_readouts(eng.acquire(r), roi, session.fov_yx(r))[0]["mean_magnitude"][0] for r in rp]
        return m[1] / m[0]
    assert left(0.0) < 0.15 and left(250.0) > 0.9 and left(250.0, fatsat_offset_hz=-250.0) == pytest.approx(left(0.0), rel=1e-9)


@pytest.mark.parametrize("direction, rotation, pe_dir", [("y", 0.0, "y"), ("x", 0.0, "y"), ("y", 0.0, "x"), ("y", 90.0, "y"), ("x", 30.0, "y")])
def test_a_frozen_displacement_moves_the_class_along_its_patient_axis(direction, rotation, pe_dir):
    """The shift model in every frame: motion along the phase axis, along the readout axis, and seen by a turned
    field of view. With f = 0 every line sees the same displacement, so the image of the class is a displaced copy."""
    w = W.flat_world(W.shapes_phantom([("blood", lambda y, x: np.hypot(y + 20.0, x - 10.0) - 8.0)]))
    w["motion"] = [dict(cls="blood", kind="shift", f_hz=0.0, amp_mm=6.0, dir=direction, phase_key=5)]
    rp = console.plan(dict(sequence="se", tr_ms=500.0, te_ms=20.0, fov_mm=256.0, matrix=128, bandwidth_hz_per_px=300.0,
                           rotation_deg=rotation, pe_dir=pe_dir), w["console"])["realised"]
    moved = np.abs(kspace.Engine(w).acquire(rp)[0])
    still = np.abs(kspace.Engine(w, effects=kspace.ALL - {"motion"}).acquire(rp)[0])
    yy, xx = session.pixel_grid(moved.shape, session.fov_yx(rp), session.centre_yx(rp), session.rotation_of(rp))
    def centre(im):
        m = np.where(im > 0.1 * im.max(), im, 0.0)
        return np.array([(m * yy).sum(), (m * xx).sum()]) / m.sum()
    assert centre(still) == pytest.approx([-20.0, 10.0], abs=0.1)
    d = centre(moved) - centre(still)
    along, across = (d[0], d[1]) if direction == "y" else (d[1], d[0])
    assert abs(across) < 0.1 and 0.3 < abs(along) <= 6.05, d
