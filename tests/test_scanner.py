"""The scanner: worlds, console, engine, session, replay, HTTP service."""
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import SEED, episode_cfg
from mrgym import validate
from mrgym.scanner import client, console, kspace, session
from mrgym.scanner import world as W

SPGR = dict(sequence="spgr", tr_ms=12.0, flip_deg=10.0, matrix=64)
ALL_SEQUENCES = (
    dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0, n_echoes=3), dict(sequence="afi", tr_ms=30.0, flip_deg=60.0),
    dict(sequence="se", tr_ms=500.0, te_ms=20.0, phase_fov=0.75, fat_sat=True), dict(sequence="ir", tr_ms=3000.0, ti_ms=800.0),
    dict(sequence="bssfp", tr_ms=8.0, flip_deg=40.0, phase_cycle_deg=0.0),
    dict(sequence="epi", tr_ms=2000.0, pe_polarity=-1, partial_fourier=0.75), dict(sequence="se", tr_ms=400.0, pe_dir="x", phase_oversampling=0.5))


# ------------------------------------------------------------------ physics against formulas
def test_every_validation_check_passes():
    rows, _ = validate.run()
    failed = [f"{r['id']} {r['check']}: {r['measured']}" for r in rows if not r["ok"]]
    assert not failed, failed
    assert len(rows) >= 63 and {r["group"] for r in rows} == set(validate.GROUPS) == {"signal", "rf", "encoding", "console"}


# ------------------------------------------------------------------ worlds
@pytest.mark.parametrize("kind", W.KINDS)
def test_world_is_a_function_of_the_seed(kind):
    a, b = W.make_world(SEED, dict(kind=kind)), W.make_world(SEED, dict(kind=kind))
    c = W.make_world(SEED[::-1], dict(kind=kind))
    assert np.array_equal(a["weights"], b["weights"]) and a["system"] == b["system"]
    assert not np.array_equal(a["weights"], c["weights"]) and a["system"] != c["system"]
    assert a["weights"].shape[1:] == (a["n"], a["n"]) and abs(a["weights"].sum(0).max() - 1) < 1e-3
    assert a["console"]["min_pixel_mm"] == pytest.approx(2 * a["extent_mm"] / a["n"], abs=1e-3)


def test_seed_must_be_128_bits():
    for bad in ("0", 0, "abc", SEED + "0", None):
        with pytest.raises(ValueError):
            W.make_world(bad)


def test_layout_only_world_has_the_same_truth(qa_world):
    lay = W.make_world(SEED, dict(kind="qa"), layout_only=True)
    full, light = W.truth(qa_world), W.truth(lay)
    assert lay["regions"] == qa_world["regions"] and "weights" not in lay
    assert all(full[k]["t1_ms"] == light[k]["t1_ms"] for k in full) and lay["system"] == qa_world["system"]


def test_field_strength_changes_tissue_and_limits():
    lo, hi = W.make_world(SEED, dict(kind="brain", b0_t=1.5)), W.make_world(SEED, dict(kind="brain", b0_t=3.0))
    assert lo["tissues"]["wm"]["t1_ms"] < hi["tissues"]["wm"]["t1_ms"]
    assert lo["console"]["b1rms_limit_ut"] > hi["console"]["b1rms_limit_ut"] and lo["system"]["sigma0"] > hi["system"]["sigma0"]
    with pytest.raises(ValueError):
        W.make_world(SEED, dict(kind="brain", b0_t=7.0))


# ------------------------------------------------------------------ console
def test_plan_fills_defaults_and_prices_the_scan():
    p = console.plan(SPGR)
    r = p["realised"]
    assert p["accept"] and r["fov_mm"] == 240.0 and r["matrix_pe"] == 64 and r["lines"] == 64
    parts = p["scan_s_parts"]
    assert parts["overhead_s"] == 3.0 and parts["acquisition_s"] == pytest.approx(0.012 * 64)
    assert p["scan_s"] == pytest.approx(sum(parts.values()))
    assert console.plan(dict(SPGR, averages=4))["scan_s_parts"]["acquisition_s"] == pytest.approx(4 * parts["acquisition_s"])
    half = console.plan(dict(SPGR, phase_fov=0.5))
    assert half["realised"]["lines"] == 32 and half["realised"]["fov_pe_mm"] == 120.0
    assert console.plan(dict(SPGR, phase_oversampling=0.5))["realised"]["lines"] == 96
    assert console.plan(dict(SPGR, partial_fourier=0.75))["realised"]["lines"] == 48


@pytest.mark.parametrize("protocol, code", [
    (dict(sequence="nope"), "UNKNOWN_SEQUENCE"), (dict(SPGR, matrx=64), "UNKNOWN_PARAMETER"),
    (dict(sequence="spgr", tr_ms=12.0), "MISSING_PARAMETER"), (dict(SPGR, flip_deg=120), "OUT_OF_RANGE"),
    (dict(SPGR, matrix=65), "OUT_OF_RANGE"), (dict(SPGR, te_ms=0.6), "TE_BELOW_MINIMUM"), (dict(SPGR, tr_ms=3.0), "TR_BELOW_MINIMUM"),
    (dict(SPGR, matrix=256, spoiler_mt_ms_per_m=10.0), "SPOILER_BELOW_MINIMUM"), (dict(SPGR, matrix=512), "RESOLUTION_LIMIT"),
    (dict(sequence="bssfp", tr_ms=3.5, flip_deg=90.0, matrix=64, bandwidth_hz_per_px=1500.0), "RF_POWER_LIMIT"),
    (dict(SPGR, matrix=256, fov_mm=200, bandwidth_hz_per_px=3000), "GRADIENT_LIMIT"),
    (dict(sequence="ir", tr_ms=3000.0, ti_ms=10.0, fat_sat=True), "TI_NOT_REALISABLE"),
    (dict(sequence="epi", tr_ms=2000.0, echo_spacing_ms=0.2), "ECHO_SPACING_BELOW_MINIMUM"),
    (dict(SPGR, fat_sat="yes"), "OUT_OF_RANGE"), (dict(SPGR, pe_dir="z"), "OUT_OF_RANGE"), ("spgr", "INVALID_REQUEST")])
def test_refusals_are_typed(protocol, code):
    p = console.plan(protocol)
    assert not p["accept"] and code in [v["code"] for v in p["violations"]], p["violations"]
    assert "scan_s" not in p or all(k in v for v in p["violations"] for k in ("code", "message"))


@pytest.mark.parametrize("protocol", ALL_SEQUENCES)
def test_a_realised_protocol_reproduces_itself(protocol):
    p = console.plan(protocol)
    assert p["accept"], p["violations"]
    again = console.plan(console.request_of(p["realised"]))
    assert again["realised"] == p["realised"] and again["scan_s"] == p["scan_s"]
    assert p["header"].startswith(protocol["sequence"].upper()) and "TA " in p["header"] and "SAR " in p["header"]


def test_plan_never_depends_on_the_world(qa_world):
    other = W.make_world(SEED[::-1], dict(kind="qa"))
    assert console.plan(SPGR, qa_world["console"]) == console.plan(SPGR, other["console"])


def test_protocol_schema_covers_every_parameter_and_the_datasheet_names_them():
    import importlib.resources
    props = console.protocol_schema()["properties"]
    names = {k for spec in console.SEQUENCES.values() for k in spec}
    assert names | {"sequence"} == set(props)
    sheet = importlib.resources.files("mrgym.scanner").joinpath("datasheet.md").read_text(encoding="utf-8")
    assert not [n for n in names if f"`{n}`" not in sheet]
    assert all(s in sheet for s in ("## 5.", "## 6.", "## 7."))
    example = console.plan(dict(sequence="se", tr_ms=500, te_ms=15, matrix=256, phase_fov=0.75))["header"]
    assert example in sheet                                             # the header shown in the datasheet is a real one


# ------------------------------------------------------------------ engine
@pytest.mark.parametrize("protocol", ALL_SEQUENCES)
def test_every_sequence_returns_images_of_the_prescribed_shape(qa_engine, protocol):
    out = qa_engine.scan(dict(protocol, matrix=64))
    rp = out["plan"]["realised"]
    n_img = 2 if rp["sequence"] == "afi" else rp.get("n_echoes", 1)
    ny, nx = (rp["matrix_pe"], rp["matrix"]) if rp["pe_dir"] == "y" else (rp["matrix"], rp["matrix_pe"])
    assert out["images"].shape == (n_img, ny, nx) and np.isfinite(out["images"]).all() and np.abs(out["images"]).max() > 0


def test_noise_is_keyed_by_the_generator_and_absent_without_one(qa_engine):
    rp = console.plan(SPGR, qa_engine.w["console"])["realised"]
    clean = qa_engine.acquire(rp)
    a, b = qa_engine.acquire(rp, np.random.default_rng(1)), qa_engine.acquire(rp, np.random.default_rng(1))
    c = qa_engine.acquire(rp, np.random.default_rng(2))
    assert np.array_equal(clean, qa_engine.acquire(rp)) and np.array_equal(a, b) and not np.array_equal(a, c)
    assert 0 < np.abs(a - clean).std() < 0.2 * np.abs(clean).max()


def test_effects_can_be_switched_off(qa_world):
    rp = console.plan(dict(sequence="spgr", tr_ms=40.0, flip_deg=20.0, matrix=64, bandwidth_hz_per_px=60.0, te_ms=15.0), qa_world["console"])["realised"]
    full, bare = kspace.Engine(qa_world), kspace.Engine(qa_world, effects=kspace.ALL - {"b0shift", "t2star", "chemshift"})
    assert not np.allclose(full.acquire(rp), bare.acquire(rp), rtol=1e-3)


# ------------------------------------------------------------------ session: budget, log, replay
def test_scan_charges_and_refusals_cost_nothing(qa_engine):
    s = session.ScannerSession(episode_cfg(budget_s=12.0), engine=qa_engine)
    plan = s.plan({"protocol": SPGR})
    assert plan["accept"] and plan["fits_budget"] and s.budget()["used_s"] == 0
    resp, arrays = s.scan({"protocol": SPGR, "rois": [dict(id="c", y_mm=0, x_mm=0, radius_mm=20)]})
    assert s.budget()["used_s"] == pytest.approx(plan["scan_s"]) == pytest.approx(resp["cost_s"])
    assert arrays["magnitude"].dtype == np.float32 and arrays["magnitude"].shape == tuple(resp["shape"])
    assert resp["roi_readouts"][0]["n_pixels"] > 50 and "header" in resp and "timing" not in resp
    for bad, err in ((dict(SPGR, tr_ms=3.0), session.InvalidRequest), (dict(SPGR, seed=1), session.InvalidRequest),
                     (dict(SPGR, averages=64), session.BudgetExhausted)):
        with pytest.raises(err):
            s.scan({"protocol": bad})
    with pytest.raises(session.InvalidRequest):
        s.scan({"protocol": SPGR, "truth": True})
    assert s.budget()["used_s"] == pytest.approx(plan["scan_s"]) and s.budget()["scans"] == 1
    assert [e["type"] for e in s.events] == ["plan", "scan", "refused", "refused", "refused", "refused"]


def test_request_limit(qa_engine):
    s = session.ScannerSession(dict(episode_cfg(), max_requests=2), engine=qa_engine)
    s.plan({"protocol": SPGR})
    s.plan({"protocol": SPGR})
    with pytest.raises(session.RequestLimit):
        s.plan({"protocol": SPGR})


def test_a_repeated_scan_is_a_new_noise_draw_and_noise_draws_are_streams(qa_engine):
    s = session.ScannerSession(episode_cfg(), engine=qa_engine)
    a, _ = s.scan({"protocol": SPGR})
    b, _ = s.scan({"protocol": SPGR})
    assert a["data_sha256"] != b["data_sha256"]
    again = session.ScannerSession(episode_cfg(), engine=qa_engine).scan({"protocol": SPGR})[0]
    other = session.ScannerSession(dict(episode_cfg(), noise_draw=1), engine=qa_engine).scan({"protocol": SPGR})[0]
    assert again["data_sha256"] == a["data_sha256"] != other["data_sha256"]


def test_a_restart_cannot_refill_the_budget(tmp_path, qa_engine):
    """A restart continues from the log only with the receipt of that log (more attacks: tests/test_log_mac.py)."""
    cfg = episode_cfg(budget_s=20.0)
    log, state = str(tmp_path / "log"), str(tmp_path / "state")
    s = session.ScannerSession(cfg, log_dir=log, engine=qa_engine, state_dir=state)
    s.scan({"protocol": SPGR})
    used, chain, head = s.used, s.chain, s.head_path
    s.close()
    with pytest.raises(session.LogTampered):                            # a used directory without its receipt: nothing is served
        session.ScannerSession(cfg, log_dir=log, engine=qa_engine)
    r = session.ScannerSession(cfg, log_dir=log, engine=qa_engine, state_dir=state, resume_from=head)
    assert r.used == used and r.n_scans == 1 and r.restarts == 1 and r.events[0]["chain"] == chain
    r.scan({"protocol": SPGR})
    r.close()
    assert [e["type"] for e in session.read_log(os.path.join(log, session.LOG_NAME))] == ["scan", "restart", "scan"]


def _episode(engine, cfg=None):
    s = session.ScannerSession(cfg or episode_cfg(), engine=engine)
    s.plan({"protocol": SPGR})
    s.scan({"protocol": SPGR, "rois": [dict(id="c", y_mm=0, x_mm=0, radius_mm=20)]})
    s.scan({"protocol": dict(sequence="afi", tr_ms=30.0, flip_deg=60.0, matrix=64)})
    with pytest.raises(session.InvalidRequest):
        s.scan({"protocol": dict(SPGR, tr_ms=3.0)})
    return s


def test_replay_is_bit_exact_on_the_same_stack(qa_engine):
    s = _episode(qa_engine)
    rep = session.replay(s.cfg, s.events, head=s.receipt())             # a fresh world and engine from the seed
    assert rep["ok"] and rep["mode"] == "bit-exact" and rep["scans"] == 2 and rep["refused"] == 1
    assert rep["integrity"] == "anchored" and not rep["sealed"]         # nothing was delivered yet
    assert not session.replay(s.cfg, s.events)["ok"]                    # without the head a prefix could not be told apart
    assert rep["used_s"] == pytest.approx(s.used) and not rep["over_budget"] and rep["engine"] == session.ENGINE_ID
    assert json.loads(json.dumps(s.events)) == s.events                 # the log is plain JSON
    assert SEED not in json.dumps(s.events) and SEED not in json.dumps(s.info()) and SEED not in json.dumps(s.head())
    assert "task_id" not in s.info()


def test_replay_rejects_forged_reordered_and_foreign_logs(qa_engine):
    s = _episode(qa_engine)
    head = s.receipt()
    assert session.replay(s.cfg, s.events, engine=qa_engine, head=head)["ok"]
    forged = copy.deepcopy(s.events)
    forged[1]["cost_s"] = 0.1
    assert not session.replay(s.cfg, forged, engine=qa_engine, head=head)["ok"]
    swapped = copy.deepcopy(s.events)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    assert not session.replay(s.cfg, swapped, engine=qa_engine, head=head)["ok"]
    dropped = copy.deepcopy(s.events)[1:]
    assert not session.replay(s.cfg, dropped, engine=qa_engine, head=head)["ok"]
    other = session.replay(dict(s.cfg, seed=SEED[::-1]), s.events, head=head)      # the log of this world against another world
    assert not other["ok"] and "data" in other["problems"][0]


def test_replay_tolerates_rounding_but_not_other_data(qa_world, qa_engine):
    """Another BLAS or FFT build changes the last bits of an image. Replay then falls back to the logged
    fingerprint: a perturbation of 1e-7 is accepted and reported, one of 1e-2 is rejected."""
    s = _episode(qa_engine)

    class Perturbed(kspace.Engine):
        def __init__(self, world, rel):
            super().__init__(world)
            self.rel = rel

        def acquire(self, rp, rng=None):
            img = super().acquire(rp, rng)
            return img * (1 + self.rel * np.cos(np.arange(img.size).reshape(img.shape)))

    close = session.replay(s.cfg, s.events, engine=Perturbed(qa_world, 1e-7), head=s.receipt())
    assert close["ok"] and close["mode"] == "tolerance" and close["scans_within_tolerance_only"] == 2
    far = session.replay(s.cfg, s.events, engine=Perturbed(qa_world, 1e-2), head=s.receipt())
    assert not far["ok"]


def test_engine_version_is_part_of_the_chain(qa_engine, monkeypatch):
    s = _episode(qa_engine)
    head = s.receipt()
    monkeypatch.setattr(session, "ENGINE_ID", "mrgym-engine-0.0.0")
    assert not session.replay(s.cfg, s.events, engine=qa_engine, head=head)["ok"]


def test_region_readouts_on_a_rectangular_field_of_view(qa_engine):
    sc = session.LocalScanner(session.ScannerSession(episode_cfg(), engine=qa_engine))
    r = sc.scan(dict(sequence="se", tr_ms=300.0, fov_mm=320.0, matrix=96, phase_fov=0.75, pe_dir="x"), [dict(id="c", y_mm=0, x_mm=0, radius_mm=15)])
    assert r["shape"] == [1, 96, 72] and r["fov_yx_mm"] == [320.0, 240.0]
    again = sc.read_regions(0, [dict(id="c", y_mm=0, x_mm=0, radius_mm=15), dict(id="air", y_mm=-140, x_mm=0, radius_mm=8)])
    assert again[0]["mean_magnitude"][0] == pytest.approx(r["roi_readouts"][0]["mean_magnitude"][0], rel=1e-6)
    assert again[1]["mean_magnitude"][0] < 0.2 * again[0]["mean_magnitude"][0]
    box = sc.view(0)["bounding_box"]
    assert box and not box["touches_edge"] and -110 < box["y_mm"][0] < box["y_mm"][1] < 110
    with pytest.raises(session.InvalidRequest):
        sc.view(3)


def test_info_card_states_the_public_instrument_only(qa_engine):
    s = session.ScannerSession(episode_cfg(), engine=qa_engine)
    card = s.info()
    text = json.dumps(card)
    hidden = [f"{v:.6g}" for v in (s.world["system"]["tx_gain"], s.world["system"]["sigma0"], s.world["system"]["cf_offset_hz"])]
    assert not [h for h in hidden if h in text] and SEED not in text
    assert card["field_strength_t"] == 3.0 and set(card["sequences"]) == set(console.SEQUENCES)
    assert card["hardware"]["b1rms_limit_ut"] == s.world["console"]["b1rms_limit_ut"]


# ------------------------------------------------------------------ HTTP service and the single-file client
@pytest.fixture()
def service(tmp_path):
    cfg = episode_cfg(budget_s=40.0)
    cfg_path, ready_file, log_dir, state_dir = tmp_path / "episode.json", tmp_path / "ready", tmp_path / "log", tmp_path / "state"
    cfg_path.write_text(json.dumps(cfg))
    cfg_path.chmod(0o600)                                               # the config holds the seed
    proc = subprocess.Popen([sys.executable, "-m", "mrgym.scanner.service", "--config", str(cfg_path), "--log-dir", str(log_dir),
                             "--state-dir", str(state_dir), "--port", "0", "--ready-file", str(ready_file)],
                            env=dict(os.environ, OMP_NUM_THREADS="1"))
    try:
        for _ in range(200):
            if ready_file.exists():
                break
            time.sleep(0.05)
        assert ready_file.exists(), "service did not start"
        ready = json.loads(ready_file.read_text())
        handle = ready["episodes"][0]
        token = (state_dir / f"{handle}.token").read_text().strip()
        yield cfg, f"http://127.0.0.1:{ready['port']}", str(log_dir / handle), token, str(state_dir / f"{handle}.head")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_http_service_gives_the_same_log_as_in_process(service, qa_engine):
    cfg, url, log_dir, token, head = service
    sc = client.Scanner(url, token=token)
    assert sc.info()["budget_s"] == 40.0 and SEED not in json.dumps(sc.info())
    sc.plan(SPGR)
    r = sc.scan(SPGR, [dict(id="c", y_mm=0, x_mm=0, radius_mm=20)])
    with pytest.raises(client.ScannerError) as e:
        sc.scan(dict(SPGR, tr_ms=3.0))
    assert e.value.type == "invalid_request" and sc.budget()["scans"] == 1
    local = session.ScannerSession(cfg, engine=qa_engine)
    local.plan({"protocol": SPGR})
    lr, _ = local.scan({"protocol": SPGR, "rois": [dict(id="c", y_mm=0, x_mm=0, radius_mm=20)]})
    with pytest.raises(session.InvalidRequest):
        local.scan({"protocol": dict(SPGR, tr_ms=3.0), "rois": []})      # the client always sends the rois key
    assert r["data_sha256"] == lr["data_sha256"] and r["roi_readouts"] == lr["roi_readouts"]
    events = session.read_log(os.path.join(log_dir, session.LOG_NAME))
    assert [e["chain"] for e in events] == [e["chain"] for e in local.events]      # the same hash chain, link by link
    assert session.replay(cfg, events, head=session.read_head(head))["ok"]


def test_http_service_has_no_route_to_truth_or_reset(service):
    import urllib.error
    import urllib.request
    _, url, _, _, _ = service
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for method, path in (("GET", "/v1/truth"), ("GET", "/v1/world"), ("GET", "/v1/seed"), ("POST", "/v1/reset"), ("POST", "/v1/budget"),
                         ("GET", "/"), ("DELETE", "/v1/scan")):
        req = urllib.request.Request(url + path, method=method, data=b"{}" if method == "POST" else None)
        with pytest.raises(urllib.error.HTTPError) as e:
            opener.open(req, timeout=10)
        assert e.value.code in (404, 501), (method, path, e.value.code)
