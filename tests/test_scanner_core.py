"""Checks of the scanner core, its physics and family F1. Run:  python3 tests/test_scanner_core.py   (pytest also works).

No network beyond loopback, no GPU. Each check states the property the environment relies on.
"""
import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mrfrl import controls  # noqa: E402
from mrfrl.phys import consistency, cost, engine, world as W  # noqa: E402
from mrfrl.scanner import client, core  # noqa: E402
from mrfrl.t1_transmit import task  # noqa: E402

SEED = "9f3c2a7d41e8b6505d1c0e7fa2b39468"
SPGR = {"sequence": "spgr", "tr_ms": 20.0, "flip_deg": 10.0, "matrix": 64, "averages": 2}


def _cfg(level=2, seed=SEED, **kw):
    return dict(task.make_instance(seed, level, task_id="t1tx-test"), **kw)


def test_tier_consistency():
    """Tiers agree where they must (closed forms vs matrix Bloch; EPG vs two independent simulations; tier 2 -> tier 1
    in the limits) and differ where they must (short TR, high flip; corrected vs draft inversion recovery)."""
    r = consistency.run(with_table=False)
    assert not r["failed"], r["failed"]
    assert r["closed_vs_matrix"]["n"] > 20000 and r["epg_vs_independent"]["cases"] >= 8


def test_world_regions_hold_60_pixels_and_do_not_touch():
    for i in range(20):
        w = W.make_world(f"{i:02x}" + SEED[2:], task.LEVELS[1 + i % 3]["world_spec"])
        pos = (np.arange(96) - 48) / 96 * w["fov_mm"]
        reg = W.regions_public(w)
        for r in reg:
            n = int(((pos[:, None] - r["y_mm"]) ** 2 + (pos[None, :] - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2).sum())
            assert n >= 60, n
        ins = [c["shape"] for c in w["classes"][1:]]
        for a in range(len(ins)):
            for b in range(a):
                d = np.hypot(ins[a]["y"] - ins[b]["y"], ins[a]["x"] - ins[b]["x"])
                assert d > max(ins[a]["a"], ins[a]["b"]) + max(ins[b]["a"], ins[b]["b"]) + 0.01
        y, x = np.array([r["y_mm"] for r in reg]) / w["fov_mm"], np.array([r["x_mm"] for r in reg]) / w["fov_mm"]
        wts = W.class_weights(w, y, x)
        assert np.allclose(wts.sum(0), 1, atol=1e-6) and np.allclose(wts[1:].max(0), 1, atol=1e-6)   # pure insert at the centre
        for c in w["classes"]:
            assert c["t2_ms"] <= c["t1_ms"] and all(k in c for k in ("pd", "t2p_ms", "adc_um2_ms"))
    try:
        W.make_world("1234")
        raise AssertionError("short seed accepted")
    except ValueError:
        pass


def test_plan_is_priced_and_never_returns_signal():
    p = cost.plan(dict(sequence="spgr", tr_ms=20, flip_deg=10))
    assert p["accept"] and p["realised"]["te_ms"] == p["timing"]["te_min_ms"]
    assert abs(p["scan_s"] - (3.0 + 4.0 + 0.020 * 96)) < 1e-9 and p["scan_s_parts"]["dummy_s"] == 4.0
    flat = json.dumps(p).lower()
    assert not any(k in flat for k in ("signal", "snr", "noise", "t1", "b1_map", "magnitude"))
    # TE is tied to the readout: narrower bandwidth and larger matrix both lengthen the minimum
    te = lambda **kw: cost.plan(dict(sequence="spgr", tr_ms=40, flip_deg=10, **kw))["timing"]["te_min_ms"]   # noqa: E731
    assert te(bandwidth_hz_per_px=100) > te(bandwidth_hz_per_px=300) > te(bandwidth_hz_per_px=1000)
    assert te(matrix=192) > te(matrix=64)
    # a stronger spoiler costs repetition time
    trm = lambda a: cost.plan(dict(sequence="spgr", tr_ms=40, flip_deg=10, spoiler_mt_ms_per_m=a))["timing"]["tr_min_ms"]  # noqa: E731
    assert trm(400) > trm(100) + 10
    codes = lambda prot, **kw: [v["code"] for v in cost.plan(prot, **kw)["violations"]]   # noqa: E731
    assert codes(dict(sequence="spgr", tr_ms=4, flip_deg=10)) == ["TR_BELOW_MINIMUM"]
    assert codes(dict(sequence="spgr", tr_ms=20, flip_deg=10, te_ms=1.0)) == ["TE_BELOW_MINIMUM"]
    assert codes(dict(sequence="spgr", tr_ms=20, flip_deg=10, seed=1)) == ["UNKNOWN_PARAMETER"]
    assert codes(dict(sequence="spgr", tr_ms=20)) == ["MISSING_PARAMETER"]
    assert codes(dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=97)) == ["OUT_OF_RANGE"]
    assert codes(dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=192, spoiler_mt_ms_per_m=10)) == ["SPOILER_BELOW_MINIMUM"]
    assert codes(dict(sequence="spgr", tr_ms=20, flip_deg=10, matrix=192, bandwidth_hz_per_px=2000)) == ["GRADIENT_LIMIT"]
    assert "RF_POWER_LIMIT" in codes(dict(sequence="spgr", tr_ms=8, flip_deg=90, spoiler_mt_ms_per_m=20), b1rms_limit_ut=1.6)
    assert codes(dict(sequence="ir", tr_ms=1000, ti_ms=995)) == ["TI_NOT_REALISABLE"]
    assert codes(dict(sequence="epi")) == ["UNKNOWN_SEQUENCE"]
    ir = cost.plan(dict(sequence="ir", tr_ms=1000, ti_ms=300, matrix=48))
    assert ir["accept"] and ir["scan_s"] == 3.0 + 1.0 + 48.0          # the cheapest inversion-recovery scan: 52 s
    # replay relies on plan(realised) == realised
    for prot in (SPGR, dict(sequence="afi", tr_ms=30, flip_deg=60), dict(sequence="spgr", tr_ms=60, flip_deg=10, n_echoes=4),
                 dict(sequence="ir", tr_ms=1500.004, ti_ms=200.003)):
        r = cost.plan(prot)["realised"]
        again = cost.plan({k: v for k, v in r.items() if v is not None and k not in ("tr2_ms", "echo_times_ms", "readout_flip_deg")})
        assert again["realised"] == r, (r, again["realised"])


def test_noise_law_small_matrices_are_not_free():
    """Measured noise SD in air follows matrix^1.5 * sqrt(bandwidth) / sqrt(averages); a region mean pays for it."""
    w = W.make_world(SEED, task.LEVELS[1]["world_spec"])
    eng, rng = engine.Engine(w), np.random.default_rng(0)
    sd = {}
    for n, bw, avg in ((48, 300.0, 1), (96, 300.0, 1), (192, 300.0, 1), (96, 1200.0, 1), (96, 300.0, 4)):
        rp = cost.plan(dict(sequence="spgr", tr_ms=30, flip_deg=10, matrix=n, bandwidth_hz_per_px=bw, averages=avg))["realised"]
        img = eng.acquire(rp, rng) - eng.clean(rp)
        sd[(n, bw, avg)] = float(np.sqrt(np.mean(np.abs(img) ** 2) / 2))
    ref = sd[(96, 300.0, 1)]
    assert abs(ref / w["system"]["noise_sd"] - 1) < 0.03
    assert abs(sd[(48, 300.0, 1)] / ref - 0.5 ** 1.5) < 0.03 and abs(sd[(192, 300.0, 1)] / ref - 2 ** 1.5) < 0.1
    assert abs(sd[(96, 1200.0, 1)] / ref - 2.0) < 0.06 and abs(sd[(96, 300.0, 4)] / ref - 0.5) < 0.02
    # region mean of a fixed physical circle: SD ~ matrix^0.5, acquisition time ~ matrix. In the uncommitted draft the
    # region-mean SD did not depend on the matrix at all.
    roi_sd = {n: sd[(n, 300.0, 1)] / np.sqrt(np.pi * (0.048 * n) ** 2) for n in (48, 96, 192)}
    assert 1.3 < roi_sd[96] / roi_sd[48] < 1.5 and 1.3 < roi_sd[192] / roi_sd[96] < 1.6


def test_engine_images_follow_the_documented_mechanisms():
    w = W.make_world(SEED, task.LEVELS[2]["world_spec"])
    eng1, eng2 = engine.Engine(w, tier=1), engine.Engine(w, tier=2)
    reg = W.regions_public(w)[0]
    cls = w["classes"][w["regions"][0]["cls"]]
    pos = (np.arange(96) - 48) / 96 * w["fov_mm"]
    mask = (pos[:, None] - reg["y_mm"]) ** 2 + (pos[None, :] - reg["x_mm"]) ** 2 <= reg["radius_mm"] ** 2
    # multi-echo: T2* decay and off-resonance phase with the documented sign
    rp = cost.plan(dict(sequence="spgr", tr_ms=60, flip_deg=10, n_echoes=5, echo_spacing_ms=5.0))["realised"]
    img = eng2.clean(rp)
    te = np.array(rp["echo_times_ms"])
    sig = np.array([img[e][mask].mean() for e in range(5)])
    t2s = 1 / (1 / cls["t2_ms"] + 1 / cls["t2p_ms"])
    fit = np.polyfit(te, np.log([np.abs(img[e][mask]).mean() for e in range(5)]), 1)   # magnitudes: no dephasing across the region
    assert abs(-1 / fit[0] / t2s - 1) < 0.02, (-1 / fit[0], t2s)
    y, x = np.meshgrid(pos / w["fov_mm"], pos / w["fov_mm"], indexing="ij")
    f_true = float(W.b0_hz(w, y, x)[mask].mean())
    f_meas = np.polyfit(te * 1e-3, np.unwrap(np.angle(sig)), 1)[0] / (2 * np.pi)
    assert abs(f_meas - f_true) < 0.5, (f_meas, f_true)
    # tier 2 differs from tier 1 at short TR and high flip, not at low flip and long TR with a strong spoiler
    hi = cost.plan(dict(sequence="spgr", tr_ms=6, flip_deg=40, matrix=64, spoiler_mt_ms_per_m=20, bandwidth_hz_per_px=600))["realised"]
    lo = cost.plan(dict(sequence="spgr", tr_ms=40, flip_deg=4, matrix=64, spoiler_mt_ms_per_m=500))["realised"]
    m64 = np.abs(eng1.clean(lo)) > 0.3 * np.abs(eng1.clean(lo)).max()
    gap = lambda rp: float(np.median(np.abs(np.abs(eng2.clean(rp))[m64] / np.abs(eng1.clean(rp))[m64] - 1)))   # noqa: E731
    assert gap(hi) > 0.03 and gap(lo) < 0.003, (gap(hi), gap(lo))


def test_same_seed_same_episode_and_noise_stream():
    cfg = _cfg()
    a, b = core.ScannerSession(cfg), core.ScannerSession(cfg)
    ra, da = a.scan({"protocol": SPGR})
    rb, db = b.scan({"protocol": SPGR})
    assert ra["data_sha256"] == rb["data_sha256"] and np.array_equal(da["magnitude"], db["magnitude"])
    ra2, da2 = a.scan({"protocol": SPGR})
    assert ra2["data_sha256"] != ra["data_sha256"], "repeating a scan must give a new noise realisation"
    c = core.ScannerSession(cfg)
    c.scan({"protocol": dict(SPGR, flip_deg=11.0)})          # a different first scan ...
    rc2, _ = c.scan({"protocol": SPGR})
    assert rc2["data_sha256"] == ra2["data_sha256"], "noise is a function of (seed, scan index, protocol)"
    d = core.ScannerSession(dict(cfg, noise_draw=1))
    assert d.scan({"protocol": SPGR})[0]["data_sha256"] != ra["data_sha256"]
    other = core.ScannerSession(_cfg(seed="00" + SEED[2:]))
    assert other.scan({"protocol": SPGR})[0]["data_sha256"] != ra["data_sha256"]
    # averaging two repeats lowers the noise by sqrt(2)
    clean = np.abs(a.engine.clean(ra["realised"]))
    n1, n2 = da["magnitude"] - clean, (da["magnitude"] + da2["magnitude"]) / 2 - clean
    inside = clean > 0.5 * clean.max()
    assert 1.25 < n1[inside].std() / n2[inside].std() < 1.6


def test_budget_refusals_limit_restart_and_replay():
    cfg = _cfg(budget_s=30.0, max_requests=12)
    with tempfile.TemporaryDirectory() as d:
        s = core.ScannerSession(cfg, d)
        rois = [dict(id="a", y_mm=0.0, x_mm=10.0, radius_mm=8.0)]
        s.plan({"protocol": SPGR})
        r1, _ = s.scan({"protocol": SPGR, "rois": rois})
        assert r1["cost_s"] == 3.0 + 4.0 + 0.02 * 64 * 2 and len(r1["roi_readouts"]) == 1
        r2, _ = s.scan({"protocol": SPGR})
        try:
            s.scan({"protocol": dict(SPGR, averages=8)})
            raise AssertionError("must be refused")
        except core.BudgetExhausted as e:
            assert e.extra["remaining_s"] == round(30.0 - 2 * r1["cost_s"], 6)
        for bad in ({"protocol": {"sequence": "spgr"}}, {"protocol": dict(SPGR, seed=1)}, {"protocol": SPGR, "truth": 1},
                    {"protocol": SPGR, "rois": [{"y_mm": 0, "x_mm": 0, "radius_mm": 500}]}, "scan please", None):
            try:
                s.scan(bad)
                raise AssertionError(f"accepted {bad!r}")
            except core.InvalidRequest:
                pass
        used, head = s.used, s.chain
        assert used == 2 * r1["cost_s"] and [e["type"] for e in s.events] == ["plan", "scan", "scan"] + ["refused"] * 7
        s.close()
        s2 = core.ScannerSession(cfg, d)                      # the service crashed and was restarted
        assert (s2.used, s2.chain, s2.n_scans) == (used, head, 2)
        try:
            for _ in range(5):
                s2.plan({"protocol": SPGR})
            raise AssertionError("request limit not enforced")
        except core.RequestLimit:
            assert len(s2.events) == 12
        s2.close()
        events = core.read_log(os.path.join(d, core.LOG_NAME))
    good = json.loads(json.dumps(events))
    rep = core.replay(cfg, good)
    assert rep["ok"] and rep["scans"] == 2 and not rep["over_budget"], rep
    cheaper = copy.deepcopy(good)
    cheaper[1]["cost_s"] = 0.1
    forged = copy.deepcopy(good)
    forged[2]["data_sha256"] = "0" * 64
    for bad in (cheaper, good[1:], [good[0], good[2], good[1]] + good[3:], forged):
        assert not core.replay(cfg, bad)["ok"]
    assert not core.replay(_cfg(seed="00" + SEED[2:], budget_s=30.0, max_requests=12), good)["ok"]   # another world


def test_nothing_secret_is_served():
    cfg = _cfg()
    s = core.ScannerSession(cfg)
    L = core.LocalScanner(s)
    resp = L.scan(SPGR, controls._rois(L.info()))
    resp.pop("arrays")
    public = json.dumps([L.info(), L.plan(SPGR), resp, L.budget(), task.instruction(cfg)])
    w = s.world
    assert SEED not in public and "seed" not in public.lower().replace("seed_attacker", "")
    hidden = [w["system"]["tx_gain"], w["system"]["noise_sd"], w["system"]["inv_eff"], w["system"]["b1rms_limit_ut"],
              w["system"]["cf_offset_hz"]] + [c[k] for c in w["classes"] for k in ("t1_ms", "t2_ms", "t2p_ms", "pd", "adc_um2_ms")]
    for v in hidden:
        assert f"{v:.5g}" not in public, v
    assert SEED not in cfg["task_id"] and SEED[:8] not in json.dumps(cfg["task"])
    assert len(core.new_seed()) == 32 and core.new_seed() != core.new_seed()


def test_client_is_self_contained():
    """The agent-side file must not import the package (it would bring the generator into the agent's sandbox)."""
    tree = ast.parse(open(client.__file__).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith("mrfrl"), node.module
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith(("mrfrl", "numpy", "scipy")) for a in node.names)


def test_reward_gates_and_zero_at_the_prior():
    cfg = _cfg()
    w = W.make_world(cfg["seed"], cfg["world_spec"])
    truth = {k: v["t1_ms"] for k, v in W.truth(w).items()}
    exact = {"units": "ms", "t1_ms": truth}
    prior = controls._prior_answer(dict(task=cfg["task"]))
    tol = 0.06
    assert task.score(cfg, exact, tol, 3)["reward"] == 1.0
    assert task.score(cfg, exact, tol, 0)["reward"] == 0.0, "the exact answer without a scan in the log earns nothing"
    assert task.score(cfg, exact, tol, 3, log_ok=False)["reward"] == 0.0
    assert task.score(cfg, prior, tol, 3)["reward"] == 0.0, "the zero-measurement prior answer is the zero of the score"
    off = lambda f: {"units": "ms", "t1_ms": {k: v * f for k, v in truth.items()}}   # noqa: E731
    e_prior = task.score(cfg, prior, tol, 3)["e_prior"]
    assert e_prior > 0.25
    rewards = [task.score(cfg, off(1 + e), tol, 3)["reward"] for e in (0.03, 0.0599, 0.1, 0.2, 0.9 * e_prior)]
    assert rewards[0] == rewards[1] == 1.0 and rewards[1] > rewards[2] > rewards[3] > rewards[4] > 0.0, rewards
    assert task.score(cfg, off(5.0), tol, 3)["reward"] == 0.0                 # worse than the prior
    sec = task.score(cfg, {"units": "ms", "t1_ms": {k: v / 1000 for k, v in truth.items()}}, tol, 3)
    assert sec["reward"] == 0.0 and "seconds" in sec["diagnostic"]
    assert "units" in task.score(cfg, {"units": "s", "t1_ms": truth}, tol, 3)["diagnostic"]
    for bad in (None, [], {"units": "ms"}, {"units": "ms", "t1_ms": {"R1": 800.0}},
                {"units": "ms", "t1_ms": dict(truth, R1=float("nan"))}, {"units": "ms", "t1_ms": truth, "extra": 1}):
        assert task.score(cfg, bad, tol, 3)["reward"] == 0.0


def test_controls_on_one_instance():
    """Reference close to truth; blind VFA, brute force, IR-only and wrong units far; IR works only when time is free."""
    cfg = _cfg(level=2)
    w = W.make_world(cfg["seed"], cfg["world_spec"])
    eng = engine.Engine(w)
    err = {}
    for name, (fn, scale) in controls.policies().items():
        answer, session, problem = controls.run_policy(cfg, fn, 0, eng, scale)
        assert problem is None, (name, problem)
        assert session.used <= session.total + 1e-9
        s = task.score(cfg, answer, 0.08, session.n_scans, world=w)
        err[name] = (s["error"], s["reward"], session.n_scans)
    assert err["reference"][0] < 0.08 and err["reference"][1] == 1.0, err
    assert err["prior"][1] == 0.0 and err["prior"][2] == 0
    assert err["blind_vfa"][0] > 0.12 and err["brute_force"][0] > 0.12, err
    assert err["ir_only"][2] <= 1 and err["ir_only"][0] > 0.3, err
    assert err["ir_unpriced"][0] < 0.03 and err["ir_unpriced"][2] >= 3, err
    assert err["wrong_units"][0] is None and err["wrong_units"][1] == 0.0
    assert err["seed_attacker"][1] < 1.0
    assert core.replay(dict(cfg, noise_draw=0), controls.run_policy(cfg, controls.reference, 0, eng)[1].events, engine=eng)["ok"]


def test_seed_attack_works_on_a_guessable_seed_only():
    """Positive control: with a seed from a small space the public generator IS the answer key."""
    weak = f"{123:032x}"
    for seed, expect in ((weak, True), (SEED, False)):
        cfg = _cfg(level=1, seed=seed)
        w = W.make_world(seed, cfg["world_spec"])
        diag = {}
        answer, session, problem = controls.run_policy(cfg, lambda sc: controls.seed_attacker(sc, diagnostics=diag), 0)
        assert problem is None, problem
        assert diag["world_recovered"] is expect and diag["repeat_scan_identical"] is False, diag
        s = task.score(cfg, answer, 0.05, session.n_scans, world=w)
        assert (s["error"] < 1e-9) is expect, s["error"]


def test_http_service_end_to_end():
    """Service in its own process (seed in a config file, not on the command line); the agent side uses only the
    single-file client; the verifier replays the log and scores the deliverable."""
    cfg = _cfg(level=2, max_requests=30)
    with tempfile.TemporaryDirectory() as d:
        hidden, logs, ws = os.path.join(d, "hidden"), os.path.join(d, "log"), os.path.join(d, "ws")
        os.makedirs(hidden), os.makedirs(ws)
        with open(os.path.join(hidden, "episode.json"), "w") as f:
            json.dump(cfg, f)
        port_file = os.path.join(d, "port")
        env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="2")
        argv = [sys.executable, "-m", "mrfrl.scanner.service", "--config", os.path.join(hidden, "episode.json"),
                "--log-dir", logs, "--port", "0", "--port-file", port_file]
        assert SEED not in " ".join(argv)
        proc = subprocess.Popen(argv, env=env, cwd=d)
        try:
            for _ in range(200):
                if os.path.exists(port_file):
                    break
                time.sleep(0.05)
            url = "http://127.0.0.1:" + open(port_file).read().strip()
            mrscan = os.path.join(ws, "mrscan.py")                 # the agent has this one file, nothing else
            with open(mrscan, "w") as f:
                f.write(open(client.__file__).read())
            cli = lambda *a: subprocess.run([sys.executable, mrscan, "--url", url, *a], capture_output=True, text=True, cwd=ws,   # noqa: E731
                                            env={"PATH": os.environ.get("PATH", "")})
            info = json.loads(cli("info").stdout)
            assert info["budget_s"] == cfg["budget_s"] and info["task"]["regions"] == cfg["task"]["regions"]
            with open(os.path.join(ws, "regions.json"), "w") as f:
                json.dump(info["task"]["regions"], f)
            p = json.loads(cli("plan", "--protocol", json.dumps(SPGR)).stdout)
            assert p["accept"] and p["fits_budget"]
            r = cli("scan", "--protocol", json.dumps(SPGR), "--rois", "regions.json", "--out", "scans/a")
            out = json.loads(r.stdout)
            assert len(r.stdout) < 6000, "images must not enter the model context"
            mag = np.load(os.path.join(ws, "scans", "a_magnitude.npy"))
            assert mag.shape == (1, 64, 64) and mag.dtype == np.float32 and len(out["roi_readouts"]) == len(info["task"]["regions"])
            bad = cli("scan", "--protocol", json.dumps(dict(SPGR, tr_ms=1)))
            assert bad.returncode == 3 and json.loads(bad.stdout)["error"]["type"] == "invalid_request"
            import urllib.error
            import urllib.request
            for path, method in (("/v1/reset", "POST"), ("/v1/truth", "GET"), ("/v1/log", "GET"), ("/v1/seed", "GET"), ("/", "GET")):
                try:
                    urllib.request.urlopen(urllib.request.Request(url + path, method=method, data=b"{}" if method == "POST" else None))
                    raise AssertionError(path)
                except urllib.error.HTTPError as e:
                    assert e.code == 404
            answer = controls.reference(client_adapter(client.Scanner(url)))      # the same policy over HTTP
            with open(os.path.join(ws, "answer.json"), "w") as f:
                json.dump(answer, f)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
        res = task.verify(cfg, logs, os.path.join(ws, "answer.json"), tolerance=0.10)
        assert res["replay"]["ok"] and res["replay"]["scans"] >= 3 and res["gate_measured"], res["replay"]
        assert res["reward"] > 0.5 and res["error"] < 0.15, res
        with open(os.path.join(ws, "answer.json"), "w") as f:
            json.dump(controls._prior_answer(dict(task=cfg["task"])), f)
        assert task.verify(cfg, logs, os.path.join(ws, "answer.json"), tolerance=0.10)["reward"] == 0.0
        assert not any(SEED in open(os.path.join(ws, n)).read() for n in os.listdir(ws) if n.endswith((".json", ".py")))


class client_adapter:
    """Gives ``client.Scanner`` the return shape the policies expect (they only read JSON fields)."""

    def __init__(self, sc):
        self.sc = sc

    def __getattr__(self, name):
        return getattr(self.sc, name)


if __name__ == "__main__":
    t_all = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            fn()
            print(f"ok   {name}  ({time.time() - t0:.1f} s)")
    print(f"all passed in {time.time() - t_all:.1f} s")
