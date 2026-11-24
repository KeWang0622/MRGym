"""Episode state of the virtual scanner: plan / scan, scan-time budget, append-only hash-chained log, replay.

The whole episode is a deterministic fold of (episode config, ordered requests):
    world              = f(seed, world spec)
    response to scan k = f(world, realised protocol, seed, noise draw, k)       k counts accepted scans
    budget, counters   = f(config, requests)
so a restarted service rebuilds its state from the log (a crash cannot refill the budget) and the verifier replays the
log from the episode config and checks every record. There is no reset and no route that returns anything derived
from the seed other than measurements.

Episode config (JSON; lives with the service and the verifier only, never in an agent-visible file):
    {"task_id": "<random token>", "seed": "<32 hex digits>", "world_spec": {...}, "budget_s": 60.0,
     "max_requests": 64, "task": {...public task card...}, "noise_draw": 0}
``noise_draw`` selects an independent noise stream for the same world; only admission and the verifier's reruns set it.
"""
import hashlib
import hmac
import json
import math
import os
import threading
import time

import numpy as np

from ..phys import cost, engine as engine_mod, world as W

LOG_VERSION = 1
LOG_NAME = "scan_log.jsonl"
MAX_BODY_BYTES = 65536
MAX_ROIS = 16


def _engine_id():
    """Hash of the physics sources: the service and the verifier must run the same engine to replay a log."""
    h = hashlib.sha256()
    d = os.path.dirname(os.path.abspath(engine_mod.__file__))
    for name in ("closed.py", "cost.py", "engine.py", "epg.py", "world.py"):
        with open(os.path.join(d, name), "rb") as f:
            h.update(f.read())
    with open(os.path.join(os.path.dirname(d), "sim.py"), "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


ENGINE_ID = _engine_id()


class ScannerError(Exception):
    """Refusal with a machine-readable type; nothing is charged."""
    type, http_status = "scanner_error", 400

    def __init__(self, message, **extra):
        super().__init__(message)
        self.message, self.extra = message, extra

    def to_json(self):
        return {"error": dict(type=self.type, message=self.message, **self.extra)}


class InvalidRequest(ScannerError):
    type, http_status = "invalid_request", 422


class BudgetExhausted(ScannerError):
    type, http_status = "budget_exhausted", 409


class RequestLimit(ScannerError):
    type, http_status = "request_limit", 429


def new_seed():
    """128 bits from the operating system."""
    return os.urandom(16).hex()


def load_config(path, environ=None):
    """Read an episode config. ``"seed_from_secret": true`` derives the seed as HMAC-SHA256(MRGYM_SECRET, task_id), so a
    published task directory holds nothing from which the world can be rebuilt. Secrets come from a file or the
    environment, never from the command line (process arguments are world-readable on a shared node)."""
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("seed_from_secret"):
        secret = (os.environ if environ is None else environ).get("MRGYM_SECRET")
        if not secret:
            raise SystemExit("mrfrl.scanner: this task needs MRGYM_SECRET in the environment of the scanner and the verifier")
        cfg["seed"] = hmac.new(secret.encode(), cfg["task_id"].encode(), hashlib.sha256).hexdigest()[:32]
    W.seed_int(cfg.get("seed"))
    return cfg


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _split_request(request):
    if not isinstance(request, dict):
        raise InvalidRequest("request must be a JSON object {\"protocol\": {...}, \"rois\": [...]}")
    unknown = sorted(set(request) - {"protocol", "rois"})
    if unknown:
        raise InvalidRequest(f"unknown request field(s) {unknown}; allowed: protocol, rois")
    rois = request.get("rois") or []
    if not isinstance(rois, list) or len(rois) > MAX_ROIS:
        raise InvalidRequest(f"rois must be a list of at most {MAX_ROIS} circles")
    out = []
    for i, r in enumerate(rois):
        if not isinstance(r, dict) or set(r) - {"id", "y_mm", "x_mm", "radius_mm"}:
            raise InvalidRequest("each roi is {\"id\": optional, \"y_mm\": .., \"x_mm\": .., \"radius_mm\": ..}")
        vals = [r.get(k) for k in ("y_mm", "x_mm", "radius_mm")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
            raise InvalidRequest("roi y_mm, x_mm, radius_mm must be finite numbers")
        if not (abs(vals[0]) <= 120 and abs(vals[1]) <= 120 and 1.0 <= vals[2] <= 120.0):
            raise InvalidRequest("roi centre must lie in the field of view and radius_mm in [1, 120]")
        out.append(dict(id=str(r.get("id", f"roi{i}"))[:32], y_mm=float(vals[0]), x_mm=float(vals[1]), radius_mm=float(vals[2])))
    return request.get("protocol"), out


def _r9(x):
    """Nine significant digits: more than the float32 images carry."""
    return float(f"{float(x):.9g}")


def roi_readouts(images, rois, fov_mm):
    """Console read-out of circular regions on complex images [n_images, n, n]. Pixel i sits at (i - n // 2) / n * FOV.

    Per region and image: pixel count, mean and SD of the magnitude, and magnitude / phase of the complex mean (the
    complex mean has no noise floor; the magnitude mean does).
    """
    n = images.shape[-1]
    pos = (np.arange(n) - n // 2) / n * fov_mm
    out = []
    for r in rois:
        mask = (pos[:, None] - r["y_mm"]) ** 2 + (pos[None, :] - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2
        npx = int(mask.sum())
        row = dict(id=r["id"], n_pixels=npx, mean_magnitude=[], sd_magnitude=[], complex_mean_magnitude=[],
                   complex_mean_phase_rad=[])
        for img in images:
            if npx == 0:
                for k in ("mean_magnitude", "sd_magnitude", "complex_mean_magnitude", "complex_mean_phase_rad"):
                    row[k].append(None)
                continue
            v = img[mask]
            cm = v.mean()
            row["mean_magnitude"].append(_r9(np.abs(v).mean()))
            row["sd_magnitude"].append(_r9(np.abs(v).std(ddof=1)) if npx > 1 else None)
            row["complex_mean_magnitude"].append(_r9(abs(cm)))
            row["complex_mean_phase_rad"].append(round(float(np.angle(cm)), 9))
        out.append(row)
    return out


def _default(sequence, d):
    if d is None:
        return {"required": True}
    if d == "min":
        return {"default": "console minimum for the requested matrix, bandwidth and spoiler"}
    if d == "seq":
        return {"default": cost.SYSTEM[f"{sequence}_phase_increment_deg"]}
    return {"default": d}


def scanner_card(cfg):
    """Everything the agent may know about the instrument and the episode. No seed, no hidden magnitude."""
    s = cost.SYSTEM
    return {
        "scanner": "mrfrl virtual scanner", "api_version": "v1", "task_id": cfg["task_id"],
        "field_of_view_mm": [s["fov_mm"], s["fov_mm"]], "slice_thickness_mm": s["slice_mm"], "slice": "single 2D slice",
        "geometry": "images are [n_images, matrix, matrix], row = y, column = x; pixel i is centred at "
                    "(i - matrix // 2) / matrix * field_of_view_mm",
        "budget_s": float(cfg["budget_s"]), "max_requests": int(cfg.get("max_requests", 64)),
        "calls": {"info": "free", "plan": "free dry run: timing, scan time, limits, accept / reject; never signal",
                  "scan": "charged scan_s; refused requests cost nothing", "budget": "free"},
        "cost": {"overhead_s_per_scan": s["overhead_s"], "dummy_time_ms_gradient_echo": s["dummy_ms"],
                 "dummy_cycles_inversion_recovery": s["ir_dummy_cycles"],
                 "rule": "scan_s = overhead + dummy cycles * cycle + cycle * matrix * averages; cycle = tr_ms (spgr, ir) "
                         "or tr_ms * (1 + tr_ratio) (afi)"},
        "hardware": {"gradient_max_mt_per_m": s["g_max_mt_m"], "slew_max_t_per_m_s": s["slew_t_m_s"],
                     "excitation_pulse_ms": s["rf_ms"], "inversion_pulse_ms": s["inv_ms"], "time_raster_ms": s["raster_ms"]},
        "sequences": {
            name: {"description": DESCRIPTIONS[name],
                   "parameters": {k: dict({"min": lo, "max": hi, "type": "integer" if kind == "i" else "number",
                                           "unit": cost.PARAMETER_UNITS[k]}, **_default(name, d))
                                  for k, (lo, hi, kind, d) in spec.items()}}
            for name, spec in cost.SEQUENCES.items()},
        "outputs": {"images": "magnitude and phase (rad), float32, one image per contrast (afi: 2; spgr: n_echoes)",
                    "roi_readouts": "per requested circle {id, y_mm, x_mm, radius_mm}: n_pixels, mean_magnitude, "
                                    "sd_magnitude, complex_mean_magnitude, complex_mean_phase_rad, one value per image",
                    "scale": "arbitrary but constant within an episode"},
        "task": cfg.get("task", {}),
    }


DESCRIPTIONS = {
    "spgr": "RF-spoiled gradient echo (optionally multi-echo); steady state with quadratic RF phase cycling and a "
            "gradient spoiler of the requested moment in every repetition",
    "afi": "actual flip-angle imaging: dual-TR RF-spoiled gradient echo, TR2 = tr_ratio * tr_ms, spoiler moments and RF "
           "phase increments in the ratio TR1 : TR2; image 1 is acquired in TR1, image 2 in TR2",
    "ir": "inversion recovery: adiabatic inversion, delay ti_ms, 90 degree (nominal) gradient-echo readout, one "
          "phase-encode line per tr_ms; signed signal (sign is in the phase image)",
}


class ScannerSession:
    def __init__(self, cfg, log_dir=None, engine=None):
        self.cfg = dict(cfg)
        self.seed = self.cfg["seed"]
        W.seed_int(self.seed)
        self.total = float(self.cfg["budget_s"])
        self.max_requests = int(self.cfg.get("max_requests", 64))
        self.noise_draw = int(self.cfg.get("noise_draw", 0))
        self.engine = engine or engine_mod.Engine(W.make_world(self.seed, self.cfg.get("world_spec")))
        self.world = self.engine.world
        self.lock = threading.Lock()
        self.events, self.used, self.n_scans = [], 0.0, 0
        self.chain = _sha(f"mrfrl-scanner-log-v{LOG_VERSION}:{self.cfg['task_id']}:{ENGINE_ID}".encode())   # no seed in here
        self.log_path = os.path.join(log_dir, LOG_NAME) if log_dir else None
        self._fh = None
        if self.log_path:
            os.makedirs(log_dir, exist_ok=True)
            if os.path.exists(self.log_path):
                for rec in read_log(self.log_path):                     # a restart must not refill the budget
                    self._apply(rec)
            self._fh = open(self.log_path, "a", buffering=1)

    # ------------------------------------------------------------------ state
    def _apply(self, rec):
        self.events.append(rec)
        self.chain = rec["chain"]
        if rec["type"] == "scan":
            self.used += rec["cost_s"]
            self.n_scans += 1

    def _append(self, body, t0):
        body = dict(v=LOG_VERSION, event=len(self.events), **body, prev=self.chain)
        rec = dict(body, chain=_sha((self.chain + _canon(body)).encode()),
                   wall={"t_utc": round(t0, 3), "ms": round((time.time() - t0) * 1e3, 2)})
        if self._fh:
            self._fh.write(_canon(rec) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        self._apply(rec)
        return rec

    @property
    def remaining(self):
        return self.total - self.used

    def info(self):
        return scanner_card(self.cfg)

    def budget(self):
        return {"budget_s": self.total, "used_s": round(self.used, 6), "remaining_s": round(self.remaining, 6),
                "scans": self.n_scans, "requests": len(self.events), "max_requests": self.max_requests}

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def _plan(self, protocol):
        return cost.plan(protocol, b1rms_limit_ut=self.world["system"]["b1rms_limit_ut"])

    def _noise_rng(self, realised, scan_index):
        h = int(_sha(_canon(realised).encode())[:32], 16)
        key = (W.STREAMS["noise"], self.noise_draw, scan_index) + tuple((h >> (32 * i)) & 0xFFFFFFFF for i in range(4))
        return np.random.default_rng(np.random.SeedSequence(entropy=W.seed_int(self.seed), spawn_key=key))

    # ------------------------------------------------------------------ the two request types
    def plan(self, request):
        """Free dry run; logged and counted against the request limit. Never touches the world's signal."""
        with self.lock:
            t0 = time.time()
            if len(self.events) >= self.max_requests:
                raise RequestLimit(f"request limit of {self.max_requests} reached", max_requests=self.max_requests)
            protocol = request.get("protocol") if isinstance(request, dict) and "protocol" in request else request
            out = self._plan(protocol)
            out["fits_budget"] = bool(out["accept"] and out["scan_s"] <= self.remaining + 1e-9)
            out["remaining_s"] = round(self.remaining, 6)
            self._append(dict(type="plan", accept=out["accept"], scan_s=out.get("scan_s"),
                              request=_canon(protocol)[:2000] if isinstance(protocol, (dict, list)) else None), t0)
            return out

    def scan(self, request):
        """Returns (response dict, {"magnitude", "phase"} float32 arrays). Raises a ScannerError on refusal."""
        with self.lock:
            t0 = time.time()
            if len(self.events) >= self.max_requests:
                raise RequestLimit(f"request limit of {self.max_requests} reached", max_requests=self.max_requests)
            try:
                protocol, rois = _split_request(request)
                pl = self._plan(protocol)
                if not pl["accept"]:
                    raise InvalidRequest("; ".join(v["message"] for v in pl["violations"]), violations=pl["violations"])
            except InvalidRequest as e:
                self._append(dict(type="refused", reason=e.type, message=e.message[:500],
                                  request=_canon(request)[:2000] if isinstance(request, (dict, list)) else None), t0)
                raise
            rp, cost_s = pl["realised"], pl["scan_s"]
            if cost_s > self.remaining + 1e-9:
                self._append(dict(type="refused", reason="budget_exhausted", realised=rp, cost_s=cost_s,
                                  remaining_s=round(self.remaining, 6)), t0)
                raise BudgetExhausted(f"scan needs {cost_s:.3f} s, {self.remaining:.3f} s left", cost_s=cost_s,
                                      remaining_s=round(self.remaining, 6))
            data = self.engine.acquire(rp, self._noise_rng(rp, self.n_scans))
            arrays = {"magnitude": np.abs(data).astype(np.float32), "phase": np.angle(data).astype(np.float32)}
            readouts = roi_readouts(data, rois, self.world["fov_mm"])
            rec = self._append(dict(type="scan", scan_index=self.n_scans, realised=rp, cost_s=cost_s,
                                    remaining_s=round(self.remaining - cost_s, 6), rois=rois,
                                    shape=list(data.shape), data_sha256=_sha(arrays["magnitude"].tobytes() + arrays["phase"].tobytes()),
                                    readouts_sha256=_sha(_canon(readouts).encode())), t0)
            resp = {k: rec[k] for k in ("event", "scan_index", "realised", "cost_s", "remaining_s", "shape", "data_sha256")}
            resp.update(timing=pl["timing"], limits=pl["limits"], scan_s_parts=pl["scan_s_parts"], roi_readouts=readouts)
            return resp, arrays


def read_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def replay(cfg, events, engine=None):
    """Re-run a log from the episode config. ``ok`` is False on any inconsistency.

    Checks: version, numbering, hash chain (whose first link carries the engine id), and for every accepted scan that
    a fresh simulation of the recorded request reproduces cost, remaining budget, image digest and read-out digest.
    Bit-exact digests need the same numpy / FFT build as the service (same image for scanner and verifier).
    """
    fresh = ScannerSession(cfg, engine=engine)
    problems = []
    for i, rec in enumerate(events):
        body = {k: v for k, v in rec.items() if k not in ("chain", "wall")}
        if rec.get("v") != LOG_VERSION or rec.get("event") != i or rec.get("prev") != fresh.chain \
                or rec.get("chain") != _sha((fresh.chain + _canon(body)).encode()):
            problems.append(f"event {i}: broken numbering or hash chain" + (" (or another engine version)" if i == 0 else ""))
            break
        if rec["type"] == "scan":
            try:
                new, _ = fresh.scan({"protocol": {k: v for k, v in rec["realised"].items()
                                                  if v is not None and k not in ("tr2_ms", "echo_times_ms", "readout_flip_deg")},
                                     "rois": rec["rois"]})
            except ScannerError as e:
                problems.append(f"event {i}: replay refused the scan ({e.type})")
                break
            last = fresh.events[-1]
            diff = [k for k in ("scan_index", "realised", "cost_s", "remaining_s", "shape", "data_sha256", "readouts_sha256")
                    if last[k] != rec[k]]
            if diff:
                problems.append(f"event {i}: replay differs in {diff}")
                break
            fresh.events[-1] = rec
            fresh.chain = rec["chain"]
        else:
            fresh._apply(rec)
    return {"ok": not problems, "problems": problems, "events": len(events), "scans": fresh.n_scans,
            "plans": sum(e["type"] == "plan" for e in events), "refused": sum(e["type"] == "refused" for e in events),
            "used_s": round(fresh.used, 6), "budget_s": fresh.total, "over_budget": fresh.used > fresh.total + 1e-6}


class LocalScanner:
    """In-process stand-in for ``client.Scanner`` with the same four calls, for controls and tests."""

    def __init__(self, session):
        self.session = session

    def info(self):
        return self.session.info()

    def budget(self):
        return self.session.budget()

    def plan(self, protocol):
        return self.session.plan({"protocol": protocol})

    def scan(self, protocol, rois=None, out=None):
        resp, arrays = self.session.scan({"protocol": protocol, "rois": rois or []})
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            for k, a in arrays.items():
                np.save(f"{out}_{k}.npy", a)
        return dict(resp, arrays=arrays)
