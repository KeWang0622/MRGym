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

Replay and floating point. Every accepted scan is logged with (a) the SHA-256 of its float32 images and (b) a
fingerprint: the mean magnitude of 8 x 8 blocks of every image. A replay on the same numerical stack reproduces the
digest ("bit-exact"). On another machine, BLAS or FFT build the images can differ in the last bits; the replay then
accepts a scan whose fingerprint agrees within ``REPLAY_RTOL`` of the image's largest block ("tolerance") and says so
in its report. The hash chain itself covers only logged text, so it never depends on arithmetic.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
import time

import numpy as np

from . import console, kspace
from . import world as W

LOG_VERSION = 2
LOG_NAME = "scan_log.jsonl"
MAX_BODY_BYTES = 65536
MAX_ROIS = 16
MAX_POSITION_MM = 300.0
REPLAY_RTOL = 1e-4
FINGERPRINT_BLOCKS = 8


# The engine version names the numerical behaviour of the scanner: world generation, signal models, k-space formation,
# console arithmetic and noise streams. It is bumped by hand whenever the same (seed, protocol) can give other data,
# and only then; tests/test_engine_version.py pins it together with reference numbers. Recorded logs replay only on
# the engine version that wrote them.
ENGINE_VERSION = "0.1.0"
ENGINE_ID = "mrgym-engine-" + ENGINE_VERSION


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
            raise SystemExit("mrgym.scanner: this task needs MRGYM_SECRET in the environment of the scanner and the verifier")
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
        if not (abs(vals[0]) <= MAX_POSITION_MM and abs(vals[1]) <= MAX_POSITION_MM and 1.0 <= vals[2] <= MAX_POSITION_MM):
            raise InvalidRequest(f"roi centre must lie within {MAX_POSITION_MM:g} mm of the isocentre and radius_mm in [1, {MAX_POSITION_MM:g}]")
        out.append(dict(id=str(r.get("id", f"roi{i}"))[:32], y_mm=float(vals[0]), x_mm=float(vals[1]), radius_mm=float(vals[2])))
    return request.get("protocol"), out


def _r9(x):
    """Nine significant digits: more than the float32 images carry."""
    return float(f"{float(x):.9g}")


def pixel_positions(shape, fov_yx_mm):
    """Centres of the pixels of an image [.., ny, nx] in mm: index i of an n-point axis sits at (i - n // 2) * FOV / n."""
    ny, nx = shape[-2:]
    return (np.arange(ny) - ny // 2) * (fov_yx_mm[0] / ny), (np.arange(nx) - nx // 2) * (fov_yx_mm[1] / nx)


def fov_yx(realised):
    """Field of view along (y, x) of the images of a realised protocol."""
    ro, pe = realised["fov_mm"], realised["fov_pe_mm"]
    return (pe, ro) if realised["pe_dir"] == "y" else (ro, pe)


def roi_readouts(images, rois, fov_yx_mm):
    """Console read-out of circular regions on complex images [n_images, ny, nx].

    Per region and image: pixel count, mean and SD of the magnitude, and magnitude / phase of the complex mean (the
    complex mean has no noise floor; the magnitude mean does. The complex mean loses signal when the phase varies
    across the region, for example at long echo times in a field gradient).
    """
    ypos, xpos = pixel_positions(images.shape, fov_yx_mm)
    out = []
    for r in rois:
        mask = (ypos[:, None] - r["y_mm"]) ** 2 + (xpos[None, :] - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2
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


def fingerprint(magnitude):
    """Mean magnitude of FINGERPRINT_BLOCKS x FINGERPRINT_BLOCKS blocks of every image, for tolerance replay."""
    out = []
    for img in magnitude:
        ny, nx = img.shape
        ey, ex = np.linspace(0, ny, FINGERPRINT_BLOCKS + 1).astype(int), np.linspace(0, nx, FINGERPRINT_BLOCKS + 1).astype(int)
        out.append([_r9(img[ey[i]:ey[i + 1], ex[j]:ex[j + 1]].mean()) for i in range(FINGERPRINT_BLOCKS)
                    for j in range(FINGERPRINT_BLOCKS)])
    return out


def fingerprints_close(a, b, rtol=REPLAY_RTOL):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return False
    return all(np.max(np.abs(np.array(x) - np.array(y))) <= rtol * max(np.max(np.abs(x)), 1e-30) for x, y in zip(a, b))


def _parameter_card(sequence, name, entry, exam):
    lo, hi, kind, d = entry
    card = {"unit": console.PARAMETER_UNITS[name]}
    if kind == "b":
        card["type"] = "boolean"
    elif kind == "c":
        card.update(type="choice", choices=list(lo))
    else:
        card.update(type="integer" if kind == "i" else "number", min=lo, max=hi)
    if d is None:
        card["required"] = True
    elif d == "min":
        card["default"] = "console minimum for the requested geometry and bandwidth"
    elif d == "seq":
        card["default"] = console.SYSTEM[f"{sequence}_phase_increment_deg"]
    elif d == "exam":
        card["default"] = exam[name]
    else:
        card["default"] = d
    return card


def scanner_card(cfg, exam):
    """Everything the agent may know about the instrument and the episode. No seed, no hidden magnitude."""
    s = console.SYSTEM
    return {
        "scanner": "MR Gym virtual scanner", "api_version": "v1", "engine": ENGINE_ID, "task_id": cfg["task_id"],
        "field_strength_t": exam["b0_t"], "slice": "single 2D slice at the isocentre",
        "geometry": "images are [n_images, ny, nx], row = y, column = x; pixel i of an n-point axis is centred at "
                    "(i - n // 2) * field of view / n; the readout runs along x when pe_dir = y (default) and along y "
                    "when pe_dir = x; the field of view is centred on the isocentre",
        "default_fov_mm": exam["fov_mm"], "min_pixel_mm": exam["min_pixel_mm"],
        "budget_s": float(cfg["budget_s"]), "max_requests": int(cfg.get("max_requests", 64)),
        "calls": {"info": "free", "plan": "free dry run: timing, scan time, limits, accept / reject; never signal",
                  "scan": "charged scan_s; refused requests cost nothing", "budget": "free"},
        "cost": {"overhead_s_per_scan": s["overhead_s"], "dummy_time_ms_steady_state_sequences": s["dummy_ms"],
                 "dummy_cycles_se_ir": s["prepared_dummy_cycles"], "dummy_shots_epi": s["epi_dummy_shots"],
                 "rule": "scan_s = overhead + dummy cycles * cycle + cycle * phase-encode lines * averages; cycle = "
                         "tr_ms (afi: tr_ms * (1 + tr_ratio)); epi: overhead + tr_ms * (averages + dummy shots); "
                         "lines = matrix * phase_fov * (1 + phase_oversampling) * partial_fourier"},
        "hardware": {"gradient_max_mt_per_m": s["g_max_mt_m"], "slew_max_t_per_m_s": s["slew_t_m_s"],
                     "excitation_pulse_ms": s["rf_ms"], "inversion_pulse_ms": s["inv_ms"], "time_raster_ms": s["raster_ms"],
                     "b1rms_limit_ut": exam["b1rms_limit_ut"], "readout_oversampling": s["ro_os"]},
        "sequences": {
            name: {"description": console.DESCRIPTIONS[name],
                   "parameters": {k: _parameter_card(name, k, e, exam) for k, e in spec.items()}}
            for name, spec in console.SEQUENCES.items()},
        "outputs": {"images": "magnitude and phase (rad), float32, one image per contrast (afi: 2; spgr: n_echoes)",
                    "roi_readouts": "per requested circle {id, y_mm, x_mm, radius_mm}: n_pixels, mean_magnitude, "
                                    "sd_magnitude, complex_mean_magnitude, complex_mean_phase_rad, one value per image",
                    "header": "the console line of the scan: sequence, field, timing, geometry, bandwidth, scan time, "
                              "RF power as percent of the limit, water-fat shift in pixels, relative SNR of the protocol",
                    "scale": "arbitrary but constant within an episode"},
        "task": cfg.get("task", {}),
    }


class ScannerSession:
    def __init__(self, cfg, log_dir=None, engine=None):
        self.cfg = dict(cfg)
        self.seed = self.cfg["seed"]
        W.seed_int(self.seed)
        self.total = float(self.cfg["budget_s"])
        self.max_requests = int(self.cfg.get("max_requests", 64))
        self.noise_draw = int(self.cfg.get("noise_draw", 0))
        self.engine = engine or kspace.Engine(W.make_world(self.seed, self.cfg.get("world_spec")))
        self.world = self.engine.world
        self.exam = self.world["console"]
        self.lock = threading.Lock()
        self.events, self.used, self.n_scans = [], 0.0, 0
        self.chain = _sha(f"mrgym-scanner-log-v{LOG_VERSION}:{self.cfg['task_id']}:{ENGINE_ID}".encode())   # no seed in here
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
        return scanner_card(self.cfg, self.exam)

    def budget(self):
        return {"budget_s": self.total, "used_s": round(self.used, 6), "remaining_s": round(self.remaining, 6),
                "scans": self.n_scans, "requests": len(self.events), "max_requests": self.max_requests}

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def _plan(self, protocol):
        return console.plan(protocol, self.exam)

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
            readouts = roi_readouts(data, rois, fov_yx(rp))
            rec = self._append(dict(type="scan", scan_index=self.n_scans, realised=rp, cost_s=cost_s,
                                    remaining_s=round(self.remaining - cost_s, 6), rois=rois,
                                    shape=list(data.shape), data_sha256=_sha(arrays["magnitude"].tobytes() + arrays["phase"].tobytes()),
                                    fingerprint=fingerprint(arrays["magnitude"]),
                                    readouts_sha256=_sha(_canon(readouts).encode())), t0)
            resp = {k: rec[k] for k in ("event", "scan_index", "realised", "cost_s", "remaining_s", "shape", "data_sha256")}
            resp.update(header=pl["header"], limits=pl["limits"], fov_yx_mm=list(fov_yx(rp)), roi_readouts=readouts)
            return resp, arrays


def read_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def replay(cfg, events, engine=None, rtol=REPLAY_RTOL):
    """Re-run a log from the episode config. ``ok`` is False on any inconsistency.

    Checks: version, numbering, hash chain (whose first link carries the engine id), and for every accepted scan that
    a fresh simulation of the recorded request reproduces the realised protocol, cost and remaining budget exactly and
    the data either bit for bit (image digest and read-out digest) or, failing that, within ``rtol`` on the logged
    fingerprint. ``mode`` reports which: "bit-exact" or "tolerance".
    """
    fresh = ScannerSession(cfg, engine=engine)
    problems, inexact = [], 0
    for i, rec in enumerate(events):
        body = {k: v for k, v in rec.items() if k not in ("chain", "wall")}
        if rec.get("v") != LOG_VERSION or rec.get("event") != i or rec.get("prev") != fresh.chain \
                or rec.get("chain") != _sha((fresh.chain + _canon(body)).encode()):
            problems.append(f"event {i}: broken numbering or hash chain" + (" (or another engine version)" if i == 0 else ""))
            break
        if rec["type"] == "scan":
            try:
                fresh.scan({"protocol": console.request_of(rec["realised"]), "rois": rec["rois"]})
            except ScannerError as e:
                problems.append(f"event {i}: replay refused the scan ({e.type})")
                break
            last = fresh.events[-1]
            diff = [k for k in ("scan_index", "realised", "cost_s", "remaining_s", "shape") if last[k] != rec[k]]
            if not diff and (last["data_sha256"] != rec["data_sha256"] or last["readouts_sha256"] != rec["readouts_sha256"]):
                if fingerprints_close(last["fingerprint"], rec.get("fingerprint", []), rtol):
                    inexact += 1
                else:
                    diff = ["data"]
            if diff:
                problems.append(f"event {i}: replay differs in {diff}")
                break
            fresh.events[-1] = rec
            fresh.chain = rec["chain"]
        else:
            fresh._apply(rec)
    return {"ok": not problems, "problems": problems, "events": len(events), "scans": fresh.n_scans,
            "plans": sum(e["type"] == "plan" for e in events), "refused": sum(e["type"] == "refused" for e in events),
            "used_s": round(fresh.used, 6), "budget_s": fresh.total, "over_budget": fresh.used > fresh.total + 1e-6,
            "mode": "tolerance" if inexact else "bit-exact", "scans_within_tolerance_only": inexact, "engine": ENGINE_ID}


class LocalScanner:
    """In-process scanner with the calls of ``client.Scanner`` (info, budget, plan, scan), plus the two free
    look-ups an environment offers on scans already acquired (``read_regions``, ``view``). It keeps the images of
    every accepted scan. Control policies and the environments both go through this class."""

    def __init__(self, session: ScannerSession):
        self.session = session
        self.scans: dict[int, dict] = {}

    def info(self) -> dict:
        return self.session.info()

    def budget(self) -> dict:
        return self.session.budget()

    def plan(self, protocol) -> dict:
        return self.session.plan({"protocol": protocol})

    def scan(self, protocol, rois=None, out=None) -> dict:
        resp, arrays = self.session.scan({"protocol": protocol, "rois": rois or []})
        self.scans[resp["scan_index"]] = dict(arrays, realised=resp["realised"], fov_yx_mm=tuple(resp["fov_yx_mm"]))
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            for k, a in arrays.items():
                np.save(f"{out}_{k}.npy", a)
        return dict(resp, arrays=arrays)

    def stored(self, scan: int) -> dict:
        if scan not in self.scans:
            raise InvalidRequest(f"no stored scan {scan}; stored scans: {sorted(self.scans)}")
        return self.scans[scan]

    def read_regions(self, scan: int, rois) -> list:
        """Read-out of other circles on a stored scan (computed from the float32 images the scan returned)."""
        s = self.stored(scan)
        _, rois = _split_request({"protocol": None, "rois": rois})
        data = (s["magnitude"] * np.exp(1j * s["phase"])).astype(np.complex64)
        return roi_readouts(data, rois, s["fov_yx_mm"])

    def view(self, scan: int, image: int = 0, kind: str = "magnitude") -> dict:
        """One stored image and what a ruler on the display gives: the extent of the bright pixels in mm."""
        from .. import render
        s = self.stored(scan)
        if not 0 <= image < s[kind].shape[0]:
            raise InvalidRequest(f"scan {scan} has {s[kind].shape[0]} image(s)")
        img = s[kind][image]
        box = render.bounding_box(s["magnitude"][image], s["fov_yx_mm"])
        if box:
            box = dict(y_mm=[round(float(v), 2) for v in box["y"]], x_mm=[round(float(v), 2) for v in box["x"]],
                       touches_edge=box["touches_edge"])
        return dict(scan=scan, image=image, kind=kind, shape=list(img.shape), fov_yx_mm=list(s["fov_yx_mm"]),
                    sequence=s["realised"]["sequence"], bounding_box=box, array=img)
