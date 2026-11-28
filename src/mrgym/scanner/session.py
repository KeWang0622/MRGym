"""Episode state of the virtual scanner: plan / scan, scan-time budget, the keyed append-only log, replay.

The whole episode is a deterministic fold of (episode config, ordered requests):
    world              = f(seed, world spec)
    response to scan k = f(world, realised protocol, seed, noise draw, k)       k counts accepted scans
    budget, counters   = f(config, requests)
The verifier replays the log from the episode config and checks every record. There is no reset and no route that
returns anything derived from the seed other than measurements.

Episode config (JSON; lives with the service and the verifier only, never in an agent-visible file):
    {"task_id": "<trainer-side label>", "seed": "<32 hex digits>", "world_spec": {...}, "budget_s": 60.0,
     "max_requests": 64, "task": {...public task card...}, "noise_draw": 0}
``noise_draw`` selects an independent noise stream for the same world; only admission and the verifier's reruns set it.

Log (version 3). A hash chain proves order, not completeness: a prefix of a chain is a chain. So the log has two parts,
held by two parties.
    record i   body_i = {v, event: i, type, ...};  line = body_i + {chain, mac, wall}
               chain_i = SHA-256(chain_{i-1} + canon(body_i))          unkeyed: the same seed and calls give the same chain
               mac_i   = HMAC(K_ep, mac_{i-1} + canon(body_i))         mac_0 input: HMAC(K_ep, "mrgym-scanner-log-v3|engine|handle")
    head       {handle, engine, events: n, used_s, scans, restarts, mac: mac_n, sealed, reason, tag = HMAC(K_ep, head)}
               the commitment to the whole log. It is rewritten after every record in ``state_dir`` (never the log
               directory) and handed to the trainer at the end of the episode (``info["log_head"]``).
    seal       the last record of an episode, {type: "seal", reason: "submit" | "close"}; nothing is accepted after it.
``K_ep`` is the episode key: derived from the secret's log sub-key and the handle, or 32 random bytes that live only in
the receipt (the head plus that key). Order per request: append the line, fsync, rewrite the head, then answer.

What this gives. A log that is shorter than its head, edited, extended by someone without the key, or that belongs to
another episode fails ``replay``. A session starts on an existing log only with the receipt of that log (``resume_from``):
a log that is shorter than the receipt or fails its MACs raises ``LogTampered`` (the service exits with status 3) and
the episode is sealed as tampered, so a restart cannot refill the budget. Records beyond the receipt that carry valid
MACs are a crash between "append" and "rewrite the head": they are kept, and the budget can only shrink. A torn last
line is moved aside. What it does not give: whoever can write the log directory and the state directory and restart
the service can roll both back to an earlier pair; only a head held elsewhere (the trainer's copy) detects that.

Replay and floating point. Every accepted scan is logged with (a) the SHA-256 of its float32 images and (b) a
fingerprint: the mean magnitude of 8 x 8 blocks of every image. A replay on the same numerical stack reproduces the
digest ("bit-exact"). On another machine, BLAS or FFT build the images can differ in the last bits; the replay then
accepts a scan whose fingerprint agrees within ``REPLAY_RTOL`` of the image's largest block ("tolerance") and says so
in its report. The hash chain itself covers only logged text, so it never depends on arithmetic.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import stat
import threading
import time

import numpy as np

from .. import keys
from . import console, kspace
from . import world as W

LOG_VERSION = 3
LOG_NAME = "scan_log.jsonl"
TORN_NAME = "scan_log.torn"
LOCK_NAME = ".lock"
HEAD_SUFFIX = ".head"
HEAD_FIELDS = ("v", "handle", "engine", "events", "used_s", "scans", "restarts", "mac", "sealed", "reason")
REQUEST_TYPES = ("plan", "scan", "refused", "design")
MAX_BODY_BYTES = 65536
MAX_ROIS = 16
MAX_POSITION_MM = 300.0
REPLAY_RTOL = 1e-4
FINGERPRINT_BLOCKS = 8


# The engine version names the numerical behaviour of the scanner: world generation, signal models, k-space formation,
# console arithmetic and noise streams. It is bumped by hand whenever the same (seed, protocol) can give other data,
# and only then; tests/test_engine_version.py pins it together with reference numbers. Recorded logs replay only on
# the engine version that wrote them.
ENGINE_VERSION = "0.3.0"
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


class EpisodeOver(ScannerError):
    """The log is sealed (the deliverable was received, or the episode was closed): nothing more is accepted."""
    type, http_status = "episode_over", 409


class LogTampered(Exception):
    """The log on disk does not match its receipt (shortened, edited, foreign), or there is a log and no receipt.
    A service exits with ``EXIT_STATUS``; the episode is sealed as tampered and scores 0."""
    EXIT_STATUS = 3


class LogSealed(LogTampered):
    """The episode on disk is already over; it is not resumed (and not marked as tampered)."""


def new_seed():
    """128 bits from the operating system."""
    return os.urandom(16).hex()


def load_config(path, environ=None):
    """Read an episode config (it holds the seed). The file must not be readable by group or others.

    ``"seed_from_secret": true`` derives the seed from the secret's world sub-key and the config's ``task_id``, so a
    published task directory holds nothing from which the world can be rebuilt. The secret comes from
    ``MRGYM_SECRET_FILE`` (or ``MRGYM_SECRET``), never from the command line: process arguments are world-readable."""
    st = os.stat(path)
    if stat.S_ISREG(st.st_mode) and st.st_mode & 0o077:
        raise SystemExit(f"mrgym.scanner: the episode config {path!r} holds the seed and is readable by group or others "
                         f"(mode {stat.S_IMODE(st.st_mode):04o}); chmod 600")
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("seed_from_secret"):
        try:
            master = keys.load_master(environ)
        except keys.Error as e:
            raise SystemExit(f"mrgym.scanner: {e}") from None
        if not master:
            raise SystemExit("mrgym.scanner: this task needs the secret (MRGYM_SECRET_FILE) on the side of the scanner and the verifier")
        cfg["seed"] = hmac.new(keys.sub(master, "world"), ("config|" + str(cfg["task_id"])).encode(), hashlib.sha256).hexdigest()[:32]
    W.seed_int(cfg.get("seed"))
    return cfg


def _mac(key: bytes, prev: bytes, body: dict) -> bytes:
    return hmac.new(key, prev + _canon(body).encode(), hashlib.sha256).digest()


def _mac0(key: bytes, handle: str) -> bytes:
    return hmac.new(key, f"mrgym-scanner-log-v{LOG_VERSION}|{ENGINE_ID}|{handle}".encode(), hashlib.sha256).digest()


def _chain0() -> str:
    return _sha(f"mrgym-scanner-log-v{LOG_VERSION}:{ENGINE_ID}".encode())          # no seed and no handle in here


def _body(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("chain", "mac", "wall")}


def head_tag(key: bytes, head: dict) -> str:
    return hmac.new(key, b"head|" + _canon({k: head.get(k) for k in HEAD_FIELDS}).encode(), hashlib.sha256).hexdigest()


def head_key(head: dict, cfg: dict, log_key: bytes | None = None) -> bytes | None:
    """The episode key that belongs to a head: the one in the receipt, or the one derived from the log sub-key."""
    try:
        if head.get("key"):
            return bytes.fromhex(head["key"])
        if log_key is not None:
            return keys.episode_key(log_key, str(head["handle"]), cfg["seed"])
    except (KeyError, TypeError, ValueError):
        pass
    return None


def read_head(path) -> dict:
    """A receipt file. Like the config it must not be readable by group or others (it may hold the episode key)."""
    st = os.stat(path)
    if stat.S_ISREG(st.st_mode) and st.st_mode & 0o077:
        raise LogTampered(f"the receipt {path!r} is readable by group or others (mode {stat.S_IMODE(st.st_mode):04o}); chmod 600")
    with open(path, encoding="utf-8") as f:
        head = json.load(f)
    if not isinstance(head, dict):
        raise LogTampered(f"the receipt {path!r} is not a JSON object")
    return head


def _write_private(path, text):
    tmp = f"{path}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _split_request(request):
    """(protocol, circles) of a scan request. Every circle is checked here, so nothing downstream meets a value that is
    not a finite number in range; a refusal names the circle and the field (``parameter``) and what would be valid."""
    if not isinstance(request, dict):
        raise InvalidRequest("request must be a JSON object {\"protocol\": {...}, \"rois\": [...]}")
    unknown = sorted(str(k)[:40] for k in set(request) - {"protocol", "rois"})
    if unknown:
        raise InvalidRequest(f"unknown request field(s) {unknown}; allowed: protocol, rois")
    rois = request.get("rois") or []
    if not isinstance(rois, list) or len(rois) > MAX_ROIS:
        raise InvalidRequest(f"rois must be a list of at most {MAX_ROIS} circles", parameter="circles")
    out = []
    for i, r in enumerate(rois):
        if not isinstance(r, dict) or set(r) - {"id", "y_mm", "x_mm", "radius_mm"}:
            raise InvalidRequest("each circle is {\"id\": optional, \"y_mm\": .., \"x_mm\": .., \"radius_mm\": ..}",
                                 parameter=f"circles[{i}]")
        vals = [r.get(k) for k in ("y_mm", "x_mm", "radius_mm")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
            raise InvalidRequest("circle y_mm, x_mm, radius_mm must be finite numbers", parameter=f"circles[{i}]")
        for k, v, lo in (("y_mm", vals[0], -MAX_POSITION_MM), ("x_mm", vals[1], -MAX_POSITION_MM), ("radius_mm", vals[2], 1.0)):
            if not lo <= v <= MAX_POSITION_MM:
                raise InvalidRequest(f"circle {k} = {v:g} is outside [{lo:g}, {MAX_POSITION_MM:g}] mm", parameter=f"circles[{i}].{k}",
                                     requested=v, low=lo, high=MAX_POSITION_MM, nearest_valid=min(max(v, lo), MAX_POSITION_MM))
        out.append(dict(id=str(r.get("id", f"roi{i}"))[:32], y_mm=float(vals[0]), x_mm=float(vals[1]), radius_mm=float(vals[2])))
    return request.get("protocol"), out


def _r9(x):
    """Nine significant digits: more than the float32 images carry."""
    return float(f"{float(x):.9g}")


def pixel_positions(shape, fov_yx_mm, centre_yx_mm=(0.0, 0.0)):
    """Centres of the pixels of an image [.., ny, nx] in mm from the isocentre: index i of an n-point axis sits at
    centre + (i - n // 2) * FOV / n."""
    ny, nx = shape[-2:]
    return (centre_yx_mm[0] + (np.arange(ny) - ny // 2) * (fov_yx_mm[0] / ny),
            centre_yx_mm[1] + (np.arange(nx) - nx // 2) * (fov_yx_mm[1] / nx))


def fov_yx(realised):
    """Field of view along (y, x) of the images of a realised protocol."""
    ro, pe = realised["fov_mm"], realised["fov_pe_mm"]
    return (pe, ro) if realised["pe_dir"] == "y" else (ro, pe)


def centre_yx(realised):
    """Centre of the field of view along (y, x), mm from the isocentre."""
    return (float(realised.get("centre_y_mm", 0.0)), float(realised.get("centre_x_mm", 0.0)))


def rotation_of(realised):
    """In-plane rotation of the field of view of a realised protocol, degrees (image rows turned from +y towards +x)."""
    return float(realised.get("rotation_deg", 0.0) or 0.0)


def pixel_grid(shape, fov_yx_mm, centre_yx_mm=(0.0, 0.0), rotation_deg=0.0):
    """Patient coordinates (y, x) in mm of every pixel of an image [.., ny, nx], as two [ny, nx] arrays: the pixel at
    (v, u) mm from the centre of the field of view along the image rows and columns lies at
    centre + (v cos r - u sin r, v sin r + u cos r)."""
    v, u = pixel_positions(shape, fov_yx_mm)
    c, s = np.cos(np.deg2rad(rotation_deg)), np.sin(np.deg2rad(rotation_deg))
    return (centre_yx_mm[0] + v[:, None] * c - u[None, :] * s, centre_yx_mm[1] + v[:, None] * s + u[None, :] * c)


def roi_readouts(images, rois, fov_yx_mm, centre_yx_mm=(0.0, 0.0), rotation_deg=0.0):
    """Console read-out of circular regions on complex images [n_images, ny, nx]. Regions are circles in patient
    coordinates (mm from the isocentre), whatever the centre and the rotation of the field of view.

    Per region and image: pixel count, mean and SD of the magnitude, and magnitude / phase of the complex mean (the
    complex mean has no noise floor; the magnitude mean does. The complex mean loses signal when the phase varies
    across the region, for example at long echo times in a field gradient).
    """
    if rotation_deg:
        ygrid, xgrid = pixel_grid(images.shape, fov_yx_mm, centre_yx_mm, rotation_deg)
    else:
        ypos, xpos = pixel_positions(images.shape, fov_yx_mm, centre_yx_mm)
        ygrid, xgrid = ypos[:, None], xpos[None, :]
    out = []
    for r in rois:
        mask = (ygrid - r["y_mm"]) ** 2 + (xgrid - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2
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
    """Everything the agent may know about the instrument and the episode. No seed, no hidden magnitude, and no
    identifier of the instance: the task id is a function of the seed on a public split and stays with the trainer."""
    s = dict(console.SYSTEM, **(exam.get("system") or {}))
    return {
        "scanner": "MR Gym virtual scanner", "api_version": "v1", "engine": ENGINE_ID,
        "field_strength_t": exam["b0_t"], "slice": "single 2D slice at the isocentre",
        "geometry": "images are [n_images, ny, nx], row = y, column = x; pixel i of an n-point axis is centred at "
                    "centre + (i - n // 2) * field of view / n; the readout runs along x when pe_dir = y (default) and along y "
                    "when pe_dir = x; the field of view is centred on (centre_y_mm, centre_x_mm), by default the isocentre; "
                    "rotation_deg turns it about that centre (image rows from +y towards +x); regions are always circles "
                    "in patient coordinates",
        "default_fov_mm": exam["fov_mm"], "min_pixel_mm": exam["min_pixel_mm"],
        "budget_s": float(cfg["budget_s"]), "max_requests": int(cfg.get("max_requests", 64)),
        "calls": {"info": "free", "plan": "free dry run: timing, scan time, limits, accept / reject; never signal",
                  "scan": "charged scan_s; refused requests cost nothing", "budget": "free"},
        "cost": {**({"overhead_s_repeat_scan": s["repeat_overhead_s"], "overhead_rule": "overhead_s_per_scan is the time of the adjustments "
                     "(centre frequency, transmit gain, shim); a scan with the same " + ", ".join(console.ADJUSTMENT_FIELDS)
                     + " as the previous scan reuses them and is charged overhead_s_repeat_scan instead"} if "repeat_overhead_s" in s else {}),
                 "overhead_s_per_scan": s["overhead_s"], "dummy_time_ms_steady_state_sequences": s["dummy_ms"],
                 "dummy_cycles_se_ir": s["prepared_dummy_cycles"], "dummy_shots_epi": s["epi_dummy_shots"],
                 "rule": "scan_s = overhead + dummy cycles * cycle + cycle * phase-encode lines * averages; cycle = "
                         "tr_ms (afi: tr_ms * (1 + tr_ratio)); epi: overhead + tr_ms * (averages + dummy shots); "
                         "lines = matrix * phase_fov * (1 + phase_oversampling) * partial_fourier"},
        "hardware": {"gradient_max_mt_per_m": s["g_max_mt_m"], "slew_max_t_per_m_s": s["slew_t_m_s"],
                     "excitation_pulses_ms_tbw": {k: list(v) for k, v in console.RF.PULSES.items()},
                     "nonselective_pulse_ms": s["rf_ms"], "slab_mm_nonselective": s["slab_mm"], "b1_peak_limit_ut": s["b1_peak_ut"],
                     "inversion_pulse_ms": s["inv_ms"], "time_raster_ms": s["raster_ms"],
                     "b1rms_limit_ut": exam["b1rms_limit_ut"], "readout_oversampling": s["ro_os"]},
        "sequences": {
            name: {"description": console.REGISTRY[name].description,
                   "parameters": {k: _parameter_card(name, k, e, exam) for k, e in spec.items()}}
            for name, spec in console.offered(exam).items()},                # the six, and what else the card of this console lists
        "outputs": {"images": "magnitude and phase (rad), float32, one image per contrast (afi: 2; spgr: n_echoes"
                              + ("; mese, megre: n_echoes" if exam.get("sequences") else "") + ")",
                    "roi_readouts": "per requested circle {id, y_mm, x_mm, radius_mm}: n_pixels, mean_magnitude, "
                                    "sd_magnitude, complex_mean_magnitude, complex_mean_phase_rad, one value per image",
                    "header": "the console line of the scan: sequence, field, timing, geometry, bandwidth, scan time, "
                              "RF power as percent of the limit, water-fat shift in pixels, relative SNR of the protocol",
                    "scale": "arbitrary but constant within an episode"},
        "task": cfg.get("task", {}),
    }


class ScannerSession:
    """One episode. ``log_dir``: where the log is written (None: memory only). ``state_dir``: where the head is rewritten
    after every record; it must not be the log directory. ``resume_from``: the receipt (a path or a dict) of an
    existing log; without it an existing log is refused. ``log_key``: the secret's log sub-key (``keys.sub(K, "log")``);
    without it the episode key is random and travels in the receipt. ``handle``: the episode's random id."""

    def __init__(self, cfg, log_dir=None, engine=None, *, state_dir=None, resume_from=None, handle=None, log_key=None):
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
        self.events, self.used, self.n_scans, self.n_requests, self.restarts, self.sealed = [], 0.0, 0, 0, 0, None
        self.n_designs = 0
        receipt = read_head(resume_from) if isinstance(resume_from, (str, os.PathLike)) else resume_from
        self.handle = str(receipt["handle"]) if receipt else (handle or keys.episode_handle())
        if receipt:
            self._key = head_key(receipt, self.cfg, log_key)
            if self._key is None:
                raise LogTampered("the receipt carries no episode key and no log key was given")
            self._key_in_receipt = bool(receipt.get("key"))
        else:
            self._key, self._key_in_receipt = keys.episode_key(log_key, self.handle, self.seed), log_key is None
        self.chain = _chain0()                                           # unkeyed content chain (deterministic)
        self._last_mac = _mac0(self._key, self.handle)                   # keyed chain (per episode)
        self.log_path = os.path.join(log_dir, LOG_NAME) if log_dir else None
        self.head_path = None
        self._fh = self._lock_fh = None
        if state_dir:
            if not log_dir:
                raise ValueError("state_dir without log_dir: there is no log on disk to commit to")
            if os.path.realpath(state_dir) == os.path.realpath(log_dir) or \
                    os.path.realpath(state_dir).startswith(os.path.realpath(log_dir) + os.sep):
                raise ValueError("state_dir must not be the log directory or lie inside it: the head is the commitment "
                                 "to the log and must be out of reach of whoever can write the log")
            os.makedirs(state_dir, mode=0o700, exist_ok=True)
            self.head_path = os.path.join(state_dir, self.handle + HEAD_SUFFIX)
        if self.log_path:
            os.makedirs(log_dir, mode=0o700, exist_ok=True)
            self._lock_directory(log_dir)
            try:
                self._open_log(receipt)
            except BaseException:
                self.close()
                raise

    # ------------------------------------------------------------------ log on disk
    def _lock_directory(self, log_dir):
        """One writer per log directory: a second session on it is refused."""
        try:
            import fcntl
        except ImportError:                                              # no advisory locks on this platform
            return
        self._lock_fh = open(os.path.join(log_dir, LOCK_NAME), "a")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_fh.close()
            self._lock_fh = None
            raise LogTampered(f"another scanner session holds the log directory {log_dir!r}") from None

    def _open_log(self, receipt):
        present = os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0
        if present and receipt is None:
            raise LogTampered(f"{self.log_path} already holds a log. A new episode needs a new directory; a restart of "
                              "this episode needs its receipt (resume_from). Nothing was served.")
        if receipt is not None:
            try:
                self._resume(receipt)
            except LogTampered as e:
                if not isinstance(e, LogSealed):
                    self._seal_tampered(receipt, str(e))
                raise
        fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._fh = os.fdopen(fd, "a", buffering=1)
        if receipt is not None:
            self._append(dict(type="restart", restarts=self.restarts + 1), time.time())
        else:
            self._write_head()

    def _resume(self, receipt):
        """Rebuild the state from the log, but only a log that the receipt vouches for."""
        if receipt.get("v") != LOG_VERSION or receipt.get("engine") != ENGINE_ID:
            raise LogTampered("the receipt was written by another log or engine version")
        if not hmac.compare_digest(str(receipt.get("tag", "")), head_tag(self._key, receipt)):
            raise LogTampered("the receipt's tag does not verify: it was edited, or it belongs to another episode or key")
        if receipt.get("sealed"):
            raise LogSealed(f"the episode is sealed ({receipt.get('reason')}); a sealed episode is not resumed")
        raw = b""
        if os.path.exists(self.log_path):
            with open(self.log_path, "rb") as f:
                raw = f.read()
        complete, _, torn = raw.rpartition(b"\n")
        lines = [ln for ln in complete.split(b"\n") if ln.strip()] if complete else []
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
                body = _body(rec)
                mac = _mac(self._key, self._last_mac, body)
                ok = (rec.get("v") == LOG_VERSION and rec.get("event") == i
                      and rec.get("chain") == _sha((self.chain + _canon(body)).encode())
                      and hmac.compare_digest(str(rec.get("mac", "")), mac.hex()))
            except (ValueError, TypeError, AttributeError):
                ok = False
            if not ok:
                raise LogTampered(f"log does not match receipt: record {i} fails its chain or MAC")
            self._apply(rec)
            if i + 1 == receipt["events"] and not hmac.compare_digest(rec["mac"], str(receipt.get("mac"))):
                raise LogTampered(f"log does not match receipt: record {i} is not the receipt's head")
        if len(lines) < int(receipt["events"]):
            raise LogTampered(f"log does not match receipt: the log holds {len(lines)} records, the receipt {receipt['events']} "
                              "(the log was shortened)")
        if int(receipt["events"]) == 0 and receipt.get("mac") != _mac0(self._key, self.handle).hex():
            raise LogTampered("log does not match receipt: the receipt's empty head is not this episode's")
        if self.sealed is not None:                                      # a stop between the seal record and its head
            self._write_head()
            raise LogSealed(f"the log ends in a seal ({self.sealed}); a sealed episode is not resumed")
        if torn.strip():                                                 # a crash in the middle of a write: never acknowledged
            with open(os.path.join(os.path.dirname(self.log_path), TORN_NAME), "ab") as f:
                f.write(torn + b"\n")
        if torn:
            with open(self.log_path, "r+b") as f:
                f.truncate(len(complete) + 1 if complete else 0)

    def _seal_tampered(self, receipt, why):
        """Mark the episode as tampered where the trusted side keeps its state. The counters stay the receipt's."""
        if not self.head_path or receipt.get("sealed"):
            return
        head = {k: receipt.get(k) for k in HEAD_FIELDS}
        head.update(sealed=True, reason="tampered")
        head["tag"] = head_tag(self._key, head)
        if receipt.get("key"):
            head["key"] = receipt["key"]
        head.update(canary=keys.CANARY, problem=why[:300])
        _write_private(self.head_path, json.dumps(head, indent=1) + "\n")

    def head(self) -> dict:
        """The commitment to the log so far. It holds no key and nothing derived from the seed."""
        h = dict(v=LOG_VERSION, handle=self.handle, engine=ENGINE_ID, events=len(self.events), used_s=round(self.used, 6),
                 scans=self.n_scans, restarts=self.restarts, mac=self._last_mac.hex(), sealed=self.sealed is not None,
                 reason=self.sealed)
        h["tag"] = head_tag(self._key, h)
        return h

    def receipt(self, with_key=None) -> dict:
        """The head, the canary and, when the episode key is not derivable from the secret (or ``with_key``), that key.
        Trainer side only: whoever holds the key before the seal can forge the log."""
        r = self.head()
        if self._key_in_receipt if with_key is None else with_key:
            r["key"] = self._key.hex()
        r["canary"] = keys.CANARY
        return r

    def _write_head(self):
        if self.head_path:
            _write_private(self.head_path, json.dumps(self.receipt(), indent=1) + "\n")

    def seal(self, reason="submit") -> dict:
        """End the episode: append the seal record; plans and scans are refused afterwards. Returns the receipt."""
        with self.lock:
            if self.sealed is None:
                self._append(dict(type="seal", events=len(self.events), used_s=round(self.used, 6), scans=self.n_scans,
                                  reason=str(reason)), time.time())
            return self.receipt()

    # ------------------------------------------------------------------ state
    def _apply(self, rec):
        self.events.append(rec)
        self.chain = rec["chain"]
        self._last_mac = bytes.fromhex(rec["mac"])
        if rec["type"] == "scan":
            self.used += rec["cost_s"]
            self.n_scans += 1
        if rec["type"] == "design":                                      # a stored design: rebuilt from the logged call
            self._store_design(rec)
        if rec["type"] in REQUEST_TYPES:
            self.n_requests += 1
        elif rec["type"] == "restart":
            self.restarts = int(rec["restarts"])
        elif rec["type"] == "seal":
            self.sealed = rec["reason"]

    def _append(self, body, t0):
        body = dict(v=LOG_VERSION, event=len(self.events), **body)
        rec = dict(body, chain=_sha((self.chain + _canon(body)).encode()), mac=_mac(self._key, self._last_mac, body).hex(),
                   wall={"t_utc": round(t0, 3), "ms": round((time.time() - t0) * 1e3, 2)})
        if self._fh:
            self._fh.write(_canon(rec) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
        self._apply(rec)
        self._write_head()                                               # after the line is durable, before the answer
        return rec

    def _open_for_request(self):
        if self.sealed is not None:
            raise EpisodeOver(f"the episode is over ({self.sealed}); nothing more is accepted")
        if self.n_requests >= self.max_requests:
            raise RequestLimit(f"request limit of {self.max_requests} reached", max_requests=self.max_requests)

    @property
    def remaining(self):
        return self.total - self.used

    def info(self):
        return scanner_card(self.cfg, self.exam)

    def budget(self):
        return {"budget_s": self.total, "used_s": round(self.used, 6), "remaining_s": round(self.remaining, 6),
                "scans": self.n_scans, "requests": self.n_requests, "max_requests": self.max_requests}

    def close(self):
        """Release the files. It does not seal: a stopped service may be resumed with its receipt."""
        for name in ("_fh", "_lock_fh"):
            fh = getattr(self, name, None)
            if fh:
                fh.close()
                setattr(self, name, None)

    def _plan(self, protocol):
        """The console's plan for the next scan. On a console that states ``repeat_overhead_s`` a scan that needs the
        same adjustments as the previous one (``console.adjustment_key``) is charged that instead of ``overhead_s``.
        A function of the log so far, so replay and resume reproduce it."""
        pl = console.plan(protocol, self.exam)
        system = self.exam.get("system") or {}
        if pl["accept"] and "repeat_overhead_s" in system:
            last = next((e["realised"] for e in reversed(self.events) if e.get("type") == "scan"), None)
            if last is not None and console.adjustment_key(last) == console.adjustment_key(pl["realised"]):
                pl = console.plan(protocol, dict(self.exam, system=dict(system, overhead_s=system["repeat_overhead_s"])))
        return pl

    def _noise_rng(self, realised, scan_index):
        h = int(_sha(_canon(realised).encode())[:32], 16)
        key = (W.STREAMS["noise"], self.noise_draw, scan_index) + tuple((h >> (32 * i)) & 0xFFFFFFFF for i in range(4))
        return np.random.default_rng(np.random.SeedSequence(entropy=W.seed_int(self.seed), spawn_key=key))

    # ------------------------------------------------------------------ designs (a console whose card holds a design table)
    def design_limits(self):
        """What a design is checked against on this console: the derated gradient limits, the peak B1, and the RF
        power limit of the task when it states one (``cfg["task"]["design"]["limits"]``: ``b1_peak_ut``, ``b1rms_ut``
        over ``tr_ms``)."""
        from ..design import tool
        s = dict(console.SYSTEM, **(self.exam.get("system") or {}))
        stated = ((self.cfg.get("task") or {}).get("design") or {}).get("limits") or {}
        return tool.Limits(b1_peak_ut=float(stated.get("b1_peak_ut", s["b1_peak_ut"])), g_max_mt_m=s["g_max_mt_m"] * s["derate"],
                           slew_t_m_s=s["slew_t_m_s"] * s["derate"], b1rms_ut=stated.get("b1rms_ut"), tr_ms=stated.get("tr_ms"))

    def _store_design(self, rec):
        """Put the design of a log record into the console's table. The pulse is rebuilt from the logged canonical call,
        so the table of a resumed, replayed or recorded session is the table of the live one."""
        from ..design import tool
        table = self.exam.get("designs")
        if table is None:
            raise ValueError("this console stores no designs")
        pulse, call = tool.build(rec["call"], self.design_limits())
        table[int(rec["index"])] = dict(index=int(rec["index"]), call=call, call_json=_canon(call), facts=tool.facts(pulse, self.design_limits()))
        self.n_designs = max(self.n_designs, int(rec["index"]))

    def design(self, call):
        """Store a design made from a generator call (``mrgym.design.tool``). Free, logged, counted as a request.
        Returns the table entry ``dict(index, call, call_json, facts)``. The same call again returns the entry it
        made the first time. Raises ``InvalidRequest`` for a call that cannot be built (nothing is logged for it)."""
        from ..design import tool
        with self.lock:
            t0 = time.time()
            self._open_for_request()
            table = self.exam.get("designs")
            if table is None:
                raise InvalidRequest("this console stores no designs")
            try:
                _, canonical = tool.build(call, self.design_limits(), lambda k: (table.get(k) or {}).get("call"))
            except tool.Refusal as e:
                raise InvalidRequest(e.message, parameter=e.parameter, **e.extra) from None
            text = _canon(canonical)
            same = next((d for d in table.values() if d["call_json"] == text), None)
            if same is not None:
                return copy.deepcopy(same)
            if self.n_designs >= tool.MAX_DESIGNS:
                raise InvalidRequest(f"{tool.MAX_DESIGNS} designs are stored, the most a session holds", stored=self.n_designs)
            self._append(dict(type="design", index=self.n_designs + 1, call=canonical), t0)
            return copy.deepcopy(table[self.n_designs])

    # ------------------------------------------------------------------ the two request types
    def plan(self, request):
        """Free dry run; logged and counted against the request limit. Never touches the world's signal."""
        with self.lock:
            t0 = time.time()
            self._open_for_request()
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
            self._open_for_request()
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
            readouts = roi_readouts(data, rois, fov_yx(rp), centre_yx(rp), rotation_of(rp))
            rec = self._append(dict(type="scan", scan_index=self.n_scans, realised=rp, cost_s=cost_s,
                                    remaining_s=round(self.remaining - cost_s, 6), rois=rois,
                                    shape=list(data.shape), data_sha256=_sha(arrays["magnitude"].tobytes() + arrays["phase"].tobytes()),
                                    fingerprint=fingerprint(arrays["magnitude"]),
                                    readouts_sha256=_sha(_canon(readouts).encode())), t0)
            resp = {k: rec[k] for k in ("event", "scan_index", "realised", "cost_s", "remaining_s", "shape", "data_sha256")}
            resp.update(header=pl["header"], limits=pl["limits"], fov_yx_mm=list(fov_yx(rp)),
                        centre_yx_mm=list(centre_yx(rp)), rotation_deg=rotation_of(rp), roi_readouts=readouts)
            return resp, arrays


def read_log(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def replay(cfg, events, engine=None, rtol=REPLAY_RTOL, head=None, log_key=None):
    """Re-run a log from the episode config and check it against its head. ``ok`` is False on any inconsistency.

    ``head``: the receipt of the episode, from the trusted side (the session object in process, the service's state
    directory or the trainer's copy otherwise). Without it a shortened log cannot be told from a whole one, so the
    result is ``ok = False`` with ``integrity = "unanchored"``. ``log_key``: the secret's log sub-key, when the receipt
    carries no episode key.

    Checks: version, numbering, the content chain, the MAC of every record under the episode key, that the log has
    exactly the head's length, last MAC and scan seconds, the head's own tag, and for every accepted scan that a fresh
    simulation of the recorded request reproduces the realised protocol, cost and remaining budget exactly and the
    data either bit for bit (image digest and read-out digest) or, failing that, within ``rtol`` on the logged
    fingerprint. ``mode`` reports which: "bit-exact" or "tolerance". ``sealed``: the log ends in a seal that the head
    confirms; a scored episode must be sealed.
    """
    fresh = ScannerSession(cfg, engine=engine)
    problems, inexact, key, anchored = [], 0, None, False
    if not isinstance(head, dict):
        problems.append("no head: truncation cannot be ruled out")
    else:
        key = head_key(head, cfg, log_key)
        if key is None:
            problems.append("head: no episode key (the receipt carries none and no log key was given)")
        elif head.get("v") != LOG_VERSION or head.get("engine") != ENGINE_ID:
            problems.append("head: written by another log or engine version")
        elif not hmac.compare_digest(str(head.get("tag", "")), head_tag(key, head)):
            problems.append("head: the tag does not verify (edited, or of another episode or key)")
        else:
            anchored = True
    chain, mac = _chain0(), _mac0(key, str(head.get("handle"))) if key else b""
    for i, rec in enumerate(events):
        if not isinstance(rec, dict) or not isinstance(rec.get("type"), str):
            problems.append(f"event {i}: not a log record")
            break
        body = _body(rec)
        chain = _sha((chain + _canon(body)).encode())
        if rec.get("v") != LOG_VERSION or rec.get("event") != i or rec.get("chain") != chain:
            problems.append(f"event {i}: broken numbering or hash chain" + (" (or another engine version)" if i == 0 else ""))
            break
        if key:
            mac = _mac(key, mac, body)
            if not hmac.compare_digest(str(rec.get("mac", "")), mac.hex()):
                problems.append(f"event {i}: the MAC does not verify (an edited record, the log of another episode, or another key)")
                break
        if fresh.sealed is not None:
            problems.append(f"event {i}: a record after the seal")
            break
        if rec["type"] == "scan":
            try:
                fresh.scan({"protocol": console.request_of(rec["realised"]), "rois": rec["rois"]})
            except (ScannerError, KeyError, TypeError, ValueError) as e:
                problems.append(f"event {i}: replay refused the scan ({getattr(e, 'type', type(e).__name__)})")
                break
            last = fresh.events[-1]
            diff = [k for k in ("scan_index", "realised", "cost_s", "remaining_s", "shape") if last[k] != rec.get(k)]
            if not diff and (last["data_sha256"] != rec.get("data_sha256") or last["readouts_sha256"] != rec.get("readouts_sha256")):
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
            try:
                fresh._apply(rec)
            except (KeyError, TypeError, ValueError):
                problems.append(f"event {i}: malformed {rec['type']} record")
                break
    if anchored and not problems:
        n = len(events)
        if n != head.get("events"):
            problems.append(f"the log holds {n} records, its head {head.get('events')}: the log was "
                            + ("shortened" if n < int(head.get("events") or 0) else "extended"))
        elif not hmac.compare_digest(mac.hex(), str(head.get("mac"))):
            problems.append("the last record is not the one the head commits to")
        elif abs(fresh.used - float(head.get("used_s") or 0.0)) > 1e-6 or fresh.n_scans != head.get("scans"):
            problems.append("scan seconds or scan count differ from the head")
    sealed = bool(anchored and not problems and head.get("sealed") and events and events[-1].get("type") == "seal"
                  and head.get("reason") == events[-1].get("reason") != "tampered")
    return {"ok": not problems, "problems": problems, "events": len(events), "scans": fresh.n_scans,
            "plans": sum(isinstance(e, dict) and e.get("type") == "plan" for e in events),
            "refused": sum(isinstance(e, dict) and e.get("type") == "refused" for e in events),
            "used_s": round(fresh.used, 6), "budget_s": fresh.total, "over_budget": fresh.used > fresh.total + 1e-6,
            "mode": "tolerance" if inexact else "bit-exact", "scans_within_tolerance_only": inexact, "engine": ENGINE_ID,
            "integrity": "anchored" if anchored and not problems else "unanchored", "sealed": sealed,
            "restarts": fresh.restarts, "handle": head.get("handle") if isinstance(head, dict) else None}


def verify_log(cfg, log_dir, head, engine=None, log_key=None):
    """``replay`` of the log in ``log_dir`` against ``head`` (a receipt dict or the path of one)."""
    if isinstance(head, (str, os.PathLike)):
        head = read_head(head)
    try:
        events = read_log(os.path.join(log_dir, LOG_NAME))
    except FileNotFoundError:
        events = []
    except ValueError:
        rep = replay(cfg, [], engine=engine, head=None)
        rep["problems"] = ["the log is not valid JSON lines"]
        return rep
    return replay(cfg, events, engine=engine, head=head, log_key=log_key)


class RecordedSession(ScannerSession):
    """A finished episode as the verifier sees it: the recorded log and its head, read-only.

    An environment bound to it (``env.use_session``) scores a deliverable through the same code as in process: the
    environment's ``submit`` asks for the seal and the receipt, which are the recorded ones here, and replays the
    recorded events against them. Nothing can be planned or scanned."""

    def __init__(self, cfg, events, head, engine=None, log_key=None):
        super().__init__(cfg, engine=engine)
        self.events = list(events)
        self._head = dict(head) if isinstance(head, dict) else None
        key = head_key(self._head, self.cfg, log_key) if self._head else None
        if key is not None:
            self._head["key"] = key.hex()
        for rec in self.events:                                          # counters for the status line; replay judges
            if isinstance(rec, dict) and rec.get("type") == "scan" and isinstance(rec.get("cost_s"), (int, float)):
                self.used += float(rec["cost_s"])
                self.n_scans += 1
            if isinstance(rec, dict) and rec.get("type") in REQUEST_TYPES:
                self.n_requests += 1
            if isinstance(rec, dict) and rec.get("type") == "design":       # the designs the scans and the deliverable refer to
                try:
                    self._store_design(rec)
                except (KeyError, TypeError, ValueError):
                    pass                                                 # replay judges a malformed record
        self.sealed = "recorded"

    def head(self):
        return {k: v for k, v in (self._head or {}).items() if k != "key"} or None

    def receipt(self, with_key=None):
        if self._head is None:
            return None
        return dict(self._head) if with_key or with_key is None else self.head()

    def seal(self, reason="submit"):
        return self.head()


MAX_PROFILE_SAMPLES = 128


class LocalScanner:
    """In-process scanner with the calls of ``client.Scanner`` (info, budget, plan, scan), plus the free look-ups an
    environment offers on scans already acquired (``read_regions``, ``image``, ``profile``). It keeps the images of
    every accepted scan. Control policies and the environments both go through this class.

    ``view`` is the call scripted policies use: it hands them the image array in process and, with it, the extent of
    the bright pixels (``bounding_box``). That measurement is not part of any agent-facing tool: the environment's
    ``view`` tool is built on ``image`` and shows the picture and its geometry only, and an agent that wants an edge
    measures it with ``profile``.
    """

    def __init__(self, session: ScannerSession):
        self.session = session
        self.scans: dict[int, dict] = {}

    def info(self) -> dict:
        return self.session.info()

    def budget(self) -> dict:
        return self.session.budget()

    def plan(self, protocol) -> dict:
        return self.session.plan({"protocol": protocol})

    def design(self, call) -> dict:
        """Store a design: ``dict(design=<number>, call, facts)``, what the ``design`` tool answers."""
        d = self.session.design(call)
        return dict(design=d["index"], call=d["call"], facts=d["facts"])

    def scan(self, protocol, rois=None, out=None) -> dict:
        resp, arrays = self.session.scan({"protocol": protocol, "rois": rois or []})
        self.scans[resp["scan_index"]] = dict(arrays, realised=resp["realised"], fov_yx_mm=tuple(resp["fov_yx_mm"]),
                                              centre_yx_mm=tuple(resp["centre_yx_mm"]), rotation_deg=resp["rotation_deg"],
                                              cost_s=resp["cost_s"], header=resp["header"])
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            for k, a in arrays.items():
                np.save(f"{out}_{k}.npy", a)
        return dict(resp, arrays=arrays)

    def stored(self, scan: int) -> dict:
        if isinstance(scan, bool) or not isinstance(scan, int) or scan not in self.scans:
            raise InvalidRequest(f"no stored scan {scan!r:.20}; stored scans: {sorted(self.scans)}", parameter="scan_index",
                                 not_found=True, allowed=sorted(self.scans))
        return self.scans[scan]

    def read_regions(self, scan: int, rois) -> list:
        """Read-out of other circles on a stored scan (computed from the float32 images the scan returned)."""
        s = self.stored(scan)
        _, rois = _split_request({"protocol": None, "rois": rois})
        data = (s["magnitude"] * np.exp(1j * s["phase"])).astype(np.complex64)
        return roi_readouts(data, rois, s["fov_yx_mm"], s["centre_yx_mm"], s["rotation_deg"])

    def image(self, scan: int, image: int = 0, kind: str = "magnitude") -> dict:
        """One stored image with its geometry: what a display shows. No measurement of what is in it."""
        s = self.stored(scan)
        if kind not in ("magnitude", "phase"):
            raise InvalidRequest("kind must be magnitude or phase", parameter="kind", allowed=["magnitude", "phase"])
        n = s[kind].shape[0]
        if isinstance(image, bool) or not isinstance(image, int) or not 0 <= image < n:
            raise InvalidRequest(f"scan {scan} has {n} image(s): image index 0{'' if n == 1 else f' to {n - 1}'}",
                                 parameter="image_index", not_found=True, allowed=list(range(n)))
        img = s[kind][image]
        return dict(scan=scan, image=image, kind=kind, shape=list(img.shape), fov_yx_mm=list(s["fov_yx_mm"]),
                    centre_yx_mm=list(s["centre_yx_mm"]), rotation_deg=s.get("rotation_deg", 0.0),
                    sequence=s["realised"]["sequence"], array=img)

    def view(self, scan: int, image: int = 0, kind: str = "magnitude") -> dict:
        """For scripted policies, in process: the image array and what a ruler on it gives, the extent of the bright
        pixels in mm (``bounding_box``). Never served by a tool."""
        v = self.image(scan, image, kind)
        v["bounding_box"] = image_extent(self.stored(scan), image)
        return v

    def profile(self, scan: int, axis: str, image: int = 0, kind: str = "magnitude", position_mm=None, start_mm=None,
                stop_mm=None, width_mm=None, samples=None, reduce: str = "mean") -> dict:
        """A line profile through a stored image: the values along ``axis`` (y: down the rows, x: along the columns)
        on a line at ``position_mm`` on the other axis, ``width_mm`` wide, between ``start_mm`` and ``stop_mm``.

        Pixels across the width and, when the range holds more pixels than ``samples``, neighbouring pixels along
        the line are combined by ``reduce`` (mean, or max: a maximum-intensity projection when the width is the whole
        field of view). At most ``MAX_PROFILE_SAMPLES`` samples. Returns positions (mm, centre of each sample) and
        values in the image's own units (phase: radians; its mean is the angle of the mean unit vector).

        A turned field of view (``rotation_deg`` = r). The line runs along the image's own rows or columns, and every
        position (``position_mm``, ``start_mm``, ``stop_mm``, the returned positions) is in the image's frame: patient
        coordinates turned back by r about the isocentre (``frame_centre``), so that r = 0 gives patient coordinates.
        ``patient_yx_mm`` then holds the patient position of every sample: (a cos r - p sin r, a sin r + p cos r) for a
        sample at ``a`` along y on a line at x = ``p``, and (p cos r - a sin r, p sin r + a cos r) along x at y = ``p``.
        """
        v = self.image(scan, image, kind)
        if axis not in ("y", "x"):
            raise InvalidRequest("axis must be y or x", parameter="axis", allowed=["y", "x"])
        if reduce not in ("mean", "max"):
            raise InvalidRequest("reduce must be mean or max", parameter="reduce", allowed=["mean", "max"])
        img = np.asarray(v["array"], float)
        rot = float(v.get("rotation_deg", 0.0) or 0.0)
        ypos, xpos = pixel_positions(img.shape, v["fov_yx_mm"], frame_centre(v["centre_yx_mm"], rot))
        along, across = (ypos, xpos) if axis == "y" else (xpos, ypos)
        if axis == "x":
            img = img.T                                                  # rows = position along the line
        d_al, d_ac = abs(along[1] - along[0]) if len(along) > 1 else 0.0, abs(across[1] - across[0]) if len(across) > 1 else 0.0
        other = "x" if axis == "y" else "y"

        def number(name, value, default, lo, hi):
            if value is None:
                return default
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InvalidRequest(f"{name} must be a finite number", parameter=name)
            if not lo - 1e-9 <= value <= hi + 1e-9:
                raise InvalidRequest(f"{name} = {value:g} is outside the image: {lo:.2f} to {hi:.2f} mm along {other if name == 'position_mm' else axis}",
                                     parameter=name, requested=value, low=round(float(lo), 2), high=round(float(hi), 2),
                                     nearest_valid=round(float(min(max(value, lo), hi)), 2))
            return float(value)

        ac_lo, ac_hi = across[0] - d_ac / 2, across[-1] + d_ac / 2
        al_lo, al_hi = along[0] - d_al / 2, along[-1] + d_al / 2
        pos = number("position_mm", position_mm, float(across[len(across) // 2]), ac_lo, ac_hi)
        start, stop = number("start_mm", start_mm, al_lo, al_lo, al_hi), number("stop_mm", stop_mm, al_hi, al_lo, al_hi)
        if stop < start:
            raise InvalidRequest(f"stop_mm {stop:g} is below start_mm {start:g}", parameter="stop_mm", requested=stop, nearest_valid=round(al_hi, 2))
        if width_mm is not None and (isinstance(width_mm, bool) or not isinstance(width_mm, (int, float)) or not math.isfinite(width_mm) or width_mm < 0):
            raise InvalidRequest("width_mm must be a number >= 0", parameter="width_mm", nearest_valid=0)
        if samples is not None and (isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= MAX_PROFILE_SAMPLES):
            raise InvalidRequest(f"samples must be an integer from 1 to {MAX_PROFILE_SAMPLES}", parameter="samples",
                                 low=1, high=MAX_PROFILE_SAMPLES, nearest_valid=MAX_PROFILE_SAMPLES)
        cols = np.flatnonzero(np.abs(across - pos) <= (width_mm or 0.0) / 2 + 1e-9)
        if not len(cols):
            cols = np.array([int(np.argmin(np.abs(across - pos)))])
        rows = np.flatnonzero((along >= start - d_al / 2 - 1e-9) & (along <= stop + d_al / 2 + 1e-9))
        if not len(rows):
            rows = np.array([int(np.argmin(np.abs(along - (start + stop) / 2)))])
        per = max(1, math.ceil(len(rows) / (samples or MAX_PROFILE_SAMPLES)))
        block = img[np.ix_(rows, cols)]
        positions, values = [], []
        for i in range(0, len(rows), per):
            part = block[i:i + per]
            if reduce == "max":
                value = float(part.max())
            elif kind == "phase":
                value = float(np.angle(np.exp(1j * part).mean()))
            else:
                value = float(part.mean())
            positions.append(round(float(along[rows[i:i + per]].mean()), 3))
            values.append(float(f"{value:.6g}"))
        turned = {}
        if rot:
            c, si, p = math.cos(math.radians(rot)), math.sin(math.radians(rot)), float(across[cols].mean())
            turned["patient_yx_mm"] = [[round(a * c - p * si, 3), round(a * si + p * c, 3)] if axis == "y" else
                                       [round(p * c - a * si, 3), round(p * si + a * c, 3)] for a in positions]
        return dict(scan=scan, image=image, kind=kind, axis=axis, reduce=reduce, rotation_deg=rot, **turned,
                    position_mm=round(float(across[cols].mean()), 3),
                    width_px=int(len(cols)), width_mm=round(float(len(cols) * d_ac), 3), start_mm=round(float(along[rows[0]] - d_al / 2), 3),
                    stop_mm=round(float(along[rows[-1]] + d_al / 2), 3), pixel_mm=round(float(d_al), 4), pixels_per_sample=per,
                    sample_mm=round(float(per * d_al), 4), sequence=v["sequence"], positions_mm=positions, values=values)


def frame_centre(centre_yx_mm, rotation_deg: float = 0.0) -> tuple:
    """The centre of a field of view in the frame of its own image: patient coordinates (y, x) turned back by the
    rotation about the isocentre. Positions along the rows and columns of a turned image are measured in this frame
    (``LocalScanner.profile``, ``image_extent``, the legend of ``view``); without a rotation it is the patient frame."""
    cy, cx = float(centre_yx_mm[0]), float(centre_yx_mm[1])
    if not rotation_deg:
        return (cy, cx)
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    return (cy * c + cx * s, -cy * s + cx * c)


def image_extent(stored: dict, image: int = 0):
    """Trainer side: the extent in mm of the pixels of a stored scan's magnitude image that are brighter than 20 % of
    its 99.5th percentile, ``{"y_mm": [lo, hi], "x_mm": [lo, hi], "touches_edge": bool}`` or None for an empty image.

    ``stored`` is an entry of ``LocalScanner.scans`` (or of ``env.stored_scans()``). Scripted control policies use it
    as their ruler. No tool returns it: what an agent reads off an image is the agent's own measurement.
    """
    from .. import render
    # for a turned field of view the extent is along the image rows and columns, about the centre of the field of view
    rot_deg = stored.get("rotation_deg", 0.0)
    rot = np.deg2rad(rot_deg)
    cy, cx = stored["centre_yx_mm"]
    frame = (cy * np.cos(rot) + cx * np.sin(rot), -cy * np.sin(rot) + cx * np.cos(rot)) if rot_deg else (cy, cx)
    box = render.bounding_box(stored["magnitude"][image], stored["fov_yx_mm"], centre_yx_mm=frame)
    if not box:
        return None
    return dict(y_mm=[round(float(v), 2) for v in box["y"]], x_mm=[round(float(v), 2) for v in box["x"]],
                touches_edge=box["touches_edge"])
