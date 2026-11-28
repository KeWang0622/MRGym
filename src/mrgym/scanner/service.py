"""HTTP front end of the scanner: an agent plane and an operator plane (standard library + numpy).

Three parties. The operator (a trainer or a harness; trusted) creates episodes and reads results. The service (trusted)
holds the secret, the worlds, the logs and their heads. The agent (untrusted) holds a URL and one bearer token.

Agent plane, ``--host`` / ``--port``. Every route except the liveness probe needs ``Authorization: Bearer <agent token>``;
the token selects the episode.
  GET  /v1/health   liveness: {"ok": true}, nothing else, no token
  GET  /v1/info     instrument card, sequences, parameter ranges and units, cost rules, budget, task card
  GET  /v1/task     the task statement and the JSON Schema of the deliverable
  GET  /v1/budget   total / used / remaining scan seconds
  POST /v1/plan     {"protocol": {...}} -> realised protocol, timing, scan seconds, limits, accept / reject. Free.
  POST /v1/scan     {"protocol": {...}, "rois": [{"id", "y_mm", "x_mm", "radius_mm"}, ...]} -> ROI read-outs + images
  POST /v1/design   a generator call {"kind": ..., ...} -> the stored design (number, canonical call, facts). Free. Only on
                    a console that stores designs (a design task); otherwise refused.
  POST /v1/submit   the deliverable -> {"received": true}. Seals the log and ends the episode. The score is not returned.
Refusals return {"error": {"type": ..., "message": ...}} with status 422 (invalid_request, invalid_deliverable),
409 (budget_exhausted, episode_over) or 429 (request_limit); they cost nothing. A body that is not JSON, or is nested or
large beyond the limits, gets 400 and is neither logged nor counted. Without a valid token: 401. Anything else: 404.

Operator plane, ``--admin-socket PATH`` (a unix socket, mode 0600) or ``--admin-port`` (bound to 127.0.0.1), with
``Authorization: Bearer <operator token>`` (``--operator-token-file``). It is not reachable through the agent port.
  POST /v1/admin/episodes               {"env_id", "level", "split", "seed"} -> {"episode", "agent_token", "statement", "submit"}
  GET  /v1/admin/episodes               the handles of the episodes of this process
  GET  /v1/admin/episodes/<handle>      {"status", "reward", "passed", "score", "replay", "head", "budget"}
  POST /v1/admin/episodes/<handle>/close   seal an episode that will not be submitted (reward 0)

What an agent can ever receive is this list: the datasheet and sequence cards, the task statement and task card, limits,
and per request the realised protocol, timing, cost, remaining budget, counters, read-outs, images, typed refusals,
{"received": true}. Never a seed, the config, a task id, a score, a gate, the head or its tag, another episode, a server
path or exception text.

Nothing secret is on the command line or in the environment of the process (both are readable by the same user):
the secret comes from ``MRGYM_SECRET_FILE``, the episode from ``--episode-file`` / ``--episode-stdin`` or the operator
plane, tokens from files. The service refuses to start with ``MRGYM_SECRET`` or ``MRGYM_SALT`` in its environment.

Run:  mrgym serve --episode-file /run/secrets/episode --agent-token-file /run/secrets/agent_token \\
                  --log-dir /var/log/scanner --state-dir /var/lib/scanner --port 8000
Exit status: 2 refused to start (isolation, secret in the environment, public split without --public-worlds-ok),
3 the log does not match its receipt.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import http.client
import io
import json
import os
import secrets
import signal
import socket
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .. import keys
from . import session as core

MAX_DEPTH, MAX_NODES = 16, 4000
EPISODE_KEYS = ("env_id", "level", "split", "seed", "obs", "epoch")
AGENT_ROUTES = "GET /v1/info, /v1/task, /v1/budget; POST /v1/plan, /v1/scan, /v1/design, /v1/submit"


class StartupRefused(Exception):
    """The service will not start (exit status 2)."""
    EXIT_STATUS = 2


class _BadBody(Exception):
    pass


def _bounded(obj):
    """True when a parsed JSON value is within the nesting and size limits (iterative: no recursion to exhaust)."""
    stack, nodes = [(obj, 1)], 0
    while stack:
        v, depth = stack.pop()
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            return False
        if isinstance(v, dict):
            stack.extend((x, depth + 1) for x in v.values())
        elif isinstance(v, list):
            stack.extend((x, depth + 1) for x in v)
    return True


def _reject_constant(name):
    raise ValueError(f"{name} is not JSON")


def parse_body(raw: bytes):
    try:
        obj = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, RecursionError, UnicodeDecodeError):
        raise _BadBody from None
    if not _bounded(obj):
        raise _BadBody
    return obj


def read_token(path: str) -> str:
    with open(path, encoding="ascii") as f:
        token = f.read().strip()
    if len(token) < 16:
        raise StartupRefused(f"the token in {path!r} is shorter than 16 characters")
    return token


class Episode:
    def __init__(self, handle, token, session, env=None, statement=None, submit_schema=None, meta=None):
        self.handle, self.token, self.session, self.env = handle, token, session, env
        self.statement, self.submit_schema, self.meta = statement, submit_schema, dict(meta or {})
        self.result = None
        self.lock = threading.Lock()


class Service:
    """The episodes of one service process. All state that matters is in the sessions (log, head)."""

    def __init__(self, log_dir, state_dir, operator_token=None, public_worlds_ok=False, log_key=None):
        if not state_dir:
            raise StartupRefused("a state directory is required: the head of every log is kept there, outside the log directory")
        self.log_dir, self.state_dir = log_dir, state_dir
        self.operator_token, self.public_worlds_ok, self.log_key = operator_token, public_worlds_ok, log_key
        self.episodes: dict[str, Episode] = {}
        self.lock = threading.Lock()
        for d in (log_dir, state_dir):
            os.makedirs(d, mode=0o700, exist_ok=True)

    # ------------------------------------------------------------------ episodes
    def _add(self, ep):
        with self.lock:
            self.episodes[ep.handle] = ep
        return ep

    def by_token(self, token: str):
        found = None
        for ep in list(self.episodes.values()):                         # constant-time comparison against every token
            if hmac.compare_digest(ep.token.encode(), token.encode()):
                found = ep
        return found

    def create(self, spec: dict, agent_token: str | None = None, resume_from=None) -> Episode:
        """An episode of a registered environment: ``{"env_id", "level", "split", "seed"}``."""
        import mrgym
        if not isinstance(spec, dict) or "env_id" not in spec or set(spec) - set(EPISODE_KEYS):
            raise core.InvalidRequest(f"an episode is an object with the keys {list(EPISODE_KEYS)}; env_id is required")
        split = spec.get("split", "train")
        try:
            public = keys.is_public(split)
            if public and not self.public_worlds_ok and not keys.load_salt():
                raise core.InvalidRequest(
                    f"split {split!r} is public: its worlds can be rebuilt from the seed by anyone who installs the package. "
                    "Serve a secret split, set a salt (MRGYM_SALT_FILE), or start the service with --public-worlds-ok")
            kw = {k: spec[k] for k in ("level", "obs") if spec.get(k) is not None}
            if spec.get("epoch") is not None and str(spec["epoch"]) != keys.epoch_of():
                raise core.InvalidRequest(f"this service serves epoch {keys.epoch_of()!r} (MRGYM_EPOCH), not {spec['epoch']!r}")
            env = mrgym.make(spec["env_id"], seed=spec.get("seed"), split=split, **kw)
            obs, info = env.reset()
        except mrgym.Error as e:
            raise core.InvalidRequest(str(e)) from None
        cfg = env.episode_config()
        receipt = core.read_head(resume_from) if isinstance(resume_from, (str, os.PathLike)) else resume_from
        handle = str(receipt["handle"]) if receipt else info["episode_id"]
        sess = core.ScannerSession(cfg, log_dir=os.path.join(self.log_dir, handle), engine=env._session.engine,
                                   state_dir=self.state_dir, resume_from=receipt, handle=handle, log_key=self.log_key)
        env.use_session(sess)
        terminal = next(t for t in env._tool_specs() if t.terminal)
        ep = Episode(handle, agent_token or secrets.token_urlsafe(24), sess, env=env, statement=obs.text,
                     submit_schema=terminal.parameters,
                     meta=dict(env_id=spec["env_id"], level=info["level"], split=split, terminal=terminal.name))
        return self._add(ep)

    def adopt(self, cfg: dict, agent_token: str | None = None, resume_from=None) -> Episode:
        """An episode from a raw session config (no environment: nothing is scored, ``submit`` only seals)."""
        receipt = core.read_head(resume_from) if isinstance(resume_from, (str, os.PathLike)) else resume_from
        handle = str(receipt["handle"]) if receipt else keys.episode_handle()
        sess = core.ScannerSession(cfg, log_dir=os.path.join(self.log_dir, handle), state_dir=self.state_dir,
                                   resume_from=receipt, handle=handle, log_key=self.log_key)
        return self._add(Episode(handle, agent_token or secrets.token_urlsafe(24), sess, meta=dict(env_id=None)))

    def submit(self, ep: Episode, answer) -> dict:
        with ep.lock:
            if ep.result is not None or ep.session.sealed is not None:
                raise core.EpisodeOver("the episode is over; the deliverable was already received or the episode was closed")
            if ep.env is None:
                ep.result = dict(reward=None, passed=None, score=None, replay=None, log_head=ep.session.seal("submit"))
                return {"received": True}
            step = ep.env.step({"name": ep.meta["terminal"], "arguments": answer})
            if not step.done:                                            # refused: nothing was delivered, the episode goes on
                err = step.observation.data.get("error") or {}
                e = core.InvalidRequest(str(err.get("message") or "the deliverable was refused"))
                e.type = "invalid_deliverable"
                raise e
            info = step.info
            score = info.get("score") or {}
            ep.result = dict(reward=float(step.reward), passed=score.get("passed"), score=score, replay=info.get("replay"),
                             log_head=info.get("log_head") or ep.session.receipt(), truncated=bool(step.truncated))
            return {"received": True}

    def close_episode(self, ep: Episode) -> dict:
        with ep.lock:
            if ep.result is None:
                ep.result = dict(reward=0.0, passed=False, score=None, replay=None, log_head=ep.session.seal("close"),
                                 closed_without_deliverable=True)
        return self.report(ep)

    def report(self, ep: Episode) -> dict:
        """Operator side. Holds the score and the head; never the seed or the config."""
        r = ep.result or {}
        return keys.stamp(dict(episode=ep.handle, status="done" if ep.result is not None else "open", **ep.meta,
                               reward=r.get("reward"), passed=r.get("passed"), score=r.get("score"), replay=r.get("replay"),
                               head=r.get("log_head") or ep.session.receipt(), budget=ep.session.budget(),
                               **{k: r[k] for k in ("truncated", "closed_without_deliverable") if k in r}))

    def close(self):
        for ep in list(self.episodes.values()):
            ep.session.close()


# ------------------------------------------------------------------------------------------------ HTTP
def make_handler(service: Service, plane: str, request_timeout: float = 30.0):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "mrgym-scanner/2"
        sys_version = ""
        timeout = request_timeout                                        # a stalled connection is closed, not kept

        def _send(self, status, payload, headers=()):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, type_, message, **kw):
            self._send(status, {"error": {"type": type_, "message": message}}, **kw)

        def _bearer(self):
            h = self.headers.get("Authorization") or ""
            return h[7:].strip() if h[:7].lower() == "bearer " else ""

        def _body(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > core.MAX_BODY_BYTES:
                self.close_connection = True
                self._error(413, "invalid_request", "body missing or too large")
                return None, False
            raw = self.rfile.read(n) if n else b""
            try:
                return parse_body(raw or b"null"), True
            except _BadBody:
                self._error(400, "bad_request", "the body is not JSON, or is nested or large beyond the limits; nothing was logged")
                return None, False

        def _route(self, method):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/v1/health" and method == "GET":
                return self._send(200, {"ok": True})
            if plane == "admin":
                return self._admin(method, path)
            return self._agent(method, path)

        # -------------------------------------------------------------- agent plane
        def _agent(self, method, path):
            routes = {"GET": ("/v1/info", "/v1/task", "/v1/budget"), "POST": ("/v1/plan", "/v1/scan", "/v1/design", "/v1/submit")}
            if path not in routes.get(method, ()):
                if method == "POST":
                    self.close_connection = True                         # an unread body must not be parsed as a request
                return self._error(404, "not_found", "routes: " + AGENT_ROUTES)
            ep = service.by_token(self._bearer())
            if ep is None:
                if method == "POST":
                    self.close_connection = True
                return self._error(401, "unauthorized", "send the agent token of the episode: Authorization: Bearer <token>",
                                   headers=(("WWW-Authenticate", "Bearer"),))
            if method == "GET":
                if path == "/v1/info":
                    return self._send(200, ep.session.info())
                if path == "/v1/task":
                    return self._send(200, {"statement": ep.statement, "submit": ep.submit_schema})
                return self._send(200, ep.session.budget())
            request, ok = self._body()
            if not ok:
                return None
            try:
                if path == "/v1/plan":
                    return self._send(200, ep.session.plan(request))
                if path == "/v1/design":
                    return self._send(200, ep.session.design(request))
                if path == "/v1/submit":
                    return self._send(200, service.submit(ep, request))
                resp, arrays = ep.session.scan(request)
            except core.ScannerError as e:
                return self._send(e.http_status, e.to_json())
            for name, a in arrays.items():
                buf = io.BytesIO()
                np.save(buf, a, allow_pickle=False)
                resp[f"{name}_npy_b64"] = base64.b64encode(buf.getvalue()).decode()
            return self._send(200, resp)

        # -------------------------------------------------------------- operator plane
        def _admin(self, method, path):
            if not path.startswith("/v1/admin/"):
                return self._error(404, "not_found", "operator routes live under /v1/admin/")
            token = service.operator_token
            if not token or not hmac.compare_digest(token.encode(), self._bearer().encode()):
                if method == "POST":
                    self.close_connection = True
                return self._error(401, "unauthorized", "send the operator token: Authorization: Bearer <token>",
                                   headers=(("WWW-Authenticate", "Bearer"),))
            parts = path.split("/")[3:]                                  # after /v1/admin
            if parts[:1] != ["episodes"] or len(parts) > 3:
                return self._error(404, "not_found", "operator routes: /v1/admin/episodes[/<handle>[/close]]")
            if method == "GET" and len(parts) == 1:
                return self._send(200, {"episodes": sorted(service.episodes)})
            if method == "POST" and len(parts) == 1:
                spec, ok = self._body()
                if not ok:
                    return None
                try:
                    ep = service.create(spec)
                except core.ScannerError as e:
                    return self._send(e.http_status, e.to_json())
                return self._send(200, keys.stamp({"episode": ep.handle, "agent_token": ep.token, "statement": ep.statement,
                                                   "submit": ep.submit_schema}))
            ep = service.episodes.get(parts[1]) if len(parts) >= 2 else None
            if ep is None:
                return self._error(404, "not_found", "no such episode")
            if method == "GET" and len(parts) == 2:
                return self._send(200, service.report(ep))
            if method == "POST" and parts[2:] == ["close"]:
                self.close_connection = True
                return self._send(200, service.close_episode(ep))
            return self._error(404, "not_found", "operator routes: /v1/admin/episodes[/<handle>[/close]]")

        def _guarded(self, method):
            try:
                self._route(method)
            except (TimeoutError, ConnectionError, BrokenPipeError):
                self.close_connection = True
            except Exception:  # noqa: BLE001  nothing about the failure goes to the client
                self.close_connection = True
                try:
                    self._send(500, {"error": {"type": "internal_error"}})
                except Exception:  # noqa: BLE001
                    pass

        def do_GET(self):
            self._guarded("GET")

        def do_POST(self):
            self._guarded("POST")

        def log_message(self, fmt, *args):      # the scan log is the record; keep stderr quiet
            pass

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def handle_error(self, request, client_address):                     # a dropped connection is not news
        pass


class _UnixServer(_Server):
    address_family = socket.AF_UNIX

    def server_bind(self):
        path = self.server_address
        if os.path.exists(path):
            os.unlink(path)
        old = os.umask(0o177)                                            # the socket is created 0600
        try:
            self.socket.bind(path)
        finally:
            os.umask(old)
        self.server_name, self.server_port = "localhost", 0

    def get_request(self):
        request, _ = self.socket.accept()
        return request, ("local", 0)


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=60.0):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def admin_request(target: str, token: str, method: str, path: str, body=None, timeout: float = 120.0):
    """Operator-side call. ``target``: ``unix:/path/to/socket`` or ``http://127.0.0.1:port``. Returns (status, JSON)."""
    if target.startswith("unix:"):
        conn = _UnixConnection(target[5:], timeout)
    else:
        host, _, port = target.split("//", 1)[-1].rstrip("/").partition(":")
        conn = http.client.HTTPConnection(host, int(port or 80), timeout=timeout)
    try:
        data = None if body is None else json.dumps(body).encode()
        conn.request(method, path, body=data, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                                                       "Connection": "close"})
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"null")
    finally:
        conn.close()


def _write_private(path, text):
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def serve(service: Service, host="127.0.0.1", port=8000, port_file=None, admin_socket=None, admin_port=None,
          ready_file=None, request_timeout=30.0):
    """Serve until SIGTERM / SIGINT. Sessions are closed, not sealed: a stopped episode can be resumed with its receipt."""
    servers = [_Server((host, port), make_handler(service, "agent", request_timeout))]
    ready = {"port": servers[0].server_address[1], "pid": os.getpid(), "episodes": sorted(service.episodes)}
    if admin_socket:
        servers.append(_UnixServer(admin_socket, make_handler(service, "admin", request_timeout)))
        ready["admin"] = "unix:" + admin_socket
    elif admin_port is not None:
        servers.append(_Server(("127.0.0.1", admin_port), make_handler(service, "admin", request_timeout)))
        ready["admin"] = f"http://127.0.0.1:{servers[-1].server_address[1]}"
    threads = [threading.Thread(target=s.serve_forever, kwargs=dict(poll_interval=0.2), daemon=True) for s in servers[1:]]
    for t in threads:
        t.start()
    if ready_file:                              # written after bind: its existence means "accepting connections"
        _write_private(ready_file, json.dumps(ready))
    if port_file:
        _write_private(port_file, str(ready["port"]))

    def stop(*_):
        for s in servers:
            threading.Thread(target=s.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        servers[0].serve_forever(poll_interval=0.2)
    finally:
        for s in servers:
            s.server_close()
        if admin_socket and os.path.exists(admin_socket):
            os.unlink(admin_socket)
        service.close()


# ------------------------------------------------------------------------------------------------ start-up
def harden():
    """Best effort, Linux: make the process non-dumpable, so that its memory and its ``/proc/<pid>/environ`` are not
    readable by other processes of the same user. Returns True when it took effect."""
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) == 0          # PR_SET_DUMPABLE = 4
    except Exception:  # noqa: BLE001
        return False


def isolation_problems(paths, agent_uid: int) -> list:
    """Why the agent's uid could read or write the service's files. Empty = the file boundary holds (for that uid)."""
    problems = []
    if agent_uid == os.geteuid():
        problems.append(f"the agent's uid {agent_uid} is the service's own uid: every file and the memory of the service are its")
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        st = os.stat(p)
        mode = stat.S_IMODE(st.st_mode)
        if st.st_uid == agent_uid:
            problems.append(f"{p} is owned by the agent's uid {agent_uid}")
        if mode & 0o077:
            problems.append(f"{p} has mode {mode:04o}: readable or writable by group or others")
    return problems


def add_arguments(ap):
    ap.add_argument("--episode-file", default=None, help="JSON {env_id, level, split, seed}: the one episode of this process "
                    "(a file of mode 0600, e.g. a container secret). Without it episodes come from the operator plane")
    ap.add_argument("--episode-stdin", action="store_true", help="read that JSON from standard input instead")
    ap.add_argument("--config", default=None, help="a raw session config (mode 0600; holds the seed) instead of an episode: "
                    "served without scoring")
    ap.add_argument("--resume-from", default=None, help="receipt (<state-dir>/<handle>.head) of the episode to continue "
                    "after a stop; the log must match it, else exit status 3")
    ap.add_argument("--agent-token-file", default=None, help="token of the start-up episode (default: a new one, written "
                    "to <state-dir>/<handle>.token)")
    ap.add_argument("--operator-token-file", default=None, help="enables the operator plane")
    ap.add_argument("--admin-socket", default=None, help="unix socket of the operator plane")
    ap.add_argument("--admin-port", type=int, default=None, help="operator plane on 127.0.0.1:<port> (0 = pick a free one)")
    ap.add_argument("--log-dir", default=os.environ.get("MRSCAN_LOG_DIR", "runs/scanner_log"))
    ap.add_argument("--state-dir", default=os.environ.get("MRSCAN_STATE_DIR"), help="where the head of every log is kept; "
                    "not the log directory, and out of the agent's reach (default: <log-dir>.state)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000, help="0 = pick a free port (see --ready-file)")
    ap.add_argument("--port-file", default=None)
    ap.add_argument("--ready-file", default=None, help="JSON {port, admin, pid, episodes}, written once the ports are bound")
    ap.add_argument("--request-timeout", type=float, default=30.0, help="seconds before a stalled connection is closed")
    ap.add_argument("--public-worlds-ok", action="store_true", help="serve public, unsalted splits (demos; the worlds can "
                    "be rebuilt by anyone who installs the package)")
    ap.add_argument("--require-isolation", action="store_true", help="exit 2 unless the files of the service are out of "
                    "reach of --agent-uid")
    ap.add_argument("--agent-uid", type=int, default=None)


def run(a) -> int:
    """Start the service from parsed arguments. Returns the exit status."""
    try:
        for name in ("MRGYM_SECRET", "MRGYM_SALT"):
            if os.environ.get(name):
                raise StartupRefused(f"{name} is set in the environment of the service, where the same user can read it "
                                     f"(/proc/<pid>/environ). Put it in a file of mode 0600 and set {name}_FILE")
        harden()
        state_dir = a.state_dir or (a.log_dir.rstrip("/") + ".state")
        try:
            master = keys.load_master()
        except keys.Error as e:
            raise StartupRefused(str(e)) from None
        if a.require_isolation:
            if a.agent_uid is None:
                raise StartupRefused("--require-isolation needs --agent-uid")
            for d in (a.log_dir, state_dir):
                os.makedirs(d, mode=0o700, exist_ok=True)
            paths = [a.log_dir, state_dir, a.config, a.episode_file, a.resume_from, a.operator_token_file,
                     os.environ.get("MRGYM_SECRET_FILE"), os.environ.get("MRGYM_SALT_FILE")]
            problems = isolation_problems(paths, a.agent_uid)
            if problems:
                raise StartupRefused("isolation check failed:\n  " + "\n  ".join(problems))
        service = Service(a.log_dir, state_dir, operator_token=read_token(a.operator_token_file) if a.operator_token_file else None,
                          public_worlds_ok=a.public_worlds_ok, log_key=keys.sub(master, "log") if master else None)
        token = read_token(a.agent_token_file) if a.agent_token_file else None
        ep = None
        if a.episode_file or a.episode_stdin:
            if a.episode_file:
                st = os.stat(a.episode_file)
                if stat.S_ISREG(st.st_mode) and st.st_mode & 0o077:
                    raise StartupRefused(f"the episode file {a.episode_file!r} holds the seed and is readable by group or others; chmod 600")
                with open(a.episode_file, encoding="utf-8") as f:
                    spec = json.load(f)
            else:
                spec = json.load(sys.stdin)
            try:
                ep = service.create(spec, agent_token=token, resume_from=a.resume_from)
            except core.ScannerError as e:
                raise StartupRefused(e.message) from None
        elif a.config:
            ep = service.adopt(core.load_config(a.config), agent_token=token, resume_from=a.resume_from)
        elif not service.operator_token:
            raise StartupRefused("nothing to serve: give --episode-file, --episode-stdin or --config, or enable the operator "
                                 "plane with --operator-token-file and --admin-socket / --admin-port")
        if service.operator_token and not (a.admin_socket or a.admin_port is not None):
            raise StartupRefused("--operator-token-file needs --admin-socket or --admin-port")
        if ep is not None and token is None:
            _write_private(os.path.join(state_dir, ep.handle + ".token"), ep.token + "\n")
    except StartupRefused as e:
        print(f"mrgym serve: {e}", file=sys.stderr)
        return StartupRefused.EXIT_STATUS
    except core.LogTampered as e:
        print(f"mrgym serve: log does not match receipt: {e}", file=sys.stderr)
        return core.LogTampered.EXIT_STATUS
    serve(service, a.host, a.port, a.port_file, a.admin_socket, a.admin_port, a.ready_file, a.request_timeout)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
