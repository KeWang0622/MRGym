"""One MR Gym episode as a Harbor task: the scanner service as a sidecar, the agent, and a separate verifier.

Three containers in a Harbor trial (``mrgym export harbor`` writes the task directories). The sidecar and the verifier
are the package's own service and verifier (``mrgym serve``, ``mrgym.epoch.verify_episode``); this module only joins
them to Harbor's file layout and adds nothing to the physics, the log or the score.

    scanner sidecar   mrgym serve --episode-file /episode/episode.json --agent-token-file /episode/agent_token
                          --log-dir /var/log/scanner --state-dir /var/lib/scanner --host 0.0.0.0 --port 8000
                      holds the hidden world. Agent plane only: info, task, budget, plan, scan, submit, each with the
                      episode's bearer token. It writes the keyed log (log directory) and, after every record, the head
                      receipt of that log (state directory). ``submit`` seals the log; the score is never returned.
    agent (main)      python -m mrgym.scanner.client info | task | budget | plan | scan      (or plain HTTP)
                      python -m mrgym.compat.harbor submit '<json>'     sends the deliverable to the scanner and writes
                                                                        it to /app/answer.json for the verifier
    verifier          python -m mrgym.compat.harbor verify --episode-file /tests/episode.json --log-dir /var/log/scanner
                          --state-dir /var/lib/scanner --answer /app/answer.json --reward-json /logs/verifier/reward.json
                      Harbor copies the sidecar's log directory and state directory into this container. The verifier
                      replays the log against the head receipt from the state directory (the trusted side: the agent's
                      container never held it), requires the seal, and scores the deliverable with the code that
                      scores in process. A log that is shortened, edited, foreign or unsealed gives the floor.

The deliverable is the agent's own statement, so it is read from the agent's container. Everything that proves what
was measured (the log, its head) comes from the sidecar.

The module is generic over the registry: an episode is ``{"env_id", "level", "split", "seed"}`` of any registered id
served by ``mrgym serve``; no tool, family or argument is named here.

    python -m mrgym.compat.harbor solve --env-id ID --out /app/answer.json     the id's scripted reference policy over HTTP
    python -m mrgym.compat.harbor lint TASK_DIR ...                            static checks that need no Docker
    python -m mrgym.compat.harbor local TASK_DIR [--policy reference|nop]      sidecar + agent side + verifier in one process
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import threading
from http.server import ThreadingHTTPServer

from . import CANARY, reference_name

ANSWER_PATH = "/app/answer.json"
DEFAULT_URL = "http://scanner:8000"
TOKEN_PATH = "/run/mrscan/token"
LOG_DIR, STATE_DIR = "/var/log/scanner", "/var/lib/scanner"
SPLIT = "test-iid"                       # a hidden-seed evaluation instance: the seed is the only thing held out
SEED_RE = re.compile(r"^[0-9a-f]{32}$")
EPISODE_KEYS = ("env_id", "level", "split", "seed")


# ------------------------------------------------------------------------------------------------ the hidden episode
def load_episode(path: str) -> dict:
    """The hidden episode file: ``{"env_id", "level", "split", "seed"}`` with a raw 128-bit seed (32 hex digits), in the
    form ``mrgym serve --episode-file`` reads."""
    with open(path, encoding="utf-8") as f:
        ep = json.load(f)
    if not isinstance(ep, dict) or set(ep) != set(EPISODE_KEYS) or not SEED_RE.match(str(ep.get("seed", ""))):
        raise SystemExit(f"{path}: not an episode file (the keys {list(EPISODE_KEYS)} with a 32-hex-digit seed)")
    return ep


# ------------------------------------------------------------------------------------------------ the agent side
class HttpSession:
    """The four calls ``scanner.session.LocalScanner`` needs from a session (``info``, ``budget``, ``plan``, ``scan``),
    served by the scanner service through ``scanner.client.Scanner``. ``LocalScanner(HttpSession(...))`` is then the
    interface the scripted policies are written against; its look-ups (regions, view, profile) are computed from the
    float32 images the service returned, exactly as in process."""

    def __init__(self, url: str | None = None, token: str | None = None):
        from ..scanner import client
        self.client = client.Scanner(url, token=token)
        self._types = None                                               # {sequence: {parameter: declared type}}, from the card

    def _call(self, method, path, body=None):
        from ..scanner import client
        from ..scanner import session as core
        try:
            return self.client._call(method, path, body)
        except client.ScannerError as e:
            types = {c.type: c for c in (core.InvalidRequest, core.BudgetExhausted, core.RequestLimit, core.EpisodeOver)}
            raise types.get(e.type, core.ScannerError)(e.message, **e.extra) from None

    def info(self) -> dict:
        return self._call("GET", "/v1/info")

    def budget(self) -> dict:
        return self._call("GET", "/v1/budget")

    def _canonical(self, request):
        """The request with every protocol number written in the type the instrument card declares (a number as a
        float, an integer as an int). The tool interface does the same before a call reaches the scanner
        (``mrgym.interface``), and the scanner logs a protocol as it was written: without this the same scan would be
        logged as ``"tr_ms": 50`` here and ``"tr_ms": 50.0`` through the tools, and the two logs would differ in text."""
        protocol = request.get("protocol") if isinstance(request, dict) else None
        if not isinstance(protocol, dict):
            return request
        if self._types is None:
            self._types = {seq: {name: p.get("type") for name, p in card.get("parameters", {}).items()}
                           for seq, card in (self.info().get("sequences") or {}).items()}
        types = self._types.get(protocol.get("sequence"), {})
        out = {}
        for name, value in protocol.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                out[name] = value
            elif types.get(name) == "number":
                out[name] = float(value)
            elif types.get(name) == "integer" and float(value) == int(value):
                out[name] = int(value)
            else:
                out[name] = value
        return dict(request, protocol=out)

    def plan(self, request) -> dict:
        return self._call("POST", "/v1/plan", self._canonical(request))

    def design(self, call) -> dict:
        return self._call("POST", "/v1/design", call)

    def scan(self, request):
        import base64

        import numpy as np
        resp = self._call("POST", "/v1/scan", self._canonical(request))
        arrays = {k[:-len("_npy_b64")]: np.load(io.BytesIO(base64.b64decode(resp.pop(k))), allow_pickle=False)
                  for k in list(resp) if k.endswith("_npy_b64")}
        return resp, arrays

    def submit(self, answer) -> dict:
        return self._call("POST", "/v1/submit", answer)


def _write_answer(out: str, answer) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(answer, f)


def submit(answer, out: str = ANSWER_PATH, url: str | None = None, token: str | None = None) -> dict:
    """Send the deliverable to the scanner; when it was received (the episode is then over), write it to ``out`` for the
    verifier. A deliverable the console cannot evaluate raises the scanner's typed refusal and the episode continues."""
    received = HttpSession(url, token).submit(answer)
    _write_answer(out, answer)
    return received


def solve(env_id: str, out: str = ANSWER_PATH, policy: str | None = None, url: str | None = None, token: str | None = None):
    """Play a scripted control policy of ``env_id`` (default: its reference) against the scanner service, deliver its
    answer and write it to ``out``. Returns the answer."""
    import mrgym

    from .. import policies
    from ..scanner import session as core
    spec = mrgym.spec(env_id)
    controls = policies.controls_of(spec)
    name = policy or reference_name(spec)
    if controls is None or name is None:
        raise SystemExit(f"{env_id} has no scripted policies")
    entry = controls.policies()[name]
    fn = entry[0] if isinstance(entry, tuple) else entry
    http = HttpSession(url, token)
    answer = fn(core.LocalScanner(http))
    http.submit(answer)
    _write_answer(out, answer)
    return answer


# ------------------------------------------------------------------------------------------------ the verifier
def verify(episode_file: str, log_dir: str, state_dir: str, answer_path: str, reward_json: str | None = None) -> dict:
    """Score the deliverable of a served episode. The log comes from ``log_dir/<handle>/``, its head receipt from
    ``state_dir/<handle>.head``: both written by the scanner service, never by the agent. Returns the report and
    writes ``{"reward": r}`` to ``reward_json`` (and a ``reward.txt`` next to it). Never raises for anything the agent
    did: every such case is the floor with the reason."""
    import mrgym

    from .. import epoch, keys
    from ..scanner import session as core
    spec = load_episode(episode_file)
    floor = float(mrgym.spec(spec["env_id"]).reward_range[0])
    report = dict(env_id=spec["env_id"], level=spec["level"], mrgym=mrgym.__version__, engine=mrgym.ENGINE_ID, reward=floor, passed=False,
                  submitted=False, sealed=False, log_replays=False, scans=0, problems=[], log_head=None)
    heads = sorted(glob.glob(os.path.join(state_dir, "*" + core.HEAD_SUFFIX))) if os.path.isdir(state_dir) else []
    head, events, answer = None, [], None
    if len(heads) != 1:
        report["problems"].append(f"expected the head receipt of one episode from the scanner's state directory, found {len(heads)}")
    else:
        try:
            # Read as JSON, not with session.read_head: Harbor normalises the permissions of copied artifacts, and this
            # container holds nothing but trusted material.
            with open(heads[0], encoding="utf-8") as f:
                head = json.load(f)
            events = core.read_log(os.path.join(log_dir, str(head["handle"]), core.LOG_NAME))
        except (OSError, ValueError, KeyError, TypeError) as e:
            head = None
            report["problems"].append(f"the log or its head cannot be read: {type(e).__name__}")
    try:
        with open(answer_path, encoding="utf-8") as f:
            answer = json.load(f)
        if not isinstance(answer, dict):
            report["problems"].append("the deliverable is not a JSON object")
            answer = None
    except FileNotFoundError:
        report["problems"].append("no deliverable was written")
    except (OSError, ValueError) as e:
        report["problems"].append(f"the deliverable cannot be read: {type(e).__name__}")
    if head is not None:
        master = keys.load_master()
        try:
            out = epoch.verify_episode(spec, events, head, answer if answer is not None else {}, log_key=keys.sub(master, "log") if master else None)
        except Exception as e:  # noqa: BLE001  a log this code cannot even walk is a failed episode, not a crashed verifier
            out = dict(reward=floor, passed=False, score=None, refused=None,
                       replay=dict(ok=False, sealed=False, problems=[f"the log cannot be replayed: {type(e).__name__}"]))
        rep = out["replay"] or {}
        report.update(log_replays=bool(rep.get("ok")), sealed=bool(rep.get("sealed")), scans=rep.get("scans", 0),
                      replay_mode=rep.get("mode"), log_head=events[-1].get("chain") if events and isinstance(events[-1], dict) else None)
        report["problems"] += [str(p) for p in rep.get("problems", [])]
        if answer is None:
            pass
        elif out.get("refused"):
            report["problems"].append("the deliverable was refused: " + str(out["refused"])[:300])
        elif not rep.get("ok"):
            pass
        elif not rep.get("sealed"):
            report["problems"].append("the episode was never sealed: no deliverable was received by the scanner")
        else:
            report["submitted"] = True
            report["reward"] = float(out["reward"])
            score = out.get("score") or {}
            report["checks"], report["diagnostic"] = score.get("checks"), score.get("diagnostic")
    spec_ = mrgym.spec(spec["env_id"])
    report["passed"] = bool(report["submitted"] and spec_.kind != "sandbox" and report["reward"] >= spec_.reward_threshold)
    if reward_json:
        d = os.path.dirname(os.path.abspath(reward_json))
        os.makedirs(d, exist_ok=True)
        with open(reward_json, "w", encoding="utf-8") as f:
            json.dump({"reward": report["reward"]}, f)
        with open(os.path.join(d, "reward.txt"), "w", encoding="utf-8") as f:
            f.write(f"{report['reward']:g}\n")
        with open(os.path.join(d, "mrgym_verify.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
    return report


# ------------------------------------------------------------------------------------------------ without Docker
class LocalService:
    """The scanner service of one episode on a free local port, in a background thread: the agent plane of
    ``mrgym serve`` with the same ``Service`` object, log directory and state directory. Use as a context manager."""

    def __init__(self, spec: dict, log_dir: str, state_dir: str, token: str | None = None):
        from ..scanner import service
        self.service = service.Service(log_dir, state_dir)
        self.episode = self.service.create(dict(spec), agent_token=token)
        self.token = self.episode.token
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), service.make_handler(self.service, "agent"))
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)
        self.service.close()


def local(task_dir: str, policy="reference", work_dir: str | None = None) -> dict:
    """Run the three sides of a task in this process, in the order of a Harbor trial: the scanner service on the
    sidecar's episode file and token, then the agent side (``policy``: the name of a scripted policy of the id, ``"nop"``
    for an agent that does nothing, or a callable ``(url, token, answer_path)``), then the verifier on the verifier's
    episode file and the service's log and state directories."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = work_dir or tmp
        log_dir, state_dir, answer_path = os.path.join(tmp, "log"), os.path.join(tmp, "state"), os.path.join(tmp, "app", "answer.json")
        spec = load_episode(os.path.join(task_dir, "environment", "scanner", "episode.json"))
        with open(os.path.join(task_dir, "environment", "scanner", "agent_token"), encoding="ascii") as f:
            token = f.read().strip()
        with LocalService(spec, log_dir, state_dir, token) as srv:
            if callable(policy):
                policy(srv.url, srv.token, answer_path)
            elif policy != "nop":
                solve(spec["env_id"], answer_path, None if policy == "reference" else policy, srv.url, srv.token)
        return verify(os.path.join(task_dir, "tests", "episode.json"), log_dir, state_dir, answer_path, os.path.join(tmp, "verifier", "reward.json"))


def _toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:                                          # Python 3.10
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return None
    with open(path, "rb") as f:
        return tomllib.load(f)


def lint(task_dir: str) -> list[tuple[str, str]]:
    """Static checks of an exported task that need neither Docker nor Harbor. Returns [(rule, message)]; empty = clean.
    They cover what Harbor's own lint covers for this layout plus what it cannot see (the hidden episode file)."""
    bad = []

    def read(rel):
        p = os.path.join(task_dir, rel)
        if not os.path.exists(p):
            bad.append(("missing-file", rel))
            return ""
        with open(p, encoding="utf-8") as f:
            return f.read()

    files = {rel: read(rel) for rel in (
        "task.toml", "instruction.md", "README.md", "environment/Dockerfile", "environment/.dockerignore", "environment/docker-compose.yaml",
        "environment/agent_token", "environment/scanner/Dockerfile", "environment/scanner/episode.json", "environment/scanner/agent_token",
        "solution/solve.sh", "tests/Dockerfile", "tests/test.sh", "tests/episode.json", "tests/export.json")}
    for rel in ("task.toml", "README.md", "solution/solve.sh", "tests/test.sh"):
        if files[rel] and CANARY not in files[rel]:
            bad.append(("canary", f"{rel} does not carry the canary string"))
    for rel in ("instruction.md", "environment/Dockerfile"):                 # the canary would tell the agent which benchmark it is in
        if CANARY in files[rel]:
            bad.append(("canary-in-agent-text", f"{rel} is read by the agent and carries the canary string"))
    if files["environment/scanner/episode.json"] != files["tests/episode.json"]:
        bad.append(("episode-mismatch", "the sidecar and the verifier hold different episode files"))
    if files["environment/agent_token"].strip() != files["environment/scanner/agent_token"].strip() or len(files["environment/agent_token"].strip()) < 16:
        bad.append(("agent-token", "the agent and the sidecar hold different tokens, or the token is shorter than 16 characters"))
    try:
        ep = json.loads(files["tests/episode.json"] or "{}")
    except ValueError:
        ep = {}
    seed = str(ep.get("seed", ""))
    if not SEED_RE.match(seed) or set(ep) != set(EPISODE_KEYS):
        bad.append(("episode-file", f"tests/episode.json is not {list(EPISODE_KEYS)} with a 32-hex-digit seed"))
    else:
        for rel, text in files.items():
            if "episode.json" not in rel and seed in text:
                bad.append(("seed-leak", f"{rel} contains the hidden seed"))
        if seed in os.path.basename(os.path.abspath(task_dir)):
            bad.append(("seed-leak", "the task directory name contains the hidden seed"))
    if "scanner" not in [ln.strip().strip("/") for ln in files["environment/.dockerignore"].splitlines()]:
        bad.append(("agent-context-leak", "environment/.dockerignore does not exclude scanner/ (the hidden episode) from the agent image"))
    agent = files["environment/Dockerfile"]
    if re.search(r"^\s*(COPY|ADD)\b.*(episode\.json|scanner/|tests|solution)", agent, re.M):
        bad.append(("agent-image-leak", "the agent Dockerfile copies the episode, the tests or the solution"))
    if not re.search(r"^\s*RUN\b.*useradd\b", agent, re.M | re.S):
        bad.append(("root-agent", "the agent image creates no unprivileged user"))
    sidecar = files["environment/scanner/Dockerfile"]
    if "--state-dir" not in sidecar or "--agent-token-file" not in sidecar or "--episode-file" not in sidecar:
        bad.append(("sidecar-command", "the sidecar does not run `mrgym serve` with an episode file, an agent token file and a state directory"))
    if re.search(r"MRGYM_SECRET=|MRGYM_SALT=", sidecar + agent + files["environment/docker-compose.yaml"]):
        bad.append(("secret-in-environment", "a secret or salt is put into a container's environment"))
    for rel in ("environment/Dockerfile", "environment/scanner/Dockerfile", "tests/Dockerfile"):
        for line in files[rel].splitlines():
            if line.strip().upper().startswith("FROM ") and "@sha256:" not in line:
                bad.append(("unpinned-image", f"{rel}: {line.strip()}"))
            if "pip install" in line and not re.search(r"mrgym==\d", line):
                bad.append(("unpinned-python-dependency", f"{rel}: {line.strip()[:120]}"))
    compose = files["environment/docker-compose.yaml"]
    for word in ("privileged", "cap_add", "volumes", "network_mode: host", "pid: host"):
        if word in compose:
            bad.append(("compose", f"environment/docker-compose.yaml uses {word!r}"))
    cfg = _toml(os.path.join(task_dir, "task.toml")) if files["task.toml"] else {}
    if cfg is None:
        bad.append(("toml", "no TOML parser on this Python (3.10 without tomli): task.toml was not checked"))
    elif cfg:
        if cfg.get("schema_version") != "1.4":
            bad.append(("task-config", "schema_version is not 1.4"))
        if cfg.get("agent", {}).get("user") in (None, "root", 0):
            bad.append(("root-agent", "[agent].user is omitted or root"))
        if not cfg.get("agent", {}).get("timeout_sec") or not cfg.get("verifier", {}).get("timeout_sec"):
            bad.append(("timeouts", "agent or verifier timeout is not declared"))
        if cfg.get("verifier", {}).get("environment_mode") != "separate":
            bad.append(("shared-verifier", "[verifier].environment_mode is not \"separate\""))
        if cfg.get("verifier", {}).get("environment", {}).get("network_mode") != "no-network":
            bad.append(("verifier-network", "the verifier environment is not no-network"))
        for section in (cfg.get("environment", {}), cfg.get("verifier", {}).get("environment", {})):
            if not all(section.get(k) for k in ("cpus", "memory_mb", "storage_mb")):
                bad.append(("resources", "cpus, memory_mb and storage_mb are not all declared"))
        sources = {a.get("source"): a.get("service") for a in cfg.get("artifacts", []) if isinstance(a, dict)}
        if ANSWER_PATH not in sources or sources.get(LOG_DIR) != "scanner" or sources.get(STATE_DIR) != "scanner":
            bad.append(("artifacts", "the deliverable, the sidecar's log and the sidecar's state directory (the head) are not all declared as artifacts"))
        md = cfg.get("metadata", {})
        if seed and any(seed in str(v) for v in md.values()):
            bad.append(("seed-leak", "[metadata] contains the hidden seed"))
    if "reward.json" not in files["tests/test.sh"] or "--state-dir" not in files["tests/test.sh"]:
        bad.append(("verifier-command", "tests/test.sh does not verify against the sidecar's state directory or does not write reward.json"))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mrgym.compat.harbor", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit", help="send the deliverable to the scanner and write it for the verifier: submit '<json object>'")
    p.add_argument("answer", help="the deliverable as a JSON object, or @FILE")
    p.add_argument("--out", default=ANSWER_PATH)
    p.add_argument("--url", default=None, help="default: $MRSCAN_URL")
    p = sub.add_parser("solve", help="a scripted control policy of the id over HTTP (default: the reference)")
    p.add_argument("--env-id", required=True)
    p.add_argument("--out", default=ANSWER_PATH)
    p.add_argument("--policy")
    p.add_argument("--url", default=None, help="default: $MRSCAN_URL")
    p = sub.add_parser("verify", help="replay the sidecar's log against its head and score the deliverable")
    p.add_argument("--episode-file", required=True)
    p.add_argument("--log-dir", default=LOG_DIR)
    p.add_argument("--state-dir", default=STATE_DIR)
    p.add_argument("--answer", default=ANSWER_PATH)
    p.add_argument("--reward-json")
    p = sub.add_parser("lint", help="static checks of exported tasks; needs no Docker")
    p.add_argument("task_dir", nargs="+")
    p = sub.add_parser("local", help="sidecar + agent side + verifier in one process, without Docker")
    p.add_argument("task_dir", nargs="+")
    p.add_argument("--policy", default="reference", help="a scripted policy of the id, or nop (no call, no deliverable)")
    a = ap.parse_args(argv)
    if a.cmd == "submit":
        from ..scanner import session as core
        try:
            if a.answer.startswith("@"):
                with open(a.answer[1:], encoding="utf-8") as f:
                    answer = json.load(f)
            else:
                answer = json.loads(a.answer)
        except (OSError, ValueError) as e:
            print(f"the deliverable is not JSON: {e}", file=sys.stderr)
            return 2
        try:
            print(json.dumps(submit(answer, a.out, a.url)))
        except core.ScannerError as e:
            print(json.dumps(e.to_json()))
            return 3
        return 0
    if a.cmd == "solve":
        solve(a.env_id, a.out, a.policy, a.url)
        return 0
    if a.cmd == "verify":
        print(json.dumps(verify(a.episode_file, a.log_dir, a.state_dir, a.answer, a.reward_json), indent=1, default=str))
        return 0
    if a.cmd == "lint":
        n = 0
        for d in a.task_dir:
            findings = lint(d)
            n += len(findings)
            print(f"{d}: {'clean' if not findings else str(len(findings)) + ' finding(s)'}")
            for rule, msg in findings:
                print(f"  {rule}: {msg}")
        return 1 if n else 0
    ok = True
    for d in a.task_dir:
        r = local(d, a.policy)
        expect = {"nop": 0.0, "reference": 1.0}.get(a.policy)
        ok = ok and (expect is None or r["reward"] == expect)
        print(f"{d}: policy {a.policy} -> reward {r['reward']:g}" + (f" (expected {expect:g})" if expect is not None else "")
              + f", {r['scans']} scan(s), log replays against its head: {r['log_replays']}, sealed: {r['sealed']}"
              + (f"; {'; '.join(r['problems'])}" if r["problems"] else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
