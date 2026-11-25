"""A stand-in for the model server in the GPU-free simulation: the package's stub endpoint plus one tunable "policy".

    python -m mrgym_verl.mixstub --port 0 --port-file PORT --nonce RUN --cells "Scanner/NoWrap-v1:2 Scanner/SNRSpec-v0:1" \
        --seeds train:9000-9007,train:10000-10063,dev:0-7,test-iid:0-7

Every model id of ``mrgym.agents.stub_server`` works (``stub:reference``, ``stub:recover``, ...). In addition

    mix:<p>[:<q>]    with probability ``q`` (default 0) the episode never calls a tool (it ends as
                     ``no_submission_stopped``); otherwise, at the first reply where the reference policy and the
                     no-scan ``prior`` policy differ, it follows the reference with probability ``p`` and the prior
                     with 1 - p. The reply at that branch says ``policy=reference`` or ``policy=prior``.

``p`` is the one parameter the simulated trainer updates: a pass rate that a policy gradient can move. Nothing here
is a model, and no number from a run against it is a result.

The package's stub replays policies on instances of the public train split only. Here the candidate instances carry
their split, so that ``dev`` rows and (with ``MRGYM_SECRET_FILE`` set for this process too) ``test-iid`` rows can be
simulated; that the stand-in knows the secret is a property of the simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import threading
from http.server import ThreadingHTTPServer

import mrgym
from mrgym import policies
from mrgym.agents import stub_server as S
from mrgym.agents.runner import parse_seeds, policy_recipe

_RNG, _RNG_LOCK = random.Random(0), threading.Lock()


class SplitTwins(S.Twins):
    """``Twins`` whose candidates are (environment, level, seed, split)."""

    def __init__(self, cells, seeds_by_split: dict):
        self.candidates = [(env_id, lv, int(s), split) for env_id, lv in cells for split, seeds in seeds_by_split.items() for s in seeds]
        self._statements, self._records, self._lock, self._by_body = {}, {}, threading.Lock(), {}

    def next_call(self, policy: str, statement: str, done):
        """``Twins.next_call`` over the candidates whose task statement is this one (an index by statement, built once,
        instead of a pass over every candidate at every request). Same rule: the candidates whose recorded episode
        agrees with the answered calls so far must all name the same next call."""
        body = S._body(statement)
        same = lambda rec: all(r[0] == d[0] and r[1] == d[1] and r[2] == S._body(d[2]) for r, d in zip(rec, done))   # noqa: E731
        with self._lock:
            if not self._by_body:
                for cand in self.candidates:
                    self._by_body.setdefault(self._statement(cand), []).append(cand)
            calls = set()
            for cand in self._by_body.get(body, []):
                rec = self._record(cand, policy)
                if done is None:
                    calls.add(json.dumps(rec[-1][:2], sort_keys=True))
                elif len(rec) > len(done) and same(rec):
                    calls.add(json.dumps(rec[len(done)][:2], sort_keys=True))
                elif len(rec) == len(done) and same(rec):
                    calls.add(json.dumps([None, None]))                  # the policy's episode ended with a refused delivery
        if len(calls) != 1:
            raise S.BadRequest(f"stub: {len(calls)} of the {len(self.candidates)} candidate instances continue this conversation as policy "
                               f"{policy!r} would; start the stand-in with the cells and seeds of the run")
        name, arguments = json.loads(calls.pop())
        return name, arguments

    def _statement(self, cand) -> str:
        if cand not in self._statements:
            env = mrgym.make(cand[0], level=cand[1], seed=cand[2], split=cand[3])
            self._statements[cand] = S._body(env.reset()[0].text)
            env.close()
        return self._statements[cand]

    def _record(self, cand, policy: str) -> list:
        if (cand, policy) not in self._records:
            env = mrgym.make(cand[0], level=cand[1], seed=cand[2], split=cand[3])
            if policy not in policies.policy_names(env.spec):
                raise S.BadRequest(f"The model `stub:{policy}` does not exist for {cand[0]}")
            policies.run(env, policy, recipe=policy_recipe(env.spec, cand[1]))
            self._records[(cand, policy)] = [(e["call"]["name"], S._norm(e["call"]["arguments"]), S._body(e["text"]))
                                             for e in env.transcript()[1:]]
            env.close()
        return self._records[(cand, policy)]


def mix_reply(p: float, q: float, body: dict) -> tuple[str, list]:
    msgs = body["messages"]
    turn = sum(m["role"] == "assistant" for m in msgs) + 1
    said = [m.get("content") or "" for m in msgs if m["role"] == "assistant"]
    with _RNG_LOCK:
        u, v = _RNG.random(), _RNG.random()
    if (turn == 1 and u < q) or any(t.startswith("policy=idle") for t in said):
        return "policy=idle. I will think about it.", []
    statement, done = S._statement(msgs), S.conversation_calls(msgs)
    nxt = {}
    for kind in ("reference", "prior"):
        try:
            nxt[kind] = S.TWINS.next_call(kind, statement, done)
        except S.BadRequest:
            pass
    if not nxt:
        raise S.BadRequest("mix: the conversation continues neither the reference nor the prior policy on a candidate instance; "
                           "start the stand-in with the cells and seeds of the run")
    branch = len(nxt) == 2 and nxt["reference"] != nxt["prior"]
    kind = ("reference" if v < p else "prior") if branch else next(iter(nxt))
    name, arguments = nxt[kind]
    if name is None:
        return "The policy has nothing further to send.", []
    return (f"policy={kind}" if branch else f"Step {turn}: {name}."), [S._call(name, arguments)]


def respond(body: dict) -> tuple[int, object]:
    model = body.get("model")
    if not isinstance(model, str) or not model.startswith("mix:"):
        return S.respond(body)
    try:
        parts = [float(x) for x in model[4:].split(":")]
        p, q = parts[0], parts[1] if len(parts) > 1 else 0.0
        assert 0.0 <= p <= 1.0 and 0.0 <= q <= 1.0 and len(parts) <= 2
    except (ValueError, AssertionError):
        raise S.BadRequest(f"The model `{model}` does not exist (mix:<p>[:<q>] with p, q in [0, 1])") from None
    text, calls = mix_reply(p, q, body)
    return _payload(body, text, calls)


def _payload(body: dict, text: str, calls: list) -> tuple[int, dict]:
    tcs = [{"id": "call_" + hashlib.md5((c["name"] + c["arguments"] + str(i) + str(len(body["messages"]))).encode()).hexdigest()[:20],
            "type": "function", "function": c} for i, c in enumerate(calls)]
    msg = {"role": "assistant", "content": text or None}
    if tcs:
        msg["tool_calls"] = tcs
    p = math.ceil((len(json.dumps(body["messages"])) + len(json.dumps(body.get("tools") or []))) / 4)
    c = math.ceil((len(text) + sum(len(x["arguments"]) + len(x["name"]) for x in calls)) / 4)
    return 200, {"id": "chatcmpl-mix", "object": "chat.completion", "model": body["model"],
                 "choices": [{"index": 0, "finish_reason": "tool_calls" if tcs else "stop", "message": msg}],
                 "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}}


NONCE = ""                                   # given at start; /health returns it, so a caller can tell its own server from another run's


class Handler(S.Handler):
    def do_GET(self):
        if self.path.rstrip("/").endswith("/health"):
            self._send(200, {"status": "ok", "nonce": NONCE})
        else:
            super().do_GET()

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        try:
            body = json.loads(raw)
            S.validate(body)
            status, payload = respond(body)
        except S.BadRequest as e:
            status, payload = 400, {"error": {"message": str(e), "type": "invalid_request_error"}}
        except ValueError as e:
            status, payload = 400, {"error": {"message": f"invalid JSON: {e}", "type": "invalid_request_error"}}
        except Exception as e:  # noqa: BLE001  a fault of the stand-in is reported as such, never as a model reply
            status, payload = 400, {"error": {"message": f"stub failure: {type(e).__name__}: {e}", "type": "stub_bug"}}
        self._send(status, payload)


def parse_split_seeds(text: str) -> dict:
    """``"train:9000-9007,train:10000-10063,dev:0-7"`` -> {"train": [...], "dev": [...]}."""
    out: dict = {}
    for part in text.split(","):
        split, _, rng = part.strip().partition(":")
        if split and rng:
            out.setdefault(split, []).extend(parse_seeds(rng))
    return {k: sorted(set(v)) for k, v in out.items()}


def start(port: int, cells, seeds_by_split: dict, host: str = "127.0.0.1", seed: int = 0, nonce: str = ""):
    """Start the stand-in in a daemon thread. ``port=0`` takes a free port. Returns ``(server, base_url)``."""
    global NONCE
    NONCE = nonce
    S.TWINS = SplitTwins(cells, seeds_by_split)
    _RNG.seed(seed)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return httpd, f"http://{host}:{httpd.server_address[1]}/v1"


def main(argv=None) -> int:
    from .instances import parse_arms
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--seeds", required=True, help="split:range[,split:range...] of the candidate instances")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonce", default="", help="returned by /health: lets a caller check that this is its own server")
    ap.add_argument("--port-file", default=None, help="write the port that was bound here (with --port 0: a free one)")
    a = ap.parse_args(argv)
    cells = []
    for part in a.cells.replace(";", " ").split():                       # the same cell may be named by several lists of a run
        cells += [c for c in parse_arms(part) if c not in cells]
    httpd, url = start(a.port, cells, parse_split_seeds(a.seeds), seed=a.seed, nonce=a.nonce)
    if a.port_file:
        with open(a.port_file + ".tmp", "w", encoding="utf-8") as f:
            f.write(str(httpd.server_address[1]))
        os.replace(a.port_file + ".tmp", a.port_file)
    print(f"stand-in endpoint at {url}; models: stub:<behaviour>, mix:<p>[:<q>]; loopback only", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
