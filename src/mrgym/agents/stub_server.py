"""A stub OpenAI-compatible chat-completions endpoint on loopback, for testing the runner without any model.

    python -m mrgym.agents.stub_server --port 4199 --seeds 0-3
    mrgym run-agent --env Scanner/NoWrap-v1 --seeds 0-3 --model stub:reference --base-url http://127.0.0.1:4199/v1

The "model" id selects a scripted behaviour, ``stub:<behaviour>[+flag+flag]``.

Behaviours that replay a control policy of the environment as tool calls
  <policy>        any control policy of the environment: ``reference``, ``prior`` (the no-scan answer),
                  ``fixed_recipe``, ...
  recover         two bad turns (cut-off JSON arguments, an unknown tool), then ``reference``
  overspend       scans until the scanner refuses for lack of budget, then delivers the ``prior`` answer

  How a replay works. The stub sees only the conversation. It holds a list of candidate instances (the scored
  environments, their levels and the public train seeds given at start). For every candidate whose task statement
  equals the first user message it plays the policy once on a twin environment made from the same public seed and
  keeps the recorded tool calls and observations. The candidates whose record agrees with the tool results in the
  conversation so far all name the same next call; that call is the reply. An instance outside the candidate list
  gets HTTP 400.

Behaviours that fail
  malformed_json  every turn a ``scan`` call whose arguments are cut-off JSON
  unknown_tool    every turn a call to a tool that does not exist
  bad_args        wrong argument types, a scan index that does not exist, a ``calc`` that is not arithmetic, a
                  protocol parameter that does not exist, a deliverable of the wrong shape, and at last a
                  deliverable that fits the schema and nothing else
  text_only       the answer as prose, never a tool call
  empty           no text and no tool call
  flood           30 tool calls in every turn

Flags (habits of real endpoints)
  split      the text in ``choices[0]``, the tool calls in ``choices[1]``
  nousage    no ``usage`` block in the response
  noparams   HTTP 400 if ``temperature`` or ``max_tokens`` is sent ("unsupported parameter")
  flaky      the first request of a conversation fails once with HTTP 500, and one later response is not JSON
  dead       always HTTP 500

Requests are validated strictly, as the strictest model behind a gateway would: unknown top-level fields, unknown
message keys, content that is not a string, a tool call whose arguments are not a JSON object, a tool message
without a matching call and unanswered tool calls are all HTTP 400. A passing run against the stub therefore also
shows that the runner sends standard fields only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOP_FIELDS = {"model", "messages", "tools", "tool_choice", "temperature", "top_p", "max_tokens", "max_completion_tokens",
              "seed", "stop", "n", "stream"}
MSG_KEYS = {"system": {"role", "content"}, "user": {"role", "content"}, "assistant": {"role", "content", "tool_calls"},
            "tool": {"role", "content", "tool_call_id"}}
FAILURES = ("malformed_json", "unknown_tool", "bad_args", "text_only", "empty", "flood")
FLAGS = ("split", "nousage", "noparams", "flaky", "dead")
OVERSPEND_SCAN = {"protocol": {"sequence": "spgr", "tr_ms": 25, "flip_deg": 12, "matrix": 96, "averages": 8}}


class BadRequest(Exception):
    pass


def validate(body) -> None:
    """Raise ``BadRequest`` unless ``body`` is a chat-completions request made of standard fields only."""
    if not isinstance(body, dict):
        raise BadRequest("body must be a JSON object")
    extra = set(body) - TOP_FIELDS
    if extra:
        raise BadRequest(f"Unrecognized request argument supplied: {sorted(extra)[0]}")
    if body.get("stream"):
        raise BadRequest("stream is not supported by the stub")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("messages must be a non-empty list")
    for t in body.get("tools") or []:
        if not isinstance(t, dict) or set(t) != {"type", "function"} or t["type"] != "function" \
                or set(t["function"]) - {"name", "description", "parameters"}:
            raise BadRequest("tools[]: only {type: function, function: {name, description, parameters}} is accepted")
    pending = []
    for i, m in enumerate(msgs):
        role = m.get("role") if isinstance(m, dict) else None
        if role not in MSG_KEYS:
            raise BadRequest(f"messages[{i}]: unknown role {role!r}")
        if set(m) - MSG_KEYS[role]:
            raise BadRequest(f"messages[{i}]: unknown field(s) {sorted(set(m) - MSG_KEYS[role])}")
        c = m.get("content")
        if not (isinstance(c, str) or (c is None and role == "assistant" and m.get("tool_calls"))):
            raise BadRequest(f"messages[{i}]: content must be a string (image and multi-part content are not supported)")
        if isinstance(c, str) and not c and role != "tool":
            raise BadRequest(f"messages[{i}]: text content must be non-empty")
        if role != "tool" and pending:
            raise BadRequest(f"messages[{i}]: tool call(s) {pending} were not answered by tool messages")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not isinstance(tc, dict) or set(tc) != {"id", "type", "function"} or not isinstance(fn, dict) \
                        or set(fn) != {"name", "arguments"}:
                    raise BadRequest(f"messages[{i}]: tool_calls[] must be {{id, type, function: {{name, arguments}}}}")
                try:
                    ok = isinstance(json.loads(fn["arguments"]), dict)
                except (ValueError, TypeError):
                    ok = False
                if not ok:
                    raise BadRequest(f"messages[{i}]: tool call arguments must be a JSON object in a string")
                pending.append(tc["id"])
        if role == "tool":
            if m.get("tool_call_id") not in pending:
                raise BadRequest(f"messages[{i}]: tool_call_id {m.get('tool_call_id')!r} matches no pending tool call")
            pending.remove(m["tool_call_id"])
    if pending:
        raise BadRequest(f"tool call(s) {pending} were not answered")


# ------------------------------------------------------------------ twins: a control policy recorded on a public seed
def _norm(x):
    return json.loads(json.dumps(x, default=float))


def _body(text: str) -> str:
    """An observation without what depends on the settings and counters of the episode it was read in (the twin plays
    on the environment's own limits and tools and never makes the bad calls): the status line, the ``charged:`` line
    of a refusal, and the Tools line and the Limits block of the task text."""
    lines = [ln for ln in text.rstrip().split("\n") if not ln.startswith(("charged: ", "Limits. ", "Tools: "))]
    return "\n".join(lines[:-1] if lines and lines[-1].startswith("[") and lines[-1].endswith("]") else lines).rstrip()


class Twins:
    """Candidate instances and, per policy, the recorded episode of that policy on each of them."""

    def __init__(self, seeds=range(4), env_ids=None, levels=None):
        import mrgym
        self.candidates = [(i, lv, int(s)) for i in (env_ids or [e for e in mrgym.list_envs() if mrgym.spec(e).family])
                           for lv in mrgym.spec(i).levels if levels is None or lv in levels for s in seeds]
        self._statements, self._records, self._lock = {}, {}, threading.Lock()

    def _statement(self, cand) -> str:
        import mrgym
        if cand not in self._statements:
            env = mrgym.make(cand[0], level=cand[1], seed=cand[2])
            self._statements[cand] = _body(env.reset()[0].text)
            env.close()
        return self._statements[cand]

    def _record(self, cand, policy: str) -> list:
        """[(tool name, arguments, observation text)] of the policy's episode on the candidate, submission included."""
        import mrgym

        from .. import policies
        from .runner import policy_recipe
        if (cand, policy) not in self._records:
            env = mrgym.make(cand[0], level=cand[1], seed=cand[2])
            if policy not in policies.policy_names(env.spec):
                raise BadRequest(f"The model `stub:{policy}` does not exist for {cand[0]}")
            policies.run(env, policy, recipe=policy_recipe(env.spec, cand[1]))
            self._records[(cand, policy)] = [(e["call"]["name"], _norm(e["call"]["arguments"]), _body(e["text"]))
                                             for e in env.transcript()[1:]]
            env.close()
        return self._records[(cand, policy)]

    def next_call(self, policy: str, statement: str, done: list | None) -> tuple[str, dict]:
        """The call the policy makes after the answered calls ``done`` = [(name, arguments, result text)]; with
        ``done=None`` its last call, the delivery, whatever was measured (for policies that deliver without data)."""
        with self._lock:
            calls = set()
            for cand in self.candidates:
                if self._statement(cand) != _body(statement):
                    continue
                rec = self._record(cand, policy)
                if done is None:
                    calls.add(json.dumps(rec[-1][:2], sort_keys=True))
                elif len(rec) > len(done) and all(r[0] == d[0] and r[1] == d[1] and r[2] == _body(d[2]) for r, d in zip(rec, done)):
                    calls.add(json.dumps(rec[len(done)][:2], sort_keys=True))
                elif len(rec) == len(done) and all(r[0] == d[0] and r[1] == d[1] and r[2] == _body(d[2]) for r, d in zip(rec, done)):
                    calls.add(json.dumps([None, None]))                  # the policy's episode ended with a refused delivery
        if len(calls) != 1:
            raise BadRequest(f"stub: {len(calls)} of the {len(self.candidates)} candidate instances continue this conversation "
                             f"as policy {policy!r} would; start the stub with the seeds of the run")
        name, arguments = json.loads(calls.pop())
        return name, arguments


TWINS: Twins | None = None
STATE, STATE_LOCK = {}, threading.Lock()


# ------------------------------------------------------------------ behaviours
def conversation_calls(msgs: list, skip_turns: int = 0) -> list:
    """[(name, arguments, result text)] of every answered tool call, in order, without the first ``skip_turns``
    assistant turns and without calls to ``calc`` (a tool of the session, not of the environment)."""
    results = {m["tool_call_id"]: m["content"] for m in msgs if m["role"] == "tool"}
    out, turn = [], 0
    for m in msgs:
        if m["role"] != "assistant":
            continue
        turn += 1
        if turn <= skip_turns:
            continue
        for tc in m.get("tool_calls") or []:
            if tc["function"]["name"] != "calc":
                out.append((tc["function"]["name"], json.loads(tc["function"]["arguments"]), results[tc["id"]]))
    return out


def _statement(msgs: list) -> str:
    return next(m["content"] for m in msgs if m["role"] == "user")


def _fill(schema: dict):
    """The smallest value that fits a tool schema: required fields only, first enum entry, 1 for numbers."""
    if "anyOf" in schema:
        return _fill(schema["anyOf"][0])
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        return {k: _fill(schema.get("properties", {}).get(k, {})) for k in schema.get("required", [])}
    return {"array": [], "string": "x", "boolean": False, "integer": 1}.get(t, 1.0)


def _terminal_schema(tools: list) -> tuple[str, dict]:
    """Name and parameters of the delivering tool (``submit``, or the last tool of the environment)."""
    fns = [t["function"] for t in tools or [] if t["function"]["name"] != "calc"]
    fn = next((f for f in fns if f["name"] == "submit"), fns[-1] if fns else {"name": "submit", "parameters": {}})
    return fn["name"], fn.get("parameters") or {}


def _call(name: str, arguments) -> dict:
    return {"name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}


def behave(kind: str, body: dict) -> tuple[str, list]:
    """(text, tool calls) of the next assistant turn."""
    msgs = body["messages"]
    turn = sum(m["role"] == "assistant" for m in msgs) + 1
    protocol = {"sequence": "spgr", "tr_ms": 20, "flip_deg": 10}
    if kind == "malformed_json":
        return "", [_call("scan", '{"protocol": {"sequence": "spgr", "tr_ms": 20, "flip_deg": ')]
    if kind == "unknown_tool":
        return "Let me run some code.", [_call("execute_bash", {"cmd": "ls"})]
    if kind == "text_only":
        return "The answer is: phase-encode along y, field of view 240 mm; T1 is about 900 ms in every region.", []
    if kind == "empty":
        return "", []
    if kind == "flood":
        return "", [_call("scanner_info", {"topic": "overview"}) for _ in range(30)]
    if kind == "bad_args":
        name, schema = _terminal_schema(body.get("tools"))
        seq = [_call("scan", {"protocol": "spgr 20 ms"}), _call("plan", {}), _call("scan", {"protocol": protocol, "regions": {"R1": 1}}),
               _call("read_regions", {"scan": 7, "regions": []}), _call("calc", {"expressions": "__import__('os').system('id')"}),
               _call("view", {"scan": "zero"}), _call("scan", '["spgr"]'), _call("scan", {"protocol": dict(protocol, banana=1)}),
               _call(name, {"T1": [1000, 1200]}), _call(name, _fill(schema))]
        return "", [seq[min(turn - 1, len(seq) - 1)]]
    if TWINS is None:
        raise BadRequest("stub: no candidate instances; start the stub with seeds")
    if kind == "recover":
        if turn == 1:
            return "Scanning.", [_call("scan", '{"protocol": {"sequence": "spgr", "tr_ms": 20, ')]
        if turn == 2:
            return "Fitting.", [_call("run_python", {"code": "import numpy"})]
        n, a = TWINS.next_call("reference", _statement(msgs), conversation_calls(msgs, skip_turns=2))
        return f"Step {turn}: {n}.", [_call(n, a)]
    if kind == "overspend":
        done = conversation_calls(msgs)
        if any(r.startswith("error budget") for _, _, r in done):
            n, a = TWINS.next_call("prior", _statement(msgs), None)
            return "The budget is spent.", [_call(n, a)]
        if not done:
            return "", [_call("scanner_info", {})]
        return "", [_call("scan", OVERSPEND_SCAN)]
    n, a = TWINS.next_call(kind, _statement(msgs), conversation_calls(msgs))
    if n is None:
        return "The policy has nothing further to send.", []
    return f"Step {turn}: {n}.", [_call(n, a)]


def respond(body: dict) -> tuple[int, object]:
    """(HTTP status, payload) for a validated request."""
    model = body.get("model")
    if not isinstance(model, str) or not model.startswith("stub:"):
        raise BadRequest(f"The model `{model}` does not exist")
    kind, *flags = model[5:].split("+")
    if set(flags) - set(FLAGS):
        raise BadRequest(f"The model `{model}` does not exist (unknown flag)")
    if "dead" in flags:
        return 500, {"error": {"message": "upstream unavailable", "type": "server_error"}}
    if "noparams" in flags:
        for p in ("temperature", "max_tokens"):
            if p in body:
                return 400, {"error": {"message": f"Unsupported parameter: '{p}' is not supported with this model."
                                       + (" Use 'max_completion_tokens' instead." if p == "max_tokens" else ""),
                                       "type": "invalid_request_error", "param": p}}
    if "flaky" in flags:
        key = hashlib.sha256(json.dumps([model, body["messages"][:2]]).encode()).hexdigest()
        with STATE_LOCK:
            n = STATE[key] = STATE.get(key, 0) + 1
        if n == 1:
            return 500, {"error": {"message": "upstream timeout", "type": "server_error"}}
        if n == 4:
            return 200, "<html><body>502 Bad Gateway</body></html>"
    text, calls = behave(kind, body)
    tcs = [{"id": "call_" + hashlib.md5((c["name"] + c["arguments"] + str(i) + str(len(body["messages"]))).encode()).hexdigest()[:20],
            "type": "function", "function": c} for i, c in enumerate(calls)]
    if "split" in flags and tcs:
        choices = [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text or "Calling a tool."}},
                   {"index": 1, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": tcs}}]
    else:
        msg = {"role": "assistant", "content": text or None}
        if tcs:
            msg["tool_calls"] = tcs
        choices = [{"index": 0, "finish_reason": "tool_calls" if tcs else "stop", "message": msg}]
    out = {"id": "chatcmpl-stub", "object": "chat.completion", "model": model, "choices": choices}
    if "nousage" not in flags:
        p = math.ceil((len(json.dumps(body["messages"])) + len(json.dumps(body.get("tools") or []))) / 4)
        c = math.ceil((len(text) + sum(len(x["arguments"]) + len(x["name"]) for x in calls)) / 4)
        out["usage"] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
    return 200, out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status, payload):
        body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": f"stub:{k}", "object": "model"}
                                                        for k in ("reference", "prior", "recover", "overspend") + FAILURES]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        try:
            body = json.loads(raw)
            validate(body)
            status, payload = respond(body)
        except BadRequest as e:
            status, payload = 400, {"error": {"message": str(e), "type": "invalid_request_error"}}
        except ValueError as e:
            status, payload = 400, {"error": {"message": f"invalid JSON: {e}", "type": "invalid_request_error"}}
        except Exception as e:  # noqa: BLE001  a fault of the stub is reported as such, not as a model reply
            status, payload = 400, {"error": {"message": f"stub failure: {type(e).__name__}: {e}", "type": "stub_bug"}}
        self._send(status, payload)

    def log_message(self, fmt, *args):
        pass


def start(port: int = 0, host: str = "127.0.0.1", seeds=range(4), env_ids=None, levels=None) -> tuple[ThreadingHTTPServer, str]:
    """Start the stub in a daemon thread on loopback. ``port=0`` takes a free port. Returns ``(server, base_url)``;
    stop it with ``server.shutdown()``. ``seeds`` (and optionally ``env_ids``, ``levels``) are the candidate
    instances of the policy replays; give the seeds of the run."""
    global TWINS
    TWINS = Twins(seeds, env_ids, levels)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return httpd, f"http://{host}:{httpd.server_address[1]}/v1"


def main(argv=None) -> int:
    from .runner import parse_seeds
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", type=int, default=4199)
    ap.add_argument("--seeds", default="0-3", help="public train seeds of the candidate instances (default 0-3)")
    a = ap.parse_args(argv)
    _, url = start(a.port, seeds=parse_seeds(a.seeds))
    print(f"stub endpoint at {url}; models: stub:<behaviour>[+flags]; loopback only; Ctrl-C to stop", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
