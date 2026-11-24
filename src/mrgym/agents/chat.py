"""OpenAI-compatible chat-completions client with tools, standard library only (``urllib``).

Written for endpoints that front several model vendors. Their habits, and what this client does about each:

* The tool call may sit in another ``choices[]`` element than the text. ``merge_choices`` reads every choice.
* Some models reject fields they do not know. A request holds ``model``, ``messages``, ``tools`` and nothing else
  unless the caller asks for a sampling parameter. A parameter the model rejects (HTTP 400 naming it) is dropped and
  the request repeated; what was finally sent and what was dropped is reported.
* Messages sent back hold only the standard keys, and the arguments of an echoed tool call are always a JSON object
  in a string: an endpoint that converts to another vendor's format parses them and fails on anything else.
* 408 / 429 / 5xx, timeouts and bodies that are not JSON are retried with exponential back-off. When the retries are
  used up the call raises ``GatewayError``: an endpoint failure, never an agent failure.
* A response without a ``usage`` block is accepted; the caller estimates tokens and flags the estimate.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})


class GatewayError(Exception):
    """The endpoint could not be used (after retries). Not an agent failure."""

    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status, self.body = status, body


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):                       # content parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def merge_choices(response: dict) -> tuple[str, list[dict], list]:
    """Text and tool calls from all choices of a response.

    Returns ``(text, tool_calls, finish_reasons)``; a tool call is ``{"id", "name", "arguments"}`` with the arguments
    as the raw string the model wrote. The legacy ``function_call`` field is accepted, the same call repeated in two
    choices is kept once, and missing or duplicate ids are replaced by unique ones.
    """
    texts, calls, finish, seen = [], [], [], set()
    for ch in response.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        finish.append(ch.get("finish_reason"))
        msg = ch.get("message") or {}
        t = _text_of(msg.get("content"))
        if t.strip() and t not in texts:
            texts.append(t)
        raw = list(msg.get("tool_calls") or [])
        if msg.get("function_call"):
            raw.append({"function": msg["function_call"]})
        for tc in raw:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args) if args is not None else ""
            key = (tc.get("id"), fn.get("name"), args)
            if key in seen:
                continue
            seen.add(key)
            calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})
    used = set()
    for i, c in enumerate(calls):
        if not isinstance(c["id"], str) or not c["id"] or c["id"] in used:
            c["id"] = f"call_{i}_{time.time_ns() % 10 ** 10}"
        used.add(c["id"])
    return "\n".join(texts), calls, finish


def assistant_message(text: str, calls: list[dict]) -> dict:
    """The assistant turn as it is sent back: standard keys only, arguments always a JSON object in a string.

    Arguments that are not a JSON object are echoed as ``{}``; the model reads what was wrong with them in the tool
    message that answers the call.
    """
    msg = {"role": "assistant", "content": text if text else None}
    if calls:
        out = []
        for c in calls:
            raw = c["arguments"] if isinstance(c["arguments"], str) else ""
            try:
                ok = isinstance(json.loads(raw) if raw.strip() else {}, dict)
            except ValueError:
                ok = False
            name = re.sub(r"[^A-Za-z0-9_-]", "_", str(c["name"] or "unknown"))[:64] or "unknown"
            out.append({"id": c["id"], "type": "function",
                        "function": {"name": name, "arguments": (raw if raw.strip() else "{}") if ok else "{}"}})
        msg["tool_calls"] = out
    elif not text:
        msg["content"] = "(empty response)"
    return msg


def tool_message(call_id: str, text: str) -> dict:
    """A tool result written by the harness itself (the environment writes its own with ``Observation.to_openai``)."""
    return {"role": "tool", "tool_call_id": call_id, "content": text}


class ChatClient:
    """One model behind one endpoint. ``params`` are optional sampling parameters; ``None`` values are not sent."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, params: dict | None = None,
                 timeout: float = 600.0, max_retries: int = 5, backoff_s: float = 2.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key or "none"
        self.params = {k: v for k, v in (params or {}).items() if v is not None}
        self.timeout, self.max_retries, self.backoff_s = timeout, max_retries, backoff_s
        self.dropped: list[str] = []                    # parameters this model refused
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, body: dict):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with self._opener.open(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        """One completion.

        Returns ``dict(text, tool_calls, finish, usage, raw, sent_params, attempts, seconds)``. Raises
        ``GatewayError`` when the endpoint cannot be used after ``max_retries`` retries.
        """
        attempt, t0, last = 0, time.time(), None
        while True:
            body = {"model": self.model, "messages": messages, "tools": tools, **self.params}
            try:
                raw = self._post(body)
                if not isinstance(raw, dict) or not isinstance(raw.get("choices"), list) or not raw["choices"]:
                    raise ValueError("response has no choices: " + json.dumps(raw)[:300])
                text, calls, finish = merge_choices(raw)
                usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
                return dict(text=text, tool_calls=calls, finish=finish, usage=usage, raw=raw,
                            sent_params=dict(self.params), attempts=attempt + 1, seconds=round(time.time() - t0, 3))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:2000]
                if e.code == 400:
                    bad = [p for p in self.params if re.search(rf"\b{re.escape(p)}\b", detail)]
                    if bad:                             # the model refuses a parameter: drop it, or use its new name
                        for p in bad:
                            v = self.params.pop(p)
                            self.dropped.append(p)
                            if p == "max_tokens" and "max_completion_tokens" in detail:
                                self.params["max_completion_tokens"] = v
                        continue
                last = GatewayError(f"HTTP {e.code}: {detail[:300]}", e.code, detail)
                if e.code not in RETRY_STATUS:
                    raise last from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, OSError) as e:
                last = GatewayError(f"{type(e).__name__}: {e}")
            attempt += 1
            if attempt > self.max_retries:
                raise last
            time.sleep(min(self.backoff_s * 2 ** (attempt - 1), 60.0))
