"""Episode bundles: one JSON file that holds everything a figure, a viewer or a reader needs to replay an episode.

    import mrgym
    from mrgym import record
    bundle = record.record("Scanner/NoWrap-v1", policy="reference", level=2, seed=0)
    record.save(bundle, "runs/episode/episode_bundle.json")

    mrgym record Scanner/NoWrap-v1 --policy reference --level 2 --seed 0 --out runs/episode

A bundle is built after an episode has ended, from the environment object that played it (``bundle_from_env``). The
same function serves a scripted policy and a language model: a model runner passes its messages as ``agent_events``.

Layout (``"format": "mrgym-episode-bundle/1"``)
    package        name, version, scanner engine id
    env            id, level, kind, observation mode, split, limits
    episode        episode id, task id, how it ended
    agent          {"kind": "scripted" | "model" | ..., "name": ...}: who played
    task           the public task card, budget, request limit, the tools
    steps          every step: the call, the console's reply as text and as data, error flag, status counters
    scans          every stored scan: shape, field of view, realised protocol, magnitude and phase as base64 float32
    scanner_log    the scanner's hash-chained log
    result         reward, the answer, the verifier's score and gates, the replay report, the delivered geometry
    agent_events   optional: model messages and tool-call text, each {"step", "role", "content", ...}
    hidden         what the agent never saw: maps of the world, system state, true extent, region truth, and a
                   noise-free re-scan of the delivered protocol where the deliverable is a protocol

Everything outside ``hidden`` and ``result`` is what crossed the tool boundary. The world seed is never written: a
bundle shows a world, it does not let anyone rebuild it.

Episode records (``"schema": "mrgym-episode/2"``), second half of this module. ``mrgym run-agent`` writes one small
``record.json`` per episode and, for a model, the whole conversation next to it (``messages.json``): the prompt, every
model turn with its reasoning text when the endpoint returns it, every tool call and every tool result, the token
usage and latency of every request. ``finish_episode`` adds to the record what makes a result checkable later: the
harness id and configuration, the sampling settings that were sent, the delivered answer, the outcome, the hash of
the conversation file, the command and the analysis plan of the run. ``missing`` names what a record lacks for a
published row; ``upgrade`` reads the records of schema 1, which carry none of this and are pilot data.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
import os
import urllib.parse

import numpy as np

from . import keys
from .canary import CANARY
from .registry import Error

FORMAT = "mrgym-episode-bundle/1"
FILE_NAME = "episode_bundle.json"
MAP_SIZE = 128                      # hidden maps are block means on at most this many points per side (limit 160)
MAX_INLINE_VALUES = 4096            # an array in an observation's data above this size is replaced by its shape
HIDDEN_NOTE = ("Scanner and verifier side only. Nothing under this key was available to the agent during the "
               "episode: the maps are block means of the master grid of the hidden world.")
MAP_INFO = dict(
    pd=dict(title="proton density of the object", unit="a.u."),
    t1_ms=dict(title="T1", unit="ms"),
    t2_ms=dict(title="T2", unit="ms"),
    b1=dict(title="B1+ (actual / nominal flip angle, with the transmit-gain error)", unit="ratio"),
    b0_hz=dict(title="off-resonance (field map plus centre-frequency offset)", unit="Hz"),
)


# ------------------------------------------------------------------------------------------------ encoding
def encode_f32(a) -> str:
    """Base64 of an array as little-endian float32 in C order."""
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def decode_f32(text: str, shape) -> np.ndarray:
    """Inverse of ``encode_f32``: a float32 array of the given shape."""
    return np.frombuffer(base64.b64decode(text), dtype="<f4").reshape(tuple(shape)).astype(np.float32)


def scan_arrays(bundle: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
    """(magnitude, phase) of stored scan ``index``, each float32 [n_images, ny, nx]."""
    for s in bundle["scans"]:
        if s["index"] == index:
            return decode_f32(s["magnitude_f32_b64"], s["shape"]), decode_f32(s["phase_f32_b64"], s["shape"])
    raise KeyError(f"the bundle holds no scan {index}; stored scans: {[s['index'] for s in bundle['scans']]}")


def hidden_map(bundle: dict, name: str) -> np.ndarray:
    """One hidden map (``pd``, ``t1_ms``, ``t2_ms``, ``b1``, ``b0_hz``) as float32 [n, n]; NaN outside the object."""
    m = bundle["hidden"]["maps"]
    return decode_f32(m[name + "_f32_b64"], m["shape"])


def _plain(obj):
    """JSON-safe copy: numpy scalars to Python, tuples to lists, non-finite numbers to None, big arrays and bytes
    replaced by a short description. Keys named ``seed`` are dropped."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items() if k != "seed"}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > MAX_INLINE_VALUES or np.iscomplexobj(obj):
            return {"omitted": "array", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        return _plain(obj.tolist())
    if isinstance(obj, (bytes, bytearray)):
        return {"omitted": "bytes", "length": len(obj)}
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _block_mean(a: np.ndarray, m: int) -> np.ndarray:
    """Mean over ``m`` x ``m`` blocks of a square array (NaN where a block holds no finite value)."""
    n = a.shape[0]
    edges = np.linspace(0, n, m + 1).astype(int)
    ok = np.isfinite(a)
    num = np.add.reduceat(np.add.reduceat(np.where(ok, a, 0.0), edges[:-1], 0), edges[:-1], 1)
    cnt = np.add.reduceat(np.add.reduceat(ok.astype(float), edges[:-1], 0), edges[:-1], 1)
    return np.where(cnt > 0, num / np.maximum(cnt, 1.0), np.nan)


# ------------------------------------------------------------------------------------------------ the hidden block
def hidden_maps(world: dict, size: int = MAP_SIZE) -> dict:
    """Downsampled maps of a world: proton density of the object, T1, T2, B1+, off-resonance. T1, T2 and the fields
    are NaN outside the object."""
    size = int(min(size, 160, world["n"]))
    w = world["weights"].astype(np.float64)
    total = w.sum(0)
    inside = total > 0.05
    par = lambda key: np.array([world["tissues"][k][key] for k in world["names"]])  # noqa: E731
    mean = lambda key: np.where(inside, np.tensordot(par(key), w, 1) / np.maximum(total, 1e-6), np.nan)  # noqa: E731
    layers = dict(
        pd=np.tensordot(par("pd"), w, 1),
        t1_ms=mean("t1_ms"), t2_ms=mean("t2_ms"),
        b1=np.where(inside, world["fields"]["b1"] * world["system"]["tx_gain"], np.nan),
        b0_hz=np.where(inside, world["fields"]["b0_hz"] + world["system"]["cf_offset_hz"], np.nan),
    )
    out = dict(shape=[size, size], extent_mm=float(world["extent_mm"]), dtype="<f4",
               geometry="pixel i of an n-point axis is centred near (i + 0.5 - n / 2) * extent_mm / n; row = y, column = x",
               layers={k: MAP_INFO[k] for k in layers})
    for k, a in layers.items():
        out[k + "_f32_b64"] = encode_f32(_block_mean(a, size))
    return out


def _task_module(env):
    family = getattr(env.spec, "family", None)
    if not family:
        return None
    try:
        return importlib.import_module(family + ".task")
    except ImportError:
        return None


def _delivered(env, answer) -> tuple[dict | None, dict | None]:
    """Where the deliverable is the geometry of a scan: (what the console makes of it, its noise-free re-scan on the
    hidden world). (None, None) for other deliverables and for a geometry the console refuses."""
    from .scanner import console, session
    task = _task_module(env)
    if task is None or not hasattr(task, "protocol_of") or not isinstance(answer, dict):
        return None, None
    world = env._session.world
    try:
        protocol = task.protocol_of(answer)
        pl = console.plan(protocol, world["console"])
    except Exception:  # noqa: BLE001  a malformed answer has no delivered protocol
        return None, None
    if not pl["accept"]:
        return None, None
    rp = pl["realised"]
    fov = [float(v) for v in session.fov_yx(rp)]
    public = dict(protocol=_plain(protocol), realised=_plain(rp), fov_yx_mm=fov, centre_yx_mm=[float(v) for v in session.centre_yx(rp)],
                  rotation_deg=session.rotation_of(rp), scan_s=pl["scan_s"], header=pl["header"])
    img = np.abs(env._session.engine.acquire(rp, None)).astype(np.float32)
    rescan = dict(note="noise-free scan of the delivered protocol on the hidden world, as the verifier sees it",
                  shape=list(img.shape), fov_yx_mm=fov, centre_yx_mm=public["centre_yx_mm"], rotation_deg=public["rotation_deg"],
                  dtype="<f4", magnitude_f32_b64=encode_f32(img))
    return public, rescan


def _hidden(env, map_size: int) -> dict:
    from .scanner import world as W
    world = env._session.world
    truth = W.truth(world)
    ext = W.extent_mm(world)
    return dict(
        note=HIDDEN_NOTE,
        world=dict(kind=world["kind"], field_t=world["b0_t"], master_grid=world["n"], extent_mm=world["extent_mm"]),
        maps=hidden_maps(world, map_size),
        classes=[dict(name=k, **{q: world["tissues"][k].get(q) for q in ("pd", "t1_ms", "t2_ms", "t2s_ms", "fat")})
                 for k in world["names"]],
        system=_plain(world["system"]),
        object_extent_mm=dict(y=[float(v) for v in ext["y"]], x=[float(v) for v in ext["x"]]),
        regions=[dict(_plain(r), **_plain(truth.get(r["id"], {}))) for r in world.get("regions", [])],
    )


# ------------------------------------------------------------------------------------------------ the bundle
def bundle_from_env(env, result, agent: dict, agent_events: list | None = None, map_size: int = MAP_SIZE) -> dict:
    """The bundle of the episode ``env`` has just played.

    env            a ``ScannerEnv`` after its last step (terminated or truncated) and before ``close`` or ``reset``
    result         the last ``StepResult`` (``policies.run`` returns it), or None to read the outcome from ``env``
    agent          who played, for example {"kind": "scripted", "name": "reference"} or {"kind": "model", "name": ...}
    agent_events   model messages and tool-call text in order; each a dict with ``role`` and ``content`` and, where
                   known, ``step``: the number of the step that the message led to
    """
    import mrgym

    ep = getattr(env, "_episode", None)
    if ep is None or getattr(env, "_session", None) is None:
        raise Error("bundle_from_env needs an environment that has played an episode and is not closed")
    if not ep["done"]:
        raise Error("the episode has not ended: a bundle is built after the last step")
    if not isinstance(agent, dict) or "name" not in agent:
        raise Error('agent must be a dict such as {"kind": "scripted", "name": "reference"}')
    cfg, spec = env.episode_config(), env.spec
    specs = env._tool_specs()
    terminal = next((t.name for t in specs if t.terminal), None)

    steps, scan_step = [], {}
    for e in env._log:
        obs = e["observation"]
        data = _plain(obs.data)
        if obs.tool == "scan" and not obs.is_error and isinstance(obs.data.get("scan_index"), int):
            scan_step[obs.data["scan_index"]] = e["step"]
        steps.append(dict(step=e["step"], call=_plain(e["call"]), tool=obs.tool, text=obs.text, data=data,
                          is_error=bool(obs.is_error), status=_plain(obs.status), reward=float(e["reward"]),
                          image_captions=[im.get("caption") for im in obs.images]))

    scans = []
    for k in sorted(env._scanner.scans):
        s = env._scanner.scans[k]
        scans.append(dict(index=int(k), step=scan_step.get(k), shape=list(s["magnitude"].shape),
                          fov_yx_mm=[float(v) for v in s["fov_yx_mm"]], centre_yx_mm=[float(v) for v in s.get("centre_yx_mm", (0.0, 0.0))],
                          rotation_deg=float(s.get("rotation_deg", 0.0)), sequence=s["realised"]["sequence"],
                          realised=_plain(s["realised"]), dtype="<f4",
                          magnitude_f32_b64=encode_f32(s["magnitude"]), phase_f32_b64=encode_f32(s["phase"])))

    info = dict(result.info) if result is not None else dict(ep["info"])
    last = steps[-1]["call"] or {}
    answer = last.get("arguments") if last.get("name") == terminal else None
    delivered, rescan = _delivered(env, answer)
    out_result = dict(
        reward=float(result.reward if result is not None else ep["reward"] or 0.0),
        terminated=bool(ep["done"] and not ep["truncated"]), truncated=bool(ep["truncated"]),
        submitted=bool(info.get("submitted", False)), answer=_plain(answer),
        score=_plain(info.get("score")), replay=_plain(info.get("replay")), budget=_plain(info.get("budget")),
        policy_error=info.get("policy_error"), reason=info.get("reason"), delivered=delivered)

    hidden = _hidden(env, map_size)
    if rescan is not None:
        hidden["rescan"] = rescan

    bundle = dict(
        format=FORMAT, canary=keys.CANARY,               # marks exported task content for people who filter corpora
        package=dict(name="mrgym", version=mrgym.__version__, engine=mrgym.ENGINE_ID),
        env=dict(id=spec.id, level=env.level, kind=spec.kind, obs=env.obs_mode, split=env.split, max_steps=env.max_steps,
                 reveal_score=bool(env.reveal_score), reward_range=list(spec.reward_range)),
        episode=dict(id=ep["id"], task_id=cfg.get("task_id"), steps=ep["steps"]),
        agent=_plain(agent),
        task=dict(card=_plain(cfg.get("task", {})), budget_s=float(cfg["budget_s"]), max_requests=int(cfg["max_requests"]),
                  field_t=float(env._session.exam["b0_t"]), default_fov_mm=float(env._session.exam["fov_mm"]),
                  regions=_plain(cfg.get("task", {}).get("regions", [])),
                  tools=[dict(name=t.name, cost=t.cost, counted=t.counted, terminal=t.terminal) for t in specs]),
        steps=steps, scans=scans, scanner_log=_plain(env.scanner_log()), result=out_result,
        agent_events=_plain(agent_events) if agent_events else None,
        hidden=hidden,
    )
    seed = cfg.get("seed")
    if seed and seed in json.dumps(bundle):
        raise Error("refusing to build the bundle: the world seed occurs in the recorded episode")
    return bundle


def record(env_id: str, policy="reference", level: int | None = None, seed=0, **make_kwargs) -> dict:
    """Play one episode of ``env_id`` with a scripted control policy (a name or a callable) and return its bundle.

    ``make_kwargs`` go to ``mrgym.make``; ``reveal_score`` defaults to True so the last reply carries the gates."""
    import mrgym

    from . import policies
    kw = dict(make_kwargs)
    kw.setdefault("reveal_score", True)
    if level is not None:
        kw["level"] = level
    env = mrgym.make(env_id, seed=seed, **kw)
    controls = policies.controls_of(env)
    if controls is None:
        raise Error(f"{env_id} has no scripted policies to record; play it and call record.bundle_from_env")
    if isinstance(policy, str) and policy not in controls.policies():
        raise Error(f"{env_id} has no policy {policy!r}; available: {sorted(controls.policies())}")
    name = policy if isinstance(policy, str) else getattr(policy, "__name__", "policy")
    try:
        result = policies.run(env, policy)
        agent = dict(kind="scripted", name=name, description=getattr(controls, "DESCRIPTIONS", {}).get(name),
                     language_model=False)
        return bundle_from_env(env, result, agent)
    finally:
        env.close()


def save(bundle: dict, path: str) -> str:
    """Write a bundle as compact JSON (``path`` may be a folder: the file is then ``episode_bundle.json``)."""
    if os.path.isdir(path) or not path.endswith(".json"):
        path = os.path.join(path, FILE_NAME)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    return path


def load(path: str) -> dict:
    """Read a bundle from a file or from the folder that holds ``episode_bundle.json``."""
    if os.path.isdir(path):
        path = os.path.join(path, FILE_NAME)
    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)
    if not isinstance(bundle, dict) or not str(bundle.get("format", "")).startswith("mrgym-episode-bundle/"):
        raise Error(f"{path} is not an episode bundle")
    if bundle["format"] != FORMAT:
        raise Error(f"{path} has format {bundle['format']}; this package reads {FORMAT}")
    return bundle


def summary(bundle: dict) -> str:
    """One line: who played what, and how it ended."""
    r, b = bundle["result"], bundle["result"].get("budget") or {}
    score = r.get("score") or {}
    verdict = "passed" if score.get("passed") else "truncated" if r["truncated"] else \
        "no pass / fail (sandbox)" if score.get("passed") is None else "not passed"
    return (f"{bundle['env']['id']} · level {bundle['env']['level']} · {bundle['agent'].get('kind', 'agent')} "
            f"{bundle['agent']['name']} · {len(bundle['steps']) - 1} tool calls · {len(bundle['scans'])} scan(s), "
            f"{b.get('used_s', 0.0):.1f} of {bundle['task']['budget_s']:.0f} s of scan time · reward {r['reward']:.3f} · {verdict}")


# ------------------------------------------------------------------------------------------------ episode records
EPISODE_SCHEMA = "mrgym-episode/2"
PILOT_SCHEMA = "mrgym-episode/1"            # records without transcript, harness id or deliverable: pilot data
TRANSCRIPT_SCHEMA = "mrgym-transcript/1"
TRANSCRIPT_FILE = "messages.json"
LOG_FILE = "scanner_log.jsonl"
# What a completed record must carry before its row may be published (``missing``).
REQUIRED = ("harness_id", "harness", "outcome", "ladder", "distance", "group", "wall_s", "command", "engine",
            "package_version", "utc", "canary")
REQUIRED_MODEL = ("model", "provider", "max_turns", "sampler", "tokens_in", "tokens_out", "transcript")
REQUIRED_KEYS = ("deliverable", "adapter_sha256", "plan_sha256")          # the key must exist; None is a value


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reasoning_of(raw: dict) -> str:
    """The reasoning text of a chat-completions response, when the endpoint returns it: ``reasoning_content``
    (vLLM, DeepSeek, Qwen), ``reasoning`` (OpenRouter), ``thinking``, or content parts of type ``thinking`` /
    ``reasoning``. Empty when the endpoint keeps it."""
    out = []
    for ch in (raw or {}).get("choices") or []:
        msg = ch.get("message") if isinstance(ch, dict) else None
        if not isinstance(msg, dict):
            continue
        for key in ("reasoning_content", "reasoning", "thinking"):
            v = msg.get(key)
            if isinstance(v, dict):
                v = v.get("content") or v.get("text") or v.get("summary")
            if isinstance(v, str) and v.strip() and v not in out:
                out.append(v)
        for part in msg.get("content") if isinstance(msg.get("content"), list) else []:
            if isinstance(part, dict) and part.get("type") in ("thinking", "reasoning"):
                v = part.get("thinking") or part.get("text")
                if isinstance(v, str) and v.strip() and v not in out:
                    out.append(v)
    return "\n".join(out)


class Recorder:
    """A chat client that keeps what the endpoint returned for every request: reasoning text, the raw text and tool
    calls, the finish reason, the usage block, latency and attempts. Everything else is the wrapped client."""

    def __init__(self, client):
        self._client, self.turns = client, []

    def complete(self, messages, tools):
        resp = self._client.complete(messages, tools)
        usage = resp.get("usage") or {}
        self.turns.append(dict(
            turn=len(self.turns) + 1, reasoning=reasoning_of(resp.get("raw")), text=resp.get("text"),
            tool_calls=[dict(id=c.get("id"), name=c.get("name"), arguments=c.get("arguments")) for c in resp.get("tool_calls") or []],
            finish=resp.get("finish"), usage=usage, model=(resp.get("raw") or {}).get("model"),
            sent_params=resp.get("sent_params"), attempts=resp.get("attempts"), seconds=resp.get("seconds")))
        return resp

    def __getattr__(self, name):
        return getattr(self._client, name)


def harness_of(job: dict, env, sink: dict | None) -> dict:
    """Everything of the harness that can move a result, without the endpoint and without the sampling settings (they
    are in ``sampler``): the kind of agent loop, its settings (turn cap, calls per turn, the session's own tool), the
    hash of the system note as sent, the tool list and the hash of the tool schemas. ``harness_id`` is the hash of
    this dict, so two runs with one harness id differ at most in the model and its sampling."""
    if job["kind"] == "scripted":
        return dict(kind="scripted", policy=job.get("policy"), recipe="tuned" if job.get("recipe") else "default",
                    max_steps=env.max_steps)
    tools = (sink or {}).get("tools") or []
    messages = (sink or {}).get("messages") or []
    system = next((m.get("content") for m in messages if m.get("role") == "system"), None)
    settings = {k: v for k, v in job["opts"]["settings"].items() if k not in ("temperature", "max_tokens")}     # sampling is not harness
    return dict(kind="chat-tools", loop="mrgym.agents.runner.play_model", settings=settings,
                system_sha256=None if system is None else hashlib.sha256(str(system).encode()).hexdigest(),
                tools=[t.get("function", {}).get("name") for t in tools], tools_sha256=_sha(tools), max_steps=env.max_steps)


def harness_id(harness: dict) -> str:
    return _sha(harness)[:16]


def deliverable_of(env) -> dict | None:
    """The answer the episode delivered: the arguments of the accepted call of the terminal tool, or None when the
    episode ended without one (turn cap, truncation, a refused deliverable)."""
    ep = getattr(env, "_episode", None)
    if not ep or not ep.get("done") or ep.get("truncated"):
        return None
    terminal = next((t.name for t in env._tool_specs() if t.terminal), None)
    for e in reversed(env._log):
        call = e["call"] or {}
        if call.get("name") == terminal and not e["observation"].is_error:
            return _plain(call.get("arguments"))
    return None


def outcome_of(rec: dict, info: dict) -> tuple[str, dict, dict]:
    """(outcome, ladder, distance) of a completed episode. The environment's own values are used when ``info``
    carries them; otherwise they are read from the record: ``pass``, ``fail:<first failed check>``,
    ``no_delivery:<stop reason>``. Ladder flags form a prefix: each implies the ones before it."""
    checks = rec.get("checks") or rec.get("gates") or {}
    submitted = rec.get("deliverable") is not None
    valid = submitted and bool(checks.get("deliverable_valid", checks.get("deliverable", True)))
    measured = valid and bool(checks.get("measured", (rec.get("scans") or 0) > 0))
    ladder = dict(submitted=submitted, valid=valid, measured=measured, passed=measured and bool(rec.get("passed")))
    if rec.get("passed"):
        outcome = "pass"
    elif not submitted:
        outcome = "no_delivery:" + str(rec.get("stop_reason"))
    else:
        outcome = "fail:" + next((k for k, ok in checks.items() if not ok), "unknown")
    return info.get("outcome") or outcome, info.get("ladder") or ladder, info.get("distance") or dict(error=rec.get("error"))


def _run_folder(directory: str) -> str | None:
    """The run folder of an episode folder ``<out>/episodes/<agent>/<env>/L<level>/s<seed>/e<k>``."""
    d = os.path.abspath(directory)
    for _ in range(8):
        parent = os.path.dirname(d)
        if os.path.basename(d) == "episodes":
            return parent
        d = parent
    return None


def _run_provenance(directory: str) -> tuple[str | None, str | None]:
    """(command of the invocation that plays this episode, sha256 of the analysis plan of the run folder or None)."""
    out = _run_folder(directory)
    command = plan = None
    if out:
        try:
            with open(os.path.join(out, "run_config.json"), encoding="utf-8") as f:
                command = (json.load(f).get("invocations") or [{}])[-1].get("command")
        except (OSError, ValueError):
            pass
        path = os.path.join(out, "analysis_plan.json")
        if os.path.exists(path):
            plan = _file_sha(path)
    return command, plan


def transcript_of(rec: dict, client, sink: dict, hid: str) -> dict:
    """The stored conversation: the tools and every message as sent to the endpoint, each assistant message with the
    reasoning text of its turn, and one entry per request with usage and latency."""
    messages = copy.deepcopy(sink.get("messages") or [])
    turns = list(getattr(client, "turns", []) or [])
    assistants = [m for m in messages if m.get("role") == "assistant"]
    for m, t in zip(assistants, turns):
        m["turn"] = t["turn"]
        if t["reasoning"]:
            m["reasoning"] = t["reasoning"]
    return dict(schema=TRANSCRIPT_SCHEMA, canary=CANARY, harness_id=hid, agent=rec.get("agent"), model=rec.get("model"),
                env_id=rec.get("env_id"), level=rec.get("level"), split=rec.get("split"), task_id=rec.get("task_id"),
                complete=len(assistants) == len(turns), tools=sink.get("tools") or [], messages=messages, turns=turns)


def _sum_usage(turns: list, *path) -> int | None:
    total, seen = 0, False
    for t in turns:
        v = t.get("usage") or {}
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
        if isinstance(v, int):
            total, seen = total + v, True
    return total if seen else None


def finish_episode(rec: dict, job: dict, env, final, client, sink: dict | None, directory: str) -> dict:
    """Complete the record of an episode that was played to its end (schema 2) and write the conversation and the
    scanner log next to it. Called once by the runner, before the environment is closed. Returns ``rec``.

    Added: ``harness`` and ``harness_id``; ``provider``; ``sampler`` (requested, sent, dropped) and ``max_turns``;
    ``group``; ``deliverable``, ``checks``, ``outcome``, ``ladder``, ``distance``; ``adapter_sha256``;
    ``tokens_reasoning``, ``tokens_cached``, ``model_s``, ``requests``; ``transcript`` (file, sha256, counts) for a
    model; ``command`` and ``plan_sha256`` of the run; ``canary``."""
    info = dict(final.info) if final is not None else {}
    scripted = job["kind"] == "scripted"
    opts = job.get("opts") or {}
    settings = opts.get("settings") or {}
    harness = harness_of(job, env, sink)
    hid = harness_id(harness)
    command, plan = _run_provenance(directory)
    rec.update(schema=EPISODE_SCHEMA, harness_id=hid, harness=harness, canary=CANARY, command=command, plan_sha256=plan,
               group=dict(env_id=rec.get("env_id"), level=rec.get("level"), split=rec.get("split"), task_id=rec.get("task_id")),
               max_turns=None if scripted else (settings.get("max_turns") or env.max_steps), max_steps=env.max_steps,
               adapter_sha256=opts.get("adapter_sha256"),
               provider=None if scripted else urllib.parse.urlsplit(opts.get("base_url") or "").netloc or None,
               sampler=None if scripted else dict(requested=dict(temperature=settings.get("temperature"), max_tokens=settings.get("max_tokens")),
                                                  sent=dict(rec.get("params_sent") or {}), dropped=list(rec.get("params_dropped") or [])))
    rec["deliverable"] = deliverable_of(env)
    if rec.get("status") == "completed":
        score = info.get("score") or {}
        if isinstance(score.get("checks"), dict):
            rec["checks"] = {k: bool(v) for k, v in score["checks"].items()}
        rec["outcome"], rec["ladder"], rec["distance"] = outcome_of(rec, info)
    log_path = os.path.join(directory, LOG_FILE)
    if not os.path.exists(log_path) and not (scripted and opts.get("records_only")):       # a model episode always keeps its log
        try:
            events = env.scanner_log()
            with open(log_path, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001  an environment without a scanner
            pass
    rec["transcript"] = None
    if not scripted and sink:
        turns = list(getattr(client, "turns", []) or [])
        rec.update(tokens_reasoning=_sum_usage(turns, "completion_tokens_details", "reasoning_tokens"),
                   tokens_cached=_sum_usage(turns, "prompt_tokens_details", "cached_tokens"),
                   model_s=round(sum(t.get("seconds") or 0.0 for t in turns), 3), requests=sum(t.get("attempts") or 0 for t in turns))
        doc = transcript_of(rec, client, sink, hid)
        text = json.dumps(doc, indent=1) + "\n"
        seed = env.episode_config().get("seed") if callable(getattr(env, "episode_config", None)) else None
        if seed and seed in text:                                        # a conversation with the seed is not kept
            rec["transcript_withheld"] = "the world seed occurs in the conversation"
        else:
            path = os.path.join(directory, TRANSCRIPT_FILE)
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(path + ".tmp", path)
            rec["transcript"] = dict(
                file=TRANSCRIPT_FILE, schema=TRANSCRIPT_SCHEMA, sha256=hashlib.sha256(text.encode()).hexdigest(), bytes=len(text.encode()),
                complete=doc["complete"], messages=len(doc["messages"]), assistant_turns=sum(m.get("role") == "assistant" for m in doc["messages"]),
                tool_results=sum(m.get("role") == "tool" for m in doc["messages"]),
                reasoning_chars=sum(len(t.get("reasoning") or "") for t in turns))
    return rec


def upgrade(rec: dict) -> dict:
    """A record of any schema in the shape the results code reads. Schema 1 (no transcript, no harness id, no
    deliverable) is marked ``tier = "pilot"``: it is kept and shown, never published."""
    rec = dict(rec)
    schema = rec.get("schema")
    rec["record_schema"] = 2 if schema == EPISODE_SCHEMA else 1
    if rec["record_schema"] == 1:
        rec["tier"] = "pilot"
    return rec


def missing(rec: dict) -> list[str]:
    """Fields a completed record lacks for a published row (empty: nothing is missing). A schema-1 record lacks all of
    them. For a model: the conversation, the token counts, the provider, the turn cap and the sampling settings."""
    if rec.get("status") != "completed":
        return []
    out = [k for k in REQUIRED if rec.get(k) is None] + [k for k in REQUIRED_KEYS if k not in rec]
    if rec.get("agent_kind") != "scripted":
        out += [k for k in REQUIRED_MODEL if rec.get(k) is None]
        t = rec.get("transcript") or {}
        if rec.get("transcript") is not None and not (t.get("sha256") and t.get("complete") and t.get("assistant_turns") == rec.get("turns")):
            out.append("transcript (incomplete)")
    if rec.get("stop_reason") == "submitted" and rec.get("agent_kind") != "scripted" and rec.get("deliverable") is None and "deliverable" in rec:
        out.append("deliverable (submitted, none stored)")
    return out


def transcript_problem(rec: dict, directory: str) -> str | None:
    """Why the stored conversation of a record cannot be trusted, or None: the file exists, its hash is the one in the
    record, and every tool message answers a tool call of an earlier assistant message."""
    t = rec.get("transcript")
    if not t:
        return "no transcript in the record"
    path = os.path.join(directory, t.get("file") or TRANSCRIPT_FILE)
    if not os.path.exists(path):
        return f"{t.get('file')} is missing"
    if _file_sha(path) != t.get("sha256"):
        return f"{t.get('file')} does not have the sha256 stored in the record"
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    called = set()
    for m in doc.get("messages") or []:
        if m.get("role") == "assistant":
            called.update(c.get("id") for c in m.get("tool_calls") or [])
        elif m.get("role") == "tool" and m.get("tool_call_id") not in called:
            return f"a tool message answers no call ({m.get('tool_call_id')})"
    return None


# ------------------------------------------------------------------------------------------------ command line
def cmd_record(a) -> int:
    import mrgym
    level = a.level if a.level is not None else mrgym.spec(a.id).default_level
    bundle = record(a.id, policy=a.policy, level=level, seed=a.seed)
    out = a.out or os.path.join("runs", f"{a.id.split('/')[-1]}_level{level}_{a.policy}")
    path = save(bundle, os.path.join(out, FILE_NAME))
    print(path)
    print(summary(bundle))
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("record", help="play a scripted policy and save the episode as one bundle file")
    p.add_argument("id", help="environment id, for example Scanner/NoWrap-v1")
    p.add_argument("--policy", default="reference", help="name of one of the environment's scripted policies")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--seed", default="0")
    p.add_argument("--out", default=None, help="folder for episode_bundle.json (default runs/<name>_level<L>_<policy>)")
    p.set_defaults(fn=cmd_record)
