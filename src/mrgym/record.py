"""Episode bundles: one JSON file that holds everything a figure, a viewer or a reader needs to replay an episode.

    import mrgym
    from mrgym import record
    bundle = record.record("Scanner/NoWrap-v0", policy="reference", level=2, seed=0)
    record.save(bundle, "runs/episode/episode_bundle.json")

    mrgym record Scanner/NoWrap-v0 --policy reference --level 2 --seed 0 --out runs/episode

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
"""
from __future__ import annotations

import base64
import importlib
import json
import os

import numpy as np

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
    public = dict(protocol=_plain(protocol), realised=_plain(rp), fov_yx_mm=fov, scan_s=pl["scan_s"], header=pl["header"])
    img = np.abs(env._session.engine.acquire(rp, None)).astype(np.float32)
    rescan = dict(note="noise-free scan of the delivered protocol on the hidden world, as the verifier sees it",
                  shape=list(img.shape), fov_yx_mm=fov, dtype="<f4", magnitude_f32_b64=encode_f32(img))
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
                          fov_yx_mm=[float(v) for v in s["fov_yx_mm"]], sequence=s["realised"]["sequence"],
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
        format=FORMAT,
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
    p.add_argument("id", help="environment id, for example Scanner/NoWrap-v0")
    p.add_argument("--policy", default="reference", help="name of one of the environment's scripted policies")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--seed", default="0")
    p.add_argument("--out", default=None, help="folder for episode_bundle.json (default runs/<name>_level<L>_<policy>)")
    p.set_defaults(fn=cmd_record)
