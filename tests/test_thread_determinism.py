"""An instance and an episode are functions of (id, level, seed) alone, also while other threads build and play.

Harnesses run episodes in threads (``agents/runner.py``: a pool of workers for model episodes; the verl adapter:
``asyncio.to_thread``; the scanner service: one thread per request). Anything a build writes outside its own frame
(a module attribute changed "for the duration of a call", a class attribute, a shared dict or array edited in place,
global numpy state) is then read by the builds next to it. The case that made this file: ``scanner/world.py::_build``
wrote the half-width of the soft boundaries of ONE world into ``anatomy.EDGE_MM``; a level-3 ArtefactClinic build
(the only spec with ``edge_mm``) changed the object of every world built at the same moment, and a build that had
read the changed value "restored" it for the rest of the process.

Every registered scored family is discovered (``golden_cases.task_ids``), none is named here.

    instances   every level of every family, seeds ``threads:<k>`` (string seeds: no stored bank holds them, so
                every instance is generated). Built serially first, then the same set from ``THREADS`` threads in
                shuffled order, ``ROUNDS`` times, the families' caches cleared before each pass. The JSON of every
                episode config (public task, hidden truth, world spec, budget) must equal the serial one.
    episodes    one reference-policy episode per family at its highest level: task text, every observation, the head
                of the scanner log's hash chain, reward and score dict, serially and then while three other threads
                play the other families (``EPISODE_ROUNDS`` times).
    writes      whether two builds meet inside a window of a few milliseconds is chance, so the construct itself is
                watched too: while the passes run, every assignment to an upper-case attribute of a module of the
                package that changes its value is recorded with the build that made it (``_watch_writes``). No
                family may make one. This part does not depend on the scheduling.
    constants   the repr of every upper-case module attribute of the package after the threads equals the one
                before (an in-place edit of a shared dict or list, a value left behind).

Cost. A generated ArtefactClinic, ProtocolToSpec or PhantomQA instance takes 0.4 to 17 s, so three seeds of every
level, serially and in several rounds, take minutes. By default seed 0 of every level is built, and seeds 1 and 2 only
where seed 0 took less than ``EXTRA_SEED_MAX_S``; two rounds of instances, one of episodes (about 80 s on two cores).
``MRGYM_THREAD_DETERMINISM=full``: three seeds of every level, three rounds of each. The seed label is one whose
searched instances are found within a few worlds; any other label is as valid and only slower.
"""
import concurrent.futures
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import random
import sys
import threading
import time
import types

import pytest

import golden_cases as G
import mrgym
from mrgym import gate, policies
from mrgym.env import derive_seed

IDS = G.task_ids()
FULL = os.environ.get("MRGYM_THREAD_DETERMINISM") == "full"
THREADS = 4
SEEDS = 3
ROUNDS = 3 if FULL else 2
EPISODE_ROUNDS = 3 if FULL else 1
EXTRA_SEED_MAX_S = float("inf") if FULL else 0.5
SWITCH_INTERVAL_S = 1e-4                 # the interpreter changes thread this often during the threaded passes (default 5 ms)

_JOB = threading.local()                 # the build or episode that runs in this thread
_WRITES: list = []                       # (job, "module.NAME", repr before, repr after)
_MISSING = object()


def _seed(k: int) -> str:
    return f"threads:{k}"


def _family_of(job) -> str:
    return job[0] if isinstance(job, tuple) else job


def _package_modules() -> dict:
    return {name: mod for name, mod in sorted(sys.modules.items()) if mod is not None and (name == "mrgym" or name.startswith("mrgym."))}


class _Watched(types.ModuleType):
    """A module whose upper-case attributes report every assignment that changes them (``module.NAME = value`` from
    anywhere; a ``global`` statement inside the module writes its dict directly and is not seen)."""

    def __setattr__(self, name, value):
        if name.isupper():
            old = self.__dict__.get(name, _MISSING)
            if old is _MISSING or (old is not value and repr(old) != repr(value)):
                _WRITES.append((getattr(_JOB, "name", None), f"{self.__name__}.{name}", "<unset>" if old is _MISSING else repr(old), repr(value)))
        super().__setattr__(name, value)


@contextlib.contextmanager
def _watch_writes():
    """Record the changing writes to module constants of the package made inside the block: the list of them."""
    mods = [m for m in _package_modules().values() if type(m) is types.ModuleType]
    start = len(_WRITES)
    out: list = []
    for m in mods:
        m.__class__ = _Watched
    try:
        yield out
    finally:
        for m in mods:
            m.__class__ = types.ModuleType
        out.extend(_WRITES[start:])


def _constants() -> dict:
    """repr of every upper-case attribute (plain data) of every loaded module of the package."""
    plain = (bool, int, float, str, bytes, tuple, frozenset, set, list, dict, type(None))
    return {f"{name}.{k}": repr(v) for name, mod in _package_modules().items() for k, v in sorted(vars(mod).items())
            if k.isupper() and isinstance(v, plain)}


def _left_changed(before: dict, moved: dict, when: str):
    after = _constants()
    moved.update({k: f"{before[k]} -> {after.get(k)} ({when})" for k in before if after.get(k) != before[k] and k not in moved})


def _clear_caches():
    """Empty every memoised function of every family's package: a pass must build, not look up what the pass before built."""
    for env_id in IDS:
        pkg = mrgym.spec(env_id).family
        for name, mod in _package_modules().items():
            if name == pkg or name.startswith(pkg + "."):
                for v in list(vars(mod).values()):
                    if callable(getattr(v, "cache_clear", None)):
                        v.cache_clear()


def _not_stored(env_id: str, level: int, seed: str):
    """A family that keeps a bank of generated instances (``<family>/bank.py``) must not serve this seed from it."""
    pkg = mrgym.spec(env_id).family
    if importlib.util.find_spec(pkg + ".bank") is not None:
        bank = importlib.import_module(pkg + ".bank")
        assert bank.get(derive_seed(env_id, level, "train", seed, environ={}), level) is None, \
            f"{env_id} level {level} seed {seed!r} is a stored instance: this test would compare two look-ups"


def _config(job) -> str:
    env_id, level, k = job
    return G.canon(G.jsonable(gate._instance(mrgym.spec(env_id), level, _seed(k), "train")))


def _episode(env_id: str) -> dict:
    spec = mrgym.spec(env_id)
    env = mrgym.make(env_id, level=max(spec.levels), seed=_seed(0))
    try:
        res = policies.run(env, getattr(policies.controls_of(spec), "REFERENCE", "reference"))
        texts = [e["text"] for e in env.transcript()]
        log = env.scanner_log()
        return dict(task_text=texts[0], texts_sha256=G.sha(G.canon(texts)), chain_head=log[-1]["chain"] if log else None,
                    reward=res.reward, policy_error=res.info.get("policy_error"), score=G.jsonable(res.info.get("score")))
    finally:
        env.close()


def _serial(fn, job):
    _JOB.name = job
    try:
        return fn(job)
    finally:
        _JOB.name = None


def _threaded(fn, jobs, order_seed: int) -> dict:
    """``fn`` on every job from ``THREADS`` threads in a shuffled order: {job: result or the text of what it raised}."""
    def one(job):
        _JOB.name = job
        try:
            return fn(job)
        except Exception as e:  # noqa: BLE001  a build that fails only next to other threads is a finding, not a crash
            return f"raised {type(e).__name__}: {e}"
        finally:
            _JOB.name = None
    order = list(jobs)
    random.Random(order_seed).shuffle(order)
    old = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL_S)
    try:
        with concurrent.futures.ThreadPoolExecutor(THREADS) as pool:
            return dict(zip(order, pool.map(one, order)))
    finally:
        sys.setswitchinterval(old)


def _where(a, b, path="") -> list:
    """Paths at which two JSON values differ (the first few are enough to see what moved)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in (_where(a[k], b[k], f"{path}/{k}") if k in a and k in b else [f"{path}/{k}"])]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _where(x, y, f"{path}/{i}")]
    return [] if a == b else [path or "/"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _passes(fn, serial_jobs, rounds: int, order_seed: int, extra=None) -> dict:
    """The serial pass over ``serial_jobs`` (``extra(job, seconds)`` may name more jobs after each), then ``rounds``
    threaded passes over the same jobs: dict(serial, rounds, writes, constants, seconds)."""
    for env_id in IDS:                                                   # everything a family imports is loaded (and watched) first
        importlib.import_module(mrgym.spec(env_id).family + ".task")
        policies.controls_of(mrgym.spec(env_id))
    _clear_caches()
    t0, serial, moved = time.time(), {}, {}
    with _watch_writes() as writes:
        todo = list(serial_jobs)
        while todo:
            job = todo.pop(0)
            t1 = time.time()
            serial[job] = _serial(fn, job)
            todo = (extra(job, time.time() - t1) if extra else []) + todo
    t_serial, before, out = time.time() - t0, _constants(), []
    for r in range(rounds):
        _clear_caches()
        with _watch_writes() as more:
            out.append(_threaded(fn, serial, order_seed + r))
        writes += more
        _left_changed(before, moved, f"after threaded round {r}")
    return dict(serial=serial, rounds=out, writes=writes, constants=moved, seconds=(round(t_serial, 1), round(time.time() - t0 - t_serial, 1)))


# ---------------------------------------------------------------------------------------------- the passes
@pytest.fixture(scope="module")
def instances():
    def more_seeds(job, seconds):
        env_id, level, k = job
        return [(env_id, level, j) for j in range(1, SEEDS)] if k == 0 and seconds < EXTRA_SEED_MAX_S else []
    first = [(env_id, level, 0) for env_id in IDS for level in mrgym.spec(env_id).levels]
    for env_id, level, _ in first:
        for k in range(SEEDS):
            _not_stored(env_id, level, _seed(k))
    return _passes(_config, first, ROUNDS, 0, more_seeds)


@pytest.fixture(scope="module")
def episodes():
    return _passes(_episode, IDS, EPISODE_ROUNDS, 100)


# ---------------------------------------------------------------------------------------------- instances
def test_every_level_of_every_registered_family_is_built(instances):
    assert IDS, "no scored family is registered"
    want = {(env_id, level) for env_id in IDS for level in mrgym.spec(env_id).levels}
    assert {(e, lv) for e, lv, _ in instances["serial"]} == want
    assert all(set(r) == set(instances["serial"]) for r in instances["rounds"]) and len(instances["rounds"]) >= 2
    if FULL:
        assert len(instances["serial"]) == SEEDS * len(want)
    print(f"\n{len(instances['serial'])} instances of {len(want)} (family, level) pairs, {ROUNDS} rounds in {THREADS} threads; "
          f"serial {instances['seconds'][0]} s, threaded {instances['seconds'][1]} s")


@pytest.mark.parametrize("env_id", IDS)
def test_an_instance_built_next_to_other_threads_is_the_serial_one(instances, env_id):
    bad = []
    for job, want in instances["serial"].items():
        if job[0] != env_id:
            continue
        for r, got in enumerate(instances["rounds"]):
            if got[job] != want:
                what = got[job] if got[job].startswith("raised ") else ", ".join(_where(json.loads(want), json.loads(got[job]))[:6])
                bad.append(f"level {job[1]} seed {_seed(job[2])!r} round {r}: config {_sha(got[job])} instead of {_sha(want)}; differs at {what}")
    assert not bad, (f"{env_id}: {len(bad)} instance(s) built in a thread next to other builds differ from the serial build of the same "
                     "(level, seed). Something a build writes is shared between threads:\n  " + "\n  ".join(bad[:8]))


# ---------------------------------------------------------------------------------------------- episodes
@pytest.mark.parametrize("env_id", IDS)
def test_a_reference_episode_played_next_to_other_families_is_the_serial_one(episodes, env_id):
    want = episodes["serial"][env_id]
    assert want["policy_error"] is None, f"the reference of {env_id} raised in the serial episode: {want['policy_error']}"
    bad = []
    for r, got in enumerate(episodes["rounds"]):
        got = got[env_id]
        if got != want:
            bad.append(f"round {r}: " + (got if isinstance(got, str) else "differs at " + ", ".join(_where(want, got)[:6])))
    assert not bad, (f"{env_id}: the reference episode of level {max(mrgym.spec(env_id).levels)}, seed {_seed(0)!r}, played while "
                     f"{THREADS - 1} other threads played other families is not the serial one:\n  " + "\n  ".join(bad))


# ---------------------------------------------------------------------------------------------- shared state
@pytest.mark.parametrize("env_id", IDS)
def test_no_build_and_no_episode_of_a_family_writes_a_module_constant(instances, episodes, env_id):
    """Scheduling-independent: the serial pass alone shows the write, and names the level that makes it."""
    mine = [(job, name, old, new) for job, name, old, new in instances["writes"] + episodes["writes"] if job is not None and _family_of(job) == env_id]
    seen = sorted({f"{name}: {old} -> {new} ({f'instance of level {job[1]}' if isinstance(job, tuple) else 'reference episode'})"
                   for job, name, old, new in mine})
    assert not seen, (f"{env_id} changes module attributes of the package while it builds or plays ({len(mine)} writes). A value "
                      "written for the duration of a call is read by every other thread, and restored by whichever thread "
                      "read it last. Pass it as an argument instead:\n  " + "\n  ".join(seen[:8]))


def test_no_module_constant_is_left_changed_by_the_threads(instances, episodes):
    moved = dict(instances["constants"], **episodes["constants"])
    assert not moved, ("module attributes of the package differ after the threaded passes from before them (a shared dict or list "
                       "edited in place, or a value written for the duration of a call and restored by another thread):\n  "
                       + "\n  ".join(f"{k}: {v}" for k, v in moved.items()))
