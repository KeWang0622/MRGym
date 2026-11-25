"""Shared fixtures. The tests import the installed package (``pip install -e .`` or the built wheel): nothing here
adds the repository to ``sys.path``."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest

SEED = "000102030405060708090a0b0c0d0e0f"


@pytest.fixture(scope="session")
def seed():
    return SEED


@pytest.fixture(scope="session")
def qa_world():
    from mrgym.scanner import world
    return world.make_world(SEED, dict(kind="qa"))


@pytest.fixture(scope="session")
def qa_engine(qa_world):
    from mrgym.scanner import kspace
    return kspace.Engine(qa_world)


def episode_cfg(seed=SEED, budget_s=60.0, **spec):
    return dict(task_id="test-episode", seed=seed, world_spec=dict(dict(kind="qa"), **spec), budget_s=budget_s, max_requests=24,
                task=dict(regions=[]))


def scan_call(env, protocol) -> dict:
    """A ``scan`` tool call for ``protocol`` in the shape the scan tool of this tree takes: a ``protocol`` object, or flat
    arguments. Tests that step a scan by hand use it, so a change of the tool's shape does not silently turn their scans
    into refused calls."""
    schema = next(t["parameters"] for t in env.tools("neutral") if t["name"] == "scan")
    return {"name": "scan", "arguments": {"protocol": dict(protocol)} if "protocol" in schema["properties"] else dict(protocol)}


def legacy_task(env_id: str, package: str):
    """(task module, controls module, "") of a v0 id that still scores the old way (``score(cfg, answer, ..., measured,
    log_ok)``), else (None, None, why not): the id is not registered any more, the package is gone, or its task follows
    the family contract. Tests of the v0 ids skip with that reason while the ids are being migrated."""
    import importlib
    import inspect

    import mrgym
    if env_id not in mrgym.registry:
        return None, None, f"{env_id} is not registered any more (retired or replaced by a family built on envs/family.py)"
    try:
        task, controls = importlib.import_module(package + ".task"), importlib.import_module(package + ".controls")
    except ImportError as e:
        return None, None, f"{package} is gone ({e})"
    if "measured" not in inspect.signature(task.score).parameters:
        return None, None, f"{package}.task follows the family contract now: its table belongs under tests/scoring/"
    return task, controls, ""


# ---------------------------------------------------------------------------------------------- a family that costs nothing
STUB_ID = "Test/Stub-v0"


@pytest.fixture
def stub_env():
    """Registers ``tests/stub_family`` as ``Test/Stub-v0`` for one test (see its docstring)."""
    import mrgym
    mrgym.register(STUB_ID, "mrgym.envs.family:FamilyEnv", "stub_family.task:card", kind="candidate", family="stub_family",
                   levels=(1, 2), default_level=1, max_steps=12)
    yield STUB_ID
    mrgym.registry.pop(STUB_ID)


class InProcessPool:
    """Stands in for ``multiprocessing`` in ``gate.run``: the jobs run in the test process, in order,
    so a test sees every call (and so does a mutation run). Records the start method and the pool size it was asked for."""

    def __init__(self):
        self.method, self.workers, self.maps = None, None, 0

    def get_context(self, method):
        assert method in ("fork", "spawn", "forkserver"), method
        self.method = method
        return self

    def Pool(self, workers):
        self.workers = workers
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs, chunksize=None):
        self.maps += 1
        return [fn(j) for j in jobs]


@pytest.fixture
def in_process_pool():
    return InProcessPool()


class FakeClock:
    """Stands in for the ``time`` module of ``gate``: every call of ``time()`` is ``step`` later."""

    def __init__(self, step=1.234):
        self.now, self.step = 1000.0, step

    def time(self):
        self.now += self.step
        return self.now


class FakeDatetime:
    """Stands in for the ``datetime`` module: 23:30 local on 24 November is already 25 November in UTC."""
    import datetime as _dt
    timezone = _dt.timezone

    class datetime:
        @staticmethod
        def now(tz=None):
            import datetime as dt
            return dt.datetime(2026, 11, 25, 7, 30, tzinfo=dt.timezone.utc) if tz is dt.timezone.utc else dt.datetime(2026, 11, 24, 23, 30)


# ---------------------------------------------------------------------------------------------- Hypothesis profiles
# dev      what a plain ``pytest`` runs: a few derandomised examples, seconds
# ci       what ``make audit`` and the CI ``quality`` job run: HYPOTHESIS_PROFILE=ci
# nightly  random, for the scheduled fuzz job
try:
    from hypothesis import HealthCheck, settings
except ImportError:                                                     # the plain suite runs without Hypothesis
    pass
else:
    _common = dict(deadline=None, database=None, print_blob=True, suppress_health_check=list(HealthCheck))
    settings.register_profile("dev", max_examples=8, stateful_step_count=8, derandomize=True, **_common)
    settings.register_profile("ci", max_examples=200, stateful_step_count=20, derandomize=True, **_common)
    settings.register_profile("nightly", max_examples=5000, stateful_step_count=20, derandomize=False, **_common)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_report_header(config):
    """Which way the golden digests are compared on this stack (tests/test_golden.py)."""
    try:
        import golden_cases as G
        recorded = G.load(G.ENGINE_FILE)
        if not recorded:
            return "golden digests: no file (python tools/audit/golden.py --write)"
        m, why = G.mode(recorded["stack"])
        return f"golden digests: mode {m} ({why}); recorded on engine {recorded['engine']}"
    except Exception as e:  # noqa: BLE001  a header must never break a test run
        return f"golden digests: unavailable ({type(e).__name__}: {e})"
