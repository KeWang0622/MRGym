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
