"""The versioning and deprecation policy as code (docs/VERSIONING.md): an id is served, deprecated or retired, and
never reused. No real id is retired here: the mechanism is exercised with ids that exist only in these tests."""
import warnings

import pytest

import mrgym
from mrgym.registry import deprecate, deprecated, retire, retired

FAKE, FAKE_NEXT = "Scanner/RetiredForTheTest-v0", "Scanner/RetiredForTheTest-v1"


@pytest.fixture
def clean():
    before = (dict(retired), dict(deprecated))
    yield
    retired.clear(), retired.update(before[0])
    deprecated.clear(), deprecated.update(before[1])
    mrgym.registry.pop(FAKE_NEXT, None)
    from mrgym.registry import _warned
    _warned.discard(FAKE_NEXT)


def test_no_real_id_is_retired_or_deprecated_by_this_package():
    """Which ids retire is the release manager's decision; the lists ship empty until it is taken."""
    assert set(retired) <= set() or all(i not in mrgym.registry for i in retired)
    assert all(i in mrgym.registry for i in deprecated)


def test_a_retired_id_raises_retired_env_naming_the_last_version_that_served_it(clean):
    retire(FAKE, "0.1.0", successor=FAKE_NEXT)
    with pytest.raises(mrgym.RetiredEnv) as e:
        mrgym.make(FAKE)
    msg = str(e.value)
    assert FAKE in msg and "mrgym 0.1.0" in msg and "pip install mrgym==0.1.0" in msg and FAKE_NEXT in msg
    assert isinstance(e.value, mrgym.UnknownEnv)                     # code that catches UnknownEnv keeps working
    with pytest.raises(mrgym.RetiredEnv):
        mrgym.spec(FAKE)
    assert FAKE not in mrgym.list_envs()


def test_a_retired_id_without_a_successor_says_so(clean):
    retire(FAKE, "0.1.0")
    with pytest.raises(mrgym.RetiredEnv, match="no successor"):
        mrgym.make(FAKE)


def test_a_retired_id_is_never_reused_and_a_served_id_cannot_be_retired(clean):
    retire(FAKE, "0.1.0")
    with pytest.raises(mrgym.Error, match="never reused"):
        mrgym.register(FAKE, "mrgym.envs.explore:ExploreEnv", "mrgym.envs.explore:card")
    served = mrgym.list_envs()[0]
    with pytest.raises(mrgym.Error, match="served or retired"):
        retire(served, "0.1.0")
    with pytest.raises(mrgym.Error):
        retire("not an id", "0.1.0")
    with pytest.raises(mrgym.Error):
        retire(FAKE_NEXT, "")


def test_an_unknown_id_is_not_reported_as_retired():
    with pytest.raises(mrgym.UnknownEnv) as e:
        mrgym.make("Scanner/NeverExisted-v0")
    assert not isinstance(e.value, mrgym.RetiredEnv)


def test_an_id_without_a_version_suffix_stays_an_error():
    """Gymnasium picks the latest version silently; that would make stored results ambiguous."""
    name = mrgym.list_envs()[0].rsplit("-v", 1)[0]
    with pytest.raises(mrgym.UnknownEnv, match="malformed"):
        mrgym.make(name)


def test_a_deprecated_id_still_runs_and_warns_once_with_its_successor(clean):
    src = mrgym.spec("Scanner/Explore-v0")
    mrgym.register(FAKE_NEXT, src.entry_point, src.card, kind=src.kind, levels=src.levels, max_steps=src.max_steps,
                   reward_range=src.reward_range, reward_threshold=src.reward_threshold)
    deprecate(FAKE_NEXT, "0.2.0", "Scanner/Explore-v0")
    with pytest.warns(DeprecationWarning, match="out of date since mrgym 0.2.0; use Scanner/Explore-v0"):
        env = mrgym.make(FAKE_NEXT, seed=0)
    env.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mrgym.make(FAKE_NEXT, seed=0)                                # once per process
    with pytest.raises(mrgym.Error):
        deprecate("Scanner/NeverExisted-v0", "0.2.0", FAKE_NEXT)
    with pytest.raises(mrgym.Error):
        deprecate(FAKE_NEXT, "0.2.0", FAKE_NEXT)


def test_the_public_names_of_the_first_snapshot_are_still_exported():
    """Interface rule: while the package is 0.y a public name may only be removed with a "Removed" entry in the
    changelog. Additions are free. This list is the public surface at the v1 base; shrink it only with that entry."""
    snapshot = {"ENGINE_ID", "ENGINE_VERSION", "Env", "EnvSpec", "Error", "Observation", "ResetNeeded", "StepResult", "ToolSpec",
                "UnknownEnv", "card", "list_envs", "make", "register", "registry", "spec", "__version__"}
    missing = sorted(n for n in snapshot if n not in mrgym.__all__ or not hasattr(mrgym, n))
    assert not missing, f"public names removed: {missing}"
    assert "RetiredEnv" in mrgym.__all__
