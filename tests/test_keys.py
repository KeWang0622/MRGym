"""Keys, splits and identifiers: nothing public is a function of the seed, and a public split never touches the secret."""
import hashlib
import json
import os
import subprocess
import sys
import types

import pytest

import mrgym
from mrgym import keys
from mrgym.env import derive_seed
from mrgym.registry import ENV_ID_RE

HERE = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(HERE, "data", "private_stub")
NULL = "Scanner/NullTissue-v0"
SECRET = {"MRGYM_SECRET": "ab" * 32}


def _digest(values):
    return hashlib.sha256("".join(values).encode()).hexdigest()


# ------------------------------------------------------------------ the public train split does not move
@pytest.mark.parametrize("environ", [{}, SECRET, {"MRGYM_SECRET": "x"}, {"MRGYM_SECRET_FILE": "/nonexistent/key"}],
                         ids=["no secret", "secret", "malformed secret", "missing secret file"])
def test_train_worlds_and_task_ids_are_the_recorded_ones_with_and_without_a_secret(environ):
    """Seeds 0-99 of every id and level against the values recorded on the base tree (tests/data/train_split_v0.json)."""
    with open(os.path.join(HERE, "data", "train_split_v0.json")) as f:
        recorded = json.load(f)["ids"]
    checked = 0
    for env_id, levels in recorded.items():
        # a retired id (Scanner/NoWrap-v0) is not served, but its seeds and labels are those of its stored records
        spec = mrgym.registry.get(env_id) or types.SimpleNamespace(name=ENV_ID_RE.match(env_id)["name"])
        for lv, rec in levels.items():
            seeds = [derive_seed(env_id, int(lv), "train", k, environ=environ) for k in range(100)]
            ids = [keys.task_id_for(spec, s, "train", environ) for s in seeds]
            assert _digest(seeds) == rec["world_seeds_sha256"] and seeds[:3] == rec["first_world_seeds"], (env_id, lv)
            assert _digest(ids) == rec["task_ids_sha256"] and ids[:3] == rec["first_task_ids"], (env_id, lv)
            checked += 100
    assert checked >= 900


def test_a_secret_in_the_process_environment_does_not_change_a_train_episode(monkeypatch):
    env = mrgym.make(NULL, seed=3)
    _, info = env.reset()
    cfg = env.episode_config()
    monkeypatch.setenv("MRGYM_SECRET", "x")                              # even a malformed one: train never reads it
    env2 = mrgym.make(NULL, seed=3)
    _, info2 = env2.reset()
    assert env2.episode_config()["seed"] == cfg["seed"] and info2["task_id"] == info["task_id"]
    monkeypatch.setenv("MRGYM_SECRET", "cd" * 32)
    env3 = mrgym.make(NULL, seed=3)
    env3.reset()
    assert env3.episode_config() == cfg


# ------------------------------------------------------------------ splits
def test_split_registry():
    assert keys.PUBLIC_SPLITS == ("train", "dev", "tune", "gate")
    assert {"test-iid", "test-range", "test-combo", "test-world", "test-text", "ood", "test-family", "test-mechanism",
            "heldout", "semi"} == set(keys.SECRET_SPLITS)
    for bad in ("tset", "", "Train", None, 3):
        with pytest.raises(mrgym.Error, match="unknown split"):
            derive_seed(NULL, 1, bad, 0, environ={})
    with pytest.raises(mrgym.Error, match="unknown split"):
        mrgym.make(NULL, seed=0, split="tset").reset()
    worlds = {s: derive_seed(NULL, 1, s, 0, environ={}) for s in keys.PUBLIC_SPLITS}
    assert len(set(worlds.values())) == 4                                # the split is in the message: worlds are disjoint
    assert worlds == {s: derive_seed(NULL, 1, s, 0, environ=SECRET) for s in keys.PUBLIC_SPLITS}


@pytest.mark.parametrize("split", keys.SECRET_SPLITS)
def test_secret_splits_refuse_without_a_secret_and_depend_on_every_byte_of_it(split, monkeypatch):
    with pytest.raises(mrgym.Error, match="needs a secret"):
        derive_seed(NULL, 1, split, 0, environ={})
    monkeypatch.delenv("MRGYM_SECRET", raising=False)
    monkeypatch.delenv("MRGYM_SECRET_FILE", raising=False)
    with pytest.raises(mrgym.Error, match="needs a secret"):
        mrgym.make(NULL, seed=0, split=split).reset()
    a = [derive_seed(NULL, 1, split, i, environ=SECRET) for i in range(200)]
    train = [derive_seed(NULL, 1, "train", i, environ=SECRET) for i in range(200)]
    assert not set(a) & set(train) and len(set(a)) == 200
    one_byte = {"MRGYM_SECRET": "ab" * 31 + "ac"}
    assert not set(a) & {derive_seed(NULL, 1, split, i, environ=one_byte) for i in range(200)}
    assert a[0] != derive_seed(NULL, 1, split, 0, environ=dict(SECRET, MRGYM_EPOCH="2026q4"))     # epochs are disjoint
    other = [s for s in keys.SECRET_SPLITS if s != split][0]
    assert a[0] != derive_seed(NULL, 1, other, 0, environ=SECRET)


def test_secrets_are_hex_long_enough_and_come_from_a_private_file(tmp_path):
    for bad, why in (("s", "hexadecimal"), ("zz" * 32, "hexadecimal"), ("ab" * 16, "at least 32")):
        with pytest.raises(mrgym.Error, match=why):
            derive_seed(NULL, 1, "heldout", 0, environ={"MRGYM_SECRET": bad})
    path = keys.keygen(str(tmp_path / "secret.key"))
    assert os.stat(path).st_mode & 0o777 == 0o600 and len(open(path).read().strip()) == 64
    with pytest.raises(FileExistsError):
        keys.keygen(path)                                                # never overwrites a key
    from_file = derive_seed(NULL, 1, "heldout", 0, environ={"MRGYM_SECRET_FILE": path})
    assert from_file == derive_seed(NULL, 1, "heldout", 0, environ={"MRGYM_SECRET": open(path).read().strip()})
    os.chmod(path, 0o644)
    with pytest.raises(mrgym.Error, match="readable by group or others"):
        derive_seed(NULL, 1, "heldout", 0, environ={"MRGYM_SECRET_FILE": path})


def test_sub_keys_are_independent():
    k = bytes.fromhex("ab" * 32)
    subs = {p: keys.sub(k, p) for p in ("world", "label", "log", "commit")}
    assert len(set(subs.values())) == 4 and all(len(v) == 32 for v in subs.values())
    assert keys.sub(k, "world") == keys.sub(k, "world") != keys.sub(bytes.fromhex("ab" * 31 + "ac"), "world")


def test_a_salt_moves_a_public_split_and_only_that(tmp_path):
    salt = {"MRGYM_SALT": "07" * 16}
    plain = derive_seed(NULL, 1, "train", 0, environ={})
    assert derive_seed(NULL, 1, "train", 0, environ=salt) not in (plain, derive_seed(NULL, 1, "train", 0, environ={"MRGYM_SALT": "08" * 16}))
    assert derive_seed(NULL, 1, "train", 0, environ=salt) == derive_seed(NULL, 1, "train", 0, environ=dict(salt))   # reproducible once published
    assert derive_seed(NULL, 1, "heldout", 0, environ=dict(SECRET, **salt)) == derive_seed(NULL, 1, "heldout", 0, environ=SECRET)
    with pytest.raises(mrgym.Error, match="at least 16"):
        derive_seed(NULL, 1, "train", 0, environ={"MRGYM_SALT": "07" * 4})


# ------------------------------------------------------------------ seeds
@pytest.mark.parametrize("seed", [0.0, 1.5, True, False, b"0", [0], (0,), {"s": 0}, "", "x" * 65, "a\nb"])
def test_seed_types_that_used_to_select_other_worlds_silently_are_refused(seed):
    with pytest.raises(mrgym.Error, match="seed must be"):
        derive_seed(NULL, 1, "train", seed, environ={})
    with pytest.raises(mrgym.Error, match="seed must be"):
        mrgym.make(NULL, seed=seed).reset()


def test_seed_forms():
    assert derive_seed(NULL, 1, "train", 7, environ={}) == derive_seed(NULL, 1, "train", "7", environ={})
    assert derive_seed(NULL, 1, "heldout", "ab" * 16, environ={}) == "ab" * 16       # an operator's own 128-bit seed
    assert derive_seed(NULL, 1, "train", None, environ={}) != derive_seed(NULL, 1, "train", None, environ={})
    assert derive_seed(NULL, 1, "train", -1, environ={}) != derive_seed(NULL, 1, "train", 1, environ={})


# ------------------------------------------------------------------ identifiers
def test_episode_ids_are_random_and_no_public_function_of_the_seed():
    env = mrgym.make(NULL, seed=0)
    ids, seeds = set(), set()
    for _ in range(3):
        _, info = env.reset()
        ids.add(info["episode_id"])
        seeds.add(env.episode_config()["seed"])
        assert env.state()["episode_id"] == info["episode_id"] and len(info["episode_id"]) == 16
    seed = seeds.pop()
    assert len(ids) == 3 and not seeds                                   # the same world, three handles
    v0 = hashlib.sha256(("episode:" + seed).encode()).hexdigest()[:12]
    assert v0 not in ids and all(i not in seed and seed[:8] not in i for i in ids)


def test_task_id_is_keyed_on_a_secret_split_and_trainer_side_only():
    spec = mrgym.spec(NULL)
    seed = derive_seed(NULL, 1, "test-iid", 0, environ=SECRET)
    label = keys.task_id_for(spec, seed, "test-iid", SECRET)
    assert label.startswith("nulltissue-") and label != keys.public_label("nulltissue", seed)
    assert label == keys.task_id_for(spec, seed, "test-iid", SECRET)            # stable under one secret
    assert label != keys.task_id_for(spec, seed, "test-iid", {"MRGYM_SECRET": "cd" * 32})
    assert keys.task_id_for(spec, seed, "train") == keys.public_label("nulltissue", seed)
    env = mrgym.make(NULL, seed=0)
    obs, info = env.reset()
    card = env.step({"name": "scanner_info", "arguments": {}}).observation
    assert info["task_id"] == env.episode_config()["task_id"]            # the trainer has it
    assert "task_id" not in card.data and info["task_id"] not in json.dumps(card.data) + card.text + obs.text


def test_the_leak_walk_finds_nothing_and_finds_a_planted_leak(monkeypatch):
    """Positive control first: with the task id back in the instrument card the walk must report it."""
    from mrgym.redteam import common, leak
    from mrgym.scanner import session
    r = leak.walk(common.Context(NULL), 3)                               # 50 episodes per id: mrgym redteam <id> --only RT16
    assert r["episodes"] == 3 and r["payloads"] > 250 and r["hits"] == [], r["hits"][:3]
    card = session.scanner_card
    monkeypatch.setattr(session, "scanner_card", lambda cfg, exam: dict(card(cfg, exam), task_id=cfg["task_id"]))
    planted = leak.walk(common.Context(NULL), 1)["hits"]
    assert {h[1] for h in planted} >= {"task id", "forbidden key"} and len(planted) >= 3
    world_leak = lambda cfg, exam: dict(card(cfg, exam), calibration=session.W.make_world(cfg["seed"], cfg["world_spec"])["system"]["tx_gain"])   # noqa: E731
    monkeypatch.setattr(session, "scanner_card", world_leak)
    assert any(h[1].startswith("hidden value system.tx_gain") for h in leak.walk(common.Context(NULL), 1)["hits"])


# ------------------------------------------------------------------ the split hook and private families
def _run(code, **env):
    full = dict(os.environ, OMP_NUM_THREADS="1", **env)
    full["PYTHONPATH"] = os.pathsep.join([STUB] + [p for p in (os.path.dirname(os.path.dirname(mrgym.__file__)), full.get("PYTHONPATH")) if p])
    return subprocess.run([sys.executable, *code], env=full, capture_output=True, text=True, timeout=120)


def test_a_private_package_registers_its_family_through_the_entry_point_group():
    listed = _run(["-m", "mrgym.cli", "list"])
    assert listed.returncode == 0 and "Scanner/PrivateStub-v0" in listed.stdout, listed.stderr
    assert "Scanner/PrivateStub-v0" not in mrgym.list_envs()             # and it is not part of the public tree
    played = _run(["-c", "import mrgym, json; from mrgym import policies\n"
                   "env = mrgym.make('Scanner/PrivateStub-v0', seed=0, split='dev')\n"
                   "r = policies.run(env, 'reference')\n"
                   "print(json.dumps([r.reward, env.episode_config()['split'], mrgym.make('Scanner/NullTissue-v0', seed=0, split='dev').reset()[1]['split']]))"])
    assert played.returncode == 0, played.stderr
    reward, split_seen, split_plain = json.loads(played.stdout.strip().splitlines()[-1])
    assert reward in (0.0, 1.0) and split_seen == "dev" and split_plain == "dev"      # the split reaches a family that takes it


def test_make_instance_gets_the_split_only_if_the_family_takes_it():
    class Old:
        @staticmethod
        def make_instance(seed, level, task_id=None):
            return dict(seed=seed, level=level, task_id=task_id)

    class New:
        @staticmethod
        def make_instance(seed, level, task_id=None, split="train"):
            return dict(seed=seed, level=level, task_id=task_id, split=split)
    assert keys.make_instance(Old, "ab" * 16, 2, "t", "test-range") == dict(seed="ab" * 16, level=2, task_id="t")
    assert keys.make_instance(New, "ab" * 16, 2, "t", "test-range")["split"] == "test-range"


def test_a_broken_plugin_is_a_warning_not_a_crash(monkeypatch):
    from importlib import metadata

    from mrgym import envs

    class Broken:
        name, value = "broken", "no_such_module.spec"

        def load(self):
            raise ModuleNotFoundError("no_such_module")
    monkeypatch.setattr(metadata, "entry_points", lambda group: [Broken()])
    with pytest.warns(RuntimeWarning, match="was not registered"):
        assert envs.load_entry_points() == []


# ------------------------------------------------------------------ canary
def test_canary_is_in_exported_records_and_never_in_agent_visible_text():
    assert keys.CANARY.startswith("MRGYM-CANARY GUID ") and len(keys.CANARY.split()[-1]) == 36
    env = mrgym.make(NULL, seed=0)
    obs, _ = env.reset()
    from mrgym import policies
    step = policies.run(env, "prior")
    assert step.info["log_head"]["canary"] == keys.CANARY                # the receipt is an exported record
    visible = obs.text + json.dumps(obs.data) + "".join(e["text"] for e in env.transcript()) + json.dumps(env.tools("openai"))
    assert "CANARY" not in visible.upper()
    for doc in ("docs/DATASHEET.md", "SECURITY.md"):
        path = os.path.join(os.path.dirname(HERE), *doc.split("/"))
        if os.path.exists(path):                                         # absent when the tests run against an installed wheel
            assert keys.CANARY in open(path, encoding="utf-8").read(), doc
