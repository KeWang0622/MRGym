"""Held-out epochs: the commitment is published first, the reveal later, and anyone can then re-score a result."""
import copy
import hashlib
import json

import pytest

import mrgym
from mrgym import cli, epoch, keys

NULL = "Scanner/NullTissue-v0"


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """One epoch through the command line: keygen, new, run (reference, 2 indices), reveal."""
    tmp = tmp_path_factory.mktemp("epoch")
    mp = pytest.MonkeyPatch()
    mp.delenv("MRGYM_SECRET", raising=False)
    p = {k: str(tmp / f"e.{k}.json") for k in ("commit", "result", "reveal")}
    p["key"] = str(tmp / "secret.key")
    assert cli.main(["epoch", "keygen", "--out", p["key"]]) == 0
    mp.setenv("MRGYM_SECRET_FILE", p["key"])
    assert cli.main(["epoch", "new", "--id", NULL, "--levels", "1", "--split", "test-iid", "--epoch", "2026q4", "--n", "8", "--out", p["commit"]]) == 0
    assert cli.main(["epoch", "run", "--commit", p["commit"], "--policy", "reference", "--n", "2", "--out", p["result"]]) == 0
    assert cli.main(["epoch", "reveal", "--commit", p["commit"], "--out", p["reveal"]]) == 0
    mp.undo()
    return p, {k: json.load(open(p[k])) for k in ("commit", "result", "reveal")}


def _verify(tmp_path, commit, reveal, result=None):
    paths = []
    for name, obj in (("commit", commit), ("reveal", reveal), ("result", result)):
        if obj is not None:
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(obj))
            paths += [f"--{name}", str(path)]
    return cli.main(["epoch", "verify", *paths])


def test_merkle_root_is_rfc_6962():
    leaf = lambda h: hashlib.sha256(b"\x00" + bytes.fromhex(h)).digest()         # noqa: E731
    node = lambda a, b: hashlib.sha256(b"\x01" + a + b).digest()                 # noqa: E731
    a, b, c = (f"{i:064x}" for i in (1, 2, 3))
    assert epoch.merkle_root([]) == hashlib.sha256(b"").hexdigest()
    assert epoch.merkle_root([a]) == leaf(a).hex()
    assert epoch.merkle_root([a, b, c]) == node(node(leaf(a), leaf(b)), leaf(c)).hex()


def test_the_commitment_binds_and_hides(files, monkeypatch):
    p, f = files
    commit, reveal = f["commit"], f["reveal"]
    text = json.dumps(commit)
    seeds = reveal["levels"]["1"]["world_seeds"]
    assert len(commit["levels"]["1"]["commitments"]) == 8 == len(set(seeds)) and commit["canary"] == keys.CANARY
    assert not [s for s in seeds if s in text or s[:12] in text]         # the published file does not give a world away
    assert commit["engine"] == mrgym.ENGINE_ID and set(commit["sources"]) == {"scanner", "family", "family_package"}
    assert epoch.check_reveal(commit, reveal) == []
    monkeypatch.delenv("MRGYM_SECRET", raising=False)
    monkeypatch.delenv("MRGYM_SECRET_FILE", raising=False)
    with pytest.raises(epoch.EpochError, match="needs the secret"):
        epoch.new(NULL, [1], "test-iid", "x", 4)
    with pytest.raises(epoch.EpochError, match="is public"):
        epoch.new(NULL, [1], "train", "x", 4, environ={"MRGYM_SECRET": "ab" * 32})
    with pytest.raises(epoch.EpochError, match="does not open"):          # another secret cannot produce a reveal
        epoch.reveal(commit, environ={"MRGYM_SECRET": "ab" * 32})
    again = epoch.new(NULL, [1], "test-iid", "2026q4", 8, environ={"MRGYM_SECRET_FILE": p["key"]})
    assert again["root"] == commit["root"]                               # the same secret and epoch commit to the same worlds
    assert epoch.new(NULL, [1], "test-iid", "2027q1", 8, environ={"MRGYM_SECRET_FILE": p["key"]})["root"] != commit["root"]


def test_verify_passes_an_honest_reveal_and_needs_no_secret(files, tmp_path, monkeypatch, capsys):
    _, f = files
    monkeypatch.delenv("MRGYM_SECRET", raising=False)
    monkeypatch.delenv("MRGYM_SECRET_FILE", raising=False)
    assert _verify(tmp_path, f["commit"], f["reveal"]) == 0
    assert _verify(tmp_path, f["commit"], f["reveal"], f["result"]) == 0
    assert '"ok": true' in capsys.readouterr().out
    v = epoch.verify(f["commit"], f["reveal"], f["result"])
    assert v["ok"] and v["checked"] == dict(commitments=8, episodes=2, rewards_equal=2)
    block = f["result"]["levels"]["1"]
    assert block["indices"] == [0, 1] and all(e["head"]["sealed"] for e in block["episodes"]) and f["result"]["canary"] == keys.CANARY


def test_verify_fails_on_one_changed_seed_log_line_index_or_reward(files, tmp_path, capsys):
    _, f = files
    commit, reveal, result = f["commit"], f["reveal"], f["result"]
    seed = copy.deepcopy(reveal)
    s = seed["levels"]["1"]["world_seeds"][5]
    seed["levels"]["1"]["world_seeds"][5] = s[:-1] + ("0" if s[-1] != "0" else "1")
    assert _verify(tmp_path, commit, seed) == 1 and "do not open" in capsys.readouterr().out
    line = copy.deepcopy(result)
    scan = next(e for e in line["levels"]["1"]["episodes"][1]["log"] if e["type"] == "scan")
    scan["realised"]["ti_ms"] = scan["realised"].get("ti_ms", 100.0) + 1.0
    assert _verify(tmp_path, commit, reveal, line) == 1 and "does not replay" in capsys.readouterr().out
    index = copy.deepcopy(result)
    index["levels"]["1"]["episodes"][1]["index"] = 2
    index["levels"]["1"]["indices"] = [0, 2]
    assert _verify(tmp_path, commit, reveal, index) == 1 and "without a gap" in capsys.readouterr().out
    reward = copy.deepcopy(result)
    e = reward["levels"]["1"]["episodes"][0]
    e["reward"] = 1.0 - e["reward"]
    v = epoch.verify(commit, reveal, reward)
    assert not v["ok"] and "recomputed" in v["problems"][0]
    swapped = copy.deepcopy(result)                                      # the log of index 1 presented as index 0
    eps = swapped["levels"]["1"]["episodes"]
    eps[0]["log"], eps[0]["head"] = eps[1]["log"], eps[1]["head"]
    assert not epoch.verify(commit, reveal, swapped)["ok"]
    drift = copy.deepcopy(commit)
    drift["sources"]["scanner"] = "0" * 64
    assert not epoch.verify(drift, reveal)["ok"] and epoch.verify(drift, reveal, allow_source_drift=True)["ok"]
