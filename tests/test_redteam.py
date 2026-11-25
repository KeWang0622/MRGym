"""The red-team suite: every row states what must happen and what happened, and a mismatch is a non-zero exit."""
import json
import os
import subprocess
import sys

import pytest

import mrgym
from mrgym import cli, keys, redteam
from mrgym.redteam import common, mutants, rows

NULL = "Scanner/NullTissue-v0"
STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "private_stub")
# on this tree two malformed tool-call shapes still raise out of step() (owned by the interface work package);
# the suite reports them as a mismatch of RT14b, and nothing else may mismatch
KNOWN = {"RT14b"}


@pytest.fixture(scope="module")
def ctx():
    return common.Context(NULL, n=2)


def test_tool_tier_rows_on_the_worked_example(ctx, tmp_path):
    got = {}
    for fn in (rows.rt00_reference, rows.rt01_null, rows.rt02_no_scan, rows.rt05_junk_scan, rows.rt14_format,
               rows.rt15_lifecycle, rows.rt16_leak, rows.rt23_inprocess_read):
        out = fn(ctx)
        got.update({r.id: r for r in (out if isinstance(out, list) else [out])})
    got["RT22"] = rows.rt22_reuse(ctx, str(tmp_path))
    got["RT04"] = rows.rt04_reward_is_pass(ctx, ctx.episodes)
    bad = {i: r.observed for i, r in got.items() if r.ok is False and i not in KNOWN}
    assert not bad, bad
    assert got["RT00"].detail["passed"] == 2 and got["RT02"].ok and got["RT02b"].ok and got["RT14"].ok
    assert got["RT05"].ok is None and "too small to decide" in got["RT05"].observed       # never "ok" on two instances
    assert got["RT23"].observed == "succeeds" and got["RT23"].tier == "inprocess"
    assert len(ctx.episodes) >= 8 and got["RT04"].ok


def test_seed_enumeration_recovers_public_worlds_and_no_salted_or_secret_one(ctx, tmp_path):
    """The positive control: the attack with the documented derivation must work on the public split."""
    r = {x.id: x for x in rows.rt17_seed_enumeration(ctx, str(tmp_path))}
    assert r["RT17"].ok and r["RT17"].detail["recovered"] == 2 and r["RT17"].detail["passed"] == 2
    for rid in ("RT17b", "RT17c"):
        assert r[rid].detail["recovered"] == 0 and r[rid].detail["passed"] == 0 and r[rid].ok is not False, r[rid].observed
    assert "task id = public derivation on 0/2" in r["RT17c"].observed
    assert not os.environ.get("MRGYM_SECRET_FILE") or "rt17" not in os.environ["MRGYM_SECRET_FILE"]    # the row cleans up


def test_log_and_stored_file_rows(ctx, tmp_path):
    log = rows.rt20_log(ctx, str(tmp_path))
    assert log.ok and len(log.detail["outcomes"]) == 10 and all(log.detail["outcomes"].values())
    assert rows.rt20_inprocess_tamper(ctx).ok and rows.rt18_stored_files(ctx).ok


def test_a_scorer_mutant_is_killed_and_the_detectors_hold_without_one(ctx):
    ref = ctx.reference_runs(range(2))
    seeds = [s for s, r in ref.items() if r["passed"]][:1]
    all_mutants = mutants.mutants(ctx)
    assert len(all_mutants) >= 12
    base = {d: mutants._detect(ctx, d, seeds, ref) for d in ("prior", "no_scan", "tampered")}
    assert base == dict(prior=(0.0,), no_scan=(0.0,), tampered=(0.0,))
    for name, detector in (("M01", "prior"), ("M04", "no_scan"), ("M12", "tampered")):
        with mutants.patched(*all_mutants[name][1]):
            assert mutants._detect(ctx, detector, seeds, ref) != base[detector], name
    assert {d: mutants._detect(ctx, d, seeds, ref) for d in base} == base                 # the patches are gone


def test_a_broken_scorer_makes_the_suite_exit_non_zero(monkeypatch, capsys):
    from mrgym.envs import family
    monkeypatch.setattr(family, "reward_of", lambda passed: 1.0)                          # a scorer that pays everything
    assert cli.main(["redteam", NULL, "--n", "2", "--only", "RT05", "--workers", "1"]) == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "reward > 0 in 2" in out and "1 mismatch" in out


def test_cli_prints_expected_against_observed_and_writes_json(tmp_path, capsys):
    path = tmp_path / "redteam.json"
    assert cli.main(["redteam", NULL, "--n", "2", "--only", "RT01,RT15,RT22", "--tier", "tool", "--json", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("expected: ") == out.count("observed: ") == 3 and "3 ok · 0 mismatches" in out
    stored = json.loads(path.read_text())
    assert stored["schema"] == redteam.SCHEMA and stored["canary"] == keys.CANARY and stored["ok"] and stored["engine"] == mrgym.ENGINE_ID
    assert [r["id"] for r in stored["rows"]] == ["RT01", "RT15", "RT22"]


def test_full_table_lists_what_is_not_covered():
    ids = {r.id for r in rows.not_covered()}
    assert ids == {"RT07", "RT08", "RT09", "RT11", "RT19"} and all(r.ok is None for r in rows.not_covered())


@pytest.mark.parametrize("env_id", ["Scanner/NoWrap-v1", "Scanner/T1Transmit-v1"])
def test_the_suite_runs_on_the_two_migrated_ids(env_id, tmp_path):
    """Both ids follow ``envs/family.py`` now, so every row of the suite applies to them (the scorer mutants of RT25
    included: ``mrgym redteam <id> --only RT25``; one mutant is run here)."""
    c = common.Context(env_id, n=2)
    assert c.family_style and "seed_attacker" not in c.must_fail and {"prior", "fixed_recipe", "wrong_units"} <= set(c.must_fail)
    got = [rows.rt01_null(c), rows.rt22_reuse(c, str(tmp_path))]
    if env_id == "Scanner/NoWrap-v1":
        got += [rows.rt15_lifecycle(c)]
    assert not [(r.id, r.observed) for r in got if r.ok is False]
    ref = c.reference_runs(range(2))
    seeds = [s for s, r in ref.items() if r["passed"]][:1]
    assert seeds, "the reference passes neither of the two instances"
    base = mutants._detect(c, "prior", seeds, ref)
    assert base == (0.0,)
    with mutants.patched(*mutants.mutants(c)["M01"][1]):                  # the reward is always 1: the no-scan answer is paid
        assert mutants._detect(c, "prior", seeds, ref) != base
    assert mutants._detect(c, "prior", seeds, ref) == base                # the patch is gone
    with pytest.raises(mrgym.Error, match="sandbox"):
        common.Context("Scanner/Explore-v0")


def test_the_suite_runs_on_a_family_from_a_private_package():
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join([STUB, os.path.dirname(os.path.dirname(mrgym.__file__)), env.get("PYTHONPATH", "")])
    out = subprocess.run([sys.executable, "-m", "mrgym.cli", "redteam", "Scanner/PrivateStub-v0", "--n", "2", "--only", "RT02",
                          "--tier", "tool", "--workers", "1"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout[-600:] + out.stderr[-600:]
    assert "Scanner/PrivateStub-v0 level 1" in out.stdout and "0 mismatches" in out.stdout and "RT02b" in out.stdout
