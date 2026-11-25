"""The scanner log, version 3: a keyed chain, a seal, and a head that someone other than the log's holder keeps.

The scenarios are the ones a holder of the log directory can try: shorten the log and restart (budget refill), present a
prefix, edit a record, append one, present another episode's log, tear the last line, start twice.
"""
import copy
import json
import os

import pytest

from conftest import SEED, episode_cfg
from mrgym.scanner import session

SPGR = dict(sequence="spgr", tr_ms=12.0, flip_deg=12.0, matrix=64)


def _start(tmp_path, qa_engine, name="ep", budget_s=30.0, **kw):
    cfg = episode_cfg(budget_s=budget_s)
    s = session.ScannerSession(cfg, log_dir=str(tmp_path / name / "log"), engine=qa_engine, state_dir=str(tmp_path / name / "state"), **kw)
    return cfg, s


def _cut(path, keep):
    with open(path) as f:
        lines = f.readlines()
    with open(path, "w") as f:
        f.writelines(lines[:keep])


def test_records_carry_a_mac_and_the_head_commits_to_the_log(tmp_path, qa_engine):
    cfg, s = _start(tmp_path, qa_engine)
    s.plan({"protocol": SPGR})
    s.scan({"protocol": SPGR})
    head, on_disk = s.head(), session.read_head(s.head_path)
    assert head["events"] == 2 and head["mac"] == s.events[-1]["mac"] and not head["sealed"] and head["v"] == session.LOG_VERSION == 3
    assert {k: on_disk[k] for k in head} == head and "key" in on_disk and "key" not in head     # the episode key is in the receipt only
    assert os.stat(s.head_path).st_mode & 0o777 == 0o600 and os.stat(s.log_path).st_mode & 0o777 == 0o600
    assert os.path.dirname(s.head_path) != os.path.dirname(s.log_path)
    assert SEED not in json.dumps(on_disk) + open(s.log_path).read()
    receipt = s.seal("submit")
    assert receipt["sealed"] and s.events[-1]["type"] == "seal" and session.read_head(s.head_path)["sealed"]
    with pytest.raises(session.EpisodeOver):
        s.scan({"protocol": SPGR})
    with pytest.raises(session.EpisodeOver):
        s.plan({"protocol": SPGR})
    rep = session.verify_log(cfg, os.path.dirname(s.log_path), s.head_path, engine=qa_engine)
    assert rep["ok"] and rep["sealed"] and rep["integrity"] == "anchored" and rep["scans"] == 1
    s.close()


def test_the_head_must_not_live_with_the_log(tmp_path, qa_engine):
    for state in (tmp_path / "log", tmp_path / "log" / "inside"):
        with pytest.raises(ValueError, match="must not be the log directory"):
            session.ScannerSession(episode_cfg(), log_dir=str(tmp_path / "log"), engine=qa_engine, state_dir=str(state))


def test_cut_log_and_restart_refills_nothing(tmp_path, qa_engine):
    """Two scans, stop, cut the log to one record, restart: refused, no third scan, and the episode is marked."""
    cfg, s = _start(tmp_path, qa_engine, budget_s=20.0)
    s.scan({"protocol": SPGR})
    s.scan({"protocol": SPGR})
    used, head_path, log_path, state = s.used, s.head_path, s.log_path, os.path.dirname(s.head_path)
    s.close()
    _cut(log_path, 1)
    with pytest.raises(session.LogTampered, match="shortened"):
        session.ScannerSession(cfg, log_dir=os.path.dirname(log_path), engine=qa_engine, state_dir=state, resume_from=head_path)
    with pytest.raises(session.LogTampered, match="already holds a log"):          # nor without the receipt
        session.ScannerSession(cfg, log_dir=os.path.dirname(log_path), engine=qa_engine)
    head = session.read_head(head_path)
    assert head["sealed"] and head["reason"] == "tampered" and head["used_s"] == pytest.approx(used) and head["scans"] == 2
    with pytest.raises(session.LogSealed):                                          # and it stays dead
        session.ScannerSession(cfg, log_dir=os.path.dirname(log_path), engine=qa_engine, state_dir=state, resume_from=head_path)
    assert len(session.read_log(log_path)) == 1                                     # 0 extra seconds were scanned
    assert session.LogTampered.EXIT_STATUS == 3


def test_honest_stop_and_resume_continues_the_budget(tmp_path, qa_engine):
    cfg, s = _start(tmp_path, qa_engine, budget_s=20.0)
    s.scan({"protocol": SPGR})
    used, head_path, log_dir, state = s.used, s.head_path, os.path.dirname(s.log_path), os.path.dirname(s.head_path)
    s.close()
    r = session.ScannerSession(cfg, log_dir=log_dir, engine=qa_engine, state_dir=state, resume_from=head_path)
    assert r.used == used and r.n_scans == 1 and r.restarts == 1 and r.budget()["requests"] == 1 and r.handle == s.handle
    r.scan({"protocol": SPGR})
    receipt = r.seal()
    r.close()
    rep = session.verify_log(cfg, log_dir, head_path, engine=qa_engine)
    assert rep["ok"] and rep["sealed"] and rep["restarts"] == 1 and rep["scans"] == 2 and receipt["restarts"] == 1


def test_verifier_rejects_prefix_edit_append_foreign_and_headless_logs(tmp_path, qa_engine):
    cfg, s = _start(tmp_path, qa_engine)
    for _ in range(3):
        s.scan({"protocol": SPGR})
    receipt, events = s.seal(), list(s.events)
    s.close()
    ok = session.replay(cfg, events, engine=qa_engine, head=receipt)
    assert ok["ok"] and ok["sealed"]

    def problem(ev, head=receipt):
        rep = session.replay(cfg, ev, engine=qa_engine, head=head)
        assert not rep["ok"] and rep["integrity"] == "unanchored" and not rep["sealed"]
        return rep["problems"][0]
    assert "shortened" in problem(events[:2])                                       # a prefix is a valid chain, not a valid log
    edited = copy.deepcopy(events)
    edited[1]["cost_s"] = 0.01
    assert "MAC" in problem(edited) or "chain" in problem(edited)
    fixed = copy.deepcopy(edited)                                                   # even with the unkeyed chain recomputed
    chain = session._chain0()
    for rec in fixed:
        chain = rec["chain"] = session._sha((chain + session._canon(session._body(rec))).encode())
    assert "MAC" in problem(fixed)
    forged = events[:-1] + [dict(copy.deepcopy(events[1]), event=len(events) - 1)]                  # a record written without the key
    assert "chain" in problem(forged)
    forged[-1]["chain"] = session._sha((events[-2]["chain"] + session._canon(session._body(forged[-1]))).encode())
    assert "MAC" in problem(forged)
    problem(events + [dict(copy.deepcopy(events[1]), event=len(events))])                         # nor anything after the seal
    assert "no head" in problem(events, head=None)
    assert "tag" in problem(events, head=dict(receipt, events=2))                   # an edited head
    assert "no episode key" in problem(events, head={k: v for k, v in receipt.items() if k != "key"})
    # another episode of the same world with the same calls: identical bodies, another handle and key
    cfg2, s2 = _start(tmp_path, qa_engine, name="other")
    for _ in range(3):
        s2.scan({"protocol": SPGR})
    receipt2, events2 = s2.seal(), list(s2.events)
    s2.close()
    assert [session._body(a) for a in events] == [session._body(b) for b in events2] and receipt2["handle"] != receipt["handle"]
    assert "MAC" in problem(events2)
    assert not session.replay(cfg, events, engine=qa_engine, head=receipt2)["ok"]


def test_unsealed_log_is_reported_as_such(tmp_path, qa_engine):
    cfg, s = _start(tmp_path, qa_engine)
    s.scan({"protocol": SPGR})
    rep = session.replay(cfg, s.events, engine=qa_engine, head=s.receipt())
    assert rep["ok"] and not rep["sealed"]
    s.close()


def test_torn_last_line_is_moved_aside_and_the_budget_is_the_same(tmp_path, qa_engine):
    cfg, s = _start(tmp_path, qa_engine, budget_s=20.0)
    s.scan({"protocol": SPGR})
    used, head_path, log_path, state = s.used, s.head_path, s.log_path, os.path.dirname(s.head_path)
    s.close()
    with open(log_path, "a") as f:
        f.write('{"v":3,"event":1,"type":"scan","cost_s":0.5,"rea')              # a crash in the middle of a write
    r = session.ScannerSession(cfg, log_dir=os.path.dirname(log_path), engine=qa_engine, state_dir=state, resume_from=head_path)
    assert r.used == used and r.n_scans == 1 and r.restarts == 1
    assert os.path.exists(os.path.join(os.path.dirname(log_path), session.TORN_NAME))
    r.scan({"protocol": SPGR})
    r.close()
    assert [e["type"] for e in session.read_log(log_path)] == ["scan", "restart", "scan"]


def test_a_crash_between_append_and_head_rolls_forward_and_cannot_refill(tmp_path, qa_engine):
    """The log is one record ahead of its head (records with valid MACs): they are kept, so the budget only shrinks."""
    cfg, s = _start(tmp_path, qa_engine, budget_s=20.0)
    s.scan({"protocol": SPGR})
    stale = open(s.head_path).read()
    s.scan({"protocol": SPGR})
    used, head_path, log_path, state = s.used, s.head_path, s.log_path, os.path.dirname(s.head_path)
    s.close()
    with open(head_path, "w") as f:
        f.write(stale)                                                             # the head of one record ago
    r = session.ScannerSession(cfg, log_dir=os.path.dirname(log_path), engine=qa_engine, state_dir=state, resume_from=head_path)
    assert r.used == used and r.n_scans == 2
    r.close()


def test_a_second_session_on_the_same_directory_is_refused(tmp_path, qa_engine):
    pytest.importorskip("fcntl")
    cfg, s = _start(tmp_path, qa_engine)
    s.scan({"protocol": SPGR})
    with pytest.raises(session.LogTampered, match="another scanner session"):
        session.ScannerSession(cfg, log_dir=os.path.dirname(s.log_path), engine=qa_engine, state_dir=os.path.dirname(s.head_path),
                               resume_from=s.head_path)
    s.close()


def test_derived_episode_key_needs_the_log_key_not_the_receipt(tmp_path, qa_engine):
    """With a secret the episode key is derived, so a reader of the state directory cannot forge the log."""
    from mrgym import keys
    log_key = keys.sub(bytes.fromhex("ab" * 32), "log")
    cfg, s = _start(tmp_path, qa_engine, log_key=log_key)
    s.scan({"protocol": SPGR})
    receipt, events = s.seal(), list(s.events)
    s.close()
    assert "key" not in receipt and "key" not in session.read_head(s.head_path)
    assert session.replay(cfg, events, engine=qa_engine, head=receipt, log_key=log_key)["ok"]
    assert not session.replay(cfg, events, engine=qa_engine, head=receipt)["ok"]
    assert not session.replay(cfg, events, engine=qa_engine, head=receipt, log_key=keys.sub(bytes.fromhex("cd" * 32), "log"))["ok"]


def test_refill_invariant_over_random_attack_sequences(tmp_path, qa_engine):
    """Over random sequences of scan / stop + restart / cut / edit / swap-head: the scan seconds accepted over all
    incarnations never exceed the budget, unless the verifier says the log is not ok."""
    import random
    rng = random.Random(7)
    for trial in range(6):
        cfg, s = _start(tmp_path, qa_engine, name=f"t{trial}", budget_s=3 * 3.9)
        log_dir, state, head_path = os.path.dirname(s.log_path), os.path.dirname(s.head_path), s.head_path
        accepted, stale_head, dead = 0.0, None, False
        for _ in range(10):
            op = rng.choice(["scan", "scan", "scan", "restart", "cut", "edit", "swap_head"])
            if op == "scan":
                if s is None:
                    continue
                try:
                    accepted += s.scan({"protocol": SPGR})[0]["cost_s"]
                except session.ScannerError:
                    pass
                stale_head = stale_head or open(head_path).read()
                continue
            if s is not None:
                s.close()
                s = None
            if op == "cut" and os.path.exists(os.path.join(log_dir, session.LOG_NAME)):
                _cut(os.path.join(log_dir, session.LOG_NAME), rng.randrange(0, 2))
            elif op == "edit":
                ev = session.read_log(os.path.join(log_dir, session.LOG_NAME))
                if ev:
                    ev[0]["cost_s"] = 0.001
                    with open(os.path.join(log_dir, session.LOG_NAME), "w") as f:
                        f.writelines(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in ev)
            elif op == "swap_head" and stale_head:
                with open(head_path, "w") as f:
                    f.write(stale_head)
            try:
                s = session.ScannerSession(cfg, log_dir=log_dir, engine=qa_engine, state_dir=state, resume_from=head_path)
            except session.LogTampered:
                dead = True
                break
        if s is not None:
            s.close()
        rep = session.verify_log(cfg, log_dir, head_path, engine=qa_engine)
        assert accepted <= cfg["budget_s"] + 1e-6 or not rep["ok"], (trial, accepted, rep)
        assert dead or rep["ok"] or accepted <= cfg["budget_s"] + 1e-6
