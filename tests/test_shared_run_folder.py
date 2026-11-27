"""Several processes that write into one run folder: what is guaranteed, under interleavings built with gates.

The run folder is how a preempted job continues and how evaluation processes share work, so the guarantee is stated
on what is on disk and on what the processes report together:

  * every episode asked for has exactly one completed record, with the conversation that belongs to it
  * an episode is played by one process at a time, and never again once it has a completed record: the numbers of
    episodes the processes say they played add up to the number asked for
  * a process that splits the seeds with others reports its own share, whatever the others have finished

Shares of the seeds that do not overlap (``integrations/verl/node/calibrate.sh``) give each episode one owner. Shares
that overlap are settled by the claim of ``mrgym.agents.runner._claim`` (the runner) and by the exclusive folder of
``mrgym_verl.episode.free_dir`` (the training adapter). Each test holds a process or a thread at the point where the
code without the claim played an episode twice, replaced a completed record or wrote two episodes into one folder.
"""
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from mrgym import calibrate
from mrgym import record as mrecord
from mrgym.agents import chat, runner, stub_server, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERL = os.path.join(ROOT, "integrations", "verl")
sys.path.insert(0, VERL)

from mrgym_verl import calsum  # noqa: E402
from mrgym_verl.episode import Episode, episode_dir, free_dir, play_chat  # noqa: E402

NOWRAP = "Scanner/NoWrap-v1"
MODEL = "stub:reference"
FAST = dict(backoff_s=0.01, max_retries=2)
WAIT_S = 120                                                              # a gate that is never opened fails the test; nothing is tried twice

# `mrgym <arguments>` with gates on the write of a completed record: HOLDS = [[end of the record's path, what to wait
# for]]. What is waited for is a file, or a record.json that must be completed. `<what>.held` says the gate is reached.
GATED_CLI = r"""
import json, os, sys, time
from mrgym import cli
from mrgym.agents import runner
HOLDS, real = json.loads(os.environ["HOLDS"]), runner.write_json
def there(path):
    return (runner._read(path) or {}).get("status") == "completed" if path.endswith("record.json") else os.path.exists(path)
def write_json(path, obj, **kw):
    if isinstance(obj, dict) and obj.get("status") == "completed":
        for end, what in HOLDS:
            if path.endswith(end):
                open(what + ".held", "w").close()
                t0 = time.time()
                while not there(what):
                    if time.time() - t0 > float(os.environ["WAIT_S"]):
                        raise SystemExit("the gate was never opened")
                    time.sleep(0.02)
    real(path, obj, **kw)
runner.write_json = write_json
raise SystemExit(cli.main(sys.argv[1:]))
"""


@pytest.fixture(scope="module")
def stub():
    httpd, url = stub_server.start(seeds=range(4), env_ids=[NOWRAP], levels=[2])
    yield url
    httpd.shutdown()


def _record_path(out, seed, episode=0, agent=MODEL):
    return os.path.join(out, "episodes", runner.slug(agent), runner.env_slug(NOWRAP), "L2", f"s{seed}", f"e{episode}", "record.json")


def _stamp(path):
    st = os.stat(path)
    return st.st_ino, st.st_mtime_ns


def _whole(out, n):
    """The folder holds exactly ``n`` episodes, each once, completed, with its own conversation; no claim is left."""
    records = summarize.load(out)
    assert len(records) == n == len({(r["agent"], r["level"], r["seed"], r["_dir"]) for r in records})
    assert all(r["status"] == "completed" and r["passed"] for r in records), [(r["_dir"], r["status"]) for r in records]
    assert [mrecord.transcript_problem(r, os.path.join(out, r["_dir"])) for r in records] == [None] * n
    assert not glob.glob(os.path.join(out, "**", "*.tmp"), recursive=True)
    return records


def _until(condition, what):
    t0 = time.time()
    while not condition():
        assert time.time() - t0 < WAIT_S, what
        time.sleep(0.02)


class Gates:
    """Holds a named thread at ``runner.<point>`` until the test opens its gate. ``make_env`` is where an episode
    starts to be played (its folder is taken, its first record written); ``_note_invocation`` is where a call has
    listed its episodes and has not started one."""

    def __init__(self, monkeypatch, point, names):
        self.reached, self.open = {n: threading.Event() for n in names}, {n: threading.Event() for n in names}
        real = getattr(runner, point)

        def gated(*a, **kw):
            name = threading.current_thread().name
            if name in self.reached:
                self.reached[name].set()
                assert self.open[name].wait(WAIT_S), f"the gate of {name} was never opened"
            return real(*a, **kw)
        monkeypatch.setattr(runner, point, gated)


def _call(results, name, out, stub, seeds, **kw):
    """One runner call (what one process does) in a thread of its own; the result, or the exception, under ``name``."""
    def body():
        try:
            results[name] = runner.run(out, [NOWRAP], levels=[2], seeds=seeds, models=[MODEL], base_url=stub, max_tokens=4096, quiet=True,
                                       **FAST, **kw)
        except BaseException as e:  # noqa: BLE001  shown by the assertion of the test
            results[name] = e
    t = threading.Thread(target=body, name=name)
    t.start()
    return t


def _counts(result):
    assert isinstance(result, dict), result
    return {k: result[k] for k in ("started", "already_finished", "infra_errors", "played_elsewhere")}


# ------------------------------------------------------------------ shares that do not overlap: the report of a process
def test_a_process_reports_its_own_share_whatever_the_other_has_finished(stub, tmp_path):
    """Two ``mrgym calibrate`` processes, one folder, a seed each. A is held until B has finished one of its two
    episodes; B's second is held until A has reported. A used to count what it found in the folder at that moment:
    three episodes of two instances under ``seeds: 0-0``."""
    cal, release = str(tmp_path / "cal"), str(tmp_path / "release")
    mine = lambda seed, episode: _record_path("", seed, episode)         # noqa: E731  the end of the path of a record
    holds = {0: [[mine(0, 1), _record_path(cal, 1, 0)], [mine(0, 1), release + ".held"]],    # A: until B has one finished and one at its end
             1: [[mine(1, 1), release]]}                                                      # B: until A has reported
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), OMP_NUM_THREADS="1", WAIT_S=str(WAIT_S))
    args = ["calibrate", NOWRAP, "--model", MODEL, "--base-url", stub, "--levels", "2", "--n", "1", "--episodes", "2", "--max-turns", "0",
            "--workers", "2", "--out", cal, "--json"]
    procs = [subprocess.Popen([sys.executable, "-c", GATED_CLI, *args, "--first-seed", str(k)], env=dict(env, HOLDS=json.dumps(holds[k])),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for k in (0, 1)]
    try:
        out_a, err_a = procs[0].communicate(timeout=WAIT_S)
        on_disk = sorted((r["seed"], r["episode"], r["status"]) for r in map(runner._read, glob.glob(_record_path(cal, "*", "*"))))
        assert on_disk == [(0, 0, "completed"), (0, 1, "completed"), (1, 0, "completed"), (1, 1, "infra_error")], err_a   # the moment A looked
        open(release, "w").close()
        out_b, err_b = procs[1].communicate(timeout=WAIT_S)
    finally:
        for p in procs:
            p.kill()
    assert [p.returncode for p in procs] == [1, 1], (err_a, err_b)        # 1: no entry rung (the reference passes); a crash would be 2 or a trace
    for k, text in enumerate((out_a, out_b)):
        result = json.loads(text)
        row = result["calibration"][0]
        assert (row["n_instances"], row["n_episodes"], row["passed"]) == (1, 2, 2) and result["infra_errors"] == 0, row
        assert result["settings"]["seeds"] == f"{k}-{k}"                  # the row is of the seeds its settings name
    _whole(cal, 4)
    with open(os.path.join(cal, "run_config.json")) as f:
        assert len(json.load(f)["invocations"]) == 2
    s = calsum.summarize(cal, MODEL, [(NOWRAP, 2)], 2, 2, 0)
    assert s["complete"] and s["episodes"] == 4


# ------------------------------------------------------------------ shares that overlap: the claim of the runner
def test_an_episode_one_process_is_playing_is_left_to_it(stub, tmp_path, monkeypatch):
    out, results = str(tmp_path / "run"), {}
    gates = Gates(monkeypatch, "make_env", ["first"])
    first = _call(results, "first", out, stub, [0])
    assert gates.reached["first"].wait(WAIT_S), results
    path = _record_path(out, 0)
    started = _stamp(path)
    assert runner._read(path)["stop_reason"] == "interrupted" and os.path.exists(os.path.dirname(path) + runner.CLAIM_SUFFIX)
    second = _call(results, "second", out, stub, [0])                    # lists the episode (no completed record) while it is being played
    second.join(WAIT_S)
    assert _counts(results["second"]) == dict(started=0, already_finished=0, infra_errors=0, played_elsewhere=1)
    assert _stamp(path) == started                                       # the folder of the running episode was not touched
    gates.open["first"].set()
    first.join(WAIT_S)
    assert _counts(results["first"]) == dict(started=1, already_finished=0, infra_errors=0, played_elsewhere=0)
    _whole(out, 1)
    with open(os.path.join(out, "run_config.json")) as f:
        assert len(json.load(f)["invocations"]) == 2


def test_an_episode_completed_after_it_was_listed_is_not_played_again(stub, tmp_path, monkeypatch):
    """Both calls list the episode as missing before either plays. The first plays it to the end; the second then
    gets to it. It used to delete the completed record and play the episode again."""
    out, results = str(tmp_path / "run"), {}
    gates = Gates(monkeypatch, "_note_invocation", ["first", "second"])
    calls = {name: _call(results, name, out, stub, [0, 1]) for name in ("first", "second")}
    assert gates.reached["first"].wait(WAIT_S) and gates.reached["second"].wait(WAIT_S), results
    gates.open["first"].set()
    calls["first"].join(WAIT_S)
    assert _counts(results["first"]) == dict(started=2, already_finished=0, infra_errors=0, played_elsewhere=0)
    stamps = [_stamp(_record_path(out, seed)) for seed in (0, 1)]
    gates.open["second"].set()
    calls["second"].join(WAIT_S)
    assert _counts(results["second"]) == dict(started=0, already_finished=0, infra_errors=0, played_elsewhere=2)
    assert [_stamp(_record_path(out, seed)) for seed in (0, 1)] == stamps
    _whole(out, 2)


def test_the_claim_holds_between_processes_and_ends_with_a_killed_process(stub, tmp_path):
    out, never = str(tmp_path / "run"), str(tmp_path / "never")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), OMP_NUM_THREADS="1", WAIT_S=str(WAIT_S),
               HOLDS=json.dumps([[_record_path("", 0), never]]))
    args = ["run-agent", "--env", NOWRAP, "--levels", "2", "--seeds", "0", "--model", MODEL, "--base-url", stub, "--max-tokens", "4096", "--out", out]
    proc = subprocess.Popen([sys.executable, "-c", GATED_CLI, *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _until(lambda: os.path.exists(never + ".held") or proc.poll() is not None, "the process never reached its last write")
        assert proc.poll() is None, proc.communicate()
        kw = dict(levels=[2], seeds=[0], models=[MODEL], base_url=stub, max_tokens=4096, quiet=True, **FAST)
        while_alive = runner.run(out, [NOWRAP], **kw)                    # the other process holds the episode
        assert _counts(while_alive) == dict(started=0, already_finished=0, infra_errors=0, played_elsewhere=1)
        assert runner._read(_record_path(out, 0))["stop_reason"] == "interrupted"
        proc.send_signal(signal.SIGKILL)
        proc.communicate(timeout=WAIT_S)
    finally:
        proc.kill()
    claim = os.path.dirname(_record_path(out, 0)) + runner.CLAIM_SUFFIX
    assert os.path.exists(claim) and runner._read(_record_path(out, 0))["status"] == "infra_error"      # what a kill leaves
    after = runner.run(out, [NOWRAP], **kw)                              # the lock went with the process: nothing to clear
    assert _counts(after) == dict(started=1, already_finished=0, infra_errors=0, played_elsewhere=0)
    assert _counts(runner.run(out, [NOWRAP], **kw)) == dict(started=0, already_finished=1, infra_errors=0, played_elsewhere=0)
    _whole(out, 1)


def test_calls_whose_shares_overlap_play_every_episode_once(stub, tmp_path):
    """Three calls at once on shares that overlap in every seed: no gate, whatever order the machine gives. Six
    episodes are asked for; six are played, by whoever got each. (One worker per call: three episodes at a time is
    what this test may hold in memory.)"""
    out, results = str(tmp_path / "run"), {}
    shares = dict(a=[0, 1], b=[1, 2], c=[0, 2])
    calls = [_call(results, name, out, stub, seeds, episodes=2) for name, seeds in shares.items()]
    for t in calls:
        t.join(WAIT_S)
    counts = {name: _counts(results[name]) for name in shares}
    assert sum(c["started"] for c in counts.values()) == 6 and not any(c["infra_errors"] for c in counts.values()), counts
    for name, seeds in shares.items():                                   # each listed episode: played here, finished before, or left to another call
        assert sum(counts[name][k] for k in ("started", "already_finished", "played_elsewhere")) == 2 * len(seeds), counts
    records = _whole(out, 6)
    assert sorted((r["seed"], r["episode"]) for r in records) == [(s, e) for s in range(3) for e in range(2)]
    played = sorted((r["seed"], r["episode"]) for name in shares for r in results[name]["records"])
    assert played == sorted((r["seed"], r["episode"]) for r in records)   # no episode in the reports of two calls
    row = calibrate.level_row(calibrate.read_records(out, NOWRAP, MODEL), 2)
    assert (row["n_instances"], row["n_episodes"]) == (3, 6)


# ------------------------------------------------------------------ a folder the code before the claim wrote
def test_a_folder_written_before_there_was_a_claim_is_continued(stub, tmp_path):
    out = str(tmp_path / "run")
    kw = dict(levels=[2], models=[MODEL], base_url=stub, temperature=1.0, max_tokens=4096, quiet=True, **FAST)   # the settings of a calibration
    runner.run(out, [NOWRAP], seeds=[0, 1], **kw)
    files = sorted(os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs)
    episodes = [os.path.dirname(os.path.relpath(_record_path(out, seed), out)) for seed in (0, 1)]
    assert files == sorted([f"{e}/{name}" for e in episodes for name in ("messages.json", "record.json", "scanner_log.jsonl")]
                           + ["run_config.json", "run_config.json.lock"])  # the files of a run are the ones they were: a claim leaves none
    interrupted = dict(runner._read(_record_path(out, 0)), seed=2, status="infra_error", stop_reason="interrupted",
                       infra_error="the runner stopped before this episode ended")
    os.makedirs(os.path.dirname(_record_path(out, 2)))
    runner.write_json(_record_path(out, 2), interrupted, indent=1)       # what a killed run left: a started episode
    before = {seed: open(_record_path(out, seed), "rb").read() for seed in (0, 1)}
    again = runner.run(out, [NOWRAP], seeds=[0, 1, 2], **kw)
    assert _counts(again) == dict(started=1, already_finished=2, infra_errors=0, played_elsewhere=0)
    assert {seed: open(_record_path(out, seed), "rb").read() for seed in (0, 1)} == before
    _whole(out, 3)
    rows = {n: calibrate.run(NOWRAP, MODEL, stub, out, levels=[2], n=n, episodes=1, first_seed=0, max_turns=0, max_tokens=4096, quiet=True, **FAST)
            for n in (2, 3)}
    assert [(rows[n]["calibration"][0]["n_instances"], rows[n]["calibration"][0]["n_episodes"]) for n in (2, 3)] == [(2, 2), (3, 3)]
    assert {seed: open(_record_path(out, seed), "rb").read() for seed in (0, 1)} == before          # read, not played again


# ------------------------------------------------------------------ the training adapter: one folder per episode
def _played(stub, seed=0):
    ep = Episode(NOWRAP, 2, seed, max_reply_tokens=4096)
    play_chat(ep, chat.ChatClient(stub, MODEL, None, dict(max_tokens=4096), **FAST))
    return ep


def _write(ep, directory, uid, session=0):
    return ep.write_record(directory, agent="run", model=MODEL, sampler=dict(max_tokens=4096), episode=session,
                           training=dict(phase="dev", step=2, uid=uid, session=session))


def test_two_episodes_that_ask_for_one_folder_at_one_moment_get_two(stub, tmp_path):
    uid = "0123456789abcdef"
    for k in range(20):                                                  # eight askers leave a barrier together
        d, got, barrier = episode_dir(str(tmp_path / f"r{k}"), "run", NOWRAP, 2, 7, 0), [], threading.Barrier(8)

        def ask(d=d, got=got, barrier=barrier):
            barrier.wait(WAIT_S)
            got.append(free_dir(d, uid))
        threads = [threading.Thread(target=ask) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(WAIT_S)
        assert sorted(os.path.basename(x) for x in got) == ["e0", "e0-01234567"] + [f"e0-01234567-{j}" for j in range(2, 8)]
        assert all(os.path.isdir(x) for x in got)
    # the whole write: two finished episodes of one instance, one session number and one group, stored at the same moment
    out, eps, barrier = str(tmp_path / "records"), [_played(stub), _played(stub)], threading.Barrier(2)
    d = episode_dir(out, "run", NOWRAP, 2, 0, 0)

    def store(ep):
        barrier.wait(WAIT_S)
        _write(ep, free_dir(d, uid), uid)
    threads = [threading.Thread(target=store, args=(ep,)) for ep in eps]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT_S)
    for ep in eps:
        ep.close()
    assert sorted(r["_dir"].rsplit("/", 1)[1] for r in _whole(out, 2)) == ["e0", "e0-01234567"]


def test_a_training_folder_written_before_is_read_and_added_to(stub, tmp_path):
    """The layout the adapter wrote before: ``e<session>``, and ``e<session>-<uid prefix>`` for the same instance and
    session played again after a restore. Reading it has not changed; a new episode never replaces one of them."""
    out = str(tmp_path / "records")
    d = episode_dir(out, "run", NOWRAP, 2, 0, 0)
    old_uid, new_uid = "aaaaaaaa11111111", "bbbbbbbb22222222"
    for folder in (d, f"{d}-{old_uid[:8]}"):                             # written as the code before did: into a name it only looked up
        ep = _played(stub)
        _write(ep, folder, old_uid)
        ep.close()
    os.makedirs(episode_dir(out, "run", NOWRAP, 2, 0, 1))                # a writer killed before its record: a folder without one
    before = {p: open(p, "rb").read() for p in glob.glob(os.path.join(out, "**", "*.json*"), recursive=True)}
    assert len(before) == 7 and len(_whole(out, 2)) == 2                 # 2 x (record, conversation, scanner log) + run_config.json
    names = []
    for uid, session in ((new_uid, 0), (old_uid, 0), (old_uid, 0), (new_uid, 1), ("Scanner/NoWrap-v1|2|0|dev", 0)):
        ep = _played(stub)
        names.append(os.path.basename(free_dir(episode_dir(out, "run", NOWRAP, 2, 0, session), uid)))
        _write(ep, os.path.join(os.path.dirname(d), names[-1]), uid, session)
        ep.close()
    assert names == ["e0-bbbbbbbb", "e0-aaaaaaaa-2", "e0-aaaaaaaa-3", "e1-bbbbbbbb", "e0-Scanner"]
    assert {p: open(p, "rb").read() for p in before} == before           # nothing that was there has changed
    records = _whole(out, 7)
    assert sorted(r["training"]["uid"][:8] for r in records) == ["Scanner/"] + ["aaaaaaaa"] * 4 + ["bbbbbbbb"] * 2
