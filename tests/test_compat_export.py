"""``mrgym export``: the generated Harbor tasks and OpenEnv manifest, and the isolation properties of a Harbor task
that can be checked without Docker."""
import glob
import json
import os
import re

import pytest

import mrgym
from mrgym import cli, compat
from mrgym.compat import export, harbor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORED = [i for i in mrgym.list_envs() if mrgym.spec(i).kind != "sandbox"]


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("datasets"))
    assert cli.main(["export", "harbor", "--out", out]) == 0
    found = export.harbor_tasks(out)
    assert len(found) == len(SCORED)                                 # one task per scored id of the registry, by discovery
    return out, found


def test_export_harbor_is_current_lints_clean_detects_an_edit_and_refresh_restores_it(tasks):
    out, found = tasks
    assert export.check_harbor(out) == []
    for t in found:
        assert harbor.lint(t) == [], t
    victim = os.path.join(found[0], "instruction.md")
    seed = open(os.path.join(found[0], "tests", "episode.json")).read()
    open(victim, "a").write("an edit\n")
    problems = export.check_harbor(out)
    assert any("differs" in p for p in problems) and any("--refresh" in p for p in problems)
    assert cli.main(["export", "harbor", "--out", out, "--refresh"]) == 0     # rendered again from the stored seeds
    assert export.check_harbor(out) == [] and export.harbor_tasks(out) == found
    assert open(os.path.join(found[0], "tests", "episode.json")).read() == seed


def test_check_and_refresh_are_functions_of_the_stored_seed_and_token_also_in_another_process(tasks):
    """The text of a task must not depend on anything a process draws at random (an episode handle, a trainer-side
    label on a secret split): a second interpreter renders the same files."""
    import subprocess
    import sys
    out, found = tasks
    r = subprocess.run([sys.executable, "-m", "mrgym.cli", "export", "harbor", "--out", out, "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:]


def test_the_hidden_seed_stays_out_of_everything_the_agent_can_read(tasks):
    for t in tasks[1]:
        seed = json.load(open(os.path.join(t, "tests", "episode.json")))["seed"]
        assert re.fullmatch(r"[0-9a-f]{32}", seed)
        for dirpath, _, names in os.walk(t):
            for n in names:
                rel = os.path.relpath(os.path.join(dirpath, n), t)
                if rel in ("tests/episode.json", "environment/scanner/episode.json"):
                    continue
                assert seed not in open(os.path.join(dirpath, n)).read(), rel
        assert seed[:6] not in os.path.basename(t)
        ignored = open(os.path.join(t, "environment", ".dockerignore")).read().split()
        assert "scanner" in ignored                                  # the agent's build context excludes the episode file
        assert compat.CANARY in open(os.path.join(t, "task.toml")).read()
        for rel in ("instruction.md", "environment/Dockerfile"):         # what the agent reads never names the benchmark
            assert compat.CANARY not in open(os.path.join(t, rel)).read()


def test_two_exports_share_neither_a_seed_nor_a_token_and_nothing_public_is_a_function_of_the_seed():
    a, b = export.new_episode(SCORED[0], 1), export.new_episode(SCORED[0], 1)
    ta, tb = export.new_token(), export.new_token()
    assert a["seed"] != b["seed"] and ta != tb and len(ta) >= 32
    assert set(a) == set(harbor.EPISODE_KEYS) and a["split"] == harbor.SPLIT       # the form `mrgym serve --episode-file` reads
    # the same seed with another token gives another directory name and no file outside the episode files changes with the seed
    assert export.task_dirname(a, ta) != export.task_dirname(a, tb)
    one, other = export.render_task(a, ta), export.render_task(dict(a, seed=b["seed"]), ta)
    statement_differs = one["instruction.md"] != other["instruction.md"]            # the statement may depend on the world
    changed = {k for k in one if one[k] != other[k]} - {"tests/episode.json", "environment/scanner/episode.json"}
    assert changed <= ({"instruction.md"} if statement_differs else set())


def test_lint_flags_a_leak_a_shared_verifier_an_unpinned_install_and_a_canary_the_agent_reads(tasks, tmp_path):
    import shutil
    t = str(tmp_path / "task")
    shutil.copytree(tasks[1][0], t)
    seed = json.load(open(os.path.join(t, "tests", "episode.json")))["seed"]
    open(os.path.join(t, "instruction.md"), "a").write(f"\nseed {seed}\n{compat.CANARY}\n")
    open(os.path.join(t, "environment", ".dockerignore"), "w").write("")
    toml, docker = os.path.join(t, "task.toml"), os.path.join(t, "tests", "Dockerfile")
    edited = open(toml).read().replace('environment_mode = "separate"', 'environment_mode = "shared"').replace(harbor.STATE_DIR, "/var/lib/other")
    open(toml, "w").write(edited)
    unpinned = re.sub(r"mrgym==\S+", "mrgym", open(docker).read())
    open(docker, "w").write(unpinned)
    rules = {r for r, _ in harbor.lint(t)}
    assert {"seed-leak", "agent-context-leak", "shared-verifier", "unpinned-python-dependency", "canary-in-agent-text", "artifacts"} <= rules


def _native(task_dir):
    ep = harbor.load_episode(os.path.join(task_dir, "tests", "episode.json"))
    return ep, compat.scripted_episode(ep["env_id"], ep["level"], ep["seed"], split=ep["split"])


def test_the_service_wants_the_token_and_serves_nothing_derived_from_the_seed(tasks, tmp_path):
    ep = harbor.load_episode(os.path.join(tasks[1][0], "tests", "episode.json"))
    with harbor.LocalService(ep, str(tmp_path / "log"), str(tmp_path / "state"), "t" * 40) as srv:
        from mrgym.scanner import session as core
        with pytest.raises(core.ScannerError):
            harbor.HttpSession(srv.url, "wrong-token").info()
        http = harbor.HttpSession(srv.url, "t" * 40)
        blob = json.dumps([http.info(), http._call("GET", "/v1/task"), http.budget()])
        assert ep["seed"] not in blob and "task_id" not in blob
        for route in ("/v1/reset", "/v1/world", "/v1/episode", "/v1/admin/episodes"):
            with pytest.raises(core.ScannerError):
                http._call("GET", route)


def _played(task, tmp_path, name, policy="reference"):
    """One trial of ``task`` kept on disk: (verifier's episode file, log dir, state dir, answer path)."""
    work = tmp_path / name
    harbor.local(task, policy=policy, work_dir=str(work))
    return os.path.join(task, "tests", "episode.json"), str(work / "log"), str(work / "state"), str(work / "app" / "answer.json")


def test_the_verifier_gives_the_floor_for_a_log_that_is_cut_edited_foreign_unanchored_or_unsealed(tasks, tmp_path):
    import shutil
    t = next((x for x in tasks[1] if _native(x)[1]["passed"]), None)
    if t is None:
        pytest.skip(f"no scored id has a scripted reference that passes natively on {mrgym.ENGINE_ID}")
    ep_path, log_dir, state_dir, answer = _played(t, tmp_path, "good")
    good = harbor.verify(ep_path, log_dir, state_dir, answer, str(tmp_path / "v" / "reward.json"))
    assert good["passed"] and good["sealed"] and json.load(open(tmp_path / "v" / "reward.json")) == {"reward": good["reward"]}
    assert open(tmp_path / "v" / "reward.txt").read().strip() == f"{good['reward']:g}"
    log = glob.glob(os.path.join(log_dir, "*", "scan_log.jsonl"))[0]
    lines = open(log).read().splitlines()
    assert len(lines) >= 2

    def reward_after(new_lines):
        open(log, "w").write("\n".join(new_lines) + "\n")
        r = harbor.verify(ep_path, log_dir, state_dir, answer)
        open(log, "w").write("\n".join(lines) + "\n")
        return r

    cut = reward_after(lines[:-1])                                       # the seal removed: shorter than its head
    assert cut["reward"] == 0.0 and not cut["log_replays"]
    assert reward_after(lines[1:])["reward"] == 0.0                       # the first record removed
    rec = json.loads(lines[0])
    rec["wall"] = dict(rec.get("wall", {}), ms=0.0)                       # outside the chain and the MAC: still the same log
    assert reward_after([json.dumps(rec)] + lines[1:])["reward"] == good["reward"]
    for key in ("cost_s", "data_sha256", "request"):
        if key in rec:
            forged = dict(rec, **{key: 0.0 if key == "cost_s" else "0" * 64 if key == "data_sha256" else {}})
            assert reward_after([json.dumps(forged)] + lines[1:])["reward"] == 0.0, key
    # the head: missing, or the head of another episode
    empty = str(tmp_path / "no_state")
    os.makedirs(empty)
    none = harbor.verify(ep_path, log_dir, empty, answer)
    assert none["reward"] == 0.0 and "head receipt" in none["problems"][0]
    _, log2, state2, _ = _played(t, tmp_path, "second")
    mixed = str(tmp_path / "mixed_state")
    shutil.copytree(state2, mixed)
    assert harbor.verify(ep_path, log_dir, mixed, answer)["reward"] == 0.0
    # a deliverable that was written but never received by the scanner: the log is not sealed
    def scans_then_only_writes(url, token, answer_path):
        os.makedirs(os.path.dirname(answer_path))
        shutil.copy(answer, answer_path)

    _, log3, state3, answer3 = _played(t, tmp_path, "unsealed", policy=scans_then_only_writes)
    unsealed = harbor.verify(ep_path, log3, state3, answer3)
    assert unsealed["reward"] == 0.0 and not unsealed["sealed"] and any("never sealed" in p for p in unsealed["problems"])
    for content in ("not json", "[1, 2]", "{}"):
        open(answer, "w").write(content)
        r = harbor.verify(ep_path, log_dir, state_dir, answer)
        assert r["reward"] == 0.0 and not r["passed"], content


def test_the_command_of_the_sidecar_image_starts_the_service_and_the_oracle_plays_against_it(tasks, tmp_path):
    """The exact ``mrgym serve`` command line of the generated sidecar Dockerfile, as a process, with the container's
    paths replaced by local ones; then the two commands of solve.sh and test.sh."""
    import shutil
    import subprocess
    import sys
    import time
    t = tasks[1][0]
    ep, native = _native(t)
    cmd = json.loads(re.search(r"^CMD (\[.*\])$", open(os.path.join(t, "environment", "scanner", "Dockerfile")).read(), re.M)[1])
    assert cmd[:2] == ["mrgym", "serve"] and "--state-dir" in cmd and "--agent-token-file" in cmd
    secret_dir = tmp_path / "episode"
    secret_dir.mkdir()
    for name in ("episode.json", "agent_token"):
        shutil.copy(os.path.join(t, "environment", "scanner", name), secret_dir / name)
        os.chmod(secret_dir / name, 0o600)                               # what `COPY --chmod=600` does in the image
    local_paths = {"/episode/episode.json": str(secret_dir / "episode.json"), "/episode/agent_token": str(secret_dir / "agent_token"),
                   harbor.LOG_DIR: str(tmp_path / "log"), harbor.STATE_DIR: str(tmp_path / "state"), "0.0.0.0": "127.0.0.1", "8000": "0"}
    argv = [sys.executable, "-m", "mrgym.cli"] + [local_paths.get(a, a) for a in cmd[1:]] + ["--ready-file", str(tmp_path / "ready.json")]
    env = {k: v for k, v in os.environ.items() if k not in ("MRGYM_SECRET", "MRGYM_SALT")}
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 300                                    # the service builds the world first: seconds, family-dependent
        while not (tmp_path / "ready.json").exists() and proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        assert proc.poll() is None, proc.stderr.read().decode()[-500:]
        assert (tmp_path / "ready.json").exists(), "the service did not come up within 300 s"
        port = json.load(open(tmp_path / "ready.json"))["port"]
        client_env = dict(env, MRSCAN_URL=f"http://127.0.0.1:{port}", MRSCAN_TOKEN_FILE=str(secret_dir / "agent_token"))
        answer = str(tmp_path / "app" / "answer.json")
        solved = subprocess.run([sys.executable, "-m", "mrgym.compat.harbor", "solve", "--env-id", ep["env_id"], "--out", answer],
                                env=client_env, capture_output=True, text=True, timeout=600)
        assert solved.returncode == 0, solved.stderr[-500:]
    finally:
        proc.terminate()
        proc.wait(timeout=20)
    out = subprocess.run([sys.executable, "-m", "mrgym.compat.harbor", "verify", "--episode-file", os.path.join(t, "tests", "episode.json"),
                          "--log-dir", str(tmp_path / "log"), "--state-dir", str(tmp_path / "state"), "--answer", answer,
                          "--reward-json", str(tmp_path / "logs" / "reward.json")], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert json.load(open(tmp_path / "logs" / "reward.json")) == {"reward": native["reward"]}
    assert json.loads(out.stdout)["log_head"] == native["log_head"]


def test_the_committed_harbor_sample_and_openenv_package_are_current():
    datasets = os.path.join(ROOT, "datasets")
    if not os.path.isdir(datasets):
        pytest.skip("needs a checkout: datasets/ is not part of the wheel")
    assert export.check_harbor(datasets) == []
    for t in export.harbor_tasks(datasets):
        assert harbor.lint(t) == [], t
    assert export.check_openenv(os.path.join(ROOT, "integrations", "openenv", "mrgym_scanner_env")) == []


def test_export_openenv_declares_the_tools_of_every_id(tmp_path):
    for env_id in mrgym.list_envs():
        files = export.render_openenv(env_id)
        env = mrgym.make(env_id, seed=0)
        env.reset()
        names = [t["name"] for t in env.tools("neutral")]
        assert f"declared_tools: [{', '.join(names)}]" in files["openenv.yaml"]
        plan = json.loads(files["validation/runtime.json"])
        assert plan["reset"]["options"]["env_id"] == env_id and len(plan["actions"]) <= 100
        assert all(a["tool_name"] in names for a in plan["actions"])
        assert f"mrgym=={mrgym.__version__}" in files["server/requirements.txt"]
