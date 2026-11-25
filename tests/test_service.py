"""The scanner service: an agent plane behind a token, an operator plane on a unix socket, nothing secret in argv or
the environment, the result out of the agent's reach."""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

import mrgym
from mrgym import keys
from mrgym.redteam import common, rows
from mrgym.scanner import client, service

NULL = "Scanner/NullTissue-v0"
SEED = 915273
TIMEOUT = 1.5


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """One service process: a start-up episode on a secret split, the operator plane on a unix socket."""
    tmp = tmp_path_factory.mktemp("service")
    key_file, op_file, sock = str(tmp / "secret.key"), str(tmp / "operator.token"), str(tmp / "admin.sock")
    keys.keygen(key_file)
    keys.keygen(op_file)
    spec = dict(env_id=NULL, level=2, split="test-iid", seed=SEED)
    proc, ready, paths = rows.start_service(str(tmp), "svc", spec, env=dict(MRGYM_SECRET_FILE=key_file),
                                            extra=("--operator-token-file", op_file, "--admin-socket", sock, "--request-timeout", str(TIMEOUT)))
    assert ready, proc.stderr.read().decode()
    handle = ready["episodes"][0]
    out = dict(proc=proc, url=f"http://127.0.0.1:{ready['port']}", port=ready["port"], handle=handle, admin=ready["admin"],
               operator=open(op_file).read().strip(), token=open(os.path.join(paths["state"], handle + ".token")).read().strip(),
               secret=open(key_file).read().strip(), key_file=key_file, paths=paths, spec=spec)
    yield out
    assert rows.stop_service(proc) in (0, -15)


def _status(served, method, path, token=None, body=None):
    return rows._http(served["url"] + path, method, token=token, body=body)[0]


def test_process_arguments_and_environment_hold_no_seed_and_no_secret(served):
    world_seed = keys.world_seed(NULL, 2, "test-iid", SEED, environ={}, master=bytes.fromhex(served["secret"]))
    cmdline = open(f"/proc/{served['proc'].pid}/cmdline", "rb").read().decode()
    assert "serve" in cmdline and "--seed" not in cmdline
    try:
        environ = open(f"/proc/{served['proc'].pid}/environ", "rb").read().decode()
    except OSError:
        environ = ""                                                     # not dumpable: the same user cannot read it at all
    for what, needle in (("integer seed", str(SEED)), ("world seed", world_seed), ("secret", served["secret"]),
                         ("agent token", served["token"]), ("operator token", served["operator"])):
        assert needle not in cmdline and needle not in environ, what


def test_every_agent_route_needs_the_token_and_operator_routes_are_not_on_the_agent_port(served):
    for path in ("/v1/info", "/v1/task", "/v1/budget"):
        assert _status(served, "GET", path) == 401
        assert _status(served, "GET", path, token="wrong" * 8) == 401
        assert _status(served, "GET", path, token=served["operator"]) == 401          # the operator token is not an agent token
        assert _status(served, "GET", path, token=served["token"]) == 200
    for path in ("/v1/plan", "/v1/scan", "/v1/submit"):
        assert _status(served, "POST", path, body=b"{}") == 401
    for path in ("/v1/admin/episodes", f"/v1/admin/episodes/{served['handle']}", "/v1/truth", "/v1/seed", "/v1/reset", "/v1/config",
                 "/v1/../v1/admin/episodes", "/"):
        assert _status(served, "GET", path, token=served["token"]) == 404, path
        assert _status(served, "GET", path, token=served["operator"]) == 404, path
    assert _status(served, "GET", "/v1/health") == 200
    assert service.admin_request(served["admin"], "wrong" * 8, "GET", "/v1/admin/episodes")[0] == 401
    assert service.admin_request(served["admin"], served["token"], "GET", "/v1/admin/episodes")[0] == 401
    status, body = service.admin_request(served["admin"], served["operator"], "GET", "/v1/admin/episodes")
    assert status == 200 and served["handle"] in body["episodes"]
    assert os.stat(served["admin"][5:]).st_mode & 0o777 == 0o600                    # the socket is the operator's alone


def test_hostile_bodies_get_400_on_a_connection_that_stays_open_and_are_not_counted(served):
    conn = http.client.HTTPConnection("127.0.0.1", served["port"], timeout=10)
    headers = {"Authorization": "Bearer " + served["token"], "Content-Type": "application/json"}
    bodies = [b"{", b"}", b"\xff\xfe\x00", b'{"a": NaN}', b'{"a": Infinity}', b"[" * 5000, b'{"a":' * 2000, b"nul", b"'x'",
              b'{"protocol": ' + b'{"a":' * 40 + b"1" + b"}" * 40 + b"}", b"[" + b"1," * 5000 + b"1]", b'{"a": 1} trailing',
              b"\x00\x01\x02", b'{"a": 1e999999999, "b": -}', b"{'a': 1}", b'{"\\ud800": }']
    assert len(bodies) == 16
    for path in ("/v1/plan", "/v1/scan", "/v1/submit"):
        for body in bodies:
            conn.request("POST", path, body=body, headers=headers)
            r = conn.getresponse()
            payload = json.loads(r.read())
            assert r.status == 400 and payload["error"]["type"] == "bad_request", (path, body[:20], r.status)
    conn.request("GET", "/v1/budget", headers=headers)                   # the same connection still answers
    budget = json.loads(conn.getresponse().read())
    conn.close()
    assert budget["requests"] == 0 and budget["scans"] == 0


def test_sixty_four_simultaneous_connections_all_get_a_status(served):
    out, gate = [], threading.Barrier(64)

    def one():
        gate.wait(timeout=20)
        out.append(_status(served, "GET", "/v1/budget", token=served["token"]))
    threads = [threading.Thread(target=one) for _ in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert out == [200] * 64


def test_stalled_connections_are_closed_and_do_not_move_the_request_counter(served):
    socks = []
    for i in range(40):
        s = socket.create_connection(("127.0.0.1", served["port"]), timeout=10)
        if i % 2:                                                        # half send nothing, half stop in the middle of a body
            s.sendall(b"POST /v1/scan HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer " + served["token"].encode()
                      + b"\r\nContent-Length: 500\r\n\r\n{\"protocol\"")
        socks.append(s)
    t0 = time.time()
    closed = 0
    for s in socks:
        s.settimeout(max(0.2, TIMEOUT + 4.0 - (time.time() - t0)))
        try:
            closed += s.recv(4096) == b""                                # an orderly close, without a response
        except (TimeoutError, ConnectionError):
            pass
        s.close()
    assert closed == 40 and time.time() - t0 < TIMEOUT + 5.0
    budget = json.loads(rows._http(served["url"] + "/v1/budget", token=served["token"])[1])
    assert budget["requests"] == 0


def test_reference_over_http_earns_the_in_process_reward_and_the_score_stays_with_the_operator(served, monkeypatch):
    """Episodes created on the operator plane, played with the single-file client, read back on the operator plane."""
    controls = common.Context(NULL).controls
    monkeypatch.setenv("MRGYM_SECRET_FILE", served["key_file"])
    for seed, policy in ((1, "reference"), (2, "reference"), (3, "prior")):
        spec = dict(env_id=NULL, level=2, split="test-iid", seed=seed)
        status, made = service.admin_request(served["admin"], served["operator"], "POST", "/v1/admin/episodes", spec)
        assert status == 200 and made["statement"].startswith("# Null") and made["canary"] == keys.CANARY
        sc = client.Scanner(served["url"], token=made["agent_token"])
        task = sc.task()
        assert task["statement"] == made["statement"] and task["submit"]["required"] == ["ti_ms"]
        answer = controls.policies()[policy](sc)
        with pytest.raises(client.ScannerError) as e:                    # not a deliverable: refused, the episode goes on
            sc.submit({"ti_ms": {}})
        assert e.value.type == "invalid_deliverable"
        got = sc.submit(answer)
        assert got == {"received": True}                                 # and nothing else: no score, no gate
        for call in (lambda: sc.submit(answer), lambda: sc.scan(dict(sequence="spgr", tr_ms=10.0, flip_deg=8.0, matrix=32))):
            with pytest.raises(client.ScannerError) as e:
                call()
            assert e.value.type == "episode_over"
        status, rep = service.admin_request(served["admin"], served["operator"], "GET", f"/v1/admin/episodes/{made['episode']}")
        local = mrgym.make(NULL, level=2, seed=seed, split="test-iid")
        from mrgym import policies
        expected = policies.run(local, policy)
        assert status == 200 and rep["status"] == "done" and rep["reward"] == expected.reward, (seed, policy)
        assert rep["replay"]["ok"] and rep["replay"]["sealed"] and rep["head"]["sealed"] and "key" not in rep["head"]
        text = json.dumps(rep)
        assert local.episode_config()["seed"] not in text and "task_id" not in rep and '"seed"' not in text
        if policy == "reference":
            assert rep["reward"] == 1.0 and rep["replay"]["scans"] == expected.info["replay"]["scans"]
    # a public split is refused unless the service was started for demos
    status, body = service.admin_request(served["admin"], served["operator"], "POST", "/v1/admin/episodes", dict(env_id=NULL, split="train", seed=0))
    assert status == 422 and "public" in body["error"]["message"]
    status, body = service.admin_request(served["admin"], served["operator"], "POST", "/v1/admin/episodes", dict(env_id=NULL, seed=0, extra=1))
    assert status == 422


def test_the_start_up_episode_can_be_closed_without_a_deliverable(served):
    status, rep = service.admin_request(served["admin"], served["operator"], "POST", f"/v1/admin/episodes/{served['handle']}/close")
    assert status == 200 and rep["status"] == "done" and rep["reward"] == 0.0 and rep["head"]["reason"] == "close"
    assert _status(served, "POST", "/v1/plan", token=served["token"], body=b'{"protocol": {"sequence": "spgr"}}') == 409


def _serve(tmp_path, name, spec, mode=0o600, extra=(), **env):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(spec))
    path.chmod(mode)
    full = {k: v for k, v in os.environ.items() if not k.startswith("MRGYM_")}
    full.update(env, OMP_NUM_THREADS="1")
    return subprocess.Popen([sys.executable, "-m", "mrgym.cli", "serve", "--episode-file", str(path), "--log-dir", str(tmp_path / name / "log"),
                             "--state-dir", str(tmp_path / name / "state"), "--port", "0", *extra],
                            env=full, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def test_the_service_refuses_to_start_when_a_boundary_is_missing(tmp_path):
    key = keys.keygen(str(tmp_path / "secret.key"))
    secret_spec = dict(env_id=NULL, split="test-iid", seed=1)
    cases = {
        "the secret in the environment": (_serve(tmp_path, "a", secret_spec, MRGYM_SECRET="ab" * 32), "MRGYM_SECRET is set"),
        "the agent has the service's uid": (_serve(tmp_path, "b", secret_spec, extra=("--require-isolation", "--agent-uid", str(os.geteuid())),
                                                   MRGYM_SECRET_FILE=key), "isolation check failed"),
        "a public split without the demo flag": (_serve(tmp_path, "c", dict(env_id=NULL, split="train", seed=1)), "is public"),
        "an episode file others can read": (_serve(tmp_path, "d", secret_spec, mode=0o644, MRGYM_SECRET_FILE=key), "chmod 600"),
        "a secret split without a secret": (_serve(tmp_path, "e", secret_spec), "needs a secret"),
        "a seed on the command line": (_serve(tmp_path, "f", secret_spec, extra=("--seed", "3"), MRGYM_SECRET_FILE=key), "unrecognized arguments"),
    }
    for what, (proc, message) in cases.items():
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 2 and message in err.decode(), (what, proc.returncode, err.decode()[-300:])


def test_service_attacks_from_the_same_user(tmp_path):
    """The suite's service rows: /proc, the HTTP surface, the bodies it returns, --require-isolation, and the budget
    refill by cutting the log and restarting (exit status 3)."""
    got = {r.id: r for r in rows.rt21_service(common.Context(NULL), str(tmp_path))}
    assert set(got) == {"RT21", "RT21b", "RT16b", "RT21c", "RT20c"}
    assert all(r.ok for r in got.values()), [(r.id, r.observed) for r in got.values() if not r.ok]
    assert "exit 3" in got["RT20c"].observed and "exits 2" in got["RT21c"].observed
