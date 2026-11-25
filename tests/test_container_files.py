"""The container tier is written, not run (no Docker daemon where this suite was developed). These tests parse the
files and check the properties the isolation rests on, so that an edit cannot quietly remove one."""
import json
import os
import py_compile
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "integrations", "container")
pytestmark = pytest.mark.skipif(not os.path.isdir(DIR), reason="the tests are not running from a checkout of the repository")


def _compose():
    with open(os.path.join(DIR, "compose.yaml"), encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)                                          # the file is written in the JSON subset of YAML


def _volume_names(service):
    return {v.split(":")[0] for v in service.get("volumes", []) if not v.startswith((".", "/"))}


def test_compose_separates_scanner_and_agent():
    c = _compose()
    scanner, agent, operator = (c["services"][k] for k in ("scanner", "agent", "operator"))
    assert not _volume_names(scanner) & _volume_names(agent) and _volume_names(agent) == set()
    assert agent["secrets"] == ["agent_token"] and {"episode", "mrgym_secret", "operator_token"} <= set(scanner["secrets"])
    assert all(v.endswith(":ro") for v in agent["volumes"])
    assert c["networks"]["episode"] == {"internal": True} and scanner["networks"] == agent["networks"] == ["episode"]
    assert operator["network_mode"] == "none" and "scanadmin" in _volume_names(operator) and operator["secrets"] == ["operator_token"]
    users = [s["user"] for s in (scanner, agent)]
    assert len(set(users)) == 2 and all(not u.startswith("0") and u != "root" for u in users)
    for s in (scanner, agent, operator):
        assert s["read_only"] is True and s["cap_drop"] == ["ALL"] and s["security_opt"] == ["no-new-privileges:true"]
        assert "privileged" not in s and "pid" not in s and "ports" not in s


def test_nothing_secret_is_in_a_command_line_or_an_environment():
    c = _compose()
    cmd = c["services"]["scanner"]["command"]
    assert cmd[0] == "serve" and "--seed" not in cmd and cmd[cmd.index("--episode-file") + 1].startswith("/run/secrets/")
    assert cmd[cmd.index("--state-dir") + 1] != cmd[cmd.index("--log-dir") + 1]
    import argparse

    from mrgym.scanner import service
    ap = argparse.ArgumentParser()
    service.add_arguments(ap)
    ap.parse_args(cmd[1:])                                               # every flag of the compose command exists
    for s in c["services"].values():
        env = s.get("environment", {})
        assert "MRGYM_SECRET" not in env and "MRGYM_SALT" not in env and "MRSCAN_TOKEN" not in env
        assert all(k.endswith("_FILE") or k in ("OMP_NUM_THREADS", "MRSCAN_URL", "MRGYM_ADMIN") for k in env), env
    assert set(c["secrets"]) == {"episode", "mrgym_secret", "operator_token", "agent_token"}
    assert all(v["file"].startswith("./secrets/") for v in c["secrets"].values())
    with open(os.path.join(DIR, "secrets", ".gitignore")) as f:
        assert "*" in f.read().split()


def test_dockerfile_and_scripts():
    with open(os.path.join(DIR, "Dockerfile"), encoding="utf-8") as f:
        docker = f.read()
    assert re.search(r"^USER 10001:10001$", docker, re.M) and re.search(r'^ENTRYPOINT \["mrgym"\]$', docker, re.M)
    assert docker.index("USER 10001") > docker.index("pip install") and "--seed" not in docker and "MRGYM_SECRET" not in docker
    assert not re.search(r"^(ENV|ARG)\s.*(SECRET|TOKEN|SEED)", docker, re.M | re.I)
    for name in ("probe.py", "operator_result.py"):
        py_compile.compile(os.path.join(DIR, name), doraise=True)
    with open(os.path.join(DIR, "probe.py"), encoding="utf-8") as f:
        probe = f.read()
    for needle in ("/var/log/scanner", "/var/lib/scanner", "/run/secrets/episode", "/run/secrets/mrgym_secret", "/run/secrets/operator_token",
                   "/v1/admin/episodes", "other_processes", "1.1.1.1"):
        assert needle in probe, needle
    assert "import mrgym" not in probe and "from mrgym" not in probe      # the agent side needs the single-file client only


def test_readme_says_the_tier_was_not_run():
    with open(os.path.join(DIR, "README.md"), encoding="utf-8") as f:
        text = f.read()
    assert "written and linted, not run" in text and "Docker client without a daemon" in text
