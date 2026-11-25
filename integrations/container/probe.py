#!/usr/bin/env python3
"""Run inside the agent container: everything the agent must not be able to do is tried, and each attempt must fail.

    docker compose -f integrations/container/compose.yaml run --rm agent

Prints one JSON line per probe and exits 0 only if every forbidden probe failed and the one allowed call (the
instrument card, with the agent token) worked. Standard library only; it uses ``mrscan.py`` next to it.
"""
import json
import os
import socket
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("MRSCAN_URL", "http://scanner:8000").rstrip("/")


def can_read(path):
    try:
        if os.path.isdir(path):
            return bool(os.listdir(path)) or True
        with open(path, "rb") as f:
            f.read(1)
        return True
    except OSError:
        return False


def other_processes():
    """Processes that are not this one or its parents: none are visible from a container of its own."""
    mine, pid = set(), os.getpid()
    while pid > 0:
        mine.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return [p for p in os.listdir("/proc") if p.isdigit() and int(p) not in mine]


def status(path, token=None):
    req = urllib.request.Request(URL + path, headers={"Authorization": "Bearer " + token} if token else {})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except OSError:
        return None


def reaches(host, port):
    try:
        socket.create_connection((host, port), timeout=3).close()
        return True
    except OSError:
        return False


def main():
    token = None
    if os.environ.get("MRSCAN_TOKEN_FILE"):
        with open(os.environ["MRSCAN_TOKEN_FILE"]) as f:
            token = f.read().strip()
    forbidden = {
        "read the scanner's log directory": can_read("/var/log/scanner"),
        "read the scanner's state directory": can_read("/var/lib/scanner"),
        "read the operator socket directory": can_read("/var/run/scanner-admin"),
        "read the episode secret": can_read("/run/secrets/episode"),
        "read the master secret": can_read("/run/secrets/mrgym_secret"),
        "read the operator token": can_read("/run/secrets/operator_token"),
        "see another process": bool(other_processes()),
        "reach a host outside the episode network (1.1.1.1:443)": reaches("1.1.1.1", 443),
        "reach a host outside the episode network (example.com:80)": reaches("example.com", 80),
        "call an operator route on the agent port": status("/v1/admin/episodes", token) != 404,
        "read the instrument card without a token": status("/v1/info") != 401,
        "reach a route that resets or reveals": any(status(p, token) != 404 for p in ("/v1/reset", "/v1/seed", "/v1/truth", "/v1/config")),
    }
    allowed = {"read the instrument card with the agent token": status("/v1/info", token) == 200}
    for name, happened in forbidden.items():
        print(json.dumps({"probe": name, "must": "fail", "happened": bool(happened), "ok": not happened}))
    for name, worked in allowed.items():
        print(json.dumps({"probe": name, "must": "work", "happened": bool(worked), "ok": bool(worked)}))
    return 0 if not any(forbidden.values()) and all(allowed.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
