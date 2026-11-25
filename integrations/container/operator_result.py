#!/usr/bin/env python3
"""Operator side of the container tier: read the result of every episode from the scanner's operator plane.

The reward never comes from a file the agent's container could write: it is asked for here, over the unix socket of
the operator plane (a volume the agent container does not mount), with the operator token.

    docker compose -f integrations/container/compose.yaml --profile operator run --rm operator

Prints one JSON object per episode: {"episode", "status", "reward", "passed", "replay", "head", ...}.
Exit status 0 when every episode is done and its log replays against its head, 1 otherwise.
"""
import json
import os
import sys

from mrgym.scanner import service


def main():
    target = os.environ.get("MRGYM_ADMIN", "unix:/var/run/scanner-admin/admin.sock")
    token = service.read_token(os.environ.get("MRGYM_OPERATOR_TOKEN_FILE", "/run/secrets/operator_token"))
    status, listing = service.admin_request(target, token, "GET", "/v1/admin/episodes")
    if status != 200:
        print(json.dumps({"error": f"operator plane answered {status}"}))
        return 1
    ok = True
    for handle in listing["episodes"]:
        status, report = service.admin_request(target, token, "GET", f"/v1/admin/episodes/{handle}")
        print(json.dumps(report))
        ok = ok and status == 200 and report.get("status") == "done" and bool((report.get("replay") or {}).get("ok"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
