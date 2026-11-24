#!/usr/bin/env python3
"""Client of the virtual scanner: a module and a command line in one file, standard library only.

This file imports nothing from the scanner package, so it can be copied alone into the agent's workspace.

    python3 mrscan.py info
    python3 mrscan.py budget
    python3 mrscan.py plan --protocol '{"sequence": "spgr", "tr_ms": 20, "flip_deg": 10}'
    python3 mrscan.py scan --protocol '{"sequence": "spgr", "tr_ms": 20, "flip_deg": 10, "averages": 4}' \\
                           --rois regions.json --out scans/fa10

``scan`` prints the realised protocol, the cost and the ROI read-outs as JSON and, with --out, writes
<out>_magnitude.npy and <out>_phase.npy (float32, [n_images, matrix, matrix]) and <out>.json. Images never go to stdout.
Exit status: 0 ok, 3 refused (invalid_request / budget_exhausted / request_limit), 4 service unreachable.

    from mrscan import Scanner
    sc = Scanner()                       # URL from MRSCAN_URL
    sc.plan({...}); r = sc.scan({...}, rois=[{"id": "R1", "y_mm": 10, "x_mm": -20, "radius_mm": 11.5}], out="scans/a")
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URLS = "http://scanner:8000,http://127.0.0.1:8000"


class ScannerError(Exception):
    def __init__(self, type_, message, **extra):
        super().__init__(f"{type_}: {message}")
        self.type, self.message, self.extra = type_, message, extra


class Scanner:
    def __init__(self, url=None, timeout=300.0):
        self.timeout = timeout
        self._candidates = [u.strip().rstrip("/") for u in (url or os.environ.get("MRSCAN_URL") or DEFAULT_URLS).split(",")
                            if u.strip()]
        self.url = self._candidates[0] if len(self._candidates) == 1 else None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))     # never through a proxy

    def _resolve(self):
        if self.url is None:
            errors = []
            for u in self._candidates:
                try:
                    with self._opener.open(u + "/v1/health", timeout=5) as r:
                        if json.load(r).get("ok"):
                            self.url = u
                            break
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{u}: {type(e).__name__}: {e}")
            else:
                raise ConnectionError("no scanner service reachable; tried " + "; ".join(errors))
        return self.url

    def _call(self, method, path, body=None):
        req = urllib.request.Request(self._resolve() + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            try:
                err = json.load(e)["error"]
            except Exception:  # noqa: BLE001
                raise ScannerError("http_error", f"HTTP {e.code}") from None
            raise ScannerError(err.pop("type", "scanner_error"), err.pop("message", ""), **err) from None
        except urllib.error.URLError as e:
            raise ConnectionError(str(e)) from None

    def info(self):
        return self._call("GET", "/v1/info")

    def budget(self):
        return self._call("GET", "/v1/budget")

    def plan(self, protocol):
        """Free dry run: realised protocol, timing, scan seconds, limits, accept / reject with reasons."""
        return self._call("POST", "/v1/plan", {"protocol": protocol})

    def scan(self, protocol, rois=None, out=None):
        """Run one acquisition. Returns the response dict; with ``out`` the images are written next to it as .npy,
        otherwise they stay in the dict under ``arrays`` as raw .npy bytes."""
        resp = self._call("POST", "/v1/scan", {"protocol": protocol, "rois": rois or []})
        arrays = {k[:-len("_npy_b64")]: base64.b64decode(resp.pop(k)) for k in list(resp) if k.endswith("_npy_b64")}
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            resp["files"] = []
            for name, raw in arrays.items():
                with open(f"{out}_{name}.npy", "wb") as f:
                    f.write(raw)
                resp["files"].append(f"{out}_{name}.npy")
            with open(f"{out}.json", "w") as f:
                json.dump(resp, f, indent=1)
        else:
            resp["arrays"] = arrays
        return resp


def _json_arg(text):
    """A JSON string, or @file / a path to a JSON file."""
    if text is None:
        return None
    path = text[1:] if text.startswith("@") else text
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return json.loads(text)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mrscan", description="command line of the virtual scanner")
    ap.add_argument("--url", default=None, help="scanner URL (default: MRSCAN_URL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    sub.add_parser("budget")
    for name in ("plan", "scan"):
        sp = sub.add_parser(name)
        sp.add_argument("--protocol", required=True, help="JSON object or a file holding one, e.g. "
                        "'{\"sequence\": \"spgr\", \"tr_ms\": 20, \"flip_deg\": 10}'")
        if name == "scan":
            sp.add_argument("--rois", default=None, help="JSON list of circles {id, y_mm, x_mm, radius_mm} or a file")
            sp.add_argument("--out", default=None, help="path prefix for <out>_magnitude.npy, <out>_phase.npy, <out>.json")
    a = ap.parse_args(argv)
    scanner = Scanner(a.url)
    try:
        if a.cmd == "scan":
            rois = _json_arg(a.rois)
            if isinstance(rois, dict):
                rois = rois.get("regions", rois.get("rois"))
            out = scanner.scan(_json_arg(a.protocol), rois, a.out)
            out.pop("arrays", None)
        elif a.cmd == "plan":
            out = scanner.plan(_json_arg(a.protocol))
        else:
            out = getattr(scanner, a.cmd)()
    except ScannerError as e:
        print(json.dumps({"error": dict(type=e.type, message=e.message, **e.extra)}))
        sys.exit(3)
    except (ConnectionError, ValueError) as e:
        print(json.dumps({"error": {"type": "unreachable" if isinstance(e, ConnectionError) else "bad_argument",
                                    "message": str(e)}}))
        sys.exit(4)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
