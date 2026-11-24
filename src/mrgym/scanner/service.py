"""HTTP front end of one scanner episode (standard library + numpy).

One process = one episode. No route creates, resets or re-seeds an episode, and none returns ground truth: a fresh
state is a fresh process on a fresh log directory.

  GET  /v1/health   liveness
  GET  /v1/info     instrument card, sequences, parameter ranges and units, cost rules, budget, task card
  GET  /v1/budget   total / used / remaining scan seconds
  POST /v1/plan     {"protocol": {...}} -> realised protocol, timing, scan seconds, limits, accept / reject. Free.
  POST /v1/scan     {"protocol": {...}, "rois": [{"id", "y_mm", "x_mm", "radius_mm"}, ...]} -> ROI read-outs + images

Refusals return {"error": {"type": ..., "message": ...}} with status 422 (invalid_request), 409 (budget_exhausted) or
429 (request_limit); they are logged and cost nothing.

Run:  python -m mrgym.scanner.service --config episode.json --log-dir /var/log/scanner --port 8000
The seed is read from the config file (or derived from MRGYM_SECRET), never from the command line.
"""
import argparse
import base64
import io
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import session as core


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "mrgym-scanner/1"

        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/v1/health":
                self._send(200, {"ok": True})
            elif path == "/v1/info":
                self._send(200, session.info())
            elif path == "/v1/budget":
                self._send(200, session.budget())
            else:
                self._send(404, {"error": {"type": "not_found", "message": "routes: GET /v1/info, /v1/budget; POST /v1/plan, /v1/scan"}})

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/")
            if path not in ("/v1/plan", "/v1/scan"):
                self._send(404, {"error": {"type": "not_found", "message": "POST /v1/plan and /v1/scan are the only actions"}})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > core.MAX_BODY_BYTES:
                self.close_connection = True
                self._send(413, {"error": {"type": "invalid_request", "message": "body missing or too large"}})
                return
            try:
                request = json.loads(self.rfile.read(n))
            except ValueError:
                request = None
            try:
                if path == "/v1/plan":
                    self._send(200, session.plan(request))
                    return
                resp, arrays = session.scan(request)
            except core.ScannerError as e:
                self._send(e.http_status, e.to_json())
                return
            for name, a in arrays.items():
                buf = io.BytesIO()
                np.save(buf, a, allow_pickle=False)
                resp[f"{name}_npy_b64"] = base64.b64encode(buf.getvalue()).decode()
            self._send(200, resp)

        def log_message(self, fmt, *args):      # the scan log is the record; keep stderr quiet
            pass

    return Handler


def serve(cfg, log_dir, host="127.0.0.1", port=8000, port_file=None):
    session = core.ScannerSession(cfg, log_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(session))
    httpd.daemon_threads = True
    if port_file:                               # written after bind: its existence means "accepting connections"
        tmp = port_file + ".tmp"
        with open(tmp, "w") as f:
            f.write(str(httpd.server_address[1]))
        os.replace(tmp, port_file)

    def stop(*_):
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        httpd.server_close()
        session.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.environ.get("MRSCAN_CONFIG"), help="episode config JSON (holds the seed)")
    ap.add_argument("--log-dir", default=os.environ.get("MRSCAN_LOG_DIR", "/var/log/scanner"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000, help="0 = pick a free port (see --port-file)")
    ap.add_argument("--port-file", default=None)
    a = ap.parse_args(argv)
    if not a.config:
        sys.exit("mrgym.scanner.service: --config or MRSCAN_CONFIG is required")
    serve(core.load_config(a.config), a.log_dir, a.host, a.port, a.port_file)


if __name__ == "__main__":
    main()
