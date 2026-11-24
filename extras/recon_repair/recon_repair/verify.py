#!/usr/bin/env python3
"""Verifier for the recon-repair family. Standalone: needs only numpy. Copied into every task's tests/ directory.

    python verify.py --submission DIR --hidden DIR --out reward.json [--details details.json]

It runs the submitted ``recon.py`` on hidden scans the agent never saw and compares each image with the ground truth
of the simulated object. Trust boundary: the submission runs in its own process and working directory that contain
only the hidden *raw* scans. When the verifier runs as root (the container case) the submission is run as an
unprivileged user and the ground truth is readable by root only. Run as an ordinary user (local development) the
separation is by directory only, which is not a security boundary.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def nrmse(img, truth):
    img = np.asarray(img, dtype=float)
    den = float(np.sum(img * img))
    scale = float(np.sum(img * truth)) / den if den > 0 else 0.0
    return float(np.linalg.norm(max(scale, 0.0) * img - truth) / np.linalg.norm(truth))


def run_submission(sub_dir, scan_path, timeout):
    """Run recon.py on one hidden scan in a scratch directory. Returns (image or None, reason)."""
    work = tempfile.mkdtemp(prefix="mrfrl_run_")
    try:
        for name in os.listdir(sub_dir):                 # the submission's own Python files only
            if name.endswith(".py") and os.path.isfile(os.path.join(sub_dir, name)):
                shutil.copy(os.path.join(sub_dir, name), os.path.join(work, name))
        if not os.path.exists(os.path.join(work, "recon.py")):
            return None, "recon.py missing"
        shutil.copy(scan_path, os.path.join(work, "scan.npz"))
        kwargs = {}
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        if os.geteuid() == 0:
            os.chmod(work, 0o777)
            for name in os.listdir(work):
                os.chmod(os.path.join(work, name), 0o644)
            kwargs = dict(user="nobody", group="nogroup")
            env = {"PATH": os.environ.get("PATH", ""), "HOME": work, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        try:
            proc = subprocess.run([sys.executable, "recon.py", "scan.npz", "image.npy"], cwd=work, env=env,
                                  capture_output=True, timeout=timeout, **kwargs)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        if proc.returncode != 0:
            return None, "exit code %d: %s" % (proc.returncode, proc.stderr.decode(errors="replace")[-300:])
        out = os.path.join(work, "image.npy")
        if not os.path.exists(out):
            return None, "no image written"
        try:
            return np.load(out, allow_pickle=False), "ok"
        except Exception as exc:                        # an unreadable file is a failed submission, not a crash
            return None, "unreadable image: %s" % exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


def grade(sub_dir, hidden_dir, timeout=120):
    meta = json.load(open(os.path.join(hidden_dir, "meta.json")))
    scans = []
    for case in meta["cases"]:
        truth = np.load(os.path.join(hidden_dir, case["truth"]))
        img, reason = run_submission(sub_dir, os.path.join(hidden_dir, case["scan"]), timeout)
        err = None
        if img is not None:
            if img.shape != truth.shape:
                reason = "shape %s, expected %s" % (img.shape, truth.shape)
            elif not np.all(np.isfinite(img)) or np.iscomplexobj(img):
                reason = "non-finite or complex values"
            else:
                err = nrmse(img, truth)
        ok = err is not None and err <= case["threshold"]
        span = case["error_as_given"] - case["error_reference"]
        close = 0.0 if err is None else float(np.clip((case["error_as_given"] - err) / span, 0.0, 1.0))
        scans.append(dict(scan=case["scan"], passed=bool(ok), error=err, threshold=case["threshold"],
                          error_reference=case["error_reference"], error_as_given=case["error_as_given"],
                          closeness=close, reason=reason))
    n_pass = sum(s["passed"] for s in scans)
    reward = {"reward": float(n_pass == len(scans)), "scans_passed": float(n_pass), "scans_total": float(len(scans)),
              "closeness": float(np.mean([s["closeness"] for s in scans]))}
    return reward, scans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--details")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    reward, scans = grade(args.submission, args.hidden, args.timeout)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(reward, fh, indent=1)
    if args.details:
        with open(args.details, "w") as fh:
            json.dump({"reward": reward, "scans": scans}, fh, indent=1)
    print(json.dumps(reward))


if __name__ == "__main__":
    main()
