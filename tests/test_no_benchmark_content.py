"""MR Gym is for practice and training; MRI Agent Bench is the held-out evaluation. Nothing of the benchmark may
enter this repository (AGENTS.md, "Hard rule"). These tests enforce the part a machine can check: the benchmark's
canary string is absent, and no environment here is named like a benchmark task."""
import os
import re
import subprocess

import pytest

import mrgym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The canary of MRI Agent Bench, written in two pieces so that this file does not contain it.
CANARY = "3b83d6a4-2edd-4dee-" + "8faa-d7db9704b608"
# Domain codes of the benchmark's task ids. An environment here must not look like one of its tasks.
DOMAIN_CODES = ("SEQ", "MRS", "FMRI", "RELAX", "PERF", "DIFF", "RECON", "HW", "STATS", "WRITE")
# a code used as a whole word or as a prefix: "Perf01", "PERF-01", "DiffMap" match; "Perfusion", "Diffusion" do not
CODE_PREFIX = re.compile("^(?i:" + "|".join(DOMAIN_CODES) + ")(?![a-z])")
ID_LIKE = re.compile(r"^([A-Za-z0-9]+)-([A-Za-z0-9]+)-v\d+$")
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "dist", "build", "runs", "mrgym_out"}


def repo_files():
    """Tracked files plus untracked files that are not ignored, as paths relative to the repository root."""
    if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        pytest.skip("the tests are not running from a checkout of the repository")
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, check=True).stdout.decode()
        names = [n for n in out.split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        names = []
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
            names += [os.path.relpath(os.path.join(base, f), ROOT).replace(os.sep, "/") for f in files]
    return sorted(n for n in names if os.path.isfile(os.path.join(ROOT, n)))


def test_the_benchmark_canary_is_nowhere_in_the_tree():
    needle = CANARY.encode()
    hits = []
    for name in repo_files():
        with open(os.path.join(ROOT, name), "rb") as f:
            if needle in f.read().lower():
                hits.append(name)
    assert not hits, f"the MRI Agent Bench canary appears in: {hits}"


def test_no_environment_id_uses_a_benchmark_domain_code():
    bad = []
    for env_id in mrgym.list_envs():
        s = mrgym.spec(env_id)
        if s.namespace.upper() in DOMAIN_CODES or CODE_PREFIX.match(s.name):
            bad.append(env_id)
    assert not bad, f"environment ids named like benchmark tasks: {bad}"


def test_no_environment_folder_uses_a_benchmark_domain_code():
    roots = [os.path.join(ROOT, "envs"), os.path.join(ROOT, "src", "mrgym", "envs"), os.path.dirname(mrgym.envs.__file__)]
    bad = []
    for root in roots:
        for base, dirs, _ in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for d in dirs:
                m = ID_LIKE.match(d)
                parts = [m.group(1), m.group(2)] if m else [d]
                if any(CODE_PREFIX.match(p) for p in parts):
                    bad.append(os.path.join(base, d))
    assert not bad, f"environment folders named like benchmark tasks: {bad}"


def test_no_path_is_laid_out_like_the_benchmark():
    marker = "families" + "/"
    bad = [n for n in repo_files() if marker in n]
    assert not bad, f"paths that contain {marker!r}: {bad}"


def test_the_prefix_rule_itself():
    """The rule must catch task-like names and must let ordinary words through."""
    for name in ("PERF", "Perf01", "perf-01-pcasl", "DiffMap", "Recon", "HW", "hw_limits", "Stats2", "WriteUp", "FMRI", "mrs"):
        assert CODE_PREFIX.match(name), name
    for name in ("Perfusion", "Diffusion", "Reconstruction", "Relaxometry", "Sequence", "Statistics", "NoWrap", "T1Transmit",
                 "Explore", "nowrap", "t1_transmit", "results"):
        assert not CODE_PREFIX.match(name), name
