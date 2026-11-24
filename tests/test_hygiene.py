"""Nothing from the workshop reaches the public tree: no former package name, no reference to private working
notes, no absolute path of a development machine.

The test reads every tracked file and every untracked file that is not ignored, skips binary files, and prints the
file and line of each hit. The patterns are assembled from pieces, so this file passes its own check.
"""
import os
import re
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "dist", "build", "runs", "mrgym_out"}
BINARY_SUFFIXES = (".png", ".gif", ".jpg", ".jpeg", ".webp", ".ico", ".pdf", ".npy", ".npz", ".whl", ".gz", ".zip", ".woff",
                   ".woff2", ".ttf", ".mp4", ".pyc")

# (label, regular expression). Case-insensitive where the case carries no meaning.
PATTERNS = [
    ("former package name", "(?i)mr" "frl"),
    ("former project name", "(?i)mrfoundry" "-rl"),
    ("reference to a private note", r"(?i)\bmemo " r"\d"),
    ("reference to a private note", "(?i)memo" "s3"),
    ("reference to a private workspace", "(?i)lens" r"\d\d"),
    ("reference to a private workspace", "(?i)round" "3"),
    ("unverified citation", "(?i)cited from " "memory"),
    ("unverified citation", "(?i)from " "memory"),
    ("absolute local path", "/opt" "/cis"),
    ("absolute local path", "/ho" "me/"),
    ("absolute local path", "/mn" "t/"),
    ("absolute local path", "/Us" "ers/"),
]
COMPILED = [(label, re.compile(rx)) for label, rx in PATTERNS]


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


def text_of(name):
    """The file as text, or None for a binary file."""
    if name.lower().endswith(BINARY_SUFFIXES):
        return None
    with open(os.path.join(ROOT, name), "rb") as f:
        raw = f.read()
    if b"\0" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def find_hits(text):
    """[(line number, label, line)] for one file's text."""
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for label, rx in COMPILED:
            if rx.search(line):
                hits.append((i, label, line.strip()[:160]))
                break
    return hits


def test_no_workshop_residue_in_the_tree():
    report, n_files = [], 0
    for name in repo_files():
        text = text_of(name)
        if text is None:
            continue
        n_files += 1
        report += [f"{name}:{i}: {label}: {line}" for i, label, line in find_hits(text)]
    assert n_files > 20, f"only {n_files} text files were read; the file listing is broken"
    shown = "\n".join(report[:60]) + (f"\n... and {len(report) - 60} more" if len(report) > 60 else "")
    assert not report, f"{len(report)} line(s) with workshop residue:\n{shown}"


def test_the_patterns_catch_what_they_are_for():
    samples = ["python3 -m mr" "frl selfcheck", "MRFoundry" "-RL families", "see memo " "17, section 2", "memo" "s3/notes.md",
               "work/lens" "26/proto", "round" "3 drafts", "value cited from " "memory", "/opt" "/cis/scratch", "/ho" "me/user/x",
               "/mn" "t/remotes", "/Us" "ers/name"]
    for s in samples:
        assert find_hits(s), s
    for s in ("MRFoundry is the companion skill library", "a memorandum", "the lens of the camera", "round(3.2)",
              "https://github.com/KeWang0622/MRGym", "read from the log", "cd extras/recon_repair"):
        assert not find_hits(s), s
