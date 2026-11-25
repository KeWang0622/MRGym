"""The wheel holds exactly the tracked files of ``src/mrgym`` (plus its own metadata), and nothing else.

    python -m build --wheel && python tools/audit/wheel_vs_git.py dist/*.whl

Fails when a tracked package file is missing from the wheel (a data file that was not declared), when the wheel holds
a file that is not tracked (a stray cache, a local result), or when anything outside ``mrgym/`` got in (tests,
results, tools). Outside a git checkout (an unpacked sdist) the files on disk under ``src/mrgym`` are used instead.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tracked() -> tuple[set, str]:
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "src/mrgym"], capture_output=True, text=True, check=True).stdout
        files = {ln[len("src/"):] for ln in out.splitlines() if ln}
        if files:
            return files, "git ls-files src/mrgym"
    except (OSError, subprocess.CalledProcessError):
        pass
    files = set()
    for base, dirs, names in os.walk(os.path.join(ROOT, "src", "mrgym")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        files |= {os.path.relpath(os.path.join(base, n), os.path.join(ROOT, "src")).replace(os.sep, "/") for n in names
                  if not n.endswith((".pyc", ".pyo"))}
    return files, "the files under src/mrgym (no git checkout)"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    with zipfile.ZipFile(argv[0]) as z:
        names = {n for n in z.namelist() if not n.endswith("/")}
    meta = {n for n in names if n.split("/")[0].endswith(".dist-info")}
    package = {n for n in names if n.startswith("mrgym/")}
    want, source = tracked()
    problems = [f"missing from the wheel: {n}" for n in sorted(want - package)]
    problems += [f"in the wheel but not tracked: {n}" for n in sorted(package - want)]
    problems += [f"outside the package: {n}" for n in sorted(names - package - meta)]
    print(f"{os.path.basename(argv[0])}: {len(package)} package files, {len(meta)} metadata files; compared with {source} ({len(want)} files)")
    for p in problems:
        print("  " + p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
