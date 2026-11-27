"""Run every command shown in a bash block of README.md and docs/TOUR.md, verbatim, and report exit code and wall time.

    python tools/run_readme_commands.py                 # from the repository root, after `pip install -e .`
    python tools/run_readme_commands.py --markdown

The three set-up lines (`git clone`, `cd MRGym`, `pip install -e .`) are listed but not run: the script is started
from an installed checkout. Commands read standard input from an empty stream, so the interactive console
(`mrgym play`) opens and closes. Exits 1 if any command fails.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = ("git clone ", "cd MRGym", "pip install -e .")
PAGES = ("README.md", os.path.join("docs", "TOUR.md"))            # the front page and the page its commands moved to


def commands(*paths):
    text = ""
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text += f.read() + "\n"
    out = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        out += [ln.strip() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", nargs="+", default=[os.path.join(ROOT, p) for p in PAGES])
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    rows, failed = [], 0
    for cmd in dict.fromkeys(commands(*a.file)):                     # a command shown on both pages runs once
        if cmd.startswith(SETUP):
            rows.append((cmd, "set-up, not run", ""))
            continue
        t0 = time.time()
        p = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        rows.append((cmd, str(p.returncode), f"{time.time() - t0:.1f}"))
        if p.returncode:
            failed += 1
            sys.stderr.write(f"FAILED: {cmd}\n{p.stderr.decode(errors='replace')[-2000:]}\n")
    if a.markdown:
        print("| Command | Exit | Seconds |\n|---|---:|---:|")
        for c, e, s in rows:
            print(f"| `{c}` | {e} | {s} |")
    else:
        for c, e, s in rows:
            print(f"{e:>16s} {s:>7s} s  {c}")
    print(f"{len(rows) - failed} of {len(rows)} commands ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
