"""Type check of the typed scope ([tool.mypy] in pyproject.toml) against a stored baseline.

    python tools/audit/typecheck.py            fail when a file has more mypy errors than tools/audit/mypy_baseline.json
    python tools/audit/typecheck.py --write    record the current counts (only ever to lower them)

The scope is the public surface and the scoring code. The baseline is a ratchet: the count of a file may fall, never
rise, and a file that reaches 0 stays at 0. The physics modules are outside the scope until someone needs them typed.
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINE = os.path.join(ROOT, "tools", "audit", "mypy_baseline.json")


def run() -> tuple[dict, str]:
    r = subprocess.run([sys.executable, "-m", "mypy", "--no-error-summary", "--no-color-output"], cwd=ROOT, capture_output=True, text=True)
    if "No module named mypy" in r.stderr:
        raise SystemExit("mypy is not installed: pip install -r tools/ci/requirements-audit.txt")
    counts = collections.Counter()
    for line in r.stdout.splitlines():
        m = re.match(r"(src/[^:]+\.py):\d+: error:", line.replace(os.sep, "/"))
        if m:
            counts[m.group(1)] += 1
    return dict(sorted(counts.items())), r.stdout


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    counts, text = run()
    if "--write" in argv:
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(dict(mypy=subprocess.run([sys.executable, "-m", "mypy", "--version"], capture_output=True, text=True).stdout.strip(),
                           errors=counts), f, indent=1)
            f.write("\n")
        print(f"wrote {os.path.relpath(BASELINE, ROOT)}: {sum(counts.values())} errors in {len(counts)} files")
        return 0
    with open(BASELINE, encoding="utf-8") as f:
        base = json.load(f)["errors"]
    worse = {k: (base.get(k, 0), v) for k, v in counts.items() if v > base.get(k, 0)}
    print(f"mypy: {sum(counts.values())} errors in the typed scope (baseline {sum(base.values())})")
    for path, (was, now) in worse.items():
        print(f"  {path}: {was} -> {now}")
        print("\n".join("    " + ln for ln in text.splitlines() if ln.replace(os.sep, "/").startswith(path))[:4000])
    better = {k: (v, counts.get(k, 0)) for k, v in base.items() if counts.get(k, 0) < v}
    if better and not worse:
        print("  fewer errors than the baseline in: " + ", ".join(better) + " (run with --write to lock that in)")
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
