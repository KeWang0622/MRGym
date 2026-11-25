"""Mutation gate on the code that decides a reward or a gate verdict (scope and test selection: ``[tool.mutmut]`` in
pyproject.toml).

    python tools/audit/mutation.py                      run, print the score and the survivors, exit 1 below the gate
    python tools/audit/mutation.py --min 0.85 --max-seconds 600 --json mutation.json

The run happens in a temporary copy of ``src/``, ``tests/`` and ``pyproject.toml`` (mutmut writes a ``mutants/`` tree
next to the sources; the working tree stays clean). Two worker processes, one thread each.

Score = killed / all generated mutants (survived, never executed, timed out and suspicious all count against it).
The gate is the score and the wall time of the mutmut run; nothing may be excluded by a pragma that is not listed in
``tools/audit/no_mutate_allowlist.txt``.

Survivors are then read one by one (``mutmut show``) and matched against ``tools/audit/mutation_classes.txt``: a rule
names a function, a pattern of the changed line and why such a mutant cannot change any result (equivalent). A
survivor no rule matches is reported as UNCLASSIFIED: a gap in the tests until someone looks at it.
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
PRAGMA = re.compile(r"#\s*pragma:\s*no mutate")


def _mutmut() -> str:
    exe = os.path.join(os.path.dirname(sys.executable), "mutmut")
    if not os.path.exists(exe):
        raise SystemExit("mutmut is not installed next to this interpreter: pip install 'mutmut==3.8.1'")
    return exe


def _limit():
    """Address-space limit of the run and of every process it forks (MRGYM_AUDIT_MEM_GB, default 4)."""
    import resource
    cap, hard = int(float(os.environ.get("MRGYM_AUDIT_MEM_GB", "4")) * 2 ** 30), resource.getrlimit(resource.RLIMIT_AS)[1]
    if cap <= 0:
        return
    cap = cap if hard == resource.RLIM_INFINITY else min(cap, hard)      # never above a limit the caller already runs under
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def check_pragmas() -> list[str]:
    """Problems with ``# pragma: no mutate`` in src/: more than 25 lines, or a line the allow-list does not name."""
    allowed = set()
    with open(os.path.join(HERE, "no_mutate_allowlist.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                allowed.add(line.split()[0])                            # "<path>:<line number>"
    found = []
    for base, _, files in os.walk(os.path.join(ROOT, "src")):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(base, name)
                with open(path, encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if PRAGMA.search(line):
                            found.append(f"{os.path.relpath(path, ROOT)}:{i}")
    problems = [f"{p}: '# pragma: no mutate' is not in tools/audit/no_mutate_allowlist.txt" for p in found if p not in allowed]
    if len(found) > 25:
        problems.append(f"{len(found)} lines carry '# pragma: no mutate' (at most 25)")
    return problems


def load_rules() -> list[tuple]:
    rules = []
    with open(os.path.join(HERE, "mutation_classes.txt"), encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [p.strip() for p in line.rstrip("\n").split(" | ")]
            if len(parts) != 4:
                raise SystemExit(f"mutation_classes.txt line {n}: expected 'function glob | regex | class | reason'")
            rules.append((parts[0], re.compile(parts[1], re.M), parts[2], parts[3]))
    return rules


def function_of(name: str) -> str:
    return re.sub(r"\.x_|\.xǁ", ".", re.sub(r"__mutmut_\d+$", "", name)).replace("ǁ", ".")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min", type=float, default=0.85, help="smallest accepted score")
    ap.add_argument("--max-seconds", type=float, default=600.0, help="longest accepted mutmut run")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keep", default=None, help="run in this directory and keep it (default: a temporary directory)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    problems = check_pragmas()
    work = a.keep or tempfile.mkdtemp(prefix="mrgym-mutation-")
    os.makedirs(work, exist_ok=True)
    for name in ("src", "tests"):
        shutil.rmtree(os.path.join(work, name), ignore_errors=True)
        shutil.copytree(os.path.join(ROOT, name), os.path.join(work, name), ignore=shutil.ignore_patterns("__pycache__"))
    shutil.rmtree(os.path.join(work, "mutants"), ignore_errors=True)
    shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("HYPOTHESIS_PROFILE", None)
    workers = str(max(1, min(a.workers, 2)))
    t0 = time.time()
    with open(os.path.join(work, "mutmut_run.log"), "w") as log:
        run = subprocess.run([_mutmut(), "run", "--max-children", workers], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                             preexec_fn=_limit)                           # inherited by every forked test process
    seconds = time.time() - t0
    res = subprocess.run([_mutmut(), "results", "--all", "true"], cwd=work, env=env, capture_output=True, text=True)
    status = {}
    for line in res.stdout.splitlines():
        m = re.match(r"\s+(\S+): (.+)$", line)
        if m:
            status[m.group(1)] = m.group(2).strip()
    if not status:
        print(f"mutmut produced no result (exit {run.returncode}); see {work}/mutmut_run.log")
        return 2
    counts = collections.Counter(status.values())
    total, killed = len(status), counts["killed"]
    score = killed / total

    rules, rows = load_rules(), []
    for name in sorted(k for k, v in status.items() if v != "killed"):
        diff = ""
        if status[name] == "survived":
            show = subprocess.run([_mutmut(), "show", name], cwd=work, env=env, capture_output=True, text=True).stdout
            diff = "\n".join(ln for ln in show.splitlines() if ln[:1] in "+-" and not ln.startswith(("---", "+++")))
        fn = function_of(name)
        cls, why = ("never executed", "no selected test calls the function") if status[name] != "survived" else ("UNCLASSIFIED", "")
        for glob, pattern, c, reason in rules:
            if status[name] == "survived" and fnmatch.fnmatch(fn, glob) and pattern.search(diff):
                cls, why = c, reason
                break
        rows.append(dict(mutant=name, function=fn, status=status[name], diff=diff, classified=cls, reason=why))

    by_fn = collections.defaultdict(collections.Counter)
    for r in rows:
        by_fn[r["function"]][r["classified"]] += 1
    print(f"mutmut run: {total} mutants, {killed} killed, {counts['survived']} survived, {total - killed - counts['survived']} other "
          f"({dict(counts)}); wall {seconds:.0f} s with {workers} workers")
    print(f"mutation score {100 * score:.2f} % (gate {100 * a.min:.0f} %), wall time gate {a.max_seconds:.0f} s")
    print("not killed, by function:")
    for fn in sorted(by_fn, key=lambda f: -sum(by_fn[f].values())):
        print(f"  {sum(by_fn[fn].values()):4d}  {fn}   " + ", ".join(f"{k} {v}" for k, v in sorted(by_fn[fn].items())))
    classes = collections.Counter(r["classified"] for r in rows)
    print("not killed, by class: " + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())))
    for r in rows:
        if r["classified"] == "UNCLASSIFIED":
            print(f"UNCLASSIFIED {r['mutant']}\n" + "\n".join("    " + ln[:200] for ln in r["diff"].splitlines()))
    for p in problems:
        print("pragma:", p)
    ok = score >= a.min and seconds <= a.max_seconds and not problems
    out = dict(total=total, killed=killed, counts=dict(counts), score=round(score, 4), seconds=round(seconds, 1), workers=int(workers),
               gate=dict(min_score=a.min, max_seconds=a.max_seconds), passed=ok, pragma_problems=problems,
               not_killed_by_class=dict(classes), not_killed=rows,
               command="python tools/audit/mutation.py" + (f" --min {a.min}" if a.min != 0.85 else ""))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
