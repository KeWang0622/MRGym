"""``make audit``: every check that stands behind the claim "the reward and the scanner are tested", in one run, with
one report (``audit_report.json``) and one exit status.

    python tools/audit/report.py                         all steps; exit 0 only if every row passes
    python tools/audit/report.py --only lint,golden      some steps
    python tools/audit/report.py --list                  the steps, their commands and their pass criteria

Rows and pass criteria
  lint          ruff check                                                       0 findings
  types         tools/audit/typecheck.py                                         no file above its mypy baseline
  suite         the whole test suite under coverage, one file per process,       0 failed; every skip and expected failure
                deprecations as errors                                           is listed in tools/audit/allowed_*.txt; no
                                                                                 test file above 1.5 GB of resident memory
  coverage      of that run                                                      total >= the ratchet in tools/audit/
                                                                                 coverage_baseline.json; scoring scope 100 %
  contract      the reset / step / submit state machine, HYPOTHESIS_PROFILE=ci   0 failed
  metamorphic   tests/test_metamorphic.py                                        every relation (20) exercised, 0 failed
  golden        tools/audit/golden.py --check + the golden and snapshot tests    files current; mode of the stack printed
  tables        every family id has a scoring table and a stored snapshot        none missing
  mutation      tools/audit/mutation.py                                          score >= 85 % of all mutants, <= 600 s
  handmutants   tools/audit/hand_mutants.py                                      16 / 16 review mutants killed (and the 6
                                                                                 further ones: constants, slice path)
  isolation     tests/test_keys.py test_log_mac.py test_redteam.py test_epoch.py  0 failed
                test_service.py
  wheel         build, twine check --strict, check-wheel-contents, wheel_vs_git  all OK
  generated     tools/generate.py --check, mrgym results --check, README cmds    current; every README command exits 0

A step whose tool or test files are not in the tree is "absent"; it counts as a failure unless it is listed in
``tools/audit/allowed_absent.txt`` with the reason. At most two processes run at a time, one thread each.

Memory. The process tree of every command is killed when its resident memory passes ``MRGYM_AUDIT_RSS_GB`` (default
2): a leak ends its own step instead of the machine. The report gives the peak resident memory of every row
(``peak_rss_gb``: the command and its children) and, for the suite, of every test file; a test file above 1.5 GB fails
the row. ``MRGYM_AUDIT_MEM_GB`` adds an address-space limit (``ulimit -v``; off by default: the service tests open
hundreds of threads, which need address space and no memory).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fnmatch
import glob
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SCORING_SCOPE = "*/mrgym/envs/family.py,*/mrgym/gate.py,*/mrgym/envs/*/task.py"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
ENV.pop("HYPOTHESIS_PROFILE", None)


MEM_GB = float(os.environ.get("MRGYM_AUDIT_MEM_GB", "0"))          # address space of every command (ulimit -v); 0 = not limited
RSS_GB = float(os.environ.get("MRGYM_AUDIT_RSS_GB", "2"))          # resident memory of a command and its children: killed above
_PEAK = threading.local()
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _limit():
    if MEM_GB <= 0:
        return
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]                     # never above a limit the caller already runs under
    cap = int(MEM_GB * 2 ** 30) if hard == resource.RLIM_INFINITY else min(int(MEM_GB * 2 ** 30), hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _memory_bytes(pid: int) -> int:
    """Memory of one process: its proportional set size (pages shared with a forked parent counted once over the
    sharers) where the kernel gives it, its resident set size otherwise."""
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    with open(f"/proc/{pid}/statm") as f:
        return int(f.read().split()[1]) * _PAGE


def tree_rss_gb(pid: int) -> float:
    """Memory held by a process and all its descendants, in GB (Linux /proc; 0.0 where that is not available). Forked
    workers share most pages with their parent: the sum uses proportional sizes, so it is what the tree really holds."""
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        try:
            total += _memory_bytes(p)
            for t in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{t}/children") as f:
                    todo += [int(c) for c in f.read().split()]
        except (OSError, ValueError, IndexError):
            pass
    return total / 2 ** 30


def sh(cmd, env=None, timeout=3600):
    """Run ``cmd`` in the repository. Returns (exit code, output, seconds) and notes the peak resident memory of the
    command and its children for the row that is running. Two limits: the address space (``MEM_GB``, set before the
    command starts) and the resident memory of the whole process tree (``RSS_GB``, watched four times a second): a
    command above it is killed and its output says so. A leak ends its own step, not the machine."""
    t0, peak, killed = time.time(), 0.0, ""
    with tempfile.TemporaryFile("w+") as out:
        p = subprocess.Popen(cmd, cwd=ROOT, env=dict(ENV, **(env or {})), stdout=out, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True, preexec_fn=_limit if hasattr(resource, "RLIMIT_AS") else None)
        while True:
            pid, status, usage = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            rss = tree_rss_gb(p.pid)
            peak = max(peak, rss)
            if rss > RSS_GB or time.time() > t0 + timeout:
                killed = (f"resident memory {rss:.2f} GB above the limit of {RSS_GB:g} GB" if rss > RSS_GB else f"no result after {timeout:g} s")
                try:
                    os.killpg(p.pid, 9)
                except OSError:
                    p.kill()
                pid, status, usage = os.wait4(p.pid, 0)
                break
            time.sleep(0.25)
        code = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status >> 8
        out.seek(0)
        text = out.read() + (f"\n[audit] killed: {killed}\n" if killed else "")
    _PEAK.gb = max(getattr(_PEAK, "gb", 0.0), peak, usage.ru_maxrss / 2 ** 20)
    return (code if not killed else -9), text, round(time.time() - t0, 1)


def patterns(name: str) -> list[str]:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.split("#")[0].strip() for ln in f if ln.split("#")[0].strip()]


def pytest_counts(text: str) -> dict:
    line = next((ln for ln in reversed(text.splitlines()) if re.search(r"\d+ (passed|failed|error)", ln)), "")
    return {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?)", line)}


def percent(text: str):
    """The number `coverage report --format=total` prints (a failed --fail-under adds a sentence after it)."""
    return next((float(ln) for ln in text.splitlines() if re.fullmatch(r"\d+(\.\d+)?", ln.strip())), None)


def row(status, command, seconds, **numbers):
    return dict(status=status, command=command if isinstance(command, str) else " ".join(command), seconds=seconds, **numbers)


def have(module: str) -> bool:
    return subprocess.run([PY, "-c", f"import {module}"], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------------------------- the steps
def step_lint(work):
    cmd = [PY, "-m", "ruff", "check"]
    code, out, s = sh(cmd)
    return row("pass" if code == 0 else "fail", cmd, s, output=out.strip().splitlines()[-1] if out.strip() else "")


def step_types(work):
    if not have("mypy"):
        return row("absent", "python tools/audit/typecheck.py", 0.0, reason="mypy is not installed")
    cmd = [PY, "tools/audit/typecheck.py"]
    code, out, s = sh(cmd)
    return row("pass" if code == 0 else "fail", cmd, s, output=out.strip().splitlines()[0] if out.strip() else "")


FILE_RSS_LIMIT_GB = 1.5                 # a test file that needs more is a defect (it holds worlds, engines or scans it should free)


def step_suite(work):
    """The whole suite under coverage, one test file per process (a file that leaks cannot take the others with it, and
    every file gets its own peak memory); the coverage row reads the data the runs leave in ``work``."""
    cov = have("coverage")
    data = os.path.join(work, ".coverage")
    base = [PY, "-m", "coverage", "run", "-m", "pytest"] if cov else [PY, "-m", "pytest"]
    files = sorted(os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    counts, skips, xfails, failed_files, rss, t0 = {}, set(), set(), [], {}, time.time()
    for f in files:
        before = getattr(_PEAK, "gb", 0.0)
        _PEAK.gb = 0.0
        code, out, _ = sh(base + ["-q", "-p", "no:cacheprovider", "-rsx", "-W", "error::DeprecationWarning:mrgym", f], env=dict(COVERAGE_FILE=data))
        rss[f] = round(_PEAK.gb, 2)
        _PEAK.gb = max(before, _PEAK.gb)
        for k, v in pytest_counts(out).items():
            counts[k] = counts.get(k, 0) + v
        skips |= set(re.findall(r"^SKIPPED \[\d+\] (.+)$", out, re.M))
        xfails |= set(re.findall(r"^XFAIL (.+?)(?: - .*)?$", out, re.M))
        if code not in (0, 5):                                           # 5: no test collected (a file of helpers)
            failed_files.append(f)
    skips, xfails = sorted(skips), sorted(xfails)
    bad_skips = [x for x in skips if not any(fnmatch.fnmatch(x, p) for p in patterns("allowed_skips.txt"))]
    bad_xfails = [x for x in xfails if not any(fnmatch.fnmatch(x, p) for p in patterns("allowed_xfails.txt"))]
    heavy = {f: v for f, v in rss.items() if v > FILE_RSS_LIMIT_GB}
    ok = not failed_files and not bad_skips and not bad_xfails and not heavy
    return row("pass" if ok else "fail",
               "for each tests/test_*.py: coverage run -m pytest -q -rsx -W error::DeprecationWarning:mrgym <file>" if cov else "pytest, one file per process",
               round(time.time() - t0, 1), files=len(files), counts=counts, files_that_fail=failed_files, skips=skips, expected_failures=xfails,
               skips_not_allowed=bad_skips, expected_failures_not_allowed=bad_xfails, peak_rss_gb_by_file=rss,
               files_over_the_memory_limit=heavy, memory_limit_per_file_gb=FILE_RSS_LIMIT_GB)


def least_covered(text: str, n: int = 3, min_statements: int = 20) -> list:
    """The ``n`` least-covered modules of a ``coverage report`` table: [{module, statements, missed, percent}]."""
    rows = []
    for line in text.splitlines():
        m = re.match(r"(\S+\.py)\s+(\d+)\s+(\d+)\s+([\d.]+)%", line)
        if m and int(m.group(2)) >= min_statements:
            rows.append(dict(module=m.group(1).split("mrgym/", 1)[-1], statements=int(m.group(2)), missed=int(m.group(3)), percent=float(m.group(4))))
    return sorted(rows, key=lambda r: (r["percent"], -r["missed"]))[:n]


def step_coverage(work):
    """Total coverage against a ratchet (the stored value: it may rise, never fall), and 100 % on the scoring scope."""
    if not have("coverage"):
        return row("absent", "coverage report", 0.0, reason="coverage is not installed")
    with open(os.path.join(HERE, "coverage_baseline.json"), encoding="utf-8") as f:
        base = json.load(f)
    env = dict(COVERAGE_FILE=os.path.join(work, ".coverage"))
    sh([PY, "-m", "coverage", "combine"], env=env)
    t0 = time.time()
    _, out1, _ = sh([PY, "-m", "coverage", "report", "--format=total", "--precision=1"], env=env)
    _, out2, _ = sh([PY, "-m", "coverage", "report", "--format=total", "--precision=1", f"--include={SCORING_SCOPE}"], env=env)
    _, table, _ = sh([PY, "-m", "coverage", "report", "--precision=1"], env=env)
    _, scope_table, _ = sh([PY, "-m", "coverage", "report", "--precision=1", "--show-missing", f"--include={SCORING_SCOPE}"], env=env)
    total, scope = percent(out1), percent(out2)
    ok = total is not None and scope is not None and total >= base["total_percent"] - 1e-9 and scope >= base["scoring_scope_percent"] - 1e-9
    return row("pass" if ok else "fail",
               f"coverage combine; coverage report (total >= {base['total_percent']}, the ratchet); coverage report --include={SCORING_SCOPE} "
               f"(>= {base['scoring_scope_percent']})", round(time.time() - t0, 1), total_percent=total, scoring_scope_percent=scope,
               gates=dict(total=base["total_percent"], scoring_scope=base["scoring_scope_percent"]), least_covered=least_covered(table),
               scoring_scope_not_covered=[ln.strip() for ln in scope_table.splitlines() if re.search(r"\.py\s", ln) and not re.search(r"\s100(\.0)?%", ln)],
               raise_the_ratchet=("python tools/audit/report.py --only suite,coverage --write-coverage"
                                  if ok and total is not None and total > base["total_percent"] + 0.05 else None))


def step_contract(work):
    if not (have("hypothesis") and have("hypothesis_jsonschema")):
        return row("absent", "HYPOTHESIS_PROFILE=ci pytest tests/test_contract_stateful.py", 0.0, reason="Hypothesis is not installed")
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rx", "tests/test_contract_stateful.py"]
    code, out, s = sh(cmd, env=dict(HYPOTHESIS_PROFILE="ci"))
    return row("pass" if code == 0 else "fail", "HYPOTHESIS_PROFILE=ci " + " ".join(cmd[1:]), s, counts=pytest_counts(out),
               profile="ci: 200 examples x 20 steps, derandomised",
               inputs_that_raise_today=sorted(set(re.findall(r"raises_today\[(.+?) \[", out))))


def step_metamorphic(work):
    table = os.path.join(work, "metamorphic.json")
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_metamorphic.py"]
    code, out, s = sh(cmd, env=dict(MRGYM_METAMORPHIC_TABLE=table))
    rows = []
    if os.path.exists(table):
        with open(table, encoding="utf-8") as f:
            for r in json.load(f):
                numeric = [c for c in r["checks"] if c["tolerance"] != "exact"]
                exact = [c for c in r["checks"] if c["tolerance"] == "exact"]
                worst = max(numeric, key=lambda c: c["measured"] / c["tolerance"]) if numeric else None
                rows.append(dict(relation=r["relation"], transformation=r["transformation"], expected=r["expected"],
                                 excitation=r["excitation"], checks=len(r["checks"]),
                                 tolerance=worst["tolerance"] if worst else "exact",
                                 measured=worst["measured"] if worst else ("identical" if all(c["measured"] == 0.0 for c in exact) else "differs"),
                                 exact_checks_identical=all(c["measured"] == 0.0 for c in exact)))
    ok = code == 0 and len(rows) >= 20 and all(r["checks"] for r in rows)
    return row("pass" if ok else "fail", "MRGYM_METAMORPHIC_TABLE=<file> " + " ".join(cmd[1:]), s, counts=pytest_counts(out), relations=rows)


def step_golden(work):
    c1, out1, s1 = sh([PY, "tools/audit/golden.py", "--check"])
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_golden.py", "tests/test_task_snapshots.py"]
    c2, out2, s2 = sh(cmd)
    _, mode, _ = sh([PY, "-c", "import sys; sys.path.insert(0, 'tests'); import golden_cases as G; e = G.load(G.ENGINE_FILE); "
                                "print(*G.mode(e['stack']), sep=': ') if e else print('no file')"])
    return row("pass" if c1 == 0 and c2 == 0 else "fail", "python tools/audit/golden.py --check; " + " ".join(cmd[1:]), round(s1 + s2, 1),
               counts=pytest_counts(out2), mode=mode.strip().splitlines()[-1] if mode.strip() else None, check=out1.strip().splitlines()[-3:])


def step_tables(work):
    """Every family id has a scoring table (tests/scoring/<family>.py) and a stored snapshot. A family without one, or a
    table that does not import, is a named entry of this row, not a crash."""
    code, out, s = sh([PY, "-c", (
        "import sys, json, importlib, pkgutil; sys.path.insert(0, 'tests'); import scoring, golden_cases as G, mrgym\n"
        "mods, broken = {}, {}\n"
        "for m in pkgutil.iter_modules(scoring.__path__):\n"
        "    if m.name.startswith('_'): continue\n"
        "    try: mods[m.name] = importlib.import_module('scoring.' + m.name)\n"
        "    except Exception as e: broken[m.name] = type(e).__name__ + ': ' + str(e)[:200]\n"
        "snap = (G.load(G.TASK_FILE) or {}).get('ids', {})\n"
        "fam = scoring.family_ids()\n"
        "name = lambda i: mrgym.spec(i).family.rsplit('.', 1)[-1]\n"
        "print(json.dumps(dict(family_ids=fam, without_table=[i for i in fam if name(i) not in mods], tables_that_do_not_import=broken,"
        " tables_of_unregistered_ids=sorted(k for k, m in mods.items() if getattr(m, 'ENV_ID', None) not in mrgym.registry),"
        " without_snapshot=[i for i in G.task_ids() if i not in snap], snapshots_with_errors=sorted(i for i, v in snap.items() if 'error' in v),"
        " rows={k: len(getattr(m, 'ROWS', [])) for k, m in mods.items()})))")])
    try:
        d = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return row("fail", "tables", s, output=out[-600:])
    bad = [k for k in ("without_table", "tables_that_do_not_import", "tables_of_unregistered_ids", "without_snapshot", "snapshots_with_errors") if d[k]]
    return row("pass" if code == 0 and not bad else "fail",
               "every family id has tests/scoring/<family>.py and a snapshot in tests/golden/task_snapshots.json", s, failing=bad, **d)


def step_mutation(work):
    if not os.path.exists(os.path.join(os.path.dirname(PY), "mutmut")):
        return row("absent", "python tools/audit/mutation.py", 0.0, reason="mutmut is not installed")
    out_json = os.path.join(work, "mutation.json")
    cmd = [PY, "tools/audit/mutation.py", "--json", out_json]
    code, out, s = sh(cmd)
    d = {}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            d = json.load(f)
    return row("pass" if code == 0 else "fail", "python tools/audit/mutation.py", s, total=d.get("total"), killed=d.get("killed"),
               score=d.get("score"), mutmut_seconds=d.get("seconds"), counts=d.get("counts"), gate=d.get("gate"),
               not_killed_by_class=d.get("not_killed_by_class"),
               unclassified=[r["mutant"] for r in d.get("not_killed", []) if r["classified"] == "UNCLASSIFIED"])


def step_handmutants(work):
    out_json = os.path.join(work, "hand_mutants.json")
    cmd = [PY, "tools/audit/hand_mutants.py", "--json", out_json]
    code, out, s = sh(cmd)
    d = {}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            d = json.load(f)
    summary = {k: v for k, v in d.get("summary", {}).items() if k != "seconds"}
    return row("pass" if code == 0 else "fail", "python tools/audit/hand_mutants.py", s, **summary,
               mutants=[dict(id=m["id"], kind=m["kind"], killed=m["killed"], by=m["by"]) for m in d.get("mutants", [])])


ISOLATION = ("tests/test_keys.py", "tests/test_log_mac.py", "tests/test_redteam.py", "tests/test_epoch.py", "tests/test_service.py")


def step_isolation(work):
    """Keys and splits, the MAC'd log with its seal and head, the red-team probes, epochs, the service."""
    missing = [f for f in ISOLATION if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        return row("absent", "pytest " + " ".join(ISOLATION), 0.0, reason="not in the tree: " + ", ".join(missing))
    cmd = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", *ISOLATION]
    code, out, s = sh(cmd)
    return row("pass" if code == 0 else "fail", cmd, s, counts=pytest_counts(out), skips=sorted(set(re.findall(r"^SKIPPED \[\d+\] (.+)$", out, re.M))))


def step_wheel(work):
    missing = [m for m in ("build", "twine", "check_wheel_contents") if not have(m)]
    if missing:
        return row("absent", "python -m build; twine check --strict; check-wheel-contents; wheel_vs_git.py", 0.0, reason="not installed: " + ", ".join(missing))
    dist = os.path.join(work, "dist")
    t0 = time.time()
    steps = {}
    c, out, _ = sh([PY, "-m", "build", "--outdir", dist])
    steps["build"] = c == 0
    wheels, sdists = glob.glob(os.path.join(dist, "*.whl")), glob.glob(os.path.join(dist, "*.tar.gz"))
    if wheels:
        steps["twine check --strict"] = sh([PY, "-m", "twine", "check", "--strict", *wheels, *sdists])[0] == 0
        steps["check-wheel-contents"] = sh([PY, "-m", "check_wheel_contents", wheels[0]])[0] == 0
        c, out, _ = sh([PY, "tools/audit/wheel_vs_git.py", wheels[0]])
        steps["wheel_vs_git"] = c == 0
    return row("pass" if steps and all(steps.values()) else "fail",
               "python -m build; twine check --strict dist/*; check-wheel-contents dist/*.whl; python tools/audit/wheel_vs_git.py dist/*.whl",
               round(time.time() - t0, 1), steps=steps, wheel=os.path.basename(wheels[0]) if wheels else None,
               wheel_vs_git=out.strip().splitlines()[0] if wheels and out.strip() else None)


SUITE_COMMANDS = re.compile(r"^\s*(pip install (-U )?pytest\b.*|pytest\b.*|python3? -m pytest\b.*|make test)\s*$")


def readme_without_the_suite(work) -> tuple[str, list]:
    """A copy of README.md and docs/TOUR.md (the pages whose commands are run) without the commands that run the test
    suite (the ``suite`` row runs it, one file per process; a second run of everything in one process is what this
    audit avoids). Returns (path, the lines left out)."""
    lines = []
    for rel in ("README.md", os.path.join("docs", "TOUR.md")):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            lines += f.read().split("\n")
    left_out = [ln.strip() for ln in lines if SUITE_COMMANDS.match(ln)]
    path = os.path.join(work, "README.without-suite.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(ln for ln in lines if not SUITE_COMMANDS.match(ln)))
    return path, left_out


def step_generated(work):
    t0 = time.time()
    steps = {}
    readme, left_out = readme_without_the_suite(work)
    for name, cmd in (("tools/generate.py --check", [PY, "tools/generate.py", "--check"]),
                      ("mrgym results --check --dir docs/results", [PY, "-m", "mrgym.cli", "results", "--check", "--dir", "docs/results"]),
                      ("tools/run_readme_commands.py", [PY, "tools/run_readme_commands.py", "--file", readme])):
        c, out, _ = sh(cmd, env=dict(PATH=os.path.dirname(PY) + os.pathsep + os.environ.get("PATH", "")))
        steps[name] = c == 0
        if c:                                                            # what failed, by name (a README command, a stale file)
            named = [ln.strip() for ln in out.splitlines() if ln.startswith("FAILED:") or "[audit] killed" in ln]
            steps[name + " (output)"] = named or out.strip().splitlines()[-6:]
    ok = all(v for k, v in steps.items() if not k.endswith("(output)"))
    return row("pass" if ok else "fail", "python tools/generate.py --check; mrgym results --check --dir docs/results; python tools/run_readme_commands.py",
               round(time.time() - t0, 1), steps=steps, readme_commands_left_to_the_suite_row=left_out)


STEPS = dict(lint=step_lint, types=step_types, suite=step_suite, coverage=step_coverage, contract=step_contract,
             metamorphic=step_metamorphic, golden=step_golden, tables=step_tables, mutation=step_mutation,
             handmutants=step_handmutants, isolation=step_isolation, wheel=step_wheel, generated=step_generated)
# Two chains run side by side (one process each); then the two steps that use both workers themselves, one after the other.
CHAINS = (("lint", "types", "suite", "coverage", "generated"), ("contract", "metamorphic", "golden", "tables", "isolation", "wheel"))
ALONE = ("mutation", "handmutants")


def versions() -> dict:
    code, out, _ = sh([PY, "-c", "import json, sys, numpy, scipy, mrgym; sys.path.insert(0, 'tests'); import golden_cases as G; "
                                 "print(json.dumps(dict(mrgym=mrgym.__version__, engine=mrgym.ENGINE_ID, numpy=numpy.__version__, "
                                 "scipy=scipy.__version__, stack=G.stack(), ids=mrgym.list_envs())))"])
    v = json.loads(out.strip().splitlines()[-1]) if code == 0 else {}
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    return dict(v, python=platform.python_version(), platform=platform.platform(), git_head=head, git_dirty=dirty)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=None, help="comma-separated step names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "audit_report.json"))
    ap.add_argument("--write-coverage", action="store_true",
                    help="after the suite and coverage rows: store the measured total as the new ratchet (only ever upwards)")
    a = ap.parse_args(argv)
    if a.list:
        print(__doc__)
        return 0
    chosen = list(STEPS) if not a.only else [s.strip() for s in a.only.split(",")]
    unknown = [s for s in chosen if s not in STEPS]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; steps: {', '.join(STEPS)}")
    if "coverage" in chosen and "suite" not in chosen:
        ap.error("the coverage row reads the data of the suite row: run both")
    t0 = time.time()
    results = {}
    work = tempfile.mkdtemp(prefix="mrgym-audit-")

    def chain(names):
        for n in names:
            if n in chosen:
                t, _PEAK.gb = time.time(), 0.0
                try:
                    results[n] = STEPS[n](work)
                except Exception as e:  # noqa: BLE001  a broken step is a failed row, not a lost report
                    results[n] = row("fail", n, round(time.time() - t, 1), error=f"{type(e).__name__}: {e}")
                results[n]["peak_rss_gb"] = round(_PEAK.gb, 2)
                print(f"  {n:<12} {results[n]['status']:<7} {results[n]['seconds']:>7.1f} s   peak {_PEAK.gb:4.2f} GB", flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(chain, CHAINS))
        chain(ALONE)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    allowed_absent = {ln.split()[0]: ln for ln in patterns("allowed_absent.txt")}
    rows, failed = {}, []
    for n in STEPS:
        if n not in results:
            continue
        r = results[n]
        if r["status"] == "absent" and n in allowed_absent:
            r["status"], r["allowed_because"] = "absent (allowed)", allowed_absent[n].split(None, 1)[1] if " " in allowed_absent[n] else ""
        if r["status"] not in ("pass", "absent (allowed)"):
            failed.append(n)
        rows[n] = r
    if a.write_coverage and "coverage" in rows and rows["coverage"].get("total_percent") is not None:
        path = os.path.join(HERE, "coverage_baseline.json")
        with open(path, encoding="utf-8") as f:
            base = json.load(f)
        if rows["coverage"]["total_percent"] > base["total_percent"]:
            base.update(total_percent=rows["coverage"]["total_percent"], recorded_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(base, f, indent=1)
                f.write("\n")
            print(f"coverage ratchet raised to {base['total_percent']} %")
    report = dict(schema="mrgym-audit/1", limits=dict(address_space_gb=MEM_GB, resident_gb=RSS_GB, resident_per_test_file_gb=FILE_RSS_LIMIT_GB), date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                  versions=versions(), wall_seconds=round(time.time() - t0, 1), steps_run=chosen, passed=not failed, failed=failed, rows=rows)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(f"audit: {len(rows) - len(failed)} of {len(rows)} rows pass in {report['wall_seconds']:.0f} s"
          + (f"; FAILED: {', '.join(failed)}" if failed else "") + f"; report: {os.path.relpath(a.out, ROOT)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
