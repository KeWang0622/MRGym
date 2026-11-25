"""The audit tools themselves (``tools/audit/``): the golden writer refuses a silent change, the wheel check sees a
missing or stray file, the release check sees versions that disagree, the hand mutants still find their lines, the
survivor rules parse. These run in the plain suite so a moved source line shows up long before ``make audit``."""
import importlib.util
import json
import os
import re
import sys
import zipfile

import pytest

import golden_cases as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools", "audit")
pytestmark = pytest.mark.skipif(not os.path.isdir(TOOLS), reason="tools/audit is not next to the tests (installed-wheel run outside a checkout)")


def tool(name):
    spec = importlib.util.spec_from_file_location(f"audit_tool_{name}", os.path.join(TOOLS, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------- golden.py
@pytest.fixture
def golden(tmp_path, monkeypatch):
    """golden.py on temporary files, with a small fake engine whose numbers the test controls."""
    g = tool("golden")
    state = dict(engine="1.0.0", stack=dict(numpy="2.2.6", scipy="1.15.3", machine="x86_64", simd=["A"]),
                 images={"qa/se": dict(sha256="aa", fingerprint=[[1.0, 2.0]], shape=[1, 2, 2])}, noisy={"0:se": dict(sha256="bb")},
                 prices={"se": dict(scan_s=67.5)})
    tasks = dict(engine="1.0.0", shared=dict(gate=dict(ALPHA=0.05)), ids={"Scanner/X-v0": dict(constants=dict(TOL=1.1))})
    monkeypatch.setattr(g.G, "ENGINE_FILE", str(tmp_path / "engine_digests.json"))
    monkeypatch.setattr(g.G, "TASK_FILE", str(tmp_path / "task_snapshots.json"))
    monkeypatch.setattr(g.G, "engine_digests", lambda: json.loads(json.dumps(state)))
    monkeypatch.setattr(g.G, "task_snapshots", lambda: json.loads(json.dumps(tasks)))
    return g, state, tasks, tmp_path


def _stored(tmp_path):
    return json.loads((tmp_path / "engine_digests.json").read_text())


def test_golden_write_then_check(golden, capsys):
    g, state, tasks, tmp = golden
    assert g.main(["--check"]) == 1                                      # no file yet
    assert g.main(["--write"]) == 0 and _stored(tmp) == state
    assert json.loads((tmp / "task_snapshots.json").read_text()) == tasks
    assert g.main(["--check"]) == 0
    with pytest.raises(SystemExit):
        g.main([])                                                      # exactly one of --write / --check


def test_golden_refuses_a_number_that_moved_under_the_same_engine_version(golden, capsys):
    g, state, tasks, tmp = golden
    assert g.main(["--write"]) == 0
    before = (tmp / "engine_digests.json").read_text()
    for section, key, field, value in (("images", "qa/se", "sha256", "cc"), ("noisy", "0:se", "sha256", "dd"), ("prices", "se", "scan_s", 39.0)):
        old = state[section][key][field]
        state[section][key][field] = value
        capsys.readouterr()
        assert g.main(["--check"]) == 1
        assert g.main(["--write"]) == 1 and (tmp / "engine_digests.json").read_text() == before        # nothing written
        assert "Bump it" in capsys.readouterr().out
        state[section][key][field] = old
    del state["images"]["qa/se"]                                         # a case that disappears is a change too
    assert g.main(["--write"]) == 1 and (tmp / "engine_digests.json").read_text() == before


def test_golden_accepts_the_same_change_after_an_engine_bump_and_new_cases_without_one(golden):
    g, state, tasks, tmp = golden
    assert g.main(["--write"]) == 0
    state["images"]["brain/se"] = dict(sha256="ee", fingerprint=[[3.0]], shape=[1, 1, 1])              # a new case: allowed
    assert g.main(["--check"]) == 1 and g.main(["--write"]) == 0 and "brain/se" in _stored(tmp)["images"]
    state["images"]["qa/se"]["sha256"] = "cc"
    state["engine"] = "1.1.0"                                            # the version moved with the number
    assert g.main(["--check"]) == 1                                      # the file is stale ...
    assert g.main(["--write"]) == 0                                      # ... and one command brings it up to date
    assert _stored(tmp)["engine"] == "1.1.0" and _stored(tmp)["images"]["qa/se"]["sha256"] == "cc"
    assert g.main(["--check"]) == 0


def test_golden_is_written_only_from_the_recorded_stack(golden, capsys):
    g, state, tasks, tmp = golden
    assert g.main(["--write"]) == 0
    state["stack"]["numpy"] = "2.5.5"
    state["images"]["qa/se"]["sha256"] = "ff"                            # another build: other last bits
    assert g.main(["--write"]) == 1 and "not the stack" in capsys.readouterr().out
    assert _stored(tmp)["stack"]["numpy"] == "2.2.6"
    assert g.main(["--write", "--new-reference-stack"]) == 0 and _stored(tmp)["stack"]["numpy"] == "2.5.5"


def test_golden_lists_the_ids_whose_snapshot_changed(golden, capsys):
    g, state, tasks, tmp = golden
    assert g.main(["--write"]) == 0
    tasks["ids"]["Scanner/X-v0"]["constants"]["TOL"] = 1.2
    capsys.readouterr()
    assert g.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "ids touched: Scanner/X-v0" in out and "changed: ids.Scanner/X-v0.constants.TOL" in out
    assert g.main(["--write"]) == 0 and g.main(["--check"]) == 0


def test_golden_mode_follows_the_stack(monkeypatch):
    now = G.stack()
    monkeypatch.delenv("MRGYM_GOLDEN_MODE", raising=False)
    assert G.mode(now)[0] == "bit-exact"
    assert G.mode(dict(now, numpy="0.0"))[0] == "tolerance" and "numpy" in G.mode(dict(now, numpy="0.0"))[1]
    assert G.mode(dict(now, simd=["other"]))[0] == "tolerance"
    monkeypatch.setenv("MRGYM_GOLDEN_MODE", "tolerance")
    assert G.mode(now) == ("tolerance", "MRGYM_GOLDEN_MODE")


def test_golden_dump_keeps_number_lists_on_one_line(tmp_path):
    path = str(tmp_path / "x.json")
    obj = dict(a=dict(fingerprint=[[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]], shape=[1, 2, 3]), b="text")
    G.dump(path, obj)
    text = open(path).read()
    assert G.load(path) == obj and "[1.5, 2.5, 3.5]" in text and text.endswith("\n")


# ---------------------------------------------------------------------------------------------- wheel_vs_git.py
def _wheel(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, "")
    return str(path)


def test_wheel_check_wants_exactly_the_tracked_package_files(tmp_path, monkeypatch, capsys):
    w = tool("wheel_vs_git")
    tracked = {"mrgym/__init__.py", "mrgym/scanner/datasheet.md"}
    monkeypatch.setattr(w, "tracked", lambda: (tracked, "a fixed list"))
    meta = ["mrgym-0.1.dist-info/METADATA", "mrgym-0.1.dist-info/RECORD"]
    assert w.main([_wheel(tmp_path / "ok.whl", [*tracked, *meta])]) == 0
    assert w.main([_wheel(tmp_path / "missing.whl", ["mrgym/__init__.py", *meta])]) == 1
    assert "missing from the wheel: mrgym/scanner/datasheet.md" in capsys.readouterr().out
    assert w.main([_wheel(tmp_path / "stray.whl", [*tracked, "mrgym/local_results.json", *meta])]) == 1
    assert "in the wheel but not tracked: mrgym/local_results.json" in capsys.readouterr().out
    assert w.main([_wheel(tmp_path / "tests.whl", [*tracked, "tests/test_env.py", *meta])]) == 1
    assert "outside the package: tests/test_env.py" in capsys.readouterr().out
    assert w.main([]) == 2


def test_wheel_check_reads_the_tracked_files_of_this_checkout():
    files, source = tool("wheel_vs_git").tracked()
    assert "mrgym/__init__.py" in files and "mrgym/scanner/datasheet.md" in files and "mrgym/envs/family.py" in files
    assert not any(f.endswith(".pyc") for f in files) and source


# ---------------------------------------------------------------------------------------------- release_check.py
def test_release_check(monkeypatch):
    r = tool("release_check")
    v = r.versions()
    assert v["pyproject"] == v["package"] and v["pyproject"] in v["changelog"]            # true of this tree
    monkeypatch.setattr(r, "versions", lambda: dict(pyproject="0.2.0", package="0.2.0", changelog=["0.2.0", "0.1.0"]))
    assert r.problems("v0.2.0") == []
    assert r.problems("0.2.0") == ["the tag '0.2.0' does not look like v<version>"]
    assert r.problems("v0.3.0") == ["the tag is v0.3.0, the package version is 0.2.0", "CHANGELOG.md has no heading '## 0.3.0' (headings: ['0.2.0', '0.1.0'])"]
    monkeypatch.setattr(r, "versions", lambda: dict(pyproject="0.2.0", package="0.2.1", changelog=["0.2.0"]))
    assert r.problems("v0.2.0") == ["pyproject.toml says 0.2.0, mrgym.__version__ says 0.2.1"]
    monkeypatch.setattr(r, "versions", lambda: dict(pyproject="0.2.0.dev0", package="0.2.0.dev0", changelog=["0.2.0.dev0"]))
    assert r.problems("v0.2.0.dev0") == ["0.2.0.dev0 is a pre-release: publish it as a GitHub pre-release only"]
    assert r.main(["v0.2.0.dev0"]) == 1 and r.main([]) == 2


# ---------------------------------------------------------------------------------------------- hand_mutants.py, mutation.py
def test_every_hand_mutant_still_finds_its_line():
    h = tool("hand_mutants")
    assert [m[0] for m in h.MUTANTS] == [f"M{i:02d}" for i in range(1, len(h.MUTANTS) + 1)] and len(h.REVIEW) == 16
    for mid, kind, what, path, old, new in h.MUTANTS:
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            text = f.read()
        assert text.count(old) == 1, f"{mid} ({what}): the line to edit occurs {text.count(old)} times in {path}; update tools/audit/hand_mutants.py"
        assert old != new and kind in ("physics", "scoring", "other data", "constant")
    assert all(os.path.exists(os.path.join(ROOT, f)) for files in h.GROUPS.values() for f in files)


def test_survivor_rules_parse_and_name_functions_of_the_mutated_files():
    m = tool("mutation")
    rules = m.load_rules()
    assert rules and all(cls in ("equivalent", "equivalent in range", "performance", "text") and reason for _, _, cls, reason in rules)
    assert all(glob.startswith("mrgym.") for glob, _, _, _ in rules)
    assert m.function_of("mrgym.gate.x__play__mutmut_35") == "mrgym.gate._play"
    assert m.function_of("mrgym.envs.null_tissue.task.x_score__mutmut_2") == "mrgym.envs.null_tissue.task.score"
    assert m.check_pragmas() == []                                       # nothing in src/ is excluded from mutation
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        selection = re.search(r"pytest_add_cli_args_test_selection = \[(.*?)\]", f.read(), re.S).group(1)
    for path in re.findall(r'"([^"]+)"', selection):
        assert os.path.exists(os.path.join(ROOT, path)), f"[tool.mutmut] selects {path}, which does not exist"


def test_audit_report_reads_a_pytest_summary_line():
    r = tool("report")
    assert r.pytest_counts("....\n12 passed, 2 skipped, 3 xfailed in 4.21s\n") == dict(passed=12, skipped=2, xfailed=3)
    assert r.pytest_counts("1 failed, 5 passed in 1.0s") == dict(failed=1, passed=5)
    assert r.pytest_counts("no tests ran") == {}
    assert r.percent("82\nCoverage failure: total of 82 is less than fail-under=90\n") == 82.0 and r.percent("99.5\n") == 99.5
    assert r.percent("No data to report.") is None
    assert r.row("pass", ["a", "b"], 1.5, n=3) == dict(status="pass", command="a b", seconds=1.5, n=3)
    assert set(r.CHAINS[0] + r.CHAINS[1] + r.ALONE) == set(r.STEPS)
    assert r.patterns("allowed_absent.txt") == [] and all(os.path.exists(os.path.join(ROOT, f)) for f in r.ISOLATION)
    assert r.patterns("allowed_xfails.txt") == [] and len(r.patterns("allowed_skips.txt")) >= 3


def test_audit_report_runs_a_command_under_the_memory_limit_and_notes_its_peak():
    r = tool("report")
    code, out, seconds = r.sh([sys.executable, "-c", "print('x' * 10)"])
    assert (code, out.strip()) == (0, "x" * 10) and r._PEAK.gb > 0.0            # (no upper bound: the peak of waited children is cumulative in this process)
    old, r.MEM_GB = r.MEM_GB, 1.0                                        # the optional address-space limit (ulimit -v)
    try:
        code, out, _ = r.sh([sys.executable, "-c", "b = bytearray(int(1.5 * 2 ** 30)); print(len(b))"])
    finally:
        r.MEM_GB = old
    assert code != 0 and "MemoryError" in out                            # a leak ends its own step, not the machine
    code, out, _ = r.sh([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert code != 0 and "[audit] killed: no result after 0.5 s" in out
    grow = "import time\nb = []\nfor _ in range(40):\n    b.append(bytearray(32 * 2 ** 20)); time.sleep(0.1)\nprint('not stopped')"
    old, r.RSS_GB = r.RSS_GB, 0.25                                       # the watchdog: resident memory of the process tree
    try:
        code, out, seconds = r.sh([sys.executable, "-c", grow])
    finally:
        r.RSS_GB = old
    assert code != 0 and "[audit] killed: resident memory" in out and "not stopped" not in out and seconds < 4.0
    assert r._PEAK.gb > 0.2 and r.tree_rss_gb(os.getpid()) > 0.02


def test_audit_report_names_the_least_covered_modules_and_holds_the_ratchet():
    r = tool("report")
    table = "\n".join(["Name   Stmts   Miss  Cover", "src/mrgym/a.py   100   60   40.0%", "src/mrgym/tiny.py   5   5   0.0%",
                        "src/mrgym/b.py   200   20   90.0%", "src/mrgym/c.py   50   25   50.0%", "src/mrgym/d.py   30   0   100.0%", "TOTAL 385 110 71.4%"])
    assert [x["module"] for x in r.least_covered(table)] == ["a.py", "c.py", "b.py"]
    assert r.least_covered(table)[0] == dict(module="a.py", statements=100, missed=60, percent=40.0)
    with open(os.path.join(TOOLS, "coverage_baseline.json"), encoding="utf-8") as f:
        base = json.load(f)
    assert base["scoring_scope_percent"] == 100.0 and 50.0 < base["total_percent"] <= 100.0
    assert "admission" not in r.SCORING_SCOPE and "family.py" in r.SCORING_SCOPE


def test_audit_leaves_the_test_suite_commands_of_the_readme_to_the_suite_row(tmp_path):
    r = tool("report")
    path, left_out = r.readme_without_the_suite(str(tmp_path))
    text = open(path, encoding="utf-8").read()
    assert left_out and all(r.SUITE_COMMANDS.match(ln) for ln in left_out)
    assert not any(r.SUITE_COMMANDS.match(ln) for ln in text.split("\n")) and "mrgym demo" in text
    for cmd, suite in (("pytest -q", True), ("python -m pytest", True), ("pip install pytest", True), ("make test", True),
                       ("mrgym demo", False), ("pip install -e .", False), ("mrgym validate", False), ("python tools/generate.py --check", False)):
        assert bool(r.SUITE_COMMANDS.match(cmd)) is suite, cmd
