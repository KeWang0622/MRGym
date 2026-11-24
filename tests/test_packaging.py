"""What a user gets from an installed package: the data files, the version, and a core that needs numpy and scipy
only. These tests must pass against a wheel installed outside the repository as well as against a checkout."""
import importlib.resources
import json
import os
import re
import subprocess
import sys

import mrgym
from mrgym import cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOT_IN_THE_CORE = ("matplotlib", "torch", "requests", "openai")


def test_datasheet_is_package_data():
    sheet = importlib.resources.files("mrgym.scanner").joinpath("datasheet.md")
    assert sheet.is_file()
    assert len(sheet.read_text(encoding="utf-8")) > 1000


def test_version_prints_package_and_engine(capsys):
    try:
        code = cli.main(["--version"])
    except SystemExit as e:                    # argparse's version action prints and exits
        code = e.code
    assert not code
    out = capsys.readouterr().out.strip()
    assert out == f"mrgym {mrgym.__version__} (engine {mrgym.ENGINE_ID})", out


def test_version_equals_pyproject():
    path = os.path.join(ROOT, "pyproject.toml")
    if not os.path.exists(path):               # an installed package tested without the repository
        return
    with open(path, encoding="utf-8") as f:
        project = f.read().split("[project]", 1)[1]
    declared = re.search(r'^version\s*=\s*"([^"]+)"', project, re.M).group(1)
    assert mrgym.__version__ == declared


def _modules_after(statement, cwd):
    """Top-level module names loaded in a fresh interpreter, started outside the repository, after ``statement``."""
    code = f"import json, sys\n{statement}\nprint(json.dumps(sorted({{m.split('.')[0] for m in sys.modules}})))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(cwd))
    assert r.returncode == 0, r.stderr[-2000:]
    return set(json.loads(r.stdout.strip().splitlines()[-1]))


def test_importing_the_package_loads_only_the_core(tmp_path):
    loaded = _modules_after("import mrgym, mrgym.cli", tmp_path)
    assert {"mrgym", "numpy"} <= loaded
    extra = sorted(loaded & set(NOT_IN_THE_CORE))
    assert not extra, f"importing mrgym loaded {extra}; the core is numpy, scipy and the standard library"


def test_the_command_line_loads_only_the_core(tmp_path):
    """``mrgym list`` builds the whole command line (recorder, viewer, model runner, results) and reads the cards."""
    loaded = _modules_after("import contextlib, io, mrgym.cli\nwith contextlib.redirect_stdout(io.StringIO()):\n    mrgym.cli.main(['list'])", tmp_path)
    extra = sorted(loaded & set(NOT_IN_THE_CORE))
    assert not extra, f"the command line loaded {extra}; the core is numpy, scipy and the standard library"
