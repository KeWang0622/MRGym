"""Every script in ``examples/textbook`` runs as a subprocess and exits 0: each one asserts its textbook result."""
import os
import pathlib
import subprocess
import sys

import pytest

FOLDER = pathlib.Path(__file__).resolve().parents[1] / "examples" / "textbook"
SCRIPTS = sorted(FOLDER.glob("[0-9][0-9]_*.py"))


def test_all_twelve_examples_are_present():
    assert [p.name[:2] for p in SCRIPTS] == [f"{i:02d}" for i in range(1, 13)]


def test_readme_lists_every_example():
    readme = (FOLDER / "README.md").read_text(encoding="utf-8")
    assert all(p.name in readme for p in SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs_and_asserts(script):
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, env=env, cwd=str(FOLDER))
    assert done.returncode == 0, f"{script.name} failed\n--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}"
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    assert 3 <= len(lines) <= 8, f"{script.name} printed {len(lines)} lines"
