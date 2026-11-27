"""examples/new_family is the skeleton of docs/ADDING_A_TASK.md: it imports, its reference policy passes, its shortcuts
fail, the package discovers a copy of it, and the code the page shows is the code of the example."""
import importlib
import os
import re
import shutil
import subprocess
import sys

import pytest

import mrgym
from mrgym import cards, gate, policies

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(EXAMPLES, "new_family")), reason="needs a checkout: examples/ is not part of the wheel")
ENV_ID = "Example/T2Vials-v0"


@pytest.fixture(scope="module")
def family():
    """The example, registered for this module only (the registry of the other tests does not see it)."""
    sys.path.insert(0, EXAMPLES)
    try:
        spec = importlib.import_module("new_family.spec")
        assert [kw["id"] for kw in spec.SPECS] == [ENV_ID] and spec.SPECS[0]["kind"] == "candidate"
        mrgym.register(**spec.SPECS[0])
        yield importlib.import_module("new_family.task"), importlib.import_module("new_family.controls")
    finally:
        mrgym.registry.pop(ENV_ID, None)
        sys.path.remove(EXAMPLES)
        for name in [n for n in sys.modules if n == "new_family" or n.startswith("new_family.")]:
            del sys.modules[name]


def test_the_example_plays_its_reference_passes_and_its_shortcuts_fail(family):
    task, controls = family
    env = mrgym.make(ENV_ID, level=1, seed=0)
    obs, _ = env.reset()
    assert "Deliver T2 of every region in milliseconds" in obs.text and len(env.episode_config()["task"]["regions"]) == 3
    for seed in (0, 1):
        step = policies.run(mrgym.make(ENV_ID, level=1, seed=seed), controls.REFERENCE)
        assert step.reward == 1.0 and step.info["policy_error"] is None and step.info["budget"]["scans"] == 2, seed
    for name in controls.MUST_FAIL:
        assert policies.run(mrgym.make(ENV_ID, level=1, seed=0), name).reward == 0.0, name
    assert set(controls.policies()) == {controls.REFERENCE, *controls.MUST_FAIL, *controls.ABLATIONS, *controls.REPORTED}
    assert "open_loop" in controls.REPORTED and set(controls.DESCRIPTIONS) == set(controls.policies())
    card = mrgym.card(ENV_ID)                                               # complete, and renders
    assert cards.lint(card) == [] and "T2 of three marked vials" in cards.markdown(card)
    refused = env.step({"name": "submit", "arguments": {"t2_ms": {"R1": 80.0}}})     # not every region: refused, the episode goes on
    assert refused.observation.is_error and not refused.terminated


def test_a_small_gate_separates_the_reference_from_the_shortcuts(family):
    result = gate.run(ENV_ID, n=6, workers=1, blocks=1)
    level = result["levels"]["1"]
    assert level["reference"]["passed"] == 6 and all(r["passed"] == 0 for r in level["must_fail"])
    assert {r["policy"].split("#")[0] for r in level["must_fail"]} == {"prior", "fixed_recipe", "wrong_units"}
    assert [r["policy"] for r in level["reported"]] == ["open_loop"] and level["reported"][0]["passed"] == 6


def test_the_page_shows_the_code_of_the_example_and_stays_short():
    with open(os.path.join(ROOT, "docs", "ADDING_A_TASK.md"), encoding="utf-8") as f:
        page = f.read()
    assert len(page.rstrip("\n").split("\n")) <= 120
    files = {}
    for name in ("spec.py", "task.py", "controls.py"):
        with open(os.path.join(EXAMPLES, "new_family", name), encoding="utf-8") as f:
            files[name] = f.read()
    blocks = re.findall(r"```python\n(.*?)```", page, re.S)
    shown = [next((n for n, text in files.items() if b in text), None) for b in blocks]
    assert shown == ["spec.py", "task.py", "controls.py"]                    # each block is a verbatim part of its file
    assert "def make_instance(" in blocks[1] and "def score(" in blocks[1] and "def reference(sc):" in blocks[2]
    for word in ("mrgym gate <id> --n 60 --workers 2", "mrgym gate <id> --n 300 --write", "`candidate`", "`admitted`", "`practice`",
                 "`open_loop`", "## Checklist", "10. "):
        assert word in page, word
    for rel in ("README.md", "CONTRIBUTING.md"):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            assert "docs/ADDING_A_TASK.md" in f.read(), rel


def test_a_copy_of_the_example_under_envs_is_found_by_the_package(tmp_path):
    """What the page says: copy the folder to src/mrgym/envs/<your_task>/ and no other file is edited."""
    shutil.copytree(os.path.join(ROOT, "src", "mrgym"), tmp_path / "mrgym", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(EXAMPLES, "new_family"), tmp_path / "mrgym" / "envs" / "t2_vials", ignore=shutil.ignore_patterns("__pycache__", "play.py"))
    code = ("import mrgym; from mrgym import policies; s = mrgym.spec('Example/T2Vials-v0'); "
            "print(s.family, s.kind, policies.run(mrgym.make(s.id, level=1, seed=3), 'reference').reward)")
    env = dict(os.environ, PYTHONPATH=str(tmp_path), OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.split() == ["mrgym.envs.t2_vials", "candidate", "1.0"]
