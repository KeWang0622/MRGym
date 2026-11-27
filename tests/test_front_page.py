"""The front page and the layout of the repository: README.md is short, runs as written and uses plain words; the
pages it links exist; the root holds what a user of the environment expects and nothing else."""
import importlib.util
import os
import re
import subprocess

import pytest

import mrgym
from mrgym import cards

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "README.md")), reason="needs a checkout: the pages are not part of the wheel")
# what the root of the repository holds (dot-files aside)
ROOT_ENTRIES = {"README.md", "LICENSE", "CITATION.cff", "CONTRIBUTING.md", "CHANGELOG.md", "SECURITY.md", "AGENTS.md", "pyproject.toml", "uv.lock",
                "Makefile", "src", "tests", "docs", "examples", "envs", "assets", "integrations", "recipes", "tools"}
# words of the project that the front page does not use; envs/README.md and docs/ADDING_A_TASK.md define them
INTERNAL = r"\b(admitted|practice|candidate|gate|gated|must[- ]fail|open[- _]loop|claim levels?|family contract|merit|pre-?regist\w*)\b"


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_python_block_of_the_front_page_runs_as_written_and_ends_with_reward_1():
    blocks = re.findall(r"```python\n(.*?)```", _read("README.md"), re.S)
    assert len(blocks) == 1 and len(blocks[0].splitlines()) <= 14
    scope = {}
    exec(compile(blocks[0], "README.md", "exec"), scope)                      # noqa: S102  the page's own example
    assert scope["reward"] == 1.0 and scope["terminated"] and not scope["truncated"]
    assert scope["tools"] and "1.0 True" in blocks[0]                         # what the comment of the last line says


def test_the_front_page_is_short_and_uses_plain_words():
    readme = _read("README.md")
    lines = readme.rstrip("\n").split("\n")
    assert len(lines) <= 120
    code = False
    for n, ln in enumerate(lines, 1):
        code = code != ln.startswith("```")
        if not code and not ln.startswith("|"):
            assert len(ln) <= 160, f"README.md:{n}: {len(ln)} characters"
    prose = re.sub(r"```.*?```|\]\([^)]*\)|<!--.*?-->", " ", readme, flags=re.S)         # code, link targets and markers are not prose
    assert re.findall(INTERNAL, prose, re.I) == []
    assert "<details>" not in readme and readme.count("<img ") in (3, 4)    # the scanner, the training figure, the licence badge; the CI badge where CI runs
    assert not re.search(r"^[ \t]+[-*] ", re.sub(r"```.*?```", "", readme, flags=re.S), re.M)       # no nested list
    for page in ("envs/README.md", "docs/ADDING_A_TASK.md"):                 # where the words of the registry are defined
        assert all(w in _read(*page.split("/")) for w in ("admitted", "practice", "candidate")), page


def test_the_task_table_has_one_row_per_registered_id_with_its_card_and_a_plain_status():
    generate = _tool("generate")
    table = generate.tasks_table()
    assert table in _read("README.md")
    rows = [ln.split(" | ") for ln in table.splitlines()[2:]]
    assert sorted(re.search(r"`([^`]+)`", r[0])[1] for r in rows) == sorted(mrgym.list_envs())
    for name, what, status in rows:
        env_id = re.search(r"`([^`]+)`", name)[1]
        assert f"](envs/{cards.slug(env_id)}/)" in name and os.path.exists(os.path.join(ROOT, "envs", cards.slug(env_id), "README.md"))
        assert status.rstrip(" |") == generate.STATUS[mrgym.spec(env_id).kind] and 3 <= len(what.split()) <= 10, env_id
    assert set(generate.STATUS) == set(cards.KINDS) and generate.STATUS["admitted"] == "ready"
    statuses = [r[2].rstrip(" |") for r in rows]
    assert statuses == sorted(statuses, key=["ready", "experimental", "no score"].index)        # ready tasks first


def test_what_left_the_front_page_is_generated_on_the_tour_page():
    generate = _tool("generate")
    assert generate.BLOCKS == {"README.md": ("TASKS",), "docs/TOUR.md": ("COUNTS", "DEMO", "RESULTS", "TIMING")}
    tour = _read("docs", "TOUR.md")
    for name in generate.BLOCKS["docs/TOUR.md"]:
        assert len(generate._get_block(tour, name).strip()) > 40, name
    for head in ("## What `mrgym demo` prints", "## Limits", "## Repository map", "## Results of API models"):
        assert head in tour


def test_every_relative_link_of_the_pages_resolves():
    links = _tool("check_links")
    import glob
    pages = sorted({p for pat in links.PAGES for p in glob.glob(os.path.join(ROOT, pat), recursive=True)})
    assert len(pages) > 30 and [b for p in pages for b in links.broken(p)] == []
    # the checker notices a missing file and a missing heading
    assert links.anchors(os.path.join(ROOT, "docs", "TOUR.md")) >= {"limits", "what-mrgym-demo-prints"}


def test_the_docs_index_is_ten_links_with_a_few_words_each():
    index = _read("docs", "README.md")
    first = index.split("\n\n")[1].splitlines()
    assert len(first) == 10
    for ln in first:
        m = re.fullmatch(r"- \[([^\]]+)\]\(([^)]+)\): (.+)", ln)
        assert m and len(m[3].split()) <= 5 and os.path.exists(os.path.normpath(os.path.join(ROOT, "docs", m[2]))), ln


def test_the_root_holds_what_a_user_expects_and_nothing_else():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("needs a git checkout")
    top = {p.split("/")[0] for p in out.stdout.splitlines()}
    assert {p for p in top if not p.startswith(".")} == ROOT_ENTRIES
