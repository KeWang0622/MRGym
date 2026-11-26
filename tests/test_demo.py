"""`mrgym demo` says what each scripted policy is for: the solution, the shortcuts that are built to fail, the controls
that may pass."""
import contextlib
import io

import mrgym
from mrgym import cli, policies

SOLUTION = "scripted solution (no language model): shows the task can be solved through the tools"
SHORTCUTS = "shortcuts that ignore the object: built to fail (this is what makes the reward mean something)"
OTHERS = "other scripted controls: may pass"
CLOSING = "a language model lands between; `mrgym run-agent` plays one"


def demo(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["demo", *argv]) == 0
    return buf.getvalue().rstrip("\n").split("\n")


def rows(lines, title):
    start = lines.index(title) + 1
    out = {}
    for ln in lines[start:]:
        if not ln.startswith("  "):
            break
        name, _, rest = ln.strip().partition("  reward ")
        out[name.strip()] = (int(rest[0]), rest[1:].strip())
    return out


def test_the_demo_labels_the_solution_the_shortcuts_and_the_other_controls():
    lines = demo()
    controls = policies.controls_of(mrgym.spec("Scanner/NoWrap-v1"))
    assert lines[0] == "Scanner/NoWrap-v1 · level 2 · seed 0" and lines[1] == SOLUTION and lines[-1] == CLOSING
    assert lines.index(SOLUTION) < lines.index(SHORTCUTS) < lines.index(OTHERS)
    assert lines[lines.index(SHORTCUTS) - 1].startswith("  reward 1 ·")                 # the solution passes on this instance
    shortcuts, others = rows(lines, SHORTCUTS), rows(lines, OTHERS)
    assert list(shortcuts) == list(controls.MUST_FAIL) and all(reward == 0 for reward, _ in shortcuts.values())
    assert list(others) == [p for p in controls.policies() if p != controls.REFERENCE and p not in controls.MUST_FAIL]
    assert set(others) == set(controls.ABLATIONS) | set(controls.REPORTED)
    assert others["open_loop"][0] == 1                                                   # a control that may pass, and does here
    for name, (_, text) in {**shortcuts, **others}.items():                              # one line each: the family's own description
        full = controls.DESCRIPTIONS[name]
        assert text == full or (text.endswith("...") and full.startswith(text[:-3]) and len(text) <= cli.DEMO_WIDTH)
    assert max(len(ln) for ln in lines[lines.index(SHORTCUTS):]) <= 2 + 16 + 12 + cli.DEMO_WIDTH


def test_the_demo_of_another_policy_or_family_keeps_the_groups():
    lines = demo("--policy", "prior")
    assert lines[1] == "policy prior (scripted, no language model)" and "reference" in rows(lines, OTHERS) and "prior" not in rows(lines, SHORTCUTS)
    lines = demo("--only")
    assert SHORTCUTS not in lines and OTHERS not in lines and CLOSING not in lines and lines[1] == SOLUTION
    lines = demo("--id", "Scanner/NullTissue-v0")
    controls = policies.controls_of(mrgym.spec("Scanner/NullTissue-v0"))
    assert list(rows(lines, SHORTCUTS)) == list(controls.MUST_FAIL) and list(rows(lines, OTHERS)) == list(controls.ABLATIONS)
    assert cli._short("x" * 200) == "x" * (cli.DEMO_WIDTH - 3) + "..." and cli._short("short") == "short"
