"""The reporting rules (``mrgym.stats.lint``) as ``mrgym results --check`` applies them.

``tests/data/lint/`` holds one valid ``results.json``, one valid page, and one file per rule with one kind of
violation; each makes ``mrgym results --check --lint FILE`` exit 1 and name the rule and the line. The repository's
own pages pass, and re-inserting two sentences that an earlier review found false fails.
"""
import importlib.util
import json
import os
import re

import pytest

import synth
from mrgym import cli, results
from mrgym.stats import lint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data", "lint")
FIXTURES = {
    "R01_rate_typed_by_hand.md": "R1", "R02_unit_is_the_instance.json": "R2", "R03_claim_without_test.md": "R3",
    "R04_seed_sets_differ.json": "R4", "R05_no_command.json": "R5", "R06_certainty_without_bound.md": "R6",
    "R07_pilot_in_comparison.json": "R7", "R08_no_ablation_row.json": "R8", "R09_selected_subset.md": "R9",
    "R10_infra_errors.json": "R10", "R11_no_plan.json": "R11", "R12_no_cost.json": "R12",
    "R13_no_conversation.json": "R13", "D01_too_few_instances.json": "D1",
}
TWELVE = [f"R{i}" for i in range(1, 13)]


def run_lint(capsys, *names, context="valid.json"):
    argv = ["results", "--check", "--lint", *[os.path.join(DATA, n) for n in names]]
    if context and any(n.endswith(".md") for n in names):
        argv += ["--context", context if os.path.isabs(context) else os.path.join(DATA, context)]
    code = cli.main(argv)
    return code, capsys.readouterr().out


def test_fixture_files_are_the_generated_ones():
    made = synth.lint_fixtures()
    assert sorted(made) == sorted(os.listdir(DATA)) and set(FIXTURES) | {"valid.json", "valid.md", "M1_no_model_has_been_run.md", "M4_same_instances.md"} == set(made)
    for name, text in made.items():
        with open(os.path.join(DATA, name), encoding="utf-8") as f:
            assert f.read() == text, name
    assert sorted(set(FIXTURES.values()) & set(TWELVE), key=TWELVE.index) == TWELVE          # one fixture for each of the twelve rules


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_each_fixture_breaks_its_rule_and_only_that_rule(name, capsys):
    rule = FIXTURES[name]
    code, out = run_lint(capsys, name)
    lines = out.strip().splitlines()
    assert code == 1 and lines
    for line in lines:
        m = re.match(rf"{re.escape(os.path.join(DATA, name))}:(\d+): (\w+) \((.+?)\): .+", line)
        assert m and m[2] == rule and m[3] == lint.RULES[rule], line
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        text = f.read().split("\n")
    number = int(re.match(r".*?:(\d+):", lines[0])[1])
    if name.endswith(".md"):
        assert number == 5 and text[number - 1].strip()                     # the sentence is on line 5 of the page
    else:
        assert text[number - 1] == "  {"                                    # the line on which the offending row starts
        if rule not in ("R7", "R8", "R11"):
            assert '"agent": "model-' in "\n".join(text[number:number + 4])


def test_valid_fixtures_pass(capsys):
    code, out = run_lint(capsys, "valid.json", "valid.md")
    assert code == 0 and "no reporting rule is broken" in out
    with open(os.path.join(DATA, "valid.json"), encoding="utf-8") as f:
        assert lint.lint_results(json.load(f)) == []


def test_sentences_an_earlier_review_found_false(capsys):
    code, out = run_lint(capsys, "M1_no_model_has_been_run.md")
    assert code == 1 and ": R1 (" in out and ":5:" in out
    repo = os.path.join(ROOT, "results", "results.json")                   # models played a subset of the scripted rows' seeds
    code, out = run_lint(capsys, "M4_same_instances.md", context=repo)
    assert code == 1 and ": R4 (" in out and "different seed sets" in out
    code, out = run_lint(capsys, "M4_same_instances.md", context=None)
    assert code == 1 and "no results support it" in out
    code, _ = run_lint(capsys, "M4_same_instances.md")                     # true of the synthetic run: every row has one seed set
    assert code == 0


def load_generate():
    spec = importlib.util.spec_from_file_location("generate", os.path.join(ROOT, "tools", "generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_repository_passes_and_reinserting_the_sentences_fails():
    generate = load_generate()
    assert generate.lint_docs({}) == []
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    with open(os.path.join(ROOT, "results", "results.json"), encoding="utf-8") as f:
        summary = json.load(f)
    assert lint.lint_text(readme, summary) == []
    anchor = "<!-- RESULTS:START -->\n"                                       # the results section of the page, wherever it is folded
    m1 = readme.replace(anchor, "No language model has been run yet.\n\n" + anchor)
    m4 = readme.replace(anchor, "Rows of kind `model` are language models; every agent plays the same instances.\n\n" + anchor)
    assert m1 != readme and [v["rule"] for v in lint.lint_text(m1, summary)] == ["R1"]
    assert [v["rule"] for v in lint.lint_text(m4, summary)] == ["R4"]
    found = generate.lint_docs({"README.md": m1})
    assert len(found) == 1 and found[0].startswith("README.md:") and ": R1 (" in found[0]
    line = int(found[0].split(":")[1])
    assert m1.split("\n")[line - 1] == "No language model has been run yet."
    block = results.START + "\nkimi-k3 passes 9/20 and is better than claude-haiku-5-5.\n" + results.END
    assert lint.lint_text(block, summary) == []                              # generated text is trusted


def test_text_rules_tell_results_from_rules_and_navigation():
    s = synth.summary()

    def broken(text):
        return [v["rule"] for v in lint.lint_text(text, s)]

    assert broken("model-a passes 7/10 episodes of level 2.") == ["R1"]
    assert broken("The reference passes 24 of 24 instances (100 %).") == ["R1", "R6"]
    assert broken("The reference passes 297/300, with a lower bound of 97.6 %.") == ["R1"]          # a bound does not make a typed rate generated
    assert broken("A level is admitted when the reference passes at least 90 % of the instances.") == []
    assert broken("A must-fail policy is cleared if it passes at most 21 of 300 instances.") == []
    assert broken("The repetition time is 24 % shorter and the matrix is 96/128 of the default.") == []     # numbers, not results
    assert broken("Run it as `mrgym run-agent --seeds 0-23`; the reference passes `24/24` in the demo.") == []     # inline code is trusted
    assert broken("```text\nreference 24/24 passed (100 %)\n```\n") == []
    assert broken("The world seed is never written, and all levels use the same tools.") == []      # not about a rate
    assert broken("The reference never fails.") == ["R6"] and broken("No model passes all instances.") == ["R6"]
    assert broken("The reference never fails: its one-sided lower bound is given in the table.") == []
    assert broken("model-a outperforms model-b.") == ["R3"] and broken("model-a is ahead of model-b on NullTissue-v0.") == ["R3"]
    assert broken("NullTissue-v0 is easier than NoWrap-v0 for every model.") == ["R3"]
    assert broken("model-a is ahead of model-b <!-- CMP:NullTissue-v0:model-a-vs-model-b -->.") == []       # p (Holm) below 0.05
    assert broken("model-a is ahead of model-b <!-- CMP:no-such-comparison -->.") == ["R3"]
    weak = json.loads(json.dumps(s))
    weak["comparisons"][0]["p_holm"] = 0.2
    assert [v["rule"] for v in lint.lint_text("model-a is ahead of model-b <!-- CMP:NullTissue-v0:model-a-vs-model-b -->.", weak)] == ["R3"]
    assert broken("See the table below for level 2 and level 3; level 1 is described above.") == []
    assert broken("The reference is better documented than the prior policy.") == ["R3"]            # a known cost: comparative words are read literally
    assert broken("On admitted instances the reference passes <!-- STAT:x -->1/2<!-- /STAT -->; over all instances "
                  "<!-- STAT:y -->1/3<!-- /STAT -->.") == []
    assert broken("With the best checkpoint the model passes <!-- STAT:x -->1/2<!-- /STAT -->.") == ["R9"]
    assert broken("<!-- Generated by tools/generate.py -->\nmodel-a passes 7/10 and is better than model-b.\n") == []


def test_line_numbers_of_text_violations():
    text = "# Title\n\nFirst paragraph\nruns over two lines.\n\n- a list item\n- model-a always passes\n\n| a | b |\n|---|---|\n| reference | passes 3/4 |\n"
    found = lint.lint_text(text, synth.summary())
    assert [(v["rule"], v["line"]) for v in found] == [("R6", 7), ("R1", 11)]
    wrapped = "Intro sentence. The next one is long and\nwraps: model-a passes\n7/10 episodes.\n"
    assert [(v["rule"], v["line"]) for v in lint.lint_text(wrapped, synth.summary())] == [("R1", 1)]
    assert lint.locate('{\n "rows": [\n  {\n   "a": 1\n  },\n  {\n   "a": 2\n  }\n ]\n}', "rows[1]") == 6 and lint.locate("", "rows[0]") == 1


def test_documents_state_the_constants_of_the_code():
    from mrgym.stats import gate, power
    assert (lint.MIN_INSTANCES, lint.MIN_EPISODES, lint.PILOT_BELOW) == (power.PUBLISHED_MIN_INSTANCES, power.PUBLISHED_MIN_EPISODES, power.PILOT_BELOW) == (300, 2, 100)
    with open(os.path.join(ROOT, "docs", "ANALYSIS_PLAN.md"), encoding="utf-8") as f:
        plan = f.read()
    assert f"{power.PUBLISHED_MIN_INSTANCES} instances per environment and {power.PUBLISHED_MIN_EPISODES} episodes per instance" in plan
    assert f"fewer than {power.PILOT_BELOW} instances" in plan and f"at most {100 * gate.ALPHA:.0f} %" in plan
    assert power.power_markdown() in plan and power.gate_text() in plan           # the generated blocks are current
    with open(os.path.join(ROOT, "results", "README.md"), encoding="utf-8") as f:
        page = f.read()
    for rule in lint.RULES:
        assert f"| {rule} |" in page, rule
    with open(os.path.join(ROOT, "DATASHEET.md"), encoding="utf-8") as f:
        assert power.power_markdown() in f.read()
