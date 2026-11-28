"""``mrgym.bestknown``: a record table changes only by the rule it states (fresh worlds, no pass below the bound,
yield, a paired margin), and the stored table of ``Scanner/RFPulse-v1`` is what the rule gives."""
import json

import numpy as np

import mrgym
from mrgym import bestknown, cli


def rows(ratios, passed=None, start=0):
    passed = [True] * len(ratios) if passed is None else passed
    return [dict(index=start + i, passed=bool(p), ratio=float(r)) for i, (r, p) in enumerate(zip(ratios, passed))]


def test_the_rule_for_a_new_record():
    rng = np.random.default_rng(5)
    record = rows(1.10 + 0.02 * rng.standard_normal(60))
    better = rows([r["ratio"] - 0.05 for r in record])
    assert bestknown.accept(record, better)["accepted"] and bestknown.accept(record, better)["margin_lower95"] > bestknown.MARGIN
    same = rows(1.10 + 0.02 * rng.standard_normal(60))                   # an equal design is not a record by chance
    assert not bestknown.accept(record, same)["accepted"] and "margin" in bestknown.accept(record, same)["reason"]
    assert not bestknown.accept(record, record)["accepted"]              # the stored record claimed again
    slightly = rows([r["ratio"] - 0.005 for r in record])                # a gain below the margin
    assert not bestknown.accept(record, slightly)["accepted"]
    below = rows([0.95] + [r["ratio"] - 0.05 for r in record[1:]])       # one pass below the bound: a defect, never a record
    got = bestknown.accept(record, below)
    assert not got["accepted"] and got["quarantined"] and "beats_bound" in got["reason"]
    failing = rows([r["ratio"] - 0.05 for r in record], [i >= 6 for i in range(60)])    # 54 of 60 pass: the yield is not shown
    assert "yield" in bestknown.accept(record, failing)["reason"]
    assert "worlds_reused" in bestknown.accept(record, better, used_indices=[3])["reason"]
    assert "not_paired" in bestknown.accept(record, rows([1.0] * 60, start=100))["reason"]
    s = bestknown.summarise(failing)
    assert (s["n"], s["passed"]) == (60, 54) and s["yield_lower95"] < 0.9 and bestknown.summarise([])["passed"] == 0


def test_a_table_is_built_by_the_rule_and_printed():
    rng = np.random.default_rng(6)
    first = {"class a": rows(1.2 + 0.01 * rng.standard_normal(40)), "class b": rows(1.2 + 0.01 * rng.standard_normal(40))}
    second = {"class a": rows([r["ratio"] - 0.1 for r in first["class a"]]), "class b": rows([r["ratio"] + 0.1 for r in first["class b"]])}
    t = bestknown.build("X/Y-v0", "0.0.0", dict(split="dev", seeds="records:0:0 .. records:0:39", n_per_level=40), dict(old=first, new=second), "cmd")
    assert t["classes"]["class a"]["record"] == "new" and t["classes"]["class b"]["record"] == "old" and not t["quarantined"]
    text = bestknown.text_table(t)
    assert "class a" in text and "new" in text and "paired gain" in text
    wrong = dict(old=first, new={"class a": rows([0.9] * 40), "class b": second["class b"]})
    t = bestknown.build("X/Y-v0", "0.0.0", dict(split="dev", seeds="s", n_per_level=40), wrong, "cmd")
    assert t["classes"]["class a"]["record"] == "old" and t["quarantined"] == [dict(claimant="new", specification_class="class a", reason="beats_bound")]
    weak = dict(old={"class a": rows([1.0] * 40, [i >= 4 for i in range(40)])}, new={"class a": rows([1.3] * 40)})     # 36 of 40: no yield shown
    t = bestknown.build("X/Y-v0", "0.0.0", dict(split="dev", seeds="s", n_per_level=40), weak, "cmd")
    assert t["classes"]["class a"]["record"] == "new" and "yield" in t["classes"]["class a"]["decisions"]["old"]["reason"]


def test_the_stored_table_of_the_design_family(capsys):
    t = bestknown.load("Scanner/RFPulse-v1")
    assert t is not None and t["engine"] == mrgym.ENGINE_ID and t["worlds"]["split"] == "dev" and not t["quarantined"]
    assert len(t["classes"]) == 9 and all(set(c["claimants"]) == {"constructive", "reference"} for c in t["classes"].values())
    for key, c in t["classes"].items():
        con, ref = c["claimants"]["constructive"], c["claimants"]["reference"]
        assert con["n"] == ref["n"] >= 30 and con["ratio_best"] >= 1.0 and ref["ratio_best"] >= 1.0, key
        assert con["ratio_median"] <= ref["ratio_median"] and c["record"] in ("constructive", "reference")
    assert json.dumps(t, indent=1) + "\n" == open(bestknown.path("Scanner/RFPulse-v1"), encoding="utf-8").read()
    assert cli.main(["records", "Scanner/RFPulse-v1"]) == 0
    out = capsys.readouterr().out
    assert "level 3 | thickness 6 mm" in out and "quarantined (a pass below the bound: a defect, not a record): 0" in out
    assert cli.main(["records", "Scanner/NullTissue-v0"]) == 2
