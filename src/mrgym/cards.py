"""Environment cards: one dict per environment id, rendered to Markdown, linted for completeness.

A card answers, in this order, what a user of an environment asks: what is on the table, what is hidden, what the
agent can do and what it costs, what it must deliver, how the reward is computed, which policies were tried and what
they scored. The authored part lives next to the environment (``card()``). The claim (``kind``) comes from the
registry, and the evidence from the stored gate result (``mrgym gate <id> --n 300 --write``, schema ``mrgym-gate/2``):
status line and gate table are read from that file and never typed. A card whose gate is not stored on the current
engine says so.
"""
from __future__ import annotations

KINDS = ("sandbox", "practice", "admitted", "candidate")
REQUIRED = ("id", "title", "kind", "status", "given", "deliver", "scored_by", "summary", "world", "hidden", "told", "levels",
            "limits", "deliverable", "reward", "controls", "gate", "determinism")
MUSTS = ("pass", "fail", "ablation", "reported")
GATE_NOT_STORED = "gate not stored on this engine"


def slug(env_id: str) -> str:
    """Folder name of an environment card: ``Scanner/NoWrap-v1`` -> ``scanner-nowrap-v1``."""
    return env_id.replace("/", "-").lower()


def claim_levels(card: dict) -> list:
    """The levels the claim of a card covers: its levels without the practice levels the family marks."""
    practice = set(card.get("practice_levels") or ())
    return [lv for lv in sorted(card["levels"]) if lv not in practice]


def gate_rows(level: dict) -> dict:
    """{policy name: row} of one level of a stored gate result. The recipes of ``fixed_recipe`` are tuned on several
    sets of other worlds and stored as ``fixed_recipe#k``: the row of the name is the one that passed most often."""
    out = {"reference": dict(level["reference"], must="pass")} if "reference" in level else {}
    for r in level.get("must_fail", ()):
        name = r["policy"].split("#")[0]
        if name not in out or r["passed"] > out[name]["passed"]:
            out[name] = dict(r, must="fail", tuned_sets=sum(x["policy"].split("#")[0] == name for x in level["must_fail"]))
    out.update({r["policy"]: dict(r, must="ablation") for r in level.get("ablations", ())})
    out.update({r["policy"]: dict(r, must="reported") for r in level.get("reported", ())})
    return out


def gate_problems(card: dict, engine_id: str | None = None) -> list[str]:
    """Why the stored gate of a card is not evidence for it on ``engine_id`` (default: the current engine); empty when
    it is, or when nothing is stored. Does not look at the verdicts (``mrgym.check`` compares them with the claim)."""
    import mrgym
    g = card.get("gate")
    if not g:
        return []
    engine_id = engine_id or mrgym.ENGINE_ID
    bad = []
    if g.get("schema") != "mrgym-gate/2":
        bad.append(f"the stored gate has schema {g.get('schema')!r}, not 'mrgym-gate/2'")
    if g.get("engine") != engine_id:
        bad.append(f"the stored gate was measured on engine {g.get('engine')}, not on {engine_id}")
    missing = [lv for lv in claim_levels(card) if str(lv) not in g.get("levels", {})]
    if missing:
        bad.append(f"the stored gate has no result for level(s) {', '.join(map(str, missing))}")
    for lv, level in g.get("levels", {}).items():
        rows = gate_rows(level)
        ref = next((c["name"] for c in card["controls"] if c["must"] == "pass"), None)
        for c in card["controls"]:
            if ("reference" if c["name"] == ref else c["name"]) not in rows:
                bad.append(f"control {c['name']} has no stored result at level {lv}")
    return bad


def lint(card: dict) -> list[str]:
    """Problems that keep a card from being published."""
    bad = [f"missing field: {k}" for k in REQUIRED if k not in card]
    if bad:
        return bad
    if card["kind"] not in KINDS:
        return [f"kind must be one of {KINDS}, not {card['kind']!r}"]
    if card["kind"] == "sandbox":
        return bad
    bad += [f"control {c.get('name')}: must is one of {MUSTS}" for c in card["controls"] if c.get("must") not in MUSTS]
    if sum(c["must"] == "pass" for c in card["controls"]) != 1:
        bad.append("exactly one reference policy (a control that must pass) is needed")
    if sum(c["must"] == "fail" for c in card["controls"]) < 3:
        bad.append("fewer than three shortcut controls that must fail")
    if not claim_levels(card):
        bad.append("every level is a practice level: the card claims nothing")
    if card["kind"] == "admitted" and not card["gate"]:
        bad.append("listed as admitted without a stored gate result (run `mrgym gate <id> --n 300 --write`)")
    if card["gate"] and card["gate"].get("schema") == "mrgym-gate/2":
        bad += [p for p in gate_problems(card, card["gate"].get("engine")) if p.startswith("control ")]
    return bad


def _rate(r: dict, n: int) -> str:
    return f"{r['passed']}/{n} ({100 * r['rate']:.1f} %"


def gate_cell(row: dict | None, n: int) -> str:
    """One policy at one level: passes over instances, the rate and the one-sided 95 % bound the gate decides on."""
    if row is None:
        return "-"
    if row["must"] == "pass":
        return _rate(row, n) + f"; lower bound {100 * row['lower95']:.1f} %)"
    if row["must"] == "fail":
        return _rate(row, n) + f"; upper bound {100 * row['upper95']:.1f} %)" + ("" if row["cleared"] else " NOT CLEARED")
    if row["must"] == "ablation":
        gap = f"; {100 * row['gap']:.1f} points below the reference, lower bound {100 * row['gap_lower95']:.1f})"
        return _rate(row, n) + gap + ("" if row["required"] else " not required here") + ("" if row["cleared"] else " NOT CLEARED")
    return _rate(row, n) + ")"


def gate_table(card: dict) -> list[str]:
    """The gate section of a card: the stored table, or the sentence that none is stored on this engine."""
    import mrgym
    g = card["gate"]
    how = f"`mrgym gate {card['id']} --n 300 --write` measures and stores it."
    if not g:
        return [f"**{GATE_NOT_STORED.capitalize()}** (`{mrgym.ENGINE_ID}`). {how}"]
    problems = [p for p in gate_problems(card) if not p.startswith("control ")]
    lvs = sorted(g["levels"], key=int)
    out = []
    if problems:
        out += [f"**{GATE_NOT_STORED.capitalize()}** (`{mrgym.ENGINE_ID}`): {'; '.join(problems)}. {how} The table below is that older result.", ""]
    t = g["thresholds"]
    out += [f"Run {g['date_utc']} on engine `{g['engine']}`, package {g['package_version']}: `{g['command']}` ({g['seeds']['gate']}; "
            f"recipes tuned on {g['seeds']['tune']}). A level is admitted when the lower bound of the reference is at least "
            f"{100 * t['reference_lower_min']:g} %, the upper bound of every must-fail policy is at most {100 * t['blind_upper_max']:g} % and every "
            f"ablation required at the level loses at least {100 * t['ablation_gap_lower_min']:g} points (lower bound of the paired difference). "
            f"Bounds are one-sided, {100 * (1 - t['alpha']):g} %. Cells: passed / instances (rate; bound).", "",
            "| Policy | Must | " + " | ".join(f"Level {lv}" for lv in lvs) + " | What it does |", "|---|---|" + "---|" * len(lvs) + "---|"]
    rows = {lv: gate_rows(g["levels"][lv]) for lv in lvs}
    ref = next((c["name"] for c in card["controls"] if c["must"] == "pass"), "reference")
    for c in card["controls"]:
        cells = [gate_cell(rows[lv].get("reference" if c["name"] == ref else c["name"]), g["levels"][lv]["n"]) for lv in lvs]
        out.append(f"| `{c['name']}` | {c['must']} | " + " | ".join(cells) + f" | {c['what']} |")
    out += ["", "Verdict per level: " + "; ".join(
        f"level {lv} **{g['levels'][lv]['verdict']}**" + ("" if not g["levels"][lv]["reasons"] else " (" + "; ".join(g["levels"][lv]["reasons"]) + ")")
        for lv in lvs) + "."]
    ceiling = [f"level {lv} `{g['levels'][lv]['blind_ceiling']['policy']}` {100 * g['levels'][lv]['blind_ceiling']['rate']:.1f} % "
               f"(upper bound {100 * g['levels'][lv]['blind_ceiling']['upper95']:.1f} %)" for lv in lvs if g["levels"][lv].get("blind_ceiling")]
    if ceiling:
        out.append("Blind ceiling (the must-fail policy that passes most often): " + "; ".join(ceiling) + ".")
    ref_rows = [(lv, g["levels"][lv]["reference"]) for lv in lvs]
    if all(r.get("error_median") is not None for _, r in ref_rows):
        out.append("Reference, error of its answer (median / 95th percentile): "
                   + "; ".join(f"level {lv} {r['error_median']:g} / {r['error_q95']:g}" for lv, r in ref_rows)
                   + "; scan time used (median): " + ", ".join(f"level {lv} {r['scan_s_median']:g} s" for lv, r in ref_rows) + ".")
    return out


def markdown(card: dict, tools: list | None = None, statement: str | None = None) -> str:
    w, r = card["world"], card["reward"]
    lv0 = card.get("default_level", sorted(card["levels"])[0])
    practice = set(card.get("practice_levels") or ())
    from . import physics_map
    place = physics_map.card_line(card["id"])                           # where the task stands on the map (docs/tasks.md); generated
    out = ["<!-- Generated by tools/generate.py from the registry; edit the card() next to the environment. -->",
           f"# `{card['id']}` · {card['title']}", ""] + ([place, ""] if place else []) + [f"**Status:** {card['status']}", "", card["summary"], ""]
    if card.get("task_kind"):
        out += [f"Kind of task: **{card['task_kind']}**" + (f" ({card['task_kind_why']})" if card.get("task_kind_why") else "") + ".", ""]
    if card.get("why_practice"):
        out += [card["why_practice"], ""]
    out += ["```python", "import mrgym", f"env = mrgym.make(\"{card['id']}\", level={lv0}, seed=0)", "obs, info = env.reset()",
            "obs, reward, terminated, truncated, info = env.step({\"name\": \"scanner_info\", \"arguments\": {}})", "```", "",
            "| Given | Deliver | Scored by |", "|---|---|---|", f"| {card['given']} | {card['deliver']} | {card['scored_by']} |", "",
            "## World", "", f"- On the table: {w['phantom']}", f"- Sequences that matter here: {', '.join(w['sequences_useful'])}",
            f"- Formulas the agent is given: {w['agent_side_formulas']}", "",
            "| Hidden (measurable only by scanning) | Told |", "|---|---|"]
    for i in range(max(len(card["hidden"]), len(card["told"]))):
        out.append(f"| {card['hidden'][i] if i < len(card['hidden']) else ''} | {card['told'][i] if i < len(card['told']) else ''} |")
    if tools:
        out += ["", "## Tools", "", "| Tool | Cost | Counts as scanner request | Ends episode |", "|---|---|---|---|"]
        out += [f"| `{t['name']}` | {t['cost'].replace('_', ' ')} | {'yes' if t['counted'] else 'no'} | {'yes' if t['terminal'] else 'no'} |" for t in tools]
        out += ["", f"Limits: {card['limits']['max_scanner_requests']} scanner requests and {card['limits']['max_tool_calls']} tool calls per episode."]
    out += ["", "## Levels", "", "| Level | Covered by the claim | Hidden-world prior | Budget (s) |", "|---|---|---|---|"]
    out += [f"| {lv} | {'no: practice level' if lv in practice else 'yes'} | {d['summary']} | {d['budget_s'][0]:g}"
            + (f" to {d['budget_s'][1]:g}" if d["budget_s"][1] != d["budget_s"][0] else "") + " |" for lv, d in sorted(card["levels"].items())]
    out += ["", "## Deliverable and reward", "", f"- Deliverable: `{card['deliverable']['tool']}` with `{card['deliverable']['format']}`",
            f"- Reward range {r['range']}; paid {r['when']}"]
    if r["gates"]:
        out += ["- Gates: " + "; ".join(r["gates"]), f"- Error: `{r['error']}`", f"- Score: `{r['score']}`", f"- Passing: {r['passing']}"]
    if card["kind"] != "sandbox":
        out += ["", "## Gate: the control policies on fresh instances", ""] + gate_table(card)
        if not card["gate"]:
            out += ["", "| Policy | Must | What it does |", "|---|---|---|"] + [f"| `{c['name']}` | {c['must']} | {c['what']} |" for c in card["controls"]]
    if card["kind"] != "sandbox" and "validity" in card:
        from . import perturb
        out += [""] + perturb.markdown(card["validity"])
    if statement:
        out += ["", f"## What the agent reads at reset (level {lv0}, seed 0)", "", "~~~text", statement.rstrip(), "~~~"]
    out += ["", "## Determinism and versions", "", card["determinism"] + ".", "",
            "Rewards are comparable only within one environment version (the `-vN` suffix) and one scanner engine "
            "(`mrgym --version`)."]
    return "\n".join(out) + "\n"
