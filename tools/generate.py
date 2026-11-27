"""Regenerate every generated file of the repository from the code, or check that none has drifted.

    python tools/generate.py            write
    python tools/generate.py --check    write nothing; exit 1 if a generated file or block differs
    python tools/generate.py --timing   also re-measure the scan timing table in the README (machine dependent)

Generated: envs/ (index.json, README.md with the full environment table, one card per environment id), docs/scanner.md
(the datasheet shipped in the package), docs/physics/validation.md (from ``mrgym validate``), the marked blocks listed
in ``BLOCKS`` (README.md: the task table; docs/TOUR.md: inventory counts, the demo transcript, the results line,
the timing table), and every ``<!-- STAT:key -->...<!-- /STAT -->`` span of README.md and docs/*.md, filled from
docs/results/results.json, and their POWER and GATE_OC blocks (the power table of the designs and the operating
characteristic of the gate). Numbers that depend on the machine (measured errors at rounding level, timings) are
written but not compared by ``--check``.

Both modes also apply the reporting rules of ``mrgym.stats.lint`` to the hand-written text of README.md,
docs/results/README.md, docs/*.md and envs/*/README.md, and exit 1 when one is broken: a rate typed by hand, a
comparison of two agents without its planned test, "always" about a rate without a bound.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.resources
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = "docs/results"                                   # the stored results: an archive of pilot rows (its README says what is kept)
# page -> the generated blocks it must hold (TIMING is rewritten only with --timing)
BLOCKS = {"README.md": ("TASKS",), "docs/TOUR.md": ("COUNTS", "DEMO", "RESULTS", "TIMING")}
# The front page says in plain words what is done in each task. A family that is not listed here is shown with the title
# of its card, so a new family needs no edit of this file.
WHAT_YOU_DO = {
    "Scanner/NoWrap-v1": "find the smallest, fastest field of view around the object",
    "Scanner/NullTissue-v0": "find the inversion time that makes each vial dark",
    "Scanner/FatSuppress-v0": "choose the fat suppression that works in each region",
    "Scanner/BandFree-v0": "keep the dark bands of balanced SSFP off the object",
    "Scanner/SNRSpec-v0": "reach a required signal-to-noise ratio in the least scan time",
    "Scanner/PhantomQA-v0": "run the phantom quality tests and name what is broken",
    "Scanner/ArtefactClinic-v0": "say what causes each artefact and fix the protocol",
    "Scanner/ProtocolToSpec-v0": "turn written requirements into a protocol that meets them",
    "Scanner/T1Transmit-v1": "measure T1 when the transmit field is miscalibrated",
    "Scanner/Explore-v0": "scan freely; nothing is asked and nothing is scored",
}
# The front page uses plain words for the kind of the registry; envs/README.md and docs/ADDING_A_TASK.md define the kinds.
STATUS = dict(admitted="ready", candidate="experimental", practice="experimental", sandbox="no score")


def _read(rel):
    try:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _block(text, name, body):
    """Replace the block between <!-- NAME:START --> and <!-- NAME:END --> (markers kept)."""
    pat = re.compile(rf"(<!-- {name}:START -->\n).*?(<!-- {name}:END -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"the page has no {name} block")
    return pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text)


def _get_block(text, name):
    m = re.search(rf"<!-- {name}:START -->\n(.*?)<!-- {name}:END -->", text, re.S)
    return m.group(1) if m else None


def _capture(argv):
    from mrgym import cli
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(argv)
    return buf.getvalue()


# ------------------------------------------------------------------ pieces
def env_cards():
    """envs/: one card per registered id, the index (served and retired ids) and the table of the README. Kind and
    status come from the registry and from the stored gate result; nothing here is typed."""
    import mrgym
    from mrgym import cards, cli
    from mrgym.registry import retired
    out, index, rows = {}, [], []
    listed = {r["id"]: r for r in cli.list_rows()}
    for env_id in mrgym.list_envs():
        spec, card = mrgym.spec(env_id), mrgym.card(env_id)
        env = mrgym.make(env_id, seed=0)
        obs, _ = env.reset()
        tools = [dict(name=t.name, cost=t.cost, counted=t.counted, terminal=t.terminal) for t in env._tool_specs()]
        slug = cards.slug(env_id)
        out[f"envs/{slug}/README.md"] = cards.markdown(card, tools, obs.text)
        r, g = listed[env_id], card["gate"]
        index.append(dict(id=env_id, folder=slug, kind=spec.kind, levels=list(spec.levels), claim_levels=r["claim_levels"],
                          practice_levels=r["practice_levels"], title=card["title"], status=card["status"], task_kind=r["task_kind"],
                          gate=None if not g else dict(engine=g["engine"], instances_per_level=min(v["n"] for v in g["levels"].values()),
                                                       verdicts={lv: v["verdict"] for lv, v in g["levels"].items()})))
        head, _, rest = card["status"].partition(": ")
        gate = "no reward" if spec.kind == "sandbox" else rest if head.lower() == spec.kind else f"{head.lower()}: {rest}"
        levels = ", ".join(map(str, r["claim_levels"])) + (f" (practice: {', '.join(map(str, r['practice_levels']))})" if r["practice_levels"] else "")
        rows.append(f"| [`{env_id}`](envs/{slug}/) | {card['given']} | {card['deliver']} | {card['scored_by']} | {spec.kind} | "
                    f"{levels if spec.family else '-'} | {gate} |")
    table = "\n".join(["| Environment | Given | Deliver | Scored by | Kind | Levels of the claim | Stored gate |", "|---|---|---|---|---|---|---|"] + rows)
    gone = [dict(id=i, last_version=v, successor=s) for i, (v, s) in sorted(retired.items())]
    out["envs/index.json"] = json.dumps(dict(environments=index, retired=gone), indent=1) + "\n"
    retired_md = "" if not gone else "\n## Retired ids\n\n" + "\n".join(
        f"- `{g['id']}`: not served by this package. `pip install mrgym=={g['last_version']}` reproduces it."
        + (f" Successor: `{g['successor']}`." if g["successor"] else "") for g in gone) + "\n"
    out["envs/README.md"] = ("<!-- Generated by tools/generate.py from the registry. -->\n# Environments\n\n"
                             + table.replace("](envs/", "](") + "\n\nAn **environment** is one job on the scanner with its own scoring "
                             "rule. A **level** sets the hidden-world prior and the budget. An **episode** is one seeded attempt. "
                             "**Kind** is the claim of the registry: `sandbox` (no reward), `candidate` (scored; its gate is not stored on "
                             "this engine yet), `admitted` (the stored gate admits every level of the claim) or `practice` (scored; "
                             "not part of a claim). The gate gives one verdict per level, never one per instance; it is described in "
                             "[docs/DESIGN.md](../docs/DESIGN.md), `mrgym gate <id> --n 300 --write` measures and stores it, and "
                             "`mrgym check <id>` compares the kind with what is stored.\n" + retired_md)
    return out, table


def tasks_table():
    """The task table of the front page: one row per registered id, linked to its card, with what is done in the task
    in at most ten plain words (``WHAT_YOU_DO``, else the title of the card) and a status from the kind of the registry
    (``STATUS``). Ready tasks first."""
    import mrgym
    from mrgym import cards
    order = list(STATUS)
    ids = sorted(mrgym.list_envs(), key=lambda i: (order.index(mrgym.spec(i).kind), list(WHAT_YOU_DO).index(i) if i in WHAT_YOU_DO else len(WHAT_YOU_DO), i))
    rows = []
    for env_id in ids:
        text = WHAT_YOU_DO.get(env_id) or mrgym.card(env_id)["title"]
        if len(text.split()) > 10:
            raise SystemExit(f"the front-page line of {env_id} has more than ten words: {text!r}")
        rows.append(f"| [`{env_id}`](envs/{cards.slug(env_id)}/) | {text} | {STATUS[mrgym.spec(env_id).kind]} |")
    return "\n".join(["| Task | What you do | Status |", "|---|---|---|"] + rows)


def datasheet():
    text = importlib.resources.files("mrgym.scanner").joinpath("datasheet.md").read_text(encoding="utf-8")
    return "<!-- Generated by tools/generate.py: a copy of src/mrgym/scanner/datasheet.md, the text the agent is given. -->\n" + text


def validation():
    import mrgym
    from mrgym import validate
    rows, _ = validate.run()
    groups = getattr(validate, "GROUPS", dict(signal="Signal models", encoding="Image formation in k-space", console="Console arithmetic"))
    out = ["<!-- Generated by tools/generate.py from `mrgym validate`. -->", "# Physics validation", "",
           f"`mrgym validate` runs {len(rows)} checks of engine `{mrgym.ENGINE_ID}`; all must pass, and the test suite asserts it. "
           "Each row compares the scanner with an analytic formula or with a calculation written separately from the engine.", "",
           "What this establishes: the scanner computes the equations it states, to the bounds listed. What it does not: "
           "agreement with a measurement or with a real scanner. Rows whose reference was written by the same authors say so. "
           "Comparisons with another group's code and with published operating points: [external.md](external.md). "
           "Whether a simplification can change a scored answer is measured per environment by `mrgym validate --perturb <id>` "
           "and printed in the Validity section of the environment's card. "
           "References: [references.md](references.md). Simplifications: [limits.md](limits.md).", ""]
    for g, title in groups.items():
        out += [f"## {title}", "", validate.markdown_table([r for r in rows if r["group"] == g]), ""]
    out += ["The measured column is from the machine that generated this page; values at rounding level differ between "
            "machines, the bounds do not.", ""]
    ext = _read("docs/physics/external_sycomore.json")
    out += ["## Another group's simulator (optional, not in the test suite)", ""]
    if ext:
        e = json.loads(ext)
        out += [f"The RF-spoiled steady state of the engine (spgr and afi) against [sycomore](https://github.com/lamyj/sycomore), an extended "
                f"phase graph simulator in C++ that is not a dependency of this package: {e['cases']} cases, largest difference "
                f"{e['max_abs_diff_no_diffusion']:.1e} of M0 without diffusion and {e['max_abs_diff_with_diffusion']:.1e} with "
                f"diffusion (bounds {e['bounds'][0]:g} and {e['bounds'][1]:g}), on engine `{e['engine']}`, {e['date_utc']}. With "
                "diffusion both codes use the same b-factor formula, so this checks the implementation, not the diffusion model.",
                "", f"Regenerate: `{e['command']}`."]
    else:
        out += ["Not run. `pip install sycomore && python tools/external/compare_sycomore.py --write` compares the RF-spoiled "
                "steady state with sycomore and stores the result."]
    return "\n".join(out) + "\n", rows


def counts(rows):
    import mrgym
    from mrgym.scanner import console, tissue
    from mrgym.scanner import world as W
    kinds = {mrgym.spec(i).kind for i in mrgym.list_envs()}
    n_env = len(mrgym.list_envs())
    return (f"| {len(console.SEQUENCES)} sequences | {len(W.KINDS)} phantoms | 2 field strengths | {len(tissue.NAMES)} tissue rows, each with a source tag "
            f"| {len(rows)} physics checks | {n_env} environments ({', '.join(sorted(kinds))}) |\n|---|---|---|---|---|---|")


def demo():
    text = _capture(["demo"])
    return "```text\n$ mrgym demo\n" + text.rstrip() + "\n```"


def results_block():
    """A compact view of the stored results.json for docs/TOUR.md: the published rows only, one table per
    environment, each cell as ``k/N (rate; 95 % interval)`` with the instance count in the header. Pilot rows are not
    shown here: they stand under their own heading in the README of the results folder. Links are relative to docs/."""
    from mrgym.agents import summarize
    from mrgym.registry import retired
    raw = _read(f"{RESULTS}/results.json")
    page = RESULTS.split("/", 1)[1] + "/README.md"
    if not raw:
        return "No results have been generated yet: `mrgym run-agent` writes episode records, `mrgym results` builds the table."
    r = json.loads(raw)
    rows = [x for x in r["rows"] if x["tier"] == "published"]
    played = r["played_utc"]
    played = "-" if not played else played[0] if played[0] == played[-1] else f"{played[0]} to {played[-1]}"
    if not rows:
        d = r["design"]
        old = " on retired ids" if r["environments"] and all(i in retired for i in r["environments"]) else ""
        kept = "" if os.path.isdir(os.path.join(ROOT, RESULTS, "episodes")) else "; the records themselves are not stored"
        return (f"No row is published yet. A published row needs at least {d['published_min_instances']} instances per environment, "
                f"{d['published_min_episodes']} episodes per instance for a model, and records that store the whole conversation. "
                f"The rows of {r['pilot_records']} pilot episode records (played {played} UTC{old}{kept}) are listed with their "
                f"intervals in [{page}]({page}); no comparison is computed from them.")
    out = [f"{r['published_records']} published episode records, played {played} (UTC) on engine `{', '.join(r['engines'])}`, package "
           f"{', '.join(r['package_versions'])}. Cells are `passed/episodes (rate; 95 % interval over instances)`. The full table, the "
           f"planned comparisons, reliability and cost are in [{page}]({page}).", ""]
    for env_id in sorted({x["env_id"] for x in rows}):
        part = [x for x in rows if x["env_id"] == env_id]
        out += [f"**`{env_id}`** ({max(x['instances'] for x in part)} instances)", "", "| Agent | Kind | Passed | Regret: median / 95th pct |", "|---|---|---:|---:|"]
        out += [f"| {x['agent']} | {x['kind']} | {summarize.cell_text(x)} | {summarize._regret(x)} |" for x in part]
        out.append("")
    return "\n".join(out).rstrip()


LINTED = ("README.md", f"{RESULTS}/README.md", "docs/*.md", "envs/*/README.md")


def lint_docs(files):
    """The reporting rules on the hand-written text of the documentation (``mrgym.stats.lint``): generated blocks and
    STAT spans are trusted, everything else is scanned. ``files`` holds the fresh content of the generated files.
    Returns ``file:line: rule: message`` lines."""
    import glob

    from mrgym.stats import lint
    raw = _read(f"{RESULTS}/results.json")
    summary = json.loads(raw) if raw else None
    out = []
    for pattern in LINTED:
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
            text = files.get(rel) or _read(rel) or ""
            out += [lint.format_violation(rel, v) for v in lint.lint_text(text, summary)]
    return out


def stats_blocks(text):
    """Fill the POWER and GATE_OC blocks of a page: the power table of the designs and the operating characteristic
    of the admission gate, computed by ``mrgym.stats.power`` (closed forms and exact binomial; nothing is typed)."""
    from mrgym.stats import power
    for name, body in (("POWER", power.power_markdown()), ("GATE_OC", "```text\n" + power.gate_text() + "\n```")):
        if f"<!-- {name}:START -->" in text:
            text = _block(text, name, body)
    return text


def fill_stats(rel, text):
    """``text`` with every ``<!-- STAT:key -->...<!-- /STAT -->`` span filled from the stored results.json."""
    from mrgym import results
    raw = _read(f"{RESULTS}/results.json")
    if "<!-- STAT:" not in text:
        return text
    if not raw:
        raise SystemExit(f"{rel} has STAT spans but {RESULTS}/results.json does not exist")
    try:
        return results.fill_stat_spans(text, json.loads(raw))
    except KeyError as e:
        raise SystemExit(f"{rel}: {e.args[0]}") from None


def timing_block():
    return _capture(["timing", "--markdown"]).rstrip()


def _strip_measured(md):
    """Validation table without the measured column (rounding-level values differ between machines)."""
    out = []
    for line in md.splitlines():
        cells = line.split(" | ")
        out.append(" | ".join(cells[:3] + cells[4:]) if line.startswith("| ") and len(cells) >= 6 else line)
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args(argv)
    files, _ = env_cards()
    files["docs/scanner.md"] = datasheet()
    val_md, rows = validation()
    files["docs/physics/validation.md"] = val_md
    bodies = dict(TASKS=tasks_table, COUNTS=lambda: counts(rows), DEMO=demo, RESULTS=results_block)
    if a.timing:
        bodies["TIMING"] = timing_block
    for rel, names in BLOCKS.items():
        text = _read(rel)
        if text is None:
            raise SystemExit(f"{rel} does not exist; it holds the generated blocks {', '.join(names)}")
        for name in names:
            if f"<!-- {name}:START -->" not in text:
                raise SystemExit(f"{rel} has no {name} block")
            if name in bodies:
                text = _block(text, name, bodies[name]())
        files[rel] = fill_stats(rel, stats_blocks(text))
    for rel in sorted(os.path.join("docs", n).replace(os.sep, "/") for n in os.listdir(os.path.join(ROOT, "docs")) if n.endswith(".md")):
        text = _read(rel)
        if text is not None and rel not in files and ("<!-- STAT:" in text or "<!-- POWER:START -->" in text or "<!-- GATE_OC:START -->" in text):
            files[rel] = fill_stats(rel, stats_blocks(text))          # hand-written pages: only their marked parts are generated
    broken = lint_docs(files)
    drift = []
    for rel, text in files.items():
        old = _read(rel)
        if rel == "docs/physics/validation.md":
            same = old is not None and _strip_measured(old) == _strip_measured(text) and all(r["ok"] for r in rows)
        else:
            same = old == text
        if not same:
            drift.append(rel)
            if not a.check:
                path = os.path.join(ROOT, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
    for line in broken:
        print(line)
    if a.check:
        if drift:
            print("generated files differ from the code; run `python tools/generate.py`:\n  " + "\n  ".join(drift))
        if drift or broken:
            return 1
        print(f"{len(files)} generated files are up to date; the hand-written text breaks no reporting rule")
        return 0
    print(f"{len(files)} generated files, {len(drift)} written" + (": " + ", ".join(drift) if drift else ""))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
