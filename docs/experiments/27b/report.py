"""The report of the first training runs of Qwen3.8-27B on MR Gym, written from the stored result files.

    python docs/experiments/27b/report.py            # writes docs/experiments/RESULTS-27b.md, the figure, the blocks
    python docs/experiments/27b/report.py --check    # exit 1 when the report, the figure's data or a block is stale
    python docs/experiments/27b/report.py --figure   # redraw assets/training_27b.png and its data file only

Every number in the report is computed here from ``docs/experiments/27b/results/``: the episode logs of the runs
(``episodes_<run>.jsonl.gz``, one line per episode as the trainer wrote it), the evaluation logs on the secret split
(``test_base_<run>.jsonl.gz`` and ``test_trained_<run>.jsonl.gz``; ``READOUT_FILES`` names the ones of the first run,
``LINE`` the readings of the first run's adapter and of its continuation against the same base log),
the calibration summaries (``calibration_<run>.json``) and ``runs.json`` (what each run was). A run has a readout when
its two evaluation logs are stored; what it trained on, for how many steps and which environments it never trained on
are read from its logs. The paired readout and the evaluation table are the functions the node scripts use
(``mrgym_verl.readout``, ``mrgym_verl.stoprules``): the newest attempt of each log, the instances both played. The
text between the tables is written here; it states settings and no result of its own.

The same numbers go to the front page. ``assets/training_27b.png`` is drawn from ``figure_data()``, which is stored
beside it as ``assets/training_27b.json``: ``--check`` compares that file with the stored logs, not the image bytes
(``FIGURE_VERSION`` is in it, so a change of the drawing makes the stored figure stale too). The blocks
``<!-- NAME:START -->`` ... ``<!-- NAME:END -->`` of README.md and docs/RECIPE.md listed in ``PAGES`` are written here.
The front page holds the figure and two plain sentences with two links (the trained level and the family that was
never trained, before and after, of the headline run ``MAIN_RUN``): ``front_block``. The recipe page holds the before / after
table of that run, the settings of ``recipes/qwen3.8-27b-nowrap.env``, what the run cost (from the ``utc`` stamps of
its logs), the pitfalls with their numbers, and which commit the recipe was run on (from ``runs.json``).
"""
from __future__ import annotations

import collections
import datetime
import gzip
import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "integrations", "verl"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mrgym_verl import readout as R  # noqa: E402
from mrgym_verl import stoprules as S  # noqa: E402

import mrgym  # noqa: E402
from mrgym import learn  # noqa: E402

RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(ROOT, "docs", "experiments", "RESULTS-27b.md")
GAIN, LOWER, HARM, MIN_ENVS = 0.15, 0.05, 0.05, 2          # the rule of PREREG-27b.md section 1
FIGURE = os.path.join(ROOT, "assets", "training_27b.png")
SIDECAR = os.path.join(ROOT, "assets", "training_27b.json")
RECIPE_ENV = os.path.join(ROOT, "recipes", "qwen3.8-27b-nowrap.env")
FIGURE_VERSION = 2                                         # raise it when draw() changes: the stored figure is then stale
MAIN_RUN = "v17"                                           # the headline: the recipe on the published code
# The engine every stored run was played on. The episode logs do not record it: it is the ENGINE_VERSION of
# src/mrgym/scanner/session.py at the commits that runs.json names for the runs (all of them the same;
# ``git show <commit>:src/mrgym/scanner/session.py``; tests/test_results_27b.py repeats that where the checkout has the
# history). Results on two engine versions are not comparable (CHANGELOG.md), so every page that shows these numbers
# says which engine they are from, and whether that is the engine of the package (``engine_sentence``).
MEASURED_ON = "mrgym-engine-0.2.0"
FIRST_RUN, MANY_CELLS_RUN, LONGER_RUN, STOPPED_RUN, RECOUNT_RUN = "v07", "v09", "v13", "v18", "v19"
READOUT_FILES = {"v07": ("test_base_57k.jsonl.gz", "test_trained_step60.jsonl.gz")}       # other runs: test_base_<run>, test_trained_<run>
READOUT_BY = {"v07": "v15"}                                # the evaluation-only run that read a run out, where it was not the run itself
# one adapter read several times against the base log of the first run: (the run that read it out, the run whose weights, the log)
LINE = (("v12", "v07", "test_trained_step40.jsonl.gz"), ("v15", "v07", "test_trained_step60.jsonl.gz"),
        ("v20", "v13", "test_trained_step80.jsonl.gz"), ("v13", "v13", "test_trained_step110.jsonl.gz"))
SELECTED = "v20"                                           # the reading whose checkpoint was chosen on the dev looks (runs.json says so)
# the stop rules of the runs that a rule stopped, as their job scripts set them (the other values are the defaults of stoprules.py)
STOP_CONFIG = {k: dict(episodes_per_step=64, group_size=8, base_rate=0.40, mixed_min=0.30, submit_min=0.40, cut_max=0.50, looks=6, futility_from_step=30)
               for k in ("v18", "v19")}
L1, L2, L3 = ("Scanner/NoWrap-v1", 1), ("Scanner/NoWrap-v1", 2), ("Scanner/NoWrap-v1", 3)
HINTED, UNHINTED, NEVER = ("Scanner/NullTissue-v0", 0), ("Scanner/NullTissue-v0", 1), ("Scanner/T1Transmit-v1", 1)
REJECTED = "no_submission_rejected"                        # the outcome of an episode ended by the rejected-call limit
# the cells of the headline readout in plain words: (short label, label in a table)
PLAIN = {L1: ("trained level", "Trained level"),
         L2: ("harder level", "Harder level (trained; reward only late)"),
         L3: ("never-trained level", "Never-trained level"),
         HINTED: ("another task family (never trained)", "Another task family (never trained)")}
# page -> (prefix of a link to the repository root, the blocks it must hold); README.md holds the short TRAINING block
FRONT = "README.md"
PAGES = {FRONT: ("", ("TRAINING",)),
         "docs/RECIPE.md": ("../", ("TRAINING", "RECIPE_NEEDS", "RECIPE_SETTINGS", "RECIPE_PITFALLS", "RECIPE_STATUS", "RECIPE_REACH"))}
# the sections of the report, in order; the text refers to them by these names so that a number never goes stale
SECTIONS = ("runs", "replications", "line", "behaviour", "second", "secret", "dev", "rollouts", "calibration", "reads", "deviations", "limits",
            "reproduce")
TITLES = dict(line="One adapter line on the same secret instances", second="A second family did not train")
SEC = {name: i + 1 for i, name in enumerate(SECTIONS)}


def rows(name):
    with gzip.open(os.path.join(RESULTS, name), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def cell(r):
    return f"{r['env_id'].split('/')[1]} L{r['level']}"


def name(key):
    return f"{key[0].split('/')[1]} L{key[1]}"


def frac(eps):
    return f"{sum(bool(r['passed']) for r in eps)} / {len(eps)}" if eps else "-"


def pts(x):
    return f"{100 * x:+.1f}"


def gain(p):
    return f"{pts(p['diff'])} [{pts(p['lo'])}, {pts(p['hi'])}]"


def table(head, lines):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]


def words(items):
    items = [str(x) for x in items]
    return "" if not items else items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def moved(a, b, verb=("fall", "rise", "stay"), fmt="{}"):
    x, y = fmt.format(a), fmt.format(b)
    return f"{verb[0]} from {x} to {y}" if b < a else f"{verb[1]} from {x} to {y}" if b > a else f"{verb[2]} at {x}"


def _of(eps, key):
    return [r for r in eps if (r["env_id"], r["level"]) == key]


def _cells(eps):
    return sorted({(r["env_id"], r["level"]) for r in eps})


def _test(eps):
    """The rows of a readout as ``mrgym_verl.readout`` takes them: the newest attempt of the log, the test phase."""
    return [r for r in S._newest_attempt(eps) if r.get("phase") == "test"]


def _train(log):
    return [r for r in S._newest_attempt(log) if r.get("phase") == "train"]


def _ended(eps, key=None):
    """(passes, ended by the rejected-call limit, cut by length, median context tokens, episodes) of a cell."""
    eps = eps if key is None else _of(eps, key)
    ctx = sorted(r["context_tokens"] for r in eps)
    return dict(passed=sum(bool(r["passed"]) for r in eps), rejected=sum(r["outcome"] == REJECTED for r in eps),
                cut=sum("length" in r["outcome"] for r in eps), context=ctx[len(ctx) // 2], episodes=len(eps))


def _response_tokens(run):
    """The response limit of a run, from its ``node`` line in runs.json (``... + 24,576 response tokens``)."""
    return int(re.search(r"\+ ([\d,]+)", run["node"]).group(1).replace(",", ""))


def _episode_tokens(run):
    """Prompt + response limit of a run, from its ``node`` line in runs.json."""
    a, b = re.search(r"([\d,]+)[^\d+]*\+ ([\d,]+)", run["node"]).groups()
    return int(a.replace(",", "")) + int(b.replace(",", ""))


def _node(run):
    return run["node"].split(";")[0]


def _attempts(log):
    """(attempts, node hours): per attempt from its start (the stamp in its id) to its last logged episode."""
    stamp = lambda s, fmt: datetime.datetime.strptime(s, fmt)                                       # noqa: E731
    last = {}
    for r in log:
        last[r["attempt"]] = max(last.get(r["attempt"], r["utc"]), r["utc"])
    seconds = sum((stamp(u, "%Y-%m-%dT%H:%M:%SZ") - stamp(a.split("-")[0], "%Y%m%dT%H%M%SZ")).total_seconds() for a, u in last.items())
    return len(last), seconds / 3600.0


# ------------------------------------------------------------------------------------------------ the stored files
def stored() -> dict:
    """Every stored result file, read once. ``logs``: the episode log of every run that has one. ``tests``: per run with
    a readout its two evaluation logs as stored (``raw``) and as the readout takes them (``base``, ``trained``)."""
    runs = load("runs.json")
    present = set(os.listdir(RESULTS))
    logs = {k: rows(f"episodes_{k}.jsonl.gz") for k in runs if f"episodes_{k}.jsonl.gz" in present}
    tests = {}
    for k in runs:
        b, t = READOUT_FILES.get(k, (f"test_base_{k}.jsonl.gz", f"test_trained_{k}.jsonl.gz"))
        if b in present and t in present:
            raw = (rows(b), rows(t))
            tests[k] = dict(files=(b, t), raw=raw, base=_test(raw[0]), trained=_test(raw[1]))
    line = []
    for reader, weights, log in LINE:
        raw = rows(log)
        line.append(dict(reader=reader, run=weights, file=log, raw=raw, rows=_test(raw), step=max(r["step"] for r in raw)))
    return dict(runs=runs, logs=logs, tests=tests, line=line, cals={k: load(f"calibration_{k}.json") for k in ("c01", "c04", "c05")})


def readout_of(d: dict, run: str) -> dict:
    """The readout of one run: what it trained on (from its log), the environments it never trained on, the paired
    result of ``mrgym_verl.readout`` with the plan's rule, and how its episodes ended per cell."""
    t = d["tests"][run]
    trained_cells = _cells(_train(d["logs"][run]))
    untrained = sorted({e for e, _ in _cells(t["base"] + t["trained"])} - {e for e, _ in trained_cells})
    r = R.readout(t["base"], t["trained"], untrained, "test", GAIN, LOWER, HARM, MIN_ENVS)
    cells = {(c["env_id"], c["level"]): c for c in r["cells"]}
    return dict(run=run, code=d["runs"][run]["code"], by=READOUT_BY.get(run, run), trained_cells=trained_cells, untrained=untrained,
                steps=max(x["step"] for x in t["trained"]), result=r, cells=cells,
                instances=sorted({c["paired"]["instances"] for c in r["cells"]}),
                episodes=len(t["base"]) // max(1, sum(c["paired"]["instances"] for c in r["cells"])),
                ended={k: (_ended(t["base"], k), _ended(t["trained"], k)) for k in cells})


def readouts(d: dict) -> list[dict]:
    """The readouts of every finished training run, the headline run first, the others in the order of runs.json."""
    order = [MAIN_RUN] + [k for k in d["runs"] if k in d["tests"] and k != MAIN_RUN]
    return [readout_of(d, k) for k in order]


def line_of(d: dict) -> dict:
    """The adapter of the first run and of its continuation, read against the same base log: per reading the paired
    readout with the plan's rule and how its episodes ended."""
    base = d["tests"][FIRST_RUN]["base"]
    trained = {k for p in d["line"] for k in _cells(_train(d["logs"][p["run"]]))}
    untrained = sorted({e for e, _ in _cells(base)} - {e for e, _ in trained})
    points = []
    for p in d["line"]:
        r = R.readout(base, p["rows"], untrained, "test", GAIN, LOWER, HARM, MIN_ENVS)
        cells = {(c["env_id"], c["level"]): c for c in r["cells"]}
        label = f"step {p['step']} (`{p['run']}`" + ("" if p["reader"] == p["run"] else f", read out by `{p['reader']}`") + ")"
        points.append(dict(p, label=label, result=r, cells=cells, trained_cells=_cells(_train(d["logs"][p["run"]])),
                           ended={k: (_ended(base, k), _ended(p["rows"], k)) for k in cells}))
    return dict(base=base, untrained=untrained, points=points)


def stop_of(d: dict, run: str) -> dict:
    """What the stop rules say on the stored log of a run, with the settings of its job script."""
    return S.decide(d["logs"][run], STOP_CONFIG[run])


def slug(title: str) -> str:
    """The anchor a Markdown heading gets."""
    return re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")


# ------------------------------------------------------------------------------------------------ sections
def runs_table(d):
    def log(k):
        if k not in d["logs"]:
            return "-"
        train = _train(d["logs"][k])
        return f"steps {min(r['step'] for r in train)}-{max(r['step'] for r in train)}, {_attempts(d['logs'][k])[0]} attempt{'s' if _attempts(d['logs'][k])[0] != 1 else ''}"
    return table(["run", "what", "cells", "node", "code", "state when the files were copied", "training steps in the stored log"],
                 [[f"`{k}`", r["what"], r["cells"], r["node"], r["code"], r["state"], log(k)] for k, r in d["runs"].items() if not k.startswith("_")])


def readout_section(r: dict, label: str):
    out = []
    for c in r["cells"]:
        b, t, p = c["base"], c["trained"], c["paired"]
        out.append([f"{c['env_id'].split('/')[1]} L{c['level']}",
                    f"{round(b['pass_rate'] * b['episodes'])} / {b['episodes']}", f"{round(t['pass_rate'] * t['episodes'])} / {t['episodes']}",
                    gain(p), p["instances"], f"{p['p']:.4f}" if p["p"] >= 0.0001 else "< 0.0001"])
    lines = table(["cell", "base", label, "paired difference in points [95 %]", "instances", "p"], out)
    env = []
    for e in r["environments"]:
        p = e["paired"]
        env.append([e["env_id"].split("/")[1], e["role"], gain(p), p["instances"], "yes" if e.get("meets_rule") else "no"])
    lines += [""] + table(["environment (levels as strata)", "role", "paired difference in points [95 %]", "instances",
                           f"trained: gain >= {100 * GAIN:.0f} points and lower bound > {100 * LOWER:.0f}; untrained: not below -{100 * HARM:.0f}"], env)
    v = r["verdict"]
    lines += ["", f"Rule of the pre-registration: {v['trained_environments_meeting_rule']} trained environment(s) meet it, "
                  f"{v['needed']} are needed; no harm on the environment(s) never trained on: {v['no_harm_on_untrained']}; "
                  f"\"it trained\" by that rule: **{v['it_trained']}**."]
    return lines


def outcomes_section(columns):
    """How the episodes ended, per cell: one column per (label, rows)."""
    cells = _cells(columns[0][1])
    names = [n for n, _ in collections.Counter(r["outcome"] for _, eps in columns for r in eps).most_common()]
    out = []
    for key in cells:
        counts = [collections.Counter(r["outcome"] for r in _of(eps, key)) for _, eps in columns]
        for n in names:
            if sum(c[n] for c in counts):
                out.append([name(key), f"`{n}`"] + [c[n] for c in counts])
    return table(["cell", "how the episode ended"] + [label for label, _ in columns], out)


def tokens_section(columns):
    out = []
    for key in _cells(columns[0][1]):
        line = [name(key)]
        for _, eps in columns:
            e = _ended(eps, key)
            line += [f"{e['context']:,}", e["cut"]]
        out.append(line)
    head = ["cell"]
    for label, _ in columns:
        head += [f"{label}: median context tokens", "cut by length"]
    return table(head, out)


def replications_section(rs: list[dict]) -> list[str]:
    """One row per finished training run, then every readout cell of every run side by side."""
    def trained_level(r):
        c = r["cells"][L1]
        return [f"{round(c['base']['pass_rate'] * c['base']['episodes'])} / {c['base']['episodes']}",
                f"{round(c['trained']['pass_rate'] * c['trained']['episodes'])} / {c['trained']['episodes']}", gain(c["paired"])]
    out = []
    for r in rs:
        v = r["result"]["verdict"]
        meets = [e["env_id"].split("/")[1] for e in r["result"]["environments"] if e["role"] == "trained" and e["meets_rule"]]
        out.append([f"`{r['run']}`" + ("" if r["by"] == r["run"] else f" (read out by `{r['by']}`)"), r["code"], ", ".join(name(k) for k in r["trained_cells"]),
                    r["steps"], f"{' or '.join(map(str, r['instances']))} x {r['episodes']}"] + trained_level(r)
                   + [f"{v['trained_environments_meeting_rule']} of {v['needed']} needed" + (f" ({', '.join(meets)})" if meets else "")])
    lines = table(["run", "code", "trained on", "steps", "secret instances x episodes per cell", f"{name(L1)}: base", "trained",
                   "paired difference in points [95 %]", "trained environments that meet the plan's rule"], out)
    keys = sorted({k for r in rs for k in r["cells"]})
    matrix = []
    for k in keys:
        line = [name(k)]
        for r in rs:
            c = r["cells"].get(k)
            if c is None:
                line.append("-")
                continue
            b, t = c["base"], c["trained"]
            line.append(f"{round(b['pass_rate'] * b['episodes'])} / {b['episodes']} -> {round(t['pass_rate'] * t['episodes'])} / {t['episodes']}: "
                        f"{gain(c['paired'])}" + (" **T**" if k in r["trained_cells"] else ""))
        matrix.append(line)
    lines += ["", "Every readout cell of every run: base -> trained passes / episodes, then the paired difference in points with its 95 % "
                  "interval. **T** marks a cell the run trained on; `-` a cell it did not read.", ""]
    lines += table(["cell"] + [f"`{r['run']}`" for r in rs], matrix)
    return lines


def findings(rs: list[dict]) -> list[str]:
    """What the replication tables show, in sentences whose numbers and directions are read from the readouts."""
    def per_run(key, pick=lambda r: True):
        return [f"{gain(r['cells'][key]['paired'])} (`{r['run']}`)" for r in rs if key in r["cells"] and pick(r)]
    l1 = [r["cells"][L1]["paired"] for r in rs]
    out = [f"- The trained level ({name(L1)}) gains on instances the model never saw in {len(l1)} of {len(rs)} runs: " + ", ".join(per_run(L1)) + ". "
           + ("Every interval lies above zero." if all(p["lo"] > 0 for p in l1) else f"{sum(p['lo'] > 0 for p in l1)} of the intervals lie above zero.")]
    hard_t, hard_u = per_run(L2, lambda r: L2 in r["trained_cells"]), per_run(L2, lambda r: L2 not in r["trained_cells"])
    out.append(f"- The harder level ({name(L2)}): " + ", ".join(hard_t) + " where it was a training cell"
               + (f"; {', '.join(hard_u)} where it was not" if hard_u else "") + f". Its reward appears late (section {SEC['rollouts']}).")
    away, on = per_run(HINTED, lambda r: HINTED not in r["trained_cells"]), per_run(HINTED, lambda r: HINTED in r["trained_cells"])
    out.append(f"- {name(HINTED)}, the practice level of another task family (its text states the method): " + ", ".join(away)
               + " in the runs that never trained on it; " + ", ".join(on) + f" in the runs that did (section {SEC['behaviour']}). "
               "What carried over in the first case is executing a stated method through the tools. No conversation of these episodes has "
               "been read, so the mechanism is not established.")
    edge = per_run(L3)
    out.append(f"- {name(L3)}, never trained on: " + ", ".join(edge) + ". It is at the edge of its intervals and is not claimed.")
    met = [r["result"]["verdict"]["trained_environments_meeting_rule"] for r in rs]
    needed = rs[0]["result"]["verdict"]["needed"]
    out.append(f"- By the rule of the pre-registration ({needed} trained environments with a gain of at least {100 * GAIN:.0f} points and a lower bound "
               f"above {100 * LOWER:.0f}) " + (f"each of the {len(rs)} runs trained {met[0]} environment that meets it"
                                             if len(set(met)) == 1 else "the runs trained " + words(met) + " environments that meet it")
               + f": \"improved on one environment\" {len(rs)} times, not the plan's \"it trained\".")
    return out


def behaviour_section(d: dict, rs: list[dict]) -> list[str]:
    """Per run and cell of the secret split, base -> trained: passes, episodes ended by the rejected-call limit, cut by
    length, median context tokens; then what the table holds, in sentences read from it."""
    out = []
    for r in rs:
        for k, (b, t) in sorted(r["ended"].items()):
            out.append([f"`{r['run']}`", name(k), "yes" if k in r["trained_cells"] else "no", f"{b['passed']} -> {t['passed']}",
                        f"{b['rejected']} -> {t['rejected']}", f"{b['cut']} -> {t['cut']}", f"{b['context']:,} -> {t['context']:,}", b["episodes"]])
    line = line_of(d)
    cont = [p for p in line["points"] if p["run"] != FIRST_RUN]
    for p in cont:
        for k, (b, t) in sorted(p["ended"].items()):
            out.append([f"`{p['run']}` at step {p['step']}" + ("" if p["reader"] == p["run"] else f" (read out by `{p['reader']}`)"), name(k), "yes" if k in p["trained_cells"] else "no", f"{b['passed']} -> {t['passed']}",
                        f"{b['rejected']} -> {t['rejected']}", f"{b['cut']} -> {t['cut']}", f"{b['context']:,} -> {t['context']:,}", b["episodes"]])
    lines = table(["run", "cell", "trained on it", "passes", "ended by the rejected-call limit", "cut by length", "median context tokens",
                   "episodes per side"], out) + [""]
    by = {r["run"]: r for r in rs}
    clean = [r for r in rs if HINTED in r["cells"] and HINTED not in r["trained_cells"] and all(e == L1[0] for e, _ in r["trained_cells"])]
    if clean:
        lines.append(f"- Runs that trained on {L1[0].split('/')[1]} only: on {name(HINTED)}, which they never trained on, the endings at the "
                     "rejected-call limit " + words(f"{moved(*[x['rejected'] for x in r['ended'][HINTED]])} of {r['ended'][HINTED][0]['episodes']} "
                                                    f"(`{r['run']}`)" for r in clean) + ".")
    for r in rs:
        if HINTED in r["trained_cells"] and UNHINTED in r["cells"] and NEVER in r["cells"]:
            e = r["ended"]
            env = next(x for x in r["result"]["environments"] if x["env_id"] == NEVER[0])
            lines.append(f"- `{r['run']}` trained on {name(HINTED)} itself, on the accounting of that time. There the endings at the rejected-call "
                         f"limit {moved(e[HINTED][0]['rejected'], e[HINTED][1]['rejected'])} of {e[HINTED][0]['episodes']} and the passes "
                         f"{moved(e[HINTED][0]['passed'], e[HINTED][1]['passed'])}; on {name(UNHINTED)} the endings "
                         f"{moved(e[UNHINTED][0]['rejected'], e[UNHINTED][1]['rejected'])}; on {name(NEVER)}, which it "
                         f"never trained on, they {moved(e[NEVER][0]['rejected'], e[NEVER][1]['rejected'])} while its "
                         f"passes {moved(e[NEVER][0]['passed'], e[NEVER][1]['passed'])}.")
            lines.append(f"- The plan's \"no harm\" reads the pass rate only and does not see this: {NEVER[0].split('/')[1]} in `{r['run']}` is "
                         f"{gain(env['paired'])} points and counts as no harm: {env['meets_rule']}.")
    many = by.get(MANY_CELLS_RUN)
    if many:
        b, t = (sum(x[i]["cut"] for x in many["ended"].values()) for i in (0, 1))
        n = sum(x[0]["episodes"] for x in many["ended"].values())
        lines.append(f"- `{many['run']}` ({len(many['trained_cells'])} training cells): the episodes cut by length {moved(b, t)} "
                     f"of {n} ({100 * t / n:.0f} % of the trained side), and the median context grows in {sum(x[1]['context'] > x[0]['context'] for x in many['ended'].values())} "
                     f"of its {len(many['ended'])} cells. Its own readout stopped on its length check for that reason.")
    ref, last = next(p for p in reversed(line["points"]) if p["run"] == FIRST_RUN), cont[-1]
    for k in (L1, HINTED):
        a, b = ref["ended"][k][1], last["ended"][k][1]
        lines.append(f"- The adapter line (section {SEC['line']}) from step {ref['step']} (`{ref['run']}`) to step {last['step']} (`{last['run']}`) on "
                     f"{name(k)}: the endings at the rejected-call limit {moved(a['rejected'], b['rejected'])}, the episodes cut by length "
                     f"{moved(a['cut'], b['cut'])} of {a['episodes']}, and the median context "
                     f"{moved(a['context'], b['context'], ('falls', 'rises', 'stays'), '{:,}')} tokens.")
    lines.append(f"- The accounting was corrected afterwards (commit {d['runs'][RECOUNT_RUN]['code']}): until then a call that the instrument refused "
                 "counted toward the rejected-call limit like a call that was written wrongly; since then only a call that is written wrongly "
                 f"does. Run `{RECOUNT_RUN}` trained {d['runs'][RECOUNT_RUN]['cells'].split(' (')[0]} on that code (section {SEC['second']}); it has no readout.")
    return lines


def line_section(d: dict, line: dict) -> list[str]:
    """One adapter on the same secret instances at every step it was read: passes per cell with the paired difference
    against base, then what the table holds, in sentences read from it."""
    points, base = line["points"], line["base"]
    first, cont = [p for p in points if p["run"] == FIRST_RUN], [p for p in points if p["run"] != FIRST_RUN]
    n = len(_of(base, L1))
    forked = min(r["step"] for r in _train(d["logs"][LONGER_RUN])) - 1
    who = lambda k: words(f"`{r}`" for r in dict.fromkeys(p["run"] for p in points if k in p["trained_cells"])) or "-"           # noqa: E731
    head = ["cell", "training cell of", "base"] + [p["label"] + (", chosen on the dev looks" if p["reader"] == SELECTED else "") for p in points]
    out = [[name(k), who(k), _ended(base, k)["passed"]] + [f"{p['ended'][k][1]['passed']}: {gain(p['cells'][k]['paired'])}" for p in points]
           for k in sorted(points[0]["cells"])]
    lines = [f"`{FIRST_RUN}` trained on {words(name(k) for k in first[0]['trained_cells'])} for {first[-1]['step']} steps. `{LONGER_RUN}` continued "
             f"its step-{forked} weights on {words(name(k) for k in cont[0]['trained_cells'])} to step {cont[-1]['step']}. Their adapter was read "
             f"{len(points)} times against one base log on the same secret instances ({len(points) + 1} readings with the base model's): passes of "
             f"{n} episodes per cell, then the paired difference against base in points with its 95 % interval. The step-"
             f"{next(p['step'] for p in points if p['reader'] == SELECTED)} checkpoint was chosen on the dev looks of `{LONGER_RUN}` (section "
             f"{SEC['dev']}); the other readings are of fixed steps.", ""] + table(head, out) + [""]
    seq = lambda k: [_ended(base, k)["passed"]] + [p["ended"][k][1]["passed"] for p in points]                                 # noqa: E731
    hard = seq(L2)
    lines.append(f"- {name(L2)} " + ("rises at every reading" if all(b > a for a, b in zip(hard, hard[1:])) else "does not rise at every reading")
                 + ": " + ", ".join(map(str, hard)) + f" of {n}.")
    for k, what in ((L1, "not a training cell of the continuation"), (HINTED, "the family never trained on")):
        ref, mid, last = first[-1], cont[0], cont[-1]
        lines.append(f"- {name(k)}, {what}: its passes {moved(ref['ended'][k][1]['passed'], last['ended'][k][1]['passed'])} of {n} between step "
                     f"{ref['step']} and step {last['step']} ({mid['ended'][k][1]['passed']} at step {mid['step']}; the base model passes "
                     f"{_ended(base, k)['passed']}).")
    harm = [(p, next(e for e in p["result"]["environments"] if e["role"] == "untrained")) for p in points]
    broken = [(p, e) for p, e in harm if not e["meets_rule"]]
    lines.append(f"- The plan's \"no harm\" (the environment never trained on is at most {100 * HARM:.0f} points below base) "
                 + ("holds at every reading." if not broken else "is violated at " + words(f"step {p['step']}" for p, _ in broken) + ": "
                    + ", ".join(f"{e['env_id'].split('/')[1]} is {gain(e['paired'])} points against base" for _, e in broken)
                    + ". It holds at " + words(f"step {p['step']}" for p, e in harm if e["meets_rule"]) + "."))
    return lines


def second_family_section(d: dict, rs: list[dict]) -> list[str]:
    """Every run that had the other family as a training cell: what it ran on, its dev looks on the practice level,
    its readout there where it has one, how it ended; for the run on the corrected accounting, its training by blocks."""
    runs, logs = d["runs"], d["logs"]
    by = {r["run"]: r for r in rs}
    family = [k for k in runs if k in logs and any(e == HINTED[0] for e, _ in _cells(_train(logs[k])))]
    out = []
    for k in family:
        looks = _of(S._looks(logs[k], "dev"), HINTED)
        dev = ", ".join(f"{frac([r for r in looks if r['step'] == st])} at {st}" for st in sorted({r["step"] for r in looks}))
        c = by[k]["cells"][HINTED] if k in by else None
        secret = "-" if c is None else (f"{by[k]['ended'][HINTED][0]['passed']} / {by[k]['ended'][HINTED][0]['episodes']} -> "
                                        f"{by[k]['ended'][HINTED][1]['passed']} / {by[k]['ended'][HINTED][1]['episodes']}: {gain(c['paired'])}")
        out.append([f"`{k}`", runs[k]["code"], "by class" if k == RECOUNT_RUN else "before the correction",
                    ", ".join(name(x) for x in _cells(_train(logs[k]))), max(r["step"] for r in _train(logs[k])), dev, secret, runs[k]["state"]])
    lines = [f"{HINTED[0].split('/')[1]} was a training cell in {len(family)} runs. Level 0 is its practice level: the text states the method.", ""]
    lines += table(["run", "code", "rejected-call accounting", "trained on", "last training step", f"dev passes of {name(HINTED)} by step",
                    f"secret split, {name(HINTED)}: base -> trained, paired difference [95 %]", "how it ended"], out) + [""]
    for k in family:
        if k in STOP_CONFIG:
            stop = stop_of(d, k)
            lines.append(f"- `{k}` stopped on `{stop['rule']}`: {stop['reason']}." if stop["stop"] else f"- `{k}`: no rule stops on its stored log.")
    lines.append(f"- In `{RECOUNT_RUN}` the rejected calls were already counted by class (commit {runs[RECOUNT_RUN]['code']}, section "
                 f"{SEC['behaviour']}), so that accounting is not the explanation there. `{STOPPED_RUN}` holds the episodes of `v16`.")
    train = _train(logs[RECOUNT_RUN])
    last, block = max(r["step"] for r in train), 5
    blocks = []
    for lo in range(0, last, block):
        eps = [r for r in train if lo < r["step"] <= lo + block]
        blocks.append([f"{lo + 1}-{min(lo + block, last)}", f"{frac(eps)} ({100 * sum(bool(r['passed']) for r in eps) / len(eps):.0f} %)",
                       f"{sum(bool(r['submitted']) for r in eps)} / {len(eps)} ({100 * sum(bool(r['submitted']) for r in eps) / len(eps):.0f} %)",
                       f"{_ended(eps)['context']:,}"])
    lines += ["", f"Training rollouts of `{RECOUNT_RUN}` in blocks of {block} steps (all its training cells):", ""]
    lines += table(["steps", "passes / rollouts", "rollouts that delivered", "median context tokens"], blocks)
    return lines


def dev_section(log):
    looks = S._looks(log, "dev")
    cells = sorted({(r["env_id"], r["level"]) for r in looks})
    steps = sorted({r["step"] for r in looks})
    table_rows = {d["step"]: d for d in S.dev_table(log, dict(S.DEFAULTS))}
    out = []
    for st in steps:
        at = [r for r in looks if r["step"] == st]
        line = [st] + [frac(_of(at, c)) for c in cells]
        lk = table_rows[st]["look"]
        line += [_ended(at)["cut"], f"{_ended(at)['context']:,}", "-" if not lk else f"{pts(lk['diff'])}"]
        out.append(line)
    return table(["step"] + [f"{e.split('/')[1]} L{lv}" for e, lv in cells]
                 + ["cut by length (all cells)", "median context tokens (all cells)", "all cells against the first look, points"], out)


def train_section(log, block=10):
    train = _train(log)
    cells = sorted({(r["env_id"], r["level"]) for r in train})
    last = max(r["step"] for r in train)
    first = min(r["step"] for r in train)
    out = []
    for lo in range(((first - 1) // block) * block, last, block):
        eps = [r for r in train if lo < r["step"] <= lo + block]
        out.append([f"{max(lo + 1, first)}-{min(lo + block, last)}"] + [frac([r for r in eps if (r["env_id"], r["level"]) == c]) for c in cells])
    return table(["steps"] + [f"{e.split('/')[1]} L{lv}" for e, lv in cells], out)


def c01_length(cal):
    med = sorted(c["stream_tokens_est"]["p50"] for c in cal["cells"] if c.get("stream_tokens_est"))
    return (f"the median episode of a cell runs from {med[0]:,} to {med[-1]:,} tokens as one stream, which the accelerators used here "
            "do not hold for training.") if med else "its episodes are too long to train on one accelerator."


def calibration_section(cals):
    keys = [(c["env_id"], c["level"]) for c in cals["c05"]["cells"]]
    by = {k: {(c["env_id"], c["level"]): c for c in v["cells"]} for k, v in cals.items()}
    show = lambda c: "-" if c is None else f"{c['passed']} / {c['n_episodes']}"                                   # noqa: E731
    rej = lambda c: "-" if c is None else c["outcome_counts"].get(REJECTED, 0)                                    # noqa: E731
    out = []
    for env, lv in sorted(keys):
        a, b, c = (by[k].get((env, lv)) for k in ("c01", "c04", "c05"))
        out.append([f"{env.split('/')[1]} L{lv}", show(a), show(b), rej(b), show(c), rej(c), f"{c['stream_tokens_est']['p50']:,} / {c['stream_tokens_est']['p90']:,}"])
    return table(["cell", "`c01` reasoning on", "`c04` thinking off", "ended by rejected calls", "`c05` thinking off, corrected interface",
                  "ended by rejected calls", "`c05` stream tokens p50 / p90"], out)


def dead_families(cal):
    """(families whose every cell is in the dead band, families) of a calibration."""
    by = collections.defaultdict(list)
    for c in cal["cells"]:
        by[c["env_id"]].append(learn.band(c["passed"] / c["n_episodes"]))
    return sum(all(b == "dead" for b in bands) for bands in by.values()), len(by)


def repeat_sentence(run, log):
    """What two evaluations of the same weights gave (a run that was preempted right after an evaluation)."""
    looks = S._looks(log, "dev")
    for step in sorted({r["step"] for r in looks}):
        attempts = sorted({r["attempt"] for r in looks if r["step"] == step})
        if step > 0 and len(attempts) > 1:
            c = sorted({(r["env_id"], r["level"]) for r in looks})[0]
            parts = [frac([r for r in looks if r["step"] == step and r["attempt"] == a and (r["env_id"], r["level"]) == c]) for a in attempts]
            return (f"In run `{run}` the weights of step {step} were evaluated {len(attempts)} times on the same instances: "
                    f"{' and '.join(parts)} on {c[0].split('/')[1]} level {c[1]}.")
    return ""


def dev_against_secret(d: dict, r: dict, key=L1):
    """(last dev look, secret-split readout) of a cell of one run after training: two rate dicts."""
    looks = _of(S._looks(d["logs"][r["run"]], "dev"), key)
    last = max(x["step"] for x in looks)
    return _rate([x for x in looks if x["step"] == last]), _rate(_of(d["tests"][r["run"]]["trained"], key)), last


# ------------------------------------------------------------------------------------------------ the report
def build(d: dict | None = None) -> str:
    d = d or stored()
    runs, logs, cals = d["runs"], d["logs"], d["cals"]
    rs = readouts(d)
    by = {r["run"]: r for r in rs}
    main, first = by[MAIN_RUN], by[FIRST_RUN]
    line = line_of(d)
    early_point = line["points"][0]
    early, early_step, early_rows = early_point["result"], early_point["step"], early_point["rows"]
    selected = next(p for p in line["points"] if p["reader"] == SELECTED)
    limit = lambda k: f"{_episode_tokens(runs[k]):,}-token"                                         # noqa: E731
    last_step = lambda k: max(r["step"] for r in _train(logs[k]))                                   # noqa: E731
    dev, secret, dev_step = dev_against_secret(d, main)
    many = by[MANY_CELLS_RUN]
    many_cut, many_n = sum(x[1]["cut"] for x in many["ended"].values()), sum(x[1]["episodes"] for x in many["ended"].values())
    base_logs = len({r["files"][0] for r in d["tests"].values()})
    trained_logs = len({r["files"][1] for r in d["tests"].values()} | {p["file"] for p in line["points"]})
    stops = {k: stop_of(d, k) for k in STOP_CONFIG}
    second = [k for k in runs if k in logs and any(e == HINTED[0] for e, _ in _cells(_train(logs[k])))]
    attempts_in_base = {k: len({r["attempt"] for r in t["raw"][0]}) for k, t in d["tests"].items()}
    dead, families = dead_families(cals["c05"])
    unhinted = [f"{r['ended'][UNHINTED][0]['passed']} / {r['ended'][UNHINTED][0]['episodes']} -> {r['ended'][UNHINTED][1]['passed']} / "
                f"{r['ended'][UNHINTED][1]['episodes']} (`{r['run']}`)" for r in rs if UNHINTED in r["cells"]]
    sizes = sorted({n for r in rs for n in r["instances"]}, reverse=True)
    training = [k for k in logs if runs[k]["state"].startswith("training") and "finished" not in runs[k]["state"]]
    L = (["<!-- Generated by docs/experiments/27b/report.py from docs/experiments/27b/results/. Do not edit: change the script. -->",
         "# Results: Qwen3.8-27B on MR Gym by reinforcement learning, first runs", "",
         f"Files copied {runs['_copied_utc']}. Some runs were still training then; their tables stop where the copied log stops.", "",
         "**Status: exploratory.** These runs came before the plan of [PREREG-27b.md](PREREG-27b.md) could be followed as written; the "
         f"deviations are listed in section {SEC['deviations']}. {len(rs)} training runs from the base model were read out on the secret split, each "
         f"with one sampler seed; a cell holds {' or '.join(map(str, sizes))} secret instances. The headline is run `{MAIN_RUN}`: the recipe of "
         f"[RECIPE.md](../RECIPE.md) on the published code (commit {runs[MAIN_RUN]['code']}). Read it as evidence that the model learns on the "
         "environment, not as the pre-registered result.", "",
         f"**Engine.** {engine_sentence()} Every run on this page was played on that engine (the commits in the table of runs). Results on "
         "two engine versions are not comparable: an engine version changes the data of the scans ([CHANGELOG.md](../../CHANGELOG.md)).", "",
         "The model is trained by reinforcement learning only: LoRA (rank 32) with GRPO in verl on episodes played through the "
         "environment's tools, reward 1 when the delivered result passes the verifier and 0 otherwise. No demonstrations, no supervised "
         "step, no reference answers. Thinking is off and a reply holds at most 2,048 tokens, because an episode has to fit one "
         f"accelerator as a single token stream (section {SEC['calibration']}).", "",
         f"## {SEC['runs']}. Runs", ""] + runs_table(d) + ["",
         f"## {SEC['replications']}. Replications", "",
         "Every finished training run, trained against the base model on the same instances of the secret split `test-iid`, in the "
         "trainer's harness. One row per run; the trained level is the cell that every run trained on.", ""] + replications_section(rs) + [
         "", "What the tables show, and what they do not:", ""] + findings(rs) + ["",
         f"## {SEC['line']}. {TITLES['line']}", ""] + line_section(d, line) + ["",
         f"## {SEC['behaviour']}. What training does to behaviour", "",
         "How the episodes of the secret split ended, base -> trained, per run and cell: passes, episodes ended by the rejected-call "
         f"limit (outcome `{REJECTED}`), episodes cut by length (a reply cut by its token limit, or a full context), and the median "
         f"context of an episode in tokens. The last rows are the continuation of the adapter line (section {SEC['line']}).", ""]
         + behaviour_section(d, rs) + ["",
         f"## {SEC['second']}. {TITLES['second']}", ""] + second_family_section(d, rs) + ["",
         f"## {SEC['secret']}. The secret split, run by run", "",
         f"### Run `{MAIN_RUN}`: the recipe on the published code", "",
         f"Trained on {words(name(k) for k in main['trained_cells'])} for {main['steps']} steps on {_node(runs[MAIN_RUN])} with a {limit(MAIN_RUN)} "
         f"episode limit, then read out by the run itself: {main['instances'][0]} instances per cell, {main['episodes']} episodes each.", ""]
         + readout_section(main["result"], f"after {main['steps']} steps") + ["", "How its episodes ended:", ""]
         + outcomes_section([("base", d["tests"][MAIN_RUN]["base"]), (f"step {main['steps']}", d["tests"][MAIN_RUN]["trained"])]) + ["",
         f"### Run `{FIRST_RUN}`: the first result", "",
         f"Trained on {words(name(k) for k in first['trained_cells'])} on {_node(runs[FIRST_RUN])} with a {limit(FIRST_RUN)} episode limit. Its "
         f"weights after {early_step} and after {first['steps']} steps were evaluated against the base model by evaluation-only runs on a "
         f"{limit(first['by'])} limit (runs `v12`, `{first['by']}`; the base half is shared).", "",
         f"After {first['steps']} steps (the final weights):", ""] + readout_section(first["result"], f"after {first['steps']} steps") + ["",
         f"After {early_step} steps:", ""] + readout_section(early, f"after {early_step} steps") + ["", "How its episodes ended:", ""]
         + outcomes_section([("base", d["tests"][FIRST_RUN]["base"]), (f"step {early_step}", early_rows),
                             (f"step {first['steps']}", d["tests"][FIRST_RUN]["trained"])]) + ["", "Length of its episodes:", ""]
         + tokens_section([("base", d["tests"][FIRST_RUN]["base"]), (f"step {early_step}", early_rows),
                           (f"step {first['steps']}", d["tests"][FIRST_RUN]["trained"])]) + [""])
    for r in rs:
        if r["run"] not in (MAIN_RUN, FIRST_RUN):
            L += [f"### Run `{r['run']}`", "",
                  f"Trained on {words(name(k) for k in r['trained_cells'])} for {r['steps']} steps (code {r['code']}); "
                  f"{r['instances'][0]} instances per cell, {r['episodes']} episodes each.", ""] + readout_section(r["result"], f"after {r['steps']} steps") + [""]
    L += [f"## {SEC['dev']}. Evaluation during training (public dev split)", "",
          "Passes / episodes per cell at each evaluation; 2 episodes per instance. When a run was preempted and restored a checkpoint, "
          "the evaluation of the same weights was repeated and both are pooled here. " + repeat_sentence(FIRST_RUN, logs[FIRST_RUN]) +
          " A single cell of a single look is a direction, not a number.", ""]
    notes = {
        MAIN_RUN: f"The recipe on the published code, {limit(MAIN_RUN)} episode limit.",
        FIRST_RUN: f"NoWrap-v1 levels 1-2 on a {limit(FIRST_RUN)} episode limit: level-2 episodes are cut there, which is why its level-2 column "
                   f"understates its readout (section {SEC['secret']}).",
        MANY_CELLS_RUN: f"Four cells at once on the {limit(MANY_CELLS_RUN)} limit, same code as `{FIRST_RUN}`.",
        "v11": "NullTissue-v0 level 0 + NoWrap-v1 level 1 on an interim code state in which only the calculator's step limit was raised.",
        LONGER_RUN: f"Continues the step-50 weights of `{FIRST_RUN}` on NoWrap-v1 levels 2 and 3 only, {limit(LONGER_RUN)} limit; level 1 is watched "
                    f"for forgetting. The log ends at step {last_step(LONGER_RUN)}. Its secret-split readings are in section {SEC['line']}.",
        STOPPED_RUN: f"NullTissue-v0 levels 0 and 1 from the base model on the corrected interface (it holds the episodes of `v16`). "
                     f"Stopped on `{stops[STOPPED_RUN]['rule']}`: {stops[STOPPED_RUN]['reason']}.",
        RECOUNT_RUN: f"The same cells on the code that counts rejected calls by class (commit {runs[RECOUNT_RUN]['code']}). "
                     f"Stopped on `{stops[RECOUNT_RUN]['rule']}`: {stops[RECOUNT_RUN]['reason']}.",
    }
    order = [k for k in notes if k in logs]
    for k in order:
        L += [f"### Run `{k}`", "", notes[k], ""] + dev_section(logs[k]) + [""]
    L += [f"## {SEC['rollouts']}. Training rollouts", "", "Passes / rollouts per cell in blocks of ten steps (8 groups of 8 rollouts per step; the sampler "
          "chooses the cell of each group).", ""]
    for k in order:
        L += [f"### Run `{k}`", ""] + train_section(logs[k]) + [""]
    L += [f"## {SEC['calibration']}. Where the untrained model stands", "",
          "Passes / episodes of the base model per cell in the chat harness (`mrgym calibrate`). `c01`: reasoning on (medium effort, "
          "16,384-token replies), an earlier interface; " + c01_length(cals["c01"]) + " `c04`: thinking off, 2,048-token replies. `c05`: the same "
          f"setting after the interface corrections of section {SEC['reads']} (twice the instances).", ""] + calibration_section(cals) + ["",
          "- With thinking off the model passes where the task text gives the method (NoWrap-v1 levels 0-1, NullTissue-v0 level 0) "
          "and almost nowhere else; with reasoning on it passes several of the unhinted cells. The tasks are within its reach; the "
          "setting that can be trained is the weak one.",
          "- The corrections moved the cells that the calculator had been blocking, and did not move the unhinted levels: there "
          "the gap is the method.",
          f"- ArtefactClinic-v0 in `c01` and `c04` was served from a build that was not thread-safe (section {SEC['reads']}); those two columns "
          "are not valid for that family.", "",
          f"## {SEC['reads']}. What reading the conversations found", "",
          "Four reads of about 250 stored conversations of `c04` against the verifiers: [NoWrap levels 2-3](27b/reads/nowrap_l2_l3.md), "
          "[PhantomQA](27b/reads/phantomqa.md), [NullTissue, T1Transmit, BandFree, ProtocolToSpec](27b/reads/fitting_families.md), "
          "[SNRSpec, FatSuppress, ArtefactClinic](27b/reads/snrspec_fatsuppress_clinic.md). They were written against the code "
          "before the corrections; paths in them are relative to the run folders.", "",
          "- No scoring defect was found in any verifier (near misses recomputed by hand and re-scored).",
          "- Defects of the environment, all corrected before `c05`: the calculator stopped legitimate fitting programs at a step limit, "
          "refused to print a non-finite number, and gave list `+` and `*` a meaning other than Python's; the statements never said "
          "that the agent's own scans may use any protocol; `plan` printed the image size in the reverse axis order and called a "
          "geometry that is never scanned \"over budget\"; `profile` clipped silently; ArtefactClinic-v0 instances depended on what "
          "other threads of the process were building.",
          "- What remains is method: without the hint the model looks at the object with the slow target sequence, probes at the "
          "target repetition time, or never measures the transmit field.",
          f"- An episode limit that is too small hides learning: on the {limit(FIRST_RUN)} limit level 2 looked dead in `{FIRST_RUN}`, on the "
          f"{limit(first['by'])} limit the same weights pass it (sections {SEC['secret']} and {SEC['dev']}).", "",
          f"## {SEC['deviations']}. Deviations from the pre-registration", "",
          "- Thinking off and 2,048-token replies instead of the planned reasoning setting: with reasoning an episode does not fit one accelerator.",
          "- Trained environments that meet the rule, per run: " + ", ".join(f"{r['result']['verdict']['trained_environments_meeting_rule']} (`{r['run']}`)" for r in rs)
          + f"; the plan requires {MIN_ENVS} for \"it trained\" (section {SEC['replications']}).",
          f"- The readout cells of `{FIRST_RUN}` did not fit the episode limit of its node and its own readout stopped on its length check; the "
          "readout was made by separate evaluation-only runs on a larger limit, with the same instances and harness.",
          f"- The readout of `{MANY_CELLS_RUN}` stopped on its length check as well ({many_cut} of its {many_n} trained episodes were cut, "
          f"{100 * many_cut / many_n:.0f} %); its two evaluation logs are complete and the paired readout here is computed from them.",
          f"- The secret split was read {trained_logs} times with trained weights ("
          + ", ".join(f"`{k}` after {words(p['step'] for p in line['points'] if p['run'] == k)} steps" for k in dict.fromkeys(p["run"] for p in line["points"]))
          + ", " + ", ".join(f"`{r['run']}` after {r['steps']}" for r in rs if r["run"] != FIRST_RUN) + f") and {base_logs} times with the base model; "
          "every reading is in this report. Where an evaluation log holds more than one attempt ("
          + (", ".join(f"the base log of `{k}`: {n}" for k, n in attempts_in_base.items() if n > 1) or "none does") + "), the newest is used, as the "
          "readout of the node scripts does.",
          f"- The adapter of `{FIRST_RUN}` and `{LONGER_RUN}` was read {len(line['points']) + 1} times on the same secret instances, the base model's "
          f"reading included; the checkpoint of step {selected['step']} was chosen on the dev looks, the other steps were fixed (section {SEC['line']}).",
          f"- {' or '.join(map(str, sizes))} secret instances per cell instead of the planned number; one sampler seed per run.",
          f"- `v11` ran on {runs['v11']['code']}.",
          "- The evaluation table pools repeated looks at the same weights after a restart; the table the job itself printed kept the newer look only.", "",
          f"## {SEC['limits']}. Limits", "",
          f"- {len(rs)} training runs from the base model have a readout. They differ in code, cells and node; none repeats another with a "
          "second sampler seed.",
          f"- The public dev split overstated the trained level in `{MAIN_RUN}`: its dev look at step {dev_step} gave {dev['passed']} / {dev['episodes']} "
          f"({100 * dev['rate']:.0f} %), the secret split {secret['passed']} / {secret['episodes']} ({100 * secret['rate']:.0f} %). The secret-split "
          "figure is the one to quote.",
          f"- The gain on NullTissue-v0 is on its practice level, whose text states the method. The level without that text was read in "
          f"{len(unhinted)} runs: " + ", ".join(unhinted) + ".",
          f"- Training can make a cell worse while the pass-rate rule reports no harm (section {SEC['behaviour']}).",
          "- All tasks of the suite are measurement tasks by the suite's own open-loop control; nothing here is evidence about decision tasks.",
          f"- {dead} of the {families} scored families have every cell below {100 * learn.BANDS[0][2]:.0f} % for the thinking-off model in `c05`: "
          "the reward has nothing to start from there.",
          f"- The step-{selected['step']} reading of the adapter line is of a checkpoint selected on the dev looks; it is not a fixed-step result "
          f"(section {SEC['line']}).",
          f"- A second family did not train: {HINTED[0].split('/')[1]} was a training cell in {len(second)} runs (section {SEC['second']}).",
          *([f"- {words(f'`{k}`' for k in training)} {'was' if len(training) == 1 else 'were'} still training when the files were copied ("
             + ", ".join(f"`{k}` at step {last_step(k)}" for k in training) + ")."] if training else []),
          "- No conversation of a readout episode has been read: the report counts how episodes ended, it does not explain why.", "",
          f"## {SEC['reproduce']}. Reproduce", "",
          "```", "python docs/experiments/27b/report.py --check          # this report from the stored files",
          "python -m mrgym_verl.readout --base <test_base.jsonl> --trained <test_trained.jsonl> --untrained Scanner/NullTissue-v0 --min-envs 2",
          "bash integrations/verl/node/make_main.sh train <settings.env> <job.sh>   # the node script of a run; settings in runs.json", "```", "",
          "The steps with one settings file, what a run costs and the pitfalls: [RECIPE.md](../RECIPE.md).", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ the front page
def _rate(eps):
    c = R._rate(eps)
    return dict(passed=sum(bool(r["passed"]) for r in eps), episodes=len(eps), rate=round(c["pass_rate"], 4), lo=round(c["lo"], 4), hi=round(c["hi"], 4))


def _curve(log, key):
    looks = S._looks(log, "dev")
    at = lambda st: [r for r in _of(looks, key) if r["step"] == st]                                  # noqa: E731
    return [dict(step=st, cut=sum("length" in r["outcome"] for r in at(st)), **_rate(at(st))) for st in sorted({r["step"] for r in _of(looks, key)})]


def figure_data(d: dict) -> dict:
    """Everything the figure and the front-page table show, from the stored logs (rounded, so that the file is the same
    on every machine): the headline run's cells on the secret split, the trained level in every run that has a
    readout, and the dev curves."""
    rs = readouts(d)
    main = rs[0]
    t = d["tests"][MAIN_RUN]
    secret = []
    for key, c in sorted(main["cells"].items()):
        p = c["paired"]
        secret.append(dict(cell=name(key), label=PLAIN[key][0], row=PLAIN[key][1], before=_rate(_of(t["base"], key)), after=_rate(_of(t["trained"], key)),
                           instances=p["instances"], diff=round(p["diff"], 4), lo=round(p["lo"], 4), hi=round(p["hi"], 4)))
    curve = lambda run, key: dict(run=run, cell=name(key), label=PLAIN[key][0], points=_curve(d["logs"][run], key))   # noqa: E731
    replications = []
    for r in rs:
        p, tt = r["cells"][L1]["paired"], d["tests"][r["run"]]
        replications.append(dict(run=r["run"], code=r["code"].strip("`"), steps=r["steps"], instances=p["instances"], before=_rate(_of(tt["base"], L1)),
                                 after=_rate(_of(tt["trained"], L1)), diff=round(p["diff"], 4), lo=round(p["lo"], 4), hi=round(p["hi"], 4)))
    return dict(schema="mrgym-27b-figure/2", version=FIGURE_VERSION, run=MAIN_RUN, code=d["runs"][MAIN_RUN]["code"].strip("`"), steps=main["steps"],
                secret=secret, replications=replications,
                rule=dict(needed=main["result"]["verdict"]["needed"], met=[r["result"]["verdict"]["trained_environments_meeting_rule"] for r in rs]),
                dev=[curve(MAIN_RUN, L1), curve(MAIN_RUN, L2)], earlier=[curve(r["run"], L1) for r in rs[1:]])


BLUE, ORANGE = "#1c5cb5", "#e07b1a"                     # distinct for colour-blind readers and by lightness in greyscale
INK, INK2, MUTED, GRID = "#0b0b0b", "#4a4946", "#807e78", "#e1e0d9"


def draw(data: dict, path: str) -> None:
    """The figure of the front page: left, before and after on the secret split with 95 % intervals; right, the public
    dev instances during training, the headline run thick and the earlier runs thin. Two colours; every series is
    labelled where it is drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.edgecolor": GRID, "axes.linewidth": 0.6,
                         "xtick.color": GRID, "ytick.color": GRID, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    pct = lambda x: f"{100 * x:.0f} %"                                                               # noqa: E731
    fig = plt.figure(figsize=(9.6, 3.7), dpi=160, facecolor="white")
    # ---- left: before and after, one row per cell
    ax = fig.add_axes([0.225, 0.15, 0.255, 0.60])
    n = len(data["secret"])
    for i, c in enumerate(data["secret"]):
        y = n - 1 - i
        for side, dy, colour, fill in (("before", 0.16, MUTED, "white"), ("after", -0.16, BLUE, BLUE)):
            x = c[side]
            ax.plot([100 * x["lo"], 100 * x["hi"]], [y + dy, y + dy], color=colour, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot([100 * x["rate"]], [y + dy], "o", ms=7.5, mfc=fill, mec=colour, mew=1.6, zorder=3)
            ax.text(100 * x["hi"] + 2.5, y + dy, pct(x["rate"]), va="center", ha="left", fontsize=8.5, color=INK if side == "after" else INK2,
                    fontweight="bold" if side == "after" else "normal")
            if i == 0:
                ax.text(100 * x["lo"] - 2.5, y + dy, "before" if side == "before" else f"after {data['steps']} steps", va="center", ha="right",
                        fontsize=8, color=INK2)
        ax.text(-0.03, y + 0.10, c["label"].split(" (")[0], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(-0.03, y - 0.22, c["cell"] + ("" if "(" not in c["label"] else ", " + c["label"].split(" (")[1].rstrip(")")),
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(-3, 112); ax.set_ylim(-0.6, n - 0.4); ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100 %"])
    for x in (0, 25, 50, 75, 100):
        ax.axvline(x, color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("episodes that pass", fontsize=8.5, color=INK2)
    one = data["secret"][0]
    fig.text(0.02, 0.93, "Before and after training, on secret test instances", fontsize=10.5, fontweight="bold", color=INK, va="center")
    fig.text(0.02, 0.855, f"run {data['run']}: {one['instances']} instances x {one['before']['episodes'] // one['instances']} episodes per row; "
             f"lines: 95 % intervals", fontsize=8, color=INK2, va="center")
    # ---- right: the public dev instances during training
    bx = fig.add_axes([0.575, 0.15, 0.30, 0.60])
    for y in (0, 25, 50, 75, 100):
        bx.axhline(y, color=GRID, lw=0.6, zorder=0)
    for series in data["earlier"]:
        bx.plot([p["step"] for p in series["points"]], [100 * p["rate"] for p in series["points"]], color=BLUE, lw=0.9, alpha=0.75, zorder=2)
    for series, colour, marker in zip(data["dev"], (BLUE, ORANGE), ("o", "s")):
        pts_ = series["points"]
        bx.plot([p["step"] for p in pts_], [100 * p["rate"] for p in pts_], color=colour, lw=2.0, marker=marker, ms=6.5, mec="white", mew=1.4,
                solid_joinstyle="round", zorder=3, clip_on=False)
        bx.text(pts_[-1]["step"] + 2.2, 100 * pts_[-1]["rate"], series["label"], va="center", ha="left", fontsize=9, color=INK)
        bx.text(pts_[-1]["step"] + 2.2, 100 * pts_[-1]["rate"] - 8.5, pct(pts_[-1]["rate"]), va="center", ha="left", fontsize=8.5, color=INK2)
    if data["earlier"]:
        bx.text(3, 30, f"thin lines: the trained level in {len(data['earlier'])} earlier runs", fontsize=8, color=INK2, ha="left", va="center")
    last = max(p["step"] for s in data["dev"] for p in s["points"])
    bx.set_xlim(-2, last + 1); bx.set_ylim(-4, 104)
    bx.set_xticks(sorted({p["step"] for p in data["dev"][0]["points"]}))
    bx.set_yticks([0, 25, 50, 75, 100]); bx.set_yticklabels(["0", "25", "50", "75", "100 %"])
    for side in ("top", "right", "left", "bottom"):
        bx.spines[side].set_visible(False)
    bx.tick_params(length=0)
    bx.set_xlabel("training step", fontsize=8.5, color=INK2)
    per = sorted({p["episodes"] for p in data["dev"][0]["points"]})
    fig.text(0.535, 0.93, "During training, on unseen public instances", fontsize=10.5, fontweight="bold", color=INK, va="center")
    fig.text(0.535, 0.855, f"run {data['run']} (thick): {' or '.join(map(str, per))} episodes per point", fontsize=8, color=INK2, va="center")
    fig.savefig(path, dpi=160, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def write_figure(d: dict) -> None:
    data = figure_data(d)
    draw(data, FIGURE)
    with open(SIDECAR, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.write("\n")


def settings(path: str = RECIPE_ENV) -> dict:
    """The ``NAME=value`` lines of a settings file, as the shell reads them (comments dropped, quotes removed)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = shlex.split(line, comments=True)
            parts = parts[1:] if parts[:1] == ["export"] else parts
            if len(parts) == 1 and re.match(r"[A-Za-z_][A-Za-z0-9_]*=", parts[0]):
                key, _, value = parts[0].partition("=")
                out[key] = value
    return out


def _figure(data: dict, prefix: str, short: str) -> list[str]:
    return ['<p align="center">',
            f'  <img src="{prefix}assets/training_27b.png" width="900" alt="Pass rate of {short} before and after {data["steps"]} steps of '
            f'reinforcement learning on {len(data["secret"])} cells of the secret split, with 95 % intervals, and on public instances during training">',
            "</p>"]


def engine_sentence(plain: bool = False) -> str:
    """Which engine the stored runs were measured on, and, when the package serves another one, that they have not
    been repeated on it. ``plain``: engine versions as numbers (the front page), otherwise as engine ids."""
    short = lambda e: e.rsplit("-", 1)[-1] if plain else f"`{e}`"       # noqa: E731
    if mrgym.ENGINE_ID == MEASURED_ON:
        return f"Measured on engine {short(MEASURED_ON)}, the engine of this package."
    return (f"Measured on engine {short(MEASURED_ON)}; this package serves engine {short(mrgym.ENGINE_ID)}, on which the runs have not been "
            "repeated.")


def front_block(data: dict, prefix: str, model: str) -> str:
    """The TRAINING block of the front page: the figure, two plain sentences, two links. The first sentence gives the
    passes before and after training on the level that was trained and on the easy level of the family that was never
    trained (its task text states the method), on the secret split; the second says how much evidence that is. The
    table with every cell and its interval is on the recipe page and in the report."""
    import textwrap
    short = model.split("/")[-1]
    by = {c["cell"]: c for c in data["secret"]}
    trained, other = by[name(L1)], by[name(HINTED)]
    n = trained["before"]["episodes"]
    assert all(c[k]["episodes"] == n for c in (trained, other) for k in ("before", "after"))
    text = (f"{short} was trained for {data['steps']} steps by reinforcement learning, on the reward alone. On {n} test episodes with "
            f"objects it had never seen, the level it trained on went from {trained['before']['passed']} passes to "
            f"{trained['after']['passed']}, and an easy level of a task it never trained on from {other['before']['passed']} to "
            f"{other['after']['passed']}. {engine_sentence(plain=True)} These are first runs on one task, and training on a second task did not work: "
            f"[the recipe]({prefix}docs/RECIPE.md) and [the full report]({prefix}docs/experiments/RESULTS-27b.md) have the details.")
    return "\n".join([
        '<p align="center">',
        f'  <img src="{prefix}assets/training_27b.png" width="900"',
        f'       alt="Pass rate of {short} before and after {data["steps"]} steps of reinforcement learning, on test objects it had never seen">',
        "</p>", "",
        *next(lines for lines in (textwrap.wrap(text, width=w, break_long_words=False, break_on_hyphens=False) for w in range(124, 80, -4))
              if not any(re.match(r"\d+\. |[-+*] ", ln) for ln in lines))])     # no line that Markdown could read as a list item


def training_block(data: dict, prefix: str, model: str) -> str:
    short = model.split("/")[-1]
    out = [f"**{short}, reinforcement learning only, {data['steps']} steps, the recipe below on the published code: tested on instances it never "
           "saw.**", "", *_figure(data, prefix, short), ""]
    show = lambda x: f"{100 * x['rate']:.0f} % ({x['passed']} / {x['episodes']})"                   # noqa: E731
    out += table(["On secret test instances", "Before training", f"After {data['steps']} steps", "Gain in points [95 % interval]"],
                 [[f"{c['row']} · `{c['cell']}`", show(c["before"]), show(c["after"]), gain(c)] for c in data["secret"]])
    one, reps, met = data["secret"][0], data["replications"], data["rule"]["met"]
    out += ["", f"Evidence: run `{data['run']}` at commit `{data['code']}`, {one['instances']} secret instances x "
                f"{one['before']['episodes'] // one['instances']} episodes per row, base and trained on the same instances. The trained level's gain "
                f"was measured in {len(reps)} independent training runs: " + ", ".join(f"{gain(r)} (`{r['run']}`)" for r in reps) + ". Exploratory: "
                f"the pre-registered rule needs {data['rule']['needed']} trained environments, "
                + (f"each run trained {met[0]} that meets it" if len(set(met)) == 1 else "the runs trained " + words(met) + " that meet it")
                + f". The other family's row is a practice level whose text states the method. {engine_sentence()} "
                  f"[Full report]({prefix}docs/experiments/RESULTS-27b.md)."]
    return "\n".join(out)


def settings_block(env: dict, prefix: str) -> str:
    n = lambda k: f"{int(env[k]):,}"                                                                # noqa: E731
    thinking = "thinking off" if env["CHAT_KWARGS"] == "enable_thinking=False" else env["CHAT_KWARGS"]
    reward = "1 when the delivered result passes the verifier, else 0" if env["SIGNAL"] == "pass" else f"the training signal `{env['SIGNAL']}`"
    lines = [f"model      {env['MODEL_HF']} · {thinking} · replies up to {n('MAX_TOKENS_PER_TURN')} tokens · episode up to "
             f"{n('MAX_PROMPT_LEN')} prompt + {n('MAX_RESPONSE_LEN')} response tokens",
             f"adapter    LoRA rank {env['LORA_RANK']}, alpha {env['LORA_ALPHA']} · learning rate {env['LR']}, {env['WARMUP_STEPS']} warm-up steps"
             + (" · no KL term" if env["KL_COEF"] == "0" else f" · KL coefficient {env['KL_COEF']}"),
             f"algorithm  GRPO in verl · {env['TOTAL_STEPS']} steps x {env['PROMPTS_PER_STEP']} groups x {env['GROUP_SIZE']} rollouts · "
             f"{env['CURRICULUM_MODE']} curriculum over the training cells",
             f"reward     {reward} · no demonstrations, no supervised step",
             f"cells      train {env['TRAIN_ARMS']} · read out {env['TEST_ARMS']} (secret split, seeds {env['TEST_SEEDS']} x {env['VAL_N']} episodes)"]
    rel = os.path.relpath(RECIPE_ENV, ROOT).replace(os.sep, "/")
    return "\n".join(["```text"] + lines + ["```", "", f"Every line as the node scripts read it: [`{rel}`]({prefix}{rel})."])


def needs_block(d: dict, env: dict, prefix: str) -> str:
    runs = d["runs"]
    plan = load(os.path.join("..", "plan.json"))
    gb = {c["name"]: c["gb"] for c in plan["hardware"]["cards"]}
    card = lambda run: (lambda kind: f"{_node(run)} ({gb[kind]} GB each)")(_node(run).split()[-1])  # noqa: E731
    attempts, hours = _attempts(d["logs"][MAIN_RUN])
    steps = max(r["step"] for r in _train(d["logs"][MAIN_RUN]))
    evals = [(_attempts(x)[1], len(x)) for x in d["tests"][MAIN_RUN]["raw"]]
    return "\n".join([
        f"- **Model:** `{env['MODEL_HF']}`, open weights. The trainer is [verl]({prefix}integrations/verl/README.md) at its pinned commit.",
        f"- **One node with 8 accelerators.** Run `{MAIN_RUN}` used {card(runs[MAIN_RUN])} for the {_response_tokens(runs[MAIN_RUN]):,}-token responses "
        f"of these settings; {_response_tokens(runs[FIRST_RUN]):,}-token responses fit {card(runs[FIRST_RUN])} (`{FIRST_RUN}`, pitfall 1).",
        f"- **Training:** {hours:.1f} node hours for {steps} steps (`{MAIN_RUN}`, {attempts} attempt{'s' if attempts != 1 else ''} after {attempts - 1} preemption{'s' if attempts != 2 else ''}).",
        f"- **Readout:** {sum(h for h, _ in evals):.1f} node hours for the base model and the final weights, {evals[0][1]} episodes each, on the same node.",
        "- Hours run from the start of each attempt to its last logged episode (the `utc` stamps of the stored logs); the first set-up "
        "of a node (environments, weights) is not in them."])


def pitfalls_block(d: dict) -> str:
    logs, runs = d["logs"], d["runs"]
    by = {r["run"]: r for r in readouts(d)}
    main, many = by[MAIN_RUN], by[MANY_CELLS_RUN]
    look = lambda run, key, step: [r for r in _of(S._looks(logs[run], "dev"), key) if r["step"] == step]            # noqa: E731
    train = lambda run, key, lo, hi: [r for r in _of(_train(logs[run]), key) if lo <= r["step"] <= hi]              # noqa: E731
    step = min(r["step"] for r in S._looks(logs[LONGER_RUN], "dev"))                    # the weights the longer run started from
    cut = sum("length" in r["outcome"] for r in look(FIRST_RUN, L2, step))
    last = max(r["step"] for r in _train(logs[MAIN_RUN]))
    lo, hi = (round(100 * x) for x in learn.ENTRY_BAND)
    groups = len({r["uid"] for r in _train(logs[MANY_CELLS_RUN]) if r["step"] == 1})
    rest = [k for k in many["trained_cells"] if k != L1]
    dev, secret, dev_step = dev_against_secret(d, main)
    walled = [r for r in by.values() if HINTED in r["trained_cells"] and UNHINTED in r["cells"]][-1]
    clean = [r for r in by.values() if HINTED in r["cells"] and HINTED not in r["trained_cells"]]
    wall = walled["ended"][HINTED]
    recount_looks = _of(S._looks(logs[RECOUNT_RUN], "dev"), HINTED)
    recount = [[r for r in recount_looks if r["step"] == st] for st in (0, max(r["step"] for r in recount_looks))]
    line = line_of(d)["points"]
    late = [next(p for p in reversed(line) if p["run"] == FIRST_RUN)] + [p for p in line if p["run"] != FIRST_RUN]
    ends = [p["ended"][L1][1] for p in late]
    return "\n".join([
        f"1. **The episode token limit.** A limit that is too small hides learning. The step-{step} weights of `{FIRST_RUN}` pass the harder level in "
        f"{frac(look(FIRST_RUN, L2, step))} dev episodes on its {_response_tokens(runs[FIRST_RUN]):,}-token response limit ({cut} cut by length) "
        f"and in {frac(look(LONGER_RUN, L2, step))} on the {_response_tokens(runs[LONGER_RUN]):,}-token limit (run `{LONGER_RUN}`).",
        f"2. **Start where there is signal.** The reward is pass or nothing, so a cell near zero gives no gradient: start where the untrained "
        f"model passes {lo}-{hi} % of the episodes with a pass and a failure in the same group, and keep the next level beside it. In "
        f"`{MAIN_RUN}` the harder level gave {frac(train(MAIN_RUN, L2, 1, 10))} passing rollouts in steps 1-10 and "
        f"{frac(train(MAIN_RUN, L2, last - 9, last))} in steps {last - 9}-{last}, after the first level had gone from "
        f"{frac(train(MAIN_RUN, L1, 1, 10))} to {frac(train(MAIN_RUN, L1, last - 9, last))}.",
        f"3. **Few cells per run.** `{MANY_CELLS_RUN}` trained {len(many['trained_cells'])} cells at {groups} groups per step: on the secret split the "
        f"first level gained {gain(many['cells'][L1]['paired'])} points, the other {len(rest)} "
        + ", ".join(f"{gain(many['cells'][k]['paired'])} ({name(k)})" for k in rest) + ".",
        f"4. **One look is noise, and the dev split flatters.** {repeat_sentence(FIRST_RUN, logs[FIRST_RUN])} The last dev look of `{MAIN_RUN}` "
        f"gave the trained level {dev['passed']} / {dev['episodes']}, its secret split {secret['passed']} / {secret['episodes']}. Judge by the "
        "paired readout on held-out instances, trained against base on the same instances.",
        f"5. **Do not train on a cell while its interface wall stands.** {name(HINTED)} ended {wall[0]['rejected']} of {wall[0]['episodes']} base "
        f"episodes at the rejected-call limit. `{walled['run']}` trained on it and it got worse (passes {wall[0]['passed']} -> {wall[1]['passed']}, "
        f"such endings {wall[0]['rejected']} -> {wall[1]['rejected']}); "
        + words(f"`{r['run']}`" for r in clean) + " trained on the clean cell only and it improved (passes "
        + ", ".join(f"{r['ended'][HINTED][0]['passed']} -> {r['ended'][HINTED][1]['passed']}" for r in clean) + ", such endings "
        + ", ".join(f"{r['ended'][HINTED][0]['rejected']} -> {r['ended'][HINTED][1]['rejected']}" for r in clean) + "). With the accounting corrected "
        f"it still did not train (`{RECOUNT_RUN}`: dev {frac(recount[0])} at step 0, {frac(recount[1])} at its last look).",
        f"6. **Episodes get longer late in training.** On the secret split the adapter of `{FIRST_RUN}` and its continuation `{LONGER_RUN}` passes the "
        f"trained level in " + ", ".join(str(e["passed"]) for e in ends) + f" of {ends[0]['episodes']} episodes at steps {words(p['step'] for p in late)}, "
        "with a median context of " + ", ".join(f"{e['context']:,}" for e in ends) + " tokens and " + ", ".join(str(e["cut"]) for e in ends)
        + " episodes cut by length. Choose the checkpoint by the dev looks, not by the last step."])


def status_block(d: dict) -> str:
    runs = d["runs"]
    main = runs[MAIN_RUN]
    then = f"The recipe first ran on an earlier commit ({runs[FIRST_RUN]['code']}, run `{FIRST_RUN}`)."
    if MAIN_RUN in d["tests"] and main["state"].startswith("finished"):
        steps = max(r["step"] for r in d["tests"][MAIN_RUN]["trained"])
        return (f"Verified on the published code: run `{MAIN_RUN}` at commit {main['code']} ({_node(main)}, {steps} steps, then its own readout). "
                + then)
    return f"Check on the published code (commit {main['code']}, run `{MAIN_RUN}`, {_node(main)}): {main['state']}. " + then


def reach_block(d: dict, prefix: str) -> str:
    cal = d["cals"]["c05"]
    second = [k for k in d["runs"] if k in d["logs"] and any(e == HINTED[0] for e, _ in _cells(_train(d["logs"][k])))]
    by = collections.defaultdict(list)
    for c in cal["cells"]:
        by[learn.band(c["passed"] / c["n_episodes"])].append(c)
    out = []
    for band, lo, hi in reversed(learn.BANDS):
        cells = sorted(by.get(band, []), key=lambda c: (c["env_id"], c["level"]))
        if not cells:
            continue
        if band == "dead":
            shown = f"{len(cells)} cells of {len({c['env_id'] for c in cells})} families"
        else:
            shown = ", ".join(f"{cell(c)} ({c['passed']} / {c['n_episodes']})" for c in cells)
        out.append([band, f"{100 * lo:.0f} - {100 * min(hi, 1.0):.0f} %", shown, learn.BAND_USE[band]])
    chat = next(c for c in cal["cells"] if (c["env_id"], c["level"]) == L1)
    step0 = [r for r in _of(S._looks(d["logs"][MAIN_RUN], "dev"), L1) if r["step"] == 0]
    return "\n".join(table(["Band", "Pass rate of the untrained model", "Cells (calibration `c05`: thinking off, chat harness, "
                            f"{cal['n']} instances x {cal['k']} episodes)", "Use"], out) + [
        "", f"The trainer's token stream is stricter than the chat harness (a reply cut by the token limit ends the episode): {cell(chat)} is "
            f"{chat['passed']} / {chat['n_episodes']} in this calibration and {frac(step0)} at step 0 of `{MAIN_RUN}`. The step-0 evaluation of a "
            f"run is the number that decides. A second family ({HINTED[0].split('/')[1]}) did not train with this recipe in {len(second)} runs: "
            f"[what happened]({prefix}docs/experiments/RESULTS-27b.md#{slug(str(SEC['second']) + '. ' + TITLES['second'])})."])


def blocks(d: dict, prefix: str, front: bool = False) -> dict:
    env, data = settings(), figure_data(d)
    return dict(TRAINING=(front_block if front else training_block)(data, prefix, env["MODEL_HF"]), RECIPE_SETTINGS=settings_block(env, prefix), RECIPE_STATUS=status_block(d),
                RECIPE_NEEDS=needs_block(d, env, prefix), RECIPE_PITFALLS=pitfalls_block(d), RECIPE_REACH=reach_block(d, prefix))


def fill(text: str, bodies: dict, needed=()) -> str:
    """``text`` with every block of ``bodies`` it holds replaced (markers kept, as tools/generate.py writes them)."""
    for block in needed:
        if f"<!-- {block}:START -->" not in text:
            raise SystemExit(f"the page has no {block} block")
    for block, body in bodies.items():
        pat = re.compile(rf"(<!-- {block}:START -->\n).*?(<!-- {block}:END -->)", re.S)
        text = pat.sub(lambda m, body=body: m.group(1) + body.rstrip() + "\n" + m.group(2), text)
    return text


def pages(d: dict) -> dict:
    """{path relative to the repository: (stored text or None, text with fresh blocks)} for the pages of ``PAGES``."""
    out = {}
    for rel, (prefix, needed) in PAGES.items():
        try:
            with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                old = f.read()
        except FileNotFoundError:
            out[rel] = (None, None)
            continue
        out[rel] = (old, fill(old, {k: v for k, v in blocks(d, prefix, front=rel == FRONT).items() if k in needed}, needed))
    return out


def stale(d: dict, text: str) -> list[str]:
    """What differs from the stored result files: the report, the data of the figure, the blocks of the pages."""
    out = []
    try:
        with open(OUT, encoding="utf-8") as f:
            if f.read() != text:
                out.append("docs/experiments/RESULTS-27b.md")
    except FileNotFoundError:
        out.append("docs/experiments/RESULTS-27b.md")
    try:
        with open(SIDECAR, encoding="utf-8") as f:
            same = json.load(f) == figure_data(d) and os.path.getsize(FIGURE) > 0
    except (OSError, ValueError):
        same = False
    if not same:
        out.append("assets/training_27b.png (its data, assets/training_27b.json)")
    out += [rel for rel, (old, new) in pages(d).items() if old is None or old != new]
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    d = stored()
    if "--figure" in argv:
        write_figure(d)
        print(f"wrote {os.path.relpath(FIGURE, ROOT)}: {os.path.getsize(FIGURE) / 1024:.0f} KB, and {os.path.relpath(SIDECAR, ROOT)}")
        return 0
    text = build(d)
    if "--check" in argv:
        old = stale(d, text)
        print("RESULTS-27b.md, the figure and the blocks are current" if not old
              else "stale: " + "; ".join(old) + ": run docs/experiments/27b/report.py")
        return 1 if old else 0
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(text.splitlines())} lines")
    write_figure(d)
    print(f"wrote {os.path.relpath(FIGURE, ROOT)}: {os.path.getsize(FIGURE) / 1024:.0f} KB, and {os.path.relpath(SIDECAR, ROOT)}")
    for rel, (old, new) in pages(d).items():
        if old is None:
            raise SystemExit(f"{rel} does not exist")
        if old != new:
            with open(os.path.join(ROOT, rel), "w", encoding="utf-8") as f:
                f.write(new)
        print(f"{rel}: blocks {', '.join(PAGES[rel][1])} {'written' if old != new else 'unchanged'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
