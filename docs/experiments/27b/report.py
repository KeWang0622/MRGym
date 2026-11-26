"""The report of the first training runs of Qwen3.8-27B on MR Gym, written from the stored result files.

    python docs/experiments/27b/report.py            # writes docs/experiments/RESULTS-27b.md, the figure, the blocks
    python docs/experiments/27b/report.py --check    # exit 1 when the report, the figure's data or a block is stale
    python docs/experiments/27b/report.py --figure   # redraw assets/training_27b.png and its data file only

Every number in the report is computed here from ``docs/experiments/27b/results/``: the episode logs of the runs
(``episodes_<run>.jsonl.gz``, one line per episode as the trainer wrote it), the evaluation logs on the secret split
(``test_*.jsonl.gz``), the calibration summaries (``calibration_<run>.json``) and ``runs.json`` (what each run was).
The paired readout and the evaluation table are the functions the node scripts use (``mrgym_verl.readout``,
``mrgym_verl.stoprules``). The text between the tables is written here; it states settings and no result of its own.

The same numbers go to the front page. ``assets/training_27b.png`` is drawn from ``figure_data()``, which is stored
beside it as ``assets/training_27b.json``: ``--check`` compares that file with the stored logs, not the image bytes
(``FIGURE_VERSION`` is in it, so a change of the drawing makes the stored figure stale too). The blocks
``<!-- NAME:START -->`` ... ``<!-- NAME:END -->`` of README.md and docs/RECIPE.md listed in ``PAGES`` are written here:
the before / after table, the settings of ``recipes/qwen3.8-27b-nowrap.env``, what the run cost (from the ``utc``
stamps of the logs), the pitfalls with their numbers, and which commit the recipe was run on (from ``runs.json``).
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

from mrgym import learn  # noqa: E402

RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(ROOT, "docs", "experiments", "RESULTS-27b.md")
GAIN, LOWER, HARM, MIN_ENVS = 0.15, 0.05, 0.05, 2          # the rule of PREREG-27b.md section 1
FIGURE = os.path.join(ROOT, "assets", "training_27b.png")
SIDECAR = os.path.join(ROOT, "assets", "training_27b.json")
RECIPE_ENV = os.path.join(ROOT, "recipes", "qwen3.8-27b-nowrap.env")
FIGURE_VERSION = 1                                         # raise it when draw() changes: the stored figure is then stale
MAIN_RUN, SECOND_RUN, LONGER_RUN, CHECK_RUN = "v07", "v09", "v13", "v17"
UNTRAINED = ["Scanner/NullTissue-v0"]
# the cells of the readout in plain words: (short label, label in a table)
PLAIN = {("Scanner/NoWrap-v1", 1): ("trained level", "Trained level"),
         ("Scanner/NoWrap-v1", 2): ("harder level", "Harder level (trained; reward only late)"),
         ("Scanner/NoWrap-v1", 3): ("never-trained level", "Never-trained level"),
         ("Scanner/NullTissue-v0", 0): ("another task family (never trained)", "Another task family (never trained)")}
# page -> (prefix of a link to the repository root, the blocks it must hold)
PAGES = {"README.md": ("", ("TRAINING", "RECIPE_SETTINGS", "RECIPE_STATUS")),
         "docs/RECIPE.md": ("../", ("TRAINING", "RECIPE_NEEDS", "RECIPE_SETTINGS", "RECIPE_PITFALLS", "RECIPE_STATUS", "RECIPE_REACH"))}


def rows(name):
    with gzip.open(os.path.join(RESULTS, name), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def cell(r):
    return f"{r['env_id'].split('/')[1]} L{r['level']}"


def frac(eps):
    return f"{sum(bool(r['passed']) for r in eps)} / {len(eps)}" if eps else "-"


def pts(x):
    return f"{100 * x:+.1f}"


def table(head, lines):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]


# ------------------------------------------------------------------------------------------------ sections
def runs_table(runs):
    return table(["run", "what", "cells", "node", "code", "state when the files were copied"],
                 [[f"`{k}`", r["what"], r["cells"], r["node"], r["code"], r["state"]] for k, r in runs.items()])


def readout_section(base, trained, label):
    r = R.readout(base, trained, ["Scanner/NullTissue-v0"], "test", GAIN, LOWER, HARM, MIN_ENVS)
    out = []
    for c in r["cells"]:
        b, t, p = c["base"], c["trained"], c["paired"]
        out.append([f"{c['env_id'].split('/')[1]} L{c['level']}",
                    f"{round(b['pass_rate'] * b['episodes'])} / {b['episodes']}", f"{round(t['pass_rate'] * t['episodes'])} / {t['episodes']}",
                    f"{pts(p['diff'])} [{pts(p['lo'])}, {pts(p['hi'])}]", p["instances"], f"{p['p']:.4f}" if p["p"] >= 0.0001 else "< 0.0001"])
    lines = table(["cell", "base", label, "paired difference in points [95 %]", "instances", "p"], out)
    env = []
    for e in r["environments"]:
        p = e["paired"]
        env.append([e["env_id"].split("/")[1], e["role"], f"{pts(p['diff'])} [{pts(p['lo'])}, {pts(p['hi'])}]", p["instances"],
                    "yes" if e.get("meets_rule") else "no"])
    lines += [""] + table(["environment (levels as strata)", "role", "paired difference in points [95 %]", "instances",
                           f"gain >= {100 * GAIN:.0f} points and lower bound > {100 * LOWER:.0f}"], env)
    v = r["verdict"]
    lines += ["", f"Rule of the pre-registration: {v['trained_environments_meeting_rule']} trained environment(s) meet it, "
                  f"{v['needed']} are needed; no harm on the environment never trained on: {v['no_harm_on_untrained']}; "
                  f"\"it trained\" by that rule: **{v['it_trained']}**."]
    return lines, r


def outcomes_section(base, t40, t60):
    cells = sorted({(r["env_id"], r["level"]) for r in base})
    names = [n for n, _ in collections.Counter(r["outcome"] for r in base + t40 + t60).most_common()]
    out = []
    for env, lv in cells:
        b, m, e = (collections.Counter(r["outcome"] for r in eps if (r["env_id"], r["level"]) == (env, lv)) for eps in (base, t40, t60))
        for n in names:
            if b[n] + m[n] + e[n]:
                out.append([f"{env.split('/')[1]} L{lv}", f"`{n}`", b[n], m[n], e[n]])
    return table(["cell", "how the episode ended", "base", "step 40", "step 60"], out)


def tokens_section(base, t40, t60):
    out = []
    for env, lv in sorted({(r["env_id"], r["level"]) for r in base}):
        line = [f"{env.split('/')[1]} L{lv}"]
        for eps in (base, t40, t60):
            ctx = sorted(r["context_tokens"] for r in eps if (r["env_id"], r["level"]) == (env, lv))
            cut = sum(1 for r in eps if (r["env_id"], r["level"]) == (env, lv) and "length" in r["outcome"])
            line += [f"{ctx[len(ctx) // 2]:,}", cut]
        out.append(line)
    return table(["cell", "base: median context tokens", "cut by length", "step 40: median", "cut", "step 60: median", "cut"], out)


def dev_section(log):
    looks = S._looks(log, "dev")
    cells = sorted({(r["env_id"], r["level"]) for r in looks})
    steps = sorted({r["step"] for r in looks})
    table_rows = {d["step"]: d for d in S.dev_table(log, dict(S.DEFAULTS))}
    out = []
    for st in steps:
        line = [st] + [frac([r for r in looks if r["step"] == st and (r["env_id"], r["level"]) == c]) for c in cells]
        cut = sum(1 for r in looks if r["step"] == st and "length" in r["outcome"])
        lk = table_rows[st]["look"]
        line += [cut, "-" if not lk else f"{pts(lk['diff'])}"]
        out.append(line)
    return table(["step"] + [f"{e.split('/')[1]} L{lv}" for e, lv in cells] + ["cut by length (all cells)", "all cells against the first look, points"], out)


def train_section(log, block=10):
    train = [r for r in S._newest_attempt(log) if r.get("phase") == "train"]
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
    rej = lambda c: "-" if c is None else c["outcome_counts"].get("no_submission_rejected", 0)                    # noqa: E731
    out = []
    for env, lv in sorted(keys):
        a, b, c = (by[k].get((env, lv)) for k in ("c01", "c04", "c05"))
        out.append([f"{env.split('/')[1]} L{lv}", show(a), show(b), rej(b), show(c), rej(c), f"{c['stream_tokens_est']['p50']:,} / {c['stream_tokens_est']['p90']:,}"])
    return table(["cell", "`c01` reasoning on", "`c04` thinking off", "ended by rejected calls", "`c05` thinking off, corrected interface",
                  "ended by rejected calls", "`c05` stream tokens p50 / p90"], out)


def repeat_sentence(log):
    """What two evaluations of the same weights gave (a run that was preempted right after an evaluation)."""
    looks = S._looks(log, "dev")
    for step in sorted({r["step"] for r in looks}):
        attempts = sorted({r["attempt"] for r in looks if r["step"] == step})
        if step > 0 and len(attempts) > 1:
            c = sorted({(r["env_id"], r["level"]) for r in looks})[0]
            parts = [frac([r for r in looks if r["step"] == step and r["attempt"] == a and (r["env_id"], r["level"]) == c]) for a in attempts]
            return (f"In run `v07` the weights of step {step} were evaluated {len(attempts)} times on the same instances: "
                    f"{' and '.join(parts)} on {c[0].split('/')[1]} level {c[1]}.")
    return ""


# ------------------------------------------------------------------------------------------------ the report
def stored() -> dict:
    """Every stored result file, read once."""
    logs = {k: rows(f"episodes_{k}.jsonl.gz") for k in ("v07", "v09", "v11", "v13")}
    if os.path.exists(os.path.join(RESULTS, f"episodes_{CHECK_RUN}.jsonl.gz")):
        logs[CHECK_RUN] = rows(f"episodes_{CHECK_RUN}.jsonl.gz")
    return dict(runs=load("runs.json"), base=rows("test_base_57k.jsonl.gz"), t40=rows("test_trained_step40.jsonl.gz"),
                t60=rows("test_trained_step60.jsonl.gz"), logs=logs, cals={k: load(f"calibration_{k}.json") for k in ("c01", "c04", "c05")})


def build(d: dict | None = None) -> str:
    d = d or stored()
    runs, base, t40, t60, logs, cals = d["runs"], d["base"], d["t40"], d["t60"], d["logs"], d["cals"]
    r60_lines, r60 = readout_section(base, t60, "after 60 steps")
    r40_lines, _ = readout_section(base, t40, "after 40 steps")
    L = ["<!-- Generated by docs/experiments/27b/report.py from docs/experiments/27b/results/. Do not edit: change the script. -->",
         "# Results: Qwen3.8-27B on MR Gym by reinforcement learning, first runs", "",
         f"Files copied {runs['_copied_utc']}. Some runs were still training then; their tables stop where the copied log stops.", "",
         "**Status: exploratory.** These runs came before the plan of [PREREG-27b.md](PREREG-27b.md) could be followed as written; the "
         "deviations are listed in section 7. One training run with one sampler seed carries the main result, the secret split was read "
         "twice with its weights (both readings are below), and a cell holds 48 secret instances. Read it as the first evidence that the "
         "model learns on the environment, not as the pre-registered result.", "",
         "The model is trained by reinforcement learning only: LoRA (rank 32) with GRPO in verl on episodes played through the "
         "environment's tools, reward 1 when the delivered result passes the verifier and 0 otherwise. No demonstrations, no supervised "
         "step, no reference answers. Thinking is off and a reply holds at most 2,048 tokens, because an episode has to fit one "
         "accelerator as a single token stream (section 6).", "",
         "## 1. Runs", ""] + runs_table({k: v for k, v in runs.items() if not k.startswith("_")}) + ["",
         "## 2. The secret split: trained against base on the same instances", "",
         "Run `v07` trained on NoWrap-v1 levels 1 and 2 only. Its weights after 40 and after 60 steps were evaluated against the base "
         "model on the secret split `test-iid`: 48 instances per cell, 2 episodes each, the same instances for both, the trainer's "
         "harness, an episode limit of about 57,000 tokens (runs `v12`, `v15`; the base half is shared). NoWrap-v1 level 3 and "
         "NullTissue-v0 were never trained on.", "", "### After 60 steps (the final weights)", ""] + r60_lines + ["",
         "### After 40 steps", ""] + r40_lines + ["",
         "### How the episodes ended", ""] + outcomes_section(base, t40, t60) + ["",
         "### Length of the episodes", ""] + tokens_section(base, t40, t60) + ["",
         "What the tables show, and what they do not:", "",
         "- The trained level gains on instances the model never saw, and the gain is larger after 60 steps than after 40.",
         "- Level 2 gains as well. It was a training cell, but it gave almost no passing rollout before step 30 (section 4): the "
         "signal there appeared after level 1 had been learned.",
         "- NullTissue-v0 level 0, another task family, gains without having been trained on. That level is the family's practice "
         "level: its text states the method. What carried over is executing a stated method through the tools; the outcome table "
         "shows fewer wrong deliveries and fewer episodes ended by rejected calls. No conversation of these episodes has been read "
         "yet, so the mechanism is not established.",
         "- Level 3 is at the edge of its interval and is not claimed.",
         "- By the rule of the pre-registration this is \"improved on one environment\": one trained environment, two were required.", "",
         "## 3. Evaluation during training (public dev split)", "",
         "Passes / episodes per cell at each evaluation; 2 episodes per instance. When a run was preempted and restored a checkpoint, "
         "the evaluation of the same weights was repeated and both are pooled here. " + repeat_sentence(logs["v07"]) +
         " A single cell of a single look is a direction, not a number.", ""]
    for k, note in (("v07", "NoWrap-v1 levels 1-2 on an episode limit of about 33,000 tokens: level-2 episodes are cut there, which is why its level-2 column understates section 2."),
                    ("v09", "Four cells at once on the 57,000-token limit, same code as `v07`."),
                    ("v13", "Continues the step-50 weights of `v07` on NoWrap-v1 levels 2 and 3 only, 57,000-token limit; level 1 is watched for forgetting."),
                    ("v11", "NullTissue-v0 level 0 + NoWrap-v1 level 1 on an interim code state in which only the calculator's step limit was raised.")):
        L += [f"### Run `{k}`", "", note, ""] + dev_section(logs[k]) + [""]
    L += ["## 4. Training rollouts", "", "Passes / rollouts per cell in blocks of ten steps (8 groups of 8 rollouts per step; the sampler "
          "chooses the cell of each group).", ""]
    for k in ("v07", "v09", "v13", "v11"):
        L += [f"### Run `{k}`", ""] + train_section(logs[k]) + [""]
    L += ["## 5. Where the untrained model stands", "",
          "Passes / episodes of the base model per cell in the chat harness (`mrgym calibrate`). `c01`: reasoning on (medium effort, "
          "16,384-token replies), an earlier interface; " + c01_length(cals["c01"]) + " `c04`: thinking off, 2,048-token replies. `c05`: the same setting after the interface corrections of section 6 "
          "(twice the instances).", ""] + calibration_section(cals) + ["",
          "- With thinking off the model passes where the task text gives the method (NoWrap-v1 levels 0-1, NullTissue-v0 level 0) "
          "and almost nowhere else; with reasoning on it passes several of the unhinted cells. The tasks are within its reach; the "
          "setting that can be trained is the weak one.",
          "- The corrections moved the cells that the calculator had been blocking, and did not move the unhinted levels: there "
          "the gap is the method.",
          "- ArtefactClinic-v0 in `c01` and `c04` was served from a build that was not thread-safe (section 6); those two columns "
          "are not valid for that family.", "",
          "## 6. What reading the conversations found", "",
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
          "- An episode limit that is too small hides learning: on the 33,000-token limit level 2 looked dead in `v07`, on the "
          "57,000-token limit the same weights pass it (sections 2 and 3).", "",
          "## 7. Deviations from the pre-registration", "",
          "- Thinking off and 2,048-token replies instead of the planned reasoning setting: with reasoning an episode does not fit one accelerator.",
          "- One trained environment (NoWrap-v1) in the run that carries the result; the plan requires two for \"it trained\".",
          "- The readout cells of `v07` did not fit the episode limit of its node and its own readout stopped on its length check; the "
          "readout was made by separate evaluation-only runs on a larger limit, with the same instances and harness.",
          "- The secret split was read at two checkpoints of the same run (steps 40 and 60); both are reported.",
          "- 48 secret instances per cell instead of the planned number; one sampler seed.",
          "- The evaluation table pools repeated looks at the same weights after a restart; the table the job itself printed kept the newer look only.", "",
          "## 8. Limits", "",
          "- One training run; no second seed. `v09` repeats the level-1 gain in direction on other hardware, on the dev split only.",
          "- The transfer to NullTissue-v0 is on its practice level and on the interface before the corrections; on the corrected "
          "interface the untrained model already stands higher in the chat harness (section 5), so the size of a training effect there is open.",
          "- All tasks of the suite are measurement tasks by the suite's own open-loop control; nothing here is evidence about decision tasks.",
          "- Six of the nine families give the thinking-off model no passing episode to learn from.", "",
          "## 9. Reproduce", "",
          "```", "python docs/experiments/27b/report.py --check          # this report from the stored files",
          "python -m mrgym_verl.readout --base <test_base.jsonl> --trained <test_trained.jsonl> --untrained Scanner/NullTissue-v0 --min-envs 2",
          "bash integrations/verl/node/make_main.sh train <settings.env> <job.sh>   # the node script of a run; settings in runs.json", "```", "",
          "The steps with one settings file, what a run costs and the pitfalls: [RECIPE.md](../RECIPE.md).", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ the front page
def _rate(eps):
    c = R._rate(eps)
    return dict(passed=sum(bool(r["passed"]) for r in eps), episodes=len(eps), rate=round(c["pass_rate"], 4), lo=round(c["lo"], 4), hi=round(c["hi"], 4))


def _of(eps, key):
    return [r for r in eps if (r["env_id"], r["level"]) == key]


def _curve(log, key):
    looks = S._looks(log, "dev")
    at = lambda st: [r for r in _of(looks, key) if r["step"] == st]                                  # noqa: E731
    return [dict(step=st, cut=sum("length" in r["outcome"] for r in at(st)), **_rate(at(st))) for st in sorted({r["step"] for r in _of(looks, key)})]


def _response_tokens(run):
    """The response limit of a run, from its ``node`` line in runs.json (``... + 24,576 response tokens``)."""
    return int(re.search(r"\+ ([\d,]+)", run["node"]).group(1).replace(",", ""))


def _node(run):
    return run["node"].split(";")[0]


def figure_data(d: dict) -> dict:
    """Everything the figure and the before / after table show, from the stored logs (rounded, so that the file is the
    same on every machine)."""
    test = lambda eps: [r for r in S._newest_attempt(eps) if r.get("phase") == "test"]              # noqa: E731
    base, t60 = test(d["base"]), test(d["t60"])
    r = R.readout(base, t60, UNTRAINED, "test", GAIN, LOWER, HARM, MIN_ENVS)
    secret = []
    for c in r["cells"]:
        key, p = (c["env_id"], c["level"]), c["paired"]
        secret.append(dict(cell=cell(dict(env_id=key[0], level=key[1])), label=PLAIN[key][0], row=PLAIN[key][1], before=_rate(_of(base, key)),
                           after=_rate(_of(t60, key)), instances=p["instances"], diff=round(p["diff"], 4), lo=round(p["lo"], 4), hi=round(p["hi"], 4)))
    l1, l2 = ("Scanner/NoWrap-v1", 1), ("Scanner/NoWrap-v1", 2)
    curve = lambda run, key: dict(run=run, cell=cell(dict(env_id=key[0], level=key[1])), label=PLAIN[key][0], points=_curve(d["logs"][run], key))   # noqa: E731
    v = r["verdict"]
    return dict(schema="mrgym-27b-figure/1", version=FIGURE_VERSION, steps=max(r["step"] for r in t60), secret=secret,
                rule=dict(met=v["trained_environments_meeting_rule"], needed=v["needed"]),
                response_tokens=dict(training=_response_tokens(d["runs"][MAIN_RUN]), readout=_response_tokens(d["runs"]["v15"])),
                dev=[curve(MAIN_RUN, l1), curve(MAIN_RUN, l2)], second=curve(SECOND_RUN, l1))


BLUE, ORANGE = "#1c5cb5", "#e07b1a"                     # distinct for colour-blind readers and by lightness in greyscale
INK, INK2, MUTED, GRID = "#0b0b0b", "#4a4946", "#807e78", "#e1e0d9"


def draw(data: dict, path: str) -> None:
    """The figure of the front page: left, before and after on the secret split with 95 % intervals; right, the public
    dev instances during training. Two colours; every series is labelled where it is drawn."""
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
    fig.text(0.02, 0.855, f"{one['instances']} instances x {one['before']['episodes'] // one['instances']} episodes per row, the same for both; "
             f"lines: 95 % intervals", fontsize=8, color=INK2, va="center")
    # ---- right: the public dev instances during training
    bx = fig.add_axes([0.575, 0.15, 0.30, 0.60])
    for y in (0, 25, 50, 75, 100):
        bx.axhline(y, color=GRID, lw=0.6, zorder=0)
    sec = data["second"]["points"]
    bx.plot([p["step"] for p in sec], [100 * p["rate"] for p in sec], color=BLUE, lw=1.0, zorder=2)
    for series, colour, marker in zip(data["dev"], (BLUE, ORANGE), ("o", "s")):
        pts = series["points"]
        bx.plot([p["step"] for p in pts], [100 * p["rate"] for p in pts], color=colour, lw=2.0, marker=marker, ms=6.5, mec="white", mew=1.4,
                solid_joinstyle="round", zorder=3, clip_on=False)
        bx.text(pts[-1]["step"] + 2.2, 100 * pts[-1]["rate"], series["label"], va="center", ha="left", fontsize=9, color=INK)
        bx.text(pts[-1]["step"] + 2.2, 100 * pts[-1]["rate"] - 8.5, pct(pts[-1]["rate"]), va="center", ha="left", fontsize=8.5, color=INK2)
    mid = sec[len(sec) // 2]
    bx.annotate(f"second run ({data['second']['run']}): thin line", xy=(mid["step"], 100 * mid["rate"]), xytext=(mid["step"] + 3, 100 * mid["rate"] - 26),
                fontsize=8, color=INK2, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=2))
    last = max(p["step"] for s in data["dev"] for p in s["points"])
    bx.set_xlim(-2, last + 1); bx.set_ylim(-4, 104)
    bx.set_xticks(sorted({p["step"] for p in data["dev"][0]["points"]}))
    bx.set_yticks([0, 25, 50, 75, 100]); bx.set_yticklabels(["0", "25", "50", "75", "100 %"])
    for side in ("top", "right", "left", "bottom"):
        bx.spines[side].set_visible(False)
    bx.tick_params(length=0)
    bx.set_xlabel("training step", fontsize=8.5, color=INK2)
    hard, per = data["dev"][1]["points"], sorted({p["episodes"] for p in data["dev"][0]["points"]})
    limits = data["response_tokens"]
    bx.text(1, 16.5, f"{sum(p['cut'] for p in hard)} of {sum(p['episodes'] for p in hard)} harder-level episodes were cut by\nthis run's "
            f"{limits['training']:,}-token limit (left: {limits['readout']:,})", fontsize=7.5, color=MUTED, ha="left", va="center", linespacing=1.25)
    fig.text(0.535, 0.93, "During training, on unseen public instances", fontsize=10.5, fontweight="bold", color=INK, va="center")
    fig.text(0.535, 0.855, f"run {data['dev'][0]['run']}; {' or '.join(map(str, per))} episodes per point", fontsize=8, color=INK2, va="center")
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
            words = shlex.split(line, comments=True)
            words = words[1:] if words[:1] == ["export"] else words
            if len(words) == 1 and re.match(r"[A-Za-z_][A-Za-z0-9_]*=", words[0]):
                name, _, value = words[0].partition("=")
                out[name] = value
    return out


def _attempts(log):
    """(attempts, node hours): per attempt from its start (the stamp in its id) to its last logged episode."""
    stamp = lambda s, fmt: datetime.datetime.strptime(s, fmt)                                       # noqa: E731
    last = {}
    for r in log:
        last[r["attempt"]] = max(last.get(r["attempt"], r["utc"]), r["utc"])
    seconds = sum((stamp(u, "%Y-%m-%dT%H:%M:%SZ") - stamp(a.split("-")[0], "%Y%m%dT%H%M%SZ")).total_seconds() for a, u in last.items())
    return len(last), seconds / 3600.0


def training_block(data: dict, prefix: str, model: str) -> str:
    name = model.split("/")[-1]
    out = [f"**{name}, reinforcement learning only, {data['steps']} steps: tested on instances it never saw.**", "",
           '<p align="center">',
           f'  <img src="{prefix}assets/training_27b.png" width="900" alt="Pass rate of {name} before and after {data["steps"]} steps of '
           f'reinforcement learning on {len(data["secret"])} cells of the secret split, with 95 % intervals, and on public instances during training">',
           "</p>", ""]
    show = lambda x: f"{100 * x['rate']:.0f} % ({x['passed']} / {x['episodes']})"                   # noqa: E731
    out += table(["On secret test instances", "Before training", f"After {data['steps']} steps", "Gain in points [95 % interval]"],
                 [[f"{c['row']} · `{c['cell']}`", show(c["before"]), show(c["after"]), f"{pts(c['diff'])} [{pts(c['lo'])}, {pts(c['hi'])}]"]
                  for c in data["secret"]])
    one = data["secret"][0]
    out += ["", f"Evidence: one run, one seed, {one['instances']} secret instances x {one['before']['episodes'] // one['instances']} episodes per "
                f"row, base and trained on the same instances; exploratory (the pre-registered rule needs {data['rule']['needed']} trained "
                f"environments, {data['rule']['met']} was trained). The other family's row is a practice level whose text states the method. "
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
    card = lambda run: (lambda name: f"{_node(run)} ({gb[name]} GB each)")(_node(run).split()[-1])  # noqa: E731
    attempts, hours = _attempts(d["logs"][MAIN_RUN])
    steps = max(r["step"] for r in d["logs"][MAIN_RUN] if r["phase"] == "train")
    evals = [(_attempts(d[k])[1], len(d[k])) for k in ("base", "t60")]
    return "\n".join([
        f"- **Model:** `{env['MODEL_HF']}`, open weights. The trainer is [verl]({prefix}integrations/verl/README.md) at its pinned commit.",
        f"- **One node with 8 accelerators.** Run `{MAIN_RUN}` trained on {card(runs[MAIN_RUN])} with {_response_tokens(runs[MAIN_RUN]):,}-token "
        f"responses; the settings of this page ({int(env['MAX_RESPONSE_LEN']):,}-token responses) run on {card(runs[CHECK_RUN])}.",
        f"- **Training:** {hours:.1f} node hours for {steps} steps (`{MAIN_RUN}`, {attempts} attempts after {attempts - 1} preemptions).",
        f"- **Readout:** {sum(h for h, _ in evals):.1f} node hours for the base model and the final weights, {evals[0][1]} episodes each, on "
        f"{_node(runs['v15'])}.",
        "- Hours run from the start of each attempt to its last logged episode (the `utc` stamps of the stored logs); the first set-up "
        "of a node (environments, weights) is not in them."])


def pitfalls_block(d: dict) -> str:
    logs, runs = d["logs"], d["runs"]
    l1, l2 = ("Scanner/NoWrap-v1", 1), ("Scanner/NoWrap-v1", 2)
    look = lambda run, key, step: [r for r in _of(S._looks(logs[run], "dev"), key) if r["step"] == step]            # noqa: E731
    train = lambda run, key, lo, hi: [r for r in _of(S._newest_attempt(logs[run]), key) if r["phase"] == "train" and lo <= r["step"] <= hi]   # noqa: E731
    step = min(r["step"] for r in S._looks(logs[LONGER_RUN], "dev"))                    # the weights the longer run started from
    cut = sum("length" in r["outcome"] for r in look(MAIN_RUN, l2, step))
    last = max(r["step"] for r in logs[MAIN_RUN] if r["phase"] == "train")
    lo, hi = (round(100 * x) for x in learn.ENTRY_BAND)
    looks9 = S._looks(logs[SECOND_RUN], "dev")
    end9 = max(r["step"] for r in looks9)
    others = [k for k in sorted({(r["env_id"], r["level"]) for r in looks9}) if k != l1]
    groups = len({r["uid"] for r in S._newest_attempt(logs[SECOND_RUN]) if r["phase"] == "train" and r["step"] == 1})
    stay = ", ".join(f"{cell(dict(env_id=k[0], level=k[1]))} {frac(look(SECOND_RUN, k, 0))} then {frac(look(SECOND_RUN, k, end9))}" for k in others)
    return "\n".join([
        f"1. **The episode token limit.** A limit that is too small hides learning. The step-{step} weights pass the harder level in "
        f"{frac(look(MAIN_RUN, l2, step))} dev episodes on the {_response_tokens(runs[MAIN_RUN]):,}-token response limit ({cut} cut by length, "
        f"run `{MAIN_RUN}`) and in {frac(look(LONGER_RUN, l2, step))} on the {_response_tokens(runs[LONGER_RUN]):,}-token limit (run `{LONGER_RUN}`).",
        f"2. **Start where there is signal.** The reward is pass or nothing, so a cell near zero gives no gradient: start where the untrained "
        f"model passes {lo}-{hi} % of the episodes with a pass and a failure in the same group, and keep the next level beside it. In "
        f"`{MAIN_RUN}` the harder level gave {frac(train(MAIN_RUN, l2, 1, 10))} passing rollouts in steps 1-10 and "
        f"{frac(train(MAIN_RUN, l2, last - 9, last))} in steps {last - 9}-{last}, after the first level had gone from "
        f"{frac(train(MAIN_RUN, l1, 1, 10))} to {frac(train(MAIN_RUN, l1, last - 9, last))}.",
        f"3. **Few cells per run.** `{SECOND_RUN}` trained {len(others) + 1} cells at {groups} groups per step: on dev the first level went from {frac(look(SECOND_RUN, l1, 0))} to {frac(look(SECOND_RUN, l1, end9))} by step {end9}, "
        f"the others stayed near their start ({stay}).",
        f"4. **One look is noise.** {repeat_sentence(logs[MAIN_RUN])} Judge by the paired readout on held-out instances, trained against "
        "base on the same instances, not by one evaluation."])


def status_block(d: dict) -> str:
    runs = d["runs"]
    state = runs[CHECK_RUN]["state"].replace(" in this report", " yet")
    out = (f"Verified on commit {runs[MAIN_RUN]['code']} (run `{MAIN_RUN}`, {_node(runs[MAIN_RUN])} with {_response_tokens(runs[MAIN_RUN]):,}-token "
           f"responses; readout on {_node(runs['v15'])} at {_response_tokens(runs['v15']):,}). "
           f"Check on the current code (commit {runs[CHECK_RUN]['code']}, run `{CHECK_RUN}`, {_node(runs[CHECK_RUN])}): {state}.")
    log = d["logs"].get(CHECK_RUN)
    if log:
        looks = S._looks(log, "dev")
        step = max(r["step"] for r in looks)
        out += f" Its newest evaluation on public instances (step {step}): " + ", ".join(
            f"{cell(dict(env_id=k[0], level=k[1]))} {frac([r for r in _of(looks, k) if r['step'] == step])}"
            for k in sorted({(r["env_id"], r["level"]) for r in looks})) + "."
    return out


def reach_block(d: dict) -> str:
    cal = d["cals"]["c05"]
    by = collections.defaultdict(list)
    for c in cal["cells"]:
        by[learn.band(c["passed"] / c["n_episodes"])].append(c)
    out = []
    for name, lo, hi in reversed(learn.BANDS):
        cells = sorted(by.get(name, []), key=lambda c: (c["env_id"], c["level"]))
        if not cells:
            continue
        if name == "dead":
            shown = f"{len(cells)} cells of {len({c['env_id'] for c in cells})} families"
        else:
            shown = ", ".join(f"{cell(c)} ({c['passed']} / {c['n_episodes']})" for c in cells)
        out.append([name, f"{100 * lo:.0f} - {100 * min(hi, 1.0):.0f} %", shown, learn.BAND_USE[name]])
    first = ("Scanner/NoWrap-v1", 1)
    chat = next(c for c in cal["cells"] if (c["env_id"], c["level"]) == first)
    step0 = [r for r in _of(S._looks(d["logs"][MAIN_RUN], "dev"), first) if r["step"] == 0]
    return "\n".join(table(["Band", "Pass rate of the untrained model", "Cells (calibration `c05`: thinking off, chat harness, "
                            f"{cal['n']} instances x {cal['k']} episodes)", "Use"], out) + [
        "", f"The trainer's token stream is stricter than the chat harness (a reply cut by the token limit ends the episode), and `{MAIN_RUN}` ran "
            f"on an earlier interface: {cell(chat)} is {chat['passed']} / {chat['n_episodes']} in this calibration and was {frac(step0)} at step 0 of "
            f"`{MAIN_RUN}`. The step-0 evaluation of a run is the number that decides."])


def blocks(d: dict, prefix: str) -> dict:
    env, data = settings(), figure_data(d)
    return dict(TRAINING=training_block(data, prefix, env["MODEL_HF"]), RECIPE_SETTINGS=settings_block(env, prefix), RECIPE_STATUS=status_block(d),
                RECIPE_NEEDS=needs_block(d, env, prefix), RECIPE_PITFALLS=pitfalls_block(d), RECIPE_REACH=reach_block(d))


def fill(text: str, bodies: dict, needed=()) -> str:
    """``text`` with every block of ``bodies`` it holds replaced (markers kept, as tools/generate.py writes them)."""
    for name in needed:
        if f"<!-- {name}:START -->" not in text:
            raise SystemExit(f"the page has no {name} block")
    for name, body in bodies.items():
        pat = re.compile(rf"(<!-- {name}:START -->\n).*?(<!-- {name}:END -->)", re.S)
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
        out[rel] = (old, fill(old, blocks(d, prefix), needed))
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
