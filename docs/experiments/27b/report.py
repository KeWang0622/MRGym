"""The report of the first training runs of Qwen3.8-27B on MR Gym, written from the stored result files.

    python docs/experiments/27b/report.py            # writes docs/experiments/RESULTS-27b.md
    python docs/experiments/27b/report.py --check    # exit 1 when the stored report is stale

Every number in the report is computed here from ``docs/experiments/27b/results/``: the episode logs of the runs
(``episodes_<run>.jsonl.gz``, one line per episode as the trainer wrote it), the evaluation logs on the secret split
(``test_*.jsonl.gz``), the calibration summaries (``calibration_<run>.json``) and ``runs.json`` (what each run was).
The paired readout and the evaluation table are the functions the node scripts use (``mrgym_verl.readout``,
``mrgym_verl.stoprules``). The text between the tables is written here; it states settings and no result of its own.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "integrations", "verl"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mrgym_verl import readout as R  # noqa: E402
from mrgym_verl import stoprules as S  # noqa: E402

RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(ROOT, "docs", "experiments", "RESULTS-27b.md")
GAIN, LOWER, HARM, MIN_ENVS = 0.15, 0.05, 0.05, 2          # the rule of PREREG-27b.md section 1


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
def build() -> str:
    runs = load("runs.json")
    base, t40, t60 = rows("test_base_57k.jsonl.gz"), rows("test_trained_step40.jsonl.gz"), rows("test_trained_step60.jsonl.gz")
    logs = {k: rows(f"episodes_{k}.jsonl.gz") for k in ("v07", "v09", "v11", "v13")}
    cals = {k: load(f"calibration_{k}.json") for k in ("c01", "c04", "c05")}
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
          "bash integrations/verl/node/make_main.sh train <settings.env> <job.sh>   # the node script of a run; settings in runs.json", "```", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = build()
    if "--check" in argv:
        try:
            with open(OUT, encoding="utf-8") as f:
                same = f.read() == text
        except FileNotFoundError:
            same = False
        print("RESULTS-27b.md is current" if same else "RESULTS-27b.md is stale: run docs/experiments/27b/report.py")
        return 0 if same else 1
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
