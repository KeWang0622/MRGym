"""Per-episode log of a training run, the per-step table computed from it, and the stop rules.

    python -m mrgym_verl.stoprules --episodes RUN/records/episodes.jsonl --config stop.json --out metrics.json
    python -m mrgym_verl.stoprules ... --watch --pid-file trainer.pid --stop-file STOP.json --interval 120

The agent loop appends one line per finished episode to ``episodes.jsonl`` (``append``): phase (train / dev / test),
trainer step, group uid, the instance, pass, the trainer's signal, the outcome, whether it submitted, cut replies,
tokens. Everything below is computed from that file, so the rules do not depend on the trainer's console format and
are the same code in the GPU-free simulation and on a node.

Rules (``DEFAULTS``; docs/experiments/PREREG-27b.md section 7 fixes the values of the run). A stop is a result.

    R0  no_reward      mean training pass rate over the first ``no_reward_steps`` complete steps below ``no_reward_min``
    F1  harness        over the first ``harness_steps`` steps: mean training pass rate below half the calibrated base
                       rate, or the share of groups with both a pass and a failure below ``mixed_min``
    F3a not_delivering submit rate below ``submit_min`` over the last ``window`` steps
    F3b cut_replies    share of episodes with a reply cut by the token limit or a full context at least ``cut_max``
                       over the last ``window`` steps
    F3c collapse       training pass rate below half its running maximum for ``collapse_steps`` steps in a row
    F2  futility       sequential test at an evaluation at or after ``futility_from_step``: the one-sided upper bound
                       of (checkpoint - base) on the dev instances, at level 1 - alpha / looks, is below ``mde``
                       (needs at least ``min_look_instances`` paired instances)
    S1  mastered       dev pass rate at least ``mastered`` at two consecutive evaluations (a success; stop early)

The sequential test is ``mrgym.stats.sequential.checkpoint_look``: paired differences of the instance means between
the checkpoint and the base model (the evaluation at the first step of the run, before any update), ``looks``
evaluations declared in advance, each at alpha / looks. ``efficacy`` (lower bound above 0) is recorded at every look
and never stops a run by itself.

Only complete training steps are judged. After a preemption the steps since the last checkpoint are played again;
of several attempts at one step the newest is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time

import numpy as np

from mrgym.stats import intervals, sequential

DEFAULTS = dict(episodes_per_step=128, group_size=8, no_reward_steps=4, no_reward_min=0.02, harness_steps=10, base_rate=None,
                mixed_min=0.40, submit_min=0.50, cut_max=0.20, window=5, collapse_steps=10, alpha=0.05, looks=10, mde=0.05,
                futility_from_step=50, mastered=0.90, min_look_instances=20)
FIELDS = ("phase", "step", "step_source", "attempt", "uid", "session", "env_id", "level", "seed", "split", "passed", "reward", "signal", "signal_mode",
          "outcome", "submitted", "stop_reason", "aborted", "cut_replies", "turns", "tool_calls", "invalid_calls", "interface_tax",
          "tokens_in", "tokens_out", "context_tokens", "wall_s", "utc")


def line(result: dict, **extra) -> dict:
    """The log line of an episode from ``Episode.result()`` and the trainer's own fields (phase, step, uid, ...)."""
    row = {k: result.get(k) for k in FIELDS if k in result}
    row.update(extra, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    row.setdefault("attempt", os.environ.get("MRGYM_ATTEMPT", "0"))
    return row


def append(path: str, row: dict) -> None:
    """One ``write`` on a descriptor opened with O_APPEND: lines of several worker processes do not interleave."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def load(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for text in f:
                try:
                    rows.append(json.loads(text))
                except ValueError:                                        # a line cut by a kill
                    continue
    except OSError:
        pass
    return rows


def _newest_attempt(rows: list[dict]) -> list[dict]:
    """Per (phase, step): the rows of the attempt whose last line is newest."""
    last = {}
    for i, r in enumerate(rows):
        last[(r.get("phase"), r.get("step"), r.get("attempt"))] = i
    keep = {}
    for (phase, step, attempt), i in last.items():
        if (phase, step) not in keep or i > keep[(phase, step)][1]:
            keep[(phase, step)] = (attempt, i)
    return [r for r in rows if keep[(r.get("phase"), r.get("step"))][0] == r.get("attempt")]


def _looks(rows: list[dict], phase: str) -> list[dict]:
    """The evaluation rows of ``phase``: per step those of the newest attempt, and those of an earlier attempt when every
    later attempt that evaluated the step did so on the same weights - it restored the checkpoint of that step (its
    first training step is the next one, or it has not trained yet and the step is the first it evaluated). A restart
    then adds episodes to a look instead of replacing it; a look repeated after more training replaces the older one."""
    order, first_train, first_dev = [], {}, {}
    for r in rows:
        a, step = r.get("attempt"), r.get("step")
        if a not in order:
            order.append(a)
        if isinstance(step, int):
            kind = first_train if r.get("phase") == "train" else first_dev if r.get("phase") == phase else None
            if kind is not None:
                kind[a] = min(kind.get(a, step), step)
    newest = [r for r in _newest_attempt(rows) if r.get("phase") == phase and isinstance(r.get("step"), int)]
    kept = {(r.get("step"), r.get("attempt")) for r in newest}
    restored = lambda b, step: first_train[b] == step + 1 if b in first_train else first_dev.get(b) == step   # noqa: E731
    evaluated: dict = {}
    for r in rows:
        if r.get("phase") == phase and isinstance(r.get("step"), int):
            evaluated.setdefault(r["step"], set()).add(r.get("attempt"))
    for step, attempts in evaluated.items():
        for a in attempts:
            later = [b for b in attempts if order.index(b) > order.index(a)]
            if later and all(restored(b, step) for b in later):
                kept.add((step, a))
    return [r for r in rows if r.get("phase") == phase and isinstance(r.get("step"), int) and (r["step"], r.get("attempt")) in kept]


def step_table(rows: list[dict], cfg: dict) -> list[dict]:
    """One row per complete training step."""
    train = [r for r in _newest_attempt(rows) if r.get("phase") == "train" and isinstance(r.get("step"), int)]
    by_step: dict = {}
    for r in train:
        by_step.setdefault(r["step"], []).append(r)
    last = max(by_step) if by_step else None
    out = []
    for step in sorted(by_step):
        eps = by_step[step]
        if len(eps) < cfg["episodes_per_step"] and step == last:          # still being played
            continue
        groups: dict = {}
        for r in eps:
            groups.setdefault(r.get("uid"), []).append(bool(r["passed"]))
        full = [g for g in groups.values() if len(g) > 1]
        cells: dict = {}
        for r in eps:
            cells.setdefault(f"{r['env_id']}/L{r['level']}", []).append(bool(r["passed"]))
        out.append(dict(step=step, episodes=len(eps), groups=len(groups), pass_rate=float(np.mean([r["passed"] for r in eps])),
                        signal=float(np.mean([r.get("signal") or 0.0 for r in eps])),
                        mixed_share=float(np.mean([0 < sum(g) < len(g) for g in full])) if full else 0.0,
                        submit_rate=float(np.mean([bool(r.get("submitted")) for r in eps])),
                        cut_share=float(np.mean([bool(r.get("cut_replies")) or r.get("aborted") == "length" for r in eps])),
                        interface_tax=float(np.mean([r.get("interface_tax") or 0.0 for r in eps])),
                        tokens_out=float(np.mean([r.get("tokens_out") or 0 for r in eps])),
                        context_tokens_max=max((r.get("context_tokens") or 0 for r in eps), default=0),
                        cells={k: dict(episodes=len(v), pass_rate=float(np.mean(v))) for k, v in sorted(cells.items())}))
    return out


def dev_table(rows: list[dict], cfg: dict, phase: str = "dev") -> list[dict]:
    """One row per evaluation: the clustered pass rate per cell and overall, and the look against the first one."""
    dev = _looks(rows, phase)
    by_step: dict = {}
    for r in dev:
        by_step.setdefault(r["step"], []).append(r)
    if not by_step:
        return []
    inst = lambda r: (r["env_id"], r["level"], r["split"], r["seed"])     # noqa: E731

    def means(eps):
        acc: dict = {}
        for r in eps:
            acc.setdefault(inst(r), []).append(float(bool(r["passed"])))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    base_step = min(by_step)
    base = means(by_step[base_step])
    out = []
    for step in sorted(by_step):
        eps = by_step[step]
        rate = intervals.cluster_rate([float(bool(r["passed"])) for r in eps], [inst(r) for r in eps], [f"{r['env_id']}/L{r['level']}" for r in eps])
        cells: dict = {}
        for r in eps:
            cells.setdefault(f"{r['env_id']}/L{r['level']}", []).append(r)
        row = dict(step=step, episodes=len(eps), instances=rate["instances"], pass_rate=rate["rate"], lo=rate["lo"], hi=rate["hi"],
                   cells={k: (lambda c: dict(episodes=c["episodes"], pass_rate=c["rate"], lo=c["lo"], hi=c["hi"]))(
                       intervals.cluster_rate([float(bool(r["passed"])) for r in v], [inst(r) for r in v])) for k, v in sorted(cells.items())},
                   look=None)
        if step != base_step:
            now = means(eps)
            common = sorted(set(now) & set(base))
            if len(common) > 1:
                row["look"] = dict(base_step=base_step, **sequential.checkpoint_look([now[k] - base[k] for k in common], cfg["alpha"], cfg["looks"], cfg["mde"]))
        out.append(row)
    return out


def decide(rows: list[dict], config: dict | None = None) -> dict:
    """``{stop, rule, reason, steps, dev}`` for the episodes logged so far."""
    cfg = dict(DEFAULTS, **{k: v for k, v in (config or {}).items() if v is not None})
    steps, dev = step_table(rows, cfg), dev_table(rows, cfg)
    stop = lambda rule, reason: dict(stop=True, rule=rule, reason=reason, steps=steps, dev=dev, config=cfg)   # noqa: E731
    rates = [s["pass_rate"] for s in steps]
    k = cfg["no_reward_steps"]
    if len(steps) >= k and float(np.mean(rates[:k])) < cfg["no_reward_min"]:
        return stop("no_reward", f"no reward: mean training pass rate of the first {k} steps is {np.mean(rates[:k]):.4f} "
                                 f"(< {cfg['no_reward_min']}); per step {[round(x, 4) for x in rates[:k]]}")
    k = cfg["harness_steps"]
    if len(steps) >= k:
        first, mixed = float(np.mean(rates[:k])), float(np.mean([s["mixed_share"] for s in steps[:k]]))
        if cfg["base_rate"] is not None and first < 0.5 * cfg["base_rate"]:
            return stop("harness", f"training pass rate of the first {k} steps is {first:.3f}, below half the calibrated base rate "
                                   f"{cfg['base_rate']:.3f}: the training harness differs from the calibration harness")
        if mixed < cfg["mixed_min"]:
            return stop("harness", f"only {100 * mixed:.0f} % of the groups of the first {k} steps hold both a pass and a failure "
                                   f"(< {100 * cfg['mixed_min']:.0f} %)")
    w = cfg["window"]
    if len(steps) >= w:
        submit, cut = float(np.mean([s["submit_rate"] for s in steps[-w:]])), float(np.mean([s["cut_share"] for s in steps[-w:]]))
        if submit < cfg["submit_min"]:
            return stop("not_delivering", f"submit rate of the last {w} steps is {submit:.3f} (< {cfg['submit_min']})")
        if cut >= cfg["cut_max"]:
            return stop("cut_replies", f"{100 * cut:.0f} % of the episodes of the last {w} steps were cut by the token limit or a full "
                                       f"context (>= {100 * cfg['cut_max']:.0f} %)")
    c = cfg["collapse_steps"]
    if len(steps) > c:
        best = max(rates[:-c])
        if best > 0 and all(r < 0.5 * best for r in rates[-c:]):
            return stop("collapse", f"training pass rate below half its earlier maximum {best:.3f} for {c} steps in a row")
    looks = [d for d in dev if d["look"]]
    for d in looks:
        if d["step"] >= cfg["futility_from_step"] and d["look"]["futility"] and d["look"]["instances"] >= cfg["min_look_instances"]:
            return stop("futility", f"evaluation at step {d['step']}: checkpoint minus base = {d['look']['diff']:+.3f}, one-sided upper bound "
                                    f"{d['look']['upper']:+.3f} below the smallest gain worth detecting {cfg['mde']} "
                                    f"(level {d['look']['level']:.4f}, {d['look']['instances']} instances)")
    for a, b in zip(looks, looks[1:]):
        if a["pass_rate"] >= cfg["mastered"] and b["pass_rate"] >= cfg["mastered"]:
            return stop("mastered", f"dev pass rate {a['pass_rate']:.3f} and {b['pass_rate']:.3f} at steps {a['step']} and {b['step']} "
                                    f"(>= {cfg['mastered']}): nothing left to learn on these cells")
    return dict(stop=False, rule=None, reason=None, steps=steps, dev=dev, config=cfg)


def text_table(decision: dict) -> str:
    out = ["step  episodes  pass   signal  mixed  submit  cut    tokens_out  context_max"]
    for s in decision["steps"]:
        out.append(f"{s['step']:>4}  {s['episodes']:>8}  {s['pass_rate']:.3f}  {s['signal']:.3f}   {s['mixed_share']:.2f}   {s['submit_rate']:.2f}    "
                   f"{s['cut_share']:.2f}   {s['tokens_out']:>9.0f}  {s['context_tokens_max']:>10}")
    out.append("evaluation: step  episodes  instances  pass [95 % interval]   look against the first evaluation")
    for d in decision["dev"]:
        look = d["look"]
        tail = "" if not look else (f"   diff {look['diff']:+.3f} bounds {look['lower']:+.3f} .. {look['upper']:+.3f} at level {look['level']:.4f}"
                                    f"  efficacy {look['efficacy']}  futility {look['futility']}")
        out.append(f"            {d['step']:>4}  {d['episodes']:>8}  {d['instances']:>9}  {d['pass_rate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]{tail}")
    out.append(f"decision: {'STOP ' + decision['rule'] + ' - ' + decision['reason'] if decision['stop'] else 'continue'}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--config", default=None, help="JSON file with values that replace DEFAULTS")
    ap.add_argument("--out", default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--pid-file", default=None, help="file with the process-group id of the trainer; it gets SIGTERM on a stop")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--interval", type=float, default=120.0)
    a = ap.parse_args(argv)
    cfg = {}
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = json.load(f)
    while True:
        decision = decide(load(a.episodes), cfg)
        if a.out:
            tmp = a.out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(decision, f, indent=1)
            os.replace(tmp, a.out)
            with open(os.path.splitext(a.out)[0] + ".txt.tmp", "w", encoding="utf-8") as f:
                f.write(text_table(decision) + "\n")
            os.replace(os.path.splitext(a.out)[0] + ".txt.tmp", os.path.splitext(a.out)[0] + ".txt")
        if not a.watch:
            print(text_table(decision))
            return 3 if decision["stop"] else 0
        if decision["stop"]:
            if a.stop_file:
                with open(a.stop_file, "w", encoding="utf-8") as f:
                    json.dump(dict(rule=decision["rule"], reason=decision["reason"], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f)
            if a.pid_file and os.path.exists(a.pid_file):
                with open(a.pid_file, encoding="utf-8") as f:
                    pgid = int(f.read().strip())
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            print(f"STOP {decision['rule']}: {decision['reason']}", flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
