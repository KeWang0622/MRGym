"""Replay the ``calc`` calls of recorded episodes on the calculator of this tree.

    python tools/audit/calc_corpus.py CORPUS.jsonl                       refused before -> after by class, wall times
    python tools/audit/calc_corpus.py CORPUS.jsonl --limit 40000000      the same with another step limit (what the stopped programs need)
    python tools/audit/calc_corpus.py CORPUS.jsonl --every 10            every tenth episode (a quick look)
    python tools/audit/calc_corpus.py CORPUS.jsonl --write-sample tests/data/calc_corpus_sample.jsonl
    python tools/audit/calc_corpus.py --messages 'runs/.../e*/messages.json'   the same on episode directories of a run
    python tools/audit/calc_corpus.py --refresh-sample tests/data/calc_corpus_sample.jsonl    replay the sample, store what it gives
    python tools/audit/calc_corpus.py --throughput                       steps per second of this machine on three fixed programs

``CORPUS.jsonl``: one line per recorded episode, ``{"episode", "env_id", "stop_reason", "calc_calls": [{"arguments":
the raw tool arguments as the model wrote them, "observation", "refused", "step_limit"}]}``. The environment keeps one
names dictionary per episode, so the calls of an episode are replayed in order with one dictionary.

What a replay cannot know: what the model would have written next after a call that is now accepted. A later call of
the episode is replayed as recorded, also when it was written as a repair of a refusal that no longer happens.

Single thread, no clock inside the calculator: the outcome of every call is the same on every machine; only the wall
times of this report are not.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from mrgym import calc as C  # noqa: E402

SLIPS = ("arguments", "not Python", "unknown name", "agent: ")           # classes that are the agent's own slips


REJECTED = re.compile(r"rejected (\d+)/(\d+)")


def from_messages(pattern: str) -> list:
    """Episodes in the corpus form from ``messages.json`` files of a run (their ``record.json`` gives the stop reason)."""
    episodes = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            doc = json.load(f)
        results = {m.get("tool_call_id"): m.get("content") or "" for m in doc["messages"] if m.get("role") == "tool"}
        calls, rejected = [], 0
        for m in doc["messages"]:
            for call in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
                seen = results.get(call.get("id"), "")
                rejected = max([rejected] + [int(k) for k, _ in REJECTED.findall(seen)])
                if call["function"]["name"] == "calc":
                    calls.append(dict(arguments=call["function"]["arguments"], observation=seen[:1200], refused=seen.startswith("error"),
                                      step_limit="stopped after" in seen))
        record = {}
        if os.path.exists(os.path.join(os.path.dirname(path), "record.json")):
            with open(os.path.join(os.path.dirname(path), "record.json")) as f:
                record = json.load(f)
        episodes.append(dict(episode=os.path.relpath(os.path.dirname(path), os.path.commonpath(glob.glob(pattern)) if len(glob.glob(pattern)) > 1
                                                     else os.path.dirname(os.path.dirname(path))),
                             env_id=doc.get("env_id") or record.get("env_id"), stop_reason=record.get("stop_reason"), calc_calls=calls,
                             rejected_total=(record.get("interface") or {}).get("rejected", rejected)))
    return episodes


def rejected_recorded(episode) -> int:
    """Rejected calls of the recorded episode: the record's count, else the last count a calc result printed (the
    limit itself when the episode ended at it)."""
    if episode.get("rejected_total") is not None:
        return int(episode["rejected_total"])
    seen = [int(k) for c in episode["calc_calls"] for k, _ in REJECTED.findall(c["observation"])]
    return REJECTED_LIMIT if episode.get("stop_reason") == "rejected_call_limit" else max(seen, default=0)


REJECTED_LIMIT = 8                                                       # of the recorded episodes (the environment's, not calc's)


def text_of(arguments: str):
    """The calculator text of a recorded call as the environment hands it over, or None when the call has none."""
    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except ValueError:
        return None
    if not isinstance(args, dict) or "expressions" not in args:
        return None
    text = args["expressions"]
    if isinstance(text, list) and all(isinstance(x, str) for x in text):          # the interface joins a list of lines
        text = "\n".join(text)
    return text if isinstance(text, str) else None


def norm(message: str) -> str:
    message = re.sub(r"'[^']*'|\"[^\"]*\"", "'..'", message)
    message = re.sub(r"-?\d+(\.\d+)?(e[-+]?\d+)?", "N", message)
    return message


# Refusals of a value that are the text's own mistake (Python refuses the same text), by the start of the message.
AGENT = (
    "index N is out of range", "index out of range", "none has no", "arithmetic with none", "cannot loop over none", "a number is needed here, not none",
    "N names on the left", "no key", "an index is a whole number", "a slice takes whole numbers", "len of", "cannot loop over a number",
    "a number has no elements", "min / max of an empty", "the text has N % place", "arithmetic needs lists of one length",
    "arithmetic needs a list of numbers of one length", "text takes no arithmetic", "a number is needed here, not text",
    "a number is needed here, not a dictionary", "a number is needed here, not a function", "functions nested more than", "assert failed",
    "none and", "a list and", "a number and", "text and", "none has no order", "pop from an empty", "a tuple cannot be changed",
    "a list of comparisons is neither", "true / false has no elements", "a function has no elements", "a list has no elements",
    "mean of an empty", "only an element of", "a key is text or a number", "range takes one to three whole numbers", "complex numbers have no order",
    "%", "the format", "interp(x, xs, ys) needs", "polyfit", "brentq(f, a, b) needs f(a) and f(b)", "dot: the shapes",
    "solve(A, b)", "curve_fit", "index: ", "remove: ", "`in` needs", "value N for %", "a range of more than N numbers can only", "a list of more than",
    "a grid of more than", "a range of more than", "these lists do not fit together", "nothing to stack", "the number of repeats",
    "these values cannot be ordered", "fsolve did not converge", "least_squares", "minimize", "interp(x, xp, fp", "text * n repeats",
    "the result has more than", "argmin / argmax of an empty", "N unknowns need", "** takes a dictionary",
)


def classify(error) -> str:
    """The class of a refusal: whose slip it is, and what it is."""
    if error is None:
        return "ok"
    if isinstance(error, SyntaxError):
        return "not Python"
    kind = getattr(error, "kind", "value")
    message = norm(str(error))
    if isinstance(error, C._Stop):
        return "step limit"
    if kind == "unknown":
        return "unknown name"
    if kind == "unsupported":
        return "calc has no: " + message[:44]
    called = re.match(r"^([\w.]+|the function) takes N argument|^[\w.]+ has no argument|^[\w.]+: \w+ is given twice|^\S+ is .*, not a function", message)
    if called or any(message.startswith(a) for a in AGENT) or re.match(r"^[\w.\[\]'() ]+: ", message):
        return "agent: " + message[:44]
    return "value: " + message[:44]


def probe(runs: list):
    """A ``calc._Run`` that records itself in ``runs``: how many steps a call took is read from it."""
    class Probe(C._Run):
        __slots__ = ()

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            runs.append(self)
    return Probe


def replay(episodes, limit=None, progress=True):
    """Replay every call. Returns one record per call: episode index, call index, recorded flags, class, seconds, steps."""
    runs = []
    saved = C._Run, C.MAX_STEPS
    C._Run = probe(runs)
    if limit:
        C.MAX_STEPS = int(limit)
    out, t_start = [], time.perf_counter()
    try:
        for e, episode in enumerate(episodes):
            names, in_step = {}, True                                    # in_step: every call so far ended as it did in the recording
            for k, call in enumerate(episode["calc_calls"]):
                text = text_of(call["arguments"])
                error, shown, steps = None, None, 0
                t0 = time.perf_counter()
                if text is None:
                    cls = "arguments"
                else:
                    del runs[:]
                    try:
                        shown = C.render(C.calc(text, names))
                    except C._CALC_ERRORS as err:
                        error = err
                    cls = classify(error)
                    steps = runs[-1].steps if runs else 0
                kind = "ok" if cls == "ok" else cls if cls in ("arguments", "not Python", "step limit") else getattr(error, "kind", "value")
                seconds = time.perf_counter() - t0
                in_step = in_step and (cls == "ok") == (not call["refused"])
                same = None
                if in_step and cls == "ok":                              # the same names as in the recording: the same answer is due
                    same = same_answer(call["observation"], shown)
                out.append(dict(episode=e, call=k, env_id=episode["env_id"], was_refused=bool(call["refused"]), was_limit=bool(call["step_limit"]),
                                cls=cls, message=None if error is None else str(error)[:200], line=getattr(error, "line", None),
                                seconds=seconds, steps=steps, text=text, shown=shown, same=same, kind=kind))
            if progress and (e + 1) % 50 == 0:
                refused = sum(r["cls"] != "ok" for r in out)
                print(f"  ... {e + 1}/{len(episodes)} episodes, {len(out)} calls, {refused} refused, {time.perf_counter() - t_start:.0f} s", flush=True)
    finally:
        C._Run, C.MAX_STEPS = saved
    return out


def same_answer(observation: str, shown: str) -> bool:
    """Whether the recorded result of an accepted call is the text this tree shows (the recording keeps the first
    1,200 characters and ends with the status line of the environment)."""
    recorded = observation
    if recorded.startswith("[") and "\n" not in recorded:
        recorded = ""                                                    # nothing but the status line: an empty print
    elif "\n[" in recorded[-120:]:
        recorded = recorded[:recorded.rfind("\n[")]
    if recorded in ("", "calc: nothing to evaluate"):
        return not shown
    tidy = lambda text: "\n".join(line.rstrip() for line in text.rstrip().split("\n"))     # noqa: E731  the environment trims line ends
    recorded, shown = tidy(recorded), tidy(shown or "")
    n = min(len(recorded), 1100)
    return recorded[:n] == shown[:n] if len(observation) >= 1150 else recorded == shown


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else 0.0


def report(episodes, records, limit) -> dict:
    n = len(records)
    before = sum(r["was_refused"] for r in records)
    after = [r for r in records if r["cls"] != "ok"]
    print(f"\ncalls {n} in {len(episodes)} episodes; step limit {limit or C.MAX_STEPS}")
    print(f"refused before {before} ({100 * before / n:.1f} %), of them {sum(r['was_limit'] for r in records)} by the old step limit")
    print(f"refused after  {len(after)} ({100 * len(after) / n:.1f} %)")
    classes = collections.Counter(r["cls"] for r in after)
    slips = sum(v for k, v in classes.items() if k.startswith(SLIPS))
    print(f"  of them the agent's own slips (no arguments, not Python, a name never assigned, a wrong index or value): {slips} "
          f"({100 * slips / max(1, len(after)):.0f} %)")
    groups = collections.Counter("agent: a wrong value / index / shape (Python refuses it too)" if k.startswith("agent: ") else
                                 "calc has no ..." if k.startswith("calc has no") else "value: other" if k.startswith("value: ") else k
                                 for k in classes.elements())
    for k, v in groups.most_common():
        print(f"  {v:5d}  {k}")
    print("  by message:")
    for k, v in classes.most_common(45):
        print(f"  {v:5d}  {k}")
    kinds = collections.Counter(r["kind"] for r in after)
    print("  by kind (CalcError.kind; arguments / not Python never reach the calculator; the step limit is of kind value):")
    for k, v in kinds.most_common():
        print(f"  {v:5d}  {k}")
    limited = [r for r in records if r["was_limit"]]
    done = sum(r["cls"] == "ok" for r in limited)
    still = sum(r["cls"] == "step limit" for r in limited)
    other = len(limited) - done - still
    only = len(limited) - other
    print(f"\nprograms the old limit stopped: {len(limited)}; now finish {done}, stopped by the new limit {still}, refused for another reason {other}")
    print(f"  of those stopped only by a limit ({only}): {100 * done / max(1, only):.1f} % finish")
    seconds = [r["seconds"] for r in records]
    worst = max(records, key=lambda r: r["seconds"])
    print(f"\nwall seconds per call: p50 {quantile(seconds, 0.5):.5f}  p90 {quantile(seconds, 0.9):.4f}  p99 {quantile(seconds, 0.99):.3f}  "
          f"max {worst['seconds']:.2f}  total {sum(seconds):.0f}")
    print(f"  the worst call: episode {episodes[worst['episode']]['episode']} call {worst['call']}: {worst['cls']}, {worst['steps']} steps")
    full = [r for r in records if r["cls"] == "step limit"]
    if full:
        print(f"  calls that ran to the limit: {len(full)}, wall p50 {quantile([r['seconds'] for r in full], 0.5):.2f} s, "
              f"max {max(r['seconds'] for r in full):.2f} s")
    steps = [r["steps"] for r in limited if r["cls"] in ("ok", "step limit")]
    if steps:
        print("  steps of the programs the old limit stopped: " + "  ".join(f"p{int(100 * q)} {quantile(steps, q):,}" for q in (0.5, 0.75, 0.9, 0.95, 0.99))
              + f"  max {max(steps):,}")
    # episodes the rejected-call limit ended
    print("\nepisodes ended by the rejected-call limit (recorded), and what the replay can say about them:")
    print(f"  {'family':28s} {'ended':>5s} {'>= 1 calc refusal lifted':>25s} {'every calc refusal lifted':>26s}")
    by_episode = collections.defaultdict(list)
    for r in records:
        by_episode[r["episode"]].append(r)
    table = collections.defaultdict(lambda: [0, 0, 0])
    for e, episode in enumerate(episodes):
        if episode.get("stop_reason") != "rejected_call_limit":
            continue
        rs = by_episode.get(e, [])
        was = sum(r["was_refused"] for r in rs)
        now = sum(r["cls"] != "ok" for r in rs)
        row = table[episode["env_id"]]
        row[0] += 1
        row[1] += now < was
        row[2] += was > 0 and now == 0
    for family, row in sorted(table.items()):
        print(f"  {family:28s} {row[0]:5d} {row[1]:25d} {row[2]:26d}")
    total = [sum(row[i] for row in table.values()) for i in range(3)]
    print(f"  {'all':28s} {total[0]:5d} {total[1]:25d} {total[2]:26d}")
    print("  (an episode with one calc refusal fewer would not have reached its eighth rejected call where it did; what the model\n"
          "   would have written after an accepted call is not known to a replay)")
    # how many episodes would still count eight rejected calls: the recorded count, less the calc refusals the replay lifts
    reach = collections.defaultdict(lambda: [0, 0, 0])
    for e, episode in enumerate(episodes):
        rs = by_episode.get(e, [])
        recorded = rejected_recorded(episode)
        after_ = recorded - sum(r["was_refused"] for r in rs) + sum(r["cls"] != "ok" for r in rs)
        row = reach[episode["env_id"]]
        row[0] += 1
        row[1] += recorded >= REJECTED_LIMIT
        row[2] += after_ >= REJECTED_LIMIT
    print(f"\nepisodes with {REJECTED_LIMIT} or more rejected calls: recorded, and with the calc refusals of the replay in place of the recorded ones;")
    print(f"episodes with {REJECTED_LIMIT} or more refused calc calls alone: recorded and replay; replay without the refusals of kind unsupported")
    print(f"  {'family':28s} {'episodes':>8s} {'recorded':>9s} {'replay':>7s} {'calc rec.':>10s} {'calc replay':>12s} {'w/o unsupported':>16s}")
    for e, episode in enumerate(episodes):
        rs = by_episode.get(e, [])
        row = reach[episode["env_id"]]
        while len(row) < 6:
            row.append(0)
        row[3] += sum(r["was_refused"] for r in rs) >= REJECTED_LIMIT
        row[4] += sum(r["cls"] != "ok" for r in rs) >= REJECTED_LIMIT
        row[5] += sum(r["cls"] != "ok" and r["kind"] != "unsupported" for r in rs) >= REJECTED_LIMIT
    for family, row in sorted(reach.items()):
        print(f"  {family:28s} {row[0]:8d} {row[1]:9d} {row[2]:7d} {row[3]:10d} {row[4]:12d} {row[5]:16d}")
    print(f"  {'all':28s} " + " ".join(f"{sum(r[i] for r in reach.values()):{w}d}" for i, w in enumerate((8, 9, 7, 10, 12, 16))))
    compared = [r for r in records if r["same"] is not None]
    differ = [r for r in compared if not r["same"]]
    print(f"\ncalls accepted in the recording and in the replay, with every earlier call of the episode ending as recorded: {len(compared)}; "
          f"the answer differs in {len(differ)}")
    worse = [r for r in records if r["cls"] != "ok" and not r["was_refused"]]
    print(f"\ncalls accepted in the recording and refused in the replay: {len(worse)}")
    for k, v in collections.Counter(r["cls"] for r in worse).most_common(12):
        print(f"  {v:5d}  {k}")
    return dict(calls=n, refused_before=before, refused_after=len(after), slips=slips, classes=dict(classes), compared=len(compared),
                differ=len(differ),
                old_limit=dict(n=len(limited), finish=done, new_limit=still, other=other), ended=dict(table), reach=dict(reach))


def write_sample(path, episodes, records, most=150, steps=12_000_000):
    """A small deterministic sample for the test suite: whole episodes (names carry over), chosen by the hash of their
    id among those with a recorded refusal, at most four of a family and ``steps`` steps in all (the test replays them
    twice); each call with the class the replay gave it."""
    by_episode = collections.defaultdict(list)
    for r in records:
        by_episode[r["episode"]].append(r)
    order = sorted(by_episode, key=lambda e: hashlib.sha256(episodes[e]["episode"].encode()).hexdigest())
    lines, calls, families, used = [], 0, collections.Counter(), 0
    for e in order:
        rs = by_episode[e]
        cost = sum(r["steps"] for r in rs)
        if not any(r["was_refused"] for r in rs) or cost > 1_500_000 or len(rs) > 12 or any(r["text"] is None for r in rs):
            continue
        if calls + len(rs) > most or used + cost > steps or families[episodes[e]["env_id"]] >= 4:
            continue
        families[episodes[e]["env_id"]] += 1
        calls += len(rs)
        used += cost
        lines.append(json.dumps(dict(episode=episodes[e]["episode"], calls=[
            dict(text=r["text"], was_refused=r["was_refused"], was_limit=r["was_limit"], cls=r["cls"],
                 shown=hashlib.sha256((r["shown"] or r["message"] or "").encode()).hexdigest()[:16]) for r in rs]), sort_keys=True))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(lines)} episodes, {calls} calls, {used} steps")


def refresh_sample(path):
    """Replay the episodes of a sample file and store the class and the answer of every call as this tree gives them."""
    with open(path) as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    changed = 0
    for episode in episodes:
        names = {}
        for call in episode["calls"]:
            try:
                shown, error = C.render(C.calc(call["text"], names)), None
            except C._CALC_ERRORS as e:
                shown, error = None, e
            new = (classify(error), hashlib.sha256(((shown if error is None else str(error)[:200]) or "").encode()).hexdigest()[:16])
            changed += new != (call["cls"], call["shown"])
            call["cls"], call["shown"] = new
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(e, sort_keys=True) for e in episodes) + "\n")
    print(f"{path}: {len(episodes)} episodes, {sum(len(e['calls']) for e in episodes)} calls, {changed} changed")


THROUGHPUT = {
    "scalar loop (grid fit, closed-form solve)": """
import math
TI = [60, 300, 500, 740, 1100]
S = [-1.072, -0.059721, 0.63779, 1.3348, 1.71]
best = None
for k in range(100000):
    T1 = 50 + 0.1 * k
    x = [math.exp(-t / T1) for t in TI]
    n = len(x); sx = sum(x); sy = sum(S); sxx = sum(v * v for v in x); sxy = sum(a * b for a, b in zip(x, S))
    den = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / den; a = (sy - b * sx) / n
    r = sum((s - a - b * v) ** 2 for s, v in zip(S, x))
    if best is None or r < best[0]:
        best = (r, T1, a, b)
""",
    "small arrays inside two loops": """
import numpy as np
t = np.array([60, 300, 500, 740, 1100]); s = np.array([-1.072, -0.059721, 0.63779, 1.3348, 1.71])
best = (1e9, 0, 0)
for T1 in np.linspace(100, 3000, 3000):
    for k in np.linspace(0.5, 1.0, 100):
        m = 1 - (1 + k) * np.exp(-t / T1)
        a = np.sum(m * s) / np.sum(m * m)
        r = np.sum((s - a * m) ** 2)
        if r < best[0]:
            best = (r, T1, k)
""",
    "function calls": """
def f(x, T1): return 1 - 2 * exp(-x / T1)
n = 0
while True:
    n += f(n, 1000.0) * 0 + 1
""",
}


def throughput():
    for name, text in THROUGHPUT.items():
        runs = []
        saved, C._Run = C._Run, probe(runs)
        t0 = time.perf_counter()
        try:
            C.calc(text)
            end = "finished"
        except C.CalcError as e:
            end = "stopped" if "stopped after" in str(e) else f"refused: {e}"
        finally:
            C._Run = saved
        dt = time.perf_counter() - t0
        steps = getattr(runs[-1], "steps", 0)
        print(f"{name:45s} {end:9s} {steps:>11,} steps  {dt:6.2f} s  {steps / dt / 1e6:5.2f} M steps/s")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("corpus", nargs="?")
    p.add_argument("--limit", type=int, default=None, help="replay with this step limit instead of calc.MAX_STEPS")
    p.add_argument("--every", type=int, default=1, help="replay every n-th episode")
    p.add_argument("--write-sample", default=None, help="write a small sample of the replay as a test fixture")
    p.add_argument("--json", default=None, help="write the per-call records (without the texts) here")
    p.add_argument("--throughput", action="store_true", help="steps per second on three fixed programs")
    p.add_argument("--refresh-sample", default=None, help="replay the episodes of this sample file and rewrite what they give")
    p.add_argument("--messages", default=None, help="a glob of messages.json files of recorded episodes, instead of a corpus file")
    a = p.parse_args(argv)
    if a.throughput:
        throughput()
        return 0
    if a.refresh_sample:
        refresh_sample(a.refresh_sample)
        return 0
    if a.messages:
        episodes = from_messages(a.messages)[::a.every]
    elif a.corpus:
        with open(a.corpus) as f:
            episodes = [json.loads(line) for line in f if line.strip()][::a.every]
    else:
        p.error("the corpus path (or --messages) is needed")
    records = replay(episodes, a.limit)
    summary = report(episodes, records, a.limit)
    if a.write_sample:
        write_sample(a.write_sample, episodes, records)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(summary=summary, records=[{k: v for k, v in r.items() if k not in ("text", "shown")} for r in records]), f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
