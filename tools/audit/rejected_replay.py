"""Stored model episodes, replayed call by call, to see what the rule on rejected calls changes for them.

    python tools/audit/rejected_replay.py RUN/episodes [--workers 2] [--no-replay] [--json OUT.json]

The rule (``mrgym.interface.counts_as_tool_call``): a call refused with class ``shape`` or ``vocabulary`` is a rejected
call, a call refused with class ``task`` is a tool call whatever its type; refused deliverables, ``episode_over`` and
``internal_error`` are rejected calls. Before, every ``invalid_value`` refusal was a rejected call, whatever its class.

Three readings of every ``messages.json`` + ``record.json`` under the folder, from the most direct to the least:

    record    the record's own counters (``interface``: shape, vocabulary, internal_errors; ``counts``:
              refused_deliverables), written by the tree that played the episode. Under the rule of this tree its
              rejected calls are shape + vocabulary + refused deliverables + internal errors: exact for that tree.
    messages  the error type in the first line of every stored tool message (``error <type> ·``), counted by the old
              table: a check that the stored rejected count is what the old table gives.
    replay    the environment of the record (id, level, seed, split) made again with no limit on calls, and every
              stored call that the environment answered sent to it in the stored order (a call the harness answered
              itself, beyond its cap of calls per reply or after the end, is not). Each answer is classed by the old
              table (the type alone) and by the rule of this tree (the class), and its first line is compared with the
              stored one. This is the tree of today, not the one that played: where its tools have changed since (the
              calculator), an answer differs and the replay says what today's tree does with the stored call.

What none of them can know. The model would have continued differently: an episode that the old rule ended has no
calls after its eighth rejected call, so "would not have ended there" says only that, not how the episode would have
gone on or whether it would have delivered. In the replay the noise of the scans may differ from the recorded episode
(the record does not carry the trainer's nonce); refusals do not depend on it.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

OLD_TOOL_CALL_TYPES = ("console_limit", "budget", "not_found")          # the old table: the types that counted as tool calls
BIG = 10 ** 9


def _harness_answered(stored: str) -> bool:
    """A tool message the harness wrote itself (the call never reached the environment)."""
    head = stored.split("\n")[0]
    return head.startswith("error episode_over") or "tool calls per reply" in head


def kind_of(tool: str, err: dict) -> str:
    """A refusal in the words of the report."""
    cls, type = err.get("class"), err.get("type")
    if type != "invalid_value" or cls != "task":
        return f"{type} ({cls})"
    msg = err.get("message") or ""
    if tool == "calc":
        return "invalid_value (task): calc run-time error" + (", step limit" if msg.startswith("stopped after") else "")
    if msg.startswith("required"):
        return "invalid_value (task): missing required"
    if "a unit is never converted" in msg:
        return "invalid_value (task): another unit"
    if any(w in msg for w in ("below the minimum", "above the maximum", "is outside", "not an integer", "even", "nearest:")):
        return "invalid_value (task): out of range / not on the grid"
    return f"invalid_value (task): other ({tool})"


def read(folder: str) -> tuple[dict, list, dict]:
    """(record, messages, the row of the two readings that need no environment)."""
    from mrgym import interface as I
    with open(os.path.join(folder, "record.json"), encoding="utf-8") as f:
        record = json.load(f)
    with open(os.path.join(folder, "messages.json"), encoding="utf-8") as f:
        messages = json.load(f)["messages"]
    stored, refused = record.get("interface") or {}, (record.get("counts") or {}).get("refused_deliverables") or 0
    max_steps = int(record.get("max_steps") or 0) or None
    rejected = sum(stored.get(k) or 0 for k in ("shape", "vocabulary", "internal_errors")) + refused
    steps = (stored.get("calls") or 0) - rejected
    # a delivery is the last call: it stands when it is at most the max_steps-th tool call; any other call ends the episode
    # when it is the max_steps-th. (Which of two limits came first inside one episode the counters cannot say; the replay can.)
    submitted = record["stop_reason"] == "submitted"
    end = ("max_steps" if max_steps and steps > max_steps - (not submitted) else "submitted" if submitted
           else "3 refused deliverables" if refused >= I.MAX_INVALID_DELIVERABLES else "8 rejected calls" if rejected >= I.MAX_REJECTED_CALLS else None)
    types = collections.Counter()
    for m in messages:
        head = (m.get("content") or "").split("\n")[0] if m.get("role") == "tool" else ""
        if head.startswith("error ") and not _harness_answered(head):
            types[head.split(" ")[1]] += 1
    row = dict(folder=folder, level=record["level"], seed=record["seed"], stop_reason=record["stop_reason"], stored=stored,
               stored_refused_deliverables=refused, by_record=dict(steps=steps, rejected=rejected, end=end), stored_types=dict(types),
               old_rejected_by_messages=sum(n for t, n in types.items() if t not in OLD_TOOL_CALL_TYPES))
    return record, messages, row


def replay(folder: str, run: bool = True) -> dict:
    import mrgym
    from mrgym import interface as I
    record, messages, row = read(folder)
    if not run:
        return row
    settings = record.get("settings") or {}
    kw = {} if settings.get("calc", True) else {"calc": False}
    env = mrgym.make(record["env_id"], level=record["level"], seed=record["seed"], split=record["split"], max_steps=BIG, max_rejected_calls=BIG, **kw)
    env.reset()
    max_steps, max_rejected, max_refused = int(record.get("max_steps") or env.spec.max_steps), I.MAX_REJECTED_CALLS, I.MAX_INVALID_DELIVERABLES
    stored = {m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"}
    cap = int(settings.get("max_calls_per_turn") or BIG)
    old = dict(steps=0, rejected=0, end=None)
    new = dict(steps=0, rejected=0, end=None)
    refused_deliverables = sent = same = 0
    refusals, differ = collections.Counter(), []
    over = False
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for k, c in enumerate(m.get("tool_calls") or []):
            was = stored.get(c["id"])
            if was is None or k >= cap or _harness_answered(was) or over:
                continue
            step = env.step(c)
            obs = step.observation
            sent += 1
            now, then = obs.text.split("\n")[0], was.split("\n")[0]
            same += now == then
            if now != then and len(differ) < 3:
                differ.append(dict(call=sent, stored=then[:160], replay=now[:160]))
            err = obs.data.get("error") if obs.is_error else None
            if err:
                refusals[kind_of(c["function"]["name"], err)] += 1
                refused_deliverables += err["type"] == "invalid_deliverable"
            for rule, counted in ((old, err is None or err["type"] in OLD_TOOL_CALL_TYPES), (new, I.counts_as_tool_call(obs.data))):
                if rule["end"]:
                    continue
                rule["steps" if counted else "rejected"] += 1
                if step.terminated:
                    rule["end"] = "submitted"
                elif rule["steps"] >= max_steps:
                    rule["end"] = "max_steps"
                elif rule["rejected"] >= max_rejected:
                    rule["end"] = "8 rejected calls"
                elif refused_deliverables >= max_refused:
                    rule["end"] = "3 refused deliverables"
                if rule["end"]:
                    rule["at_call"] = sent
            over = step.done
    counts = env.interface_counts()
    env.close()
    return dict(row, calls_sent=sent, first_lines_same=same, differ=differ, old=old, new=new, refusals=dict(refusals),
                refused_deliverables=refused_deliverables, replay_counts=counts)


def _worker(job) -> dict:
    folder, run = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        return replay(folder, run)
    except Exception as e:  # noqa: BLE001  one episode that cannot be replayed is reported, not fatal
        return dict(folder=folder, failed=f"{type(e).__name__}: {e}"[:300])


def _table(rows: list, key: str, title: str) -> None:
    print(f"\n{title}")
    head = ("level", "episodes", "ended by the limit (stored)", "NEW: 8 rejected", "3 refused deliv.", "tool-call limit", "would not have ended")
    print("  " + " | ".join(head))
    for level in sorted({r["level"] for r in rows}) + ["all"]:
        of = [r for r in rows if level == "all" or r["level"] == level]
        lim = [r for r in of if r["stop_reason"] == "rejected_call_limit"]
        ends = collections.Counter(r[key]["end"] for r in lim)
        cells = (level, len(of), len(lim), ends["8 rejected calls"], ends["3 refused deliverables"], ends["max_steps"], ends[None])
        assert sum(cells[3:]) == len(lim), ends
        print("  " + " | ".join(str(c).rjust(len(h)) for c, h in zip(cells, head)))


def report(rows: list) -> None:
    failed = [r for r in rows if r.get("failed")]
    rows = [r for r in rows if not r.get("failed")]
    print(f"episodes read: {len(rows)}" + (f"; could not be read or replayed: {len(failed)}" if failed else ""))
    for r in failed[:5]:
        print(f"  {r['folder']}: {r['failed']}")
    limit = [r for r in rows if r["stop_reason"] == "rejected_call_limit"]
    # ---- the record's own counters
    ok = sum(r["old_rejected_by_messages"] == r["stored"].get("rejected") for r in rows)
    print(f"check, old table: the error types of the stored tool messages give the record's rejected count in {ok} of {len(rows)} episodes")
    print(f"stored as rejected_call_limit: {len(limit)}; of them with 8 rejected calls in the record {sum(r['stored'].get('rejected', 0) >= 8 for r in limit)}, "
          f"with 3 refused deliverables {sum(r['stored_refused_deliverables'] >= 3 for r in limit)}")
    total = collections.Counter()
    for r in limit:
        total.update(dict(shape=r["stored"].get("shape", 0), vocabulary=r["stored"].get("vocabulary", 0),
                          refused_deliverables=r["stored_refused_deliverables"], internal_errors=r["stored"].get("internal_errors", 0),
                          rejected=r["stored"].get("rejected", 0)))
    rest = total["rejected"] - total["shape"] - total["vocabulary"] - total["refused_deliverables"] - total["internal_errors"]
    print(f"their {total['rejected']} rejected calls (record): shape {total['shape']}, vocabulary {total['vocabulary']}, refused deliverables "
          f"{total['refused_deliverables']}, internal errors {total['internal_errors']}, class task of another type (invalid_value) {rest}")
    _table(rows, "by_record", "FROM THE RECORDS (the classes the playing tree gave): the stored calls under the rule of this tree")
    still = [r for r in limit if r["by_record"]["end"] is None]
    if still:
        left = sorted(r["by_record"]["rejected"] for r in still)
        print(f"  the {len(still)} that would not have ended: rejected calls at the last stored call min {left[0]}, median {left[len(left) // 2]}, "
              f"max {left[-1]} of 8; tool calls at most {max(r['by_record']['steps'] for r in still)}")
    other = collections.Counter((r["stop_reason"], r["by_record"]["end"]) for r in rows if r["stop_reason"] != "rejected_call_limit")
    print("  episodes not stored as rejected_call_limit: stored stop reason -> end under the rule of this tree within the stored calls")
    for (was, now), n in sorted(other.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"    {was:22s} -> {str(now):24s} {n}")
    if not rows or "new" not in rows[0]:
        return
    # ---- the replay on this tree
    sent, same = sum(r["calls_sent"] for r in rows), sum(r["first_lines_same"] for r in rows)
    print(f"\nREPLAY ON THIS TREE. calls sent to the environment: {sent}; first line of the answer the same as stored: {same} "
          f"({100 * same / max(1, sent):.2f} %)")
    for r in [r for r in rows if r["differ"]][:4]:
        d = r["differ"][0]
        print(f"  differs, e.g. {os.path.relpath(r['folder'], os.path.dirname(os.path.dirname(os.path.dirname(r['folder']))))} call {d['call']}:"
              f"\n    stored: {d['stored']}\n    replay: {d['replay']}")
    agree = sum(r["old"]["end"] in ("8 rejected calls", "3 refused deliverables") and r["old"].get("at_call") == r["calls_sent"] for r in limit)
    counts = sum(r["old"]["rejected"] == r["stored"].get("rejected") for r in rows)
    print(f"  old table on the replay: rejected calls equal to the record's in {counts} of {len(rows)} episodes; of the {len(limit)} episodes "
          f"stored as rejected_call_limit, {agree} end by that limit at their last stored call (the others: an answer of today's tree differs)")
    _table(rows, "new", "replay: the stored calls on this tree, under its rule")
    both = sum(r["new"]["end"] == r["by_record"]["end"] for r in limit)
    print(f"  the replay and the records give the same end for {both} of {len(limit)} of these episodes")
    for title, of in (("refused calls of the episodes stored as rejected_call_limit, as this tree answers them", limit),
                      ("refused calls of all episodes, as this tree answers them", rows)):
        total = collections.Counter()
        for r in of:
            total.update(r["refusals"])
        print(f"\n{title}: {sum(total.values())}")
        for kind, n in sorted(total.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:5d}  {kind}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episodes", help="a folder with episode folders (messages.json + record.json) anywhere below it")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-replay", action="store_true", help="the records and the stored messages only: no environment is made")
    ap.add_argument("--json", help="write the per-episode rows here")
    a = ap.parse_args(argv)
    folders = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.episodes, "**", "messages.json"), recursive=True)
                     if os.path.isfile(os.path.join(os.path.dirname(p), "record.json")))
    if not folders:
        print(f"no episodes under {a.episodes}", file=sys.stderr)
        return 2
    t0, rows = time.time(), []
    folders = [(f, not a.no_replay) for f in folders]
    if a.workers <= 1 or a.no_replay:
        rows = [_worker(f) for f in folders]
    else:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(min(2, a.workers)) as pool:      # never more than two: the pod is shared
            for i, row in enumerate(pool.imap(_worker, folders, chunksize=1)):
                rows.append(row)
                if (i + 1) % 16 == 0:
                    print(f"  ... {i + 1}/{len(folders)} episodes, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    report(rows)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
