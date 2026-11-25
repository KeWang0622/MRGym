"""Where the tokens of an episode go: recorded conversations, replayed call by call on the environment of this tree.

    python tools/obs_budget.py --run RUN_FOLDER [--tokenizer tokenizer.json] [--json OUT.json] [--calls OUT.jsonl]
    python tools/obs_budget.py --sizes                      characters of tools + statement of every registered id
    python tools/obs_budget.py --run RUN_FOLDER --corpus tests/data/NAME_interface_calls.jsonl
                                                            the calls of the run as a replay corpus of ``mrgym iface-tax``

``RUN_FOLDER`` holds the episodes of ``mrgym run-agent`` / ``mrgym calibrate`` with full conversations
(``episodes/<agent>/<env>/L<level>/s<seed>/e<k>/{record.json, messages.json}``). For every completed episode the
script makes the same environment (id, level, seed, split, limits and reply limit of the record), sends it the tool
calls of the recorded conversation in their order, and splits both conversations, the recorded one and the replayed
one, into the pieces a trainer has to hold:

    tools           the tool definitions (JSON, as the chat template prints them), per tool
    system          the session note of the harness            statement   the task text up to its Limits block
    limits          the Limits block                           status      the status line under every result
    result:<tool>   what a tool answered (``scanner_info`` per topic)
    error:<tool>    a refusal: its lines up to the fix         charged     the line that says what a refusal cost
    assistant       what the model wrote: text and calls (the same in both conversations)
    harness         notes of the harness between turns (the same in both conversations)

The scanner is deterministic given the calls, so on the tree an episode was recorded with the replay reproduces the
recorded text; on a tree whose text differs, the two columns are the before and the after on the same conversation.
What the replay cannot be: the conversation the model would have had with the new text. It is the same calls, read
again.

Tokens. With ``--tokenizer`` (a ``tokenizer.json`` of the Hugging Face ``tokenizers`` package, which must be
importable) every piece is tokenized on its own. Without it a token is taken as ``CHARS_PER_TOKEN`` characters and
the output says so. The stream of an episode is the sum of its pieces plus the framing of the chat template, which
is measured per episode as the recorded prompt of the last request minus the pieces that were in it, and is the
same before and after (the number of messages does not change). A second column gives the estimate the calibration
summary prints (prompt of the last request plus every completion token), moved by the same difference.

``--corpus`` writes every call that reached the environment with the class the replay gave it, so it is meant to be
run on the tree the conversations were recorded with: there the replay reproduces the recorded answers, and "lost"
means lost to the interface the model met. The file holds calls and nothing else of the conversations.

What is not replayed, and counted: a call beyond the per-reply cap of the harness and a call after the end of the
recorded episode (the harness answered those; their recorded text is kept), an episode that did not complete.
When the replayed episode reaches a limit before the recorded calls are used up (calls that were refused when
recorded and are executed now count as tool calls), the replay goes on past the limit and the episode is counted
under ``past_limit``: the sizes are those of the same calls, not of an episode that would have ended.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

CHARS_PER_TOKEN = 3.3
MAX_CALLS_PER_TURN = 12                       # the cap of the reference runner (mrgym.agents.runner)


# ------------------------------------------------------------------------------------------------ tokens
class Counter:
    """Token counts of text pieces, cached (the tool definitions and datasheet topics repeat in every episode)."""

    def __init__(self, tokenizer_path: str | None):
        self.exact, self._tk, self._cache = False, None, {}
        if tokenizer_path:
            try:
                from tokenizers import Tokenizer
                self._tk = Tokenizer.from_file(tokenizer_path)
                self.exact = True
            except Exception as e:  # noqa: BLE001  no tokenizer is a stated fallback, not a failure
                print(f"obs_budget: no tokenizer ({type(e).__name__}: {e}); a token is taken as {CHARS_PER_TOKEN} characters", file=sys.stderr)

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        n = self._cache.get(text)
        if n is None:
            n = len(self._tk.encode(text, add_special_tokens=False).ids) if self._tk else int(round(len(text) / CHARS_PER_TOKEN))
            if len(text) > 200:
                self._cache[text] = n
        return n


# ------------------------------------------------------------------------------------------------ pieces
def split_result(name: str, arguments, text: str) -> list:
    """One tool answer as ``[(category, text)]``: the status line, the charge line of a refusal and the body."""
    lines = text.split("\n")
    out = []
    if lines and lines[-1].startswith("[") and lines[-1].endswith("]"):
        out.append(("status", "\n" + lines.pop()))
    keep = []
    for ln in lines:
        if ln.startswith("charged: ") or ln.startswith("episode truncated: "):
            out.append(("charged", "\n" + ln))
        else:
            keep.append(ln)
    body = "\n".join(keep)
    if body.startswith("error "):
        kind = f"error:{name}"
    elif name == "scanner_info":
        topic = "overview"
        if isinstance(arguments, dict) and arguments:
            topic = "+".join(str(arguments[k]) for k in sorted(arguments) if isinstance(arguments[k], (str, int, float)))[:40] or "overview"
        kind = f"result:scanner_info:{topic}"
    else:
        kind = f"result:{name}"
    return [(kind, body)] + out


def split_reset(text: str) -> list:
    lines = text.split("\n")
    out = []
    if lines and lines[-1].startswith("[") and lines[-1].endswith("]"):
        out.append(("status", "\n" + lines.pop()))
    body = "\n".join(lines)
    at = body.rfind("\nLimits")
    if at >= 0:
        return [("statement", body[:at]), ("limits", body[at:])] + out
    return [("statement", body)] + out


def tool_pieces(tools: list) -> list:
    return [(f"tools:{t['function']['name']}", json.dumps(t, ensure_ascii=False)) for t in tools]


def _arguments(text):
    try:
        v = json.loads(text) if isinstance(text, str) and text.strip() else {}
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


# ------------------------------------------------------------------------------------------------ one episode
def replay_episode(directory: str, count: Counter) -> dict | None:
    """Replay one recorded episode. Returns its row, or None when it cannot be replayed (not completed, no conversation)."""
    from mrgym.agents import runner
    with open(os.path.join(directory, "record.json"), encoding="utf-8") as f:
        rec = json.load(f)
    path = os.path.join(directory, "messages.json")
    if rec.get("status") != "completed" or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        conv = json.load(f)
    messages, turns = conv["messages"], conv.get("turns") or []
    settings = dict(rec.get("settings") or {})
    if rec.get("max_steps") and settings.get("max_turns") is None:
        settings["max_turns"] = rec["max_steps"]
    env = runner.make_env(rec["env_id"], rec["level"], rec["seed"], settings, split=rec.get("split", "train"))
    obs, _ = env.reset(options=runner.reset_options(settings))
    before, after, calls = [], [], []                                   # [(category, text)] recorded and replayed; one row per call
    before += tool_pieces(conv.get("tools") if isinstance(conv.get("tools"), list) else [])
    after += tool_pieces(env.tools("openai"))
    first_user = True
    answers = {m["tool_call_id"]: m.get("content") or "" for m in messages if m.get("role") == "tool"}
    done, past_limit, not_replayed = False, 0, 0
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    in_last_prompt = 0                                                  # recorded pieces that were in the prompt of the last request
    for i, m in enumerate(messages):
        role = m.get("role")
        start = len(before)
        if role == "system":
            before.append(("system", m.get("content") or ""))
            after.append(("system", runner.first_messages(env, obs.text)[0]["content"]))
        elif role == "user" and first_user:
            first_user = False
            before += split_reset(m.get("content") or "")
            after += split_reset(obs.text)
        elif role == "user":
            before.append(("harness", m.get("content") or ""))
            after.append(("harness", m.get("content") or ""))
        elif role == "assistant":
            written = (m.get("content") or "") + "".join(
                "\n" + json.dumps({"name": c["function"]["name"], "arguments": _arguments(c["function"]["arguments"]) or c["function"]["arguments"]},
                                  ensure_ascii=False) for c in m.get("tool_calls") or [])
            before.append(("assistant", written))
            after.append(("assistant", written))
            for k, c in enumerate(m.get("tool_calls") or []):
                name, raw = c["function"]["name"], c["function"]["arguments"]
                args = _arguments(raw)
                recorded = answers.get(c["id"], "")
                row = dict(turn=m.get("turn"), name=name, arguments=raw, recorded=recorded.split("\n")[0][:200])
                parts = split_result(name, args, recorded)
                row["recorded_error"] = parts[0][0].startswith("error:")
                before += parts
                if done or k >= MAX_CALLS_PER_TURN or recorded.startswith("error episode_over") or "tool calls per reply" in recorded.split("\n")[0]:
                    after += parts                                      # answered by the harness, not by the environment
                    not_replayed += 1
                    row["replayed"] = None
                    calls.append(row)
                    continue
                step = env.step({"id": c["id"], "type": "function", "function": {"name": name, "arguments": raw}})
                o = step.observation
                after += split_result(name, args, o.text)
                err = (o.data.get("error") or {}) if o.is_error else {}
                row.update(replayed=o.text.split("\n")[0][:200], is_error=bool(o.is_error), type=err.get("type"), cls=err.get("class"),
                           parameter=err.get("parameter"), code=err.get("code"), fix=err.get("fix") is not None,
                           coerced=[x["rule"] for x in o.data.get("coercions", [])], same=o.text == recorded, chars=[len(recorded), len(o.text)])
                calls.append(row)
                if step.done:
                    if step.truncated and any(x.get("role") == "assistant" for x in messages[i + 1:]) or \
                            (step.truncated and k + 1 < len(m.get("tool_calls") or [])):
                        env._episode.update(done=False, truncated=False, reward=None, info={})     # go on past the limit (see the module text)
                        past_limit = 1
                    else:
                        done = True
        if i <= last_assistant and role != "tool":
            in_last_prompt += sum(count(t) for _, t in before[start:start + 1]) if role == "assistant" and i == last_assistant else \
                sum(count(t) for _, t in before[start:])
    env.close()
    # the recorded prompt of the last request holds everything before the last reply; tool answers are counted where issued
    usage = [(t.get("usage") or {}) for t in turns]
    last = next((u for u in reversed(usage) if isinstance(u.get("prompt_tokens"), int)), None)
    tok = lambda pieces: sum(count(t) for _, t in pieces)                # noqa: E731
    tb, ta = tok(before), tok(after)
    framing = None
    if last is not None and last_assistant >= 0:
        cut = _pieces_before_last_reply(messages, before)
        framing = last["prompt_tokens"] + last.get("completion_tokens", 0) - tok(before[:cut])
    cats = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])           # n before, chars, tokens; n after, chars, tokens
    for col, pieces in ((0, before), (3, after)):
        for k, t in pieces:
            c = cats[k]
            c[col] += 1
            c[col + 1] += len(t)
            c[col + 2] += count(t)
    return dict(env_id=rec["env_id"], level=rec["level"], seed=rec["seed"], episode=rec.get("episode"), passed=bool(rec.get("passed")), agent=rec.get("agent"),
                tokens_out=rec.get("tokens_out"), calsum_before=(last["prompt_tokens"] + rec["tokens_out"]) if last and rec.get("tokens_out") else None,
                framing=framing, tokens_before=tb, tokens_after=ta, chars_before=sum(len(t) for _, t in before),
                chars_after=sum(len(t) for _, t in after), cats={k: v for k, v in cats.items()}, calls=calls,
                past_limit=past_limit, not_replayed=not_replayed, turns=len(turns), interface=rec.get("interface"))


def _pieces_before_last_reply(messages: list, before: list) -> int:
    """Index into ``before`` just after the last assistant piece (the stream the last request and its reply make)."""
    n = sum(1 for m in messages if m.get("role") == "assistant")
    seen = 0
    for i, (k, _) in enumerate(before):
        if k == "assistant":
            seen += 1
            if seen == n:
                return i + 1
    return len(before)


# ------------------------------------------------------------------------------------------------ report
def _q(values, q):
    return int(round(float(np.percentile(values, q)))) if len(values) else None


def family_of(env_id: str) -> str:
    return env_id.split("/")[-1].split("-")[0]


def summarise(rows: list, exact: bool) -> dict:
    out = dict(exact_tokens=exact, chars_per_token=None if exact else CHARS_PER_TOKEN, episodes=len(rows), families={}, cells={})
    groups = collections.defaultdict(list)
    for r in rows:
        groups[("family", family_of(r["env_id"]))].append(r)
        groups[("cell", f"{r['env_id']} L{r['level']}")].append(r)
    for (kind, name), rs in sorted(groups.items()):
        framing = [r["framing"] or 0 for r in rs]
        sb = [r["tokens_before"] + f for r, f in zip(rs, framing)]
        sa = [r["tokens_after"] + f for r, f in zip(rs, framing)]
        cb = [r["calsum_before"] for r in rs if r["calsum_before"]]
        ca = [r["calsum_before"] - (r["tokens_before"] - r["tokens_after"]) for r in rs if r["calsum_before"]]
        cut = [1.0 - a / b for a, b in zip(sa, sb)]
        row = dict(episodes=len(rs), stream_before=dict(p50=_q(sb, 50), p90=_q(sb, 90), p99=_q(sb, 99), max=_q(sb, 100)),
                   stream_after=dict(p50=_q(sa, 50), p90=_q(sa, 90), p99=_q(sa, 99), max=_q(sa, 100)),
                   calsum_before=dict(p50=_q(cb, 50), p90=_q(cb, 90), p99=_q(cb, 99)), calsum_after=dict(p50=_q(ca, 50), p90=_q(ca, 90), p99=_q(ca, 99)),
                   cut_median=round(float(np.median(cut)), 4), cut_of_medians=round(1.0 - _q(sa, 50) / max(1, _q(sb, 50)), 4),
                   framing_median=_q(framing, 50), past_limit=sum(r["past_limit"] for r in rs), not_replayed=sum(r["not_replayed"] for r in rs),
                   calls=sum(len(r["calls"]) for r in rs),
                   same_text=sum(bool(c.get("same")) for r in rs for c in r["calls"]))
        if kind == "family":
            cats = collections.defaultdict(lambda: [0] * 6)
            for r in rs:
                for k, v in r["cats"].items():
                    for j in range(6):
                        cats[k][j] += v[j]
            row["pieces"] = {k: dict(n_before=v[0], chars_before=v[1], tokens_before=v[2], n_after=v[3], chars_after=v[4], tokens_after=v[5])
                             for k, v in sorted(cats.items(), key=lambda kv: -kv[1][2])}
            out["families"][name] = row
        else:
            out["cells"][name] = row
    return out


def _merge_topics(pieces: dict, top: int = 22) -> list:
    """Rows of the per-family table: the largest pieces by tokens before; the tool definitions as one row plus per tool."""
    rows = sorted(pieces.items(), key=lambda kv: -max(kv[1]["tokens_before"], kv[1]["tokens_after"]))
    return rows[:top], rows[top:]


def report(summary: dict) -> str:
    unit = "tokens" if summary["exact_tokens"] else f"tokens (characters / {CHARS_PER_TOKEN}; no tokenizer)"
    out = [f"observation budget of {summary['episodes']} replayed episodes; {unit}", ""]
    for name, f in summary["families"].items():
        b, a = f["stream_before"], f["stream_after"]
        out.append(f"== {name}: {f['episodes']} episodes, {f['calls']} calls ({f['same_text']} answered with the recorded text, "
                   f"{f['not_replayed']} answered by the harness, {f['past_limit']} episodes replayed past a limit)")
        out.append(f"   stream p50 {b['p50']} -> {a['p50']}   p90 {b['p90']} -> {a['p90']}   p99 {b['p99']} -> {a['p99']}   "
                   f"median cut {100 * f['cut_median']:.1f} %   (calibration-summary estimate p50 {f['calsum_before']['p50']} -> {f['calsum_after']['p50']}, "
                   f"p90 {f['calsum_before']['p90']} -> {f['calsum_after']['p90']})")
        total_b = sum(v["tokens_before"] for v in f["pieces"].values()) or 1
        total_a = sum(v["tokens_after"] for v in f["pieces"].values()) or 1
        out.append(f"   {'piece':<34}{'n':>6}{'chars each':>12}{'tokens each':>12}{'share':>8}   |{'n':>6}{'chars each':>12}{'tokens each':>12}{'share':>8}")
        head, tail = _merge_topics(f["pieces"])
        for k, v in head:
            nb, na = max(1, v["n_before"]), max(1, v["n_after"])
            out.append(f"   {k:<34}{v['n_before']:>6}{v['chars_before'] / nb:>12.0f}{v['tokens_before'] / nb:>12.0f}{100 * v['tokens_before'] / total_b:>7.1f}%   |"
                       f"{v['n_after']:>6}{v['chars_after'] / na:>12.0f}{v['tokens_after'] / na:>12.0f}{100 * v['tokens_after'] / total_a:>7.1f}%")
        if tail:
            tb, ta = sum(v["tokens_before"] for _, v in tail), sum(v["tokens_after"] for _, v in tail)
            out.append(f"   {'(' + str(len(tail)) + ' smaller pieces)':<34}{'':>6}{'':>12}{'':>12}{100 * tb / total_b:>7.1f}%   |{'':>6}{'':>12}{'':>12}{100 * ta / total_a:>7.1f}%")
        out.append(f"   {'all pieces, per episode':<34}{'':>6}{'':>12}{total_b / f['episodes']:>12.0f}{'':>8}   |{'':>6}{'':>12}{total_a / f['episodes']:>12.0f}")
        out.append("")
    out.append(f"{'cell':<34}{'n':>4}{'p50 before':>12}{'after':>8}{'p90 before':>12}{'after':>8}{'p99 before':>12}{'after':>8}{'median cut':>12}")
    for name, c in summary["cells"].items():
        b, a = c["stream_before"], c["stream_after"]
        out.append(f"{name:<34}{c['episodes']:>4}{b['p50']:>12}{a['p50']:>8}{b['p90']:>12}{a['p90']:>8}{b['p99']:>12}{a['p99']:>8}{100 * c['cut_median']:>11.1f}%")
    return "\n".join(out)


def write_corpus(path: str, rows: list, name: str) -> dict:
    """The calls of the replayed episodes as a corpus of ``mrgym iface-tax --replay``: one header line, then one line
    per call, ``[episode, turn, name, arguments, recorded, lost]``. ``arguments`` are the model's own, as an object
    where it wrote JSON and as its text where it did not; ``recorded`` is ``ok`` or ``<error type>/<class>`` as the
    replay on this tree answered; ``lost`` is 1 for the classes ``shape`` and ``vocabulary``."""
    from mrgym import __version__, canary
    from mrgym import interface as I
    lines, lost_total = [], 0
    for e, r in enumerate(rows):
        for c in r["calls"]:
            if c.get("replayed") is None:                               # answered by the harness: it never reached the environment
                continue
            lost = int(bool(c.get("is_error")) and c.get("cls") in I.INTERFACE_CLASSES)
            lost_total += lost
            args = _arguments(c["arguments"])
            lines.append([e, max(0, int(c["turn"] or 1) - 1), c["name"], c["arguments"] if args is None else args,
                          f"{c['type']}/{c['cls']}" if c.get("is_error") else "ok", lost])
    header = dict(corpus=name, canary=canary.CANARY, format="rows are [episode, turn, name, arguments, recorded, lost]",
                  description=f"every tool call of {len(rows)} recorded episodes of {rows[0]['agent']} on {len({r['env_id'] for r in rows})} "
                              "environments, with the arguments as the model wrote them (calls only: no task text, no answers)",
                  recorded_interface=f"mrgym {__version__}, engine and interface of the recording", agent=rows[0]["agent"], episodes=len(rows),
                  calls=len(lines), lost_when_recorded=lost_total,
                  lost_rule="answered with an error of class shape or vocabulary, in the episode's own state, by the interface it was recorded on",
                  instances=[[r["env_id"], r["level"], r["seed"]] for r in rows], turns=[r["turns"] for r in rows])
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for row in lines:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    return header


def sizes() -> list:
    """Characters of the tool definitions (OpenAI wire format, ``json.dumps``) and of the task text of every id and level."""
    import mrgym
    rows = []
    for env_id in mrgym.list_envs():
        spec = mrgym.spec(env_id)
        for level in spec.levels:
            env = mrgym.make(env_id, level=level, seed=9000)
            obs, _ = env.reset()
            tools = env.tools("openai")
            per = {t["function"]["name"]: len(json.dumps(t)) for t in tools}
            rows.append(dict(env_id=env_id, level=level, kind=getattr(spec, "kind", None), tools=len(json.dumps(tools)), statement=len(obs.text),
                             total=len(json.dumps(tools)) + len(obs.text), per_tool=per))
            env.close()
    return rows


def sizes_report(rows: list) -> str:
    out = [f"{'id':<28}{'level':>6}{'tools':>8}{'statement':>11}{'total':>8}   per tool"]
    for r in rows:
        out.append(f"{r['env_id']:<28}{r['level']:>6}{r['tools']:>8}{r['statement']:>11}{r['total']:>8}   "
                   + " ".join(f"{k}={v}" for k, v in r["per_tool"].items()))
    worst = collections.defaultdict(int)
    for r in rows:
        worst[r["env_id"]] = max(worst[r["env_id"]], r["total"])
    out.append("largest tools + statement per id: " + ", ".join(f"{k.split('/')[-1]} {v}" for k, v in worst.items()))
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", metavar="FOLDER", help="run folder with episodes/<agent>/<env>/L<level>/s<seed>/e<k>/")
    p.add_argument("--sizes", action="store_true", help="characters of tools + statement of every registered id and level")
    p.add_argument("--tokenizer", metavar="tokenizer.json", default=os.environ.get("MRGYM_TOKENIZER"),
                   help=f"tokenizer file of the model family (default: $MRGYM_TOKENIZER; without it: characters / {CHARS_PER_TOKEN})")
    p.add_argument("--only", metavar="TEXT", default=None, help="only episode folders whose path contains TEXT")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="at most N episodes (the first N in path order)")
    p.add_argument("--json", metavar="OUT.json", help="write the summary as JSON")
    p.add_argument("--calls", metavar="OUT.jsonl", help="write one line per recorded call: how it was answered then and now")
    p.add_argument("--corpus", metavar="OUT.jsonl", help="write the calls as a replay corpus of mrgym iface-tax (run it on the recorded tree)")
    a = p.parse_args(argv)
    if not a.run and not a.sizes:
        p.error("give --run FOLDER or --sizes")
    if a.sizes:
        rows = sizes()
        print(sizes_report(rows))
        if a.json and not a.run:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1)
    if not a.run:
        return 0
    count = Counter(a.tokenizer)
    folders = sorted(os.path.dirname(x) for x in glob.glob(os.path.join(a.run, "**", "episodes", "*", "*", "*", "*", "*", "record.json"), recursive=True))
    folders = [d for d in folders if not a.only or a.only in d][:a.limit]
    rows, skipped = [], 0
    for d in folders:
        row = replay_episode(d, count)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    if not rows:
        print(f"no completed episode with a conversation under {a.run}")
        return 1
    summary = summarise(rows, count.exact)
    summary["skipped_episodes"] = skipped
    print(report(summary))
    print(f"episodes not replayed (not completed or no conversation): {skipped}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    if a.corpus:
        header = write_corpus(a.corpus, rows, os.path.basename(os.path.normpath(a.run)))
        print(f"corpus {a.corpus}: {header['calls']} calls of {header['episodes']} episodes, {header['lost_when_recorded']} lost when recorded")
    if a.calls:
        with open(a.calls, "w", encoding="utf-8") as f:
            for r in rows:
                for k, c in enumerate(r["calls"]):
                    f.write(json.dumps(dict(env_id=r["env_id"], level=r["level"], seed=r["seed"], episode=r["episode"], index=k, **c)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
