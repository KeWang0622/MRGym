"""Episode records -> rows -> a Markdown table and a JSON file.

    records = summarize.load("results")            # every <dir>/episodes/<agent>/<env>/L<level>/s<seed>/e<k>/record.json
    summary = summarize.summarise(records)         # the content of results.json
    print(summarize.markdown(summary))

One row per environment, level and agent: episodes, passed as ``k/N (p%)`` with the Wilson 95 % interval, mean
reward, median error, median scans, median share of the scan budget used, median turns, tokens in and out, and the
episodes lost to endpoint or runner failures. Pass rate, mean reward and median error stand next to each other on
purpose: a pass threshold alone hides how far a failing agent is from it.

Rules of the table
  * An ``infra_error`` record is not an agent result. It is counted in its own column and left out of N.
  * Every other started episode counts, with reward 0 if nothing was delivered.
  * Agents are listed alphabetically within kind, scripted policies first, then models. The table does not rank.
  * Everything here is a pure function of the records (no clock, no random numbers), so the stored table can be
    checked against a fresh build (``mrgym results --check``).
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics

from .runner import seeds_text

SCHEMA = "mrgym-results/1"
Z95 = 1.959963984540054
KINDS = ("scripted", "model", "stub")
PATTERN = os.path.join("episodes", "*", "*", "L*", "s*", "e*", "record.json")


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion. Returns (low, high); (0, 1) for n = 0."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> str:
    """A rate as ``k/N (p%)``."""
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def load(directory: str) -> list[dict]:
    """All episode records under ``directory``, in a fixed order."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, PATTERN))):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return sorted(out, key=lambda r: (r["env_id"], r["level"], r["agent_kind"], r["agent"], r["seed"], r["episode"]))


def _r(x, digits: int = 6):
    return None if x is None else round(float(x), digits)


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _row(env_id: str, level: int, agent: str, kind: str, recs: list[dict]) -> dict:
    done = [r for r in recs if r["status"] == "completed"]
    n, k = len(done), sum(bool(r["passed"]) for r in done)
    lo, hi = wilson(k, n)
    errors = [math.inf if r.get("error") is None else r["error"] for r in done]
    error = statistics.median(errors) if errors else None
    shares = [r["scan_s_used"] / r["scan_s_budget"] for r in done if r.get("scan_s_budget")]
    stops = sorted({r["stop_reason"] for r in done})
    return dict(
        env_id=env_id, level=level, agent=agent, kind=kind, episodes=n, instances=len({r["seed"] for r in done}), passed=k,
        pass_rate=_r(k / n) if n else None, pass_wilson95=[_r(lo), _r(hi)],
        reward_mean=_r(math.fsum(r["reward"] for r in done) / n) if n else None,
        error_median=None if error is None or math.isinf(error) else _r(error),
        valid_deliverables=sum(r.get("error") is not None for r in done),
        scans_median=_r(_median(r.get("scans") for r in done), 1),
        budget_share_median=_r(_median(shares), 4),
        turns_median=_r(_median(r.get("turns") for r in done), 1),
        tokens_in_median=_r(_median(r.get("tokens_in") for r in done), 0),
        tokens_out_median=_r(_median(r.get("tokens_out") for r in done), 0),
        tokens_estimated=any(r.get("tokens_estimated") for r in done),
        invalid_calls_median=_r(_median(r.get("invalid_calls") for r in done), 1),
        stop_reasons={s: sum(r["stop_reason"] == s for r in done) for s in stops},
        policy_errors=sum(bool(r.get("policy_error")) for r in done),
        admitted_episodes=sum(bool(r.get("admitted")) for r in done),
        admitted_passed=sum(bool(r.get("admitted")) and bool(r["passed"]) for r in done),
        infra_errors=len(recs) - n)


def _admitted(records: list[dict]) -> dict:
    """{environment: {level: admitted instances}} for the instances whose records carry the stored gate verdict."""
    out = {}
    for r in records:
        if r.get("admitted") is not None:
            out.setdefault(r["env_id"], {}).setdefault(str(r["level"]), set())
            if r["admitted"]:
                out[r["env_id"]][str(r["level"])].add(r["seed"])
    return {e: {lv: len(seeds) for lv, seeds in sorted(levels.items())} for e, levels in sorted(out.items())}


def summarise(records: list[dict]) -> dict:
    """The content of ``results.json``: provenance read from the records, then one row per environment, level and
    agent."""
    groups = {}
    for r in records:
        groups.setdefault((r["env_id"], r["level"], KINDS.index(r["agent_kind"]) if r["agent_kind"] in KINDS else len(KINDS),
                           r["agent"], r["agent_kind"]), []).append(r)
    rows = [_row(env_id, level, agent, kind, recs) for (env_id, level, _, agent, kind), recs in sorted(groups.items())]
    days = sorted({r["utc"][:10] for r in records})
    settings = {}
    for r in records:
        if r.get("settings"):
            settings.setdefault(r["agent"], r["settings"])
    seeds = sorted({r["seed"] for r in records})
    return dict(
        schema=SCHEMA, source=PATTERN.replace(os.sep, "/"), records=len(records),
        completed=sum(r["status"] == "completed" for r in records), infra_errors=sum(r["status"] != "completed" for r in records),
        played_utc=[days[0], days[-1]] if days else [], engines=sorted({r["engine"] for r in records}),
        package_versions=sorted({r["package_version"] for r in records}), split="train", seeds=seeds_text(seeds) if seeds else "",
        environments={e: sorted({r["level"] for r in records if r["env_id"] == e}) for e in sorted({r["env_id"] for r in records})},
        admitted_instances=_admitted(records),
        agents={kind: sorted({r["agent"] for r in records if r["agent_kind"] == kind}) for kind in KINDS},
        model_settings=settings, rows=rows)


# ------------------------------------------------------------------ text
def dateline(summary: dict) -> str:
    """One line: what was run, when, on which engine and package version, on which seeds."""
    if not summary["records"]:
        return "No episode records yet."
    days = summary["played_utc"]
    when = days[0] if days[0] == days[-1] else f"{days[0]} to {days[-1]}"
    envs = "; ".join(f"`{e}` level{'s' if len(lv) > 1 else ''} {', '.join(map(str, lv))}" for e, lv in summary["environments"].items())
    ag = summary["agents"]
    who = "scripted control policies" if ag["scripted"] else ""
    if ag["model"]:
        who += (" and " if who else "") + f"{len(ag['model'])} language model{'s' if len(ag['model']) > 1 else ''} ({', '.join(ag['model'])})"
    else:
        who += " only, no language model" if who else "no language model"
    if ag["stub"]:
        who += f"; {len(ag['stub'])} stub agent{'s' if len(ag['stub']) > 1 else ''} of the test endpoint"
    return (f"{summary['records']} episodes, played {when} (UTC) on engine `{', '.join(summary['engines'])}`, package version "
            f"`{', '.join(summary['package_versions'])}`. Environments: {envs}. Instances: public train split, seeds {summary['seeds']}. "
            f"Agents: {who}.")


def _status(summary: dict) -> str:
    ag = summary["agents"]
    if not ag["model"]:
        text = "These rows are scripted control policies only. No language model has been run on these instances yet."
    else:
        text = (f"Rows of kind `scripted` are control policies of the environment; rows of kind `model` are language models "
                f"({len(ag['model'])} so far), played on the same instances.")
    if ag["stub"]:
        text += " Rows of kind `stub` come from the test endpoint: scripted replies sent through the model path, not a model."
    return text


def _fmt(x, pattern: str = "{:g}") -> str:
    return "-" if x is None else pattern.format(x)


def _percent(x, digits: int = 0) -> str:
    if x is None:
        return "-"
    text = f"{100 * x:.{digits}f}"
    return (text.lstrip("-") if float(text) == 0 else text) + " %"       # never "-0.0 %"


def _tokens(row: dict) -> str:
    if row["tokens_in_median"] is None:
        return "-"
    return f"{row['tokens_in_median'] / 1e3:.1f}k / {row['tokens_out_median'] / 1e3:.1f}k" + (" (est.)" if row["tokens_estimated"] else "")


HEADER = ("| Agent | Kind | Episodes | Passed | 95 % interval | Mean reward | Median error | Median scans | Median budget used "
          "| Median turns | Tokens in / out | Infra errors |")
NOTES = ("Passed: episodes that meet the pass rule of the environment, as `k/N (p%)`, with the Wilson 95 % interval. "
         "N counts every started episode except those that ended in an endpoint or runner failure; these are not agent "
         "results and stand under Infra errors. An episode without a delivery has reward 0. "
         "The rows cover every instance, admitted or not: on an instance the admission gate rejects, a shortcut policy "
         "can pass, and that is why the gate rejects it (`results.json` also counts the passes on admitted instances). "
         "Median error: the error measure of the environment (`mrgym card <id>`), in percent; an episode without a valid "
         "deliverable counts as infinite, and `-` means the median is infinite. "
         "Median budget used: scan seconds used over the scan-time budget. "
         "Turns and tokens are per episode and exist for models only; `(est.)` marks counts estimated from characters "
         "because the endpoint reported no usage. Agents are in alphabetical order within kind; the table does not rank.")


def markdown(summary: dict, agents=None) -> str:
    """The table block: a status sentence, one table per environment and level, the reading notes.

    ``agents`` restricts the rows to the named agents (for a short excerpt); None keeps all.
    """
    if not summary["records"]:
        return "No episode records yet.\n"
    out = [_status(summary), ""]
    if len(summary["engines"]) > 1 or len(summary["package_versions"]) > 1:
        out += ["Note: the records come from more than one engine or package version. Rows are comparable only within one.", ""]
    rows = [r for r in summary["rows"] if agents is None or r["agent"] in agents]
    for key in sorted({(r["env_id"], r["level"]) for r in rows}):
        part = [r for r in rows if (r["env_id"], r["level"]) == key]
        gate = summary["admitted_instances"].get(key[0], {}).get(str(key[1]))
        out += [f"**`{key[0]}`, level {key[1]}** ({max(r['instances'] for r in part)} instances"
                + (f"; the admission gate admits {gate} of them)" if gate is not None else ")"), "", HEADER,
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in part:
            lo, hi = r["pass_wilson95"]
            out.append(
                f"| {r['agent']} | {r['kind']} | {r['episodes']} | {rate(r['passed'], r['episodes'])} | "
                f"{'-' if not r['episodes'] else f'{100 * lo:.0f}-{100 * hi:.0f} %'} | {_fmt(r['reward_mean'], '{:.3f}')} | "
                f"{_percent(r['error_median'], 1)} | {_fmt(r['scans_median'])} | {_percent(r['budget_share_median'])} | "
                f"{_fmt(r['turns_median'])} | {_tokens(r)} | {r['infra_errors']} |")
        out.append("")
    shown = {a: s for a, s in summary["model_settings"].items() if agents is None or a in agents}
    for text in sorted({json.dumps(s, sort_keys=True) for s in shown.values()}):
        s = json.loads(text)
        names = ", ".join(f"`{a}`" for a in sorted(shown) if shown[a] == s)
        out.append(f"Settings of {names}: at most {s.get('max_turns')} turns per episode, session `calc` tool "
                   f"{'on' if s.get('calc') else 'off'}, temperature {_fmt(s.get('temperature'))}, max tokens {_fmt(s.get('max_tokens'))} "
                   "(`-` = not sent, the endpoint's default).")
    if shown:
        out.append("")
    out.append(NOTES)
    return "\n".join(out) + "\n"


def json_text(summary: dict) -> str:
    return json.dumps(summary, indent=1) + "\n"
