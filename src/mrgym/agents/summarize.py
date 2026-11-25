"""Episode records -> rows -> a Markdown table and a JSON file (``results.json``, schema ``mrgym-results/2``).

    records = summarize.load("results")            # every <dir>/episodes/<agent>/<env>/L<level>/s<seed>/e<k>/record.json
    summary = summarize.summarise(records, plan)   # the content of results.json
    print(summarize.markdown(summary))

The unit is the instance. One row per (environment, agent): the cell of the published design, with its levels as
strata. Every rate carries episodes, instances and the clustered Wilson interval of ``mrgym.stats.cluster_rate``;
per-level rates are printed for description and no comparison is computed from them.

Tiers
  published   every record is of schema 2 (stored conversation, harness id, deliverable) and the cell has at least
              100 instances. ``mrgym results --check`` then holds the row to the reporting rules R1-R13 (300 instances,
              2 episodes per instance for a model, the seed set of the analysis plan, settings that were sent, ...).
  pilot       anything else: a record of schema 1, or fewer than 100 instances. Pilot rows are listed under their own
              heading, are absent from the README table and never enter a comparison.

Rules of the table
  * An ``infra_error`` record is not an agent result. It is counted in its own column and left out of N.
  * Every other started episode counts, with reward 0 if nothing was delivered.
  * Agents are listed alphabetically within kind, scripted policies first, then models. The table does not rank.
  * Comparisons are the family fixed in ``analysis_plan.json`` before the run: paired differences on the instances
    both agents played, Holm within an environment. A difference that is not in that table is not a finding.
  * Everything here is a pure function of the records, the plan and the optional price and oracle files (no clock,
    no random numbers), so the stored table can be checked against a fresh build (``mrgym results --check``).
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import statistics

from .. import record as _record
from .. import stats
from ..stats import power, reliability

SCHEMA = "mrgym-results/2"
KINDS = ("scripted", "model", "stub")
PATTERN = os.path.join("episodes", "*", "*", "L*", "s*", "e*", "record.json")
PLAN_FILE, PRICES_FILE, ORACLE_FILE = "analysis_plan.json", "prices.json", "oracle.json"
PLAN_SCHEMA = "mrgym-analysis-plan/1"
BETA_BINOMIAL = "beta-binomial-grid"


def rate(k: int, n: int) -> str:
    """A rate as ``k/N (p%)``."""
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def load(directory: str) -> list[dict]:
    """All episode records under ``directory``, in a fixed order. Records of schema 1 are read too and marked as
    pilot data (``mrgym.record.upgrade``). ``_dir`` is the episode folder relative to ``directory``."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, PATTERN))):
        with open(path, encoding="utf-8") as f:
            rec = _record.upgrade(json.load(f))
        rec["_dir"] = os.path.relpath(os.path.dirname(path), directory).replace(os.sep, "/")
        out.append(rec)
    return sorted(out, key=lambda r: (r["env_id"], r["level"], r["agent_kind"], r["agent"], str(r["seed"]), r["episode"]))


def read_json(directory: str, name: str):
    try:
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def file_sha256(directory: str, name: str) -> str | None:
    try:
        with open(os.path.join(directory, name), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def load_plan(directory: str) -> dict | None:
    """The analysis plan of a results folder with its file hash under ``sha256``, or None."""
    plan = read_json(directory, PLAN_FILE)
    if not isinstance(plan, dict):
        return None
    return dict(plan, sha256=file_sha256(directory, PLAN_FILE))


# ------------------------------------------------------------------ small helpers
def _r(x, digits: int = 6):
    return None if x is None else round(float(x), digits)


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def instance_key(r: dict) -> str:
    return f"L{r['level']}:{r['seed']}"


def seed_set_sha256(env_id: str, instances) -> str:
    """Hash of a set of instances of one environment: ``instances`` are (level, split, seed) triples. Two rows of a
    table played the same instances exactly when their hashes are equal; a suite manifest stores the same hash."""
    lines = sorted({f"{env_id}|{level}|{split}|{seed}" for level, split, seed in instances})
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _ranges(numbers) -> str:
    numbers, parts, i = sorted(set(numbers)), [], 0
    while i < len(numbers):
        j = i
        while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
            j += 1
        parts.append(str(numbers[i]) if i == j else f"{numbers[i]}-{numbers[j]}")
        i = j + 1
    return ",".join(parts)


def seeds_text(pairs) -> str:
    """(level, seed) pairs -> ``"L1: 0-23; L2: report:0-149"``. Integer seeds and ``prefix:<int>`` seeds are written
    as ranges."""
    out = []
    for level in sorted({lv for lv, _ in pairs}):
        groups = {}
        for lv, seed in pairs:
            if lv != level:
                continue
            m = re.fullmatch(r"(.*:)?(\d+)", str(seed))
            if m:
                groups.setdefault(m[1] or "", []).append(int(m[2]))
            else:
                groups.setdefault(None, []).append(str(seed))
        parts = [prefix + _ranges(v) for prefix, v in sorted((k, v) for k, v in groups.items() if k is not None)]
        parts += sorted(groups.get(None, []))
        out.append(f"L{level}: " + ",".join(parts))
    return "; ".join(out)


def _sampling(recs: list[dict]) -> dict:
    """What was actually sent, per setting: the single value, ``None`` when it was not sent, ``"mixed"`` otherwise."""
    out = {}
    for name in ("temperature", "max_tokens", "max_turns"):
        values = set()
        for r in recs:
            sent = (r.get("sampler") or {}).get("sent") or r.get("params_sent") or {}
            v = (r.get("settings") or {}).get("max_turns") if name == "max_turns" else sent.get(name, sent.get("max_completion_tokens") if name == "max_tokens" else None)
            values.add(json.dumps(v))
        out[name] = json.loads(values.pop()) if len(values) == 1 else "mixed" if values else None
    return out


def _one(values):
    values = sorted({json.dumps(v, sort_keys=True) for v in values if v is not None})
    return None if not values else json.loads(values[0]) if len(values) == 1 else [json.loads(v) for v in values]


def _price(prices: dict | None, model: str | None) -> dict | None:
    for p in (prices or {}).get("prices", []) if isinstance(prices, dict) else []:
        if p.get("model") == model:
            return p
    return None


# ------------------------------------------------------------------ one row
def _row(env_id: str, agent: str, kind: str, recs: list[dict], prices=None, oracle=None) -> dict:
    done = [r for r in recs if r["status"] == "completed"]
    n, k = len(done), sum(bool(r["passed"]) for r in done)
    y = [float(bool(r["passed"])) for r in done]
    keys = [instance_key(r) for r in done]
    levels = sorted({r["level"] for r in done})
    strata = [r["level"] for r in done] if len(levels) > 1 else None
    cr = stats.cluster_rate(y, keys, strata)
    by = {}
    for r in sorted(done, key=lambda r: r["episode"]):
        by.setdefault(instance_key(r), []).append(float(bool(r["passed"])))
    m = len(by)
    ci, method = [cr["lo"], cr["hi"]], cr["method"]
    wilson_ci, bb_ci = list(ci), None
    if 0 < m < power.PILOT_BELOW:                                # small cell: also the Beta-binomial interval; print the wider
        ordered = sorted(by)
        bb_ci = list(stats.beta_binomial_interval([sum(by[i]) for i in ordered], [len(by[i]) for i in ordered]))
        if bb_ci[1] - bb_ci[0] > ci[1] - ci[0]:
            ci, method = bb_ci, BETA_BINOMIAL
    counts = [len(v) for v in by.values()]
    pairs = [v[:2] for _, v in sorted(by.items()) if len(v) >= 2]
    retest = stats.retest_agreement([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
    reward = stats.cluster_mean([r["reward"] for r in done], keys, strata)
    errors = [math.inf if r.get("error") is None else r["error"] for r in done]
    error = statistics.median(errors) if errors else None
    shares = [r["scan_s_used"] / r["scan_s_budget"] for r in done if r.get("scan_s_budget")]
    stops = sorted({r["stop_reason"] for r in done})
    by_level = {}
    for lv in levels:
        part = [r for r in done if r["level"] == lv]
        c = stats.cluster_rate([float(bool(r["passed"])) for r in part], [instance_key(r) for r in part])
        by_level[str(lv)] = dict(instances=c["instances"], episodes=c["episodes"], passed=sum(bool(r["passed"]) for r in part),
                                 pass_rate=_r(c["rate"]), pass_ci95=[_r(c["lo"]), _r(c["hi"])])
    regrets = None
    opt = (oracle or {}).get(env_id) if isinstance(oracle, dict) else None
    if opt:
        from ..envs import oracle as O
        values = [O.regret(r.get("error"), opt["instances"].get(instance_key(r)), opt.get("kind", "difference")) for r in done]
        regrets = O.regret_report(values, [bool(r["passed"]) for r in done])
    tok = lambda name: [r[name] for r in done if isinstance(r.get(name), (int, float))]          # noqa: E731
    tokens_in, tokens_out, walls = tok("tokens_in"), tok("tokens_out"), tok("wall_s")
    model = _one(r.get("model") for r in recs)
    price = _price(prices, model if isinstance(model, str) else None)
    cost = None
    if price and tokens_in and tokens_out:
        cost = _r((sum(tokens_in) * price["usd_per_million_in"] + sum(tokens_out) * price["usd_per_million_out"]) / 1e6, 2)
    lacking = {}
    for r in done:
        for name in _record.missing(r):
            lacking[name] = lacking.get(name, 0) + 1
    schemas = sorted({r["record_schema"] for r in recs})
    if 1 in schemas:
        tier, why = "pilot", "schema 1"
    elif m < power.PILOT_BELOW:
        tier, why = "pilot", f"< {power.PILOT_BELOW} instances"
    else:
        tier, why = "published", None
    days = sorted(r["utc"] for r in recs if r.get("utc"))
    return dict(
        env_id=env_id, agent=agent, kind=kind, tier=tier, tier_reason=why, record_schemas=schemas,
        split=_one(r.get("split") for r in recs), levels=levels,
        seeds=seeds_text(sorted({(r["level"], r["seed"]) for r in done}, key=lambda p: (p[0], str(p[1])))),
        seed_set_sha256=seed_set_sha256(env_id, {(r["level"], r.get("split"), r["seed"]) for r in done}),
        instances=m, episodes=n, episodes_per_instance=[min(counts), max(counts)] if counts else [0, 0], passed=k,
        pass_rate=_r(cr["rate"]), pass_ci95=[_r(ci[0]), _r(ci[1])], ci_method=method,
        pass_ci95_wilson=[_r(wilson_ci[0]), _r(wilson_ci[1])], pass_ci95_beta_binomial=None if bb_ci is None else [_r(bb_ci[0]), _r(bb_ci[1])],
        pass_se=_r(cr["se"]), n_eff=_r(cr["n_eff"], 2), deff=_r(cr["deff"], 4), one_sided_bound=_r(cr["one_sided_bound"]),
        icc=_r(stats.icc_oneway(y, keys)) if n else None,
        pass_all_k=_r(sum(min(v) == 1 for v in by.values()) / m) if m and max(counts) > 1 else None,     # needs repeats
        pass_any_k=_r(sum(max(v) == 1 for v in by.values()) / m) if m and max(counts) > 1 else None,
        retest_agreement=None if retest is None else _r(retest["agreement"]),
        reward_mean=_r(reward["mean"]), reward_ci95=[_r(max(0.0, reward["lo"])), _r(min(1.0, reward["hi"]))] if n and reward["lo"] is not None and math.isfinite(reward["lo"]) else None,
        error_median=None if error is None or math.isinf(error) else _r(error),
        regret_median=None if not regrets else _r(regrets["median"]), regret_q95=None if not regrets else _r(regrets["q95"]),
        regret_max=None if not regrets else _r(regrets["max"]), regret_negative_passing=None if not regrets else regrets["negative_passing"],
        valid_deliverables=sum(r.get("error") is not None for r in done),
        by_level=by_level,
        scans_median=_r(_median(r.get("scans") for r in done), 1),
        budget_share_median=_r(_median(shares), 4),
        turns_median=_r(_median(r.get("turns") for r in done), 1),
        tokens_in_median=_r(_median(r.get("tokens_in") for r in done), 0),
        tokens_out_median=_r(_median(r.get("tokens_out") for r in done), 0),
        tokens_estimated=any(r.get("tokens_estimated") for r in done),
        tokens_in_total=sum(tokens_in) if tokens_in else None, tokens_out_total=sum(tokens_out) if tokens_out else None,
        wall_s_total=_r(math.fsum(walls), 1) if walls else None, cost_usd=cost,
        price=None if price is None else dict(date=price.get("date"), source=price.get("source")),
        invalid_calls_median=_r(_median(r.get("invalid_calls") for r in done), 1),
        stop_reasons={s: sum(r["stop_reason"] == s for r in done) for s in stops},
        policy_errors=sum(bool(r.get("policy_error")) for r in done),
        infra_errors=len(recs) - n, infra_error_rate=_r((len(recs) - n) / len(recs), 4) if recs else None,
        command=_one(r.get("command") for r in recs), model=model, provider=_one(r.get("provider") for r in recs),
        utc=[days[0], days[-1]] if days else None, engine=_one(r.get("engine") for r in recs),
        package_version=_one(r.get("package_version") for r in recs),
        sampling=None if kind == "scripted" else _sampling(done), harness_id=_one(r.get("harness_id") for r in done),
        adapter_sha256=_one(r.get("adapter_sha256") for r in done),
        plan_sha256=_one(r.get("plan_sha256") for r in done), transcripts=sum(bool(r.get("transcript")) for r in done),
        records_missing=dict(sorted(lacking.items())))


# ------------------------------------------------------------------ comparisons
def _instance_means(recs: list[dict]) -> tuple[dict, dict, bool]:
    """({instance: mean pass}, {instance: level}, one episode per instance) of the completed records of a cell."""
    by, level = {}, {}
    for r in recs:
        if r["status"] == "completed":
            by.setdefault(instance_key(r), []).append(float(bool(r["passed"])))
            level[instance_key(r)] = r["level"]
    return {k: sum(v) / len(v) for k, v in by.items()}, level, all(len(v) == 1 for v in by.values())


def comparisons(plan: dict | None, rows: list[dict], cells: dict) -> list[dict]:
    """The planned comparisons that can be computed: both rows exist and are published. Paired on the instances both
    agents played, strata = levels, Holm within an environment over the comparisons computed."""
    tiers = {(r["env_id"], r["agent"]): r["tier"] for r in rows}
    out = []
    for c in (plan or {}).get("comparisons") or []:
        pair = [(c["env_id"], c["a"]), (c["env_id"], c["b"])]
        if any(tiers.get(p) != "published" for p in pair):
            continue
        (a, la, ka), (b, lb, kb) = (_instance_means(cells[p]) for p in pair)
        res = stats.paired_compare(a, b, {**la, **lb} if len(set(la.values()) | set(lb.values())) > 1 else None,
                                   alternative=c.get("alternative", "two-sided"), binary=ka and kb)
        if not res["instances"]:                                 # the two rows share no instance: nothing to compare
            continue
        out.append(dict(id=c["id"], env_id=c["env_id"], a=c["a"], b=c["b"], alternative=res["alternative"], test=res["test"],
                        instances=res["instances"], diff=_r(res["diff"], 9), ci95=[_r(res["lo"], 9), _r(res["hi"], 9)], se=_r(res["se"], 9),
                        p=res["p"], p_holm=None, corr=_r(res["corr"], 9), wins=res["wins"], losses=res["losses"], ties=res["ties"]))
    for env_id in {c["env_id"] for c in out}:
        part = [c for c in out if c["env_id"] == env_id]
        for c, p in zip(part, stats.holm([c["p"] for c in part])):
            c["p_holm"] = p
    return out


def floors(env_ids) -> dict:
    """{environment: {reference, no_scan, must_fail, ablations}}: the rows a published table of the environment must
    hold beside the models (R8), read from the family's control policies; None for an environment that is not
    registered or does not declare them. Only policies that play on the environment's own budget are listed."""
    import mrgym

    from .. import policies
    out = {}
    for env_id in env_ids:
        try:
            controls = policies.controls_of(mrgym.spec(env_id))
        except mrgym.Error:
            controls = None
        if controls is None or not hasattr(controls, "REFERENCE") or not hasattr(controls, "MUST_FAIL"):
            out[env_id] = None
            continue
        table = controls.policies()
        playable = [n for n, e in table.items() if not isinstance(e, tuple) or e[1] == 1.0]
        out[env_id] = dict(reference=controls.REFERENCE, no_scan="prior" if "prior" in playable else None,
                           must_fail=[n for n in controls.MUST_FAIL if n in playable],
                           ablations=[n for n in getattr(controls, "ABLATIONS", ()) if n in playable])
    return out


# ------------------------------------------------------------------ the summary
def summarise(records: list[dict], plan: dict | None = None, prices: dict | None = None, oracle: dict | None = None) -> dict:
    """The content of ``results.json``: provenance read from the records, one row per environment and agent, the
    planned comparisons and the reliability of the published rows."""
    cells = {}
    for r in records:
        cells.setdefault((r["env_id"], KINDS.index(r["agent_kind"]) if r["agent_kind"] in KINDS else len(KINDS), r["agent"], r["agent_kind"]), []).append(r)
    rows = [_row(env_id, agent, kind, recs, prices, oracle) for (env_id, _, agent, kind), recs in sorted(cells.items())]
    by_cell = {(env_id, agent): recs for (env_id, _, agent, _), recs in cells.items()}
    days = sorted({r["utc"][:10] for r in records})
    stamps = sorted(r["utc"] for r in records if r.get("utc"))
    settings = {}
    for r in records:
        if r.get("settings"):
            settings.setdefault(r["agent"], r["settings"])
    published = {(r["env_id"], r["agent"]) for r in rows if r["tier"] == "published"}
    pub_records = [r for r in records if (r["env_id"], r["agent"]) in published]
    return dict(
        schema=SCHEMA, canary=_record.CANARY, source=PATTERN.replace(os.sep, "/"), records=len(records),
        completed=sum(r["status"] == "completed" for r in records), infra_errors=sum(r["status"] != "completed" for r in records),
        published_records=len(pub_records), pilot_records=len(records) - len(pub_records),
        played_utc=[days[0], days[-1]] if days else [], records_utc=[stamps[0], stamps[-1]] if stamps else [],
        engines=sorted({r["engine"] for r in records}), package_versions=sorted({r["package_version"] for r in records}),
        design=dict(published_min_instances=power.PUBLISHED_MIN_INSTANCES, published_min_episodes=power.PUBLISHED_MIN_EPISODES,
                    pilot_below_instances=power.PILOT_BELOW, level=0.95, interval=stats.intervals.METHOD_RATE),
        plan_sha256=None if plan is None else plan.get("sha256"),
        plan=None if plan is None else dict(registered_utc=plan.get("registered_utc"), design=plan.get("design"),
                                            comparisons=[c["id"] for c in plan.get("comparisons") or []],
                                            alpha=plan.get("alpha"), correction=plan.get("correction"), mde_points=plan.get("mde_points")),
        environments={e: sorted({r["level"] for r in records if r["env_id"] == e}) for e in sorted({r["env_id"] for r in records})},
        agents={kind: sorted({r["agent"] for r in records if r["agent_kind"] == kind}) for kind in KINDS},
        floors=floors(sorted({r["env_id"] for r in records})), model_settings=settings, rows=rows, comparisons=comparisons(plan, rows, by_cell),
        reliability=reliability.report(pub_records))


# ------------------------------------------------------------------ text
def dateline(summary: dict) -> str:
    """One line: what was run, when, on which engine and package version, and how much of it is published."""
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
    return (f"{summary['records']} episode records, played {when} (UTC) on engine `{', '.join(summary['engines'])}`, package version "
            f"`{', '.join(summary['package_versions'])}`. Environments: {envs}. Agents: {who}. "
            f"Published: {summary['published_records']} records; pilot: {summary['pilot_records']} records.")


def _fmt(x, pattern: str = "{:g}") -> str:
    return "-" if x is None else pattern.format(x)


def _percent(x, digits: int = 0) -> str:
    if x is None:
        return "-"
    text = f"{100 * x:.{digits}f}"
    return (text.lstrip("-") if float(text) == 0 else text) + " %"       # never "-0.0 %"


def cell_text(row: dict) -> str:
    """A rate as ``k/N (p %; lo-hi)``; at 0 or 100 % the one-sided 95 % bound is added."""
    if not row["episodes"]:
        return "-"
    lo, hi = row["pass_ci95"]
    text = f"{row['passed']}/{row['episodes']} ({100 * row['pass_rate']:.1f} %; {100 * lo:.1f}-{100 * hi:.1f}"
    if row.get("one_sided_bound") is not None:
        text += f"; one-sided bound {'<=' if row['pass_rate'] == 0 else '>='} {100 * row['one_sided_bound']:.1f} %"
    return text + ")"


def _level_cells(row: dict) -> str:
    return "; ".join(f"L{lv}: {v['passed']}/{v['episodes']} ({100 * v['pass_ci95'][0]:.0f}-{100 * v['pass_ci95'][1]:.0f} %)" for lv, v in row["by_level"].items())


def _tokens_total(row: dict) -> str:
    if row["tokens_in_total"] is None:
        return "-"
    return f"{row['tokens_in_total'] / 1e6:.2f} M / {row['tokens_out_total'] / 1e6:.2f} M" + (" (est.)" if row["tokens_estimated"] else "")


def _tokens_median(row: dict) -> str:
    if row["tokens_in_median"] is None:
        return "-"
    return f"{row['tokens_in_median'] / 1e3:.1f}k / {row['tokens_out_median'] / 1e3:.1f}k" + (" (est.)" if row["tokens_estimated"] else "")


def _regret(row: dict) -> str:
    return "-" if row["regret_median"] is None else f"{100 * row['regret_median']:.2f} / {100 * row['regret_q95']:.2f} %"


PUBLISHED_HEADER = ("| Agent | Kind | Passed: k/N (rate; 95 % interval) | n_eff | pass^2 | Mean reward (95 % interval) | Regret: median / 95th pct "
                    "| By level (descriptive) | Tokens in / out, total | Wall, total | Cost | Infra errors |")
PILOT_HEADER = ("| Agent | Kind | Instances | Passed: k/N (rate; 95 % interval) | Mean reward | Median error | Median scans "
                "| Median turns | Tokens in / out, median | Infra errors | Why pilot |")
NOTES = ("How to read a row. The unit is the instance: N counts episodes, the interval counts instances. A rate is printed as "
         "`k/N (rate; 95 % interval)`; the interval is the Wilson interval on the Korn-Graubard effective sample size with a t "
         "quantile on instances minus strata (`mrgym.stats.cluster_rate`); a cell with fewer than 100 instances also gets the "
         "Beta-binomial interval and the wider of the two is printed. At 0 or 100 % the one-sided 95 % bound stands beside the "
         "rate. N counts every started episode except those that ended in an endpoint or runner failure (Infra errors); an "
         "episode without a delivery has reward 0. pass^2: share of instances whose episodes all pass. Regret: distance of the "
         "delivered answer from the oracle's optimum of the same instance (`-`: no oracle values stored). Per-level cells are "
         "descriptive: no comparison is computed from them. Agents are in alphabetical order within kind; the table does not "
         "rank. A difference that is not in the comparison table is not a finding.")


def _published_block(summary: dict, rows: list[dict]) -> list[str]:
    out = []
    for env_id in sorted({r["env_id"] for r in rows}):
        part = [r for r in rows if r["env_id"] == env_id]
        inst = max(r["instances"] for r in part)
        k = sorted({tuple(r["episodes_per_instance"]) for r in part if r["kind"] != "scripted"})
        out += [f"**`{env_id}`** ({inst} instances, levels {', '.join(map(str, part[0]['levels']))} as strata"
                + (f"; models play {k[0][0] if len(k) == 1 and k[0][0] == k[0][1] else 'a varying number of'} episodes per instance" if k else "")
                + "; scripted policies are deterministic and play one)", "", PUBLISHED_HEADER, "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
        for r in part:
            ci = "-" if r["reward_ci95"] is None else f" ({r['reward_ci95'][0]:.3f}-{r['reward_ci95'][1]:.3f})"
            out.append(f"| {r['agent']} | {r['kind']} | {cell_text(r)} | {r['n_eff']:.0f} | {_percent(r['pass_all_k'], 1)} | "
                       f"{_fmt(r['reward_mean'], '{:.3f}')}{ci} | {_regret(r)} | {_level_cells(r)} | {_tokens_total(r)} | "
                       f"{_fmt(r['wall_s_total'], '{:.0f} s')} | {_fmt(r['cost_usd'], '{:.2f} USD')} | {r['infra_errors']} |")
        out.append("")
        comps = [c for c in summary["comparisons"] if c["env_id"] == env_id]
        if comps:
            out += [f"Planned comparisons on `{env_id}` (paired on common instances; Holm within the environment):", "",
                    "| Id | A | B | Instances | A - B, points (95 % interval) | Test | p | p (Holm) | Correlation | Wins / losses / ties |",
                    "|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
            for c in comps:
                out.append(f"| {c['id']} | {c['a']} | {c['b']} | {c['instances']} | {100 * c['diff']:+.1f} ({100 * c['ci95'][0]:+.1f} to {100 * c['ci95'][1]:+.1f}) | "
                           f"{c['test']}{', one-sided' if c['alternative'] == 'greater' else ''} | {c['p']:.2g} | {c['p_holm']:.2g} | "
                           f"{_fmt(c['corr'], '{:.2f}')} | {c['wins']} / {c['losses']} / {c['ties']} |")
            out.append("")
    return out


def _pilot_block(rows: list[dict]) -> list[str]:
    out = []
    for env_id in sorted({r["env_id"] for r in rows}):
        part = [r for r in rows if r["env_id"] == env_id]
        out += [f"**`{env_id}`** (pilot; levels {', '.join(map(str, sorted({lv for r in part for lv in r['levels']})))} pooled as strata)", "",
                PILOT_HEADER, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in part:
            out.append(f"| {r['agent']} | {r['kind']} | {r['instances']} | {cell_text(r)} | {_fmt(r['reward_mean'], '{:.3f}')} | "
                       f"{_percent(r['error_median'], 1)} | {_fmt(r['scans_median'])} | {_fmt(r['turns_median'])} | {_tokens_median(r)} | "
                       f"{r['infra_errors']} | {r['tier_reason']} |")
        out.append("")
    return out


def markdown(summary: dict, agents=None) -> str:
    """The table block: the published table per environment with its planned comparisons, the pilot rows under their
    own heading, the reliability of the published rows, the design and the reading notes.

    ``agents`` restricts the rows to the named agents (for a short excerpt); None keeps all."""
    if not summary["records"]:
        return "No episode records yet.\n"
    rows = [r for r in summary["rows"] if agents is None or r["agent"] in agents]
    pub, pilot = [r for r in rows if r["tier"] == "published"], [r for r in rows if r["tier"] == "pilot"]
    out = ["### Published table", ""]
    if len(summary["engines"]) > 1 or len(summary["package_versions"]) > 1:
        out += ["Note: the records come from more than one engine or package version. Rows are comparable only within one.", ""]
    if pub:
        out += _published_block(summary, pub)
    else:
        d = summary["design"]
        out += [f"No row is published yet. A published row needs records of schema 2 (stored conversation, harness id, deliverable), "
                f"at least {d['published_min_instances']} instances per environment on the reporting instances of the analysis plan, "
                f"and {d['published_min_episodes']} episodes per instance for a model.", ""]
    if pilot:
        out += ["### Pilot rows (not published)", "",
                "Kept for the record. They carry intervals, but they are too small or lack what a published row must store, "
                "they are absent from the README table, and no comparison is computed from them. Why pilot: `schema 1` = the "
                f"records hold no stored conversation, harness id or deliverable; `< {summary['design']['pilot_below_instances']} "
                "instances` = too few instances for a comparison.", ""]
        out += _pilot_block(pilot)
    out += ["### Reliability of the published rows", "", reliability.markdown(summary["reliability"]).rstrip(), ""]
    plan = summary.get("plan")
    out += ["### Design and planned comparisons", "",
            "What the designs detect (points of pass rate; p = 0.5, two-sided alpha = 0.05, power 0.80; computed by `mrgym stats power`):", "",
            power.power_markdown(), ""]
    if plan:
        out += [f"Analysis plan registered {plan['registered_utc']} (sha256 `{summary['plan_sha256'][:12]}`): {len(plan['comparisons'])} planned "
                f"comparison(s), {plan['correction']} correction at alpha {plan['alpha']}; computed so far: {len(summary['comparisons'])}.", ""]
    else:
        out += ["No analysis plan is registered in this folder (`analysis_plan.json`): no comparison is computed.", ""]
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
