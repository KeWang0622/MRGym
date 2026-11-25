"""``mrgym calibrate``: where on the ladder of an environment a model can start training.

    mrgym calibrate Scanner/NullTissue-v0 --model MODEL --base-url https://HOST/v1 --levels 0,1,2 --n 64 --episodes 8 --out runs/cal

For a model behind an OpenAI-compatible endpoint it plays ``n`` instances per level, ``episodes`` (the group size, 8)
times each, at the sampling temperature of training (1.0). ``--max-turns`` is the runner's: it becomes the tool-call
limit of the environment, which the task text and every status line state. Per level it reports

    pass                the pass rate over all completed episodes
    ci95                its 95 % interval by bootstrap over instances (episodes of one instance are not independent)
    mixed_group_share   the share of instances whose group holds both a pass and a failure: the groups with a gradient
    icc                 the intraclass correlation of the pass over instances (one-way ANOVA estimate)
    outcome_counts      how the episodes ended, in the environment's closed taxonomy
    band                dead < 2 % | frontier 2 - 20 % | entry 20 - 60 % | comfortable 60 - 80 % | mastered > 80 %

and which levels are in the entry band. The entry rung is the hardest level whose pass rate is in [0.20, 0.60] and whose
mixed-group share is at least 0.80. A run starts there. Below band a pass / fail reward gives almost no group with
spread; above it most groups are all passes.

The episodes go through ``mrgym.agents.runner`` (same harness, same records, resumable: repeat the command). Seeds are
integers of the public train split from ``--first-seed`` (9000): disjoint from the seeds a training sampler issues
(``mrgym.curriculum``: from 10,000) and from the seeds of the stored baselines.

``--write`` stores the rows as package data (``envs/results/<id>.calibration.json``). A card says a model class can
train on an environment only through that file (``card["learnability"]["trainable_by"]``); nothing is typed.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os

import numpy as np

from . import learn

SCHEMA = "mrgym-calibration/1"
FIRST_SEED = 9000
BOOTSTRAP = 2000


def read_records(out: str, env_id: str, model: str) -> list[dict]:
    """The completed episode records of ``model`` on ``env_id`` under the run folder ``out``."""
    from .agents import runner
    pattern = os.path.join(out, "episodes", runner.slug(model), runner.env_slug(env_id), "L*", "s*", "e*", "record.json")
    recs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("status") == "completed" and r.get("agent") == model:
            recs.append(r)
    return recs


def outcome_of(record: dict) -> str:
    """The outcome label of a record: the environment's own when the record has it, else from how the harness stopped."""
    if record.get("outcome"):
        return record["outcome"]
    if record.get("passed"):
        return "pass"
    return {"submitted": "check_failed", "truncated": "no_submission_cap", "rejected_call_limit": "no_submission_rejected",
            "deliverable_refused": "invalid_deliverable", "no_tool_calls": "no_submission_stopped"}.get(record.get("stop_reason"), "no_submission_stopped")


def icc(groups: list) -> float | None:
    """Intraclass correlation of a 0 / 1 outcome over groups (one-way ANOVA estimator); None when it is undefined."""
    groups = [np.asarray(g, float) for g in groups if len(g) > 1]
    n, total = len(groups), sum(len(g) for g in groups)
    if n < 2 or total <= n:
        return None
    grand = sum(g.sum() for g in groups) / total
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (n - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (total - n)
    k0 = (total - sum(len(g) ** 2 for g in groups) / total) / (n - 1)
    denom = msb + (k0 - 1.0) * msw
    return None if denom <= 0 else float((msb - msw) / denom)


def level_row(records: list, level: int, seed: int = 0) -> dict:
    """The calibration row of one level from its completed records (the measured fields only)."""
    by_seed = {}
    for r in records:
        if r["level"] == level:
            by_seed.setdefault(r["seed"], []).append(bool(r["passed"]))
    groups = [by_seed[s] for s in sorted(by_seed)]
    flat = [x for g in groups for x in g]
    if not flat:
        return dict(level=level, n_instances=0, n_episodes=0)
    rate = float(np.mean(flat))
    rng = np.random.default_rng(seed)
    sums, sizes = np.array([sum(g) for g in groups], float), np.array([len(g) for g in groups], float)
    pick = rng.integers(0, len(groups), size=(BOOTSTRAP, len(groups)))
    boot = sums[pick].sum(1) / sizes[pick].sum(1)
    full = [g for g in groups if len(g) > 1]
    counts = {}
    for r in records:
        if r["level"] == level:
            o = outcome_of(r)
            counts[o] = counts.get(o, 0) + 1
    rho = icc(groups)
    return dict(level=level, n_instances=len(groups), n_episodes=len(flat), k=int(max(len(g) for g in groups)), passed=int(sum(flat)),
                **{"pass": round(rate, 4)}, ci95=[round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)],
                mixed_group_share=round(float(np.mean([0 < sum(g) < len(g) for g in full])), 4) if full else None,
                icc=None if rho is None else round(rho, 4), outcome_counts=dict(sorted(counts.items())), band=learn.band(rate))


def decide(rows: list) -> dict:
    """Which levels are in the entry band, and the entry rung (None when no level qualifies)."""
    lo, hi = learn.ENTRY_BAND
    played = [r for r in rows if r.get("n_episodes")]
    in_band = [r["level"] for r in played if lo <= r["pass"] <= hi]
    ok = [r["level"] for r in played if r["level"] in in_band and (r["mixed_group_share"] or 0.0) >= learn.MIXED_MIN]
    rates = [r["pass"] for r in played]
    if ok:
        advice = f"start training at level {max(ok)}"
    elif played and all(p < lo for p in rates):
        advice = "every level is below the entry band: add an easier level (more hints, more acquired at reset) or warm-start the model"
    elif played and all(p > hi for p in rates):
        advice = "every level is above the entry band: the model has little to learn here; add a harder level"
    else:
        advice = "no level qualifies as an entry rung: the ladder has a gap between its levels for this model"
    return dict(in_band=in_band, entry_rung=max(ok) if ok else None, advice=advice)


def run(env_id: str, model: str, base_url: str, out: str, *, levels=None, n: int = 64, episodes: int = learn.GROUP_SIZE,
        first_seed: int = FIRST_SEED, temperature: float = 1.0, max_turns: int | None = 12, max_tokens: int | None = None,
        reasoning: str | None = None, workers: int = 4, api_key: str | None = None, quiet: bool = False, **runner_kw) -> dict:
    """Play the calibration episodes (resumable) and return the result: ``calibration`` rows and the decision."""
    import mrgym

    from .agents import runner
    spec = mrgym.spec(env_id)
    levels = [int(lv) for lv in (levels or spec.levels)]
    seeds = list(range(first_seed, first_seed + n))
    own = not max_turns                                                  # 0 or None: the environment's own tool-call limit
    done = runner.run(out, [env_id], levels=levels, seeds=seeds, episodes=episodes, models=[model], base_url=base_url, api_key=api_key,
                      max_turns=None if own else max_turns, temperature=temperature, max_tokens=max_tokens, workers=workers, quiet=quiet,
                      command=f"mrgym calibrate {env_id} --model {model} --levels {','.join(map(str, levels))} "
                                                     f"--n {n} --episodes {episodes} --first-seed {first_seed} --out {out}", **runner_kw)
    max_turns = spec.max_steps if own else max_turns                      # the rows state the limit that applied
    records = read_records(out, env_id, model)
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    settings = dict(model=model, reasoning=reasoning, temperature=temperature, max_turns=max_turns, max_tokens=max_tokens,
                    seeds=f"{seeds[0]}-{seeds[-1]}", episodes=episodes)
    run_id = hashlib.sha256(json.dumps([env_id, mrgym.ENGINE_ID, settings], sort_keys=True).encode()).hexdigest()[:12]
    rows = [dict(model=model, reasoning=reasoning, temperature=temperature, max_turns=max_turns, run_id=run_id, date=date,
                 engine=mrgym.ENGINE_ID, **level_row(records, lv)) for lv in levels]
    verdict = decide(rows)
    for r in rows:
        r["entry_rung"] = r["level"] == verdict["entry_rung"]
    return dict(schema=SCHEMA, env_id=env_id, engine=mrgym.ENGINE_ID, package_version=mrgym.__version__, date_utc=date, run_id=run_id,
                settings=settings, calibration=rows, infra_errors=done["infra_errors"], out=out, **verdict)


def write(result: dict, directory: str | None = None) -> str:
    """Store the rows of ``result``, replacing earlier rows of the same model, reasoning setting, tool-call limit and level.
    The stub endpoint's models are never stored as package data: they are not models."""
    if result["settings"]["model"].startswith("stub:") and directory is None:
        raise ValueError("a calibration of the stub endpoint is not stored with the package; give a directory")
    old = learn.load(result["env_id"], "calibration", directory) or dict(schema=SCHEMA, env_id=result["env_id"], calibration=[])
    key = lambda r: (r["model"], r["reasoning"], r["max_turns"], r["temperature"], r["level"])   # noqa: E731
    new = {key(r) for r in result["calibration"]}
    rows = [r for r in old["calibration"] if key(r) not in new] + [r for r in result["calibration"] if r.get("n_episodes")]
    return learn.write(dict(schema=SCHEMA, env_id=result["env_id"], calibration=rows), "calibration", directory)


def text_table(result: dict) -> str:
    s = result["settings"]
    out = [f"== calibration of {s['model']} on {result['env_id']} (engine {result['engine']}): temperature {s['temperature']}, tool-call limit "
           f"{s['max_turns']}, seeds {s['seeds']} x {s['episodes']} episodes, run {result['run_id']}",
           f"   {'level':>5} {'instances':>9} {'episodes':>8} {'pass':>7} {'95 % interval':>16} {'mixed groups':>12} {'ICC':>6}  {'band':<12} outcomes"]
    for r in result["calibration"]:
        if not r.get("n_episodes"):
            out.append(f"   {r['level']:>5} {'0':>9} {'0':>8}  no completed episode")
            continue
        mixed = "-" if r["mixed_group_share"] is None else f"{100 * r['mixed_group_share']:.0f} %"
        interval = f"{100 * r['ci95'][0]:.1f} .. {100 * r['ci95'][1]:.1f}"
        rho = "-" if r["icc"] is None else f"{r['icc']:.2f}"
        band = r["band"] + (" *" if r["entry_rung"] else "")
        out.append(f"   {r['level']:>5} {r['n_instances']:>9} {r['n_episodes']:>8} {100 * r['pass']:>6.1f}% {interval:>16} {mixed:>12} "
                   f"{rho:>6}  {band:<12} " + ", ".join(f"{k} {v}" for k, v in r["outcome_counts"].items()))
    out.append(f"   levels in the entry band (20 - 60 %): {result['in_band'] or 'none'}; entry rung (* = in band and at least "
               f"{100 * learn.MIXED_MIN:.0f} % mixed groups): {result['entry_rung'] if result['entry_rung'] is not None else 'none'}")
    out.append(f"   {result['advice']}" + (f"; {result['infra_errors']} episode(s) failed on the endpoint and are not counted" if result["infra_errors"] else ""))
    return "\n".join(out)


def add_parser(sub) -> None:
    p = sub.add_parser("calibrate", help="pass rate of a model endpoint per level, in groups of 8; which levels are in the entry band")
    p.add_argument("id")
    p.add_argument("--model", required=True, help="model id of the endpoint")
    p.add_argument("--base-url", required=True, help="base URL of an OpenAI-compatible endpoint, e.g. http://HOST:8000/v1")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable that holds the key (default OPENAI_API_KEY)")
    p.add_argument("--levels", default=None, help="comma-separated levels (default: every level of the id)")
    p.add_argument("--n", type=int, default=64, help="instances per level (default 64)")
    p.add_argument("--episodes", type=int, default=learn.GROUP_SIZE, help="episodes per instance: the group size (default 8)")
    p.add_argument("--first-seed", type=int, default=FIRST_SEED)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-turns", type=int, default=12, help="the runner's turn setting: the tool-call limit of every episode (default 12);\n"
                                                             "0 = the environment's own limit")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--reasoning", default=None, help="label of the model's reasoning setting, stored with the rows")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True, help="run folder (episode records and transcripts); repeat the command to resume")
    p.add_argument("--write", action="store_true", help="store the rows as package data (the evidence behind a card's trainable_by)")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_calibrate)


def cmd_calibrate(a) -> int:
    levels = [int(x) for x in a.levels.split(",")] if a.levels else None
    result = run(a.id, a.model, a.base_url, a.out, levels=levels, n=a.n, episodes=a.episodes, first_seed=a.first_seed,
                 temperature=a.temperature, max_turns=a.max_turns, max_tokens=a.max_tokens, reasoning=a.reasoning, workers=a.workers,
                 api_key=os.environ.get(a.api_key_env), quiet=a.json)
    print(json.dumps(result, indent=1) if a.json else text_table(result))
    if a.write:
        print("written:", write(result, a.results_dir))
    return 0 if result["entry_rung"] is not None else 1
