"""Reliability and discrimination of a set of episode records. Classical statistics only.

A two-parameter item-response fit is not offered: the published uses fit it on the results of many models, and with a
handful of agents its parameters are noise. Difficulty and discrimination below are the classical ones, they are
reported, and they are never used to drop an instance (filtering on the outcome is selection).

    icc_oneway          between-instance share of the variance of one agent's episodes: (MSB - MSW) / (MSB + (k0 - 1) MSW)
    retest_agreement    share of instances whose first two episodes agree, phi between them, pass^2 and pass-any
    separation          G = SD of the agents' pass rates corrected for their sampling error / RMS of their standard
                        errors; reliability = G^2 / (1 + G^2)
    cronbach_alpha      internal consistency of the agents x instances matrix of instance means (agents are the persons)
    split_half          correlation of the agents' scores on odd and even instances, with the Spearman-Brown step
    item_table          per instance: difficulty, discrimination (correlation with the agent's score on the other
                        instances), dead (every agent passes every episode, or none does)
    difficulty_drivers  logistic regression of pass on given columns, cluster-robust standard errors
    threshold_profile   share of episodes within tolerance tau, for tau on a grid
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as _st

from .intervals import cluster_rate, instance_means, wilson

MIN_AGENTS = 5                       # agents with a non-zero pass rate needed for separation, alpha and split-half
THRESHOLDS = dict(n_eff_min=200, separation_reliability_order=0.80, separation_reliability_adjacent=0.90,
                  alpha_min=0.80, dead_share_max=0.20, negative_discrimination_share_max=0.05)


def icc_oneway(y, instance) -> float | None:
    """One-way random-effects intraclass correlation (unbalanced form). None when it is not identified: one episode
    per instance, a single instance, or no variance at all."""
    y = np.asarray(y, float)
    _, mean, count, _ = instance_means(y, instance)
    m, n = len(mean), len(y)
    if m < 2 or n <= m:
        return None
    grand = y.mean()
    msb = float((count * (mean - grand) ** 2).sum()) / (m - 1)
    msw = float((y ** 2).sum() - (count * mean ** 2).sum()) / (n - m)
    k0 = (n - float((count ** 2).sum()) / n) / (m - 1)
    den = msb + (k0 - 1) * msw
    return None if den <= 0 else (msb - msw) / den


def icc_interval(y, instance, level: float = 0.95) -> tuple[float, float] | None:
    """Interval of ``icc_oneway`` from the F distribution of MSB / MSW (exact for a balanced design, approximate with
    the mean episode count otherwise)."""
    y = np.asarray(y, float)
    _, mean, count, _ = instance_means(y, instance)
    m, n = len(mean), len(y)
    if m < 2 or n <= m:
        return None
    grand = y.mean()
    msb = float((count * (mean - grand) ** 2).sum()) / (m - 1)
    msw = float((y ** 2).sum() - (count * mean ** 2).sum()) / (n - m)
    if msw <= 0 or msb <= 0:
        return None
    k0 = (n - float((count ** 2).sum()) / n) / (m - 1)
    f, a = msb / msw, (1 - level) / 2
    fl, fu = f / float(_st.f.ppf(1 - a, m - 1, n - m)), f / float(_st.f.ppf(a, m - 1, n - m))
    return (fl - 1) / (fl + k0 - 1), (fu - 1) / (fu + k0 - 1)


def retest_agreement(first, second) -> dict:
    """Two episodes per instance (0 / 1 arrays in instance order): ``{instances, agreement, phi, pass_both, pass_any}``."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    n = len(a)
    if n == 0:
        return dict(instances=0, agreement=None, phi=None, pass_both=None, pass_any=None)
    phi = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return dict(instances=n, agreement=float((a == b).mean()), phi=phi, pass_both=float((a * b).mean()),
                pass_any=float(np.maximum(a, b).mean()))


def separation(rates, ses, min_agents: int = MIN_AGENTS) -> dict:
    """How well a set of agents is told apart: ``{agents, g, reliability, estimable, reason}``.

    G = sqrt(max(0, variance of the observed rates - mean squared standard error)) / RMS standard error, the ratio of
    the spread of the true rates to the noise of one rate; reliability = G^2 / (1 + G^2). Not estimable with fewer
    than ``min_agents`` agents that pass anything."""
    rates, ses = np.asarray(rates, float), np.asarray(ses, float)
    live = int((rates > 0).sum())
    if live < min_agents:
        return dict(agents=len(rates), g=None, reliability=None, estimable=False,
                    reason=f"not estimable (< {min_agents} agents with a non-zero rate)")
    noise = float((ses ** 2).mean())
    if noise <= 0:
        return dict(agents=len(rates), g=None, reliability=None, estimable=False, reason="not estimable (no sampling error)")
    g = math.sqrt(max(0.0, float(rates.var(ddof=1)) - noise) / noise)
    return dict(agents=len(rates), g=g, reliability=g * g / (1 + g * g), estimable=True, reason=None)


def cronbach_alpha(matrix) -> float | None:
    """Cronbach's alpha of an agents x instances matrix (rows = agents, the persons; columns = instances, the items)."""
    x = np.asarray(matrix, float)
    a, i = x.shape
    if a < 2 or i < 2:
        return None
    total = float(x.sum(1).var(ddof=1))
    return None if total <= 0 else i / (i - 1) * (1 - float(x.var(0, ddof=1).sum()) / total)


def split_half(matrix) -> dict:
    """Odd / even split of the instances (columns in the plan's order): correlation of the agents' half scores and its
    Spearman-Brown step-up ``2 r / (1 + r)``."""
    x = np.asarray(matrix, float)
    if x.shape[0] < 3 or x.shape[1] < 2:
        return dict(r=None, spearman_brown=None)
    a, b = x[:, 0::2].mean(1), x[:, 1::2].mean(1)
    if a.std() == 0 or b.std() == 0:
        return dict(r=None, spearman_brown=None)
    r = float(np.corrcoef(a, b)[0, 1])
    return dict(r=r, spearman_brown=2 * r / (1 + r) if r > -1 else None)


def item_table(matrix, episodes=None) -> dict:
    """Per instance (column) of an agents x instances matrix of instance means: difficulty = 1 - mean over the agents,
    its Wilson interval on ``episodes`` (episodes per instance summed over agents; default: one per agent),
    discrimination = correlation between the instance's scores and the agents' mean score on the other instances
    (Madaan et al.), dead = every agent passes every episode or none does."""
    x = np.asarray(matrix, float)
    a, i = x.shape
    mean = x.mean(0)
    n = np.full(i, float(a)) if episodes is None else np.asarray(episodes, float) * np.ones(i)
    ci = [wilson((1 - p) * k, k) for p, k in zip(mean, n)]
    rest = (x.sum(1, keepdims=True) - x) / max(i - 1, 1)
    disc = []
    for j in range(i):
        ok = a > 2 and x[:, j].std() > 0 and rest[:, j].std() > 0
        disc.append(float(np.corrcoef(x[:, j], rest[:, j])[0, 1]) if ok else None)
    return dict(difficulty=(1 - mean).tolist(), difficulty_ci95=[list(c) for c in ci], discrimination=disc,
                dead=((mean <= 0) | (mean >= 1)).tolist())


def difficulty_drivers(X, y, cluster, names=None, level: float = 0.95, iterations: int = 50) -> dict | None:
    """Logistic regression of pass (0 / 1) on the columns of ``X`` (an intercept is added), by iteratively reweighted
    least squares, with cluster-robust (sandwich) standard errors over ``cluster``. Returns ``{names, coef, se, lo, hi,
    clusters, converged}`` or None when the fit is not identified (no variance in y, a singular design)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    if y.min() == y.max() or len(y) <= X.shape[1] + 1:
        return None
    A = np.c_[np.ones(len(y)), X]
    beta = np.zeros(A.shape[1])
    converged = False
    for _ in range(iterations):
        eta = np.clip(A @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        H = A.T @ (A * w[:, None])
        try:
            step = np.linalg.solve(H, A.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            converged = True
            break
    if not converged or np.abs(beta).max() > 25:                    # separation: the estimate runs away
        return None
    mu = 1 / (1 + np.exp(-(A @ beta)))
    bread = np.linalg.inv(A.T @ (A * (mu * (1 - mu))[:, None]))
    keys = [repr(c) for c in cluster]
    index = {k: i for i, k in enumerate(sorted(set(keys)))}
    g = len(index)
    scores = np.zeros((g, A.shape[1]))
    np.add.at(scores, [index[k] for k in keys], A * (y - mu)[:, None])
    cov = bread @ (scores.T @ scores) @ bread * (g / (g - 1) if g > 1 else 1.0)
    se = np.sqrt(np.diag(cov))
    q = float(_st.t.ppf(0.5 + level / 2, max(g - 1, 1)))
    return dict(names=["intercept"] + list(names or [f"x{i}" for i in range(X.shape[1])]), coef=beta.tolist(), se=se.tolist(),
                lo=(beta - q * se).tolist(), hi=(beta + q * se).tolist(), clusters=g, converged=True)


def threshold_profile(errors, valid, taus) -> list[float]:
    """Share of episodes with a valid deliverable and error <= tau, for each tau (the performance profile of the
    error). ``errors`` may hold None; ``valid`` says whether every check other than the tolerance passed."""
    e = np.array([math.inf if x is None else float(x) for x in errors])
    v = np.asarray(valid, bool)
    return [float(((e <= t) & v).mean()) if len(e) else None for t in taus]


# ------------------------------------------------------------------ a report from episode records
def _key(r: dict) -> str:
    return f"L{r['level']}:{r['seed']}"


def _tau0(recs: list[dict]) -> float | None:
    """The tolerance the records were scored with: the most frequent stored value; a factor above 1 (time <= 1.1 x
    the optimum) is read as factor - 1."""
    values = [r["tolerance"] for r in recs if isinstance(r.get("tolerance"), (int, float))]
    if not values:
        return None
    t = max(sorted(set(values)), key=values.count)
    return round(t - 1, 9) if t > 1 else t


def _cell(env_id: str, agent: str, kind: str, recs: list[dict]) -> dict:
    y = [float(bool(r["passed"])) for r in recs]
    keys, levels = [_key(r) for r in recs], [r["level"] for r in recs]
    strata = levels if len(set(levels)) > 1 else None
    rate = cluster_rate(y, keys, strata)
    icc, ci = icc_oneway(y, keys), icc_interval(y, keys)
    by = {}
    for r in sorted(recs, key=lambda r: r["episode"]):
        by.setdefault(_key(r), []).append(float(bool(r["passed"])))
    pairs = [v[:2] for _, v in sorted(by.items()) if len(v) >= 2]
    counts = [len(v) for v in by.values()]
    kbar = len(recs) / len(by)
    out = dict(env_id=env_id, agent=agent, kind=kind, instances=len(by), episodes=len(recs),
               episodes_per_instance=[min(counts), max(counts)], pass_rate=rate["rate"], se=rate["se"], n_eff=rate["n_eff"],
               icc=icc, icc_ci95=None if ci is None else list(ci), deff_from_icc=None if icc is None else 1 + (kbar - 1) * icc,
               retest=retest_agreement([p[0] for p in pairs], [p[1] for p in pairs]),
               all_fail_instances=sum(max(v) == 0 for v in by.values()), all_pass_instances=sum(min(v) == 1 for v in by.values()))
    tau0 = _tau0(recs)
    if tau0:
        valid = [bool(r.get("gates") or r.get("checks")) and all((r.get("gates") or r.get("checks") or {}).values()) for r in recs]
        out["threshold_profile"] = dict(tau0=tau0, multipliers=[0.5, 1.0, 1.5, 2.0],
                                        pass_share=threshold_profile([r.get("error") for r in recs], valid, [m * tau0 for m in (0.5, 1.0, 1.5, 2.0)]))
    else:
        out["threshold_profile"] = None
    return out


def _environment(env_id: str, cells: list[dict], recs: list[dict]) -> dict:
    """Panel statistics of the language-model rows of one environment, on the instances every model played."""
    models = sorted({r["agent"] for r in recs if r["agent_kind"] == "model"})
    by = {a: {} for a in models}
    for r in recs:
        if r["agent"] in by:
            by[r["agent"]].setdefault(_key(r), []).append(float(bool(r["passed"])))
    common = sorted(set.intersection(*[set(v) for v in by.values()])) if models else []
    rows = [c for c in cells if c["agent"] in models]
    out = dict(env_id=env_id, agents=models, instances_common=len(common),
               separation=separation([c["pass_rate"] for c in rows], [c["se"] for c in rows]))
    live = sum(c["pass_rate"] > 0 for c in rows)
    if common and models:
        matrix = np.array([[np.mean(by[a][k]) for k in common] for a in models])
        episodes = np.array([sum(len(by[a][k]) for a in models) for k in common])
        items = item_table(matrix, episodes)
        disc = [d for d in items["discrimination"] if d is not None]
        hist = np.histogram(items["difficulty"], bins=10, range=(0, 1))[0].tolist()
        negative = [k for k, d in zip(common, items["discrimination"]) if d is not None and d < 0]
        out["items"] = dict(instances=len(common), dead=int(sum(items["dead"])), dead_share=float(np.mean(items["dead"])),
                            dead_instances=[k for k, dead in zip(common, items["dead"]) if dead],
                            difficulty_histogram=hist,
                            discrimination_median=float(np.median(disc)) if disc else None,
                            discrimination_negative_share=float(np.mean([d < 0 for d in disc])) if disc else None,
                            discrimination_negative=negative)
        enough = live >= MIN_AGENTS
        out["cronbach_alpha"] = cronbach_alpha(matrix) if enough else None
        out["split_half"] = split_half(matrix) if enough else dict(r=None, spearman_brown=None)
    else:
        out.update(items=None, cronbach_alpha=None, split_half=dict(r=None, spearman_brown=None))
    levels = sorted({r["level"] for r in recs})
    panel = [r for r in recs if r["agent"] in models]
    drivers = None
    if len(levels) > 1 and panel:
        X = np.array([[float(r["level"] == lv) for lv in levels[1:]] for r in panel])
        drivers = difficulty_drivers(X, [float(bool(r["passed"])) for r in panel], [_key(r) for r in panel],
                                     names=[f"level {lv} against level {levels[0]}" for lv in levels[1:]])
    out["difficulty_drivers"] = drivers
    return out


def report(records: list[dict]) -> dict:
    """Reliability of the completed records given: one entry per (environment, agent) and one per environment.
    A pure function of the records."""
    done = [r for r in records if r.get("status") == "completed"]
    groups = {}
    for r in done:
        groups.setdefault((r["env_id"], r["agent_kind"], r["agent"]), []).append(r)
    cells = [_cell(env_id, agent, kind, recs) for (env_id, kind, agent), recs in sorted(groups.items())]
    envs = [_environment(e, [c for c in cells if c["env_id"] == e], [r for r in done if r["env_id"] == e])
            for e in sorted({r["env_id"] for r in done})]
    return dict(thresholds=THRESHOLDS, cells=cells, environments=envs)


def _f(x, pattern="{:.3f}") -> str:
    return "-" if x is None else pattern.format(x)


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def markdown(rep: dict) -> str:
    """The reliability section as Markdown (generated from ``report``)."""
    if not rep["cells"]:
        return "No records to compute reliability from.\n"
    out = ["| Environment | Agent | Instances x episodes | Pass rate | n_eff | ICC (95 % interval) | Retest agreement | phi | pass^2 "
           "| All-fail / all-pass instances |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in rep["cells"]:
        if c["kind"] == "scripted":
            continue
        k = c["episodes_per_instance"]
        ci = "" if c["icc_ci95"] is None else f" ({c['icc_ci95'][0]:.3f} to {c['icc_ci95'][1]:.3f})"
        rt = c["retest"]
        out.append(f"| `{c['env_id']}` | {c['agent']} | {c['instances']} x {k[0] if k[0] == k[1] else f'{k[0]}-{k[1]}'} | "
                   f"{_pct(c['pass_rate'])} | {c['n_eff']:.0f} | {_f(c['icc'])}{ci} | {_pct(rt['agreement'])} | "
                   f"{_f(rt['phi'], '{:.2f}')} | {_pct(rt['pass_both'])} | {c['all_fail_instances']} / {c['all_pass_instances']} |")
    out.append("")
    for e in rep["environments"]:
        sep = e["separation"]
        out.append(f"`{e['env_id']}`, {len(e['agents'])} language model(s) on {e['instances_common']} common instances. Separation: "
                   + (f"G = {sep['g']:.2f}, reliability {sep['reliability']:.2f}" if sep["estimable"] else sep["reason"]) + ". "
                   + f"Cronbach's alpha: {_f(e['cronbach_alpha'], '{:.2f}')}; split-half (Spearman-Brown): {_f(e['split_half']['spearman_brown'], '{:.2f}')}. "
                   + ("" if not e["items"] else f"Dead instances (every model passes every episode, or none does): {e['items']['dead']} of "
                      f"{e['items']['instances']} ({_pct(e['items']['dead_share'])}); negative discrimination: "
                      f"{_pct(e['items']['discrimination_negative_share'])}. "))
        d = e["difficulty_drivers"]
        if d:
            out.append("Level effect on the log-odds of passing (model panel, cluster-robust): "
                       + "; ".join(f"{n}: {c:+.2f} ({lo:+.2f} to {hi:+.2f})" for n, c, lo, hi in list(zip(d["names"], d["coef"], d["lo"], d["hi"]))[1:]) + ".")
        out.append("")
    for c in rep["cells"]:
        p = c.get("threshold_profile")
        if p and c["kind"] != "scripted":
            out.append(f"Tolerance sensitivity, {c['agent']} on `{c['env_id']}`: pass share at 0.5 / 1 / 1.5 / 2 times the tolerance "
                       f"({p['tau0']:g}): " + " / ".join(_pct(x) for x in p["pass_share"]) + ".")
    return "\n".join(out).rstrip() + "\n"


def text(rep: dict) -> str:
    """Plain-text form for the terminal (``mrgym stats reliability``)."""
    out = []
    for c in rep["cells"]:
        if c["kind"] == "scripted":
            continue
        rt = c["retest"]
        out.append(f"{c['env_id']}  {c['agent']}: {c['instances']} instances, {c['episodes']} episodes, pass {_f(c['pass_rate'], '{:.3f}')}, "
                   f"ICC {_f(c['icc'])}, n_eff {c['n_eff']:.0f}, retest agreement {_f(rt['agreement'])}, pass^2 {_f(rt['pass_both'])}, "
                   f"all-fail {c['all_fail_instances']}, all-pass {c['all_pass_instances']}")
    for e in rep["environments"]:
        sep = e["separation"]
        out.append(f"{e['env_id']}  separation: " + (f"G {sep['g']:.2f}, reliability {sep['reliability']:.2f}" if sep["estimable"] else sep["reason"])
                   + f"; alpha {_f(e['cronbach_alpha'], '{:.2f}')}"
                   + ("" if not e["items"] else f"; dead instances {e['items']['dead']} of {e['items']['instances']}"))
    return "\n".join(out) if out else "no records"
