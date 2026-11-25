"""Comparisons of two agents. A cross-agent claim is a paired difference on the instances both played.

    d_i = ybar_A,i - ybar_B,i      difference of the instance means
    estimate, SE, interval         as ``cluster_mean`` with d in place of ybar (strata = levels, t on M - H)
    p-value                        both agents played one episode per instance: exact McNemar on the discordant
                                   instances; otherwise a t test on d
    always returned                Pearson correlation of the instance means, wins / losses / ties

Multiplicity: ``holm`` over the family of comparisons fixed in the analysis plan.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as _st

from .intervals import _stratified, cluster_rate


def mcnemar_exact(b: int, c: int, alternative: str = "two-sided") -> float:
    """Exact McNemar test on the discordant instances: ``b`` where A passed and B failed, ``c`` the reverse.
    Two-sided binomial p-value by default; ``alternative="greater"`` tests A > B."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    if alternative == "greater":
        return float(_st.binom.sf(int(b) - 1, n, 0.5))
    return float(min(1.0, 2 * _st.binom.cdf(min(int(b), int(c)), n, 0.5)))


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the order given."""
    order = sorted(range(len(p)), key=lambda i: p[i])
    out, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def paired_compare(a: dict, b: dict, stratum: dict | None = None, level: float = 0.95, *,
                   alternative: str = "two-sided", binary: bool | None = None) -> dict:
    """A minus B on the instances both played. ``a`` and ``b`` are ``{instance: mean score}``; ``stratum`` is
    ``{instance: stratum}`` or None.

    Returns ``{diff, lo, hi, se, instances, p, test, corr, wins, losses, ties, alternative}``. ``binary`` says whether
    each agent has one 0/1 outcome per instance (then the test is McNemar's); None reads it from the values.
    ``alternative="greater"`` gives the one-sided p-value of A > B (a model against a floor); the interval stays
    two-sided."""
    common = sorted(set(a) & set(b), key=repr)
    m = len(common)
    if m == 0:
        return dict(diff=None, lo=None, hi=None, se=None, instances=0, p=None, test=None, corr=None, wins=0, losses=0,
                    ties=0, alternative=alternative)
    xa, xb = np.array([float(a[k]) for k in common]), np.array([float(b[k]) for k in common])
    d = xa - xb
    strata = [None if stratum is None else stratum[k] for k in common]
    diff, var, _, h = _stratified(d, strata)
    se, dof = math.sqrt(var), m - h
    half = float(_st.t.ppf(0.5 + level / 2, dof)) * se if dof >= 1 else math.inf
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    if binary is None:
        binary = bool(np.isin(xa, (0.0, 1.0)).all() and np.isin(xb, (0.0, 1.0)).all())
    if binary:
        test, p = "mcnemar-exact", mcnemar_exact(wins, losses, alternative)
    elif se > 0 and dof >= 1:
        t = diff / se
        test, p = "paired-t", float(_st.t.sf(t, dof) if alternative == "greater" else 2 * _st.t.sf(abs(t), dof))
    else:                                                               # every difference equal: no variance to test against
        test, p = "sign-exact", mcnemar_exact(wins, losses, alternative)   # two equal gains on two instances are p = 0.5, not 0
    corr = float(np.corrcoef(xa, xb)[0, 1]) if m > 2 and xa.std() > 0 and xb.std() > 0 else None
    return dict(diff=diff, lo=diff - half, hi=diff + half, se=se, instances=m, p=p, test=test, corr=corr, wins=wins,
                losses=losses, ties=m - wins - losses, alternative=alternative)


def unpaired_cluster_diff(ya, instance_a, yb, instance_b, level: float = 0.95) -> dict:
    """Difference of two pass rates measured on different instances (two levels, two environments): A minus B with
    the clustered standard errors combined and Welch-Satterthwaite degrees of freedom. Descriptive only: the planned
    comparisons of a results table are paired."""
    ra, rb = cluster_rate(ya, instance_a, level=level), cluster_rate(yb, instance_b, level=level)
    se = math.hypot(ra["se"], rb["se"])
    diff = ra["rate"] - rb["rate"]
    if se == 0:
        return dict(diff=diff, lo=diff, hi=diff, se=0.0, dof=None, p=None if diff == 0 else 0.0)
    dof = se ** 4 / sum(r["se"] ** 4 / (r["instances"] - 1) for r in (ra, rb) if r["instances"] > 1 and r["se"] > 0)
    half = float(_st.t.ppf(0.5 + level / 2, dof)) * se
    return dict(diff=diff, lo=diff - half, hi=diff + half, se=se, dof=float(dof), p=float(2 * _st.t.sf(abs(diff / se), dof)))
