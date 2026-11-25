"""Intervals for a rate and for a mean. The unit is the instance, never the episode.

``cluster_rate`` is the one interval a pass rate is printed with. It is the Wilson interval on the Korn-Graubard
effective sample size with a t quantile (what Inspect calls ``ci_wilson(cluster=)``; Franco et al. 2019 for the
justification), with one deliberate change: when every episode passes or every episode fails, the effective sample
size falls back to the number of instances, not of episodes, because eight episodes of one instance are not eight
units.

    instance mean        ybar_i   = mean of the K_i episodes of instance i
    estimate             p        = sum_h w_h * mean_{i in h} ybar_i                  (strata h, weights 1/H)
    variance             v        = sum_h w_h^2 * s_h^2 / M_h,   s_h^2 = sample variance of ybar_i in stratum h
    effective size       n_eff    = min(episodes, p (1 - p) / v);   p in {0, 1} or v = 0:  n_eff = M
    interval             Wilson(k = p * n_eff, n = n_eff) with the t quantile on M - H degrees of freedom
    design effect        deff     = episodes / n_eff

Nothing here draws a random number: the stored table can be compared byte for byte with a fresh build.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special
from scipy import stats as _st

METHOD_RATE = "wilson-kg-t"
METHOD_MEAN = "t-instance-means"


def wilson(k: float, n: float, level: float = 0.95, crit: float | None = None) -> tuple[float, float]:
    """Wilson score interval of a proportion ``k / n``; ``k`` and ``n`` may be effective (non-integer) counts.
    ``crit`` replaces the normal quantile (a t quantile for a clustered sample). (0, 1) for n <= 0."""
    if n <= 0:
        return 0.0, 1.0
    z = float(_st.norm.ppf(0.5 + level / 2)) if crit is None else float(crit)
    p = min(1.0, max(0.0, k / n))
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson(k: int, n: int, level: float = 0.95, side: str = "two") -> tuple[float, float]:
    """Exact binomial interval. ``side``: "two" (both tails at (1 - level) / 2), "upper" (0, one-sided upper bound),
    "lower" (one-sided lower bound, 1)."""
    if side not in ("two", "upper", "lower"):
        raise ValueError("side is two, upper or lower")
    if n <= 0:
        return 0.0, 1.0
    a = (1 - level) / 2 if side == "two" else 1 - level
    lo = 0.0 if k <= 0 or side == "upper" else float(_st.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k >= n or side == "lower" else float(_st.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def instance_means(values, instance, stratum=None):
    """(keys, means, episode counts, stratum per key) with one row per instance, in sorted key order.
    Keys are compared as strings, so (level, seed) tuples and string seeds both work."""
    values = np.asarray(values, float)
    names = [repr(k) if not isinstance(k, str) else k for k in instance]
    if len(names) != len(values):
        raise ValueError("one instance key per value")
    keys = sorted(set(names))
    index = {k: i for i, k in enumerate(keys)}
    inv = np.array([index[k] for k in names], int)
    count = np.bincount(inv, minlength=len(keys)).astype(float)
    mean = np.bincount(inv, values, len(keys)) / np.maximum(count, 1)
    strata = [None] * len(keys)
    if stratum is not None:
        for k, s in zip(names, stratum):
            if strata[index[k]] not in (None, s):
                raise ValueError(f"instance {k} lies in two strata")
            strata[index[k]] = s
    return keys, mean, count, strata


def _stratified(mean: np.ndarray, strata: list) -> tuple[float, float, int, int]:
    """(estimate, variance of the estimate, instances M, strata H) from instance means."""
    labels = sorted(set(strata), key=repr)
    h = len(labels)
    est = var = 0.0
    for lab in labels:
        m = mean[[s == lab for s in strata]]
        est += m.mean() / h
        if len(m) > 1:
            var += m.var(ddof=1) / len(m) / h ** 2
    return float(est), float(var), len(mean), h


def cluster_rate(y, instance, stratum=None, level: float = 0.95) -> dict:
    """Pass rate of a cell with its interval; ``y`` in {0, 1} per episode, ``instance`` the instance key per episode,
    ``stratum`` the stratum (level) per episode or None for one stratum.

    Returns ``{rate, lo, hi, method, episodes, instances, se, n_eff, deff, one_sided_bound}``. ``one_sided_bound`` is
    None unless the rate is 0 (then the one-sided upper bound ``1 - alpha^(1/M)``) or 1 (the lower bound
    ``alpha^(1/M)``), alpha = 1 - level: what "0 %" and "100 %" are allowed to mean."""
    y = np.asarray(y, float)
    if not len(y):
        return dict(rate=None, lo=0.0, hi=1.0, method=METHOD_RATE, episodes=0, instances=0, se=None, n_eff=0.0, deff=None,
                    one_sided_bound=None)
    _, mean, _, strata = instance_means(y, instance, stratum)
    p, v, m, h = _stratified(mean, strata)
    p = min(1.0, max(0.0, p))
    n = len(y)
    degenerate = p <= 0.0 or p >= 1.0 or v <= 0.0
    n_eff = float(m) if degenerate else min(float(n), p * (1 - p) / v)
    dof = m - h
    crit = float(_st.t.ppf(0.5 + level / 2, dof)) if dof >= 1 else math.inf
    lo, hi = wilson(p * n_eff, n_eff, crit=crit) if math.isfinite(crit) else (0.0, 1.0)
    alpha = 1 - level
    bound = 1 - alpha ** (1 / m) if p <= 0.0 else alpha ** (1 / m) if p >= 1.0 else None
    return dict(rate=p, lo=lo, hi=hi, method=METHOD_RATE, episodes=n, instances=m, se=math.sqrt(v), n_eff=n_eff,
                deff=n / n_eff, one_sided_bound=bound)


def cluster_mean(x, instance, stratum=None, level: float = 0.95) -> dict:
    """Mean of a continuous quantity (reward, error, tokens) with a t interval on instance means, same strata.
    Returns ``{mean, lo, hi, se, method, episodes, instances}``."""
    x = np.asarray(x, float)
    if not len(x):
        return dict(mean=None, lo=None, hi=None, se=None, method=METHOD_MEAN, episodes=0, instances=0)
    _, mean, _, strata = instance_means(x, instance, stratum)
    est, v, m, h = _stratified(mean, strata)
    se = math.sqrt(v)
    dof = m - h
    half = float(_st.t.ppf(0.5 + level / 2, dof)) * se if dof >= 1 else math.inf
    return dict(mean=est, lo=est - half, hi=est + half, se=se, method=METHOD_MEAN, episodes=len(x), instances=m)


def beta_binomial_interval(passes_per_instance, episodes_per_instance, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed credible interval of the pass rate under the clustered model of Bowyer et al. (Eq. 8):

        d ~ Gamma(1, 1),  theta ~ Beta(1, 1),  theta_i ~ Beta(d theta, d (1 - theta)),  Y_i ~ Binomial(K_i, theta_i)

    with theta_i integrated out (Y_i is Beta-binomial). The posterior of theta is integrated on a fixed grid of
    200 values of theta and 60 quantiles of the prior of d: no random numbers. A cross-check for cells with fewer
    than 100 instances, where the effective sample size of ``cluster_rate`` is itself uncertain."""
    y = np.asarray(passes_per_instance, float)
    k = np.asarray(episodes_per_instance, float) * np.ones_like(y)
    theta = (np.arange(200) + 0.5) / 200
    d = -np.log1p(-(np.arange(60) + 0.5) / 60)                       # quantiles of Gamma(1, 1), equal prior weight
    a = d[None, :, None] * theta[:, None, None]                      # [theta, d, 1]
    b = d[None, :, None] * (1 - theta[:, None, None])
    log_like = (special.betaln(a + y, b + k - y) - special.betaln(a, b)).sum(-1)       # binomial coefficients cancel
    log_like -= log_like.max()
    post = np.exp(log_like).sum(1)
    cdf = np.cumsum(post) / post.sum()
    edges = np.arange(1, 201) / 200                                   # cdf[i] is the mass up to the right edge of cell i
    tail = (1 - level) / 2
    lo = float(np.interp(tail, np.r_[0.0, cdf], np.r_[0.0, edges]))
    hi = float(np.interp(1 - tail, np.r_[0.0, cdf], np.r_[0.0, edges]))
    return lo, hi
