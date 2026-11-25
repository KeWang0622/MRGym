"""Stop rules that keep their error rate when the data are looked at more than once.

``betting_test``  early stop inside one evaluation (Waudby-Smith & Ramdas, testing by betting). The instances come in
                  the fixed order of the analysis plan; x_i in [0, 1]; H0: mean <= m. Wealth

                      W_t = prod_{i <= t} (1 + lambda_i (x_i - m)),   lambda_i from the instances before i only

                  is a non-negative supermartingale under H0, so by Ville's inequality P(sup_t W_t >= 1 / alpha)
                  <= alpha at any stopping time and for any such lambda; the choice of lambda moves the power only.
                  For a paired comparison x_i = (d_i + 1) / 2 with d_i the difference of the instance means, m = 1/2.
``alpha_spend``   stop rule across checkpoints: at most J looks declared in advance, each a one-sided paired test at
                  alpha / J. ``checkpoint_look`` applies it and the futility rule of the plan.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as _st


def betting_test(x, m: float = 0.5, alpha: float = 0.05, cap: float = 0.5) -> dict:
    """One-sided sequential test of H0: mean(x) <= m for x in [0, 1].

    ``lambda_i = clip((mu - m) / (var + (mu - m)^2), 0, cap / m)`` with ``mu`` and ``var`` the running mean and
    variance of the instances before i (started at m and 1/4). Returns ``{reject, stopped_at, wealth, max_wealth,
    n, threshold}``; ``stopped_at`` is the 1-based index at which the wealth first reached 1 / alpha, else None."""
    x = np.asarray(x, float)
    if len(x) and (x.min() < 0 or x.max() > 1):
        raise ValueError("x must lie in [0, 1]")
    threshold, top = 1 / alpha, cap / m
    log_w, best, stopped = 0.0, 0.0, None
    total, total_sq = m, 0.25 + m * m                        # one pseudo-observation at m with variance 1/4
    for i, v in enumerate(x):
        mu = total / (i + 1)
        var = max(total_sq / (i + 1) - mu * mu, 1e-12)
        lam = min(max((mu - m) / (var + (mu - m) ** 2), 0.0), top)
        log_w += math.log1p(lam * (v - m))
        best = max(best, log_w)
        if stopped is None and log_w >= math.log(threshold):
            stopped = i + 1
        total += v
        total_sq += v * v
    return dict(reject=stopped is not None, stopped_at=stopped, wealth=math.exp(log_w), max_wealth=math.exp(best), n=len(x),
                threshold=threshold)


def alpha_spend(alpha: float = 0.05, looks: int = 1) -> list[float]:
    """Level of each of ``looks`` pre-declared looks (Bonferroni: alpha / J each; valid for any dependence)."""
    if looks < 1:
        raise ValueError("at least one look")
    return [alpha / looks] * looks


def checkpoint_look(d, alpha: float = 0.05, looks: int = 1, mde: float | None = None) -> dict:
    """One look at a checkpoint against the base: ``d`` are the paired differences of the instance means (checkpoint
    minus base) on the validation instances. ``efficacy``: the one-sided lower bound at 1 - alpha / J is above 0.
    ``futility``: the one-sided upper bound at 1 - alpha / J is below ``mde`` (the plan's smallest difference worth
    detecting). Returns ``{diff, lower, upper, level, efficacy, futility, instances}``."""
    d = np.asarray(d, float)
    n, a = len(d), alpha_spend(alpha, looks)[0]
    se = float(d.std(ddof=1)) / math.sqrt(n) if n > 1 else math.inf
    q = float(_st.t.ppf(1 - a, n - 1)) if n > 1 else math.inf
    half = 0.0 if se == 0 else q * se
    diff = float(d.mean())
    return dict(diff=diff, lower=diff - half, upper=diff + half, level=1 - a, efficacy=bool(diff - half > 0),
                futility=bool(mde is not None and diff + half < mde), instances=n)
