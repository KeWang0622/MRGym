"""What a design can detect, stated before the run. Closed forms only.

Variance of the paired difference of two cell means with M instances and K episodes each (Miller 2024, Eq. 9-10, for
a pass rate p, a between-instance share of variance rho and a correlation r of the two agents' instance effects):

    Var = 2 p (1 - p) (rho (1 - r) + (1 - rho) / K) / M

The detectable difference at two-sided level alpha (divided by the number of comparisons, the Bonferroni bound of
Holm's first step) and the given power is ``(z_alpha + z_power) * sqrt(Var)``. Instances are generated, so a new
instance costs what a repeat costs: for a fixed number of episodes K = 1 is the most precise design, and K = 2 is the
smallest that also gives retest agreement, the ICC and pass^2.
"""
from __future__ import annotations

import math

from scipy import stats as _st

from .gate import lower_bound, upper_bound

DESIGNS = ((10, 1, "none"), (24, 1, "none"), (96, 8, "pilot for rho"), (100, 2, "descriptive"), (300, 1, ""),
           (300, 2, "published minimum"), (600, 2, "trained against base"), (1500, 2, "suite score"))
PUBLISHED_MIN_INSTANCES, PUBLISHED_MIN_EPISODES, PILOT_BELOW = 300, 2, 100


def _z(alpha: float, power: float, comparisons: int) -> float:
    return float(_st.norm.ppf(1 - alpha / (2 * comparisons)) + _st.norm.ppf(power))


def _unit_var(K: int, p: float, rho: float, r: float) -> float:
    return 2 * p * (1 - p) * (rho * (1 - r) + (1 - rho) / K)


def mde_paired(M, K, p=0.5, rho=0.07, r=0.0, alpha=0.05, power=0.8, comparisons=1) -> float:
    """Smallest paired difference of two pass rates (as a fraction) that M instances x K episodes detect."""
    return _z(alpha, power, comparisons) * math.sqrt(_unit_var(K, p, rho, r) / M)


def instances_needed(delta, K, p=0.5, rho=0.07, r=0.0, alpha=0.05, power=0.8, comparisons=1) -> int:
    """Instances needed to detect a paired difference ``delta`` with K episodes per instance."""
    return math.ceil(_z(alpha, power, comparisons) ** 2 * _unit_var(K, p, rho, r) / delta ** 2)


def halfwidth(M, K, p=0.5, rho=0.07, level=0.95) -> float:
    """Half-width of the interval of one pass rate: t quantile on M - 1 times the clustered standard error."""
    return float(_st.t.ppf(0.5 + level / 2, M - 1)) * math.sqrt(p * (1 - p) * (rho + (1 - rho) / K) / M)


def icc_se(M, K, rho=0.07) -> float | None:
    """Approximate standard error of the one-way ICC (Fisher); None for K = 1, where it is not identified."""
    return None if K < 2 else (1 - rho) * (1 + (K - 1) * rho) * math.sqrt(2 / (K * (K - 1) * (M - 1)))


def design_row(M, K, rho=0.07, r=0.0, comparisons=10, rho_alt=0.30, r_alt=0.5) -> dict:
    """One row of the power table, in points (percent of pass rate)."""
    return dict(instances=M, episodes=K, halfwidth=100 * halfwidth(M, K, rho=rho), mde=100 * mde_paired(M, K, rho=rho, r=r),
                mde_holm=100 * mde_paired(M, K, rho=rho, r=r, comparisons=comparisons), comparisons=comparisons,
                mde_alt=100 * mde_paired(M, K, rho=rho_alt, r=r_alt), rho=rho, r=r, rho_alt=rho_alt, r_alt=r_alt)


def cleared_at_most(n: int, alpha: float = 0.05, s_max: float = 0.10) -> int:
    """Largest number of passes with which a must-fail policy is cleared (one-sided upper bound <= ``s_max``); -1 if
    not even zero passes clear it."""
    k = -1
    while k + 1 <= n and upper_bound(k + 1, n, alpha) <= s_max:
        k += 1
    return k


def reference_at_least(n: int, alpha: float = 0.05, ref_min: float = 0.90) -> int | None:
    """Smallest number of passes with which the reference is accepted (one-sided lower bound >= ``ref_min``)."""
    return next((k for k in range(n + 1) if lower_bound(k, n, alpha) >= ref_min), None)


def gate_power(n: int, rates=(0.02, 0.03, 0.05, 0.07, 0.10, 0.12), alpha=0.05, s_max=0.10, ref_min=0.90,
               reference_rates=(0.90, 0.95, 0.97, 0.99)) -> dict:
    """Operating characteristic of the admission gate at ``n`` instances, by exact binomial: the pass count that
    clears a must-fail policy and the probability of clearing it at each true pass rate; the pass count the reference
    needs and the probability of reaching it at each true rate."""
    k = cleared_at_most(n, alpha, s_max)
    need = reference_at_least(n, alpha, ref_min)
    return dict(n=n, alpha=alpha, s_max=s_max, ref_min=ref_min, cleared_at_most=k,
                p_cleared={p: float(_st.binom.cdf(k, n, p)) if k >= 0 else 0.0 for p in rates},
                reference_at_least=need,
                p_reference={p: float(_st.binom.sf(need - 1, n, p)) if need is not None else 0.0 for p in reference_rates})


def ablation_gap_bound(n: int, p_ref: float, p_abl: float, alpha: float = 0.05) -> float:
    """Expected one-sided lower bound of the paired gap reference - ablation when the two outcomes are independent
    given the instance (the conservative case: positive dependence makes the bound larger)."""
    sd = math.sqrt(p_ref * (1 - p_ref) + p_abl * (1 - p_abl))
    return (p_ref - p_abl) - float(_st.t.ppf(1 - alpha, n - 1)) * sd / math.sqrt(n)


# ------------------------------------------------------------------ text
def power_text(M: int, K: int, rho=0.07, r=0.0, comparisons=10) -> str:
    d = design_row(M, K, rho, r, comparisons)
    se = icc_se(M, K, rho)
    lines = [
        f"design: {M} instances x {K} episode{'s' if K != 1 else ''} = {M * K} episodes per cell; p = 0.5 (worst case), rho = {rho:g}, r = {r:g}, "
        "two-sided alpha = 0.05, power 0.80",
        f"95 % half-width of one pass rate            {d['halfwidth']:5.1f} points",
        f"detectable paired difference                {d['mde']:5.1f} points",
        f"same, Holm over {comparisons} comparisons{'':<{max(0, 13 - len(str(comparisons)))}}{d['mde_holm']:5.1f} points",
        f"instances needed at K = {K} for 5 / 8 / 10 / 15 points: "
        + " / ".join(str(instances_needed(x, K, rho=rho, r=r)) for x in (0.05, 0.08, 0.10, 0.15)),
        "standard error of the ICC                   " + ("not identified at K = 1" if se is None else f"{se:5.3f}"),
    ]
    if M < PILOT_BELOW:
        lines.append(f"tier: pilot (fewer than {PILOT_BELOW} instances): no comparison is computed from this cell")
    elif M < PUBLISHED_MIN_INSTANCES or K < PUBLISHED_MIN_EPISODES:
        lines.append(f"tier: below the published minimum of {PUBLISHED_MIN_INSTANCES} instances x {PUBLISHED_MIN_EPISODES} episodes")
    else:
        lines.append("tier: meets the published minimum")
    return "\n".join(lines)


def power_markdown(rho=0.07, r=0.0, comparisons=10, rho_alt=0.30, r_alt=0.5, designs=DESIGNS) -> str:
    """The power table of the design, as Markdown (generated; never typed)."""
    out = [f"| Design (instances x episodes) | 95 % half-width of one rate | Detectable paired difference, rho = {rho:g}, r = {r:g} "
           f"| same, Holm over {comparisons} | rho = {rho_alt:g}, r = {r_alt:g} | Use |", "|---|---:|---:|---:|---:|---|"]
    for M, K, use in designs:
        d = design_row(M, K, rho, r, comparisons, rho_alt, r_alt)
        out.append(f"| {M} x {K} | {d['halfwidth']:.1f} | {d['mde']:.1f} | {d['mde_holm']:.1f} | {d['mde_alt']:.1f} | {use} |")
    return "\n".join(out)


def gate_text(ns=(100, 300, 1000), alpha=0.05, s_max=0.10, ref_min=0.90) -> str:
    """The operating-characteristic table of the admission gate (``mrgym stats power --gate``)."""
    rows = [gate_power(n, alpha=alpha, s_max=s_max, ref_min=ref_min) for n in ns]
    rates = list(rows[0]["p_cleared"])
    out = [f"admission gate, alpha = {alpha:g}: a must-fail policy is cleared when its one-sided upper bound is <= {s_max:g}; "
           f"the reference is accepted when its lower bound is >= {ref_min:g} (Clopper-Pearson, exact binomial)", "",
           "P(a must-fail policy is cleared | its true pass rate)",
           f"{'n':>6} {'cleared at':>11} " + " ".join(f"{100 * p:>6.0f} %" for p in rates)]
    for g in rows:
        out.append(f"{g['n']:>6} {'<= ' + str(g['cleared_at_most']):>11} " + " ".join(f"{g['p_cleared'][p]:>8.3f}" for p in rates))
    refs = list(rows[0]["p_reference"])
    out += ["", "P(the reference is accepted | its true pass rate)",
            f"{'n':>6} {'needs':>11} " + " ".join(f"{100 * p:>6.0f} %" for p in refs)]
    for g in rows:
        out.append(f"{g['n']:>6} {'>= ' + str(g['reference_at_least']):>11} " + " ".join(f"{g['p_reference'][p]:>8.3f}" for p in refs))
    out += ["", "expected lower bound of the gap reference - ablation, reference at 98 %, independent outcomes (points)",
            f"{'n':>6} " + " ".join(f"{'abl ' + str(int(100 * p)) + ' %':>10}" for p in (0.5, 0.6, 0.7))]
    for n in ns:
        out.append(f"{n:>6} " + " ".join(f"{100 * ablation_gap_bound(n, 0.98, p, alpha):>10.1f}" for p in (0.5, 0.6, 0.7)))
    return "\n".join(out)
