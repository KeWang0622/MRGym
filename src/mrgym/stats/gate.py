"""The statistics of the admission gate: one verdict per (environment, level), with a stated error rate.

    for each must-fail (tier-1) member c:   k_c passes of N;  U_c = Clopper-Pearson one-sided upper bound at 1 - alpha
    for each ablation a:                    d_i = ref_i - a_i;  D_a = mean(d) - t_{1 - alpha, N - 1} sd(d) / sqrt(N)
    reference:                              L_ref = Clopper-Pearson one-sided lower bound at 1 - alpha
    ADMIT  <=>  max_c U_c <= s_max   and   min_a D_a >= sep_min   and   L_ref >= ref_min

Error rate. If one member truly passes more than ``s_max``, P(admit) <= P(U_c <= s_max) <= alpha; the same holds for
a reference below ``ref_min`` and for an ablation closer than ``sep_min``. Every condition must hold, so the decision
needs no correction over the members (an intersection-union test): a family that violates any one condition is
admitted with probability at most alpha. The *reported* ceiling of the blind policies is another statement and is
corrected: ``shortcut_ceiling`` uses bounds at 1 - alpha / J for J members.

``mrgym.gate`` (the runner of the gate and its file) imports the three bound functions from here.
"""
from __future__ import annotations

import functools

import numpy as np
from scipy import stats as _st

ALPHA, REF_MIN, S_MAX, SEP_MIN = 0.05, 0.90, 0.10, 0.25


@functools.lru_cache(maxsize=65536)
def _upper(k: int, n: int, alpha: float) -> float:
    return 1.0 if k >= n else float(_st.beta.ppf(1 - alpha, k + 1, n - k))


@functools.lru_cache(maxsize=65536)
def _lower(k: int, n: int, alpha: float) -> float:
    return 0.0 if k <= 0 else float(_st.beta.ppf(alpha, k, n - k + 1))


def upper_bound(k: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided Clopper-Pearson upper bound of a binomial rate."""
    return _upper(int(k), int(n), float(alpha))


def lower_bound(k: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided Clopper-Pearson lower bound of a binomial rate."""
    return _lower(int(k), int(n), float(alpha))


def gap_lower_bound(ref, abl, alpha: float = ALPHA) -> tuple[float, float]:
    """(mean, one-sided lower bound) of the paired difference in pass rate, reference minus ablation."""
    d = np.asarray(ref, float) - np.asarray(abl, float)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return float(d.mean()), float(d.mean() - _st.t.ppf(1 - alpha, n - 1) * sd / np.sqrt(n)) if n > 1 else float(d.mean())


def shortcut_ceiling(tier1: dict, alpha: float = ALPHA) -> dict:
    """The reported ceiling of the policies that ignore the object: the member with the largest upper bound, its rate,
    its bound at 1 - alpha, and the bound that holds for all J members at once (1 - alpha / J)."""
    if not tier1:
        return dict(member=None, passed=None, rate=None, upper95=None, upper95_simultaneous=None, members=0)
    counts = {name: (int(np.asarray(v, bool).sum()), len(v)) for name, v in tier1.items()}
    worst = max(sorted(counts), key=lambda name: upper_bound(*counts[name], alpha))
    k, n = counts[worst]
    return dict(member=worst, passed=k, rate=k / n, upper95=upper_bound(k, n, alpha),
                upper95_simultaneous=upper_bound(k, n, alpha / len(counts)), members=len(counts))


def family_gate(tier1: dict[str, np.ndarray], reference: np.ndarray, ablations: dict[str, np.ndarray],
                alpha: float = ALPHA, s_max: float = S_MAX, ref_min: float = REF_MIN, sep_min: float = SEP_MIN) -> dict:
    """The verdict of one level from its pass matrix: ``tier1`` {member: bool per instance} for the policies that must
    fail, ``reference`` bool per instance, ``ablations`` {name: bool per instance}. All arrays are over the same
    instances, in the same order.

    Returns ``{n, verdict ("admitted" | "practice"), reasons, reference{passed, rate, lower95, cleared},
    tier1[{member, passed, rate, upper95, cleared}], tier2[{ablation, passed, rate, gap, gap_lower95, cleared}],
    shortcut_ceiling{...}, alpha, s_max, ref_min, sep_min}``."""
    ref = np.asarray(reference, bool)
    n = len(ref)
    k = int(ref.sum())
    out = dict(n=n, alpha=alpha, s_max=s_max, ref_min=ref_min, sep_min=sep_min, reasons=[],
               reference=dict(passed=k, rate=k / n, lower95=lower_bound(k, n, alpha)), tier1=[], tier2=[])
    out["reference"]["cleared"] = out["reference"]["lower95"] >= ref_min
    if not out["reference"]["cleared"]:
        out["reasons"].append(f"the reference passes {k} of {n} (lower bound {100 * out['reference']['lower95']:.1f} % < {100 * ref_min:.0f} %)")
    for name in sorted(tier1):
        v = np.asarray(tier1[name], bool)
        if len(v) != n:
            raise ValueError(f"{name}: {len(v)} instances, the reference has {n}")
        kc = int(v.sum())
        row = dict(member=name, passed=kc, rate=kc / n, upper95=upper_bound(kc, n, alpha))
        row["cleared"] = row["upper95"] <= s_max
        out["tier1"].append(row)
        if not row["cleared"]:
            out["reasons"].append(f"{name} passes {kc} of {n} (upper bound {100 * row['upper95']:.1f} % > {100 * s_max:.0f} %)")
    for name in sorted(ablations):
        v = np.asarray(ablations[name], bool)
        if len(v) != n:
            raise ValueError(f"{name}: {len(v)} instances, the reference has {n}")
        gap, lo = gap_lower_bound(ref, v, alpha)
        row = dict(ablation=name, passed=int(v.sum()), rate=float(v.mean()), gap=gap, gap_lower95=lo, cleared=lo >= sep_min)
        out["tier2"].append(row)
        if not row["cleared"]:
            out["reasons"].append(f"{name} loses only {100 * gap:.1f} points against the reference (lower bound {100 * lo:.1f} < {100 * sep_min:.0f})")
    out["shortcut_ceiling"] = shortcut_ceiling(tier1, alpha)
    out["verdict"] = "admitted" if not out["reasons"] else "practice"
    return out


def stability(tier1: dict, reference, ablations: dict, blocks: int = 5, **thresholds) -> dict:
    """Leave-one-block-out: the instances in ``blocks`` consecutive blocks, the verdict recomputed without each.
    ``{block_verdicts, verdict, stable}``; any disagreement makes the level ``borderline`` (listed as not admitted).
    The verdict uses no tuning set, so it cannot depend on one; this is the only re-draw it can be sensitive to."""
    ref = np.asarray(reference, bool)
    n = len(ref)
    full = family_gate(tier1, ref, ablations, **thresholds)["verdict"]
    if n < 10 * blocks:
        return dict(block_verdicts=[], verdict=full, stable=None)
    size = n // blocks
    verdicts = []
    for b in range(blocks):
        keep = np.ones(n, bool)
        keep[b * size:(b + 1) * size] = False
        verdicts.append(family_gate({k: np.asarray(v, bool)[keep] for k, v in tier1.items()}, ref[keep],
                                    {k: np.asarray(v, bool)[keep] for k, v in ablations.items()}, **thresholds)["verdict"])
    stable = len(set(verdicts + [full])) == 1
    return dict(block_verdicts=verdicts, verdict=full if stable else "borderline", stable=stable)


def tuned_constant(rates) -> dict:
    """Spread of "the best constant tuned on other worlds" over disjoint tuning sets: the pass rate on the gate
    instances of the constant each set selects. ``{sets, mean, sd, min, max}``."""
    r = np.asarray(rates, float)
    return dict(sets=len(r), mean=float(r.mean()), sd=float(r.std(ddof=1)) if len(r) > 1 else 0.0, min=float(r.min()),
                max=float(r.max()))
