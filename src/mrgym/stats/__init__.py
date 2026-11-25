"""Statistics of MR Gym: one interval, one paired test, one place. numpy and scipy only; the primary path draws no
random numbers, so every stored number can be rebuilt byte for byte.

    intervals     wilson, clopper_pearson, cluster_rate, cluster_mean, beta_binomial_interval
    compare       paired_compare, mcnemar_exact, holm, unpaired_cluster_diff
    power         mde_paired, instances_needed, halfwidth, gate_power
    reliability   icc_oneway, retest_agreement, cronbach_alpha, split_half, item_table, separation, difficulty_drivers
    gate          family_gate, shortcut_ceiling, stability, and the three bound functions of the admission gate
    sequential    betting_test, alpha_spend
    lint          lint_text, lint_results: the reporting rules R1-R12

The unit of every interval and test is the instance. ``mrgym stats power`` and ``mrgym stats reliability`` print the
design and the reliability tables; ``mrgym results --check`` applies the reporting rules.
"""
from .compare import holm, mcnemar_exact, paired_compare, unpaired_cluster_diff
from .gate import family_gate, gap_lower_bound, lower_bound, shortcut_ceiling, stability, upper_bound
from .intervals import beta_binomial_interval, clopper_pearson, cluster_mean, cluster_rate, wilson
from .lint import lint_results, lint_text
from .power import gate_power, halfwidth, instances_needed, mde_paired
from .reliability import (
    cronbach_alpha,
    difficulty_drivers,
    icc_oneway,
    item_table,
    retest_agreement,
    separation,
    split_half,
)
from .sequential import alpha_spend, betting_test

__all__ = [
    "alpha_spend", "beta_binomial_interval", "betting_test", "clopper_pearson", "cluster_mean", "cluster_rate",
    "cronbach_alpha", "difficulty_drivers", "family_gate", "gap_lower_bound", "gate_power", "halfwidth", "holm",
    "icc_oneway", "instances_needed", "item_table", "lint_results", "lint_text", "lower_bound", "mcnemar_exact",
    "mde_paired", "paired_compare", "retest_agreement", "separation", "shortcut_ceiling", "split_half", "stability",
    "unpaired_cluster_diff", "upper_bound", "wilson",
]
