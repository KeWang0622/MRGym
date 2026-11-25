"""``mrgym stats``: the design and the reliability of a results folder, printed.

    mrgym stats power --instances 300 --episodes 2     what a design detects (half-width, paired difference, Holm)
    mrgym stats power --table                          the power table of the designs, as Markdown
    mrgym stats power --gate                           operating characteristic of the admission gate
    mrgym stats reliability --dir results              ICC, retest agreement, separation, dead instances (--dir is repeatable)
    mrgym stats regret Scanner/NullTissue-v0 --n 20    a scripted policy against the oracle of each instance
"""
from __future__ import annotations

import json


def cmd_stats(a) -> int:
    from . import power, reliability
    if a.what == "power":
        if a.gate:
            print(power.gate_text(tuple(a.n) if a.n else (100, 300, 1000)))
        elif a.table:
            print(power.power_markdown(a.rho, a.r, a.comparisons))
        else:
            print(power.power_text(a.instances, a.episodes, a.rho, a.r, a.comparisons))
        return 0
    if a.what == "reliability":
        from ..agents import summarize
        records = [r for d in (a.dir or ["results"]) for r in summarize.load(d)]
        if not a.all_tiers:
            pilot = [r for r in records if r["record_schema"] == 1]
            if pilot:
                print(f"note: {len(pilot)} of {len(records)} records are of schema 1 (pilot); they are included here and "
                      "excluded from results.json[\"reliability\"]")
        rep = reliability.report(records)
        print(json.dumps(rep, indent=1) if a.json else reliability.text(rep))
        return 0
    if a.what == "regret":
        import mrgym

        from ..envs import oracle as O
        spec = mrgym.spec(a.id)
        for level in ([a.level] if a.level is not None else spec.levels):
            rep = O.policy_regret(a.id, level, [f"{a.prefix}:{i}" for i in range(a.n[0] if a.n else 20)], a.policy)
            print(json.dumps(rep, indent=1) if a.json else O.regret_text(rep))
        return 0
    return 1


def add_parser(sub) -> None:
    """Register ``mrgym stats``."""
    p = sub.add_parser("stats", help="power of a design, reliability of a results folder, regret against the oracle")
    p.add_argument("what", choices=("power", "reliability", "regret"))
    p.add_argument("id", nargs="?", help="regret: environment id")
    p.add_argument("--instances", type=int, default=300, help="power: instances per cell (default 300)")
    p.add_argument("--episodes", type=int, default=2, help="power: episodes per instance (default 2)")
    p.add_argument("--rho", type=float, default=0.07, help="power: between-instance share of variance (default 0.07, measured on the pilot)")
    p.add_argument("--r", type=float, default=0.0, help="power: correlation of the two agents' instance effects (default 0)")
    p.add_argument("--comparisons", type=int, default=10, help="power: size of the Holm family (default 10)")
    p.add_argument("--table", action="store_true", help="power: the table of designs as Markdown")
    p.add_argument("--gate", action="store_true", help="power: the operating characteristic of the admission gate")
    p.add_argument("--n", type=int, action="append", help="power --gate: instances (repeatable; default 100, 300, 1000); regret: instances")
    p.add_argument("--dir", action="append", help="reliability: results folder; repeat to pool several run folders (default results)")
    p.add_argument("--all-tiers", action="store_true", help="reliability: do not print the pilot note")
    p.add_argument("--level", type=int, default=None, help="regret: one level (default: all)")
    p.add_argument("--policy", default=None, help="regret: a scripted policy (default: the reference)")
    p.add_argument("--prefix", default="gate", help="regret: seed prefix (default gate)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_stats)
