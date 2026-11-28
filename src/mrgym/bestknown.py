"""Record tables of design families: the best figure of merit any stored claim has reached, per class of instance.

    mrgym records Scanner/RFPulse-v1                      print the stored table
    mrgym records Scanner/RFPulse-v1 --run --write        measure the family's own claimants on its committed worlds

A record is never paid: the reward of an episode is pass or fail. The figure of merit of a design family is
duration / bound (>= 1 when the bound is a theorem; 1 is the bound itself). A table holds, per class, one row per
claimant measured on the SAME committed worlds: how many worlds, how many passed, the one-sided 95 % lower bound of
the pass rate, and the median, the 95th percentile and the smallest ratio among the passes. The record of a class is
the first claimant that passes below the bound nowhere and whose yield is shown (rule 3); a later one replaces it only
by ``accept``:

  1. fresh worlds     the claim is measured on indices of the committed epoch that no earlier claim has used, and
                      the stored record is measured again on the same indices (the comparison is paired)
  2. the bound        a world on which the claim passes with a ratio below 1 quarantines the claim with the reason
                      ``beats_bound``: a defect of the bound or of the simulator, to be found, never a record
  3. yield            the lower bound of the claim's pass rate is at least ``YIELD_MIN`` (the gate's rule for a reference)
  4. margin           on the worlds where both pass, the mean of (record ratio - claim ratio) has a one-sided 95 %
                      lower bound above ``MARGIN``: equal designs are not exchanged for each other by chance

The worlds of the tables stored with this package are PUBLIC seeds (split ``dev``, seeds ``records:<epoch>:<i>``):
committed, never used for tuning or for the gate, but visible. A claim from outside the package needs worlds the
claimant cannot have seen (a secret epoch, ``mrgym epoch``); that path is not built here.
"""
from __future__ import annotations

import importlib
import importlib.resources
import json
import math
import os

import numpy as np

from .stats.gate import lower_bound

SCHEMA = "mrgym-records/1"
YIELD_MIN, MARGIN, BOUND_EPS = 0.90, 0.01, 1e-9


def path(env_id: str, directory: str | None = None) -> str:
    d = directory or str(importlib.resources.files("mrgym.envs").joinpath("results"))
    return os.path.join(d, env_id.replace("/", "-") + ".records.json")


def load(env_id: str, directory: str | None = None) -> dict | None:
    try:
        with open(path(env_id, directory), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError):
        return None


def summarise(rows) -> dict:
    """One claimant on one class: ``rows`` = [dict(index, passed, ratio)]."""
    ratios = np.array([r["ratio"] for r in rows if r["passed"]], float)
    n, k = len(rows), int(sum(bool(r["passed"]) for r in rows))
    return dict(n=n, passed=k, yield_lower95=round(lower_bound(k, n), 4) if n else 0.0,
                ratio_median=round(float(np.median(ratios)), 4) if k else None, ratio_q95=round(float(np.quantile(ratios, 0.95)), 4) if k else None,
                ratio_best=round(float(ratios.min()), 4) if k else None)


def accept(record_rows, claim_rows, used_indices=()) -> dict:
    """Whether a claim replaces the record of a class. ``record_rows`` and ``claim_rows``: [dict(index, passed, ratio)]
    on the same indices. Returns dict(accepted, reason, quarantined, margin_lower95)."""
    idx = [r["index"] for r in claim_rows]
    if sorted(idx) != sorted(r["index"] for r in record_rows) or len(set(idx)) != len(idx):
        return dict(accepted=False, reason="not_paired: the record was not measured again on the worlds of the claim", quarantined=False, margin_lower95=None)
    if set(idx) & set(used_indices):
        return dict(accepted=False, reason="worlds_reused: an earlier claim was measured on some of these worlds", quarantined=False, margin_lower95=None)
    if any(r["passed"] and r["ratio"] < 1.0 - BOUND_EPS for r in claim_rows):
        return dict(accepted=False, reason="beats_bound: a defect of the bound or of the simulator", quarantined=True, margin_lower95=None)
    s = summarise(claim_rows)
    if s["yield_lower95"] < YIELD_MIN:
        return dict(accepted=False, reason=f"yield: {s['passed']} of {s['n']} pass (lower bound {s['yield_lower95']:.3f} < {YIELD_MIN})", quarantined=False,
                    margin_lower95=None)
    old = {r["index"]: r for r in record_rows}
    diff = np.array([old[r["index"]]["ratio"] - r["ratio"] for r in claim_rows if r["passed"] and old[r["index"]]["passed"]], float)
    if diff.size < 2:
        return dict(accepted=False, reason="margin: fewer than two worlds on which both pass", quarantined=False, margin_lower95=None)
    low = float(diff.mean() - 1.6449 * diff.std(ddof=1) / math.sqrt(diff.size))
    ok = low > MARGIN
    return dict(accepted=ok, reason="accepted" if ok else f"margin: the paired gain has a lower bound of {low:.4f}, not above {MARGIN}", quarantined=False,
                margin_lower95=round(low, 4))


def build(env_id: str, engine: str, worlds: dict, claims: dict, command: str, note: str = "") -> dict:
    """The table of ``claims`` = {claimant: {class: rows}}, claimants in the order they were made. The record of a
    class is the first claimant unless a later one is accepted against the record so far (``accept``)."""
    classes, quarantined = {}, []
    for key in sorted({c for rows in claims.values() for c in rows}):
        best, table, decisions = None, {}, {}
        for name, by_class in claims.items():
            rows = by_class.get(key, [])
            table[name] = summarise(rows)
            if best is None:                                            # the first claimant that may hold a record at all
                if any(r["passed"] and r["ratio"] < 1.0 - BOUND_EPS for r in rows):
                    quarantined.append(dict(claimant=name, specification_class=key, reason="beats_bound"))
                elif table[name]["yield_lower95"] >= YIELD_MIN:
                    best = name
                else:
                    decisions[name] = dict(accepted=False, quarantined=False, margin_lower95=None,
                                           reason=f"yield: {table[name]['passed']} of {table[name]['n']} pass (lower bound {table[name]['yield_lower95']:.3f} < {YIELD_MIN})")
                continue
            decisions[name] = accept(claims[best][key], rows)
            if decisions[name]["quarantined"]:
                quarantined.append(dict(claimant=name, specification_class=key, reason="beats_bound"))
            if decisions[name]["accepted"]:
                best = name
        classes[key] = dict(record=best, record_ratio_median=table[best]["ratio_median"] if best else None, claimants=table, decisions=decisions)
    return dict(schema=SCHEMA, env_id=env_id, engine=engine, command=command, figure_of_merit="duration / bound (1 is the bound; never paid)",
                worlds=worlds, rule=dict(yield_lower95_min=YIELD_MIN, paired_margin=MARGIN, bound_eps=BOUND_EPS), note=note,
                classes=classes, quarantined=quarantined)


def text_table(t: dict) -> str:
    out = [f"{t['env_id']}: best duration / bound per class of specification (engine {t['engine']}; {t['worlds']['n_per_level']} committed worlds per level: "
           f"split {t['worlds']['split']}, seeds {t['worlds']['seeds']})",
           f"{'class':<30}{'record':<14}{'median':>7}   " + "   ".join(f"{name}: passed/n, median, q95, best" for name in next(iter(t["classes"].values()))["claimants"])]
    for key, c in t["classes"].items():
        cells = [f"{s['passed']}/{s['n']}, {s['ratio_median']}, {s['ratio_q95']}, {s['ratio_best']}" for s in c["claimants"].values()]
        out.append(f"{key:<30}{str(c['record']):<14}{str(c['record_ratio_median']):>7}   " + "   ".join(cells))
    out.append(f"quarantined (a pass below the bound: a defect, not a record): {len(t['quarantined'])}")
    out.append(f"a new record needs: fresh committed worlds, no pass below the bound, yield lower bound >= {t['rule']['yield_lower95_min']}, "
               f"paired gain with a 95 % lower bound above {t['rule']['paired_margin']}")
    if t.get("note"):
        out.append(t["note"])
    return "\n".join(out)


def cmd_records(a) -> int:
    import mrgym
    spec = mrgym.spec(a.id)
    if a.run:
        try:
            family = importlib.import_module(spec.family + ".records")
        except ModuleNotFoundError:
            print(f"{a.id} has no record table (its family declares no claimants)")
            return 2
        t = family.run(n=a.n, workers=a.workers)
        if a.write:
            with open(path(a.id, a.results_dir), "w", encoding="utf-8") as f:
                json.dump(t, f, indent=1)
                f.write("\n")
    else:
        t = load(a.id, a.results_dir)
        if t is None:
            print(f"{a.id}: no stored record table")
            return 2
    print(text_table(t))
    return 1 if t["quarantined"] else 0


def add_parser(sub):
    p = sub.add_parser("records", help="the record table of a design family: best duration / bound per class (never paid)")
    p.add_argument("id")
    p.add_argument("--run", action="store_true", help="measure the family's own claimants on its committed worlds")
    p.add_argument("--n", type=int, default=180, help="with --run: committed worlds per level")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--write", action="store_true", help="with --run: store the table next to the package")
    p.add_argument("--results-dir")
    p.set_defaults(fn=cmd_records)
