"""An independent oracle of ``Scanner/FatSuppress``: the answer computed from the hidden world in closed form.

The verifier (``task``) measures the delivered methods on noise-free scans. The reference policy (``controls``)
measures with probes and fits. This module scans nothing: it reads the hidden field, the hidden pulse and the hidden
fat of every region and evaluates the saturation band and the inversion-recovery steady state directly. It shares no
code with the other two, so an error in the verifier's scans or in the policy's model shows up as a disagreement.

    python -m mrgym.envs.fat_suppress.agreement --level 2 --n 60

prints, on fresh instances, how often the oracle's own answer passes the verifier, how often oracle and verifier
agree on the method of a region, and the regret of the must-pass policy (that module, not this one, plays the
policies). ``solve`` follows the contract of ``mrgym.envs.oracle``: ``value`` is the largest residual of the
oracle's answer in its own closed form, ``argmin`` that answer.
"""
from __future__ import annotations

import math

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import oracle as O

REGRET_KIND = "difference"                           # error of an answer minus the oracle's value
FAT_HZ_PER_T = -console.FAT_PPM * 1e-6 * console.GAMMA_HZ_PER_T


def _mz(events, t1):
    """Longitudinal steady state before the first event of a cycle of ('rf', factor) and ('wait', ms)."""
    a, b = 1.0, 0.0
    for kind, v in events:
        if kind == "rf":
            a, b = a * v, b * v
        else:
            e = math.exp(-v / t1)
            a, b = a * e, b * e + (1 - e)
    return b / (1 - a)


def spin_echo(flip, tr, te, t1):
    return _mz([("rf", math.cos(flip)), ("wait", te / 2), ("rf", math.cos(2 * flip)), ("wait", tr - te / 2)], t1)


def inversion_recovery(flip, tr, te, ti, t1, k):
    return _mz([("rf", math.cos(flip)), ("wait", te / 2), ("rf", math.cos(2 * flip)), ("wait", tr - ti - te / 2),
                ("rf", -k), ("wait", ti)], t1)


def truth(cfg: dict, world: dict) -> dict:
    """Per marked region: what fat saturation leaves (mean over the spins that appear inside the circle), the T1 of
    its fat, the flip angle at its centre, and the inversion time of its null in the target scan."""
    t, sy = cfg["task"], world["system"]
    n, d = world["n"], world["extent_mm"] / world["n"]
    geo = dict(matrix=t["target_scan"]["matrix"], bandwidth_hz_per_px=t["target_scan"]["bandwidth_hz_per_px"])
    tr = t["target_scan"]["tr_ms"]
    te = console.plan(dict(geo, sequence="se", tr_ms=tr), world["console"])["realised"]["te_ms"]
    px = world["console"]["fov_mm"] / geo["matrix"]
    df = world["fields"]["b0_hz"] + sy["cf_offset_hz"]
    cls = {r["id"]: world["tissues"][r["cls"]] for r in world["regions"]}
    at = lambda a, y, x: a[np.clip(np.rint(y / d).astype(int) + n // 2, 0, n - 1), np.clip(np.rint(x / d).astype(int) + n // 2, 0, n - 1)]   # noqa: E731
    out = {}
    for r in t["regions"]:
        g = np.arange(-r["radius_mm"], r["radius_mm"] + 1e-9, 1.0)
        yy, xx = np.meshgrid(g, g, indexing="ij")
        keep = yy ** 2 + xx ** 2 <= r["radius_mm"] ** 2
        y, x0 = r["y_mm"] + yy[keep], r["x_mm"] + xx[keep]
        x = x0.copy()
        for _ in range(3):                                               # where the fat seen at this pixel really is
            x = x0 + (at(df, y, x) + FAT_HZ_PER_T * world["b0_t"]) / geo["bandwidth_hz_per_px"] * px
        off = at(df, y, x) - sy.get("fatsat_offset_hz", 0.0)
        hit = 0.5 * (1 - np.tanh((np.abs(off) - sy["fatsat_bw_hz"] / 2) / 8.0))
        c = cls[r["id"]]
        flip = math.pi / 2 * float(at(world["fields"]["b1"], np.array([r["y_mm"]]), np.array([r["x_mm"]]))[0]) * sy["tx_gain"]
        row = dict(fat_sat_left=float(np.mean(1 - sy["fatsat_eff"] * hit)), t1_ms=c["t1_ms"], flip_rad=flip, tr_ms=tr, te_ms=te,
                   inv_eff=sy["inv_eff"])
        lo, hi = 10.0, tr - te - 20.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if inversion_recovery(flip, tr, te, mid, c["t1_ms"], sy["inv_eff"]) < 0 else (lo, mid)
        row["null_ms"] = round(0.5 * (lo + hi), 2)
        out[r["id"]] = row
    return out


def left_with(row: dict, entry: dict) -> float:
    """Signal of a region with the delivered method relative to the plain spin echo."""
    if entry["method"] == "fat_sat":
        return row["fat_sat_left"]
    return abs(inversion_recovery(row["flip_rad"], row["tr_ms"], row["te_ms"], entry["ti_ms"], row["t1_ms"], row["inv_eff"])
               / spin_echo(row["flip_rad"], row["tr_ms"], row["te_ms"], row["t1_ms"]))


def passes(t: dict, truths: dict, answer: dict) -> bool:
    """The two pass rules of the task card on the closed-form quantities."""
    for i, e in answer.items():
        if left_with(truths[i], e) > t["fat_max"]:
            return False
        if e["method"] == "stir" and truths[i]["fat_sat_left"] <= t["fat_sat_sufficient"]:
            return False
    return True


def answer(cfg: dict, world: dict, truths: dict | None = None) -> dict:
    """The oracle's deliverable: fat saturation where it leaves less than the middle of the two thresholds, else stir
    at the null."""
    t, tr = cfg["task"], truths or truth(cfg, world)
    switch = 0.5 * (t["fat_max"] + t["fat_sat_sufficient"])
    return {i: dict(method="fat_sat") if row["fat_sat_left"] < switch else dict(method="stir", ti_ms=row["null_ms"]) for i, row in tr.items()}


@O.timed
def solve(cfg: dict, world: dict | None = None) -> dict:
    """The oracle of an instance, by the contract of ``mrgym.envs.oracle``: ``argmin`` is ``answer``; ``value`` the
    largest residual it leaves, in closed form; ``certificate``: the same answer with the method of one region
    swapped, and the check that then fails by the closed form (None where the swap passes too: between the two
    thresholds both methods pass)."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    t, truths = cfg["task"], truth(cfg, world)
    best = answer(cfg, world, truths)
    certificate = []
    for i, e in best.items():
        other = dict(method="stir", ti_ms=truths[i]["null_ms"]) if e["method"] == "fat_sat" else dict(method="fat_sat")
        left = left_with(truths[i], other)
        fails = f"suppressed_{i}" if left > t["fat_max"] else f"method_{i}" if other["method"] == "stir" and truths[i]["fat_sat_left"] <= t["fat_sat_sufficient"] else None
        certificate.append(dict(deliverable={**best, i: other}, fails=fails))
    return dict(value=round(max(left_with(truths[i], e) for i, e in best.items()), 5), argmin=best, method="exact",
                grid=dict(axes=dict(method=2, region=len(best)), size=2 ** len(best), evaluated=2 * len(best)), certificate=certificate, truth=truths)
