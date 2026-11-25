"""Build and check the instance bank of ``Scanner/ArtefactClinic-v0``.

    python tools/build_clinic_bank.py --level 2 --n 300              generate, report, check; write nothing
    python tools/build_clinic_bank.py --level 2 --n 64 --write       also store the instances in the package bank
    python tools/build_clinic_bank.py --level 2 --n 300 --seeds gate the gate's instances (split gate, seeds 0..n-1)
    python tools/build_clinic_bank.py --level 3 --n 48 --seeds tune --tune-open-loop
                                                                     also print the open-loop control and the fixed
                                                                     doses tuned on them (``controls.OPEN_LOOP``,
                                                                     ``controls.TUNED_DOSE``)

Report, all computed here from the generated instances:
  rejections        attempts the generator threw away, by reason
  complaint         share of findings with a cause whose error on the complaint protocol is >= ``present`` for every
                    one of its mechanisms (taken alone), and of benign findings below ``absent`` for every mechanism
  remedy            share of findings that are clean under the verified remedy at ``VERIFY`` of the thresholds
                    (``VERIFY_BUILT`` at a level with constructed sites, whose doses are measured)
  metric            per cause: smallest and median error on its target in the complaint protocol, largest errors
                    after the remedy; benign findings: largest error before and after
  sheet classifier  5-fold cross-validated accuracy of a multinomial logistic regression that predicts the cause of
                    a finding from the protocol sheet and the requirements (and, second row, also from the position
                    of the finding), against the rate of the commonest cause. The check: accuracy <= chance + 0.10
                    for the sheet at every level, and with the position as well at the levels whose findings lie on
                    constructed sites (where a finding lies is then drawn before its cause)
  benign            share of findings that are the object itself, overall and split by what the sheet shows (field,
                    phase-encode axis, matrix) and by look
  budget            levels diagnosed by look: median of "cost of the minimal battery / budget", the largest budget in
                    units of the cost every scan has (overhead and dummy cycles), share of instances that needed
                    another world than the first (``attempts``), share that took causes the world offered (fallback)
Exit status 1 when a check fails.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import sys
import time

import numpy as np

ENV_ID = "Scanner/ArtefactClinic-v0"
LEAK_MAX = 0.10


def _one(args):
    level, seed, split = args
    import mrgym
    from mrgym.env import derive_seed
    from mrgym.envs.artefact_clinic import task
    t0 = time.time()
    seed_hex = derive_seed(ENV_ID, level, split, seed, environ={})
    task._instance.cache_clear()
    from mrgym.envs.artefact_clinic import bank
    stored = bank.get(seed_hex, level) is not None
    inst = task._instance(seed_hex, level)
    assert mrgym.spec(ENV_ID).family
    return dict(seed=seed, seed_hex=seed_hex, instance=inst, seconds=round(time.time() - t0, 2), from_bank=stored)


def features(inst: dict, with_position: bool) -> list:
    """What the card states about one instance: protocol sheet and requirements (one row per finding)."""
    t = inst["task"]
    p, req = t["current_protocol"], t["requirements"]
    from mrgym.envs.artefact_clinic import catalogue as C
    field = t["exam"]["field_t"]
    sheet = [field == 3.0, p["matrix"] / 100.0, p["phase_fov"], np.log(p["bandwidth_hz_per_px"]), np.log(C.fat_hz(field) / p["bandwidth_hz_per_px"]),
             p["te_ms"] / C.fat_period_ms(field), C.echo_class(p["te_ms"], field) == "in", p["tr_ms"] / 100.0, p["flip_deg"] / 90.0,
             p["averages"] / 4.0, p["pe_dir"] == "y", p["ro_polarity"] == 1, t["current_scan_s"] / 100.0,
             req["max_scan_s"] / t["current_scan_s"], np.log(req["min_snr_reference"]) if "min_snr_reference" in req else req["snr_share"],
             req["max_pixel_mm"], len(t["findings"]) / 4.0]
    rows = []
    for f in t["findings"]:
        row = [float(v) for v in sheet]
        if with_position:
            pe, ro = (f["y_mm"], f["x_mm"]) if p["pe_dir"] == "y" else (f["x_mm"], f["y_mm"])
            half = 0.5 * p["fov_mm"] * p["phase_fov"]
            row += [abs(pe) / half, abs(ro) / 100.0, float(np.hypot(pe, ro)) / 100.0, (half - abs(pe)) / 20.0]
        rows.append(row)
    return rows


def _softmax_fit(x, y, k, l2=1.0, steps=400):
    w = np.zeros((x.shape[1], k))
    onehot = np.eye(k)[y]
    lr = 0.5
    for _ in range(steps):
        z = x @ w
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        w -= lr * (x.T @ (p - onehot) / len(y) + l2 * w / len(y))
    return w


def classifier_accuracy(x, y, groups, folds=5, seed=0) -> float:
    """Cross-validated accuracy; findings of one instance stay in one fold."""
    x, y, groups = np.asarray(x, float), np.asarray(y), np.asarray(groups)
    k = int(y.max()) + 1
    order = np.random.default_rng(seed).permutation(np.unique(groups))
    hits = 0
    for f in range(folds):
        test = np.isin(groups, order[f::folds])
        mu, sd = x[~test].mean(0), x[~test].std(0) + 1e-9
        xa = np.c_[(x[~test] - mu) / sd, np.ones((~test).sum())]
        xb = np.c_[(x[test] - mu) / sd, np.ones(test.sum())]
        w = _softmax_fit(xa, y[~test], k)
        hits += int(np.sum(np.argmax(xb @ w, 1) == y[test]))
    return hits / len(y)


def metric_table(rows: list) -> dict:
    """Per cause, over every finding that carries it: the error its mechanism alone makes on the complaint protocol
    (smallest and median) and the largest errors left under the verified remedy; for benign findings the largest
    error on the complaint protocol and under the remedy. Numbers stored by the generator from the twins."""
    from mrgym.envs.artefact_clinic import catalogue as C
    acc = collections.defaultdict(lambda: dict(solo=[], mech=[], trunc=[], worlds=set()))
    for row in rows:
        h = row["instance"]["hidden"]
        for fid, causes in h["causes"].items():
            for c in causes or ["none"]:
                a = acc[c]
                a["worlds"].add(row["instance"]["seed"])
                a["solo"].append(h["solo"][fid][C.CATALOGUE[c].mechanism] if c != "none" else max(h["complaint"][fid].values()))
                a["mech"].append(h["remedied"][fid]["mech"])
                a["trunc"].append(h["remedied"][fid]["trunc"])
    out = {}
    for c, a in sorted(acc.items()):
        key = "largest_error_on_complaint" if c == "none" else "smallest_error_on_target"
        out[c] = {"findings": len(a["solo"]), "worlds": len(a["worlds"]),
                  key: round(max(a["solo"]) if c == "none" else min(a["solo"]), 4), "median_error_on_complaint": round(float(np.median(a["solo"])), 4),
                  "largest_twin_error_after_remedy": round(max(a["mech"]), 4), "largest_object_error_after_remedy": round(max(a["trunc"]), 4)}
    return out


def report(rows: list, level: int) -> dict:
    from mrgym.envs.artefact_clinic import catalogue as C
    from mrgym.envs.artefact_clinic import generator as G
    from mrgym.envs.artefact_clinic import task
    rej = collections.Counter(r for row in rows for r in row["instance"]["rejected"])
    margin = G.VERIFY_BUILT if task.LEVELS[level].get("constructed") else G.VERIFY   # how far below the thresholds the remedy was verified
    labels, present, benign, clean, n_caused, n_benign = [], 0, 0, 0, 0, 0
    for row in rows:
        h = row["instance"]["hidden"]
        for fid, causes in h["causes"].items():
            labels.append("+".join(causes) or "none")
            solo = h["solo"][fid]
            if causes:
                n_caused += 1
                present += all(solo[C.CATALOGUE[c].mechanism] >= C.CATALOGUE[c].present for c in causes)
            else:
                n_benign += 1
                benign += all(v < C.BY_MECHANISM[m].absent for m, v in solo.items())
            clean += G.is_clean(h["remedied"][fid], margin)
    table = metric_table(rows)
    names = sorted(set(labels))
    y = [names.index(v) for v in labels]
    groups = [i for i, row in enumerate(rows) for _ in row["instance"]["task"]["findings"]]
    counts = collections.Counter(labels)
    chance = max(counts.values()) / len(labels)
    acc = {name: classifier_accuracy([f for row in rows for f in features(row["instance"], pos)], y, groups)
           for name, pos in (("sheet", False), ("sheet_and_position", True))}
    attempts = sum(len(r["instance"]["rejected"]) + 1 for r in rows)
    constructed = bool(task.LEVELS[level].get("constructed"))

    def benign_share(key):
        acc = collections.defaultdict(list)
        for row in rows:
            inst = row["instance"]
            acc[str(key(inst))] += [not c for c in inst["hidden"]["causes"].values()]
        return {k: [round(sum(v) / len(v), 3), len(v)] for k, v in sorted(acc.items())}
    proto = lambda inst: inst["task"]["current_protocol"]                 # noqa: E731
    by_look = [r["instance"] for r in rows if "look" in r["instance"]["hidden"]]
    ratio = sorted(i["hidden"]["battery_over_budget"] for i in by_look if "battery_over_budget" in i["hidden"])
    out = dict(level=level, instances=len(rows), findings=len(labels), attempts=attempts,
               rejection_rate=round(1 - len(rows) / attempts, 3), rejections=dict(rej.most_common()),
               instances_with_redrawn_causes=sum(r["instance"].get("redraws", 0) > 0 for r in rows),
               causes={k: counts[k] for k in names}, chance=round(chance, 3),
               complaint_present=f"{present}/{n_caused}", benign_absent=f"{benign}/{n_benign}", remedy_clean=f"{clean}/{len(labels)}",
               metric=table, classifier={k: round(v, 3) for k, v in acc.items()},
               classifier_gain={k: round(v - chance, 3) for k, v in acc.items()},
               seconds_per_instance=round(float(np.mean([r["seconds"] for r in rows if not r["from_bank"]] or [0.0])), 2),
               seconds_per_instance_median=round(float(np.median([r["seconds"] for r in rows if not r["from_bank"]] or [0.0])), 2),
               fallback_share=round(sum(r["instance"].get("redraws", 0) > 0 for r in rows) / len(rows), 3),
               other_world_share=round(sum(len(r["instance"]["rejected"]) > 0 for r in rows) / len(rows), 3),
               benign_share=round(counts["none"] / len(labels), 3),
               benign_share_by=dict(field_t=benign_share(lambda i: i["task"]["exam"]["field_t"]), pe_dir=benign_share(lambda i: proto(i)["pe_dir"]),
                                    matrix=benign_share(lambda i: proto(i)["matrix"]),
                                    look=benign_share(lambda i: "/".join(sorted(set(i["hidden"].get("look", {}).values()))) or "-")),
               from_bank=sum(r["from_bank"] for r in rows))
    if ratio:
        out.update(battery_over_budget_median=ratio[len(ratio) // 2], battery_over_budget_min=ratio[0],
                   largest_budget_in_scan_floors=round(max(i["budget_s"] for i in by_look) / G.SCAN_FLOOR_S, 3),
                   looks=dict(collections.Counter("/".join(sorted(set(i["hidden"]["look"].values()))) for i in by_look)))
    out["checks"] = dict(complaint_present=present == n_caused, benign_absent=benign == n_benign, remedy_clean=clean == len(labels),
                         sheet_classifier_within_chance=acc["sheet"] <= chance + LEAK_MAX)
    if constructed:
        out["checks"].update(position_classifier_within_chance=acc["sheet_and_position"] <= chance + LEAK_MAX, no_fallback=out["fallback_share"] == 0.0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", choices=("train", "gate", "tune"), default="train", help="the public split whose seeds 0..n-1 are generated")
    ap.add_argument("--tune-open-loop", action="store_true", help="print the open-loop control tuned on these instances (use --seeds tune)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--write", action="store_true", help="store the instances in src/mrgym/envs/artefact_clinic/bank/")
    ap.add_argument("--json", help="write the report to this file")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with multiprocessing.get_context("fork").Pool(max(1, min(a.workers, 2))) as pool:
        rows = pool.map(_one, [(a.level, f"clinic:{i}" if a.seeds == "tune" else i, a.seeds) for i in range(a.n)], chunksize=1)
    if a.tune_open_loop:
        from mrgym.envs.artefact_clinic import controls
        print(json.dumps(controls.tune_open_loop([r["instance"] for r in rows])))
        print(json.dumps(controls.tune_doses([r["instance"] for r in rows])))
    rep = report(rows, a.level)
    print(json.dumps(rep, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    if a.write:
        from mrgym.envs.artefact_clinic import bank
        path = os.path.join(os.path.dirname(bank.__file__), "bank", f"level{a.level}.jsonl")
        old = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                old = {json.loads(ln)["key"]: ln.rstrip("\n") for ln in f if ln.strip()}
        for r in rows:
            old[bank.key(r["seed_hex"], a.level)] = bank.line(r["seed_hex"], a.level, r["instance"])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(old[k] for k in sorted(old)) + "\n")
        print(f"wrote {len(old)} instances to {path}")
    return 0 if all(rep["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
