"""The final readout of a training run: trained minus base on the same held-out instances.

    python -m mrgym_verl.readout --base RUN/test_base/episodes.jsonl --trained RUN/test_trained/episodes.jsonl \
        --untrained "Scanner/PhantomQA-v0" --out RUN/readout.json

Both files are episode logs (``mrgym_verl.stoprules``) of evaluation-only runs through the same agent loop on the same
rows of the secret split ``test-iid``: the base model, and the model with the final adapter. Per cell and per
environment (levels as strata) it reports the pass rate of each with the clustered interval of
``mrgym.stats.intervals.cluster_rate`` and the paired difference of the instance means with
``mrgym.stats.compare.paired_compare``; Holm-adjusted p-values over the trained environments.

The verdict is the rule of docs/experiments/PREREG-27b.md section 1, with its numbers as arguments:

    trained    an environment counts when the paired difference is at least ``--gain`` (0.15) and its lower 95 % bound
               is above ``--lower`` (0.05); "it trained" needs ``--min-envs`` (2) such environments
    no harm    on every environment named by ``--untrained`` (never trained on) the paired difference is not below
               ``-harm`` (-0.05)

Nothing is selected here: one base log, one trained log, the instances both played.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from mrgym.stats import compare, intervals

from . import stoprules


def _means(rows):
    acc: dict = {}
    for r in rows:
        acc.setdefault((r["env_id"], r["level"], r["split"], r["seed"]), []).append(float(bool(r["passed"])))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _rate(rows):
    if not rows:
        return None
    c = intervals.cluster_rate([float(bool(r["passed"])) for r in rows], [(r["env_id"], r["level"], r["seed"]) for r in rows],
                               [(r["env_id"], r["level"]) for r in rows])
    return dict(pass_rate=c["rate"], lo=c["lo"], hi=c["hi"], episodes=c["episodes"], instances=c["instances"])


def _pair(base, trained):
    a, b = _means(trained), _means(base)
    c = compare.paired_compare(a, b, {k: (k[0], k[1]) for k in set(a) & set(b)})
    return {k: c[k] for k in ("diff", "lo", "hi", "se", "instances", "p", "test", "wins", "losses", "ties")}


def readout(base: list[dict], trained: list[dict], untrained=(), phase: str = "test", gain: float = 0.15, lower: float = 0.05,
            harm: float = 0.05, min_envs: int = 2) -> dict:
    base = [r for r in stoprules._newest_attempt(base) if r.get("phase") == phase]
    trained = [r for r in stoprules._newest_attempt(trained) if r.get("phase") == phase]
    cells = sorted({(r["env_id"], r["level"]) for r in base + trained})
    envs = sorted({e for e, _ in cells})
    pick = lambda rows, env, level=None: [r for r in rows if r["env_id"] == env and (level is None or r["level"] == level)]   # noqa: E731
    out_cells = [dict(env_id=e, level=lv, base=_rate(pick(base, e, lv)), trained=_rate(pick(trained, e, lv)),
                      paired=_pair(pick(base, e, lv), pick(trained, e, lv))) for e, lv in cells]
    out_envs = []
    for e in envs:
        p = _pair(pick(base, e), pick(trained, e))
        role = "untrained" if e in untrained else "trained"
        ok = p["diff"] is not None and ((p["diff"] >= gain and p["lo"] > lower) if role == "trained" else p["diff"] >= -harm)
        out_envs.append(dict(env_id=e, role=role, base=_rate(pick(base, e)), trained=_rate(pick(trained, e)), paired=p, meets_rule=bool(ok)))
    tr = [x for x in out_envs if x["role"] == "trained" and x["paired"]["p"] is not None]
    for x, adj in zip(tr, compare.holm([x["paired"]["p"] for x in tr])):
        x["paired"]["p_holm"] = adj
    n_ok = sum(x["meets_rule"] for x in out_envs if x["role"] == "trained")
    harm_ok = all(x["meets_rule"] for x in out_envs if x["role"] == "untrained")
    has_untrained = any(x["role"] == "untrained" for x in out_envs)
    verdict = dict(trained_environments_meeting_rule=n_ok, needed=min_envs, no_harm_on_untrained=harm_ok if has_untrained else None,
                   it_trained=bool(n_ok >= min_envs and harm_ok and has_untrained),
                   note=None if has_untrained else "no untrained environment was evaluated: the transfer condition is not assessed and the verdict is not given")
    return dict(schema="mrgym-verl-readout/1", phase=phase, rule=dict(gain=gain, lower=lower, harm=harm, min_envs=min_envs),
                episodes=dict(base=len(base), trained=len(trained)), environments=out_envs, cells=out_cells, verdict=verdict)


def text_table(r: dict) -> str:
    fmt = lambda x: "-" if x is None else f"{100 * x['pass_rate']:.1f} % [{100 * x['lo']:.1f}, {100 * x['hi']:.1f}] ({x['episodes']} ep, {x['instances']} inst)"   # noqa: E731
    out = ["environment / level        role       base                                   trained                                paired difference (trained - base)"]
    for x in r["environments"] + r["cells"]:
        name = x["env_id"] + (f" L{x['level']}" if "level" in x else "")
        p = x["paired"]
        d = "-" if p["diff"] is None else f"{100 * p['diff']:+.1f} points [{100 * p['lo']:+.1f}, {100 * p['hi']:+.1f}], {p['instances']} instances, p = {p['p']:.4f} ({p['test']})"
        out.append(f"{name:<26} {x.get('role', ''):<10} {fmt(x['base']):<38} {fmt(x['trained']):<38} {d}" + ("  MEETS RULE" if x.get("meets_rule") else ""))
    v = r["verdict"]
    out.append(f"verdict: {v['trained_environments_meeting_rule']} trained environment(s) meet the rule (needed {v['needed']}); no harm on the untrained "
               f"environment: {v['no_harm_on_untrained']}; it trained: {v['it_trained']}" + (f" ({v['note']})" if v["note"] else ""))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--untrained", default="", help="environment ids never trained on, space separated")
    ap.add_argument("--phase", default="test")
    ap.add_argument("--gain", type=float, default=0.15)
    ap.add_argument("--lower", type=float, default=0.05)
    ap.add_argument("--harm", type=float, default=0.05)
    ap.add_argument("--min-envs", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = readout(stoprules.load(a.base), stoprules.load(a.trained), [e.split(":")[0] for e in a.untrained.split()], a.phase, a.gain, a.lower, a.harm, a.min_envs)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    print(text_table(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
