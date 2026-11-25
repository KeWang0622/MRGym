"""Oracle of ``Scanner/BandFree`` against the verifier and against the must-pass policy, on fresh instances.

    python -m mrgym.envs.band_free.agreement --level 2 --n 60 [--seed-prefix fresh] [--no-reference]

prints per level: how often the oracle's own answer passes the verifier; how often oracle and verifier agree on
whether each image needs two acquisitions; the verifier's lowest band-free bandwidth over the oracle's; and, unless
``--no-reference``, the pass rate of the reference policy and its delivered bandwidth relative to the oracle's lowest
band-free bandwidth. (``oracle.py`` itself never imports the policies; this module plays them.)
"""
from __future__ import annotations

import numpy as np

import mrgym
from mrgym import policies

from . import oracle, task


def compare(level: int, n: int, seed_prefix: str = "oracle", split: str = "train", reference: bool = True) -> dict:
    """Oracle against verifier (and, with ``reference``, against the reference policy) on ``n`` fresh instances of
    ``level`` (see the module docstring)."""
    rows = []
    for i in range(n):
        env = mrgym.make("Scanner/BandFree-v0", level=level, seed=f"{seed_prefix}:{i}", split=split)
        r = policies.run(env, "reference") if reference else (env.reset(), None)[1]
        cfg, world, engine = env.episode_config(), env.hidden_world(), env._session.engine
        o = oracle.solve(cfg, world)
        v = task.reference_protocols(cfg, world)
        s = task.score(cfg, o["answer"], world, engine) if o["answer"] else dict(passed=False)
        mine = r.info["score"].get("images", {}) if reference else {}
        same, ratios, lowest_ratio = True, [], []
        for name, oi in o["images"].items():
            one = v[name]["single_bandwidth"] is not None and v[name]["single_bandwidth"] <= cfg["task"]["single_up_to"]
            same &= oi["needs_two"] != one
            kind = "single_bandwidth" if one else "pair_bandwidth"
            if v[name][kind] and oi[kind]:
                lowest_ratio.append(v[name][kind] / oi[kind])
            lowest = oi["single_bandwidth"] if mine.get(name, {}).get("acquisitions") == 1 else oi["pair_bandwidth"]
            if lowest and name in mine:
                ratios.append(mine[name]["bandwidth_hz_per_px"] / lowest)
        rows.append(dict(seed=i, oracle_passes=bool(s["passed"]), reference_passes=bool(reference and r.reward >= 1.0), same_acquisitions=bool(same),
                         ratios=ratios, lowest_ratio=lowest_ratio, oracle_diagnostic=s.get("diagnostic", "no answer")))
        env.close()
    k = lambda key: int(sum(x[key] for x in rows))                                       # noqa: E731
    low = [x for row in rows for x in row["lowest_ratio"]]
    out = dict(level=level, n=n, seeds=f"split {split}, seeds {seed_prefix}:0..{n - 1}", oracle_passes=k("oracle_passes"),
               same_acquisitions=k("same_acquisitions"),
               verifier_lowest_over_oracle_lowest=dict(median=round(float(np.median(low)), 3), q05=round(float(np.quantile(low, 0.05)), 3),
                                                       q95=round(float(np.quantile(low, 0.95)), 3)),
               disagreements=[dict(seed=x["seed"], why=x["oracle_diagnostic"]) for x in rows if not x["oracle_passes"] or not x["same_acquisitions"]][:6])
    if reference:
        ratio = [x for row in rows for x in row["ratios"]]
        out.update(reference_passes=k("reference_passes"), pass_regret=round((k("oracle_passes") - k("reference_passes")) / n, 4),
                   reference_bandwidth_over_oracle_lowest=dict(median=round(float(np.median(ratio)), 3), q95=round(float(np.quantile(ratio, 0.95)), 3)))
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--level", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed-prefix", default="oracle")
    ap.add_argument("--no-reference", action="store_true", help="oracle against verifier only (the reference policy is not played)")
    a = ap.parse_args()
    for lv in a.level:
        print(json.dumps(compare(lv, a.n, a.seed_prefix, reference=not a.no_reference)), flush=True)
