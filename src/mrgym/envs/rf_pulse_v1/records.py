"""The claimants of the record table of ``Scanner/RFPulse-v1`` (``mrgym records Scanner/RFPulse-v1 --run``).

Committed worlds: split ``dev``, seeds ``records:0:<i>``, the same ``n`` indices at every level. Not used for the
factors (split ``tune``) and not by the gate (split ``gate``).
  reference      the reference policy of ``controls.py``, played through the tools: it measures the scanner
  constructive   verifier side, no scan: the design rule of ``model.py`` with the scanner's true transmit scale,
                 off-resonance and delay and the whole of both tolerances, the corrections unrounded; when that
                 pulse misses the verifier's mask (a pulse that uses the whole of a tolerance has no room for the
                 last digit of its gradient), the same with one, two or three samples more. What a perfect
                 measurement would give with the generator ``minphase``; where the gradient limit decides it is the
                 pulse of the bound's own proof
The class of an instance: its level (which limit can bind) and the slice thickness.
"""
from __future__ import annotations

import multiprocessing

ENV_ID = "Scanner/RFPulse-v1"
EPOCH, SPLIT = 0, "dev"
CLAIMANTS = ("constructive", "reference")


def class_of(cfg: dict) -> str:
    return f"level {cfg['level']} | thickness {cfg['task']['design']['specification']['thickness_mm']:g} mm"


def _one(args):
    import mrgym
    from mrgym import gate, policies
    from mrgym.scanner import world as W

    from ...design import tool
    from . import controls, model, task
    level, i = args
    seed = f"records:{EPOCH}:{i}"
    cfg = gate._instance(mrgym.spec(ENV_ID), level, seed, SPLIT)
    r = policies.run(mrgym.make(ENV_ID, level=level, seed=seed, split=SPLIT), "reference")
    s = r.info.get("score", {})
    out = dict(specification_class=class_of(cfg), index=i,
               reference=dict(index=i, passed=bool(r.reward >= 1.0), ratio=s.get("duration_over_bound"), quarantine=s.get("quarantine")))
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    chain = task.chain_of(world)
    out["constructive"] = dict(index=i, passed=False, ratio=None, quarantine=None)
    try:
        spec, limits = model.spec_of(cfg["task"]), model.limits_of(cfg["task"])
        first, start = model.design_rule(spec, limits, chain.tx_scale, chain.df_hz, chain.grad_delay_us, (1.0, 1.0))
        g = float(first.g_mt_m[0])
        for extra in range(4):
            n = first.n + extra
            base, _ = tool.build(dict(model.call_of(spec, n, g, (1.0, 1.0)), amplitude=1.0 / chain.tx_scale, freq_hz=chain.df_hz), limits)
            fraction = base.call["design"]["rephase_fraction"] + chain.grad_delay_us * 1e-3 / base.duration_ms
            call = dict(model.call_of(spec, n, g, (1.0, 1.0)), amplitude=1.0 / chain.tx_scale, freq_hz=chain.df_hz, rephase_fraction=fraction)
            ok = controls.passes(cfg, world, call)
            s = task.score(cfg, dict(design=1), world)
            out["constructive"] = dict(index=i, passed=bool(ok), ratio=s.get("duration_over_bound"), quarantine=s.get("quarantine"), extra_samples=extra)
            if ok or s.get("quarantine"):
                break
    except (ValueError, tool.Refusal):
        pass
    return out


def run(n: int = 180, workers: int = 2, levels=(1, 2, 3)) -> dict:
    import mrgym

    from ... import bestknown
    with multiprocessing.get_context("fork").Pool(max(1, min(int(workers), 4))) as pool:
        rows = pool.map(_one, [(lv, i) for lv in levels for i in range(n)], chunksize=1)
    claims = {name: {} for name in CLAIMANTS}
    for r in rows:
        for name in CLAIMANTS:
            claims[name].setdefault(r["specification_class"], []).append(r[name])
    return bestknown.build(ENV_ID, mrgym.ENGINE_ID, dict(split=SPLIT, seeds=f"records:{EPOCH}:0 .. records:{EPOCH}:{n - 1}", n_per_level=n, epoch=EPOCH),
                           claims, f"mrgym records {ENV_ID} --run --n {n} --write",
                           note="constructive: the design rule with the scanner's true chain (verifier side, no scan); reference: the scripted policy that measures.")
