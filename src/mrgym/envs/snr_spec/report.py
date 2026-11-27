"""Numbers about ``Scanner/SNRSpec-v0`` that the card and a build report quote, computed by a command.

    python -m mrgym.envs.snr_spec.report noise                          the analytic noise formula against repeated scans
    python -m mrgym.envs.snr_spec.report regret --n 40 --workers 2      regret of the reference against the oracle, seconds per episode
    python -m mrgym.envs.snr_spec.report ladder --n 60 --workers 2      the development gate, the ladder table and the task kind
    python -m mrgym.envs.snr_spec.report ladder --from gate.json        the same tables from a stored gate result

Seeds: ``regret`` plays string seeds ``report:0 ..``; ``ladder`` runs ``mrgym.gate`` on its ``gate:`` seeds.
Nothing is written into the package; ``--out`` names a JSON file for the raw result.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time

import numpy as np

ENV = "Scanner/SNRSpec-v0"
NOISE_SEED = "000102030405060708090a0b0c0d0e0f"
NOISE_PROTOCOLS = [
    dict(sequence="se", tr_ms=300.0, matrix=64, phase_fov=0.875, bandwidth_hz_per_px=100.0),
    dict(sequence="se", tr_ms=300.0, matrix=128, bandwidth_hz_per_px=50.0, averages=3),
    dict(sequence="se", tr_ms=300.0, matrix=96, fov_mm=260.0, phase_fov=0.75, bandwidth_hz_per_px=150.0, partial_fourier=0.75),
    dict(sequence="spgr", tr_ms=12.0, flip_deg=15.0, matrix=64, phase_oversampling=0.25, averages=2),
    dict(sequence="se", tr_ms=50.0, matrix=256),
]


def noise_check(world: dict, engine, protocol: dict, repeats: int = 64, seed: int = 7) -> dict:
    """SD of one pixel, one channel: the scatter over ``repeats`` noisy scans of ``protocol`` against ``Engine.noise_sd``
    (what the verifier divides by) and against the law the control policies scale noise with."""
    from ...scanner import console
    from . import controls
    rp = console.plan(protocol, world["console"])["realised"]
    rng = np.random.default_rng(seed)
    stack = np.array([engine.acquire(rp, rng)[0] for _ in range(repeats)])
    if rp["lines"] < rp["lines_full"]:
        # partial Fourier (homodyne detection): the reconstruction keeps the component in phase with the signal, so the
        # noise of a pixel is the scatter of that component where there is signal
        clean = engine.acquire(rp)[0]
        signal = np.abs(clean) > 0.5 * np.abs(clean).max()
        measured = float(np.sqrt((stack * np.conj(clean / np.maximum(np.abs(clean), 1e-30))).real.var(axis=0, ddof=1)[signal].mean()))
    else:
        measured = float(np.sqrt(0.5 * (stack.real.var(axis=0, ddof=1).mean() + stack.imag.var(axis=0, ddof=1).mean())))
    return dict(measured=measured, formula=engine.noise_sd(rp), policy_law=controls.noise_factor(rp) * world["system"]["sigma0"],
                header=console.plan(protocol, world["console"])["header"])


def noise_table() -> str:
    from ...scanner import kspace
    from ...scanner import world as W
    world = W.make_world(NOISE_SEED, dict(kind="qa"))
    engine = kspace.Engine(world)
    out = ["noise SD of one pixel, one channel: 64 repeated scans against Engine.noise_sd (QA world, seed of the test suite)",
           "measured    formula     ratio    protocol"]
    for p in NOISE_PROTOCOLS:
        r = noise_check(world, engine, p)
        out.append(f"{r['measured']:.6f}  {r['formula']:.6f}  {r['measured'] / r['formula']:.4f}   {r['header'].split(' | WFS')[0]}")
    return "\n".join(out)


def _episode(args):
    import mrgym
    from mrgym import policies
    level, seed, policy = args
    t0 = time.time()
    r = policies.run(mrgym.make(ENV, level=level, seed=seed), policy)
    s = r.info["score"]
    return dict(level=level, seed=seed, passed=bool(r.reward >= 1.0), seconds=round(time.time() - t0, 3),
                used_s=r.info["budget"]["used_s"], policy_error=r.info.get("policy_error"), checks=s["checks"], regret=s.get("regret"),
                images={k: dict(time_ratio=v["time_ratio"], snr_margin=v["snr_margin"], shortest_s=v["shortest_s"],
                                averages=v["optimum"]["averages"], bandwidth=v["optimum"]["bandwidth_hz_per_px"])
                        for k, v in s.get("images", {}).items()})


def regret(n: int, levels, workers: int, policy: str = "reference") -> dict:
    jobs = [(lv, f"report:{i}", policy) for lv in levels for i in range(n)]
    with multiprocessing.get_context("fork").Pool(max(1, min(workers, 2))) as pool:
        rows = pool.map(_episode, jobs, chunksize=1)
    return dict(env_id=ENV, policy=policy, n=n, rows=rows)


def regret_table(result: dict) -> str:
    q = (0.0, 0.1, 0.5, 0.9, 1.0)
    out = [f"{result['policy']} against the oracle, {result['n']} instances per level (seeds report:0 .. report:{result['n'] - 1}); "
           "regret = delivered scan time / shortest sufficient scan time - 1, per image",
           "level  passed   regret % of the images that meet the SNR: min / q10 / median / q90 / max   SNR margin %: min / median / max   "
           "s per episode: mean / max"]
    for lv in sorted({r["level"] for r in result["rows"]}):
        rows = [r for r in result["rows"] if r["level"] == lv]
        img = [(k, v, r["checks"].get(f"snr_{k}")) for r in rows for k, v in r["images"].items()]
        reg = np.array([100 * (v["time_ratio"] - 1) for _, v, ok in img if ok])
        mar = np.array([100 * v["snr_margin"] for _, v, _ in img])
        sec = np.array([r["seconds"] for r in rows])
        out.append(f"{lv:>5}  {sum(r['passed'] for r in rows):>3}/{len(rows):<3}  " + " / ".join(f"{v:5.1f}" for v in np.quantile(reg, q))
                   + f"   ({len(reg)} of {len(img)} images)   " + " / ".join(f"{v:5.1f}" for v in np.quantile(mar, (0.0, 0.5, 1.0)))
                   + f"   {sec.mean():.2f} / {sec.max():.2f}")
        errs = sorted({r["policy_error"] for r in rows if r["policy_error"]})
        if errs:
            out.append(f"       policy errors: {errs[:3]}")
    return "\n".join(out)


def ladder_table(gate_result: dict) -> str:
    """Pass rate of the rung policy of each level at every level: a ladder when the policy that suffices at level n
    (>= 0.85) does not at level n + 1 (<= 0.60)."""
    from . import controls
    lvs = sorted(gate_result["levels"], key=int)

    def rate(policy, lv):
        s = gate_result["levels"][lv]
        if policy == controls.REFERENCE:
            return s["reference"]["rate"]
        return {r["policy"]: r["rate"] for r in s["ablations"] + s["reported"]}[policy]
    out = ["rung policy (suffices at)   " + "   ".join(f"L{lv}" for lv in lvs) + "    ladder step"]
    for lv, policy in sorted(controls.LADDER.items()):
        rates = [rate(policy, k) for k in lvs]
        here = rates[lvs.index(str(lv))] if str(lv) in lvs else float("nan")
        nxt = rates[lvs.index(str(lv + 1))] if str(lv + 1) in lvs else None
        verdict = "top rung" if nxt is None else ("holds" if here >= 0.85 and nxt <= 0.60 else "DOES NOT HOLD")
        out.append(f"{policy:<18} (L{lv})    " + "  ".join(f"{100 * r:4.0f}" for r in rates) + f"    {verdict}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m mrgym.envs.snr_spec.report", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("noise")
    a = sub.add_parser("regret")
    a.add_argument("--policy", default="reference")
    b = sub.add_parser("ladder")
    b.add_argument("--from", dest="source", help="a stored gate result (JSON) instead of a run")
    for q in (a, b):
        q.add_argument("--n", type=int, default=40)
        q.add_argument("--levels", type=int, nargs="*", default=[1, 2, 3])
        q.add_argument("--workers", type=int, default=2)
        q.add_argument("--out", help="write the raw result to this JSON file")
    args = p.parse_args(argv)
    if args.cmd == "noise":
        print(noise_table())
        return 0
    if args.cmd == "regret":
        result = regret(args.n, args.levels, args.workers, args.policy)
        print(regret_table(result))
    else:
        from ... import gate
        from . import task
        if args.source:
            with open(args.source, encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = gate.run(ENV, n=args.n, levels=args.levels, workers=min(args.workers, 2))
        print(gate.text_table(result))
        print()
        print(ladder_table(result))
        print()
        kind = task.task_kind(result)
        print(f"task kind by the open-loop rule: {kind['kind']} ({kind['evidence']})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
