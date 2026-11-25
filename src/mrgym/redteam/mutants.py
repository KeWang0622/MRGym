"""RT25: scorer mutants. Each mutant breaks the scoring path in one way; a cheap subset of the suite must notice.

A suite that passes on a broken scorer measures nothing. The mutants are applied at run time (attributes of the family
module, of ``envs.family`` and of the scanner session), never to files, and each is run against five detectors on two
instances:

    reference     the reference passes (on instances where it passes without the mutant)
    prior         the no-scan prior answer earns 0
    no_scan       the reference's answer without any scan earns 0
    junk_prior    the prior answer after the cheapest scan earns 0 (on instances where it does without the mutant)
    tampered      the reference's answer after an in-memory edit of the log earns 0

A mutant is killed when a detector gives another outcome than without the mutant; the detectors run cheapest first and
stop at the first that notices (``killed_by`` names it).
"""
from __future__ import annotations

import contextlib

from . import common, rows


@contextlib.contextmanager
def patched(*patches):
    old = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    try:
        for obj, name, new in patches:
            setattr(obj, name, new)
        yield
    finally:
        for obj, name, value in old:
            setattr(obj, name, value)


def _score_with(task, change):
    original = task.score

    def score(cfg, answer, *a, **kw):
        return change(dict(original(cfg, answer, *a, **kw)), cfg, answer, original)
    return (task, "score", score)


def _set(**kw):
    return lambda s, cfg, answer, original: dict(s, **kw)


def mutants(ctx) -> dict:
    """{name: (description, patches)}."""
    from ..envs import family as F
    from ..scanner import session as core
    task = ctx.task
    replay = core.replay

    def blind_replay(*a, **kw):
        return dict(replay(*a, **kw), ok=True, over_budget=False)

    def other_world(s, cfg, answer, original):
        return dict(original(dict(cfg, seed=cfg["seed"][::-1]), answer))

    def first_check(s, cfg, answer, original):
        return dict(s, passed=bool(next(iter(s.get("checks", {}).values()), False)))

    return {
        "M01": ("reward is always 1", [(F, "reward_of", lambda passed: 1.0)]),
        "M02": ("reward is always 0", [(F, "reward_of", lambda passed: 0.0)]),
        "M03": ("reward is inverted", [(F, "reward_of", lambda passed: 0.0 if passed else 1.0)]),
        "M04": ("the gate 'the log shows a measurement' is dropped", [(F, "GATES", ("log_replays_within_budget",))]),
        "M05": ("the gate 'the log replays within budget' is dropped", [(F, "GATES", ("measured",))]),
        "M06": ("both log gates are dropped", [(F, "GATES", ())]),
        "M07": ("the verifier always says passed", [_score_with(task, _set(passed=True))]),
        "M08": ("the verifier never says passed", [_score_with(task, _set(passed=False))]),
        "M09": ("the verifier's verdict is negated", [_score_with(task, lambda s, c, a, o: dict(s, passed=not s["passed"]))]),
        "M10": ("one check is enough (any instead of all)",
                [_score_with(task, lambda s, c, a, o: dict(s, passed=any(s.get("checks", {}).values())))]),
        "M11": ("only the first check counts (a valid deliverable passes)", [_score_with(task, first_check)]),
        "M12": ("the log replay always reports ok", [(core, "replay", blind_replay)]),
        "M13": ("the verifier scores on another world", [_score_with(task, other_world)]),
    }


DETECTORS = ("prior", "no_scan", "junk_prior", "reference", "tampered")        # cheapest first


def _detect(ctx, name, seeds, ref) -> tuple:
    """Outcome of one detector: the rewards on ``seeds``."""
    if name == "reference":
        return tuple(common.job(j)["reward"] for j in ctx.jobs("policy", seeds, name=ctx.reference))
    if name == "prior":
        return tuple(common.job(j)["reward"] for j in ctx.jobs("policy", seeds, name="prior"))
    if name == "no_scan":
        return tuple(common.job(ctx.jobs("noscan_answer", [s], answer=ref[s]["answer"])[0])["reward"] for s in seeds)
    if name == "junk_prior":
        return tuple(common.job(j)["reward"] for j in ctx.jobs("junk_prior", seeds))
    return tuple(rows.tampered_submit(ctx, s, ref[s]["answer"]) for s in seeds)


def run(ctx, n_instances: int = 2) -> rows.Row:
    if not ctx.family_style:
        return rows.Row("RT25", "tool", "scorer mutants", "-", "not applicable: this id does not follow envs/family.py", None)
    ref = ctx.reference_runs(range(min(ctx.n, 4 * n_instances)))
    seeds = [s for s, r in ref.items() if r["passed"]][:n_instances]
    base = {d: _detect(ctx, d, seeds, ref) for d in DETECTORS}
    expected = dict(reference=(1.0,) * len(seeds), prior=(0.0,) * len(seeds), no_scan=(0.0,) * len(seeds), tampered=(0.0,) * len(seeds))
    wrong = [k for k, v in expected.items() if base[k] != v]
    if wrong or not seeds:
        return rows.Row("RT25", "tool", "scorer mutants", "the detectors hold without a mutant",
                        f"without a mutant these detectors already fail: {wrong or 'no instance the reference passes'}", False)
    killed, survived = {}, []
    for name, (what, patches) in mutants(ctx).items():
        with patched(*patches):
            by = next((d for d in DETECTORS if _detect(ctx, d, seeds, ref) != base[d]), None)      # the first detector that notices
        if by:
            killed[name] = by
        else:
            survived.append(f"{name} ({what})")
    total = len(killed) + len(survived)
    return rows.Row("RT25", "tool", f"{total} scorer mutants (reward rule, each gate, the verdict, the log replay, the world) against five "
                    f"detectors on {len(seeds)} instances", f"{total}/{total} killed",
                    f"{len(killed)}/{total} killed" + (f"; survived: {survived}" if survived else ""), not survived,
                    dict(killed_by=killed, survived=survived))
