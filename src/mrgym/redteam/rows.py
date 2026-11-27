"""The rows of the red-team suite. Each returns one or more ``Row``: what was attacked, what must happen, what happened.

Row ids follow the suite's specification (RT01-RT25). Rows that belong to reward modes the package does not have
(the reward is pass / fail), or that need a hook the family contract does not define, are listed as not applicable with
the reason, so the table shows what is not covered.
"""
from __future__ import annotations

import copy
import dataclasses
import gc
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .. import keys
from . import common, leak

TIERS = ("tool", "sandbox", "inprocess")


@dataclasses.dataclass
class Row:
    id: str
    tier: str            # tool: a model that sends tool calls · sandbox: code in another process · inprocess: code in the env's process
    attack: str
    expected: str
    observed: str
    ok: bool | None      # None: reported or not applicable; it does not decide the exit status
    detail: dict = dataclasses.field(default_factory=dict)

    @property
    def verdict(self):
        return "ok" if self.ok else "reported" if self.ok is None else "MISMATCH"


def _bound(ctx):
    from .. import gate
    return gate.BLIND_MAX, gate.upper_bound


def _decide(ctx, k, n):
    """(ok, note) of "the one-sided 95 % upper bound of k passes in n is within the gate's limit". With too few
    instances the bound cannot clear even at zero passes: the row is then reported, never called ok."""
    limit, ub = _bound(ctx)
    if k == 0 and ub(0, n) > limit:
        return None, f" (n = {n} is too small to decide: even 0 passes leave a bound of {ub(0, n):.3f})"
    return ub(k, n) <= limit, ""


def _blind_row(ctx, rid, attack, results, tier="tool"):
    """A policy that ignores the object must fail: one-sided 95 % upper bound of its pass rate <= the gate's limit."""
    limit, ub = _bound(ctx)
    k, n = sum(r["passed"] for r in results), len(results)
    paid = sum(r["reward"] > 0 for r in results)
    ok, note = _decide(ctx, k, n)
    return Row(rid, tier, attack, f"upper bound <= {limit:.2f}", f"{k}/{n} pass, bound {ub(k, n):.3f}; reward > 0 in {paid}{note}",
               False if paid != k else ok, dict(passed=k, n=n, upper95=round(ub(k, n), 4)))


# ------------------------------------------------------------------------------------------------ tool tier
def rt00_reference(ctx):
    from .. import gate
    runs = ctx.reference_runs(range(ctx.n))
    k = sum(r["passed"] for r in runs.values())
    errors = sorted({r["error"] for r in runs.values() if r["error"]})[:2]
    return Row("RT00", "tool", "reference policy (positive control: the task can be done)", f"pass rate >= {gate.REF_MIN:.2f}",
               f"{k}/{ctx.n} pass" + (f"; policy errors: {errors}" if errors else ""), k / ctx.n >= gate.REF_MIN,
               dict(passed=k, n=ctx.n, lower95=round(gate.lower_bound(k, ctx.n), 4)))


def rt01_null(ctx):
    import mrgym
    n, bad, detail = min(ctx.n, 10), 0, []
    for seed in range(n):
        env = ctx.make(seed)
        env.reset()
        if env.state()["reward"] is not None or env.state()["done"]:      # never submits: no reward exists
            bad += 1
        step = env.step({"name": common.terminal_name(env), "arguments": {}})
        if step.reward != 0.0 or (step.done and common.passed_of(ctx.spec, step)):
            bad += 1
            detail.append(("empty submit", seed, step.reward))
        if not step.done:                                                # run into the tool-call limit without a deliverable
            last = step
            for _ in range(env.max_steps):
                try:
                    last = env.step({"name": "scanner_info", "arguments": {}})
                except mrgym.ResetNeeded:
                    break
            if not last.truncated or last.reward != 0.0:
                bad += 1
                detail.append(("no truncation", seed, last.reward))
        env.close()
    return Row("RT01", "tool", "never submits; first call is submit with an empty object; runs into the call limit",
               "no reward", f"{bad} of {n} episodes paid or misbehaved", bad == 0, dict(cases=detail))


def rt02_no_scan(ctx):
    seeds = list(range(ctx.n))
    prior = ctx.map(ctx.jobs("policy", seeds, name="prior"))
    ref = ctx.reference_runs(seeds)
    right = ctx.map([j for s in seeds for j in ctx.jobs("noscan_answer", [s], answer=ref[s]["answer"])])
    k1, k2 = sum(r["reward"] > 0 for r in prior), sum(r["reward"] > 0 for r in right)
    return [Row("RT02", "tool", "no scan, the family's prior answer", "reward 0 on every instance", f"{k1}/{ctx.n} paid", k1 == 0),
            Row("RT02b", "tool", "no scan, the reference's own answer for this instance (an answer without a measurement)",
                "reward 0 on every instance", f"{k2}/{ctx.n} paid", k2 == 0)]


def rt03_blind_constant(ctx):
    if "fixed_recipe" not in ctx.controls.policies():
        return Row("RT03", "tool", "best blind answer, tuned for pass", "n.a.", "the family has no fixed_recipe policy", None)
    recipe = ctx.recipe()
    res = ctx.map(ctx.jobs("policy", range(ctx.n), name="fixed_recipe", recipe=recipe))
    row = _blind_row(ctx, "RT03", "best answer that ignores the object, tuned for pass on 16 other worlds (split tune)", res)
    row.detail["recipe"] = recipe
    return row


def rt04_reward_is_pass(ctx, results):
    """The reward has one mode. Over every episode the suite played: reward is 0 or 1 and equals passed."""
    bad = [r for r in results if r["reward"] not in (0.0, 1.0) or (r["reward"] == 1.0) != bool(r["passed"])]
    return Row("RT04", "tool", "reward against pass on every episode of this run (covers RT10 and RT12: one reward mode)",
               "reward in {0, 1} and equal to passed", f"{len(bad)} of {len(results)} episodes differ", not bad)


def rt05_junk_scan(ctx):
    return _blind_row(ctx, "RT05", "the cheapest accepted scan, then the prior answer (the log shows a measurement)",
                      ctx.map(ctx.jobs("junk_prior", range(ctx.n))))


def rt06_random_and_replayed(ctx):
    seeds = list(range(ctx.n))
    rnd = ctx.map(ctx.jobs("random", seeds))
    ref = ctx.reference_runs(seeds)
    other = ctx.map([j for i, s in enumerate(seeds) for j in ctx.jobs("junk_answer", [s], answer=ref[seeds[(i + 1) % len(seeds)]]["answer"])])
    return [_blind_row(ctx, "RT06", "junk scan, then a random valid deliverable", rnd),
            _blind_row(ctx, "RT06b", "junk scan, then the reference's answer of another instance (replay)", other)]


def rt13_wrong_units(ctx):
    if "wrong_units" not in ctx.controls.policies():
        return Row("RT13", "tool", "the right numbers in the wrong unit", "n.a.", "the family has no wrong_units policy", None)
    res = ctx.map(ctx.jobs("policy", range(ctx.n), name="wrong_units"))
    k = sum(r["passed"] or r["reward"] > 0 for r in res)
    return Row("RT13", "tool", "the reference's answer in the wrong unit", "0 passes", f"{k}/{ctx.n} pass", k == 0)


def malformed(template) -> list:
    """Deliverables that are not deliverables, built from a well-formed one."""
    out = [{}, [], None, "text", 0, True, [template], {"answer": template}, dict(template or {}, unknown_field=1)]
    for path, v in common.leaves(template)[:3]:
        for bad in (float("nan"), float("inf"), -float("inf"), 10 ** 309, -(10 ** 309), -1.0, 0, "12", [v], {"v": v}, None, True, "1e400"):
            out.append(common.with_leaf(template, path, bad))
    if isinstance(template, dict):
        for k in list(template)[:3]:
            out.append({a: b for a, b in template.items() if a != k})
            out.append(dict(template, **{k: None}))
            out.append(dict(template, **{k: []}))
    out.append({"x": "y" * 100_000})
    return out


def hostile_calls(terminal) -> list:
    """Tool calls whose shape is wrong (not the deliverable's content)."""
    deep = {}
    node = deep
    for _ in range(3000):
        node["a"] = {}
        node = node["a"]
    return [[], None, "submit", 7, {"name": terminal}, {"arguments": {}}, {"name": terminal, "arguments": '{"x": NaN}'},
            {"name": terminal, "arguments": "{" * 5000}, {"name": terminal, "arguments": deep},
            {"name": terminal, "arguments": [[[]]]}, {"method": "tools/call", "params": 5},
            {"type": "function", "function": 5}, {"name": 5, "arguments": {}}, {"name": terminal * 2000, "arguments": {}}]


def _fire(ctx, calls, seed=0):
    """Send each call to a fresh or continuing episode. Returns (exceptions, paid, n)."""
    import mrgym
    env = ctx.make(seed)
    env.reset()
    exceptions, paid = [], 0
    for call in calls:
        try:
            step = env.step(call)
        except mrgym.ResetNeeded:
            env.reset()
            continue
        except BaseException as e:  # noqa: BLE001  the point of the row: nothing may escape step()
            exceptions.append(type(e).__name__)
            env.reset()
            continue
        paid += step.reward != 0.0
        if step.done:
            env.reset()
    env.close()
    return exceptions, paid, len(calls)


def rt14_format(ctx):
    from .. import policies
    env = ctx.make(0)
    env.reset()
    terminal = common.terminal_name(env)
    template = ctx.controls.prior(policies.ToolScanner(env))
    env.close()
    ex, paid, n = _fire(ctx, [{"name": terminal, "arguments": a} for a in malformed(template)])
    ex2, paid2, n2 = _fire(ctx, hostile_calls(terminal))
    return [Row("RT14", "tool", f"{n} malformed deliverables (missing and unknown fields, NaN, Infinity, 1e400, 309-digit integers, containers for scalars)",
                "0 exceptions, 0 reward", f"{len(ex)} exceptions{' (' + ', '.join(sorted(set(ex))) + ')' if ex else ''}, {paid} paid",
                not ex and paid == 0, dict(exceptions=sorted(set(ex)))),
            Row("RT14b", "tool", f"{n2} tool calls of the wrong shape (not an object, deep nesting, huge names, text that is not JSON)",
                "0 exceptions, 0 reward", f"{len(ex2)} exceptions{' (' + ', '.join(sorted(set(ex2))) + ')' if ex2 else ''}, {paid2} paid",
                not ex2 and paid2 == 0, dict(exceptions=sorted(set(ex2))))]


def rt15_lifecycle(ctx):
    import mrgym

    from .. import policies
    from ..scanner import session as core
    problems = []
    env = ctx.make(0)
    step = policies.run(env, lambda sc: (common.junk_scan(sc), ctx.controls.prior(sc))[1])
    first = env.state()["reward"]
    try:
        env.step({"name": common.terminal_name(env), "arguments": env.transcript()[-1]["call"].get("arguments", {})})
        problems.append("a second submit was accepted")
    except mrgym.ResetNeeded:
        pass
    try:
        env.step({"name": "scanner_info", "arguments": {}})
        problems.append("a call after the end was accepted")
    except mrgym.ResetNeeded:
        pass
    if env.state()["reward"] != first or not step.done:
        problems.append("the first reward did not stand")
    env.reset()
    sc = policies.ToolScanner(env)
    before = sc.budget()
    try:
        sc.scan(common.HUGE_PROTOCOL)
        problems.append("a scan beyond the budget was accepted")
    except core.ScannerError as e:
        if e.type not in ("budget_exhausted", "invalid_request"):
            problems.append(f"over-budget scan refused as {e.type}")
    if sc.budget()["used_s"] != before["used_s"] or sc.budget()["scans"] != before["scans"]:
        problems.append("a refused scan was charged")
    plans, limit, refused = 0, before["max_requests"], False
    for _ in range(limit + 3):
        try:
            sc.plan(common.JUNK_PROTOCOLS[-1])
            plans += 1
        except core.ScannerError as e:
            refused = e.type == "request_limit"
            break
        except mrgym.ResetNeeded:
            break
    if plans > limit or (not refused and not env.state()["done"]):
        problems.append(f"{plans} scanner requests accepted with a limit of {limit}")
    env.close()
    return Row("RT15", "tool", "second submit; a call after the end; a scan beyond the budget; requests beyond the limit",
               "refused; the first reward stands; nothing charged", "; ".join(problems) or "all refused, nothing charged", not problems)


def rt16_leak(ctx):
    r = leak.walk(ctx, ctx.n)
    return Row("RT16", "tool", f"leak walk over every agent-visible payload of {r['episodes']} episodes ({r['payloads']} payloads): seed, "
               "ids derived from it, task id, canary, hidden values", "0 hits",
               f"{len(r['hits'])} hits" + (f", first: {r['hits'][0]}" if r["hits"] else "")
               + (f"; {r['chance_candidates']} single 6-digit coincidence(s), not recurring" if r["chance_candidates"] else ""),
               not r["hits"], dict(hits=[list(map(str, h)) for h in r["hits"][:20]]))


# ------------------------------------------------------------------------------------------------ lookup
def _enumeration_table(ctx, n_enum, split="train"):
    """What an attacker with the package computes from integer seeds and the documented public derivation:
    {public task card: [[world seed, fingerprint], ...]}. Where the card alone tells the worlds apart the fingerprint
    is None; otherwise it is the read-out of the junk scan on the rebuilt world (the attacker repeats that scan)."""
    from ..env import derive_seed
    from ..scanner import session as core
    cfgs = []
    for i in range(n_enum):
        seed_hex = derive_seed(ctx.env_id, ctx.level, split, i, environ={})
        cfgs.append(ctx.task.make_instance(seed_hex, ctx.level))
    cards = [json.dumps(c["task"], sort_keys=True) for c in cfgs]
    distinct = len(set(cards)) == len(cards)
    table = {}
    for cfg, card in zip(cfgs, cards):
        fp = None if distinct else common.fingerprint(common.junk_scan(core.LocalScanner(core.ScannerSession(cfg)), common.PROBE_ROI))
        table.setdefault(card, []).append([cfg["seed"], fp])
    return table, distinct


def rt17_seed_enumeration(ctx, tmp):
    limit, ub = _bound(ctx)
    n, n_enum = ctx.n, ctx.n + max(4, ctx.n // 4)                       # the played seeds and some that were not played
    table, by_card = _enumeration_table(ctx, n_enum)
    key_file, salt_file = os.path.join(tmp, "rt17.secret"), os.path.join(tmp, "rt17.salt")
    for f in (key_file, salt_file):
        if not os.path.exists(f):
            keys.keygen(f)
    rows = []
    cases = (("RT17", "train", "public split, integer seeds (positive control: the attack must work here)", {}, True),
             ("RT17b", "train", "the same seeds under a trainer's salt", dict(MRGYM_SALT_FILE=salt_file), False),
             ("RT17c", "test-iid", "a secret split", dict(MRGYM_SECRET_FILE=key_file), False))
    for rid, split, what, env, must_work in cases:
        with common.environ(MRGYM_SECRET=None, MRGYM_SALT=None, MRGYM_SECRET_FILE=None, MRGYM_SALT_FILE=None):
            with common.environ(**env):
                res = ctx.map(ctx.jobs("lookup", range(n), split, table=table))
        rec = sum(r.get("claimed") == r["world_seed"] for r in res)
        k = sum(r["passed"] for r in res)
        v0 = sum(r["task_id"] == keys.public_label(ctx.spec.name.lower(), r["world_seed"]) for r in res)
        if must_work:
            ok = rec / n >= 0.99
            exp = "worlds recovered >= 0.99 (public by design: a code-executing agent can look the answer up)"
            obs = f"{rec}/{n} worlds recovered; the looked-up answer passes {k}/{n}"
        else:
            ok, note = _decide(ctx, k, n)
            if rec or (v0 and split not in keys.PUBLIC_SPLITS):
                ok = False
            exp = f"0 recovered; pass bound <= {limit:.2f}" + ("; task id is not the public derivation" if split not in keys.PUBLIC_SPLITS else "")
            obs = f"{rec}/{n} recovered; {k}/{n} pass (bound {ub(k, n):.3f})" + (f"; task id = public derivation on {v0}/{n}"
                                                                               if split not in keys.PUBLIC_SPLITS else "") + note
        rows.append(Row(rid, "sandbox", f"seed enumeration with the documented derivation of the train split ({n_enum} integer seeds), "
                        f"matching {'the task card' if by_card else 'the read-out of one junk scan'}: {what}", exp, obs, ok,
                        dict(recovered=rec, passed=k, n=n)))
    return rows


def rt18_stored_files(ctx):
    """Nothing that ships (package results) or sits in the repository's results may belong to a secret split."""
    import importlib.resources
    roots = [str(importlib.resources.files("mrgym.envs").joinpath("results"))]
    repo = os.path.abspath(os.path.join(os.path.dirname(roots[0]), "..", "..", "..", "results"))
    if os.path.isdir(repo):
        roots.append(repo)
    needles = [f'"split": "{s}"' for s in keys.SECRET_SPLITS] + [f'"split":"{s}"' for s in keys.SECRET_SPLITS]
    files, bad = 0, []
    for root in roots:
        for d, _, names in os.walk(root):
            for name in names:
                if name.endswith((".json", ".jsonl", ".md")):
                    files += 1
                    with open(os.path.join(d, name), encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    if '"schema": "mrgym-manifest/' in text:            # a suite manifest names a split and seed labels; it
                        continue                                        # holds no world, task id or episode of that split
                    if any(x in text for x in needles):
                        bad.append(os.path.join(d, name))
    return Row("RT18", "sandbox", f"static scan of the stored results ({files} files) for records of a secret split",
               "0 files", f"{len(bad)} files" + (f": {bad[:2]}" if bad else ""), not bad)


# ------------------------------------------------------------------------------------------------ log
def _logged_episode(ctx, tmp, name, scans=2):
    from ..scanner import session as core
    env = ctx.make(0)
    env.reset()
    cfg, engine = env.episode_config(), env._session.engine
    log_dir, state_dir = os.path.join(tmp, name, "log"), os.path.join(tmp, name, "state")
    s = core.ScannerSession(cfg, log_dir=log_dir, engine=engine, state_dir=state_dir)
    sc = core.LocalScanner(s)
    for _ in range(scans):
        common.junk_scan(sc)
    env.close()
    return cfg, engine, s, log_dir


def rt20_log(ctx, tmp):
    from ..scanner import session as core
    cfg, engine, s, log_dir = _logged_episode(ctx, tmp, "rt20a")
    used, receipt, events, path = s.used, s.receipt(), list(s.events), s.log_path
    s.close()
    outcomes = {}
    # control: the untouched log replays against its head
    outcomes["control: the untouched log replays"] = core.replay(cfg, events, engine=engine, head=receipt)["ok"]
    # a prefix presented to the verifier; an edited record; a forged record appended; no head at all
    outcomes["prefix presented to the verifier"] = not core.replay(cfg, events[:1], engine=engine, head=receipt)["ok"]
    edited = copy.deepcopy(events)
    edited[-1]["cost_s"] = 0.01
    outcomes["one record edited"] = not core.replay(cfg, edited, engine=engine, head=receipt)["ok"]
    forged = copy.deepcopy(events) + [dict(copy.deepcopy(events[-1]), event=len(events))]
    outcomes["a record appended without the key"] = not core.replay(cfg, forged, engine=engine, head=receipt)["ok"]
    outcomes["log without a head"] = not core.replay(cfg, events, engine=engine)["ok"]
    # the log of another episode of the same world and the same calls
    cfg2, engine2, s2, _ = _logged_episode(ctx, tmp, "rt20b")
    foreign = list(s2.events)
    s2.close()
    outcomes["log of another episode of the same world"] = not core.replay(cfg, foreign, engine=engine, head=receipt)["ok"]
    # honest stop and resume: the budget continues
    r = core.ScannerSession(cfg, log_dir=log_dir, engine=engine, state_dir=os.path.dirname(s.head_path), resume_from=s.head_path)
    outcomes["control: honest resume keeps the budget"] = r.used == used and r.restarts == 1 and r.n_scans == 2
    head_path = r.head_path
    r.close()
    # truncate + restart: with the receipt it is refused; without it the directory is refused
    with open(path) as f:
        lines = f.readlines()
    with open(path, "w") as f:
        f.writelines(lines[:1])
    refill = None
    try:
        r = core.ScannerSession(cfg, log_dir=log_dir, engine=engine, state_dir=os.path.dirname(head_path), resume_from=head_path)
        refill = r.remaining
        r.close()
    except core.LogTampered:
        pass
    outcomes["log cut to one record, restart with the receipt"] = refill is None
    try:
        core.ScannerSession(cfg, log_dir=log_dir, engine=engine).close()
        outcomes["log cut to one record, restart without a receipt"] = False
    except core.LogTampered:
        outcomes["log cut to one record, restart without a receipt"] = True
    after = core.read_head(head_path)
    outcomes["the episode is then sealed as tampered"] = bool(after.get("sealed")) and after.get("reason") == "tampered"
    bad = [k for k, v in outcomes.items() if not v]
    return Row("RT20", "sandbox", "log attacks: prefix, edit, forged record, no head, foreign log, cut + restart (budget refill)",
               f"{len(outcomes)}/{len(outcomes)} as required (attacks refused, controls accepted)",
               f"{len(outcomes) - len(bad)}/{len(outcomes)}" + (f"; failed: {bad}" if bad else ""), not bad, dict(outcomes=outcomes))


def rt20_inprocess_tamper(ctx):
    """The environment's own gate: a log edited before the deliverable arrives must cost the reward."""
    ref = ctx.reference_runs(range(min(ctx.n, 8)))
    seeds = [s for s, r in ref.items() if r["passed"]][:3]
    paid = sum(tampered_submit(ctx, s, ref[s]["answer"]) > 0 for s in seeds)
    return Row("RT20b", "inprocess", f"a scan record edited in memory before a correct deliverable is submitted ({len(seeds)} instances)",
               "reward 0", f"{paid}/{len(seeds)} paid", paid == 0 and bool(seeds))


def tampered_submit(ctx, seed, answer) -> float:
    """Repeat the reference's scans, edit the last scan record of the log in memory, deliver the reference's answer.
    The earlier scans still replay, so only the log gate stands between this episode and the reward."""
    from .. import policies
    env = ctx.make(seed)

    def policy(sc):
        fn = ctx.controls.policies()[ctx.reference]
        (fn[0] if isinstance(fn, tuple) else fn)(sc)
        for e in reversed(env._session.events):
            if e["type"] == "scan":
                e["cost_s"] = round(e["cost_s"] * 0.5, 6)
                break
        return answer
    step = policies.run(env, policy)
    env.close()
    return float(step.reward)


def rt22_reuse(ctx, tmp):
    from .. import policies
    from ..scanner import session as core
    log_root = os.path.join(tmp, "rt22")
    env = ctx.make(0, log_dir=log_root)
    env.reset()
    sc = policies.ToolScanner(env)
    common.junk_scan(sc)
    used = sc.budget()["used_s"]
    first_dir = os.path.dirname(env._session.log_path)
    env.reset()                                                          # no seed: the same world again, a fresh episode
    b = policies.ToolScanner(env).budget()
    problems = []
    if b["used_s"] != 0 or b["scans"] != 0 or b["requests"] != 0 or used <= 0:
        problems.append("scans or budget carried over to the next episode")
    if os.path.dirname(env._session.log_path) == first_dir:
        problems.append("the second episode writes into the first one's log directory")
    cfg = env.episode_config()
    env.close()
    try:
        core.ScannerSession(cfg, log_dir=first_dir).close()
        problems.append("a new session was accepted on a used log directory")
    except core.LogTampered:
        pass
    return Row("RT22", "tool", "reset() without a seed; a second episode on the same log_dir; a new session on a used directory",
               "a fresh episode; the used directory is refused", "; ".join(problems) or "fresh episode, used directory refused", not problems)


def rt23_inprocess_read(ctx):
    """Code in the environment's process reads the world. This must succeed: the documentation says it does."""
    env = ctx.make(0)
    env.reset()
    seed = env.episode_config()["seed"]
    found = any(isinstance(o, dict) and o.get("seed") == seed and "world_spec" in o for o in gc.get_objects())
    env.close()
    return Row("RT23", "inprocess", "code in the environment's process finds the episode config (seed) with gc.get_objects()",
               "succeeds (documented limit: tier 1 has no boundary against code in the process)",
               "succeeds" if found else "did not find it", True if found else None, dict(found=found))


def rt24_epoch(ctx, tmp):
    from .. import epoch
    key_file = os.path.join(tmp, "rt24.secret")
    if not os.path.exists(key_file):
        keys.keygen(key_file)
    with common.environ(MRGYM_SECRET=None, MRGYM_SECRET_FILE=key_file, MRGYM_SALT=None, MRGYM_SALT_FILE=None):
        commit = epoch.new(ctx.env_id, [ctx.level], "test-iid", "rt24", 4)
        result = epoch.run(commit, ctx.reference, 2, [ctx.level])
        rev = epoch.reveal(commit)
    lv = str(ctx.level)
    out = {"control: the honest reveal and result verify": epoch.verify(commit, rev, result)["ok"]}
    bad = copy.deepcopy(rev)
    bad["levels"][lv]["world_seeds"][1] = bad["levels"][lv]["world_seeds"][1][::-1]
    out["one revealed seed changed"] = not epoch.verify(commit, bad, result)["ok"]
    bad = copy.deepcopy(result)
    scan = next(e for e in bad["levels"][lv]["episodes"][0]["log"] if e["type"] == "scan")
    scan["cost_s"] = round(scan["cost_s"] + 1.0, 6)
    out["one log line changed"] = not epoch.verify(commit, rev, bad)["ok"]
    bad = copy.deepcopy(result)
    bad["levels"][lv]["episodes"][1]["index"] = 3
    bad["levels"][lv]["indices"] = [0, 3]
    out["one reported index changed"] = not epoch.verify(commit, rev, bad)["ok"]
    bad = copy.deepcopy(result)
    del bad["levels"][lv]["episodes"][0]
    bad["levels"][lv]["indices"] = [1]
    out["a started index left out"] = not epoch.verify(commit, rev, bad)["ok"]
    bad = copy.deepcopy(result)
    e = bad["levels"][lv]["episodes"][0]
    e["reward"] = 1.0 - e["reward"]
    out["one reward changed"] = not epoch.verify(commit, rev, bad)["ok"]
    failed = [k for k, v in out.items() if not v]
    return Row("RT24", "sandbox", "held-out epoch: tampered reveal, changed log line, changed or missing index, changed reward",
               f"{len(out)}/{len(out)} as required (tampering detected, the honest files verify)",
               f"{len(out) - len(failed)}/{len(out)}" + (f"; failed: {failed}" if failed else ""), not failed, dict(outcomes=out))


# ------------------------------------------------------------------------------------------------ service
def _wait(path, proc, seconds=30.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        if proc.poll() is not None:
            return None
        time.sleep(0.05)
    return None


def _ends_with_its_starter():
    """``preexec_fn`` of a service process, Linux: the kernel sends the service SIGTERM when the thread that started
    it ends, however it ends (``PR_SET_PDEATHSIG``). A test run or a suite that is killed before it could stop its
    service then leaves none behind (two had been serving for days). None where the call does not exist: there the
    starter has to stop the service itself (``stop_service``)."""
    if not sys.platform.startswith("linux"):
        return None
    import ctypes
    import signal
    try:
        prctl = ctypes.CDLL(None).prctl
    except (OSError, AttributeError):
        return None
    starter = os.getpid()

    def arm():
        prctl(1, int(signal.SIGTERM))                                   # 1 = PR_SET_PDEATHSIG; it survives the exec
        if os.getppid() != starter:                                      # the starter ended before the request was in place
            os._exit(0)
    return arm


def start_service(tmp, name, spec, env=None, extra=(), resume=None):
    """Start ``mrgym serve`` for one episode in a child process. Returns (process, ready dict or None, paths). The
    process ends with the thread that called this (Linux); call from the thread that will stop it."""
    base = os.path.join(tmp, name)
    os.makedirs(base, mode=0o700, exist_ok=True)
    paths = dict(base=base, log=os.path.join(base, "log"), state=os.path.join(base, "state"), ready=os.path.join(base, f"ready{time.time_ns()}.json"),
                 episode=os.path.join(base, "episode.json"))
    fd = os.open(paths["episode"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(spec, f)
    cmd = [sys.executable, "-m", "mrgym.cli", "serve", "--episode-file", paths["episode"], "--log-dir", paths["log"],
           "--state-dir", paths["state"], "--port", "0", "--ready-file", paths["ready"], *extra]
    if resume:
        cmd += ["--resume-from", resume]
    child_env = {k: v for k, v in os.environ.items() if k not in ("MRGYM_SECRET", "MRGYM_SALT", "MRGYM_SECRET_FILE", "MRGYM_SALT_FILE")}
    child_env.update(env or {}, OMP_NUM_THREADS="1")
    proc = subprocess.Popen(cmd, env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, preexec_fn=_ends_with_its_starter())
    return proc, _wait(paths["ready"], proc), paths


def stop_service(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if proc.stderr:
        proc.stderr.close()
    return proc.returncode


def _http(url, method="GET", token=None, body=None):
    req = urllib.request.Request(url, method=method, data=body,
                                 headers=dict({"Content-Type": "application/json"}, **({"Authorization": "Bearer " + token} if token else {})))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def rt21_service(ctx, tmp):
    """One served episode on a secret split, attacked from another process of the same user."""
    from ..scanner import client
    key_file = os.path.join(tmp, "rt21.secret")
    if not os.path.exists(key_file):
        keys.keygen(key_file)
    with open(key_file) as f:
        secret = f.read().strip()
    seed = 915273
    spec = dict(env_id=ctx.env_id, level=ctx.level, split="test-iid", seed=seed)
    world_seed = keys.world_seed(ctx.env_id, ctx.level, "test-iid", seed, environ={}, master=bytes.fromhex(secret))
    proc, ready, paths = start_service(tmp, "rt21", spec, env=dict(MRGYM_SECRET_FILE=key_file))
    rows = []
    if not ready:
        code = stop_service(proc)
        return [Row("RT21", "sandbox", "service start on a secret split", "starts", f"did not start (exit {code})", False)]
    try:
        url, handle = f"http://127.0.0.1:{ready['port']}", ready["episodes"][0]
        with open(os.path.join(paths["state"], handle + ".token")) as f:
            token = f.read().strip()
        # --- what the process shows to the same user
        needles = {"integer seed": str(seed), "world seed": world_seed, "secret": secret, "agent token": token}
        if os.path.exists(f"/proc/{proc.pid}/cmdline"):
            with open(f"/proc/{proc.pid}/cmdline", "rb") as f:
                cmdline = f.read().decode(errors="replace")
            try:
                with open(f"/proc/{proc.pid}/environ", "rb") as f:
                    environ_text, environ_note = f.read().decode(errors="replace"), "readable"
            except OSError:
                environ_text, environ_note = "", "not readable (the service is not dumpable)"
            shown = [k for k, v in needles.items() if v in cmdline or v in environ_text]
            rows.append(Row("RT21", "sandbox", "same user reads /proc/<pid>/cmdline and /proc/<pid>/environ of the service",
                            "no seed, world seed, secret or token",
                            f"{len(shown)} found{': ' + ', '.join(shown) if shown else ''}; environ {environ_note}", not shown))
        else:                                                            # no /proc: the arguments as the process table shows them
            ps = subprocess.run(["ps", "-o", "args=", "-p", str(proc.pid)], capture_output=True, text=True).stdout
            shown = [k for k, v in needles.items() if v in ps]
            rows.append(Row("RT21", "sandbox", "same user reads the arguments of the service (ps; no /proc on this platform)",
                            "no seed, world seed, secret or token", f"{len(shown)} found; the environment was not read",
                            False if shown else None))
        # --- the HTTP surface
        probes = {}
        for path in ("/v1/info", "/v1/task", "/v1/budget"):
            probes[f"GET {path} without a token"] = _http(url + path)[0] == 401
            probes[f"GET {path} with a wrong token"] = _http(url + path, token="x" * 32)[0] == 401
        for path in ("/v1/plan", "/v1/scan", "/v1/submit"):
            probes[f"POST {path} without a token"] = _http(url + path, "POST", body=b"{}")[0] == 401
        for path in ("/v1/admin/episodes", f"/v1/admin/episodes/{handle}", "/v1/truth", "/v1/world", "/v1/seed", "/v1/reset", "/v1/config",
                     "/v1/../v1/admin/episodes", "/v1/info/../../state", "/"):
            probes[f"GET {path}"] = _http(url + path, token=token)[0] == 404
        probes["POST /v1/reset"] = _http(url + "/v1/reset", "POST", token=token, body=b"{}")[0] == 404
        for body in (b"{", b"\xff\xfe", b'{"a": NaN}', b"[" * 5000, b'{"protocol": ' + b'{"a":' * 40 + b"1" + b"}" * 40 + b"}"):
            probes[f"POST /v1/plan with the hostile body {body[:12]!r}"] = _http(url + "/v1/plan", "POST", token=token, body=body)[0] == 400
        requests_before = json.loads(_http(url + "/v1/budget", token=token)[1])["requests"]
        probes["hostile bodies are not counted as requests"] = requests_before == 0
        bodies = [("GET " + p, _http(url + p, token=token)[1].decode()) for p in ("/v1/info", "/v1/task", "/v1/budget")]
        sc = client.Scanner(url, token=token)
        junk = next(p for p in common.JUNK_PROTOCOLS if sc.plan(p)["accept"])
        r = sc.scan(junk)
        r.pop("arrays", None)
        bodies.append(("POST /v1/scan", json.dumps(r)))
        bodies.append(("POST /v1/submit refused", _http(url + "/v1/submit", "POST", token=token, body=b"{}")[1].decode()))
        bad = [k for k, v in probes.items() if not v]
        rows.append(Row("RT21b", "sandbox", f"{len(probes)} probes of the agent port: no token, wrong token, operator routes, undocumented routes, "
                        "'..' paths, hostile bodies", "401 without the token, 404 off the list, 400 for hostile bodies",
                        f"{len(probes) - len(bad)}/{len(probes)}" + (f"; failed: {bad[:3]}" if bad else ""), not bad))
        env = ctx.make(world_seed, split="test-iid")
        with common.environ(MRGYM_SECRET=None, MRGYM_SECRET_FILE=key_file):
            env.reset()
        h, cands = leak.hits(bodies, leak.needles(world_seed, env.episode_config().get("task_id"), env.hidden_world()),
                             [json.loads(b) for _, b in bodies])
        env.close()
        rows.append(Row("RT16b", "sandbox", f"leak walk over {len(bodies)} bodies the service returns (info, task, budget, scan, refusal)",
                        "0 hits", f"{len(h)} hits" + (f", first: {h[0]}" if h else ""), not h))
        # --- files: one uid has no file boundary; the service says so when asked
        readable = [n for n, p in (("log", os.path.join(paths["log"], handle, "scan_log.jsonl")), ("receipt", os.path.join(paths["state"], handle + ".head")),
                                   ("episode file", paths["episode"]), ("secret file", key_file)) if os.access(p, os.R_OK)]
        used = sc.budget()["used_s"]
    finally:
        stop_service(proc)
    iso, ready2, _ = start_service(tmp, "rt21iso", spec, env=dict(MRGYM_SECRET_FILE=key_file),
                                   extra=("--require-isolation", "--agent-uid", str(os.geteuid())))
    code = iso.wait(timeout=60) if ready2 is None else None
    stop_service(iso)
    rows.append(Row("RT21c", "sandbox", "same user reads the service's files (log, receipt, episode file, secret file)",
                    "readable: one uid is no boundary (documented); --require-isolation with the agent's uid = the service's must exit 2",
                    f"readable: {', '.join(readable) or 'none'}; --require-isolation exits {code}", code == 2, dict(readable=readable)))
    # --- budget refill by cutting the log and restarting the service
    log_path = os.path.join(paths["log"], handle, "scan_log.jsonl")
    head_path = os.path.join(paths["state"], handle + ".head")
    again, ready3, _ = start_service(tmp, "rt21", spec, env=dict(MRGYM_SECRET_FILE=key_file), extra=("--agent-token-file", os.path.join(paths["state"], handle + ".token")),
                                     resume=head_path)
    resumed = None
    if ready3:
        sc = client.Scanner(f"http://127.0.0.1:{ready3['port']}", token=token)
        resumed = sc.budget()["used_s"] == used
    stop_service(again)
    with open(log_path) as f:
        lines = f.readlines()
    with open(log_path, "w") as f:
        f.writelines(lines[:1])
    cut, ready4, _ = start_service(tmp, "rt21", spec, env=dict(MRGYM_SECRET_FILE=key_file), resume=head_path)
    code = cut.wait(timeout=60) if ready4 is None else None
    stop_service(cut)
    rows.append(Row("RT20c", "sandbox", "service stopped, log cut to one record, service restarted with its receipt (budget refill by restart)",
                    "honest restart continues the budget; after the cut the service exits 3 and serves nothing",
                    f"honest restart keeps used_s: {resumed}; after the cut: exit {code}", resumed is True and code == 3))
    return rows


def not_covered():
    na = (("RT07", "regressor from public features to the answer", "not implemented: needs a per-family feature hook"),
          ("RT08", "black-box search for a constant outside the lattice", "not implemented: the family contract has no answer lattice (blind_candidates)"),
          ("RT09", "impossible instance", "not implemented: the family contract has no accepts() predicate to find worlds without an answer"),
          ("RT11", "accepted answer cheaper than the optimum", "not applicable: pass / fail reward, no optimum cost in the contract"),
          ("RT19", "submit latency against hidden truth", "not implemented"))
    return [Row(i, "tool", a, "-", why, None) for i, a, why in na]
