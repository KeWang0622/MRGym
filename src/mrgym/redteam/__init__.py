"""``mrgym redteam <id>``: the attack suite of an environment. Every row states what must happen and what happened.

    mrgym redteam Scanner/NullTissue-v0 --n 50                  every row, on the default level
    mrgym redteam Scanner/NullTissue-v0 --only RT17,RT20        some rows
    mrgym redteam Scanner/NullTissue-v0 --tier tool             only what a tool-calling model can try

Tiers (who attacks): ``tool`` a model that only sends tool calls; ``sandbox`` code in another process, with the package
installed and the same user as the service; ``inprocess`` code in the environment's own process. In-process attacks
succeed and are reported, not failed: Python has no private memory, and the documentation says so.

The suite works on any family that follows ``envs/family.py``. It plays on the public split ``dev`` (seeds 0 .. n-1),
tunes the blind answer on ``tune`` and uses its own temporary secret for the rows on a secret split.

Exit status 0 only if no row is a MISMATCH. A row is "reported" when it has no threshold (documented limits, rows the
family contract cannot support); those never decide the exit status and are always printed.
"""
from __future__ import annotations

import datetime
import json
import tempfile
import textwrap
import time

from .. import keys
from . import common, leak, mutants, rows  # noqa: F401
from .rows import TIERS, Row

SCHEMA = "mrgym-redteam/1"


def _suite(ctx, tmp):
    """(id prefix, tier, callable) in the order of the table. A callable returns a Row or a list of Rows."""
    return [
        ("RT00", "tool", lambda: rows.rt00_reference(ctx)),
        ("RT01", "tool", lambda: rows.rt01_null(ctx)),
        ("RT02", "tool", lambda: rows.rt02_no_scan(ctx)),
        ("RT03", "tool", lambda: rows.rt03_blind_constant(ctx)),
        ("RT05", "tool", lambda: rows.rt05_junk_scan(ctx)),
        ("RT06", "tool", lambda: rows.rt06_random_and_replayed(ctx)),
        ("RT13", "tool", lambda: rows.rt13_wrong_units(ctx)),
        ("RT14", "tool", lambda: rows.rt14_format(ctx)),
        ("RT15", "tool", lambda: rows.rt15_lifecycle(ctx)),
        ("RT16", "tool", lambda: rows.rt16_leak(ctx)),
        ("RT17", "sandbox", lambda: rows.rt17_seed_enumeration(ctx, tmp)),
        ("RT18", "sandbox", lambda: rows.rt18_stored_files(ctx)),
        ("RT20", "sandbox", lambda: [rows.rt20_log(ctx, tmp), rows.rt20_inprocess_tamper(ctx)]),
        ("RT21", "sandbox", lambda: rows.rt21_service(ctx, tmp)),
        ("RT22", "tool", lambda: rows.rt22_reuse(ctx, tmp)),
        ("RT23", "inprocess", lambda: rows.rt23_inprocess_read(ctx)),
        ("RT24", "sandbox", lambda: rows.rt24_epoch(ctx, tmp)),
        ("RT25", "tool", lambda: mutants.run(ctx)),
    ]


def run(env_id: str, n: int = 50, level: int | None = None, tiers=TIERS, only=None, workers: int = 2) -> dict:
    """Run the suite. Returns ``dict(schema, env_id, level, n, rows=[...], mismatches, ok, ...)``."""
    import mrgym
    ctx = common.Context(env_id, level=level, n=n, workers=workers)
    only = [x.strip().upper() for x in only] if only else None
    out, t0 = [], time.time()
    with tempfile.TemporaryDirectory(prefix="mrgym-redteam-") as tmp:
        for rid, tier, fn in _suite(ctx, tmp):
            if only and not any(rid.startswith(o) or o.startswith(rid) for o in only):
                continue
            if tier not in tiers and rid != "RT20":
                continue
            t1 = time.time()
            try:
                got = fn()
            except Exception as e:  # noqa: BLE001  a row that cannot run is a mismatch, never a silent skip
                got = Row(rid, tier, "this row", "runs", f"raised {type(e).__name__}: {e}", False)
            got = got if isinstance(got, list) else [got]
            for r in got:
                r.detail["seconds"] = round((time.time() - t1) / len(got), 1)
            out += [r for r in got if r.tier in tiers]
        if "tool" in tiers and ctx.episodes and (not only or "RT04" in only):
            out.append(rows.rt04_reward_is_pass(ctx, ctx.episodes))
    if not only:
        out += rows.not_covered()
    out.sort(key=lambda r: r.id)
    bad = [r.id for r in out if r.ok is False]
    return keys.stamp(dict(
        schema=SCHEMA, env_id=env_id, level=ctx.level, n=ctx.n, split=ctx.split, tiers=list(tiers), engine=mrgym.ENGINE_ID,
        package_version=mrgym.__version__, date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        command=f"mrgym redteam {env_id} --n {n}" + (f" --level {level}" if level else ""),
        rows=[dict(id=r.id, tier=r.tier, attack=r.attack, expected=r.expected, observed=r.observed, verdict=r.verdict, detail=r.detail)
              for r in out], mismatches=bad, ok=not bad, wall_seconds=round(time.time() - t0, 1)))


def text_table(result: dict, width: int = 150) -> str:
    out = [f"== red team: {result['env_id']} level {result['level']} · n = {result['n']} · split {result['split']} · {result['engine']}",
           f"{'id':<7}{'tier':<11}{'verdict':<10}attack / expected / observed"]
    pad = " " * 28
    for r in result["rows"]:
        first = True
        for label, text in (("", r["attack"]), ("expected: ", r["expected"]), ("observed: ", r["observed"])):
            for line in textwrap.wrap(label + text, width - 28, subsequent_indent="  ") or [""]:
                out.append((f"{r['id']:<7}{r['tier']:<11}{r['verdict']:<10}" if first else pad) + line)
                first = False
    n_ok = sum(r["verdict"] == "ok" for r in result["rows"])
    n_rep = sum(r["verdict"] == "reported" for r in result["rows"])
    out.append(f"{n_ok} ok · {len(result['mismatches'])} mismatch{'' if len(result['mismatches']) == 1 else 'es'}"
               + (f" ({', '.join(result['mismatches'])})" if result["mismatches"] else "") + f" · {n_rep} reported · {result['wall_seconds']} s")
    return "\n".join(out)


def add_parser(sub):
    p = sub.add_parser("redteam", help="the attack suite of an environment: expected against observed, per row",
                       description=__doc__.split("\n\n")[0])
    p.add_argument("id")
    p.add_argument("--n", type=int, default=50, help="instances per row (default 50)")
    p.add_argument("--level", type=int, default=None, help="default: the environment's default level")
    p.add_argument("--tier", default=None, choices=TIERS, help="only the rows of one attacker tier (default: all)")
    p.add_argument("--only", default=None, help="comma-separated row ids, e.g. RT03,RT17")
    p.add_argument("--workers", type=int, default=2, choices=(1, 2))
    p.add_argument("--json", default=None, metavar="FILE", help="also write the result as JSON")
    return p


def main(a) -> int:
    result = run(a.id, n=a.n, level=a.level, tiers=(a.tier,) if a.tier else TIERS, only=a.only.split(",") if a.only else None,
                 workers=a.workers)
    print(text_table(result))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, default=str)
            f.write("\n")
        print("written:", a.json)
    return 0 if result["ok"] else 1
