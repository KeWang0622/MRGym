"""The ``mrgym`` command.

    mrgym --version                         package version and scanner engine
    mrgym list                              registered environments and their admission status
    mrgym demo                              one real episode on the scanner, printed as a transcript
    mrgym scan sequence=se tr_ms=500 ...    one scan of a phantom from the command line; writes a PNG
    mrgym play                              operate the scanner yourself (plan / scan / view / budget)
    mrgym record <id> --out DIR             play a scripted policy and save the episode as one bundle file
    mrgym view BUNDLE ... --out FILE.html   one-file HTML viewer of recorded episodes
    mrgym validate                          physics checks against analytic formulas, as a table
    mrgym timing                            seconds per scan, per sequence and matrix
    mrgym card <id> | tools <id>            the environment card; the tool schemas in a wire format
    mrgym check <id> [--admission]          the gate for publishing an id; with --admission, rerun the control policies
    mrgym run-agent ...                     run a language model on an environment through an OpenAI-compatible endpoint
    mrgym results [--check]                 rebuild (or check) the results table from the saved episode records
    mrgym serve ...                         the scanner as an HTTP service, one process per episode
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _kv(pairs):
    """``key=value`` arguments -> dict; values are JSON literals when they parse, strings otherwise."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"expected key=value, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def _write_pngs(scanner, scan_index, out_dir, stem, regions=None):
    from . import render
    s = scanner.scans[scan_index]
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(s["magnitude"]):
        png, _ = render.image_png(img, "magnitude", min_side=512, circles=regions, fov_yx_mm=s["fov_yx_mm"])
        path = os.path.join(out_dir, f"{stem}{'' if len(s['magnitude']) == 1 else f'_{i}'}.png")
        with open(path, "wb") as f:
            f.write(png)
        paths.append(path)
    return paths


# ------------------------------------------------------------------ list / card / tools / check
def cmd_list(a):
    import mrgym
    rows = [(i, mrgym.spec(i), mrgym.card(i)) for i in mrgym.list_envs()]
    if a.json:
        print(json.dumps([dict(id=i, kind=s.kind, levels=list(s.levels), title=c["title"], status=c["status"]) for i, s, c in rows], indent=1))
        return 0
    for i, s, c in rows:
        print(f"{i:24s} levels {','.join(map(str, s.levels)):6s} {c['title']}\n{'':24s} {c['status']}")
    return 0


def cmd_card(a):
    import mrgym
    from . import cards
    c = mrgym.card(a.id)
    if a.json:
        print(json.dumps(c, indent=1))
        return 0
    env = mrgym.make(a.id, seed=0)
    obs, _ = env.reset()
    tools = [dict(name=t.name, cost=t.cost, counted=t.counted, terminal=t.terminal) for t in env._tool_specs()]
    print(cards.markdown(c, tools, obs.text), end="")
    return 0


def cmd_tools(a):
    import mrgym
    env = mrgym.make(a.id, seed=0)
    env.reset()
    print(json.dumps(env.tools(a.format), indent=1))
    return 0


def cmd_check(a):
    import mrgym
    from . import admission, check
    if a.admission:
        result = admission.run(a.id, n=a.n, levels=[a.level] if a.level else None, workers=a.workers)
        print(admission.text_table(result))
        if a.write:
            print("written:", admission.write(a.id, result, a.results_dir))
        return 0
    res = check.check_env(a.id, level=a.level, verbose=True)
    bad = [r for r in res if not r[1]]
    print(f"{len(res) - len(bad)} / {len(res)} checks passed · {mrgym.card(a.id)['status']}")
    return 1 if bad else 0


# ------------------------------------------------------------------ demo
def _compact(env, max_lines=3):
    out = []
    for e in env.transcript():
        if e["call"] is None:
            continue
        name = e["call"].get("name", "?")
        args = json.dumps(e["call"].get("arguments", {}), separators=(",", ":"))
        out.append(f"  agent -> {name} {args if len(args) < 150 else args[:147] + '...'}")
        lines = [ln for ln in e["text"].splitlines() if ln and not ln.startswith("[") and not ln.isdigit()]
        keep = lines[:max_lines] if name not in ("scanner_info",) else [f"(instrument card, {len(e['text'])} characters)"]
        out += ["      " + (ln if len(ln) < 170 else ln[:167] + "...") for ln in keep]
    return "\n".join(out)


def cmd_demo(a):
    import mrgym
    from . import policies
    spec = mrgym.spec(a.id)
    level = a.level or spec.default_level
    env = mrgym.make(a.id, level=level, seed=a.seed, reveal_score=True)
    t0 = time.time()
    r = policies.run(env, a.policy)
    b = env.state()
    print(f"{a.id} · level {level} · seed {a.seed} · policy {a.policy} (scripted, no language model)")
    print(_compact(env))
    print(f"  reward {r.reward:.3f} · {b['scans']} scan(s), {b['scan_time_used_s']:.1f} of {b['scan_time_budget_s']:.0f} s of scan time · "
          f"log replays: {r.info['replay']['ok']} ({r.info['replay']['mode']}) · {time.time() - t0:.1f} s wall")
    if r.info.get("policy_error"):
        print("  policy error:", r.info["policy_error"])
    others = [p for p in policies.policy_names(spec) if p not in (a.policy, "seed_attacker") and not a.only]
    if others:
        print("the same hidden world, other scripted policies:")
        for p in others:
            rr = policies.run(mrgym.make(a.id, level=level, seed=a.seed, reveal_score=True), p)
            s = rr.info["score"]
            why = "passed" if s.get("passed") else (s.get("diagnostic") if s.get("diagnostic") not in (None, "ok") else "not within tolerance")
            print(f"  {p:16s} reward {rr.reward:.3f}   {why}")
    return 0


# ------------------------------------------------------------------ scan / play
def _explore(a):
    import mrgym
    return mrgym.make("Scanner/Explore-v0", phantom=a.phantom, field_t=a.field, lesion=a.lesion, seed=a.seed)


def cmd_scan(a):
    env = _explore(a)
    env.reset()
    protocol = _kv(a.protocol)
    step = env.step({"name": "scan", "arguments": {"protocol": protocol}})
    print(step.observation.text.rsplit("\n", 1)[0])
    if step.observation.is_error:
        return 1
    out = a.out or os.path.join("mrgym_out", f"{a.phantom}_{protocol.get('sequence', 'scan')}.png")
    for p in _write_pngs(env._scanner, 0, os.path.dirname(out) or ".", os.path.splitext(os.path.basename(out))[0]):
        print("image written:", p)
    return 0


PLAY_HELP = """commands
  info [sequences|physics]        the instrument card
  plan key=value ...              dry run of a protocol (free), e.g.  plan sequence=se tr_ms=500 matrix=256
  scan key=value ...              run it (costs scan time); the image is written as a PNG
  view N                          digit map and extent of scan N
  budget                          scan time used and left
  help | quit"""


def cmd_play(a):
    env = _explore(a)
    obs, _ = env.reset()
    print(obs.text + "\n\n" + PLAY_HELP)
    interactive = sys.stdin.isatty()
    while True:
        if interactive:
            sys.stdout.write("scanner> ")
            sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            break
        words = line.split()
        if not words:
            continue
        if not interactive:
            print("scanner> " + line.strip())
        cmd, rest = words[0], words[1:]
        if cmd in ("quit", "exit", "q"):
            break
        try:
            if cmd == "help":
                print(PLAY_HELP)
                continue
            if cmd == "info":
                call = {"name": "scanner_info", "arguments": {"topic": rest[0]} if rest else {}}
            elif cmd in ("plan", "scan"):
                call = {"name": cmd, "arguments": {"protocol": _kv(rest)}}
            elif cmd == "view":
                call = {"name": "view", "arguments": {"scan": int(rest[0])}}
            elif cmd == "budget":
                s = env.state()
                print(f"{s['scan_time_used_s']:.1f} s used, {s['scan_time_left_s']:.1f} s left, {s['scans']} scans")
                continue
            else:
                print(f"unknown command {cmd!r}; type help")
                continue
        except (SystemExit, ValueError, IndexError) as e:
            print(f"could not read that: {e}")
            continue
        step = env.step(call)
        print(step.observation.text)
        if cmd == "scan" and not step.observation.is_error:
            k = step.observation.data["scan_index"]
            for p in _write_pngs(env._scanner, k, a.out, f"scan{k:02d}_{step.observation.data['realised']['sequence']}"):
                print("image written:", p)
    env.step({"name": "finish", "arguments": {}})
    print("session closed")
    return 0


# ------------------------------------------------------------------ validate / timing
def cmd_validate(a):
    from . import validate
    rows, seconds = validate.run()
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(validate.text_table(rows))
        print(f"{seconds:.0f} s")
    return 0 if all(r["ok"] for r in rows) else 1


TIMING_PROTOCOLS = (
    ("spgr", dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0)), ("spgr, 4 echoes", dict(sequence="spgr", tr_ms=60.0, flip_deg=20.0, n_echoes=4)),
    ("afi", dict(sequence="afi", tr_ms=30.0, flip_deg=60.0)), ("se", dict(sequence="se", tr_ms=500.0)),
    ("ir", dict(sequence="ir", tr_ms=3000.0, ti_ms=800.0)), ("bssfp", dict(sequence="bssfp", tr_ms=6.0, flip_deg=40.0)),
    ("epi", dict(sequence="epi", tr_ms=2000.0)))


def timing_table(repeats=3, matrices=(64, 128, 256)):
    """Seconds of computation per scan (median of ``repeats``, cold cache) and per world. Returns (rows, worlds)."""
    import numpy as np

    from .scanner import console, kspace
    from .scanner import world as W
    seed = "00000000000000000000000000000001"
    worlds = {}
    for kind in ("brain", "abdomen", "qa"):
        t = time.perf_counter()
        w = W.make_world(seed, dict(kind=kind))
        worlds[kind] = time.perf_counter() - t
        if kind == "brain":
            eng = kspace.Engine(w)
    rows = []
    for label, prot in TIMING_PROTOCOLS:
        row = [label]
        for m in matrices:
            pl = console.plan(dict(prot, matrix=m), eng.w["console"])
            if not pl["accept"]:
                row.append(None)
                continue
            ts = []
            for i in range(repeats):
                eng._clean.clear()
                eng._splat.clear()
                t = time.perf_counter()
                eng.acquire(pl["realised"], np.random.default_rng(i))
                ts.append(time.perf_counter() - t)
            row.append(float(np.median(ts)))
        rows.append(row)
    return rows, worlds, matrices


def cmd_timing(a):
    rows, worlds, matrices = timing_table(a.repeats)
    cell = lambda v: "refused" if v is None else f"{v:.2f}"   # noqa: E731
    if a.markdown:
        print("| Sequence | " + " | ".join(f"matrix {m}" for m in matrices) + " |")
        print("|---|" + "---:|" * len(matrices))
        for r in rows:
            print(f"| {r[0]} | " + " | ".join(cell(v) for v in r[1:]) + " |")
    else:
        print(f"{'seconds per scan':18s}" + "".join(f"{'matrix ' + str(m):>12s}" for m in matrices))
        for r in rows:
            print(f"{r[0]:18s}" + "".join(f"{cell(v):>12s}" for v in r[1:]))
    print("\nworld generation, once per episode: " + ", ".join(f"{k} {v:.2f} s" for k, v in worlds.items()))
    print(f"brain phantom at 3 T, one thread, median of {a.repeats}, noise-free data not cached; a repeated protocol reuses it")
    return 0


# ------------------------------------------------------------------ serve
def cmd_serve(a):
    from .scanner import service
    from .scanner import session as core
    if a.config:
        cfg = core.load_config(a.config)
    else:
        import mrgym
        env = mrgym.make(a.id, level=a.level, seed=a.seed) if a.level else mrgym.make(a.id, seed=a.seed)
        env.reset()
        cfg = env.episode_config()
        env.close()
    service.serve(cfg, a.log_dir, a.host, a.port, a.port_file)
    return 0


def main(argv=None) -> int:
    import mrgym
    ap = argparse.ArgumentParser(prog="mrgym", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--version", action="version", version=f"mrgym {mrgym.__version__} (engine {mrgym.ENGINE_ID})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="registered environments")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("card", help="environment card as Markdown")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_card)
    p = sub.add_parser("tools", help="tool schemas")
    p.add_argument("id")
    p.add_argument("--format", default="openai", choices=("openai", "mcp", "anthropic", "neutral"))
    p.set_defaults(fn=cmd_tools)
    p = sub.add_parser("check", help="the gate for publishing an id")
    p.add_argument("id")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--admission", action="store_true", help="rerun the control policies (minutes)")
    p.add_argument("--n", type=int, default=8, help="instances per level for --admission")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--write", action="store_true", help="store the --admission result as package data")
    p.add_argument("--results-dir", default=None)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("demo", help="one real episode, printed")
    p.add_argument("--id", default="Scanner/NoWrap-v0")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--seed", default="0")
    p.add_argument("--policy", default="reference")
    p.add_argument("--only", action="store_true", help="do not run the other scripted policies")
    p.set_defaults(fn=cmd_demo)

    for name, fn, hlp in (("scan", cmd_scan, "one scan from the command line"), ("play", cmd_play, "operate the scanner yourself")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--phantom", default="brain", choices=("brain", "abdomen", "qa"))
        p.add_argument("--field", type=float, default=3.0, choices=(1.5, 3.0))
        p.add_argument("--lesion", default=None, choices=("ms", "tumour", "liver"))
        p.add_argument("--seed", default="1")
        if name == "scan":
            p.add_argument("protocol", nargs="+", help="key=value protocol parameters, e.g. sequence=se tr_ms=500 matrix=256")
            p.add_argument("--out", default=None, help="PNG path (default mrgym_out/<phantom>_<sequence>.png)")
        else:
            p.add_argument("--out", default="mrgym_out", help="folder for the PNG of every scan")
        p.set_defaults(fn=fn)

    p = sub.add_parser("validate", help="physics checks against analytic formulas")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("timing", help="seconds per scan")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(fn=cmd_timing)

    p = sub.add_parser("serve", help="HTTP scanner service for one episode")
    p.add_argument("--config", default=None, help="episode config JSON (holds the seed)")
    p.add_argument("--id", default="Scanner/Explore-v0")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--seed", default=None)
    p.add_argument("--log-dir", default="runs/scanner_log")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--port-file", default=None)
    p.set_defaults(fn=cmd_serve)

    from . import record, results, viewer
    from .agents import runner
    record.add_parser(sub)
    viewer.add_parser(sub)
    runner.add_parser(sub)
    results.add_parser(sub)

    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
