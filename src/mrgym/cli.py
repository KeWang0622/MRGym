"""The ``mrgym`` command.

    mrgym --version                         package version and scanner engine
    mrgym list                              registered environments by subject: kind of task, kind, levels, the stored gate
    mrgym map                               the map: subjects x kinds of task, which physics is formed, which task asks for it
    mrgym demo                              one real episode on the scanner, printed as a transcript
    mrgym scan sequence=se tr_ms=500 ...    one scan of a phantom from the command line; writes a PNG
    mrgym play                              operate the scanner yourself (plan / scan / view / budget)
    mrgym record <id> --out DIR             play a scripted policy and save the episode as one bundle file
    mrgym view BUNDLE ... --out FILE.html   one-file HTML viewer of recorded episodes
    mrgym validate [--group rf]             physics checks against analytic formulas, as a table
    mrgym validate --perturb <id>           the reference policy of an environment re-scored under perturbed physics
    mrgym timing                            seconds per scan, per sequence and matrix
    mrgym card <id> | tools <id>            the environment card; the tool schemas in a wire format
    mrgym check <id>                        what must hold before an id is published; the claim against the stored gate
    mrgym gate <id> [--n 300] [--write]     the gate of a family: reference, blind policies and ablations on fresh instances
    mrgym check <id> --learnability         train a small scripted policy family on the real reward, on CPU (minutes)
    mrgym calibrate <id> --model M ...      pass rate of a model endpoint per level, in groups of 8; which levels are in band
    mrgym run-agent ...                     run a language model on an environment through an OpenAI-compatible endpoint
    mrgym results [--check]                 rebuild (or check) the results table from the saved episode records
    mrgym stats power | reliability | regret   what a design detects; reliability of a results folder; regret against the oracle
    mrgym manifest <suite>                  the frozen instance list of a suite (report, validation, public-test, smoke)
    mrgym export [episodes] --dir DIR       saved episodes as JSON lines: messages, deliverable, harness id, canary
    mrgym export harbor | openenv           write the Harbor tasks or the OpenEnv manifest generated from the registry
    mrgym serve ...                         the scanner as an HTTP service: agent plane, operator plane, tokens
    mrgym verify ...                        replay the log of a served episode against its head and score the deliverable
    mrgym epoch keygen|new|run|reveal|verify   secrets, and the commitment of a held-out epoch (publish, later reveal, verify)
    mrgym redteam <id> [--n 50]             the attack suite of an environment: expected against observed, per row
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
        png, _ = render.image_png(img, "magnitude", min_side=512, circles=regions, fov_yx_mm=s["fov_yx_mm"],
                                  centre_yx_mm=s.get("centre_yx_mm", (0.0, 0.0)), rotation_deg=s.get("rotation_deg", 0.0))
        path = os.path.join(out_dir, f"{stem}{'' if len(s['magnitude']) == 1 else f'_{i}'}.png")
        with open(path, "wb") as f:
            f.write(png)
        paths.append(path)
    return paths


# ------------------------------------------------------------------ list / card / tools / check
def list_rows() -> list[dict]:
    """One row per registered id: the claim (``kind``), its levels and what the stored gate says per claim level."""
    import mrgym

    from . import gate
    out = []
    for i in mrgym.list_envs():
        s, c = mrgym.spec(i), mrgym.card(i)
        claim = gate.claim_levels(s) if s.family else []
        g = c["gate"] if c["gate"] and c["gate"].get("engine") == mrgym.ENGINE_ID else None
        out.append(dict(id=i, kind=s.kind, levels=list(s.levels), claim_levels=claim, practice_levels=[lv for lv in s.levels if s.family and lv not in claim],
                        title=c["title"], status=c["status"], task_kind=c.get("task_kind"),
                        gate=None if not s.family else {str(lv): ((g or {}).get("levels", {}).get(str(lv)) or {}).get("verdict") for lv in claim}))
    return out


def cmd_list(a):
    from .registry import retired as retired_ids
    rows = list_rows()
    retired = [dict(id=i, last_version=v, successor=s) for i, (v, s) in sorted(retired_ids.items())]
    if a.json:
        print(json.dumps(rows if not a.retired else dict(environments=rows, retired=retired), indent=1))
        return 0
    def show(r, task=""):
        levels = ",".join(map(str, r["levels"])) + (f" (claim: {','.join(map(str, r['claim_levels']))})" if r["practice_levels"] else "")
        print(f"{r['id']:24s} {r['kind']:9s} levels {levels:22s} {task}{r['title']}\n{'':24s} {r['status']}")

    if a.flat:
        for r in rows:
            show(r)
    else:                                               # by subject, with the kind of task beside each id (mrgym.physics_map)
        from . import physics_map
        by_id = {r["id"]: r for r in rows}
        for letter, title, ids in physics_map.groups():
            print(f"{letter}  {title}")
            for env_id, number in ids:
                show(by_id[env_id], f"{f'{number} {physics_map.kind(number).name}' if number else '-':<11}")
    for r in retired:
        print(f"{r['id']:24s} retired   last served by mrgym {r['last_version']}" + (f"; successor {r['successor']}" if r["successor"] else ""))
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

    from . import check
    if a.learnability:
        from . import learn
        from .learn import check as learnability
        report = learnability.run(a.id, level=a.level, quick=a.quick, workers=min(a.workers, 2))
        print(learnability.text_table(report))
        if a.write:
            print("written:", learn.write(report, "learnability", a.results_dir))
        return 0 if report["verdict"] == "pass" else 1
    res = check.check_env(a.id, level=a.level, verbose=True)
    bad = [r for r in res if not r[1]]
    print(f"{len(res) - len(bad)} / {len(res)} checks passed · {mrgym.card(a.id)['status']}")
    return 1 if bad else 0


def cmd_gate(a):
    from . import gate
    result = gate.run(a.id, n=a.n, levels=[a.level] if a.level is not None else None, workers=a.workers)
    print(gate.text_table(result))
    if a.write:
        print("written:", gate.write(result, a.results_dir))
    return 0 if all(v["verdict"] == "admitted" for v in result["levels"].values()) else 1


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
        out += ["      " + (ln if len(ln) < 170 or ln.startswith("reward") else ln[:167] + "...") for ln in keep]
    return "\n".join(out)


DEMO_WIDTH = 100                                # characters of a control's description in the demo


def _short(text, width=DEMO_WIDTH):
    return text if len(text) <= width else text[:width - 3].rstrip() + "..."


def cmd_demo(a):
    """One episode of a scripted policy, then every other control policy on the same hidden world, in the two groups
    of the family's ``controls``: the shortcuts that have to fail (``MUST_FAIL``) and the controls that may pass
    (ablations of the reference and the reported policies)."""
    import mrgym

    from . import policies
    spec = mrgym.spec(a.id)
    level = a.level or spec.default_level
    env = mrgym.make(a.id, level=level, seed=a.seed, reveal_score=True)
    r = policies.run(env, a.policy)
    b = env.state()
    controls = policies.controls_of(spec)
    print(f"{a.id} · level {level} · seed {a.seed}")
    print("scripted solution (no language model): shows the task can be solved through the tools" if a.policy == controls.REFERENCE
          else f"policy {a.policy} (scripted, no language model)")
    print(_compact(env))
    print(f"  reward {r.reward:.0f} · {b['scans']} scan(s), {b['scan_time_used_s']:.1f} of {b['scan_time_budget_s']:.0f} s of scan time · "
          f"scanner log replays: {r.info['replay']['ok']}")
    if r.info.get("policy_error"):
        print("  policy error:", r.info["policy_error"])
    if a.only:
        return 0
    others = [p for p in controls.policies() if p != a.policy]
    describe = getattr(controls, "DESCRIPTIONS", {})
    groups = (("shortcuts that ignore the object: built to fail (this is what makes the reward mean something)",
               [p for p in others if p in controls.MUST_FAIL]),
              ("other scripted controls: may pass", [p for p in others if p not in controls.MUST_FAIL]))
    width = max([len(p) for p in others] + [0])
    for title, names in groups:
        if not names:
            continue
        print(title)
        for p in names:
            rr = policies.run(mrgym.make(a.id, level=level, seed=a.seed, reveal_score=True), p)
            print(f"  {p:{width}s}  reward {rr.reward:.0f}   {_short(describe.get(p, ''))}".rstrip())
    print("a language model lands between; `mrgym run-agent` plays one")
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
  view N                          digit map of scan N
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
    if a.perturb:
        from . import perturb
        result = perturb.run(a.perturb, n=a.n, level=a.level, draws=a.draws, only=a.only.split(",") if a.only else None,
                             workers=a.workers, all_policies=a.all_policies)
        print(json.dumps(result, indent=1) if a.json else perturb.text_table(result))
        if a.write:
            print("written:", perturb.write(result, a.results_dir))
        return 0
    rows, seconds = validate.run(a.group or None)
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(validate.text_table(rows))
        print(f"{seconds:.0f} s")
    return 0 if all(r["ok"] for r in rows) else 1


TIMING_PROTOCOLS = (
    ("spgr", dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0)), ("spgr, 4 echoes", dict(sequence="spgr", tr_ms=60.0, flip_deg=20.0, n_echoes=4)),
    ("afi", dict(sequence="afi", tr_ms=30.0, flip_deg=60.0)), ("se", dict(sequence="se", tr_ms=500.0)),
    ("ir", dict(sequence="ir", tr_ms=3000.0, ti_ms=800.0)), ("bssfp", dict(sequence="bssfp", tr_ms=8.0, flip_deg=40.0)),
    ("epi", dict(sequence="epi", tr_ms=2000.0)),
    # the sequences served from the block-list core (a console that lists them; ``scanner/served.py``)
    ("fse, 8 echoes", dict(sequence="fse", tr_ms=3000.0, etl=8, te_ms=80.0, echo_spacing_ms=10.0, echo_order="linear")),
    ("fse, 16 echoes", dict(sequence="fse", tr_ms=3000.0, etl=16, te_ms=80.0, echo_order="linear")),
    ("fse, 32 echoes", dict(sequence="fse", tr_ms=3000.0, etl=32, te_ms=80.0, echo_order="linear")),
    ("mese, 8 echoes", dict(sequence="mese", tr_ms=2000.0, n_echoes=8)), ("mese, 32 echoes", dict(sequence="mese", tr_ms=2000.0, n_echoes=32)),
    ("megre, 4 echoes", dict(sequence="megre", tr_ms=60.0, flip_deg=20.0, n_echoes=4)),
    ("megre, 16 echoes", dict(sequence="megre", tr_ms=120.0, flip_deg=20.0, n_echoes=16, bandwidth_hz_per_px=600.0)),
    ("dwse, b 0", dict(sequence="dwse", tr_ms=3000.0, b_value=0.0)), ("dwse, b 1000", dict(sequence="dwse", tr_ms=3000.0, b_value=1000.0)))


def timing_table(repeats=3, matrices=(64, 128, 256), cpu=False):
    """Seconds of computation per scan (median of ``repeats``, cold cache) and per world. Returns (rows, worlds).
    ``cpu``: processor seconds of this process instead of elapsed seconds (the number to read on a busy machine)."""
    import numpy as np

    from .scanner import console, kspace
    from .scanner import world as W
    seed = "00000000000000000000000000000001"
    worlds = {}
    for kind in ("brain", "abdomen", "qa"):
        t = time.perf_counter()
        w = W.make_world(seed, dict(kind=kind, console_sequences="all") if kind == "brain" else dict(kind=kind))
        worlds[kind] = time.perf_counter() - t
        if kind == "brain":                                             # its console lists every sequence; the world is the same
            eng = kspace.Engine(w)
    clock = time.process_time if cpu else time.perf_counter
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
                t = clock()
                eng.acquire(pl["realised"], np.random.default_rng(i))
                ts.append(clock() - t)
            row.append(float(np.median(ts)))
        rows.append(row)
    return rows, worlds, matrices


def cmd_timing(a):
    rows, worlds, matrices = timing_table(a.repeats, cpu=a.cpu)
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
    print(f"brain phantom at 3 T, one thread, median of {a.repeats}, {'processor' if a.cpu else 'elapsed'} seconds, noise-free data not cached; "
          "a repeated protocol reuses it")
    return 0


# ------------------------------------------------------------------ serve / verify / epoch / redteam
def cmd_serve(a):
    from .scanner import service
    return service.run(a)


def cmd_verify(a):
    """Replay the scanner log of a served episode against its head and score the deliverable.

    Exit status: 0 the log replays, is anchored by its head and sealed; 1 it does not replay or is not anchored (the
    reason is printed); 2 it replays but the episode was never sealed (only with --allow-unsealed)."""
    from . import epoch, keys
    from .scanner import session as core
    with open(a.episode_file, encoding="utf-8") as f:
        spec = json.load(f)
    with open(a.answer, encoding="utf-8") as f:
        answer = json.load(f)
    try:
        events = core.read_log(os.path.join(a.log_dir, core.LOG_NAME))
    except (OSError, ValueError) as e:
        print(json.dumps({"replay": {"ok": False, "problems": [f"the log cannot be read: {type(e).__name__}"]}}, indent=1))
        return 1
    master = keys.load_master()
    out = epoch.verify_episode(spec, events, core.read_head(a.head), answer, log_key=keys.sub(master, "log") if master else None)
    rep = out["replay"]
    if rep["ok"] and not rep["sealed"] and not a.allow_unsealed:
        rep["ok"] = False
        rep["problems"].append("the episode was never sealed: no deliverable was received by the scanner (--allow-unsealed scores it anyway)")
        out.update(reward=0.0, passed=False)
    print(json.dumps(keys.stamp(out), indent=1))
    return 1 if not rep["ok"] else 2 if not rep["sealed"] else 0


def cmd_epoch(a):
    from . import epoch
    return epoch.main(a)


def cmd_redteam(a):
    from . import redteam
    return redteam.main(a)


def main(argv=None) -> int:
    import mrgym
    ap = argparse.ArgumentParser(prog="mrgym", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--version", action="version", version=f"mrgym {mrgym.__version__} (engine {mrgym.ENGINE_ID})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="registered environments")
    p.add_argument("--json", action="store_true")
    p.add_argument("--retired", action="store_true", help="with --json: an object with the served and the retired ids")
    p.add_argument("--flat", action="store_true", help="one list in the order of the ids, without the subjects and the kinds of task")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("card", help="environment card as Markdown")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_card)
    p = sub.add_parser("tools", help="tool schemas")
    p.add_argument("id")
    p.add_argument("--format", default="openai", choices=("openai", "mcp", "anthropic", "neutral"))
    p.set_defaults(fn=cmd_tools)
    p = sub.add_parser("check", help="what must hold before an id is published")
    p.add_argument("id")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--learnability", action="store_true", help="the CPU learnability check of the family (minutes; at most 2 workers)")
    p.add_argument("--quick", action="store_true", help="with --learnability: one seed, fewer instances and steps")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--write", action="store_true", help="store the --learnability result as package data")
    p.add_argument("--results-dir", default=None)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("gate", help="the gate of a family: reference, blind policies and ablations on fresh instances")
    p.add_argument("id")
    p.add_argument("--n", type=int, default=300, help="fresh instances per level (default 300)")
    p.add_argument("--level", type=int)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--write", action="store_true", help="store the result next to the package")
    p.add_argument("--results-dir")
    p.set_defaults(fn=cmd_gate)
    p = sub.add_parser("demo", help="one real episode, printed")
    p.add_argument("--id", default="Scanner/NoWrap-v1")
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

    p = sub.add_parser("validate", help="physics checks against analytic formulas; with --perturb, an environment under perturbed physics")
    p.add_argument("--json", action="store_true")
    p.add_argument("--group", action="append", choices=("signal", "rf", "encoding", "console", "core"), help="only this group of checks (repeatable)")
    p.add_argument("--perturb", metavar="ID", help="re-score the reference policy of this environment under physics perturbations")
    p.add_argument("--level", type=int, help="--perturb: level (default: the environment's default level)")
    p.add_argument("--n", type=int, default=30, help="--perturb: instances (default 30)")
    p.add_argument("--draws", type=int, default=1, help="--perturb: noise draws per instance")
    p.add_argument("--only", help="--perturb: comma-separated perturbation names")
    p.add_argument("--workers", type=int, default=1, help="--perturb: worker processes (at most 2)")
    p.add_argument("--all-policies", action="store_true", help="--perturb: play every control policy, for the ranking under perturbed physics")
    p.add_argument("--write", action="store_true", help="--perturb: store the result next to the package")
    p.add_argument("--results-dir")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("timing", help="seconds per scan")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--cpu", action="store_true", help="processor seconds of this process instead of elapsed seconds")
    p.set_defaults(fn=cmd_timing)

    p = sub.add_parser("serve", help="the scanner as an HTTP service: an agent plane and an operator plane")
    from .scanner import service as _service
    _service.add_arguments(p)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("verify", help="replay a service log against its head and score a deliverable")
    p.add_argument("--episode-file", required=True, help="JSON {env_id, level, split, seed} of the episode (hidden from the agent)")
    p.add_argument("--log-dir", required=True, help="directory that holds the episode's scan_log.jsonl")
    p.add_argument("--head", required=True, help="the receipt of the episode: <state-dir>/<handle>.head, or the trainer's copy")
    p.add_argument("--answer", required=True, help="JSON file with the deliverable")
    p.add_argument("--allow-unsealed", action="store_true", help="score a log without a seal (exit status 2)")
    p.set_defaults(fn=cmd_verify)
    from . import epoch as _epoch
    from . import redteam as _redteam
    _epoch.add_parser(sub).set_defaults(fn=cmd_epoch)
    _redteam.add_parser(sub).set_defaults(fn=cmd_redteam)

    from . import calibrate, export, manifest, record, results, viewer
    from .agents import runner
    from .stats import cli as stats_cli
    record.add_parser(sub)
    viewer.add_parser(sub)
    runner.add_parser(sub)
    results.add_parser(sub)
    stats_cli.add_parser(sub)
    manifest.add_parser(sub)
    calibrate.add_parser(sub)
    from .compat import export as compat_export
    p = sub.add_parser("export", help="saved episodes as JSON lines (default), or the generated Harbor tasks / OpenEnv manifest")
    p.add_argument("target", nargs="?", default="episodes", choices=("episodes", "harbor", "openenv"))
    p.add_argument("--out", default=None, help="episodes: output file (default standard output); harbor: the datasets folder "
                   "(default datasets); openenv: the package folder")
    p.add_argument("--dir", default="results", help="episodes: run folder with episodes/ (default results)")
    p.add_argument("--format", default="rl", choices=export.FORMATS, help="episodes: rl = one object per episode; sft = messages and tools of passing episodes")
    p.add_argument("--agent", action="append", metavar="NAMES", help="episodes: only these agents (comma list)")
    p.add_argument("--env", action="append", metavar="IDS", help="episodes: only these environments (comma list)")
    p.add_argument("--include-scripted", action="store_true", help="episodes: also export scripted policies (no conversation)")
    p.add_argument("--id", action="append", help="harbor / openenv: environment id; repeatable for harbor (default: every scored id)")
    p.add_argument("--level", type=int, default=None, help="harbor / openenv: level (default: the id's default level)")
    p.add_argument("--n", type=int, default=1, help="harbor: new tasks per id and level")
    p.add_argument("--from-secret", action="store_true", help="harbor: derive the hidden seeds from MRGYM_SECRET instead of fresh entropy")
    p.add_argument("--check", action="store_true", help="harbor / openenv: fail when a generated file on disk is stale or was edited")
    p.add_argument("--refresh", action="store_true", help="harbor: render the existing tasks again from their stored seeds (no new instance)")
    p.set_defaults(fn=lambda a: export.cmd_export(a) if a.target == "episodes" else compat_export.cmd_export(a))

    from . import coverage, physics_map, qa
    qa.add_parser(sub)
    coverage.add_parser(sub)
    physics_map.add_parser(sub)

    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
