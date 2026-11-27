"""The conformance scorecard: 108 items in 11 themes, each a command with an expected result or a documented action.

    python tools/conformance/run.py                      run every checker, print the scorecard, write tools/conformance/report.json
    python tools/conformance/run.py --only A2,H5         run some items (nothing is written)
    python tools/conformance/run.py --docker             also run what needs a Docker daemon (image builds, Harbor oracle / nop runs,
                                                   OpenEnv runtime level, the Compose lint rules)
    python tools/conformance/run.py --online             also ask the package index and the code host (release, tag, CI state)
    python tools/conformance/run.py --check-readme       fail when the generated block of integrations/README.md is stale
    python tools/conformance/run.py --accept             store the passing items of the last report as the baseline
    make conformance                               the same as the first line; exit 1 below 90 %, on a regression against
                                                   tools/conformance/baseline.json, or when an adapter is not equal to the native API

The registry is ``tools/conformance/items.json`` (stable ids, source of each item, tier, checker). Checker kinds:

    check          lines of ``mrgym check <id>`` on every registered id
    tests          named tests of the test suite pass (one pytest run for all items)
    tests_matching at least one shipped test matches a name pattern, and all of them pass
    cmd            a command exits 0 (and its output matches ``expect``)
    files          files exist and hold (or do not hold) patterns
    native         a function in ``tools/conformance/checks.py`` that reads the registry, stored results or a validator's output
    evidence       ``tools/conformance/evidence/<ID>.json`` with the keys claim, n, result, who, date, artefact_sha256; items
                   B12, B13, D9, F8 and I5 need a named person. Without the file the item fails and prints its action.
    online         a fact on a package index or a code host; not asked without ``--online``

    score = (PASS + 0.5 * PARTIAL) / applicable.  A checker that could not run is a FAIL, never a PASS.

Each framework's validator runs in its own environment (their pins collide). The command prefixes are the
environment variables CONF_GYM, CONF_VF, CONF_INSPECT, CONF_OPENENV and CONF_HARBOR; the defaults are throw-away
``uv run --isolated`` environments with the pinned versions (see ``DEFAULT_TOOLS`` in checks.py).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import checks as K  # noqa: E402

THEMES = {"A": "Definition", "B": "Verification", "C": "Anti-cheating", "D": "Difficulty and learnability", "E": "Diversity and scale",
          "F": "Documentation", "G": "Reproducibility", "H": "Packaging and integration", "I": "Baselines", "J": "Maintenance",
          "K": "Ethics and licence"}
VALUE = {K.PASS: 1.0, K.PARTIAL: 0.5, K.FAIL: 0.0}
TARGET = 0.90
TIERS = {"T1": "engineering, no Docker, no model", "T2": "CI with Docker", "OWN": "owner action", "MOD": "model runs",
         "TRN": "training run or outside use", None: "-"}
BEGIN, END = "<!-- BEGIN GENERATED: conformance -->", "<!-- END GENERATED: conformance -->"


def load_items():
    with open(os.path.join(HERE, "items.json"), encoding="utf-8") as f:
        return json.load(f)


def _online(ctx, check):
    if not ctx.online:
        return K.FAIL, f"not asked (run with --online); action: {check.get('action')}"
    import urllib.request

    import mrgym
    try:
        if check["what"] == "pypi":
            with urllib.request.urlopen("https://pypi.org/pypi/mrgym/json", timeout=20) as r:
                ok = mrgym.__version__ in json.load(r).get("releases", {})
            return (K.PASS if ok else K.FAIL), f"mrgym {mrgym.__version__} on the package index: {ok}"
        url = re.search(r'Repository = "https://github.com/([^"]+)"', open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read())[1]
        with urllib.request.urlopen(f"https://api.github.com/repos/{url}/actions/runs?branch=main&per_page=20", timeout=20) as r:
            runs = json.load(r).get("workflow_runs", [])
        last = {}
        for run in runs:
            last.setdefault(run["name"], run["conclusion"])
        ok = bool(last) and all(v == "success" for v in last.values())
        return (K.PASS if ok else K.FAIL), f"last run per workflow on main: {last or 'none'}"
    except Exception as e:  # noqa: BLE001
        return K.FAIL, f"could not be asked: {type(e).__name__}: {e}"[:200]


def run_check(ctx, item, check=None):
    check = check or item["check"]
    kind = check["kind"]
    try:
        if kind == "native":
            status, detail = K.registry(check["fn"])(ctx)
        elif kind == "evidence":
            status, detail = K.kind_evidence(ctx, dict(check, action=item.get("action", check.get("action"))), item["id"])
        elif kind == "online":
            status, detail = _online(ctx, dict(check, action=item.get("action")))
        else:
            status, detail = getattr(K, "kind_" + kind)(ctx, check)
    except Exception as e:  # noqa: BLE001  a checker that crashes is a failed item with the reason, not a crashed scorecard
        status, detail = K.FAIL, f"the checker could not run: {type(e).__name__}: {e}"[:300]
    if status == K.PASS and check.get("then"):
        s2, d2 = run_check(ctx, item, check["then"])
        if s2 != K.PASS:
            status, detail = s2, f"{detail}; then: {d2}"
    return status, detail


def scorecard(results):
    rows, tot = [], dict(n=0, PASS=0, PARTIAL=0, FAIL=0, points=0.0)
    for t, name in THEMES.items():
        rs = [r for r in results if r["theme"] == t]
        c = {s: sum(r["status"] == s for r in rs) for s in VALUE}
        points = sum(VALUE[r["status"]] for r in rs)
        rows.append(dict(theme=t, name=name, n=len(rs), points=points, percent=round(100 * points / len(rs), 1) if rs else None, **c))
        tot = dict(n=tot["n"] + len(rs), points=tot["points"] + points, **{s: tot[s] + c[s] for s in VALUE})
    return rows, tot


def text_table(rows, tot, title):
    out = [title, f"{'theme':<32}{'items':>6}{'pass':>6}{'part':>6}{'fail':>6}{'score':>9}"]
    for r in rows:
        pc = f"{r['percent']:>8.1f} %" if r["percent"] is not None else f"{'-':>10}"
        out.append(f"{r['theme']}. {r['name']:<29}{r['n']:>6}{r['PASS']:>6}{r['PARTIAL']:>6}{r['FAIL']:>6}{pc}")
    pc = 100 * tot["points"] / tot["n"] if tot["n"] else 0.0
    out.append(f"{'All':<32}{tot['n']:>6}{tot['PASS']:>6}{tot['PARTIAL']:>6}{tot['FAIL']:>6}{pc:>8.1f} %   target >= {100 * TARGET:.0f} %  ->  "
               f"{'OK' if pc >= 100 * TARGET else 'BELOW'}")
    return "\n".join(out)


def git_head():
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return head + ("+uncommitted" if dirty else "")
    except OSError:
        return None


def readme_block(report) -> str:
    """The generated block of integrations/README.md: what each validator said, from the report."""
    fw = report["frameworks"]
    t = report["total"]
    rows = [("Gymnasium", "`src/mrgym/compat/gymnasium.py` (extra `mrgym[gymnasium]`)", "`python -m mrgym.compat.gymnasium --check`", fw.get("gymnasium")),
            ("verifiers", "`integrations/verifiers/mrgym_scanner/`", "`vf-validate mrgym-scanner -n N --runtime.type subprocess`", fw.get("verifiers")),
            ("Inspect AI", "`integrations/inspect/mrgym_inspect.py`", "`python integrations/inspect/mock_run.py`", fw.get("inspect")),
            ("OpenEnv", "`integrations/openenv/mrgym_scanner_env/`", "`openenv validate <dir> --level static --skip-build`; `python integrations/openenv/demo.py`",
             fw.get("openenv")),
            ("Harbor", "`mrgym export harbor` -> `datasets/mrgym-<family>-v<N>/`", "`python -m mrgym.compat.harbor lint` and `local`; `harbor lint`", fw.get("harbor"))]
    out = [BEGIN, f"Measured by `make conformance` on {report['date_utc']} at commit `{report['commit']}` (mrgym {report['mrgym']}, engine "
           f"`{report['engine']}`, {len(report['ids'])} registered ids). Conformance: {report['percent']:.1f} % of {t['n']} items "
           f"({t['PASS']} pass, {t['PARTIAL']} partial, {t['FAIL']} fail).", "",
           "| Target | Artifact | Validator | Result of the last run |", "|---|---|---|---|"]
    for name, artifact, cmd, summary in rows:
        out.append(f"| {name} | {artifact} | {cmd} | {summary or 'not run in the last report'} |")
    out += ["", "Adapter parity (`tests/test_adapters_parity.py`, 3 seeds per id, in each framework's environment): "
            + "; ".join(f"{k} {'equal to the native API' if v else 'NOT EQUAL or not run'}" for k, v in report["parity"].items()) + ".", END]
    return "\n".join(out)


def write_readme(report, check=False) -> bool:
    path = os.path.join(ROOT, "integrations", "README.md")
    text = open(path, encoding="utf-8").read()
    block = readme_block(report)
    new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda m: block, text, flags=re.S) if BEGIN in text else text.rstrip() + "\n\n" + block + "\n"
    if check:
        return new == text
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated item ids or theme letters; nothing is written")
    ap.add_argument("--docker", action="store_true")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "report.json"))
    ap.add_argument("--no-write", action="store_true", help="print only")
    ap.add_argument("--check-readme", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--accept", action="store_true", help="store the passing items of the last report as tools/conformance/baseline.json")
    ap.add_argument("--verbose", action="store_true", help="print every item, not only the ones that do not pass")
    a = ap.parse_args(argv)
    items = load_items()
    if a.check_readme:
        try:
            with open(a.out, encoding="utf-8") as f:
                ok = write_readme(json.load(f), check=True)
        except (OSError, ValueError):                                    # no report yet: the block must be empty
            text = open(os.path.join(ROOT, "integrations", "README.md"), encoding="utf-8").read()
            ok = BEGIN + "\n" + END in text
        print("integrations/README.md: the generated block is " + ("current" if ok else "stale or the report is missing; run `make conformance`"))
        return 0 if ok else 1
    if a.accept:
        with open(a.out, encoding="utf-8") as f:
            report = json.load(f)
        passed = sorted((r["id"] for r in report["items"] if r["status"] == K.PASS), key=lambda i: (i[0], int(i[1:])))
        with open(os.path.join(HERE, "baseline.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"from_report": report["date_utc"], "commit": report["commit"], "pass": passed}, f, indent=1)
            f.write("\n")
        print(f"tools/conformance/baseline.json: {len(passed)} passing items of the report of {report['date_utc']}")
        return 0
    if a.list:
        for it in items:
            c = it["check"]
            what = c.get("fn") or c.get("cmd") or c.get("pattern") or ", ".join(c.get("names", c.get("lines", [r["path"] for r in c.get("require", [])])))[:90] or c.get("what", "")
            print(f"{it['id']:<4} {it['tier'] or '-':<4} {c['kind']:<14} {what[:100]}")
        return 0
    if a.only:
        want = {w.strip() for w in a.only.split(",")}
        items = [it for it in items if it["id"] in want or it["theme"] in want]
    names = [n for it in items for c in (it["check"], it["check"].get("then") or {}) if c.get("kind") == "tests" for n in c["names"]]
    names.append("test_harbor_service_and_verifier_match_the_native_api")
    ctx = K.Ctx(docker=a.docker, online=a.online, test_names=names)
    import mrgym
    results = []
    try:
        for it in items:
            status, detail = run_check(ctx, it)
            detail = ctx.public(detail)
            results.append(dict(id=it["id"], theme=it["theme"], item=it["item"], tier=it["tier"], kind=it["check"]["kind"], status=status,
                                detail=detail, action=it.get("action")))
            print(f"{it['id']:<4} {status:<8} {detail[:200]}" if a.verbose or status != K.PASS else f"{it['id']:<4} {status}", flush=True)
        for name in ("gymnasium", "verifiers", "inspect", "openenv", "harbor"):      # every validator runs, also one no item reads
            ctx.framework(name)
        frameworks = {k[1]: ctx.public(v.get("summary") or "") for k, v in ctx._cache.items() if isinstance(k, tuple) and k[0] == "framework"}
        fw = {k[1]: v for k, v in ctx._cache.items() if isinstance(k, tuple) and k[0] == "framework"}
        parity = {"harbor": ctx.tests(["test_harbor_service_and_verifier_match_the_native_api"]).get("test_harbor_service_and_verifier_match_the_native_api") == "passed"
                  if "tests" in ctx._cache else None}
        parity.update({k: bool(v.get("parity_ok", v.get("demo_ok", v.get("ok")))) for k, v in fw.items() if k != "harbor"})
        commands = ctx.log
    finally:
        ctx.close()
    rows, tot = scorecard(results)
    percent = 100 * tot["points"] / tot["n"] if tot["n"] else 0.0
    report = dict(schema="mrgym-conformance/1", date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), commit=git_head(),
                  mrgym=mrgym.__version__, engine=mrgym.ENGINE_ID, ids=mrgym.list_envs(), docker=a.docker, online=a.online, python=sys.version.split()[0],
                  score=round(tot["points"] / tot["n"], 4) if tot["n"] else 0.0, percent=round(percent, 1), target=TARGET, total=tot, themes=rows,
                  frameworks=frameworks, parity=parity, items=results, commands=[dict(c, tail=c.get("tail", "")[-300:]) for c in commands])
    print()
    print(text_table(rows, tot, f"MR Gym conformance, {report['date_utc']}, commit {report['commit']}, mrgym {mrgym.__version__}, "
                                f"{len(report['ids'])} ids" + (", with Docker" if a.docker else ", without Docker") + (", online" if a.online else ", offline")))
    print("\nvalidators:")
    for name, summary in frameworks.items():
        print(f"  {name:<10} {summary}")
    print("  parity     " + "; ".join(f"{k}: {'equal to the native API' if v else 'NOT EQUAL or not run'}" for k, v in parity.items()))
    open_items = [r for r in results if r["status"] != K.PASS]
    by_tier = {}
    for r in open_items:
        by_tier.setdefault(r["tier"], []).append(f"{r['id']}{'~' if r['status'] == K.PARTIAL else ''}")
    print("\nnot passing (~ = partial), by what it takes:")
    for tier, ids in sorted(by_tier.items(), key=lambda kv: str(kv[0])):
        print(f"  {tier or '-':<4} {TIERS.get(tier, tier):<34} {' '.join(ids)}")
    regress = []
    base_path = os.path.join(HERE, "baseline.json")
    if os.path.exists(base_path) and not a.only:
        with open(base_path, encoding="utf-8") as f:
            base = set(json.load(f)["pass"])
        regress = sorted(base - {r["id"] for r in results if r["status"] == K.PASS})
        if regress:
            print(f"\nREGRESSION: items that pass in tools/conformance/baseline.json and do not pass now: {' '.join(regress)}")
    broken = [k for k, v in parity.items() if not v] if not a.only else []
    if broken:
        print(f"\nVALIDATOR OR PARITY FAILURE: {' '.join(broken)}")
    if not a.only and not a.no_write:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
        write_readme(report)
        print(f"\nwritten: {os.path.relpath(a.out, ROOT)}, integrations/README.md (generated block)")
    return 0 if percent >= 100 * TARGET and not regress and not broken else 1


if __name__ == "__main__":
    sys.exit(main())
