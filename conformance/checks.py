"""Checkers of the conformance registry (``conformance/items.json``). See ``conformance/run.py``.

Every checker returns ``(status, detail)`` with status PASS, PARTIAL or FAIL. A checker reads the tree as it is: the
registry (``mrgym.list_envs()``), stored gate and control results, files, and the output of commands it runs. Nothing
here knows a family or a tool by name, so new families are scored the moment they are registered.

Shared, expensive facts are probes on the context (``Ctx``) and are computed once per run:
    ctx.check_rows(id)       the lines of ``mrgym check <id>``
    ctx.tests(names)         outcomes of named tests of the test suite (one pytest run for all items)
    ctx.framework(name)      the result of a framework's own validator, run in that framework's environment
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

PASS, PARTIAL, FAIL = "PASS", "PARTIAL", "FAIL"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each framework's validator runs in its own throw-away environment (their pins collide: OpenEnv and verifiers pin
# different Harbor versions). A value is a command prefix; override with the environment variable CONF_<NAME>, for
# example CONF_GYM="env PATH=/path/to/venv/bin:$PATH".
UV = "uv run --isolated --no-project -p 3.12 --with pytest --with-editable ."
DEFAULT_TOOLS = {
    "GYM": f"{UV} --with gymnasium==1.4.0",
    "VF": f"{UV} --with verifiers==0.4.1 --with-editable integrations/verifiers/mrgym_scanner",
    "INSPECT": f"{UV} --with inspect-ai==0.3.292",
    "OPENENV": f"{UV} --with uvicorn --with 'openenv @ git+https://github.com/meta-pytorch/OpenEnv@3659dc7'",
    "HARBOR": "uv run --isolated --no-project -p 3.12 --with harbor==0.27.0",
}
PERSON_ITEMS = ("B12", "B13", "D9", "F8", "I5")           # evidence that only a named person can give
EVIDENCE_KEYS = ("claim", "n", "result", "who", "date", "artefact_sha256")


def agg(oks: dict, what: str = "") -> tuple[str, str]:
    """PASS when every entry holds, FAIL when none does, PARTIAL otherwise. ``oks``: {name: bool or (bool, detail)}."""
    if not oks:
        return FAIL, f"nothing to check{': ' + what if what else ''}"
    flat = {k: (v if isinstance(v, tuple) else (v, "")) for k, v in oks.items()}
    good = [k for k, (ok, _) in flat.items() if ok]
    bad = [f"{k}{' (' + d + ')' if d else ''}" for k, (ok, d) in flat.items() if not ok]
    status = PASS if not bad else FAIL if not good else PARTIAL
    return status, f"{len(good)} of {len(flat)} {what}".strip() + (f"; not: {'; '.join(bad)[:400]}" if bad else "")


class Ctx:
    def __init__(self, tools=None, docker=False, online=False, test_names=()):
        self.tools = dict(DEFAULT_TOOLS, **{k: os.environ[f"CONF_{k}"] for k in DEFAULT_TOOLS if os.environ.get(f"CONF_{k}")}, **(tools or {}))
        self.docker, self.online = docker, online
        self.tmp = tempfile.mkdtemp(prefix="mrgym_conformance_")
        self.env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=os.path.join(ROOT, "src"), MRGYM_PARITY_SEEDS="3",
                        PATH=os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + os.environ.get("PATH", ""))
        self.env.pop("MRGYM_SECRET", None)
        self._cache, self._test_names = {}, sorted(set(test_names))
        self.log = []

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ commands
    def sh(self, cmd: str, timeout: float = 900, env=None) -> tuple[int, str]:
        t0 = time.time()
        try:
            r = subprocess.run(cmd, shell=True, cwd=ROOT, env=env or self.env, capture_output=True, text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL)
            code, out = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            code, out = 124, f"timed out after {timeout:.0f} s"
        self.log.append(dict(cmd=self.public(cmd), exit=code, seconds=round(time.time() - t0, 1), tail=self.public(out[-600:])))
        return code, out

    def public(self, text: str) -> str:
        """A command or an output as it may be stored: tool prefixes by name, no path of the machine it ran on."""
        for name, prefix in self.tools.items():
            text = text.replace(prefix, "{" + name + "}")
        text = text.replace(shlex.quote(sys.executable), "python").replace(sys.executable, "python")
        text = text.replace(self.tmp, "$TMP").replace(ROOT + os.sep, "").replace(ROOT, ".")
        return re.sub(r"/(?:opt|home|mnt|Users|tmp|root|usr/local)/[^\s:'\"]*", "<path>", text)

    def py(self, args: str, **kw) -> tuple[int, str]:
        return self.sh(f"{shlex.quote(sys.executable)} {args}", **kw)

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ------------------------------------------------------------------ registry
    @property
    def ids(self):
        import mrgym
        return mrgym.list_envs()

    @property
    def scored(self):
        import mrgym
        return [i for i in self.ids if mrgym.spec(i).kind != "sandbox"]

    def controls(self, env_id):
        import mrgym
        from mrgym import policies
        return policies.controls_of(mrgym.spec(env_id))

    def task(self, env_id):
        import mrgym
        return importlib.import_module(mrgym.spec(env_id).family + ".task")

    def card(self, env_id):
        import mrgym
        return self.cached(("card", env_id), lambda: mrgym.card(env_id))

    def gate(self, env_id):
        from mrgym import gate
        return gate.load(env_id)

    def episodes(self, env_id, n=6):
        """Scripted reference episodes of an id at its default level on seeds 0..n-1 (calls, reward, seconds)."""
        from mrgym import compat

        def run():
            out = []
            for seed in range(n):
                t0 = time.process_time()
                ep = compat.scripted_episode(env_id, None, seed)
                out.append(dict(ep, cpu_s=time.process_time() - t0))
            return out
        return self.cached(("episodes", env_id, n), run)

    # ------------------------------------------------------------------ probes
    def check_rows(self, env_id) -> dict:
        def run():
            from mrgym import check
            try:
                return {name: (ok, detail) for name, ok, detail in check.check_env(env_id)}
            except Exception as e:  # noqa: BLE001  a crash of the check is a finding, not a crash of the scorecard
                return {"mrgym check ran to the end": (False, f"{type(e).__name__}: {e}"[:300])}
        return self.cached(("check", env_id), run)

    def tests(self, names) -> dict:
        """{name: 'passed' | 'failed' | 'skipped' | 'missing'} for test function names (parametrised cases are merged)."""
        def run():
            if not self._test_names:
                return {}
            xml = os.path.join(self.tmp, "junit.xml")
            expr = " or ".join(self._test_names)
            self.py(f"-m pytest -q -p no:cacheprovider tests -k {shlex.quote(expr)} --junitxml={shlex.quote(xml)}", timeout=1800)
            out = {}
            try:
                for case in ET.parse(xml).getroot().iter("testcase"):
                    name = case.get("name", "").split("[")[0]
                    state = "failed" if case.find("failure") is not None or case.find("error") is not None else \
                        "skipped" if case.find("skipped") is not None else "passed"
                    rank = {"failed": 3, "skipped": 2, "passed": 1}
                    if rank[state] > rank.get(out.get(name), 0):
                        out[name] = state
            except (OSError, ET.ParseError):
                pass
            return out
        res = self.cached("tests", run)
        return {n: res.get(n, "missing") for n in names}

    def framework(self, name) -> dict:
        return self.cached(("framework", name), lambda: getattr(self, "_fw_" + name)())

    def _json_tail(self, out):
        """The last JSON object in a command's output."""
        start = out.find("{")
        while start != -1:
            try:
                return json.loads(out[start:out.rindex("}") + 1])
            except ValueError:
                start = out.find("{", start + 1)
        return None

    def _fw_gymnasium(self):
        code, out = self.sh(f"{self.tools['GYM']} python -m mrgym.compat.gymnasium --check --json")
        data = self._json_tail(out) or {}
        rows = data.get("rows", [])
        pcode, pout = self.sh(f"{self.tools['GYM']} python -m pytest -q -p no:cacheprovider tests/test_adapters_parity.py -k gymnasium")
        return dict(ran=bool(rows), exit=code, version=data.get("gymnasium"), rows=rows, ok=code == 0 and bool(rows) and all(r["ok"] for r in rows),
                    parity_ok=pcode == 0 and " passed" in pout and "skipped" not in pout.splitlines()[-1],
                    summary=f"gymnasium {data.get('gymnasium')}: check_env PASS with 0 warnings on {sum(r['ok'] for r in rows)} of {len(rows)} ids"
                    if rows else "the Gymnasium check did not run: " + out[-200:].strip())

    def native_passes(self, env_id, n=3) -> list:
        """Whether the scripted reference passes natively on seeds 0..n-1 of the id's default level."""
        return [bool(e["passed"]) for e in self.episodes(env_id)[:n]]

    def _fw_verifiers(self):
        """``vf-validate`` once per scored id, 3 instances each. Gold must be valid on every instance whose native
        reference passes (and invalid where it does not: the validator and the native API agree); no-op must be valid
        with reward 0.0 everywhere."""
        n, rows = 3, {}
        for env_id in self.scored:
            out_dir = os.path.join(self.tmp, "vf", env_id.replace("/", "-"))
            code, out = self.sh(f"{self.tools['VF']} vf-validate mrgym-scanner -n {n} -c 2 --runtime.type subprocess "
                                f"--taskset.env-id {shlex.quote(env_id)} -o {shlex.quote(out_dir)}")
            summary, rewards = None, []
            for path in glob.glob(os.path.join(out_dir, "*", "summary.json")):
                with open(path, encoding="utf-8") as f:
                    summary = json.load(f)
            for path in glob.glob(os.path.join(out_dir, "*", "results.jsonl")):
                with open(path, encoding="utf-8") as f:
                    rewards = [(json.loads(line).get("noop") or {}).get("reward") for line in f if line.strip()]
            checks = (summary or {}).get("checks", {})
            rows[env_id] = dict(ran=bool(summary), gold=checks.get("gold", {}).get("valid"), noop=checks.get("noop", {}).get("valid"),
                                noop_zero=sum(r == 0.0 for r in rewards), native=sum(self.native_passes(env_id, n)))
        vcode, vout = self.sh(f"{self.tools['VF']} python -c \"import importlib.metadata as m; print(m.version('verifiers'))\"")
        pcode, pout = self.sh(f"{self.tools['VF']} python -m pytest -q -p no:cacheprovider tests/test_adapters_parity.py -k verifiers")
        ran = bool(rows) and all(r["ran"] for r in rows.values())
        ok = ran and all(r["gold"] == r["native"] and r["noop"] == n and r["noop_zero"] == n for r in rows.values()) and any(r["native"] for r in rows.values())
        skipped = [f"{i} ({r['native']} of {n})" for i, r in rows.items() if r["native"] < n]
        version = vout.strip().splitlines()[-1] if vcode == 0 and vout.strip() else "?"
        return dict(ran=ran, version=version, n=n, rows=rows, ok=ok,
                    parity_ok=pcode == 0 and " passed" in pout and "skipped" not in pout.splitlines()[-1],
                    summary=(f"verifiers {version} vf-validate, {n} instances per scored id: gold valid "
                             + ", ".join(f"{i} {r['gold']}/{n}" for i, r in rows.items()) + f"; no-op valid with reward 0.0 on "
                             f"{sum(r['noop_zero'] for r in rows.values())} of {n * len(rows)}"
                             + (f"; gold is valid exactly where the native reference passes; it does not pass natively on every instance of: {', '.join(skipped)}"
                                if skipped else "")) if ran else "vf-validate did not run")

    def _fw_inspect(self):
        code, out = self.sh(f"{self.tools['INSPECT']} python integrations/inspect/mock_run.py --seeds 3 --json", timeout=1800)
        data = self._json_tail(out) or {}
        ran = "episodes" in data
        return dict(ran=ran, exit=code, version=data.get("inspect_ai"), data={k: v for k, v in data.items() if k != "rows"},
                    ok=ran and code == 0 and data["ok"] == data["episodes"] and data["reference_reward_1"] == data["scored_episodes"]
                    and data["noop_reward_0"] == data["episodes"] and data["scored_episodes"] > 0,
                    undescribed=data.get("undescribed_parameters"),
                    summary=(f"inspect-ai {data.get('inspect_ai')} mock model: {data.get('ok')} of {data.get('episodes')} episodes equal the native API; "
                             f"reference reward 1.0 on {data.get('reference_reward_1')} of {data.get('scored_episodes')} episodes whose native "
                             f"reference passes; no-op reward 0.0 on {data.get('noop_reward_0')} of {data.get('episodes')}"
                             + (f"; the reference does not pass natively for: {', '.join(data['native_reference_does_not_pass'])}"
                                if data.get("native_reference_does_not_pass") else ""))
                    if ran else "the Inspect mock run did not run: " + out[-200:].strip())

    def _fw_openenv(self):
        report = os.path.join(self.tmp, "openenv.json")
        pkg = "integrations/openenv/mrgym_scanner_env"
        code, out = self.sh(f"{self.tools['OPENENV']} openenv validate {pkg} --level static --skip-build --json --output {shlex.quote(report)}")
        checks = {}
        try:
            with open(report, encoding="utf-8") as f:
                data = json.load(f)
            for c in data.get("checks", data.get("results", [])):
                checks[c.get("check_id", c.get("id"))] = str(c.get("status", "")).upper().split(".")[-1]
        except (OSError, ValueError):
            data = None
        dcode, dout = self.sh(f"{self.tools['OPENENV']} python integrations/openenv/demo.py")
        n = {s: sum(v == s for v in checks.values()) for s in ("PASS", "FAIL", "SKIP", "WARN")}
        static_ok = code == 0 and checks.get("static.manifest") == "PASS" and n["FAIL"] == 0
        runtime = None
        if self.docker:
            rcode, rout = self.sh(f"{self.tools['OPENENV']} openenv validate {pkg} --level runtime", timeout=3600)
            runtime = rcode == 0 and "FAIL" not in rout
        return dict(ran=data is not None, exit=code, checks=checks, static_ok=static_ok, demo_ok=dcode == 0, runtime_ok=runtime,
                    summary=(f"openenv validate --level static --skip-build: static.manifest {checks.get('static.manifest')}, {n['SKIP']} SKIP "
                             f"(graders not implemented in that build), {n['FAIL']} FAIL, exit {code}; demo in process and over HTTP + WebSocket: "
                             f"{'same reward and digest as the native API' if dcode == 0 else 'MISMATCH'}; runtime level: "
                             f"{'not run (needs Docker)' if runtime is None else 'PASS' if runtime else 'FAIL'}")
                    if data is not None else "openenv validate did not run: " + out[-200:].strip())

    def _fw_harbor(self):
        from mrgym.compat import export, harbor
        tasks = export.harbor_tasks(os.path.join(ROOT, "datasets"))
        stale = export.check_harbor(os.path.join(ROOT, "datasets"))
        own = {t: harbor.lint(t) for t in tasks}
        from mrgym import compat
        local, oracle_pass, native_pass = {}, 0, 0
        for t in tasks:
            try:
                ep = harbor.load_episode(os.path.join(t, "tests", "episode.json"))
                native = compat.scripted_episode(ep["env_id"], ep["level"], ep["seed"], split=ep["split"])
                o, z = harbor.local(t, "reference"), harbor.local(t, "nop")
                local[t] = ((o["reward"], o["log_head"], o["passed"]) == (native["reward"], native["log_head"], native["passed"])
                            and o["log_replays"] and o["sealed"] and z["reward"] == 0.0 and not z["passed"])
                oracle_pass, native_pass = oracle_pass + bool(o["passed"]), native_pass + bool(native["passed"])
            except Exception as e:  # noqa: BLE001
                local[t] = False
                self.log.append(dict(cmd=f"harbor.local {os.path.basename(t)}", exit=1, tail=f"{type(e).__name__}: {e}"[:300]))
        # Harbor's own lint. Its Compose rules need the Docker CLI with the Compose plugin; without it the lint exits 2,
        # so the remaining 13 rules are run on a copy without the Compose file and the Compose rules are reported as not run.
        rel = [os.path.relpath(t, ROOT) for t in tasks]
        code_full, out_full = self.sh(f"{self.tools['HARBOR']} harbor lint --extend-select ALL {' '.join(map(shlex.quote, rel))}") if tasks else (1, "")
        copy = os.path.join(self.tmp, "harbor_nocompose")
        shutil.copytree(os.path.join(ROOT, "datasets"), copy)
        for path in glob.glob(os.path.join(copy, "*", "tasks", "*", "environment", "docker-compose.yaml")):
            os.remove(path)
        nc = export.harbor_tasks(copy)
        code, out = self.sh(f"{self.tools['HARBOR']} harbor lint --json --extend-select ALL {' '.join(map(shlex.quote, nc))}") if nc else (1, "")
        try:
            findings = json.loads(out[out.index("["):out.rindex("]") + 1])
        except ValueError:
            findings = None
        docker_runs = None
        if self.docker:
            docker_runs = code_full == 0
            for t in rel:
                ep = harbor.load_episode(os.path.join(ROOT, t, "tests", "episode.json"))
                passes = compat.scripted_episode(ep["env_id"], ep["level"], ep["seed"], split=ep["split"])["passed"]
                for agent, want in (("oracle", "1" if passes else "0"), ("nop", "0")):
                    c, o = self.sh(f"{self.tools['HARBOR']} harbor run -p {shlex.quote(t)} -a {agent}", timeout=3600)
                    docker_runs = docker_runs and c == 0 and re.search(rf"reward\D+{want}(\.0)?\b", o) is not None
        return dict(ran=findings is not None, tasks=len(tasks), stale=stale, own_findings=sum(map(len, own.values())), local_ok=bool(local) and all(local.values()),
                    harbor_findings=findings, compose_lint_exit=code_full, docker_ok=docker_runs,
                    static_ok=bool(tasks) and not stale and not any(own.values()) and findings == [] and code == 0,
                    summary=(f"{len(tasks)} exported task(s): `python -m mrgym.compat.harbor lint` {sum(map(len, own.values()))} finding(s); the scanner "
                             f"service, the reference over HTTP and the verifier without Docker give the native reward and scanner-log head on "
                             f"{sum(local.values())} of {len(local)} (oracle passes on {oracle_pass}, as natively on {native_pass}; no-op 0 on all); "
                             f"`harbor lint --extend-select ALL` on a copy "
                             f"without the Compose file: {len(findings) if findings is not None else 'did not run'} finding(s), exit {code}; with the Compose "
                             f"file: exit {code_full} ({'clean' if code_full == 0 else 'the Compose rules need the Docker CLI with the Compose plugin'}); "
                             f"`harbor run -a oracle|nop`: {'not run (needs a Docker daemon)' if docker_runs is None else 'PASS' if docker_runs else 'FAIL'}"))


# ====================================================================================================== generic kinds
def kind_tests(ctx, check):
    res = ctx.tests(check["names"])
    return agg({n: (s == "passed", s if s != "passed" else "") for n, s in res.items()}, "named tests pass")


def kind_tests_matching(ctx, check):
    """At least one test function whose name matches ``pattern`` exists; every one of them passes."""
    pattern = re.compile(check["pattern"])
    names = set()
    for path in glob.glob(os.path.join(ROOT, "tests", "test_*.py")):
        with open(path, encoding="utf-8") as f:
            names |= {m for m in re.findall(r"^def (test_\w+)", f.read(), re.M) if pattern.search(m)}
    if not names:
        return FAIL, f"no shipped test matches /{check['pattern']}/"
    code, out = ctx.py(f"-m pytest -q -p no:cacheprovider tests -k {shlex.quote(' or '.join(sorted(names)))}", timeout=1800)
    return (PASS if code == 0 else FAIL), f"{len(names)} test(s) matching /{check['pattern']}/: " + out.strip().splitlines()[-1][:200]


def kind_check(ctx, check):
    """Lines of ``mrgym check <id>``: every listed pattern must match an ok line, for every id it applies to."""
    oks = {}
    for env_id in (ctx.scored if check.get("scored_only") else ctx.ids):
        rows = ctx.check_rows(env_id)
        for pat in check["lines"]:
            hit = [(ok, d) for name, (ok, d) in rows.items() if re.search(pat, name)]
            oks[f"{env_id}: {pat[:40]}"] = (bool(hit) and all(ok for ok, _ in hit), "no such line" if not hit else "; ".join(d for ok, d in hit if not ok)[:160])
    return agg(oks, "check lines ok")


def kind_cmd(ctx, check):
    cmd = check["cmd"].replace("{PY}", shlex.quote(sys.executable))
    for k, v in ctx.tools.items():
        cmd = cmd.replace("{" + k + "}", v)
    code, out = ctx.sh(cmd, timeout=check.get("timeout", 900))
    ok = code == 0 and (not check.get("expect") or re.search(check["expect"], out) is not None)
    return (PASS if ok else FAIL), f"`{check['cmd'][:120]}` -> exit {code}" + ("" if ok else f"; {out.strip().splitlines()[-1][:200] if out.strip() else 'no output'}")


def kind_files(ctx, check):
    oks = {}
    for req in check["require"]:
        paths = glob.glob(os.path.join(ROOT, req["path"]))
        if len(paths) < req.get("min", 1):
            oks[req["path"]] = (False, "missing")
            continue
        text = ""
        for p in paths:
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    text += f.read()
        missing = [pat for pat in req.get("contains", []) if not re.search(pat, text, re.I | re.M)]
        present = [pat for pat in req.get("absent", []) if re.search(pat, text, re.I | re.M)]
        oks[req["path"]] = (not missing and not present, "; ".join([f"lacks /{m}/" for m in missing] + [f"has /{p}/" for p in present]))
    return agg(oks, "file requirements hold")


def kind_evidence(ctx, check, item_id):
    path = os.path.join(ROOT, "conformance", "evidence", f"{item_id}.json")
    if not os.path.exists(path):
        return FAIL, f"no evidence file conformance/evidence/{item_id}.json; action: {check.get('action', 'see items.json')}"
    try:
        with open(path, encoding="utf-8") as f:
            ev = json.load(f)
    except ValueError:
        return FAIL, "the evidence file is not JSON"
    missing = [k for k in EVIDENCE_KEYS if not ev.get(k) and ev.get(k) != 0]
    if missing:
        return FAIL, f"the evidence file lacks {missing}"
    who = str(ev["who"])
    if item_id in PERSON_ITEMS and (ev.get("who_is") != "person" or re.search(r"\b(agent|model|claude|gpt|gemini|bot|llm)\b", who, re.I)):
        return FAIL, "this item needs a named person (\"who_is\": \"person\"); an agent or a model cannot sign it"
    return (PASS if ev.get("result") == "pass" else PARTIAL if ev.get("result") == "partial" else FAIL), f"evidence by {who}, {ev['date']}: {str(ev['claim'])[:160]}"


# ====================================================================================================== native checks
def card_states_the_skill(ctx):
    return agg({i: all(ctx.card(i).get(k) for k in ("summary", "given", "deliver", "scored_by")) for i in ctx.scored}, "scored cards state skill, given, deliver, scored_by")


def gymnasium_check_env(ctx):
    fw = ctx.framework("gymnasium")
    if not fw["ran"]:
        return FAIL, fw["summary"]
    return agg({r["id"]: (r["ok"], r.get("error") or f"{len(r['warnings'])} warning(s)" if not r["ok"] else "") for r in fw["rows"]},
               f"ids pass gymnasium {fw['version']} check_env with 0 warnings")


def gymnasium_registered(ctx):
    fw = ctx.framework("gymnasium")
    if not fw["ran"]:
        return FAIL, fw["summary"]
    ok = fw["ok"] and fw["parity_ok"] and {r["id"] for r in fw["rows"]} == set(ctx.ids)
    return (PASS if ok else FAIL), f"gym.make works for {len(fw['rows'])} of {len(ctx.ids)} registered ids after `import mrgym.compat.gymnasium`; parity test: {fw['parity_ok']}"


def rules_in_statement(ctx):
    """A6: every number the verifier uses is in a table ``task.RULES`` and each value appears in the reset text."""
    import mrgym
    oks = {}
    for i in ctx.scored:
        rules = getattr(ctx.task(i), "RULES", None)
        if not isinstance(rules, dict) or not rules:
            oks[i] = (False, "no task.RULES table")
            continue
        env = mrgym.make(i, seed=0)
        text = env.reset()[0].text
        env.close()
        missing = [k for k, v in rules.items() if f"{v:g}" not in text and str(v) not in text]
        oks[i] = (not missing, f"not in the reset text: {missing[:4]}")
    return agg(oks, "scored ids render every rule value into the reset text")


UNIT_SUFFIX = re.compile(r"_(ms|s|mm|cm|m|deg|rad|hz|khz|ppm|pct|percent|t|ut|px|n|count|index|dir|ratio|fraction)$")
UNIT_WORD = re.compile(r"\b(ms|milliseconds?|seconds?|mm|millimet\w+|degrees?|deg|rad|hz|khz|ppm|percent|%|tesla|pixels?|lines?|index|count|"
                       r"ratio|fraction|unitless|dimensionless|number of)\b", re.I)


def deliverable_has_units(ctx):
    """A7: every numeric field of the deliverable schema names its unit (in its name or its description)."""
    import mrgym
    oks = {}

    def walk(schema, path, bad, inherited=""):
        desc = schema.get("description", "") or inherited
        named = any(UNIT_SUFFIX.search(part) for part in path.lower().split("."))        # ti_ms.R1: the unit is in the parent's name
        if schema.get("type") in ("number", "integer") and not (named or UNIT_WORD.search(desc)):
            bad.append(path)
        for k, sub in (schema.get("properties") or {}).items():
            walk(sub, f"{path}.{k}" if path else k, bad, "")
        if isinstance(schema.get("additionalProperties"), dict):
            walk(schema["additionalProperties"], path, bad, desc)
        if isinstance(schema.get("items"), dict):
            walk(schema["items"], path, bad, desc)
        for sub in schema.get("anyOf", []):
            walk(sub, path, bad, desc)

    for i in ctx.scored:
        env = mrgym.make(i, seed=0)
        env.reset()
        term = next(t for t in env.tools("neutral") if t["terminal"])
        env.close()
        bad = []
        walk(term["parameters"], "", bad)
        oks[i] = (not bad, f"no unit for: {bad[:5]}")
    return agg(oks, "deliverable schemas name a unit for every number")


def outcome_not_process(ctx):
    """A8: no process gate. The family rule keeps one: the log must show a measurement (lead's decision 1)."""
    from mrgym.envs import family
    if "measured" in getattr(family, "GATES", ()):
        return PARTIAL, "one process gate is kept by design: the scanner log must show a measurement within the budget (mrgym.envs.family.GATES)"
    return PASS, "no process gate in mrgym.envs.family.GATES"


def reset_stream(ctx):
    """A9: reset(seed), reset(), reset() is a reproducible stream of distinct episodes."""
    import mrgym
    from mrgym import compat

    def stream(i):
        env = mrgym.make(i, seed=0)
        ids = [compat.instance_label(env.reset()[1]) for _ in range(3)]           # the instance, not the (random) episode id
        env.close()
        return ids
    core = {i: len(set(stream(i))) == 3 and stream(i) == stream(i) for i in ctx.ids[:2]}
    fw = ctx.framework("gymnasium")
    adapter = fw["ran"] and all(r["distinct_worlds"] == 3 and r["reproducible"] for r in fw["rows"])
    if all(core.values()):
        return PASS, "mrgym.Env: reset() without a seed gives distinct, reproducible worlds"
    if adapter:
        return PARTIAL, ("the Gymnasium adapter gives a reproducible stream of distinct worlds on every id; mrgym.Env.reset() without a seed "
                         "repeats the constructor seed's world")
    return FAIL, "neither the core nor the Gymnasium adapter gives a seeded stream"


def reference_on_fresh_instances(ctx, n_min=200, rate_min=0.97):
    """B2: the stored gate holds the reference on >= 200 fresh instances per level, >= 97 %, on the current engine."""
    import mrgym
    oks, any_stored = {}, False
    for i in ctx.scored:
        g = ctx.gate(i)
        if not g:
            oks[i] = (False, "no stored gate result")
            continue
        any_stored = True
        bad = [f"L{lv}: {v['reference']['passed']}/{v['n']}" for lv, v in g["levels"].items()
               if v["n"] < n_min or v["reference"]["rate"] < rate_min]
        oks[i] = (not bad and g.get("engine") == mrgym.ENGINE_ID, "; ".join(bad) or "another engine")
    status, detail = agg(oks, f"scored ids store the reference on >= {n_min} fresh instances per level at >= {100 * rate_min:.0f} %")
    return (PARTIAL if status == FAIL and any_stored else status), detail


def second_reference(ctx):
    oks = {}
    for i in ctx.scored:
        names = ctx.controls(i).policies()
        oks[i] = ("reference_b" in names, "no policy named reference_b")
    return agg(oks, "scored ids ship a second, independently written reference (reference_b)")


def mutants(ctx, n_min=8):
    """B6: >= 8 known-wrong variants of the reference per family (``controls.MUTANTS``, or policies named wrong_*)."""
    oks, some = {}, False
    for i in ctx.scored:
        c = ctx.controls(i)
        names = list(getattr(c, "MUTANTS", ())) or [n for n in c.policies() if n.startswith(("wrong_", "mutant_"))]
        some = some or bool(names)
        oks[i] = (len(names) >= n_min, f"{len(names)} mutant(s)")
    status, detail = agg(oks, f"scored ids ship >= {n_min} mutants of the reference")
    return (PARTIAL if status == FAIL and some else status), detail


def verifier_rescans(ctx):
    """B9, B14: the scoring function is handed the hidden world or the engine and uses it (a re-scan, the console, or
    the world's own truth): it measures, it does not match text or apply the formula given to the agent."""
    import inspect
    oks = {}
    for i in ctx.scored:
        score = getattr(ctx.task(i), "score", None)
        if score is None:
            oks[i] = (False, "the task module has no score function")
            continue
        hidden = [p for p in inspect.signature(score).parameters if p in ("world", "engine")]
        src = inspect.getsource(ctx.task(i))
        used = re.search(r"noise_free\(|region_complex\(|\.acquire\(|console\.plan\(|\bworld\[|\bworld\.get\(", src) is not None
        oks[i] = (bool(hidden) and used, "score() takes neither the hidden world nor the engine" if not hidden else "the task module never reads them")
    return agg(oks, "task modules score on the hidden world (a noise-free re-scan, the console or the world's truth)")


def no_llm_or_network_in_scoring(ctx):
    bad = []
    for path in glob.glob(os.path.join(ROOT, "src", "mrgym", "envs", "**", "*.py"), recursive=True) + [os.path.join(ROOT, "src", "mrgym", "env.py")]:
        with open(path, encoding="utf-8") as f:
            if re.search(r"^\s*(import|from)\s+(openai|anthropic|requests|httpx|urllib|http\.client|socket)\b", f.read(), re.M):
                bad.append(os.path.relpath(path, ROOT))
    return (PASS if not bad else FAIL), "no model client and no network import in the environment and scoring code" if not bad else f"network imports in {bad}"


def tolerance_provenance(ctx):
    oks = {}
    for i in ctx.scored:
        rows = ctx.card(i).get("tolerances")
        oks[i] = (bool(rows) and all(r.get("derivation") for r in rows), "the card has no `tolerances` table with a derivation per constant")
    return agg(oks, "scored cards carry a tolerance table with provenance")


def _gate_levels(ctx):
    """[(id, level, level result)] of every stored gate; and the scored ids without one."""
    rows, missing = [], []
    for i in ctx.scored:
        g = ctx.gate(i)
        if not g:
            missing.append(i)
        else:
            rows += [(i, lv, v) for lv, v in g["levels"].items()]
    return rows, missing


def _gate_bound(ctx, policy_pattern, limit, what):
    rows, missing = _gate_levels(ctx)
    oks = {i: (False, "no stored gate result") for i in missing}
    for i, lv, v in rows:
        hit = [r for r in v.get("must_fail", []) if re.search(policy_pattern, r["policy"])]
        oks[f"{i} L{lv}"] = (bool(hit) and all(r["upper95"] <= limit for r in hit),
                             "no such policy in the gate" if not hit else ", ".join(f"{r['policy']} {r['passed']}/{v['n']}" for r in hit if r["upper95"] > limit))
    return agg(oks, what)


def best_constant_fails(ctx):
    return _gate_bound(ctx, r"^fixed_recipe", 0.20, "levels where the best constant answer has a 95 % upper bound <= 20 %")


def blind_sweep_fails(ctx):
    return _gate_bound(ctx, r"^brute_force", 0.10, "levels where the blind sweep has a 95 % upper bound <= 10 %")


def swap_fails(ctx):
    rows, missing = _gate_levels(ctx)
    oks = {i: (False, "no stored gate result") for i in missing}
    for i, lv, v in rows:
        s = v.get("swap")
        oks[f"{i} L{lv}"] = (bool(s) and s.get("rate", 1.0) <= 0.01, "the gate stores no swap probe" if not s else f"swap passes {100 * s['rate']:.1f} % of pairs")
    return agg(oks, "levels where another instance's answer passes <= 1 % of pairs")


def decision_blind_fails(ctx):
    rows, missing = _gate_levels(ctx)
    oks = {i: (False, "no stored gate result") for i in missing}
    for i, lv, v in rows:
        abl = v.get("ablations", [])
        oks[f"{i} L{lv}"] = (bool(abl) and all(r.get("cleared") for r in abl), "no ablation listed" if not abl else
                             ", ".join(f"{r['policy']} loses {100 * r['gap']:.0f} points" for r in abl if not r.get("cleared")))
    return agg(oks, "levels where every listed ablation of the reference's decision loses >= 25 points")


def copy_rule_fails(ctx):
    """C3: the task statement carries no instance-specific number to copy (it is the same text for every instance of
    a level), or the family lists a copy policy among its must-fail controls."""
    import mrgym
    oks = {}
    for i in ctx.scored:
        texts = set()
        for seed in range(4):
            env = mrgym.make(i, seed=seed)
            texts.add(env.reset()[0].text)
            env.close()
        # an answer built only from what the statement says: the gate's tuned constant (`fixed_recipe`) or a policy named copy*
        g = ctx.gate(i)
        blind = [r for v in (g or {}).get("levels", {}).values() for r in v.get("must_fail", []) if re.search(r"^fixed_recipe|copy", r["policy"])]
        cleared = bool(blind) and all(r["upper95"] <= 0.10 for r in blind)
        oks[i] = (len(texts) == 1 or cleared, f"{len(texts)} distinct statements on 4 seeds, and no stored gate result clears a statement-only policy")
    return agg(oks, "scored ids where a rule that uses only the statement has nothing to copy or fails the stored gate")


def seed_attack(ctx, n=6):
    """C5: a seed_attacker policy exists and passes nowhere on a secret split."""
    import mrgym
    from mrgym import policies
    oks, some = {}, False
    old = os.environ.get("MRGYM_SECRET")
    os.environ["MRGYM_SECRET"] = os.urandom(32).hex()                    # a throw-away secret of the required length
    try:
        for i in ctx.scored:
            table = ctx.controls(i).policies()
            if "seed_attacker" not in table:
                oks[i] = (False, "no seed_attacker policy")
                continue
            entry = table["seed_attacker"]
            fn = entry[0] if isinstance(entry, tuple) else entry
            passes = 0
            for seed in range(n):
                try:
                    r = policies.run(mrgym.make(i, seed=seed, split="heldout"), fn)
                    passes += r.reward >= mrgym.spec(i).reward_threshold
                except Exception as e:  # noqa: BLE001
                    oks[i] = (False, f"the policy crashed: {type(e).__name__}")
                    break
            else:
                some = True
                oks[i] = (passes == 0, f"passes {passes} of {n} secret instances")
    finally:
        if old is None:
            os.environ.pop("MRGYM_SECRET", None)
        else:
            os.environ["MRGYM_SECRET"] = old
    status, detail = agg(oks, f"scored ids whose seed_attacker passes 0 of {n} instances of a secret split")
    return (PARTIAL if status == FAIL and some else status), detail


def verifier_isolated(ctx):
    fw = ctx.framework("harbor")
    if fw["docker_ok"]:
        return PASS, fw["summary"]
    if fw["static_ok"] and fw["local_ok"]:
        return PARTIAL, "separate verifier and non-root agent are declared and lint clean; containers were not run. " + fw["summary"]
    return FAIL, fw["summary"]


def canary_and_honeypot(ctx):
    from mrgym import compat
    from mrgym.compat import export
    tasks = export.harbor_tasks(os.path.join(ROOT, "datasets"))
    missing = []
    for t in tasks:
        for rel in ("task.toml", "README.md", "solution/solve.sh", "tests/test.sh"):
            with open(os.path.join(t, rel), encoding="utf-8") as f:
                if compat.CANARY not in f.read():
                    missing.append(os.path.join(t, rel))
        with open(os.path.join(t, "instruction.md"), encoding="utf-8") as f:        # never in what the agent reads
            if compat.CANARY in f.read():
                missing.append(os.path.join(t, "instruction.md") + " (carries the canary; the agent reads this file)")
    shortcuts = os.path.exists(os.path.join(ROOT, "conformance", "shortcuts.json"))
    honeypot = any("honeypot" in open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(ROOT, "src", "mrgym", "envs", "**", "*.py"), recursive=True))
    canary = bool(tasks) and not missing
    if canary and shortcuts and honeypot:
        return PASS, "canary in every exported task file; shortcut registry and decoy answer present"
    if canary:
        return PARTIAL, (f"canary in the task, solution and verifier files of {len(tasks)} exported task(s) and not in the agent's instruction; shortcut registry conformance/shortcuts.json: "
                         f"{'present' if shortcuts else 'absent'}; decoy answer (honeypot): {'present' if honeypot else 'absent'}")
    return FAIL, f"canary missing in {missing[:3]}" if tasks else "no exported task"


def concentration_reported(ctx):
    rows, missing = _gate_levels(ctx)
    oks = {i: (False, "no stored gate result") for i in missing}
    for i, lv, v in rows:
        oks[f"{i} L{lv}"] = ("concentration" in v, "the gate stores no concentration of passing answers")
    return agg(oks, "levels whose gate result reports the concentration of passing answers")


def reward_means_solved(ctx, cap=0.3):
    """C13: no policy that does not solve the task earns more than 0.3 on average (stored control results)."""
    oks = {}
    for i in ctx.scored:
        g = ctx.gate(i)
        if g:                                                # pass-only reward: the mean reward of a policy is its pass rate
            worst = max((r["rate"] for v in g["levels"].values() for r in v.get("must_fail", [])), default=None)
            oks[i] = (worst is not None and worst <= cap, f"a must-fail policy passes {100 * (worst or 0):.0f} %")
        else:
            oks[i] = (False, "no stored control results")
    return agg(oks, f"scored ids where no non-reference policy earns a mean reward above {cap}")


def multi_step(ctx, n_min=8):
    oks = {}
    for i in ctx.scored:
        med = statistics.median(len(e["calls"]) for e in ctx.episodes(i))
        oks[i] = (med >= n_min, f"the reference needs a median of {med:g} tool calls")
    return agg(oks, f"scored ids whose reference needs >= {n_min} tool calls")


def several_families(ctx, n_min=4):
    import mrgym
    admitted = [i for i in ctx.scored if mrgym.spec(i).kind == "admitted" and (ctx.gate(i) or {}).get("levels")
                and all(v["verdict"] == "admitted" for v in ctx.gate(i)["levels"].values())]
    status = PASS if len(admitted) >= n_min else PARTIAL if len(admitted) >= 2 else FAIL
    return status, f"{len(admitted)} families are listed as admitted with a stored gate that admits every level (need {n_min}); {len(ctx.scored)} scored ids registered"


def enough_gated_instances(ctx, n_min=200):
    import mrgym
    admitted = [i for i in ctx.scored if mrgym.spec(i).kind == "admitted"]
    if not admitted:
        return FAIL, "no id is listed as admitted"
    oks = {}
    for i in admitted:
        g = ctx.gate(i)
        oks[i] = (bool(g) and all(v["n"] >= n_min and v["verdict"] == "admitted" for v in g["levels"].values()),
                  "no stored gate result" if not g else "a level has fewer instances or is not admitted")
    return agg(oks, f"admitted ids with a stored gate on >= {n_min} fresh instances per level")


def instance_changes_answer(ctx):
    oks = {}
    for i in ctx.scored:
        answers = {json.dumps(e["calls"][-1].get("arguments"), sort_keys=True) for e in ctx.episodes(i) if e["calls"]}
        oks[i] = (len(answers) >= 3, f"{len(answers)} distinct reference answers on {len(ctx.episodes(i))} seeds")
    return agg(oks, "scored ids whose reference answer differs between instances")


def heldout(ctx):
    import mrgym
    try:
        for name in ("MRGYM_SECRET", "MRGYM_SECRET_FILE"):
            os.environ.pop(name, None)
        mrgym.make(ctx.scored[0], seed=0, split="heldout").reset()
        return FAIL, "the held-out split runs without a secret"
    except mrgym.Error:
        mech = True
    regime = {i: bool(ctx.card(i).get("heldout_regime")) for i in ctx.scored}
    if mech and all(regime.values()):
        return PASS, "secret split and a declared held-out parameter regime on every scored card"
    return PARTIAL, (f"the held-out split refuses to run without MRGYM_SECRET; a held-out parameter regime is declared on "
                     f"{sum(regime.values())} of {len(regime)} scored cards and no held-out result row is checked")


def shared_tools(ctx):
    import mrgym
    admitted = [i for i in ctx.scored if mrgym.spec(i).kind == "admitted"]
    sets = {}
    for i in admitted:
        env = mrgym.make(i, seed=0)
        env.reset()
        sets[i] = tuple(t["name"] for t in env.tools("neutral") if not t["terminal"])
        env.close()
    ok = len(admitted) >= 2 and len(set(sets.values())) == 1
    return (PASS if ok else FAIL), f"{len(admitted)} admitted id(s); they share one tool set: {len(set(sets.values())) == 1 if sets else 'n/a'} (need >= 2 admitted)"


def cards_complete(ctx):
    from mrgym import cards
    authored = [k for k in cards.REQUIRED if k not in ("gate", "status")]
    return agg({i: (all(k in ctx.card(i) for k in authored), "missing " + str([k for k in authored if k not in ctx.card(i)])) for i in ctx.ids},
               "cards carry every authored field")


DATASHEET_SECTIONS = ("Motivation", "Composition", "Collection process", "Preprocessing", "Uses", "Distribution", "Maintenance")


def standard_metadata(ctx):
    path = os.path.join(ROOT, "DATASHEET.md")
    text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    heads = [s for s in DATASHEET_SECTIONS if re.search(rf"^##+ .*{s}", text, re.M | re.I)]
    card = os.path.exists(os.path.join(ROOT, "BENCHMARK_CARD.md"))
    croissant = glob.glob(os.path.join(ROOT, "docs", "croissant", "*.json"))
    if len(heads) == len(DATASHEET_SECTIONS) and card and croissant:
        return PASS, "datasheet in the seven standard sections, benchmark card and Croissant files present"
    if text:
        return PARTIAL, (f"DATASHEET.md exists with {len(heads)} of the 7 standard section headings; BENCHMARK_CARD.md: {'present' if card else 'absent'}; "
                         f"Croissant files: {len(croissant)}")
    return FAIL, "no datasheet"


def docs_agree(ctx):
    code1, out1 = ctx.py("tools/generate.py --check")
    code2, out2 = ctx.py("-m mrgym.cli export harbor --check")
    code3, out3 = ctx.py("-m mrgym.cli export openenv --check")
    code4, out4 = ctx.py("conformance/run.py --check-readme")
    stale = []
    for rel in ("README.md", "DATASHEET.md", "results/README.md", "integrations/README.md"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            text = re.sub(r"<!-- BEGIN GENERATED.*?<!-- END GENERATED[^>]*-->", "", open(p, encoding="utf-8").read(), flags=re.S)
            stale += [f"{rel}: '{m}'" for m in re.findall(r"No language model has been run|Not run in this tree|Not written\.|has not run yet", text)]
    gen = all(c == 0 for c in (code1, code2, code3, code4))
    if gen and not stale:
        return PASS, "generated files are current (cards, tables, Harbor tasks, OpenEnv manifest, integrations table); no known stale sentence"
    if gen:
        return PARTIAL, f"generated files are current; sentences outside generated blocks that state a stale fact: {stale[:4]}"
    return FAIL, "a generated file is stale: " + "; ".join(o.strip().splitlines()[-1][:120] for c, o in ((code1, out1), (code2, out2), (code3, out3), (code4, out4)) if c and o.strip())


def per_task_explanations(ctx):
    from mrgym.compat import export
    heads = ("## Difficulty", "## Reference solution", "## Verification")
    tasks = export.harbor_tasks(os.path.join(ROOT, "datasets"))
    task_ok = bool(tasks) and all(all(h in open(os.path.join(t, "README.md"), encoding="utf-8").read() for h in heads) for t in tasks)
    cards = glob.glob(os.path.join(ROOT, "envs", "*", "README.md"))
    card_ok = bool(cards) and all(all(h in open(c, encoding="utf-8").read() for h in heads) for c in cards)
    if task_ok and card_ok:
        return PASS, "Difficulty, Reference solution and Verification sections in every environment card and every exported task"
    if task_ok:
        return PARTIAL, f"the three sections are in the README of all {len(tasks)} exported Harbor task(s); the environment cards under envs/ do not have them"
    return FAIL, "the sections are in neither the cards nor the exported tasks"


def code_documented(ctx):
    import ast
    files = glob.glob(os.path.join(ROOT, "src", "mrgym", "**", "*.py"), recursive=True)
    bad = [os.path.relpath(p, ROOT) for p in files if os.path.getsize(p) > 200 and not ast.get_docstring(ast.parse(open(p, encoding="utf-8").read()))]
    return (PASS if not bad else PARTIAL), f"{len(files) - len(bad)} of {len(files)} modules have a module docstring" + (f"; not: {bad[:5]}" if bad else "")


def cross_stack(ctx):
    """G3: the stored fingerprints (they include an observation digest and a scanner-log head per level) reproduce on
    this Python and on the Gymnasium environment's Python."""
    import numpy
    import scipy
    code1, out1 = ctx.py("-m mrgym.compat.fingerprint --check")
    code2, out2 = ctx.sh(f"{ctx.tools['GYM']} python -m mrgym.compat.fingerprint --check")
    c3, v = ctx.sh(f"{ctx.tools['GYM']} python -c \"import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)\"")
    here = f"{sys.version.split()[0]} / numpy {numpy.__version__} / scipy {scipy.__version__}"
    there = v.strip().splitlines()[-1] if c3 == 0 and v.strip() else "?"
    two = there.split()[:1] != [sys.version.split()[0]] or there.split()[1:2] != [numpy.__version__]
    if code1 == 0 and code2 == 0 and two:
        return PASS, f"identical episode digests and log heads on Python {here} and on Python/numpy/scipy {there} (one host)"
    if code1 == 0 and code2 == 0:
        return PARTIAL, f"fingerprints reproduce, but both environments are the same stack ({here})"
    return FAIL, f"fingerprint check: this Python exit {code1}, second stack exit {code2}"


def dependencies_pinned(ctx):
    lock = os.path.exists(os.path.join(ROOT, "uv.lock"))
    code, out = ctx.sh("uv lock --check") if lock else (1, "no uv.lock")
    fw = ctx.framework("harbor")
    images = fw["static_ok"]
    if lock and code == 0 and images:
        return PASS, "uv.lock is current (`uv lock --check`); exported images pin the base image by digest and mrgym by exact version"
    if lock and code == 0 or images:
        return PARTIAL, f"uv.lock current: {lock and code == 0}; exported images pinned (digest + exact version): {images}"
    return FAIL, "no current lock file and no pinned images"


def results_name_versions(ctx):
    p = os.path.join(ROOT, "results", "results.json")
    if not os.path.exists(p):
        return FAIL, "no results/results.json"
    r = json.load(open(p, encoding="utf-8"))
    ok = bool(r.get("engines")) and bool(r.get("package_versions")) and bool(r.get("environments"))
    return (PASS if ok else FAIL), f"results.json names engines {r.get('engines')}, package versions {r.get('package_versions')}, ids {sorted(r.get('environments', {}))}"


def episode_evidence(ctx):
    """G10: every model episode behind a number has its conversation and scanner log stored."""
    base = os.path.join(ROOT, "results", "episodes")
    records = glob.glob(os.path.join(base, "*", "*", "L*", "s*", "e*", "record.json"))
    if not records:
        return FAIL, "no stored episode records"
    model, full = 0, 0
    for path in records:
        try:
            rec = json.load(open(path, encoding="utf-8"))
        except ValueError:
            continue
        if rec.get("agent_kind") != "model":
            continue
        model += 1
        others = [n for n in os.listdir(os.path.dirname(path)) if n != "record.json"]
        full += bool(others or rec.get("transcript_sha256"))
    if model and full == model:
        return PASS, f"{model} model episodes, each with a stored conversation"
    return (PARTIAL if records else FAIL), f"{len(records)} episode records; {model} model episodes of which {full} have a stored conversation and scanner log"


def wheel_installs(ctx):
    d = os.path.join(ctx.tmp, "wheel")
    code, out = ctx.sh(f"uv build --wheel -o {shlex.quote(d)} . && uv venv -q -p {shlex.quote(sys.executable)} {shlex.quote(d)}/v && "
                       f"uv pip install -q --python {shlex.quote(d)}/v/bin/python {shlex.quote(d)}/*.whl && cd {shlex.quote(d)} && "
                       f"v/bin/python -c \"import mrgym, json, importlib.metadata as m; "
                       f"print(json.dumps([mrgym.list_envs(), sorted(r.split()[0] for r in (m.requires('mrgym') or []) if 'extra' not in r)]))\" && "
                       f"v/bin/mrgym export harbor --out {shlex.quote(d)}/ds --id {shlex.quote(ctx.scored[0])} && v/bin/python -m mrgym.compat.harbor local {shlex.quote(d)}/ds/*/tasks/*",
                       env={k: v for k, v in ctx.env.items() if k != "PYTHONPATH"})
    return (PASS if code == 0 else FAIL), ("the wheel builds, installs into a fresh environment with its declared dependencies only, lists the ids, "
                                           "exports a Harbor task and plays it" if code == 0 else out.strip()[-300:])


def openenv_package(ctx):
    fw = ctx.framework("openenv")
    if not fw["ran"]:
        return FAIL, fw["summary"]
    if fw["static_ok"] and fw["demo_ok"] and fw["runtime_ok"]:
        return PASS, fw["summary"]
    if fw["static_ok"] and fw["demo_ok"]:
        return PARTIAL, fw["summary"]
    return FAIL, fw["summary"]


def verifiers_package(ctx):
    fw = ctx.framework("verifiers")
    return (PASS if fw["ok"] and fw["parity_ok"] else FAIL), fw["summary"] + f"; in-process toolset parity: {fw['parity_ok']}"


def inspect_task(ctx):
    fw = ctx.framework("inspect")
    return (PASS if fw["ok"] else FAIL), fw["summary"]


def harbor_export(ctx):
    fw = ctx.framework("harbor")
    if fw["docker_ok"]:
        return PASS, fw["summary"]
    if fw["static_ok"] and fw["local_ok"]:
        return PARTIAL, fw["summary"]
    return FAIL, fw["summary"]


def container_images(ctx):
    from mrgym.compat import export, harbor
    tasks = export.harbor_tasks(os.path.join(ROOT, "datasets"))
    declared = bool(tasks) and all(not [f for f in harbor.lint(t) if f[0] in ("resources", "timeouts", "unpinned-image", "unpinned-python-dependency")] for t in tasks)
    openenv = os.path.exists(os.path.join(ROOT, "integrations", "openenv", "mrgym_scanner_env", "server", "Dockerfile"))
    if ctx.docker:
        built = all(ctx.sh(f"docker build -q {shlex.quote(os.path.join(t, sub))}", timeout=3600)[0] == 0
                    for t in tasks for sub in ("environment", "environment/scanner", "tests"))
        return (PASS if built and declared else FAIL), f"docker build of the three images of {len(tasks)} task(s): {built}"
    if declared and openenv:
        return PARTIAL, (f"three Dockerfiles per exported task ({len(tasks)} task(s)) and one for the OpenEnv server, base image pinned by digest, "
                         "resources and timeouts declared; no image was built (no Docker daemon here, and mrgym is not on a package index yet)")
    return FAIL, "no Dockerfiles with declared resources"


def trainer_integration(ctx):
    train = glob.glob(os.path.join(ROOT, "examples", "train", "*"))
    fw = ctx.framework("verifiers")
    if train and fw["ok"]:
        return PASS, f"examples/train/ ships {len(train)} file(s) and the verifiers taskset validates"
    if fw["ok"]:
        return PARTIAL, "the verifiers taskset (the form prime-rl trains on) validates; no training config or smoke run ships under examples/train/"
    return FAIL, "no trainer integration"


def concurrent_sessions(ctx):
    """H12: two episodes interleaved in one process are the same two episodes played one after the other."""
    import mrgym
    from mrgym import compat
    i = ctx.scored[0]
    eps = [compat.scripted_episode(i, None, s) for s in (0, 1)]
    envs = [mrgym.make(i, seed=s) for s in (0, 1)]
    texts = [[e.reset()[0].text] for e in envs]
    rewards = [0.0, 0.0]
    for k in range(max(len(e["calls"]) for e in eps)):
        for j in (0, 1):
            if k < len(eps[j]["calls"]):
                r = envs[j].step(eps[j]["calls"][k])
                texts[j].append(r.observation.text)
                rewards[j] = r.reward
    ok = all(compat.text_digest(texts[j]) == eps[j]["text_sha256"] and rewards[j] == eps[j]["reward"] and compat.log_head(envs[j]) == eps[j]["log_head"]
             for j in (0, 1))
    fw = ctx.framework("openenv")
    return (PASS if ok else FAIL), f"two interleaved episodes of {i} equal their sequential runs: {ok}; OpenEnv sessions over one server: {fw.get('demo_ok')}"


def fast_enough(ctx, limit=5.0):
    oks = {}
    for i in ctx.scored:
        cpu = statistics.median(e["cpu_s"] for e in ctx.episodes(i))
        oks[i] = (cpu <= limit, f"{cpu:.1f} CPU s per reference episode")
    return agg(oks, f"scored ids whose reference episode takes <= {limit:g} CPU s on one thread")


def _results(ctx):
    p = os.path.join(ROOT, "results", "results.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def controls_reported(ctx):
    """I1-I3: the stored control results of every id with results hold a no-scan policy, >= 3 must-fail policies and
    the reference, and the cards print them."""
    oks = {}
    for i in ctx.scored:
        g = ctx.gate(i)
        if g:
            lv = next(iter(g["levels"].values()))
            names = {r["policy"].split("#")[0] for r in lv.get("must_fail", [])}
            oks[i] = ("prior" in names and len(names) >= 3 and "reference" in lv, f"must-fail policies stored: {sorted(names)}")
        else:
            oks[i] = (False, "no stored control results")
    return agg(oks, "scored ids with stored floor, shortcut and reference rows")


def model_baselines(ctx):
    r = _results(ctx)
    models = (r.get("agents") or {}).get("model") or []
    readme = os.path.join(ROOT, "results", "README.md")
    wilson = os.path.exists(readme) and "Wilson" in open(readme, encoding="utf-8").read()
    stale_ids = [i for i in r.get("environments", {}) if i not in ctx.ids]
    if models and wilson and not stale_ids:
        return PASS, f"{len(models)} model rows with counts and Wilson intervals, on ids of the current registry"
    if models and wilson:
        return PARTIAL, f"{len(models)} model rows with Wilson intervals, but on ids that are no longer registered: {stale_ids}"
    return FAIL, "no model rows with intervals in results/"


def open_model_rates(ctx):
    r = _results(ctx)
    models = [str(m) for m in (r.get("agents") or {}).get("model") or []]
    open_models = [m for m in models if re.search(r"qwen|llama|mistral|gemma|phi|deepseek", m, re.I)]
    if len(open_models) >= 3:
        return PASS, f"open-model rows in results/: {open_models}"
    return (PARTIAL if models else FAIL), f"model rows in results/: {models}; open models one would train: {open_models} (need 3)"


def recently_checked(ctx):
    return PARTIAL, "this command ran the tests and every validator on the date of the report; there is no record of a CI run on the host"


def registry(name):
    return globals()[name]
