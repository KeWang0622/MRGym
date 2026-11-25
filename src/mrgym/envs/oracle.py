"""Oracles: the true optimum of an instance, found with the hidden world known, by code that shares nothing with the
policy it certifies.

Why. A reference policy that passes 300 of 300 instances shows that the task is solvable by that policy. It does not
show what the best possible answer is, and when the verifier's optimum and the reference are one formula the pass
rate is circular. An oracle is the other side of the comparison: it reads the hidden world, searches the deliverable
exhaustively on the noise-free engine, and returns the optimum together with the neighbours that fail.

Contract. A scored family may ship ``<family>/oracle.py`` with

    def solve(cfg: dict, world: dict | None = None) -> dict:
        {"value": float,          # T* in seconds (cost tasks) or E*, the smallest attainable error (estimation tasks)
         "argmin": dict,          # a deliverable that attains it; ``task.score`` passes it
         "method": "exhaustive" | "bound" | "exact",       # "exact": read from the hidden world, nothing searched
         "grid": {"axes": {...}, "size": int, "evaluated": int},
         "certificate": [{"deliverable": dict, "fails": "<check>"}, ...],   # nearest answers that fail, and why
         "seconds": float}

``oracle.py`` may import ``task`` and the scanner. It must not import ``controls`` (the policies) and neither it nor
``task.py`` may mention the reference; ``independence_problems`` checks both by reading the syntax tree, and a test
runs it on every family.

Helpers here are generic: ``exhaustive_first`` (cost tasks: candidates in ascending cost, the first that passes),
``exhaustive_min`` (estimation tasks: the smallest error on a grid), ``refine_root`` (a sign change located to the
console raster), ``regret`` and ``regret_report`` (how far a policy or a model is from the optimum), and
``policy_regret`` (plays a scripted policy and compares it with the oracle of the same instance).
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import time

import numpy as np

from ..scanner import kspace, tissue
from ..scanner import signal as S

MAX_POINTS = 80                     # points of a region on which ``RegionPoints`` evaluates the engine
FORBIDDEN_IMPORTS = ("controls",)
FORBIDDEN_NAMES = ("reference", "_geometry", "tolerance_from", "REFERENCE")


def load(env_or_spec):
    """The oracle module of an environment (``<family>.oracle``), or None when the family ships none."""
    spec = getattr(env_or_spec, "spec", env_or_spec)
    if isinstance(spec, str):
        import mrgym
        spec = mrgym.spec(spec)
    if not spec.family or importlib.util.find_spec(spec.family + ".oracle") is None:
        return None
    return importlib.import_module(spec.family + ".oracle")


# ------------------------------------------------------------------ search on the noise-free engine
def exhaustive_first(candidates, fails, keep: int = 6) -> dict:
    """Cost-minimisation: ``candidates`` is an iterable of ``(deliverable, cost)`` in ascending cost; ``fails(d)``
    returns None when ``d`` passes every check on the noise-free engine, else the name of the first failed check.

    Returns ``{value, argmin, method, grid, certificate}``: the first candidate that passes, and the last ``keep``
    cheaper candidates with the check each fails. ``value`` is None when nothing passes."""
    seen, last_cost = [], -np.inf
    for size, (deliverable, cost) in enumerate(candidates, 1):
        if cost < last_cost:
            raise ValueError("candidates must come in ascending cost")
        last_cost = cost
        why = fails(deliverable)
        if why is None:
            return dict(value=float(cost), argmin=deliverable, method="exhaustive", grid=dict(size=size, evaluated=size),
                        certificate=[dict(deliverable=d, cost=float(c), fails=w) for d, c, w in seen[-keep:]])
        seen.append((deliverable, cost, why))
    return dict(value=None, argmin=None, method="exhaustive", grid=dict(size=len(seen), evaluated=len(seen)),
                certificate=[dict(deliverable=d, cost=float(c), fails=w) for d, c, w in seen[-keep:]])


def exhaustive_min(grid, error) -> dict:
    """Estimation: evaluate ``error(x)`` on every point of ``grid`` and return ``{value, argmin, index, errors,
    evaluated}`` of the smallest."""
    errors = [float(error(x)) for x in grid]
    i = int(np.argmin(errors))
    return dict(value=errors[i], argmin=grid[i], index=i, errors=errors, evaluated=len(errors))


def sign_changes(values) -> list[int]:
    """Indices i where ``values[i]`` and ``values[i + 1]`` have opposite signs (a zero counts as positive)."""
    s = np.where(np.asarray(values, float) >= 0, 1, -1)
    return [int(i) for i in np.flatnonzero(s[:-1] != s[1:])]


def refine_root(f, lo: float, hi: float, tol: float, max_evaluations: int = 60) -> tuple[float, int]:
    """Root of ``f`` in [lo, hi] (``f(lo)`` and ``f(hi)`` of opposite sign) to ``tol``, by the Illinois variant of
    regula falsi: bracketing like bisection, a handful of evaluations on a smooth function. Returns (root,
    evaluations)."""
    fa, fb = f(lo), f(hi)
    n = 2
    if fa == 0:
        return lo, n
    if fb == 0:
        return hi, n
    if (fa > 0) == (fb > 0):
        raise ValueError("the interval does not bracket a sign change")
    side = 0
    while hi - lo > tol and n < max_evaluations:
        x = (lo * fb - hi * fa) / (fb - fa)
        if not lo < x < hi:
            x = 0.5 * (lo + hi)
        fx = f(x)
        n += 1
        if fx == 0:
            return x, n
        if (fx > 0) == (fb > 0):
            hi, fb = x, fx
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            lo, fa = x, fx
            if side == 1:
                fb *= 0.5
            side = 1
    return 0.5 * (lo + hi), n


# ------------------------------------------------------------------ one region of the hidden world
class RegionPoints:
    """Amplitude-only model of one region: the engine's steady state on the region's points, no image formation.
    Verifier side (it reads the hidden world); shared by the oracles of the families."""

    def __init__(self, world: dict, region: dict):
        cls = region["cls"]
        if "fields" in world:
            n, d = world["n"], world["extent_mm"] / world["n"]
            ax = (np.arange(n) - n // 2) * d
            iy = np.flatnonzero(np.abs(ax - region["y_mm"]) <= region["radius_mm"])
            ix = np.flatnonzero(np.abs(ax - region["x_mm"]) <= region["radius_mm"])
            step = max(1, int(np.ceil(np.sqrt(len(iy) * len(ix) * np.pi / 4 / MAX_POINTS))))
            yy, xx = np.meshgrid(iy[step // 2::step], ix[step // 2::step], indexing="ij")
            inside = (ax[yy] - region["y_mm"]) ** 2 + (ax[xx] - region["x_mm"]) ** 2 <= region["radius_mm"] ** 2
            yy, xx = yy[inside], xx[inside]
            weights = world["weights"][world["names"].index(cls)][yy, xx].astype(float)
            fields = {k: world["fields"][k][yy, xx] for k in ("b0_hz", "b1")}
            rx = world["fields"]["rx"][yy, xx]
        else:                                                           # material values only
            weights, fields, rx = np.ones(1), dict(b0_hz=np.zeros(1), b1=np.ones(1)), np.ones(1, complex)
        points = dict(n=len(weights), extent_mm=float(len(weights)), b0_t=world["b0_t"], names=[cls], tissues={cls: world["tissues"][cls]},
                      weights=weights[None, :], fields=fields, system=world["system"])
        self.engine = kspace.Engine(points)
        self.rx = rx[self.engine.sup[0]] / len(weights)
        self.cls, self.b0_t = world["tissues"][cls], world["b0_t"]

    def amplitude(self, realised: dict) -> float:
        """|mean echo amplitude of the region| for one realised protocol, without the echo-time decay."""
        return float(abs((self.engine.amplitudes(realised, 0)[0] * self.rx).sum()))

    def decay(self, realised: dict) -> float:
        return float(S.time_function(self.cls, np.zeros(1), np.zeros(1), realised["te_ms"], "se", self.b0_t, tissue.FAT_PEAKS)[0].real)

    def signal(self, realised: dict) -> float:
        """|mean signal of the region| of one realised protocol, no image formation."""
        return self.amplitude(realised) * self.decay(realised) * realised["slice_mm"]

    def noise_sd(self, realised: dict) -> float:
        """SD of the noise in one pixel of a realised protocol (``Engine.noise_sd`` on the hidden system)."""
        return self.engine.noise_sd(realised)


# ------------------------------------------------------------------ regret
def regret(value: float | None, optimum: float | None, kind: str = "ratio") -> float | None:
    """How far a result is from the oracle's optimum: ``value / optimum - 1`` for a cost (kind "ratio": scan time
    over T*), ``value - optimum`` for an error whose optimum may be zero (kind "difference"). None when either is
    missing. A negative regret means the result beats the "optimum": the oracle or the verifier is wrong."""
    if value is None or optimum is None:
        return None
    if kind == "ratio":
        return None if optimum == 0 else float(value) / float(optimum) - 1.0
    if kind == "difference":
        return float(value) - float(optimum)
    raise ValueError("kind is ratio or difference")


def regret_report(regrets, passed=None) -> dict:
    """Summary of the regrets of one policy or model over instances: ``{n, missing, median, q95, max, negative,
    negative_passing}``. ``negative_passing`` counts passing episodes that beat the optimum: it must be 0."""
    values = [r for r in regrets if r is not None]
    out = dict(n=len(values), missing=len(regrets) - len(values), median=None, q95=None, max=None, negative=0, negative_passing=0)
    if values:
        v = np.asarray(values, float)
        out.update(median=float(np.median(v)), q95=float(np.quantile(v, 0.95)), max=float(v.max()), negative=int((v < -1e-9).sum()))
    if passed is not None:
        out["negative_passing"] = sum(1 for r, p in zip(regrets, passed) if p and r is not None and r < -1e-9)
    return out


def policy_regret(env_id: str, level: int, seeds, policy: str | None = None) -> dict:
    """Play a scripted policy (default: the family's must-pass policy) on ``seeds`` and compare each episode with the
    oracle of the same instance. Returns ``{env_id, level, policy, instances, passed, oracle_passes, kind, regret
    {n, median, q95, max, ...}, rows[...]}``. The policy is run through the tools, as in the gate."""
    import mrgym
    from mrgym import policies
    spec = mrgym.spec(env_id)
    module = load(spec)
    if module is None:
        raise mrgym.Error(f"{env_id} ships no oracle (no {spec.family or '<family>'}.oracle module)")
    name = policy or policies.controls_of(spec).REFERENCE
    task = importlib.import_module(spec.family + ".task")
    rows = []
    for seed in seeds:
        env = mrgym.make(env_id, level=level, seed=seed)
        result = policies.run(env, name)
        cfg, world = env.episode_config(), env.hidden_world()
        solution = module.solve(cfg, world)
        score = result.info.get("score") or {}
        check = task.score(cfg, solution["argmin"], world=world)
        rows.append(dict(seed=seed, passed=bool(score.get("passed")), error=score.get("error"), optimum=solution["value"],
                         regret=regret(score.get("error"), solution["value"], module.REGRET_KIND),
                         oracle_passes=bool(check["passed"]), oracle_seconds=solution["seconds"]))
        env.close()
    return dict(env_id=env_id, level=level, policy=name, instances=len(rows), passed=sum(r["passed"] for r in rows),
                oracle_passes=sum(r["oracle_passes"] for r in rows), kind=module.REGRET_KIND,
                regret=regret_report([r["regret"] for r in rows], [r["passed"] for r in rows]), rows=rows)


def regret_text(rep: dict) -> str:
    r = rep["regret"]
    unit = "of the optimum (value / optimum - 1)" if rep["kind"] == "ratio" else "above the optimum (value - optimum)"

    def fmt(x):
        return "-" if x is None else f"{100 * x:.2f} %"

    return (f"{rep['env_id']} level {rep['level']}, policy {rep['policy']}, {rep['instances']} instances: passes {rep['passed']}, "
            f"the oracle's answer passes {rep['oracle_passes']}\n"
            f"regret {unit}: median {fmt(r['median'])}, 95th percentile {fmt(r['q95'])}, maximum {fmt(r['max'])}; "
            f"passing episodes that beat the optimum: {r['negative_passing']}")


# ------------------------------------------------------------------ independence, checked on the syntax tree
def _tree(path: str) -> ast.AST:
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def imported_modules(path: str) -> set[str]:
    """Last components of every module a file imports, and every name it imports from a module."""
    out = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[-1] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[-1])
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in ("import_module", "__import__"):
            out.update(str(a.value).split(".")[-1] for a in node.args if isinstance(a, ast.Constant))
    return out


def referenced_names(path: str) -> set[str]:
    """Every identifier and attribute name a file uses in code (docstrings and comments are not code)."""
    out = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
    return out


def independence_problems(family: str) -> list[str]:
    """Why the oracle of ``family`` (a package name) is not independent of its reference policy; empty when it is.
    ``oracle.py`` must not import ``controls`` or use a name of the reference in code, and no function of ``task.py``
    other than ``card`` (which lists the policies for the reader) may do so either."""
    directory = os.path.dirname(importlib.util.find_spec(family).origin)
    path = os.path.join(directory, "oracle.py")
    if not os.path.exists(path):
        return [f"{family}: no oracle.py"]
    out = [f"{family}/oracle.py imports {name}" for name in sorted(imported_modules(path) & set(FORBIDDEN_IMPORTS))]
    out += [f"{family}/oracle.py uses the name {name}" for name in sorted(referenced_names(path) & set(FORBIDDEN_NAMES))]
    task = os.path.join(directory, "task.py")
    for node in _tree(task).body if os.path.exists(task) else []:        # the card may list the policies; scoring may not
        if isinstance(node, ast.FunctionDef) and node.name != "card":
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            used |= {a.name for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
            out += [f"{family}/task.py: {node.name}() uses {name}" for name in sorted(used & (set(FORBIDDEN_NAMES) | set(FORBIDDEN_IMPORTS)))]
    return out


def timed(fn):
    """Decorator for ``solve``: adds ``seconds`` to the returned dict."""
    def wrapper(*args, **kw):
        t0 = time.time()
        out = fn(*args, **kw)
        out["seconds"] = round(time.time() - t0, 3)
        return out
    wrapper.__doc__, wrapper.__name__ = fn.__doc__, fn.__name__
    return wrapper
