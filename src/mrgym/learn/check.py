"""``mrgym check <id> --learnability``: does the reward lead from a weak policy to the intended procedure? CPU only.

No model is involved. A family contributes a small parametric family of scripted policies around its reference
(``<family>.controls.policy_space()``); a distribution over those parameters is trained on the real environment, with
the group-normalised policy gradient a trainer uses (groups of 8 episodes on one instance, advantage = (signal - group
mean) / group SD, groups without spread dropped and counted), and, independently of the gradient estimator, with the
cross-entropy method. Two training signals are compared:

    paid        the environment's reward: 1 for a pass, 0 otherwise
    progress    ``info["score"]["progress"]``: the class-ordered diagnostic (never paid by the environment)

What is run (``run``)
    A  progress, start ``weak``        must pass T1, T2, T3: the diagnostic has a slope where the paid reward has none
    B  paid,     start ``in_band``     must pass T1, T2, T3: from an entry-rung start the paid reward alone is enough
    C  paid,     start ``weak``        printed. It is expected to fail T1: this is how the check shows that it can fail
    D  cross-entropy method from ``weak``, on progress and on the paid reward     printed with T4, for every seed. It
       does not enter the verdict: with a population of 32 the method can lose a decision that the start makes rare
       before any member of the population shows it, whatever the reward (the toy family of the tests shows it)
    E  the best answer that ignores the object, mean progress on fresh ``gate:`` instances   must stay <= BLIND_PROGRESS_MAX

Thresholds (``THRESHOLDS``; with ``n`` training steps)
    T1 no plateau     over the first 10 steps at most a quarter of the groups have no spread on average, and there is no
                      run of more than 3 consecutive steps in which every group is dead while the policy still fails
                      (a step in which at least 95 % of the training episodes pass is mastery, not a plateau). The run
                      over every all-dead step, converged ones included, is stored beside it
                      (``longest_all_dead_run_literal``, ``ok_literal``): a policy family that converges to a single
                      passing corner fails that reading by succeeding
    T2 climbs         pass rate on held-out instances >= 0.50 at 60 % of the steps and >= 0.95 at the end. The value
                      at the end is the exact expectation of the final distribution over the grid of policies
                      (``evaluate_exact``; a lower bound, short by at most 0.5 points); the values on the way are
                      sampled (``eval_k`` draws per instance). The sampled value at the end is stored beside the exact
                      one (``pass_end_sampled``, ``ok_sampled``): on a few dozen episodes it cannot resolve 0.95
    T3 the procedure  under the final distribution every decision the family lists as its procedure has probability
                      >= 0.95, and every shortcut it lists has probability <= 0.02
    T4 CEM            the cross-entropy method reaches a held-out pass rate >= 0.90 (exact, as for T2); reported
The pass rates are those of the policy distribution on instances that training never sees (seeds ``learn:eval:<i>``).

The contract of ``policy_space()`` (a dict):
    parameters   [{name, kind: "choice" | "bool" | "float", values (choice) | low, high, step (float), what}]
    build        callable(params: {name: value}) -> policy(scanner) -> answer, playing through the tools only
    reference    the parameter values of the family's reference policy
    starts       {"weak": {...}, "in_band": {...}}: per parameter a list of probabilities (choice), P(True) (bool) or
                 (mean, standard deviation) (float)
    procedure    {name: predicate(params) -> bool}: the decisions the task is about
    shortcuts    {name: predicate(params) -> bool}: what must not survive training
A float parameter is played on the grid ``low, low + step, ..., high``; the policy over it is a Gaussian whose samples
are snapped to that grid. Episodes are deterministic given (instance, parameter values), so each distinct pair is played
once and memoised: every arm, seed and optimiser of one check shares the cache. Each one is a real episode through
``mrgym.make`` and the tools; there is no fast path around the verifier.

    mrgym check Scanner/NullTissue-v0 --learnability --level 2 [--quick] [--write]
"""
from __future__ import annotations

import datetime
import importlib
import itertools
import math
import multiprocessing
import time

import numpy as np

SCHEMA = "mrgym-learnability/1"
BLIND_PROGRESS_MAX = 0.35
THRESHOLDS = dict(dead_share_max=0.25, dead_run_max=3, first_steps=10, mid_fraction=0.6, pass_mid_min=0.50, pass_end_min=0.95,
                  procedure_min=0.95, shortcut_max=0.02, cem_pass_min=0.90, blind_progress_max=BLIND_PROGRESS_MAX)
SIGNALS = ("paid", "progress")
SPREAD_MIN = 1e-3
LOG_SIGMA_MIN = math.log(0.03)
CEM_PSEUDO = 0.05                       # pseudo-count per value in the cross-entropy refit: keeps every value possible. A larger
                                        # one caps the mass the final distribution can put on the elite's choice (0.25 capped the
                                        # exact pass rate of a three-way choice near 0.9 whatever the reward)
TOKEN_SCAN = {"sequence": "spgr", "matrix": 48, "tr_ms": 12.0, "flip_deg": 8.0}
DEFAULTS = dict(steps=120, groups=8, k=8, lr=0.1, train_instances=8, eval_instances=8, eval_k=6, eval_every=20, seeds=3,
                cem_iterations=6, cem_population=32, cem_elite=8, cem_instances=3, n_blind=60)
QUICK = dict(DEFAULTS, steps=80, lr=0.15, seeds=1, train_instances=6, eval_instances=6, cem_iterations=5, n_blind=12)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ------------------------------------------------------------------------------------------------ the policy space
class Space:
    """A factorised distribution over the parameters of ``policy_space()``: logits for a choice, one logit for a
    bool, (mean, log standard deviation) in units of the range for a float."""

    def __init__(self, spec: dict):
        self.spec, self.params, self.grids, self.slices, n = spec, spec["parameters"], [], [], 0
        for p in self.params:
            if p["kind"] == "choice":
                grid, width = list(p["values"]), len(p["values"])
            elif p["kind"] == "bool":
                grid, width = [False, True], 1
            elif p["kind"] == "float":
                steps = int(round((p["high"] - p["low"]) / p.get("step", (p["high"] - p["low"]) / 8.0)))
                grid, width = [round(p["low"] + j * (p["high"] - p["low"]) / steps, 9) for j in range(steps + 1)], 2
            else:
                raise ValueError(f"parameter {p['name']}: kind must be choice, bool or float")
            self.grids.append(grid)
            self.slices.append(slice(n, n + width))
            n += width
        self.n_theta = n
        self.names = [p["name"] for p in self.params]
        if int(np.prod([len(g) for g in self.grids])) > 50_000:
            raise ValueError("the policy space has more than 50,000 grid points; coarsen it")

    def values(self, idx) -> dict:
        return {n: g[i] for n, g, i in zip(self.names, self.grids, idx)}

    def index(self, params: dict) -> tuple:
        return tuple(min(range(len(g)), key=lambda j: abs(float(g[j]) - float(params[n]))) if p["kind"] == "float" else g.index(params[n])
                     for p, n, g in zip(self.params, self.names, self.grids))

    def theta(self, start: dict) -> np.ndarray:
        th = np.zeros(self.n_theta)
        for p, sl in zip(self.params, self.slices):
            v = start[p["name"]]
            if p["kind"] == "choice":
                th[sl] = np.log(np.maximum(np.asarray(v, float), 1e-4))
            elif p["kind"] == "bool":
                th[sl] = math.log(max(v, 1e-4) / max(1.0 - v, 1e-4))
            else:
                span = p["high"] - p["low"]
                th[sl] = [(v[0] - p["low"]) / span, math.log(max(v[1] / span, math.exp(LOG_SIGMA_MIN)))]
        return th

    def marginals(self, th) -> list:
        """Probability of every grid value of every parameter."""
        out = []
        for p, sl, g in zip(self.params, self.slices, self.grids):
            if p["kind"] == "choice":
                out.append(_softmax(th[sl]))
            elif p["kind"] == "bool":
                q = _sigmoid(th[sl][0])
                out.append(np.array([1.0 - q, q]))
            else:
                mu, sd, n = th[sl][0], math.exp(th[sl][1]), len(g) - 1
                edges = [-math.inf] + [(j + 0.5) / n for j in range(n)] + [math.inf]
                out.append(np.array([_phi((edges[j + 1] - mu) / sd) - _phi((edges[j] - mu) / sd) for j in range(n + 1)]))
        return out

    def sample(self, th, rng) -> tuple:
        """(grid index per parameter, gradient of the log-probability of the draw with respect to theta)."""
        idx, grad = [], np.zeros(self.n_theta)
        for p, sl, g in zip(self.params, self.slices, self.grids):
            if p["kind"] == "choice":
                pr = _softmax(th[sl])
                i = int(rng.choice(len(pr), p=pr))
                grad[sl] = -pr
                grad[sl.start + i] += 1.0
            elif p["kind"] == "bool":
                q = _sigmoid(th[sl][0])
                i = int(rng.random() < q)
                grad[sl] = (1.0 - q) if i else -q
            else:
                mu, sd, z = th[sl][0], math.exp(th[sl][1]), float(rng.standard_normal())
                i = int(round(min(max(mu + sd * z, 0.0), 1.0) * (len(g) - 1)))
                grad[sl] = [z / sd, z * z - 1.0]
            idx.append(i)
        return tuple(idx), grad

    def clip(self, th) -> np.ndarray:
        for p, sl in zip(self.params, self.slices):
            if p["kind"] == "float":
                th[sl.start] = min(max(th[sl.start], -0.25), 1.25)
                th[sl.start + 1] = max(th[sl.start + 1], LOG_SIGMA_MIN)
        return th

    def prob(self, th, predicate) -> float:
        """Probability, under theta, that a draw satisfies ``predicate(params)`` (exact: the grid is enumerated)."""
        m, total = self.marginals(th), 0.0
        for idx in itertools.product(*[range(len(g)) for g in self.grids]):
            if predicate(self.values(idx)):
                total += float(np.prod([m[k][i] for k, i in enumerate(idx)]))
        return total

    def describe(self, th) -> dict:
        return {n: {str(v): round(float(q), 4) for v, q in zip(g, m)} for n, g, m in zip(self.names, self.grids, self.marginals(th))}

    def fit(self, th, elite, smoothing: float) -> np.ndarray:
        """Cross-entropy update: move the distribution towards the elite draws (probabilities are mixed with weight
        ``smoothing``; a pseudo-count keeps every value possible)."""
        new, m = np.array(th, float), self.marginals(th)
        for k, (p, sl, g) in enumerate(zip(self.params, self.slices, self.grids)):
            picks = np.array([e[k] for e in elite])
            if p["kind"] == "float":
                u = picks / (len(g) - 1)
                mu = smoothing * float(u.mean()) + (1.0 - smoothing) * th[sl.start]
                sd = smoothing * max(float(u.std()), 0.5 / (len(g) - 1)) + (1.0 - smoothing) * math.exp(th[sl.start + 1])
                new[sl] = [mu, math.log(sd)]
                continue
            freq = (np.bincount(picks, minlength=len(g)) + CEM_PSEUDO) / (len(picks) + CEM_PSEUDO * len(g))
            q = smoothing * freq + (1.0 - smoothing) * m[k]
            new[sl] = math.log(q[1] / q[0]) if p["kind"] == "bool" else np.log(q)
        return self.clip(new)


# ------------------------------------------------------------------------------------------------ episodes
def _play_env(args) -> dict:
    """One real episode: the policy ``build(params)`` on one instance, through the tools. Runs in a worker."""
    import mrgym

    from .. import policies
    env_id, level, seed, params = args
    spec = mrgym.spec(env_id)
    build = importlib.import_module(spec.family + ".controls").policy_space()["build"]
    env = mrgym.make(env_id, level=level, seed=seed)
    r = policies.run(env, build(params))
    env.close()
    s = r.info.get("score") or {}
    return dict(passed=bool(s.get("passed", False)), progress=float(s.get("progress", 0.0)),
                outcome=s.get("outcome") or r.info.get("outcome") or "invalid_deliverable", policy_error=r.info.get("policy_error"))


class Oracle:
    """Memoised episodes: ``(seed, grid index) -> {passed, progress, outcome}``. ``play`` maps one key to that dict; by
    default it plays the environment, in at most two worker processes."""

    def __init__(self, space: Space, env_id: str | None = None, level: int | None = None, play=None, workers: int = 1):
        self.space, self.env_id, self.level, self.play = space, env_id, level, play
        self.workers = max(1, min(int(workers), 2))
        self.cache, self.seconds, self._pool = {}, 0.0, None

    def resolve(self, keys) -> None:
        todo = sorted({k for k in keys if k not in self.cache})
        if not todo:
            return
        t0 = time.time()
        if self.play is not None:
            out = [self.play(k) for k in todo]
        else:
            jobs = [(self.env_id, self.level, seed, self.space.values(idx)) for seed, idx in todo]
            if self.workers > 1 and len(jobs) > 1:
                if self._pool is None:
                    self._pool = multiprocessing.get_context("fork").Pool(self.workers)
                out = self._pool.map(_play_env, jobs, chunksize=1)
            else:
                out = [_play_env(j) for j in jobs]
        self.cache.update(zip(todo, out))
        self.seconds += time.time() - t0

    def __getitem__(self, key) -> dict:
        return self.cache[key]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def signal_of(episode: dict, signal: str) -> float:
    return float(episode["passed"]) if signal == "paid" else float(episode["progress"])


def evaluate(space: Space, oracle: Oracle, th, seeds, k: int, rng) -> dict:
    """The sampled policy on held-out instances: ``k`` draws per instance."""
    keys = [(s, space.sample(th, rng)[0]) for s in seeds for _ in range(k)]
    oracle.resolve(keys)
    eps = [oracle[key] for key in keys]
    return dict(pass_rate=float(np.mean([e["passed"] for e in eps])), progress=float(np.mean([e["progress"] for e in eps])))


def evaluate_exact(space: Space, oracle: Oracle, th, seeds, coverage: float = 0.995) -> dict:
    """The expected pass rate of the distribution on held-out instances, without sampling: the most probable grid
    points that together hold ``coverage`` of the probability are each played on every instance; the rest counts as
    failing, so the value is a lower bound that is short by at most ``1 - coverage``."""
    m = space.marginals(th)
    points = sorted(((float(np.prod([m[k][i] for k, i in enumerate(idx)])), idx)
                     for idx in itertools.product(*[range(len(g)) for g in space.grids])), reverse=True)
    kept, mass = [], 0.0
    for q, idx in points:
        if mass >= coverage:
            break
        kept.append((q, idx))
        mass += q
    oracle.resolve([(s, idx) for _, idx in kept for s in seeds])
    rate = sum(q * float(np.mean([oracle[(s, idx)]["passed"] for s in seeds])) for q, idx in kept)
    return dict(pass_rate=float(rate), points=len(kept), mass=float(mass))


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999):
        self.m, self.v, self.t, self.lr, self.b1, self.b2 = np.zeros(n), np.zeros(n), 0, lr, b1, b2

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        return self.lr * (self.m / (1 - self.b1 ** self.t)) / (np.sqrt(self.v / (1 - self.b2 ** self.t)) + 1e-8)


def group_gradient(x, grads, groups: int, k: int) -> tuple:
    """Policy gradient with group-normalised advantages. ``x`` and ``grads`` hold ``groups * k`` episodes, group after
    group. Returns (gradient, number of groups without spread)."""
    x, adv, dead = np.asarray(x, float).reshape(groups, k), np.zeros((groups, k)), 0
    for g in range(groups):
        sd = x[g].std()
        if sd < SPREAD_MIN:
            dead += 1
            continue
        adv[g] = (x[g] - x[g].mean()) / (sd + 1e-6)
    return (adv.reshape(-1, 1) * np.asarray(grads)).mean(0), dead


def reinforce(space: Space, oracle: Oracle, start: str, signal: str, seed: int, train_seeds, eval_seeds, *, steps, groups, k, lr,
              eval_k, eval_every, say=None, **_) -> dict:
    """Train the distribution from ``start`` on ``signal``. Returns the curve and the final distribution."""
    rng, eval_rng = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
    th, opt, curve = space.theta(space.spec["starts"][start]), Adam(space.n_theta, lr), []
    for step in range(steps + 1):
        row = dict(step=step)
        if step % eval_every == 0 or step == steps:
            ev = evaluate(space, oracle, th, eval_seeds, eval_k, eval_rng)
            row.update(eval_pass=ev["pass_rate"], eval_progress=ev["progress"])
        if step < steps:
            inst = [train_seeds[i] for i in rng.choice(len(train_seeds), groups, replace=groups > len(train_seeds))]
            draws = [space.sample(th, rng) for _ in inst for _ in range(k)]
            keys = [(s, d[0]) for s, d in zip([s for s in inst for _ in range(k)], draws)]
            oracle.resolve(keys)
            eps = [oracle[key] for key in keys]
            g, dead = group_gradient([signal_of(e, signal) for e in eps], [d[1] for d in draws], groups, k)
            row.update(dead=dead, train_pass=float(np.mean([e["passed"] for e in eps])), train_progress=float(np.mean([e["progress"] for e in eps])))
            th = space.clip(th + opt.step(g))
        curve.append(row)
        if say and "eval_pass" in row:
            say(f"  [{signal:8s} from {start:8s} seed {seed}] step {step:3d}  held-out pass {row['eval_pass']:.3f}  progress {row['eval_progress']:.3f}"
                + (f"  dead groups {row['dead']}/{groups}" if "dead" in row else "") + f"  episodes played {len(oracle.cache)}")
    exact = evaluate_exact(space, oracle, th, eval_seeds)
    return dict(optimiser="policy_gradient", signal=signal, start=start, seed=seed, groups=groups, k=k, steps=steps, curve=curve,
                pass_end_exact=exact, theta=[round(float(v), 6) for v in th], final=space.describe(th), **_landing(space, th))


def cem(space: Space, oracle: Oracle, start: str, signal: str, seed: int, train_seeds, eval_seeds, *, cem_iterations, cem_population,
        cem_elite, cem_instances, eval_k, smoothing: float = 0.7, say=None, **_) -> dict:
    """The cross-entropy method on the same distribution: draw a population of parameter settings, play each on a few
    training instances, refit the distribution to the best ones. An iteration whose population has no spread in the
    signal changes nothing (and is counted)."""
    rng, eval_rng = np.random.default_rng([seed, 3]), np.random.default_rng([seed, 4])
    th, curve = space.theta(space.spec["starts"][start]), []
    for it in range(cem_iterations + 1):
        ev = evaluate(space, oracle, th, eval_seeds, eval_k, eval_rng)
        row = dict(iteration=it, eval_pass=ev["pass_rate"], eval_progress=ev["progress"])
        if it < cem_iterations:
            inst = [train_seeds[i] for i in rng.choice(len(train_seeds), min(cem_instances, len(train_seeds)), replace=False)]
            pop = [space.sample(th, rng)[0] for _ in range(cem_population)]
            oracle.resolve([(s, idx) for idx in pop for s in inst])
            fit = np.array([np.mean([signal_of(oracle[(s, idx)], signal) for s in inst]) for idx in pop])
            row.update(dead=bool(fit.std() < SPREAD_MIN), best=float(fit.max()), mean=float(fit.mean()))
            if not row["dead"]:
                order = np.argsort(-fit, kind="stable")[:cem_elite]
                th = space.fit(th, [pop[i] for i in order], smoothing)
        curve.append(row)
        if say:
            say(f"  [CEM {signal:8s} from {start:8s} seed {seed}] iteration {it}  held-out pass {row['eval_pass']:.3f}  episodes played {len(oracle.cache)}")
    exact = evaluate_exact(space, oracle, th, eval_seeds)
    return dict(optimiser="cem", signal=signal, start=start, seed=seed, population=cem_population, elite=cem_elite, iterations=cem_iterations,
                curve=curve, pass_end_exact=exact, theta=[round(float(v), 6) for v in th], final=space.describe(th), **_landing(space, th))


def _landing(space: Space, th) -> dict:
    return dict(procedure={n: round(space.prob(th, f), 4) for n, f in space.spec.get("procedure", {}).items()},
                shortcuts={n: round(space.prob(th, f), 4) for n, f in space.spec.get("shortcuts", {}).items()})


# ------------------------------------------------------------------------------------------------ thresholds
def judge(run: dict, t: dict = THRESHOLDS) -> dict:
    """T1 - T3 for a policy-gradient run, T4 for a cross-entropy run: ``{name: {ok, ...measured values}}``."""
    curve = run["curve"]
    sampled = curve[-1]["eval_pass"]
    end = run["pass_end_exact"]["pass_rate"] if run.get("pass_end_exact") else sampled
    if run["optimiser"] == "cem":
        return dict(T4=dict(ok=end >= t["cem_pass_min"], pass_end=round(end, 4), pass_end_sampled=round(sampled, 4),
                            ok_sampled=sampled >= t["cem_pass_min"]))
    dead = [r["dead"] for r in curve if "dead" in r]
    longest = cur = literal = cur_literal = 0
    for r in curve[:-1]:
        all_dead = r["dead"] == run["groups"]
        cur_literal = cur_literal + 1 if all_dead else 0                 # every all-dead step, converged ones included
        literal = max(literal, cur_literal)
        cur = cur + 1 if all_dead and r["train_pass"] < t["pass_end_min"] else 0   # a step whose groups all pass is mastery, not a plateau
        longest = max(longest, cur)
    share = float(np.mean(dead[:t["first_steps"]])) / run["groups"]
    evals = [(r["step"], r["eval_pass"]) for r in curve if "eval_pass" in r]
    mid = next(p for s, p in evals if s >= t["mid_fraction"] * run["steps"])
    proc, short = run["procedure"], run["shortcuts"]
    return dict(
        T1=dict(ok=share <= t["dead_share_max"] and longest <= t["dead_run_max"], dead_share_first_steps=round(share, 4),
                dead_groups_first_steps=round(share * run["groups"], 2), longest_all_dead_run=longest,
                longest_all_dead_run_literal=literal,
                ok_literal=share <= t["dead_share_max"] and literal <= t["dead_run_max"]),
        T2=dict(ok=mid >= t["pass_mid_min"] and end >= t["pass_end_min"], pass_mid=round(mid, 4), pass_end=round(end, 4),
                pass_end_sampled=round(sampled, 4), ok_sampled=mid >= t["pass_mid_min"] and sampled >= t["pass_end_min"]),
        T3=dict(ok=all(v >= t["procedure_min"] for v in proc.values()) and all(v <= t["shortcut_max"] for v in short.values()),
                procedure_min=min(proc.values(), default=1.0), shortcut_max=max(short.values(), default=0.0)))


# ------------------------------------------------------------------------------------------------ the blind bound
def _blind_instance(args) -> dict:
    """Progress of every blind answer on one instance, after one token scan (so that the measurement gate is met)."""
    import mrgym

    from .. import policies
    from ..envs import family as F
    env_id, level, seed = args
    spec = mrgym.spec(env_id)
    controls = importlib.import_module(spec.family + ".controls")
    env = mrgym.make(env_id, level=level, seed=seed)
    env.reset()
    sc = policies.ToolScanner(env)
    sc.scan(TOKEN_SCAN)
    answers = controls.blind_answers(sc.info()["task"])
    cfg, out = env.episode_config(), {}
    for label, answer in answers.items():
        s = dict(env.task.score(cfg, answer, world=env.hidden_world(), engine=env._session.engine))
        s["checks"] = dict(s["checks"], log_replays_within_budget=True, measured=True)
        out[label] = F.progress_of(s)
    label, answer = next(reversed(answers.items()))                      # one of them through the environment itself
    end = env.step({"name": "submit", "arguments": answer})
    same = bool(end.done and abs(end.info["score"]["progress"] - out[label]) < 1e-12)
    env.close()
    return dict(progress=out, same_through_env=same)


def blind_progress(env_id: str, level: int, n: int = 60, workers: int = 2) -> dict:
    """Mean ``progress`` of each answer that uses the task card only (``controls.blind_answers``) on ``n`` fresh
    instances (seeds ``gate:0 .. gate:n-1``), each after one token scan. The best of them must stay at or below
    ``BLIND_PROGRESS_MAX``: the diagnostic may not pay an answer that ignores the object."""
    jobs = [(env_id, level, f"gate:{i}") for i in range(n)]
    if workers > 1 and n > 1:
        with multiprocessing.get_context("fork").Pool(min(int(workers), 2)) as pool:
            rows = pool.map(_blind_instance, jobs, chunksize=1)
    else:
        rows = [_blind_instance(j) for j in jobs]
    means = {label: round(float(np.mean([r["progress"][label] for r in rows])), 4) for label in rows[0]["progress"]}
    best = max(means, key=means.get)
    return dict(n=n, seeds=f"gate:0..gate:{n - 1}", answers=means, best=dict(answer=best, mean_progress=means[best]),
                bound=BLIND_PROGRESS_MAX, ok=means[best] <= BLIND_PROGRESS_MAX,
                computed_equals_environment=all(r["same_through_env"] for r in rows))


# ------------------------------------------------------------------------------------------------ the check
def run(env_id: str, level: int | None = None, quick: bool = False, workers: int = 2, verbose: bool = True, oracle: Oracle | None = None,
        space: Space | None = None, blind: bool = True, **overrides) -> dict:
    """Run the learnability check of ``env_id`` at ``level``. Returns the report (``verdict`` is "pass" or "fail").

    ``oracle`` / ``space`` replace the environment by any memoised episode function (used by the tests of the trainer
    itself); ``overrides`` replace entries of ``DEFAULTS``."""
    import mrgym
    cfg = dict(QUICK if quick else DEFAULTS, **overrides)
    say = (lambda s: print(s, flush=True)) if verbose else None
    t0 = time.time()
    if space is None:
        spec = mrgym.spec(env_id)
        level = spec.default_level if level is None else level
        controls = importlib.import_module(spec.family + ".controls") if spec.family else None
        if controls is None or not hasattr(controls, "policy_space"):
            raise mrgym.Error(f"{env_id} has no policy_space(): its family does not offer a learnability check")
        space = Space(controls.policy_space())
    own = oracle is None
    oracle = oracle or Oracle(space, env_id, level, workers=workers)
    train_seeds = [f"learn:train:{i}" for i in range(cfg["train_instances"])]
    eval_seeds = [f"learn:eval:{i}" for i in range(cfg["eval_instances"])]
    try:
        ref_idx = space.index(space.spec["reference"])
        oracle.resolve([(s, ref_idx) for s in eval_seeds])
        ref = float(np.mean([oracle[(s, ref_idx)]["passed"] for s in eval_seeds]))
        if say:
            say(f"{env_id} level {level}: {int(np.prod([len(g) for g in space.grids]))} policies in the family; its reference passes "
                f"{100 * ref:.0f} % of the {len(eval_seeds)} held-out instances")
        runs = []
        for signal, start, role in (("progress", "weak", "required"), ("paid", "in_band", "required"), ("paid", "weak", "expected to fail T1")):
            for seed in range(cfg["seeds"]):
                r = reinforce(space, oracle, start, signal, seed, train_seeds, eval_seeds, say=say, **cfg)
                r.update(role=role, tests=judge(r))
                runs.append(r)
        for signal in ("progress", "paid"):
            for seed in range(cfg["seeds"]):
                r = cem(space, oracle, "weak", signal, seed, train_seeds, eval_seeds, say=say, **cfg)
                r.update(role="reported", tests=judge(r))
                runs.append(r)
        played, seconds = len(oracle.cache), oracle.seconds
    finally:
        if own:
            oracle.close()
    report = dict(schema=SCHEMA, env_id=env_id, level=level, engine=mrgym.ENGINE_ID, package_version=mrgym.__version__,
                  date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
                  command=f"mrgym check {env_id} --learnability --level {level}" + (" --quick" if quick else ""),
                  settings=cfg, thresholds=dict(THRESHOLDS), parameters=[{k: v for k, v in p.items()} for p in space.params],
                  reference=dict(parameters=space.spec["reference"], held_out_pass=round(ref, 4)),
                  seeds=dict(train=f"learn:train:0..{cfg['train_instances'] - 1}", held_out=f"learn:eval:0..{cfg['eval_instances'] - 1}"),
                  runs=runs, episodes_played=played, episode_seconds=round(seconds, 1), workers=oracle.workers)
    report["blind"] = blind_progress(env_id, level, cfg["n_blind"], workers) if blind else None
    report["summary"] = summarise(report)
    weak_paid = [r for r in runs if r["optimiser"] == "policy_gradient" and r["role"] != "required"]
    report["can_fail"] = dict(what="paid reward from the weak start", fails_T1=[not r["tests"]["T1"]["ok"] for r in weak_paid])
    failed = [f"{r['optimiser']} {r['signal']} from {r['start']} seed {r['seed']}: {name}"
              for r in runs if r["role"] == "required" for name, t in r["tests"].items() if not t["ok"]]
    if report["blind"] and not report["blind"]["ok"]:
        failed.append(f"blind progress {report['blind']['best']['mean_progress']} > {BLIND_PROGRESS_MAX}")
    report.update(failed=failed, verdict="pass" if not failed else "fail", wall_seconds=round(time.time() - t0, 1))
    return report


def summarise(report: dict) -> list:
    """One row per run: what the acceptance of the check reads."""
    rows = []
    for r in report["runs"]:
        t = r["tests"]
        last = t.get("T2") or t["T4"]
        row = dict(optimiser=r["optimiser"], signal=r["signal"], start=r["start"], seed=r["seed"], role=r["role"],
                   pass_start=round(r["curve"][0]["eval_pass"], 4), pass_end=last["pass_end"], pass_end_sampled=last["pass_end_sampled"],
                   end_ok_sampled=last["ok_sampled"], ok={k: v["ok"] for k, v in t.items()})
        if "T1" in t:
            row.update(dead_groups_first_steps=t["T1"]["dead_groups_first_steps"], groups=r["groups"],
                       longest_all_dead_run=t["T1"]["longest_all_dead_run"],
                       longest_all_dead_run_literal=t["T1"]["longest_all_dead_run_literal"], T1_literal_ok=t["T1"]["ok_literal"],
                       pass_mid=t["T2"]["pass_mid"],
                       procedure_min=t["T3"]["procedure_min"], shortcut_max=t["T3"]["shortcut_max"])
        rows.append(row)
    return rows


def text_table(report: dict) -> str:
    out = [f"== learnability of {report['env_id']} level {report['level']}: {report['verdict'].upper()}  "
           f"({report['episodes_played']} distinct episodes, {report['wall_seconds']} s wall on {report['workers']} worker(s))",
           f"   reference policy on the held-out instances: {100 * report['reference']['held_out_pass']:.0f} %",
           f"   {'optimiser':<16}{'signal':<10}{'start':<9}{'seed':>4}  {'dead groups, first 10 steps':>28}  {'longest dead run':>16}  "
           f"{'pass start':>10} {'mid':>6} {'end':>6}  {'procedure':>9} {'shortcut':>8}  tests"]
    for r in report["summary"]:
        tests = " ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in r["ok"].items()) + ("" if r["role"] == "required" else f"  ({r['role']})")
        if r["optimiser"] == "cem":
            out.append(f"   {'cross-entropy':<16}{r['signal']:<10}{r['start']:<9}{r['seed']:>4}  {'':>28}  {'':>16}  {r['pass_start']:>10.3f} {'':>6} "
                       f"{r['pass_end']:>6.3f}  {'':>9} {'':>8}  {tests}")
        else:
            out.append(f"   {'policy gradient':<16}{r['signal']:<10}{r['start']:<9}{r['seed']:>4}  {str(r['dead_groups_first_steps']) + ' of ' + str(r['groups']):>28}  "
                       f"{r['longest_all_dead_run']:>16}  {r['pass_start']:>10.3f} {r['pass_mid']:>6.3f} {r['pass_end']:>6.3f}  "
                       f"{r['procedure_min']:>9.3f} {r['shortcut_max']:>8.3f}  {tests}")
    b = report.get("blind")
    if b:
        out.append(f"   best answer that ignores the object ({b['best']['answer']}): mean progress {b['best']['mean_progress']:.3f} on {b['n']} "
                   f"fresh instances (bound {b['bound']}): {'ok' if b['ok'] else 'FAIL'}")
    req = [r for r in report["summary"] if r["role"] == "required"]
    out.append(f"   pass at the end is exact over the policy grid; sampled on the held-out draws it reads "
               f"{[r['pass_end_sampled'] for r in req]} for the required runs ({sum(r['end_ok_sampled'] for r in req)} of {len(req)} meet their threshold)")
    lit = [r for r in report["summary"] if r["optimiser"] == "policy_gradient" and r["role"] == "required"]
    out.append(f"   T1 with every all-dead step counted (converged steps included), required runs: "
               f"{sum(r['T1_literal_ok'] for r in lit)} of {len(lit)} ok; longest runs {[r['longest_all_dead_run_literal'] for r in lit]}")
    out.append(f"   paid reward from the weak start fails T1 (the check can fail): {report['can_fail']['fails_T1']}")
    out += [f"   failed: {f}" for f in report["failed"]]
    return "\n".join(out)
