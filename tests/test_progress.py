"""The progress diagnostic and the outcome label of the family framework (``mrgym.envs.family``).

What is shown
  * order: over random answers the largest progress of a class of outcome lies below the smallest of the next class,
    every failing answer scores below 1, and exactly 1 is a pass
  * slope: on the engine, moving one inversion time towards the null never lowers progress and raises it while the
    region is neither nulled nor far away
  * the generic default (no family value) is a staircase over the checks in their declared order
  * gates: nothing measured, or a log outside the budget, scores 0 whatever the answer is
  * the outcome label is one name of a closed, ordered taxonomy
  * the paid reward stays pass / fail: progress never reaches ``reward``
"""
import numpy as np
import pytest

import mrgym
from mrgym import policies
from mrgym.envs import family as F
from mrgym.envs.null_tissue import task
from mrgym.learn import check as learnability
from mrgym.learn import hooks

NULL = "Scanner/NullTissue-v0"


@pytest.mark.parametrize("n_regions", [2, 3])
def test_progress_order_classes_never_overlap(n_regions):
    rng, eps = np.random.default_rng(n_regions), 0.05
    by_class = {}
    for _ in range(2000):
        ratios = {f"R{i + 1}": float(np.exp(rng.uniform(np.log(1e-3), np.log(1.0)))) for i in range(n_regions)}
        p = task.progress(ratios, eps)
        k = sum(v <= eps for v in ratios.values())
        by_class.setdefault(k, []).append(p)
        assert (p == 1.0) == (k == n_regions) and 0.0 <= p <= 1.0
        if k < n_regions:
            assert p < F.PROGRESS_FAIL_MAX
    assert sorted(by_class) == list(range(n_regions + 1))               # every class occurred
    for k in range(n_regions):
        assert max(by_class[k]) < min(by_class[k + 1]), f"class {k} reaches into class {k + 1}"


def test_progress_order_closer_is_never_worse():
    eps = 0.05
    for far, near in ((0.9, 0.5), (0.5, 0.3), (0.3, 0.1), (0.1, 0.051)):
        assert task.progress({"R1": far, "R2": 0.01}, eps) <= task.progress({"R1": near, "R2": 0.01}, eps)
    assert task.progress({"R1": 0.3, "R2": 0.01}, eps) < task.progress({"R1": 0.1, "R2": 0.01}, eps)   # a slope below the pass line
    assert task.progress({"R1": 0.9, "R2": 0.9}, eps) == 0.0             # half the signal or more left everywhere: nothing


def test_generic_default_is_a_staircase_over_the_declared_checks():
    gates = dict(log_replays_within_budget=True, measured=True)
    def score(*own, **kw):
        checks = dict({f"c{i}": ok for i, ok in enumerate(own)}, **dict(gates, **kw))
        return dict(passed=all(checks.values()), checks=checks)
    steps = [F.progress_of(score(*([True] * i + [False] * (4 - i)))) for i in range(4)]
    assert steps == sorted(steps) and len(set(steps)) == 4 and steps[0] == 0.0 and steps[-1] < F.PROGRESS_FAIL_MAX
    assert F.progress_of(score(True, True, True, True)) == 1.0
    assert F.progress_of(score(True, False, True, True)) == F.progress_of(score(True, False, False, False))   # the first failure decides
    assert F.progress_of(score(True, True, True, True, measured=False)) == 0.0
    assert F.progress_of(score(True, True, True, False, log_replays_within_budget=False)) == 0.0
    # a family's own value is used for a failing answer, never above the supremum, never for a pass or a failed gate
    own = dict(passed=False, checks=dict(deliverable_valid=True, c=False, **gates))
    assert F.progress_of(dict(own, progress=0.3)) == 0.3 and F.progress_of(dict(own, progress=7.0)) == F.PROGRESS_SUP < 1.0
    assert F.progress_of(dict(own, progress=0.3, checks=dict(own["checks"], measured=False))) == 0.0
    assert F.progress_of(dict(own, progress=0.3, checks=dict(own["checks"], deliverable_valid=False))) == 0.0


def test_outcome_is_one_name_of_a_closed_ordered_taxonomy():
    names = F.outcomes(task)
    assert names[:len(hooks.NO_SUBMISSION)] == hooks.NO_SUBMISSION and names[-1] == "pass" and len(set(names)) == len(names)
    assert names.index("no_measurement") < names.index("log_or_budget") < names.index("none_nulled") < names.index("some_nulled")
    assert F.outcomes(None)[-2:] == ("check_failed", "pass")
    base = dict(deliverable_valid=True, nulled_R1=True, nulled_R2=False, log_replays_within_budget=True, measured=True)
    assert F.outcome_of(dict(passed=False, checks=base), task) == "some_nulled"
    assert F.outcome_of(dict(passed=False, checks=dict(base, nulled_R1=False)), task) == "none_nulled"
    assert F.outcome_of(dict(passed=False, checks=dict(base, measured=False)), task) == "no_measurement"
    assert F.outcome_of(dict(passed=False, checks=dict(base, log_replays_within_budget=False)), task) == "log_or_budget"
    assert F.outcome_of(dict(passed=False, checks=dict(base, deliverable_valid=False)), task) == "invalid_deliverable"
    assert F.outcome_of(dict(passed=True, checks=dict(base, nulled_R2=True)), task) == "pass"
    assert F.outcome_of(dict(passed=False, checks=base), None) == "check_failed"


@pytest.fixture(scope="module")
def played():
    """The control policies of one level-2 instance, and the environment of the reference's episode."""
    out = {}
    for name in ("reference", "prior", "textbook", "wrong_units"):
        env = mrgym.make(NULL, level=2, seed=3)
        out[name] = (policies.run(env, name), env)
    return out


def test_progress_and_outcome_of_the_control_policies(played):
    ref, prior, textbook, units = (played[k][0] for k in ("reference", "prior", "textbook", "wrong_units"))
    assert ref.reward == 1.0 and ref.info["score"]["progress"] == 1.0 and ref.info["outcome"] == "pass"
    assert prior.reward == 0.0 and prior.info["score"]["progress"] == 0.0 and prior.info["outcome"] == "no_measurement"
    assert textbook.reward == 0.0 and 0.0 < textbook.info["score"]["progress"] < F.PROGRESS_FAIL_MAX
    assert textbook.info["outcome"] in task.OUTCOMES
    # the wrong unit is refused by the console (the value is below its range): nothing is delivered, the episode has no last step
    end = played["wrong_units"][1].end_without_submission("stopped")
    assert units.reward == 0.0 and not units.done or units.truncated
    assert end["outcome"] in hooks.NO_SUBMISSION and end["progress"] == 0.0 and not end["ladder"]["submitted"]
    for r in (ref, prior, textbook):                                     # the paid reward is the pass and nothing else
        assert r.reward == (1.0 if r.info["score"]["passed"] else 0.0) and r.info["progress"] == r.info["score"]["progress"]
        assert r.info["outcome"] in F.outcomes(task) and not hooks.validate(r.info, F.outcomes(task), final=True)


def test_progress_slope_towards_the_null_on_the_engine(played):
    ref, env = played["reference"]
    cfg, world, engine = env.episode_config(), env.hidden_world(), env._session.engine
    answer = env.transcript()[-1]["call"]["arguments"]["ti_ms"]         # what the reference delivered (it passes)
    assert ref.reward == 1.0 and set(answer) == {r["id"] for r in cfg["task"]["regions"]}
    first = sorted(answer)[0]

    def progress_at(factor):
        s = dict(task.score(cfg, dict(ti_ms=dict(answer, **{first: round(answer[first] * factor, 1)})), world=world, engine=engine))
        s["checks"] = dict(s["checks"], log_replays_within_budget=True, measured=True)
        return F.progress_of(s), s["null_ratio"][first]

    below = [progress_at(f) for f in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    above = [progress_at(f) for f in (1.0, 1.1, 1.25, 1.5, 1.8, 2.2)]
    assert below[-1][0] == 1.0 and above[0][0] == 1.0
    for seq in (below, above[::-1]):                                     # towards the null: never down
        values = [p for p, _ in seq]
        assert values == sorted(values)
        rising = [p for p, ratio in seq if cfg["task"]["null_max"] < ratio < task.FAR_RATIO]
        assert len(rising) >= 2 and all(b > a for a, b in zip(rising, rising[1:]))   # and strictly up where the region is approached
        assert max(p for p, ratio in seq if ratio > cfg["task"]["null_max"]) < F.PROGRESS_FAIL_MAX


def test_blind_answers_stay_under_the_bound_on_a_few_instances():
    """The mechanics of the bound on 3 instances; the 60-instance number comes from ``mrgym check <id> --learnability``."""
    out = learnability.blind_progress(NULL, 2, n=3, workers=1)
    assert out["computed_equals_environment"] and out["n"] == 3 and len(out["answers"]) >= 10
    assert out["best"]["mean_progress"] == max(out["answers"].values()) and out["bound"] == learnability.BLIND_PROGRESS_MAX
    assert all(0.0 <= v < F.PROGRESS_FAIL_MAX for v in out["answers"].values())
