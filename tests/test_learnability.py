"""The CPU learnability check (``mrgym.learn.check``): the trainer on a toy family, the family of ``Scanner/NullTissue``.

What is shown
  * the policy space: sampling, exact probabilities, the gradient of the log-probability
  * on a toy family whose progress has a slope the check passes; when the progress is replaced by the paid reward the
    same check fails T1 from the weak start: the check can fail. The cross-entropy rows are printed and do not decide
  * the thresholds read a curve as documented; a converged policy is not a plateau, and the literal reading is kept
  * the NullTissue family: its reference corner plays exactly like the reference policy, the textbook corner like the
    textbook ablation, and the declared starts are below band / in band by their own parameters
The check of the real environment takes minutes and is a command (``mrgym check <id> --learnability``), not a test.
"""
import numpy as np
import pytest

import mrgym
from mrgym import policies
from mrgym.envs.null_tissue import controls
from mrgym.learn import check as C

NULL = "Scanner/NullTissue-v0"


def toy_space():
    """Three decisions: measure (0, 1 or 3 scans), use the model (bool), a dose in [0, 1] whose best value is 0.75."""
    return dict(
        parameters=[dict(name="scans", kind="choice", values=(0, 1, 3)), dict(name="model", kind="bool"),
                    dict(name="dose", kind="float", low=0.0, high=1.0, step=0.25)],
        build=None, reference=dict(scans=3, model=True, dose=0.75),
        starts=dict(weak=dict(scans=[0.6, 0.3, 0.1], model=0.05, dose=(0.2, 0.25)),
                    in_band=dict(scans=[0.1, 0.2, 0.7], model=0.7, dose=(0.7, 0.2))),
        procedure=dict(measures=lambda p: p["scans"] == 3, models=lambda p: bool(p["model"])),
        shortcuts=dict(no_scan=lambda p: p["scans"] == 0, no_model=lambda p: not p["model"]))


def toy_play(space, with_slope=True):
    def play(key):
        p = space.values(key[1])
        passed = p["scans"] == 3 and p["model"] and abs(p["dose"] - 0.75) < 0.3
        stage = (p["scans"] > 0) + (p["scans"] == 3) + bool(p["model"]) + (abs(p["dose"] - 0.75) < 0.3)
        return dict(passed=bool(passed), progress=1.0 if passed else (0.12 * stage if with_slope else 0.0), outcome="pass" if passed else "x")
    return play


def test_space_sampling_probabilities_and_gradient():
    space = C.Space(toy_space())
    assert space.n_theta == 3 + 1 + 2 and [len(g) for g in space.grids] == [3, 2, 5]
    th = space.theta(space.spec["starts"]["weak"])
    m = space.marginals(th)
    assert m[0] == pytest.approx([0.6, 0.3, 0.1]) and m[1][1] == pytest.approx(0.05) and all(x.sum() == pytest.approx(1.0) for x in m)
    assert space.prob(th, lambda p: True) == pytest.approx(1.0)
    assert space.prob(th, lambda p: p["scans"] == 0) == pytest.approx(0.6)
    assert space.prob(th, lambda p: p["scans"] == 3 and p["model"]) == pytest.approx(0.1 * 0.05)
    assert space.values(space.index(space.spec["reference"])) == space.spec["reference"]
    rng, n = np.random.default_rng(0), 4000
    draws = [space.sample(th, rng) for _ in range(n)]
    freq = np.bincount([d[0][2] for d in draws], minlength=5) / n
    assert np.allclose(freq, m[2], atol=0.03)                           # the snapped Gaussian matches its exact marginal
    assert np.allclose(np.mean([d[1] for d in draws], 0), 0.0, atol=0.08)    # the score function has mean zero
    # the gradient of the log-probability, against a finite difference of the exact marginals (choice and bool)
    idx, grad = draws[0]
    for j in range(4):
        e = np.zeros(space.n_theta)
        e[j] = 1e-5
        lp = lambda t: sum(np.log(space.marginals(t)[k][idx[k]]) for k in (0, 1))    # noqa: E731
        assert (lp(th + e) - lp(th - e)) / 2e-5 == pytest.approx(grad[j], abs=1e-6)
    with pytest.raises(ValueError):
        C.Space(dict(parameters=[dict(name="x", kind="other")]))


def test_group_gradient_drops_and_counts_dead_groups():
    grads = np.ones((16, 2))
    g, dead = C.group_gradient([0.0] * 8 + [1.0, 0, 0, 0, 0, 0, 0, 0], grads, groups=2, k=8)
    assert dead == 1 and np.allclose(g, 0.0)                             # advantages of a group sum to zero
    grads[8] = [5.0, -1.0]
    g, dead = C.group_gradient([0.0] * 8 + [1.0, 0, 0, 0, 0, 0, 0, 0], grads, groups=2, k=8)
    assert dead == 1 and g[0] > 0 > g[1]                                 # the rewarded episode's direction, the dead group silent


def test_the_check_passes_with_a_slope_and_fails_without_one():
    space = C.Space(toy_space())
    settings = dict(steps=120, seeds=2, train_instances=6, eval_instances=6)
    report = C.run("Toy/Slope-v0", 1, space=space, oracle=C.Oracle(space, play=toy_play(space)), blind=False, verbose=False, **settings)
    assert report["verdict"] == "pass", report["failed"]
    rows = {(r["optimiser"], r["signal"], r["start"], r["seed"]): r for r in report["summary"]}
    assert all(rows[("policy_gradient", "progress", "weak", s)]["ok"] == dict(T1=True, T2=True, T3=True) for s in (0, 1))
    assert all(rows[("policy_gradient", "paid", "in_band", s)]["ok"] == dict(T1=True, T2=True, T3=True) for s in (0, 1))
    cem = [r for r in report["summary"] if r["optimiser"] == "cem"]
    assert len(cem) == 4 and all(r["role"] == "reported" and set(r["ok"]) == {"T4"} for r in cem)       # printed, not in the verdict
    assert any(r["ok"]["T4"] for r in cem if r["signal"] == "progress")
    assert report["can_fail"]["fails_T1"] == [True, True]                # the paid reward from the weak start has dead groups
    assert report["episodes_played"] <= (6 + 6) * 3 * 2 * 5              # every distinct (instance, policy) is played once
    assert "PASS" in C.text_table(report)
    # the same family when the diagnostic has no slope (it equals the paid reward): the required arm fails T1
    flat = C.run("Toy/Flat-v0", 1, space=space, oracle=C.Oracle(space, play=toy_play(space, with_slope=False)), blind=False,
                 verbose=False, **settings)
    assert flat["verdict"] == "fail" and any("progress from weak" in f and f.endswith("T1") for f in flat["failed"])


def test_thresholds_read_a_curve_as_documented():
    def run(dead, train_pass, evals, procedure=0.99, shortcut=0.01, groups=8):
        steps = len(dead)
        curve = [dict(step=i, dead=d, train_pass=p) for i, (d, p) in enumerate(zip(dead, train_pass))] + [dict(step=steps)]
        for s, v in evals.items():
            curve[s]["eval_pass"] = v
        return dict(optimiser="policy_gradient", groups=groups, steps=steps, curve=curve, procedure=dict(a=procedure), shortcuts=dict(b=shortcut))

    good = C.judge(run([1] * 10 + [8] * 10, [0.3] * 10 + [1.0] * 10, {0: 0.1, 12: 0.6, 20: 0.97}))
    assert good["T1"]["ok"] and good["T2"]["ok"] and good["T3"]["ok"]
    assert good["T1"]["longest_all_dead_run"] == 0 and good["T1"]["longest_all_dead_run_literal"] == 10 and not good["T1"]["ok_literal"]
    plateau = C.judge(run([8] * 10 + [0] * 10, [0.0] * 10 + [0.5] * 10, {0: 0.0, 12: 0.6, 20: 0.97}))
    assert not plateau["T1"]["ok"] and plateau["T1"]["longest_all_dead_run"] == 10 and plateau["T1"]["dead_groups_first_steps"] == 8.0
    late_plateau = C.judge(run([0] * 10 + [8] * 4 + [0] * 6, [0.3] * 20, {0: 0.1, 12: 0.6, 20: 0.97}))
    assert not late_plateau["T1"]["ok"] and late_plateau["T1"]["longest_all_dead_run"] == 4
    slow = C.judge(run([0] * 20, [0.3] * 20, {0: 0.1, 12: 0.4, 20: 0.97}))
    assert slow["T1"]["ok"] and not slow["T2"]["ok"]
    short = C.judge(run([0] * 20, [0.3] * 20, {0: 0.1, 12: 0.6, 20: 0.94}))
    assert not short["T2"]["ok"]
    assert not C.judge(run([0] * 20, [0.3] * 20, {0: 0.1, 12: 0.6, 20: 0.97}, shortcut=0.05))["T3"]["ok"]
    assert not C.judge(run([0] * 20, [0.3] * 20, {0: 0.1, 12: 0.6, 20: 0.97}, procedure=0.90))["T3"]["ok"]
    cem = dict(optimiser="cem", curve=[dict(eval_pass=0.0), dict(eval_pass=0.89)])
    assert not C.judge(cem)["T4"]["ok"] and C.judge(dict(cem, curve=[dict(eval_pass=0.9)]))["T4"]["ok"]


def test_null_tissue_family_corners_are_the_control_policies():
    spec = controls.policy_space()
    space = C.Space(spec)
    assert len(space.grids) == 5 and int(np.prod([len(g) for g in space.grids])) == 128
    assert space.values(space.index(spec["reference"])) == spec["reference"]

    def answer(policy, level=2, seed=7):
        env = mrgym.make(NULL, level=level, seed=seed)
        env.reset()
        return policy(policies.ToolScanner(env)), env.scanner_log()[-1]["chain"]

    # the family's best corner is the transfer procedure (four probes, recovery model, check scan); the control reference
    # also measures at the target's own repetition time when the budget pays that, which the family does not parametrise
    transfer = lambda sc: controls._reference_transfer(sc, controls._task(sc))       # noqa: E731
    assert answer(spec["build"](spec["reference"])) == answer(transfer)                 # same answer, same log
    textbook = dict(spec["reference"], finite_tr_model=False, check_scan=True)
    assert answer(spec["build"](textbook))[0] == answer(controls.textbook)[0]
    env = mrgym.make(NULL, level=2, seed=7)
    env.reset()
    assert spec["build"](dict(spec["reference"], n_probes=0))(policies.ToolScanner(env)) == controls.prior(policies.ToolScanner(env))
    assert env.scanner_log() == [] or all(e["type"] != "scan" for e in env.scanner_log())      # no scan was made
    # the declared starts, by their own parameters: weak is below 2 % on the two decisions, in_band inside 20 - 60 %
    both = lambda p: all(f(p) for f in spec["procedure"].values())      # noqa: E731
    assert space.prob(space.theta(spec["starts"]["weak"]), both) < 0.02
    assert 0.20 <= space.prob(space.theta(spec["starts"]["in_band"]), both) <= 0.60
    assert all(not f(spec["reference"]) for f in spec["shortcuts"].values()) and both(spec["reference"])


def test_oracle_plays_the_environment_and_memoises():
    spec = controls.policy_space()
    space = C.Space(spec)
    oracle = C.Oracle(space, NULL, 2, workers=1)
    ref, none = space.index(spec["reference"]), space.index(dict(spec["reference"], n_probes=0))
    oracle.resolve([("learn:train:0", ref), ("learn:train:0", none), ("learn:train:0", ref)])
    assert len(oracle.cache) == 2
    assert oracle[("learn:train:0", ref)] == dict(passed=True, progress=1.0, outcome="pass", policy_error=None)
    assert oracle[("learn:train:0", none)]["outcome"] == "no_measurement" and oracle[("learn:train:0", none)]["progress"] == 0.0
    before = oracle.seconds
    oracle.resolve([("learn:train:0", ref)])
    assert oracle.seconds == before
    with pytest.raises(mrgym.Error):
        C.run("Scanner/T1Transmit-v1", 1, verbose=False)                   # a family without a policy_space
