"""The curriculum sampler (``mrgym.curriculum``): numpy only, no environment, no model.

What is shown
  * the learning-progress reward of a group: 0 without spread, 1 for a half-passed group
  * (a) on a simulated learner the bandit brings the hardest reachable arm to p >= 0.6 in at most 70 % of the steps that
    uniform sampling needs, and from step 20 until then gives at most 20 % of the groups to an arm that is never
    passed. After everything reachable is mastered it tends to uniform sampling, and the share over a long run is
    about a fifth
  * (b) ``state_dict`` / ``load_state_dict`` resume exactly
  * (c) no seed is issued twice in 10,000 draws, and none below ``first_seed``
  * the ladder mode promotes by threshold; a dead arm starts at the floor; the row hook groups rollouts
"""
import json

import numpy as np
import pytest

from mrgym.curriculum import Curriculum, RowHook, assert_disjoint, learning_progress, rows

ENV = "Scanner/NullTissue-v0"
ARMS = [(ENV, 0), (ENV, 1), (ENV, 2), (ENV, 3)]
DIFFICULTY = {0: -4.0, 1: 0.0, 2: 2.5, 3: 14.0}        # level 0 is mastered from the start, level 3 is out of reach
G, GROUPS, ETA = 8, 16, 0.02


def _simulate(mode, steps, seed=0, track=None):
    """A learner whose pass rate on an arm is sigmoid(skill - difficulty) and whose skill grows with the number of
    episodes that are neither surely passed nor surely failed: skill += ETA * sum over groups of p (1 - p).
    Returns the first step at which p >= 0.6 on level 2 (None if never)."""
    cur, rng, skill, reached = Curriculum(ARMS, mode=mode, seed=seed), np.random.default_rng(1000 + seed), 0.0, None
    for step in range(steps):
        p_of = {lv: 1.0 / (1.0 + np.exp(-(skill - d))) for lv, d in DIFFICULTY.items()}
        if reached is None and p_of[2] >= 0.6:
            reached = step
        gain = 0.0
        for row in cur.sample(GROUPS):
            p = p_of[row["level"]]
            cur.update(row["group_id"], (rng.random(G) < p).astype(float))
            gain += p * (1.0 - p)
            if track is not None and step >= 20:
                track.append(row["level"])
        cur.end_step()
        skill += ETA * gain
    return reached


def test_learning_progress_of_a_group():
    assert learning_progress([0.0] * 8) == 0.0 and learning_progress([1.0] * 8) == 0.0
    assert learning_progress([1, 1, 1, 1, 0, 0, 0, 0]) == pytest.approx(1.0)
    assert learning_progress([1, 0, 0, 0, 0, 0, 0, 0]) == pytest.approx(2 * np.sqrt(1 / 8 * 7 / 8))    # 2 sqrt(p (1 - p))
    assert learning_progress([0.1, 0.2, 0.3, 0.4]) == pytest.approx(learning_progress([1.0, 2.0, 3.0, 4.0]))   # scale-free


def test_bandit_is_faster_than_uniform_and_leaves_the_dead_arm():
    ratios, learning, whole = [], [], []
    for seed in range(5):
        uniform = _simulate("uniform", 400, seed)
        track = []
        bandit = _simulate("bandit", 400, seed, track)
        assert uniform is not None and bandit is not None
        ratios.append(bandit / uniform)
        track = np.array(track)                                          # the levels of the groups from step 20 on
        learning.append(float(np.mean(track[:(bandit - 20) * GROUPS] == 3)))
        whole.append(float(np.mean(track == 3)))
    assert max(ratios) <= 0.70, ratios
    # while there is something to learn (step 20 to the step at which level 2 reaches 0.6) the dead arm gets its floor
    assert max(learning) <= 0.20, learning
    # once every reachable level is mastered no arm has learning progress left and the sampler tends to uniform (a quarter
    # per arm): over the whole 400 steps the dead arm's share is therefore about a fifth, not below it
    assert 0.15 <= min(whole) and max(whole) <= 0.25, whole


def test_state_dict_round_trips_and_resumes_exactly():
    a = Curriculum(ARMS, seed=3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        for row in a.sample(6):
            a.update(row["group_id"], (rng.random(G) < 0.4).astype(float))
        a.end_step()
    pending = a.sample(4)                                                # a checkpoint in the middle of a step
    a.update(pending[0]["group_id"], [1, 0, 1, 0, 0, 0, 0, 0])
    state = json.loads(json.dumps(a.state_dict()))                       # survives JSON
    b = Curriculum(ARMS, seed=99)
    b.load_state_dict(state)
    assert b.state_dict() == a.state_dict()
    for cur in (a, b):
        for row in pending[1:]:
            cur.update(row["group_id"], [1, 1, 0, 0, 0, 0, 0, 0])
        cur.end_step()
    assert a.sample(32) == b.sample(32) and np.array_equal(a.Q, b.Q) and a.metrics() == b.metrics()
    with pytest.raises(ValueError):
        Curriculum(ARMS[:2]).load_state_dict(state)


def test_no_seed_is_issued_twice():
    cur = Curriculum(ARMS, seed=1, first_seed=10_000)
    seen = [r["seed"] for _ in range(100) for r in cur.sample(100)]
    assert len(seen) == 10_000 == len(set(seen)) and min(seen) == 10_000
    assert len({r["group_id"] for r in cur.sample(50)}) == 50
    with pytest.raises(KeyError):
        cur.update("unknown-group", [0.0] * 8)


def test_probabilities_floor_and_dead_arms():
    cur = Curriculum(ARMS, floor=0.10, dead=[(ENV, 3)])
    p = cur.probs()
    assert p.sum() == pytest.approx(1.0) and p[3] == pytest.approx(0.10 / 4, abs=1e-3) and p[0] == pytest.approx(p[1])
    cur = Curriculum(ARMS, mode="uniform")
    assert np.allclose(cur.probs(), 0.25)
    for bad in (dict(mode="other"), dict(alpha=0.0), dict(tau=0.0), dict(floor=1.5)):
        with pytest.raises(ValueError):
            Curriculum(ARMS, **bad)
    with pytest.raises(ValueError):
        Curriculum([(ENV, 1), (ENV, 1)])


def test_ladder_mode_promotes_by_threshold():
    cur = Curriculum(ARMS[:3], mode="ladder", promote=0.70, window=3)
    rate = {0: 7, 1: 3, 2: 0}                                            # passes of 8 per level: only level 0 is above the threshold

    def step(n):
        for row in cur.sample(n):
            cur.update(row["group_id"], [1.0] * rate[row["level"]] + [0.0] * (8 - rate[row["level"]]))
        cur.end_step()

    assert cur.probs().tolist() == [1.0, 0.0, 0.0]                      # nothing played: the lowest level
    step(4)
    step(4)
    assert cur.probs().tolist() == [1.0, 0.0, 0.0]                      # above the threshold, but the window is not full yet
    step(4)
    assert cur.probs().tolist() == pytest.approx([0.2, 0.8, 0.0])        # promoted; a fifth stays below
    for _ in range(6):
        step(40)
    assert cur.probs().tolist() == pytest.approx([0.2, 0.8, 0.0])        # level 1 stays below the threshold: no promotion
    rate[1] = 8
    for _ in range(3):
        step(40)
    assert cur.probs().tolist() == pytest.approx([0.1, 0.1, 0.8])        # promoted again; the fifth is spread over the levels below


def test_metrics_and_the_progress_signal():
    cur = Curriculum(ARMS[:2], seed=0)
    for row in cur.sample(8):
        cur.update(row["group_id"], [0.1, 0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], passed=[0] * 8)      # a trainer's own signal
    m = cur.end_step()
    names = [f"{ENV}/L0", f"{ENV}/L1"]
    assert m["curriculum/step"] == 1 and m["curriculum/seeds_issued"] == 8
    assert sum(m[f"curriculum/groups[{n}]"] for n in names) == 8
    played = [n for n in names if m[f"curriculum/groups[{n}]"]]
    assert all(m[f"curriculum/pass_rate[{n}]"] == 0.0 and m[f"curriculum/mixed_share[{n}]"] == 1.0 and m[f"curriculum/Q[{n}]"] > 0 for n in played)


def test_row_hook_groups_the_rollouts_of_a_row():
    cur = Curriculum(ARMS[:2], seed=0)
    hook = RowHook(cur, group_size=4, groups_per_step=2)
    first = hook.assign("row-a")
    assert hook.assign("row-a") == first and hook.assign("row-b") != first      # one instance per row, another per other row
    for uid in ("row-a", "row-b"):
        for k in range(4):
            assert cur.steps == 0
            hook.report(uid, float(k % 2), bool(k % 2))
    assert cur.steps == 1 and hook.last_metrics["curriculum/step"] == 1 and cur.issued == 2
    assert hook.assign("row-a") != first                                 # the next step: a fresh instance for the same row


def test_static_rows_and_disjoint_splits():
    train, dev = rows(ENV, [1, 2], range(10_000, 10_004)), rows(ENV, [1, 2], range(0, 4))
    assert len(train) == 8 and train[0] == dict(env_id=ENV, level=1, seed=10_000, split="train", group_id=f"{ENV}|L1|train|s10000")
    assert_disjoint(train, dev)
    with pytest.raises(ValueError):
        assert_disjoint(train, rows(ENV, [2], [10_003]))
