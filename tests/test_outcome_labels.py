"""The outcome labels of ``Scanner/SNRSpec-v0``, ``BandFree-v0``, ``ArtefactClinic-v0`` and ``ProtocolToSpec-v0``.

These four reported every delivered, measured answer that fails as ``check_failed``. Each now names the first
requirement of its pass rule that the delivery misses (``task.REQUIREMENTS``, ``task.OUTCOMES``, ``task.outcome``).

  pinned     the label tuples are stable strings; ``family.outcomes(task)`` and ``env.outcomes()`` list them
  function   the label is a function of the checks: every check of the verifier belongs to exactly one requirement;
             the first missed requirement wins whatever fails after it; the image, finding or region does not matter
  episodes   on ``SEEDS`` of the level at which every ablation is named: the reference policy ends in ``pass``; every
             shortcut (``MUST_FAIL``) and every ablation that names a level ends in a label of the taxonomy that is
             never ``check_failed``; a delivered, measured, failed answer ends in the label of its first missed requirement
  unchanged  the pass, the reward and the progress of a failed delivery do not depend on the label

The episodes are nearly all of the time of this file: the band-free ones, and the table that the ``spec_only`` shortcut
of SNRSpec tunes once per process. The seeds are ``MRGYM_OUTCOME_SEEDS`` (default ``0``; ``0,1`` for the full check: instances the reference
passes); ``tools/audit/policy_identity.py`` plays every policy on every level.
"""
import importlib
import itertools
import os

import pytest

import mrgym
from mrgym import policies
from mrgym.envs import family as F
from mrgym.learn import hooks

GATES = dict(log_replays_within_budget=True, measured=True)
SEEDS = tuple(int(x) for x in os.environ.get("MRGYM_OUTCOME_SEEDS", "0").split(","))
# id: (package, the pinned labels worst first, {label: the checks behind it, as the verifier names them}, the level played)
FAMILIES = {
    "Scanner/SNRSpec-v0": (
        "mrgym.envs.snr_spec", ("snr_too_low", "scan_too_long"),
        dict(snr_too_low=("snr_A", "snr_B"), scan_too_long=("time_A", "time_B")), 3),
    "Scanner/BandFree-v0": (
        "mrgym.envs.band_free", ("band_on_region", "second_acquisition_not_needed", "bandwidth_too_high"),
        dict(band_on_region=("band_free_A", "band_free_B"), second_acquisition_not_needed=("acquisitions_needed_A", "acquisitions_needed_B"),
             bandwidth_too_high=("bandwidth_low_A", "bandwidth_low_B")), 3),
    "Scanner/ArtefactClinic-v0": (
        "mrgym.envs.artefact_clinic",
        ("diagnosis_wrong", "remedy_leaves_artefact", "locked_parameter_changed", "parameter_outside_window", "pixel_too_large",
         "scan_too_long", "snr_too_low"),
        dict(diagnosis_wrong=("diagnosis_F1", "diagnosis_F2", "diagnosis_F3", "diagnosis_F4"),
             remedy_leaves_artefact=("clean_F1", "clean_F2", "clean_F3", "clean_F4"), locked_parameter_changed=("locked_parameters",),
             parameter_outside_window=("te_window", "slice_window"), pixel_too_large=("pixel_size",), scan_too_long=("scan_time",),
             snr_too_low=("snr_reference",)), 3),
    "Scanner/ProtocolToSpec-v0": (
        "mrgym.envs.protocol_to_spec", ("t1_contrast_too_low", "t2_contrast_too_low", "snr_too_low", "scan_too_long"),
        dict(t1_contrast_too_low=("t1_contrast_A", "t1_contrast_B"), t2_contrast_too_low=("t2_contrast_A", "t2_contrast_B"),
             snr_too_low=("snr_A", "snr_B"), scan_too_long=("time",)), 3),
}
IDS = [pytest.param(env_id, id=env_id.split("/")[1]) for env_id in FAMILIES]


def _task(env_id):
    return importlib.import_module(FAMILIES[env_id][0] + ".task")


def _score(passed=False, **checks):
    return dict(passed=passed, checks=dict(dict(deliverable_valid=True, **checks), **{k: checks.get(k, v) for k, v in GATES.items()}))


def _expected(env_id, checks) -> str:
    """The label by the table of this file: the first label, in the pinned order, with a failed check."""
    _, labels, behind, _ = FAMILIES[env_id]
    return next(label for label in labels if not all(checks.get(k, True) for k in behind[label]))


# ---------------------------------------------------------------------------------------------- pinned
@pytest.mark.parametrize("env_id", IDS)
def test_the_labels_are_pinned_and_listed_by_the_family_and_the_environment(env_id):
    task, (_, labels, behind, level) = _task(env_id), FAMILIES[env_id]
    assert task.OUTCOMES == labels and tuple(label for label, _ in task.REQUIREMENTS) == labels and tuple(behind) == labels
    assert 2 <= len(labels) <= 8 and len(set(labels)) == len(labels) and all(x == x.lower() and " " not in x for x in labels)
    assert F.outcomes(task) == hooks.NO_SUBMISSION + F.GATE_OUTCOMES + labels + ("pass",)
    assert "check_failed" not in F.outcomes(task) and len(set(F.outcomes(task))) == len(F.outcomes(task))
    env = mrgym.make(env_id, level=level, seed=0)
    assert env.outcomes() == F.outcomes(task)
    env.close()


def test_the_window_checks_of_the_clinic_are_the_ones_its_verifier_can_name():
    """``<parameter>_window`` for every parameter the requirements may bound (``WINDOWED``): the echo time at every
    level, the slice thickness at level 3, the repetition time at none today."""
    task = _task("Scanner/ArtefactClinic-v0")
    stems = dict(task.REQUIREMENTS)["parameter_outside_window"]
    assert stems == ("te_window", "tr_window", "slice_window") == tuple(f"{k.split('_')[0]}_window" for k in task.WINDOWED)
    for name in stems:
        assert F.outcome_of(_score(**{name: False, "pixel_size": False}), task) == "parameter_outside_window"


# ---------------------------------------------------------------------------------------------- a function of the checks
@pytest.mark.parametrize("env_id", IDS)
def test_the_label_is_the_first_missed_requirement_of_the_checks(env_id):
    task, (_, labels, behind, _) = _task(env_id), FAMILIES[env_id]
    names = [k for label in labels for k in behind[label]]
    ok = dict.fromkeys(names, True)
    assert F.outcome_of(_score(passed=True, **ok), task) == "pass"
    for i, label in enumerate(labels):
        for k in behind[label]:                                          # one failed check: its requirement, whichever image or finding
            assert task.outcome(_score(**dict(ok, **{k: False}))) == label == F.outcome_of(_score(**dict(ok, **{k: False})), task)
        every = dict(ok, **dict.fromkeys(behind[label], False))
        assert task.outcome(_score(**every)) == label
        later = [k for other in labels[i + 1:] for k in behind[other]]   # whatever fails after it does not change the label
        for n in range(1, len(later) + 1):
            for more in (later[:n], later[-n:]):
                assert task.outcome(_score(**dict(every, **dict.fromkeys(more, False)))) == label
                assert task.outcome(_score(**dict(ok, **{behind[label][-1]: False}, **dict.fromkeys(more, False)))) == label
    for a, b in itertools.combinations(names, 2):                        # every pair of failed checks: the earlier requirement
        checks = dict(ok, **{a: False, b: False})
        assert task.outcome(_score(**checks)) == _expected(env_id, checks)
    # the order in which the verifier lists the checks does not matter, and nothing but the checks is read
    scrambled = dict(reversed(list(dict(ok, **{names[-1]: False, names[0]: False}).items())))
    assert task.outcome(dict(checks=scrambled)) == labels[0]
    assert task.outcome(dict(passed=False, checks=dict(ok, **{names[-1]: False}), diagnostic="x", error=3.0, outcome="pass")) == labels[-1]
    # the gates come before the family's labels
    bad = dict(ok, **{names[0]: False})
    assert F.outcome_of(dict(passed=False, checks=dict(bad, deliverable_valid=False, **GATES)), task) == "invalid_deliverable"
    assert F.outcome_of(_score(**dict(bad, measured=False)), task) == "no_measurement"
    assert F.outcome_of(_score(**dict(bad, log_replays_within_budget=False)), task) == "log_or_budget"


# ---------------------------------------------------------------------------------------------- episodes
def _controls(env_id):
    return importlib.import_module(FAMILIES[env_id][0] + ".controls")


def _policies(env_id) -> list:
    """The reference, every shortcut and every ablation that names a level (the ones the gate holds to a loss)."""
    c = _controls(env_id)
    return [c.REFERENCE, *c.MUST_FAIL, *(name for name, levels in c.ABLATIONS.items() if levels)]


@pytest.fixture(scope="module")
def played():
    """``played(id)``: the policies of ``_policies`` on ``SEEDS`` of the family's level, played once per module:
    [(policy, seed, final info, score, taxonomy, info of the last step or None)]. A policy whose deliverable the
    console refuses delivers nothing: the harness ends that episode, which then has no last step."""
    cache = {}

    def play(env_id):
        if env_id not in cache:
            rows = []
            for seed in SEEDS:
                for name in _policies(env_id):
                    env = mrgym.make(env_id, level=FAMILIES[env_id][3], seed=seed)
                    res = policies.run(env, name)
                    last = res.info if "outcome" in res.info else None
                    final = last or env.end_without_submission("stopped")
                    rows.append((name, seed, final, res.info.get("score"), env.outcomes(), last))
                    env.close()
            cache[env_id] = rows
        return cache[env_id]
    return play


@pytest.mark.parametrize("env_id", IDS)
def test_the_reference_passes_and_no_shortcut_or_ablation_ends_in_check_failed(played, env_id):
    task, controls, labels, rows = _task(env_id), _controls(env_id), FAMILIES[env_id][1], played(env_id)
    named = {name for name, levels in controls.ABLATIONS.items() if FAMILIES[env_id][3] in levels}
    assert len(rows) == len(SEEDS) * len(_policies(env_id))
    assert {name for name, *_ in rows} >= set(controls.MUST_FAIL) | named | {controls.REFERENCE} and named
    seen = set()
    for name, seed, final, score, taxonomy, last in rows:
        where = f"{env_id} {name} seed {seed}"
        assert final["outcome"] in taxonomy and final["outcome"] != "check_failed", where
        assert last is None or hooks.validate(last, taxonomy, final=True) == [], where
        assert (final["outcome"] == "pass") == final["passed"] == (final["reward"] == 1.0), where
        if name == controls.REFERENCE:                                   # on these seeds; the gate holds its rate, and the shortcuts' rates
            assert final["outcome"] == "pass", where
        if final["ladder"]["in_budget"] and not final["passed"]:         # delivered, valid, measured, within budget, and wrong
            assert final["outcome"] in labels and final["outcome"] == task.outcome(score) == _expected(env_id, score["checks"]), where
            seen.add(final["outcome"])
        elif not final["passed"]:
            assert final["outcome"] not in labels, where
    assert seen, f"{env_id}: no shortcut or ablation delivered a measured answer that fails"


@pytest.mark.parametrize("env_id", IDS)
def test_every_check_the_verifier_names_belongs_to_exactly_one_requirement(played, env_id):
    task, (_, labels, behind, _) = _task(env_id), FAMILIES[env_id]
    belongs = lambda k, stems: any(k == s or k.startswith(s + "_") for s in stems)   # noqa: E731
    scored = [(name, seed, score) for name, seed, _, score, _, _ in played(env_id) if score and score["checks"].get("deliverable_valid", True)]
    assert scored
    for name, seed, score in scored:
        own = [k for k in score["checks"] if k not in F.GATES + ("deliverable_valid",)]
        assert own == [k for k in own if k in {c for label in labels for c in behind[label]}], f"{env_id} {name} seed {seed}: {own}"
        for k in own:
            assert [label for label, stems in task.REQUIREMENTS if belongs(k, stems)] == [label for label in labels if k in behind[label]]
        # the order in which the verifier declares the requirements is the order of the labels
        order = [next(i for i, label in enumerate(labels) if k in behind[label]) for k in own]
        first = [i for n, i in enumerate(order) if i not in order[:n]]
        assert first == sorted(first) and len(first) == len(labels), f"{env_id}: {own}"


# ---------------------------------------------------------------------------------------------- unchanged
@pytest.mark.parametrize("env_id", IDS)
def test_pass_reward_and_progress_do_not_depend_on_the_label(played, env_id):
    """The same finished scores with the family's labels removed: only ``outcome`` differs (``check_failed``)."""
    task = _task(env_id)
    scores = [s for _, _, _, s, _, _ in played(env_id) if s and "outcome" in s]
    assert scores
    without = type("Task", (), {})
    for s in scores:
        raw = {k: v for k, v in s.items() if k not in ("progress", "outcome")}
        assert F.progress_of(raw) == s["progress"] and F.reward_of(s["passed"]) == s["reward"] and s["passed"] == all(s["checks"].values())
        assert F.outcome_of(raw, task) == s["outcome"]
        plain = F.outcome_of(raw, without)
        assert plain == (s["outcome"] if s["outcome"] in ("pass", "invalid_deliverable") + F.GATE_OUTCOMES else "check_failed")
