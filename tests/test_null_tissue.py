"""``Scanner/NullTissue-v0`` beyond its scoring table (``tests/scoring/null_tissue.py``): the edge of the pass rule, the
verifier's own null search, and the target protocol. The expected numbers come from the pulse-chain iteration of the
table module, not from the task module."""
import math
import re

import numpy as np
import pytest

import scoring
from mrgym.envs.null_tissue import task
from mrgym.scanner import console
from scoring import null_tissue as table

CASE = lambda key="L1": scoring.case(table, key)       # noqa: E731


def test_target_protocol_is_the_stated_scan_with_the_delivered_inversion_time():
    case = CASE()
    assert task.target_protocol(case.cfg["task"], 412.5) == dict(matrix=128, bandwidth_hz_per_px=300.0, sequence="ir", tr_ms=6000.0, ti_ms=412.5)
    assert case.cfg["task"]["target_scan"] == dict(matrix=128, bandwidth_hz_per_px=300.0, sequence="ir", tr_ms=6000.0,
                                                   te_ms="console minimum", averages=1, slice_mm=5.0)
    rp = console.plan(task.target_protocol(case.cfg["task"], 412.5), case.exam)["realised"]
    assert rp["te_ms"] == table.TE_MS                                    # the echo time the table's chain uses
    # ... and the pulse it simulates: the deliverable cannot change the excitation, so the verifier scores this one
    assert (rp["excitation"], rp["rf_pulse"], rp["rf_tbw"]) == ("selective", "normal", table.TBW)
    assert set(task.submit_schema(case.cfg)["properties"]) == {"ti_ms"}


def test_an_instance_without_a_task_id_gets_a_random_token():
    a, b = (task.make_instance(CASE().seed_hex, 1)["task_id"] for _ in range(2))
    assert re.fullmatch(r"nulltissue-[0-9a-f]{12}", a) and re.fullmatch(r"nulltissue-[0-9a-f]{12}", b) and a != b


def test_the_pass_threshold_is_inclusive(monkeypatch):
    case = CASE()
    answer = table.at()(case)
    monkeypatch.setattr(task, "null_ratio", lambda cfg, region, ti, world, engine: dict(ratio=0.05, sign=1.0))
    s = task.score(case.cfg, answer, world=case.world, engine=case.engine)
    assert s["passed"] is True and s["diagnostic"] == "" and s["error"] == 0.05
    over = 0.05 * (1 + 1e-12)
    monkeypatch.setattr(task, "null_ratio", lambda cfg, region, ti, world, engine: dict(ratio=over if region == "R2" else 0.05, sign=1.0))
    s = task.score(case.cfg, answer, world=case.world, engine=case.engine)
    assert s["passed"] is False and s["checks"] == dict(deliverable_valid=True, nulled_R1=True, nulled_R2=False)
    assert s["diagnostic"] == "R2 keeps 5.0 %"


def test_the_verifier_passes_each_region_its_own_inversion_time(monkeypatch):
    case = CASE()
    seen = []
    monkeypatch.setattr(task, "null_ratio", lambda cfg, region, ti, world, engine: seen.append((region, ti, world, engine)) or dict(ratio=0.0, sign=1.0))
    task.score(case.cfg, {"ti_ms": {"R2": 700, "R1": 300.5}}, world=case.world, engine=case.engine)
    assert seen == [("R2", 700.0, case.world, case.engine), ("R1", 300.5, case.world, case.engine)]
    assert all(type(ti) is float for _, ti, _, _ in seen)


def _receive_phase(case, region):
    """Phase of the receive profile at the centre of a region (hidden world, nearest master-grid point)."""
    r = next(x for x in case.cfg["task"]["regions"] if x["id"] == region)
    n, d = case.world["n"], case.world["extent_mm"] / case.world["n"]
    return float(np.angle(case.world["fields"]["rx"][int(round(r["y_mm"] / d)) + n // 2, int(round(r["x_mm"] / d)) + n // 2]))


def test_null_ratio_is_signed_like_the_recovery_curve_whatever_the_receive_phase():
    """Negative before the null, positive after it. The receive coil gives every region its own phase; the sign must not
    depend on it, so one region is taken where cos(2 phase) < 0 (where a product instead of a quotient would flip it)."""
    regions = [(c, i) for c in (CASE("L1"), CASE("L2"), CASE("L3")) for i in table.ids(c)]
    flipping = next((c, i) for c, i in regions if math.cos(2 * _receive_phase(c, i)) < -0.3)
    plain = next((c, i) for c, i in regions if math.cos(2 * _receive_phase(c, i)) > 0.3)
    for case, region in (flipping, plain):
        null = table.null_ms(case, region)
        for factor in (0.8, 1.25):
            got = task.null_ratio(case.cfg, region, round(null * factor, 2), case.world, case.engine)
            want = table.ratio(case, region, round(null * factor, 2))
            assert set(got) == {"ratio", "sign"} and type(got["ratio"]) is float and type(got["sign"]) is float
            assert got["sign"] == math.copysign(1.0, want) == (1.0 if factor > 1 else -1.0)
            assert got["ratio"] == pytest.approx(abs(want), abs=table.RATIO_ATOL)


def test_the_null_search_bisects_between_the_console_limits(monkeypatch):
    case = CASE("L2")                                                    # repetition time 2160 ms
    calls = []

    def fake(null):
        def null_ratio(cfg, region, ti, world, engine):
            calls.append(ti)
            return dict(ratio=abs(ti - null), sign=1.0 if ti > null else -1.0)
        return null_ratio
    lo, hi = 6.04 + 0.01, 2160.0 - table.TE_MS - 20.0                    # shortest inversion time + 0.01; TR - TE - 20
    for null in (6.5, 300.0, 1234.567, 2131.0):
        calls.clear()
        monkeypatch.setattr(task, "null_ratio", fake(null))
        got = task.true_null_ms(case.cfg, "R1", case.world, case.engine)
        assert got == pytest.approx(null, abs=(hi - lo) / 2 ** 18 + 0.006) and got == round(got, 2)
        assert calls[:2] == [lo, hi] and len(calls) == 20 and all(lo <= t <= hi for t in calls)
    for null in (5.0, 2135.0):                                           # no sign change inside the limits: no null
        monkeypatch.setattr(task, "null_ratio", fake(null))
        assert math.isnan(task.true_null_ms(case.cfg, "R1", case.world, case.engine))


def test_the_null_search_agrees_with_the_pulse_chain():
    case = CASE("L1")
    assert task.true_null_ms(case.cfg, "R1", case.world, case.engine) == pytest.approx(table.null_ms(case, "R1"), abs=1.0)


# ---------------------------------------------------------------------------------------------- progress, outcome, hints
def test_progress_orders_the_classes_by_the_number_of_regions_nulled():
    """Module docstring: 0.5 (k + c) / n with c the mean closeness ln(0.5 / ratio) / ln(0.5 / eps) of the regions not nulled."""
    eps, c = 0.05, lambda v: math.log(0.5 / v) / math.log(0.5 / 0.05)                     # noqa: E731
    assert task.progress(dict(R1=0.01, R2=0.05), eps) == 1.0                              # at the threshold counts as nulled
    assert task.progress(dict(R1=0.01, R2=0.2), eps) == pytest.approx(0.5 * (1 + c(0.2)) / 2, abs=1e-12)
    assert task.progress(dict(R1=0.2, R2=0.01), eps) == task.progress(dict(R1=0.01, R2=0.2), eps)
    assert task.progress(dict(R1=0.1, R2=0.3), eps) == pytest.approx(0.5 * ((c(0.1) + c(0.3)) / 2) / 2, abs=1e-12)
    assert task.progress(dict(R1=0.5, R2=0.9), eps) == 0.0                                # half the signal or more: nothing
    assert task.progress(dict(R1=0.01, R2=0.7), eps) == 0.25                              # one of two nulled, the other far
    assert task.progress(dict(R1=0.01, R2=0.02, R3=0.4), eps) == pytest.approx(0.5 * (2 + c(0.4)) / 3, abs=1e-12)
    just_over = task.progress(dict(R1=0.01, R2=0.05 * (1 + 1e-9)), eps)                   # closeness capped at 0.999
    assert just_over == pytest.approx(0.5 * (1 + 0.999) / 2, abs=1e-12) and just_over < 0.5
    best_none, worst_one = task.progress(dict(R1=0.0501, R2=0.0501), eps), task.progress(dict(R1=0.0, R2=0.9), eps)
    assert best_none < worst_one < just_over < 1.0                                        # a class never overtakes the next
    assert (task.FAR_RATIO, task.OUTCOMES) == (0.5, ("none_nulled", "some_nulled"))


def test_outcome_says_whether_any_region_was_nulled():
    assert task.outcome(dict(checks=dict(deliverable_valid=True, nulled_R1=True, nulled_R2=False))) == "some_nulled"
    assert task.outcome(dict(checks=dict(deliverable_valid=True, nulled_R1=False, nulled_R2=False))) == "none_nulled"
    assert task.outcome(dict(checks=dict(deliverable_valid=True, measured=True))) == "none_nulled"     # only its own checks count


def test_potential_counts_distinct_inversion_times_probed_on_regions_up_to_three():
    ir = lambda ti, rois=("R1",), seq="ir", kind="scan": dict(type=kind, realised=dict(sequence=seq, ti_ms=ti), rois=list(rois))   # noqa: E731
    cfg = CASE().cfg
    assert task.potential(cfg, []) == 0.0
    assert task.potential(cfg, [ir(100.0)]) == pytest.approx(0.1)
    assert task.potential(cfg, [ir(100.0), ir(100.04), ir(400.0)]) == pytest.approx(0.2)              # the same time to 0.1 ms counts once
    assert task.potential(cfg, [ir(100.0), ir(100.3)]) == pytest.approx(0.2)                          # 0.3 ms apart: two times
    assert task.potential(cfg, [ir(100.0), ir(400.0), ir(900.0), ir(1500.0), ir(2500.0)]) == pytest.approx(0.3)   # three are enough
    assert task.potential(cfg, [ir(100.0, rois=()), ir(400.0, seq="se"), ir(900.0, kind="plan")]) == 0.0           # not a probe


def test_prefill_acquires_the_first_probe_only_where_the_level_says_so():
    class Scanner:
        calls = []

        def scan(self, protocol, rois=None):
            self.calls.append((protocol, rois))
            return dict(scan_index=0)
    level0, level1 = CASE("L0").cfg, CASE("L1").cfg
    assert level0["task"]["hints"] == ["rule", "first_measurement"] and "hints" not in level1["task"]
    assert task.prefill(level1, Scanner()) is None and Scanner.calls == []
    assert task.prefill(level0, Scanner()) == [dict(scan_index=0)]
    assert Scanner.calls == [(task.FIRST_PROBE, level0["task"]["regions"])] and Scanner.calls[0][0] is not task.FIRST_PROBE
    assert task.FIRST_PROBE == dict(sequence="ir", matrix=64, partial_fourier=0.6, bandwidth_hz_per_px=300.0, tr_ms=800.0, ti_ms=740.0)
    assert task.statement(level1) == task.statement(level1, [dict(scan_index=0)]) and task.hint_text(level1) == ""
    only_rule = dict(level0, task=dict(level0["task"], hints=["rule"]))
    assert task.hint_text(only_rule, [dict()]) == task.hint_text(level0) and task.hint_text(level0).startswith("How to do it")


def test_the_hints_state_the_first_probe_as_it_was_acquired():
    cfg = CASE("L0").cfg
    given = [dict(scan_index=4, cost_s=12.3456, realised=dict(tr_ms=800.0, ti_ms=740.0, te_ms=8.4, matrix=64),
                  roi_readouts=[dict(id="R1", complex_mean_magnitude=[1.2345678, 9.0], complex_mean_phase_rad=[0.123456, 9.0]),
                                dict(id="R2", complex_mean_magnitude=[0.5, 9.0], complex_mean_phase_rad=[-2.0, 9.0])]),
             dict(scan_index=99, cost_s=99.0, realised=dict(tr_ms=1.0, ti_ms=1.0, te_ms=1.0, matrix=1), roi_readouts=[])]
    text = task.hint_text(cfg, given)
    rule, probe = text.split("\n\n")
    assert rule == task.hint_text(cfg) and len(text.split("\n\n")) == 2
    assert probe == ("The first probe is already acquired: scan 4 (`ir`, tr_ms 800, ti_ms 740, te_ms 8.4, matrix 64),\n"
                     "12.35 s of the budget. Its read-out (the longest inversion time of that repetition time: every signal is positive):\n"
                     "  R1   |complex mean| 1.23457   phase_rad 0.1235\n"
                     "  R2   |complex mean| 0.5   phase_rad -2.0000")
    assert task.statement(cfg, given) == task.statement(cfg, None) + "\n\n" + probe
    assert task.statement(cfg).endswith("An answer in the wrong unit is wrong.\n\n" + rule)
