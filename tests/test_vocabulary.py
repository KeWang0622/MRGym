"""Console vocabulary: vendor names of existing parameters, parameters of real consoles the scanner does not model,
vendor terms in another unit, and the computed presets."""
import math

import pytest

import mrgym
from mrgym import interface as I
from mrgym.scanner import console, presets, tissue
from mrgym.scanner import signal as S

EXAM = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=6.0, b0_t=1.5)
NOWRAP = "Scanner/NoWrap-v1"


def _plan(env, **arguments):
    return env.step({"name": "plan", "arguments": arguments})[0]


@pytest.fixture
def env():
    e = mrgym.make(NOWRAP, level=1, seed=0)
    e.reset()
    yield e
    e.close()


def test_one_protocol_in_three_dialects_is_one_realised_protocol(env):
    ours = _plan(env, sequence="ir", tr_ms=2000, te_ms=20, ti_ms=500, averages=2, slice_mm=4, bandwidth_hz_per_px=200, pe_dir="x", fat_sat=True)
    ge = _plan(env, sequence="ir", tr=2000, te=20, ti=500, nex=2, slice_thickness=4, pixel_bandwidth=200, phase_encoding_direction="x", fat_saturation=True)
    philips = _plan(env, sequence="ir", repetition_time_ms=2000, echo_time_ms=20, inversion_time_ms=500, nsa=2, slice_thickness_mm=4,
                    pixel_bandwidth_hz=200, fold_over_direction="x", fat_suppression=True)
    assert not ours.is_error and ours.data["realised"] == ge.data["realised"] == philips.data["realised"]
    assert {c["rule"] for c in ge.data["coercions"]} == {"C10"} and len(ge.data["coercions"]) == 8
    assert set(console.ALIASES.values()) <= set(console.parameter_names())
    assert not set(console.ALIASES) & set(console.parameter_names()) and not set(console.ALIASES) & set(console.NOT_SIMULATED)


@pytest.mark.parametrize("word", sorted(console.NOT_SIMULATED))
def test_a_real_console_parameter_the_scanner_lacks_is_named_as_such(env, word):
    code = console.validate(dict(sequence="se", tr_ms=500, **{word: 2}))[2][0]
    assert code["code"] == "NOT_SIMULATED" and code["parameter"] == word and "does not simulate" in code["message"] and "limits.md" in code["message"]
    obs = _plan(env, sequence="se", tr_ms=500, **{word: 2})
    err = obs.data["error"]
    assert obs.is_error and (err["type"], err["code"], err["parameter"]) == ("unknown_parameter", "NOT_SIMULATED", word) and err["fix"] == {word: None}
    assert I.error_class(obs.data) == "vocabulary" and "does not simulate" in obs.text and not I.counts_as_tool_call(obs.data)
    assert not _plan(env, **I.apply_fix(dict(sequence="se", tr_ms=500, **{word: 2}), err["fix"])).is_error


def test_the_words_of_the_memo_are_covered_and_a_misspelling_is_still_unknown(env):
    assert {"etl", "turbo_factor", "slices", "sense", "sat_band", "flow_comp", "b_value", "shim", "gating", "venc", "concatenations"} <= set(console.NOT_SIMULATED)
    assert len(console.NOT_SIMULATED) >= 20 and "NOT_SIMULATED" in I.CONSOLE_CODES
    err = _plan(env, sequence="se", tr_ms=500, matrx=64).data["error"]
    assert err["type"] == "unknown_parameter" and err.get("code") is None and err["did_you_mean"] == "matrix"
    assert console.validate(dict(sequence="se", tr_ms=500, matrx=64))[2][0]["code"] == "UNKNOWN_PARAMETER"


def test_a_vendor_term_in_another_unit_is_answered_with_the_conversion_and_never_applied(env):
    obs = _plan(env, sequence="se", tr_ms=500, matrix=256, rbw_khz=15.63)
    err = obs.data["error"]
    assert obs.is_error and err["code"] == "VENDOR_TERM" and err["did_you_mean"] == "bandwidth_hz_per_px"
    assert err["fix"] == {"rbw_khz": None, "bandwidth_hz_per_px": pytest.approx(2000 * 15.63 / 256, abs=1e-3)} and "never converted" in obs.text
    fixed = _plan(env, **I.apply_fix(dict(sequence="se", tr_ms=500, matrix=256, rbw_khz=15.63), err["fix"]))
    assert not fixed.is_error and fixed.data["realised"]["bandwidth_hz_per_px"] == pytest.approx(122.109)
    assert _plan(env, sequence="se", tr_ms=500, fov_phase_percent=75).data["error"]["fix"] == {"fov_phase_percent": None, "phase_fov": 0.75}
    assert _plan(env, sequence="se", tr_ms=500, no_phase_wrap=True).data["error"]["fix"] == {"no_phase_wrap": None, "phase_oversampling": 1.0}
    assert _plan(env, sequence="se", tr_ms=500, rbw_khz=15.63).data["error"]["fix"] == {"rbw_khz": None}      # no matrix written: no number
    wfs = _plan(env, sequence="se", tr_ms=500, wfs_px=1.5).data["error"]
    assert wfs["fix"] == {"wfs_px": None} and "217 Hz / wfs_px at 1.5 T" in wfs["message"]
    assert set(console.VENDOR_TERMS) & set(console.ALIASES) == set() and {v[0] for v in console.VENDOR_TERMS.values()} <= set(console.parameter_names())


@pytest.mark.parametrize("b0_t", [1.5, 3.0])
def test_presets_are_computed_from_the_tissue_table_and_the_console_accepts_them(b0_t):
    exam = dict(EXAM, b0_t=b0_t, b1rms_limit_ut={1.5: 6.0, 3.0: 3.0}[b0_t])
    rows = presets.presets(b0_t, exam)
    t = tissue.table(b0_t)
    assert set(rows) == {"t1w_se", "pdw_se", "t2w_se", "flair", "stir", "t1w_spgr", "in_phase", "opposed_phase", "acr_localizer", "acr_t1", "acr_t2"}
    plans = {k: console.plan(v["protocol"], exam) for k, v in rows.items()}
    assert all(p["accept"] for p in plans.values()), {k: p["violations"] for k, p in plans.items() if not p["accept"]}
    for name, cls in (("stir", "subcut_fat"), ("flair", "csf")):         # the preset nulls what its name says: under 5 % of the signal without inversion
        rp = plans[name]["realised"]
        ir = S.ir_se(math.pi / 2, rp["tr_ms"], rp["te_ms"], rp["ti_ms"], t[cls]["t1_ms"], 1.0)
        se = S.se(math.pi / 2, rp["tr_ms"], rp["te_ms"], t[cls]["t1_ms"])
        assert abs(ir / se) < 0.05, (name, ir / se)
        assert rp["ti_ms"] == pytest.approx(presets.null_ti_ms(t[cls]["t1_ms"], rp["tr_ms"]), abs=0.5)
    df = 3.4e-6 * 42.577478518e6 * b0_t
    cycles = {k: rows[k]["protocol"]["te_ms"] * df / 1e3 for k in ("in_phase", "opposed_phase")}     # cycles of water-fat phase at the echo
    assert abs(cycles["in_phase"] - round(cycles["in_phase"])) < 0.01 and abs(cycles["opposed_phase"] % 1 - 0.5) < 0.01
    assert all(rows[k]["protocol"]["te_ms"] >= plans[k]["timing"]["te_min_ms"] for k in cycles)
    assert rows["t1w_spgr"]["protocol"]["flip_deg"] == pytest.approx(math.degrees(math.acos(math.exp(-20.0 / t["wm"]["t1_ms"]))), abs=0.06)
    assert rows["t1w_se"]["protocol"]["tr_ms"] < rows["pdw_se"]["protocol"]["tr_ms"] == rows["t2w_se"]["protocol"]["tr_ms"]


def test_presets_differ_with_the_field_and_are_served_by_scanner_info(env):
    assert presets.presets(1.5)["stir"]["protocol"]["ti_ms"] < presets.presets(3.0)["stir"]["protocol"]["ti_ms"]
    obs = env.step({"name": "scanner_info", "arguments": {"topic": "presets"}})[0]
    field = env.hidden_world()["b0_t"]
    assert not obs.is_error and obs.data["presets"] == presets.presets(field, env.hidden_world()["console"]) and "stir:" in obs.text and "not an answer" in obs.text
    assert not env.step({"name": "plan", "arguments": obs.data["presets"]["flair"]["protocol"]})[0].is_error
