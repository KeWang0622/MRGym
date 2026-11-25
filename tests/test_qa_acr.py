"""``mrgym qa acr``: the measurements on synthetic images with known answers, the healthy system against the limits of
the guidance, and the stored fault x test matrix."""
import json

import numpy as np
import pytest

import mrgym
from mrgym import cli, qa

FOV = (250.0, 250.0)


def _disc(n=256, radius=95.0, centre=(0.0, 0.0), value=100.0):
    y, x = qa.positions((n, n), FOV)
    r = np.hypot(y[:, None] - centre[0], x[None, :] - centre[1])
    return value * np.clip((radius - r) / (FOV[0] / n) + 0.5, 0.0, 1.0), y, x       # an edge one pixel wide: half at the radius


# ---------------------------------------------------------------------------------------------- the measurements
def test_the_limits_are_those_of_the_guidance():
    lim = qa.LIMITS
    assert lim["geometry_mm"] == 3.0 and lim["resolution_mm"] == 1.0 and lim["ghosting_ratio"] == 0.030
    assert lim["piu_percent"] == {1.5: 85.0, 3.0: 80.0} and lim["piu_should_percent"] == {1.5: 87.5, 3.0: 82.0}
    assert lim["lcd_spokes"] == {1.5: dict(t1=30, t2=25), 3.0: dict(t1=37, t2=37)}
    assert lim["slice_thickness_mm"] == (5.0, 0.7) and lim["slice_position_mm"] == 5.0
    assert qa.GUIDANCE_SCAN_S == dict(localizer=56.0, t1=136.0, t2=536.0)                # 0:56, 2:16, 8:56
    assert {k: (v["tr_ms"], v["te_ms"], v["fov_mm"], v["matrix"], v["slice_mm"], v["averages"]) for k, v in qa.SERIES.items()} == dict(
        localizer=(200.0, 20.0, 250.0, 256, 10.0, 1), t1=(500.0, 20.0, 250.0, 256, 5.0, 1), t2=(2000.0, 80.0, 250.0, 256, 5.0, 1))
    assert set(qa.NOT_SIMULATED) == {"slice_thickness", "slice_position"} and len(qa.TESTS) == 8


def test_diameter_and_centre_of_a_disc():
    img, _, _ = _disc(centre=(1.7, -2.3), radius=95.0)
    d = qa.diameters(img, FOV, angles=(90.0, 0.0, 45.0, 135.0))
    assert d["lengths_mm"] == pytest.approx([190.0] * 4, abs=0.15) and d["centre_mm"] == pytest.approx((1.7, -2.3), abs=0.1)
    assert d["level"] == pytest.approx(50.0)


def test_uniformity_of_a_linear_shading_is_the_analytic_value():
    img, y, x = _disc()
    tilt = 0.2                                                           # signal 100 (1 + tilt x / 95) inside the disc
    img = img * (1.0 + tilt * x[None, :] / 95.0)
    u = qa.uniformity(img, FOV)
    reach = (np.sqrt(20000.0 / np.pi) - np.sqrt(100.0 / np.pi)) / 95.0  # the 1 cm^2 circle stays inside the 200 cm^2 circle
    want = 100.0 * (1.0 - tilt * reach)                                  # (high - low) / (high + low) = tilt x reach
    assert u["piu_percent"] == pytest.approx(want, abs=0.25) and u["high_at_mm"][1] > 70 and u["low_at_mm"][1] < -70
    assert qa.uniformity(_disc()[0], FOV)["piu_percent"] == pytest.approx(100.0, abs=1e-6)


def test_ghosting_ratio_of_a_known_background():
    img, y, x = _disc()
    rois = qa.background_rois(FOV)
    assert {k: (r[1] - r[0], r[3] - r[2]) for k, r in rois.items()} == dict(top=(16.0, 63.0), bottom=(16.0, 63.0), left=(63.0, 16.0), right=(63.0, 16.0))
    for name, value in (("top", 6.0), ("bottom", 4.0), ("left", 1.0), ("right", 1.0)):
        r = rois[name]
        img[np.ix_((y >= r[0]) & (y <= r[1]), (x >= r[2]) & (x <= r[3]))] = value
    g = qa.ghosting(img, FOV)
    assert g["large_mean"] == pytest.approx(100.0) and g["ratio"] == pytest.approx(abs((6.0 + 4.0) - (1.0 + 1.0)) / 200.0)


def test_the_line_detector_sees_a_line_and_not_the_edge_of_a_ghost():
    rng = np.random.default_rng(0)
    noise = np.abs(rng.normal(0, 1, (256, 256)) + 1j * rng.normal(0, 1, (256, 256)))
    assert qa.artifact_line(noise, FOV)["z"] < qa.LIMITS["artifact_z"]
    line = noise.copy()
    line[:, 140] += 3.0
    assert qa.artifact_line(line, FOV)["z"] > qa.LIMITS["artifact_z"]
    step = noise.copy()
    step[:, :100] += 3.0                                                 # the edge of a broad ghost: above one neighbour only
    assert qa.artifact_line(step, FOV)["z"] < qa.LIMITS["artifact_z"]
    assert qa.noise_sd(noise, FOV) == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------------------------- the series
@pytest.fixture(scope="module")
def healthy():
    return qa.run_acr(1.5, 0)


def test_a_healthy_system_passes_every_simulated_test_at_1_5_t(healthy):
    assert healthy["passed"] == healthy["simulated"] == 6 and healthy["failed"] == [] and healthy["engine"] == mrgym.ENGINE_ID
    rows = {r["test"]: r for r in healthy["tests"]}
    assert list(rows) == list(qa.TESTS)
    assert rows["slice_thickness"]["passed"] is None and rows["slice_position"]["passed"] is None
    lengths = rows["geometric_accuracy"]["detail"]["lengths_mm"]
    assert len(lengths) == 7 and lengths["localizer length"] == pytest.approx(148.0, abs=0.5)
    assert all(v == pytest.approx(190.0, abs=0.5) for k, v in lengths.items() if "top-bottom" in k or "left-right" in k)
    res = rows["spatial_resolution"]["detail"]
    for s in ("t1", "t2"):                                               # 1.0 mm is resolved, 0.9 mm (below two pixels per pitch) is not
        assert [res[s][k]["resolved_mm"] for k in ("left_right", "top_bottom")] == [1.0, 1.0]
        assert all(res[s][k]["depth"]["0.9"] < qa.RESOLVED_DEPTH < res[s][k]["depth"]["1.0"] for k in ("left_right", "top_bottom"))
    assert {k: v["ratio"] for k, v in healthy["series"].items()} == pytest.approx(dict(localizer=1.0, t1=1.0, t2=1.0), abs=0.05)
    text = qa.text_table(healthy)
    assert "PASS 6 / 6 simulated tests" in text and text.count("not simulated") >= 4 and "t1 2:11.5 (2:16.0" in text


def test_one_fault_fails_the_test_the_guidance_names_and_no_other():
    rep = qa.run_acr(1.5, 0, "gradient_x", series=("localizer", "t1"))
    assert rep["failed"] == ["geometric_accuracy"] == list(qa.FAULTS["gradient_x"]["fails"])
    worst = next(r for r in rep["tests"] if r["test"] == "geometric_accuracy")["detail"]["lengths_mm"]
    assert worst["slice 1 left-right"] == pytest.approx(190.0 * 1.025, abs=0.7) and worst["slice 1 top-bottom"] == pytest.approx(190.0, abs=0.5)
    with pytest.raises(ValueError, match="unknown fault"):
        qa.system_spec(1.5, 0, "poltergeist")


def test_the_stored_fault_matrix_is_current_and_every_row_is_as_expected():
    m = qa.stored_matrix()
    assert m is not None, "run `mrgym qa acr --matrix --write`"
    assert m["engine"] == mrgym.ENGINE_ID and m["schema"] == qa.SCHEMA and m["all_as_expected"]
    rows = {(r["field_t"], r["fault"]): r for r in m["rows"]}
    assert set(rows) == {(f, name) for f in (1.5, 3.0) for name in (None, *qa.FAULTS)}
    for (field, name), r in rows.items():
        assert r["failed"] == r["expected"] == sorted(qa.FAULTS[name]["fails"] if name else []), (field, name)
    assert set(qa.MATRIX_FAULTS) <= set(qa.FAULTS) and len(qa.MATRIX_FAULTS) == 7
    for name in qa.MATRIX_FAULTS:                                        # each of the seven mechanisms fails something, at both fields
        assert rows[(1.5, name)]["failed"] and rows[(3.0, name)]["failed"]
    assert "every row as expected: True" in qa.matrix_table(m)


def test_the_page_under_docs_is_generated_from_the_stored_matrix():
    import pathlib
    page = pathlib.Path(__file__).resolve().parents[1] / "docs" / "physics" / "phantoms.md"
    assert page.read_text(encoding="utf-8") == qa.doc_page(qa.stored_matrix()), "regenerate: mrgym qa acr --doc docs/physics/phantoms.md"


def test_the_command_line(capsys):
    assert cli.main(["qa", "acr", "--field", "3", "--fault", "receiver_noise", "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["field_t"] == 3.0 and rep["failed"] == ["low_contrast"] and rep["fault"] == "receiver_noise"
