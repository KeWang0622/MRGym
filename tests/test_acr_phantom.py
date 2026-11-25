"""The two reference objects (``anatomy.acr``, ``anatomy.nist_t1``), the world kinds that carry them and the gradient
calibration error of the hidden system.

The dimensions asserted here are those of the sources named in ``anatomy.ACR`` / ``anatomy.NIST_T1``: the ACR phantom
test guidance (Table 1, sections 2.2 and 7.2) and Stupic 2021 with the reference list of the ISMRM reproducibility
challenge."""
import numpy as np
import pytest

from mrgym import qa
from mrgym.scanner import anatomy, console, kspace
from mrgym.scanner import world as W

SEED = "0f0e0d0c0b0a09080706050403020100"
QUIET = dict(tx_gain_dev=(0.0, 0.0), cf_offset_hz=0.0, sigma0=0.0)


def _image(spec, protocol):
    w = W.make_world(SEED, spec)
    rp = console.plan(protocol, w["console"])["realised"]
    return np.abs(kspace.Engine(w).acquire(rp, None)[0]), w


def test_the_numbers_of_the_guidance_are_the_ones_in_the_table():
    a = anatomy.ACR
    assert (a["diameter_mm"], a["length_mm"]) == (190.0, 148.0)                              # Table 1
    assert a["hole_mm"] == (1.1, 1.0, 0.9) and a["hole_pitch"] == 2.0 and a["holes"] == 4     # Table 1, section 2.2
    assert a["lcd_contrast"] == {8: 0.014, 9: 0.025, 10: 0.036, 11: 0.051}                    # section 7.2
    assert (a["lcd_spokes"], a["lcd_disks"], a["lcd_disk_mm"]) == (10, 3, (7.0, 1.5))
    assert (a["slices"] - 1) * (a["slice_mm"] + a["gap_mm"]) == 100.0                         # section 0.3
    d = anatomy.acr_lcd_disk_mm()
    assert d[0] == 7.0 and d[-1] == pytest.approx(1.5) and all(x > y for x, y in zip(d, d[1:]))


def test_layout_of_the_resolution_insert_and_of_the_low_contrast_section():
    lay = anatomy.acr_layout(dict(slice=1))
    assert lay["centre_mm"] == (0.0, 0.0) and lay["rotation_deg"] == 0.0 and not lay["disks"]
    for d in anatomy.ACR["hole_mm"]:
        ul = [h for h in lay["holes"] if h["d_mm"] == d and h["array"] == "UL"]
        lr = [h for h in lay["holes"] if h["d_mm"] == d and h["array"] == "LR"]
        assert len(ul) == 16 and len(lr) == 16                           # 4 x 4 each; the shared hole is in both lists
        shared = next(h for h in lr if h.get("shared"))
        corner = next(h for h in ul if (h["line"], h["k"]) == (3, 3))
        assert (shared["y_mm"], shared["x_mm"]) == pytest.approx((corner["y_mm"], corner["x_mm"]))
        row = sorted((h for h in ul if h["line"] == 0), key=lambda h: h["k"])
        assert np.diff([h["x_mm"] for h in row]) == pytest.approx(2 * d) and len({h["y_mm"] for h in row}) == 1
        col = sorted((h for h in lr if h["line"] == 2), key=lambda h: h["k"])
        assert np.diff([h["y_mm"] for h in col]) == pytest.approx(2 * d) and len({h["x_mm"] for h in col}) == 1
        rows = [min(h["x_mm"] for h in ul if h["line"] == r) for r in range(4)]
        assert np.diff(rows) == pytest.approx(anatomy.ACR["hole_stagger"] * d)   # each row a little to the right
    lay = anatomy.acr_layout(dict(slice=11))
    assert len(lay["disks"]) == 30 and not lay["holes"]
    first = [dk for dk in lay["disks"] if dk["spoke"] == 1]
    assert all(dk["y_mm"] < 0 and 0 < dk["x_mm"] < 8 for dk in first)    # 12 o'clock is up (-y), slightly to the right
    with pytest.raises(ValueError):
        anatomy.acr_layout(dict(slice=3))


def test_the_phantom_measures_190_and_148_mm_on_a_perfect_system():
    proto = dict(qa.SERIES["t1"])
    img, w = _image(dict(QUIET, kind="acr", b0_t=1.5, acr=dict(slice=7)), proto)
    assert w["kind"] == "acr" and w["names"] == ["fill"] and w["console"]["fov_mm"] == 250.0 and w["regions"] == []
    d = qa.diameters(img, (250.0, 250.0))["lengths_mm"]
    assert d == pytest.approx([190.0, 190.0], abs=0.3)
    img, _ = _image(dict(QUIET, kind="acr", b0_t=1.5, acr=dict(slice="sagittal")), dict(qa.SERIES["localizer"]))
    d = qa.diameters(img, (250.0, 250.0))["lengths_mm"]
    assert d == pytest.approx([148.0, 190.0], abs=0.3)                   # length along y, diameter along x


def test_a_gradient_calibration_error_scales_the_image_along_its_axis_and_is_absent_by_default():
    base = dict(QUIET, kind="acr", b0_t=1.5, n=640, acr=dict(slice=7, offset_mm=0.0, rotation_deg=0.0))
    proto = dict(qa.SERIES["t1"], matrix=128)
    plain, w0 = _image(base, proto)
    assert "grad_scale" not in w0["system"] and W.grad_scale_of(SEED, base) is None and W._warp(None, None) is None
    zero, w1 = _image(dict(base, grad_scale=dict(y=0.0, x=0.0)), proto)
    assert w1["system"]["grad_scale"] == (1.0, 1.0) and np.array_equal(zero, plain)
    img, w = _image(dict(base, grad_scale=dict(y=-0.02, x=0.03)), proto)
    assert w["system"]["grad_scale"] == pytest.approx((0.98, 1.03))
    d = qa.diameters(img, (250.0, 250.0))["lengths_mm"]
    assert d == pytest.approx([190.0 * 0.98, 190.0 * 1.03], abs=0.4)
    yy, xx = W.grid(w)                                                    # the grid point (y, x) shows the object point (y / s_y, x / s_x)
    assert yy[0, 0] == pytest.approx(-128.0 / 0.98) and xx[0, 0] == pytest.approx(-128.0 / 1.03)
    drawn = W.make_world(SEED, dict(base, grad_scale=dict(y=(-0.01, 0.01), x=None)), )["system"]["grad_scale"]
    assert abs(drawn[0] - 1.0) <= 0.01 and drawn[1] == 1.0 and drawn == W.grad_scale_of(SEED, dict(grad_scale=dict(y=(-0.01, 0.01), x=None)))


def test_worlds_of_the_other_kinds_do_not_see_the_new_keys(qa_world):
    assert "grad_scale" not in qa_world["system"] and "layout" not in qa_world
    again = W.make_world("000102030405060708090a0b0c0d0e0f", dict(kind="qa"))
    assert np.array_equal(again["weights"], qa_world["weights"]) and again["system"] == qa_world["system"]
    assert W.KINDS[:3] == ("brain", "abdomen", "qa") and W.GRID["qa"] == (640, 256.0) and W.EXAM_FOV_MM["qa"] == 240.0


def test_the_nist_t1_plate_carries_the_opened_reference_values_and_exists_at_3_t_only():
    w = W.make_world(SEED, dict(kind="nist_t1", b0_t=3.0))
    n = anatomy.NIST_T1
    assert [r["id"] for r in w["regions"]] == [f"T1-{i}" for i in range(1, 15)] and len(n["t1_ms"]) == 14
    truth = W.truth(w)
    assert [truth[f"T1-{i}"]["t1_ms"] for i in range(1, 15)] == list(n["t1_ms"])
    assert n["t1_ms"][11] == 44.53 and truth["T1-12"]["t2_ms"] == pytest.approx(31.86)       # Stupic 2021: Ni-12 at 3 T, 20 C
    (cy, cx), rot = w["layout"]["centre_mm"], w["layout"]["rotation_deg"]
    assert abs(cy) <= 1.0 and abs(cx) <= 1.0 and abs(rot) <= 0.3            # the plate is placed like the ACR phantom
    ring = [np.hypot(r["y_mm"] - cy, r["x_mm"] - cx) for r in w["regions"][:10]]
    assert ring == pytest.approx([50.0] * 10, abs=0.02)
    inner = [np.hypot(r["y_mm"] - cy, r["x_mm"] - cx) for r in w["regions"][10:]]
    assert inner == pytest.approx([20.0 * np.sqrt(2)] * 4, abs=0.02)                         # the corners of a 40 mm grid
    sides = [np.hypot(a["y_mm"] - b["y_mm"], a["x_mm"] - b["x_mm"]) for a, b in zip(w["regions"][10:], w["regions"][11:] + w["regions"][10:11])]
    assert sides == pytest.approx([40.0] * 4, abs=0.02)
    assert w["weights"].shape[0] == 15 and float(w["weights"][0][320, 320]) == pytest.approx(1.0)
    for r in w["regions"]:                                               # a region lies inside its sphere, away from the wall
        i, j = int(round(r["y_mm"] / 0.4)) + 320, int(round(r["x_mm"] / 0.4)) + 320
        assert float(w["weights"][w["names"].index(r["cls"])][i, j]) == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError, match="3 T only"):
        W.make_world(SEED, dict(kind="nist_t1", b0_t=1.5))
