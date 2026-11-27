"""The volume worlds ``head3d`` and ``qa3d`` (``mrgym.scanner.volume``): the acceptance tests of work package W1.

The acceptance list of the package: plane = points <= 1e-4; weights sum <= 1; same bytes in two threads; two
planes agree on their common line; lattice volume of a sphere and a cylinder within 0.5 %; an axial plane of
``qa3d`` through its uniform part is the 2D ``qa`` drawing rule; build <= 1.2 s; one plane <= 100 ms on the master
grid; memory reported. The numbers are computed by ``mrgym.scanner.volume_check`` (the same functions write
``docs/physics/volume.md``); the limits are ``volume_check.LIMITS``. No scan is formed of a volume world here.
"""
import concurrent.futures
import json
import os
import random
import sys

import numpy as np
import pytest

import golden_cases as G
from mrgym.scanner import anatomy, kspace, volume
from mrgym.scanner import volume_check as VC
from mrgym.scanner import world as W

LIM = VC.LIMITS
SEED = VC.SEEDS[0]


@pytest.fixture(scope="module")
def head():
    return VC.make("head3d")


@pytest.fixture(scope="module")
def qa():
    return VC.make("qa3d")


def _world(request, kind):
    return request.getfixturevalue("head" if kind == "head3d" else "qa")


# ---------------------------------------------------------------------------------------------- the world dict
@pytest.mark.parametrize("kind", volume.KINDS)
def test_a_volume_world_is_a_world_with_a_volume_and_a_reference_plane(request, kind):
    w = _world(request, kind)
    vol = w["volume"]
    assert kind in W.KINDS and w["kind"] == kind and isinstance(vol, volume.Volume)
    assert w["weights"].dtype == np.float32 and w["weights"].shape == (len(w["names"]), w["n"], w["n"]) and (w["n"], w["extent_mm"]) == volume.GRID[kind]
    assert tuple(w["names"]) == vol.names and vol.matter == vol.names + vol.solid and set(w["tissues"]) == set(vol.names)
    assert np.array_equal(w["weights"], vol.eval_plane(volume.Frame.plane("axial"), dtype=np.float64))      # the reference plane: axial, z = 0
    assert w["masks"]["body"].dtype == bool and w["masks"]["body"].sum() > 20000 and w["masks"]["brain"].sum() > 5000
    assert w["console"] == dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0) and w["schema"] == W.SCHEMA
    assert w["source"] == dict(kind="seed", seed=SEED, spec=w["source"]["spec"]) and w["regions"] is vol.regions
    assert set(vol.susceptibility_ppm) == set(vol.matter) | {"air"} and vol.susceptibility_ppm["air"] == pytest.approx(W.DCHI_AIR * 1e6)
    other = VC.make(kind, SEED[::-1])
    assert not np.array_equal(w["weights"], other["weights"]) and other["system"] != w["system"] and other["volume"].layout != vol.layout


@pytest.mark.parametrize("kind", volume.KINDS)
def test_no_scan_is_formed_of_a_volume_world_yet(request, kind):
    """The world has no fields (they come from its lattice, another package), so the 2D engine does not take it."""
    w = _world(request, kind)
    assert "fields" not in w
    with pytest.raises(KeyError, match="fields"):
        kspace.Engine(w)
    for key, value in (("gnl", 0.01), ("motion", [dict(cls="*")]), ("dbdz", dict(n=1)), ("grad_scale", dict(y=0.01))):
        with pytest.raises(ValueError, match="not defined for a volume"):
            W.make_world(SEED, {"kind": kind, key: value})
    with pytest.raises(ValueError, match="1.5 or 3.0"):
        W.make_world(SEED, dict(kind=kind, b0_t=7.0))


def test_layout_only_gives_the_same_truth_without_arrays(head, qa):
    for w in (head, qa):
        light = W.make_world(SEED, w["source"]["spec"], layout_only=True)
        assert "weights" not in light and light["regions"] == w["regions"] and light["layout"] == w["layout"] and light["system"] == w["system"]
        assert W.truth(light) == W.truth(w) and light["volume"].truth() == w["volume"].truth()


# ---------------------------------------------------------------------------------------------- frames
def test_the_axial_frame_is_the_grid_of_the_2d_worlds():
    n, ext = 64, 256.0
    c = volume.grid_axis(n, ext)
    for rot in (0.0, 30.0, -75.0):
        y, x, z = volume.Frame.plane("axial", rot_deg=rot).points(c, c)
        v, u = anatomy.axes(n, ext, rot)
        assert np.abs(y - v).max() < 1e-12 and np.abs(x - u).max() < 1e-12 and not z.any()
    y, x, z = volume.Frame.plane("axial").points(c, c)
    v, u = anatomy.axes(n, ext)
    assert np.array_equal(y, v) and np.array_equal(x, u)                 # upright: the same numbers, bit for bit


def test_the_three_standard_planes_and_their_signs():
    """Axes: +y posterior, +x left, +z superior; vectors are (y, x, z)."""
    ax, co, sa = (volume.Frame.plane(k, 20.0) for k in ("axial", "coronal", "sagittal"))
    assert ax.row.tolist() == [1, 0, 0] and ax.col.tolist() == [0, 1, 0] and ax.normal.tolist() == [0, 0, 1] and ax.origin.tolist() == [0, 0, 20]
    assert co.row.tolist() == [0, 0, -1] and co.col.tolist() == [0, 1, 0] and co.normal.tolist() == [1, 0, 0] and co.origin.tolist() == [20, 0, 0]
    assert sa.row.tolist() == [0, 0, -1] and sa.col.tolist() == [1, 0, 0] and sa.normal.tolist() == [0, 1, 0] and sa.origin.tolist() == [0, 20, 0]
    f = volume.Frame.plane("axial", 5.0, tilt_deg=(20.0, -15.0), rot_deg=40.0)
    m = np.stack([f.row, f.col, f.normal])
    assert np.abs(m @ m.T - np.eye(3)).max() < 1e-12 and abs(np.linalg.det(m) - 1.0) < 1e-12
    g = f.shifted(7.5)
    assert np.allclose(g.origin - f.origin, 7.5 * f.normal) and g.key() != f.key() and f.key() == volume.Frame.plane("axial", 5.0, (20.0, -15.0), 40.0).key()
    y, x, z = f.points([3.0, -2.0], [1.0, 4.0, 9.0])
    v, u, d = f.coords(y, x, z)
    assert np.allclose(v, [[3.0] * 3, [-2.0] * 3]) and np.allclose(u, [[1.0, 4.0, 9.0]] * 2) and np.abs(d).max() < 1e-12
    with pytest.raises(ValueError, match="perpendicular"):
        volume.Frame(row=(1, 0, 0), col=(1, 1, 0))
    with pytest.raises(ValueError):
        volume.Frame.plane("oblique")


def test_the_head_is_the_right_way_up(head):
    """The top of the head is at +z, the neck at -z, the eyes and the frontal sinus at -y (anterior), left is +x."""
    vol, lay = head["volume"], head["layout"]
    body = lambda y, x, z: float(vol.eval_points(y, x, z).sum())        # noqa: E731
    assert body(0, 0, lay["zc"] + lay["az"] - 15) > 0.99 and body(0, 0, lay["zc"] + lay["az"] + 15) < 1e-6
    assert body(0, 30, -125.0) > 0.99                                    # the neck leaves the imaged cube downwards
    eyes = {s["name"]: s["c_mm"] for s in lay["eyes"]}
    assert eyes["eye_L"][1] > 20 and eyes["eye_R"][1] < -20 and all(c[0] < -35 and c[2] < lay["zc"] - 40 for c in eyes.values())
    assert next(s for s in lay["air"] if s["name"] == "frontal_sinus")["c_mm"][0] < -60
    assert next(s for s in lay["air"] if s["name"] == "mastoid_L")["c_mm"][1] > 40
    csf = vol.names.index("csf")
    assert all(vol.eval_points(*c)[csf] > 0.9 for c in eyes.values())    # the eyes are where the layout says (vitreous: the CSF class)


# ---------------------------------------------------------------------------------------------- acceptance: evaluation
@pytest.mark.parametrize("kind", volume.KINDS)
@pytest.mark.parametrize("seed", VC.SEEDS[:3])
def test_a_plane_equals_the_recipe_at_its_points(kind, seed):
    vol = VC.make(kind, seed)["volume"]
    d = VC.plane_against_points(vol)
    print(f"\n{kind} {seed[-6:]}: plane against points max {d['max']:.2e} rms {d['rms']:.2e}")
    assert d["max"] <= LIM["plane_points"]
    for frame in (volume.Frame.plane("sagittal", 11.0), volume.Frame.plane("coronal", -20.0, rot_deg=25.0)):
        assert VC.plane_against_points(vol, frame, n=128)["max"] <= LIM["plane_points"]


def test_the_master_grid_plane_equals_the_recipe_too(head):
    """Grid 640 (strips of rows, the float32 path) on one axial and one sagittal plane."""
    for frame in (volume.Frame.plane("axial", 12.0), volume.Frame.plane("sagittal", -9.0)):
        assert VC.plane_against_points(head["volume"], frame, n=640)["max"] <= LIM["plane_points"]


@pytest.mark.parametrize("kind", volume.KINDS)
def test_weights_sum_to_one_in_matter_and_never_exceed_one(request, kind):
    s = VC.weight_sums(_world(request, kind)["volume"])
    print(f"\n{kind}: largest sum - 1 = {s['max_sum'] - 1:.2e}, smallest weight {s['min_weight']:.2e}, inside |sum - 1| <= {s['inside_off_one']:.2e}")
    assert s["max_sum"] <= 1 + LIM["weight_sum"] and s["min_weight"] >= -LIM["weight_sum"] and s["inside_off_one"] <= LIM["weight_sum"]


@pytest.mark.parametrize("seed", range(8))
def test_the_head_is_a_partition_for_other_seeds_and_options(seed):
    """The construction is a partition: more seeds, with an insert and a thinner edge, in float64 at points."""
    spec = dict(lesion="ms" if seed % 2 else None, edge_mm=0.25 if seed % 3 == 0 else None,
                volume=dict(vessels=seed % 4, inserts=[dict(y_mm=10.0, x_mm=25.0, z_mm=30.0, radius_mm=6.0, tissue="tumour")] if seed % 2 == 0 else None))
    vol = VC.make("head3d", "%032x" % (seed * 7919 + 5), **spec)["volume"]
    rng = np.random.default_rng(seed)
    p = rng.uniform(-120, 120, (4000, 3))
    w = vol.eval_points(p[:, 0], p[:, 1], p[:, 2], solid=True)
    assert w.min() >= 0 and w.sum(0).max() <= 1 + 1e-12


@pytest.mark.parametrize("kind", volume.KINDS)
def test_two_planes_agree_on_their_common_line(request, kind):
    d = VC.common_line(_world(request, kind)["volume"])
    print(f"\n{kind}: two planes on their common line differ by at most {d:.2e}")
    assert d <= LIM["common_line"]


def test_lattice_volume_of_a_ball_and_a_cylinder():
    rows = VC.lattice_volumes()
    for r in rows:
        print(f"\n{r['shape']} [{r['sampling']}]: lattice {r['lattice_mm3']:.0f} mm^3, {100 * r['rel_sharp']:+.3f} % of the sharp form, {100 * r['rel_soft']:+.3f} % of the soft form")
    mean = [r for r in rows if r["sampling"].startswith("mean")]
    point = [r for r in rows if r["sampling"] == "voxel centre"]
    assert len(mean) == len(point) == 4
    for r in mean:                                                       # partial volume per voxel: every shape, against the object that is drawn
        assert abs(r["rel_soft"]) <= LIM["lattice_volume"], r
    for r in mean[:2]:                                                   # radius 24 mm: against the sharp-edged closed form as well
        assert abs(r["rel_sharp"]) <= LIM["lattice_volume"], r
    for r in point:                                                      # the value at the voxel centre: balls within the limit ...
        if "ball" in r["shape"]:
            assert abs(r["rel_soft"]) <= LIM["lattice_volume"], r
        else:                                                            # ... a flat end parallel to the lattice planes within the spacing only
            assert abs(r["rel_soft"]) <= 0.6 * r["spacing_mm"] / float(r["shape"].split("length ")[1].split(" mm")[0]), r
    assert max(abs(r["rel_soft"]) for r in point if "ball" not in r["shape"]) > LIM["lattice_volume"]       # the page says so: keep it true
    # the soft-edge term itself: a fine supersampled lattice converges to the soft form, not to the sharp one
    ball = volume.shapes_volume([("wm", lambda y, x, z: volume.sd_sphere(y, x, z, (0.3, 0.2, -0.4), 10.0))])
    v = float(ball.lattice(1.0, box={a: (-14.0, 14.0) for a in "zyx"}, supersample=3).weights.sum(dtype=np.float64))
    assert abs(v / VC.ball_volume(10.0) - 1) < 2e-4 and v / (4 / 3 * np.pi * 1000.0) - 1 == pytest.approx(np.pi ** 2 / 4 * 0.05 ** 2, rel=0.05)
    cyl = volume.shapes_volume([("wm", lambda y, x, z: volume.sd_cylinder(y, x, z, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 12.0, 20.5))])
    box = dict(z=(-26.0, 26.0), y=(-16.0, 16.0), x=(-16.0, 16.0))
    along = [float(cyl.lattice(2.0, box=box, supersample=k).weights.sum(dtype=np.float64)) * 8 / (VC.disc_area(12.0) * 41.0) - 1 for k in (1, (2, 1, 1))]
    assert abs(along[0]) > 0.005 > abs(along[1])                         # two samples along z place a flat end; one does not


@pytest.mark.parametrize("qa_spec", [None, dict(n_oil=2), dict(body_aspect=(1.2, 1.5), body_rotation_deg=(0.0, 90.0)),
                                     dict(inserts=[dict(y_mm=10.0, x_mm=-20.0, radius_mm=8.0, tissue="oil", weight=0.4),
                                                   dict(y_mm=10.0, x_mm=-20.0, radius_mm=8.0, tissue=dict(pd=0.9, t1_ms=900.0, t2_ms=80.0, t2p_ms=100.0), weight=0.6)])],
                         ids=["default", "oil", "elongated", "placed discs"])
def test_the_axial_plane_of_qa3d_through_its_vial_section_is_the_2d_qa_world(qa_spec):
    """Compared: the reference plane of ``qa3d`` (axial, z = 0, master grid) against ``world["weights"]`` of the 2D
    ``qa`` world of the same seed and spec, i.e. against what ``anatomy.qa`` draws; also turned by 30 degrees."""
    for seed in VC.SEEDS:
        for rot in (0.0, 30.0):
            d = VC.flat_plane_against_2d(seed, qa_spec, rot)
            assert d["max"] <= LIM["flat_2d"] and d["rest_absent"] and d["names"] and d["tissues"] and d["regions"], (seed, rot, d)
            assert d["bitwise"], (seed, rot, d)                          # in fact the same bytes
    # and through the slice: every plane of the uniform part of the section is that plane
    w = VC.make("qa3d", VC.SEEDS[1], qa=qa_spec)
    lay, vol = w["layout"], w["volume"]
    reach = lay["vial_half_length"] - 20 * anatomy.EDGE_MM
    for dz in (-0.9 * reach, 0.5 * reach, 0.9 * reach):
        assert np.array_equal(vol.eval_plane(volume.Frame.plane("axial", lay["body"]["z"] + dz), dtype=np.float64), w["weights"])
    assert np.abs(vol.eval_plane(volume.Frame.plane("axial"), dtype=np.float32) - w["weights"]).max() < 1e-4       # the float32 path: rounding only


# ---------------------------------------------------------------------------------------------- acceptance: one function of the seed
JOBS = [("head3d", 0, {}), ("head3d", 1, dict(lesion=None)), ("head3d", 2, dict(edge_mm=0.25)), ("head3d", 0, dict(volume=dict(vessels=1, pose_deg=5.0))),
        ("qa3d", 0, {}), ("qa3d", 1, dict(edge_mm=0.25)), ("qa3d", 2, dict(qa=dict(n_oil=2))), ("qa3d", 0, dict(volume=dict(pose_deg=(2.0, 6.0)))),
        ("qa3d", 3, dict(volume=dict(features=False))), ("head3d", 3, dict(b0_t=1.5)), ("qa", 0, {}), ("brain", 0, {})]


def _job(job):
    kind, k, spec = job
    w = W.make_world(VC.SEEDS[k], dict(dict(kind=kind, lesion="ms" if kind in ("head3d", "brain") else None), **spec))
    if kind in volume.KINDS:
        return VC.digest(w)
    return VC.hashlib.sha256(w["weights"].tobytes() + json.dumps(w["system"], sort_keys=True).encode()).hexdigest()


def _constants():
    plain = (bool, int, float, str, bytes, tuple, frozenset, set, list, dict, type(None))
    return {f"{name}.{k}": repr(v) for name, mod in sorted(sys.modules.items()) if name.startswith("mrgym.scanner") and mod is not None
            for k, v in sorted(vars(mod).items()) if k.isupper() and isinstance(v, plain)}


def test_a_world_is_the_same_bytes_from_any_thread_in_any_order():
    """As ``tests/test_thread_determinism.py`` does for families: serial digests first, then the same worlds (two
    half-widths of the soft edge, options on and off, 2D kinds between them) from four threads in two shuffled
    orders with a short switch interval; and no module constant of the scanner is left changed."""
    serial = {i: _job(job) for i, job in enumerate(JOBS)}
    assert len(set(serial.values())) == len(JOBS)                        # every job is another world
    before, old = _constants(), sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    try:
        for r in range(2):
            order = list(serial)
            random.Random(r).shuffle(order)
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                got = dict(zip(order, pool.map(lambda i: _job(JOBS[i]), order)))
            bad = [JOBS[i][:2] for i in order if got[i] != serial[i]]
            assert not bad, f"round {r}: worlds built next to other threads differ from the serial build: {bad}"
    finally:
        sys.setswitchinterval(old)
    assert _constants() == before
    assert {i: _job(job) for i, job in list(enumerate(JOBS))[::-1]} == serial          # and in the reverse order


def test_an_option_moves_no_other_draw():
    """Lesions, vessels, the 3D inserts and the pose each draw from a stream of their own."""
    base = W.make_world(SEED, dict(kind="head3d"), layout_only=True)
    les = W.make_world(SEED, dict(kind="head3d", lesion="ms"), layout_only=True)
    ves = W.make_world(SEED, dict(kind="head3d", volume=dict(vessels=0)), layout_only=True)
    strip = lambda lay, *keys: {k: v for k, v in lay.items() if k not in keys}                  # noqa: E731
    assert strip(base["layout"], "lesions", "lesion_chi_ppm") == strip(les["layout"], "lesions", "lesion_chi_ppm") and not base["layout"]["lesions"]
    assert 4 <= len(les["layout"]["lesions"]) <= 8 and "lesion_ms" in les["names"] and "lesion_ms" not in base["names"]
    assert all(base["tissues"][k] == les["tissues"][k] for k in base["names"]) and base["system"] == les["system"]
    assert strip(base["layout"], "vessels") == strip(ves["layout"], "vessels") and len(base["layout"]["vessels"]) == 3 and not ves["layout"]["vessels"]
    full = W.make_world(SEED, dict(kind="qa3d"), layout_only=True)
    bare = W.make_world(SEED, dict(kind="qa3d", volume=dict(features=False)), layout_only=True)
    assert all(full["layout"][k] == bare["layout"][k] for k in bare["layout"]) and bare["volume"].solid == () and full["volume"].solid == ("plastic",)
    assert all(full["tissues"][k] == bare["tissues"][k] for k in bare["names"]) and bare["names"] == full["names"][:len(bare["names"])]
    turned = W.make_world(SEED, dict(kind="qa3d", volume=dict(pose_deg=4.0)), layout_only=True)
    assert strip(turned["layout"], "pose_deg") == strip(full["layout"], "pose_deg") and any(turned["layout"]["pose_deg"]) and not any(full["layout"]["pose_deg"])


# ---------------------------------------------------------------------------------------------- acceptance: cost
@pytest.mark.parametrize("kind", volume.KINDS)
def test_build_time_plane_time_and_memory(kind):
    t = VC.timings(kind)
    m = VC.memory(kind)
    print(f"\n{kind}: build {t['build_s']} s (recipe and regions {t['recipe_and_regions_s']} s); one plane at {t['grid'][0]} x {t['grid'][0]}: {t['plane_ms']} ms; "
          f"memory (tracemalloc) build peak {m['build_peak_mb']} MB, held {m['world_held_mb']} MB, one plane {m['plane_peak_mb']} MB, "
          f"lattice 2 mm {m['lattice_2mm_shape']} {m['lattice_2mm_held_mb']} MB held, {m['lattice_2mm_peak_mb']} MB peak, {m['lattice_2mm_s']} s")
    assert min(t["build_s"]) <= LIM["build_s"] and max(t["plane_ms"].values()) <= LIM["plane_ms"] and t["grid"][0] == 640
    assert max(m["build_peak_mb"], m["plane_peak_mb"], m["lattice_2mm_peak_mb"]) < 600          # far below the 3 GB this package may use


# ---------------------------------------------------------------------------------------------- regions, truth, layout
@pytest.mark.parametrize("kind", volume.KINDS)
def test_every_region_lies_in_its_class_and_truth_is_its_tissue(request, kind):
    w = _world(request, kind)
    vol, rng = w["volume"], np.random.default_rng(3)
    assert len({r["id"] for r in vol.regions}) == len(vol.regions) >= 5
    truth, flat = vol.truth(), W.truth(w)
    for r in vol.regions:
        d = rng.normal(size=(400, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        if r["shape"] == "sphere":
            p = d * r["radius_mm"] * rng.uniform(0, 1, (400, 1)) ** (1 / 3)
        else:
            a = np.array(r["axis"])
            d -= np.outer(d @ a, a)
            p = d / np.linalg.norm(d, axis=1)[:, None] * r["radius_mm"] * np.sqrt(rng.uniform(0, 1, (400, 1))) + np.outer(rng.uniform(-1, 1, 400) * r["half_length_mm"], a)
        p = p + [r["y_mm"], r["x_mm"], r["z_mm"]]
        assert vol.inside(r, p[:, 0], p[:, 1], p[:, 2]).mean() > 0.99 and np.array_equal(vol.inside(r["id"], *p.T), vol.inside(r, *p.T))
        far = p + 3.0 * (r["radius_mm"] + r.get("half_length_mm", 0.0))
        assert not vol.inside(r, far[:, 0], far[:, 1], far[:, 2]).any()
        share = float(vol.eval_points(p[:, 0], p[:, 1], p[:, 2], solid=True)[vol.matter.index(r["cls"])].mean())
        assert share >= volume.PURITY_MIN - 0.03 and r["purity"] >= volume.PURITY_MIN, (r["id"], r["name"], share)
        c = w["tissues"][r["cls"]]
        assert truth[r["id"]] == dict(t1_ms=c["t1_ms"], t2_ms=c["t2_ms"], t2s_ms=c["t2s_ms"], pd=c["pd"], fat=c["fat"], chi_ppm=vol.susceptibility_ppm[r["cls"]])
        assert flat[r["id"]] == {k: v for k, v in truth[r["id"]].items() if k != "chi_ppm"}
    assert {r["shape"] for r in vol.regions} == {"sphere", "cylinder"}
    assert [dict(id=r["id"], y_mm=r["y_mm"], x_mm=r["x_mm"], radius_mm=r["radius_mm"]) for r in vol.regions] == W.regions_public(w)


def test_the_layout_of_the_head_says_where_things_are(head):
    """Positions in the layout are patient coordinates: the recipe, asked there, answers with the structure."""
    vol, lay = head["volume"], head["layout"]
    at = lambda c, cls: float(vol.eval_points(*c)[vol.names.index(cls)])                         # noqa: E731
    assert len(lay["deep_nuclei"]) == 6 and all(at(s["c_mm"], "deep_gm") > 0.95 for s in lay["deep_nuclei"])
    assert all(at(s["c_mm"], "lesion_ms") + at(s["c_mm"], "blood") > 0.99 for s in lay["lesions"]) and 4 <= len(lay["lesions"]) <= 8     # a vessel may cross one
    assert sum(at(s["c_mm"], "csf") > 0.9 for s in lay["ventricles"]) >= 6
    assert all(float(vol.eval_points(*s["c_mm"]).sum()) < 0.05 for s in lay["air"] if s["name"].endswith("sinus"))       # air: no class
    for v in lay["vessels"]:
        p0, d = np.array(v["p0_mm"]), np.array(v["direction"])
        assert abs(np.linalg.norm(d) - 1) < 1e-9 and d[2] > 0.9 and 40 <= abs(v["speed_mm_s"]) <= 250
        assert np.allclose(np.array(v["p1_mm"]) - p0, d * np.linalg.norm(np.array(v["p1_mm"]) - p0))
        on = [p0 + d * s for s in np.linspace(60, 170, 23)]             # on the axis: through neck, skull base and brain
        assert sum(at(p, "blood") > 0.97 for p in on) >= 20              # blood (an air cavity in its way has no blood)
        side = np.cross(d, [1.0, 0.0, 0.0])
        off = [p + side / np.linalg.norm(side) * (v["radius"] + 2.5) for p in on]
        assert sum(at(p, "blood") < 0.01 for p in off) >= 17             # and straight: 2.5 mm outside the wall there is none (unless another vessel comes close)
    # the warp and its inverse
    s = np.array([[10.0, -20.0, 30.0], [-60.0, 5.0, -40.0]])
    assert np.abs(vol._warped(vol._unwarped(s)) - s).max() < 1e-9 and 0.3 < np.abs(vol._unwarped(s) - s).max() < 12.0
    json.dumps(lay)                                                      # plain numbers and lists


def test_the_3d_inserts_of_qa3d_are_where_the_layout_says(qa):
    """The ball by its radius in a plane; the wedge faces and the ramps by the positions of their edges."""
    vol, lay = qa["volume"], qa["layout"]
    idx = {k: i for i, k in enumerate(vol.matter)}
    sp = lay["sphere"]
    for h in (0.0, 0.5 * sp["radius"]):                                  # a plane at height h cuts a disc of radius sqrt(R^2 - h^2)
        w = vol.eval_plane(volume.Frame.plane("axial", sp["c"][2] + h), 640, solid=True)[idx["sphere"]]
        area = float(w.sum(dtype=np.float64)) * 0.4 ** 2
        assert area == pytest.approx(VC.disc_area(float(np.sqrt(sp["radius"] ** 2 - h ** 2))), rel=3e-3)
    assert abs(sp["c"][2] - lay["body"]["z"]) > lay["vial_half_length"] + sp["radius"]             # clear of the vial section
    # wedges: along the lane, at height d above the centre the face of lane 1 sits at v = +d / tan(angle), of lane 2 at -d / tan(angle)
    wd = lay["wedges"]
    co, si, tb = np.cos(np.deg2rad(wd["rot_deg"])), np.sin(np.deg2rad(wd["rot_deg"])), np.tan(np.deg2rad(wd["angle_deg"]))
    v = np.linspace(-17.0, 17.0, 3401)
    for d in (-3.0, 0.0, 4.0):
        edges = []
        for u in (-wd["lane_mm"], wd["lane_mm"]):
            w = vol.eval_points(wd["c"][0] + v * co - u * si, wd["c"][1] + v * si + u * co, wd["c"][2] + d, solid=True)[idx["wedge"]]
            edges.append(float(v[np.argmin(np.abs(w - 0.5))]))           # the half-height point of the tilted face
        assert edges[0] == pytest.approx(d / tb, abs=0.03) and edges[1] == pytest.approx(-d / tb, abs=0.03)
        assert (18.0 - edges[0]) - (18.0 - edges[1]) == pytest.approx(-2 * d / tb, abs=0.06)       # bar lengths differ by 2 d / tan(angle)
    # ramps: in a plane at height d the sheet of band 1 sits at u = d / slope, of band 2 at -d / slope
    bk = lay["ramp_block"]
    co, si = np.cos(np.deg2rad(bk["rot_deg"])), np.sin(np.deg2rad(bk["rot_deg"]))
    u = np.linspace(-bk["reach"] + 2, bk["reach"] - 2, 5601)
    for d in (-2.0, 0.0, 3.0):
        for vb, sign in ((-bk["band_centre_mm"], 1.0), (bk["band_centre_mm"], -1.0)):
            w = vol.eval_points(bk["c"][0] + vb * co - u * si, bk["c"][1] + vb * si + u * co, bk["c"][2] + d, solid=True)
            slot, plastic = w[idx["slot"]], w[idx["plastic"]]
            assert float((slot * u).sum() / slot.sum()) == pytest.approx(sign * d / bk["slope"], abs=0.05) and slot.max() > 0.99
            assert float(slot.sum() * (u[1] - u[0])) == pytest.approx(bk["sheet_mm"] * np.hypot(1.0, bk["slope"]) / bk["slope"], rel=0.01)   # bar length of a thin plane
            assert np.abs(slot + plastic - 1).max() < 1e-9               # in the block: plastic or fluid, nothing else
    assert vol.susceptibility_ppm["plastic"] == 0.0 and "plastic" not in qa["names"] and "plastic" in vol.matter


def test_a_pose_turns_the_object_not_the_recipe():
    """The object turned about its centre, seen in a plane = the upright object seen in the plane turned back."""
    for kind in volume.KINDS:
        up = VC.make(kind, VC.SEEDS[1])
        posed = VC.make(kind, VC.SEEDS[1], volume=dict(pose_deg=(3.0, 8.0)))
        vol, pv = up["volume"], posed["volume"]
        assert all(3.0 <= abs(a) <= 8.0 for a in posed["layout"]["pose_deg"]) and pv._pose is not None and vol._pose is None
        f = volume.Frame.plane("coronal", 6.0, tilt_deg=(5.0, 0.0))
        g = volume.Frame(pv.to_object(f.origin), *(pv.to_object(v, direction=True) for v in (f.row, f.col, f.normal)))
        assert np.abs(pv.eval_plane(f, 128) - vol.eval_plane(g, 128)).max() < 1e-4
        c = volume.grid_axis(48, 256.0)
        assert np.abs(pv.eval_plane(f, 48) - pv.eval_points(*f.points(c, c))).max() < 1e-4
        q = np.array([12.0, -30.0, 44.0])
        assert np.allclose(pv.to_patient(pv.to_object(q)), q) and not np.allclose(pv.to_object(q), q)
        for a, b in zip(up["regions"], posed["regions"]):                # regions follow the object
            assert a["id"] == b["id"] and np.allclose(pv.to_object([b["y_mm"], b["x_mm"], b["z_mm"]]), [a["y_mm"], a["x_mm"], a["z_mm"]], atol=0.02)


def test_turned_reference_plane_and_placed_inserts(head):
    t = W.turned(head, 25.0)
    assert np.array_equal(t["weights"], head["volume"].eval_plane(volume.Frame.plane("axial", rot_deg=25.0), dtype=np.float64))
    assert t["volume"] is head["volume"] and t["rot_deg"] == 25.0 and t["tissues"] is head["tissues"]
    ins = [dict(shape="sphere", y_mm=0.0, x_mm=30.0, z_mm=40.0, radius_mm=7.0, tissue="tumour", chi_ppm=0.5),
           dict(shape="cylinder", y_mm=0.0, x_mm=0.0, z_mm=140.0, radius_mm=5.0, half_length_mm=10.0, axis=(0, 1, 0), tissue="oil")]
    w = W.make_world(SEED, dict(kind="head3d", lesion="ms", volume=dict(inserts=ins)))
    vol = w["volume"]
    assert w["names"][-2:] == ["placed0", "placed1"] and vol.susceptibility_ppm["placed0"] == 0.5 and w["tissues"]["placed1"]["fat"]
    assert vol.eval_points(0.0, 30.0, 40.0)[-2] == pytest.approx(1.0, abs=1e-6) and vol.eval_points(0.0, 30.0, 49.0)[-2] < 0.01
    assert not vol.eval_points(0.0, 0.0, 140.0).any()                    # outside the head an insert adds no matter
    assert w["layout"] == head["layout"]                                 # placed, not drawn


# ---------------------------------------------------------------------------------------------- the lattice (interface of the field builder)
def test_lattice_interface(head):
    vol = head["volume"]
    lat = vol.lattice(4.0)
    weights, origin, spacing = lat                                       # unpacks as (weights, origin_mm, spacing_mm)
    assert weights.dtype == np.float32 and weights.shape[0] == len(vol.matter) and spacing == (4.0, 4.0, 4.0)
    z, y, x = lat.axes()
    assert (len(z), len(y), len(x)) == weights.shape[1:] and (z[0], y[0], x[0]) == origin and 0.0 in z and 0.0 in y and 0.0 in x
    assert z[0] == -128.0 and z[-1] <= vol.bounds_mm["z"][1] and y[0] >= vol.bounds_mm["y"][0]
    pts = vol.eval_points(y[None, :, None], x[None, None, :], z[::6, None, None], solid=True)
    assert np.abs(pts - weights[:, ::6]).max() < 1e-4                    # voxel [k, i, j] is the recipe at (z_k, y_i, x_j)
    assert vol.lattice(4.0) is lat                                       # the last lattice is kept
    lab = vol.lattice(4.0, labels=True).weights
    assert lab.dtype == np.uint8 and lab.shape == weights.shape[1:] and np.array_equal(lab, vol.labels(weights)) and lab.max() <= len(vol.matter)
    chi = vol.chi_ppm(weights)
    assert chi.max() == pytest.approx(9.4, abs=1e-5) and chi[0, 0, 0] == pytest.approx(9.4) and -2.0 <= chi.min() < -1.5
    organ = vol.organ_mask(weights)
    assert vol.organ == ("wm", "gm", "deep_gm", "lesion_ms") and 8000 < organ.sum() < 30000 and abs(float(chi[organ].mean())) < 0.05
    fine = vol.lattice((4.0, 2.0, 2.0), box=dict(z=(0.0, 8.0)))
    fz, fy, fx = fine.axes()
    same = fine.weights[:, :, np.isin(fy, y)][:, :, :, np.isin(fx, x)]                  # the points the two lattices share
    assert fz.tolist() == [0.0, 4.0, 8.0] and fine.spacing_mm == (4.0, 2.0, 2.0) and np.abs(same - weights[:, np.isin(z, fz)]).max() < 1e-5
    box = dict(z=(0.0, 4.0), y=(-20.0, 20.0), x=(-20.0, 20.0))
    mean = vol.lattice(4.0, box=box, supersample=2).weights
    half = (np.arange(-5, 6)[:, None] * 4.0 + np.array([-1.0, 1.0])).ravel()                        # the 8 samples of each voxel: at +-1 mm
    sub = vol.eval_points(half[None, :, None], half[None, None, :], np.array([-1.0, 1.0, 3.0, 5.0])[:, None, None], solid=True)
    assert np.abs(mean - sub.reshape(len(vol.matter), 2, 2, 11, 2, 11, 2).mean((2, 4, 6))).max() < 1e-4
    with pytest.raises(ValueError, match="labels=True"):
        vol.lattice(0.5)
    with pytest.raises(ValueError, match="no lattice point"):
        vol.lattice(4.0, box=dict(z=(1.0, 3.0)))


def test_hand_made_volume_later_shapes_on_top_and_solid_gives_no_signal():
    vol = volume.shapes_volume([("gm", lambda y, x, z: volume.sd_sphere(y, x, z, (0, 0, 0), 30.0)),
                                ("plastic", lambda y, x, z: volume.sd_cylinder(y, x, z, (0, 0, 0), (0, 0, 1), 10.0, 20.0)),
                                ("wm", lambda y, x, z: volume.sd_halfspace(y, x, z, (0, 0, 25.0), (0, 0, -1)))], solid=("plastic",), chi_ppm=dict(plastic=-0.5))
    assert vol.names == ("gm", "wm") and vol.solid == ("plastic",) and vol.susceptibility_ppm == dict(gm=0.0, wm=0.0, plastic=-0.5, air=9.4)
    at = lambda *p: vol.eval_points(*p, solid=True).round(4).tolist()   # noqa: E731
    assert at(20, 0, 0) == [1, 0, 0] and at(0, 0, 0) == [0, 0, 1] and at(0, 0, 28) == [0, 1, 0] and at(40, 0, 0) == [0, 0, 0] and at(40, 0, 40) == [0, 1, 0]
    assert vol.eval_points(0, 0, 0).tolist() == [0, 0] and vol.eval_plane(volume.Frame.plane("sagittal"), 64).shape == (2, 64, 64)
    assert vol.chi_ppm(np.array([[0.0], [0.0], [1.0]]))[0] == -0.5 and vol.chi_ppm(np.zeros((3, 1)))[0] == pytest.approx(9.4)
    # a plane interface tilted by 30 degrees against the normal of an axial plane: its soft edge is the world's, along its own normal
    tilt = volume.shapes_volume([("wm", lambda y, x, z: volume.sd_halfspace(y, x, z, (0, 0, 0), (np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6))))])
    y = np.linspace(-5, 5, 2001)
    w = tilt.eval_points(y, 0.0, 0.0)[0]
    assert w[0] > 0.99 and w[-1] < 0.01 and abs(float(y[np.argmin(np.abs(w - 0.5))])) < 0.01
    assert np.allclose(w, 0.5 * (1 + np.tanh(-y * np.sin(np.pi / 6) / anatomy.EDGE_MM)))


# ---------------------------------------------------------------------------------------------- golden digests
@pytest.mark.parametrize("key", list(G.VOLUME_WORLDS))
def test_golden_digest_of_a_volume_world(key):
    """``tests/golden/volume_digests.json``: bit for bit on the stack it was recorded on, class volumes elsewhere."""
    stored = G.load(G.VOLUME_FILE)
    assert stored and sorted(stored["worlds"]) == sorted(G.VOLUME_WORLDS), "run `python tools/audit/golden.py --write`"
    got, want = G.volume_case(key), stored["worlds"][key]
    assert got["classes"] == want["classes"] and got["regions"] == want["regions"]
    if G.mode(stored["stack"])[0] == "bit-exact":
        assert got["sha256"] == want["sha256"], f"the volume world {key} changed: if that is intended, `python tools/audit/golden.py --write`"
    assert got["class_mm3"] == pytest.approx(want["class_mm3"], rel=1e-4, abs=2.0)


# ---------------------------------------------------------------------------------------------- the page
def test_the_page_is_the_stored_measurement():
    """docs/physics/volume.md is ``doc_page`` of docs/physics/volume.json, and the stored numbers that are functions
    of the code (not of the machine) are the ones this tree gives."""
    store, page = VC.paths()
    with open(store, encoding="utf-8") as f:
        m = json.load(f)
    with open(page, encoding="utf-8") as f:
        assert f.read() == VC.doc_page(m), "regenerate: python -m mrgym.scanner.volume_check --write"
    assert m["limits"] == VC.LIMITS and m["seeds"] == list(VC.SEEDS) and set(m["kinds"]) == set(volume.KINDS)
    for kind in volume.KINDS:
        w, row = VC.make(kind), m["kinds"][kind]
        assert row["names"] == list(w["names"]) and row["regions"] == len(w["regions"]) and row["layout_keys"] == sorted(w["layout"])
        assert row["plane_points"]["max"] <= LIM["plane_points"] and row["common_line"] <= LIM["common_line"]
        assert max(row["timings"]["plane_ms"].values()) <= LIM["plane_ms"] and max(row["timings"]["build_s"]) <= LIM["build_s"]
        assert row["common_line"] == pytest.approx(VC.common_line(w["volume"]), rel=1e-3, abs=1e-12)
    assert m["flat_2d"]["bitwise"] and m["flat_2d"]["max"] == 0.0 and m["flat_2d"]["all_else"]
    assert [r["rel_soft"] for r in m["lattice_volumes"]] == pytest.approx([r["rel_soft"] for r in VC.lattice_volumes()], abs=1e-6)
    assert [r[0] for r in m["kinds"]["qa3d"]["ranges"]] == [r[0] for r in VC.hidden_ranges("qa3d", 2)]
    assert os.path.getsize(store) < 30_000
