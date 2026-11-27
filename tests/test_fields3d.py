"""Fields in three dimensions (``mrgym.scanner.fields3d``): the validation table passes and the document shows it;
the conventions (sign, k = 0, axes); the surface step; shim limits; the interfaces the slice engine will call."""
import math
import os
import re

import numpy as np
import pytest

from mrgym.scanner import fields3d as F
from mrgym.scanner import fields3d_validate as V
from mrgym.scanner import world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = {}


def rows(group):
    if group not in _cache:
        _cache[group] = V.CHECKS[group]()
    return _cache[group]


# ------------------------------------------------------------------------------------------------ the table
@pytest.mark.parametrize("group", list(V.CHECKS))
def test_every_row_passes(group):
    bad = [(r["id"], r["measured"], r["tolerance"]) for r in rows(group) if not r["ok"]]
    assert not bad, bad
    assert len(rows(group)) >= 6


def test_row_ids_are_unique_and_groups_known():
    ids = [r["id"] for g in V.CHECKS for r in rows(g)]
    assert len(ids) == len(set(ids))
    assert set(V.CHECKS) == set(V.GROUPS)


def _strip_numbers(line):
    """A table line without its measured figures: numbers inside a quantity text and the error column are rounding
    level or timing and differ between machines."""
    cells = line.split(" | ")
    if len(cells) < 6:
        return line
    cells[1] = re.sub(r"\d[\d.]*(e[+-]?\d+)?", "#", cells[1])
    cells[2] = re.sub(r"\d[\d.]*(e[+-]?\d+)?", "#", cells[2])
    cells[3] = ""
    return " | ".join(cells)


def test_the_document_shows_the_table_that_the_command_prints():
    path = os.path.join(ROOT, "docs", "physics", "fields3d.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    block = text.split("<!-- validation table: begin -->")[1].split("<!-- validation table: end -->")[0]
    want = V.markdown_table([r for g in V.CHECKS for r in rows(g)])
    got = [_strip_numbers(line) for line in block.strip().splitlines()]
    assert got == [_strip_numbers(line) for line in want.strip().splitlines()]
    assert "FAIL" not in block


def test_the_document_shows_the_head_figures_of_the_command():
    """The head table of the document is the output of ``--head``: every figure within 2 % or 0.2 of a unit (the
    1 mm comparison is left out here: it takes 1.7 GB)."""
    path = os.path.join(ROOT, "docs", "physics", "fields3d.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        block = f.read().split("<!-- head figures: begin -->")[1].split("<!-- head figures: end -->")[0].strip().splitlines()
    want = V.head_markdown(V.head_numbers(fine=False)).splitlines()
    fine = [line for line in block if line.startswith("| The 2 mm lattice against a 1 mm lattice")]
    assert len(fine) == 1                                               # the document has the 1 mm row as well
    block = [line for line in block if line not in fine]
    assert len(block) == len(want)
    number = re.compile(r"[-+]?\d+\.?\d*")
    for a, b in zip(block, want):
        assert number.sub("#", a) == number.sub("#", b)
        for u, v in zip(number.findall(a), number.findall(b)):
            assert abs(float(u) - float(v)) <= max(0.2, 0.02 * abs(float(v))), (a, b)


# ------------------------------------------------------------------------------------------------ conventions
def test_sign_units_and_the_engine_sphere():
    """An air sphere in tissue: positive off-resonance on the B0 axis, negative on the equator, in Hz at the world's
    field strength, equal to the engine's declared sphere (``world.dipole_hz``) outside it."""
    y, x, z = np.array([0.0, 30.0, 12.0]), np.array([0.0, 0.0, -9.0]), np.array([30.0, 0.0, 21.0])
    for b0_t in (1.5, 3.0):
        ph = dict(b0_t=b0_t, air=[(0.0, 0.0, 0.0, 12.0)])
        mine = world.GAMMA_HZ_PER_T * b0_t * 1e-6 * F.sphere_field(y, x, z, (0, 0, 0), 12.0, F.DCHI_AIR_PPM)
        np.testing.assert_allclose(mine, [world.dipole_hz(ph, y[i], x[i], z[i]) for i in range(3)], rtol=1e-12)
        assert mine[0] > 0 > mine[1]
    lat = F.Lattice.centred((48,) * 3, 2.0)
    w = V.ball(lat, 12.0)
    fld = F.Field3D(F.DCHI_AIR_PPM * w, lat, 3.0, tissue=1 - w, background_ppm=0.0)
    assert fld.hz_per_ppm == pytest.approx(127.732, abs=1e-3)
    assert fld.hz(0.0, 0.0, 20.0) == pytest.approx(world.dipole_hz(dict(b0_t=3.0, air=[(0, 0, 0, 12.0)]), 0.0, 0.0, 20.0), rel=0.03)
    assert fld.hz(0.0, 0.0, 20.0) > 0 > fld.hz(20.0, 0.0, 0.0)


def test_k0_only_differences_of_chi_matter_and_the_mean_is_zero():
    lat = F.Lattice.centred((32,) * 3, 2.0)
    w = V.ball(lat, 10.0, (3.0, -2.0, 1.0)).astype(float)
    f = F.dipole_field(w, 2.0, pad=1.0, background=0.0, images=False)
    assert abs(f.mean()) < 1e-15
    np.testing.assert_allclose(F.dipole_field(w + 7.0, 2.0, pad=1.0, background=7.0, images=False), f, atol=1e-13)
    np.testing.assert_allclose(F.dipole_field(w + 7.0, 2.0, pad=1.0, images=False), f, atol=1e-13)      # the faces name the surroundings
    assert F.dipole_kernel((8, 8, 8), 2.0)[0, 0, 0] == 0.0


def test_axes_origin_and_spacing_follow_the_array_and_vectors_are_y_x_z():
    lat = F.Lattice.centred((10, 20, 30), (3.0, 2.0, 1.0))
    z, y, x = lat.axes()
    assert (z[5], y[10], x[15]) == (0.0, 0.0, 0.0) and (z[1] - z[0], y[1] - y[0], x[1] - x[0]) == (3.0, 2.0, 1.0)
    np.testing.assert_allclose(lat.index(y[3], x[7], z[2]), [2, 3, 7])
    # a sphere displaced along +y on the lattice is found along +y by the closed form
    lat = F.Lattice.centred((40,) * 3, 2.0)
    w = V.ball(lat, 8.0, (14.0, 0.0, 0.0))
    f = F.dipole_field(w, 2.0, (0, 0, 1), background=0.0)
    yy, xx, zz = lat.mesh()
    ex = F.sphere_field(yy, xx, zz, (14.0, 0.0, 0.0), 8.0) + 0 * f
    far = np.sqrt((yy - 14.0) ** 2 + xx ** 2 + zz ** 2) + 0 * f > 12.0
    assert np.abs(f - ex)[far].max() < 0.01 and np.abs(ex[far]).max() > 0.15
    with pytest.raises(ValueError):
        F.dipole_field(w, 2.0, (0, 0, 0))
    with pytest.raises(ValueError):
        F.dipole_field(w, 2.0, pad=0.5)
    with pytest.raises(ValueError):
        F.dipole_field(w, 2.0, extrude=("w+",))


def test_lorentz_term_is_two_thirds_of_the_local_susceptibility():
    lat = F.Lattice.centred((32,) * 3, 2.0)
    w = V.ball(lat, 10.0).astype(float)
    with_l = F.dipole_field(w, 2.0, pad=1.0, background=0.0, images=False)
    without = F.dipole_field(w, 2.0, pad=1.0, background=0.0, images=False, lorentz=False)
    np.testing.assert_allclose(without - with_l, 2.0 / 3.0 * (w - w.mean()), atol=1e-13)
    assert F.sphere_field(0, 0, 0, (0, 0, 0), 5.0, lorentz=False) == pytest.approx(2.0 / 3.0)
    for theta, inside in ((0.0, 1.0), (math.pi / 2, 0.5)):              # Schenck's values without the correction
        b = (math.sin(theta), 0.0, math.cos(theta))
        assert F.cylinder_field(0, 0, 0, (0, 0, 0), (0, 0, 1), 5.0, 1.0, b, lorentz=False) == pytest.approx(inside)


def test_closed_forms_against_the_separate_derivation():
    """Values of a derivation that shares no code with this package (surface-charge quadrature, no k-space kernel)."""
    a = 24.0
    assert F.sphere_field(0, 0, 1.5 * a, (0, 0, 0), a) == pytest.approx(0.197531, abs=1e-6)
    assert F.sphere_field(1.5 * a, 0, 0, (0, 0, 0), a) == pytest.approx(-0.098765, abs=1e-6)
    assert abs(F.sphere_field(1.25 * a * math.sqrt(2 / 3), 0, 1.25 * a * math.sqrt(1 / 3), (0, 0, 0), a)) < 1e-12
    for theta, inside in ((0.0, 1 / 3), (math.radians(30), 0.208333), (V.MAGIC, 0.0), (math.pi / 2, -1 / 6)):
        b = (math.sin(theta), 0.0, math.cos(theta))
        assert F.cylinder_field(1.0, 2.0, 3.0, (0, 0, 0), (0, 0, 1), a, 1.0, b) == pytest.approx(inside, abs=1e-6)
        out = F.cylinder_field(1.5 * a, 0.0, 7.0, (0, 0, 0), (0, 0, 1), a, 1.0, b)            # phi = 0: along the projection of B0
        assert out == pytest.approx(0.5 * math.sin(theta) ** 2 / 1.5 ** 2, abs=1e-12)
        assert F.cylinder_field(0.0, 1.5 * a, 7.0, (0, 0, 0), (0, 0, 1), a, 1.0, b) == pytest.approx(-out, abs=1e-12)
    assert F.shell_field(0, 0, 20.0, (0, 0, 0), 16.0, a) == pytest.approx(-0.341333, abs=1e-6)
    assert F.shell_field(20.0, 0, 0, (0, 0, 0), 16.0, a) == pytest.approx(0.170667, abs=1e-6)
    assert F.shell_field(0, 0, 36.0, (0, 0, 0), 16.0, a) == pytest.approx(0.139003, abs=1e-6)
    assert F.shell_field(3.0, -2.0, 1.0, (0, 0, 0), 16.0, a) == 0.0
    assert F.demagnetising_factors((18.0, 18.0, 36.0))[2] == pytest.approx(0.173564, abs=1e-6)
    assert F.demagnetising_factors((24.0, 24.0, 12.0))[2] == pytest.approx(0.527200, abs=1e-6)
    assert F.ellipsoid_inside((18.0, 18.0, 36.0), (0, 0, 1)) == pytest.approx(0.159769, abs=1e-6)
    assert F.ellipsoid_inside((24.0, 24.0, 12.0), (0, 0, 1)) == pytest.approx(-0.193867, abs=1e-6)
    assert sum(F.demagnetising_factors((10.0, 17.0, 31.0))) == pytest.approx(1.0, abs=1e-9)
    hz = world.GAMMA_HZ_PER_T * 3.0 * 1e-6 * F.DCHI_AIR_PPM                                 # an air sphere of 12 mm at 3 T
    for d, along, across in ((15.0, 409.83, -204.92), (20.0, 172.90, -86.45), (30.0, 51.23, -25.61), (40.0, 21.61, -10.81)):
        assert hz * F.sphere_field(0, 0, d, (0, 0, 0), 12.0) == pytest.approx(along, abs=0.006)
        assert hz * F.sphere_field(d, 0, 0, (0, 0, 0), 12.0) == pytest.approx(across, abs=0.006)
    grad = hz * (F.sphere_field(0, 0, 20.0 + 1e-4, (0, 0, 0), 12.0) - F.sphere_field(0, 0, 20.0 - 1e-4, (0, 0, 0), 12.0)) / 2e-4
    assert grad == pytest.approx(-25.93, abs=0.01)


def test_lattice_spheroid_exterior_against_the_separate_derivation():
    """The whole chain (cell means, kernel, image term, step taken out, interpolation) at points off the lattice
    around two spheroids, against exterior values from the confocal potential of a derivation that shares no code
    with this package. Points are (distance from the axis, height); B0 along the axis."""
    lat = F.Lattice.centred((128,) * 3, 2.0)
    cases = (((18.0, 36.0), ((0.0, 54.0, 0.079885), (27.0, 0.0, -0.088803))),         # prolate: on the axis at 1.5 c, on the equator at 1.5 a
             ((24.0, 12.0), ((0.0, 18.0, 0.278822), (36.0, 0.0, -0.070535))))         # oblate
    for (a, c), points in cases:
        w = V.spheroid(lat, a, c)
        fld = F.Field3D(w, lat, 3.0, tissue=1 - w, background_ppm=0.0)
        for rho, z, want in points:
            assert float(fld.ppm(rho, 0.0, z)) == pytest.approx(want, abs=3e-3)
        inside = float(fld.delta[V.spheroid(lat, a - 6, c - 6) > 0.999].mean())
        assert inside == pytest.approx(F.ellipsoid_inside((a, a, c), (0, 0, 1)), abs=2e-4)


# ------------------------------------------------------------------------------------------------ the surface
def test_the_step_is_taken_out_at_a_plane_interface_exactly():
    """A slab of air normal to B0, and one along it: the lattice field is kappa x chi with kappa = -2/3 and +1/3, and
    the one-sided field in the tissue is flat up to the surface, where plain interpolation ramps by half the step."""
    lat = F.Lattice.centred((64, 8, 8), 2.0)
    zc = lat.axes()[0]
    air = np.clip((zc - 10.0) / 2.0 + 0.5, 0, 1)[:, None, None] * np.ones((64, 8, 8))      # air above z = 10 mm: the surface cuts a cell in half
    args = dict(lattice=lat, b0_t=3.0, background_ppm=0.0, pad=1.0, images=False)
    for b, kappa in (((0, 0, 1), -2.0 / 3.0), ((1, 0, 0), 1.0 / 3.0)):
        good = F.Field3D(air, b0_dir=b, tissue=1 - air, **args)
        naive = F.Field3D(air, b0_dir=b, tissue=None, **args)
        z = np.linspace(-20.0, 9.99, 60)
        flat = good.ppm(0.0, 0.0, z)
        assert np.ptp(flat) < 1e-5
        assert naive.ppm(0.0, 0.0, 9.99) - flat[0] == pytest.approx(0.5 * kappa, abs=0.01)
        assert good.kappa[37, 4, 4] == pytest.approx(kappa, abs=1e-6)                # the cell the surface cuts


def test_near_surface_field_fails_with_plain_interpolation():
    """The guard of the specification's risk 2: tissue points within 1 mm of an air sphere. With the step taken out
    the field is that of the closed form; plain interpolation of the lattice is five times worse."""
    rng = np.random.default_rng(5)
    good, naive = V._surface_errors(12.0, True, rng, n=64)
    assert good < 0.03 and naive > 0.08 and naive > 4 * good


def test_chi_seen_from_the_plane_weights_sharpens_a_tissue_interface():
    """Bone against soft tissue, plane normal to B0: the field of each side is flat, and with the susceptibility of
    the material at each point the step sits on the surface, not smeared over a lattice cell."""
    lat = F.Lattice.centred((64, 8, 8), 2.0)
    zc = lat.axes()[0]
    bone = np.clip((zc - 1.0) / 2.0 + 0.5, 0, 1)[:, None, None] * np.ones((64, 8, 8))      # bone above z = 1 mm
    w = np.stack([1 - bone, bone]).astype(np.float32)
    fld = F.from_weights(w, (0.0, -2.0), lat, 3.0, pad=1.0, images=False)
    z = np.array([0.6, 0.9, 1.1, 1.4])
    seen = F.seen_chi(np.stack([z < 1.0, z >= 1.0]).astype(np.float32), (0.0, -2.0))
    sharp = fld.ppm(0.0, 0.0, z, chi_seen=seen)
    assert sharp[1] - sharp[0] == pytest.approx(0.0, abs=1e-4) and sharp[3] - sharp[2] == pytest.approx(0.0, abs=1e-4)
    assert sharp[2] - sharp[1] == pytest.approx(-2.0 * (-2.0 / 3.0), abs=1e-3)              # (1/3 - 1) x (-2 ppm)
    smeared = fld.ppm(0.0, 0.0, z)
    assert abs(smeared[2] - smeared[1]) < 0.3


def test_volume_fraction_is_a_cell_mean():
    lat = F.Lattice.centred((40,) * 3, 2.0)
    w = V.ball(lat, 13.0, (1.3, -0.4, 0.7))
    assert w.sum() * 8.0 == pytest.approx(4 / 3 * math.pi * 13.0 ** 3, rel=2e-3)
    assert 0.0 <= w.min() and w.max() <= 1.0 and ((w > 0) & (w < 1)).sum() > 500
    chi, tissue, chi_t = F.susceptibility(np.stack([0.7 * w, 0.2 * w]), (0.0, -2.0))
    np.testing.assert_allclose(chi, 0.2 * w * -2.0 + (1 - 0.9 * w) * F.DCHI_AIR_PPM, atol=1e-5)
    np.testing.assert_allclose(chi_t[w > 0.5], -2.0 * 0.2 / 0.9, atol=1e-5)
    assert F.lattice_or_closed_form(3.0, 2.0) == "closed form" and F.lattice_or_closed_form(4.0, (2.0, 2.0, 2.0)) == "lattice"


# ------------------------------------------------------------------------------------------------ planes
def test_sample_returns_the_fields_of_a_world_on_any_plane():
    lat = F.Lattice.centred((64,) * 3, 2.0)
    w = V.ball(lat, 12.0)
    fld = F.Field3D(F.DCHI_AIR_PPM * w, lat, 3.0, tissue=1 - w, background_ppm=0.0)
    hz = F.DCHI_AIR_PPM * fld.hz_per_ppm
    assert F.plane("sagittal", 22.0)[0] == (0.0, 22.0, 0.0) and F.plane("coronal", -22.0)[0] == (-22.0, 0.0, 0.0)
    for frame in (F.plane("axial", 22.0), F.plane("coronal", -22.0), F.plane("sagittal", 22.0),
                  ((3.0, -2.0, 20.0), (1.0, 0.0, 0.3), (0.0, 1.0, 0.0))):
        s = F.sample(fld, frame, 96, 96.0, 4.0, offsets=(-1.0, 0.0, 1.0))
        py, px, pz = F.plane_points(frame, 96, 96.0)
        assert s["b0_hz"].shape == s["dbdz_hz_per_mm"].shape == (96, 96) and s["stack"].shape == (3, 96, 96)
        r = np.sqrt(py ** 2 + px ** 2 + pz ** 2)
        ok = r > 18.0
        assert np.abs(s["b0_hz"] - hz * F.sphere_field(py, px, pz, (0, 0, 0), 12.0))[ok].max() < 8.0
        np.testing.assert_allclose(s["stack"][1], s["b0_hz"])
        n = F._frame(frame)[3]
        secant = hz * (F.sphere_field(py + 2 * n[0], px + 2 * n[1], pz + 2 * n[2], (0, 0, 0), 12.0)
                       - F.sphere_field(py - 2 * n[0], px - 2 * n[1], pz - 2 * n[2], (0, 0, 0), 12.0)) / 4.0
        assert np.abs(s["dbdz_hz_per_mm"] - secant)[ok].max() < 4.0
    # the engine's grid rule and the axial frame: rows along y, columns along x
    py, px, pz = F.plane_points(F.plane("axial", 5.0), 8, 16.0)
    assert py[3, 0] == -2.0 and px[0, 5] == 2.0 and np.all(pz == 5.0) and py[4, 4] == px[4, 4] == 0.0
    with pytest.raises(ValueError):
        F.sample(fld, ((0, 0, 0), (1, 0, 0), (1, 1, 0)), 8, 16.0, 4.0)
    with pytest.raises(ValueError):
        F.sample(fld, F.plane("axial"), 8, 16.0, 0.0)


def test_sample_accepts_the_frame_object_of_the_volume_world():
    class Frame:                                                         # the attributes of the specification's prototype
        o, r, c, n = np.array([0.0, 0.0, 4.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    lat = F.Lattice.centred((24,) * 3, 2.0)
    fld = F.Field3D(F.DCHI_AIR_PPM * V.ball(lat, 6.0), lat, 1.5, background_ppm=0.0)
    np.testing.assert_allclose(F.sample(fld, Frame, 16, 32.0, 3.0)["b0_hz"], F.sample(fld, F.plane("axial", 4.0), 16, 32.0, 3.0)["b0_hz"])


def test_the_seeded_volume_world_end_to_end():
    """The calls of docs/physics/fields3d.md section 11 on the 3D head of ``volume.py``: lattice, field, shim, and
    the two field entries of a slice world on its own frames."""
    from mrgym.scanner import volume
    vol = world.make_world("0123456789abcdef0123456789abcdef", dict(kind="head3d"))["volume"]
    L = vol.lattice(4.0, supersample=2)
    lat = F.Lattice(L.weights.shape[1:], L.spacing_mm, L.origin_mm)
    field = F.from_weights(L.weights, [vol.susceptibility_ppm[k] for k in vol.matter], lat, 3.0, extrude=("z-",))
    np.testing.assert_allclose(field.chi, vol.chi_ppm(L.weights), atol=1e-5)
    organ = vol.organ_mask(L.weights)
    rep = field.shim(organ, 1)
    assert rep["sd_after"] < rep["sd_before"] and rep["kkt"] < 1e-9
    for name in ("axial", "coronal", "sagittal"):
        frame = volume.Frame.plane(name, 10.0)
        weights = vol.eval_plane(frame, 96, 256.0)
        seen = F.seen_chi(weights, [vol.susceptibility_ppm[k] for k in vol.names])
        got = F.sample(field, frame, 96, 256.0, 5.0, chi_seen=seen)
        assert got["b0_hz"].shape == got["dbdz_hz_per_mm"].shape == weights.shape[1:]
        assert np.isfinite(got["b0_hz"]).all() and np.isfinite(got["dbdz_hz_per_mm"]).all()
        brain = weights[[vol.names.index(k) for k in vol.organ]].sum(0) > 0.5
        assert brain.sum() > 500 and got["b0_hz"][brain].std() < rep["sd_before"]
        py, px, pz = F.plane_points(frame, 96, 256.0)                    # the same grid points as the volume's frame
        qy, qx, qz = frame.points((np.arange(96) - 48) * 256.0 / 96, (np.arange(96) - 48) * 256.0 / 96)
        np.testing.assert_allclose([py, px, pz], np.broadcast_arrays(qy, qx, qz), atol=1e-9)


# ------------------------------------------------------------------------------------------------ shim
def test_shim_limits_bind_and_the_stored_shim_is_applied_once():
    lat = F.Lattice.centred((40,) * 3, 4.0)
    y, x, z = lat.mesh()
    w = V.ball(lat, 60.0)
    fld = F.Field3D(np.zeros(lat.shape, np.float32), lat, 3.0, tissue=w, background_ppm=0.0)
    voi = (np.sqrt(y * y + x * x + z * z) + 0 * w) < 50.0
    lim = F.shim_limits_hz()
    target = 0.5 * lim[1] * x + 3.0 * lim[4] * (z * z - 0.5 * (x * x + y * y)) + 0 * w       # needs three times the Z2 channel
    fld.smooth = (target / fld.hz_per_ppm).astype(np.float32)
    free = fld.shim(voi, 2, bounded=False)
    assert free["sd_after"] < 1e-3 * free["sd_before"] and free["at_limit"] == []
    np.testing.assert_allclose(fld.hz(10.0, -20.0, 30.0), 0.0, atol=1e-2 * free["sd_before"])
    held = fld.shim(voi, 2)
    assert held["at_limit"] == ["Z2"] and held["kkt"] < 1e-9 and held["sd_after"] > 100 * free["sd_after"]
    assert held["coef_ut"][4] == pytest.approx(F.SHIM_LIMITS_UT["Z2"], rel=1e-9)
    again = fld.shim(voi, 2)                                             # fitting twice gives the same absolute coefficients
    np.testing.assert_allclose(again["coef"], held["coef"], rtol=1e-12)
    first = fld.shim(voi, 1)
    assert len(first["coef"]) == 4 and np.all(fld.shim_coef[4:] == 0)
    with pytest.raises(ValueError):
        fld.shim(np.zeros(lat.shape, bool))
    with pytest.raises(ValueError):
        F.shim_basis(0.0, 0.0, 0.0, 3)


def test_shim_basis_follows_b0():
    """The harmonics are those of the magnet: with B0 along patient y the Z term is the patient y coordinate, and the
    nine functions span the same space as for B0 along z."""
    p = np.random.default_rng(1).uniform(-50, 50, (3, 300))
    tilted, upright = F.shim_basis(*p, 2, (1, 0, 0)), F.shim_basis(*p, 2)
    np.testing.assert_allclose(tilted[3], p[0])
    coef = np.linalg.lstsq(upright.T, tilted.T, rcond=None)[0]
    np.testing.assert_allclose(upright.T @ coef, tilted.T, atol=1e-8)


def test_engine_shim_is_reproduced():
    assert V.engine_shim_pair("fedcba9876543210fedcba9876543210") < 1e-9


# ------------------------------------------------------------------------------------------------ profiles, QSM
@pytest.mark.parametrize("kind", ["brain", "abdomen"])
def test_declared_profiles_continue_the_2d_world(kind):
    assert V.profile_pair(kind, "00112233445566778899aabbccddeeff") < 1e-12


def test_profiles_along_z_and_the_interface():
    rng, rng_z = np.random.default_rng(3), np.random.default_rng(4)
    prof = F.declared_profiles("brain", 3.0, (180.0, 150.0, 200.0), rng, rng_z, z_centre_mm=-20.0)
    y, x = np.meshgrid(np.linspace(-60, 60, 7), np.linspace(-50, 50, 7), indexing="ij")
    centre, off = prof.b1(y, x, -20.0), prof.b1(y, x, 40.0)
    assert np.abs(off - centre).max() > 0.02 and off[3, 3] < centre[3, 3]            # the central brightening falls off along z
    assert np.iscomplexobj(prof.rx(y, x, 0.0)) and np.all(prof.b1(y, x, 300.0) >= 0.3)
    flat = F.declared_profiles("brain", 3.0, (180.0, 150.0, 200.0), np.random.default_rng(3))
    np.testing.assert_allclose(flat.b1(y, x, 0.0), F.declared_profiles("brain", 3.0, (180.0, 150.0, 200.0), np.random.default_rng(3),
                                                                      np.random.default_rng(9)).b1(y, x, 0.0), atol=1e-12)
    on = prof.on_plane(F.plane("sagittal", 10.0), 16, 200.0)
    assert on["b1"].shape == on["rx"].shape == (16, 16)
    with pytest.raises(ValueError):
        F.declared_profiles("qa", 3.0, (1, 1, 1), rng)


def test_phase_and_truncated_division():
    assert F.to_phase(25.0, 10.0)[0] == pytest.approx(math.pi / 2)                   # +25 Hz for 10 ms: a quarter turn, positive
    assert F.to_phase(-25.0, [10.0, 30.0], wrap=False).tolist() == pytest.approx([-math.pi / 2, -3 * math.pi / 2])
    assert F.to_phase(0.0, 10.0, phase0=math.pi)[0] == pytest.approx(math.pi)        # the branch is (-pi, pi]
    assert abs(F.to_phase(75.0, 10.0)[0] + math.pi / 2) < 1e-12                      # three quarter turns wrap to minus a quarter
    assert F.tkd_factor(0.1) == pytest.approx(0.91290, abs=1e-5) and F.tkd_factor(0.2) == pytest.approx(0.82241, abs=1e-5)
    assert F.tkd_factor(1e-6) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError):
        F.tkd_factor(0.5)
    lat = F.Lattice.centred((48,) * 3, 2.0)
    w = V.ball(lat, 12.0).astype(float)
    field = F.dipole_field(w, 2.0, pad=1.0, background=0.0, images=False)
    y, x, z = lat.mesh()
    inside = (np.sqrt(y * y + x * x + z * z) + 0 * w) < 8.0
    low, high = (F.tkd(field, 2.0, threshold=t)[inside].mean() for t in (0.02, 0.2))
    assert 0.75 < high < low < 1.0                                                   # a higher threshold underestimates more


def test_validate_prints_and_returns(capsys):
    assert V.main(["shim"]) == 0
    out = capsys.readouterr().out
    assert "rows pass" in out and "F33" in out
    assert V.main(["shim", "--markdown"]) == 0
    assert "| F33 |" in capsys.readouterr().out
