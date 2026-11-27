"""Receive arrays (``scanner/array.py``) and the reconstruction mathematics (``scanner/recon.py``).

The validation table (``recon.validate()``, docs/physics/array.md) is run group by group; the other tests pin what
the table does not: a world without the key is the world of before, the array is a function of (seed, spec), the
single-channel scanner of an array world is untouched, the fast formation holds across sequences and geometries.
"""
import os
import pickle
import re

import numpy as np
import pytest

import golden_cases as G
from mrgym.scanner import array as A
from mrgym.scanner import array_checks, console, kspace, recon
from mrgym.scanner import world as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = "000102030405060708090a0b0c0d0e0f"
_rows = {}


def rows(group):
    if group not in _rows:
        _rows[group] = array_checks.CHECKS[group]()
    return _rows[group]


@pytest.fixture(scope="module")
def brain():
    return W.make_world(SEED, dict(kind="brain", n=320, array=dict(n=4)))


def _all_arrays(w, prefix="w"):
    out = {}
    for k, v in w.items():
        if isinstance(v, np.ndarray):
            out[f"{prefix}.{k}"] = v.tobytes()
        elif isinstance(v, dict) and k != "array":
            out.update(_all_arrays(v, f"{prefix}.{k}"))
        elif k not in ("array", "source"):
            out[f"{prefix}.{k}"] = repr(v)
    return out


# ---------------------------------------------------------------------------------------------- the validation table
@pytest.mark.parametrize("group", list(array_checks.CHECKS))
def test_every_row_of_the_validation_table_passes(group):
    # T1 is a ratio of two timings on a shared machine: the table holds the design target (1.5), the test
    # fails only on a gross regression
    bad = [(r["id"], r["measured"], r["tolerance"]) for r in rows(group) if not r["ok"] and not (r["id"] == "T1" and r["measured"] <= 2.5)]
    assert not bad, bad


def test_row_ids_are_unique_and_groups_known():
    ids = [r["id"] for g in array_checks.CHECKS for r in rows(g)]
    assert len(ids) == len(set(ids)) and set(array_checks.CHECKS) == set(array_checks.GROUPS)


def test_the_document_shows_the_table_that_the_command_prints():
    """Quantity, reference, tolerance and verdict of every row of docs/physics/array.md are those of
    ``python -m mrgym.scanner.array_checks --markdown``; the column of found numbers differs between machines."""
    path = os.path.join(ROOT, "docs", "physics", "array.md")
    if not os.path.exists(path):
        pytest.skip("the tests are not running from a checkout of the repository")
    with open(path, encoding="utf-8") as f:
        block = f.read().split("<!-- validation table: begin -->")[1].split("<!-- validation table: end -->")[0]

    def strip(md):
        out = []
        for line in md.strip().splitlines():
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)]
            if len(cells) == 8 and re.fullmatch(r"[A-Z]\d+", cells[1]):
                out.append((cells[1], cells[2], cells[3], cells[5], "pass" if cells[1] == "T1" else cells[6]))
        return out

    made = strip(array_checks.markdown_table([r for g in array_checks.CHECKS for r in rows(g)]))
    assert strip(block) == made
    assert made and all(row[4] in ("pass", "") for row in made)


def test_a_failing_row_is_reported_as_failing():
    assert not array_checks._row("X1", "sense", "q", "r", 2e-3, 1e-3)["ok"]
    assert not array_checks._row("X2", "sense", "q", "r", 0.5, 1.0, kind="min")["ok"]
    assert not array_checks._row("X3", "sense", "q", "r", float("nan"), 1.0)["ok"]
    assert "FAIL" in array_checks.markdown_table([array_checks._row("X1", "sense", "q", "r", 2e-3, 1e-3)])


# ---------------------------------------------------------------------------------------------- the additive key
def test_a_world_without_the_key_has_no_array_and_the_key_changes_nothing_else(brain):
    plain = W.make_world(SEED, dict(kind="brain", n=320))
    assert "array" not in plain and "array" not in W.DEFAULT_SPEC
    assert A.public(plain) == dict(channels=1) and A.public(brain) == dict(channels=4)
    a, b = _all_arrays(plain), _all_arrays(brain)
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)
    with pytest.raises(ValueError):
        A.ArrayEngine(plain)


def test_the_single_channel_scanner_of_an_array_world_is_untouched(brain):
    plain = W.make_world(SEED, dict(kind="brain", n=320))
    rp = console.plan(G.PROTOCOLS["spgr"], plain["console"])["realised"]
    a = kspace.Engine(plain).acquire(rp, np.random.default_rng(3))
    b = kspace.Engine(brain).acquire(rp, np.random.default_rng(3))
    assert a.tobytes() == b.tobytes()


def test_the_array_is_a_function_of_seed_and_spec_and_a_fault_changes_its_own_term():
    ph = dict(size_mm=(180.0, 140.0))
    y, x = np.meshgrid(np.linspace(-60, 60, 7), np.linspace(-50, 50, 7), indexing="ij")

    def make(seed, **spec):
        return A.make_array(dict(n=8, **spec), ph, W.rng_for(seed, "fields", 2))

    a, b, other = make(SEED), make(SEED), make("0123456789abcdef0123456789abcdef")
    assert np.array_equal(a["maps"](y, x), b["maps"](y, x)) and np.array_equal(a["psi"], b["psi"])
    assert not np.allclose(a["maps"](y, x), other["maps"](y, x))
    assert np.linalg.eigvalsh(a["psi"]).min() > 0 and np.allclose(a["psi"], a["psi"].conj().T)
    snr0 = 1 / recon.roemer_noise(a["maps"](np.zeros((1, 1)), np.zeros((1, 1))), a["psi"])[0, 0]
    assert snr0 == pytest.approx(1.0, abs=1e-12)                        # centre_snr: the body coil's sensitivity at the isocentre
    k = make(SEED, faults=dict(dead=True))["faults"]["dead"][0]         # which element dies is drawn
    assert make(SEED, faults=dict(dead=True, noisy=True))["faults"]["dead"] == [k]
    n, g = [i for i in range(8) if i != k][:2]
    dead = make(SEED, faults=dict(dead=True, noisy=dict(element=n, factor=9.0), gain=dict(element=g, gain=1.3)))
    keep = [i for i in range(8) if i not in (k, n, g)]
    assert np.all(dead["maps"](y, x)[k] == 0) and np.array_equal(dead["maps"](y, x)[keep], a["maps"](y, x)[keep])
    assert np.allclose(dead["psi"][np.ix_(keep, keep)], a["psi"][np.ix_(keep, keep)])
    others = [i for i in range(8) if i != k]
    assert np.all(dead["psi"][k, others] == 0) and dead["psi"][k, k] == pytest.approx(a["psi"][k, k])
    assert dead["psi"][n, n].real == pytest.approx(9.0 * a["psi"][n, n].real)
    assert np.allclose(dead["maps"](y, x)[g], 1.3 * a["maps"](y, x)[g]) and dead["psi"][g, g].real == pytest.approx(1.69 * a["psi"][g, g].real)
    drawn = make(SEED, faults=dict(noisy=True, gain=True))["faults"]
    assert A.NOISY_FACTOR[0] <= list(drawn["noisy"].values())[0] <= A.NOISY_FACTOR[1]
    assert A.GAIN_ERROR[0] <= abs(list(drawn["gain"].values())[0] - 1) <= A.GAIN_ERROR[1]
    assert dead["healthy"]["maps"] is a["maps"] or np.array_equal(dead["healthy"]["maps"](y, x), a["maps"](y, x))
    with pytest.raises(ValueError):
        make(SEED, layout="spiral")
    with pytest.raises(ValueError):
        make(SEED, loops=8)


def test_layouts_and_hidden_choices():
    ph = dict(size_mm=(180.0, 140.0))
    sides = {A.make_array(dict(n=4, layout="row"), ph, W.rng_for(f"{i:032x}", "fields", 2))["side"] for i in range(24)}
    assert sides == set(A.SIDES)                                         # which side the row is on is drawn
    row = A.make_array(dict(n=4, layout="row", side="posterior"), ph, W.rng_for(SEED, "fields", 2))
    assert all(e["y_mm"] > 90 for e in row["elements"]) and len({round(e["x_mm"], -1) for e in row["elements"]}) == 4
    pair = A.make_array(dict(n=8, layout="pair", side="left"), ph, W.rng_for(SEED, "fields", 2))
    assert sum(e["x_mm"] > 0 for e in pair["elements"]) == 4 and sum(e["x_mm"] < 0 for e in pair["elements"]) == 4
    drawn = {A.make_array(dict(n=4, layout=["ring", "row"]), ph, W.rng_for(f"{i:032x}", "fields", 2))["layout"] for i in range(16)}
    assert drawn == {"ring", "row"}


def test_the_array_survives_a_turned_grid_and_pickling(brain):
    assert W.turned(brain, 30.0)["array"] is brain["array"]
    arr = pickle.loads(pickle.dumps(brain["array"]))
    y, x = np.meshgrid(np.linspace(-60, 60, 5), np.linspace(-50, 50, 5), indexing="ij")
    assert np.array_equal(arr["maps"](y, x), brain["array"]["maps"](y, x))


def test_maps_given_from_outside_on_a_grid(brain):
    c = np.linspace(-120.0, 120.0, 121)
    grid = brain["array"]["maps"](c[:, None], c[None, :])
    arr = A.given(grid, np.eye(4), y_mm=c, x_mm=c)
    y, x = np.meshgrid(np.linspace(-70, 70, 9), np.linspace(-55, 55, 9), indexing="ij")
    assert np.abs(arr["maps"](y, x) - brain["array"]["maps"](y, x)).max() < 1e-4 * np.abs(grid).max()
    assert A.ArrayEngine(A.attach(brain, arr)).n_coils == 4 and "array" in brain
    with pytest.raises(ValueError):
        A.given(grid, np.ones((4, 4)), y_mm=c, x_mm=c)                  # not positive definite


def test_golden_array_section():
    """The additive section ``array`` of tests/golden/engine_digests.json (``golden_cases.array_cases``): bit for bit
    on the recorded stack, fingerprints within the replay tolerance elsewhere."""
    from mrgym.scanner import session
    golden = G.load(G.ENGINE_FILE)
    assert golden and "array" in golden, "run `python tools/audit/golden.py --write`"
    mode = G.mode(golden["stack"])[0]
    got, want = G.array_cases(), golden["array"]
    assert sorted(got) == sorted(want)
    for key in got:
        if key == "psi":
            assert np.allclose(got[key], want[key], rtol=0, atol=1e-9 if mode == "bit-exact" else 1e-6)
            continue
        assert got[key]["shape"] == want[key]["shape"], key
        if mode == "bit-exact":
            assert got[key]["sha256"] == want[key]["sha256"], key
        else:
            assert session.fingerprints_close(got[key]["fingerprint"], want[key]["fingerprint"]), key


# ---------------------------------------------------------------------------------------------- formation
FORMATION = dict(
    spgr_two_echoes=dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0, matrix=64, n_echoes=2, ro_polarity=-1),
    se_swapped_shifted=dict(sequence="se", tr_ms=500.0, matrix=64, pe_dir="x", centre_x_mm=12.0, phase_fov=0.75),
    se_turned=dict(sequence="se", tr_ms=500.0, matrix=64, rotation_deg=30.0, centre_y_mm=-8.0),
    ir_fatsat=dict(sequence="ir", tr_ms=1500.0, ti_ms=400.0, matrix=64, fat_sat=True),
    # engine 0.3.0 saturates the lines of fat one by one: each line has its own amplitude map and is formed by itself
    spgr_fatsat=dict(sequence="spgr", tr_ms=30.0, flip_deg=20.0, matrix=64, n_echoes=2, fat_sat=True),
    se_fatsat_retuned=dict(sequence="se", tr_ms=500.0, matrix=64, fat_sat=True, fatsat_offset_hz=40.0),
    bssfp_oversampled=dict(sequence="bssfp", tr_ms=8.0, flip_deg=30.0, matrix=64, phase_oversampling=0.25),
    afi_partial=dict(sequence="afi", tr_ms=25.0, flip_deg=55.0, matrix=64, partial_fourier=0.75),
)


@pytest.mark.parametrize("name", list(FORMATION))
def test_fast_formation_against_per_coil(brain, name):
    ae = A.ArrayEngine(brain)
    rp = console.plan(FORMATION[name], brain["console"])["realised"]
    assert ae.method_for(rp) == "fast"
    assert ae.formation_error(rp) <= 5e-3
    assert ae.clean_kspace(rp).shape == (len(kspace.Engine(brain).clean_kspace(rp)), 4, rp["lines_full"], 2 * rp["matrix"])


def test_epi_and_moving_worlds_are_formed_per_coil(brain):
    ae = A.ArrayEngine(brain)
    rp = console.plan(G.PROTOCOLS["epi"], brain["console"])["realised"]
    assert ae.method_for(rp) == "per_coil"
    ks = ae.clean_kspace(rp)
    j = 2
    ref = kspace.Engine(dict(brain, fields=dict(brain["fields"], rx=brain["array"]["maps"](*W.grid(brain))[j]))).clean_kspace(rp)
    assert np.array_equal(ks[:, j], ref)
    moving = W.make_world(SEED, dict(kind="abdomen", n=160, motion=[dict(cls="*", kind="shift")], array=dict(n=2)))
    rp = console.plan(G.PROTOCOLS["spgr"], moving["console"])["realised"]
    assert A.ArrayEngine(moving).method_for(rp) == "per_coil"


def test_noise_is_drawn_channel_by_channel_in_the_receivers_order(brain):
    ae = A.ArrayEngine(brain)
    rp = console.plan(dict(G.PROTOCOLS["spgr"], n_echoes=1), brain["console"])["realised"]
    nz = ae.noise(rp, np.random.default_rng(11))
    rng = np.random.default_rng(11)
    shape = (rp["lines_full"], 2 * rp["matrix"])
    first = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    assert np.allclose(nz[0], ae.sample_sd(rp) * ae.chol[0, 0] * first, rtol=1e-13, atol=0)       # channel 0 = the single-channel draws
    lines = recon.sampling_mask(rp["lines_full"], 2)
    full = ae.acquire_kspace(rp, np.random.default_rng(5))
    half = ae.acquire_kspace(rp, np.random.default_rng(5), lines)
    assert np.array_equal(half[:, :, lines], full[:, :, lines]) and not half[:, :, ~lines].any()   # the same stream, fewer lines
    assert ae.acquire_kspace(rp).tobytes() == ae.clean_kspace(rp).tobytes()                        # no generator, no noise


def test_the_verifiers_noise_map_at_full_sampling_is_the_matched_filters(brain):
    ae = A.ArrayEngine(brain)
    rp = console.plan(G.PROTOCOLS["se"], brain["console"])["realised"]
    sd = ae.noise_sd_map(rp)
    assert sd.shape == (rp["matrix"], rp["matrix_pe"])                  # pe_dir x: [y, x] with the phase direction along x
    expect = ae.image_sd(rp) * ae.to_image(recon.roemer_noise(ae.pixel_maps(rp), ae.psi), ae.engine._frame(rp)[1])
    assert np.allclose(sd, expect, rtol=1e-12)


# ---------------------------------------------------------------------------------------------- recon, without the engine
def test_sampling_masks_and_their_reading():
    for n, r, acs, off in ((96, 2, 0, 0), (96, 3, 24, 1), (64, 4, 12, 3), (48, 1, 0, 0)):
        lines = recon.sampling_mask(n, r, acs, off)
        assert lines[n // 2] or off
        got = recon.acceleration_of(lines)
        assert got[:2] == (r, off % r) and (got[2] >= acs if r > 1 else got[2] == 0)
    assert recon.sampling_mask(96, 2).sum() == 48 and recon.sampling_mask(96, 3, 24).sum() == 32 + 16
    with pytest.raises(ValueError):
        recon.acceleration_of(np.array([1, 1, 0, 1, 0, 0, 1, 0], bool))
    ks = np.arange(2 * 12 * 4).reshape(2, 12, 4).astype(complex)
    block = recon.calibration_block(ks, recon.sampling_mask(12, 3, 4))
    assert block.shape == (2, 3, 4) and np.array_equal(block, ks[:, 5:8])     # symmetric about k = 0 (index 6)
    assert np.array_equal(recon.resize_k(recon.resize_k(ks, (16, 8)), (12, 4)), ks)
    x = np.random.default_rng(0).standard_normal((3, 8, 6)) + 0j
    assert np.allclose(recon.ifft2c(recon.fft2c(x)), x)


def test_sense_plan_refuses_what_it_cannot_unfold_and_leaves_out_what_is_empty():
    rng = np.random.default_rng(4)
    maps = rng.standard_normal((4, 12, 5)) + 1j * rng.standard_normal((4, 12, 5))
    with pytest.raises(ValueError):
        recon.sense_plan(maps, None, 5)                                 # 5 does not divide 12
    with pytest.raises(ValueError):
        recon.sense_plan(maps, None, 6)                                 # 6 folded pixels, 4 channels
    support = np.ones((12, 5), bool)
    support[:6] = False                                                 # the object fills half the field of view: R = 2 folds nothing
    plan = recon.sense_plan(maps, None, 2, support=support)
    assert np.isnan(plan["g"][:6]).all() and np.allclose(plan["g"][6:], 1.0)
    obj = (rng.standard_normal((12, 5)) + 0j) * support
    lines = recon.sampling_mask(12, 2)
    assert np.allclose(recon.sense(recon.fft2c(maps * obj) * lines[:, None], plan), obj)
    reg = recon.sense_plan(maps, None, 4, reg=0.1)
    assert np.all(reg["g_reg"] < recon.sense_plan(maps, None, 4)["g"])
    with pytest.raises(NotImplementedError):
        recon.grappa(maps, lines)


def test_maps_estimated_from_low_resolution_data_of_a_smooth_array():
    n = 64
    c = (np.arange(n) - n // 2) / n
    y, x = np.meshgrid(c, c, indexing="ij")
    maps = np.array([np.exp(-((y - a) ** 2 + (x - b) ** 2) / 0.5) * np.exp(1j * (a * 3 * y + b * 2 * x)) for a, b in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5))])
    obj = ((y * y + x * x) < 0.16) * (1 + 0.5 * np.cos(9 * x)) * np.exp(0.4j * y)
    body = 1 + 0.2 * y
    cal = recon.resize_k(recon.fft2c(maps * obj), (24, 24))
    est, support = recon.estimate_maps(cal, recon.resize_k(recon.fft2c(body * obj), (24, 24)), (n, n))
    inside = (y * y + x * x) < 0.12
    assert support[inside].all()
    assert np.abs(est - maps / body)[:, inside].max() < 0.03 * np.abs(maps).max()
    est, _ = recon.estimate_maps(cal, None, (n, n))                     # no body coil: relative to the root sum of squares
    assert np.abs(np.sqrt((np.abs(est) ** 2).sum(0)) - 1)[inside].max() < 0.03


def test_chi_moments_of_one_channel_are_rayleigh():
    mean, sd = recon.chi_moments(1, 2.0)
    assert mean == pytest.approx(2.0 * np.sqrt(np.pi / 2)) and sd == pytest.approx(2.0 * np.sqrt(2 - np.pi / 2))
