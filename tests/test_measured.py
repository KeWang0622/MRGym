"""The comparison with measured data (``tools/measured/run_all.py``, ``docs/physics/measured.md``).

The stored table is current: it was computed by this engine, by this code and from these extracts, the page is the
page of the table, and the rungs that take seconds give the stored rows again. Declared differences have not grown.
Every cached extract carries its licence fields and the checksum of its original. The two defects the comparison found
outside the engine core are fixed: QA regions follow the phantom, and the NIST T2 plate with the published values of
both plates is in the package.
"""
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pytest

import mrgym
from mrgym import qa
from mrgym.scanner import anatomy, console, kspace
from mrgym.scanner import world as W

ROOT = pathlib.Path(__file__).resolve().parents[1]
STORE = ROOT / "docs" / "physics" / "measured"
TOOLS = ROOT / "tools" / "measured"
FIELDS = ("source", "url", "licence", "citation", "retrieved", "sha256", "sha256_of", "stored")


def _tool(name):
    """A module of tools/measured, loaded by path (the folder is put on the path once: its modules import each other)."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(f"measured_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def R():
    return _tool("run_all")


@pytest.fixture(scope="module")
def table():
    return json.loads((STORE / "results.json").read_text(encoding="utf-8"))


def _rows(table):
    return [q for r in table["rungs"] for q in r["rows"]]


# ---------------------------------------------------------------------------------------------- the stored table
def test_the_stored_table_is_current(R, table):
    assert table["engine"] == mrgym.ENGINE_ID, "the engine has changed: python tools/measured/run_all.py --write (M5 needs --fetch)"
    assert table["tool"] == R.tool_digest(), "tools/measured/ has changed since the table was stored: python tools/measured/run_all.py --write"
    assert table["extracts"] == R.extract_digests(), "a cached extract is not the one the table was computed from"
    assert [r["rung"] for r in table["rungs"]] == [name for name, _ in R.RUNGS] and all(r["rows"] for r in table["rungs"])
    assert all(r["engine"] == mrgym.ENGINE_ID for r in table["rungs"])
    m5 = next(r for r in table["rungs"] if r["rung"] == "M5")
    assert m5["fetch_only"] and m5["tissue_digest"] == R.tissue_digest(), "the tissue table has changed since M5 was fetched"
    assert table["known"] == R.known_list(table["rungs"]) and table["not_run"] == [dict(rung=a, what=b, state=c) for a, b, c in R.LADDER_NOT_RUN]
    assert table["sources"] == R.source_rows()
    page = _tool("page")
    assert (ROOT / "docs" / "physics" / "measured.md").read_text(encoding="utf-8") == page.markdown(table), "regenerate: python tools/generate.py"


def test_the_fast_rungs_give_the_stored_rows_again(R, table):
    """M0, M2b, M7 and M9 take seconds; the others are checked by ``run_all.py --check`` (minutes)."""
    stored = {r["rung"]: r["rows"] for r in table["rungs"]}
    fns = dict(R.RUNGS)
    for rung in ("M0", "M2b", "M7", "M9"):
        fresh = fns[rung]()["rows"]
        assert R.same_rows(stored[rung], fresh), rung
        assert [q["quantity"] for q in fresh] == [q["quantity"] for q in stored[rung]], rung


def test_every_row_says_what_was_measured_and_how_it_is_judged(table):
    rows = _rows(table)
    assert len({q["id"] for q in rows}) == len(rows) >= 50
    for q in rows:
        assert all(isinstance(q[k], str) and q[k] for k in ("id", "quantity", "measured", "engine", "difference", "tolerance")), q["id"]
        assert q["declared"] == (q["tolerance"] == "declared difference"), q["id"]
    failed = [q["id"] for q in rows if not q["declared"] and not q["passed"]]
    assert failed == [], "a row with a tolerance fails: it must be listed on the page and explained, not hidden"
    text = (ROOT / "docs" / "physics" / "measured.md").read_text(encoding="utf-8")
    for q in rows:                                                       # every row is on the page with its verdict
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| {q['id']} | "))
        assert line.endswith("| declared difference |" if q["declared"] else "| pass |"), q["id"]
    first = text.split("\n\n")[1]
    assert "**What is shown:** statistics of images and signals from public phantom sessions on real scanners" in first
    assert "**What is not shown:** a scan of this console's sequences on one physical object" in first


def test_declared_differences_have_not_grown(R, table):
    declared = [q for q in _rows(table) if q["declared"]]
    assert {q["id"] for q in declared} <= set(R.CEILINGS) and len(declared) >= 20
    for q in declared:
        assert q["ceiling"] == R.CEILINGS[q["id"]] and q["passed"] and q["size"] <= q["ceiling"], f"{q['id']} has grown: {q['size']} > {q['ceiling']}"
        assert q["size"] >= 0.5 * q["ceiling"] or q["size"] == 0.0, f"{q['id']} has shrunk to less than half its ceiling: lower the ceiling"
    # the twelve known differences of the research pass, and the two rung M9 added: each has a number read from rows that exist
    known = table["known"]
    assert [k["n"] for k in known] == list(range(1, 15)) and all(k["numbers"] and k["closes"] and k["what"] for k in known)
    ids = {q["id"] for q in _rows(table)}
    assert all(set(k["rows"]) <= ids for k in known)
    named = {q["known"] for q in declared if q["known"] is not None}
    assert named == set(range(1, 15)), "every known difference is carried by a declared row, and every such row names one"
    page = (ROOT / "docs" / "physics" / "measured.md").read_text(encoding="utf-8")
    section = page.split("## Known differences from measurement")[1].split("\n## ")[0]
    assert len([ln for ln in section.splitlines() if re.match(r"^\| \d+ \| ", ln)]) == 14


# ---------------------------------------------------------------------------------------------- the cached extracts
def test_every_cached_extract_has_its_licence_fields_and_its_checksum(R, table):
    S = _tool("sources")
    files = sorted(p.name for p in STORE.iterdir())
    assert files == sorted([*R.EXTRACTS.values(), R.SLICE_FILE, "results.json"])
    assert all((STORE / f).stat().st_size < 200_000 for f in files)                      # nothing large in the repository
    of = dict(nist_tables="stupic", rrsg="rrsg", acr="hazen", gel="gel", pdff="pdff", scholand="scholand")
    for name, file in R.EXTRACTS.items():
        x = json.loads((STORE / file).read_text(encoding="utf-8"))
        assert all(x.get(k) for k in FIELDS), file
        src = S.SOURCES[of[name]]
        assert (x["source"], x["url"], x["licence"], x["citation"]) == (src["source"], src["url"], src["licence"], src["citation"]), file
        assert x["sha256"] == src["sha256"], f"{file}: the checksum of the original is not the recorded one"
        sums = list(x["sha256"].values()) if isinstance(x["sha256"], dict) else [x["sha256"]]
        assert all(re.fullmatch(r"[0-9a-f]{64}", s) for s in sums), file
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", x["retrieved"]) and x["url"].startswith("https://"), file
    assert R.extract_digests() == table["extracts"] and set(table["extracts"]) == {*R.EXTRACTS.values(), R.SLICE_FILE}
    # what is kept follows the licence: an original only under CC0, CC BY or Apache-2.0; derived numbers where there is none; nothing where it is unclear
    licence = {name: json.loads((STORE / file).read_text(encoding="utf-8"))["licence"] for name, file in R.EXTRACTS.items()}
    assert licence["pdff"].startswith("CC0") and licence["gel"].startswith("CC BY") and licence["scholand"].startswith("CC BY")
    assert licence["acr"].startswith("Apache-2.0") and licence["nist_tables"].startswith("CC BY") and licence["rrsg"].startswith("none")
    rrsg = json.loads((STORE / R.EXTRACTS["rrsg"]).read_text(encoding="utf-8"))
    for part in ("all", "complex", "magnitude"):                                          # medians and quartiles of 14 spheres, no value of one submission
        assert set(rrsg[part]) == {"n", "error_percent_q25_median_q75", "in_roi_cov_percent_q25_median_q75"} and rrsg[part]["n"] >= 9
        assert np.array(rrsg[part]["error_percent_q25_median_q75"]).shape == (3, 14)
    assert S.SOURCES["itis"]["licence"].startswith("unclear") and S.SOURCES["itis"]["stored"].startswith("nothing")
    assert not [f for f in files if "itis" in f.lower()] and table["sources"][-1]["file"] is None
    # the one stored image: its checksum is in the extract, and it is the pixels only
    acr = json.loads((STORE / R.EXTRACTS["acr"]).read_text(encoding="utf-8"))
    assert hashlib.sha256((STORE / R.SLICE_FILE).read_bytes()).hexdigest() == acr["off_centre_slice"]["sha256"]
    with np.load(STORE / R.SLICE_FILE) as z:
        assert z.files == ["pixels"] and z["pixels"].dtype == np.uint16 and z["pixels"].shape == (256, 256)
    assert "mrgym" not in S.AGENT.lower()                                                 # no request names this project


def test_the_cfl_reader_reads_column_major_complex64(tmp_path):
    cfl = _tool("cfl")
    a = (np.arange(24) + 1j * np.arange(24)[::-1]).astype(np.complex64).reshape(2, 3, 4, order="F")
    (tmp_path / "x.hdr").write_text("# Dimensions\n2 3 4 1 1 \n", encoding="ascii")
    a.ravel(order="F").tofile(tmp_path / "x.cfl")
    b = cfl.read(str(tmp_path / "x"))
    assert b.shape == (2, 3, 4) and b.dtype == np.complex64 and np.array_equal(np.asarray(b), a)


# ---------------------------------------------------------------------------------------------- QA regions follow the phantom
def _regions_from_the_isocentre(img, fov):
    """The ghosting ratio and the line detector as they were before the fix: regions and bands from the isocentre."""
    y, x = qa.positions(img.shape, fov)
    w, ln, c = qa.BACKGROUND_ROI_MM[0] / 2, qa.BACKGROUND_ROI_MM[1] / 2, qa.BACKGROUND_AT_MM
    rects = dict(top=(-c - w, -c + w, -ln, ln), bottom=(c - w, c + w, -ln, ln), left=(-ln, ln, -c - w, -c + w), right=(-ln, ln, c - w, c + w))
    m = {k: float(img[np.ix_((y >= r[0]) & (y <= r[1]), (x >= r[2]) & (x <= r[3]))].mean()) for k, r in rects.items()}
    centre = qa.find_centre(img, fov, 0.5 * qa.water_level(img))
    large = float(img[np.hypot(y[:, None] - centre[0], x[None, :] - centre[1]) <= np.sqrt(qa.LARGE_ROI_CM2 * 100 / np.pi)].mean())
    edge, best = qa.A["diameter_mm"] / 2 + 12.0, 0.0
    for prof in (img[np.abs(y) > edge].mean(0), img[:, np.abs(x) > edge].mean(1)):
        d = np.minimum(prof[3:-3] - prof[:-6], prof[3:-3] - prof[6:])
        best = max(best, float((d.max() - np.median(d)) / max(1.4826 * np.median(np.abs(d - np.median(d))), 1e-30)))
    return abs((m["top"] + m["bottom"]) - (m["left"] + m["right"])) / (2.0 * large), best


def test_on_the_real_off_centre_image_the_ghosting_ratio_is_the_hand_computed_one(R):
    acr = json.loads((STORE / R.EXTRACTS["acr"]).read_text(encoding="utf-8"))
    with np.load(STORE / R.SLICE_FILE) as z:
        img = z["pixels"].astype(float)
    fov = tuple(acr["off_centre_slice"]["fov_mm"])
    centre = qa.find_centre(img, fov, 0.5 * qa.water_level(img))
    assert 20.0 < np.hypot(*centre) < 22.0                                               # the phantom lies 21 mm from the centre of the image
    px = fov[0] / img.shape[0]
    shift = qa.follow(img.shape, fov, centre)
    assert shift == pytest.approx(tuple(round(c / px) * px for c in centre)) and all(abs(s - c) <= px / 2 for s, c in zip(shift, centre))
    g = qa.ghosting(img, fov)
    assert round(g["ratio"], 4) == 0.0016 and g["ratio"] < qa.LIMITS["ghosting_ratio"]   # the figure the research pass found by re-centring the image by hand
    before, _ = _regions_from_the_isocentre(img, fov)
    assert round(before, 2) == 0.18 and before > qa.LIMITS["ghosting_ratio"]             # the regions from the isocentre: one lies on the phantom, a healthy scanner fails
    # the same on the image rolled so that the phantom is central: the measurement does not depend on where the phantom lies
    rolled = np.roll(img, (-round(centre[0] / px), -round(centre[1] / px)), axis=(0, 1))
    assert _regions_from_the_isocentre(rolled, fov)[0] == pytest.approx(g["ratio"], rel=1e-9)
    z = qa.artifact_line(img, fov)["z"]
    assert z == pytest.approx(qa.artifact_line(rolled, fov)["z"], rel=1e-9) and z < qa.LIMITS["artifact_z"]
    stored = acr["series"][acr["off_centre_slice"]["series"]]
    assert stored["ghost"] == pytest.approx(g["ratio"], rel=1e-5) and stored["ghost_regions_from_isocentre"] == pytest.approx(before, rel=1e-5)


def test_results_on_phantoms_placed_within_the_tolerance_do_not_move():
    """A phantom within ``BACKGROUND_KEEP_MM`` of the isocentre keeps the regions of the guidance, so every number of
    a world drawn within its 1 mm positioning error is the number it was: bit for bit, on engine images."""
    fov = (250.0, 250.0)
    assert qa.follow((256, 256), fov, (0.0, 0.0)) == (0.0, 0.0) and qa.follow((256, 256), fov, (2.4, -2.5)) == (0.0, 0.0)
    assert qa.follow((256, 256), fov, None) == (0.0, 0.0) and qa.follow((256, 256), fov, (2.6, -2.5)) != (0.0, 0.0)
    assert qa.background_rois(fov, (0.0, 0.0)) == qa.background_rois(fov) == dict(
        top=(-118.0, -102.0, -31.5, 31.5), bottom=(102.0, 118.0, -31.5, 31.5), left=(-31.5, 31.5, -118.0, -102.0), right=(-31.5, 31.5, 102.0, 118.0))
    assert anatomy.ACR_SPEC["offset_mm"] * np.sqrt(2) < qa.BACKGROUND_KEEP_MM            # every drawn position lies inside
    for field, seed in ((1.5, 0), (3.0, 1)):
        sp, grads = qa.system_spec(field, seed)
        img, f, _ = qa.scan(qa.engine_of(sp, grads, 7, seed), "t1", 7, seed)
        before = _regions_from_the_isocentre(img, f)
        assert qa.ghosting(img, f)["ratio"] == before[0] and qa.artifact_line(img, f)["z"] == before[1], (field, seed)


FLAT = dict(b1_amp={1.5: 0.0, 3.0: 0.0}, rx_bowl={1.5: 0.0, 3.0: 0.0}, tilt=0.0, b0_ppm=0.0)      # fields are fixed to the scanner: a moved phantom sees other values


def _acr_measures(seed, offset_mm, noise_seed):
    sp, grads = qa.system_spec(3.0, seed)
    out = {}
    for section in (7, 1):
        spec = dict(qa.world_spec(dict(sp, acr_fields=FLAT), grads, section), acr=dict(slice=section, offset_mm=offset_mm))
        w = W.make_world(qa.seed_hex(seed), spec)
        img, fov, _ = qa.scan(kspace.Engine(w), "t1", section, noise_seed)
        if section == 7:
            g = qa.ghosting(img, fov)
            out.update(centre=w["layout"]["centre_mm"], ghost=g["ratio"], piu=qa.uniformity(img, fov)["piu_percent"], snr=g["large_mean"] / qa.noise_sd(img, fov),
                       z=qa.artifact_line(img, fov)["z"], d7=qa.diameters(img, fov, angles=(90.0, 0.0, 45.0, 135.0))["lengths_mm"])
        else:
            r = qa.resolution(img, fov)
            out.update(d1=qa.diameters(img, fov)["lengths_mm"], res=(r["left_right"]["resolved_mm"], r["top_bottom"]["resolved_mm"]))
    return out


def test_the_acr_measures_of_a_shifted_phantom_equal_those_of_the_centred_one():
    """The same system and the same noise, the phantom once within 1 mm and once up to 25 mm from the isocentre (flat
    fields, so that the object is the only thing that moves). Yardstick: a second noise realisation of the centred phantom."""
    seed = 0
    a, again, b = _acr_measures(seed, 1.0, seed), _acr_measures(seed, 1.0, seed + 100), _acr_measures(seed, 25.0, seed)
    assert np.hypot(*a["centre"]) < 1.5 and 20.0 < np.hypot(*b["centre"]) < 36.0 and max(abs(c) for c in b["centre"]) <= 25.0
    noise = dict(ghost=abs(a["ghost"] - again["ghost"]), piu=abs(a["piu"] - again["piu"]), snr=abs(a["snr"] / again["snr"] - 1))
    assert abs(b["piu"] - a["piu"]) <= max(3 * noise["piu"], 0.05)                       # percent integral uniformity, points
    assert abs(b["snr"] / a["snr"] - 1) <= max(3 * noise["snr"], 0.01)
    # ghosting: both far below the limit, and no further apart than healthy systems are from each other (0.0002 to 0.0004 at this noise)
    assert max(a["ghost"], b["ghost"]) < 0.001 < qa.LIMITS["ghosting_ratio"] and abs(b["ghost"] - a["ghost"]) <= max(3 * noise["ghost"], 0.0005)
    assert max(a["z"], b["z"]) < qa.LIMITS["artifact_z"]
    assert np.abs(np.array(b["d7"]) - a["d7"]).max() <= 0.15 and np.abs(np.array(b["d1"]) - a["d1"]).max() <= 0.15      # mm: an edge at another place in its pixel
    assert b["res"] == a["res"] == (1.0, 1.0)


# ---------------------------------------------------------------------------------------------- the NIST plates from the published tables
def test_the_nist_arrays_are_the_cached_table(R):
    t = json.loads((STORE / R.EXTRACTS["nist_tables"]).read_text(encoding="utf-8"))
    assert [list(r) for r in anatomy.NIST_S1] == t["table_s1_nicl2"] and [list(r) for r in anatomy.NIST_S2] == t["table_s2_mncl2"]
    assert [list(p) for p in anatomy.NIST_XZ_MM] == t["xz_mm"] and anatomy.NIST_PLATE_Y_MM == t["plate_y_mm"]
    assert {str(k): list(v) for k, v in anatomy.NIST_WATER_RATE_PER_S.items()} == t["table_s4_water_rate_per_s"]
    s1, s2 = np.array(t["table_s1_nicl2"]), np.array(t["table_s2_mncl2"])
    assert s1.shape == s2.shape == (14, 4) and np.all(np.diff(s1, axis=0) < 0) and np.all(np.diff(s2, axis=0) < 0)      # every column falls from sphere 1 to 14
    for table, ref in ((anatomy.NIST_T1_PUBLISHED, s1), (anatomy.NIST_T2, s2)):
        assert np.array_equal(table["t1_ms"][1.5], ref[:, 0]) and np.array_equal(table["t2_ms"][1.5], ref[:, 1])
        assert np.array_equal(table["t1_ms"][3.0], ref[:, 2]) and np.array_equal(table["t2_ms"][3.0], ref[:, 3])
    assert np.all(s1[:, 1] < s1[:, 0]) and np.all(s2[:, 3] < s2[:, 2])                    # T2 below T1 in every sphere
    # the older world is not changed here: its list still has 496.0 where the table has 496.7 (row M0.2), the other 13 are the table
    assert anatomy.NIST_T1["t1_ms"] == (1989.0, 1454.0, 984.1, 706.0, 496.0, 351.5, 247.13, 175.3, 125.9, 89.0, 62.7, 44.53, 30.84, 21.719)
    assert [k for k in range(14) if anatomy.NIST_T1["t1_ms"][k] != s1[k, 2]] == [4] and s1[4, 2] == 496.7
    assert anatomy.NIST_T1["t2_ms_sphere_12"] == s1[11, 3] and anatomy.NIST_T1["fill"]["t1_ms"] == 2800.0


def test_the_nist_t2_plate_holds_the_published_values_at_both_fields():
    for b0 in (1.5, 3.0):
        ph = anatomy.nist_t2(np.random.default_rng(0), b0, n=320)
        assert ph["kind"] == "nist_t2" and ph["names"] == ["fill"] + [f"sphere{i}" for i in range(1, 15)] and ph["weights"].shape == (15, 320, 320)
        assert [ph["tissues"][f"sphere{i + 1}"]["t1_ms"] for i in range(14)] == list(anatomy.NIST_T2["t1_ms"][b0])
        assert [ph["tissues"][f"sphere{i + 1}"]["t2_ms"] for i in range(14)] == list(anatomy.NIST_T2["t2_ms"][b0])
        r1, r2 = anatomy.NIST_WATER_RATE_PER_S[b0]
        assert ph["tissues"]["fill"]["t1_ms"] == pytest.approx(1e3 / r1) and ph["tissues"]["fill"]["t2_ms"] == pytest.approx(1e3 / r2)
        assert ph["size_mm"][0] == pytest.approx(2 * np.sqrt(100.0 ** 2 - 16.5 ** 2)) and [r["id"] for r in ph["regions"]] == [f"T2-{i}" for i in range(1, 15)]
        oy, ox = ph["layout"]["centre_mm"]
        assert abs(oy) <= 1.0 and abs(ox) <= 1.0
        for r, (x, z) in zip(ph["regions"], anatomy.NIST_XZ_MM):                         # image y = -Z, x = X, within the drawn offset and 0.3 degrees
            assert np.hypot(r["y_mm"] - oy + z, r["x_mm"] - ox - x) < 0.35 and r["radius_mm"] == 5.0
    assert ph["regions"][0]["y_mm"] < -48 and abs(ph["regions"][0]["x_mm"]) < 2            # sphere 1 at 12 o'clock
    same = anatomy.nist_t2(np.random.default_rng(1), 3.0, n=320, spec=dict(offset_mm=0.0, rotation_deg=0.0))
    assert [(r["y_mm"], r["x_mm"]) for r in same["regions"]] == [(-z, x) for x, z in anatomy.NIST_XZ_MM]
    with pytest.raises(ValueError, match="1.5 T and 3 T"):
        anatomy.nist_t2(np.random.default_rng(0), 7.0)


def test_the_scanner_images_the_nist_t2_plate_with_its_t2():
    """The plate in a world with ideal fields (``world.py`` does not list the kind yet): two spin echoes give the T2 of the table."""
    ph = anatomy.nist_t2(np.random.default_rng(0), 3.0, spec=dict(offset_mm=0.0, rotation_deg=0.0))
    w = W.flat_world(ph)
    eng = kspace.Engine(w)
    sig = []
    for te in (20.0, 100.0):
        pl = console.plan(dict(sequence="se", tr_ms=8000.0, te_ms=te, fov_mm=250.0, matrix=128, slice_mm=5.0), dict(w["console"], fov_mm=250.0))
        assert pl["accept"], pl["violations"]
        img = np.abs(eng.acquire(pl["realised"], None)[0])
        y, x = qa.positions(img.shape, (250.0, 250.0))
        sig.append([img[np.hypot(y[:, None] - r["y_mm"], x[None, :] - r["x_mm"]) <= 4.0].mean() for r in ph["regions"]])
    t2 = 80.0 / np.log(np.array(sig[0]) / np.array(sig[1]))
    want = np.array(anatomy.NIST_T2["t2_ms"][3.0])
    assert np.abs(t2[:6] / want[:6] - 1).max() < 0.01                                     # spheres 1 to 6: T2 97 ms and longer, both echoes far above the ringing of the water
