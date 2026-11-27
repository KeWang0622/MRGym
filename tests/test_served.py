"""The sequence registry and the sequences a console serves from the block-list core (``scanner/served.py``).

What is pinned here, next to the validation rows of ``mrgym validate`` (group ``core``), which this file also runs:
  * a console whose card lists no sequences is the console of before: tables, schema, refusals;
  * the registry is one source: every entry plans, has a block list and a sampling plan;
  * the four sequences of ``console.EXTENDED`` scan through ``Engine.scan`` and through a session, are replayed bit for
    bit, and their noise-free data do not depend on the averages;
  * the line path equals the engine's own formation where the amplitude does not change from line to line.
"""
import numpy as np
import pytest

import golden_cases as G
from conftest import episode_cfg
from mrgym.core import blocks as B
from mrgym.scanner import console, kspace, served, served_checks, session
from mrgym.scanner import world as W

EXAM = dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0)
ALL = dict(EXAM, sequences="all")
PROTOCOLS = dict(
    fse=dict(sequence="fse", tr_ms=2500.0, etl=8, te_ms=50.0, echo_order="linear", matrix=64),
    mese=dict(sequence="mese", tr_ms=1500.0, n_echoes=6, matrix=64, refocus_deg=150.0),
    megre=dict(sequence="megre", tr_ms=60.0, flip_deg=20.0, n_echoes=3, matrix=64, readout="bipolar"),
    dwse=dict(sequence="dwse", tr_ms=2500.0, b_value=800.0, matrix=64),
)
_rows = {}


def rows(fn):
    if fn not in _rows:
        _rows[fn] = fn()
    return _rows[fn]


@pytest.fixture(scope="module")
def world():
    return W.make_world("c0" * 16, dict(kind="brain", n=256, console_sequences="all"))


@pytest.fixture(scope="module")
def engine(world):
    return kspace.Engine(world)


# ---------------------------------------------------------------------------------------------- the console of before
def test_a_card_without_sequences_offers_the_six_and_nothing_else():
    assert console.offered(None) is console.SEQUENCES and console.offered(EXAM) is console.SEQUENCES
    assert console.CLASSIC == tuple(console.SEQUENCES) == ("spgr", "afi", "se", "ir", "bssfp", "epi")
    assert set(console.REGISTRY) == set(console.SEQUENCES) | set(console.EXTENDED)
    for name, prot in PROTOCOLS.items():
        pl = console.plan(prot, EXAM)
        assert not pl["accept"] and pl["violations"][0]["code"] == "UNKNOWN_SEQUENCE" and name not in pl["violations"][0]["message"]
    assert console.protocol_schema(EXAM)["properties"]["sequence"]["enum"] == sorted(console.SEQUENCES)
    assert not set(console.protocol_schema(EXAM)["properties"]) & {"etl", "b_value", "refocus_deg", "echo_order", "readout", "diffusion_dir"}
    # the words of real consoles keep their refusal there, and get the parameter that provides them where it exists
    old = console.plan(dict(sequence="se", tr_ms=500.0, turbo_factor=4), EXAM)["violations"][0]
    new = console.plan(dict(sequence="se", tr_ms=500.0, turbo_factor=4), ALL)["violations"][0]
    assert old["code"] == "NOT_SIMULATED" and new["code"] == "UNKNOWN_PARAMETER" and "etl of fse" in new["message"]


def test_a_card_lists_the_sequences_it_offers():
    assert list(console.offered(dict(sequences=["fse"]))) == list(console.SEQUENCES) + ["fse"]
    assert set(console.offered(ALL)) == set(console.REGISTRY)
    with pytest.raises(ValueError):
        console.offered(dict(sequences=["haste"]))
    schema = console.protocol_schema(ALL)
    assert schema["properties"]["sequence"]["enum"] == sorted(console.REGISTRY)
    names = {k for spec in console.offered(ALL).values() for k in spec}
    assert set(schema["properties"]) == names | {"sequence"} and names <= set(console.PARAMETER_UNITS) and names <= set(console.PARAMETER_MEANING)
    assert console.plan(PROTOCOLS["fse"], dict(EXAM, sequences=["fse"]))["accept"]
    assert not console.plan(PROTOCOLS["dwse"], dict(EXAM, sequences=["fse"]))["accept"]


# ---------------------------------------------------------------------------------------------- one source
@pytest.mark.parametrize("name", sorted(console.REGISTRY))
def test_every_entry_plans_and_has_a_block_list_and_a_sampling_plan(name):
    entry = console.REGISTRY[name]
    prot = PROTOCOLS.get(name) or dict(dict(sequence=name, tr_ms=dict(bssfp=8.0, afi=25.0, spgr=30.0).get(name, 1500.0), matrix=64),
                                       **({"flip_deg": 30.0} if "flip_deg" in entry.parameters and name not in console.FIXED else {}),
                                       **({"ti_ms": 400.0} if name == "ir" else {}))
    pl = console.plan(prot, ALL)
    assert pl["accept"], pl["violations"]
    rp = pl["realised"]
    assert entry.serves == ("core" if name in console.EXTENDED else "closed form") and entry.parameters is console.offered(ALL)[name]
    seq, labels = entry.blocks(rp)
    plan = console.sampling(rp)
    assert isinstance(seq, B.Sequence) and len(labels) == plan.samples and len(plan.images) == len(rp["echo_times_ms"]) * (2 if name == "afi" else 1)
    assert console.plan(console.request_of(rp), ALL)["realised"] == rp      # replay plans the realised protocol again
    assert pl["header"].startswith(name.upper())


def test_the_acquired_lines_of_the_console_are_those_the_reconstruction_weights():
    for lines_full in (32, 48, 130):
        for lines in range(lines_full // 2 + 2, lines_full + 1):
            rp = dict(sequence="se", lines_full=lines_full, lines=lines)
            assert console.acquired_lines(lines_full, lines) == list(kspace.Engine.line_weights(rp) > 0) and sum(console.acquired_lines(lines_full, lines)) == lines


# ---------------------------------------------------------------------------------------------- scans
@pytest.mark.parametrize("name", sorted(PROTOCOLS))
def test_scan_of_a_core_sequence(engine, name):
    out = engine.scan(PROTOCOLS[name])
    rp, img = out["plan"]["realised"], out["images"]
    assert img.shape == (len(rp["echo_times_ms"]), rp["matrix_pe"], rp["matrix"]) and np.all(np.isfinite(img)) and np.abs(img).max() > 0.1
    # the noise-free data are a function of the physics key: the averages do not enter, and a second engine gives the same bytes
    again = kspace.Engine(engine.w).acquire(console.plan(dict(PROTOCOLS[name], averages=4), engine.w["console"])["realised"])
    assert np.array_equal(again, img)
    noisy = engine.acquire(rp, np.random.default_rng(1))
    sd = float(np.std((noisy - img)[0].real))
    assert 0.8 < sd / engine.noise_sd(rp) < 1.25                        # the noise of a pixel is the one the engine states


def test_later_echoes_decay_and_the_diffusion_weighted_image_is_darker(engine):
    mese = engine.scan(PROTOCOLS["mese"])["images"]
    means = np.abs(mese).mean((1, 2))
    assert np.all(np.diff(means[1:]) < 0)                               # (the first echo may be the lower one: stimulated echoes)
    lo = engine.scan(dict(PROTOCOLS["dwse"], b_value=0.0, te_ms=engine.scan(PROTOCOLS["dwse"])["plan"]["realised"]["te_ms"]))["images"]
    hi = engine.scan(PROTOCOLS["dwse"])["images"]
    assert np.abs(hi).mean() < 0.7 * np.abs(lo).mean()


def test_a_world_with_periodic_motion_is_refused_for_core_sequences():
    w = W.make_world("c1" * 16, dict(kind="brain", n=128, console_sequences="all", motion=[dict(kind="pulse", cls="csf", f_hz=1.1, depth=0.2)]))
    eng = kspace.Engine(w)
    assert eng.scan(dict(sequence="se", tr_ms=500.0, matrix=32))["images"].shape == (1, 32, 32)
    with pytest.raises(ValueError, match="periodic motion"):
        eng.scan(dict(PROTOCOLS["fse"], matrix=32))


def test_the_line_path_is_the_engines_formation_where_nothing_changes_from_line_to_line(engine):
    """A table with one sample and the amplitude of ``se``: ``form_lines`` against ``Engine.kspace`` (other order of
    the transforms, other factorisation, cropped grid)."""
    w = engine.w
    rp = console.plan(dict(sequence="dwse", tr_ms=800.0, te_ms=40.0, b_value=0.0, matrix=64, pe_dir="x", centre_y_mm=12.0, phase_fov=0.75), w["console"])["realised"]
    se = console.plan(dict(sequence="se", tr_ms=800.0, te_ms=40.0, matrix=64, pe_dir="x", centre_y_mm=12.0, phase_fov=0.75), w["console"])["realised"]
    a, b = engine.clean_kspace(rp)[0], engine.clean_kspace(se)[0]
    assert np.abs(a - b).max() < 1e-4 * np.abs(b).max()                 # the engine cuts its readout basis at 1e-4
    det = {}
    served.form_lines(engine, rp, served.table(engine, rp), console.sampling(rp), details=det)
    assert 0 < det["terms"] <= 16 and det["bound"] <= det["tolerance"] == served.LINE_NOISE_FRACTION * served.noise_per_sample(engine, rp, 64)


def test_session_scans_and_replays_core_sequences(world):
    cfg = episode_cfg(seed="c0" * 16, budget_s=4000.0, kind="brain", n=256, console_sequences="all")
    s = session.ScannerSession(cfg, engine=kspace.Engine(world))
    card = s.info()
    assert set(card["sequences"]) == set(console.REGISTRY)
    for name in ("fse", "dwse", "megre"):
        s.scan({"protocol": PROTOCOLS[name], "rois": [dict(id="c", y_mm=0, x_mm=0, radius_mm=20)]})
    with pytest.raises(session.InvalidRequest):
        s.scan({"protocol": dict(PROTOCOLS["fse"], etl=100)})
    rep = session.replay(s.cfg, s.events, head=s.receipt())             # a fresh world and engine from the seed
    assert rep["ok"] and rep["mode"] == "bit-exact" and rep["scans"] == 3 and rep["refused"] == 1


def test_a_session_without_the_key_does_not_know_them():
    s = session.ScannerSession(episode_cfg(), engine=None)
    with pytest.raises(session.InvalidRequest):
        s.scan({"protocol": PROTOCOLS["fse"]})


# ---------------------------------------------------------------------------------------------- the validation rows
@pytest.mark.parametrize("check", served_checks.CHECKS, ids=lambda f: f.__name__)
def test_every_row_passes(check):
    bad = [(r["id"], r["measured"], r["limit"]) for r in rows(check) if not r["ok"]]
    assert not bad, bad


def test_row_ids_are_unique_and_every_sequence_has_rows():
    ids = [r["id"] for fn in served_checks.CHECKS for r in rows(fn)]
    assert len(ids) == len(set(ids)) and all(i.startswith("J") for i in ids)
    text = " ".join(r["check"] for fn in served_checks.CHECKS for r in rows(fn))
    assert all(f"`{name}`" in text for name in console.EXTENDED)


# ---------------------------------------------------------------------------------------------- golden data (additive section)
def test_golden_section_of_the_core_sequences():
    """``tests/golden/engine_digests.json``, section ``core``: bit for bit on the recorded stack, block fingerprints
    elsewhere; prices and header lines equal on every stack (rules of ``tests/test_golden.py``). Here the first
    world of the section (the others and the noisy session: ``python tools/audit/golden.py --check``)."""
    golden = G.load(G.ENGINE_FILE)
    assert golden and "core" in golden, "run `python tools/audit/golden.py --write` (adding a section under the same engine is allowed)"
    want, got = golden["core"], G.core_cases(worlds=G.CORE_WORLDS[:1], noisy=False)
    exact = G.mode(golden["stack"])[0] == "bit-exact"
    assert got["prices"] == want["prices"] and set(got["images"]) < set(want["images"]) and len(want["images"]) == len(G.CORE_WORLDS) * len(G.CORE_PROTOCOLS)
    assert {k.split("/")[2].split()[0] for k in got["images"]} == set(console.EXTENDED) and len(want["noisy"]) == 6
    for key, val in got["images"].items():
        if exact:
            assert val["sha256"] == want["images"][key]["sha256"], key
        else:
            assert val["shape"] == want["images"][key]["shape"] and session.fingerprints_close(val["fingerprint"], want["images"][key]["fingerprint"]), key
    assert want["noisy"]["0:fse"]["sha256"] != want["noisy"]["4:fse"]["sha256"]          # a repeated scan draws new noise


# ---------------------------------------------------------------------------------------------- the manual and the tissue sources
def test_the_supplement_names_every_parameter_of_the_core_sequences_with_its_range():
    import importlib.resources
    import os
    text = importlib.resources.files("mrgym.scanner").joinpath("datasheet_core.md").read_text(encoding="utf-8")
    for seq, spec in console.EXTENDED.items():
        own = {k: v for k, v in spec.items() if k not in console._GEOM or v != console._GEOM[k]}
        assert own, seq
        for name, (lo, hi, kind, _) in own.items():
            row = next((ln for ln in text.splitlines() if ln.startswith(f"| `{seq}` | `{name}` |")), None)
            assert row, f"{seq}.{name} has no row in the supplement"
            assert (f"| {lo:g} to {hi:g} |" in row) if kind in "fi" else all(f"{c}" in row.rsplit("|", 3)[1] for c in lo), row
    copy = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "scanner-core.md")
    with open(copy, encoding="utf-8") as f:
        assert f.read().split("-->\n", 1)[1] == text                       # the docs page is a generated copy
    # the datasheet every console serves does not know them: an agent of an existing task reads what it read
    old = importlib.resources.files("mrgym.scanner").joinpath("datasheet.md").read_text(encoding="utf-8")
    assert not any(f"`{name}`" in old for name in console.EXTENDED)


def test_every_diffusion_coefficient_has_a_source_line_and_the_table_did_not_move():
    from mrgym.scanner import tissue
    rep = {r["name"]: r for r in tissue.adc_report()}
    assert set(rep) == set(tissue.NAMES) == set(tissue.ADC_SOURCES)
    assert sum(r["tag"] == "A" for r in rep.values()) == 8 and sum(r["tag"] == "C" for r in rep.values()) == 16 and rep["csf"]["tag"] == "R"
    assert all(r["source"] and r["low"] < r["high"] for r in rep.values() if r["tag"] != "C")
    outside = sorted(n for n, r in rep.items() if r["inside"] is False)
    assert outside == ["liver"]                                           # declared in limits.md, block 3a: not corrected, the table is read by spgr
    # the column itself: the numbers of engine 0.3.0 (a change would move the data of spgr and afi)
    assert [tissue.table(3.0)[n]["adc_um2_ms"] for n in ("wm", "gm", "csf", "scalp_fat", "muscle", "liver")] == [0.75, 0.85, 3.0, 0.05, 1.4, 1.1]
    assert abs(tissue.water_adc_um2_ms(25.0) - 2.299) < 2e-3 and abs(tissue.water_adc_um2_ms(20.0) - 2.023) < 2e-3 and 3.0 < tissue.water_adc_um2_ms(37.0) < 3.1
    with pytest.raises(ValueError):
        tissue.water_adc_um2_ms(-5.0)
