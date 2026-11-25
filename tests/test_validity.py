"""``mrgym validate --perturb`` (the validity table of an environment) and the external validation ladder."""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

import mrgym
from mrgym import cards, perturb

ROOT = pathlib.Path(__file__).resolve().parents[1]
NULL = "Scanner/NullTissue-v0"


def _double_t1(world):
    for t in world["tissues"].values():
        t["t1_ms"] *= 2.0


# ------------------------------------------------------------------ statistics
def test_paired_interval_brackets_the_difference_and_covers_it():
    base, pert = np.array([1] * 40 + [0] * 10, bool), np.array([1] * 30 + [0] * 10 + [1] * 5 + [0] * 5, bool)
    d, lo, hi = perturb.paired_difference(base, pert)
    assert d == pytest.approx(-0.1) and lo < d < hi and -0.25 < lo and hi < 0.08
    same = perturb.paired_difference(np.ones(100, bool), np.ones(100, bool))
    assert same[0] == 0.0 and -0.06 < same[1] < 0 < same[2] < 0.06
    rng = np.random.default_rng(0)
    covered, n = 0, 60
    for _ in range(1500):                                               # correlated pairs: true rates 0.9 and 0.8
        u = rng.uniform(size=n)
        a, b = u < 0.9, np.where(rng.uniform(size=n) < 0.7, u < 0.8, rng.uniform(size=n) < 0.8)
        _, lo, hi = perturb.paired_difference(a, b)
        covered += lo <= -0.1 <= hi
    assert covered / 1500 >= 0.93


def test_classes():
    assert perturb.classify(0.0, -0.05, 0.05, 0.0, 0.0) == "robust"
    assert perturb.classify(-0.3, -0.45, -0.15, 2.0, 0.0) == "flips"
    assert perturb.classify(0.0, -0.05, 0.05, 0.0, 0.5) == "flips"       # the right answer moved, whatever the pass rate
    assert perturb.classify(-0.08, -0.2, 0.04, 0.1, 0.0) == "noise-like"
    assert perturb.classify(-0.08, -0.2, 0.04, 0.6, 0.0) == "undecided"


def test_every_perturbation_is_described_and_builds_a_world():
    assert len({p["name"] for p in perturb.PERTURBATIONS}) == len(perturb.PERTURBATIONS) >= 20
    cfg = dict(task_id="t", seed="000102030405060708090a0b0c0d0e0f", world_spec=dict(kind="qa"), budget_s=60.0, task={})
    names = perturb.applicable(cfg["world_spec"])
    assert "tissue_brainweb" not in names and {"rf_fast", "noise_x2", "edge_soft", "gnl", "motion_ghost"} <= set(names)
    _, plain, _ = perturb.build(cfg, None)
    for name in names:
        p = perturb.BY_NAME[name]
        assert all(p[k] for k in ("what", "size", "source")) and p["kind"] in ("world_spec", "world", "engine", "readout")
        _, world, engine = perturb.build(cfg, p)
        assert world["regions"] == plain["regions"] and engine.world is world
    _, world, _ = perturb.build(cfg, perturb.BY_NAME["b0_x5"])
    from mrgym.scanner import world as W
    turned = W.turned(world, 20.0)                                       # an edit of the fields holds on a turned grid
    assert np.abs(turned["fields"]["b0_hz"]).max() == pytest.approx(5 * np.abs(W.turned(plain, 20.0)["fields"]["b0_hz"]).max())


# ------------------------------------------------------------------ an environment under perturbed physics
@pytest.fixture(scope="module")
def result():
    planted = dict(name="planted_t1_x2", what="T1 of every material, in the measurement only", size="x 2", source="test", kind="world",
                   world=_double_t1, score_on="default")
    return perturb.run(NULL, n=3, level=1, only=["noise_x0.5", "relax_shift", "planted_t1_x2"], extra=[planted])


def test_a_planted_change_of_the_measured_physics_is_classified_as_a_flip(result):
    rows = {r["name"]: r for r in result["rows"]}
    assert result["engine"] == mrgym.ENGINE_ID and result["default"]["passed"] == 3 and result["instances"] == 3
    planted = rows["planted_t1_x2"]
    assert planted["class"] == "flips" and planted["passed"] == 0 and not planted["follows"]
    assert rows["noise_x0.5"]["passed"] == 3 and rows["noise_x0.5"]["answer_moved"] == 0.0 and rows["noise_x0.5"]["class"] != "flips"
    moved = rows["relax_shift"]                                         # the object changed: the right answer moves, the reference measures it
    assert moved["answer_moved"] == 1.0 and moved["passed"] == 3 and moved["class"] == "flips" and moved["follows"]
    assert perturb.problems(result, mrgym.ENGINE_ID) == ["planted_t1_x2 flips the answer and is neither modelled, randomised nor declared"]
    assert perturb.problems(result, "mrgym-engine-0.0.0")[0].startswith("the validity result was measured on")
    assert perturb.problems(None, mrgym.ENGINE_ID)


def test_the_table_and_the_card_section_render(result, tmp_path):
    text = perturb.text_table(result)
    assert "planted_t1_x2" in text and "flips" in text and "the reference follows" in text
    path = perturb.write(result, str(tmp_path))
    assert json.load(open(path))["rows"] == result["rows"]
    section = "\n".join(perturb.markdown(result))
    assert section.startswith("## Validity") and "| `relax_shift`" in section and "never been compared with data from one" in section
    card = dict(mrgym.card(NULL), validity=result)
    assert "## Validity" in cards.markdown(card)
    assert "Not measured" in "\n".join(perturb.markdown(None))


def test_unknown_names_and_sandboxes_are_refused():
    with pytest.raises(mrgym.Error):
        perturb.run(NULL, n=1, only=["no_such_row"])
    with pytest.raises(mrgym.Error):
        perturb.run("Scanner/Explore-v0", n=1)


# ------------------------------------------------------------------ the external ladder
def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "external" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_independent_integrator_shares_nothing_with_the_package():
    source = (ROOT / "tools" / "external" / "independent_bloch.py").read_text(encoding="utf-8")
    assert "mrgym" not in source.split('"""', 2)[2] and "import numpy as np" in source
    ib = _load("independent_bloch")
    from mrgym.scanner import epg, rf
    ours = epg.steady_state(np.deg2rad(30.0), 8.0, 300.0, 30.0, 117.0, (1,))
    assert abs(ours[0] - ib.spoiled_gre(np.deg2rad(30.0), 8.0, 300.0, 30.0, 117.0, n_iso=256)[0]) < 1e-6
    z = np.linspace(-1.5, 1.5, 31)
    a, b = rf.cayley_klein(np.deg2rad(70.0), 4.0, z)
    a2, b2, ab = ib.slice_rotation_parameters(np.deg2rad(70.0), 4.0, z)
    assert max(np.abs(a * a - a2).max(), np.abs(b * b - b2).max(), np.abs(np.conj(a) * b - ab).max()) < 1e-10


def test_the_stored_ladder_is_for_this_engine_and_says_that_no_scanner_was_measured():
    ext = ROOT / "docs" / "physics" / "external"
    for rung in ("L1", "L2", "L3"):
        stored = json.loads((ext / f"{rung}.json").read_text(encoding="utf-8"))
        assert stored["engine"] == mrgym.ENGINE_ID and stored["rows"] and all(r["passed"] for r in stored["rows"]), rung
    page = (ROOT / "docs" / "physics" / "external.md").read_text(encoding="utf-8")
    assert "No scanner measurement exists" in page and "no scanner measurement exists" in page and mrgym.ENGINE_ID in page
    ref = json.loads((ext / "sycomore_reference.json").read_text(encoding="utf-8"))
    assert len(ref["spgr"]) == 13 and len(ref["afi"]) == 4 and ref["sycomore_version"]
