"""Oracles: the optimum of an instance found on the hidden world, by code that shares nothing with the reference.

What is shown
  * no ``oracle.py`` imports the control policies or names the reference, and no scoring function of a ``task.py``
    does (read from the syntax tree, for every family that ships an oracle)
  * the search helpers return the known optimum of small problems, with the failing neighbours as a certificate
  * ``Scanner/NullTissue``: the oracle's answer passes the verifier, agrees with the task's own bisection to the
    console raster, the ends of each pass window pass and the raster neighbours outside fail
  * the reference policy has a measured regret against the oracle, and no passing episode beats the optimum
"""
import importlib
import pkgutil

import pytest

import mrgym
from mrgym import cli, gate
from mrgym.envs import oracle as O
from mrgym.scanner import kspace
from mrgym.scanner import world as W

NULL = "Scanner/NullTissue-v0"


def families_with_oracle():
    import mrgym.envs as envs
    names = [m.name for m in pkgutil.iter_modules(envs.__path__) if m.ispkg]
    return [f"mrgym.envs.{n}" for n in names if importlib.util.find_spec(f"mrgym.envs.{n}.oracle") is not None]


def test_oracles_are_independent_of_the_policies():
    families = families_with_oracle()
    assert {"mrgym.envs.null_tissue", "mrgym.envs.nowrap", "mrgym.envs.t1_transmit"} <= set(families)
    for family in families:
        assert O.independence_problems(family) == [], family
        module = importlib.import_module(family + ".oracle")
        assert callable(module.solve) and module.REGRET_KIND in ("ratio", "difference")


def test_the_independence_check_catches_an_import_of_the_policies(tmp_path, monkeypatch):
    pkg = tmp_path / "fam"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "task.py").write_text("def score(cfg, answer):\n    from . import controls\n    return controls.reference(None)\n\n\ndef card():\n    from . import controls\n    return controls.REFERENCE\n")
    (pkg / "oracle.py").write_text("import importlib\nfrom .controls import tune\n\n\ndef solve(cfg, world=None):\n    return importlib.import_module('fam.controls')._geometry(cfg)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    problems = O.independence_problems("fam")
    assert "fam/oracle.py imports controls" in problems and "fam/oracle.py uses the name _geometry" in problems
    assert "fam/task.py: score() uses controls" in problems and "fam/task.py: score() uses reference" in problems
    assert not any("card()" in p for p in problems)                       # the card may list the policies
    (pkg / "oracle.py").unlink()
    assert O.independence_problems("fam") == ["fam: no oracle.py"]


def test_exhaustive_search_helpers():
    cost = lambda n: 2.0 * n                                                # noqa: E731
    fails = lambda d: None if d["lines"] >= 37 else "coverage"              # noqa: E731
    out = O.exhaustive_first((({"lines": n}, cost(n)) for n in range(30, 60, 2)), fails, keep=3)
    assert out["value"] == 76.0 and out["argmin"] == {"lines": 38} and out["method"] == "exhaustive" and out["grid"]["evaluated"] == 5
    assert [c["deliverable"]["lines"] for c in out["certificate"]] == [32, 34, 36] and {c["fails"] for c in out["certificate"]} == {"coverage"}
    assert O.exhaustive_first([({"lines": 2}, 4.0)], lambda d: "coverage")["value"] is None
    with pytest.raises(ValueError):
        O.exhaustive_first([({}, 2.0), ({}, 1.0)], lambda d: "x")
    best = O.exhaustive_min([0.0, 1.0, 2.0, 3.0], lambda x: (x - 1.8) ** 2)
    assert best["argmin"] == 2.0 and best["evaluated"] == 4
    assert O.sign_changes([-3, -1, 2, 5, -1]) == [1, 3]
    root, n = O.refine_root(lambda x: x ** 3 - 2.0, 0.0, 3.0, 1e-6)
    assert root == pytest.approx(2 ** (1 / 3), abs=1e-6) and n < 40
    with pytest.raises(ValueError):
        O.refine_root(lambda x: x * x + 1, -1.0, 1.0, 1e-3)


def test_regret_and_its_report():
    assert O.regret(33.0, 30.0) == pytest.approx(0.1) and O.regret(0.03, 0.001, "difference") == pytest.approx(0.029)
    assert O.regret(None, 30.0) is None and O.regret(1.0, None) is None and O.regret(1.0, 0.0) is None
    rep = O.regret_report([0.0, 0.02, 0.10, None, -0.03], passed=[True, True, False, False, True])
    assert (rep["n"], rep["missing"], rep["negative"], rep["negative_passing"]) == (4, 1, 1, 1)
    assert rep["median"] == pytest.approx(0.01) and rep["max"] == pytest.approx(0.10)
    assert O.regret_report([])["median"] is None


@pytest.fixture(scope="module")
def instance():
    spec = mrgym.spec(NULL)
    cfg = gate._instance(spec, 1, "gate:3", split="train")
    world = W.make_world(cfg["seed"], cfg["world_spec"])
    module = O.load(NULL)
    return cfg, world, kspace.Engine(world), module, module.solve(cfg, world)


def test_null_tissue_oracle_answer_passes_the_verifier(instance):
    cfg, world, engine, module, sol = instance
    task = importlib.import_module("mrgym.envs.null_tissue.task")
    ids = [r["id"] for r in cfg["task"]["regions"]]
    assert sol["method"] == "exhaustive" and sol["unique"] and set(sol["argmin"]["ti_ms"]) == set(ids)
    assert sol["grid"]["evaluated"] < 200 < sol["grid"]["size"] and sol["seconds"] > 0
    score = task.score(cfg, sol["argmin"], world, engine)
    assert score["passed"] and score["error"] == pytest.approx(sol["value"]) and sol["value"] < 0.002
    i = ids[0]                                                              # two independent root finders agree to the raster
    assert sol["null_ms"][i] == pytest.approx(task.true_null_ms(cfg, i, world, engine), abs=0.03)


def test_null_tissue_pass_window_and_certificate(instance):
    cfg, world, engine, module, sol = instance
    task = importlib.import_module("mrgym.envs.null_tissue.task")
    best = sol["argmin"]["ti_ms"]
    for i, (lo, hi) in sol["window_ms"].items():
        assert lo < best[i] < hi
        for ti in (lo, hi):                                                 # the ends of the window pass
            assert task.score(cfg, {"ti_ms": dict(best, **{i: ti})}, world, engine)["passed"], (i, ti)
    assert len(sol["certificate"]) == 2 * len(best)
    for c in sol["certificate"]:                                            # the raster neighbours just outside fail, on that region
        (i, ti), = c["deliverable"]["ti_ms"].items()
        score = task.score(cfg, {"ti_ms": dict(best, **{i: ti})}, world, engine)
        assert not score["passed"] and [k for k, ok in score["checks"].items() if not ok] == [c["fails"]]
    miss = module.miss(cfg, {"ti_ms": {i: w[1] for i, w in sol["window_ms"].items()}}, sol)
    assert all(v == pytest.approx(1.0, abs=0.01) for v in miss.values())
    far = module.miss(cfg, {"ti_ms": {i: 2.0 * v for i, v in best.items()}}, sol)
    assert all(v > 1.0 for v in far.values())


def test_reference_regret_against_the_oracle(instance, capsys, monkeypatch):
    module, sol = instance[3], instance[4]
    monkeypatch.setattr(module, "solve", lambda cfg, world=None: sol)        # the same instance: do not search it twice
    rep = O.policy_regret(NULL, 1, ["gate:3"])
    assert rep["policy"] == "reference" and rep["instances"] == 1 and rep["oracle_passes"] == 1 and rep["kind"] == "difference"
    row = rep["rows"][0]
    assert row["regret"] is not None and row["regret"] >= 0 and rep["regret"]["negative_passing"] == 0
    assert row["optimum"] < 0.002 and (not row["passed"] or row["error"] <= 0.05)
    assert "regret above the optimum" in O.regret_text(rep)
    with pytest.raises(mrgym.Error):
        O.policy_regret("Scanner/Explore-v0", 1, [0])                        # a sandbox ships no oracle
    assert cli.main(["stats", "regret", NULL, "--n", "1", "--level", "1", "--prefix", "gate"]) == 0      # gate:0, with the patched optimum
    assert "policy reference, 1 instances" in capsys.readouterr().out
