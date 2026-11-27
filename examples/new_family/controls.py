"""Scripted policies that play through the scanner's tools only. ``mrgym gate`` runs them on fresh instances."""
import math

from mrgym.scanner import world as W

REFERENCE = "reference"                                 # must pass: shows the task can be solved through the tools
MUST_FAIL = ("prior", "fixed_recipe", "wrong_units")    # shortcuts that ignore the object: each must fail
ABLATIONS = {}                                          # {name: levels}: the reference minus one decision, must lose there
REPORTED = ("open_loop",)                               # reported, no requirement
PROBE = dict(sequence="se", tr_ms=2500.0, matrix=64)
ECHOES_MS = (20.0, 100.0)


def reference(sc):
    """``sc`` is the agent's side of the scanner: info, budget, plan, scan, read_regions, view. Returns the deliverable."""
    regions = sc.info()["task"]["regions"]
    signal = [{r["id"]: r["complex_mean_magnitude"][0] for r in sc.scan(dict(PROBE, te_ms=te), rois=regions)["roi_readouts"]}
              for te in ECHOES_MS]
    return {"t2_ms": {i: (ECHOES_MS[1] - ECHOES_MS[0]) / math.log(max(signal[0][i] / signal[1][i], 1.001)) for i in signal[0]}}


def prior(sc):
    return {"t2_ms": {r["id"]: 90.0 for r in sc.info()["task"]["regions"]}}


def fixed_recipe(recipe):
    def play(sc):
        sc.scan(dict(PROBE, tr_ms=300.0, te_ms=20.0))                  # a token scan, so that only the answer is tested
        return {"t2_ms": {r["id"]: recipe["t2_ms"] for r in sc.info()["task"]["regions"]}}
    return play


def wrong_units(sc):
    return {"t2_ms": {i: v / 1000.0 for i, v in reference(sc)["t2_ms"].items()}}


def policies(recipe=None) -> dict:
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or dict(t2_ms=90.0)),
                wrong_units=wrong_units, open_loop=reference)


def tune(cfgs) -> dict:
    """The strongest constant answer on other worlds: the geometric mean of their true T2 (verifier side, never in an episode)."""
    t2 = [v["t2_ms"] for c in cfgs for v in W.truth(W.make_world(c["seed"], c["world_spec"], layout_only=True)).values()]
    return dict(t2_ms=math.exp(sum(map(math.log, t2)) / len(t2)))


DESCRIPTIONS = dict(reference="two spin-echo scans at one long repetition time; T2 from the ratio of the two echoes",
                    prior="no scan; the middle of the plausible T2 range for every region",
                    fixed_recipe="one token scan, then one constant T2 tuned on other worlds",
                    wrong_units="the reference's values in seconds",
                    open_loop="a fixed scan list and a fixed estimator: the reference itself (no scan depends on a result)")
