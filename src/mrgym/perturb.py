"""Does a simplification of the scanner change a scored answer? Re-score an environment under perturbed physics.

    mrgym validate --perturb Scanner/NullTissue-v0 --n 30 --write

For every instance (string seeds ``validity:0 ..``, never the gate's or the tuning seeds) and every perturbation in
``PERTURBATIONS`` the hidden world and the engine are built with one piece of physics changed, the family's reference
policy plays through the scanner interface, and its deliverable is scored against the perturbed world. Seeds and
noise draws are the same in every row, so each row is paired with the unperturbed one.

Per row
  pass rate, and its paired difference to the unperturbed row with a 95 % interval (Newcombe 1998: Wilson intervals of the two rates, combined with their correlation)
  shift          median over instances of (error - unperturbed error) / tolerance, where the family's score states both
  answer moved   share of the instances the reference passed unperturbed whose unperturbed deliverable fails on the
                 perturbed world: the perturbation moved the set of right answers
Class
  flips          the pass rate falls by more than 0.10 with an interval that excludes 0, or the answer moved on more
                 than 10 % of the instances. ``follows`` says whether the reference still passes (it measured the change)
  robust         difference >= -0.05 and the interval above -0.10
  noise-like     neither, and the error moves by less than a quarter of the tolerance
  undecided      neither: more instances are needed
What the table is: a statement about this reference policy on this engine. A ``flips`` row says the effect can change
what is right; whether that is modelled (it is in the default engine), randomised (it is in the instance distribution)
or declared (the task card restricts its claim) is the family's to state (``task.VALIDITY["handled_by"]``).
What it is not: a comparison with a scanner. Effects the engine cannot emulate are listed in ``UNTESTABLE``.

A family needs nothing beyond the contract of ``envs/family.py``. Optional: ``task.VALIDITY = dict(error=<key of the
score dict>, tolerance=<key>, handled_by={row name: "modelled" | "randomised" | "declared"}, claim=..., not_claimed=[..])``.
Without it the score's ``error`` and ``tolerance`` (or its single ``*_max`` / ``*_tol`` entry) are used.
"""
from __future__ import annotations

import datetime
import importlib
import importlib.resources
import json
import math
import multiprocessing
import os
import time

import numpy as np

SCHEMA = "mrgym-validity/1"
Z95 = 1.959963984540054
UNTESTABLE = (
    "magnetisation transfer and exchange", "restricted or anisotropic diffusion", "flow, perfusion and physiological motion beyond one periodic "
    "line-modulation model", "eddy currents, concomitant fields, gradient impulse response", "SAR in W/kg, local SAR, nerve stimulation",
    "B0 drift and heating", "anything through the slice beyond independent columns: slice positioning, multi-slice cross-talk, slab wrap of the 3D mode",
    "parallel transmit", "vendor pulses, filters and distortion correction", "coil loading and receive arrays",
)
NOT_BUILT = dict(
    transient="needs the approach to steady state in the engine; the engine serves the periodic state",
    array8_sos="needs a receive array with root-sum-of-squares combination; the engine has one channel",
    spoiler_lobe="needs the b-factor of a spoiler lobe instead of a gradient that is constant over TR",
    afi_phase_per_pulse="needs an RF phase counted per pulse in the EPG (the engine counts raster positions)",
    fov_free="a verifier-side row of the NoWrap family (optimum with free centre and rotation); not a physics perturbation",
)


# ------------------------------------------------------------------------------------------------ the perturbations
def _system(key, fn):
    def edit(world):
        world["system"][key] = fn(world["system"][key], world)
    return edit


def _tissues(fn):
    def edit(world):
        for t in world["tissues"].values():
            fn(t)
            t["t2s_ms"] = min(t["t2s_ms"], 0.98 * t["t2_ms"]) if "t2p_ms" not in t else 1.0 / (1.0 / t["t2_ms"] + 1.0 / t["t2p_ms"])
    return edit


def _fields(fn):
    """An edit of arrays on the grid: registered in ``world["post_build"]`` so that it holds for a turned field of view."""
    def edit(world):
        world.setdefault("post_build", []).append(fn)
        fn(world)
    return edit


def _b0_x5(world):
    world["fields"] = dict(world["fields"], b0_hz=5 * world["fields"]["b0_hz"], dbdz_hz_per_mm=5 * world["fields"]["dbdz_hz_per_mm"])


def _b1_x3(world):
    world["fields"] = dict(world["fields"], b1=np.clip(1 + 3 * (world["fields"]["b1"] - 1), 0.3, 1.7))


def _rx_surface(world):
    from .scanner import world as W
    y, _ = W.grid(world)
    world["fields"] = dict(world["fields"], rx=world["fields"]["rx"] * np.exp(-(y + world["size_mm"][0] / 2) / 70.0))


def _relax(t):
    t["t1_ms"], t["t2_ms"] = 1.2 * t["t1_ms"], 0.8 * t["t2_ms"]


BRAINWEB_1P5T = dict(csf=(2569.0, 329.0, 58.0, 1.0), gm=(833.0, 83.0, 69.0, 0.86), wm=(500.0, 70.0, 61.0, 0.77))   # T1, T2, T2*, PD


def _brainweb(world):
    for k, (t1, t2, t2s, pd) in BRAINWEB_1P5T.items():
        if k in world["tissues"]:
            world["tissues"][k].update(t1_ms=t1, t2_ms=t2, t2s_ms=t2s, pd=pd, t2p_ms=1.0 / (1.0 / t2s - 1.0 / t2))


def _motion(world):
    world["motion"] = [dict(cls="*", kind="pulse", f_hz=0.25, depth=0.1, phase_key=1)]


def _p(name, what, size, source, kind, **how):
    return dict(name=name, what=what, size=size, source=source, kind=kind, **how)


_QA = lambda spec: (spec or {}).get("kind", "qa") == "qa"                   # noqa: E731
PERTURBATIONS = (
    _p("rf_fast", "excitation pulse", "time-bandwidth 2 instead of the console's", "Malik 2011; Svedin & Parker 2017", "engine", engine=dict(rf=dict(tbw=2.0))),
    _p("rf_sharp", "excitation pulse", "time-bandwidth 8", "Malik 2011; Svedin & Parker 2017", "engine", engine=dict(rf=dict(tbw=8.0))),
    _p("rf_hamming", "excitation pulse", "Hamming instead of Hann window", "Malik 2011; Svedin & Parker 2017", "engine", engine=dict(rf=dict(window="hamming"))),
    _p("rf_rect", "excitation pulse", "ideal rectangular slice profile (the engine of 0.1.0)", "the hard-pulse model", "engine", engine=dict(rf=dict(profile="rect"))),
    _p("ideal_spoiling", "steady state of spgr / afi", "ideal spoiling instead of the EPG", "tier 1 of the engine", "engine", engine=dict(tier=1)),
    _p("fat_singlepeak", "fat spectrum", "one line at -3.4 ppm instead of six", "choice", "engine", engine=dict(fat_model="single")),
    _p("noise_x0.5", "receiver noise", "x 0.5", "choice", "world", world=_system("sigma0", lambda v, w: 0.5 * v)),
    _p("noise_x2", "receiver noise", "x 2", "choice", "world", world=_system("sigma0", lambda v, w: 2.0 * v)),
    _p("magnitude_only", "what a region read-out returns", "no complex mean (magnitude data only)", "Gudbjartsson & Patz 1995", "readout"),
    _p("tx_gain_x2", "transmit-gain error", "x 2", "choice", "world", world=_system("tx_gain", lambda v, w: 1 + 2 * (v - 1))),
    _p("b1_strong", "B1+ pattern", "amplitude 0.35 to 0.45 (QA phantom) / x 3 (anatomy)", "choice", "world_spec",
       spec=lambda s: dict(s, b1_shape_amp=(0.35, 0.45)) if _QA(s) else s, world=lambda w: None if w["kind"] == "qa" else _fields(_b1_x3)(w)),
    _p("b0_x5", "off-resonance field and its through-slice gradient", "x 5", "choice", "world", world=_fields(_b0_x5)),
    _p("cf_x5", "centre-frequency error", "x 5", "choice", "world", world=_system("cf_offset_hz", lambda v, w: 5 * v)),
    _p("rx_surface", "receive profile", "x exp(-(y - y_top) / 70 mm): a surface coil", "choice", "world", world=_fields(_rx_surface)),
    _p("edge_sharp", "soft boundaries of the object", "half-width 0.05 mm instead of 0.5 mm", "choice", "world_spec", spec=lambda s: dict(s, edge_mm=0.05)),
    _p("edge_soft", "soft boundaries of the object", "half-width 2.0 mm", "choice", "world_spec", spec=lambda s: dict(s, edge_mm=2.0)),
    _p("adc_low", "diffusion coefficient", "0.05 um^2/ms in every material (oil-like)", "choice", "world",
       world=_tissues(lambda t: t.__setitem__("adc_um2_ms", 0.05))),
    _p("relax_shift", "relaxation times", "T1 x 1.2, T2 x 0.8 in every material", "choice", "world", world=_tissues(_relax)),
    _p("tissue_brainweb", "tissue table", "BrainWeb 1.5 T values for white matter, grey matter and CSF", "brainweb.bic.mni.mcgill.ca", "world",
       world=_brainweb, applies=lambda s: (s or {}).get("kind") == "brain" and float((s or {}).get("b0_t", 3.0)) == 1.5),
    _p("inv_weak", "inversion efficiency", "0.80", "choice", "world", world=_system("inv_eff", lambda v, w: 0.80)),
    _p("fatsat_weak", "fat-saturation efficiency", "0.75", "choice", "world", world=_system("fatsat_eff", lambda v, w: 0.75)),
    _p("fatsat_off", "centre frequency of the fat-saturation pulse", "+60 Hz at 3 T", "choice", "world",
       world=_system("fatsat_offset_hz", lambda v, w: 60.0 * w["b0_t"] / 3.0)),
    _p("grad_delay_x5", "residual gradient delay (EPI ghost)", "x 5", "choice", "world", world=_system("grad_delay_us", lambda v, w: 5 * v)),
    _p("gnl", "gradient non-linearity", "positions scaled by 1 - 0.03 (r / 250 mm)^2", "choice", "world_spec", spec=lambda s: dict(s, gnl=0.03)),
    _p("motion_ghost", "periodic motion", "every line x (1 + 0.1 cos(2 pi 0.25 Hz t))", "Zaitsev 2015", "world", world=_motion),
)
BY_NAME = {p["name"]: p for p in PERTURBATIONS}


class _MagnitudeOnly:
    """A scanner whose region read-outs carry no complex mean (what a console that exports magnitude images gives)."""

    def __init__(self, scanner):
        self._s = scanner

    def __getattr__(self, name):
        return getattr(self._s, name)

    @staticmethod
    def _strip(rows):
        return [{k: v for k, v in r.items() if not k.startswith("complex_mean")} for r in rows]

    def scan(self, protocol, rois=None, out=None):
        r = dict(self._s.scan(protocol, rois, out))
        r["roi_readouts"] = self._strip(r["roi_readouts"])
        return r

    def read_regions(self, scan, rois):
        return self._strip(self._s.read_regions(scan, rois))


# ------------------------------------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of a binomial rate."""
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def paired_difference(base, pert, z: float = Z95) -> tuple[float, float, float]:
    """(difference, lower, upper) of pass rate ``pert`` minus ``base`` for paired binary outcomes, after Newcombe
    1998: the Wilson intervals of the two rates are combined by squaring and adding, with the correlation of the pair."""
    base, pert = np.asarray(base, bool), np.asarray(pert, bool)
    n = len(base)
    a, b, c, d = (int(np.sum(base & pert)), int(np.sum(base & ~pert)), int(np.sum(~base & pert)), int(np.sum(~base & ~pert)))
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    theta = p2 - p1
    lo = theta - math.sqrt(max(0.0, (p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2))
    hi = theta + math.sqrt(max(0.0, (u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2))
    return theta, lo, hi


def classify(delta: float, lo: float, hi: float, shift, moved: float) -> str:
    if (delta < -0.10 and hi < 0) or moved > 0.10:
        return "flips"
    if delta >= -0.05 and lo > -0.10:
        return "robust"
    if shift is not None and abs(shift) < 0.25:
        return "noise-like"
    return "undecided"


def spearman(x, y) -> float:
    from scipy import stats
    return float(stats.spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")


# ------------------------------------------------------------------------------------------------ one instance
def _scored(task, s: dict):
    """(error, tolerance) of a score dict, by the family's ``VALIDITY`` keys or by convention; None where absent."""
    v = getattr(task, "VALIDITY", {})
    err = s.get(v.get("error", "error"))
    tol = s.get(v.get("tolerance", "tolerance"))
    if tol is None:
        keys = [k for k, x in s.items() if (k.endswith("_max") or k.endswith("_tol")) and isinstance(x, (int, float)) and not isinstance(x, bool)]
        tol = s[keys[0]] if len(keys) == 1 else None
    num = lambda x: float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None   # noqa: E731
    return num(err), num(tol)


def build(cfg: dict, pert: dict | None):
    """(episode config, world, engine) of an instance under a perturbation (None: unperturbed)."""
    from .scanner import kspace
    from .scanner import world as W
    pert = pert or {}
    spec = pert["spec"](dict(cfg.get("world_spec") or {})) if pert.get("spec") else cfg.get("world_spec")
    cfg = dict(cfg, world_spec=spec)
    world = W.make_world(cfg["seed"], spec)
    if pert.get("world"):
        pert["world"](world)
    return cfg, world, kspace.Engine(world, **pert.get("engine", {}))


def play(task, policy, cfg: dict, world: dict, engine, pert: dict | None = None, score_on=None) -> dict:
    """One episode of ``policy`` on a given world and engine, through ``LocalScanner``; scored on that world, or on
    ``score_on`` = (config, world, engine) when the perturbation is one of the measurement only."""
    from .scanner import session as core
    sess = core.ScannerSession(cfg, engine=engine)
    sc = core.LocalScanner(sess)
    if pert and pert["kind"] == "readout":
        sc = _MagnitudeOnly(sc)
    out = dict(passed=False, error=None, tolerance=None, answer=None, policy_error=None)
    try:
        out["answer"] = policy(sc)
    except Exception as e:  # noqa: BLE001  a policy that cannot play under the perturbation has failed the instance
        out["policy_error"] = f"{type(e).__name__}: {e}"[:200]
    out["used_s"] = round(sess.used, 3)
    if out["answer"] is not None:
        cfg_s, world_s, engine_s = score_on or (cfg, world, engine)
        out.update(score_answer(task, cfg_s, out["answer"], world_s, engine_s, measured=sess.n_scans > 0 and sess.used <= sess.total + 1e-6))
    return out


def score_answer(task, cfg, answer, world, engine, measured=True) -> dict:
    problem = task.check(cfg, answer, world["console"]) if hasattr(task, "check") else None
    if problem:
        return dict(passed=False, error=None, tolerance=None, refused=problem[:200])
    s = task.score(cfg, answer, world=world, engine=engine)
    err, tol = _scored(task, s)
    return dict(passed=bool(s["passed"] and measured), error=err, tolerance=tol)


def _instance(args):
    """Every row for one instance: {row: {policy: outcome}}; the reference's row also holds ``answer_moved``."""
    import mrgym

    from . import gate
    env_id, level, index, draw, names, policies, extra = args
    spec = mrgym.spec(env_id)
    task = importlib.import_module(spec.family + ".task")
    controls = importlib.import_module(spec.family + ".controls")
    table, known = controls.policies(), dict(BY_NAME, **{p["name"]: p for p in extra})
    cfg0 = dict(gate._instance(spec, level, f"validity:{index}"), noise_draw=draw)
    out, base_answer, base_passed, default = {}, None, False, None
    for name in ["default", *names]:
        pert = known.get(name)
        cfg, world, engine = build(cfg0, pert)
        if name == "default":
            default = (cfg, world, engine)
        row = {}
        for pol in policies:
            entry = table[pol]
            r = play(task, entry[0] if isinstance(entry, tuple) else entry, cfg, world, engine, pert,
                     score_on=default if pert and pert.get("score_on") == "default" else None)
            row[pol] = {k: r[k] for k in ("passed", "error", "tolerance", "policy_error", "used_s")}
            if pol == controls.REFERENCE:
                if name == "default":
                    base_answer, base_passed = r["answer"], r["passed"]
                elif base_passed:
                    row[pol]["answer_moved"] = not score_answer(task, cfg, base_answer, world, engine)["passed"]
        out[name] = row
    return out


# ------------------------------------------------------------------------------------------------ the run
def applicable(spec_of_world: dict | None, only=None) -> list[str]:
    return [p["name"] for p in PERTURBATIONS if (not only or p["name"] in only) and p.get("applies", lambda s: True)(spec_of_world)]


def run(env_id: str, n: int = 30, level: int | None = None, draws: int = 1, only=None, workers: int = 1, all_policies: bool = False,
        extra=()) -> dict:
    """Re-score ``env_id`` under every applicable perturbation. Returns the result dict (see ``write``).

    ``extra``: further perturbation dicts (the fields of ``PERTURBATIONS``). ``score_on="default"`` in one of them
    scores the deliverable on the unperturbed world: a perturbation of the measurement only, as a planted test.
    With more than one worker the functions in ``extra`` have to be importable (module level), not lambdas."""
    import mrgym

    from . import gate
    spec = mrgym.spec(env_id)
    if not spec.family:
        raise mrgym.Error(f"{env_id} is a sandbox: nothing is scored")
    task = importlib.import_module(spec.family + ".task")
    controls = importlib.import_module(spec.family + ".controls")
    level = level or spec.default_level
    known = dict(BY_NAME, **{p["name"]: p for p in extra})
    unknown = sorted(set(only or ()) - set(known))
    if unknown:
        raise mrgym.Error(f"unknown perturbation(s) {unknown}; known: {sorted(known)}")
    names = applicable(gate._instance(spec, level, "validity:0").get("world_spec"), only) + [p["name"] for p in extra if not only or p["name"] in only]
    policies = [controls.REFERENCE]
    if all_policies:
        policies += [p for p in controls.policies() if p != controls.REFERENCE and not isinstance(controls.policies()[p], tuple)]
    t0 = time.time()
    jobs = [(env_id, level, i, d, names, policies, tuple(extra)) for i in range(n) for d in range(draws)]
    workers = max(1, min(int(workers), 2))
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            inst = pool.map(_instance, jobs, chunksize=1)
    else:
        inst = [_instance(j) for j in jobs]
    ref = controls.REFERENCE
    base = np.array([r["default"][ref]["passed"] for r in inst])
    lo0, hi0 = wilson(int(base.sum()), len(base))
    handled = getattr(task, "VALIDITY", {}).get("handled_by", {})
    rows = []
    for name in names:
        p = known[name]
        ok = np.array([r[name][ref]["passed"] for r in inst])
        delta, lo, hi = paired_difference(base, ok)
        shifts = [(r[name][ref]["error"] - r["default"][ref]["error"]) / r["default"][ref]["tolerance"] for r in inst
                  if r[name][ref]["error"] is not None and r["default"][ref]["error"] is not None and r["default"][ref]["tolerance"]]
        shift = float(np.median(shifts)) if shifts else None
        moved = [r[name][ref]["answer_moved"] for r in inst if "answer_moved" in r[name][ref]]
        moved_share = float(np.mean(moved)) if moved else 0.0
        cls = classify(delta, lo, hi, shift, moved_share)
        errors = sorted({r[name][ref]["policy_error"] for r in inst if r[name][ref]["policy_error"]})
        rows.append(dict(name=name, what=p["what"], size=p["size"], kind=p["kind"], source=p["source"], passed=int(ok.sum()), n=len(ok),
                         **{"pass": round(float(ok.mean()), 4)}, delta=round(delta, 4), ci95=[round(lo, 4), round(hi, 4)],
                         shift_in_tolerances=None if shift is None else round(shift, 3), answer_moved=round(moved_share, 4),
                         **{"class": cls}, follows=bool(cls == "flips" and not (delta < -0.10 and hi < 0)), handled_by=handled.get(name),
                         policy_errors=errors[:2]))
    v = getattr(task, "VALIDITY", {})
    result = dict(
        schema=SCHEMA, env_id=env_id, level=level, engine=mrgym.ENGINE_ID, package_version=mrgym.__version__,
        date_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        command=f"mrgym validate --perturb {env_id} --level {level} --n {n}" + (f" --draws {draws}" if draws != 1 else "")
                + (" --only " + ",".join(only) if only else "") + (" --all-policies" if all_policies else "") + " --write",
        seeds=f"validity:0..validity:{n - 1}", draws=draws, instances=len(inst), reference=ref,
        default={"passed": int(base.sum()), "pass": round(float(base.mean()), 4), "ci95": [round(lo0, 4), round(hi0, 4)]},
        rows=rows, claim=v.get("claim"), not_claimed=list(v.get("not_claimed", [])),
        not_run={k: w for k, w in NOT_BUILT.items() if not only},
        not_applicable=[p["name"] for p in PERTURBATIONS if p["name"] not in names and not only],
        untestable=list(UNTESTABLE), wall_seconds=round(time.time() - t0, 1))
    if all_policies:
        rates = {pol: {name: float(np.mean([r[name][pol]["passed"] for r in inst])) for name in ["default", *names]} for pol in policies}
        mean = {pol: float(np.mean([rates[pol][name] for name in names])) for pol in policies}
        must_fail = [pol for pol in policies if pol in controls.MUST_FAIL]
        result["predictivity"] = dict(
            policies={pol: dict(default=round(rates[pol]["default"], 4), perturbed_mean=round(mean[pol], 4),
                                perturbed_max=round(max(rates[pol][name] for name in names), 4)) for pol in policies},
            spearman=round(spearman([rates[pol]["default"] for pol in policies], [mean[pol] for pol in policies]), 4),
            must_fail_max=round(max((rates[pol][name] for pol in must_fail for name in names), default=0.0), 4),
            note="fixed_recipe plays the family's default recipe here, not one tuned per level as in the gate")
    return result


# ------------------------------------------------------------------------------------------------ storage and text
def _path(env_id: str, directory: str | None = None) -> str:
    d = directory or str(importlib.resources.files("mrgym.envs").joinpath("results"))
    return os.path.join(d, env_id.replace("/", "-") + ".validity.json")


def load(env_id: str) -> dict | None:
    try:
        with open(_path(env_id), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError):
        return None


def write(result: dict, directory: str | None = None) -> str:
    path = _path(result["env_id"], directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    return path


def problems(result: dict | None, engine_id: str) -> list[str]:
    """What keeps a stored result from backing a card: none stored, another engine, a ``flips`` row nobody handles."""
    if not result:
        return ["no stored validity result (run `mrgym validate --perturb <id> --write`)"]
    bad = []
    if result["engine"] != engine_id:
        bad.append(f"the validity result was measured on {result['engine']}, the engine is {engine_id}")
    bad += [f"{r['name']} flips the answer and is neither modelled, randomised nor declared" for r in result["rows"]
            if r["class"] == "flips" and not r["follows"] and not r["handled_by"]]
    return bad


def text_table(result: dict) -> str:
    d = result["default"]
    out = [f"== {result['env_id']} level {result['level']} on {result['engine']}: reference `{result['reference']}` under perturbed physics, "
           f"{result['instances']} paired instance(s) ({result['seeds']}), {result['wall_seconds']} s",
           f"   {'unperturbed':<16}{d['passed']:>4}/{result['instances']:<4} {100 * d['pass']:5.1f} %   95 % interval {100 * d['ci95'][0]:.1f} to {100 * d['ci95'][1]:.1f} %",
           f"   {'row':<16}{'passed':>9} {'rate':>7}  {'difference (95 % interval)':<28}{'shift/tol':>10}{'moved':>7}  class"]
    for r in result["rows"]:
        shift = "-" if r["shift_in_tolerances"] is None else f"{r['shift_in_tolerances']:+.2f}"
        cls = r["class"] + (" (the reference follows)" if r["follows"] else "") + (f" [{r['handled_by']}]" if r["handled_by"] else "")
        out.append(f"   {r['name']:<16}{r['passed']:>4}/{r['n']:<4} {100 * r['pass']:5.1f} %  {100 * r['delta']:+6.1f} ({100 * r['ci95'][0]:+6.1f} to {100 * r['ci95'][1]:+6.1f})"
                   f"{'':6}{shift:>10}{100 * r['answer_moved']:6.0f}%  {cls}" + (f"   ! {r['policy_errors'][0][:60]}" if r["policy_errors"] else ""))
    if result.get("predictivity"):
        p = result["predictivity"]
        out.append(f"   predictivity over {len(p['policies'])} policies: Spearman rho {p['spearman']} between the pass rate on default physics and the mean over "
                   f"the rows; largest pass rate of a must-fail policy in any row {100 * p['must_fail_max']:.0f} %")
    if result.get("not_run"):
        out.append("   not run (physics the engine does not have): " + ", ".join(result["not_run"]))
    open_rows = [r["name"] for r in result["rows"] if r["class"] == "flips" and not r["follows"] and not r["handled_by"]]
    out.append("   flips without a statement of how it is handled: " + (", ".join(open_rows) if open_rows else "none"))
    return "\n".join(out)


def markdown(result: dict | None) -> list[str]:
    """The "Validity" section of an environment card, from the stored result."""
    if not result:
        return ["## Validity", "", "Not measured: `mrgym validate --perturb <id> --write` has not been run for this id."]
    d = result["default"]
    out = ["## Validity", "",
           f"What could change the right answer of this environment. Measured {result['date_utc']} on engine `{result['engine']}`: `{result['command']}`. "
           f"The reference policy `{result['reference']}` replayed level {result['level']} on {result['instances']} instances ({result['seeds']}) with one piece "
           f"of physics changed per row, and was scored against the changed world. Unperturbed it passes {d['passed']} of {result['instances']} "
           f"({100 * d['ci95'][0]:.0f} to {100 * d['ci95'][1]:.0f} %, 95 % interval).", ""]
    if result.get("claim"):
        out += [f"Claim: {result['claim']}." + (" Not claimed: " + "; ".join(result["not_claimed"]) + "." if result.get("not_claimed") else ""), ""]
    out += ["| Perturbation | Size | Passed | Difference (95 % interval) | Error shift (tolerances) | Answer moved | Class | Handled by |", "|---|---|---:|---|---:|---:|---|---|"]
    for r in result["rows"]:
        shift = "-" if r["shift_in_tolerances"] is None else f"{r['shift_in_tolerances']:+.2f}"
        cls = r["class"] + (", the reference follows" if r["follows"] else "")
        out.append(f"| `{r['name']}`: {r['what']} | {r['size']} | {r['passed']}/{r['n']} | {100 * r['delta']:+.0f} ({100 * r['ci95'][0]:+.0f} to {100 * r['ci95'][1]:+.0f}) points "
                   f"| {shift} | {100 * r['answer_moved']:.0f} % | {cls} | {r['handled_by'] or '-'} |")
    out += ["", "Classes: *flips* = the pass rate falls by more than 10 points (interval excluding 0) or the unperturbed deliverable fails on more than 10 % "
            "of the perturbed worlds; *robust* = difference at least -5 points with the interval above -10; *noise-like* = neither, and the error moves by less "
            "than a quarter of the tolerance; *undecided* = more instances are needed. \"The reference follows\" = the right answer moves and the reference "
            "still passes, because it measures the quantity."]
    if result.get("predictivity"):
        p = result["predictivity"]
        out += ["", f"Policy ranking under perturbed physics ({len(p['policies'])} policies): Spearman correlation {p['spearman']} between the pass rate on default "
                f"physics and the mean over the rows; the largest pass rate of a must-fail policy in any row is {100 * p['must_fail_max']:.0f} %."]
    if result.get("not_run"):
        out += ["", "Not run, because the engine does not have the physics: " + "; ".join(f"`{k}` ({w})" for k, w in result["not_run"].items()) + "."]
    out += ["", "Cannot be tested here at all (not modelled, no emulation): " + "; ".join(result["untestable"]) + ".",
            "", "No row is a comparison with a scanner: MR Gym has never been compared with data from one."]
    return out
