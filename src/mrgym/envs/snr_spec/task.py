"""``Scanner/SNRSpec``: instances and the verifier.

The job: an examination needs two spin-echo images of prescribed geometry; in each, two marked regions must reach a
required signal-to-noise ratio. Three fields of each image are free: the repetition time, the number of averages and
the receiver bandwidth. Among the protocols that meet the requirement the scan time decides.

What makes it a measurement and a decision. The requirement is stated as a number; how much signal a region gives and
how much noise the receiver adds are hidden, so both have to be measured on cheap scans and scaled to the image with
the SNR laws (voxel volume, phase-encode lines, averages, bandwidth). How fast the signal recovers with the repetition
time depends on the hidden T1 of each material and on the hidden transmit error, how much of it survives the longer
echo time of a narrow bandwidth depends on the hidden T2, and the number of averages is an integer: the shortest scan
is a joint choice of the three fields for the region that binds, which differs from object to object.

Deliverable    {"A": {"tr_ms": 640.0, "averages": 2, "bandwidth_hz_per_px": 50.0}, "B": {...}}
Pass rule      for each image: every one of its regions reaches the required SNR on a noise-free scan of the delivered
               protocol with the analytic noise of this scanner (``Engine.noise_sd``), AND the scan time is at most
               ``TIME_FACTOR`` times the shortest scan time that does (``oracle``: exhaustive search on the hidden world).

Why two images with different regions. One protocol is one point on a scan-time axis: a protocol that ignores the
object passes whenever the hidden optimum happens to lie within the allowed band below it, which is not rare (the
single-image pass rates of the tuned constant protocols are stored with every gate result, ``fixed_recipes``). Two
images whose requirements bind on different materials make that a product.

Levels differ in the hidden-world prior and in how demanding the requirement is, never in the deliverable or the rule:
level 1 can be met with one average (the question is the repetition time), level 2 needs several (repetition time and
averages are chosen together), level 3 adds short-T2 materials (the bandwidth becomes a decision).
"""
from __future__ import annotations

import collections
import os

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import family as F
from . import oracle

FAMILY = "snr_spec"
SUBMIT_DOC = ("Deliver, for each image of the examination, the repetition time (milliseconds), the number of averages and the "
              "receiver bandwidth (hertz per pixel) of its scan.")
FIELDS = ("tr_ms", "averages", "bandwidth_hz_per_px")
IMAGES = ("A", "B")
MATRICES = dict(A=(96, 112, 128), B=(144, 160, 192))                   # readout matrix of the two images (field of view 240 mm)
SCAN = dict(sequence="se", fov_mm=240.0, slice_mm=5.0)                 # the fixed fields of both images; the rest at the console default
TIME_FACTOR = 1.15                                                     # allowed scan time relative to the shortest sufficient one
SCAN_S_RANGE = (12.0, 2400.0)                                          # instances are drawn so that the shortest scan lies in here
SNR_RANGE = (8.0, 400.0)                                               # ... and so that the required SNR is one somebody would ask for
MAX_ATTEMPTS = 60
# Examined competencies this family exercises (rows of ``mrgym.coverage``): SNR against scan time through the
# repetition time, the averages and the receiver bandwidth, at a prescribed matrix and voxel.
COMPETENCIES = ("parameter.tr", "parameter.averages", "parameter.receive_bandwidth", "parameter.matrix", "parameter.pixel",
                "parameter.voxel", "parameter.te", "qc.snr")
# Outcome labels of a delivered, measured answer that fails (worst first) and the checks behind each: (label, stems),
# a check belongs to a requirement when its name is ``<stem>_<image>``. The label is the first requirement of the pass
# rule that the delivery misses, in the order in which ``score`` declares the checks of one image; it names the
# requirement, never the image (``snr_too_low`` whether image A, image B or both miss it, whatever their scan times).
REQUIREMENTS = (("snr_too_low", ("snr",)), ("scan_too_long", ("time",)))
OUTCOMES = tuple(label for label, _ in REQUIREMENTS)
BASE = dict(kind="qa", b0_t=1.5, sigma0=0.12, sigma0_log_sd=0.6, b0_hz=20.0, tx_gain_dev=(0.05, 0.20))
QA = dict(n_ring=(4, 5), vial_t1_ms=(300.0, 2000.0))
# ``demand``: the required SNR as a multiple of what ONE fully relaxed average gives in the weaker region of the
# image at its best bandwidth, drawn log-uniformly. Below about 0.8 one average is the shortest scan; above, several.
LEVELS = {
    1: dict(world_spec=dict(BASE, qa=dict(QA, vial_t2_ms=(40.0, 200.0))), demand=(0.2, 0.7), budget_s=330.0,
            summary="one average is enough: how short can the repetition time be for the hidden T1 and the hidden noise"),
    2: dict(world_spec=dict(BASE, qa=dict(QA, vial_t2_ms=(40.0, 200.0))), demand=(1.05, 2.2), budget_s=330.0,
            summary="several averages are needed: repetition time and the integer number of averages are chosen together"),
    3: dict(world_spec=dict(BASE, qa=dict(QA, vial_t2_ms=(12.0, 60.0))), demand=(0.4, 2.2), budget_s=330.0,
            summary="as levels 1 and 2 with short-T2 materials: the narrowest bandwidth is no longer the best one"),
}


# ------------------------------------------------------------------------------------------------ instances
def _sig3(x: float) -> float:
    return float(f"{x:.3g}")


def _draw(seed: str, level: int):
    """One draw of the task on the world of ``seed``: the images, or None when a requirement would not be a sensible
    one (the required SNR outside ``SNR_RANGE``, its shortest scan outside ``SCAN_S_RANGE`` or on the edge of the
    search grid). The decision uses what the task states and the optimum on the material values, never a policy."""
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    world = W.make_world(seed, lv["world_spec"], layout_only=True)
    marked = sorted(world["regions"], key=lambda r: int(r["id"][1:]))[:2 * len(IMAGES)]   # ids are a random permutation of the vials
    images = {}
    for k, name in enumerate(IMAGES):
        matrix = int(rng.choice(MATRICES[name]))
        demand = float(np.exp(rng.uniform(*np.log(lv["demand"]))))
        img = dict(scan=dict(SCAN, matrix=matrix), regions=[r["id"] for r in marked[2 * k:2 * k + 2]], snr_min=1.0)
        table = oracle.snr_table(world, img)
        img["snr_min"] = _sig3(demand * table["snr"][:, :, -1].max(1).min())
        if not SNR_RANGE[0] <= img["snr_min"] <= SNR_RANGE[1]:
            return None
        best = oracle.optimum(table, img)
        if best is None or best["at_grid_edge"] or not SCAN_S_RANGE[0] <= best["scan_s"] <= SCAN_S_RANGE[1]:
            return None
        images[name] = img
    return images, F.public_regions(marked)


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    for attempt in range(MAX_ATTEMPTS):
        world_seed = F.reseed(seed, attempt)
        drawn = _draw(world_seed, level)
        if drawn:
            break
    else:
        raise RuntimeError(f"no instance with an answer in {MAX_ATTEMPTS} draws")
    images, regions = drawn
    lv = LEVELS[level]
    task = dict(family=FAMILY, level=level, regions=regions, time_factor=TIME_FACTOR,
                images={k: dict(scan=dict(v["scan"], te_ms="console minimum", partial_fourier=1.0, phase_fov=1.0),
                                pixel_mm=v["scan"]["fov_mm"] / v["scan"]["matrix"], regions=v["regions"], snr_min=v["snr_min"])
                        for k, v in images.items()},
                snr_definition="magnitude of the mean complex signal of the region's pixels / standard deviation of the noise "
                               "in one pixel, one channel (real or imaginary part)",
                deliverable={k: {"tr_ms": "<number, milliseconds>", "averages": "<integer>",
                                 "bandwidth_hz_per_px": "<number, hertz per pixel>"} for k in images})
    return dict(task_id=task_id or "snrspec-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=world_seed,
                world_spec=lv["world_spec"], budget_s=float(lv["budget_s"]), max_requests=40, task=task)


image_of = oracle.image_of


def statement(cfg: dict) -> str:
    t = cfg["task"]
    lines = []
    for name, img in t["images"].items():
        s = img["scan"]
        lines.append(f"    image {name}: sequence {s['sequence']}, fov_mm {s['fov_mm']:g}, matrix {s['matrix']} (pixel {img['pixel_mm']:.4g} mm), "
                     f"slice_mm {s['slice_mm']:g}; SNR at least {img['snr_min']:g} in {' and '.join(img['regions'])}")
    names = " and ".join(t["images"])
    return f"""# Meet a signal-to-noise requirement in the shortest scan time
A test object with vials of unknown materials lies in the field of
view. The examination needs {len(t['images'])} spin-echo images, {names}. Each has a prescribed geometry and a required signal-to-noise
ratio (SNR) in two marked regions:
{chr(10).join(lines)}
For each image you choose tr_ms, averages and bandwidth_hz_per_px. Everything else is fixed: the fields above, te_ms at
the console minimum for the bandwidth you choose, full k-space, every other parameter at its default.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
{F.region_table(t['regions'])}
SNR of a region = {t['snr_definition']}.
Deliver the three numbers for each image with `submit`. The images are not run in this episode: the scanner evaluates
the protocols you deliver. {F.OWN_SCANS} {F.NOT_CHARGED}
Nothing about the materials, the noise or the state of the system is known in advance.

Passing: for each image, (1) every one of its regions reaches the required SNR, evaluated on a noise-free scan of your
protocol with the noise level of this scanner, and (2) its scan time is at most {t['time_factor']:g} times the shortest scan time
with which any choice of tr_ms, averages and bandwidth_hz_per_px meets (1). The scanner's log must show that you
measured. An answer in the wrong unit is wrong."""


def submit_schema(cfg: dict) -> dict:
    spec = console.SEQUENCES["se"]
    one = {"type": "object", "additionalProperties": False, "required": list(FIELDS), "properties": {
        "tr_ms": {"type": "number", "minimum": spec["tr_ms"][0], "maximum": spec["tr_ms"][1], "description": "repetition time, milliseconds"},
        "averages": {"type": "integer", "minimum": spec["averages"][0], "maximum": spec["averages"][1], "description": "number of averages"},
        "bandwidth_hz_per_px": {"type": "number", "minimum": spec["bandwidth_hz_per_px"][0], "maximum": spec["bandwidth_hz_per_px"][1],
                                "description": "receiver bandwidth, hertz per pixel"}}}
    names = list(cfg["task"]["images"])
    return {"type": "object", "additionalProperties": False, "required": names,
            "properties": {k: dict(one, description=f"the scan of image {k}") for k in names}}


# ------------------------------------------------------------------------------------------------ the verifier
def _request(task: dict, name: str, fields) -> dict:
    return dict(image_of(task, name)["scan"], **fields)


def check(cfg: dict, answer, exam: dict):
    names = list(cfg["task"]["images"])
    if not isinstance(answer, dict) or set(answer) != set(names) \
            or any(not isinstance(answer[k], dict) or set(answer[k]) != set(FIELDS) for k in names):
        return "the deliverable is {" + ", ".join(f'"{k}": {{"tr_ms": <number>, "averages": <integer>, "bandwidth_hz_per_px": <number>}}'
                                                  for k in names) + "}"
    for k in names:
        pl = console.plan(_request(cfg["task"], k, answer[k]), exam)
        if not pl["accept"]:
            return f"the console rejects the scan of image {k}: " + "; ".join(v["message"] for v in pl["violations"])
    return None


_OPTIMA = collections.OrderedDict()


def optima(cfg: dict, world: dict, engine) -> dict:
    """The shortest sufficient scan of every image on the hidden world, {image: ``oracle.optimum``}. Cached per
    instance: every episode of the same instance in this process shares one search."""
    key = (cfg["seed"], cfg["level"], repr(sorted(cfg["task"]["images"].items())))
    if key not in _OPTIMA:
        _OPTIMA[key] = {k: oracle.shortest(world, image_of(cfg["task"], k), engine) for k in cfg["task"]["images"]}
        while len(_OPTIMA) > 16:
            _OPTIMA.popitem(last=False)
    return _OPTIMA[key]


def regions_scored(task: dict, name: str) -> list:
    """The circles whose SNR is checked for image ``name``."""
    return [r for r in task["regions"] if r["id"] in task["images"][name]["regions"]]


def meets(snr: float, snr_min: float) -> bool:
    return bool(snr >= snr_min)


def in_time(scan_s: float, shortest_s: float, factor: float) -> bool:
    return bool(scan_s <= factor * shortest_s)


def measure(cfg: dict, name: str, fields: dict, world: dict, engine) -> dict:
    """Verifier side: scan time and the SNR of every scored region of image ``name`` for the delivered ``fields``,
    on a noise-free scan with the analytic noise of the realised protocol."""
    pl = console.plan(_request(cfg["task"], name, fields), world["console"])
    rp = pl["realised"]
    sig, sd = F.region_complex(engine, rp, regions_scored(cfg["task"], name)), engine.noise_sd(rp)
    return dict(scan_s=pl["scan_s"], snr={i: float(abs(v) / sd) for i, v in sig.items()}, realised=rp)


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t, best = cfg["task"], optima(cfg, world, engine)
    checks, bad, detail = dict(deliverable_valid=True), [], {}
    for name, img in t["images"].items():
        m = measure(cfg, name, answer[name], world, engine)
        shortest = best[name]["scan_s"] if best[name] else float("nan")
        snr_ok = all(meets(v, img["snr_min"]) for v in m["snr"].values())
        time_ok = bool(best[name]) and in_time(m["scan_s"], shortest, t["time_factor"])
        checks[f"snr_{name}"], checks[f"time_{name}"] = snr_ok, time_ok
        low = min(m["snr"], key=m["snr"].get)
        if not snr_ok:
            bad.append(f"image {name}: SNR {m['snr'][low]:.3g} in {low}, required {img['snr_min']:g}")
        if not time_ok:
            bad.append(f"image {name}: scan time {m['scan_s']:.1f} s, more than {t['time_factor']:g} x the shortest sufficient scan")
        detail[name] = dict(scan_s=m["scan_s"], shortest_s=shortest, time_ratio=m["scan_s"] / shortest, snr=m["snr"],
                            snr_margin=m["snr"][low] / img["snr_min"] - 1.0, optimum=best[name])
    # ``error`` is the cost of the delivered examination: the summed scan time, to be compared with the oracle's T*;
    # ``regret`` is what the pass rule looks at: the largest scan time relative to the shortest sufficient one, minus 1
    return F.result(checks, "; ".join(bad), error=float(sum(d["scan_s"] for d in detail.values())), images=detail,
                    regret=max(d["time_ratio"] for d in detail.values()) - 1.0, time_factor=t["time_factor"])


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``, the first requirement
    of ``REQUIREMENTS`` with a failed check. A function of the names and values of ``score["checks"]`` and nothing else."""
    failed = [k for k, ok in score["checks"].items() if not ok]
    missed = lambda stems: any(k == s or k.startswith(s + "_") for k in failed for s in stems)   # noqa: E731
    return next((label for label, stems in REQUIREMENTS if missed(stems)), OUTCOMES[-1])


def task_kind(stored: dict | None = None) -> dict:
    """Measurement task or decision task, by the rule of this package: when a fixed list of scans with a fixed
    estimator (the ``open_loop`` control) passes within 25 points of the reference at every level, the task is a
    measurement task; otherwise a decision task. Read from a gate result (default: the stored one), never typed."""
    from ... import gate
    stored = stored or gate.load("Scanner/SNRSpec-v0")
    if not stored:
        return dict(kind="undetermined", evidence="no stored gate result",
                    sentence="Measurement task or decision task (does a fixed list of scans pass within 25 points of the "
                             "reference?): not determined, the gate result is not stored yet.")
    gaps = {}
    for lv, s in stored["levels"].items():
        rate = {r["policy"]: r["rate"] for r in s["reported"]}.get("open_loop")
        if rate is None:
            return dict(kind="undetermined", evidence="the stored gate result has no open_loop row",
                        sentence="Measurement task or decision task: not determined, the gate result has no open-loop control.")
        gaps[lv] = round(100 * (s["reference"]["rate"] - rate), 1)
    kind = "measurement" if max(gaps.values()) < 25.0 else "decision"
    n = min(s["n"] for s in stored["levels"].values())
    points = ", ".join(f"{v:g} at level {lv}" for lv, v in sorted(gaps.items()))
    return dict(kind=kind, evidence=f"reference minus open_loop, points of pass rate: {points} (n = {n} per level, engine {stored['engine']})",
                sentence=f"This is a {kind} task: a fixed list of scans with a fixed solver passes "
                         + ("within 25 points of the reference at every level" if kind == "measurement" else
                            "more than 25 points below the reference at some level")
                         + f" (reference minus open loop, points: {points}; {n} instances per level).")


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = task_kind()
    return dict(
        id="Scanner/SNRSpec-v0", title="Meet a signal-to-noise requirement in the shortest scan time", kind="candidate",
        given="A test object with vials of hidden materials; the scanner; a scan-time budget for probing",
        deliver="For each of two spin-echo images of prescribed geometry: repetition time, number of averages and receiver bandwidth",
        scored_by="Noise-free scans of the delivered protocols on the hidden object with the analytic noise of the hidden "
                  "receiver: every marked region must reach its required SNR, in a scan time within a stated factor of the "
                  "shortest that does (exhaustive search on the hidden world)",
        summary="The everyday protocol question: this much SNR at this pixel size, as fast as possible. Signal and noise "
                "have to be measured on cheap scans and scaled with the SNR laws (voxel volume, lines, averages, bandwidth); "
                "the repetition time trades recovery of the hidden T1 against time, the bandwidth trades noise against decay "
                "of the hidden T2 over a longer echo time, and averages come in integers. " + kind["sentence"],
        task_kind=dict(kind=kind["kind"], evidence=kind["evidence"]),
        world=dict(phantom="QA phantom (a disc of gel with vials), 1.5 T", sequences_useful=["se"],
                   agent_side_formulas="from the datasheet: the spin-echo signal mechanism (recovery with the repetition time, "
                                       "exp(-TE / T2), the slice profile), the scan-time rule, noise variance proportional to "
                                       "the receiver bandwidth. Not given: how SNR scales with voxel size, lines and averages, "
                                       "and the statistics of the magnitude of noise"),
        hidden=["T1, T2 and proton density of every material", "receiver noise level", "transmit gain and B1 pattern",
                "receive profile", "off-resonance field", "the 128-bit world seed"],
        told=["the geometry of the two images", "the regions and the required SNR of each image", "the definition of SNR",
              "the allowed scan-time factor", "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=40, max_tool_calls=48),
        deliverable={"tool": "submit", "format": {k: {"tr_ms": "<milliseconds>", "averages": "<integer>",
                                                      "bandwidth_hz_per_px": "<hertz per pixel>"} for k in IMAGES}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan",
                           "the console accepts both delivered scans"],
                    error="summed scan time of the delivered scans, seconds (the oracle's T* is the shortest sufficient sum)",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for each image, SNR >= the requirement in both regions and scan time <= {TIME_FACTOR:g} x the shortest sufficient one"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scans are noise-free and its noise is analytic",
    )
