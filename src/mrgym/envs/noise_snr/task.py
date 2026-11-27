"""``Scanner/NoiseSNR``: instances and the verifier.

The job: say what signal-to-noise ratio (SNR) a stated protocol gives in three marked regions of a uniform phantom.
The signal of a region is read off one image. The noise is not: the standard deviation of the pixels of a region
holds the shading of the receive coil and whatever structure lies in it, and the background of a magnitude image is
not Gaussian. The standards give two ways (NEMA MS 1, AAPM Report 100):

    two acquisitions   scan the protocol twice, subtract the images pixel by pixel; the object cancels, the noise of
                       the two adds in quadrature: noise = SD(difference) / sqrt(2)
    background         SD of a region of air in one magnitude image. The magnitude of complex Gaussian noise is
                       Rayleigh distributed: SD = 0.655 x noise, mean = 1.253 x noise (their ratio 1.91 is the check).
                       Wrong as soon as anything but noise lies in that region: a ghost, an interference line

Deliverable    {"snr": {"R1": 212.0, "R2": 171.0, "R3": 140.5}}        one SNR per marked region
Pass rule      every delivered SNR is within ``TOLERANCE`` (10 %) of: the region's mean magnitude in a noise-free scan
               of the stated protocol on the hidden system, divided by the true noise standard deviation of one
               pixel of that protocol (``Engine.noise_sd``: one component of the complex noise, which is the SD of a
               magnitude pixel where the signal is many times the noise).

What is hidden and decides the answer: the noise density of the receiver (drawn so that the SNR of the middle region
lies anywhere between 60 and 600), the size and the direction of the fall of the receive sensitivity across the
phantom (the largest of the three signals exceeds the smallest by more than the pass window, so no single number
fits them: tests/test_noise_snr.py), the transmit gain.
Why three regions: one SNR in a known range is passed by a constant one time in ten; three regions on a ring under a
hidden shading direction are not. The noise is the same in every region, so one noise measurement serves all three.

Levels
    1   the two-acquisition method is stated in the task text; field of view 250 mm: the corners are clean air
    2   the method is withheld; the same worlds. Both methods above pass here
    3   the field of view is 200 mm (the phantom fills it) and the signal is unstable from pulse to pulse: a ghost
        of the phantom lies over every piece of background that is left. Scan time for the protocol twice and
        nothing else. The background method has no clean region; the two acquisitions still work

A limit of the engine, stated: the instability of a session repeats identically in a repeated scan (the phase of a
mover is a function of the protocol, ``kspace.Engine._motion``), so the ghost cancels exactly in the difference image.
On a real scanner the ghost changes from scan to scan and the difference image then counts it as noise.
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import family as F

FAMILY = "noise_snr"
ENV_ID = "Scanner/NoiseSNR-v0"
SUBJECT = "C"                            # the system: characterising the instrument
KIND = "measure"
ISOLATES = "receiver noise: its level and its dependence on bandwidth"
SUBMIT_DOC = "Deliver, for each marked region, the signal-to-noise ratio of the stated protocol (a plain ratio)."
COMPETENCIES = ("qc.snr", "qc.ghosting")     # the SNR test; at level 3 a ghost has to be recognised in the background
OUTCOMES = ("no_region_within", "some_regions_within")     # of a delivered, measured answer that fails (worst first)
TOLERANCE = 0.10                         # relative. A noise SD from N differences scatters by 1 / sqrt(2 N) of its value
GRID_N = 800                             # master grid of the uniform section: 0.32 mm (smallest pixel 0.64 mm)
DIAMETER_MM = 190.0                      # inner diameter of the phantom (``anatomy.ACR``)
RING_MM, REGION_RADIUS_MM = 55.0, 18.0   # the three regions: on a ring round the isocentre, 120 degrees apart
SNR_RANGE = (60.0, 600.0)                # SNR of the middle region (log-uniform)
FALL = (0.60, 0.95)                      # fall of the receive sensitivity across the phantom (``acr_fields`` rx_fault)
GHOST_SHIFT = (0.30, 0.50)               # level 3: displacement of the ghost, as a share of the phase field of view
GHOST_DEPTH = (0.15, 0.25)               # level 3: depth of the pulse-to-pulse modulation of the signal
SYSTEM = dict(tx_gain_dev=(0.0, 0.03), cf_offset_hz=5.0, grad_delay_us=0.4)     # a healthy system otherwise
# The stated protocol is drawn per instance from these (a spin echo; every combination is accepted by the console).
PROTOCOL = dict(tr_ms=(300.0, 400.0, 500.0, 600.0), te_ms=(20.0, 30.0), matrix=(192, 224, 256),
                bandwidth_hz_per_px=(130.0, 200.0, 260.0), averages=(1, 2), pe_dir=("y", "x"))
LEVELS = {
    1: dict(fov_mm=250.0, ghost=False, scans=3.0, hints=("method",),
            summary="the two-acquisition method is stated; the corners of the image are clean air; scan time for the protocol three times"),
    2: dict(fov_mm=250.0, ghost=False, scans=3.0,
            summary="the method is withheld; the same worlds"),
    3: dict(fov_mm=200.0, ghost=True, scans=2.0,
            summary="the phantom fills the field of view and a ghost lies over the background that is left; scan time for the protocol twice"),
}
BUDGET_SPARE_S = 1.0


def protocol_of(task: dict) -> dict:
    """The stated protocol as a request to the console."""
    return dict(task["protocol"])


def budget_s(level: int, scan_s: float) -> float:
    return round(LEVELS[level]["scans"] * scan_s + BUDGET_SPARE_S, 1)


def _regions(angle_deg: float) -> list:
    out = []
    for k in range(3):
        a = math.radians(angle_deg + 120.0 * k)
        out.append(dict(id=f"R{k + 1}", y_mm=round(RING_MM * math.sin(a), 1), x_mm=round(RING_MM * math.cos(a), 1), radius_mm=REGION_RADIUS_MM))
    return out


def truth(engine, protocol: dict, regions) -> dict:
    """Verifier side: dict(snr={region id: noise-free mean magnitude / noise SD}, signal=..., noise_sd=...) of the
    protocol on the hidden system. The noise SD is the engine's own (``Engine.noise_sd``): receiver noise density x
    sqrt(bandwidth of the readout / averages), carried through the reconstruction."""
    rp = console.plan(protocol, engine.w["console"])["realised"]
    _, rows = F.noise_free(engine, rp, regions)
    sd = engine.noise_sd(rp)
    signal = {r["id"]: float(rows[r["id"]]["mean_magnitude"][0]) for r in regions}
    return dict(snr={k: v / sd for k, v in signal.items()}, signal=signal, noise_sd=sd)


@functools.lru_cache(maxsize=256)
def _instance(seed: str, level: int) -> tuple:
    """(protocol items, regions, world spec, scan seconds, hidden SNR per region) of an instance. Every draw comes
    from the task stream of the seed; the noise density is then set so that the middle region has the drawn SNR (the
    noise-free signal does not depend on it). Cached: every policy of a gate run asks for the same instance."""
    from ...scanner import kspace
    lv = LEVELS[level]
    rng = W.rng_for(seed, "task")
    field = (1.5, 3.0)[int(rng.integers(2))]
    protocol = dict(sequence="se", fov_mm=lv["fov_mm"], slice_mm=5.0)
    for name, values in PROTOCOL.items():
        protocol[name] = values[int(rng.integers(len(values)))]
    angle, fall = float(rng.uniform(0.0, 120.0)), float(rng.uniform(*FALL))
    target = float(math.exp(rng.uniform(math.log(SNR_RANGE[0]), math.log(SNR_RANGE[1]))))
    shift, depth = float(rng.uniform(*GHOST_SHIFT)), float(rng.uniform(*GHOST_DEPTH))     # drawn at every level, used at level 3
    regions = _regions(angle)
    spec = dict(SYSTEM, kind="acr", n=GRID_N, b0_t=field, acr=dict(slice=7), acr_fields=dict(rx_fault=fall), sigma0=1.0, sigma0_log_sd=0.0)
    if lv["ghost"]:
        # amplitude x (1 + depth cos(2 pi f t)): a copy of the phantom ``shift`` of the phase field of view away, both ways
        spec["motion"] = [dict(cls="*", kind="pulse", f_hz=shift / (protocol["averages"] * protocol["tr_ms"] * 1e-3), depth=depth)]
    world = W.make_world(seed, spec)
    pl = console.plan(protocol, world["console"])
    if not pl["accept"]:
        raise RuntimeError("the stated protocol is not accepted: " + "; ".join(v["message"] for v in pl["violations"]))
    t = truth(kspace.Engine(world), protocol, regions)                   # with the placeholder noise density 1.0
    middle = float(np.median(list(t["signal"].values())))
    spec["sigma0"] = middle / (target * t["noise_sd"])                   # the noise SD is linear in the density
    snr = {k: round(v / (spec["sigma0"] * t["noise_sd"]), 4) for k, v in t["signal"].items()}
    return tuple(sorted(protocol.items())), tuple(tuple(sorted(r.items())) for r in regions), spec, float(pl["scan_s"]), tuple(sorted(snr.items())), field


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    protocol, regions, spec, scan_s, snr, field = _instance(seed, level)
    lv = LEVELS[level]
    task = dict(family=FAMILY, phantom=f"uniform section of a test phantom: a disc of solution, {DIAMETER_MM:g} mm inner diameter",
                field_strength_t=field, protocol=dict(protocol), protocol_scan_s=scan_s, regions=[dict(r) for r in regions],
                tolerance_percent=100 * TOLERANCE, deliverable={"snr": {"<region id>": "<number, a plain ratio>"}})
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    world_spec = dict(spec, acr=dict(spec["acr"]), acr_fields=dict(spec["acr_fields"]))
    if "motion" in spec:
        world_spec["motion"] = [dict(m) for m in spec["motion"]]
    return dict(task_id=task_id or "noisesnr-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=world_spec,
                budget_s=budget_s(level, scan_s), max_requests=40, task=task, hidden=dict(snr=dict(snr)))


def hint_text(cfg: dict) -> str:
    """The hints of the level, as the paragraph appended to the statement ("" when the level has none)."""
    if "method" not in cfg["task"].get("hints", ()):
        return ""
    return """How to do it (this level states the procedure: two acquisitions and their difference image).
  1. Scan the protocol twice, unchanged. The object is the same in both images, the noise is not.
  2. Signal of a region: its mean_magnitude in one of the two scans.
  3. Noise: read the same pixels of both scans with `profile`, on lines inside the phantom (the same axis, position_mm,
     start_mm and stop_mm for both scans; no width_mm; every sample must be one pixel, "samples of 1 px" in the
     answer, which a line of 128 pixels or fewer gives). Subtract the two lists value by value: the object cancels.
     Noise = standard deviation of the differences / sqrt(2).
     With N differences the result scatters by 1 / sqrt(2 N) of its value: take 500 or more.
  4. SNR of a region = its signal / the noise. The noise is the same in every region."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    p = t["protocol"]
    ids = ", ".join(r["id"] for r in t["regions"])
    hints = hint_text(cfg)
    return f"""# Measure the signal-to-noise ratio of a protocol
A test phantom lies at the isocentre of a {t['field_strength_t']:g} T scanner: a disc of solution, {DIAMETER_MM:g} mm inner
diameter, one material. The examination needs the signal-to-noise ratio (SNR) that this protocol gives in each
marked region:
    sequence {p['sequence']}, tr_ms {p['tr_ms']:g}, te_ms {p['te_ms']:g}, fov_mm {p['fov_mm']:g}, matrix {p['matrix']}, slice_mm {p['slice_mm']:g},
    bandwidth_hz_per_px {p['bandwidth_hz_per_px']:g}, averages {p['averages']}, pe_dir {p['pe_dir']}   (one scan: {t['protocol_scan_s']:.1f} s)
Regions (circles; centre and radius in mm from the isocentre; each lies inside the phantom):
{F.region_table(t['regions'])}
SNR of a region: its mean signal in the magnitude image of this protocol, divided by the standard deviation of the
noise in one pixel of that image.
Deliver one SNR per region ({ids}) with `submit`. {F.OWN_SCANS}
Every scan you run is charged to the scan-time budget, a scan of this protocol too.
Nothing about the receiver, the coil or the state of the system is known in advance.

Passing: for every region, the delivered SNR is within {t['tolerance_percent']:g} % of the scanner's own value: the mean signal of the
region in a noise-free scan of the protocol, divided by the true noise standard deviation of a pixel of that protocol.
The scanner's log must show that you measured. An answer in the wrong unit is wrong (a plain ratio, not decibels).""" \
        + ("\n\n" + hints if hints else "")


def submit_schema(cfg: dict) -> dict:
    ids = [r["id"] for r in cfg["task"]["regions"]]
    return {"type": "object", "additionalProperties": False, "required": ["snr"],
            "properties": {"snr": {"type": "object", "additionalProperties": False, "required": ids,
                                   "description": "signal-to-noise ratio of the stated protocol per region, a plain ratio",
                                   "properties": {i: {"type": "number"} for i in ids}}}}      # the range is ``check``'s: its refusal names the unit


def check(cfg: dict, answer, exam: dict):
    ids = [r["id"] for r in cfg["task"]["regions"]]
    if not isinstance(answer, dict) or set(answer) != {"snr"} or not isinstance(answer["snr"], dict) or set(answer["snr"]) != set(ids):
        return 'the deliverable is {"snr": {' + ", ".join(f'"{i}": <number>' for i in ids) + "}}"
    for i in ids:
        v = answer["snr"][i]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.01 <= v <= 1e6:
            return f"snr.{i} must be a number from 0.01 to 1000000 (a plain ratio)"
    return None


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    t = cfg["task"]
    tol = t["tolerance_percent"] / 100.0
    want = truth(engine, protocol_of(t), t["regions"])["snr"]          # re-measured on the hidden system, noise-free
    off = {i: float(answer["snr"][i]) / want[i] - 1.0 for i in want}
    ok = {i: abs(v) <= tol for i, v in off.items()}
    notes = [f"{i}: delivered {float(answer['snr'][i]):.4g}, the scanner's value is {want[i]:.4g} ({100 * off[i]:+.1f} %)" for i in want if not ok[i]]
    return F.result(dict(deliverable_valid=True, **{f"snr_{i}": ok[i] for i in want}), "; ".join(notes),
                    error=float(max(abs(v) for v in off.values())), snr={i: round(v, 4) for i, v in want.items()}, tolerance=tol,
                    distance=dict(regions=len(ok), within=sum(ok.values()), worst_relative_error=round(float(max(abs(v) for v in off.values())), 5)))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: one of ``OUTCOMES``."""
    return "some_regions_within" if any(v for k, v in score["checks"].items() if k.startswith("snr_")) else "no_region_within"


def budget_range(level: int) -> list:
    """[shortest, longest] scan-time budget of a level over the protocols it draws (console arithmetic only)."""
    times = [console.plan(dict(sequence="se", fov_mm=LEVELS[level]["fov_mm"], slice_mm=5.0, tr_ms=tr, te_ms=PROTOCOL["te_ms"][0], matrix=m,
                               bandwidth_hz_per_px=PROTOCOL["bandwidth_hz_per_px"][0], averages=a), dict(min_pixel_mm=0.5))["scan_s"]
             for tr in (min(PROTOCOL["tr_ms"]), max(PROTOCOL["tr_ms"])) for m in (min(PROTOCOL["matrix"]), max(PROTOCOL["matrix"]))
             for a in (min(PROTOCOL["averages"]), max(PROTOCOL["averages"]))]
    return [budget_s(level, min(times)), budget_s(level, max(times))]


def card() -> dict:
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    return dict(
        id=ENV_ID, title="Measure the signal-to-noise ratio of a protocol", kind="candidate",
        given="A uniform phantom in a scanner with a hidden receiver noise level and a hidden receive-coil shading; a stated "
              "spin-echo protocol; three marked regions; scan time for the protocol two or three times",
        deliver="For each marked region, the signal-to-noise ratio of the stated protocol",
        scored_by="A noise-free scan of the stated protocol on the hidden system and the true noise standard deviation of a pixel "
                  "of that protocol: every delivered ratio must agree with signal / noise",
        summary="One test of acceptance and quality control, on its own. The signal is read off the image; the noise is not: "
                "the pixel scatter inside a region holds the coil shading, and the background of a magnitude image is Rayleigh "
                "distributed and carries ghosts. The standards answer with two acquisitions and their difference image "
                "(noise = SD of the difference / sqrt(2)) or with a background region and the factor 0.655. At level 3 the "
                "phantom fills the field of view and a ghost covers the background, so of these two only the first is left.",
        task_kind="measurement",
        task_kind_why="the reference is a fixed scan list (the protocol twice) with a fixed estimator; `open_loop` plays it: no scan depends on an earlier one",
        competencies=list(COMPETENCIES),
        world=dict(phantom="ACR-type phantom, uniform section (world kind acr, slice 7), 1.5 T or 3 T", sequences_useful=["se"],
                   agent_side_formulas="noise = SD(image 1 - image 2) / sqrt(2); background of a magnitude image: SD = 0.655 x noise, "
                                       "mean = 1.253 x noise; SNR = mean signal / noise (level 1 states the first)"),
        hidden=["receiver noise density (the SNR of the middle region lies between 60 and 600)",
                "the receive sensitivity, which falls across the phantom in a hidden direction", "transmit gain",
                "level 3: the pulse-to-pulse instability that puts a ghost over the background", "the 128-bit world seed"],
        told=["the phantom", "the protocol and its scan time", "the three regions", "the definition of the SNR and the tolerance",
              "level 1: the two-acquisition procedure", "the instrument datasheet", "realised protocol, timing and cost of every plan and scan"],
        levels={lv: dict(summary=d["summary"], budget_s=budget_range(lv)) for lv, d in LEVELS.items()},
        limits=dict(max_scanner_requests=40, max_tool_calls=64),
        deliverable={"tool": "submit", "format": {"snr": {"<region id>": "<plain ratio>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan"],
                    error="largest over the regions of |delivered / (noise-free signal / true noise SD) - 1|",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing=f"all gates and, for every region, error <= {100 * TOLERANCE:g} %"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's scan is noise-free",
    )
