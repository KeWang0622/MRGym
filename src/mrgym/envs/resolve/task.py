"""``Scanner/Resolve``: instances and the verifier.

The job: choose the pixel of a scan from the detail it has to show. A test object carries three pairs of small bright
features (A, B, C) in signal-free inserts. The centre-to-centre spacing of each pair is hidden. For each pair the
examination needs one spin-echo image of the whole object in which the two features are separate, and it wants the
fastest such scan. The scan time of the target scan grows with the number of phase-encode lines, lines = phase field
of view / pixel, so the fastest scan has the coarsest pixel that still separates the pair, and the phase-encode axis
along the short side of the object.

Deliverable    {"A": {"pe_dir": "y", "fov_mm": 232.5, "phase_fov": 0.533333, "matrix": 60}, "B": {...}, "C": {...}}
Pass rule      for each pair, on a noise-free re-scan of the delivered geometry on the hidden world:
                 1. the whole object lies inside the image (then nothing wraps around: the phase field of view holds it)
                 2. the two features are separated: ``separation`` >= ``MIN_DIP``
                 3. the signal-to-noise ratio of the reference region is at least ``MIN_SNR``
                 4. the scan takes at most ``MAX_SCAN_S``
                 5. the scan takes at most ``TOL`` times the scan time of the fastest geometry that meets 1 to 4
                    (``oracle.solve``: a search of the console's geometries on the hidden world)

What "separated" means (``separation``). The image a scan delivers is band-limited: its k-space ends at 1 / (2 pixel).
Between the pixel centres it is defined by the same data (zero filling: the Fourier sum evaluated anywhere). The
magnitude of that image is read along the line through the centres of the two features, from half a spacing before
the first to half a spacing behind the second. With P1 and P2 the highest value on the half of the line nearer to
each feature and V the lowest value between those two maxima, the dip is 1 - V / min(P1, P2). Two features that have
merged into one maximum have dip 0. Read between the pixels, the dip does not depend on where the pixel centres fall.

Why a pixel as large as the spacing does not separate. The point-spread function of the console reconstruction is a
sinc with its first zero one pixel from the centre. Two points one pixel apart add to 2 sinc(1 / 2) = 1.27 half-way
between them and to 1 at each of them: one maximum, no dip. The dip of two points reaches one half at a spacing of
1.63 pixels, a pixel of 0.61 spacings (``point_pair_dip``; a test compares it with the engine); the features of this
family are discs half as wide as their spacing and need 0.57 to 0.60. Every finer pixel passes as well: from a
spacing of two pixels on, the image falls to zero between the two (the side lobe that then rises half-way between
them lies between two zeros and is not a third feature of the measure).

Why three pairs, drawn log-uniformly. One pair gives one hidden number (its spacing) and one pixel within a factor
``TOL``: a constant pixel then separates and is fast enough for the share ln(TOL) / ln(range) of the spacings, one
pair in six, whatever else is measured. Three independent spacings make the product small, and the localiser that
shows one pair shows the others. (Drawn uniformly, the largest spacings would be the likeliest window.)

Why the budget pays one localiser. A probe scan at a pixel p shows directly whether p separates a pair, and the time
factor lets every pixel within about 20 % of the best one pass. A budget of three probe scans lets a fixed list of
three pixels cover a large share of the range of spacings: a sweep without any measurement of a spacing then passes
too often to be called a shortcut. ``BUDGET_S`` pays the localiser of the method and no second scan of that kind.

Not delivered, on purpose. ``bandwidth_hz_per_px``: at a fixed repetition time it does not change the scan time, so
it is part of the stated scan. ``partial_fourier`` (the one setting of this console that makes the resolution
anisotropic): the target scan samples all of k-space. The pixel of this console is square, so the phase-encode axis
acts through the phase field of view only: at level 3 the object (and its inserts with it) lies along x or along y.

Levels. 1: the long axis of the object lies along x and the method is stated. 2: the method is withheld. 3: the
orientation of the object and of its pairs (along x or along y) is hidden as well, so the phase-encode axis has to be
chosen from the localiser.
"""
from __future__ import annotations

import collections
import json
import math
import os

import numpy as np

from ...scanner import console
from ...scanner import world as W
from .. import family as F
from ..nowrap import task as N

FAMILY = "resolve"
ENV_ID = "Scanner/Resolve-v0"
SUBMIT_DOC = ("Deliver, for each marked pair, the geometry of the target spin-echo scan that separates it: phase-encode axis, readout "
              "field of view (millimetres), phase field of view (a fraction of it) and matrix.")
TARGET = dict(sequence="se", tr_ms=500.0, bandwidth_hz_per_px=300.0)
FIELDS = ("pe_dir", "fov_mm", "phase_fov", "matrix")
PAIRS = ("A", "B", "C")
REFERENCE_REGION = "REF"
MIN_DIP = 0.5                               # the profile between the two features falls to at most half of the lower peak
MIN_SNR = 12.0                              # of the reference region in the delivered scan
MAX_SCAN_S = 60.0                           # what the examination affords for one scan
TOL = 1.20                                  # a delivered scan may take 20 % longer than the fastest valid one
COVER_SLACK_MM = N.COVER_SLACK_MM
PROFILE_SAMPLES = 241                       # points of the line profile (120 per spacing)
OUTCOMES = ("object_cut_off", "not_separated", "snr_too_low", "over_time_cap", "too_slow")   # of a delivered, measured answer that fails
REQUIREMENTS = ("whole_object_inside", "features_separated", "snr_floor", "within_time_cap", "scan_time_within_factor")
# examined competencies this family exercises (ids of ``mrgym.coverage.COMPETENCIES``)
COMPETENCIES = ("parameter.matrix", "parameter.pixel", "parameter.fov", "parameter.phase_frequency_direction",
                "option.rectangular_fov", "artefact.gibbs")
# ---- the object: an elongated disc of gel, three signal-free inserts along its long axis, a pair of bright features in each
SPACING_MM = (3.5, 10.0)                    # centre-to-centre spacing of a pair, drawn per pair, log-uniformly
FEATURE_RADIUS = 0.25                       # radius of a feature as a share of the spacing of its pair (gap = half the spacing)
INSERT_RADIUS_MM = 23.0
INSERT_OFFSET_MM = (52.0, 58.0)             # distance of the two outer inserts from the centre of the object along its long axis
INSERT_CENTRE_MM = 2.0                      # the middle insert lies within this of the centre of the object along that axis
INSERT_ACROSS_MM = 8.0                      # the largest distance of an insert from the long axis
PAIR_SHIFT_MM = 1.5                         # the midpoint of a pair lies within this of the centre of its insert, along the pair
REGION_RADIUS_MM = 11.0                     # the marked circle around a pair
REF_RADIUS_MM = 8.0
REF_ALONG_MM = (24.0, 28.0)                 # the reference region: in the gel between two inserts, this far along the long axis
REF_ACROSS_MM = (36.0, 39.0)                # ... and this far from it
CENTRED_WITHIN_MM = 2.0                     # the centre of the object lies within this of the isocentre along each axis (told)
EDGE_BAND_MM = 20.0                         # width of the profile through the isocentre on which the method reads the edges
# Declared materials of the inserts (the same in every instance; values that no console number equals by chance, so that
# ``mrgym check`` can look for them in the observations):
FEATURE = dict(pd=1.0, t1_ms=203.7, t2_ms=118.3, t2p_ms=400.0)      # a doped solution: bright at the target's repetition time
PLASTIC = dict(pd=0.0, t1_ms=1003.7, t2_ms=100.0, t2p_ms=100.0)     # no signal
BASE = dict(kind="qa", b0_t=1.5, tx_gain_dev=(0.0, 0.05), edge_mm=0.3)      # sharper boundaries than the default 0.5 mm: the features are small
# long semi-axis 104 to 114 mm, short semi-axis = long / (1.35 to 1.6) = 65 to 84 mm; inside the 256 mm master grid.
# (With a ratio of 1.35 or more the wrong phase-encode axis costs more than the time factor allows at any number of lines;
# with a short axis of 130 mm and more the fastest scan has 22 lines or more, so that two lines are less than a tenth of it.)
_QA = dict(body_radius_mm=(104.0, 114.0), body_aspect=(1.35, 1.6), n_ring=(0, 0), p_centre=0.0, offset_mm=CENTRED_WITHIN_MM)
_ALONG_X = dict(BASE, qa=dict(_QA, body_rotation_deg=(90.0,)))
BUDGET_S = 15.5                             # scan time for looking: one localiser of the whole object at a pixel of 1 mm (12.6 s)
LEVELS = {
    1: dict(hints=("rule",), world_spec=_ALONG_X, budget_s=BUDGET_S,
            summary="the object and its pairs lie along x; the method is stated"),
    2: dict(world_spec=_ALONG_X, budget_s=BUDGET_S,
            summary="the object and its pairs lie along x; the method is withheld"),
    3: dict(world_spec=dict(BASE, qa=dict(_QA, body_rotation_deg=(0.0, 90.0))), budget_s=BUDGET_S,
            summary="the object and its pairs lie along x or along y (hidden): the phase-encode axis has to be chosen"),
}
LOCALISER = dict(sequence="spgr", tr_ms=11.0, flip_deg=20.0, fov_mm=256.0, matrix=256, averages=2)
RULE_PIXEL = 0.55                           # pixel = this share of the measured spacing (the rule level 1 states)
RULE_MARGIN_MM = 2.0                        # kept between the measured edge of the object and the edge of the image


def make_instance(seed: str, level: int, task_id: str | None = None) -> dict:
    lv = LEVELS[level]
    body = W.make_world(seed, lv["world_spec"], layout_only=True)["body"]
    along_x = body["rot"] == 90.0                                        # the long axis of the object
    rng = W.rng_for(seed, "task")

    def place(u, v):                                                     # (along the long axis, across it) -> patient (y, x)
        return (body["y"] + v, body["x"] + u) if along_x else (body["y"] + u, body["x"] + v)

    order = [int(k) for k in rng.permutation(len(PAIRS))]                # which pair sits in which insert
    along = [-float(rng.uniform(*INSERT_OFFSET_MM)), float(rng.uniform(-INSERT_CENTRE_MM, INSERT_CENTRE_MM)), float(rng.uniform(*INSERT_OFFSET_MM))]
    inserts, dots, regions, hidden = [], [], {}, {}
    for k, u in zip(order, along):
        name = PAIRS[k]
        s = float(np.exp(rng.uniform(math.log(SPACING_MM[0]), math.log(SPACING_MM[1]))))
        v, shift = float(rng.uniform(-INSERT_ACROSS_MM, INSERT_ACROSS_MM)), float(rng.uniform(-PAIR_SHIFT_MM, PAIR_SHIFT_MM))
        centre, a, b = place(u, v), place(u + shift - s / 2, v), place(u + shift + s / 2, v)
        centre, a, b = ([round(float(c), 2) for c in p] for p in (centre, a, b))
        inserts.append(dict(y_mm=centre[0], x_mm=centre[1], radius_mm=INSERT_RADIUS_MM, tissue=PLASTIC))
        dots += [dict(y_mm=p[0], x_mm=p[1], radius_mm=round(FEATURE_RADIUS * s, 3), tissue=FEATURE) for p in (a, b)]
        regions[name] = dict(id=name, y_mm=round(centre[0], 1), x_mm=round(centre[1], 1), radius_mm=REGION_RADIUS_MM)
        hidden[name] = dict(a_yx_mm=a, b_yx_mm=b, spacing_mm=round(float(math.hypot(a[0] - b[0], a[1] - b[1])), 4))
    sign = [1.0 if rng.uniform() < 0.5 else -1.0 for _ in range(2)]
    ref = place(sign[0] * float(rng.uniform(*REF_ALONG_MM)), sign[1] * float(rng.uniform(*REF_ACROSS_MM)))
    regions = [regions[name] for name in PAIRS] + [dict(id=REFERENCE_REGION, y_mm=round(float(ref[0]), 1), x_mm=round(float(ref[1]), 1), radius_mm=REF_RADIUS_MM)]
    hidden = {name: hidden[name] for name in PAIRS}
    spec = dict(lv["world_spec"], qa=dict(lv["world_spec"]["qa"], inserts=inserts + dots))
    task = dict(family=FAMILY, target_scan=dict(TARGET, te_ms="console minimum", averages=1, slice_mm=5.0, centre_y_mm=0.0, centre_x_mm=0.0),
                regions=regions, pairs=list(PAIRS), reference_region=REFERENCE_REGION, min_dip=MIN_DIP, min_snr=MIN_SNR,
                max_scan_s=MAX_SCAN_S, max_time_factor=TOL,
                deliverable={"<pair id>": {"pe_dir": "y or x", "fov_mm": "<mm>", "phase_fov": "<fraction, 0.25 to 1>", "matrix": "<even integer>"}})
    if level != 3:
        task["long_axis"] = "x"                                          # told at levels 1 and 2
    if lv.get("hints"):
        task["hints"] = list(lv["hints"])
    return dict(task_id=task_id or "resolve-" + os.urandom(6).hex(), family=FAMILY, level=level, seed=seed, world_spec=spec,
                budget_s=float(lv["budget_s"]), max_requests=32, task=task,
                hidden=dict(pairs=hidden, long_axis="x" if along_x else "y"))


def hint_text(cfg: dict) -> str:
    """The method, as the paragraph appended to the statement of a level that states it ("" otherwise)."""
    if "rule" not in cfg["task"].get("hints", ()):
        return ""
    loc, t = LOCALISER, cfg["task"]
    cost = console.plan(loc)["scan_s"]
    return f"""How to do it (this level states the method).
  1. Take one localiser that shows the whole object with a pixel of 1 mm: `spgr` with tr_ms {loc['tr_ms']:g}, flip_deg {loc['flip_deg']:g},
     fov_mm {loc['fov_mm']:g}, matrix {loc['matrix']}, averages {loc['averages']}. It takes {cost:.1f} s: the budget pays one such scan.
  2. Edges of the object with `profile`: a profile along y through the isocentre (position_mm 0, width_mm {EDGE_BAND_MM:g}). Take
     low and high where it first and last crosses half of the level of the gel; the same along x (the dark inserts
     lie on that line: only the outermost crossings count).
  3. Spacing of a pair: `profile` along x through the centre of its region (position_mm = the region's y, width_mm 4,
     from the region's x - {REGION_RADIUS_MM:g} to x + {REGION_RADIUS_MM:g}). It shows two peaks. Let h be half of the highest value. The samples above
     h form two groups; the position of a group is the mean of its positions weighted with (value - h).
     Spacing S = the difference of the two positions.
  4. Pixel P = {RULE_PIXEL:g} x S (a pixel as large as the spacing merges the two; {RULE_PIXEL:g} x S gives a dip of at least {t['min_dip']:g}).
  5. Phase-encode along y, the short side: pe_dir y. The image reaches from -FOV / 2 - P / 2 to FOV / 2 - P / 2 about the
     isocentre, so along each axis half = max(-low - P / 2, high + P / 2) + {RULE_MARGIN_MM:g} mm.
     lines = 2 x ceil(half_y / P), matrix = max(2 x ceil(half_x / P), lines, 32), fov_mm = P x matrix,
     phase_fov = lines / matrix."""


def statement(cfg: dict) -> str:
    t = cfg["task"]
    where = ("The long axis of the object lies along x,\nand so does the line through the two features of each pair."
             if t.get("long_axis") else
             "The long axis of the object lies along x or along y,\nand the line through the two features of each pair lies along it.")
    hints = hint_text(cfg)
    return f"""# Prescribe the fastest scan that separates each marked pair of features
A test object lies in the scanner: an elongated disc of gel with three signal-free inserts, centred on the isocentre to
within {CENTRED_WITHIN_MM:g} mm. Each insert holds a pair of small bright features. {where}
For each pair the examination needs one spin-echo image of the whole object in which the two features are separate:
    sequence se, tr_ms {TARGET['tr_ms']:g}, te_ms at the console minimum, bandwidth_hz_per_px {TARGET['bandwidth_hz_per_px']:g}, 1 average, slice 5 mm,
    field of view centred on the isocentre, not rotated, all of k-space sampled.
Regions (circles; centre and radius in mm from the isocentre; A, B and C each hold one pair, REF lies in the gel):
{F.region_table(t['regions'])}
Deliver one geometry per pair ({', '.join(t['pairs'])}) with `submit`: the phase-encode axis (`pe_dir`), the readout field of view
(`fov_mm`), `phase_fov` (the phase field of view as a fraction of `fov_mm`) and `matrix`. The pixel is fov_mm / matrix
along both axes. The target scans are not run in this episode: the scanner evaluates the geometries you deliver.
{F.OWN_SCANS}
{F.NOT_CHARGED}
The size of the object and the spacing of each pair are not known in advance.

Passing, for each pair, on the scan with the geometry you delivered for it:
  the whole object lies inside the image (so nothing wraps around);
  the two features are separated: along the line through their centres the image falls, between its two peaks, to at
  most {100 * (1 - t['min_dip']):g} % of the lower peak (read on the image interpolated between its pixels, so it does not matter where
  the pixel centres fall);
  the signal-to-noise ratio of region {t['reference_region']} is at least {t['min_snr']:g};
  the scan takes at most {t['max_scan_s']:g} s;
  the scan takes at most {round(100 * (t['max_time_factor'] - 1))} % longer than the fastest geometry that meets the four requirements above.
The scanner's log must show that you measured. A length in the wrong unit is wrong.""" + ("\n\n" + hints if hints else "")


def _geometry_schema() -> dict:
    g = console.SEQUENCES[TARGET["sequence"]]
    num = lambda k, what: {"type": "integer" if g[k][2] == "i" else "number", "minimum": g[k][0], "maximum": g[k][1], "description": what}   # noqa: E731
    return {"type": "object", "additionalProperties": False, "required": list(FIELDS),
            "properties": {"pe_dir": {"type": "string", "enum": ["y", "x"], "description": "phase-encode axis"},     # (the task text says
                           "fov_mm": num("fov_mm", "readout field of view"),                                             # what each is)
                           "phase_fov": num("phase_fov", "fraction of fov_mm"),
                           "matrix": num("matrix", "readout pixels, even")}}


def submit_schema(cfg: dict) -> dict:
    ids = list(cfg["task"]["pairs"])
    return {"type": "object", "additionalProperties": False, "required": ids,
            "properties": {i: dict(_geometry_schema(), description=f"geometry of the target scan for pair {i}") for i in ids}}


def protocol_of(geometry: dict) -> dict:
    """The target scan with one delivered geometry."""
    return dict(TARGET, **{k: geometry[k] for k in FIELDS})


def check(cfg: dict, answer, exam: dict):
    ids = list(cfg["task"]["pairs"])
    shape = "the deliverable is {" + ", ".join(f'"{i}": {{' + ", ".join(f'"{k}"' for k in FIELDS) + "}" for i in ids) + "}"
    if not isinstance(answer, dict) or set(answer) != set(ids):
        return shape
    for i in ids:
        g = answer[i]
        if not isinstance(g, dict) or set(g) != set(FIELDS):
            return shape
        pl = console.plan(protocol_of(g), exam)
        if not pl["accept"]:
            fov = g.get("fov_mm")
            hint = " (a field of view of this size looks like centimetres)" if isinstance(fov, (int, float)) and not isinstance(fov, bool) \
                and 0 < fov < console.SEQUENCES[TARGET["sequence"]]["fov_mm"][0] else ""
            return f"the console rejects the geometry of pair {i}: " + "; ".join(v["message"] for v in pl["violations"]) + hint
    return None


# ------------------------------------------------------------------------------------------------ the measures, verifier side
def point_pair_dip(spacing_px: float, samples: int = 4001) -> float:
    """Closed form: the dip between two equal points ``spacing_px`` pixels apart behind an ideal low-pass (the image of
    a point is sinc(x / pixel)), measured as ``separation`` measures it."""
    t = np.linspace(-0.5, 1.5, samples) * spacing_px
    return _dip(np.abs(np.sinc(t) + np.sinc(t - spacing_px)), samples)


def _dip(profile: np.ndarray, samples: int) -> float:
    half = samples // 2                                                  # the sample half-way between the two features
    i, j = int(np.argmax(profile[:half])), half + int(np.argmax(profile[half:]))
    low = float(min(profile[i], profile[j]))
    return 0.0 if low <= 0 else float(1.0 - profile[i:j + 1].min() / low)


def image_at(engine, realised: dict, points_yx: np.ndarray) -> np.ndarray:
    """Verifier side: the complex image of a noise-free scan of ``realised`` (no rotation) at patient points [n, 2]
    (y, x) in mm: the Fourier sum of the acquired k-space evaluated there. At a pixel centre it is the pixel of the
    console reconstruction (a test compares the two)."""
    ks = engine.clean_kspace(realised)[0]
    k_pe, k_ro = ks.shape
    fov_pe, fov_ro = realised["fov_pe_mm"] * k_pe / realised["matrix_pe"], console.SYSTEM["ro_os"] * realised["fov_mm"]
    y, x = points_yx[:, 0] - realised["centre_y_mm"], points_yx[:, 1] - realised["centre_x_mm"]
    v, u = (y, x) if realised["pe_dir"] == "y" else (x, y)
    ev = np.exp(2j * np.pi * np.outer(v, np.arange(k_pe) - k_pe // 2) / fov_pe)
    eu = np.exp(2j * np.pi * np.outer(u, np.arange(k_ro) - k_ro // 2) / fov_ro)
    return np.einsum("ja,ab,jb->j", ev, ks, eu) / (fov_pe * fov_ro)


def separation(engine, realised: dict, a_yx, b_yx, samples: int = PROFILE_SAMPLES) -> float:
    """Verifier side: the dip between the two features at ``a_yx`` and ``b_yx`` in a noise-free scan of ``realised``
    (see the module docstring)."""
    a, b = np.asarray(a_yx, float), np.asarray(b_yx, float)
    t = np.linspace(-0.5, 1.5, samples)
    return _dip(np.abs(image_at(engine, realised, a[None, :] + t[:, None] * (b - a)[None, :])), samples)


def reference_snr(engine, realised: dict, region: dict) -> float:
    """Verifier side: |mean of the noise-free image over the region| / standard deviation of the noise in one pixel
    (one component, ``Engine.noise_sd``). 0 for a region that holds no pixel centre of the image."""
    row = F.noise_free(engine, realised, [region])[1][region["id"]]
    sd = engine.noise_sd(realised)
    if not row["n_pixels"] or sd <= 0:                                   # no pixel centre of the image lies in the region
        return 0.0
    return float(row["complex_mean_magnitude"][0] / sd)


def pair_checks(cfg: dict, pair: str, geometry: dict, world: dict, engine, outline: np.ndarray | None = None, first_failure: bool = False) -> dict:
    """Verifier side: requirements 1 to 4 for one pair, in their declared order, with the numbers behind them.
    ``outline``: ``nowrap.task.outline(world)``. ``first_failure``: stop at the first requirement that fails."""
    t = cfg["task"]
    pl = console.plan(protocol_of(geometry), world["console"])
    rp, grid_mm = pl["realised"], world["extent_mm"] / world["n"]
    outline = N.outline(world) if outline is None else outline
    over = N.overhang_mm(outline, rp, grid_mm)
    out = dict(scan_s=pl["scan_s"], lines=rp["lines"], pixel_mm=round(rp["fov_mm"] / rp["matrix"], 4), overhang_mm=round(over, 3),
               dip=None, snr=None)
    checks = dict(whole_object_inside=over <= COVER_SLACK_MM)
    if not first_failure or checks["whole_object_inside"]:
        h = cfg["hidden"]["pairs"][pair]
        out["dip"] = separation(engine, rp, h["a_yx_mm"], h["b_yx_mm"])
        checks["features_separated"] = out["dip"] >= t["min_dip"]
    if not first_failure or all(checks.values()):
        ref = next(r for r in t["regions"] if r["id"] == t["reference_region"])
        out["snr"] = reference_snr(engine, rp, ref)
        checks["snr_floor"] = out["snr"] >= t["min_snr"]
    if not first_failure or all(checks.values()):
        checks["within_time_cap"] = pl["scan_s"] <= t["max_scan_s"] + 1e-9
    return dict(out, checks=checks)


_OPTIMA: collections.OrderedDict = collections.OrderedDict()


def optimum(cfg: dict, world: dict, engine) -> dict:
    """Verifier side: per pair, the fastest geometry that meets requirements 1 to 4 on this world, from the oracle. Kept
    for the last instances scored in this process (the optimum of an instance is a function of its seed and level)."""
    from . import oracle
    key = (cfg["seed"], json.dumps(cfg["world_spec"], sort_keys=True))
    if key not in _OPTIMA:
        if len(_OPTIMA) >= 32:
            _OPTIMA.popitem(last=False)
        _OPTIMA[key] = oracle.solve(cfg, world, engine)["pairs"]
    return _OPTIMA[key]


DIAGNOSTICS = dict(whole_object_inside="part of the object lies outside the image", features_separated="the two features are not separated",
                   snr_floor="the signal-to-noise ratio of the reference region is too low", within_time_cap="the scan is longer than the examination affords",
                   scan_time_within_factor="the scan is slower than allowed")


def score(cfg: dict, answer, world: dict | None = None, engine=None) -> dict:
    from ...scanner import kspace
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = engine or kspace.Engine(world)
    problem = check(cfg, answer, world["console"])
    if problem:
        return F.result(dict(deliverable_valid=False), problem, error=None)
    outline, best, ids = N.outline(world), optimum(cfg, world, engine), list(cfg["task"]["pairs"])
    rows = {i: pair_checks(cfg, i, answer[i], world, engine, outline) for i in ids}
    for i in ids:
        rows[i]["checks"]["scan_time_within_factor"] = rows[i]["scan_s"] <= TOL * best[i]["scan_s"] + 1e-9
        rows[i]["excess"] = rows[i]["scan_s"] / best[i]["scan_s"] - 1.0
    checks = dict(deliverable_valid=True, **{f"{name}_{i}": rows[i]["checks"][name] for name in REQUIREMENTS for i in ids})   # requirement by requirement
    first = {i: next((name for name in REQUIREMENTS if not rows[i]["checks"][name]), None) for i in ids}        # per pair: the first that fails
    bad = [f"pair {i}: {DIAGNOSTICS[name]}" for i, name in first.items() if name]
    excess = max(r["excess"] for r in rows.values())
    return F.result(checks, "; ".join(bad), error=float(excess), tolerance=TOL - 1.0,
                    pairs={i: dict(scan_s=r["scan_s"], scan_s_fastest=best[i]["scan_s"], lines=r["lines"], lines_fastest=best[i]["lines"],
                                   pixel_mm=r["pixel_mm"], pixel_mm_fastest=best[i]["pixel_mm"], dip=r["dip"], snr=r["snr"],
                                   overhang_mm=r["overhang_mm"]) for i, r in rows.items()},
                    distance=dict(excess_scan_time=float(excess), smallest_dip=min(r["dip"] for r in rows.values()),
                                  largest_overhang_mm=max(r["overhang_mm"] for r in rows.values())))


def outcome(score: dict) -> str:
    """The outcome label of a delivered, measured answer that fails a check: the first requirement that fails for a pair."""
    labels = dict(zip(REQUIREMENTS, OUTCOMES))
    return next((labels[k.rsplit("_", 1)[0]] for k, ok in score["checks"].items() if not ok and k.rsplit("_", 1)[0] in labels), OUTCOMES[-1])


def card() -> dict:
    from ... import gate, learn
    from . import controls
    rows = [(controls.REFERENCE, "pass")] + [(n, "fail") for n in controls.MUST_FAIL] + [(n, "ablation") for n in controls.ABLATIONS] \
        + [(n, "reported") for n in controls.REPORTED]
    kind = F.kind_by_open_loop(gate.load(ENV_ID))
    return dict(
        id=ENV_ID, title="The fastest scan that separates a detail",
        given=f"A test object with three pairs of small features at hidden spacings; the scanner; {BUDGET_S:g} s of scan time for looking",
        deliver="Per pair, the geometry of a spin-echo scan: phase-encode axis, field of view, phase field of view, matrix",
        scored_by="Re-scan on the hidden object: whole object inside, the pair separated (dip of at least one half between the "
                  f"two peaks), an SNR floor, a scan-time cap, scan time within {round(100 * (TOL - 1))} % of the fastest such geometry",
        summary="The pixel follows from the detail. Three pairs of small bright features sit in signal-free inserts of a gel "
                "phantom; the spacing of each pair is hidden. The agent looks (a localiser costs scan time), measures the "
                "spacings and the size of the object, and prescribes for each pair the spin-echo scan with the coarsest pixel "
                "that still separates it. A pixel as large as the spacing does not: the point-spread function is a sinc.",
        task_kind=kind["kind"], task_kind_why=kind["evidence"], task_kind_levels=kind["levels"], competencies=list(COMPETENCIES),
        world=dict(phantom="QA phantom (an elliptical disc of gel), three signal-free inserts with a pair of bright discs each, 1.5 T",
                   sequences_useful=["spgr", "se"],
                   agent_side_formulas="pixel = fov_mm / matrix along both axes; lines = matrix x phase_fov; scan time = overhead + "
                                       "TR x (1 + lines); the image of a point is sinc(x / pixel): two points merge at a spacing "
                                       "of one pixel and show a dip of one half at 1.63 pixels; pixel i of n is centred at "
                                       "(i - n / 2) x pixel"),
        hidden=["spacing and position of the two features of each pair", "size of the object, and at level 3 its orientation",
                "its materials", "fields and system state", "noise level", "the 128-bit world seed"],
        told=["the target sequence", "the marked regions (a circle around each pair, a reference circle in the gel)",
              "the pass rule: dip, SNR floor, time cap, time factor", "the instrument datasheet",
              "realised protocol, timing, limits and cost of every plan and scan",
              "any stored image, and the numbers along any line through it"],
        levels={lv: dict(summary=d["summary"], budget_s=[d["budget_s"], d["budget_s"]]) for lv, d in LEVELS.items()},
        practice_levels=[lv for lv, d in LEVELS.items() if d.get("kind") == "practice"],
        learnability=learn.stored(ENV_ID),
        limits=dict(max_scanner_requests=32, max_tool_calls=40),
        deliverable={"tool": "submit", "format": {"<pair id>": {"pe_dir": "y | x", "fov_mm": "<mm>", "phase_fov": "<fraction>", "matrix": "<even integer>"}}},
        reward=dict(range=[0.0, 1.0], when="only at submit; 0 on every other step",
                    gates=["the scanner log replays and is within budget", "at least one accepted scan", "the console accepts each geometry",
                           f"the whole object lies inside the image (hidden outline, {COVER_SLACK_MM:g} mm slack)",
                           f"dip between the two features >= {MIN_DIP:g} on the noise-free image read between its pixels",
                           f"SNR of the reference region >= {MIN_SNR:g} (noise-free signal over the analytic noise)",
                           f"scan time <= {MAX_SCAN_S:g} s",
                           f"scan time <= {TOL:g} x the scan time of the fastest geometry that passes the gates above (search on the hidden world)"],
                    error="largest over the pairs of T / T_opt - 1: scan time of the delivered geometry over the oracle's optimum",
                    score="none: the reward is 1 for a pass and 0 otherwise",
                    passing="all gates, for every pair"),
        controls=[dict(name=n, must=m, what=controls.DESCRIPTIONS[n]) for n, m in rows],
        determinism="same (seed, level, tool calls) -> same observations, log hash chain and reward; a repeated scan draws new noise; "
                    "the verifier's re-scan is noise-free",
    )
