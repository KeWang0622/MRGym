"""Instance generator of the clinic: a hidden object, a complaint protocol, marked findings with known causes, a
verified remedy, and the requirements a corrected protocol must meet.

Two ways to an instance. Levels 1 and 2 search a drawn object for sites (``build``, described below). Level 3
draws the causes first and builds the object for them (``build_constructed``, described above that function).

One searched instance (levels in ``task.LEVELS``):

    object      QA phantom: a gel body with a ring of vials, two of them oil; the materials are hidden
    protocol    one complaint protocol for the whole examination, drawn from one distribution per level. It arms
                wrap-around and truncation always and one of the two fat mechanisms; which marked finding carries
                which cause is decided by where the finding lies on the hidden object
    findings    circles of one radius. Candidate sites are laid along the border of every vial (inside, across and
                outside); every site is classified by the twins of ``metrics``: the cause set of a site is the set of
                mechanisms that alone make E >= ``present``; sites with any E between ``absent`` and ``present``
                are never used. A wanted cause set is drawn per finding and a site with exactly that set is picked.
                A benign finding (no cause) is a site where every E is below ``absent`` and the object itself has an
                edge or another material (the dip is in the object)
    remedy      the catalogue's remedy for the union of the causes, time kept by lowering the averages; verified on
                the twins with a margin (``VERIFY``) on every finding
    position    wrap-around can only occur in the strip into which the object folds, and where that strip ends
                depends on the hidden extent of the object. Findings of every cause are drawn close to that border
                (``fold_depth``), so that position and sheet together name the folded findings less well (they
                still do, see the position row of ``tools/build_clinic_bank.py``: on a searched object every cause
                has its own kind of place). The noise level of a world is drawn over a factor of five, so the SNR
                requirement cannot be compared with the sheet
    well-posed  a finding is served only where the catalogue's differential experiments (``experiments``) are
                diagnostic: taking a mechanism away by its remedy changes the finding clearly when it carries the
                cause and hardly when it does not (``experiments_agree``)
    budget      ``budget_factor`` x the scan time of the reference's experiments (``experiments``), and always less
                than one run of the complaint protocol
    requirements  scan time <= that of the complaint protocol x U(1.0, 1.12); pixel <= that of the complaint
                protocol; SNR in the reference region >= that of the verified remedy x U(slack); the weighting and
                the readout / phase directions locked; TE inside one fat period of the complaint protocol

A world on which a wanted cause has no site, or whose remedy is not clean with the margin, is rejected and the next
attempt (``family.reseed``) is drawn: generation is deterministic in (seed, level).
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np

from ...scanner import console, kspace, rf, tissue
from ...scanner import signal as S
from ...scanner import world as W
from . import catalogue as C
from . import metrics as M

FOV_MM = 240.0
REF = dict(id="REF", y_mm=0.0, x_mm=0.0, radius_mm=8.0)
REF_CENTRES = ((0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (0.0, -10.0), (-10.0, 0.0))   # the reference region is the first clean one of these
LOCKED = ("sequence", "tr_ms", "flip_deg", "slice_mm", "fov_mm", "pe_dir", "ro_polarity", "fat_sat", "centre_y_mm", "centre_x_mm")
WINDOWS = dict(slice_mm=(None, 1.0))   # where a level unlocks a parameter: the window, as fractions of the complaint value
CLEAN_MECH = min(c.clean for c in C.CATALOGUE.values() if c.mechanism != "truncation")
CLEAN_TRUNC = C.BY_MECHANISM["truncation"].clean
VERIFY = (0.58, 0.85)            # the reference remedy must be clean at these fractions of the two thresholds (twin, object): both errors < 0.03
RADIUS_PX, RADIUS_MIN_MM = 1.3, 3.0
SITE_ANGLES = 20
SITE_OFFSETS_PX = (-3.0, -2.2, -1.6, -1.1, -0.6, 0.0, 0.6, 1.3, 2.0, 3.0)
NOISE_SPREAD = 0.8               # the noise level of a world is drawn log-uniformly within exp(+-this) of the level value
LOOK = dict(edge=0.30, smooth=0.22, bright=1.20, dark=0.88)   # what the generator requires of a served finding (``look``)
LOOK_SPLIT = dict(texture=0.26, level=1.0)                   # where an experimenter draws the lines
PHASE_FOV = (0.46, 0.66)          # phase field of view of a complaint protocol: the object reaches beyond it on most worlds
GRID_STEP_MM = 11.0              # extra sites on a grid over the whole object
FOLD_PREFERENCE_MM = (8.0, 15.0) # findings are drawn near the hidden border of the folded strip: wrap-around just inside, the others just outside
BENIGN_CONTRAST = 0.15
CHAIN_PRESENT, CHAIN_ABSENT = 0.085, 0.035   # the differential experiments must agree with the twins (see ``experiment_changes``)
DARK_VIAL = 0.36                 # a ripple reaches ``present`` only inside a vial darker than this fraction of the gel
PICK_TRIES = 40
PROBE_TR_FRACTION, PROBE_TR_MIN_MS = 0.25, 20.0
WFS_ARMED_MIN_PX = 0.4            # below this water-fat shift the displacement cannot reach ``present``
BUDGET_MAX = 0.9                 # the budget stays below this fraction of the scan time of the complaint protocol
# ---- constructed levels (``build_constructed``): every finding lies on a site built for its cause
FEWER_AVERAGES = 3               # the echo and slice experiments of an adaptive level run at one in this many of the averages
DISC_MM = 12.0                   # radius of a constructed disc (vial, tube, patch of field gradient); the finding is its flat core
SOFT_EDGE_MM = 2.5               # half-width of every boundary of a constructed object: no ringing inside a disc at these pixels
REF_RADIUS_MM = 6.0
COLUMN_GAP_MM = 2 * DISC_MM + 6.0   # every finding has a stripe of its own along the phase axis for whatever folds or moves onto it
COLUMN_REACH = 0.5               # the stripes lie within this fraction of the object's half-length along the readout
BODY_RO_MM = (93.0, 100.0)       # half-length of the object along the readout (the ends of the readout axis stay empty: noise is read there)
BODY_PE_MARGIN_MM = (5.0, 8.0)   # the object ends this far inside the phase field of view: only what lies beside it folds
PE_INNER_MM = 14.0               # findings keep this distance from the centre line along the phase axis (one rule for every cause)
TUBE_REACH_MM = 111.0            # a tube beside the object lies within this distance of the isocentre (inside the modelled bore)
DARK_LEVEL = (0.25, 0.72)        # level of a constructed dark finding in the complaint image, whatever its cause
VERIFY_BUILT = (0.9, 0.85)       # the remedy of a constructed level is clean at these fractions of the thresholds: its doses are
#                                  measured, and a dose much stronger than needed must cost the examination its SNR (``snr_share``)
GHOST_EDGE_MM = 0.5              # the averages of the ghost remedy must not change when the clearance is moved by this much
BRIGHT_LEVEL = (1.6, 2.3)        # ... of a bright one
EMULSION_IN_PHASE = (0.95, 1.2)  # level of a water-fat mixture at an in-phase echo
GHOST_DEPTH = 0.85               # modulation depth of a pulsing vial
GHOST_ALIGN = 0.95               # the copy adds to the gel almost in phase: |1 + g exp(i phi)| = this share of 1 + g
GHOST_KEYS = 256                 # phase keys tried for a copy of the wanted level
GHOST_OWN_MM = DISC_MM + 2.0 * SOFT_EDGE_MM + 0.5   # the copy that lay on a finding ends within this distance of its centre
GHOST_MISFIT = 0.6               # the four-copy model explains the change when its mean absolute misfit is below this share of the change
GHOST_ALIAS_MM = 26.0            # the second copy of a pulsing vial, folded, stays this far from the finding
GHOST_MARGIN_MM = 3.0
SCAN_FLOOR_S = console.SYSTEM["overhead_s"] + 1e-3 * console.SYSTEM["dummy_ms"]   # what any scan costs before its first line
LOOK_CAUSES = dict(dark=("susceptibility_signal_loss", "opposed_phase_cancellation"), bright=("wrap_around", "periodic_motion_ghost"))


class Reject(Exception):
    """This world cannot host the drawn instance; the generator moves to the next attempt."""


def world_spec(field_t: float, sigma0: float, grid_n: int, extra: dict | None = None) -> dict:
    return dict(kind="qa", n=int(grid_n), b0_t=float(field_t), b0_hz=5.0, cf_offset_hz=2.0, sigma0=float(sigma0),
                qa=dict(n_oil=2, n_ring=(5, 6), p_centre=0.0, vial_t1_ms=(200.0, 5000.0), vial_pd=(0.12, 1.0),
                        vial_t2_ms=(30.0, 200.0), body_radius_mm=(74.0, 98.0), ring_radius_mm=(44.0, 52.0),
                        vial_radius_mm=(14.0, 18.0)), **(extra or {}))


def _plan(world):
    return lambda p: console.plan(p, world["console"])


def adaptive(task: dict) -> bool:
    """Levels whose findings look alike run every experiment at the timing of the complaint protocol (each costs
    about one examination) and are diagnosed from the first look: the card says so (``diagnosis``)."""
    return task.get("diagnosis") == "by_look"


# ------------------------------------------------------------------------------------------------ protocols
def fit_echo(p, te0, echo, window, field_t, plan, earlier=False):
    """TE of protocol ``p``: the complaint TE when it is realisable and of the wanted class, else the nearest
    realisable echo time of that class (in phase: k fat periods; opposed: the complaint TE plus whole periods).
    ``earlier``: of the in-phase echoes the last one before the complaint TE, when one is realisable (less
    dephasing, more signal)."""
    te_min = plan({k: v for k, v in p.items() if k != "te_ms"})
    if not te_min.get("timing"):
        return None
    te_min, period = te_min["timing"]["te_min_ms"], C.fat_period_ms(field_t)
    if C.echo_class(te0, field_t) == echo:
        cand = [te0 + n * period for n in range(0, 12)]
    else:
        cand = sorted((k * period for k in range(1, 40)), key=lambda t: (earlier and t > te0, abs(t - te0)))
    lo, hi = window if window else (0.0, 1e9)
    ok = [t for t in cand if t >= te_min + 0.005 and lo <= round(t, 2) <= hi]
    return round(ok[0], 2) if ok else None


def assemble(task: dict, causes, plan, doses: dict | None = None):
    """The catalogue's remedy for a set of causes on the complaint protocol of ``task``, fitted to its requirements:
    bandwidth under the gradient limit, a realisable TE of the right class, averages lowered until the scan time
    fits. ``plan``: the console's dry run (protocol -> plan dict). ``doses``: the doses of the remedies that have
    one. Measured (``measured_doses``): ``slice_ratio`` (the slice follows at the echo time of the remedy) and
    ``ghosts`` (the averages follow from the tracked copies; ``ghost_clear_mm`` overrides the clearance). Or stated
    by the caller: ``slice_mm``, ``averages``. Without them the catalogue's textbook dose. Returns a protocol or None
    when nothing fits."""
    p0, req, field_t = task["current_protocol"], task["requirements"], task["exam"]["field_t"]
    doses = doses or {}
    p, ctx = dict(p0), dict(doses, field_t=field_t, echo=C.echo_class(p0["te_ms"], field_t))
    locked = req.get("locked", {})
    for cause in C.CATALOGUE.values():
        if cause.label in causes:
            allowed = [fn for knob, fn in cause.remedy if knob not in locked]
            if not allowed:
                return None
            allowed[0](p, ctx)
    if "bandwidth_hz_per_px" not in locked:
        p["bandwidth_hz_per_px"] = float(min(p["bandwidth_hz_per_px"], C.bandwidth_cap(p["matrix"], p["fov_mm"])))
    p["te_ms"] = fit_echo(p, p0["te_ms"], ctx["echo"], req.get("te_ms"), field_t, plan, earlier=adaptive(task))
    if p["te_ms"] is None:
        return None
    measured = {lab for lab in causes if lab in C.CATALOGUE}
    lossy = C.BY_MECHANISM.get("dropout") and C.BY_MECHANISM["dropout"].label in measured and "slice_mm" not in locked
    if lossy and doses.get("slice_ratio"):
        tbw = (plan(p0).get("realised") or {}).get("rf_tbw")              # the measured dose, at the echo time of the remedy
        p["slice_mm"] = C.slice_dose(p0["slice_mm"], doses["slice_ratio"], p["te_ms"] / p0["te_ms"], tbw)
    elif lossy and doses.get("slice_mm"):                                 # a dose stated by the caller (a control with a fixed dose)
        p["slice_mm"] = float(doses["slice_mm"])
    pl = plan(p)
    if not pl["accept"]:
        return None
    acq = pl["scan_s_parts"]["acquisition_s"]
    n = int(math.floor((req["max_scan_s"] - (pl["scan_s"] - acq)) / (acq / p["averages"]) + 1e-9))
    if n < 1:
        return None
    moving = C.BY_MECHANISM.get("motion") and C.BY_MECHANISM["motion"].label in measured and "averages" not in locked
    if moving and doses.get("ghosts"):
        axis = "x_mm" if p0["pe_dir"] == "x" else "y_mm"                    # the measured dose: no copy on any finding
        best = C.ghost_averages(doses["ghosts"], [f[axis] for f in task["findings"]], pl["realised"]["fov_pe_mm"]
                                * pl["realised"]["lines_full"] / pl["realised"]["matrix_pe"], n, doses.get("ghost_clear_mm", C.GHOST_CLEAR_MM))
        if best is None:
            return None
        p["averages"] = int(best)
        return p
    if moving and doses.get("averages"):                                  # a dose stated by the caller
        p["averages"] = int(doses["averages"])
    p["averages"] = int(min(p["averages"], n))
    return p


def armed_on_sheet(task: dict) -> dict:
    """What the complaint protocol can produce, read from the sheet alone: {mechanism: bool}."""
    if "_armed" in task:                                                  # stated by the caller (no sheet at hand)
        return dict(dict(wrap=False, chemshift=False, fatphase=False, truncation=True), **task["_armed"])
    p0, field_t = task["current_protocol"], task["exam"]["field_t"]
    out = dict(wrap=p0["phase_fov"] < 1.0, chemshift=C.fat_hz(field_t) / p0["bandwidth_hz_per_px"] >= WFS_ARMED_MIN_PX,
               fatphase=C.echo_class(p0["te_ms"], field_t) == "opposed", truncation=True)
    return dict(out, **{m: True for m in C.mechanisms(task["causes_vocabulary"]) if m not in out})   # what only the world can arm


def experiments(task: dict, plan, every: bool = False) -> list:
    """The differential experiments on a complaint protocol, as a function of the card alone:
    [{"name", "protocol", "baseline", "shape_only", "scale", "relative", "scan_s"}]. An experiment is named by the
    mechanism it takes away and is compared with its ``baseline`` scan: the change inside a finding is what that
    mechanism does. Levels diagnosed from the first look (``adaptive``): ``_single_changes``. The other levels:

        complaint    the complaint protocol itself
        wrap         + phase oversampling up to the full field of view
        chemshift    + the bandwidth of the catalogue's remedy          (only when the sheet arms it, or ``every``)
        fatphase     + the nearest in-phase echo time, at that bandwidth (only when the sheet arms it, or ``every``)
        truncation   twice the matrix (the finer image, averaged over each coarse pixel, stands for the object)

    all at a quarter of the repetition time with the flip angle lowered to keep the T1 weighting (1 - cos(flip)
    proportional to TR) and one average; each experiment adds to the one before it, which is its baseline."""
    p0, field_t = task["current_protocol"], task["exam"]["field_t"]
    armed = armed_on_sheet(task)
    if adaptive(task):
        return _single_changes(task, plan, armed)
    steps = [("complaint", dict(p0), None, False)]
    p = dict(p0, phase_oversampling=round(min(1.0, 1.0 / p0["phase_fov"] - 1.0), 2))
    steps.append(("wrap", dict(p), "complaint", False))
    last = "wrap"
    if armed["chemshift"] or every:
        C.BY_MECHANISM["chemshift"].remedy[0][1](p, dict(field_t=field_t))
        p["bandwidth_hz_per_px"] = float(min(p["bandwidth_hz_per_px"], C.bandwidth_cap(p["matrix"], p["fov_mm"])))
        steps.append(("chemshift", dict(p), last, False))
        last = "chemshift"
    if armed["fatphase"] or every:
        te = fit_echo(p, p0["te_ms"], "in", None, field_t, plan)
        if te is not None and abs(te - p["te_ms"]) > 1e-9:
            p["te_ms"] = te
            steps.append(("fatphase", dict(p), last, True))
            last = "fatphase"
    fine = dict(p, matrix=2 * p["matrix"])
    fine["bandwidth_hz_per_px"] = float(min(fine["bandwidth_hz_per_px"], C.bandwidth_cap(fine["matrix"], fine["fov_mm"])))
    te = fit_echo(fine, p["te_ms"], C.echo_class(p["te_ms"], field_t), None, field_t, plan)
    fine["te_ms"] = te if te is not None else p["te_ms"]
    steps.append(("truncation", fine, last, abs(fine["te_ms"] - p["te_ms"]) > 1e-9))
    tr = max(round(p0["tr_ms"] * PROBE_TR_FRACTION), PROBE_TR_MIN_MS)
    for _, q, *_ in steps:                                                # one repetition time for all of them
        pl = plan(dict(q, tr_ms=tr, averages=1))
        if pl.get("timing"):
            tr = max(tr, math.ceil(pl["timing"]["tr_min_ms"] + 0.5))
    flip = math.degrees(math.acos(1.0 - (1.0 - math.cos(math.radians(p0["flip_deg"]))) * min(1.0, tr / p0["tr_ms"])))
    common = dict(tr_ms=float(tr), flip_deg=round(flip, 1), averages=1)
    out = []
    for name, q, baseline, shape_only in steps:
        q = dict(q, **common)
        pl = plan(q)
        out.append(dict(name=name, protocol=q, baseline=baseline, shape_only=shape_only, scale=1.0, relative=False,
                        scan_s=pl["scan_s"] if pl["accept"] else None))
    return out


def _single_changes(task: dict, plan, armed: dict) -> list:
    """The experiments of a level whose findings are diagnosed from the first look. A ghost copy lies where the
    repetition time and the averages put it, so the first look is the complaint protocol itself, and every experiment
    is one change against it at the repetition time of the complaint:

        wrap         phase oversampling up to the full field of view, one average (a folded copy leaves; a ghost copy
                     moves with the encoded field of view and the averages)
        motion       one average: both copies of a pulsing structure move next to it, one on either side; nothing
                     else changes (``ghost_track`` reads the displacement from this scan)
        chemshift    the bandwidth of the catalogue's remedy            (only when the sheet arms it)
        fatphase     the last in-phase echo time before the complaint's (only when the sheet arms it); compared
                     relative to the reference region (``relative``), so that the decay of the gel between the two
                     echo times is not a change
        dropout      half the slice (compared per mm of slice: ``scale``)

    The experiments that do not depend on the timing run at one in ``FEWER_AVERAGES`` of the averages. Each scan
    costs at least the console's overhead and dummy cycles, which is most of its cost at these short repetition times."""
    p0, field_t = task["current_protocol"], task["exam"]["field_t"]
    few = max(1, p0["averages"] // FEWER_AVERAGES)
    steps = [dict(name="complaint", protocol=dict(p0), baseline=None)]
    if armed["wrap"]:
        steps.append(dict(name="wrap", protocol=dict(p0, phase_oversampling=round(min(1.0, 1.0 / p0["phase_fov"] - 1.0), 2), averages=1)))
    if armed.get("motion"):
        steps.append(dict(name="motion", protocol=dict(p0, averages=1)))
    if armed["chemshift"]:
        q = dict(p0, averages=few)
        C.BY_MECHANISM["chemshift"].remedy[0][1](q, dict(field_t=field_t))
        q["bandwidth_hz_per_px"] = float(min(q["bandwidth_hz_per_px"], C.bandwidth_cap(q["matrix"], q["fov_mm"])))
        steps.append(dict(name="chemshift", protocol=q))
    if armed["fatphase"]:
        te = fit_echo(dict(p0, averages=few), p0["te_ms"], "in", None, field_t, plan, earlier=True)
        if te is not None and abs(te - p0["te_ms"]) > 1e-9:
            steps.append(dict(name="fatphase", protocol=dict(p0, te_ms=te, averages=few), relative=True))
    if armed.get("dropout"):
        steps.append(dict(name="dropout", protocol=dict(p0, slice_mm=0.5 * p0["slice_mm"], averages=few), scale=0.5))
    out = []
    for e in steps:
        pl = plan(e["protocol"])
        out.append(dict(dict(baseline="complaint", shape_only=False, scale=1.0, relative=False), **e, scan_s=pl["scan_s"] if pl["accept"] else None))
    return out


def exam_card(spec: dict) -> dict:
    """The console card of a world spec (what ``world["console"]`` will hold), without building the world."""
    n, extent = W.GRID[spec["kind"]]
    return dict(fov_mm=W.EXAM_FOV_MM[spec["kind"]], min_pixel_mm=round(2.0 * extent / int(spec.get("n") or n), 3),
                b1rms_limit_ut=W.B1RMS_LIMIT_UT[spec["b0_t"]], b0_t=spec["b0_t"])


def draw_protocol(rng, exam, armed, family: dict, vocabulary, by_look: bool = False) -> dict:
    """The complaint protocol: one distribution for every cause; ``armed`` = the mechanisms it lets through. Every
    experiment of ``experiments`` must be realisable on it."""
    field_t, plan = exam["b0_t"], lambda p: console.plan(p, exam)
    for _ in range(80):
        p = dict(sequence="spgr", fov_mm=FOV_MM, matrix=int(rng.choice(family["matrix"])),
                 phase_fov=float(round(rng.uniform(*family.get("phase_fov", PHASE_FOV)), 2)),
                 phase_oversampling=0.0, partial_fourier=1.0, slice_mm=float(family.get("slice_mm", 5.0)), pe_dir=str(rng.choice(["y", "x"])),
                 ro_polarity=int(rng.choice([1, -1])), tr_ms=float(round(rng.uniform(*family["tr_ms"]))),
                 flip_deg=float(round(rng.uniform(*family["flip_deg"]))), averages=int(rng.choice(family["averages"])), fat_sat=False,
                 centre_y_mm=0.0, centre_x_mm=0.0)
        ctx = dict(field_t=field_t, echo="in")
        for cause in C.CATALOGUE.values():
            (cause.arm if cause.mechanism in armed else cause.disarm)(p, ctx, rng)
        period = C.fat_period_ms(field_t)
        k = max(0, math.ceil(family.get("te_min_ms", 0.0) / period - (0.5 if ctx["echo"] == "opposed" else 1.0)))
        te0 = (k + (0.5 if ctx["echo"] == "opposed" else 1.0)) * period * float(rng.uniform(0.98, 1.02))
        p["te_ms"] = fit_echo(p, te0, ctx["echo"], None, field_t, plan)
        pl = plan(p)
        if p["te_ms"] is not None and pl["accept"] and family["scan_s"][0] <= pl["scan_s"] <= family["scan_s"][1]:
            task = dict(exam=dict(field_t=field_t), current_protocol=p, causes_vocabulary=list(vocabulary),
                        diagnosis="by_look" if by_look else "chain")
            if all(e["scan_s"] is not None for e in experiments(task, plan)):   # (the vocabulary only says whether the level is adaptive)
                return p
    raise Reject("no complaint protocol")


# ------------------------------------------------------------------------------------------------ findings
def _border_mm(world, cls, cy, cx, ang):
    n, d = world["n"], world["extent_mm"] / world["n"]
    r = np.arange(4.0, 40.0, 0.2)
    iy = np.round((cy + r * np.sin(ang)) / d).astype(int) + n // 2
    ix = np.round((cx + r * np.cos(ang)) / d).astype(int) + n // 2
    out = np.flatnonzero(world["weights"][cls][iy, ix] < 0.5)
    return float(r[out[0]]) if len(out) else None


def sites(world, rp, radius_mm, rng) -> list:
    """Candidate findings: circles along the border of every vial, from well inside to well outside, and on a grid
    over the object, that lie inside the image and inside the object."""
    px = rp["fov_mm"] / rp["matrix"]
    fy, fx = (rp["fov_pe_mm"], rp["fov_mm"]) if rp["pe_dir"] == "y" else (rp["fov_mm"], rp["fov_pe_mm"])
    n, d, body = world["n"], world["extent_mm"] / world["n"], world["masks"]["body"]
    out = []
    for reg in world["regions"]:
        cls = world["names"].index(reg["cls"])
        for ang in float(rng.uniform(0, 2 * np.pi)) + np.arange(SITE_ANGLES) * 2 * np.pi / SITE_ANGLES:
            rb = _border_mm(world, cls, reg["y_mm"], reg["x_mm"], ang)
            if rb is None:
                continue
            for off in SITE_OFFSETS_PX:
                rr = rb + off * px
                y, x = reg["y_mm"] + rr * math.sin(ang), reg["x_mm"] + rr * math.cos(ang)
                edge = radius_mm + px
                if abs(y) + edge > fy / 2 or abs(x) + edge > fx / 2:
                    continue
                far = rr + radius_mm + px                                 # the far side of the circle must be in the object
                iy = int(round((reg["y_mm"] + far * math.sin(ang)) / d)) + n // 2
                ix = int(round((reg["x_mm"] + far * math.cos(ang)) / d)) + n // 2
                if not (0 <= iy < n and 0 <= ix < n and body[iy, ix]):
                    continue
                out.append(dict(y_mm=round(y, 1), x_mm=round(x, 1), radius_mm=radius_mm))
    jy, jx = (float(v) for v in rng.uniform(0, GRID_STEP_MM, 2))            # a grid over the whole object: mostly plain gel
    for y in np.arange(-fy / 2 + radius_mm + px + jy, fy / 2 - radius_mm - px, GRID_STEP_MM):
        for x in np.arange(-fx / 2 + radius_mm + px + jx, fx / 2 - radius_mm - px, GRID_STEP_MM):
            iy, ix = int(round(y / d)) + n // 2, int(round(x / d)) + n // 2
            if 0 <= iy < n and 0 <= ix < n and body[iy, ix]:
                out.append(dict(y_mm=round(float(y), 1), x_mm=round(float(x), 1), radius_mm=radius_mm))
    return out


def fold_depth(world, rp, site) -> float:
    """How far (mm) a site lies inside the strip into which the object folds along the phase-encode axis (negative:
    outside it). Wrap-around can only occur inside the strip, and where the strip ends depends on the hidden extent
    of the object; findings of every cause are therefore drawn close to that border, so that the position of a
    finding together with the sheet does not say whether it is folded."""
    axis = "x" if rp["pe_dir"] == "x" else "y"
    lo, hi = W.extent_mm(world)[axis]
    pe = site[axis + "_mm"]
    return (pe - (rp["fov_pe_mm"] + lo)) if pe > 0 else ((hi - rp["fov_pe_mm"]) - pe)


def look(image, rp, finding, reference) -> dict:
    """What the complaint image shows of a finding: its mean relative to the reference region (``level``) and its
    relative spread (``texture``), and the class an experimenter reads from them: "edge" (a border runs through
    it), else "bright" or "dark" against the gel. The class says which experiments can explain the finding."""
    m, r = M.area_mask(rp, finding), M.area_mask(rp, reference)
    mean = float(np.sum(m * image) / np.sum(m))
    level = mean / max(float(np.sum(r * image) / np.sum(r)), 1e-30)
    texture = float(np.sqrt(np.sum(m * (image - mean) ** 2) / np.sum(m))) / max(mean, 1e-30)
    cls = "edge" if texture >= LOOK_SPLIT["texture"] else "bright" if level >= LOOK_SPLIT["level"] else "dark"
    return dict(cls=cls, level=level, texture=texture)


def look_is_clear(lk: dict) -> bool:
    """Served findings are well inside their class: an experimenter with noise reads the same class."""
    if lk["cls"] == "edge":
        return lk["texture"] >= LOOK["edge"]
    return lk["texture"] <= LOOK["smooth"] and (lk["level"] >= LOOK["bright"] if lk["cls"] == "bright" else lk["level"] <= LOOK["dark"])


def class_plan(task: dict) -> dict:
    """Which experiments explain a finding of each look class: {class: (first, second or None, run the second when
    the first changed the finding (True) or when it did not (False))}.

        edge      the experiment of the fat mechanism the sheet arms (other bandwidth, or other echo time)
        bright    phase oversampling; where the motion ghost is served and oversampling changed the finding, other
                  averages tell a ghost from a folded copy
        dark      half the slice: signal loss; where the slice did not change the finding and the sheet arms the
                  cancellation, the in-phase echo tells a mixture of water and fat from a material that is dark"""
    armed = armed_on_sheet(task)
    return dict(edge=("chemshift" if armed["chemshift"] else "fatphase", None, True),
                bright=("wrap", "motion" if armed.get("motion") else None, True),
                dark=("dropout", "fatphase" if armed["fatphase"] and not armed["chemshift"] else None, False))


def class_experiments(plan_of_class) -> tuple:
    return tuple(e for e in plan_of_class[:2] if e)


def next_experiments(plans: dict, classes: dict, flags: dict) -> list:
    """The adaptive plan after the first look: for every finding the first experiment of its class, and the second
    one when the first leaves the finding open (``class_plan``). ``classes``: {finding: class}; ``flags``:
    {experiment: {finding: changed}} for the experiments already run. Returns the experiments still to run, in
    catalogue order; empty when every finding is settled."""
    need = set()
    for fid, cls in classes.items():
        first, second, if_changed = plans[cls]
        if first not in flags:
            need.add(first)
        elif second and flags[first][fid] == if_changed and second not in flags:
            need.add(second)
    return [m for m in C.MECHANISMS if m in need]


def settle(plans: dict, classes: dict, flags: dict, labels) -> dict:
    """The causes of every finding from the experiments of its class that were run: {finding: [causes]}."""
    out = {}
    for fid, cls in classes.items():
        first, second, if_changed = plans[cls]
        mine = {first} if first in flags and flags[first][fid] else set()
        if second and first in flags and flags[first][fid] == if_changed and second in flags and flags[second][fid]:
            mine.add(second)                                              # the second experiment counts only where the first calls for it
        out[fid] = C.decode(mine, labels)
    return out


def cause_set(solo: dict):
    """The causes a site carries, or None when any mechanism is between ``absent`` and ``present``."""
    out = []
    for mech, e in solo.items():
        cause = C.BY_MECHANISM[mech]
        if e >= cause.present:
            out.append(cause.label)
        elif e >= cause.absent:
            return None
    return tuple(sorted(out))


def is_clean(err: dict, margin=(1.0, 1.0)) -> bool:
    return err["mech"] <= margin[0] * CLEAN_MECH and err.get("trunc", 0.0) <= margin[1] * CLEAN_TRUNC


def draw_wanted(seed: str, level_cfg: dict) -> dict:
    """What the instance of ``seed`` must contain, drawn before any world. A constructed level:
    ``_draw_look_alikes``. A searched level: the fat mechanism the protocol arms and the cause set of every finding
    (independent between findings; uniform over the level's options, a benign finding with weight
    ``benign_weight``). The attempts of an instance keep this draw, so a world that cannot host a cause does not
    shift the distribution of causes; only after ``task.ATTEMPTS_PER_DRAW`` worlds have failed does ``build`` take
    what a world offers."""
    lv, rng = level_cfg, W.rng_for(seed, "task", 1)
    if lv.get("constructed"):
        return _draw_look_alikes(lv, rng)
    served = [C.CATALOGUE[lab] for lab in lv["causes"]]
    fat = [c for c in served if c.group == "fat"]
    armed = {c.mechanism for c in served if c.group is None}
    armed |= {c.mechanism for c in fat} if lv["arm_all"] else {fat[int(rng.integers(len(fat)))].mechanism}
    options = [(c.label,) for c in served if c.mechanism in armed and c.mechanism not in lv.get("never", ())
               and (c.group is None or not lv["arm_all"] or c.mechanism == "chemshift")]
    if lv["pairs"]:
        options.append(tuple(sorted(c.label for c in fat)))
    if lv["benign"]:
        options.append(())
    weights = np.array([lv.get("benign_weight", 1.0) if not o else 1.0 for o in options])
    wanted = [options[int(rng.choice(len(options), p=weights / weights.sum()))] for _ in range(lv["n_findings"])]
    return dict(armed=sorted(armed), wanted=wanted, look=None, options=options, level_options=options, field_t=float(rng.choice(lv["field_t"])),
                lock_bandwidth=False)


def look_alike_sets(cls: str, n: int) -> list:
    """The cause sets an episode of look class ``cls`` with ``n`` findings can hold, in a fixed order: every finding
    carries one of the two artefact causes of the look or none, and each of the two causes occurs at least once."""
    a, b = ((lab,) for lab in LOOK_CAUSES[cls])
    return [seq for seq in itertools.product((a, b, ()), repeat=n) if a in seq and b in seq]


def _draw_look_alikes(lv: dict, rng) -> dict:
    """The draw of a constructed level: the look class of the episode (uniform), then the causes of its findings,
    uniform over ``look_alike_sets``: each finding has one of the two artefact causes that look like this or is the
    object itself, and both causes occur. Position and sheet are drawn independently of this (``layout``)."""
    served = [lab for lab in lv["causes"] if C.CATALOGUE[lab].mechanism not in lv.get("never", ())]
    classes = sorted(k for k, v in LOOK_CAUSES.items() if all(lab in served for lab in v))
    cls = classes[int(rng.integers(len(classes)))]
    sets = look_alike_sets(cls, lv["n_findings"])
    wanted = list(sets[int(rng.integers(len(sets)))])
    options = [(lab,) for lab in LOOK_CAUSES[cls]] + [()]
    return dict(armed=sorted(C.mechanisms(served)), wanted=wanted, look=cls, options=options, level_options=[(lab,) for lab in served] + [()],
                field_t=float(rng.choice(lv["field_t"])), lock_bandwidth=False)


def infer(classes: dict, flags: dict, labels) -> dict:
    """The most probable causes of every finding of a constructed level from the first look and whatever experiments
    were run: {finding: [causes]}. ``classes``: {finding: look class}; ``flags``: {experiment: {finding: changed}}.
    Among the cause sets the level can hold (``look_alike_sets``) those that agree with every flag of the look's
    own experiments are equally probable; the first one in the fixed order is returned (with every experiment of the
    look run there is exactly one). Findings of mixed looks, or flags that agree with no set (a misread image), are
    settled finding by finding (``settle``)."""
    ids = sorted(classes)
    looks = set(classes.values())
    cls = next(iter(looks)) if len(looks) == 1 else None
    if cls in LOOK_CAUSES and all(lab in labels for lab in LOOK_CAUSES[cls]):
        mine = [e for e in C.mechanisms(LOOK_CAUSES[cls]) if e in flags]
        for seq in look_alike_sets(cls, len(ids)):
            if all(C.expected(c, e) in (None, bool(flags[e][i])) for i, c in zip(ids, seq) for e in mine):
                return {i: list(c) for i, c in zip(ids, seq)}
    plans = class_plan(dict(causes_vocabulary=list(labels), _armed={m: True for m in C.mechanisms(labels)}))
    return settle(plans, classes, flags, labels)


def count_consistent(cls: str, truth: dict, run) -> int:
    """How many cause sets of ``look_alike_sets`` agree with what the experiments ``run`` (names) show on an episode
    whose hidden causes are ``truth`` ({finding: causes}): the open loop's guess is right with probability 1 / this."""
    ids = sorted(truth)
    mine = [e for e in C.mechanisms(LOOK_CAUSES[cls]) if e in run]
    seen = {e: [C.expected(truth[i], e) for i in ids] for e in mine}
    return sum(all(C.expected(c, e) in (None, seen[e][k]) or seen[e][k] is None for k, c in enumerate(seq) for e in mine)
               for seq in look_alike_sets(cls, len(ids)))


def _precheck(world, rp0, wanted, radius, px, tw):
    """Cheap reasons to give up on a world before any twin is scanned."""
    labels = {lab for w in wanted for lab in w}
    if C.BY_MECHANISM["wrap"].label in labels:
        lo, hi = W.extent_mm(world)["x" if rp0["pe_dir"] == "x" else "y"]
        if max(abs(lo), abs(hi)) < rp0["fov_pe_mm"] / 2 + 2 * radius:
            raise Reject("the object does not reach beyond the phase field of view")
    if C.BY_MECHANISM["truncation"].label in labels:
        truth, ref = tw.truth(rp0), M.area_mask(rp0, REF)
        gel = float(np.sum(ref * truth) / np.sum(ref))
        dark = []
        for reg in world["regions"]:
            m = M.area_mask(rp0, dict(reg, radius_mm=6.0))
            dark.append(float(np.sum(m * truth) / np.sum(m)) / gel)
        if min(dark) > DARK_VIAL:
            raise Reject("no vial is dark enough for a visible ripple")


def build(seed: str, level_cfg: dict, draw: dict, fallback: bool = False) -> dict:
    """One attempt at an instance of a searched level on the world of ``seed``. Raises ``Reject`` when this world
    cannot host it. ``fallback``: a wanted cause set this world has no site for is replaced by one it has (drawn
    among the level's options that the world offers), so that the attempt can only fail for the remedy."""
    lv = level_cfg
    rng = W.rng_for(seed, "task")
    field_t, armed, wanted = draw["field_t"], set(draw["armed"]), [tuple(w) for w in draw["wanted"]]
    spec = world_spec(field_t, lv["sigma0"] * float(np.exp(rng.uniform(-NOISE_SPREAD, NOISE_SPREAD))), lv["protocol"]["grid_n"])
    vocabulary = [lab for lab in lv["causes"] if C.CATALOGUE[lab].mechanism not in lv.get("never", ())]
    p0 = draw_protocol(rng, exam_card(spec), armed, lv["protocol"], vocabulary)
    world = W.make_world(seed, spec)
    mechs = C.mechanisms(lv["causes"])
    plan = _plan(world)
    pl0 = plan(p0)
    rp0 = pl0["realised"]
    px = rp0["fov_mm"] / rp0["matrix"]
    radius = round(max(RADIUS_PX * px, RADIUS_MIN_MM), 1)
    tw = M.Twins(world, mechs)
    _precheck(world, rp0, wanted, radius, px, tw)
    task = dict(exam=dict(object="QA phantom", field_t=field_t), current_protocol=p0, causes_vocabulary=vocabulary, diagnosis="chain")
    truth, ref_mask = tw.truth(rp0), M.area_mask(rp0, REF)
    gel = float(np.sum(ref_mask * truth) / np.sum(ref_mask))
    ref = next((r for r in (dict(REF, y_mm=y, x_mm=x) for y, x in REF_CENTRES)
                if all(v < C.BY_MECHANISM[m].absent for m, v in tw.solo(rp0, r).items())
                and M.rel_l1(truth, np.full_like(truth, gel), M.area_mask(rp0, r)) < 0.03), None)
    if ref is None:
        raise Reject("no clean reference region")
    # ---- classify every candidate site on the complaint protocol
    pool = {}
    for s in sites(world, rp0, radius, rng):
        solo = tw.solo(rp0, s)
        cs = cause_set(solo)
        if cs is None:
            continue
        if not cs and M.rel_l1(truth, np.full_like(truth, gel), M.area_mask(rp0, s)) < BENIGN_CONTRAST:
            continue                                                      # nothing to see: not a finding
        pool.setdefault(cs, []).append(dict(s, solo={k: round(v, 4) for k, v in solo.items()}))
    for i, want in enumerate(wanted):
        if want not in pool:
            offered = [tuple(o) for o in draw["options"] if tuple(o) in pool]
            if not fallback or not offered:
                raise Reject(f"no site for {'+'.join(want) or 'a benign finding'}")
            wanted[i] = offered[int(rng.integers(len(offered)))]
    union = sorted({lab for w in wanted for lab in w})
    cap = round(pl0["scan_s"] * float(rng.uniform(*lv["time_slack"])), 1)
    window = [round(max(0.5, p0["te_ms"] - 1.05 * C.fat_period_ms(field_t)), 2), round(p0["te_ms"] + 1.05 * C.fat_period_ms(field_t), 2)]
    chain = experiment_images(task, plan, tw)
    task["requirements"] = dict(max_scan_s=cap, te_ms=window, locked={k: p0[k] for k in LOCKED})
    chosen, remedy = _pick(task, wanted, union, pool, tw, plan, rp0, radius, px, np.random.default_rng(rng.integers(2 ** 32)), chain)
    if chosen is None:
        raise Reject("no remedy that fits the scan time and cleans every finding")
    rp1 = plan(remedy)["realised"]
    order = rng.permutation(len(chosen))
    findings = [dict(chosen[int(j)], id=f"F{i + 1}") for i, j in enumerate(order)]
    engine = tw.engine()
    snr0, snr1 = M.snr(engine, rp0, ref), M.snr(engine, rp1, ref)
    req = dict(task["requirements"], max_pixel_mm=math.ceil(px * 1e4) / 1e4, min_snr_reference=round(snr1 * float(rng.uniform(*lv["snr_slack"])), 1))
    task["requirements"] = req
    exps = experiments(task, plan)
    cost = {e["name"]: e["scan_s"] or 0.0 for e in exps}
    battery = sum(v for k, v in cost.items() if k == "complaint" or k in C.minimal_battery(vocabulary, [tuple(o) for o in draw["level_options"]]))
    budget = math.ceil(lv["budget_factor"] * battery)
    if budget > BUDGET_MAX * pl0["scan_s"]:
        raise Reject("the experiments are not cheaper than the examination")
    return dict(
        seed=seed, world_spec=spec, budget_s=float(budget),
        task=dict(exam=task["exam"], current_protocol=p0, current_header=pl0["header"], current_scan_s=pl0["scan_s"],
                  findings=[dict(id=f["id"], y_mm=f["y_mm"], x_mm=f["x_mm"], radius_mm=f["radius_mm"]) for f in findings],
                  reference_region=dict(ref), requirements=req, causes_vocabulary=vocabulary, diagnosis=task["diagnosis"]),
        hidden=dict(mechanisms=list(mechs), causes={f["id"]: f["causes"] for f in findings}, armed=sorted(armed), remedy=remedy,
                    remedy_scan_s=plan(remedy)["scan_s"], battery_s=round(battery, 2), experiment_s={k: round(v, 3) for k, v in cost.items()},
                    snr_complaint=round(snr0, 1), snr_remedy=round(snr1, 1),
                    solo={f["id"]: f["solo"] for f in findings}, complaint={f["id"]: f["complaint"] for f in findings},
                    remedied={f["id"]: f["remedied"] for f in findings},
                    pool={"+".join(k) or "benign": len(v) for k, v in pool.items()}))


def reference_scale(before, after, rp, reference) -> float:
    """Signal of the reference region after an experiment over its signal before: what a uniform change (decay
    between two echo times) does to the gel."""
    m = M.area_mask(rp, reference)
    return float(np.sum(m * after) / max(float(np.sum(m * before)), 1e-30))


def experiment_images(task, plan, tw, reference=None, complex_scans=None) -> dict:
    """Noise-free images of the differential experiments of ``experiments`` at the repetition time and flip angle of
    the complaint protocol: {mechanism: (image before, image after on the grid before, realised before, shape_only)}.
    ``reference``: the reference region, for the experiments that are compared relative to it. ``complex_scans``:
    a dict that receives the complex image of every scan by name."""
    p0 = task["current_protocol"]
    engine, scans, out = tw.engine(), {}, {}
    for e in experiments(task, plan):
        q = e["protocol"] if adaptive(task) else dict(e["protocol"], tr_ms=p0["tr_ms"], flip_deg=p0["flip_deg"])
        rp = M.realise(q, tw.world)
        data = engine.acquire(rp, None)[0]
        scans[e["name"]] = (np.abs(data), rp)
        if complex_scans is not None:
            complex_scans[e["name"]] = data
        if e["baseline"]:
            before, rp_b = scans[e["baseline"]]
            after = M.block_average(scans[e["name"]][0], rp, rp_b) if rp["matrix"] != rp_b["matrix"] else scans[e["name"]][0]
            scale = e["scale"] * (reference_scale(e["scale"] * before, after, rp_b, reference) if e.get("relative") else 1.0)
            out[e["name"]] = (scale * before, after, rp_b, e["shape_only"])
    return out


def experiments_agree(chain: dict, site, want, only=None) -> bool:
    """A finding is served only where the catalogue's differential experiments are diagnostic: an experiment changes
    the finding by at least ``CHAIN_PRESENT`` when the finding carries a cause that responds to it and by less than
    ``CHAIN_ABSENT`` when it does not. ``only``: the experiments that count (those of the finding's look class)."""
    for mech, (before, after, rp, shape_only) in chain.items():
        if mech not in C.BY_MECHANISM or (only is not None and mech not in only):
            continue
        c, should = M.change(before, after, M.area_mask(rp, site), shape_only), C.expected(want, mech)
        if should is not None and ((c < CHAIN_PRESENT) if should else (c >= CHAIN_ABSENT)):
            return False
    return True


def slice_ratio(chain: dict, site) -> float:
    """Signal per mm of the half slice over that of the full slice inside a finding (the dose experiment)."""
    before, after, rp, _ = chain["dropout"]
    m = M.area_mask(rp, site)
    return float(np.sum(m * after) / max(float(np.sum(m * before)), 1e-30))


def _pick(task, wanted, union, pool, tw, plan, rp0, radius, px, rng, chain):
    """A site per finding with the wanted cause set on the complaint protocol, diagnostic experiments, and clean
    under the remedy for the union of the causes. Returns (sites, remedy) or (None, None)."""

    def candidates(want):
        cand = list(pool.get(want, []))
        if cand:                                                          # near the border of the folded strip first (see ``fold_depth``)
            depth = np.array([fold_depth(tw.world, rp0, s) for s in cand])
            scale = FOLD_PREFERENCE_MM[0] if C.BY_MECHANISM["wrap"].label in want else FOLD_PREFERENCE_MM[1]
            cand = [cand[int(j)] for j in np.argsort(-(-np.abs(depth) / scale + rng.gumbel(size=len(cand))))]
        return [s for s in cand[:PICK_TRIES] if experiments_agree(chain, s, want)]

    def place(k, chosen, remedy, rp1):
        if k == len(wanted):
            return chosen
        want = wanted[k]
        for s in candidates(want):
            if any(math.hypot(s["y_mm"] - o["y_mm"], s["x_mm"] - o["x_mm"]) < 2 * radius + 2 * px for o in chosen):
                continue
            err0, err1 = tw.errors(rp0, s), tw.errors(rp1, s)
            if is_clean(err1, VERIFY) and (not want or not is_clean(err0)) and (want or is_clean(err0, VERIFY)):
                done = place(k + 1, chosen + [dict(s, causes=list(want), complaint={a: round(b, 4) for a, b in err0.items()},
                                                   remedied={a: round(b, 4) for a, b in err1.items()})], remedy, rp1)
                if done is not None:
                    return done
                if k > 0:                                                 # one level of backtracking is enough for a few findings
                    return None
        return None

    remedy = assemble(task, union, plan)
    if remedy is None:
        return None, None
    chosen = place(0, [], remedy, plan(remedy)["realised"])
    return (chosen, remedy) if chosen is not None else (None, None)


# ------------------------------------------------------------------------------------------------ constructed sites
# A level with ``constructed`` draws the cause of every finding first and then builds an object that realises it:
#
#     object      an elongated body of gel that ends inside the phase field of view, so that it does not fold itself
#     position    one rule for every cause and both looks (``layout``): each finding has a stripe of its own along the
#                 phase axis, and lies between ``PE_INNER_MM`` of the centre line and the rim of the body
#     dark look   a disc darker than the gel by a drawn amount (``DARK_LEVEL``), because
#                   susceptibility_signal_loss    a patch of through-slice field gradient over plain gel
#                   opposed_phase_cancellation    a vial that holds water and fat in the same voxels
#                   (none)                        a vial of gel with fewer protons
#     bright look a disc brighter than the gel (``BRIGHT_LEVEL``), because
#                   wrap_around                   a tube beside the body, one phase field of view away, folds onto gel
#                   periodic_motion_ghost         a vial on the finding's phase-encode line pulses; one copy lands here
#                   (none)                        a vial of doped gel (shorter T1)
#     remedy      the catalogue's, with the doses the reference measures: the slice from the half-slice experiment at
#                 the echo time of the remedy, the averages from the displacement of the ghost copy (the largest
#                 number that keeps every copy off every finding)
#     SNR         the delivered protocol needs a stated share (``snr_slack``) of the SNR of the verified remedy. The
#                 share is on the card, the number is hidden: a dose stronger than needed (a thinner slice, fewer
#                 averages) fails, and the card does not name the dose
# Nothing is searched: an attempt fails only when a constructed site does not verify on the twins.
def _unit_signal(rp, cls: dict, flip_rad: float, te_ms: float, field_t: float) -> complex:
    """Echo signal of one material per unit of volume share, at a flip angle (rad, slice centre) and an echo time."""
    cls = dict(dict(fat=False), **cls)
    cls["t2s_ms"] = 1.0 / (1.0 / cls["t2_ms"] + 1.0 / cls["t2p_ms"])
    amp = kspace.slice_signal(dict(rp, te_ms=te_ms), cls, flip_rad, rp["rf_tbw"])[0][0] if rp.get("rf_tbw") else \
        kspace.gre_steady(dict(rp, te_ms=te_ms), cls, np.array([flip_rad]))[0][0]
    return complex(cls["pd"] * amp * S.time_function(cls, np.zeros(1), np.zeros(1), te_ms, "gre", field_t, tissue.FAT_PEAKS)[0])


def _doped(gel: dict, t1_ms: float) -> dict:
    return dict(gel, t1_ms=float(t1_ms), t2_ms=float(min(gel["t2_ms"], 0.6 * t1_ms)))


T1_NODES_MS = (20.0, 35.0, 60.0, 100.0, 160.0, 250.0, 400.0, 650.0, 1000.0, 1600.0)


def _brighter(gel: dict, signal):
    """Gel doped to a shorter T1 (and, if that is not enough, with more protons): returns the function
    ratio -> material that gives ``ratio`` times the signal of the gel (the brightest such material when the ratio
    is out of reach). The signal is tabulated on ``T1_NODES_MS`` once and interpolated."""
    base = abs(signal(gel))
    nodes = [t for t in T1_NODES_MS if t < gel["t1_ms"]] + [gel["t1_ms"]]
    table = np.array([abs(signal(dict(_doped(gel, t), pd=1.0))) / base for t in nodes])   # falls with T1

    def material(ratio):
        for pd in (gel["pd"], 1.0):
            if table[0] * pd >= ratio:
                t1 = float(np.exp(np.interp(-ratio, -table * pd, np.log(nodes))))
                return dict(_doped(gel, t1), pd=pd)
        return dict(_doped(gel, nodes[0]), pd=1.0)
    return material


def _emulsion(level: float, in_phase: float, water, fat) -> tuple:
    """(volume share of fat, proton density of the water part) of a mixture that shows ``level`` x the gel at the
    complaint echo and ``in_phase`` x the gel at the in-phase echo. ``water(te_index)``, ``fat(te_index)``: complex
    signal per unit share of water of proton density 1 and of oil, relative to the gel, at the complaint echo (0)
    and at the in-phase echo (1)."""
    def mix(phi):
        lo, hi = 0.0, 4.0                                                 # water amount that gives ``in_phase`` at the in-phase echo
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if abs(mid * water(1) + phi * fat(1)) < in_phase else (lo, mid)
        a = 0.5 * (lo + hi)
        return a, abs(a * water(0) + phi * fat(0))
    lo, hi = 0.0, 0.6
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mix(mid)[1] > level else (lo, mid)
    phi = 0.5 * (lo + hi)
    a = mix(phi)[0]
    if abs(mix(phi)[1] - level) > 0.02 or not 0.0 < phi < 0.55:
        raise Reject("no water-fat mixture of this level")
    return phi, a / (1.0 - phi)


def layout(rng, n: int, fov_pe_mm: float) -> dict:
    """Where the findings of a constructed level lie, in (phase, readout) coordinates, drawn before and independently
    of their causes: ``n`` stripes along the phase axis at least ``COLUMN_GAP_MM`` apart, uniform among such
    arrangements, one finding per stripe at a phase coordinate uniform between ``PE_INNER_MM`` (or, on a larger phase
    field of view, the distance at which a tube that folds onto the finding still lies inside the bore) and the rim
    of the body, on either side. Also the half-axes of the body."""
    inner = max(PE_INNER_MM, fov_pe_mm - TUBE_REACH_MM + 0.5)
    a_pe = fov_pe_mm / 2.0 - float(rng.uniform(*BODY_PE_MARGIN_MM))
    a_ro = float(rng.uniform(*BODY_RO_MM))
    reach = COLUMN_REACH * a_ro
    slack = 2.0 * reach - (n - 1) * COLUMN_GAP_MM
    if slack < 0:
        raise Reject("no room for the stripes")
    ro = -reach + np.sort(rng.uniform(0.0, slack, n)) + COLUMN_GAP_MM * np.arange(n)
    sites = []
    for x in ro[rng.permutation(n)]:
        top = a_pe * math.sqrt(1.0 - (float(x) / a_ro) ** 2) - DISC_MM - 5.0
        if top < inner + 2.0:
            raise Reject("the body is too thin for a finding at this stripe")
        sites.append((float(rng.choice([-1.0, 1.0]) * rng.uniform(inner, top)), float(x)))
    return dict(a_pe=a_pe, a_ro=a_ro, sites=sites)


def ghost_phase(phase_key: int, rp: dict) -> float:
    """Phase (rad) of a pulsing structure at the first line of the scan ``rp`` (``kspace.Engine._motion``)."""
    h = hashlib.sha256(repr((int(phase_key), kspace.physics_key(rp, averages=True))).encode()).digest()
    return 2.0 * math.pi * int.from_bytes(h[:4], "big") / 2 ** 32


def ghost_share(cycles_per_tr: float, averages: int) -> float:
    """Amplitude of a first-order ghost copy relative to depth / 2 x its source: |sin(pi f TR N) / (N sin(pi f TR))|."""
    return abs(math.sin(math.pi * cycles_per_tr * averages) / (averages * math.sin(math.pi * cycles_per_tr)))


def ghost_track(before, after, rp: dict, finding: dict, averages_after: int = 1, span_after: float | None = None):
    """Where the pulsing source of a ghost copy lies, from two complex images of the complaint protocol: the complaint
    itself and the motion experiment (``averages_after`` averages). A copy lies frac(f TR averages) x the phase
    field of view from its source on either side, so the complex difference of the two scans (the object cancels)
    holds four copies along the phase-encode line of the finding, symmetric about the source: the two of the
    complaint, one of them on the finding, and the two of the experiment, closer to the source and stronger by the
    ratio of ``ghost_share``. The copy on the finding gives the shape of a copy; the source is the position for
    which the four copies, placed by the formula and folded into the field of view, fit the difference best.
    ``span_after``: the encoded phase field of view of the experiment when it differs from the complaint's (phase
    oversampling: its copies lie that much further out; both images are on the complaint's grid).
    Returns ``pe_source_mm`` and ``cycles_per_tr`` (f x TR, the smallest value that puts a copy on the finding), or
    None when no position explains the difference (``GHOST_MISFIT``)."""
    y, x = M.pixel_axes(rp)
    pe_y = rp["pe_dir"] == "y"
    pe, ro, axis = (y, x, "y_mm") if pe_y else (x, y, "x_mm")
    here, span = finding[axis], rp["fov_pe_mm"] * rp["lines_full"] / rp["matrix_pe"]
    band = np.abs(ro - finding["x_mm" if pe_y else "y_mm"]) <= max(finding["radius_mm"], 0.51 * abs(ro[1] - ro[0]))
    diff = np.abs(np.asarray(after) - np.asarray(before))
    q = diff[:, band].mean(1) if pe_y else diff[band, :].mean(0)
    copy = float(np.interp(here, pe, q))
    if copy <= 0.0:
        return None
    q = q / copy
    u = np.arange(0.0, GHOST_OWN_MM + 4.0, 0.5)                           # the copy on the finding, both sides: the shape of a copy
    shape = np.minimum(np.interp(here + u, pe, q, left=0.0, right=0.0), np.interp(here - u, pe, q, left=0.0, right=0.0))
    shape[u > GHOST_OWN_MM] = 0.0
    fold = lambda d: (d + span / 2.0) % span - span / 2.0                 # noqa: E731
    best = None
    for source in np.arange(pe[0], pe[-1] + 1e-9, 0.5):
        d0 = abs(fold(here - source))
        if d0 < GHOST_OWN_MM:
            continue
        nu = d0 / (span * rp["averages"])
        d1, gain = (nu * averages_after) % 1.0 * (span_after or span), ghost_share(nu, averages_after) / ghost_share(nu, rp["averages"])
        model = sum(np.interp(np.abs(fold(pe - source - k * d0)), u, shape, right=0.0) for k in (1.0, -1.0)) + \
            gain * sum(np.interp(np.abs(pe - source - k * d1 if span_after else fold(pe - source - k * d1)), u, shape, right=0.0) for k in (1.0, -1.0))
        misfit = float(np.sum(np.abs(q - model)) / np.sum(q))
        if best is None or misfit < best[0]:
            best = (misfit, float(source), nu)
    if best is None or best[0] > GHOST_MISFIT:
        return None
    return dict(pe_source_mm=best[1], cycles_per_tr=best[2])


def ghost_doses(findings, tracks: dict) -> list:
    """What ``catalogue.ghost_averages`` needs, from the tracked copies ({finding id: ``ghost_track`` result}): each
    with the findings of its own phase-encode line (``column``)."""
    out = []
    for fid, t in tracks.items():
        me = next(f for f in findings if f["id"] == fid)
        ro = "y_mm" if "x_mm" == t["axis"] else "x_mm"
        out.append(dict(pe_source_mm=t["pe_source_mm"], cycles_per_tr=t["cycles_per_tr"],
                        column=[i for i, f in enumerate(findings) if abs(f[ro] - me[ro]) < C.GHOST_CLEAR_MM]))
    return out


def _ghost_site(rng, pe, top_source, fov_pe, averages, fov_mm):
    """Displacement (mm) of the copy that makes a ghost finding at phase coordinate ``pe``: far enough for the two
    copies of the one-average experiment (next to the source) to stay clear of the finding, and for the one-average
    remedy on the full field of view to take the copy off the finding; short enough for the second copy to stay
    away when it folds; the source inside the body."""
    lo = max((GHOST_OWN_MM + DISC_MM + GHOST_MARGIN_MM) / (1.0 - 1.0 / averages),
             (C.GHOST_CLEAR_MM + GHOST_MARGIN_MM) / abs(1.0 - fov_mm / (fov_pe * averages)), abs(pe) - top_source)
    hi = min((fov_pe - GHOST_ALIAS_MM) / 2.0, abs(pe) + top_source)
    if hi < lo + 0.5:
        raise Reject("no displacement of a ghost copy fits this sheet")
    return float(rng.uniform(lo, hi))


def _reference_region(lay, structures, pe_dir):
    """Candidates for the reference region: plain gel beyond the outermost stripe, on either side."""
    edge = max(abs(x) for _, x in lay["sites"]) + DISC_MM + REF_RADIUS_MM + 4.0
    out = []
    for side in (1.0, -1.0):
        for pe in (0.0, 8.0, -8.0):
            ro = side * edge
            if (pe / lay["a_pe"]) ** 2 + (ro / lay["a_ro"]) ** 2 < 0.7 and \
                    all(math.hypot(pe - a, ro - b) >= DISC_MM + REF_RADIUS_MM + 4.0 for a, b in structures):
                y, x = (pe, ro) if pe_dir == "y" else (ro, pe)
                out.append(dict(id="REF", y_mm=round(y, 1), x_mm=round(x, 1), radius_mm=REF_RADIUS_MM))
    return out


def _tune_ghosts(world, rp0, movers, sites):
    """Phase keys of the pulsing vials for which each copy shows its drawn level on the complaint protocol. A copy
    adds to the gel with the phase of the pulsation at the start of the scan (``ghost_phase``), turned one way or
    the other depending on which of the two copies it is. Three scans at most: the copies as they come, with the
    keys chosen for one direction, and with the other direction where that missed. ``movers``: indices into
    ``world["motion"]``; ``sites``: per mover (finding circle, wanted level relative to the gel under it).
    Returns the keys, or None."""
    off = np.asarray(kspace.Engine(world, effects=kspace.ALL - {"motion", "noise"}).acquire(rp0, None)[0])
    masks = [M.area_mask(rp0, f) for f, _ in sites]
    gel = [complex(np.sum(m * off) / np.sum(m)) for m in masks]

    def scan(keys):
        for j, k in zip(movers, keys):
            world["motion"][j]["phase_key"] = int(k)
        on = np.asarray(kspace.Engine(world, effects=kspace.ALL - {"noise"}).acquire(rp0, None)[0])
        return [complex(np.sum(m * on) / np.sum(m)) - g for m, g in zip(masks, gel)]

    keys = [0] * len(movers)
    seen = scan(keys)                                                    # complex amplitude of every copy with key 0
    phase = [ghost_phase(k, rp0) for k in range(GHOST_KEYS)]

    def choose(i, turn):
        want = sites[i][1] * abs(gel[i])
        return min(range(1, GHOST_KEYS), key=lambda k: abs(abs(gel[i] + seen[i] * np.exp(1j * turn * (phase[k] - phase[0]))) - want))

    trial = [choose(i, 1.0) for i in range(len(movers))]
    got = scan(trial)
    final = []
    for i in range(len(movers)):
        ok = abs(abs(gel[i] + got[i]) - sites[i][1] * abs(gel[i])) <= 0.08 * abs(gel[i])
        final.append(trial[i] if ok else choose(i, -1.0))
    if final != trial:
        got = scan(final)
        if any(abs(abs(gel[i] + got[i]) - sites[i][1] * abs(gel[i])) > 0.08 * abs(gel[i]) for i in range(len(movers))):
            return None
    return final


def build_constructed(seed: str, level_cfg: dict, draw: dict) -> dict:
    """One attempt at an instance of a constructed level on the world of ``seed`` (see the comment above). Raises
    ``Reject`` when a constructed site does not verify; the wanted causes are never changed."""
    lv = level_cfg
    rng = W.rng_for(seed, "task")
    field_t, armed, wanted, cls = draw["field_t"], set(draw["armed"]), [tuple(w) for w in draw["wanted"]], draw["look"]
    base = dict(kind="qa", n=int(lv["protocol"]["grid_n"]), b0_t=float(field_t), b0_hz=5.0, cf_offset_hz=2.0,
                sigma0=float(lv["sigma0"] * np.exp(rng.uniform(-NOISE_SPREAD, NOISE_SPREAD))), edge_mm=SOFT_EDGE_MM)
    vocabulary = [lab for lab in lv["causes"] if C.CATALOGUE[lab].mechanism not in lv.get("never", ())]
    exam = exam_card(base)
    p0 = draw_protocol(rng, exam, armed, lv["protocol"], vocabulary, True)
    plan = lambda p: console.plan(p, exam)                                # noqa: E731
    pl0 = plan(p0)
    rp0 = pl0["realised"]
    px, fov_pe, te0 = rp0["fov_mm"] / rp0["matrix"], rp0["fov_pe_mm"], p0["te_ms"]
    radius = round(max(RADIUS_PX * px, RADIUS_MIN_MM), 1)
    task = dict(exam=dict(object="QA phantom", field_t=field_t), current_protocol=p0, causes_vocabulary=vocabulary, diagnosis="by_look")
    exps = experiments(task, plan)
    cost = {e["name"]: e["scan_s"] for e in exps}
    te1 = next((e["protocol"]["te_ms"] for e in exps if e["name"] == "fatphase"), None)
    if te1 is None or te1 >= te0:
        raise Reject("no earlier in-phase echo")
    # ---- positions first, for every cause alike; then the object
    lay = layout(rng, len(wanted), fov_pe)
    order = rng.permutation(len(wanted))                                  # the ids of the findings, drawn before anything that depends on a cause
    pe_y = p0["pe_dir"] == "y"
    yx = lambda pe, ro: (pe, ro) if pe_y else (ro, pe)                    # noqa: E731
    ba, bb = (lay["a_pe"], lay["a_ro"]) if pe_y else (lay["a_ro"], lay["a_pe"])   # half-axes along y and x
    qa = dict(n_oil=0, n_ring=(0, 0), p_centre=0.0, offset_mm=0.0, body_rotation_deg=[0.0], body_radius_mm=(ba, ba), body_aspect=(ba / bb, ba / bb))
    plain = W.make_world(seed, dict(base, qa=qa), layout_only=True)
    gel = {k: plain["tissues"]["body"][k] for k in ("pd", "t1_ms", "t2_ms", "t2p_ms", "adc_um2_ms")}
    oil = dict(tissue.table(field_t)["subcut_fat"], fat=True)
    flip = math.radians(p0["flip_deg"]) * plain["system"]["tx_gain"]
    sig = lambda c, te=te0: _unit_signal(rp0, c, flip, te, field_t)       # noqa: E731
    gel0, gel1 = sig(gel), sig(gel, te1)
    brighter = _brighter(gel, sig) if cls == "bright" else None
    level_range = DARK_LEVEL if cls == "dark" else BRIGHT_LEVEL
    inserts, patches, motion, ghosts, structures, levels = [], [], [], [], [], []

    def disc(pe, ro, material, weight=1.0):
        y, x = yx(pe, ro)
        inserts.append(dict(y_mm=round(y, 2), x_mm=round(x, 2), radius_mm=DISC_MM, tissue=material, weight=round(float(weight), 4)))
        return len(inserts) - 1

    for (pe, ro), want in zip(lay["sites"], wanted):
        level = float(rng.uniform(*level_range))
        label = want[0] if want else None
        structures.append((pe, ro))
        if label is None:
            disc(pe, ro, dict(gel, pd=level * gel["pd"]) if cls == "dark" else brighter(level))
        elif C.CATALOGUE[label].mechanism == "dropout":                   # the loss factor of the console's pulse at this slice and echo
            turns = C._cycles(level, lambda c: float(rf.dephasing_factor(np.array([c]), rp0["rf_tbw"])[0]), hi=0.45 * rp0["rf_tbw"])
            y, x = yx(pe, ro)
            patches.append(dict(y_mm=round(y, 2), x_mm=round(x, 2), sigma_mm=SOFT_EDGE_MM, flat_mm=round(DISC_MM - 0.84 * SOFT_EDGE_MM, 2),
                                peak_hz_per_mm=round(float(rng.choice([-1.0, 1.0])) * turns / (p0["slice_mm"] * te0 * 1e-3), 4)))
        elif C.CATALOGUE[label].mechanism == "fatphase":
            water = (sig(dict(gel, pd=1.0)) / gel0, sig(dict(gel, pd=1.0), te1) / gel1)
            fat = (sig(oil) / gel0, sig(oil, te1) / gel1)
            share, pd = _emulsion(level, float(rng.uniform(*EMULSION_IN_PHASE)), water.__getitem__, fat.__getitem__)
            disc(pe, ro, dict(gel, pd=pd), 1.0 - share)
            disc(pe, ro, "oil", share)
        elif C.CATALOGUE[label].mechanism == "wrap":                      # a tube one phase field of view away, on the far side
            away = pe - math.copysign(fov_pe, pe)
            if abs(away) > TUBE_REACH_MM:
                raise Reject("the tube would lie outside the bore")
            disc(away, ro, dict(gel, pd=(level - 1.0) * gel["pd"]))
        else:                                                             # a pulsing vial on this phase-encode line
            top = lay["a_pe"] * math.sqrt(1.0 - (ro / lay["a_ro"]) ** 2) - DISC_MM - 4.0
            d0 = _ghost_site(rng, pe, top, fov_pe, p0["averages"], p0["fov_mm"])
            source = pe - math.copysign(d0, pe)
            nu = d0 / (fov_pe * p0["averages"])
            share = 0.5 * GHOST_DEPTH * ghost_share(nu, p0["averages"])
            material = brighter((level - 1.0) / GHOST_ALIGN / share)
            level = min(level, 1.0 + GHOST_ALIGN * share * abs(sig(material)) / abs(gel0))
            k = disc(source, ro, material)
            motion.append(dict(cls=f"insert{k}", kind="pulse", f_hz=round(nu / (p0["tr_ms"] * 1e-3), 5), depth=GHOST_DEPTH, phase_key=0))
            ghosts.append((len(levels), len(motion) - 1))
            structures.append((source, ro))
            structures.extend((c, ro) for c in C.ghost_copies(source, nu, p0["averages"], fov_pe))
        levels.append(level)
    spec = dict(base, qa=dict(qa, inserts=inserts))
    if patches:
        spec["dbdz"] = dict(n=0, patches=patches)
    sites = [dict(zip(("y_mm", "x_mm"), (round(v, 1) for v in yx(pe, ro))), radius_mm=radius) for pe, ro in lay["sites"]]
    if motion:                                                            # every copy with the phase that shows its drawn level
        spec["motion"] = motion
        keys = _tune_ghosts(W.make_world(seed, spec), rp0, [j for _, j in ghosts], [(sites[i], levels[i]) for i, _ in ghosts])
        if keys is None:
            raise Reject("a ghost copy does not show its level")
        spec["motion"] = [dict(m, phase_key=int(keys[[j for _, j in ghosts].index(n)])) for n, m in enumerate(motion)]
    world = W.make_world(seed, spec)
    mechs = C.mechanisms(lv["causes"])
    tw = M.Twins(world, mechs)
    complaint = tw.image(rp0)
    ref = next((r for r in _reference_region(lay, structures, p0["pe_dir"])
                if all(v < C.BY_MECHANISM[m].absent for m, v in tw.solo(rp0, r).items())), None)
    if ref is None:
        raise Reject("no clean reference region")
    # ---- every site carries exactly its causes, looks as drawn, and answers the experiments of its look
    chosen = []
    for site, want in zip(sites, wanted):
        solo = tw.solo(rp0, site)
        if cause_set(solo) != tuple(sorted(want)):
            raise Reject(f"the site built for {'+'.join(want) or 'a benign finding'} carries {cause_set(solo)}")
        lk = look(complaint, rp0, site, ref)
        if lk["cls"] != cls or not look_is_clear(lk):
            raise Reject(f"a {cls} finding looks {lk['cls']} ({lk['level']:.2f})")
        chosen.append(dict(site, look=cls, causes=list(want), solo={k: round(v, 4) for k, v in solo.items()}, level=round(lk["level"], 3)))
    plans = class_plan(task)
    scans = {}
    chain = experiment_images(task, plan, tw, ref, scans)
    for s in chosen:
        if not experiments_agree(chain, s, s["causes"], class_experiments(plans[cls])):
            raise Reject("an experiment of the look is not diagnostic on a constructed site")
    # ---- the remedy with the doses the reference measures, verified on the twins
    cap = round(pl0["scan_s"] * float(rng.uniform(*lv["time_slack"])), 1)
    window = [round(max(0.5, te0 - 1.05 * C.fat_period_ms(field_t)), 2), round(te0 + 1.05 * C.fat_period_ms(field_t), 2)]
    free = tuple(lv.get("unlocked", ()))
    findings = [dict(chosen[int(j)], id=f"F{i + 1}") for i, j in enumerate(order)]
    task["findings"] = [dict(id=f["id"], y_mm=f["y_mm"], x_mm=f["x_mm"], radius_mm=f["radius_mm"]) for f in findings]
    task["requirements"] = dict(max_scan_s=cap, te_ms=window, locked={k: p0[k] for k in LOCKED if k not in free})
    for k in free:
        lo, hi = WINDOWS[k]
        task["requirements"][k] = [C.SLICE_MIN_MM if lo is None else round(lo * p0[k], 2), round(hi * p0[k], 2)]
    union = sorted({lab for f in findings for lab in f["causes"]})
    doses = measured_doses(task, findings, chain, scans)
    remedy = assemble(task, union, plan, doses)
    if remedy is None:
        raise Reject("no remedy that fits the scan time")
    if "ghosts" in doses and any(assemble(task, union, plan, dict(doses, ghost_clear_mm=C.GHOST_CLEAR_MM + d)) != remedy
                                 for d in (-GHOST_EDGE_MM, GHOST_EDGE_MM)):
        raise Reject("the averages of the ghost remedy are on the edge")   # a measurement with noise could decide either way
    rp1 = plan(remedy)["realised"]
    for f in findings:
        err0, err1 = tw.errors(rp0, f), tw.errors(rp1, f)
        if not is_clean(err1, VERIFY_BUILT) or (f["causes"] and is_clean(err0)) or (not f["causes"] and not is_clean(err0, VERIFY)):
            raise Reject("the remedy does not clean every finding")
        f.update(complaint={a: round(b, 4) for a, b in err0.items()}, remedied={a: round(b, 4) for a, b in err1.items()})
    engine = tw.engine()
    snr0, snr1 = M.snr(engine, rp0, ref), M.snr(engine, rp1, ref)
    # the SNR floor is a share of what the verified remedy has; the share is on the card, the number is not (it would name the doses)
    share = round(float(rng.uniform(*lv["snr_slack"])), 2)
    task["requirements"].update(max_pixel_mm=math.ceil(px * 1e4) / 1e4, snr_share=share)
    # ---- budget: the first look and the two experiments of the costlier look; no four scans of any protocol fit
    battery = cost["complaint"] + sum(cost[e] for e in C.minimal_battery(vocabulary, [tuple(o) for o in draw["level_options"]]))
    budget = math.ceil(cost["complaint"] + max(sum(cost[e] for e in class_experiments(plans[k])) for k in LOOK_CAUSES))
    if budget > 4.0 * SCAN_FLOOR_S:
        raise Reject("the budget would hold four scans")
    classes = {f["id"]: f["look"] for f in findings}
    flags, used = {}, cost["complaint"]
    while True:                                                           # what the adaptive diagnosis costs on this instance
        todo = next_experiments(plans, classes, flags)
        if not todo:
            break
        for name in todo:
            flags[name] = {f["id"]: bool(C.expected(f["causes"], name)) for f in findings}
            used += cost[name]
    return dict(
        seed=seed, world_spec=spec, budget_s=float(budget),
        task=dict(exam=task["exam"], current_protocol=p0, current_header=pl0["header"], current_scan_s=pl0["scan_s"],
                  findings=task["findings"], reference_region=dict(ref), requirements=task["requirements"], causes_vocabulary=vocabulary,
                  diagnosis=task["diagnosis"]),
        hidden=dict(look=classes, adaptive_s=round(used, 2), battery_over_budget=round(battery / budget, 3), mechanisms=list(mechs),
                    causes={f["id"]: f["causes"] for f in findings}, armed=sorted(armed), remedy=remedy, remedy_scan_s=plan(remedy)["scan_s"],
                    battery_s=round(battery, 2), experiment_s={k: round(v, 3) for k, v in cost.items()},
                    snr_complaint=round(snr0, 1), snr_remedy=round(snr1, 1), min_snr_reference=round(share * snr1, 1),
                    level={f["id"]: f["level"] for f in findings},
                    solo={f["id"]: f["solo"] for f in findings}, complaint={f["id"]: f["complaint"] for f in findings},
                    remedied={f["id"]: f["remedied"] for f in findings}, pool={}))


def measured_doses(task: dict, findings, chain: dict, complex_scans: dict) -> dict:
    """The doses of the remedies from the experiment images, as the reference policy measures them: the slice from
    the largest half-slice ratio among the findings with signal loss, the averages from the tracked ghost copies.
    ``findings``: dicts with ``id``, the circle and ``causes``; ``chain``: what ``experiment_images`` returns for
    the experiments that were run; ``complex_scans``: complex images of the scans by name."""
    out = {}
    loss, ghost = (C.BY_MECHANISM[m].label if m in C.BY_MECHANISM else None for m in ("dropout", "motion"))
    lossy = [f for f in findings if loss in f["causes"]]
    if lossy and "dropout" in chain:
        out["slice_ratio"] = max(slice_ratio(chain, f) for f in lossy)
    moved = [f for f in findings if ghost in f["causes"]]
    if moved and "motion" in chain and "motion" in complex_scans:
        rp = chain["motion"][2]
        tracks = {f["id"]: ghost_track(complex_scans["complaint"], complex_scans["motion"], rp, f, 1) for f in moved}
        if all(tracks.values()):
            axis = "x_mm" if rp["pe_dir"] == "x" else "y_mm"
            out["ghosts"] = ghost_doses(findings, {k: dict(v, axis=axis) for k, v in tracks.items()})
    return out
