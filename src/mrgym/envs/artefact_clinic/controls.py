"""Control policies of ``Scanner/ArtefactClinic``. Every policy plays through the agent's interface.

Must pass
  reference       levels 1 and 2: the differential experiments at a quarter of the repetition time, each added to
                  the one before (oversampling, remedy bandwidth, in-phase echo, twice the matrix); a finding carries
                  the mechanisms whose removal changed it. Level 3: a first look at the complaint image, then per
                  finding the experiments its look calls for (darker than the gel: half the slice, then the in-phase
                  echo if the slice did not change it; brighter: oversampling, then one average if oversampling
                  changed it); the slice dose from the measured loss at the echo time of the remedy, the averages
                  from the tracked ghost copy. Then the catalogue's remedy for exactly the causes found, fitted with
                  the dry run
Must fail
  prior           no scan; every finding gets the fat mechanism the protocol arms; no remedy
  fixed_recipe    one token scan, then the best answer that uses only the card, tuned on other worlds: one cause for
                  every finding, wrap-around for the findings an object of a tuned size would fold onto, the remedy;
                  at level 3 also every assignment of causes to the findings in the order of their id, of either
                  coordinate, of the distance from the phase axis or from the isocentre (``tune``)
  remedy_battery  one scan, then every remedy of the catalogue at once (the textbook battery), one cause named
  brute_force     trial and error at full length: the complaint, then one remedy after the other, while the budget
                  lasts; a finding is blamed on the first remedy that changed its mean by 10 %
  wrong_units     the reference's answer with the echo time in seconds
Ablations (the reference with one decision removed)
  first_look      diagnosis from the complaint image alone, without a differential experiment
  told_cause_fixed_dose   is told the hidden causes, applies the first-line remedy of each at the catalogue's dose,
                  whatever the examination locks and without measuring a dose (required at level 3)
  half_budget     the reference with half of the scan time (required at levels 1 and 2)
  battery_within_budget   the minimal battery in the open loop's order as far as the budget goes, without the first
                  look: a finding carries whatever experiment changed it (required at level 3)
Reported
  told_cause_tuned_dose   is told the hidden causes, applies the first-line remedy of each with the best fixed dose:
                  the slice as the fraction of the complaint slice, and the number of averages, that pass on most
                  worlds of the ``tune`` split (``tune_doses``; stored in ``TUNED_DOSE``). What is left of the task
                  when the diagnosis is given and nothing is measured
  open_loop       the best fixed scan list within the budget and a fixed estimator. Levels 1 and 2: every experiment
                  the sheet arms, whatever the earlier ones showed.
                  Level 3: whichever of two fixed lists diagnoses more worlds of the ``tune`` split
                  (``tune_open_loop``; stored in ``OPEN_LOOP``). "single": the experiments of the level, one change
                  each, in the best order, run while the budget lasts; the estimator reads the look of the findings
                  from the first image, as the reference does, and names the most probable causes given the look,
                  the level's rule for an episode and every experiment that was run (``generator.infer``).
                  "several": three scans, the complaint, one scan that carries half the slice, full phase
                  oversampling and one average at once, and the in-phase echo (``_several``): the experiments of
                  the two looks do not disturb each other, so one scan can hold a change for each. Doses are measured
                  where the scan for them was run
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ...scanner import console
from . import catalogue as C
from . import generator as G
from . import metrics as M

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "remedy_battery", "brute_force", "wrong_units")
ABLATIONS = {"first_look": (1, 2, 3), "told_cause_fixed_dose": (3,), "half_budget": (1, 2), "battery_within_budget": (3,)}
REPORTED = ("told_cause_tuned_dose", "open_loop")
DESCRIPTIONS = dict(
    reference="differential experiments, each finding carries the mechanisms whose removal changed it; at level 3 chosen "
              "finding by finding from the first look, with measured doses (slice, averages); the catalogue's remedy for exactly those",
    prior="no scan; every finding blamed on the fat mechanism the protocol arms; no remedy",
    fixed_recipe="one token scan, then the best answer from the card alone, tuned on other worlds: one cause for every finding (wrap-around where an object of a tuned size would fold), at level 3 also causes by the order of the findings' ids or positions; the remedy at the catalogue's dose",
    remedy_battery="one scan, then every remedy of the catalogue at once; one cause named",
    brute_force="full-length scans of the complaint and of one remedy after the other while the budget lasts; no model, no cheap experiment",
    wrong_units="the reference's answer with the echo time in seconds",
    first_look="diagnosis from the complaint image alone (signal and texture of each finding; at level 3 the most probable causes of its look), no differential experiment",
    told_cause_fixed_dose="is told the hidden causes; applies the first-line remedy of each at the catalogue's dose, whatever the examination locks, no measured dose",
    half_budget="the reference with half of the scan time",
    told_cause_tuned_dose="is told the hidden causes; applies the first-line remedy of each with the fixed dose that passes on most tune worlds (level 3: slice fraction, averages), no measured dose",
    battery_within_budget="the minimal battery in the open loop's order as far as the budget goes, without the first look",
    open_loop="the best fixed scan list within the budget and a fixed estimator (level 3: tuned on the tune split between the single-change experiments in their best order and three scans that carry several changes at once)",
)
THRESHOLD = 0.06                   # a finding carries a mechanism when its removal changed the finding by this much
# The open-loop control of level 3, tuned by ``tune_open_loop`` on worlds of the public ``tune`` split
# (``python tools/build_clinic_bank.py --level 3 --n 48 --seeds tune --tune-open-loop``): the order in which the
# single-change experiments are run while the budget lasts, and which of the two fixed lists the control plays.
OPEN_LOOP = dict(order=("wrap", "motion", "fatphase", "dropout"), list="several")   # 48 tune worlds: single 26.44 in expectation, several 48
# The best fixed doses of the remedies that have one, tuned by ``tune_doses`` on the same worlds: the slice of the
# signal-loss remedy as a fraction of the complaint slice, the averages of the ghost remedy.
TUNED_DOSE = dict(slice_fraction=0.31, averages=1)                       # right on 9 of 22 tune worlds with signal loss, 14 of 26 with a ghost


def _task(sc):
    return sc.info()["task"]


def _image(sc, scan: int) -> np.ndarray:
    """Magnitude image of a stored scan (``ToolScanner.view`` hands a scripted policy the array). A text-only agent
    reads the same pixels with ``read_regions``: 1 mm circles on pixel centres, see the card."""
    return np.asarray(sc.view(scan)["array"], float)


def _noise(img, rp):
    """Noise SD of a pixel from the two ends of the readout axis, which the receiver filter keeps free of signal."""
    y, x = M.pixel_axes(rp)
    ro = x if rp["pe_dir"] == "y" else y
    far = np.abs(ro) >= rp["fov_mm"] / 2 - 10.0
    strip = img[:, far] if rp["pe_dir"] == "y" else img[far, :]
    return float(strip.std() / 0.655)


class _Clinic:
    """The experiments of one episode: scans by name, and what an experiment changed inside a finding."""

    def __init__(self, sc, budget_fraction=1.0):
        self.sc, self.t = sc, _task(sc)
        self.limit = budget_fraction * sc.budget()["budget_s"]
        self.spent = 0.0
        self.scans = {}
        self.exps = {e["name"]: e for e in G.experiments(self.t, lambda p: sc.plan(p))}

    def run(self, name, protocol=None):
        """Scan an experiment if it fits the allowance. Returns True when the scan is stored under ``name``."""
        protocol = protocol or self.exps[name]["protocol"]
        pl = self.sc.plan(protocol)
        if not pl["accept"] or self.spent + pl["scan_s"] > self.limit + 1e-9 or pl["scan_s"] > self.sc.budget()["remaining_s"]:
            return False
        r = self.sc.scan(protocol, rois=[])
        self.spent += r["cost_s"]
        img = _image(self.sc, r["scan_index"])
        self.scans[name] = dict(img=img, rp=r["realised"], sd=_noise(img, r["realised"]), index=r["scan_index"])
        return True

    def complex(self, name):
        """Complex image of a stored scan: magnitude x exp(i phase), both read with ``view``."""
        s = self.scans[name]
        if "complex" not in s:
            s["complex"] = s["img"] * np.exp(1j * np.asarray(self.sc.view(s["index"], kind="phase")["array"], float))
        return s["complex"]

    def pair(self, name):
        """(baseline image scaled, image of the experiment on the baseline grid, realised baseline, noise SD of the difference)."""
        e = self.exps[name]
        a, b = self.scans[e["baseline"]], self.scans[name]
        img_b, sd_b = b["img"], b["sd"]
        if b["rp"]["matrix"] != a["rp"]["matrix"]:
            img_b, sd_b = M.block_average(b["img"], b["rp"], a["rp"]), b["sd"] * 0.375
        scale = e["scale"]
        if e.get("relative"):                                           # relative to the gel: its own change between the scans is not a finding
            scale *= G.reference_scale(scale * a["img"], img_b, a["rp"], self.t["reference_region"])
        return scale * a["img"], img_b, a["rp"], math.hypot(scale * a["sd"], sd_b)

    def doses(self, causes):
        """What the experiments that were run measured for the remedies with a dose (``generator.measured_doses``)."""
        findings = [dict(f, causes=causes[f["id"]]) for f in self.t["findings"]]
        chain = {n: self.pair(n)[:3] + (False,) for n in ("dropout", "motion") if n in self.scans}
        ghost = C.BY_MECHANISM["motion"].label if "motion" in C.BY_MECHANISM else None
        scans = {n: self.complex(n) for n in ("complaint", "motion")} if "motion" in chain and any(ghost in f["causes"] for f in findings) else {}
        return G.measured_doses(self.t, findings, chain, scans) or None

    def change(self, name, finding):
        before, after, rp, sd = self.pair(name)
        return M.change(before, after, M.area_mask(rp, finding), self.exps[name]["shape_only"], noise_sd=sd)

    def flags(self, name):
        return {f["id"]: self.change(name, f) >= THRESHOLD for f in self.t["findings"]}


def _fat_label(t):
    return C.BY_MECHANISM["chemshift" if G.armed_on_sheet(t)["chemshift"] else "fatphase"].label


def _chain(sc, budget_fraction=1.0, every_probe=False):
    """Levels 1 and 2: the experiments the sheet arms, in order, each against the one before; the costly last one
    (twice the matrix) only when a finding is still unexplained, or always (``every_probe``: a fixed scan list).
    Returns (task, {finding: [causes]}, doses)."""
    cl = _Clinic(sc, budget_fraction)
    t = cl.t
    found = {f["id"]: set() for f in t["findings"]}
    for e in cl.exps.values():
        if e["name"] == "truncation" and not every_probe and all(found.values()):
            break
        if (e["baseline"] and e["baseline"] not in cl.scans) or not cl.run(e["name"]):
            break
        if e["baseline"]:
            for fid, changed in cl.flags(e["name"]).items():
                if changed:
                    found[fid].add(e["name"])
    return t, {i: C.decode(v, t["causes_vocabulary"]) for i, v in found.items()}, None


def _by_look(sc, order=None, budget_fraction=1.0, look=True):
    """Level 3. ``order`` None: the adaptive diagnosis (``generator.next_experiments``), each finding settled by the
    experiments of its look. Otherwise a fixed list of experiments, run in that order while the budget lasts, and the
    most probable causes given the looks and what was run (``generator.infer``). ``look`` False: the first image is
    not used; a finding carries whatever experiment changed it. Doses are measured where their experiment was run."""
    cl = _Clinic(sc, budget_fraction)
    t = cl.t
    ids = [f["id"] for f in t["findings"]]
    if not cl.run("complaint"):
        return t, {i: [] for i in ids}, None
    s = cl.scans["complaint"]
    classes = {f["id"]: G.look(s["img"], s["rp"], f, t["reference_region"])["cls"] for f in t["findings"]}
    plans, flags = G.class_plan(t), {}
    if order is None:
        while True:
            todo = G.next_experiments(plans, classes, flags)
            if not todo or not all(cl.run(name) for name in todo):
                break
            flags.update({name: cl.flags(name) for name in todo})
        causes = G.settle(plans, classes, flags, t["causes_vocabulary"])
    else:
        for name in order:
            if name in cl.exps and cl.exps[name]["baseline"] in cl.scans and cl.run(name):
                flags[name] = cl.flags(name)
        causes = G.infer(classes, flags, t["causes_vocabulary"]) if look else \
            {fid: C.decode({e for e, v in flags.items() if v[fid]}, t["causes_vocabulary"]) for fid in ids}
    return t, causes, cl.doses(causes)


def _several(sc):
    """Level 3, a fixed list of three scans with a fixed estimator. The experiments of the two looks do not disturb
    each other (half the slice changes nothing in a folded or a ghost copy beyond its scale; oversampling and the
    averages change nothing in a dark finding), so one scan carries half the slice, full phase oversampling and one
    average, and a third scan the in-phase echo:

        dark      the per-mm signal of the second scan rose: signal loss (and its ratio is the slice dose); else the
                  in-phase echo changed it: cancellation; else the object
        bright    the second scan left it as it was: the object; else the complex difference of the second scan and
                  the complaint shows the two copies of a pulsing source next to it (``generator.ghost_track``): a
                  ghost, with its displacement for the averages; else a folded copy"""
    cl = _Clinic(sc)
    t = cl.t
    p0, findings = t["current_protocol"], t["findings"]
    causes = {f["id"]: [] for f in findings}
    if not cl.run("complaint"):
        return t, causes, None
    s = cl.scans["complaint"]
    classes = {f["id"]: G.look(s["img"], s["rp"], f, t["reference_region"])["cls"] for f in findings}
    labels = {m: C.BY_MECHANISM[m].label for m in ("dropout", "fatphase", "wrap", "motion")}
    several = dict(p0, slice_mm=0.5 * p0["slice_mm"], phase_oversampling=round(min(1.0, 1.0 / p0["phase_fov"] - 1.0), 2), averages=1)
    doses, tracks = {}, {}
    if cl.run("several", several):
        b = cl.scans["several"]
        sd = math.hypot(0.5 * s["sd"], b["sd"])
        span = b["rp"]["fov_pe_mm"] * b["rp"]["lines_full"] / b["rp"]["matrix_pe"]
        ratios = []
        for f in findings:
            m = M.area_mask(s["rp"], f)
            if M.change(0.5 * s["img"], b["img"], m, noise_sd=sd) < THRESHOLD:
                continue
            if classes[f["id"]] == "dark":
                causes[f["id"]] = [labels["dropout"]]
                ratios.append(float(np.sum(m * b["img"]) / max(float(np.sum(m * 0.5 * s["img"])), 1e-30)))
            elif classes[f["id"]] == "bright":
                track = G.ghost_track(cl.complex("complaint"), 2.0 * cl.complex("several"), s["rp"], f, 1, span)
                causes[f["id"]] = [labels["motion" if track else "wrap"]]
                if track:
                    tracks[f["id"]] = dict(track, axis="x_mm" if s["rp"]["pe_dir"] == "x" else "y_mm")
        if ratios:
            doses["slice_ratio"] = max(ratios)
        if tracks:
            doses["ghosts"] = G.ghost_doses(findings, tracks)
    if "fatphase" in cl.exps and cl.run("fatphase"):
        for fid, changed in cl.flags("fatphase").items():
            if changed and classes[fid] == "dark" and not causes[fid]:
                causes[fid] = [labels["fatphase"]]
    return t, causes, doses or None


def _answer(sc, t, causes, doses=None):
    union = {c for v in causes.values() for c in v}
    protocol = G.assemble(t, union, lambda p: sc.plan(p), doses) or dict(t["current_protocol"])
    return dict(causes={i: list(v) for i, v in causes.items()}, protocol=protocol)


def reference(sc, budget_fraction=1.0):
    t = _task(sc)
    if not G.adaptive(t):
        return _answer(sc, *_chain(sc, budget_fraction))
    return _answer(sc, *_by_look(sc, budget_fraction=budget_fraction))


def half_budget(sc):
    return reference(sc, budget_fraction=0.5)


def open_loop(sc):
    if G.adaptive(_task(sc)):
        return _answer(sc, *(_several(sc) if OPEN_LOOP["list"] == "several" else _by_look(sc, order=OPEN_LOOP["order"])))
    return _answer(sc, *_chain(sc, every_probe=True))


def battery_within_budget(sc):
    if G.adaptive(_task(sc)):
        return _answer(sc, *_by_look(sc, order=OPEN_LOOP["order"], look=False))
    return _answer(sc, *_chain(sc, every_probe=True))


def prior(sc):
    t = _task(sc)
    lab = _fat_label(t)
    return dict(causes={f["id"]: [lab] for f in t["findings"]}, protocol=dict(t["current_protocol"]))


def _token_scan(sc):
    cl = _Clinic(sc)
    cl.run("complaint")
    return cl


DEFAULT_RECIPE = dict(cause="armed_fat", extent_mm=0.0)
EXTENT_GRID_MM = (0.0, 76.0, 80.0, 84.0, 88.0, 92.0, 96.0, 100.0, 104.0)


RANKS = ("id", "phase", "readout", "phase_distance", "centre_distance")


def _ranked(t, rank: str) -> list:
    """The ids of the findings in an order the card states: by id, by either coordinate, by the distance from the
    phase axis or from the isocentre."""
    pe, ro = ("x_mm", "y_mm") if t["current_protocol"]["pe_dir"] == "x" else ("y_mm", "x_mm")
    value = dict(id=lambda f: f["id"], phase=lambda f: f[pe], readout=lambda f: f[ro], phase_distance=lambda f: abs(f[pe]),
                 centre_distance=lambda f: math.hypot(f["y_mm"], f["x_mm"]))[rank]
    return [f["id"] for f in sorted(t["findings"], key=value)]


def _constant_causes(t, recipe) -> dict:
    """The answer of the best constant: one cause for every finding, except that a finding which an object of
    half-extent ``extent_mm`` would fold onto is called wrap-around (the one cause whose place the card constrains:
    a point at phase-encode coordinate p receives what lies at p -+ the phase field of view). A recipe with
    ``assign`` instead names the causes of the findings taken in the order ``rank`` (``_ranked``)."""
    if recipe.get("assign") is not None:
        return {fid: list(c) for fid, c in zip(_ranked(t, recipe["rank"]), recipe["assign"])}
    p0 = t["current_protocol"]
    lab = _fat_label(t) if recipe["cause"] == "armed_fat" else recipe["cause"]
    out = {}
    for f in t["findings"]:
        pe = abs(f["x_mm"] if p0["pe_dir"] == "x" else f["y_mm"])
        near = pe > p0["fov_mm"] * p0["phase_fov"] - recipe.get("extent_mm", 0.0)
        out[f["id"]] = [C.BY_MECHANISM["wrap"].label] if near else ([lab] if lab else [])
    return out


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _token_scan(sc)
        return _answer(sc, t, _constant_causes(t, recipe))
    return policy


def remedy_battery(sc):
    t = _task(sc)
    _token_scan(sc)
    lab = _fat_label(t)
    protocol = G.assemble(t, set(t["causes_vocabulary"]), lambda p: sc.plan(p)) or dict(t["current_protocol"])
    return dict(causes={f["id"]: [lab] for f in t["findings"]}, protocol=protocol)


def brute_force(sc):
    """Trial and error at full length: the complaint, then the complaint with one remedy at a time in catalogue
    order, for as long as the budget lasts; a finding is blamed on the first remedy that changed its mean by 10 %."""
    t = _task(sc)
    regions = t["findings"]

    def means(protocol):
        pl = sc.plan(protocol)
        if not pl["accept"] or pl["scan_s"] > sc.budget()["remaining_s"]:
            return None
        return {r["id"]: r["mean_magnitude"][0] for r in sc.scan(protocol, rois=regions)["roi_readouts"]}
    base = means(dict(t["current_protocol"]))
    causes = {f["id"]: [] for f in regions}
    if base:
        for lab in t["causes_vocabulary"]:
            trial = G.assemble(dict(t, requirements=dict(t["requirements"], max_scan_s=1e9)), {lab}, lambda p: sc.plan(p))
            got = means(trial) if trial else None
            if not got:
                continue
            for i in causes:
                if not causes[i] and base[i] and got[i] is not None and abs(got[i] / base[i] - 1.0) > 0.10:
                    causes[i] = [lab]
    return _answer(sc, t, causes)


def wrong_units(sc):
    a = reference(sc)
    return dict(a, protocol=dict(a["protocol"], te_ms=a["protocol"]["te_ms"] / 1000.0))


def first_look(sc):
    """One scan of the complaint; each finding is classified from its own signal and texture with the reference's
    knowledge of what the sheet arms. No experiment separates the look-alikes."""
    t = _task(sc)
    if G.adaptive(t):
        return _answer(sc, *_by_look(sc, order=()))
    cl = _token_scan(sc)
    if "complaint" not in cl.scans:
        return prior(sc)
    s = cl.scans["complaint"]
    causes = {}
    for f in t["findings"]:
        lk = G.look(s["img"], s["rp"], f, t["reference_region"])
        if lk["level"] < 0.45:
            causes[f["id"]] = [C.BY_MECHANISM["truncation"].label]       # inside a dark material: ripple from its edge
        elif lk["texture"] > 0.25:
            causes[f["id"]] = [_fat_label(t)]                             # a strong edge inside the finding
        elif lk["level"] > 1.5:
            causes[f["id"]] = [C.BY_MECHANISM["wrap"].label]              # brighter than anything homogeneous nearby
        else:
            causes[f["id"]] = []
    return _answer(sc, t, causes)


def told_cause_fixed_dose(sc):
    """The hidden causes, then the first-line remedy of each at the catalogue's dose, whatever the examination locks."""
    t = _task(sc)
    _token_scan(sc)
    causes = sc.env.episode_config()["hidden"]["causes"]
    free = dict(t, requirements=dict(t["requirements"], locked={}))
    protocol = G.assemble(free, {c for v in causes.values() for c in v}, lambda p: sc.plan(p)) or dict(t["current_protocol"])
    return dict(causes={i: list(v) for i, v in causes.items()}, protocol=protocol)


def told_cause_tuned_dose(sc):
    """The hidden causes, then the first-line remedy of each with the fixed doses of ``TUNED_DOSE`` (the catalogue's
    dose where a remedy has none of them), within what the examination locks."""
    t = _task(sc)
    _token_scan(sc)
    causes = sc.env.episode_config()["hidden"]["causes"]
    doses = dict(slice_mm=round(TUNED_DOSE["slice_fraction"] * t["current_protocol"]["slice_mm"], 1), averages=TUNED_DOSE["averages"])
    return _answer(sc, t, {i: list(v) for i, v in causes.items()}, doses)


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), remedy_battery=remedy_battery,
                brute_force=brute_force, wrong_units=wrong_units, first_look=first_look, told_cause_fixed_dose=told_cause_fixed_dose,
                half_budget=half_budget, battery_within_budget=battery_within_budget, told_cause_tuned_dose=told_cause_tuned_dose,
                open_loop=open_loop)


def tune(cfgs):
    """The strongest answer that uses only the card, on other worlds: one cause for every finding (a named cause,
    none, or "the fat mechanism the sheet arms"), with wrap-around for the findings that an object of a tuned
    half-extent would fold onto; at a level diagnosed by look also every assignment of a cause or none to the
    findings in each order of ``RANKS``. Chosen by the number of worlds on which every finding is diagnosed, ties by
    findings."""
    best = None

    def judge(recipe):
        nonlocal best
        whole = part = 0
        for c in cfgs:
            guess = _constant_causes(c["task"], recipe)
            hits = [sorted(guess[i]) == sorted(v) for i, v in c["hidden"]["causes"].items()]
            whole, part = whole + all(hits), part + sum(hits)
        if best is None or (whole, part) > best[:2]:
            best = (whole, part, recipe)
    for cand in ["armed_fat", "", *cfgs[0]["task"]["causes_vocabulary"]]:
        for extent in EXTENT_GRID_MM:
            judge(dict(cause=cand, extent_mm=extent))
    if G.adaptive(cfgs[0]["task"]) and len({len(c["task"]["findings"]) for c in cfgs}) == 1:
        options = [[]] + [[lab] for lab in cfgs[0]["task"]["causes_vocabulary"]]
        for rank in RANKS:
            for assign in itertools.product(options, repeat=len(cfgs[0]["task"]["findings"])):
                judge(dict(rank=rank, assign=[list(a) for a in assign]))
    return dict(best[2], tuned_worlds_all_findings=best[0], tuned_findings=best[1])


def tune_open_loop(cfgs) -> dict:
    """The best open-loop control of an adaptive level on other worlds, from their hidden truth (no scan is run):
    for every order of the experiments, which of them the budget pays for on each world, what they would show, and
    the probability that the most probable causes given the look and those experiments are the true ones (one over
    the number of cause sets that agree, ``generator.count_consistent``). Returns the order with the most worlds
    diagnosed in expectation, and that number."""
    names = [n for n in cfgs[0]["hidden"]["experiment_s"] if n != "complaint"]
    best = None
    for order in itertools.permutations(names):
        total = 0.0
        for c in cfgs:
            h = c["hidden"]
            cost, used, run = h["experiment_s"], h["experiment_s"]["complaint"], []
            for name in order:
                if name in cost and used + cost[name] <= c["budget_s"] + 1e-9:
                    used += cost[name]
                    run.append(name)
            looks = set(h["look"].values())
            total += 1.0 / G.count_consistent(next(iter(looks)), h["causes"], run) if len(looks) == 1 else 0.0
        if best is None or total > best[0] + 1e-9:
            best = (total, order)
    # the list of ``_several``: on a world whose budget holds its three scans every cause set of the level shows
    # another pattern (see the table in its docstring); counted as diagnosed there, as not diagnosed elsewhere
    fits = 0
    for c in cfgs:
        p0, cost = c["task"]["current_protocol"], c["hidden"]["experiment_s"]
        several = dict(p0, slice_mm=0.5 * p0["slice_mm"], phase_oversampling=round(min(1.0, 1.0 / p0["phase_fov"] - 1.0), 2), averages=1)
        fits += cost["complaint"] + console.plan(several, G.exam_card(c["world_spec"]))["scan_s"] + cost.get("fatphase", 1e9) <= c["budget_s"] + 1e-9
    return dict(order=list(best[1]), expected_worlds_all_findings=round(best[0], 2), several_worlds=fits, worlds=len(cfgs),
                list="several" if fits > best[0] else "single")


SLICE_FRACTIONS = tuple(round(0.15 + 0.01 * k, 2) for k in range(56))    # candidates of the tuned slice dose: 0.15 ... 0.70 of the complaint slice


def tune_doses(cfgs) -> dict:
    """The best fixed doses on other worlds, from their hidden remedies (no scan is run). Slice: a fraction f of the
    complaint slice passes on a world with signal loss when it is thin enough to be clean and thick enough for the
    SNR floor: the loss the verified remedy leaves (``catalogue.DROPOUT_RESIDUAL`` at its thickness d) grows with
    the thickness as the dephasing factor of the console's pulse does, and clean is ``generator.CLEAN_MECH``; the SNR
    is proportional to the thickness and needs the stated share of the remedy's. Averages: the number the verified
    remedies of the worlds with a ghost use most."""
    from ...scanner import rf
    factor = lambda c: float(rf.dephasing_factor(np.array([c]), 4.0)[0])   # noqa: E731  the console's normal pulse
    thicker = C._cycles(1.0 - G.CLEAN_MECH, factor, hi=1.6) / C._cycles(1.0 - C.DROPOUT_RESIDUAL, factor, hi=1.6)
    loss, ghost = C.BY_MECHANISM["dropout"].label, C.BY_MECHANISM["motion"].label
    lossy = [c for c in cfgs if any(loss in v for v in c["hidden"]["causes"].values())]
    moved = [c for c in cfgs if any(ghost in v for v in c["hidden"]["causes"].values())]
    best = None
    for f in SLICE_FRACTIONS:
        n = 0
        for c in lossy:
            d, full = c["hidden"]["remedy"]["slice_mm"], c["task"]["current_protocol"]["slice_mm"]
            n += c["task"]["requirements"].get("snr_share", 0.0) * d <= round(f * full, 1) <= thicker * d
        if best is None or n > best[0]:
            best = (n, f)
    averages = [c["hidden"]["remedy"]["averages"] for c in moved]
    return dict(slice_fraction=best[1] if lossy else TUNED_DOSE["slice_fraction"], slice_worlds=best[0] if lossy else 0, worlds_with_loss=len(lossy),
                averages=max(set(averages), key=averages.count) if averages else TUNED_DOSE["averages"],
                averages_worlds=averages.count(max(set(averages), key=averages.count)) if averages else 0, worlds_with_ghost=len(moved))
