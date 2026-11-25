"""Control policies of ``Scanner/ProtocolToSpec``. Every policy plays through the agent's interface only.

All policies that measure share one signal model and one solver; they differ only in WHICH scans they make.

Must pass
  reference         scans the standard protocol; reads which contrast requirement it misses; makes the one scan that
                    requirement needs (another echo time when a T2 contrast fails, otherwise another inversion time);
                    fits, per region, the signed inversion recovery (amplitude and T1) or the decay with the echo time;
                    delivers the examination with the largest predicted margin: per image the inversion and echo time
                    for every number of averages, then the split of the averages the scan-time limit pays for
Must fail
  prior             no scan; the standard protocol
  wrong_units       the reference's answer with the times in seconds
  fixed_recipe      one scan, unused; then the best answer that ignores the object, tuned on other worlds (``tune``):
                    a constant inversion time and echo time per image, with constant numbers of averages or with the
                    averages the stated time limit pays for split by a fixed rule (a function of the task card)
  card_rule         one scan, unused; a typical inversion time and echo time, the averages of the time limit split evenly
  brute_force       no model: scans the standard protocol and one guessed alternative, delivers per image whichever
                    of the two looked better on its own read-outs
Ablations: the open-loop family (fixed lists of scans with the reference's solver) and the reference without a scan
  always_ti         second scan: always another inversion time (the policy that suffices at level 1)
  always_te         second scan: always another echo time (the policy that suffices at level 2)
  open_loop_card    second scan chosen by a rule on the task card (the scan-time limit), tuned on ``CARD_WORLDS`` worlds
                    of the tuning split; where the card says nothing it is the more frequent of the two
  open_loop_all     THREE scans that fit the budget because they are cheaper: the standard protocol, another inversion
                    time and another echo time on a coarser grid with fewer lines; both curves fitted
  open_loop_short   the same three scans at the scout resolution with half the repetition time; the fitted recovery is
                    carried to the repetition time of the images with the signal model
  trial_only        no second scan: solves from the scan of the standard protocol with typical values
Reported
  characterise_all  measure everything at the reference's resolution and repetition time: three scans, more than the
                    budget pays for (``battery_cost``), so the third is refused
``OPEN_LOOP`` names the fixed lists; the kind of task (``task.task_kind``) is read from the best of them.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from ...scanner import kspace
from ...scanner import signal as S
from . import oracle, task

REFERENCE = "reference"
MUST_FAIL = ("prior", "wrong_units", "fixed_recipe", "card_rule", "brute_force")
OPEN_LOOP = task.OPEN_LOOP
# The open-loop lists are not ablations of the reference: they decide the word on the card (measurement or decision).
# On 300 gate instances the best of them passes within 25 points of the reference at every level, so no level claims to
# be a decision task and none of them is required to lose (empty tuple = reported with its paired gap).
ABLATIONS = {"always_ti": (2, 3), "always_te": (1, 3), "open_loop_card": (), "open_loop_all": (), "open_loop_short": (),
             "trial_only": (1, 2, 3)}
REPORTED = ("characterise_all",)
LADDER = {1: "always_ti", 2: "always_te", 3: "reference"}                # the policy that suffices at each level
DESCRIPTIONS = dict(
    reference="scan of the standard protocol, then the one scan the failing contrast needs (another inversion time or "
              "another echo time), model fit per region, examination with the largest predicted margin",
    prior="no scan; the standard protocol",
    wrong_units="the reference's inversion and echo times in seconds",
    fixed_recipe="one unused scan, then the best constant protocols (constant averages, or the averages of the time limit "
                 "split by a fixed rule), tuned on other worlds",
    card_rule="one unused scan, then a typical inversion time and echo time, the averages of the time limit split evenly",
    brute_force="scans the standard protocol and one guessed alternative, delivers per image the one that looked better",
    always_ti="fixed list: the standard protocol and another inversion time; the reference's solver",
    always_te="fixed list: the standard protocol and another echo time; the reference's solver",
    open_loop_card="fixed list chosen by a rule on the task card tuned on the tuning split; the reference's solver",
    open_loop_all="fixed list of three cheaper scans (coarser, fewer lines): standard, another inversion time, another echo time",
    open_loop_short="fixed list of three scans at half the repetition time: standard, another inversion time, another echo time",
    trial_only="the reference without a second scan (typical values instead of a measurement)",
    characterise_all="standard protocol + another inversion time + another echo time at the reference's resolution: three "
                     "scans, more than the budget pays for",
)

RAYLEIGH_MEAN = math.sqrt(math.pi / 2.0)                                # mean of the magnitude of unit complex noise
SCOUT = dict(fov_mm=400.0, matrix=96, phase_fov=0.5, partial_fourier=0.75)   # the image's sequence, TR, slice and bandwidth at 4.2 mm
COARSE = dict(fov_mm=400.0, matrix=64, phase_fov=0.5, partial_fourier=0.65)  # 6.3 mm, 21 lines: three of them fit the budget
SHORT_TR_MS = 200.0                                                     # the shortest repetition time of the sequence: three such scans fit
BACKGROUND = [dict(id=f"bg{k}", y_mm=y, x_mm=x, radius_mm=22.0) for k, (y, x) in
              enumerate((y, x) for x in (-170.0, 170.0) for y in (-66.0, -22.0, 22.0, 66.0))]
SCOUT_TI_MS, SCOUT_TE_MS = 160.0, 60.0                                  # the second scan: another inversion time / another echo time
KAPPA = 0.95                                                            # assumed efficiency of the inversion pulse
T1_GRID = np.exp(np.linspace(math.log(20.0), math.log(3000.0), 600))
TI_SEARCH = 20.0 * 1.01 ** np.arange(232)                               # 20 .. 199 ms
TE_SEARCH = 12.0 * 1.025 ** np.arange(96)                               # 12 .. 125 ms
TI_GUESS_MS, TE_GUESS_MS = 55.0, 60.0                                   # typical values where nothing was measured
REACH = 1.5                         # the echo time is searched up to this factor beyond the first one predicted to pass


# ------------------------------------------------------------------------------------------------ measurement
def _task(sc):
    return sc.info()["task"]


def noise_factor(rp) -> float:
    """SD of the noise in one pixel per unit receiver noise density, from the realised protocol (the SNR laws)."""
    return rp["matrix"] * math.sqrt(rp["lines"] * rp["bandwidth_hz_per_px"] / (rp["averages"] * rp.get("partitions", 1))) \
        / (rp["fov_mm"] * rp["fov_pe_mm"] * rp["lines_full"] / rp["matrix_pe"])


def _protocol(t, geometry=None, **fields):
    s = t["scan"]
    return dict({k: s[k] for k in ("sequence", "tr_ms", "slice_mm", "bandwidth_hz_per_px")}, **(geometry or SCOUT),
                **dict(t["standard_protocol"], averages=1, **fields))


def _scan(sc, t, geometry=None, **fields):
    """One low-resolution scan: complex mean of every region, and the noise density from the background."""
    r = sc.scan(_protocol(t, geometry, **fields), t["regions"] + BACKGROUND)
    rows = {row["id"]: row for row in r["roi_readouts"]}
    z = {q["id"]: rows[q["id"]]["complex_mean_magnitude"][0] * np.exp(1j * rows[q["id"]]["complex_mean_phase_rad"][0]) for q in t["regions"]}
    n = np.array([rows[b["id"]]["n_pixels"] for b in BACKGROUND], float)
    sd = float(np.sum(n * [rows[b["id"]]["mean_magnitude"][0] for b in BACKGROUND]) / n.sum()) / RAYLEIGH_MEAN
    return dict(ti=r["realised"]["ti_ms"], te=r["realised"]["te_ms"], z=z, sigma0=sd / noise_factor(r["realised"]),
                tbw=r["realised"]["rf_tbw"], tr=r["realised"]["tr_ms"])


@functools.lru_cache(maxsize=4)
def _profile(tbw: float):
    p = kspace.slice_profile(np.array([math.pi / 2.0]), tbw, "hann", refocus=True)
    return p["theta"][0], p["w"][0], p["echo"][0]


def ir_shape(ti_ms, te_ms, t1_ms, tr_ms, tbw=4.0, kappa=KAPPA):
    """Signed signal of the inversion-recovery spin echo per unit magnetisation, without the T2 decay: the steady state
    of [excitation, TE/2, refocusing, TR - TI - TE/2, inversion, TI] through the nominal slice profile (datasheet)."""
    ti, te, t1 = (np.asarray(v, float)[..., None] for v in np.broadcast_arrays(ti_ms, te_ms, t1_ms))
    th, w, echo = _profile(float(tbw))
    mz = S.mz_steady([("rf", np.cos(th)), ("wait", te / 2.0), ("rf", 1.0 - 2.0 * echo), ("wait", tr_ms - ti - te / 2.0),
                      ("rf", -kappa), ("wait", ti)], t1)
    return (w * mz * np.sin(th) * echo).sum(-1)


def _fit_ti(a, b):
    """Per region: amplitude and T1 of the signed recovery through two scans at different inversion times (same TE).
    The signal rises with the inversion time: the relative phase of the two read-outs and their sizes give the signs."""
    lo, hi = (a, b) if a["ti"] < b["ti"] else (b, a)
    out = {}
    for i in lo["z"]:
        y_lo, y_hi = abs(lo["z"][i]), abs(hi["z"][i])
        if (lo["z"][i] * np.conj(hi["z"][i])).real < 0:
            y_lo = -y_lo
        elif y_lo > y_hi:
            y_lo, y_hi = -y_lo, -y_hi
        f = np.array([ir_shape(s["ti"], s["te"], T1_GRID, s["tr"], s["tbw"]) for s in (lo, hi)])     # [scan, T1]
        y = np.array([y_lo, y_hi])[:, None]
        amp = np.maximum((f * y).sum(0) / (f * f).sum(0), 0.0)
        k = int(np.argmin(((amp * f - y) ** 2).sum(0)))
        out[i] = dict(amp=float(amp[k]), t1=float(T1_GRID[k]))
    return out


def _fit_te(a, b):
    """Per region: the rate at which the signal falls with the echo time, from two scans at the same inversion time."""
    return {i: math.log(max(abs(a["z"][i]), 1e-12) / max(abs(b["z"][i]), 1e-12)) / (b["te"] - a["te"]) for i in a["z"]}


# ------------------------------------------------------------------------------------------------ the solver
def _facts(sc, t):
    """Console facts of the images: noise factor of one average, cycles per average, the time of an image without
    averages (its adjustments as a scan of its own and the dummy cycle), shortest echo time."""
    p = sc.plan(dict(t["scan"], **t["standard_protocol"]))
    cost = sc.info()["cost"]
    tr = p["realised"]["tr_ms"]
    return dict(noise=noise_factor(dict(p["realised"], averages=1)), lines=p["realised"]["lines"], tr=tr,
                base=cost["overhead_s_per_scan"] + tr * cost["dummy_cycles_se_ir"] / 1000.0, te_min=p["timing"]["te_min_ms"])


def exam_s(facts, images: int, total_averages):
    """Scan time of the examination: every image a scan of its own, ``total_averages`` in all."""
    return images * facts["base"] + facts["tr"] * facts["lines"] * np.asarray(total_averages) / 1000.0


def averages_total(facts, t) -> int:
    """The averages the stated scan-time limit pays for, both images together (at least one each)."""
    k = len(t["images"])
    n = math.floor((t["scan_time_max_s"] - k * facts["base"]) * 1000.0 / (facts["tr"] * facts["lines"]) + 1e-9)
    return int(min(max(n, k), 64 * k))


def split(facts, t, rule) -> dict:
    """Averages per image by a rule that ignores the object: ("split", d) gives the first image half of what the time
    limit pays for plus d and the others the rest; ("const", a, b, ...) are constants."""
    names, total = list(t["images"]), averages_total(facts, t)
    if rule[0] == "const":
        return {k: int(n) for k, n in zip(names, rule[1:])}
    first = int(min(max(total // 2 + rule[1], 1), total - 1))
    return {names[0]: first, names[1]: total - first}


def _solve(sc, t, trial, sigma0, ti_fit=None, te_fit=None):
    """The examination with the largest predicted margin over the fields that were measured (the others stay at the
    standard protocol). Signal of a region: its recovery curve along the inversion time (``ti_fit``, evaluated at the
    repetition time of the images) or its value in ``trial``, times its decay along the echo time (``te_fit``). The T1
    contrast is predicted only with a recovery curve: the read-out of a region near its null in one low-resolution scan
    carries signal of its surroundings, and without a curve the inversion time stays where it is. Per image the best
    inversion and echo time is found for every number of averages; the averages the scan-time limit pays for are then
    shared out so that the smaller of the two margins is largest. Where the requirement that fails belongs to a field
    that was not measured, that field is set to a typical value. With both curves measured (the three-scan lists) the
    solver moves the field whose contrast the standard protocol misses, as the reference does. The echo time is not moved further than ``REACH`` times
    the first one predicted to pass: far beyond it the decay of a region near its null is no longer the fitted exponential."""
    facts, std = _facts(sc, t), t["standard_protocol"]
    curve = {i: (lambda x, f=f: np.abs(f["amp"] * ir_shape(x, std["te_ms"], f["t1"], facts["tr"], trial["tbw"]))) for i, f in (ti_fit or {}).items()}
    # signals at the standard protocol: the first scan, or (a first scan at another repetition time) the fitted curve
    trial = dict(trial, z={i: (z if trial["tr"] == facts["tr"] else complex(curve[i](std["ti_ms"]))) for i, z in trial["z"].items()})
    if ti_fit and te_fit:                                                # both curves measured: move the field whose contrast fails
        ti_fit, te_fit = (None, te_fit) if misses_t2_contrast(t, trial) else (ti_fit, None)
    ti = TI_SEARCH if ti_fit else np.array([std["ti_ms"]])
    te = TE_SEARCH[TE_SEARCH >= facts["te_min"]] if te_fit else np.array([std["te_ms"]])
    sig = {}
    for i, z in trial["z"].items():
        along_ti = curve[i](ti) if ti_fit else np.array([abs(z)])
        along_te = np.exp(-te_fit[i] * (te - std["te_ms"])) if te_fit else np.ones(1)
        sig[i] = np.maximum(along_te[:, None] * along_ti[None, :], 1e-12)                      # [te, ti]
    total, picks, curves = averages_total(facts, t), {}, {}
    for name, q in t["images"].items():
        m = np.log(sig[q["t2_contrast"]["brighter"]] / sig[q["t2_contrast"]["darker"]] / q["t2_contrast"]["min_ratio"])
        if ti_fit:                                                       # the T1 contrast is predicted only from a recovery curve
            m = np.minimum(m, np.log(sig[q["t1_contrast"]["brighter"]] / sig[q["t1_contrast"]["darker"]] / q["t1_contrast"]["min_ratio"]))
        snr1 = np.log(sig[q["snr"]["region"]] / (sigma0 * facts["noise"]) / q["snr"]["min"])
        m[te[:, None] + ti[None, :] > facts["tr"] - 15.0] = -np.inf
        picks[name], curves[name] = [], []
        for n in range(1, total):
            mn = np.minimum(m, snr1 + 0.5 * math.log(n))
            passing = np.flatnonzero((mn >= 0).any(1))
            if len(passing):                                             # no further from the standard echo time than the requirement asks
                mn[te > REACH * te[passing[0]]] = -np.inf
            k, j = np.unravel_index(int(np.argmax(mn)), mn.shape)
            picks[name].append((float(ti[j]), float(te[k])))
            curves[name].append(float(mn[k, j]))
    shares, _ = oracle.allocate(curves, lambda n: np.where(exam_s(facts, len(curves), n) <= t["scan_time_max_s"], np.inf, -np.inf))
    out = {}
    for name, q in t["images"].items():
        n = shares[name]
        best = dict(ti_ms=picks[name][n - 1][0], te_ms=picks[name][n - 1][1], averages=n)
        if not ti_fit and not misses_t2_contrast(dict(images={name: q}), trial):
            best["ti_ms"] = TI_GUESS_MS                                  # the T2 contrast holds at the standard protocol: the T1 contrast is what fails
        elif curves[name][n - 1] < 0 and not te_fit:
            best["te_ms"] = TE_GUESS_MS
        out[name] = dict(ti_ms=round(best["ti_ms"], 2), te_ms=round(best["te_ms"], 2), averages=best["averages"])
    return out


def misses_t2_contrast(t, trial) -> bool:
    """Whether the scan of the standard protocol shows a T2 contrast below its requirement in some image. (The T1
    contrast is not read off that scan: see ``_solve``. One of the two fails at the standard protocol.)"""
    z = trial["z"]
    return any(abs(z[q["t2_contrast"]["brighter"]]) < q["t2_contrast"]["min_ratio"] * abs(z[q["t2_contrast"]["darker"]])
               for q in t["images"].values())


def _play(sc, second, geometry=None, tr_ms=None):
    """Scan the standard protocol, then the scans ``second(task, trial)`` names ("ti", "te"), then solve. ``geometry``
    and ``tr_ms``: every scan of the list on another grid or at another repetition time than the reference's."""
    t = _task(sc)
    more = dict(tr_ms=tr_ms) if tr_ms else {}
    trial = _scan(sc, t, geometry, **more)
    scans, fits = [trial], {}
    for kind in second(t, trial):
        other = _scan(sc, t, geometry, **more, **(dict(ti_ms=SCOUT_TI_MS) if kind == "ti" else dict(te_ms=SCOUT_TE_MS)))
        scans.append(other)
        fits[kind] = _fit_ti(trial, other) if kind == "ti" else _fit_te(trial, other)
    return _solve(sc, t, trial, float(np.mean([s["sigma0"] for s in scans])), fits.get("ti"), fits.get("te"))


# ------------------------------------------------------------------------------------------------ the policies
def reference(sc):
    return _play(sc, lambda t, trial: ("te",) if misses_t2_contrast(t, trial) else ("ti",))


def always_ti(sc):
    return _play(sc, lambda t, trial: ("ti",))


def always_te(sc):
    return _play(sc, lambda t, trial: ("te",))


def trial_only(sc):
    return _play(sc, lambda t, trial: ())


def characterise_all(sc):
    return _play(sc, lambda t, trial: ("ti", "te"))


def open_loop_all(sc):
    return _play(sc, lambda t, trial: ("ti", "te"), geometry=COARSE)


def open_loop_short(sc):
    return _play(sc, lambda t, trial: ("ti", "te"), tr_ms=SHORT_TR_MS)


def open_loop_card(sc):
    return _play(sc, lambda t, trial: (card_list(t["level"], t["scan_time_max_s"]),))


def wrong_units(sc):
    return {k: dict(v, ti_ms=v["ti_ms"] / 1000.0, te_ms=v["te_ms"] / 1000.0) for k, v in reference(sc).items()}


def prior(sc):
    t = _task(sc)
    return {name: dict(t["standard_protocol"]) for name in t["images"]}


def _constant(sc, t, times: dict, rule):
    shares = split(_facts(sc, t), t, rule)
    return {name: dict(ti_ms=float(times[name]["ti_ms"]), te_ms=float(times[name]["te_ms"]), averages=shares[name]) for name in t["images"]}


def card_rule(sc):
    t = _task(sc)
    _scan(sc, t)
    return _constant(sc, t, {name: dict(ti_ms=TI_GUESS_MS, te_ms=TE_GUESS_MS) for name in t["images"]}, ("split", 0))


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _scan(sc, t)
        return _constant(sc, t, recipe, tuple(recipe["averages"]))
    return policy


DEFAULT_RECIPE = dict({name: dict(ti_ms=TI_GUESS_MS, te_ms=12.0) for name in task.IMAGES}, averages=("split", 0))


def brute_force(sc):
    """No model: the standard protocol and one guessed alternative are scanned at low resolution; per image, the scan on
    which more of the two contrast requirements held is delivered; the averages of the time limit are split evenly."""
    t = _task(sc)
    scans = [_scan(sc, t), _scan(sc, t, ti_ms=TI_GUESS_MS, te_ms=TE_GUESS_MS)]
    shares, out = split(_facts(sc, t), t, ("split", 0)), {}
    for name, q in t["images"].items():
        held = [sum(abs(s["z"][q[k]["brighter"]]) >= q[k]["min_ratio"] * abs(s["z"][q[k]["darker"]]) for k in ("t1_contrast", "t2_contrast"))
                for s in scans]
        s = scans[1] if held[1] >= held[0] else scans[0]
        out[name] = dict(ti_ms=s["ti"], te_ms=s["te"], averages=shares[name])
    return out


def policies(recipe=None):
    return dict(reference=reference, prior=prior, wrong_units=wrong_units, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE),
                card_rule=card_rule, brute_force=brute_force, always_ti=always_ti, always_te=always_te, trial_only=trial_only,
                open_loop_card=open_loop_card, open_loop_all=open_loop_all, open_loop_short=open_loop_short,
                characterise_all=characterise_all)


def battery_cost(sc) -> dict:
    """What ``characterise_all`` would cost at the start of an episode, from the console's plan and its stated rule: the
    first scan with its adjustments, the two repeats without; their sum, the budget and the ratio. Nothing is scanned."""
    t, cost = _task(sc), sc.info()["cost"]
    first = sc.plan(_protocol(t))["scan_s"]
    repeat = first - cost["overhead_s_per_scan"] + cost.get("overhead_s_repeat_scan", cost["overhead_s_per_scan"])
    scans = dict(standard=first, inversion_time=repeat, echo_time=repeat)
    budget = sc.budget()["budget_s"]
    return dict(scans=scans, total_s=round(sum(scans.values()), 3), budget_s=budget, ratio=round(sum(scans.values()) / budget, 3))


# ------------------------------------------------------------------------------------------------ tuning on other worlds
CARD_WORLDS = 48            # worlds of the tuning split on which the rule of ``open_loop_card`` is tuned
CANDIDATES = dict(ti_ms=np.round(np.exp(np.linspace(math.log(30.0), math.log(170.0), 18)), 1),
                  te_ms=np.array([12.0, 20.0, 27.0, 35.0, 45.0, 57.0, 72.0, 90.0]))
RULES = [("split", d) for d in (-1, 0, 1)] + [("const", a, b) for a in range(1, 6) for b in range(1, 6)]


def moving_field(cfg: dict, model=None) -> str:
    """Which field has to move on an instance, read off the hidden world: "ti" when protocols meet the image
    requirements at the standard echo time, "te" when at the standard inversion time. Verifier side; for tests."""
    from ...scanner import world as W
    model = model or oracle.Model(W.make_world(cfg["seed"], cfg["world_spec"]))
    w = model.windows(next(iter(cfg["task"]["images"].values())), int(oracle.AVERAGES[-1]))
    return "ti" if w["ti_ms"] else "te"


@functools.lru_cache(maxsize=8)
def card_table(level: int, n_worlds: int | None = None) -> dict:
    """The rule of ``open_loop_card``, tuned on ``CARD_WORLDS`` instances of the tuning split: the more frequent of the
    two second scans, and the threshold on the stated scan-time limit (with the scan on either side of it) that names
    the field that has to move most often; the threshold is used only when it beats the more frequent one."""
    import mrgym
    rows = []
    for i in range(n_worlds or CARD_WORLDS):
        seed = mrgym.env.derive_seed("Scanner/ProtocolToSpec-v0", level, "tune", f"card:{i}", environ={})
        inst = task._instance(seed, level)
        rows.append((inst[4], inst[5]))
    limit, kind = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    major = "ti" if 2 * int((kind == "ti").sum()) >= len(kind) else "te"
    best = dict(threshold=None, below=major, above=major, accuracy=float((kind == major).mean()), majority=major, worlds=len(rows))
    for th in np.unique(limit)[1:]:
        for below, above in (("ti", "te"), ("te", "ti")):
            acc = float(np.mean(np.where(limit < th, below, above) == kind))
            if acc > best["accuracy"] + 1e-9:
                best.update(threshold=float(th), below=below, above=above, accuracy=acc)
    return best


def card_list(level: int, scan_time_max_s: float) -> str:
    rule = card_table(level)
    return rule["majority"] if rule["threshold"] is None else rule["below"] if scan_time_max_s < rule["threshold"] else rule["above"]


def _candidate(index: int) -> dict:
    i, j = np.unravel_index(int(index), (len(CANDIDATES["te_ms"]), len(CANDIDATES["ti_ms"])))
    return dict(ti_ms=float(CANDIDATES["ti_ms"][j]), te_ms=float(CANDIDATES["te_ms"][i]))


def tune(cfgs):
    """The strongest answer that ignores the object, on other worlds: per image one inversion time and one echo time,
    with the averages by one of ``RULES`` (constants, or the averages the stated time limit pays for split by a fixed
    rule); the combination that passes on the largest number of the tuning worlds, by the task's rule on the signal
    model of those worlds (ties: the larger sum of the single-image pass rates)."""
    from ...scanner import world as W
    j = [int(np.argmin(np.abs(oracle.TI_GRID - v))) for v in CANDIDATES["ti_ms"]]
    i = [int(np.argmin(np.abs(oracle.TE_GRID - v))) for v in CANDIDATES["te_ms"]]
    worlds = []
    for c in cfgs:
        model = oracle.Model(W.make_world(c["seed"], c["world_spec"]))
        t = c["task"]
        facts = dict(base=float(model.scan_s(0)), tr=oracle.SCAN["tr_ms"], lines=model.lines)
        worlds.append((t, facts, [model.margin(q)[:, i][:, :, j] for q in t["images"].values()]))      # [averages, te, ti]
    best = None
    for rule in RULES:
        tables = [[], []]
        for t, facts, margins in worlds:
            shares = list(split(facts, t, rule).values())
            fits = exam_s(facts, len(shares), sum(shares)) <= t["scan_time_max_s"]
            for k in range(2):
                tables[k].append((margins[k][shares[k] - 1] >= 0.0).ravel() & bool(fits))
        a, b = (np.array(v, np.float32).T for v in tables)
        joint = a @ b.T + 1e-3 * (a.sum(1)[:, None] + b.sum(1)[None, :]) / a.shape[1]
        p, q = np.unravel_index(int(np.argmax(joint)), joint.shape)
        if best is None or joint[p, q] > best[0]:
            best = (float(joint[p, q]), rule, int(p), int(q), float((a[p] * b[q]).mean()), [float(a[p].mean()), float(b[q].mean())])
    return dict(A=_candidate(best[2]), B=_candidate(best[3]), averages=list(best[1]), tuned_on_worlds=len(worlds),
                tuned_joint_pass_rate=round(best[4], 4), tuned_single_pass_rate=[round(v, 4) for v in best[5]])


def blind_answers(t: dict) -> dict:
    """Answers that use the task card only: one inversion time and one echo time for both images, the averages the stated
    time limit pays for split evenly. {label: answer}; the same labels on every instance."""
    s = t["scan"]
    facts = dict(base=t["adjustments"]["new_scan_s"] + s["tr_ms"] / 1000.0, tr=s["tr_ms"], lines=round(s["matrix"] * s.get("phase_fov", 1.0)))
    shares = split(facts, t, ("split", 0))
    return {f"ti_ms={ti:g},te_ms={te:g}": {name: dict(ti_ms=float(ti), te_ms=float(te), averages=shares[name]) for name in t["images"]}
            for ti in (35.0, 50.0, 70.0, 100.0, 140.0) for te in (12.0, 30.0, 50.0, 75.0)}
