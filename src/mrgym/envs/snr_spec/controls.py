"""Control policies of ``Scanner/SNRSpec``. Every policy plays through the agent's interface only.

Must pass
  reference         measures the noise (background of a short scan on a wide field of view) and, on low-resolution
                    spin echoes, each region's signal at four repetition times and two echo times; fits the recovery
                    (amplitude, T1 and an effective flip angle: the slice profile and the transmit error bend the
                    curve) and T2; scales signal and noise to each image with the SNR laws; searches repetition time x
                    averages x bandwidth for the shortest scan that meets the requirement with a small margin. Its
                    long scout is placed after a first look at T1.
Must fail
  prior             no scan; one typical protocol for both images
  wrong_units       the reference with the noise taken as the SD of the background MAGNITUDE (0.655 of the channel SD)
  fixed_recipe      one token scan, then the best pair of constant protocols, tuned on other worlds (``tune``)
  spec_only         one token scan, then the best protocol that is a function of what the task states (matrix and
                    required SNR of each image, the level), tuned on other worlds
  brute_force       spends the budget on one scan of each image itself, reads the SNR off it, squares for the averages;
                    no model of the repetition time or the bandwidth
Ablations (the reference with one decision removed; each must lose where that decision matters)
  tr_fixed          one scout at a fixed repetition time, scaled with the SNR laws, averages squared: no T1
  rule_of_thumb     measures like the reference, then TR = 1.3 x T1 of the weakest region and the smallest sufficient
                    number of averages: repetition time and averages are not chosen together
  single_average    the reference restricted to one average and the narrowest bandwidth (the level-1 rung)
  lowest_bandwidth  the reference restricted to the narrowest bandwidth (the level-2 rung): no T2
Reported
  open_loop         the reference's scans as a fixed list (no look at the data before the last scan) and the same solver
  ideal_pulses      the reference with the recovery curve of ideal 90 and 180 degree pulses instead of a fitted shape:
                    what ignoring the hidden transmit error and the slice profile costs
"""
from __future__ import annotations

import functools
import hashlib
import math

import numpy as np

from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "wrong_units", "fixed_recipe", "spec_only", "brute_force")
ABLATIONS = {"tr_fixed": (1, 2, 3), "rule_of_thumb": (1, 2, 3), "single_average": (2, 3), "lowest_bandwidth": (3,)}
REPORTED = ("open_loop", "ideal_pulses")
LADDER = {1: "single_average", 2: "lowest_bandwidth", 3: "reference"}      # the policy that suffices at each level
MARGIN = 0.03                                                           # SNR asked for above the requirement
DESCRIPTIONS = dict(
    reference="noise, recovery curve (T1 and its shape) and T2 of every region from cheap scans; SNR laws; joint search of "
              f"repetition time, averages and bandwidth with a {100 * MARGIN:g} % margin",
    prior="no scan; one typical protocol for both images",
    wrong_units="the reference with the noise read as the SD of the background magnitude (0.655 of the channel SD)",
    fixed_recipe="one token scan, then the best pair of constant protocols tuned on other worlds",
    spec_only="one token scan, then the best protocol per (matrix, range of the required SNR) of each image, tuned on other worlds",
    brute_force="one scan of each image itself on the whole budget, averages squared from the SNR read off it",
    tr_fixed="one scout at a fixed repetition time scaled with the SNR laws, averages squared (no T1)",
    rule_of_thumb="measures T1, then TR = 1.3 x T1 and the smallest sufficient number of averages",
    single_average="the reference restricted to one average and the narrowest bandwidth",
    lowest_bandwidth="the reference restricted to the narrowest bandwidth (no T2)",
    open_loop="the reference's scans as a fixed list, then the same solver",
    ideal_pulses="the reference with the recovery curve of ideal 90 and 180 degree pulses",
)

RAYLEIGH_MEAN, RAYLEIGH_SD = math.sqrt(math.pi / 2.0), math.sqrt(2.0 - math.pi / 2.0)   # of the magnitude of unit noise
SCOUT = dict(sequence="se", matrix=64, phase_fov=0.875, bandwidth_hz_per_px=100.0)
# noise: a short-TR scan (little signal) on a field of view wider than the object, read in four large corner discs
NOISE_SCAN = dict(sequence="se", fov_mm=360.0, matrix=384, tr_ms=50.0)
NOISE_DISC_MM = 55.0
SCOUT_TR_MS = (120.0, 300.0, 800.0, 2400.0)                             # the last one is the fixed choice of ``open_loop``
ECHO_TE_MS = 32.0                                                       # second echo time, at the third repetition time
SHAPE_GRID = np.radians(np.arange(60.0, 121.0, 5.0))                    # effective flip angles of the recovery curve
TR_GRID = np.exp(np.linspace(math.log(50.0), math.log(16000.0), 581))
BW_STEP = 1.05
PLAN_BW = (50.0, 100.0, 200.0, 400.0)
T1_GRID = np.exp(np.linspace(math.log(100.0), math.log(6000.0), 600))
TR_FIXED_MS = 1000.0
PRIOR = dict(tr_ms=800.0, averages=1, bandwidth_hz_per_px=50.0)


# ------------------------------------------------------------------------------------------------ measurement
def _task(sc):
    return sc.info()["task"]


def _circles(t, ids):
    return [r for r in t["regions"] if r["id"] in ids]


def _fov(t):
    return next(iter(t["images"].values()))["scan"]["fov_mm"]


def _complex(resp, image=0):
    return {row["id"]: row["complex_mean_magnitude"][image] * np.exp(1j * row["complex_mean_phase_rad"][image])
            for row in resp["roi_readouts"]}


def noise_factor(rp) -> float:
    """SD of the noise in one pixel per unit receiver noise density, from the realised protocol: every acquired
    sample carries noise of variance ~ bandwidth / averages (a 3D scan also averages over its partitions), and the
    reconstruction sums them into pixels of the acquired field of view."""
    return rp["matrix"] * math.sqrt(rp["lines"] * rp["bandwidth_hz_per_px"] / (rp["averages"] * rp.get("partitions", 1))) \
        / (rp["fov_mm"] * rp["fov_pe_mm"] * rp["lines_full"] / rp["matrix_pe"])


def _corners(fov, radius, gap=3.0):
    c = fov / 2.0 - radius - gap
    return [dict(id=f"bg{k}", y_mm=sy * c, x_mm=sx * c, radius_mm=radius) for k, (sy, sx) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))]


def _noise_scan(sc):
    return sc.scan(dict(NOISE_SCAN), _corners(NOISE_SCAN["fov_mm"], NOISE_DISC_MM))


def _noise(sc, statistic="mean"):
    """Receiver noise density from the corners of ``NOISE_SCAN`` (tens of thousands of pixels without signal).
    The magnitude of pure noise is Rayleigh distributed: its mean is 1.2533 and its SD 0.6551 of the channel SD."""
    r = _noise_scan(sc)
    n = np.array([row["n_pixels"] for row in r["roi_readouts"]], float)
    if statistic == "mean":
        sd = float(np.sum(n * [row["mean_magnitude"][0] for row in r["roi_readouts"]]) / n.sum()) / RAYLEIGH_MEAN
    else:                                                               # the unit error of ``wrong_units``
        sd = float(np.sqrt(np.sum(n * np.square([row["sd_magnitude"][0] for row in r["roi_readouts"]])) / n.sum()))
    return sd / noise_factor(r["realised"])


def recovery(a, tr_ms, te_ms, t1_ms):
    """Longitudinal magnetisation before the excitation of a spin echo with excitation ``a`` and refocusing 2a (rad),
    all transverse magnetisation crushed: the steady state of [a, TE/2, 2a, TR - TE/2]."""
    e1, e2 = np.exp(-te_ms / 2.0 / t1_ms), np.exp(-(tr_ms - te_ms / 2.0) / t1_ms)
    c1, c2 = math.cos(a), math.cos(2.0 * a)
    return (1.0 - e2 + c2 * e2 * (1.0 - e1)) / (1.0 - c1 * c2 * e1 * e2)


def _fit_recovery(tr, te, y, shapes):
    """Least squares of y = A recovery(a, TR; T1) over T1 on a grid and the angles ``shapes`` (A is linear).
    Returns (A, T1, a). With one shape the curve is that of a known flip angle; with ``SHAPE_GRID`` the angle is an
    effective one that takes up what the slice profile and the transmit error do to the curve."""
    tr, te, y = np.asarray(tr, float), np.asarray(te, float), np.asarray(y, float)
    best = None
    for a in np.atleast_1d(shapes):
        m = recovery(float(a), tr[None, :], te[None, :], T1_GRID[:, None])  # [T1, scan]
        amp = (m @ y) / np.sum(m * m, axis=1)
        err = np.sum((amp[:, None] * m - y[None, :]) ** 2, axis=1)
        k = int(np.argmin(err))
        if best is None or err[k] < best[0]:
            best = (float(err[k]), float(amp[k]), float(T1_GRID[k]), float(a))
    return best[1:]


def _scout(sc, t, tr_ms, **kw):
    r = sc.scan(dict(SCOUT, fov_mm=_fov(t), tr_ms=tr_ms, **kw), t["regions"])
    return dict(tr=r["realised"]["tr_ms"], te=r["realised"]["te_ms"], y={k: abs(v) for k, v in _complex(r).items()})


def _long_tr(sc, t, first):
    """Repetition time of the last scout: about 1.6 x the longest T1 seen so far, as far as the budget allows."""
    t1 = max(_fit_recovery([s["tr"] for s in first], [s["te"] for s in first], [s["y"][i] for s in first], math.pi / 2)[1]
             for i in first[0]["y"])
    overhead = sc.info()["cost"]["overhead_s_per_scan"]
    cycles = sc.plan(dict(SCOUT, fov_mm=_fov(t), tr_ms=1000.0))["scan_s"] - overhead          # seconds per second of TR
    affordable = math.floor((sc.budget()["remaining_s"] - overhead - 0.5) / cycles * 100.0) * 10.0
    return float(np.clip(round(1.6 * t1 / 50.0) * 50.0, 1000.0, min(3200.0, affordable)))


def _measure(sc, t, adaptive=True, noise="mean", shape="free"):
    """Everything the solver needs: dict(sigma0, te_ref, regions={id: dict(amp, t1, a, t2)}).

    ``shape``: how the recovery curve is modelled. "free": an effective flip angle per region, fitted with amplitude
    and T1 to the four repetition times. "ideal": 90 degrees."""
    sigma0 = _noise(sc, noise)
    shapes = SHAPE_GRID if shape == "free" else math.pi / 2
    scouts = [_scout(sc, t, tr) for tr in SCOUT_TR_MS[:3]]
    echo = _scout(sc, t, SCOUT_TR_MS[2], te_ms=ECHO_TE_MS)
    scouts.append(_scout(sc, t, _long_tr(sc, t, scouts) if adaptive else SCOUT_TR_MS[3]))
    out = {}
    for i in scouts[0]["y"]:
        amp, t1, a = _fit_recovery([s["tr"] for s in scouts], [s["te"] for s in scouts], [s["y"][i] for s in scouts], shapes)
        mid = scouts[2]
        ratio = echo["y"][i] / mid["y"][i] * recovery(a, mid["tr"], mid["te"], t1) / recovery(a, echo["tr"], echo["te"], t1)
        t2 = (echo["te"] - mid["te"]) / max(-math.log(min(ratio, 0.999)), 1e-3)
        out[i] = dict(amp=amp, t1=t1, a=a, t2=float(np.clip(t2, 2.0, 5000.0)))
    return dict(sigma0=sigma0, te_ref=scouts[0]["te"], regions=out)


# ------------------------------------------------------------------------------------------------ the solver
def _geometry(sc, t, name, bandwidths):
    """Console facts of an image for the candidate bandwidths: echo time (interpolated in the readout duration between
    a few plans), noise factor of one average, phase-encode lines, shortest repetition time."""
    scan = {k: v for k, v in t["images"][name]["scan"].items() if k in ("sequence", "fov_mm", "matrix", "slice_mm")}
    lo, hi = float(np.min(bandwidths)), float(np.max(bandwidths))
    nodes = sorted({lo, hi, *[b for b in PLAN_BW if lo < b < hi]})
    plans = [sc.plan(dict(scan, tr_ms=2000.0, bandwidth_hz_per_px=b)) for b in nodes]
    inv = np.array([1.0 / b for b in nodes])
    te = np.interp(1.0 / bandwidths, inv[::-1], np.array([p["realised"]["te_ms"] for p in plans])[::-1])
    rp = plans[0]["realised"]
    return dict(scan=scan, te=te, lines=rp["lines"], tr_min=max(p["timing"]["tr_min_ms"] for p in plans),
                noise=noise_factor(rp) * np.sqrt(bandwidths / rp["bandwidth_hz_per_px"]))


def _snr1(m, geo, ids, tr=TR_GRID):
    """Predicted SNR of one average: array [region, bandwidth, tr]."""
    out = []
    for i in ids:
        r = m["regions"][i]
        sig = r["amp"] * recovery(r["a"], tr[None, :], geo["te"][:, None], r["t1"]) * np.exp(-(geo["te"][:, None] - m["te_ref"]) / r["t2"])
        out.append(sig / (m["sigma0"] * geo["noise"][:, None]))
    return np.array(out)


def _bandwidths(sc, lowest_only=False):
    p = sc.info()["sequences"]["se"]["parameters"]["bandwidth_hz_per_px"]
    lo = float(p["min"])
    return np.array([lo]) if lowest_only else lo * BW_STEP ** np.arange(int(math.log(400.0 / lo) / math.log(BW_STEP)) + 1)


def _deliver(tr_ms, averages, bw):
    return dict(tr_ms=float(math.ceil(tr_ms * 10.0 - 1e-6) / 10.0), averages=int(averages), bandwidth_hz_per_px=round(float(bw), 1))


def _solve(sc, t, name, m, averages=None, lowest_bandwidth=False, margin=MARGIN):
    """Shortest scan of image ``name`` that the model ``m`` predicts to meet the requirement with ``margin``."""
    img = t["images"][name]
    bws = _bandwidths(sc, lowest_bandwidth)
    geo = _geometry(sc, t, name, bws)
    g = np.maximum.accumulate(_snr1(m, geo, img["regions"]).min(0), axis=1)     # binding region, [bandwidth, tr]
    n = np.arange(1, 65) if averages is None else np.asarray(averages)
    need = img["snr_min"] * (1.0 + margin) / np.sqrt(n)
    best = None
    for j, bw in enumerate(bws):
        ok = need <= g[j, -1]
        if not ok.any():
            continue
        tr = np.maximum(np.interp(need[ok], g[j], TR_GRID), geo["tr_min"])
        cyc = tr * (1 + geo["lines"] * n[ok])
        k = int(np.argmin(cyc))
        if best is None or cyc[k] < best[0]:
            best = (float(cyc[k]), float(tr[k]), int(n[ok][k]), float(bw))
    if best is None:                                                    # not reachable under the restriction: the most it can do
        j = int(np.argmax(g[:, -1]))
        best = (0.0, 5.0 * max(r["t1"] for r in m["regions"].values()), int(n[-1]), float(bws[j]))
    return _deliver(*best[1:])


def _answer(sc, t, m, **kw):
    return {name: _solve(sc, t, name, m, **kw) for name in t["images"]}


# ------------------------------------------------------------------------------------------------ the policies
def reference(sc):
    t = _task(sc)
    return _answer(sc, t, _measure(sc, t))


def open_loop(sc):
    t = _task(sc)
    return _answer(sc, t, _measure(sc, t, adaptive=False))


def wrong_units(sc):
    t = _task(sc)
    return _answer(sc, t, _measure(sc, t, noise="sd"))


def ideal_pulses(sc):
    t = _task(sc)
    return _answer(sc, t, _measure(sc, t, shape="ideal"))


def single_average(sc):
    t = _task(sc)
    return _answer(sc, t, _measure(sc, t), averages=[1], lowest_bandwidth=True)


def lowest_bandwidth(sc):
    t = _task(sc)
    return _answer(sc, t, _measure(sc, t), lowest_bandwidth=True)


def rule_of_thumb(sc):
    t = _task(sc)
    m = _measure(sc, t)
    out = {}
    for name, img in t["images"].items():
        bws = _bandwidths(sc, True)
        geo = _geometry(sc, t, name, bws)
        relaxed = _snr1(m, geo, img["regions"], tr=np.array([TR_GRID[-1]]))[:, 0, 0]
        weakest = img["regions"][int(np.argmin(relaxed))]
        tr = max(1.3 * m["regions"][weakest]["t1"], geo["tr_min"])
        snr = float(_snr1(m, geo, img["regions"], tr=np.array([tr])).min())
        out[name] = _deliver(tr, np.clip(math.ceil((img["snr_min"] * (1.0 + MARGIN) / snr) ** 2), 1, 64), bws[0])
    return out


def tr_fixed(sc):
    """Measure the SNR where it will be used (the fixed repetition time, the narrowest bandwidth), scale it to the
    image with the SNR laws, square for the averages."""
    t = _task(sc)
    sigma0, bw = _noise(sc), float(_bandwidths(sc, True)[0])
    y = _scout(sc, t, TR_FIXED_MS, bandwidth_hz_per_px=bw)["y"]
    out = {}
    for name, img in t["images"].items():
        geo = _geometry(sc, t, name, np.array([bw]))
        snr = min(y[i] for i in img["regions"]) / (sigma0 * geo["noise"][0])
        out[name] = _deliver(TR_FIXED_MS, np.clip(math.ceil((img["snr_min"] * (1.0 + MARGIN) / snr) ** 2), 1, 64), bw)
    return out


def brute_force(sc):
    """No scout, no model: one scan of each image itself, as long a repetition time as half the budget pays for."""
    t = _task(sc)
    share, out = sc.budget()["remaining_s"] / len(t["images"]), {}
    for name, img in t["images"].items():
        scan = {k: v for k, v in img["scan"].items() if k in ("sequence", "fov_mm", "matrix", "slice_mm")}
        p = sc.plan(dict(scan, tr_ms=1000.0))
        tr = math.floor((share - 3.5) / (p["scan_s"] - 3.0) * 1000.0 / 10.0) * 10.0
        r = sc.scan(dict(scan, tr_ms=tr), _circles(t, img["regions"]) + _corners(scan["fov_mm"], 24.0))
        rows = {row["id"]: row for row in r["roi_readouts"]}
        sd = float(np.mean([rows[f"bg{k}"]["mean_magnitude"][0] for k in range(4)])) / RAYLEIGH_MEAN
        snr = min(rows[i]["complex_mean_magnitude"][0] for i in img["regions"]) / sd
        out[name] = _deliver(tr, np.clip(math.ceil((img["snr_min"] / snr) ** 2), 1, 64), r["realised"]["bandwidth_hz_per_px"])
    return out


def prior(sc):
    return {name: dict(PRIOR) for name in _task(sc)["images"]}


def fixed_recipe(recipe):
    def policy(sc):
        t = _task(sc)
        _noise_scan(sc)
        return {name: {k: recipe[name][k] for k in task.FIELDS} for name in t["images"]}
    return policy


DEFAULT_RECIPE = dict(A=dict(PRIOR), B=dict(PRIOR, averages=2))


def spec_only(sc):
    t = _task(sc)
    _noise_scan(sc)
    table = spec_table(t["level"])
    return {name: dict(table["protocols"][name][_cell(table, name, img)]) for name, img in t["images"].items()}


def policies(recipe=None):
    return dict(reference=reference, prior=prior, wrong_units=wrong_units, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE),
                spec_only=spec_only, brute_force=brute_force, tr_fixed=tr_fixed, rule_of_thumb=rule_of_thumb,
                single_average=single_average, lowest_bandwidth=lowest_bandwidth, open_loop=open_loop, ideal_pulses=ideal_pulses)


# ------------------------------------------------------------------------------------------------ tuning on other worlds
CANDIDATES = dict(averages=np.arange(1, 13), tr_ms=np.exp(np.linspace(math.log(120.0), math.log(4500.0), 90)),
                  bandwidth_hz_per_px=np.array([50.0, 75.0, 110.0, 160.0]))
TUNE_EXTRA = 7              # further worlds drawn per tuning instance (``tune`` gets 16 instances from the gate)
SPEC_WORLDS = 240           # worlds ``spec_only`` is tuned on, per level
SNR_BINS = 2                # ``spec_only`` distinguishes this many ranges of the required SNR per matrix


def _other(seed: str, k: int) -> str:
    return hashlib.sha256(f"{seed}|snrspec-tuning|{k}".encode()).hexdigest()[:32]


def _pass_table(cfgs):
    """For every image: which candidate protocols pass on which tuning world, by the task's own rule evaluated on the
    material values of those worlds. Returns {image: dict(passes [candidate, world], matrix [world], snr_min [world])}."""
    from ...scanner import world as W
    from . import oracle
    n, tr, bw = (CANDIDATES[k] for k in ("averages", "tr_ms", "bandwidth_hz_per_px"))
    jbw = [int(np.argmin(np.abs(oracle.BANDWIDTHS - b))) for b in bw]
    out = {name: dict(passes=[], matrix=[], snr_min=[]) for name in task.IMAGES}
    for c in cfgs:
        world = W.make_world(c["seed"], c["world_spec"], layout_only=True)
        for name in task.IMAGES:
            img = task.image_of(c["task"], name)
            table = oracle.snr_table(world, img)
            best = oracle.optimum(table, img)
            g = table["snr"].min(0)[jbw]                                                    # [bandwidth, TR grid]
            snr = np.array([np.interp(tr, oracle.TR_GRID, row) for row in g])               # [bandwidth, tr]
            snr = snr[None, :, :] * np.sqrt(n)[:, None, None]                               # [averages, bandwidth, tr]
            scan_s = 3.0 + tr[None, None, :] * oracle.cycles(table, n)[:, None, None] / 1000.0 + 0.0 * snr
            ok = (snr >= img["snr_min"]) & (scan_s <= c["task"]["time_factor"] * best["scan_s"])
            out[name]["passes"].append(ok.ravel())
            out[name]["matrix"].append(img["scan"]["matrix"])
            out[name]["snr_min"].append(img["snr_min"])
    return {k: dict(passes=np.array(v["passes"]).T, matrix=np.array(v["matrix"]), snr_min=np.array(v["snr_min"])) for k, v in out.items()}


def _candidate(index: int) -> dict:
    n, tr, bw = (CANDIDATES[k] for k in ("averages", "tr_ms", "bandwidth_hz_per_px"))
    i, j, k = np.unravel_index(int(index), (len(n), len(bw), len(tr)))
    return _deliver(tr[k], n[i], bw[j])


def _more(cfgs, extra):
    return list(cfgs) + [task.make_instance(_other(c["seed"], k), c["level"]) for c in cfgs for k in range(extra)]


def _best_pair(tab):
    """Indices of the pair of constant protocols that passes BOTH images on the largest number of the worlds of ``tab``
    (ties: the larger sum of the two single-image pass rates), and that joint pass rate."""
    a, b = (tab[name]["passes"].astype(np.float32) for name in task.IMAGES)
    joint = a @ b.T + 1e-3 * (a.sum(1)[:, None] + b.sum(1)[None, :]) / a.shape[1]
    i, j = np.unravel_index(int(np.argmax(joint)), joint.shape)
    return int(i), int(j), float((a[i] * b[j]).mean())


def tune(cfgs):
    """The strongest pair of constant protocols on other worlds (``_best_pair``): the recipe of ``fixed_recipe``."""
    tab = _pass_table(_more(cfgs, TUNE_EXTRA))
    i, j, rate = _best_pair(tab)
    single = [float(tab[name]["passes"][k].mean()) for name, k in zip(task.IMAGES, (i, j))]
    return dict(A=_candidate(i), B=_candidate(j), tuned_on_worlds=int(tab["A"]["passes"].shape[1]), tuned_joint_pass_rate=round(rate, 4),
                tuned_single_pass_rate=[round(v, 4) for v in single])


def _cell(table, name, img) -> str:
    edges = table["snr_edges"][name][str(img["scan"]["matrix"])]
    return f"{img['scan']['matrix']}:{int(np.searchsorted(edges, img['snr_min']))}"


@functools.lru_cache(maxsize=8)
def spec_table(level: int, n_worlds: int | None = None) -> dict:
    """The lookup of ``spec_only``: for each image and each cell (matrix, lower or upper range of the required SNR) one
    protocol, chosen on ``SPEC_WORLDS`` other worlds of the level (the task card states the level) to pass both images
    as often as possible. It starts from the best pair of constant protocols on those worlds and changes the protocol
    of a cell only when that passes more of them (coordinate ascent), so on its tuning worlds it is never worse than
    the best constant."""
    cfgs = [task.make_instance(_other(f"{level:032x}", k), level) for k in range(n_worlds or SPEC_WORLDS)]
    tab = _pass_table(cfgs)
    table = dict(snr_edges={}, protocols={}, tuned_on_worlds=len(cfgs))
    cells, choice = {}, {}
    for name, start in zip(task.IMAGES, _best_pair(tab)[:2]):
        t = tab[name]
        table["snr_edges"][name] = {str(m): np.quantile(t["snr_min"][t["matrix"] == m], np.arange(1, SNR_BINS) / SNR_BINS).tolist()
                                    if (t["matrix"] == m).any() else [] for m in task.MATRICES[name]}
        cells[name] = np.array([_cell(table, name, dict(scan=dict(matrix=int(m)), snr_min=s)) for m, s in zip(t["matrix"], t["snr_min"])])
        choice[name] = {f"{m}:{b}": start for m in task.MATRICES[name] for b in range(SNR_BINS)}

    def won(name):
        return np.array([tab[name]["passes"][choice[name][c], w] for w, c in enumerate(cells[name])], float)
    for _ in range(3):
        for name, other in (task.IMAGES, task.IMAGES[::-1]):
            partner = won(other)
            for c in choice[name]:
                sel = cells[name] == c
                p = tab[name]["passes"][:, sel]
                gain = p @ partner[sel] + 1e-3 * p.sum(1)
                k = int(np.argmax(gain))
                if gain[k] > gain[choice[name][c]] + 1e-9:
                    choice[name][c] = k
    table["tuned_joint_pass_rate"] = round(float((won("A") * won("B")).mean()), 4)
    for name in task.IMAGES:
        table["protocols"][name] = {c: _candidate(i) for c, i in choice[name].items()}
    return table
