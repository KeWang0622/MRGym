"""The feasible protocols of ``Scanner/ProtocolToSpec``, found on the hidden world by exhaustive search.

Verifier side. Independent of the control policies (it must not import ``controls``): it reads the hidden world.

One image of the examination is an inversion-recovery spin echo of fixed geometry and repetition time (``SCAN``)
with three free fields: ``ti_ms``, ``te_ms`` and ``averages``. Three requirements are stated on marked regions of each
image (``CHECKS``) and one on the examination:

    t1_contrast   signal of one region / signal of another >= a ratio   (met near the inversion time that nulls the other)
    t2_contrast   signal of one region / signal of another >= a ratio   (a longer echo time raises it)
    snr           signal of a region / noise SD of one pixel >= a number  (shorter TE, more averages; moves with TI)
    time          scan time of both images together <= a number of seconds (the averages are shared out between them)

The search is exhaustive over ``TE_GRID`` x ``TI_GRID`` x ``AVERAGES`` (2.5 % steps of the echo time, 1 % steps of the
inversion time, every integer number of averages). The signal of a region is the engine's own inversion-recovery
steady state through the slice profile on points of the region in the hidden world (``mrgym.envs.oracle.RegionPoints``), signed, on ``TI_NODES`` and three echo times and interpolated (it is smooth in both), times the
ratio ``scan / model`` of one noise-free engine scan of the image at the standard protocol (what image formation adds;
it does not depend on TI or TE). The noise is ``Engine.noise_sd``, the scan time the console's.

The margin of an examination is the smallest logarithmic slack of any requirement (ln of achieved / required, for the
time ln of allowed / used): it passes exactly when the margin is >= 0. ``solve`` returns the examination with the
largest margin: a positive value proves that the instance has an answer and says how much room there is.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ...scanner import console, kspace
from ...scanner import world as W
from .. import family as F
from .. import oracle as O

SCAN = dict(sequence="ir", fov_mm=240.0, matrix=128, slice_mm=5.0, bandwidth_hz_per_px=200.0, tr_ms=400.0)   # the fixed fields of an image
STANDARD = dict(ti_ms=80.0, te_ms=12.0, averages=1)                     # the standard protocol of the examination
FIELDS = ("ti_ms", "te_ms", "averages")
TI_GRID = np.round(STANDARD["ti_ms"] * 1.01 ** np.arange(-140, 93), 2)  # 20 .. 200 ms; the standard TI is a grid point
TE_GRID = np.round(STANDARD["te_ms"] * 1.025 ** np.arange(96), 2)       # 12 .. 125 ms; the standard TE is the first point
TI_NODES = np.exp(np.linspace(np.log(TI_GRID[0]), np.log(TI_GRID[-1]), 18))
AVERAGES = np.arange(1, 65)
I_TI0, I_TE0 = int(np.argmin(np.abs(TI_GRID - STANDARD["ti_ms"]))), 0
TI_PLUS_TE_MAX_MS = SCAN["tr_ms"] - 15.0    # the console needs tr_ms >= ti_ms + te_ms + about 9 ms
CHECKS = ("t1_contrast", "t2_contrast", "snr")          # per image; the scan time is one requirement of the examination
REGRET_KIND = "difference"          # the value is -margin: the regret of an answer is how much margin it gives away


def protocol(fields: dict) -> dict:
    """The scan of one image: the fixed fields plus the three delivered ones."""
    return dict(SCAN, ti_ms=float(fields["ti_ms"]), te_ms=float(fields["te_ms"]), averages=int(fields["averages"]))


def _signed(model, realised: dict) -> complex:
    """Mean echo amplitude of a region (complex: the receive phase; its sign flips at the null), without the decay."""
    return complex((model.engine.amplitudes(realised, 0)[0] * model.rx).sum())


def _table(model, exam: dict) -> np.ndarray:
    """Signed signal of a region on ``TE_GRID`` x ``TI_GRID`` (positive beyond its null), no image formation."""
    nodes = [0, len(TE_GRID) // 2, len(TE_GRID) - 1]
    amp = []
    for i in nodes:
        rp = console.plan(protocol(dict(STANDARD, te_ms=TE_GRID[i])), exam)["realised"]
        row = np.array([_signed(model, dict(rp, ti_ms=float(ti))) for ti in TI_NODES])
        amp.append((row * np.conj(row[-1]) / abs(row[-1])).real)        # the longest inversion time is past every null
    amp = CubicSpline(np.log(TI_NODES), np.array(amp), axis=1)(np.log(TI_GRID))              # [te node, ti]
    tn = TE_GRID[nodes]
    lag = np.stack([np.prod([(TE_GRID - tn[m]) / (tn[k] - tn[m]) for m in range(3) if m != k], axis=0) for k in range(3)], axis=1)
    decay = np.array([model.decay(dict(te_ms=float(te))) for te in TE_GRID])
    return (lag @ amp) * decay[:, None] * SCAN["slice_mm"]


class Model:
    """Signed signal of every region of the object for the image geometry on ``TE_GRID`` x ``TI_GRID``, the noise SD of
    one pixel for one average, and the console's timing. With ``engine``: scaled to one noise-free scan (see the module)."""

    def __init__(self, world: dict, engine=None):
        exam = world["console"]
        plan = console.plan(protocol(STANDARD), exam)
        if not plan["accept"]:
            raise ValueError("the console rejects the standard protocol")
        self.ids = [r["id"] for r in world["regions"]]
        models = {r["id"]: O.RegionPoints(world, r) for r in world["regions"]}
        self.sig = {i: _table(m, exam) for i, m in models.items()}                           # [te, ti]
        self.sd1 = float(models[self.ids[0]].noise_sd(plan["realised"]))                     # one average
        self.lines, self.overhead_s = plan["realised"]["lines"], plan["scan_s_parts"]["overhead_s"]
        self.dummies = console.SYSTEM["prepared_dummy_cycles"]
        self.valid = TI_GRID[None, :] + TE_GRID[:, None] <= TI_PLUS_TE_MAX_MS
        self.scale = None
        if engine is not None:
            scan = F.region_complex(engine, plan["realised"], F.public_regions(world["regions"]))
            self.scale = {i: float(abs(scan[i]) / abs(self.sig[i][I_TE0, I_TI0])) for i in self.ids}
            self.sig = {i: v * self.scale[i] for i, v in self.sig.items()}

    def scan_s(self, averages):
        """Scan time of one image, planned as a new scan (with its adjustments)."""
        return self.overhead_s + SCAN["tr_ms"] * (self.dummies + self.lines * np.asarray(averages)) / 1000.0

    def slacks(self, req: dict) -> tuple:
        """Logarithmic slack of the two contrast requirements, [te, ti], and the SNR of one average, [te, ti]."""
        a, b = req["t1_contrast"]["brighter"], req["t1_contrast"]["darker"]
        c, d = req["t2_contrast"]["brighter"], req["t2_contrast"]["darker"]
        s = {i: np.maximum(np.abs(self.sig[i]), 1e-12) for i in {a, b, c, d, req["snr"]["region"]}}
        m1 = np.log(s[a] / s[b]) - np.log(req["t1_contrast"]["min_ratio"])
        m2 = np.log(s[c] / s[d]) - np.log(req["t2_contrast"]["min_ratio"])
        return m1, m2, s[req["snr"]["region"]] / self.sd1

    def margin(self, req: dict) -> np.ndarray:
        """The smallest slack of the three image requirements for every protocol of the grid, [averages, te, ti]
        (-inf where the console has no such scan). The scan time is a requirement of the examination: ``solve``."""
        m1, m2, snr1 = self.slacks(req)
        m3 = 0.5 * np.log(AVERAGES[:, None, None]) + np.log(snr1 / req["snr"]["min"])[None]
        m = np.minimum(m3, np.minimum(m1, m2)[None])
        m[:, ~self.valid] = -np.inf
        return m

    def feasible(self, req: dict, averages: int) -> np.ndarray:
        """Whether the image requirements hold with ``averages`` at each echo time and inversion time, [te, ti]."""
        m1, m2, snr1 = self.slacks(req)
        return self.valid & (m1 >= 0) & (m2 >= 0) & (np.sqrt(averages) * snr1 >= req["snr"]["min"])

    def best(self, req: dict, averages: int | None = None, margin=None) -> dict:
        """The protocol of the grid with the largest margin (with ``averages``: among those with that many)."""
        m = self.margin(req) if margin is None else margin
        m = m if averages is None else m[averages - 1:averages]
        k, i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        n = int(AVERAGES[k]) if averages is None else int(averages)
        return dict(ti_ms=float(TI_GRID[j]), te_ms=float(TE_GRID[i]), averages=n, margin=float(m[k, i, j]))

    def windows(self, req: dict, averages: int) -> dict:
        """Where the protocols that meet the image requirements with ``averages`` lie: the range of inversion times at
        the standard echo time, the range of echo times at the standard inversion time (None when there is none)."""
        ok = self.feasible(req, averages)
        ti, te = np.flatnonzero(ok[I_TE0, :]), np.flatnonzero(ok[:, I_TI0])
        return dict(ti_ms=(float(TI_GRID[ti[0]]), float(TI_GRID[ti[-1]])) if len(ti) else None,
                    te_ms=(float(TE_GRID[te[0]]), float(TE_GRID[te[-1]])) if len(te) else None, any=bool(ok.any()))

    def averages_max(self, time_max_s: float, images: int = 2) -> int:
        """The largest total number of averages of ``images`` images that the scan-time limit of the examination allows."""
        n = np.floor(((time_max_s - images * self.overhead_s) * 1000.0 / SCAN["tr_ms"] - images * self.dummies) / self.lines + 1e-9)
        return int(max(n, 0))


def first_failure(model: Model, req: dict, fields: dict) -> str | None:
    """The first image requirement that the model says a protocol misses (nearest grid point), or None."""
    i, j = int(np.argmin(np.abs(TE_GRID - fields["te_ms"]))), int(np.argmin(np.abs(TI_GRID - fields["ti_ms"])))
    m1, m2, snr1 = model.slacks(req)
    slack = (m1[i, j], m2[i, j], 0.5 * np.log(fields["averages"]) + np.log(snr1[i, j] / req["snr"]["min"]))
    return next((name for name, v in zip(CHECKS, slack) if v < 0.0), None)


def allocate(curves: dict, time_slack) -> tuple:
    """The numbers of averages of the images that maximise the smallest slack of the examination. ``curves``: per image
    the best margin for 1, 2, ... averages; ``time_slack(total averages)``: the slack of the scan-time limit. Exhaustive
    over every combination. Returns ({image: averages}, margin)."""
    names = list(curves)
    grids = np.meshgrid(*[np.arange(1, len(curves[k]) + 1) for k in names], indexing="ij")
    m = np.minimum.reduce([np.asarray(curves[k])[g - 1] for k, g in zip(names, grids)])
    m = np.minimum(m, time_slack(sum(grids)))
    idx = np.unravel_index(int(np.argmax(m)), m.shape)
    return {k: int(g[idx]) for k, g in zip(names, grids)}, float(m[idx])


@O.timed
def solve(cfg: dict, world: dict | None = None) -> dict:
    """The optimum of an instance (the contract is in ``mrgym.envs.oracle``): ``argmin`` is the examination with the
    largest margin (for each image the best inversion and echo time for every number of averages, then the split of
    the averages the scan-time limit allows, both exhaustive); ``value`` is minus that margin (negative: the instance
    has an answer); ``windows`` the feasible ranges of each image at its number of averages; ``certificate`` the same
    protocols moved along the inversion time and the echo time, and with one average more, with the requirement each
    then misses."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    model = Model(world, kspace.Engine(world))
    t = cfg["task"]
    margins = {name: model.margin(req) for name, req in t["images"].items()}
    cap = min(model.averages_max(t["scan_time_max_s"], len(margins)), int(AVERAGES[-1]))
    curves = {name: m.reshape(len(AVERAGES), -1).max(1)[:max(cap, 1)] for name, m in margins.items()}
    time_slack = lambda total: np.log(t["scan_time_max_s"] / (len(margins) * model.scan_s(0) + (model.scan_s(total) - model.scan_s(0))))   # noqa: E731
    split, value = allocate(curves, time_slack)
    images = {name: model.best(req, split[name], margins[name]) for name, req in t["images"].items()}
    windows = {name: model.windows(req, split[name]) for name, req in t["images"].items()}
    certificate = []
    for name, req in t["images"].items():
        for field, factor in (("ti_ms", 1.6), ("ti_ms", 0.6), ("te_ms", 2.0)):
            moved = {k: images[name][k] for k in FIELDS}
            moved[field] = round(moved[field] * factor, 2)
            why = first_failure(model, req, moved)
            if why and TI_GRID[0] <= moved["ti_ms"] <= TI_GRID[-1] and moved["te_ms"] <= TE_GRID[-1] \
                    and moved["ti_ms"] + moved["te_ms"] <= TI_PLUS_TE_MAX_MS:
                certificate.append(dict(deliverable={name: moved}, fails=f"{why}_{name}"))
    first = next(iter(images))
    if time_slack(sum(split.values()) + 1) < 0:
        certificate.append(dict(deliverable={first: dict({k: images[first][k] for k in FIELDS}, averages=split[first] + 1)}, fails="time"))
    size = len(images) * len(AVERAGES) * len(TE_GRID) * len(TI_GRID)
    return dict(value=-value, argmin={k: {f: v[f] for f in FIELDS} for k, v in images.items()}, method="exhaustive",
                grid=dict(axes=dict(ti_ms=dict(start=float(TI_GRID[0]), stop=float(TI_GRID[-1]), points=len(TI_GRID)),
                                    te_ms=dict(start=float(TE_GRID[0]), stop=float(TE_GRID[-1]), points=len(TE_GRID)),
                                    averages=dict(start=int(AVERAGES[0]), stop=int(AVERAGES[-1]), points=len(AVERAGES))),
                          size=size, evaluated=size),
                certificate=certificate, images=images, windows=windows, scan_over_model=model.scale)
