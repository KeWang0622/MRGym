"""The optimum of ``Scanner/SNRSpec``: the shortest scan that meets an SNR requirement, by exhaustive search.

Verifier side. Independent of the control policies (it must not import ``controls``): it reads the hidden world.

For one image of the examination (a spin echo with a fixed geometry) the free fields are ``tr_ms``, ``averages`` and
``bandwidth_hz_per_px``. The search is exhaustive over

    bandwidth    ``BANDWIDTHS``: from the console minimum, 50 Hz/px, upward in steps of 6 %
    averages     1 .. 64, every integer
    tr_ms        continuous: for a given bandwidth and number of averages the signal-to-noise ratio of every region
                 grows with the repetition time, so the shortest repetition time that meets the requirement is the
                 point where the binding region crosses it (found on ``TR_GRID``, 1 % steps, by interpolation)

Signal-to-noise ratio of a region for one candidate protocol:

    signal   ``Engine.amplitudes`` on points of the region in the hidden world (the engine's own spin-echo steady
             state through the slice profile at the local flip angle, times proton density), averaged with the receive
             profile, times the engine's echo-time decay. The amplitudes are evaluated on a lattice (``TR_NODES`` x
             three echo times that span the bandwidths) and interpolated: they are smooth in both. The result is
             multiplied by the ratio ``scan / amplitudes`` measured on two noise-free engine scans of the image's
             geometry. That ratio holds what image formation adds (displacement and compression by the off-resonance
             field, truncation, partial volume); it is linear in the readout duration and does not depend on the
             repetition time. The two scans are taken at the repetition time and bandwidth of a first search without
             the ratio and at a second, distant bandwidth, so the ratio is measured where the optimum lies.
    noise    ``Engine.noise_sd``: the analytic standard deviation of one pixel, one channel

The search asks for ``MARGIN`` more SNR than the requirement, so that the protocol it returns passes the verifier's own
noise-free scan (the model above agrees with that scan to better than ``MARGIN``; ``tests/test_snr_spec.py``).

With ``engine=None`` (and a world made with ``layout_only=True``) the same search runs on the material values alone,
with a flat receive profile and a flat transmit pattern: the instance generator uses it to set a requirement that has
an answer, and the tuning of the control policies uses it on other worlds. It is never used for scoring.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ...scanner import console, kspace
from ...scanner import world as W
from .. import family as F
from .. import oracle as O

AVERAGES = np.arange(1, 65)
BANDWIDTHS = np.round(50.0 * 1.06 ** np.arange(48), 2)                  # 50 .. 773 Hz/px
TR_GRID = np.exp(np.linspace(np.log(50.0), np.log(16000.0), 581))       # ms, 1 % steps, from the console minimum
TR_NODES = np.unique(np.append(TR_GRID[::15], TR_GRID[-1]))            # where the amplitudes are evaluated (16 % steps)
MARGIN = 1e-3
MAX_POINTS = 80
REGRET_KIND = "ratio"               # T* is a scan time: the regret of an answer is its scan time over T*, minus 1
SCAN_FIELDS = ("sequence", "fov_mm", "matrix", "slice_mm")              # what the task fixes of an image, besides the defaults


def image_of(task: dict, name: str) -> dict:
    """One image of the task card as the search sees it: its fixed fields, regions and requirement."""
    img = task["images"][name]
    return dict(scan={k: img["scan"][k] for k in SCAN_FIELDS}, regions=list(img["regions"]), snr_min=img["snr_min"])


def protocol(image: dict, tr_ms, averages, bandwidth_hz_per_px) -> dict:
    """The scan of one image: the task's fixed fields plus the three delivered ones."""
    return dict(image["scan"], tr_ms=float(tr_ms), averages=int(averages), bandwidth_hz_per_px=float(bandwidth_hz_per_px))


class RegionModel(O.RegionPoints):
    """``RegionPoints`` with the table of this family: the signal on the bandwidth grid at every repetition time."""

    def signals(self, realised: list) -> np.ndarray:
        """``signal`` for the realised protocols of the bandwidth grid (they differ in bandwidth and echo time) at every
        repetition time of ``TR_GRID``: array [bandwidth, tr]. Amplitudes on ``TR_NODES`` x three echo times, a cubic
        spline in log TR and a parabola in the echo time between them."""
        te = np.array([rp["te_ms"] for rp in realised])
        nodes = sorted({int(np.argmax(te)), int(np.argmin(np.abs(te - 0.5 * (te.max() + te.min())))), int(np.argmin(te))})
        amp = np.array([[self.amplitude(dict(realised[j], tr_ms=float(tr))) for tr in TR_NODES] for j in nodes])
        amp = CubicSpline(np.log(TR_NODES), amp, axis=1)(np.log(TR_GRID))                    # [echo-time node, tr]
        tn = te[nodes]
        if len(nodes) == 1:
            lag = np.ones((len(te), 1))
        else:                                                                                 # Lagrange weights in the echo time
            lag = np.stack([np.prod([(te - tn[m]) / (tn[k] - tn[m]) for m in range(len(tn)) if m != k], axis=0)
                            for k in range(len(tn))], axis=1)
        scale = np.array([self.decay(rp) * rp["slice_mm"] for rp in realised])
        return (lag @ amp) * scale[:, None]

    def noise_sd(self, realised: dict) -> float:
        return self.engine.noise_sd(realised)


def _plan(image, exam, tr_ms, averages, bw):
    pl = console.plan(protocol(image, tr_ms, averages, bw), exam)
    if not pl["accept"]:
        raise ValueError("the console rejects a protocol of the search grid: " + "; ".join(v["message"] for v in pl["violations"]))
    return pl


def _nodes(first: dict | None) -> tuple:
    """Where the image-formation ratio is measured: (tr_ms, the two bandwidths)."""
    if first is None:
        return 1000.0, (BANDWIDTHS[0], 300.0)
    bw = first["bandwidth_hz_per_px"]
    return first["tr_ms"], (bw, 3.0 * bw if bw < 150.0 else BANDWIDTHS[0])


def calibration(engine, image: dict, regions: list, models: list, tr_ms: float, bandwidths) -> np.ndarray:
    """``scan / amplitudes`` of every region on noise-free engine scans at ``bandwidths``: array [region, node]."""
    exam = engine.world["console"]
    out = np.ones((len(regions), len(bandwidths)))
    for j, bw in enumerate(bandwidths):
        rp = _plan(image, exam, tr_ms, 1, bw)["realised"]
        scan = F.region_complex(engine, rp, regions)
        for i, (r, m) in enumerate(zip(regions, models)):
            out[i, j] = abs(scan[r["id"]]) / m.signal(rp)
    return out


def snr_table(world: dict, image: dict, engine=None) -> dict:
    """SNR of every region of ``image`` for ONE average on the grid: ``snr`` [region, bandwidth, tr]."""
    regions = [r for r in world["regions"] if r["id"] in image["regions"]]
    exam = engine.world["console"] if engine is not None else dict(b0_t=world["b0_t"], fov_mm=image["scan"]["fov_mm"])
    models = [RegionModel(world, r) for r in regions]
    plans = [_plan(image, exam, TR_GRID[-1], 1, bw) for bw in BANDWIDTHS]
    realised = [pl["realised"] for pl in plans]
    sd = np.array([models[0].noise_sd(rp) for rp in realised])
    snr = np.array([m.signals(realised) / sd[:, None] for m in models])
    table = dict(snr=snr, tr_min_ms=np.array([pl["timing"]["tr_min_ms"] for pl in plans]), regions=[r["id"] for r in regions],
                 exam=exam, lines=realised[0]["lines"], calibration=None)
    if engine is not None:                                              # image formation, measured where the optimum lies
        tr_ms, nodes = _nodes(optimum(table, image))
        cal = calibration(engine, image, F.public_regions(regions), models, tr_ms, nodes)
        lo, hi = (1.0 / b for b in nodes)
        ratio = cal[:, :1] + (cal[:, 1:] - cal[:, :1]) * ((1.0 / BANDWIDTHS)[None, :] - lo) / (hi - lo)   # [region, bandwidth]
        table.update(snr=snr * ratio[:, :, None], calibration=dict(tr_ms=tr_ms, bandwidths=[float(b) for b in nodes], ratio=cal.tolist()))
    return table


def cycles(table: dict, averages) -> np.ndarray:
    """Repetitions a scan plays: the dummy cycles plus one per phase-encode line and average."""
    return console.SYSTEM["prepared_dummy_cycles"] + table["lines"] * np.asarray(averages)


def snr_of(table: dict, tr_ms, averages, bandwidth_hz_per_px) -> np.ndarray:
    """Model SNR of every region for one protocol (nearest grid bandwidth, interpolated in TR). For tuning and tests."""
    j = int(np.argmin(np.abs(BANDWIDTHS - bandwidth_hz_per_px)))
    return np.array([np.interp(tr_ms, TR_GRID, row[j]) for row in table["snr"]]) * np.sqrt(averages) \
        * np.sqrt(BANDWIDTHS[j] / bandwidth_hz_per_px)


def optimum(table: dict, image: dict, bandwidths=None, averages=None) -> dict | None:
    """The shortest scan of ``image`` whose every region has SNR >= ``image["snr_min"]``; None when no protocol of the
    grid does. ``bandwidths`` / ``averages`` restrict the search (for what-if questions: a fixed bandwidth, one average)."""
    need = image["snr_min"] * (1.0 + MARGIN) / np.sqrt(AVERAGES)        # what one average must give
    best = None
    for j, bw in enumerate(BANDWIDTHS):
        if bandwidths is not None and not np.any(np.isclose(bw, bandwidths)):
            continue
        g = np.maximum.accumulate(table["snr"][:, j].min(0))             # the binding region, non-decreasing in TR
        ok = need <= g[-1]
        if averages is not None:
            ok &= np.isin(AVERAGES, averages)
        if not ok.any():
            continue
        tr = np.maximum(np.interp(need[ok], g, TR_GRID), max(table["tr_min_ms"][j], TR_GRID[0]))
        cyc = tr * cycles(table, AVERAGES[ok])
        k = int(np.argmin(cyc))
        if best is None or cyc[k] < best[0]:
            best = (float(cyc[k]), float(tr[k]), int(AVERAGES[ok][k]), float(bw), j)
    if best is None:
        return None
    _, tr, n, bw, j = best
    raster = console.SYSTEM["raster_ms"]
    tr = float(np.ceil(tr / raster - 1e-9) * raster)
    pl = _plan(image, table["exam"], tr, n, bw)
    snr = snr_of(table, tr, n, bw)
    return dict(scan_s=pl["scan_s"], tr_ms=pl["realised"]["tr_ms"], averages=n, bandwidth_hz_per_px=bw, te_ms=pl["realised"]["te_ms"],
                snr=dict(zip(table["regions"], (float(v) for v in snr))), binding=table["regions"][int(np.argmin(snr))],
                at_grid_edge=bool(j == len(BANDWIDTHS) - 1 or n == AVERAGES[-1] or tr >= TR_GRID[-2]))


def shortest(world: dict, image: dict, engine=None) -> dict | None:
    """``optimum`` of one image on ``world`` (with ``engine``: the verifier's; without: material values only)."""
    return optimum(snr_table(world, image, engine), image)


def measured_snr(engine, image: dict, regions: list, fields: dict) -> tuple:
    """(scan seconds, {region: SNR}) of one protocol of ``image`` on a noise-free engine scan with the analytic noise."""
    pl = _plan(image, engine.world["console"], fields["tr_ms"], fields["averages"], fields["bandwidth_hz_per_px"])
    rp = pl["realised"]
    sd = engine.noise_sd(rp)
    return pl["scan_s"], {i: float(abs(v) / sd) for i, v in F.region_complex(engine, rp, regions).items()}


@O.timed
def solve(cfg: dict, world: dict | None = None) -> dict:
    """The optimum of an instance (the contract is in ``mrgym.envs.oracle``): ``value`` is T*, the summed scan time of
    the shortest sufficient scan of every image; ``argmin`` the deliverable; ``images`` the optimum of each image;
    ``certificate`` the same protocols with a repetition time 1 % shorter, each with the check it fails on a
    noise-free engine scan."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    engine = kspace.Engine(world)
    images, certificate = {}, []
    for name in cfg["task"]["images"]:
        image = image_of(cfg["task"], name)
        best = shortest(world, image, engine)
        images[name] = best
        if best is None:
            continue
        regions = [r for r in cfg["task"]["regions"] if r["id"] in image["regions"]]
        shorter = dict(tr_ms=round(best["tr_ms"] * 0.99, 2), averages=best["averages"], bandwidth_hz_per_px=best["bandwidth_hz_per_px"])
        if min(measured_snr(engine, image, regions, shorter)[1].values()) < image["snr_min"]:
            certificate.append(dict(deliverable={name: shorter}, fails=f"snr_{name}"))
    found = all(v is not None for v in images.values())
    fields = ("tr_ms", "averages", "bandwidth_hz_per_px")
    return dict(value=float(sum(v["scan_s"] for v in images.values())) if found else None,
                argmin={k: {f: v[f] for f in fields} for k, v in images.items()} if found else None, method="exhaustive",
                grid=dict(axes=dict(bandwidth_hz_per_px=dict(start=float(BANDWIDTHS[0]), stop=float(BANDWIDTHS[-1]), points=len(BANDWIDTHS)),
                                    averages=dict(start=int(AVERAGES[0]), stop=int(AVERAGES[-1]), points=len(AVERAGES)),
                                    tr_ms=dict(start=float(TR_GRID[0]), stop=float(TR_GRID[-1]), points=len(TR_GRID),
                                               refined="interpolated to the crossing of the requirement")),
                          size=len(cfg["task"]["images"]) * len(BANDWIDTHS) * len(AVERAGES) * len(TR_GRID),
                          evaluated=len(cfg["task"]["images"]) * len(BANDWIDTHS) * len(AVERAGES) * len(TR_GRID)),
                certificate=certificate, images=images)
