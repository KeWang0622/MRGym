"""An independent oracle of ``Scanner/BandFree``: the answer computed from the hidden world by another route.

The verifier (``task``) reads the steady state of every region from the engine (``Engine.amplitudes``, whose slice
integral is ``scanner.rf``: Cayley-Klein parameters, flip-angle tables, barycentric interpolation) and scans the
delivered protocols. The reference policy (``controls``) measures and fits a band model. This module does neither:
it reads the true off-resonance at the pixels of each region and the true material, writes the steady state
down again in closed form, and integrates it over the slice with the 3 x 3 rotation cascade of ``scanner.bloch``
(the package's second, separately written pulse simulation) on a grid of its own. It shares no code with ``task``
or ``controls``, so an error in the verifier's table, in its drift rule or in the policy's model shows up as a
disagreement.

    python -m mrgym.envs.band_free.agreement --level 2 --n 60

prints, on fresh instances, how often the oracle's own answer passes the verifier, how often the two agree on the
number of acquisitions of both images, and the regret of the must-pass policy (that module, not this one, plays the
policies). ``solve`` follows the contract of ``mrgym.envs.oracle``: ``value`` is the smallest error an answer can
have here in the oracle's own arithmetic (1 - the smallest kept fraction of its answer over regions and drifts).
"""
from __future__ import annotations

import functools

import numpy as np

from ...scanner import bloch, console
from ...scanner import world as W
from .. import oracle as O

REGRET_KIND = "difference"                           # error of an answer minus the oracle's value
STEP_DEG = 0.5
PULSE_TBW = 4.0                                      # the console's default selective pulse
ZETA = np.linspace(-2.0, 2.0, 161)                   # heights through the slice, in slice thicknesses


@functools.lru_cache(maxsize=64)
def _slice(flip_rad: float):
    """(cos and sin of the flip angle at every height, phase factor of the transverse magnetisation relative to the
    slice centre, trapezoid weights)."""
    theta, mxy = bloch.slice_profile(flip_rad, PULSE_TBW, ZETA)
    centre = mxy[len(ZETA) // 2]
    phase = np.where(np.abs(mxy) > 1e-12, mxy / np.maximum(np.abs(mxy), 1e-12), 1.0) * np.conj(centre / abs(centre))
    w = np.full(len(ZETA), ZETA[1] - ZETA[0])
    w[[0, -1]] *= 0.5
    return np.cos(theta), np.sin(theta), phase, w


def _steady_state(flip_rad, tr_ms, t1_ms, t2_ms):
    """Complex transverse magnetisation of a 2D balanced SSFP voxel right after the pulse on a grid of the phase per
    TR (0 .. 360 degrees in steps of ``STEP_DEG``): the exact steady state of every column through the slice at its
    own flip angle, summed with the column's transverse phase."""
    th = np.deg2rad(np.arange(0.0, 360.0, STEP_DEG))[:, None]
    cos_a, sin_a, phase, w = _slice(round(float(flip_rad), 9))
    e1, e2 = np.exp(-tr_ms / t1_ms), np.exp(-tr_ms / t2_ms)
    c = e2 * (e1 - 1) * (1 + cos_a[None, :])
    d = (1 - e1 * cos_a[None, :]) - (e1 - cos_a[None, :]) * e2 * e2
    columns = (1 - e1) * sin_a[None, :] * (1 - e2 * np.exp(-1j * th)) / (c * np.cos(th) + d)
    return (columns * (phase * w)[None, :]).sum(1)


def _pixels(region: dict, fov_mm: float, matrix: int):
    """Centres (y, x in mm) of the pixels of the target image that lie in the circle: what the console reads a region from."""
    pos = (np.arange(matrix) - matrix // 2) * fov_mm / matrix
    y, x = np.meshgrid(pos, pos, indexing="ij")
    inside = np.hypot(y - region["y_mm"], x - region["x_mm"]) <= region["radius_mm"]
    return y[inside], x[inside]


def points(cfg: dict, world: dict) -> dict:
    """True state of every marked region, by id: the material (T1, T2 in ms), its chemical shift, the actual flip
    angle at its centre (rad) and the positions of its sample points; the off-resonance is read there by ``_field``."""
    n, d = world["n"], world["extent_mm"] / world["n"]
    cls = {r["id"]: world["tissues"][r["cls"]] for r in world["regions"]}
    out = {}
    for r in cfg["task"]["regions"]:
        i, j = int(round(r["y_mm"] / d)) + n // 2, int(round(r["x_mm"] / d)) + n // 2
        c = cls[r["id"]]
        y, x = _pixels(r, world["console"]["fov_mm"], cfg["task"]["target_scan"]["matrix"])
        out[r["id"]] = dict(y_mm=y, x_mm=x,
                            shift_hz=-3.4e-6 * console.GAMMA_HZ_PER_T * world["b0_t"] if c["fat"] else 0.0,
                            flip_rad=float(np.deg2rad(cfg["task"]["target_scan"]["flip_deg"]) * world["fields"]["b1"][i, j] * world["system"]["tx_gain"]),
                            t1_ms=c["t1_ms"], t2_ms=c["t2_ms"])
    return out


def _field(world: dict, y_mm, x_mm):
    """Off-resonance in Hz at positions in mm (nearest point of the master grid), frequency error included."""
    n, d = world["n"], world["extent_mm"] / world["n"]
    i = np.clip(np.rint(np.asarray(y_mm) / d).astype(int) + n // 2, 0, n - 1)
    j = np.clip(np.rint(np.asarray(x_mm) / d).astype(int) + n // 2, 0, n - 1)
    return world["fields"]["b0_hz"][i, j] + world["system"]["cf_offset_hz"]


def _tr_ms(cfg, bandwidth, exam):
    s = cfg["task"]["target_scan"]
    pl = console.plan(dict(sequence="bssfp", flip_deg=s["flip_deg"], matrix=s["matrix"], bandwidth_hz_per_px=float(bandwidth), tr_ms=50.0), exam)
    return pl["timing"]["tr_min_ms"]


def curves(cfg: dict, world: dict, pts: list, bandwidth: float) -> tuple:
    """([region, phase cycle] signal of every region on a grid of ``STEP_DEG``, relative to its best; repetition time
    in ms). A spin that is off resonance appears displaced along the readout (x) by off-resonance / bandwidth pixels,
    so a region shows the spins that far from it."""
    exam = world["console"]
    tr = _tr_ms(cfg, bandwidth, exam)
    px = exam["fov_mm"] / cfg["task"]["target_scan"]["matrix"]
    n = int(round(360.0 / STEP_DEG))
    rows = []
    for p in pts:
        x = p["x_mm"]
        for _ in range(2):
            x = p["x_mm"] + (_field(world, p["y_mm"], x) + p["shift_hz"]) / bandwidth * px
        f = _field(world, p["y_mm"], x) + p["shift_hz"]
        k = np.rint(360.0 * f * tr * 1e-3 / STEP_DEG).astype(int)
        s = _steady_state(p["flip_rad"], tr, p["t1_ms"], p["t2_ms"])
        mean = np.abs((s[(k[:, None] - np.arange(n)[None, :]) % n] * np.exp(1j * np.pi * f * tr * 1e-3)[:, None]).mean(0))   # index = phase cycle
        rows.append(mean / mean.max())
    return np.array(rows), tr


def allowed(cfg: dict, world: dict, pts: list, bandwidth: float, drift_hz: float) -> np.ndarray:
    """[region, phase cycle] bool on a grid of ``STEP_DEG``: the mean signal of the region's spins keeps enough at
    this phase cycle for every drift between -drift_hz and +drift_hz."""
    rel, tr = curves(cfg, world, pts, bandwidth)
    n = rel.shape[1]
    m = int(np.ceil(360.0 * drift_hz * tr * 1e-3 / STEP_DEG - 1e-9))
    rows = []
    for good in rel >= cfg["task"]["keep_min"]:
        run = np.cumsum(np.concatenate([[0], np.tile(good, 3).astype(int)]))            # good for the whole window -m .. +m
        i = np.arange(n, 2 * n)
        rows.append(run[i + m + 1] - run[i - m] == 2 * m + 1 if 2 * m + 1 < n else np.zeros(n, bool))
    return np.array(rows)


def kept(cfg: dict, world: dict, pts: list, bandwidth: float, phase_cycles: list) -> float:
    """The smallest kept fraction of the regions ``pts`` for a protocol, over every drift up to the task's: each region
    is read from the acquisition in which its smallest value over the drifts is larger."""
    rel, tr = curves(cfg, world, pts, bandwidth)
    n = rel.shape[1]
    m = int(np.ceil(360.0 * cfg["task"]["drift_hz"] * tr * 1e-3 / STEP_DEG - 1e-9))
    worst = 1.0
    for row in rel:
        per = [row[(int(round(phi / STEP_DEG)) + np.arange(-m, m + 1)) % n].min() if 2 * m + 1 < n else row.min() for phi in phase_cycles]
        worst = min(worst, max(per))
    return float(worst)


def _pair(ok: np.ndarray):
    """Two phase-cycle indices that cover every region between them, or None."""
    miss = ((~ok) * (1 << np.arange(ok.shape[0]))[:, None]).sum(0)
    kinds = np.unique(miss)
    for a in kinds:
        for b in kinds:
            if a & b == 0:
                return _middle(miss == a), _middle(miss == b)
    return None


def _middle(mask: np.ndarray) -> int:
    """Centre of the longest circular run of True."""
    n = len(mask)
    if mask.all():
        return 0
    start = int(np.flatnonzero(~mask)[0])
    m = np.roll(mask, -start)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    runs = edges.reshape(-1, 2)
    a, b = runs[np.argmax(runs[:, 1] - runs[:, 0])]
    return int((start + (a + b) // 2) % n)


def _lowest(cfg, world, pts, grid, drift_hz, stop_single_at):
    """(lowest bandwidth of the grid with a band-free single acquisition, with a band-free pair) at ``drift_hz``."""
    single = pair = None
    for bw in grid:
        ok = allowed(cfg, world, pts, bw, drift_hz)
        if single is None and ok.all(0).any():
            single = float(bw)
        if pair is None and _pair(ok) is not None:
            pair = float(bw)
        if single is not None and (pair is not None or single <= stop_single_at):
            break
    return single, pair


def solve_image(cfg: dict, world: dict, pts: list) -> dict:
    """One image (``pts``: the true state of its regions). dict(single_bandwidth, pair_bandwidth: lowest band-free
    bandwidth of each kind at the reference drift, or None; needs_two: no single acquisition up to ``single_up_to`` at
    the reference drift; answer: the oracle's own protocol; acquisitions: of that answer).

    The answer is built at the drift halfway between the task's and the reference's: just above the lowest bandwidth
    that is band-free there, where the drift margin is widest. Whatever is band-free at the reference drift is
    band-free there too, so the answer is never above 1.07 times the lowest comparison protocol."""
    t = cfg["task"]
    lo, hi = t["bandwidth_hz_per_px"]
    grid = np.exp(np.arange(np.log(lo), np.log(hi) + 1e-9, 0.01))
    single, pair = _lowest(cfg, world, pts, grid, t["reference_drift_hz"], t["single_up_to"])
    work = 0.5 * (t["drift_hz"] + t["reference_drift_hz"])
    w_single, w_pair = _lowest(cfg, world, pts, grid, work, t["single_up_to"])
    kind, lowest = ("single", w_single) if w_single is not None and (w_single <= t["single_up_to"] or w_pair is None) else ("pair", w_pair)
    answer = None
    if lowest is not None:
        best = None
        for bw in grid[(grid >= lowest) & (grid <= 1.07 * lowest)][::2]:
            for drift in np.arange(work, 60.0, 2.0):                                     # the widest margin this bandwidth has
                ok = allowed(cfg, world, pts, bw, drift)
                found = [_middle(ok.all(0))] if kind == "single" and ok.all(0).any() else _pair(ok) if kind == "pair" else None
                if not found:
                    break
                if best is None or drift > best[0]:
                    best = (drift, float(bw), [float(i * STEP_DEG) for i in found])
        answer = dict(bandwidth_hz_per_px=round(best[1], 1), phase_cycle_deg=best[2])
    return dict(single_bandwidth=single, pair_bandwidth=pair, needs_two=single is None or single > t["single_up_to"],
                acquisitions=1 if kind == "single" else 2, answer=answer)


@O.timed
def solve(cfg: dict, world: dict | None = None) -> dict:
    """The oracle of an instance, by the contract of ``mrgym.envs.oracle``: ``argmin`` (also under ``answer``) is the
    oracle's deliverable, None when an image has no band-free protocol within the allowed bandwidths; ``value`` its
    error in the oracle's arithmetic; ``images``: per image what ``solve_image`` returns; ``certificate``: the same
    answer with one image half a turn of the phase cycle away, which puts its regions in the band."""
    world = world or W.make_world(cfg["seed"], cfg["world_spec"])
    pts = points(cfg, world)
    groups = {name: [pts[i] for i in ids] for name, ids in cfg["task"]["images"].items()}
    out = {name: solve_image(cfg, world, group) for name, group in groups.items()}
    answers = {name: s["answer"] for name, s in out.items()}
    answer = answers if all(answers.values()) else None
    lo, hi = cfg["task"]["bandwidth_hz_per_px"]
    n_bw, n_phi = int(np.log(hi / lo) / 0.01) + 1, int(round(360.0 / STEP_DEG))
    value, certificate = None, []
    if answer:
        value = round(1.0 - min(kept(cfg, world, groups[name], a["bandwidth_hz_per_px"], a["phase_cycle_deg"]) for name, a in answer.items()), 5)
        certificate = [dict(deliverable={**answer, name: dict(a, phase_cycle_deg=[(p + 180.0) % 360.0 for p in a["phase_cycle_deg"]])},
                            fails=f"band_free_{name}") for name, a in answer.items()]
    return dict(value=value, argmin=answer, answer=answer, images=out, method="exhaustive",
                grid=dict(axes=dict(bandwidth_hz_per_px=n_bw, phase_cycle_deg=n_phi, image=len(groups)), size=n_bw * n_phi * len(groups),
                          evaluated=n_bw * n_phi * len(groups)),
                certificate=certificate)
