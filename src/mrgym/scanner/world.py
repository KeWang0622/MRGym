"""The hidden world: one definition that every sequence images.

A world is a 2D slice on a master grid that never changes with the protocol:
  anatomy    soft partial-volume weights of tissue classes (``anatomy.brain``, ``anatomy.abdomen``, ``anatomy.qa``)
  tissue     per class: PD, T1, T2, T2' (T2* derived), diffusion coefficient, fat or water (``tissue.table``)
  fields     dB0 in Hz and its through-slice gradient, B1+ ratio, complex receive profile, all on the master grid
  system     hidden state of the scanner for this session: centre-frequency offset, transmit-gain error, inversion
             efficiency, receiver noise density, interference line, gradient delay, fat-saturation efficiency
Public (told to the agent by ``scanner_info``): field strength, the RF power limit, console defaults.

Randomness: a 128-bit secret seed and one independent sub-stream per component (geometry, tissue, fields, system,
noise, task), so that what is visible in an image (geometry) says nothing about the other draws.

Positions are in mm from the isocentre, arrays are [y, x], index i of an n-point axis sits at (i - n // 2) * extent / n.
dB0 of an air sphere of radius a in a field along z (Schenck 1996): outside the sphere
dB / B0 = (dchi / 3) (a / r)^3 (3 cos^2(theta) - 1), dchi(air - tissue) = 9.4 ppm.
"""
from __future__ import annotations

import numpy as np

from . import anatomy

SCHEMA = 3
GAMMA_HZ_PER_T = 42.577478518e6
DCHI_AIR = 9.4e-6
STREAMS = dict(geometry=0, tissue=1, fields=2, system=3, noise=4, task=5)
KINDS = ("brain", "abdomen", "qa")
GRID = dict(brain=(640, 256.0), abdomen=(640, 420.0), qa=(640, 256.0))      # master grid points, extent in mm
EXAM_FOV_MM = dict(brain=240.0, abdomen=400.0, qa=240.0)                    # console default field of view
B1_SHAPE = {"brain": {1.5: 0.03, 3.0: 0.11}, "abdomen": {1.5: 0.07, 3.0: 0.26}}
B1RMS_LIMIT_UT = {1.5: 6.0, 3.0: 3.0}                                        # public; the RF power figure is relative to it

DEFAULT_SPEC = dict(
    kind="qa", b0_t=3.0, lesion=None, qa=None,
    # QA phantom fields (anatomical kinds use the dipole and field-strength models below)
    b1_shape_amp=(0.01, 0.04), b0_hz=40.0,
    # system state
    tx_gain_dev=(0.03, 0.12), cf_offset_hz=15.0, inv_eff=(0.92, 0.98), sigma0=0.030, sigma0_log_sd=0.1,
    rfi=None, grad_delay_us=0.4, fatsat_eff=(0.88, 0.97),
    effects=None,
)


def seed_int(seed: str) -> int:
    """The secret seed is 32 hex digits (128 bits). Anything else is refused: a short seed is a guessable seed."""
    if not (isinstance(seed, str) and len(seed) == 32):
        raise ValueError("seed must be a string of 32 hex digits")
    return int(seed, 16)


def rng_for(seed: str, stream: str, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=seed_int(seed), spawn_key=(STREAMS[stream], *key)))


# ------------------------------------------------------------------------------------------------ fields
def dipole_hz(ph, y, x, z):
    f0 = GAMMA_HZ_PER_T * ph["b0_t"]
    out = np.zeros(np.broadcast(y, x).shape)
    for ys, xs, zs, a in ph["air"]:
        r2 = np.maximum((y - ys) ** 2 + (x - xs) ** 2 + (z - zs) ** 2, a * a)
        out += f0 * DCHI_AIR / 3.0 * a ** 3 * (3 * (z - zs) ** 2 / r2 - 1) / r2 ** 1.5
    return out


def anatomy_fields(ph, rng):
    """dB0 from air cavities below the slice after a first-order shim over the organ, its through-slice gradient,
    a B1+ pattern whose size depends on the field strength, and a receive profile. Excludes the system offsets."""
    n, ext = ph["n"], ph["extent_mm"]
    y, x = anatomy.axes(n, ext)
    raw = dipole_hz(ph, y, x, 0.0)
    dz = (dipole_hz(ph, y, x, 1.0) - dipole_hz(ph, y, x, -1.0)) / 2.0
    m = ph["masks"]["brain"]
    a = np.stack([np.ones(n * n), y.ravel(), x.ravel()], 1)
    coef = np.linalg.lstsq(a[m.ravel()], raw[m], rcond=None)[0] if m.any() and ph["air"] else np.zeros(3)
    b0 = raw - (a @ coef).reshape(n, n)
    dz = dz - (dz[m].mean() if m.any() else 0.0)
    b0 = b0 + anatomy.smooth_field(rng, 4, 150, 500, 0.02 * GAMMA_HZ_PER_T * ph["b0_t"] * 1e-6)(y, x)
    ry, rx_ = ph["size_mm"][0] / 2, ph["size_mm"][1] / 2
    rho2 = (y / ry) ** 2 + (x / rx_) ** 2
    amp = B1_SHAPE[ph["kind"]][ph["b0_t"]] * rng.uniform(0.8, 1.2)
    if ph["kind"] == "abdomen":                                        # quadrupolar shading of a wide body
        phi0 = rng.uniform(-0.5, 0.5)
        b1 = 1 + amp * rho2 * np.cos(2 * (np.arctan2(y / ry, x / rx_) - phi0)) + 0.3 * amp * (0.5 - rho2)
    else:                                                              # central brightening
        b1 = 1 + amp * (1 - 2 * rho2) + amp * 0.3 * (rng.uniform(-1, 1) * y / ry + rng.uniform(-1, 1) * x / rx_)
    b1 = np.clip(b1, 0.3, 1.7)
    rxm = 1 + rng.uniform(0.1, 0.3) * rho2 + rng.uniform(-0.1, 0.1) * y / ry + rng.uniform(-0.1, 0.1) * x / rx_
    rxp = rng.uniform(0, 2 * np.pi) + rng.uniform(-1, 1) * y / ry + rng.uniform(-1, 1) * x / rx_
    return dict(b0_hz=b0, dbdz_hz_per_mm=dz, b1=b1, rx=rxm * np.exp(1j * rxp))


def qa_fields(ph, rng, sp):
    """Smooth closed-form fields over a test object: a bowl-shaped B1+ pattern with a tilt, a linear plus saddle dB0,
    a linear receive profile with a phase ramp. Lengths are scaled by 240 mm, the exam field of view."""
    y, x = anatomy.axes(ph["n"], ph["extent_mm"])
    v, u = y / 240.0, x / 240.0
    amp = float(rng.uniform(*sp["b1_shape_amp"])) * (1 if rng.uniform() < 0.8 else -1)
    tilt = rng.uniform(-0.3, 0.3, 2) * abs(amp)
    cy, cx, sy, sx = rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(0.26, 0.34), rng.uniform(0.26, 0.34)
    b1 = 1 + amp * (0.5 - ((v - cy) / sy) ** 2 - ((u - cx) / sx) ** 2) + tilt[0] * v + tilt[1] * u
    g = rng.uniform(-1, 1, 3) * sp["b0_hz"]
    qy, qx = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
    b0 = g[0] * v + g[1] * u + g[2] * ((v - qy) ** 2 - (u - qx) ** 2) / 0.09
    r = rng.uniform(-0.4, 0.4, 2)
    ph0, py, px = rng.uniform(0, 2 * np.pi), rng.uniform(-1, 1), rng.uniform(-1, 1)
    rx = (1 + r[0] * v + r[1] * u) * np.exp(1j * (ph0 + py * v + px * u))
    return dict(b0_hz=b0, dbdz_hz_per_mm=np.zeros_like(b0), b1=np.clip(b1, 0.3, 1.7), rx=rx)


def make_system(ph, rng, sp):
    """Hidden state of the scanner for one session. Sizes are choices of this package, not measurements."""
    b0_t = ph["b0_t"]
    lo, hi = sp["tx_gain_dev"]
    dev = float(rng.uniform(lo, hi)) * (1 if rng.uniform() < 0.5 else -1)
    s = dict(
        cf_offset_hz=float(rng.uniform(-1, 1) * sp["cf_offset_hz"] * b0_t / 3.0),     # error of the frequency adjustment
        tx_gain=1.0 + dev,                                                           # error of the transmitter adjustment
        inv_eff=float(rng.uniform(*sp["inv_eff"])),
        sigma0=float(sp["sigma0"] * 3.0 / b0_t * np.exp(rng.normal(0, sp["sigma0_log_sd"]))),   # SNR is linear in B0
        grad_delay_us=float(rng.uniform(-1, 1) * sp["grad_delay_us"]),               # residual after the console correction
        fatsat_eff=float(rng.uniform(*sp["fatsat_eff"])), fatsat_bw_hz=float(1.3e-6 * GAMMA_HZ_PER_T * b0_t),
        rfi=None,
    )
    if sp["rfi"] == "random":
        s["rfi"] = dict(freq_hz=float(rng.uniform(-2e4, 2e4)), amp=float(rng.uniform(100, 400) * s["sigma0"]))
    elif isinstance(sp["rfi"], dict):
        s["rfi"] = dict(freq_hz=float(sp["rfi"]["freq_hz"]), amp=float(sp["rfi"]["amp"]))
    return s


def make_world(seed: str, spec: dict | None = None, layout_only: bool = False) -> dict:
    """Build the hidden world of one session from the secret seed and a world spec (see ``DEFAULT_SPEC``).

    ``layout_only`` (QA phantom): regions and tissue values without the master-grid arrays. The draws are the same,
    so the verifier can read the truth of an instance in a millisecond."""
    sp = dict(DEFAULT_SPEC, **(spec or {}))
    kind, b0_t = sp["kind"], float(sp["b0_t"])
    if kind not in KINDS:
        raise ValueError(f"world kind must be one of {KINDS}")
    if b0_t not in B1RMS_LIMIT_UT:
        raise ValueError("field strength must be 1.5 or 3.0 T")
    n, extent = GRID[kind]
    n = int(sp.get("n") or n)
    g, t, f, s = (rng_for(seed, k) for k in ("geometry", "tissue", "fields", "system"))
    if kind == "qa":
        ph = anatomy.qa(g, b0_t=b0_t, n=n, extent_mm=extent, spec=sp["qa"], tissue_rng=t, layout_only=layout_only)
        if layout_only:
            return dict(ph, schema=SCHEMA, system=make_system(ph, s, sp))
        fields = qa_fields(ph, f, sp)
    else:
        ph = getattr(anatomy, kind)(g, b0_t=b0_t, n=n, extent_mm=extent, lesion=sp["lesion"], tissue_rng=t)
        fields = anatomy_fields(ph, f)
    w = dict(ph, schema=SCHEMA, fields=fields, system=make_system(ph, s, sp))
    w["effects"] = None if sp["effects"] is None else sorted(sp["effects"])
    w["console"] = dict(fov_mm=EXAM_FOV_MM[kind], min_pixel_mm=round(2.0 * extent / n, 3),
                        b1rms_limit_ut=B1RMS_LIMIT_UT[b0_t], b0_t=b0_t)
    w.setdefault("regions", [])
    return w


def flat_world(ph: dict, **system) -> dict:
    """A world with perfect fields and an ideal, noiseless system around a phantom dict (for checks and examples)."""
    n = ph["n"]
    w = dict(ph, schema=SCHEMA)
    w["fields"] = dict(b0_hz=np.zeros((n, n)), dbdz_hz_per_mm=np.zeros((n, n)), b1=np.ones((n, n)), rx=np.ones((n, n), complex))
    w["system"] = dict(cf_offset_hz=0.0, tx_gain=1.0, inv_eff=1.0, sigma0=0.0, rfi=None, grad_delay_us=0.0,
                       fatsat_eff=1.0, fatsat_bw_hz=200.0)
    w["system"].update(system)
    w["effects"] = None
    w["console"] = dict(fov_mm=240.0, min_pixel_mm=round(2.0 * ph["extent_mm"] / n, 3), b1rms_limit_ut=1e3, b0_t=ph["b0_t"])
    w.setdefault("regions", [])
    return w


def shapes_phantom(shapes, b0_t=3.0, n=640, extent_mm=256.0, tissues=None, edge=anatomy.EDGE_MM) -> dict:
    """Hand-made phantom for checks. ``shapes``: list of (tissue name, callable(y_mm, x_mm) -> signed distance in mm,
    negative inside). Later shapes sit on top of earlier ones."""
    from . import tissue as T
    y, x = anatomy.axes(n, extent_mm)
    tab = dict(T.table(b0_t), **(tissues or {}))
    w = {}
    for name, f in shapes:
        m = anatomy.sig(-f(y, x), edge)
        for k in w:
            w[k] = w[k] * (1 - m)
        w[name] = w.get(name, 0) + m
    names = list(w)
    body = sum(w.values()) > 0.5
    return dict(kind="brain", b0_t=b0_t, n=n, extent_mm=extent_mm, names=names,
                weights=np.stack([w[k] for k in names]).astype(np.float32), tissues={k: dict(tab[k]) for k in names},
                air=[], masks=dict(body=body, brain=body, lesion=np.zeros((n, n), np.float32)), size_mm=(200.0, 200.0))


# ------------------------------------------------------------------------------------------------ what leaves the world
def regions_public(world: dict) -> list:
    """What the agent is told about the scored regions: id, centre and radius in mm. No tissue."""
    return [dict(id=r["id"], y_mm=r["y_mm"], x_mm=r["x_mm"], radius_mm=r["radius_mm"]) for r in world["regions"]]


def b1_at(world: dict, y_mm: float, x_mm: float) -> float:
    """Actual / nominal flip angle at a position, including the transmit-gain error (nearest master-grid point)."""
    n, d = world["n"], world["extent_mm"] / world["n"]
    i, j = int(round(y_mm / d)) + n // 2, int(round(x_mm / d)) + n // 2
    return float(world["fields"]["b1"][i, j] * world["system"]["tx_gain"])


def truth(world: dict) -> dict:
    """Scanner and verifier side only: the true tissue values of every scored region."""
    out = {}
    for r in world["regions"]:
        c = world["tissues"][r["cls"]]
        out[r["id"]] = dict(t1_ms=c["t1_ms"], t2_ms=c["t2_ms"], t2s_ms=c["t2s_ms"], pd=c["pd"], fat=c["fat"])
        if "fields" in world:
            out[r["id"]]["b1"] = b1_at(world, r["y_mm"], r["x_mm"])
    return out


def extent_mm(world: dict, threshold: float = 0.05) -> dict:
    """Bounding box of the object (proton-density weighted, above ``threshold`` of its maximum), in mm from the
    isocentre: dict(y=(lo, hi), x=(lo, hi)). Scanner and verifier side only."""
    pd = np.array([world["tissues"][k]["pd"] for k in world["names"]], np.float32)
    m = np.tensordot(pd, world["weights"], 1)
    m = m > threshold * m.max()
    d, n = world["extent_mm"] / world["n"], world["n"]
    ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    return dict(y=((ys[0] - n // 2 - 0.5) * d, (ys[-1] - n // 2 + 0.5) * d),
                x=((xs[0] - n // 2 - 0.5) * d, (xs[-1] - n // 2 + 0.5) * d))
