"""The hidden world: one definition that every sequence images.

A world is a 2D slice on a master grid that never changes with the protocol:
  anatomy    soft partial-volume weights of tissue classes (``anatomy.brain``, ``anatomy.abdomen``, ``anatomy.qa``;
             two reference objects with published dimensions or values: ``anatomy.acr``, ``anatomy.nist_t1``)
  tissue     per class: PD, T1, T2, T2' (T2* derived), diffusion coefficient, fat or water (``tissue.table``)
  fields     dB0 in Hz and its through-slice gradient, B1+ ratio, complex receive profile, all on the master grid
  system     hidden state of the scanner for this session: centre-frequency offset, transmit-gain error, inversion
             efficiency, receiver noise density, interference line, gradient delay, fat-saturation efficiency, the
             error of the fat-saturation pulse's centre frequency and the bandwidth of that pulse; on request a
             calibration error of the two in-plane gradients (``grad_scale``)
  motion     optional: tissue classes whose signal pulses or whose position shifts periodically (``make_motion``)
Public (told to the agent by ``scanner_info``): field strength, the RF power limit, console defaults.

A world is evaluated on the upright master grid. The scanner turns its field of view by asking for the same world on
a grid turned about the isocentre (``turned``): geometry and fields are functions of the patient coordinates, so
nothing is interpolated. What does not live on the grid (tissue values, system state, regions) is shared with the
upright world. Array edits made to a world after it was built have to be registered in ``world["post_build"]`` (a
list of functions of the world, see ``grid``) to hold on a turned grid as well.

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
KINDS = ("brain", "abdomen", "qa", "acr", "nist_t1", "head3d", "qa3d")     # the last two are volumes: ``volume.py``
GRID = dict(brain=(640, 256.0), abdomen=(640, 420.0), qa=(640, 256.0),      # master grid points, extent in mm
            acr=(1024, 256.0), nist_t1=(640, 256.0))                        # acr: 0.25 mm, for the 0.9 mm holes of its resolution insert
EXAM_FOV_MM = dict(brain=240.0, abdomen=400.0, qa=240.0, acr=250.0, nist_t1=250.0)   # console default field of view
B1_SHAPE = {"brain": {1.5: 0.03, 3.0: 0.11}, "abdomen": {1.5: 0.07, 3.0: 0.26}}
B1RMS_LIMIT_UT = {1.5: 6.0, 3.0: 3.0}                                        # public; the RF power figure is relative to it

DEFAULT_SPEC = dict(
    kind="qa", b0_t=3.0, lesion=None, qa=None,
    # QA phantom fields (anatomical kinds use the dipole and field-strength models below)
    b1_shape_amp=(0.01, 0.04), b0_hz=40.0,
    # system state
    tx_gain_dev=(0.03, 0.12), cf_offset_hz=15.0, inv_eff=(0.92, 0.98), sigma0=0.030, sigma0_log_sd=0.1,
    rfi=None, grad_delay_us=0.4, fatsat_eff=(0.88, 0.97), fatsat_offset_hz=0.0,
    fatsat_bw_hz=None,      # bandwidth of the fat-saturation pulse at 3 T: None = 1.3 ppm (166 Hz), a number, or a (low, high) range
    effects=None,
    # optional physics (None = absent; none of them draws from a stream another component uses)
    motion=None,            # list of movers, see ``make_motion``
    edge_mm=None,           # half-width of the soft boundaries (None = anatomy.EDGE_MM)
    gnl=None,               # gradient non-linearity: positions scaled by 1 - gnl * (r / 250 mm)^2
    dbdz=None,              # QA phantom: through-slice field gradient of a few smooth patches, see ``make_dbdz``
)
# Additive key that is not listed above (a spec without it is the spec of before): ``array`` = dict, a receive array
# around the object (``array.SPEC``). The world then has ``world["array"]``; nothing else in it changes.
MOTION_HZ = dict(respiration=(0.2, 0.35), cardiac=(0.9, 1.5))


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


def anatomy_fields(ph, rng, rot_deg=0.0, warp=None, shim=None):
    """dB0 from air cavities below the slice after a first-order shim over the organ, its through-slice gradient,
    a B1+ pattern whose size depends on the field strength, and a receive profile. Excludes the system offsets.
    ``shim``: the shim of the upright world (``fields["shim"]``), reused on a turned grid."""
    n, ext = ph["n"], ph["extent_mm"]
    y, x = anatomy.axes(n, ext, rot_deg, warp)
    raw = dipole_hz(ph, y, x, 0.0)
    dz = (dipole_hz(ph, y, x, 1.0) - dipole_hz(ph, y, x, -1.0)) / 2.0
    m = ph["masks"]["brain"]
    a = np.stack([np.ones(n * n), y.ravel(), x.ravel()], 1)
    if shim is None:
        coef = np.linalg.lstsq(a[m.ravel()], raw[m], rcond=None)[0] if m.any() and ph["air"] else np.zeros(3)
        shim = dict(coef=[float(v) for v in coef], dz_mean=float(dz[m].mean()) if m.any() else 0.0)
    b0 = raw - (a @ np.array(shim["coef"])).reshape(n, n)
    dz = dz - shim["dz_mean"]
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
    return dict(b0_hz=b0, dbdz_hz_per_mm=dz, b1=b1, rx=rxm * np.exp(1j * rxp), shim=shim)


def qa_fields(ph, rng, sp, rot_deg=0.0, warp=None):
    """Smooth closed-form fields over a test object: a bowl-shaped B1+ pattern with a tilt, a linear plus saddle dB0,
    a linear receive profile with a phase ramp. Lengths are scaled by 240 mm, the exam field of view."""
    y, x = anatomy.axes(ph["n"], ph["extent_mm"], rot_deg, warp)
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


# Fields over the ACR phantom. Sizes are declared (there is no measurement behind them): a transmit pattern that
# brightens the centre, more at 3 T (the guidance names the dielectric effect of the conductive solution, section 5.1),
# a receive profile that is a little brighter towards the coil at 1.5 T and towards the centre at 3 T, a small
# residual field gradient. The three fault knobs are sizes in the same units: ``b1_fault`` is a hump (or a dip) of the
# transmit field near the centre, 43 mm wide, that leaves the rim alone; ``rx_fault`` is a linear fall of the
# receive sensitivity across the phantom (a dead coil element); ``shim_hz_per_mm``
# a linear field gradient (y, x) left by a wrong shim. Lengths are scaled by the radius of the phantom, 95 mm.
ACR_FIELDS = dict(b1_amp={1.5: 0.03, 3.0: 0.15}, rx_bowl={1.5: -0.06, 3.0: 0.22}, tilt=0.012, b0_ppm=0.02,
                  b1_fault=None, rx_fault=None, shim_hz_per_mm=None)


def acr_fields(ph, rng, sp, rot_deg=0.0, warp=None):
    """Closed-form fields over the ACR phantom (``ACR_FIELDS``; ``sp["acr_fields"]`` overrides). Every draw is made
    whether or not a fault is asked for, so a fault changes its own term and nothing else."""
    a = dict(ACR_FIELDS, **(sp.get("acr_fields") or {}))
    b0_t = ph["b0_t"]
    y, x = anatomy.axes(ph["n"], ph["extent_mm"], rot_deg, warp)
    v, u = y / 95.0, x / 95.0
    rho2 = v * v + u * u
    amp = a["b1_amp"][b0_t] * rng.uniform(0.8, 1.2)
    t = rng.uniform(-1, 1, 6) * a["tilt"]
    ang = rng.uniform(0, 2 * np.pi, 2)                                  # direction of the transmit fault, of the receive fault
    b1 = 1 + amp * (0.5 - rho2) + t[0] * v + t[1] * u
    rxm = 1 + a["rx_bowl"][b0_t] * rng.uniform(0.8, 1.2) * (0.5 - rho2) + t[2] * v + t[3] * u
    ph0, py, px = rng.uniform(0, 2 * np.pi), rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)
    g = rng.uniform(-1, 1, 2) * a["b0_ppm"] * 1e-6 * GAMMA_HZ_PER_T * b0_t / 95.0    # Hz per mm
    if a["b1_fault"]:
        f = _draw(rng, a["b1_fault"])
        b1 = b1 + f * np.exp(-((v - 0.2 * np.sin(ang[0])) ** 2 + (u - 0.2 * np.cos(ang[0])) ** 2) / (2 * 0.45 ** 2))
    if a["rx_fault"]:
        f = _draw(rng, a["rx_fault"])
        rxm = rxm * (1 + f * (np.sin(ang[1]) * v + np.cos(ang[1]) * u) / 2) / (1 + abs(f) / 2)
    if a["shim_hz_per_mm"]:
        g = g + np.array([_draw(rng, c) for c in a["shim_hz_per_mm"]])
    b0 = g[0] * y + g[1] * x
    rx = np.clip(rxm, 0.05, 3.0) * np.exp(1j * (ph0 + py * v + px * u))
    return dict(b0_hz=b0, dbdz_hz_per_mm=np.zeros_like(b0), b1=np.clip(b1, 0.3, 1.7), rx=rx)


def make_system(ph, rng, sp, extra=None):
    """Hidden state of the scanner for one session. Sizes are choices of this package, not measurements.

    ``extra``: a callable that returns the generator of the additive keys (the sub-stream ``system``, key 1). A key
    that is absent draws nothing, so a world without it is bit-identical to one built before the key existed."""
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
    # drawn last, so that the draws above are those of a world without it: the centre of the fat-saturation pulse
    # misses the fat line by up to ``fatsat_offset_hz`` (at 3 T; it scales with the field like the frequency error)
    s["fatsat_offset_hz"] = float(rng.uniform(-1, 1) * sp["fatsat_offset_hz"] * b0_t / 3.0)
    # additive: the bandwidth of the fat-saturation pulse of this scanner (given at 3 T; it scales with the field like the
    # chemical shift it has to cover). A number, or a range drawn from a sub-stream of its own.
    if sp.get("fatsat_bw_hz") is not None:
        s["fatsat_bw_hz"] = float(_draw(extra() if extra else rng, sp["fatsat_bw_hz"]) * b0_t / 3.0)
    return s


def _draw(rng, v):
    """A number, or a (low, high) range drawn uniformly."""
    return float(rng.uniform(*v)) if isinstance(v, (tuple, list)) else float(v)


def make_motion(movers, ph, rng):
    """Periodic motion of a session, from ``spec["motion"]``: a list of dicts, one per mover.

        cls      the tissue class that moves: a class name, the id of a region (its class), or "*" for the whole object
        kind     "pulse": the signal of the class is multiplied by 1 + depth cos(2 pi f t + phase)   (pulsatile flow)
                 "shift": the class is displaced by amp_mm cos(2 pi f t + phase) along ``dir``        (respiration)
        f_hz     a number, a (low, high) range, or "respiration" (0.2 to 0.35 Hz) / "cardiac" (0.9 to 1.5 Hz)
        depth    0.1 to 0.6 by default (pulse);  amp_mm  1 to 8 mm by default, dir "y" or "x" (drawn if absent) (shift)
        phase_key  optional integer: the key of the mover's phase (drawn if absent)
    Ranges are drawn from the ``system`` stream after everything else in it. A declared model: the k-space lines of
    one class are modulated with the time of their acquisition (``kspace.Engine._motion``); there is no spin
    history, no inflow and nothing moves through the plane."""
    by_region = {r["id"]: r["cls"] for r in ph.get("regions", [])}
    out = []
    for m in movers:
        cls = by_region.get(m.get("cls", "*"), m.get("cls", "*"))
        if cls != "*" and cls not in ph["names"]:
            raise ValueError(f"motion: no tissue class or region {m.get('cls')!r} in this world")
        kind = m.get("kind", "pulse")
        if kind not in ("pulse", "shift"):
            raise ValueError("motion kind must be 'pulse' or 'shift'")
        f = m.get("f_hz", "respiration")
        mv = dict(cls=cls, kind=kind, f_hz=_draw(rng, MOTION_HZ[f] if isinstance(f, str) else f),
                  phase_key=int(rng.integers(0, 2 ** 32)))
        if "phase_key" in m:                                             # a caller that places a ghost fixes its phase (the draw above is kept)
            mv["phase_key"] = int(m["phase_key"])
        if kind == "pulse":
            mv["depth"] = _draw(rng, m.get("depth", (0.1, 0.6)))
        else:
            mv["amp_mm"] = _draw(rng, m.get("amp_mm", (1.0, 8.0)))
            mv["dir"] = m["dir"] if m.get("dir") in ("y", "x") else ("y", "x")[int(rng.integers(0, 2))]
        out.append(mv)
    return out


def make_dbdz(spec, ph, rng, rot_deg=0.0, warp=None):
    """Through-slice field gradient (Hz/mm) of a test object, from ``spec["dbdz"]``: a sum of Gaussian patches, what
    an air bubble or a susceptibility insert above or below the slice leaves behind.

        n                 number of patches, a number or a (low, high) range of integers
        peak_hz_per_mm    gradient at the centre of a patch, a number or a (low, high) range (the sign is drawn)
        sigma_mm          width of a patch, a number or a (low, high) range
        patches           optional list of placed patches, added to the drawn ones: dicts ``y_mm, x_mm,
                          peak_hz_per_mm`` (signed), ``sigma_mm`` and optionally ``flat_mm`` (the gradient keeps its
                          peak value within this distance of the centre and falls off as a Gaussian beyond it)
    Patch centres are uniform inside 0.8 of the object's smaller half-axis. Drawn from a stream of its own
    (``fields``, key 1), so a world without ``dbdz`` is unchanged; placed patches draw nothing."""
    y, x = anatomy.axes(ph["n"], ph["extent_mm"], rot_deg, warp)
    n = spec.get("n", 2)
    n = int(rng.integers(n[0], n[1] + 1)) if isinstance(n, (tuple, list)) else int(n)
    reach = 0.8 * min(ph["size_mm"]) / 2
    out = np.zeros(y.shape)
    for _ in range(n):
        r, a = reach * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        peak, sigma = _draw(rng, spec.get("peak_hz_per_mm", (3.0, 6.0))), _draw(rng, spec.get("sigma_mm", (10.0, 16.0)))
        sign = 1.0 if rng.uniform() < 0.5 else -1.0
        out += sign * peak * np.exp(-((y - r * np.sin(a)) ** 2 + (x - r * np.cos(a)) ** 2) / (2 * sigma ** 2))
    for p in spec.get("patches") or ():
        far = np.maximum(np.hypot(y - p["y_mm"], x - p["x_mm"]) - float(p.get("flat_mm", 0.0)), 0.0)
        out += float(p["peak_hz_per_mm"]) * np.exp(-far ** 2 / (2 * float(p["sigma_mm"]) ** 2))
    return out


def _warp(gnl, grad_scale=None):
    """Coordinate map of a gradient non-linearity of relative size ``gnl`` at 250 mm from the isocentre and of a
    gradient calibration error ``grad_scale`` = (s_y, s_x): a gradient that is s times its nominal strength encodes
    the position r as s r, so the image point r shows the object point r / s. (A declared simplification: the image
    intensity is not divided by s as the k-space scaling would: a relative error of the size of the calibration error.)"""
    if not gnl and not grad_scale:
        return None
    if not grad_scale:
        return lambda y, x: [v * (1.0 - gnl * (y * y + x * x) / 250.0 ** 2) for v in (y, x)]
    sy, sx = grad_scale
    g = gnl or 0.0
    return lambda y, x: [v * (1.0 - g * (y * y + x * x) / 250.0 ** 2) / s for v, s in ((y, sy), (x, sx))]


def grad_scale_of(seed: str, sp: dict):
    """The gradient calibration error of a session, (s_y, s_x), or None when the spec asks for none.
    ``sp["grad_scale"]``: dict(y=, x=), each a number or a (low, high) range of the relative error (0.01 = the
    gradient is 1 % too strong, the object appears 1 % longer along that axis). Drawn from a sub-stream of its own."""
    gs = sp.get("grad_scale")
    if not gs:
        return None
    rng = rng_for(seed, "system", 1)
    return tuple(1.0 + (_draw(rng, gs[k]) if gs.get(k) is not None else 0.0) for k in ("y", "x"))


def grid(world: dict):
    """Patient coordinates (y, x) in mm of the grid points of a world (upright, or turned by ``world["rot_deg"]``)."""
    return anatomy.axes(world["n"], world["extent_mm"], world.get("rot_deg", 0.0),
                        _warp(world.get("gnl"), (world.get("system") or {}).get("grad_scale")))


def make_world(seed: str, spec: dict | None = None, layout_only: bool = False) -> dict:
    """Build the hidden world of one session from the secret seed and a world spec (see ``DEFAULT_SPEC``).

    ``layout_only`` (QA phantom): regions and tissue values without the master-grid arrays. The draws are the same,
    so the verifier can read the truth of an instance in a millisecond."""
    return _build(seed, dict(DEFAULT_SPEC, **(spec or {})), layout_only)


def _build(seed, sp, layout_only=False, rot_deg=0.0, base=None):
    """The world of (seed, full spec) on a grid turned by ``rot_deg``; ``base`` is the upright world when turning."""
    kind, b0_t = sp["kind"], float(sp["b0_t"])
    if kind not in KINDS:
        raise ValueError(f"world kind must be one of {KINDS}")
    if b0_t not in B1RMS_LIMIT_UT:
        raise ValueError("field strength must be 1.5 or 3.0 T")
    if kind in ("head3d", "qa3d"):                                      # a volume: its own module, before any draw of the 2D kinds
        from . import volume
        return volume.build(seed, sp, layout_only, rot_deg, base)
    n, extent = GRID[kind]
    n = int(sp.get("n") or n)
    g, t, f, s = (rng_for(seed, k) for k in ("geometry", "tissue", "fields", "system"))
    # Additive keys of the system state. Known limit: ``fatsat_bw_hz`` and ``grad_scale`` both draw from (system, 1), so a
    # world spec with BOTH keys would get correlated draws. No family sets both; a family that needs both must move one
    # of them to its own sub-stream (which changes that family's worlds and voids its stored gate).
    extra = lambda: rng_for(seed, "system", 1)                          # noqa: E731
    scale = grad_scale_of(seed, sp)
    # The half-width of the soft boundaries is an argument of the build. (It used to be written into
    # ``anatomy.EDGE_MM`` for the duration of the call: a build in another thread then read this build's value, and
    # restored it for good. A world must be a function of (seed, spec) alone: tests/test_thread_determinism.py.)
    warp, edge = _warp(sp.get("gnl"), scale), float(sp.get("edge_mm") or anatomy.EDGE_MM)
    if kind == "qa":
        ph = anatomy.qa(g, b0_t=b0_t, n=n, extent_mm=extent, spec=sp["qa"], tissue_rng=t, layout_only=layout_only,
                        rot_deg=rot_deg, warp=warp, edge_mm=edge)
        if layout_only:
            return dict(ph, schema=SCHEMA, system=make_system(ph, s, sp, extra))
        fields = qa_fields(ph, f, sp, rot_deg, warp)
    elif kind == "acr":
        ph = anatomy.acr(g, b0_t=b0_t, n=n, extent_mm=extent, spec=sp.get("acr"), tissue_rng=t, rot_deg=rot_deg, warp=warp,
                         edge_mm=edge)
        fields = acr_fields(ph, f, sp, rot_deg, warp)
    elif kind == "nist_t1":
        ph = anatomy.nist_t1(g, b0_t=b0_t, n=n, extent_mm=extent, spec=None, tissue_rng=t, rot_deg=rot_deg, warp=warp,
                             edge_mm=edge)
        fields = qa_fields(ph, f, sp, rot_deg, warp)
    else:
        more = dict(picks=base["picks"]) if base is not None and kind == "brain" else {}
        ph = getattr(anatomy, kind)(g, b0_t=b0_t, n=n, extent_mm=extent, lesion=sp["lesion"], tissue_rng=t, rot_deg=rot_deg,
                                    warp=warp, edge_mm=edge, **more)
        fields = anatomy_fields(ph, f, rot_deg, warp, shim=base["fields"]["shim"] if base is not None else None)
    if sp.get("dbdz") and kind == "qa":
        fields["dbdz_hz_per_mm"] = make_dbdz(sp["dbdz"], ph, rng_for(seed, "fields", 1), rot_deg, warp)
    w = dict(ph, schema=SCHEMA, fields=fields, system=make_system(ph, s, sp, extra))
    if scale:
        w["system"]["grad_scale"] = scale
    w["effects"] = None if sp["effects"] is None else sorted(sp["effects"])
    w["console"] = dict(fov_mm=EXAM_FOV_MM[kind], min_pixel_mm=round(2.0 * extent / n, 3),
                        b1rms_limit_ut=B1RMS_LIMIT_UT[b0_t], b0_t=b0_t)
    if sp.get("console_system"):                                        # console constants of this scanner (e.g. overhead_s per scan)
        w["console"]["system"] = dict(sp["console_system"])
    w.setdefault("regions", [])
    if sp.get("motion"):
        w["motion"] = make_motion(sp["motion"], w, s)
    if sp.get("array"):
        # additive: receive elements around the object, drawn from a generator of their own (``array.make_array``)
        from . import array as _array
        w["array"] = (base["array"] if base is not None and base.get("array")
                      else _array.make_array(sp["array"], w, rng_for(seed, "fields", 2)))
    w.update(rot_deg=float(rot_deg), gnl=sp.get("gnl"), source=dict(kind="seed", seed=seed, spec=sp))
    return w


def turned(world: dict, rot_deg: float) -> dict:
    """The same world on a grid turned by ``rot_deg`` about the isocentre (image rows from +y towards +x).

    Geometry and fields are re-evaluated at the patient coordinates of the turned grid points; nothing is
    interpolated. Everything that does not live on the grid is the upright world's own object, so a change made to
    it holds at every angle. Array edits registered in ``world["post_build"]`` are applied again."""
    src = world.get("source")
    if not src:
        raise ValueError("this world has no recipe (seed and spec, or shapes) from which it can be evaluated on a turned grid")
    if src["kind"] == "seed":
        t = _build(src["seed"], src["spec"], rot_deg=rot_deg, base=world)
    else:
        if any(np.ptp(np.abs(v)) > 0 or np.ptp(np.angle(v)) > 0 for k, v in world["fields"].items() if isinstance(v, np.ndarray)):
            raise ValueError("the fields of this hand-made world are not uniform: it cannot be evaluated on a turned grid")
        ph = shapes_phantom(**src["phantom"], rot_deg=rot_deg)
        t = dict(world, weights=ph["weights"], masks=ph["masks"])
    for k in ("names", "tissues", "system", "effects", "console", "regions", "motion"):
        if k in world:
            t[k] = world[k]
    t.update(rot_deg=float(rot_deg), source=None)
    for fn in world.get("post_build", ()):
        fn(t)
    return t


def flat_world(ph: dict, **system) -> dict:
    """A world with perfect fields and an ideal, noiseless system around a phantom dict (for checks and examples)."""
    n = ph["n"]
    w = dict(ph, schema=SCHEMA)
    w["fields"] = dict(b0_hz=np.zeros((n, n)), dbdz_hz_per_mm=np.zeros((n, n)), b1=np.ones((n, n)), rx=np.ones((n, n), complex))
    w["system"] = dict(cf_offset_hz=0.0, tx_gain=1.0, inv_eff=1.0, sigma0=0.0, rfi=None, grad_delay_us=0.0,
                       fatsat_eff=1.0, fatsat_bw_hz=200.0, fatsat_offset_hz=0.0)
    w["system"].update(system)
    w["effects"] = None
    w["console"] = dict(fov_mm=240.0, min_pixel_mm=round(2.0 * ph["extent_mm"] / n, 3), b1rms_limit_ut=1e3, b0_t=ph["b0_t"])
    w.setdefault("regions", [])
    return w


def shapes_phantom(shapes, b0_t=3.0, n=640, extent_mm=256.0, tissues=None, edge=anatomy.EDGE_MM, rot_deg=0.0) -> dict:
    """Hand-made phantom for checks. ``shapes``: list of (tissue name, callable(y_mm, x_mm) -> signed distance in mm,
    negative inside). Later shapes sit on top of earlier ones. ``rot_deg``: evaluate on a turned grid."""
    from . import tissue as T
    y, x = anatomy.axes(n, extent_mm, rot_deg)
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
                air=[], masks=dict(body=body, brain=body, lesion=np.zeros((n, n), np.float32)), size_mm=(200.0, 200.0),
                rot_deg=float(rot_deg),
                source=dict(kind="shapes", phantom=dict(shapes=shapes, b0_t=b0_t, n=n, extent_mm=extent_mm, tissues=tissues, edge=edge)))


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
