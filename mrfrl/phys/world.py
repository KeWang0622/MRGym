"""The hidden world: one definition that every tier projects.

A world is a 2D slice described without a grid:
  geometry   analytic shapes with the soft edges of ``mrfrl.sim`` (a body and inserts); each shape is one tissue class
  tissue     per class: PD, T1, T2, T2' (T2* is derived), apparent diffusion coefficient (needed by the spoiling model)
  fields     smooth closed-form functions of position: B1+ ratio, dB0 in Hz, complex receive profile
  system     transmit-gain error, centre-frequency offset, inversion efficiency, receiver noise level, RF power limit;
             hooks (present, inert, asserted inert by the engine): gradient delay, RF leak, coil faults
It is a plain JSON-serialisable dict so it can be hashed, logged server-side and exported.

Randomness: a 128-bit secret seed, and one independent sub-stream per component (geometry, tissue, fields, system,
noise, task), so that what is visible in an image (geometry) says nothing about the other draws.

Positions are in FOV units, (y, x), index i of an n-point axis at (i - n // 2) / n (``mrfrl.sim`` conventions).
"""
import hashlib
import json

import numpy as np

from .. import sim

SCHEMA = 2
FOV_MM = 240.0
SLICE_MM = 5.0
ROI_RADIUS = 0.048           # scored circular region, FOV units (11.5 mm): 66 pixels at the default 96 matrix
STREAMS = dict(geometry=0, tissue=1, fields=2, system=3, noise=4, task=5)

DEFAULT_SPEC = dict(
    n_ring=(4, 6), p_centre=0.5,
    insert_t1_ms=(300.0, 2500.0), insert_t2_ms=(40.0, 200.0), insert_t2p_ms=(40.0, 300.0), insert_pd=(0.55, 1.0),
    insert_adc_um2_ms=(0.6, 2.2), t2_over_t1_max=0.6,
    b1_shape_amp=(0.01, 0.04), tx_gain_dev=(0.08, 0.25), b0_hz=40.0, noise_sd=(0.008, 0.012),
)


def seed_int(seed):
    """The secret seed is 32 hex digits (128 bits). Anything else is refused: a short seed is a guessable seed."""
    if not (isinstance(seed, str) and len(seed) == 32):
        raise ValueError("seed must be a string of 32 hex digits")
    return int(seed, 16)


def rng_for(seed, stream, *key):
    return np.random.default_rng(np.random.SeedSequence(entropy=seed_int(seed), spawn_key=(STREAMS[stream], *key)))


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def make_world(seed, spec=None):
    sp = dict(DEFAULT_SPEC, **(spec or {}))
    g, t, f, s = (rng_for(seed, k) for k in ("geometry", "tissue", "fields", "system"))

    # ---- geometry: a body and well separated inserts, each large enough to hold the scored region with margin
    body = dict(y=float(g.uniform(-0.015, 0.015)), x=float(g.uniform(-0.015, 0.015)), a=float(g.uniform(0.37, 0.41)),
                b=float(g.uniform(0.37, 0.41)), th=float(g.uniform(0, np.pi)))
    n_ring = int(g.integers(sp["n_ring"][0], sp["n_ring"][1] + 1))
    ring, start = float(g.uniform(0.22, 0.245)), float(g.uniform(0, 2 * np.pi))
    centres = [(body["y"] + ring * np.sin(a), body["x"] + ring * np.cos(a))
               for a in start + 2 * np.pi * np.arange(n_ring) / n_ring + g.uniform(-0.06, 0.06, n_ring)]
    if g.uniform() < sp["p_centre"]:
        centres.append((body["y"] + float(g.uniform(-0.01, 0.01)), body["x"] + float(g.uniform(-0.01, 0.01))))
    shapes = [body]
    for cy, cx in centres:
        r, e = g.uniform(0.074, 0.084), g.uniform(-0.05, 0.05)
        shapes.append(dict(y=float(cy), x=float(cx), a=float(r * (1 + e)), b=float(r * (1 - e)), th=float(g.uniform(0, np.pi))))
    order = g.permutation(len(centres))                   # region ids carry no information about the layout draw order

    # ---- tissue classes (class 0 = body). Phantom-like: values are arbitrary, not anatomical
    classes = [dict(name="body", shape=body, pd=float(t.uniform(0.7, 0.9)), t1_ms=float(t.uniform(700, 1300)),
                    t2_ms=float(t.uniform(50, 110)), t2p_ms=float(t.uniform(60, 200)), adc_um2_ms=float(t.uniform(0.8, 2.0)))]
    for i in range(len(centres)):
        t1 = _logu(t, *sp["insert_t1_ms"])
        classes.append(dict(name=f"insert{i}", shape=shapes[1 + i], pd=float(t.uniform(*sp["insert_pd"])), t1_ms=t1,
                            t2_ms=min(_logu(t, *sp["insert_t2_ms"]), sp["t2_over_t1_max"] * t1),
                            t2p_ms=_logu(t, *sp["insert_t2p_ms"]), adc_um2_ms=float(t.uniform(*sp["insert_adc_um2_ms"]))))
    regions = [dict(id=f"R{k + 1}", cls=1 + int(i), y=shapes[1 + i]["y"], x=shapes[1 + i]["x"], r=ROI_RADIUS)
               for k, i in enumerate(order)]

    # ---- fields
    amp = float(f.uniform(*sp["b1_shape_amp"])) * (1 if f.uniform() < 0.8 else -1)
    tilt = f.uniform(-0.3, 0.3, 2) * abs(amp)
    b1 = dict(amp=amp, cy=float(f.uniform(-0.08, 0.08)), cx=float(f.uniform(-0.08, 0.08)),
              sy=float(f.uniform(0.26, 0.34)), sx=float(f.uniform(0.26, 0.34)), gy=float(tilt[0]), gx=float(tilt[1]))
    b0 = dict(gy=float(f.uniform(-1, 1) * sp["b0_hz"]), gx=float(f.uniform(-1, 1) * sp["b0_hz"]),
              q=float(f.uniform(-1, 1) * sp["b0_hz"]), cy=float(f.uniform(-0.1, 0.1)), cx=float(f.uniform(-0.1, 0.1)))
    rx = dict(gy=float(f.uniform(-0.4, 0.4)), gx=float(f.uniform(-0.4, 0.4)), ph=float(f.uniform(0, 2 * np.pi)),
              py=float(f.uniform(-1, 1)), px=float(f.uniform(-1, 1)))

    # ---- system state
    dev = float(s.uniform(*sp["tx_gain_dev"])) * (1 if s.uniform() < 0.5 else -1)
    system = dict(tx_gain=1.0 + dev, cf_offset_hz=float(s.uniform(-25, 25)), inv_eff=float(s.uniform(0.90, 0.98)),
                  noise_sd=_logu(s, *sp["noise_sd"]), b1rms_limit_ut=float(s.uniform(1.6, 2.4)),
                  grad_delay_us=0.0, rf_leak=None, coil_faults=[])          # hooks, not implemented
    return dict(schema=SCHEMA, fov_mm=FOV_MM, slice_mm=SLICE_MM, classes=classes, regions=regions,
                fields=dict(b1=b1, b0=b0, rx=rx), system=system)


def world_hash(world):
    return hashlib.sha256(json.dumps(world, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------ evaluation at arbitrary positions
def class_weights(world, y, x):
    """Soft partial-volume weights [n_class, ...]: inserts sit on top of the body; outside everything all are 0."""
    ind = [sim.eval_object([dict(c["shape"], amp=1.0)], y, x) for c in world["classes"]]
    inserts = np.clip(np.sum(ind[1:], axis=0), 0, 1) if len(ind) > 1 else 0.0
    return np.array([ind[0] * (1 - inserts)] + ind[1:])


def b1(world, y, x):
    """B1+ ratio (actual / nominal flip angle), including the global transmit-gain error."""
    f = world["fields"]["b1"]
    rho2 = ((y - f["cy"]) / f["sy"]) ** 2 + ((x - f["cx"]) / f["sx"]) ** 2
    return world["system"]["tx_gain"] * (1 + f["amp"] * (0.5 - rho2) + f["gy"] * y + f["gx"] * x)


def b0_hz(world, y, x):
    """Off-resonance in Hz relative to the receiver reference, including the centre-frequency offset."""
    f = world["fields"]["b0"]
    return (world["system"]["cf_offset_hz"] + f["gy"] * y + f["gx"] * x
            + f["q"] * (((y - f["cy"]) ** 2 - (x - f["cx"]) ** 2) / 0.09))


def rx(world, y, x):
    f = world["fields"]["rx"]
    return (1 + f["gy"] * y + f["gx"] * x) * np.exp(1j * (f["ph"] + f["py"] * y + f["px"] * x))


def regions_public(world):
    """What the agent is told about the scored regions: id, centre and radius in mm. No tissue."""
    return [dict(id=r["id"], y_mm=round(r["y"] * world["fov_mm"], 2), x_mm=round(r["x"] * world["fov_mm"], 2),
                 radius_mm=round(r["r"] * world["fov_mm"], 2)) for r in world["regions"]]


def truth(world):
    """Server / verifier side only."""
    return {r["id"]: dict(t1_ms=world["classes"][r["cls"]]["t1_ms"], b1=float(b1(world, r["y"], r["x"])))
            for r in world["regions"]}
