"""Scoring table of ``Scanner/NoiseSNR-v0`` (format: ``tests/scoring/__init__.py``).

Pass rule (task statement): for every marked region the delivered SNR is within 10 % of the mean signal of the region
in a noise-free scan of the stated protocol, divided by the true noise standard deviation of a pixel of that protocol.

Nothing here comes from the task module or from ``Engine.noise_sd``. The expected values are derived in this file:

  signal   mean magnitude of the noise-free image over the pixels whose centres lie in the circle (a mask written here)
  noise    from the documented receiver model (docs/scanner.md, "Noise"): every k-space sample carries complex noise of
           SD (per component) density x sqrt(readout samples x bandwidth per pixel / averages); the console's inverse
           FFT sums the acquired samples and divides by the oversampled field-of-view area, so one image pixel has
           SD = that x sqrt(lines x readout samples) / (2 x fov x phase fov). The density is the hidden system's
"""
import math

import numpy as np

from mrgym.scanner import console
from scoring import Row

ENV_ID = "Scanner/NoiseSNR-v0"
INSTANCES = {"L1": dict(level=1, seed=0), "L2": dict(level=2, seed=3), "L3": dict(level=3, seed=1)}
TOLERANCE = 0.10                # the pass window of the task statement
READOUT_OVERSAMPLING = 2        # the console samples the readout twice as densely as the matrix (datasheet)
AGREE = 1e-6                    # this file against the verifier, relative
_CACHE = {}


def truth(case) -> dict:
    """{region id: noise-free mean magnitude / pixel noise SD} of the stated protocol on the hidden system."""
    if case.seed_hex not in _CACHE:
        rp = console.plan(dict(case.cfg["task"]["protocol"]), case.exam)["realised"]
        img = np.abs(case.engine.acquire(rp, None)[0])
        ny, nx = img.shape
        fov_y, fov_x = (rp["fov_pe_mm"], rp["fov_mm"]) if rp["pe_dir"] == "y" else (rp["fov_mm"], rp["fov_pe_mm"])
        y, x = (np.arange(ny) - ny // 2) * fov_y / ny, (np.arange(nx) - nx // 2) * fov_x / nx
        samples = READOUT_OVERSAMPLING * rp["matrix"]
        per_sample = case.world["system"]["sigma0"] * math.sqrt(samples * rp["bandwidth_hz_per_px"] / rp["averages"])
        area = (READOUT_OVERSAMPLING * rp["fov_mm"]) * (rp["fov_pe_mm"] * rp["lines_full"] / rp["matrix_pe"])
        sd = per_sample * math.sqrt(rp["lines"] * samples) / area
        _CACHE[case.seed_hex] = {r["id"]: float(img[(y[:, None] - r["y_mm"]) ** 2 + (x[None, :] - r["x_mm"]) ** 2 <= r["radius_mm"] ** 2].mean() / sd)
                                 for r in case.cfg["task"]["regions"]}
    return _CACHE[case.seed_hex]


def at(factor=1.0, **per_region):
    """Deliverable: every region at ``factor`` times its SNR; ``R2=1.11`` overrides the factor of one region."""
    return lambda case: dict(snr={i: v * per_region.get(i, factor) for i, v in truth(case).items()})


def agrees(case, score):
    """The verifier's own SNR is the one derived here, and the reported error is the largest relative deviation."""
    want = truth(case)
    assert set(score["snr"]) == set(want) == {"R1", "R2", "R3"}
    assert all(abs(score["snr"][i] / want[i] - 1.0) <= AGREE + 1e-4 / want[i] for i in want)      # the score rounds to 4 decimals
    worst = max(abs(score["answer"]["snr"][i] / want[i] - 1.0) for i in want)
    assert abs(score["error"] - worst) <= 1e-9 + AGREE and score["tolerance"] == TOLERANCE
    assert score["distance"]["regions"] == 3 and score["distance"]["within"] == sum(k.startswith("snr_") and ok for k, ok in score["checks"].items())


ROWS = [
    Row("L1", "the scanner's own value in every region", at(), passed=True, expect=agrees),
    Row("L1", "9.9 % high in every region", at(1.099), passed=True, expect=agrees),
    Row("L1", "9.9 % low in one region", at(R2=0.901), passed=True, expect=agrees),
    Row("L1", "10.1 % high in one region", at(R2=1.101), passed=False, failing=("snr_R2",), reason="R2: delivered", expect=agrees),
    Row("L1", "10.1 % low in one region", at(R3=0.899), passed=False, failing=("snr_R3",), reason="R3: delivered", expect=agrees),
    Row("L1", "the noise without the factor sqrt(2)", at(1.0 / math.sqrt(2.0)), passed=False, failing=("snr_R1", "snr_R2", "snr_R3"), reason="(-29.3 %)"),
    Row("L1", "the background without the Rayleigh factor", at(1.0 / 0.6551), passed=False, failing=("snr_R1", "snr_R2", "snr_R3"), reason="(+52.6 %)"),
    Row("L2", "the scanner's own value in every region", at(), passed=True, expect=agrees),
    Row("L2", "in decibels", lambda case: dict(snr={i: 20.0 * math.log10(v) for i, v in truth(case).items()}), passed=False,
        failing=("snr_R1", "snr_R2", "snr_R3"), reason="delivered"),
    Row("L3", "the scanner's own value in every region, ghost included", at(), passed=True, expect=agrees),
    Row("L3", "10.1 % high in one region", at(R1=1.101), passed=False, failing=("snr_R1",), reason="R1: delivered", expect=agrees),
    Row("L1", "a region is missing", lambda case: dict(snr={i: v for i, v in truth(case).items() if i != "R3"}), refused=True,
        reason='the deliverable is {"snr": {"R1": <number>, "R2": <number>, "R3": <number>}}'),
    Row("L1", "one number, no regions", dict(snr=150.0), refused=True, reason='the deliverable is {"snr"'),
    Row("L1", "a field too many", lambda case: dict(snr=truth(case), noise=0.01), refused=True, reason='the deliverable is {"snr"'),
    Row("L1", "zero", lambda case: dict(snr=dict(truth(case), R1=0.0)), refused=True, reason="snr.R1 must be a number from 0.01 to 1000000"),
    Row("L1", "a text", lambda case: dict(snr=dict(truth(case), R2="high")), refused=True, reason="snr.R2 must be a number"),
    Row("L1", "true", lambda case: dict(snr=dict(truth(case), R2=True)), refused=True, reason="snr.R2 must be a number"),
    Row("L1", "not a number", lambda case: dict(snr=dict(truth(case), R3=float("nan"))), refused=True, reason="snr.R3 must be a number"),
]
