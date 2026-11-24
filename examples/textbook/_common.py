"""Shared helpers of the textbook examples: open a session, scan, read regions, look up the truth, assert.

The examples drive the scanner the way an agent does: ``mrgym.make("Scanner/Explore-v0", ...)`` and tool calls.
They know the seed of their own session, so they can rebuild the hidden world and compare with it. An agent cannot.
"""
from __future__ import annotations

import argparse

import numpy as np

import mrgym
from mrgym.scanner import session as _session
from mrgym.scanner import world as W


def parse_args(doc: str):
    """Every example takes one optional argument: ``--plot FILE.png`` (needs matplotlib). Nothing is written without it."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot", metavar="FILE.png", help="write a figure to this path (needs matplotlib)")
    return p.parse_args()


def session(phantom: str = "qa", seed=1, field_t: float = 3.0, budget_s: float = 1e6):
    """A sandbox session on the virtual scanner, with a scan-time budget that never binds."""
    env = mrgym.make("Scanner/Explore-v0", phantom=phantom, field_t=field_t, seed=seed, budget_s=budget_s)
    env.reset()
    return env


def call(env, name: str, **arguments) -> dict:
    """One tool call; returns the observation's data and raises if the scanner refused."""
    obs = env.step({"name": name, "arguments": arguments}).observation
    if obs.is_error:
        raise RuntimeError(obs.text)
    return obs.data


def scan(env, regions="task", **protocol) -> dict:
    """Run a protocol. ``regions``: "task" (the marked vials of the QA phantom) or a list of circles."""
    return call(env, "scan", protocol=protocol, regions=regions)


def column(readouts: list, key: str) -> np.ndarray:
    """One read-out column as an array [region, image]."""
    return np.array([r[key] for r in readouts], float)


def complex_mean(readouts: list) -> np.ndarray:
    """Complex mean of every region and image, [region, image]."""
    return column(readouts, "complex_mean_magnitude") * np.exp(1j * column(readouts, "complex_mean_phase_rad"))


def hidden_world(env) -> dict:
    """The hidden world of this session, rebuilt from the episode config (holds the seed; operator side only)."""
    cfg = env.episode_config()
    return W.make_world(cfg["seed"], cfg["world_spec"])


def water_regions(world: dict) -> list:
    """The marked regions of a QA phantom that hold water (not oil), as circles for the ``regions`` argument."""
    return [dict(id=r["id"], y_mm=r["y_mm"], x_mm=r["x_mm"], radius_mm=r["radius_mm"])
            for r in world["regions"] if not world["tissues"][r["cls"]]["fat"]]


def engine_scan(engine, rois=(), rng=None, **protocol) -> dict:
    """A scan on a ``kspace.Engine`` built around a known world (no session, no budget): images, plan and read-outs.
    ``rng`` None gives noise-free data."""
    out = engine.scan(protocol, rng)
    out["roi_readouts"] = _session.roi_readouts(out["images"], list(rois), _session.fov_yx(out["plan"]["realised"]))
    return out


def upsample(image: np.ndarray, factor: int = 8) -> np.ndarray:
    """Sinc interpolation of a complex image by zero-filling its spectrum (what a console's interpolation does)."""
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    ny, nx = k.shape
    z = np.zeros((ny * factor, nx * factor), complex)
    y0, x0 = ny * factor // 2 - ny // 2, nx * factor // 2 - nx // 2
    z[y0:y0 + ny, x0:x0 + nx] = k
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(z))) * factor * factor


def check(label: str, value: float, expected: float, tol: float, unit: str = "", relative: bool = False):
    """Print one line and assert |value - expected| <= tol (or tol * |expected| when ``relative``)."""
    bound = tol * abs(expected) if relative else tol
    how = f"within {100 * tol:g} %" if relative else f"within {tol:g} {unit}".rstrip()
    print(f"{label}: {value:.4g} {unit}".rstrip() + f"  (expected {expected:.4g}, {how})")
    assert abs(value - expected) <= bound, f"{label}: {value} differs from {expected} by more than {bound}"


def save(fig, path: str):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print("figure written:", path)
