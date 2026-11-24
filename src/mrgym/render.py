"""Observation renderers: numbers as aligned text tables, images as PNG (vision models) or digit maps (text models).

No dependency beyond numpy and the standard library: the PNG encoder is 20 lines of zlib + struct.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np


def sig(x, n=5):
    """A number with ``n`` significant digits, as short text. None stays '-'."""
    if x is None:
        return "-"
    return f"{float(x):.{n}g}"


def table(header, rows):
    """Fixed-width text table (first column left-aligned, the others right-aligned)."""
    rows = [[str(c) for c in r] for r in rows]
    width = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(header)]
    line = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, width)))  # noqa: E731
    return "\n".join([line([str(h) for h in header])] + [line(r) for r in rows])


def kv(d, keys=None):
    """``key=value`` pairs on one line; numbers shortened to 6 significant digits."""
    out = []
    for k in keys or d:
        v = d.get(k)
        if v is None:
            continue
        out.append(f"{k}={sig(v, 6) if isinstance(v, float) else v}")
    return " ".join(out)


def window(img, kind):
    """Map an image to 0..1 for display: magnitude 0..99.5th percentile, phase -pi..pi."""
    img = np.asarray(img, float)
    if kind == "phase":
        return (img + np.pi) / (2 * np.pi), "-pi..pi rad"
    top = float(np.percentile(img, 99.5)) or 1.0
    return np.clip(img / top, 0, 1), f"0..{sig(top, 4)}"


def digit_map(img, kind="magnitude", cells=16, fov_yx_mm=(240.0, 240.0)):
    """Text stand-in for an image: about ``cells`` blocks along the longer side, block means on a 0-9 scale (one
    character per block). Returns (grid text, legend)."""
    v, rng = window(img, kind)
    ny, nx = v.shape
    cy, cx = (cells, max(1, round(cells * nx / ny))) if ny >= nx else (max(1, round(cells * ny / nx)), cells)
    cy, cx = min(cy, ny), min(cx, nx)
    ey, ex = np.linspace(0, ny, cy + 1).astype(int), np.linspace(0, nx, cx + 1).astype(int)
    rows = []
    for i in range(cy):
        rows.append("".join(str(int(min(9, np.floor(10 * v[ey[i]:ey[i + 1], ex[j]:ex[j + 1]].mean())))) for j in range(cx)))
    legend = (f"{cy}x{cx} block means, one digit per {fov_yx_mm[0] / cy:.1f} x {fov_yx_mm[1] / cx:.1f} mm block (y, x), 0-9 spans {rng}; "
              f"row = y (top = {-fov_yx_mm[0] / 2:.0f} mm), column = x (left = {-fov_yx_mm[1] / 2:.0f} mm)")
    return "\n".join(rows), legend


def bounding_box(img, fov_yx_mm, fraction=0.2):
    """Extent of the pixels brighter than ``fraction`` of the 99.5th percentile, in mm: dict(y=(lo, hi), x=(lo, hi)) with
    the outer pixel edges, or None for an empty image. Isolated bright pixels (noise) are ignored. What a ruler on
    the console's image display gives."""
    a = np.asarray(img, float)
    top = float(np.percentile(a, 99.5))
    if top <= 0:
        return None
    m = a > fraction * top
    ny, nx = a.shape
    if ny >= 3 and nx >= 3:                                              # drop isolated noise pixels: keep a pixel when at
        pad = np.pad(m.astype(np.int8), 1)                              # least 6 of the 9 pixels around it are bright
        m = sum(pad[i:i + ny, j:j + nx] for i in range(3) for j in range(3)) >= 6
    if not m.any():
        return None
    dy, dx = fov_yx_mm[0] / ny, fov_yx_mm[1] / nx
    ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    return dict(y=((ys[0] - ny // 2 - 0.5) * dy, (ys[-1] - ny // 2 + 0.5) * dy),
                x=((xs[0] - nx // 2 - 0.5) * dx, (xs[-1] - nx // 2 + 0.5) * dx),
                touches_edge=bool(ys[0] == 0 or ys[-1] == ny - 1 or xs[0] == 0 or xs[-1] == nx - 1))


def _chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def png_gray(v01, scale=1, circles=None, fov_yx_mm=(240.0, 240.0)):
    """8-bit greyscale PNG of a 0..1 image [ny, nx], upsampled by ``scale`` (nearest neighbour); optional region outlines.

    circles: [{"y_mm", "x_mm", "radius_mm"}], drawn as white rings with a dark halo so a reader sees where the
    task's regions lie.
    """
    a = np.repeat(np.repeat(np.clip(np.asarray(v01, float), 0, 1), scale, 0), scale, 1)
    ny, nx = a.shape
    if circles:
        ypos = (np.arange(ny) - ny // 2) / ny * fov_yx_mm[0]
        xpos = (np.arange(nx) - nx // 2) / nx * fov_yx_mm[1]
        w = max(fov_yx_mm[0] / ny, fov_yx_mm[1] / nx)
        for c in circles:
            d = np.hypot(ypos[:, None] - c["y_mm"], xpos[None, :] - c["x_mm"])
            a[(d - c["radius_mm"] > w) & (d - c["radius_mm"] <= 2 * w)] = 0.0       # dark halo: visible on bright vials too
            a[np.abs(d - c["radius_mm"]) <= w] = 1.0
    px = (a * 255).astype(np.uint8)
    raw = b"".join(b"\x00" + px[i].tobytes() for i in range(ny))
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", nx, ny, 8, 0, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b""))


def image_png(img, kind="magnitude", min_side=384, circles=None, fov_yx_mm=(240.0, 240.0)):
    """PNG of one image, upsampled so the shorter side has at least ``min_side`` pixels. Returns (png bytes, range text)."""
    v, rng = window(img, kind)
    scale = max(1, int(np.ceil(min_side / min(v.shape))))
    return png_gray(v, scale=scale, circles=circles, fov_yx_mm=fov_yx_mm), rng


def png_size(png):
    """(width, height) read back from a PNG header; used by the tests."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR"
    return struct.unpack(">II", png[16:24])
