"""A second, independently written correct reconstruction for the Cartesian scenario.

It interprets the dialect at run time instead of being generated from it, keeps the last record of a repeated line
instead of averaging, moves the prescribed centre to the image centre *before* the SENSE solve (and samples the coil
maps at the shifted positions), and uses a different regularisation. The admission gate requires it to pass, which
checks two things at once: the pass threshold tolerates legitimate variation, and two separately written decoders
agree on what the format means.
"""
import json

import numpy as np
from scipy.ndimage import map_coordinates

from . import sim
from .cartesian import KIND_BITS


def reconstruct(npz, d):
    N = d["names"]
    hdr = json.loads(str(npz[N["header"]]))
    raw = np.asarray(npz[N["data"]])
    if d["complex"] == "split":
        raw = raw[..., 0] + 1j * raw[..., 1]
    raw = np.moveaxis(raw, [d["layout"].index(c) for c in "RCS"], [0, 1, 2])
    n_pe, n_ro, sig = hdr[N["n_pe"]], hdr[N["n_ro"]], d["ft_sign"]
    n_s, echo = raw.shape[2], raw.shape[2] // 2
    ky = np.asarray(npz[N["ky"]], int) - (0 if d["ky_conv"] == "signed" else n_pe // 2)
    kind = np.asarray(npz[N["kind"]])
    imaging = (kind == "img") if d["kind_conv"] == "str" else (kind & sum(KIND_BITS.values())) == 0
    rev = np.asarray(npz[N["dir"]])
    rev = rev.astype(bool) if d["dir_conv"] == "bool" else rev == -1
    gain = np.asarray(npz[N["gain"]], float)
    gain = 10 ** (gain / 20) if d["gain_conv"] == "db" else gain

    lines = {}
    for r in range(raw.shape[0]):
        if not imaging[r]:
            continue
        step = sig * (-1 if rev[r] else 1)                # stored sample s holds kx = step * (s - echo) / os
        src = (echo + step * (np.arange(n_s) - echo)) % n_s
        lines[(sig * ky[r] + n_pe // 2) % n_pe] = raw[r][:, src] / gain[r]
    have = np.zeros(n_pe, bool)
    k = np.zeros((raw.shape[1], n_pe, n_s), complex)
    for row, line in lines.items():
        have[row], k[:, row] = True, line
    k = sim.cfft(sim.crop_centre(sim.cifft(k, (2,)), n_ro, 2), (2,))

    along_x = hdr[N["phase_axis"]] == "x"
    ny, nx = (n_ro, n_pe) if along_x else (n_pe, n_ro)
    c = dict(zip(d["shift_order"], hdr[N["centre"]]))
    scale = {"mm": (hdr[N["fov"]], hdr[N["fov"]]), "frac": (1, 1), "pix": (ny, nx)}[d["shift_conv"]]
    cy, cx = c["y"] / scale[0], c["x"] / scale[1]
    d_pe, d_ro = (cx, cy) if along_x else (cy, cx)
    k_pe, k_ro = np.arange(n_pe) - n_pe // 2, np.arange(n_ro) - n_ro // 2
    k = k * np.exp(2j * np.pi * (k_pe[None, :, None] * d_pe + k_ro[None, None, :] * d_ro))

    s = np.asarray(npz[N["sens"]])
    s = s if d["sens_layout"] == "CYX" else np.moveaxis(s, 2, 0)
    cmap = hdr[N["coil_map"]]
    if d["chan_conv"] == "data_to_sens":
        s = np.array([s[j] for j in cmap])
    else:
        by_channel = [None] * len(cmap)
        for j, channel in enumerate(cmap):
            by_channel[channel] = s[j]
        s = np.array(by_channel)
    if along_x:
        s = np.swapaxes(s, 1, 2)
    m_pe, m_ro = s.shape[1:]
    pts = np.meshgrid((k_pe / n_pe + d_pe) * m_pe + m_pe // 2, (k_ro / n_ro + d_ro) * m_ro + m_ro // 2, indexing="ij")
    s = np.array([map_coordinates(c_.real, pts, order=3, mode="nearest")
                  + 1j * map_coordinates(c_.imag, pts, order=3, mode="nearest") for c_ in s])

    w = have[None, :, None]

    def normal(x):
        return np.sum(s.conj() * sim.cifft(w * sim.cfft(s * x, (1, 2), "ortho"), (1, 2), "ortho"), 0) + 2e-3 * x

    b = np.sum(s.conj() * sim.cifft(w * k, (1, 2), "ortho"), 0)
    x, r = np.zeros_like(b), b.copy()
    p, rs = r.copy(), np.vdot(r, r).real
    for _ in range(80):
        q = normal(p)
        a = rs / np.vdot(p, q).real
        x, r = x + a * p, r - a * q
        rs, old = np.vdot(r, r).real, rs
        if rs < 1e-14 * np.vdot(b, b).real:
            break
        p = r + rs / old * p
    img = np.abs(x)
    return (img.T if along_x else img).astype(np.float32)
