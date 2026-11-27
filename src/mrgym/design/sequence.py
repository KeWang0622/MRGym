"""Whole sequences: a time-stepping extended phase graph (EPG) over a typed block list.

A sequence is a list of blocks, made by the functions below:

    rf(flip_deg, phase_deg)       an instantaneous pulse of the given flip angle about the axis (cos p, sin p, 0)
    rf_rotation(a, b)             an instantaneous pulse with Cayley-Klein parameters (a, b), e.g. of a designed pulse
    rf_pulse(pulse, ...)          the rotation of a designed ``pulses.Pulse``: on resonance, or one column per position
    delay(ms)                     relaxation (T1 with regrowth, T2) and precession by the off-resonance
    gradient(cycles)              dephasing by a whole number of cycles across the voxel (0 allowed), optionally with
                                  a duration: then also relaxation, precession and, with ``bd``, diffusion
    spoil()                       ideal spoiler: every transverse state is set to zero
    readout(index)                samples the transverse magnetisation F_0, demodulated with a receiver phase

``run`` returns the samples in the order of the readout blocks, with their ``index`` labels.

States and conventions (Weigel 2015; the code agrees with ``scanner/bloch.py::rf_spoiled_epg_stepping`` and with
``scanner/epg.py``, rows E8, E9 of ``mrgym.design.validate``)
  The transverse magnetisation of a voxel is  Mxy(phi) = sum_k F_k exp(i k phi),  phi = the dephasing angle across
  the voxel left by the gradients in units of one cycle, and Mz(phi) = sum_k Z_k exp(i k phi). Stored: ``fp[k]`` = F_k
  and ``fm[k]`` = conj(F_-k) for k >= 0 (so fp[0] = conj(fm[0])), ``z[k]`` = Z_k (Z_-k = conj(Z_k)).
  rf          Mxy' = conj(a)^2 Mxy - b^2 conj(Mxy) + 2 conj(a) b Mz,  Mz' = (|a|^2 - |b|^2) Mz - 2 Re(a b conj(Mxy))
              applied to every order k (``rotation`` docstring). A pulse of flip angle alpha and phase p has
              a = cos(alpha / 2), b = -i exp(i p) sin(alpha / 2): from equilibrium F_0 = -i exp(i p) sin(alpha).
  gradient    +n cycles moves F_k to F_(k+n) for every k. Z does not move.
  delay       F_k x E2 exp(+2 pi i df t), Z_k x E1, Z_0 + (1 - E1) M0.  E = exp(-t / T).
  readout     the sample is  F_0 exp(-i receiver phase) x ``demodulation``; the default demodulation +i makes the
              signal of a pulse about x real and positive, as the engine reports it (conventions.md, section 5).
  diffusion   a block of duration t during which the dephasing changes linearly by n cycles multiplies a transverse
              state that enters with order k by exp(-bd (k^2 + k n + n^2 / 3)) and a longitudinal state of order k by
              exp(-bd k^2), with bd = D kappa^2 t and kappa the wavenumber of one cycle per voxel (Weigel et al.
              2010). n = 0 is a delay. ``bd_of_b`` converts a b-value: a lobe that takes order 0 to order n has
              b = kappa^2 n^2 t / 3, so bd = 3 b D / n^2.
  truncation  orders above ``kmax`` are dropped. The default keeps every order the gradients can reach, so nothing
              is dropped; a smaller ``kmax`` is an approximation the caller chooses.

Columns. Flip angles, (a, b), T1, T2, off-resonance and the B1 scale may be arrays of one common length: the sequence
is then run for every column at once (isochromats through a slice profile, several tissues). ``weights`` sums the
columns of every sample.

Formed physics: the Bloch equations with instantaneous pulses, mono-exponential T1 and T2, free Gaussian diffusion.
Not in it: T2' (add exp(-|t| / T2') yourself at a gradient echo), magnetisation transfer, exchange, flow, finite
pulse duration other than through ``rf_pulse``.
"""
from __future__ import annotations

import dataclasses

import numpy as np

KINDS = ("rf", "delay", "gradient", "spoil", "readout")


@dataclasses.dataclass(frozen=True, eq=False)
class Block:
    """One block. Use the constructor functions; fields that a kind does not use stay at their defaults."""
    kind: str
    a: object = None                  # rf: Cayley-Klein parameters (numbers or per-column arrays)
    b: object = None
    scales: bool = False              # rf: the flip angle scales with the B1 of ``run`` (only for rf(flip, phase))
    flip_rad: object = None
    phase_rad: float = 0.0            # rf(flip, phase): pulse phase; readout: receiver phase
    ms: float = 0.0                   # delay, gradient: duration
    cycles: int = 0                   # gradient: dephasing in cycles per voxel
    bd: float = 0.0                   # delay, gradient: D kappa^2 t of the block
    index: object = None              # readout: label of the sample
    weights: object = None            # rf_pulse with columns: weights of the columns


def rf(flip_deg, phase_deg=0.0):
    """Instantaneous pulse. ``flip_deg`` may be an array with one entry per column. The actual angle is
    flip x the ``b1_scale`` of ``run``."""
    return Block("rf", scales=True, flip_rad=np.radians(np.asarray(flip_deg, float)), phase_rad=float(np.radians(phase_deg)))


def rf_rotation(a, b, phase_deg=0.0):
    """Instantaneous pulse with Cayley-Klein parameters (a, b), numbers or one per column; ``phase_deg`` adds a pulse
    phase (b -> b exp(i p)). It does not scale with ``b1_scale``: (a, b) is the rotation as it is."""
    a, b = np.asarray(a, complex), np.asarray(b, complex) * np.exp(1j * np.radians(phase_deg))
    if np.abs(np.abs(a) ** 2 + np.abs(b) ** 2 - 1).max() > 1e-9:
        raise ValueError("|a|^2 + |b|^2 must be 1")
    return Block("rf", a=a, b=b)


def rf_pulse(pulse, pos_mm=None, df_hz=0.0, b1_scale=1.0, phase_deg=0.0, weights=None):
    """The rotation of a designed pulse as one instantaneous block. Without ``pos_mm``: the rotation on resonance at
    the slice centre. With positions (one column each): the rotation at every position after the pulse's rephasing
    lobe, so that a run over the columns, summed with ``weights`` (e.g. the spacing divided by the thickness), is
    the signal of the slice profile. The duration of the pulse is NOT added to the timing: put half of it into the
    neighbouring delays if it matters. Off-resonance and B1 scale are those given here, not those of ``run``."""
    a, b = pulse.rotate(pos_mm, df_hz, b1_scale)
    blk = rf_rotation(a, b, phase_deg)
    return dataclasses.replace(blk, weights=None if weights is None else np.asarray(weights, float))


def delay(ms, bd=0.0):
    """Free evolution for ``ms``: T1 recovery, T2 decay, precession by the off-resonance; with ``bd`` = D kappa^2 t
    every order k also decays by exp(-bd k^2)."""
    if not ms >= 0 or bd < 0:
        raise ValueError("a delay and its bd are not negative")
    return Block("delay", ms=float(ms), bd=float(bd))


def gradient(cycles, ms=0.0, bd=0.0):
    """Dephasing by ``cycles`` (an integer, may be negative or zero) across the voxel. With ``ms`` the block also
    takes that time (relaxation, precession) and with ``bd`` = D kappa^2 ms it attenuates by diffusion under a
    constant gradient during the block."""
    if int(cycles) != cycles:
        raise ValueError("the dephasing of a gradient block is a whole number of cycles per voxel")
    if not ms >= 0 or bd < 0:
        raise ValueError("duration and bd are not negative")
    return Block("gradient", cycles=int(cycles), ms=float(ms), bd=float(bd))


def spoil():
    """Ideal spoiler: all transverse states are set to zero. Longitudinal states of every order are kept."""
    return Block("spoil")


def readout(index=None, phase_deg=0.0):
    """Sample F_0 with the receiver phase ``phase_deg``. ``index`` labels the sample (echo number, line, ...)."""
    return Block("readout", index=index, phase_rad=float(np.radians(phase_deg)))


def bd_of_b(b_s_mm2, adc_um2_ms, cycles):
    """``bd`` of a gradient block that takes order 0 to order ``cycles`` with the b-value ``b_s_mm2``:
    bd = 3 b D / cycles^2  (D in um^2/ms = 1e-3 mm^2/s)."""
    if cycles == 0:
        raise ValueError("a block without dephasing has no b-value of its own")
    return 3.0 * b_s_mm2 * adc_um2_ms * 1e-3 / cycles ** 2


@dataclasses.dataclass(frozen=True, eq=False)
class Result:
    signal: np.ndarray                # complex [readouts] (weighted sum over columns) or [readouts, columns]
    index: tuple                      # the labels of the readout blocks
    fp: np.ndarray                    # final state [orders, columns]
    fm: np.ndarray
    z: np.ndarray


def _columns(blocks, *arrays):
    n = 1
    for x in list(arrays) + [v for blk in blocks if blk.kind == "rf" for v in (blk.a, blk.b, blk.flip_rad) if v is not None]:
        size = np.size(x)
        if size != 1:
            if n != 1 and size != n:
                raise ValueError(f"per-column arrays have different lengths ({n} and {size})")
            n = size
    return n


def run(blocks, t1_ms, t2_ms, df_hz=0.0, b1_scale=1.0, m0=1.0, kmax=None, weights=None, state=None, demodulation=1j):
    """Run the block list from equilibrium (or from ``state`` = (fp, fm, z) of an earlier ``Result``).

    ``t1_ms``, ``t2_ms``, ``df_hz``, ``b1_scale``, ``m0``: numbers or per-column arrays. ``weights``: per-column
    weights; the samples are then summed over the columns (also taken from an ``rf_pulse`` block that carries
    weights). Without weights and with more than one column the samples are returned per column.
    """
    blocks = list(blocks)
    for blk in blocks:
        if not isinstance(blk, Block) or blk.kind not in KINDS:
            raise ValueError("a sequence is a list of blocks made by rf, rf_rotation, rf_pulse, delay, gradient, spoil, readout")
    if weights is None:
        weights = next((blk.weights for blk in blocks if blk.weights is not None), None)
    cols = _columns(blocks, t1_ms, t2_ms, df_hz, b1_scale, m0, *([] if weights is None else [weights]))
    t1, t2, df, sc, m_eq = (np.broadcast_to(np.asarray(x, float), (cols,)) for x in (t1_ms, t2_ms, df_hz, b1_scale, m0))
    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("t1_ms and t2_ms must be positive")
    reach = sum(abs(blk.cycles) for blk in blocks if blk.kind == "gradient")
    if state is not None:
        reach += np.shape(state[0])[0] - 1
    orders = reach + 1 if kmax is None else min(int(kmax), reach) + 1
    fp, fm, z = (np.zeros((orders, cols), complex) for _ in range(3))
    if state is None:
        z[0] = m_eq
    else:
        keep = min(orders, np.shape(state[0])[0])
        for dst, src in zip((fp, fm, z), state):
            dst[:keep] = np.asarray(src, complex).reshape(np.shape(src)[0], -1)[:keep]
    k = np.arange(orders, dtype=float)[:, None]
    has_df = bool(np.any(df != 0))
    out, labels = [], []
    for blk in blocks:
        kind = blk.kind
        if kind == "rf":
            if blk.scales:
                half = 0.5 * blk.flip_rad * sc
                a, b = np.cos(half), -1j * np.exp(1j * blk.phase_rad) * np.sin(half)
            else:
                a, b = blk.a, blk.b
            ac, ab = np.conj(a), a * b
            fp, fm, z = (ac * ac * fp - b * b * fm + 2 * ac * b * z,
                         -np.conj(b * b) * fp + a * a * fm + 2 * a * np.conj(b) * z,
                         -np.conj(ab) * fp - ab * fm + (np.abs(a) ** 2 - np.abs(b) ** 2) * z)
        elif kind in ("delay", "gradient"):
            n = blk.cycles
            if blk.ms > 0:
                e1, e2 = np.exp(-blk.ms / t1), np.exp(-blk.ms / t2)
                if has_df:
                    rot = np.exp(2j * np.pi * 1e-3 * df * blk.ms)
                    fp, fm = fp * (e2 * rot), fm * (e2 * np.conj(rot))
                else:
                    fp, fm = fp * e2, fm * e2
                z = z * e1
                z[0] += (1 - e1) * m_eq
            if blk.bd > 0:
                fp = fp * np.exp(-blk.bd * (k * k + k * n + n * n / 3.0))
                fm = fm * np.exp(-blk.bd * (k * k - k * n + n * n / 3.0))
                z = z * np.exp(-blk.bd * k * k)
            if n:
                fp, fm = _shift(fp, fm, n) if n > 0 else _shift(fm, fp, -n)[::-1]
        elif kind == "spoil":
            fp, fm = np.zeros_like(fp), np.zeros_like(fm)
        else:
            out.append(fp[0] * (np.exp(-1j * blk.phase_rad) * demodulation))
            labels.append(blk.index)
    sig = np.array(out).reshape(len(out), cols)
    if weights is not None:
        sig = sig @ np.broadcast_to(np.asarray(weights, float), (cols,))
    elif cols == 1:
        sig = sig[:, 0]
    return Result(sig, tuple(labels), fp, fm, z)


def _shift(up, down, n):
    """Dephasing by n > 0 cycles: the family ``up`` (orders k >= 0 of the sign that grows) moves to k + n, the family
    ``down`` (stored conjugated, orders -k) moves towards zero and through it into ``up``."""
    orders = up.shape[0]
    new_up, new_down = np.zeros_like(up), np.zeros_like(down)
    if n < orders:
        new_up[n:] = up[:orders - n]                            # order k >= 0 -> k + n
        new_down[:orders - n] = down[n:]                        # order -k -> -(k - n), still <= 0
    j = np.arange(max(1, n - orders + 1), min(n, orders - 1) + 1)
    new_up[n - j] = np.conj(down[j])                            # order -j, 1 <= j <= n, -> n - j >= 0
    new_down[0] = np.conj(new_up[0])
    return new_up, new_down


def repeat(blocks, times):
    """The block list repeated ``times`` times (a plain list)."""
    return list(blocks) * int(times)


def magnetisation(result, phase_cycles=0.0):
    """(Mxy, Mz) of the isochromat at the dephasing ``phase_cycles`` (in cycles) for every column of a result:
    the sum over orders of the state. For checks against isochromat simulations."""
    orders = result.fp.shape[0]
    k = np.arange(orders)[:, None]
    e = np.exp(2j * np.pi * k * phase_cycles)
    mxy = (result.fp * e).sum(0) + (np.conj(result.fm[1:]) * np.conj(e[1:])).sum(0)
    mz = (result.z[0] + 2 * (result.z[1:] * e[1:]).sum(0)).real
    return mxy, mz


__all__ = ["Block", "Result", "bd_of_b", "delay", "gradient", "magnetisation", "readout", "repeat", "rf", "rf_pulse", "rf_rotation", "run",
           "spoil"]
