"""Scans served from the block-list core (``mrgym.core``): amplitudes of a sequence that is data, and image formation
with an amplitude that changes from one phase-encode line to the next.

A sequence of the registry with ``serves = "core"`` (``console.Sequence``) has no closed form in the engine. Its
realised protocol is turned into a block list (``mrgym.core.library.served``), the periodic state of one repetition is
solved (``mrgym.core.periodic``), and the samples of that repetition are the amplitudes of the scan: one per echo of
an echo train, one for the echo of a diffusion-weighted spin echo, one for a gradient echo. The sampling plan
(``console.Sampling``) says which sample makes which image and which sample fills which phase-encode line.

Amplitudes (``table``). For every tissue class with a voxel in the world, on Chebyshev nodes of the transmit scale
(nominal x B1+ x transmit gain x ``tx_scale``) that span the object:
  train     echo trains and the diffusion-weighted spin echo (a train of one echo): the rotation of the excitation and
            of the refocusing pulse at every slice column (Cayley-Klein parameters; the refocusing pulse on a slice
            ``refocus_slice_ratio`` times as thick) reduces to the refocusing flip angle, the periodic state of the
            repetition with the gratings along z that survive it is one small linear solve per column
            (``train_response``, ``carry_columns``), and the samples are integrated through the slice. Imperfect
            refocusing therefore gives the stimulated echoes it gives on a scanner, inside a repetition and from
            one repetition to the next: they are not a correction, they are in the states.
  columns   every slice column solved by ``mrgym.core.periodic`` with the rotation of the pulse at that column: an
            echo train whose refocusing angle changes along the train, ``satrec``, and the reference of the above.
  flip      the RF-spoiled gradient echo (``megre``): a slice column is a hard pulse of the column's flip angle, its
            transverse phase a weight (exact for this sequence: ``mrgym.core.validate`` row Q3d). The steady state
            with ideal spoiling is solved in every column; what RF spoiling adds to it is solved on 16 Chebyshev
            nodes of the flip angle per class and interpolated (the rule of ``kspace.gre_steady``).
A voxel's amplitude is the barycentric interpolation of its class's table over the transmit scale, times weight and
proton density: the engine's treatment of every other sequence.

Image formation with line-dependent amplitudes (``form_lines``). With a(c, j, s) the table (class c, node j, sample
s), s(p) the sample that fills line p, and g_c(tau) the evolution of class c around the echo centre,

    S(p, x) = sum_c  K[ w_c pd_c  sum_j Q_j(b1(r)) a(c, j, s(p))  g_c(tau(x)) ](p, x),

K being the Fourier integral of ``kspace.py`` and x the readout sample. Evaluating it line by line costs one image
formation per sample of the cycle. It is made cheap by one low-rank factorisation of a small matrix, rows (class,
node), columns (sample of the cycle, readout sample):

    a(c, j, s) g_c(x) = sum_t U[c, j, t] H[t, s, x] + R[c, j, s, x]

so that S(p, x) = sum_t H[t, s(p), x] K[image of U[., ., t]](p, x): one coefficient image per term, transformed along
the phase-encode axis FIRST (the expensive transform, master grid -> lines) and then along the readout, and
multiplied sample by sample with the weight of its line. An image that is one sample of the cycle on every line (an
echo of a multi-echo scan) is the same sum with a constant s: all echoes share the transforms.

The truncation is bounded, not estimated. For every line and every readout sample,

    |error of S| <= kappa max_(s, x) sum_c mass_c Lambda_c max_j |R[c, j, s, x]|                 (``line_basis``)

with mass_c = dA dz sum_r w_c pd_c |receive field| (the largest k-space sample class c can make with unit amplitude),
Lambda_c the Lebesgue constant of the interpolation over the transmit scale on the voxels of the class, and kappa
the gain of the displacement operator (1 without it). The ranks are the smallest for which this bound is at most
``LINE_NOISE_FRACTION`` of the standard deviation of the receiver noise of one k-space sample of the scan at the
largest number of averages the console accepts (so the bound does not depend on the averages, and neither does the
noise-free data); in a world without noise, ``LINE_REL_TOL`` of the largest sample the object can make. The readout
evolution is inside the same factorisation, so this is the only truncation of the path (the six sequences with a
closed form cut their readout basis at ``kspace.SVD_TOL`` of its largest singular value instead).
The bound is one of k-space, sample by sample. In the image the error of the samples adds coherently where the
noise does not, so a pixel can be off by more noise standard deviations than a sample: measured 0.016 of the noise
of a pixel at 64 averages (0.002 at one average) over worlds from nearly noiseless to useless (``mrgym validate``,
row J14, which holds it below 0.1); that is a measurement, not a bound. (At a fraction of 0.5 the independent
review measured 0.8: the fraction was lowered to 0.05 for that reason.) ``scanner/served_checks.py`` (rows J12 to J14)
compares with the line-by-line Fourier sum of the signal equation.

Not here: fat saturation, and chemical-shift or J-coupling effects inside an echo train (fat is one species with its
T1 and T2; its lines dephase and rephase around every echo). The amplitude of a spin-echo sequence is that of spins
on resonance, and of the state that survives a repetition only the gratings along z are carried into the next one:
the transverse magnetisation at the end of a repetition is taken as dephased (``carry_of``). Periodic motion is the engine's modulation of k-space
lines with the time each line is read (``form``, ``line_times``): the echoes are those of the object at rest.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from ..core import blocks as B
from ..core import epg, library, periodic
from ..core import operators as O
from ..core import periodic as P
from . import console, kspace, tissue
from . import signal as S

LINE_NOISE_FRACTION = 0.05          # bound of the line-basis truncation, in noise standard deviations of one k-space sample at MAX_AVERAGES
LINE_REL_TOL = 1e-7                 # the same bound for a world without noise, relative to the largest sample the object can make
MAX_AVERAGES = console._GEOM["averages"][1]
B1_NODES = (5, 16)                  # fewest and most Chebyshev nodes of the transmit scale for a smooth dependence (the engine's rule)
TRAIN_NODES = (1.0, 6, 72)          # echo trains: nodes = slope x echoes x refocusing angle (rad) x width of the range + offset, at most ``most``
TRAIN_COLUMNS = 4                   # echo trains: slice columns per column of the engine's quadrature
FLIP_NODES = kspace.PROFILE_NODES   # nodes of the flip angle on which the departure from ideal spoiling is solved (megre)
CARRY_TOL = 1e-3                    # spin-echo sequences: a grating along z is carried into the next repetition when relaxation and diffusion leave it at least this
CARRY_CAP = (16, 1)                 # and its order is at most this on the crusher axis (and on a second axis that is dephased before the last pulse)
CARRY_DIM = 96                      # and the box has at most this many real coordinates (the crusher axis gives way)
CARRY_BYTES = 96e6                  # working memory of one chunk of slice columns
RESPONSE_BYTES = 256e6              # working memory of one run of ``train_response``
CARRY_COLUMNS = (4, 32)             # the carried part is integrated through the slice on 1, 2, 4 x the engine's columns: up to this many echoes on 1 x, on 2 x


# ---------------------------------------------------------------------------------------------- amplitudes
def _classes(eng):
    """Indices of the tissue classes that have a voxel in the world."""
    return [c for c in range(len(eng.w["names"])) if eng.sup[c].any()]


def node_count(rp, lo, hi):
    """Number of Chebyshev nodes of the transmit scale on [lo, hi].

    A world with one transmit scale needs one. Otherwise the engine's rule for a smooth dependence (5 to 16 nodes with
    the relative width of the range), and for an echo train more: echo n depends on the refocusing angle beta like a
    polynomial of degree n (it is one in cos beta), so the nodes must resolve n x beta x (hi - lo) radians.
    ``TRAIN_NODES`` = (slope, offset, most): counts that keep the interpolated echoes within 1e-6 of the largest echo
    for trains of 8 to 64 echoes at 60 to 180 degrees over a transmit range of 25 % (``mrgym validate``, row J10).
    Beyond ``most`` nodes the rule stops: 32 echoes of 180 degrees over a transmit scale of 1.2 to 2.6 (a range of
    0.6 to 1.3 at ``tx_scale`` 2) would ask for more, and the error there is 4e-5 (found by the independent review)."""
    if hi - lo < 1e-12:
        return 1
    n = int(np.clip(np.ceil(30 * (hi - lo) / (hi + lo)) + 4, *B1_NODES))
    echoes = rp.get("etl") or (rp.get("n_echoes") if rp["sequence"] == "mese" else None)
    if echoes:
        slope, offset, most = TRAIN_NODES
        beta = math.radians(float(np.max(rp["refocus_deg"])))
        n = int(min(max(n, math.ceil(slope * echoes * beta * (hi - lo)) + offset), most))
    return n


def _nodes(eng, rp, count=None):
    """Chebyshev nodes and barycentric weights of the transmit scale (B1+ x transmit gain x ``tx_scale``) that span the
    object, padded as the engine pads its flip-angle nodes."""
    tx = rp.get("tx_scale", 1.0)
    lo, hi = (tx * v for v in eng.b1_range)
    pad = 0.01 * (hi - lo) + 1e-4
    return kspace.cheb_nodes(lo - pad, hi + pad, node_count(rp, lo, hi) if count is None else count)


def _tissue(cls):
    return dict(t1_ms=cls["t1_ms"], t2_ms=cls["t2_ms"], adc_um2_ms=cls["adc_um2_ms"])


def _class_spins(eng, classes):
    return [_tissue(eng.w["tissues"][eng.w["names"][c]]) for c in classes]


def _system(eng):
    """What a block list takes from the system of the world: the efficiency of its inversion and saturation pulses
    (a world that does not state the latter saturates completely)."""
    return dict(inv_eff=eng.w["system"]["inv_eff"], sat_eff=eng.w["system"].get("sat_eff", 1.0))


def _crushed(seq):
    """True when the repetition of the block list ends with the ideal crusher (a protocol written by hand)."""
    blocks = seq.blocks[0].body if len(seq.blocks) == 1 and seq.blocks[0].kind == "repeat" else seq.blocks
    return any(blk.kind == "crush" for blk in blocks)


def carry_of(seq, tissue, tol=None, cap=None):
    """What reaches the next excitation of the block list ``seq`` for one tissue, as it is SERVED: (largest order of
    the gratings along z per axis, False).

    The gratings: ``periodic.carry_orders`` with ``CARRY_TOL``, at most ``CARRY_CAP`` orders on the first axis (crushers
    or diffusion lobes). A grating exists only on an axis that is dephased before the last pulse, so an axis of the
    spoiler alone carries none.

    The transverse states are NOT carried (the second value is always False). What is transverse when the last echo
    has been read crosses the spoiler and hundreds of milliseconds of free precession before the next excitation. Its
    contribution to the next train depends on the phase it gathers there: on the off-resonance of the voxel within a
    fraction of 1 / TR, and on the spread of off-resonance inside the voxel (T2'), neither of which the amplitude of
    these sequences holds (it is solved on resonance). Serving it for spins exactly on resonance would serve one
    member of a family whose spread is as large as the effect (found by the independent review: for cerebrospinal
    fluid the periodic echo swings by half of itself with the off-resonance at TR 1.5 s). It is taken as dephased for
    good, which static dephasing does to tissue (T2' of a fraction of a second) and pulsation to fluid in a body;
    ``mrgym validate`` row J24c prints what a static fluid exactly on resonance would add. The core computes it
    (``periodic``, method carry with ``carry_transverse``; ``table(..., carry=(orders, True))``)."""
    if _crushed(seq):
        return (0,) * B.lattice(seq)[0], False
    sp = O.grid([tissue], (0.0,), (1.0,))
    cap = CARRY_CAP if cap is None else cap
    orders, _ = P.carry_orders(seq, sp, CARRY_TOL if tol is None else tol, max(cap), transverse=False)
    prog = P.period(seq)[0]
    last = max(i for i, op in enumerate(prog.ops) if op.kind == "rf")
    crusher = [any(op.kind == "free" and op.n[a] for op in prog.ops[:last]) for a in range(prog.axes)]
    orders = [min(k, cap[0] if a == 0 else cap[1]) if c else 0 for a, (k, c) in enumerate(zip(orders, crusher))]
    while P.carry_box(orders, False)["d"] > CARRY_DIM and orders[0] > 0:
        orders[0] -= 1
    return tuple(orders), False


def _table_columns(eng, rp, classes, gn, carry=None):
    """[class, node, sample]: every slice column solved with its rotation, integrated through the slice. A block list
    whose repetition ends with a real spoiler is solved with the states that reach the next excitation
    (``periodic``, method "carry", the box of ``carry_of`` per class; ``carry`` = (orders, transverse) for all)."""
    entry = console.REGISTRY[rp["sequence"]]
    tbw, window = eng.pulse(rp)
    tissues = _class_spins(eng, classes)
    kw = {} if tbw is None else dict(selective=B.Selective(tbw, window))
    seq, labels = entry.blocks(rp, **kw, **_system(eng))
    still = not any(B.compile_ops(seq, frame=False).reach)              # no gradient: nothing but Mz crosses an ideal spoiler
    # z >= 0 where the mirror column gives the complex conjugate: true while only Mz reaches the next excitation, NOT
    # when gratings or transverse states do (``_table_train``); then both sides of the slice are solved
    hard, _ = entry.blocks(rp, **_system(eng))
    boxes = [(None, False) if _crushed(seq) or still else (carry_of(hard, tis) if carry is None else carry) for tis in tissues]
    mirror = not (any(tv for _, tv in boxes) and _tail_turns(hard))
    zeta, wz = ((0.0,), None) if tbw is None else O.slice_columns(tbw, two_sided=not mirror, window=window)
    out = np.empty((len(classes), gn.size, len(labels)), complex)
    for i, tis in enumerate(tissues):
        sp = O.grid([tis], zeta, gn)
        if _crushed(seq) or still:
            sig = periodic.steady_state(seq, sp).signal
        else:
            orders, transverse = boxes[i]
            sig = periodic.steady_state(seq, sp, method="carry", carry=orders, carry_transverse=transverse).signal
        sig = sig.reshape(len(labels), len(zeta), gn.size)
        out[i] = np.moveaxis(sig[:, 0] if tbw is None else ((sig.real if mirror else sig) * wz[None, :, None]).sum(1), 0, -1)
    return out


def _with_rotations(seq, excitation, refocusing):
    """The block list ``seq`` with its first rf block replaced by the rotation ``excitation`` (None: left out) and
    every later one by ``refocusing`` (rows of Cayley-Klein parameters per column); delays, gradients and samples
    stay what they are."""
    seen = []

    def walk(blocks):
        out = []
        for blk in blocks:
            if blk.kind == "repeat":
                out.append(dataclasses.replace(blk, body=walk(blk.body), tables=()))
            elif blk.kind == "rf":
                if seen or excitation is not None:
                    out.append(B.rf(0.0, blk.phase_deg, crushed=blk.crushed, ck=refocusing if seen else excitation))
                seen.append(blk)
            else:
                out.append(blk)
        return tuple(out)
    return B.sequence(walk(seq.blocks), cycle_mm=seq.cycle_mm, name=seq.name)


def train_response(seq, tissues, n_echoes, carry=None, transverse=False):
    """What a spin-echo train does to the state it is given, as functions of the refocusing flip angle alone.

    In a train with equal delays and equal crushers either side of every refocusing pulse, a refocusing rotation
    (a, b) acts on the sampled echoes as the hard pulse of flip angle theta = 2 asin|b| about the transverse axis of
    phase arg(b) + 90 degrees: the phase of a is a turn about z that is the same before and after the pulse, and a
    turn about z next to a crusher is a translation along the crusher axis, which no sample at order 0 and no
    relaxation or diffusion sees. (A spoiler on the crusher axis after the last echo is not paired with such a turn:
    what is transverse there keeps the phase of the translation, ``carry_columns``.)

    ``seq``: the block list of one repetition with hard pulses; its first rf is the excitation, every other one a
    refocusing pulse of the one angle theta. The repetition is cut in two: the excitation, which acts on the state
    x that the last repetition left (the state of the box ``carry``: the gratings Z_k, and with ``transverse`` the
    states F_k) and gives y = E x, and everything after it, which is real-linear in y:

        x' = K(theta) y + c(theta),        echo_n = L_n(theta) y + l_n(theta).

    Every pulse enters with cos(theta) and sin(theta), and turning the train by 180 degrees about z changes the sign
    of theta, of every transverse state and of nothing else. So an entry that connects two states of the same kind
    (transverse to transverse, longitudinal to longitudinal) is a polynomial of degree <= n in cos(theta), and an
    entry that connects the two kinds is sin(theta) times such a polynomial. The tables are computed on ``n_echoes``
    + 1 Chebyshev points of cos(theta) by stepping the train once from nothing and once from every real coordinate of
    y, and hold the mixed entries divided by sin(theta): interpolation over cos(theta) is then exact.
    ``tissues``: one tissue, or a list of tissues that share the box (they are stepped together).

    Returns dict(x, w: the points and their barycentric weights; K [point, coordinate of x, coordinate of y];
    c [point, coordinate of x]; L [point, echo, coordinate of y] and l [point, echo], complex; odd_K, odd_c, odd_L:
    which entries are held divided by sin(theta) (l always is); box_x, box_y: the boxes (``periodic.carry_box``);
    pairs: the number of pairs of orders (k, -k)). The coordinates are ordered by pair: those of order 0 first (y:
    Re F_0, Im F_0, Z_0; x: the same, or Z_0 alone), then the six of every pair (Re F_k, Im F_k, Re F_-k, Im F_-k,
    Re Z_k, Im Z_k; x without transverse states: Re Z_k, Im Z_k)."""
    one = isinstance(tissues, dict)
    tissues = [tissues] if one else list(tissues)
    axes = B.lattice(seq)[0]
    carry = (0,) * axes if carry is None else tuple(int(k) for k in carry)
    m = int(n_echoes) + 1
    reach = B.compile_ops(seq, frame=False).reach
    cells = (m * (3 * int(np.prod([2 * k + 1 for k in carry])) + 1)) * int(np.prod([2 * (k + r) + 1 for k, r in zip(carry, reach)]))
    most = max(1, int(RESPONSE_BYTES / (16.0 * 8 * cells)))             # tissues stepped together: bounded memory
    if len(tissues) > most:
        out = [t for i in range(0, len(tissues), most) for t in train_response(seq, tissues[i:i + most], n_echoes, carry, transverse)]
        return out
    n_t = len(tissues)
    xn, xw = kspace.cheb_nodes(-1.0, 1.0, m)
    half = 0.5 * np.arccos(xn)
    sin = np.sin(2 * half)
    ref = np.stack([np.cos(half), 0 * half, 0 * half, -np.sin(half)], 1)        # hard pulse about x of angle theta: a = cos, b = -i sin
    prog = B.compile_ops(_with_rotations(seq, None, ref), frame=False)
    box_y, box_x = P.carry_box(carry, True), P.carry_box(carry, transverse)
    d_y, sizes = box_y["d"], box_y["sizes"]
    f0, z0, _ = P.box_states(box_y)
    lead = np.zeros((f0.shape[0], 1), complex)                          # the first run starts from nothing
    f0, z0 = np.concatenate([lead, f0], 1), np.concatenate([lead, z0], 1)
    sp = O.take(O.grid(tissues, np.zeros(m), (1.0,)), np.repeat(np.arange(n_t * m), d_y + 1))      # rf blocks given by ck read ``column``: the point
    tile = lambda a: np.tile(a, (1, n_t * m)).reshape(sizes + (n_t * m * (d_y + 1),))      # noqa: E731
    res = epg.run(prog, sp, state=epg.State(tile(f0), tile(z0), None, None, carry, prog.unit, carry), prune=carry)
    end = epg._embed(res.state, carry)
    out = P.box_coordinates(box_x, end.f.reshape(-1, n_t, m, d_y + 1), end.z.reshape(-1, n_t, m, d_y + 1))      # [coordinate of x, tissue, point, run]
    sig = res.signal.reshape(-1, n_t, m, d_y + 1)
    # ---- coordinates grouped by pair of orders (``_pair_index``), so that the excitation acts on blocks of 6 and 3
    py, zy = _pair_index(box_y)
    px, zx = _pair_index(box_x)
    order_y, order_x = np.concatenate([zy, py.ravel()]), np.concatenate([zx[zx >= 0], px[px >= 0]])
    ky, kx = np.array([k == "F" for k in box_y["kinds"]])[order_y], np.array([k == "F" for k in box_x["kinds"]])[order_x]
    odd_k, odd_c, odd_l = kx[:, None] != ky[None, :], kx, ~ky
    tables = []
    for i in range(n_t):
        big_k = np.moveaxis(out[:, i, :, 1:] - out[:, i, :, :1], 1, 0)[:, order_x][:, :, order_y]          # [point, x, y]
        big_l = np.moveaxis(sig[:, i, :, 1:] - sig[:, i, :, :1], 1, 0)[:, :, order_y]                      # [point, echo, y]
        const = out[:, i, :, 0].T[:, order_x]
        tables.append(dict(x=xn, w=xw, K=np.where(odd_k[None], big_k / sin[:, None, None], big_k), c=np.where(odd_c[None], const / sin[:, None], const),
                           L=np.where(odd_l[None, None], big_l / sin[:, None, None], big_l), l=sig[:, i, :, 0].T / sin[:, None],
                           odd_K=odd_k, odd_c=odd_c, odd_L=odd_l, box_x=box_x, box_y=box_y, pairs=py.shape[0], transverse=bool(box_x["transverse"])))
    return tables[0] if one else tables


def _excitation_blocks(box, a, b, phase, pairs=True):
    """The excitation (a, b) [N] as real matrices on the coordinates of one pair of orders (k, -k) and of the order 0:
    (pair [N, 6, 6], zero [N, 3, 3]). Coordinates of a pair: Re F_k, Im F_k, Re F_-k, Im F_-k, Re Z_k, Im Z_k; of the
    order 0: Re F_0, Im F_0, Z_0. ``phase`` [N]: a turn about z of the transverse states before the pulse. ``pairs``
    False: the block of the order 0 alone (pair None)."""
    ca2, b2, cab2, d, ab = np.conj(a) ** 2, b * b, 2 * np.conj(a) * b, np.abs(a) ** 2 - np.abs(b) ** 2, a * b
    e = np.exp(1j * phase)
    pair, zero = (np.empty((a.size, 6, 6)) if pairs else None), np.empty((a.size, 3, 3))
    for j, (fk, fn, zk) in enumerate(((e, 0, 0), (1j * e, 0, 0), (0, e, 0), (0, 1j * e, 0), (0, 0, 1.0), (0, 0, 1j)) if pairs else ()):
        fk, fn, zk = (np.broadcast_to(np.asarray(v, complex), a.shape) for v in (fk, fn, zk))
        of = ca2 * fk - b2 * np.conj(fn) + cab2 * zk
        on = ca2 * fn - b2 * np.conj(fk) + cab2 * np.conj(zk)
        oz = d * zk - ab * np.conj(fn) - np.conj(ab) * fk
        pair[:, :, j] = np.stack([of.real, of.imag, on.real, on.imag, oz.real, oz.imag], 1)
    for j, (f0, z0) in enumerate(((e, 0), (1j * e, 0), (0, 1.0))):
        f0, z0 = (np.broadcast_to(np.asarray(v, complex), a.shape) for v in (f0, z0))
        of = ca2 * f0 - b2 * np.conj(f0) + cab2 * z0
        oz = d * z0 - ab * np.conj(f0) - np.conj(ab) * f0
        zero[:, :, j] = np.stack([of.real, of.imag, oz.real], 1)
    return pair, zero


def _pair_index(box):
    """Coordinates of ``box`` (``periodic.carry_box``) grouped by pair of orders: (pair [pairs, 6], zero [3]) in the
    order of ``_excitation_blocks``; -1 where the box does not hold the coordinate (no transverse states)."""
    p, reps, neg, n_z = box["index"].shape[0], box["reps"], box["neg"], box["n_z"]
    j = np.arange(reps.size)
    if box["transverse"]:
        pair = np.stack([n_z + reps, n_z + p + reps, n_z + neg[reps], n_z + p + neg[reps], 1 + j, 1 + reps.size + j], 1)
        return pair, np.array([n_z + box["zero"], n_z + p + box["zero"], 0])
    return np.stack([np.full(j.size, -1)] * 4 + [1 + j, 1 + reps.size + j], 1), np.array([-1, -1, 0])


def base_response(resp):
    """The tables of ``train_response`` for the box that holds Mz alone (what an ideal crusher at the end of the
    repetition leaves): the entries of ``resp`` between the coordinates of order 0."""
    zero = (0,) * len(resp["box_y"]["k"])
    ix = [2] if resp["transverse"] else [0]                              # Z_0 among the coordinates of x
    return dict(resp, K=resp["K"][:, ix, :3], c=resp["c"][:, ix], L=resp["L"][:, :, :3], odd_K=resp["odd_K"][ix, :3], odd_c=resp["odd_c"][ix],
                odd_L=resp["odd_L"][:3], box_x=P.carry_box(zero, False), box_y=P.carry_box(zero, True), pairs=0, transverse=False)


def _interpolate(q, qs, table, odd):
    """``table`` [point, ...] at the columns of ``q`` [N, point]: the entries marked ``odd`` are held divided by
    sin(theta) and get it back (``qs`` = q x sin(theta) of the column). [N, ...]. Two products of whole tables, one
    with the odd entries set to zero and one with the others: no entry is picked out one by one."""
    flat = table.reshape(table.shape[0], -1)
    odd = np.broadcast_to(odd, table.shape[1:]).ravel()
    if not odd.any():
        out = q @ flat
    elif odd.all():
        out = qs @ flat
    else:
        out = q @ np.where(odd[None, :], 0.0, flat) + qs @ np.where(odd[None, :], flat, 0.0)
    return out.reshape((q.shape[0],) + table.shape[1:])


def carry_columns(resp, ae, be, ar, br, tail_turns=0.0, weights=None):
    """The periodic echoes of slice columns from the tables of ``train_response``.

    ``ae``, ``be``, ``ar``, ``br`` [column, node]: the rotation of the excitation and of the refocusing pulse (played
    about x; the block list turns it by 90 degrees) at every slice column and transmit scale. ``tail_turns``: the
    dephasing of the spoiler on the crusher axis in crusher lobes (0 when the spoiler plays on an axis of its own).
    In the frame in which the refocusing axis is y and which is translated along the crusher axis by the phase of
    ``ar`` per crusher, the train is the one of the tables; the excitation is (ae, be x conj(turn)), and the
    transverse part of the state it meets has been turned about z by ``tail_turns`` x arg(ar), the translation that
    the spoiler was not paired with. Then x = K E x + c is one dense solve per column, and echo = turn x (L E x + l).

    Returns the echoes [column, node, echo]; with ``weights`` [column] instead the sum over the columns of weights x
    echoes, [node, echo], complex (the integral through the slice)."""
    d_x, d_y, pairs = resp["K"].shape[1], resp["K"].shape[2], resp["pairs"]
    n_z, n_g = ae.shape
    chunk = max(1, int(CARRY_BYTES / (8.0 * 3 * d_x * d_y * n_g)))
    if n_z > chunk:                                                     # bounded memory: the matrices of a chunk of columns at a time
        parts = [carry_columns(resp, ae[i:i + chunk], be[i:i + chunk], ar[i:i + chunk], br[i:i + chunk], tail_turns,
                               None if weights is None else weights[i:i + chunk]) for i in range(0, n_z, chunk)]
        return np.concatenate(parts, 0) if weights is None else sum(parts)
    ae, be, ar, br = (v.ravel() for v in (ae, be, ar, br))
    n = ae.size
    turn = np.exp(1j * (np.angle(br) + 0.5 * np.pi))
    cos = 1 - 2 * np.abs(br) ** 2
    sin = 2 * np.abs(br) * np.sqrt(np.clip(1 - np.abs(br) ** 2, 0.0, None))
    q = kspace.bary_matrix(resp["x"], resp["w"], cos)                    # [N, point]
    qs = q * sin[:, None]
    pair, zero = _excitation_blocks(resp["box_y"], ae, be * np.conj(turn), tail_turns * np.angle(ar), pairs=bool(pairs))
    if not resp["transverse"]:                                          # x holds the gratings only: Z_0, and Re Z_k, Im Z_k of every pair
        pair, zero = (None if pair is None else pair[:, :, 4:]), zero[:, :, 2:]
    big_k = _interpolate(q, qs, resp["K"], resp["odd_K"])                # [N, x, y]
    const = _interpolate(q, qs, resp["c"], resp["odd_c"])
    step = big_k[:, :, :3] @ zero                                        # K E on the coordinates of x
    if pairs:
        step = np.concatenate([step, (big_k[:, :, 3:].reshape(n, d_x, pairs, 6) @ pair[:, None]).reshape(n, d_x, -1)], 2)
    if d_x == 1:
        x = const / (1 - step[:, :, 0])
    else:
        step *= -1
        step[:, np.arange(d_x), np.arange(d_x)] += 1.0
        x = np.linalg.solve(step, const[..., None])[..., 0]
    w0 = zero.shape[2]
    y = np.einsum("nij,nj->ni", zero, x[:, :w0])                         # y = E x
    if pairs:
        y = np.concatenate([y, np.einsum("nij,npj->npi", pair, x[:, w0:].reshape(n, pairs, -1)).reshape(n, -1)], 1)
    y = np.where(resp["odd_L"][None], y * sin[:, None], y)
    if weights is None:
        big_l = (q @ resp["L"].reshape(q.shape[1], -1)).reshape(n, -1, d_y)
        return (turn[:, None] * (np.einsum("nej,nj->ne", big_l, y) + (q @ resp["l"]) * sin[:, None])).reshape(n_z, n_g, -1)
    y_g = np.ascontiguousarray(y.reshape(n_z, n_g, d_y).transpose(1, 0, 2))             # [node, column, y]
    wq = ((np.repeat(weights, n_g) * turn)[:, None] * q).reshape(n_z, n_g, -1)           # weights x turn x interpolation: [column, node, point], complex
    mom = np.matmul(np.ascontiguousarray(wq.transpose(1, 2, 0)), y_g.astype(complex))    # [node, point, y]: the sum over the columns
    return np.tensordot(mom, resp["L"], axes=([1, 2], [0, 2])) + (wq * sin.reshape(n_z, n_g, 1)).sum(0) @ resp["l"]


def carry_columns_factor(echoes):
    """Slice columns of the carried part per column of the engine's quadrature: an echo of a train oscillates through
    the slice with the local refocusing angle, the faster the longer the train (``CARRY_COLUMNS``: echoes up to which
    the engine's own points, twice and four times as many are used)."""
    return 1 if echoes <= CARRY_COLUMNS[0] else 2 if echoes <= CARRY_COLUMNS[1] else 4


def _train_columns(tbw, window, flip, beta, gn, points, width, two_sided=False):
    """Rotations of the excitation and of the refocusing pulse at every slice column and transmit scale:
    (ae, be, ar, br [column, node], weights of the integral through the slice or None for a hard pulse). ``two_sided``
    False: z >= 0 with the weight of both sides, for echoes whose mirror column is the complex conjugate."""
    if tbw is None:                                                     # hard pulses: one column
        he, hr = 0.5 * flip * gn[None, :], 0.5 * beta * gn[None, :]
        return np.cos(he) + 0j, -1j * np.sin(he), np.cos(hr) + 0j, -1j * np.sin(hr), None
    sel = B.Selective(tbw, window)
    zeta, wz = O.slice_columns(tbw, two_sided=two_sided, window=window, points_per_thickness=points)
    ae, be = O.selective_rotation(sel, flip * gn[None, :], zeta[:, None])
    ar, br = O.selective_rotation(library.refocusing_pulse(sel, width), beta * gn[None, :], zeta[:, None])
    return ae, be, ar, br, wz


def _slice_sum(resp, rot, tail_turns, mirror=False):
    """[node, echo]: ``carry_columns`` integrated through the slice. ``mirror``: the columns are those of z >= 0 and
    the mirror column gives the complex conjugate (true behind an ideal crusher: the real part is the integral)."""
    ae, be, ar, br, wz = rot
    if wz is None:
        return carry_columns(resp, ae, be, ar, br, tail_turns)[0]
    out = carry_columns(resp, ae, be, ar, br, tail_turns, wz)
    return out.real.astype(complex) if mirror else out


def _tail_turns(seq):
    """Dephasing of the spoiler on the first axis, in crusher lobes (0 when it plays on another axis)."""
    prog = B.compile_ops(seq, frame=False)
    rf = [i for i, op in enumerate(prog.ops) if op.kind == "rf"]
    if len(rf) < 2:
        return 0.0
    lobe = sum(op.n[0] for op in prog.ops[rf[-1] + 1:] if op.kind == "free")
    adc = max(i for i, op in enumerate(prog.ops) if op.kind == "adc")
    crusher = sum(op.n[0] for op in prog.ops[rf[-1] + 1:adc] if op.kind == "free")
    return 0.0 if not crusher else float(lobe - crusher) / crusher


def _table_train(eng, rp, classes, gn, columns=None, carry=None, split=True):
    """[class, node, echo]: a spin-echo train with one refocusing angle (``fse``, ``mese``, and ``dwse``, a train of
    one echo), through ``train_response`` and ``carry_columns``: the rotation of the excitation and of the refocusing
    pulse at every slice column and transmit scale, the periodic state with what reaches the next excitation, and
    the integral through the slice.

    Two grids through the slice. The echoes behind an ideal crusher (only Mz carried) are integrated on
    ``TRAIN_COLUMNS`` times the points of the engine's quadrature (an echo oscillates through the slice), over z >= 0:
    there the mirror column gives the complex conjugate. What the gratings and the transverse states that survive the
    repetition add to them is solved on a coarser grid. Where transverse states are carried past a spoiler on the
    crusher axis, a column and its mirror are NOT conjugates (the turn about z that the spoiler was not paired with has
    the other sign at -z): BOTH sides of the slice are solved then, on ``CARRY_COLUMNS`` times the engine's points;
    otherwise z >= 0 on the engine's own points (``mrgym validate``, rows J23c and J24c, measure what the coarser grid
    costs; ``tests/test_served_prepared.py`` holds the two-sided integral).
    ``carry`` = (orders, transverse) for every class instead of ``carry_of``; ``columns``: the finer grid in multiples
    of the engine's; ``split`` False: everything on the finer grid."""
    entry = console.REGISTRY[rp["sequence"]]
    tbw, window = eng.pulse(rp)
    seq, labels = entry.blocks(rp, **_system(eng))
    flip = math.radians(rp["flip_deg"])
    beta = math.radians(rp["refocus_deg"] if "refocus_deg" in rp else 2 * rp["flip_deg"])
    width = rp.get("refocus_slice_ratio")
    fine = _train_columns(tbw, window, flip, beta, gn, 6 * (columns or TRAIN_COLUMNS), width)
    factor = carry_columns_factor(len(labels))
    coarse = fine if tbw is None else _train_columns(tbw, window, flip, beta, gn, 6 * factor, width)
    grids = {}

    def both(points):                                                   # both sides of the slice, ``points`` per thickness and time-bandwidth
        if points not in grids:
            grids[points] = _train_columns(tbw, window, flip, beta, gn, points, width, two_sided=True)
        return grids[points]
    turns = _tail_turns(seq)
    out = np.empty((len(classes), gn.size, len(labels)), complex)
    tissues = _class_spins(eng, classes)
    boxes = [carry_of(seq, tis) if carry is None else carry for tis in tissues]
    for box in sorted(set(boxes)):                                      # the classes of one box are stepped together
        group = [i for i, b in enumerate(boxes) if b == box]
        orders, transverse = box
        for i, full in zip(group, train_response(seq, [tissues[i] for i in group], len(labels), orders, transverse)):
            base = base_response(full)
            carried = any(orders) or transverse
            if tbw is None:                                             # hard pulses: one column
                out[i] = _slice_sum(full if carried else base, fine, turns)
            elif not carried:
                out[i] = _slice_sum(base, fine, turns, mirror=True)
            elif transverse and turns:                                  # no mirror symmetry: both sides of the slice
                grid = both(6 * ((columns or TRAIN_COLUMNS) if not split else factor))
                out[i] = _slice_sum(full, grid, turns) if not split else \
                    _slice_sum(base, fine, turns, mirror=True) + _slice_sum(full, grid, turns) - _slice_sum(base, grid, turns)
            elif not split:
                out[i] = _slice_sum(full, fine, turns, mirror=True)
            else:
                out[i] = _slice_sum(base, fine, turns, mirror=True) + _slice_sum(full, coarse, turns, mirror=True) - _slice_sum(base, coarse, turns, mirror=True)
    return out


def _table_flip(eng, rp, classes, gn):
    """[class, node, 1]: the RF-spoiled gradient echo. Ideal spoiling in every column; the departure from it on
    ``FLIP_NODES`` Chebyshev nodes of the flip angle per class, interpolated (``kspace.gre_steady`` does the same with
    the engine's own phase graph)."""
    entry = console.REGISTRY[rp["sequence"]]
    tbw, window = eng.pulse(rp)
    tissues = [_tissue(eng.w["tissues"][eng.w["names"][c]]) for c in classes]
    flip = math.radians(rp["flip_deg"])
    if tbw is None:
        theta, wgt = (flip * gn)[:, None], np.ones((gn.size, 1))        # [node, column]
    else:
        sel = B.Selective(tbw, window)
        zeta, wz = O.slice_columns(tbw, two_sided=False, window=window)
        a, b = O.selective_rotation(sel, (flip * gn)[:, None], zeta[None, :])
        theta = 2 * np.arcsin(np.clip(np.abs(b), 0.0, 1.0))             # the flip angle of every column
        wgt = wz[None, :] * (1j * np.exp(1j * np.angle(2 * np.conj(a) * b))).real      # its transverse phase, both sides of the slice
    scale = theta.ravel() / flip                                         # a hard pulse of the column's angle, as a transmit scale
    ideal_seq, _ = entry.blocks(rp, tier=1)
    ideal = periodic.steady_state(ideal_seq, O.grid(tissues, (0.0,), scale)).signal[0].reshape(len(classes), *theta.shape)
    out = (ideal * wgt[None]).sum(2)
    if eng.tier == 2:
        tn, tw = kspace.cheb_nodes(0.0, 1.02 * float(theta.max()) + 1e-3, FLIP_NODES) if tbw is not None else (theta.ravel(), None)
        seq, _ = entry.blocks(rp, tier=2)
        sp = O.grid(tissues, (0.0,), tn / flip)
        gap = (periodic.steady_state(seq, sp).signal[0] - periodic.steady_state(ideal_seq, sp).signal[0]).reshape(len(classes), tn.size)
        if tbw is None:
            out = out + gap
        else:
            q = kspace.bary_matrix(tn, tw, theta.ravel())               # [column, flip node]
            out = out + ((gap @ q.T).reshape(len(classes), *theta.shape) * wgt[None]).sum(2)
    return out[:, :, None]


SOLVERS = dict(columns=_table_columns, flip=_table_flip, train=_table_train)
SOLVER_OF = dict(megre="flip", fse="train", mese="train", dwse="train")


def solver_of(rp):
    """How the table of a realised protocol is solved. An echo train whose refocusing angle changes from echo to echo
    (a realised protocol written by hand: the design layer) has no single angle to reduce to: every column."""
    name = SOLVER_OF.get(rp["sequence"], "columns")
    return "columns" if name == "train" and np.ndim(rp.get("refocus_deg", 0.0)) else name


# What of a realised protocol does NOT enter the amplitude table (the geometry of the image, its sampling and its
# adjustments of the field; the transmit scale does enter). Everything else is part of the key under which an engine
# keeps a table, so that a parameter added later cannot be forgotten here.
NOT_IN_TABLE = frozenset({"fov_mm", "phase_fov", "fov_pe_mm", "matrix", "matrix_pe", "lines_full", "lines", "slice_mm", "bandwidth_hz_per_px", "pe_dir",
                          "ro_polarity", "partial_fourier", "phase_oversampling", "averages", "centre_y_mm", "centre_x_mm", "rotation_deg", "cf_shift_hz",
                          "shim_y_hz_per_mm", "shim_x_hz_per_mm", "echo_order", "shots", "echo_polarities", "readout", "diffusion_dir", "partitions",
                          "fat_sat", "fatsat_offset_hz", "echo_times_ms"})
TABLES_KEPT = 6                     # amplitude tables an engine keeps (a scan that changes the geometry only finds its table)


def table_key(rp):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in rp.items() if k not in NOT_IN_TABLE))


def table(eng, rp, solver=None, nodes=None, columns=None, carry=None, split=True):
    """Amplitudes of a scan served from the core, per unit of weight x proton density:
    dict(a [class, node, sample] complex, classes (indices into the world's names), gn, gw (nodes of the transmit
    scale and their barycentric weights)). Classes with equal relaxation and diffusion numbers are solved once, and
    the engine keeps the tables of its last ``TABLES_KEPT`` protocols (``table_key``).
    ``solver``, ``nodes``, ``columns``, ``carry``, ``split``: another solver of ``SOLVERS``, another number of nodes,
    (echo trains) of slice columns per column of the engine's quadrature, another box of carried states (orders,
    transverse), and everything on one grid through the slice, for checks; such a table is not kept."""
    plain = solver is None and nodes is None and columns is None and carry is None and split
    kept = eng.__dict__.setdefault("_tables", {})
    key = table_key(rp)
    if plain and key in kept:
        return kept[key]
    classes = _classes(eng)
    gn, gw = _nodes(eng, rp, nodes)
    keys = [tuple(sorted(_tissue(eng.w["tissues"][eng.w["names"][c]]).items())) for c in classes]
    first = {k: i for i, k in reversed(list(enumerate(keys)))}          # the first class with each set of numbers
    distinct = sorted(set(first.values()))
    name = solver_of(rp) if solver is None else solver
    more = dict(carry=carry) if name in ("train", "columns") else {}
    solved = SOLVERS[name](eng, rp, [classes[i] for i in distinct], gn, **(dict(more, columns=columns, split=split) if name == "train" else more))
    a = solved[[distinct.index(first[k]) for k in keys]]
    out = dict(a=a, classes=classes, gn=gn, gw=gw)
    if plain:
        if len(kept) >= TABLES_KEPT:
            kept.pop(next(iter(kept)))
        kept[key] = out
    return out


def _flip_scale(eng, rp, c):
    """Transmit scale of every voxel of class ``c``."""
    return rp.get("tx_scale", 1.0) * eng.b1[eng.sup[c]]


def amplitudes(eng, rp, c, tab=None):
    """Amplitude maps of class ``c`` over its support, one per sample of the cycle (``console.Sampling.samples``):
    weight x proton density x the table interpolated at the voxel's transmit scale. For a spin-echo sequence the
    sample carries the T2 decay up to the centre of its echo (the six with a closed form leave it to the readout
    stage); for a gradient echo it is the amplitude at the pulse."""
    tab = table(eng, rp) if tab is None else tab
    if c not in tab["classes"]:
        return [np.zeros(0, complex) for _ in range(tab["a"].shape[2])]
    cls = eng.w["tissues"][eng.w["names"][c]]
    q = kspace.bary_matrix(tab["gn"], tab["gw"], _flip_scale(eng, rp, c))
    maps = q @ tab["a"][tab["classes"].index(c)]                         # [support, sample]
    scale = eng.w["weights"][c][eng.sup[c]] * cls["pd"]
    return [maps[:, s] * scale for s in range(maps.shape[1])]


# ---------------------------------------------------------------------------------------------- image formation
def readout_function(eng, rp, cls, tau):
    """g_c(tau) of a spin-echo sample: evolution of a class around the centre of its echo, exp(-tau / T2)
    exp(-|tau| / T2') x the spectrum of fat at tau. The decay up to the echo centre is in the sample."""
    eff = eng.effects
    t_dec, t_spec = (tau if "t2star" in eff else 0 * tau), (tau if "chemshift" in eff else 0 * tau)
    f = np.exp(-1e3 * t_dec / cls["t2_ms"]) * np.exp(-np.abs(1e3 * t_dec) / cls["t2p_ms"])
    if not cls["fat"]:
        return f.astype(complex)
    peaks = tissue.FAT_PEAKS if eng.fat_model == "multipeak" else ((tissue.FAT_PPM, 1.0),)
    return f * S.fat_spectrum(t_spec, eng.w["b0_t"], peaks)


def noise_per_sample(eng, rp, averages=None):
    """Standard deviation of the receiver noise in one component of one k-space sample (``Engine.acquire_kspace``)."""
    k_ro = console.SYSTEM["ro_os"] * rp["matrix"]
    nex = rp["averages"] if averages is None else averages
    return eng.w["system"]["sigma0"] * math.sqrt(k_ro * rp["bandwidth_hz_per_px"] / (nex * rp.get("partitions", 1)))


def _splat_gain(eng, rp, k_ro, c_ro):
    """kappa: the largest factor by which the displacement operator and its deapodisation can raise a k-space sample
    (the kernel spreads a source over ``KB_W`` grid points with positive weights; the k-space is divided by the
    transform of the kernel). 1 when the displacement is switched off."""
    if "b0shift" not in eng.effects:
        return 1.0
    frac = np.linspace(0.0, 1.0, 65)[:, None]
    u = (np.arange(kspace.KB_W)[None, :] - (kspace.KB_W // 2 - 1) - frac) / (kspace.KB_W / 2)
    spread = (np.i0(kspace.KB_BETA * np.sqrt(np.clip(1 - u * u, 0, None))) / np.i0(kspace.KB_BETA)).sum(1).max()
    return float(spread * np.abs(kspace._deapodise(k_ro, c_ro)).max())


def _interpolation(eng, rp, tab):
    """Per class of the table: the barycentric matrix Q [support, node] that takes the table to the voxels."""
    return [kspace.bary_matrix(tab["gn"], tab["gw"], _flip_scale(eng, rp, c)) for c in tab["classes"]]


def line_basis(eng, rp, tab, g, kappa, tol=None, interp=None):
    """The joint basis of the amplitude along the phase-encode lines and of the evolution along the readout.

    ``g`` [class, readout sample]. The matrix m[(class, node); (sample of the cycle, readout sample)] = a(c, j, s)
    g_c(x), each row weighted with what its class can add to a k-space sample, is factorised by an SVD and cut at the
    smallest rank T for which ``bound`` <= ``tol`` (None: the rule of the module docstring):

        m = sum_t U[c, j, t] H[t, s, x] + R,       bound = kappa max_(s, x) sum_c mass_c Lambda_c max_j |R[c, j, s, x]|.

    Returns (U [class, node, T], H [T, sample, readout sample], bound reached, tolerance)."""
    a, classes = tab["a"], tab["classes"]
    n_c, n_j, n_s = a.shape
    d = eng.d
    interp = _interpolation(eng, rp, tab) if interp is None else interp
    mass, leb = np.empty(n_c), np.empty(n_c)
    rx = np.abs(eng.w["fields"]["rx"])
    for i, c in enumerate(classes):
        sup = eng.sup[c]
        cls = eng.w["tissues"][eng.w["names"][c]]
        mass[i] = d * d * rp["slice_mm"] * float((eng.w["weights"][c][sup] * rx[sup]).sum()) * cls["pd"]
        leb[i] = float(np.abs(interp[i]).sum(1).max())
    wgt = mass * leb                                                     # what one unit of residual of class c can add to a sample
    if tol is None:
        sd = noise_per_sample(eng, rp, MAX_AVERAGES)
        tol = LINE_NOISE_FRACTION * sd if sd > 0 else LINE_REL_TOL * float((wgt * np.abs(a).max((1, 2)) * np.abs(g).max(1)).sum())
    # g has at most as many independent rows as there are classes: g = gc gb with orthonormal rows gb, nothing dropped.
    # The matrix to factorise is then m' [(class, node); (sample, row of gb)], and m = m' (1 x gb) has its singular values.
    gu, gs, gb = np.linalg.svd(g, full_matrices=False)
    gc = gu * gs                                                         # [class, L]
    n_l = gb.shape[0]
    m = ((wgt[:, None, None] * a)[:, :, :, None] * gc[:, None, None, :]).reshape(n_c * n_j, n_s * n_l)
    u, sv, vh = np.linalg.svd(m, full_matrices=False)

    def bound(rank):                                                    # the residual where it acts: on (sample, readout sample)
        resid = (m - (u[:, :rank] * sv[:rank]) @ vh[:rank]).reshape(n_c, n_j, n_s, n_l)
        worst = 0.0
        for s in range(n_s):
            worst = max(worst, float(np.abs(resid[:, :, s, :] @ gb).max(1).sum(0).max()))
        return kappa * worst

    # the smallest rank whose bound fits, by bisection (the bound is checked at the rank returned; at full rank the
    # residual is rounding)
    lo, hi = -1, sv.size
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if bound(mid) <= tol else (mid, hi)
    rank = hi
    uu = u[:, :rank].reshape(n_c, n_j, rank) / np.where(wgt > 0, wgt, 1.0)[:, None, None]
    return uu, (sv[:rank, None] * vh[:rank]).reshape(rank, n_s, n_l) @ gb, bound(rank), tol


def form_lines(eng, rp, tab, plan, tol=None, details=None, only=None):
    """Noise-free k-space of every image of a spin-echo scan served from the core, [image, lines_full, os * matrix].

    ``tol``: the bound of the truncation (None: the rule of the module docstring; 0: no truncation, every term kept).
    ``details``: a dict that receives the number of terms, the bound and the tolerance. ``only``: the index of one
    tissue class: the k-space of that class alone (what a mover modulates)."""
    w, n, d, eff = eng.w, eng.n, eng.d, eng.effects
    if only is not None:
        if only not in tab["classes"]:
            return np.zeros((len(plan.images), rp["lines_full"], console.SYSTEM["ro_os"] * rp["matrix"]), complex)
        i_c = tab["classes"].index(only)
        tab = dict(tab, a=tab["a"][i_c:i_c + 1], classes=[only])
    classes = tab["classes"]
    ro_os = console.SYSTEM["ro_os"]
    k_ro, k_pe = ro_os * rp["matrix"], rp["lines_full"]
    fov_ro_os = ro_os * rp["fov_mm"]
    fov_pe_os = rp["fov_pe_mm"] * k_pe / rp["matrix_pe"]
    c_ro, c_pe = d / fov_ro_os, d / fov_pe_os
    tau = rp["ro_polarity"] * (np.arange(k_ro) - k_ro // 2) / (k_ro * rp["bandwidth_hz_per_px"])
    g = np.array([readout_function(eng, rp, w["tissues"][w["names"][c]], tau) for c in classes])
    # ---- which sample fills which line of which image: [image, lines_full], -1 = not acquired
    acquired = eng._acquired(rp)
    if plan.lines is None:
        which = np.array([np.where(acquired, s, -1) for s, _ in plan.images])
    else:
        which = np.array([np.where(acquired, np.asarray(ln), -1) for ln in plan.lines])
    clock = [time.process_time()]
    kappa = _splat_gain(eng, rp, k_ro, c_ro)
    interp = _interpolation(eng, rp, tab)
    uu, h, reached, tol_used = line_basis(eng, rp, tab, g, kappa, tol, interp)
    n_t = uu.shape[2]
    clock.append(time.process_time())
    if details is not None:
        details.update(terms=n_t, bound=reached, tolerance=tol_used, kappa=kappa, noise_per_sample=noise_per_sample(eng, rp))
    out = np.zeros((len(plan.images), k_pe, k_ro), complex)
    if n_t == 0:
        return out
    # ---- coefficient images on the master grid, one per term: [pixel, term]
    swap = rp["pe_dir"] == "x"                                          # the acquisition frame is [pe, ro]: master arrays transposed
    comp = np.zeros((n * n, n_t), complex)
    for i, c in enumerate(classes):
        sup = eng.sup[c].T if swap else eng.sup[c]
        wt = (w["weights"][c].T if swap else w["weights"][c])[sup] * w["tissues"][w["names"][c]]["pd"]
        q = interp[i]
        if swap:                                                        # the voxels of the support in the order of the transposed grid
            order = np.empty(eng.sup[c].shape, np.int64)
            order[eng.sup[c]] = np.arange(q.shape[0])
            q = q[order.T[sup]]
        comp[np.flatnonzero(sup)] += (q @ uu[i]) * wt[:, None]
    rx = w["fields"]["rx"].astype(complex)
    comp *= (rx.T if swap else rx).reshape(-1, 1)
    axis = None
    if "b0shift" in eff:                                                # displacement along the readout by off-resonance
        op, axis = eng._shift_operator(rp, False, fov_pe_os)
        comp = np.asarray(op @ comp)
    comp = comp.reshape(n, n, n_t)
    cy, cx = rp.get("centre_y_mm", 0.0), rp.get("centre_x_mm", 0.0)
    p0, r0 = (cx, cy) if swap else (cy, cx)
    if "aa_filter" in eff:
        x = (np.arange(n) - n // 2) * d
        comp[:, np.abs(x - r0) > fov_ro_os / 2] = 0
    # ---- the Fourier integrals: along the phase-encode axis first (master grid -> lines), then along the readout.
    # Rows and columns of the master grid that are empty in every term add nothing to the sums: they are left out,
    # and the centre of the field of view is expressed in the cropped grid.
    filled = np.any(comp != 0, axis=2)
    rows, cols = np.flatnonzero(filled.any(1)), np.flatnonzero(filled.any(0))
    if rows.size == 0:
        return out
    (ra, rb), (ca, cb) = (rows[0], rows[-1] + 1), (cols[0], cols[-1] + 1)
    p_c = p0 / d + n // 2 - ra - (rb - ra) // 2
    r_c = r0 / d + n // 2 - ca - (cb - ca) // 2
    hpad = np.concatenate([h, np.zeros((n_t, 1, k_ro))], 1)              # sample -1: a line that is not acquired
    clock.append(time.process_time())
    spent = 0.0
    for t in range(n_t):                                                 # one term at a time: bounded memory
        t0 = time.process_time()
        one = np.ascontiguousarray(comp[ra:rb, ca:cb, t])
        k_t = kspace.zoom_dft(kspace.zoom_dft(one, 0, k_pe, c_pe, centre=p_c), 1, k_ro, c_ro, centre=r_c)
        spent += time.process_time() - t0
        for i in range(len(plan.images)):
            out[i] += hpad[t][which[i]] * k_t                            # the weight of line p is H[t, s(p), x]
    if axis is not None:
        out *= kspace._deapodise(k_ro, c_ro)[None, :]
    clock.append(time.process_time())
    if details is not None:                                             # CPU seconds: the basis, the coefficient images, the transforms, the weighting
        details.update(cpu_s=dict(basis=clock[1] - clock[0], images=clock[2] - clock[1], transforms=spent, weighting=clock[3] - clock[2] - spent),
                       grid=(int(rb - ra), int(cb - ca)))
    return out * (d * d * rp["slice_mm"])


def line_times(rp, plan, image):
    """Time (s) at which every phase-encode line of image ``image`` is acquired in every average, [lines_full,
    averages], counted from the first excitation. One line per repetition (``mese``, ``dwse``): linear order, the
    averages of a line in succession, plus the echo time of the image (the engine's rule for ``se``). An echo train
    (``fse``): the lines of one band are read in successive shots, a shot reads one line of every band, and a line
    is read at the echo time of the echo that fills it:
        t = ((index of the line in its band) x averages + average) x TR + echo time of its echo."""
    k_pe, nex = rp["lines_full"], rp["averages"]
    if plan.lines is None:
        order = np.arange(k_pe) - k_pe // 2 + (rp["lines"] - k_pe // 2)
        return ((order[:, None] * nex + np.arange(nex)[None, :]) * rp["tr_ms"] + plan.images[image][1]) * 1e-3
    lines = np.asarray(plan.lines[image])
    acquired = lines >= 0
    shot = np.zeros(k_pe, np.int64)
    shot[acquired] = np.arange(int(acquired.sum())) % plan.shots
    return ((shot[:, None] * nex + np.arange(nex)[None, :]) * rp["tr_ms"] + ((lines + 1) * rp["echo_spacing_ms"])[:, None]) * 1e-3


def form(eng, rp, key, moving):
    """``kspace.FORMATION["core"]``: the noise-free k-space of every image of a scan served from the core.

    Periodic motion (``moving``) is the engine's line modulation (``Engine._motion``): every line of a moving class
    is multiplied by the factor of the time at which the line is read (``line_times``). The amplitudes are those of
    the object at rest: motion during an echo train does not change its echoes, and nothing is dephased by moving
    along a crusher or a diffusion lobe (docs/physics/limits.md, block 7)."""
    entry, plan = console.REGISTRY[rp["sequence"]], console.sampling(rp)
    tab = table(eng, rp)
    if entry.echo == "gre":                                             # one sample, read at several echo times: the engine's readout stage
        n_c = len(eng.w["names"])
        amps = [amplitudes(eng, rp, c, tab) for c in range(n_c)]
        polarity = rp.get("echo_polarities") or [rp["ro_polarity"]] * len(plan.images)
        ks = [None] * len(plan.images)
        for pol in sorted(set(polarity), reverse=True):                 # one displacement operator per readout direction
            for i, (s, te) in enumerate(plan.images):
                if polarity[i] == pol:
                    one, rp_i = [amps[c][s] for c in range(n_c)], dict(rp, ro_polarity=pol)
                    ks[i] = eng.kspace(rp_i, te, one)
                    if moving:
                        ks[i] = eng._motion(ks[i], rp_i, te, s, one, key)
        return ks
    ks = form_lines(eng, rp, tab, plan)
    if not moving:
        return list(ks)
    alone = {}

    def of_class(c):
        if c not in alone:
            alone[c] = form_lines(eng, rp, tab, plan, only=c)
        return alone[c]
    return [eng._motion(ks[i], rp, te, s, None, key, times=line_times(rp, plan, i), of_class=lambda c, i=i: of_class(c)[i])
            for i, (s, te) in enumerate(plan.images)]


kspace.FORMATION["core"] = form
