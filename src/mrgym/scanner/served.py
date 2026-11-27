"""Scans served from the block-list core (``mrgym.core``): amplitudes of a sequence that is data, and image formation
with an amplitude that changes from one phase-encode line to the next.

A sequence of the registry with ``serves = "core"`` (``console.Sequence``) has no closed form in the engine. Its
realised protocol is turned into a block list (``mrgym.core.library.served``), the periodic state of one repetition is
solved (``mrgym.core.periodic``), and the samples of that repetition are the amplitudes of the scan: one per echo of
an echo train, one for the echo of a diffusion-weighted spin echo, one for a gradient echo. The sampling plan
(``console.Sampling``) says which sample makes which image and which sample fills which phase-encode line.

Amplitudes (``table``). For every tissue class with a voxel in the world, on Chebyshev nodes of the transmit scale
(nominal x B1+ x transmit gain x ``tx_scale``) that span the object:
  columns   echo trains and the diffusion-weighted spin echo: every slice column is solved with the rotation of the
            pulse at that column (Cayley-Klein parameters of the excitation and of the refocusing pulse on its 1.5 x
            slice), and the samples are integrated through the slice. Imperfect refocusing therefore gives the
            stimulated echoes it gives on a scanner: they are not a correction, they are in the states.
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

Not here: fat saturation, periodic motion (refused), and chemical-shift or J-coupling effects inside an echo train
(fat is one species with its T1 and T2; its lines dephase and rephase around every echo).
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from ..core import blocks as B
from ..core import epg, library, periodic
from ..core import operators as O
from . import console, kspace, tissue
from . import signal as S

LINE_NOISE_FRACTION = 0.05          # bound of the line-basis truncation, in noise standard deviations of one k-space sample at MAX_AVERAGES
LINE_REL_TOL = 1e-7                 # the same bound for a world without noise, relative to the largest sample the object can make
MAX_AVERAGES = console._GEOM["averages"][1]
B1_NODES = (5, 16)                  # fewest and most Chebyshev nodes of the transmit scale for a smooth dependence (the engine's rule)
TRAIN_NODES = (1.0, 6, 72)          # echo trains: nodes = slope x echoes x refocusing angle (rad) x width of the range + offset, at most ``most``
TRAIN_COLUMNS = 4                   # echo trains: slice columns per column of the engine's quadrature
FLIP_NODES = kspace.PROFILE_NODES   # nodes of the flip angle on which the departure from ideal spoiling is solved (megre)


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


def _table_columns(eng, rp, classes, gn):
    """[class, node, sample]: every slice column solved with its rotation, integrated through the slice."""
    entry = console.REGISTRY[rp["sequence"]]
    tbw, window = eng.pulse(rp)
    tissues = [_tissue(eng.w["tissues"][eng.w["names"][c]]) for c in classes]
    if tbw is None:                                                     # hard pulses: one flip angle per voxel
        seq, labels = entry.blocks(rp)
        sp = O.grid(tissues, (0.0,), gn)
        sig = periodic.steady_state(seq, sp).signal                     # [sample, column]
        return np.moveaxis(sig.reshape(len(labels), len(classes), gn.size), 0, -1)
    seq, labels = entry.blocks(rp, selective=B.Selective(tbw, window))
    zeta, wz = O.slice_columns(tbw, two_sided=False, window=window)     # z >= 0; the mirror column gives the complex conjugate
    sp = O.grid(tissues, zeta, gn)
    sig = periodic.steady_state(seq, sp).signal.reshape(len(labels), len(classes), zeta.size, gn.size)
    return np.moveaxis((sig.real * wz[None, None, :, None]).sum(2), 0, -1).astype(complex)


def _with_rotations(seq, excitation, refocusing):
    """The block list ``seq`` with its first rf block replaced by the rotation ``excitation`` and every later one by
    ``refocusing`` (rows of Cayley-Klein parameters per column); delays, gradients and samples stay what they are."""
    seen = []

    def walk(blocks):
        out = []
        for blk in blocks:
            if blk.kind == "repeat":
                out.append(dataclasses.replace(blk, body=walk(blk.body), tables=()))
            elif blk.kind == "rf":
                out.append(B.rf(0.0, blk.phase_deg, crushed=blk.crushed, ck=refocusing if seen else excitation))
                seen.append(blk)
            else:
                out.append(blk)
        return tuple(out)
    return B.sequence(walk(seq.blocks), cycle_mm=seq.cycle_mm, name=seq.name)


def train_response(seq, tissues, n_echoes):
    """What an echo train does to the magnetisation it is given, as functions of the refocusing flip angle alone.

    In a train with equal delays and equal crushers either side of every refocusing pulse, a refocusing rotation
    (a, b) acts on the sampled echoes as the hard pulse of flip angle theta = 2 asin|b| about the transverse axis of
    phase arg(b) + 90 degrees: the phase of a is a turn about z that is the same before and after the pulse, a
    precession, and the echo refocuses it. The states split into two families that the pulses never mix: the one
    that starts from the transverse magnetisation after the excitation (odd orders at the pulses: the echoes) and
    the one that starts from the longitudinal magnetisation (even orders: free induction decays that are never at
    order 0 when an echo is sampled, and Mz itself). With m0 the transverse and c0 the longitudinal magnetisation right
    after the excitation, per unit of Mz before it, in the frame in which the refocusing axis is y:

        echo_n = Mz (u A_n(theta) + v B_n(theta)),   u = -Im m0, v = Re m0        (real-linear in m0)
        Mz at the start of the next repetition = c0 P(theta) Mz + R(theta)

    and every one of A_n, B_n, P, R is a polynomial of degree <= n in cos(theta) (each pulse enters with cos theta,
    sin theta and the echoes are even in theta). They are therefore known exactly from three single shots (excitation
    about x, about y, none) on ``n_echoes`` + 3 Chebyshev points of cos(theta): no interpolation error.

    ``seq``: the block list of the train with hard pulses (its structure is what is used). Returns dict(x, w:
    the nodes of cos(theta) and their barycentric weights; A, B [tissue, node, echo]; P, R [tissue, node])."""
    m = int(n_echoes) + 3
    xn, xw = kspace.lobatto_nodes(-1.0, 1.0, m)
    half = 0.5 * np.arccos(np.clip(xn, -1.0, 1.0))
    ref = np.stack([np.cos(half), 0 * half, 0 * half, -np.sin(half)], 1)        # hard pulse about x of angle theta: a = cos, b = -i sin
    r2 = math.sqrt(0.5)
    sp = O.grid(tissues, np.zeros(m), (1.0,))                           # one column per node; rf blocks given by ck read ``column``
    out = {}
    for name, row in (("A", (r2, 0.0, 0.0, -r2)), ("B", (r2, 0.0, r2, 0.0)), ("C", (1.0, 0.0, 0.0, 0.0))):
        prog = B.compile_ops(_with_rotations(seq, [row] * m, ref), frame=False)
        kk = tuple((r + 1) // 2 for r in prog.reach)                    # higher orders cannot return before the crusher (``periodic``)
        res = epg.run(prog, sp, kmax=kk, prune=True)
        out[name] = (res.signal.reshape(-1, len(tissues), m), res.state.z[tuple(kk)].real.reshape(len(tissues), m))
    return dict(x=xn, w=xw, A=np.moveaxis(out["A"][0], 0, -1), B=np.moveaxis(out["B"][0], 0, -1), R=out["A"][1], P=out["C"][1] - out["A"][1],
                R_other=out["B"][1])


def _table_train(eng, rp, classes, gn, columns=None):
    """[class, node, echo]: an echo train with one refocusing angle, through ``train_response``: the rotation of the
    excitation and of the refocusing pulse at every slice column and transmit scale gives m0, c0, theta and the phase
    of the refocusing axis; the periodic Mz and the echoes follow in closed form, and are integrated through the
    slice on ``TRAIN_COLUMNS`` times the points of the engine's quadrature (an echo oscillates through the slice)."""
    entry = console.REGISTRY[rp["sequence"]]
    tbw, window = eng.pulse(rp)
    tissues = [_tissue(eng.w["tissues"][eng.w["names"][c]]) for c in classes]
    seq, labels = entry.blocks(rp)
    resp = train_response(seq, tissues, len(labels))
    flip, beta = math.radians(rp["flip_deg"]), math.radians(rp["refocus_deg"])
    if tbw is None:                                                     # hard pulses: [1, node]
        wz = np.ones(1)
        he, hr = 0.5 * flip * gn[None, :], 0.5 * beta * gn[None, :]
        (ae, be), (ar, br) = (np.cos(he) + 0j, -1j * np.sin(he)), (np.cos(hr) + 0j, -1j * np.sin(hr))
    else:
        sel = B.Selective(tbw, window)
        zeta, wz = O.slice_columns(tbw, two_sided=False, window=window, points_per_thickness=6 * (columns or TRAIN_COLUMNS))
        ae, be = O.selective_rotation(sel, flip * gn[None, :], zeta[:, None])
        ar, br = O.selective_rotation(library.refocusing_pulse(sel), beta * gn[None, :], zeta[:, None])
    del ar                                                              # the phase of a is a precession the echo refocuses
    m0, c0 = 2 * np.conj(ae) * be, np.abs(ae) ** 2 - np.abs(be) ** 2    # after the excitation, per unit Mz
    turn = np.exp(1j * (np.angle(br) + 0.5 * np.pi))                    # the refocusing axis against y
    m0 = m0 * np.conj(turn)
    q = kspace.bary_matrix(resp["x"], resp["w"], (1 - 2 * np.abs(br) ** 2).ravel())      # cos(theta) of every column: [column, node of cos theta]
    shape = m0.shape
    out = np.empty((len(classes), gn.size, len(labels)), complex)
    for i in range(len(classes)):
        mz = (q @ resp["R"][i]) / (1 - c0.ravel() * (q @ resp["P"][i]))
        echo = (mz * turn.ravel())[:, None] * (-m0.imag.ravel()[:, None] * (q @ resp["A"][i]) + m0.real.ravel()[:, None] * (q @ resp["B"][i]))
        echo = echo.reshape(*shape, len(labels))
        # z >= 0 with the weight of both sides: the mirror column gives the complex conjugate
        out[i] = (echo.real * wz[:, None, None]).sum(0) if tbw is not None else echo[0]
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
SOLVER_OF = dict(megre="flip", fse="train", mese="train")               # every other sequence: "columns"


def solver_of(rp):
    """How the table of a realised protocol is solved. An echo train whose refocusing angle changes from echo to echo
    (a realised protocol written by hand: the design layer) has no single angle to reduce to: every column."""
    name = SOLVER_OF.get(rp["sequence"], "columns")
    return "columns" if name == "train" and np.ndim(rp["refocus_deg"]) else name


def table(eng, rp, solver=None, nodes=None, columns=None):
    """Amplitudes of a scan served from the core, per unit of weight x proton density:
    dict(a [class, node, sample] complex, classes (indices into the world's names), gn, gw (nodes of the transmit
    scale and their barycentric weights)). Classes with equal relaxation and diffusion numbers are solved once.
    ``solver``, ``nodes``, ``columns``: another solver of ``SOLVERS``, another number of nodes and (echo trains) of
    slice columns per column of the engine's quadrature, for checks."""
    classes = _classes(eng)
    gn, gw = _nodes(eng, rp, nodes)
    keys = [tuple(sorted(_tissue(eng.w["tissues"][eng.w["names"][c]]).items())) for c in classes]
    first = {k: i for i, k in reversed(list(enumerate(keys)))}          # the first class with each set of numbers
    distinct = sorted(set(first.values()))
    name = solver_of(rp) if solver is None else solver
    solved = SOLVERS[name](eng, rp, [classes[i] for i in distinct], gn, **(dict(columns=columns) if name == "train" else {}))
    a = solved[[distinct.index(first[k]) for k in keys]]
    return dict(a=a, classes=classes, gn=gn, gw=gw)


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


def form_lines(eng, rp, tab, plan, tol=None, details=None):
    """Noise-free k-space of every image of a spin-echo scan served from the core, [image, lines_full, os * matrix].

    ``tol``: the bound of the truncation (None: the rule of the module docstring; 0: no truncation, every term kept).
    ``details``: a dict that receives the number of terms, the bound and the tolerance."""
    w, n, d, eff = eng.w, eng.n, eng.d, eng.effects
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


def form(eng, rp, key, moving):
    """``kspace.FORMATION["core"]``: the noise-free k-space of every image of a scan served from the core."""
    if moving:
        raise ValueError(f"{rp['sequence']} is served from the block-list core, which does not model periodic motion of the object "
                         "(docs/physics/limits.md, block 7)")
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
                    ks[i] = eng.kspace(dict(rp, ro_polarity=pol), te, [amps[c][s] for c in range(n_c)])
        return ks
    return list(form_lines(eng, rp, tab, plan))


kspace.FORMATION["core"] = form
