"""Spin operators and Hamiltonians of coupled spin-1/2 systems (protons).

A ``SpinSystem`` is one network of coupled sites. A site is one proton or a group of magnetically equivalent protons
(a methyl group: ``counts`` = 3). ``compile`` turns it, for a field strength and a reference frequency, into a list of
``Sector`` objects that hold the Hamiltonian in its eigenbasis.

Conventions (those of ``mrgym.design``; ``mrgym.spins.validate`` rows O3, E2, E3b and M1 compare the two)
  basis      product states |m_1 m_2 ...>, site 0 the most significant index, m descending (for a proton: alpha =
             +1/2 first). Spin matrices are the usual ones: Iz = diag(m), I+ |m> = sqrt(j(j+1) - m(m+1)) |m+1>.
  frequency  site i precesses at  nu_i = (delta_i - delta_ref) x 1e-6 x gamma x B0  (Hz) in the frame of the reference:
             a larger chemical shift is a higher frequency, and a positive frequency gains positive phase
             (docs/physics/conventions.md, section 5). The Hamiltonian, in Hz, is
                 H = sum_i nu_i I_iz + sum_(i<j) J_ij I_i . I_j           (strong coupling, the default)
                 H = sum_i nu_i I_iz + sum_(i<j) J_ij I_iz I_jz           (weak coupling, for tests)
             and the propagator of a time t is exp(-2 pi i H t): a right-handed rotation about +z, so that the
             transverse magnetisation <I+> of an uncoupled spin turns as exp(+2 pi i nu t).
  rotation   the right-handed rotation by phi about the unit axis n is exp(-i phi n.I). For one proton this is the
             spinor matrix [[a, -conj(b)], [b, conj(a)]] of ``mrgym.design.rotation`` with its Cayley-Klein
             parameters (a, b); for a group of total spin j it is the Wigner matrix of the same rotation
             (``wigner``).
  coherence order of the matrix element |r><s| is p = M_r - M_s (M = total magnetic quantum number). A rotation by
             psi about z multiplies it by exp(-i p psi). The detected operator F+ = sum_i (I_ix + i I_iy) reads the
             elements of order -1 (the I- part of the density matrix: "detection of I-"), and these turn as
             exp(+2 pi i nu t).

Magnetically equivalent groups (composite particles). n equivalent protons couple to a total spin j = n/2, n/2 - 1,
..., and the state space is the direct sum of these, each j occurring
    g(n, j) = C(n, n/2 - j) - C(n, n/2 - j - 1)
times (3 protons: one quartet j = 3/2 and two doublets j = 1/2). All operators that act on the group as a whole
(its chemical shift, its couplings to other sites, an RF pulse) leave j unchanged, and the coupling inside the group
is a constant in every sector, so it does not appear in any signal. A system is therefore simulated as one small
problem per combination of group spins (a ``Sector``), weighted by the product of the multiplicities; ``expand``
gives the same system with every proton as its own site, and rows O7 and M3 of the validation compare the two.

The Hamiltonian commutes with the total Fz. It is diagonalised block by block in M, so every eigenvector has a
definite M and coherence orders stay defined in the eigenbasis, where all propagation is done.
"""
from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np

from ..design.units import GAMMA_HZ_PER_T

MAX_SPINS = 12            # protons in one coupled network after counting equivalent groups
MAX_DIM = 256             # largest sector: 2^8; seven single protons are 128
COUPLINGS = ("strong", "weak")


# ---------------------------------------------------------------------------------------------- description
@dataclasses.dataclass(frozen=True)
class SpinSystem:
    """One coupled network. ``j_hz`` lists (i, j, J in Hz) between sites; ``counts`` the number of magnetically
    equivalent protons of every site; ``groups`` the name under which a site's relaxation times are looked up;
    ``scale`` a weight of the whole network (the amplitude of a peak in a spectral model)."""
    name: str
    shifts_ppm: tuple
    j_hz: tuple = ()
    counts: tuple = None
    labels: tuple = None
    groups: tuple = None
    scale: float = 1.0

    def __post_init__(self):
        shifts = tuple(float(s) for s in self.shifts_ppm)
        n = len(shifts)
        if n == 0:
            raise ValueError("a spin system has at least one site")
        counts = tuple(int(c) for c in (self.counts or (1,) * n))
        labels = tuple(self.labels or (f"{self.name}:{i}" for i in range(n)))
        groups = tuple(self.groups or (self.name,) * n)
        if not (len(counts) == len(labels) == len(groups) == n):
            raise ValueError("shifts_ppm, counts, labels and groups must have one entry per site")
        if min(counts) < 1:
            raise ValueError("counts must be positive")
        if sum(counts) > MAX_SPINS:
            raise ValueError(f"{sum(counts)} protons in one network; the limit is {MAX_SPINS}")
        pairs = []
        for i, j, value in self.j_hz:
            i, j = int(i), int(j)
            if not (0 <= i < n and 0 <= j < n) or i == j:
                raise ValueError(f"coupling ({i}, {j}) does not name two different sites")
            pairs.append((min(i, j), max(i, j), float(value)))
        if len({p[:2] for p in pairs}) != len(pairs):
            raise ValueError("a pair of sites is coupled twice")
        if not all(math.isfinite(x) for x in shifts + tuple(p[2] for p in pairs)) or not math.isfinite(self.scale):
            raise ValueError("shifts, couplings and scale must be finite")
        for name, value in (("shifts_ppm", shifts), ("j_hz", tuple(sorted(pairs))), ("counts", counts), ("labels", labels),
                            ("groups", groups), ("scale", float(self.scale))):
            object.__setattr__(self, name, value)

    @property
    def n_sites(self):
        return len(self.shifts_ppm)

    @property
    def n_spins(self):
        return sum(self.counts)

    def expand(self, intra_j_hz=0.0):
        """The same network with every proton as its own site (brute force). ``intra_j_hz`` is the coupling put
        between the protons of one group: with the full coupling I.I it changes no signal (the weak form Iz Iz is not
        valid inside a group and would split the lines)."""
        first = np.concatenate([[0], np.cumsum(self.counts)])
        shifts, labels, groups, pairs = [], [], [], []
        for i, c in enumerate(self.counts):
            shifts += [self.shifts_ppm[i]] * c
            labels += [f"{self.labels[i]}/{k}" for k in range(c)]
            groups += [self.groups[i]] * c
            if intra_j_hz:
                pairs += [(first[i] + p, first[i] + q, intra_j_hz) for p in range(c) for q in range(p + 1, c)]
        for i, j, value in self.j_hz:
            pairs += [(first[i] + p, first[j] + q, value) for p in range(self.counts[i]) for q in range(self.counts[j])]
        return SpinSystem(self.name, tuple(shifts), tuple(pairs), None, tuple(labels), tuple(groups), self.scale)


def larmor_hz_per_ppm(b0_t):
    """Hz per ppm at the field ``b0_t`` (T): 1e-6 x gamma x B0."""
    return 1e-6 * GAMMA_HZ_PER_T * float(b0_t)


def frequencies_hz(shifts_ppm, b0_t, ref_ppm):
    """Precession frequencies (Hz) relative to the reference at ``ref_ppm``."""
    return (np.asarray(shifts_ppm, float) - float(ref_ppm)) * larmor_hz_per_ppm(b0_t)


# ---------------------------------------------------------------------------------------------- single spins
def spin_matrices(j):
    """(Jx, Jy, Jz, J+, J-) of a spin j (a multiple of 1/2), basis m = j, j - 1, ..., -j."""
    d = int(round(2 * j)) + 1
    if d < 1 or abs(2 * j - (d - 1)) > 1e-12:
        raise ValueError("j must be a non-negative multiple of 1/2")
    m = j - np.arange(d)
    jp = np.zeros((d, d))
    for k in range(1, d):                                       # J+ |m_k> = sqrt(j(j+1) - m_k (m_k + 1)) |m_(k-1)>
        jp[k - 1, k] = math.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    jm = jp.T.copy()
    return 0.5 * (jp + jm), -0.5j * (jp - jm), np.diag(m), jp, jm


def multiplicities(n):
    """[(j, g)] for n equivalent spins 1/2: the total spins j and how often each occurs."""
    out = []
    for k in range(n // 2 + 1):
        g = math.comb(n, k) - (math.comb(n, k - 1) if k else 0)
        out.append((0.5 * n - k, g))
    return out


_JY_CACHE = {}


def wigner(j, a, b):
    """Matrix D^j of the rotation with Cayley-Klein parameters (a, b) on a spin j; ``a``, ``b`` may be arrays, the
    result has their shape + (2j + 1, 2j + 1). For j = 1/2 it is [[a, -conj(b)], [b, conj(a)]].

    The rotation is written with Euler angles, R = Rz(alpha) Ry(beta) Rz(gamma):
        a = exp(-i (alpha + gamma) / 2) cos(beta / 2),     b = exp(i (alpha - gamma) / 2) sin(beta / 2),
    so that D[m', m] = exp(-i alpha m') d(beta)[m', m] exp(-i gamma m) with d(beta) = exp(-i beta Jy), computed
    from the eigenvectors of Jy."""
    a, b = np.broadcast_arrays(np.asarray(a, complex), np.asarray(b, complex))
    d = int(round(2 * j)) + 1
    if d == 1:
        return np.ones(a.shape + (1, 1), complex)
    if d == 2:
        out = np.empty(a.shape + (2, 2), complex)
        out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = a, -np.conj(b), b, np.conj(a)
        return out
    if d not in _JY_CACHE:
        _JY_CACHE[d] = np.linalg.eigh(spin_matrices(j)[1])
    mu, y = _JY_CACHE[d]
    m = j - np.arange(d)
    beta = 2 * np.arctan2(np.abs(b), np.abs(a))
    arg_a, arg_b = np.angle(a), np.angle(b)
    alpha, gamma = arg_b - arg_a, -arg_a - arg_b
    small = (y * np.exp(-1j * beta[..., None, None] * mu)) @ y.conj().T
    return np.exp(-1j * alpha[..., None, None] * m[:, None]) * small * np.exp(-1j * gamma[..., None, None] * m[None, :])


def kron_all(mats):
    """Kronecker product of square matrices with equal leading (ensemble) shapes: site 0 most significant."""
    out = mats[0]
    for m in mats[1:]:
        lead = np.broadcast_shapes(out.shape[:-2], m.shape[:-2])
        p, q = out.shape[-1], m.shape[-1]
        out = (out[..., :, None, :, None] * m[..., None, :, None, :]).reshape(lead + (p * q, p * q))
    return out


# ---------------------------------------------------------------------------------------------- sectors
@dataclasses.dataclass(frozen=True, eq=False)
class Sector:
    """One block of a compiled system: fixed total spin of every site. Everything is in the eigenbasis of H."""
    js: tuple                 # total spin of every site
    weight: int               # how often this block occurs
    energy_hz: np.ndarray     # [d] eigenvalues of H (Hz)
    vec: np.ndarray           # [d, d] real: columns are the eigenvectors in the product basis
    m: np.ndarray             # [d] total magnetic quantum number of every eigenvector
    order: np.ndarray         # [d, d] int: coherence order M_r - M_s
    ip: np.ndarray            # [sites, d, d] raising operator of every site, eigenbasis
    iz_product: np.ndarray    # [sites, d] diagonal of every site's Iz in the product basis
    h_j: np.ndarray           # [d, d] the coupling part of H in the eigenbasis (Hz)
    detect: tuple             # per relaxation group: (r, s, coefficient, frequency in Hz) of every observable line

    @property
    def dim(self):
        return self.energy_hz.size

    def to_eigen(self, op):
        """A product-basis operator (with any leading shape) in the eigenbasis."""
        return self.vec.T @ op @ self.vec

    def to_product(self, op):
        return self.vec @ op @ self.vec.T

    def fz(self):
        return np.diag(self.m).astype(complex)

    def site_operator(self, site, which):
        """Ix, Iy, Iz, I+ or I- of one site in the eigenbasis."""
        p = self.ip[site]
        return {"x": 0.5 * (p + p.T), "y": -0.5j * (p - p.T), "z": 0.5 * (p @ p.T - p.T @ p), "+": p, "-": p.T}[which].astype(complex)

    def rotation(self, a, b):
        """Eigenbasis matrix of the rotation that turns site i by the Cayley-Klein parameters (a[..., i], b[..., i])."""
        a, b = np.asarray(a, complex), np.asarray(b, complex)
        return self.to_eigen(kron_all([wigner(j, a[..., i], b[..., i]) for i, j in enumerate(self.js)]))


@dataclasses.dataclass(frozen=True, eq=False)
class Compiled:
    """A spin system at a field strength, relative to a reference frequency."""
    system: SpinSystem
    b0_t: float
    ref_ppm: float
    coupling: str
    freq_hz: np.ndarray       # [sites]
    groups: tuple             # names of the relaxation groups, in the order of ``Sector.detect``
    sectors: tuple

    @property
    def norm(self):
        """Tr(Iz^2) of one proton in the full space: divide a trace by it to get magnetisation per proton."""
        return 2.0 ** (self.system.n_spins - 2)


def _site_embedding(js):
    dims = [int(round(2 * j)) + 1 for j in js]
    eye = [np.eye(d) for d in dims]
    iz, ip = [], []
    for i, j in enumerate(js):
        _, _, jz, jp, _ = spin_matrices(j)
        iz.append(np.diag(kron_all(eye[:i] + [jz] + eye[i + 1:])))
        ip.append(kron_all(eye[:i] + [jp] + eye[i + 1:]))
    return np.array(iz), np.array(ip)


def hamiltonian(iz, ip, freq_hz, j_hz, coupling="strong"):
    """(H, H_J) in Hz in the product basis from the site operators ``iz`` [sites, d] and ``ip`` [sites, d, d]."""
    if coupling not in COUPLINGS:
        raise ValueError(f"coupling must be one of {COUPLINGS}")
    d = iz.shape[1]
    h_j = np.zeros((d, d))
    for i, j, value in j_hz:
        h_j += value * np.diag(iz[i] * iz[j])
        if coupling == "strong":
            flip = ip[i] @ ip[j].T
            h_j += 0.5 * value * (flip + flip.T)
    return np.diag(np.asarray(freq_hz, float) @ iz) + h_j, h_j


def eigensystem(h, m):
    """Eigenvalues and real eigenvectors of the symmetric ``h``, block by block in the quantum number ``m`` of the
    basis states, so that every eigenvector has a definite m. Returns (energies, vectors, m of the eigenvectors)."""
    key = np.round(2 * np.asarray(m)).astype(int)
    energy, vec, m_out = np.zeros(h.shape[0]), np.zeros(h.shape), np.zeros(h.shape[0])
    at = 0
    for k in sorted(set(key.tolist()), reverse=True):
        idx = np.nonzero(key == k)[0]
        e, v = np.linalg.eigh(h[np.ix_(idx, idx)])
        energy[at:at + idx.size] = e
        vec[idx, at:at + idx.size] = v
        m_out[at:at + idx.size] = 0.5 * k
        at += idx.size
    return energy, vec, m_out


def compile(system, b0_t=3.0, ref_ppm=4.7, coupling="strong", composite=True):
    """The sectors of ``system`` at ``b0_t`` tesla, frequencies relative to ``ref_ppm``.

    ``composite`` False treats every proton as its own site (one sector of dimension 2^n); True uses the total-spin
    decomposition of equivalent groups. Both give the same signals."""
    if not composite and max(system.counts) > 1:
        system = system.expand()
    freq = frequencies_hz(system.shifts_ppm, b0_t, ref_ppm)
    groups = tuple(dict.fromkeys(system.groups))
    sectors = []
    for combo in itertools.product(*[multiplicities(c) for c in system.counts]):
        js = tuple(j for j, _ in combo)
        weight = math.prod(g for _, g in combo)
        iz, ip = _site_embedding(js)
        if iz.shape[1] > MAX_DIM:
            raise ValueError(f"a sector of dimension {iz.shape[1]} is above the limit {MAX_DIM}")
        h, h_j = hamiltonian(iz, ip, freq, system.j_hz, coupling)
        energy, vec, m = eigensystem(h, iz.sum(axis=0))
        order = np.round(m[:, None] - m[None, :]).astype(int)
        ip_eig = np.array([vec.T @ p @ vec for p in ip])
        detect = []
        for name in groups:                                     # Tr(rho F+) = sum_(r, s) rho[r, s] F+[s, r]
            fp = sum(ip_eig[i] for i in range(system.n_sites) if system.groups[i] == name)
            s_idx, r_idx = np.nonzero((np.abs(fp) > 1e-12) & (order == 1))
            detect.append((r_idx, s_idx, fp[s_idx, r_idx], energy[s_idx] - energy[r_idx]))
        sectors.append(Sector(js, weight, energy, vec, m, order, ip_eig, iz, vec.T @ h_j @ vec, tuple(detect)))
    return Compiled(system, float(b0_t), float(ref_ppm), coupling, freq, groups, tuple(sectors))
