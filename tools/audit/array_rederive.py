"""Independent re-derivation of the formulas of ``mrgym.scanner.recon`` and ``mrgym.scanner.array``.

    python tools/audit/array_rederive.py            prints one line per check; exit 1 if one fails

Nothing here calls the function it checks to make its reference. The references are: dense encoding matrices written
out with explicit exponentials (no FFT, no per-pixel unfolding), generalised eigenproblems, numerical quadrature of
the Biot-Savart integral and of the chi density, Monte Carlo, the numbers printed in the papers (Constantinides 1997,
Kellman 2005), and, when the folder exists, the tested helpers of another code base (``--helpers <folder>`` with
``mrfoundry_rawrecon.py``). Not part of the test suite (the last group needs a folder outside the repository).
"""
from __future__ import annotations

import math
import sys

import numpy as np
import scipy.integrate
import scipy.linalg

from mrgym.scanner import array as A
from mrgym.scanner import recon

rng = np.random.default_rng(20261127)
results = []


def check(name, err, tol):
    ok = bool(np.isfinite(err) and err <= tol)
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {err:9.2e} <= {tol:7.1e}  {name}")


def crandn(*shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def random_psi(c):
    q = crandn(c, c)
    return q @ q.conj().T + c * np.eye(c)


# ---------------------------------------------------------------------------------------------- 1. SENSE, dense
def dense_encoding(maps, lines):
    """E[(channel, acquired line, sample), (row, column)] = S_c(y, x) exp(-2 pi i (q (y - Y//2) / Y + p (x - X//2) / X))."""
    c, ny, nx = maps.shape
    q = np.flatnonzero(lines) - ny // 2
    p = np.arange(nx) - nx // 2
    y, x = np.arange(ny) - ny // 2, np.arange(nx) - nx // 2
    fy = np.exp(-2j * np.pi * np.outer(q, y) / ny)                      # [line, row]
    fx = np.exp(-2j * np.pi * np.outer(p, x) / nx)                      # [sample, column]
    e = np.einsum("cyx,qy,px->cqpyx", maps, fy, fx)
    return e.reshape(c * len(q) * nx, ny * nx), (c, len(q), nx)


def sense_dense():
    c, ny, nx = 7, 12, 3
    maps, psi = crandn(c, ny, nx), random_psi(c)
    obj = crandn(ny, nx)
    full, _ = dense_encoding(maps, np.ones(ny, bool))
    wk = lambda shape: np.kron(np.linalg.inv(psi), np.eye(shape[1] * shape[2]))      # noqa: E731  psi^-1 (x) 1 over the samples
    cov_full = np.linalg.inv(full.conj().T @ wk((c, ny, nx)) @ full)
    worst = dict(g=0.0, image=0.0, noise=0.0, full=0.0, reg=0.0, cg=0.0)

    def up(key, *values):                                               # a nan is a failure, not a skipped comparison
        for v in values:
            v = float(v)
            worst[key] = max(worst[key], v if np.isfinite(v) else np.inf)

    for r in (2, 3, 4, 6):
        for off in range(r):
            lines = recon.sampling_mask(ny, r, 0, off)
            e, shape = dense_encoding(maps, lines)
            w = wk(shape)
            cov = np.linalg.inv(e.conj().T @ w @ e)                     # covariance of the best linear unbiased estimate, unit sample noise
            g_dense = np.sqrt((np.diag(cov).real / np.diag(cov_full).real) / r).reshape(ny, nx)
            plan = recon.sense_plan(maps, psi, r, off)
            up("g", np.abs(plan["g"] / g_dense - 1).max())
            d = (e @ obj.ravel()).reshape(shape)
            ks = np.zeros((c, ny, nx), complex)
            ks[:, lines] = d
            up("image", np.abs(recon.sense(ks, plan) - obj).max())
            # absolute noise: samples with covariance 2 sd^2 psi, sd = 1 -> one component of rho has variance cov_pp;
            # recon: R x sd_z x unfold_noise with sd_z = sd sqrt(acquired samples) / (all samples) for ifft2c
            sd_z = math.sqrt(lines.sum() * nx) / (ny * nx)
            up("noise", np.abs(r * sd_z * plan["unfold_noise"] / np.sqrt(np.diag(cov).real).reshape(ny, nx) - 1).max())
            up("full", np.abs(plan["full_noise"] * math.sqrt(1.0 / (ny * nx)) / np.sqrt(np.diag(cov_full).real).reshape(ny, nx) - 1).max())
            # Tikhonov: the same estimator written densely, lam chosen as the plan defines it
            reg = 0.07
            plan_r = recon.sense_plan(maps, psi, r, off, reg=reg)
            m_diag = np.einsum("cyx,cd,dyx->yx", maps.conj(), np.linalg.inv(psi), maps).real
            lam = np.zeros((ny, nx))
            for j in range(ny // r):
                rows = (j + np.arange(r) * (ny // r)) % ny
                lam[rows] = reg * m_diag[rows].mean(0)
            n_s = lines.sum() * nx                                      # E^H W E = n_s x (per-pixel-set matrix)
            u = np.linalg.inv(e.conj().T @ w @ e + n_s * np.diag(lam.ravel())) @ e.conj().T @ w
            up("reg", np.abs(recon.sense(ks, plan_r) - (u @ d.ravel()).reshape(ny, nx)).max(),
                               np.abs(r * sd_z * plan_r["unfold_noise"] / np.sqrt(np.diag(u @ np.kron(psi, np.eye(shape[1] * shape[2])) @ u.conj().T).real).reshape(ny, nx) - 1).max())
        lines = recon.sampling_mask(ny, 3, 4, 1)                        # calibration lines: the conjugate-gradient solution = the dense one
        e, shape = dense_encoding(maps, lines)
        ks = np.zeros((c, ny, nx), complex)
        ks[:, lines] = (e @ obj.ravel()).reshape(shape)
        up("cg", np.abs(recon.sense_cg(ks, lines, maps, psi, n_iter=400, tol=1e-13) - obj).max())
    check("SENSE: g of sense_plan = sqrt(cov_R / cov_full / R) of the dense best linear unbiased estimate (R 2,3,4,6, every offset)", worst["g"], 1e-9)
    check("SENSE: recon.sense returns the object for dense-encoded data, every offset (the phase of the replicas)", worst["image"], 1e-10)
    check("SENSE: absolute noise R sd_z unfold_noise = sqrt(diag) of the dense covariance", worst["noise"], 1e-9)
    check("Roemer: full_noise x coil-image noise = sqrt(diag) of the dense full-sampling covariance", worst["full"], 1e-9)
    check("Tikhonov: image and noise of the regularised plan = the dense regularised estimator", worst["reg"], 1e-9)
    check("CG-SENSE with calibration lines = the object", worst["cg"], 1e-8)


# ---------------------------------------------------------------------------------------------- 2. combination
def combination():
    worst = 0.0
    for _ in range(20):
        c = 6
        s, psi = crandn(c), random_psi(c)
        top = scipy.linalg.eigh(np.outer(s, s.conj()), psi, eigvals_only=True)[-1]      # max_w |w^H s|^2 / w^H psi w
        worst = max(worst, abs(1 / recon.roemer_noise(s.reshape(c, 1, 1), psi)[0, 0] ** 2 / top - 1))
    check("Roemer: SNR^2 = the largest generalised eigenvalue of (s s^H, psi): no linear combination does better", worst, 1e-10)
    c, n = 4, 400000
    s, psi = crandn(c), random_psi(c)
    amp = 600.0
    noise = np.linalg.cholesky(psi) @ crandn(c, n)                      # each component: variance psi_ii
    mag = np.sqrt((np.abs(amp * s[:, None] + noise) ** 2).sum(0))
    check("root sum of squares: Monte Carlo SNR per unit magnetisation against S^H S / sqrt(S^H psi S) (high SNR, 400,000 draws)",
          abs((mag.mean() / mag.std()) / amp / recon.sos_snr(s.reshape(c, 1, 1), psi)[0, 0] - 1), 5e-3)
    w = np.linalg.inv(np.linalg.cholesky(psi))
    comb = (np.conj(w @ s)[:, None] * (w @ (amp * s[:, None] + noise))).sum(0) / np.linalg.norm(w @ s)
    check("Roemer: Monte Carlo SNR (real-part noise of the matched filter of whitened data) against 1 / roemer_noise",
          abs(amp * np.linalg.norm(w @ s) / comb.real.std() / (amp / recon.roemer_noise(s.reshape(c, 1, 1), psi)[0, 0]) - 1), 5e-3)
    est = recon.noise_covariance(noise)
    check("noise_covariance of 400,000 samples = psi (largest element error / sampling error)",
          (np.abs(est - psi) / np.sqrt(np.outer(np.diag(psi).real, np.diag(psi).real) / (2 * n))).max(), 5.0)
    white = recon.prewhiten(noise, psi)
    check("prewhiten: covariance of the whitened samples = 1", np.abs(recon.noise_covariance(white) - np.eye(c)).max(), 5 / math.sqrt(n))


# ---------------------------------------------------------------------------------------------- 3. chi statistics
def chi():
    worst, lit = 0.0, 0.0
    for n in (1, 2, 4, 8, 16, 32):
        def pdf(m, n=n, k=0):                                           # m^k x the density of Constantinides eq. 4, sd = 1
            return m ** k * 2.0 / (2.0 ** n * math.factorial(n - 1)) * m ** (2 * n - 1) * math.exp(-m * m / 2)

        norm, m1, m2 = (scipy.integrate.quad(pdf, 0, 40, args=(n, k))[0] for k in (0, 1, 2))
        mean, sd = recon.chi_moments(n)
        worst = max(worst, abs(norm - 1), abs(mean / m1 - 1), abs(sd / math.sqrt(m2 - m1 * m1) - 1), abs(m2 / (2 * n) - 1))
    for n, v in ((1, 1.25), (2, 1.88), (4, 2.74), (8, 3.94), (16, 5.61), (32, 7.97)):          # Kellman 2005: the bias of the background
        lit = max(lit, abs(recon.chi_moments(n)[0] - v))
    for n, v in ((2, 0.682), (4, 0.695)):                               # Constantinides 1997: its standard deviation
        lit = max(lit, abs(recon.chi_moments(n)[1] - v))
    check("chi_moments: mean, standard deviation and second moment 2 n against quadrature of the density of Constantinides eq. 4 (n = 1 ... 32)", worst, 1e-8)
    check("chi_moments against the numbers printed by Kellman 2005 (1.25 ... 7.97 sd) and Constantinides 1997 (0.682, 0.695 sd)", lit, 0.006)


# ---------------------------------------------------------------------------------------------- 4. the field of a loop
def biot_savart():
    a, n = 50.0, 20000
    t = (np.arange(n) + 0.5) * 2 * np.pi / n
    src = np.stack([a * np.cos(t), a * np.sin(t), np.zeros(n)], 1)      # loop in the plane z = 0, axis z
    dl = np.stack([-a * np.sin(t), a * np.cos(t), np.zeros(n)], 1) * (2 * np.pi / n)
    worst = 0.0
    for rho, zeta in ((0.0, 30.0), (20.0, 10.0), (35.0, -25.0), (80.0, 40.0), (120.0, 5.0), (10.0, 150.0), (49.0, 30.0)):
        p = np.array([rho, 0.0, zeta])
        d = p - src
        b = (np.cross(dl, d) / np.linalg.norm(d, axis=1)[:, None] ** 3).sum(0) / 2      # in units of mu0 I / (2 pi): mu0 I / (4 pi) x integral
        b_ax, b_rad = A.loop_field(a, np.array(rho), np.array(zeta))
        worst = max(worst, abs(b_ax - b[2]) / np.linalg.norm(b), abs(b_rad - b[0]) / np.linalg.norm(b))
    check("loop_field (elliptic integrals) against the Biot-Savart line integral over 20,000 segments, 7 points on and off the axis", worst, 1e-7)
    el = dict(y_mm=100.0, x_mm=0.0, z_mm=0.0, normal_yx=(-1.0, 0.0), radius_mm=a)
    s = A.LoopMaps([el])(np.array([40.0, 100.0 - 30.0]), np.array([0.0, 0.0]))[0]
    on_axis = np.pi * a * a / (a * a + np.array([60.0, 30.0]) ** 2) ** 1.5
    # on the axis the field is along the normal (-y): S = (Bx - i By) / 2 = +i B / 2
    check("LoopMaps on the axis of a loop: (Bx - i By) / 2 with the field along the loop's normal", np.abs(s - 0.5j * on_axis).max() / on_axis.max(), 1e-12)


# ---------------------------------------------------------------------------------------------- 5. other helpers
def helpers(folder):
    sys.path.insert(0, folder)
    import mrfoundry_rawrecon as H
    c, ny, nx = 6, 24, 16
    maps, psi, obj = crandn(c, ny, nx), random_psi(c), crandn(ny, nx)
    worst = 0.0
    for r, acs, off in ((2, 0, 0), (3, 6, 1), (4, 8, 2), (3, 0, 2)):
        lines = recon.sampling_mask(ny, r, acs, off)
        pat = H.sampling_pattern(lines)
        got = recon.acceleration_of(lines)
        worst = max(worst, abs(pat["R"] - got[0]), abs((pat["offset"] - ny // 2) % r - got[1]))
    check("acceleration_of against the other code base's sampling_pattern (R and lattice offset, with and without calibration lines)", worst, 0)
    imgs = maps * obj
    check("roemer against the other code base's sense_combine (with psi)", np.abs(recon.roemer(imgs, maps, psi) - H.sense_combine(imgs, maps, psi)).max(), 1e-10)
    lines = recon.sampling_mask(ny, 3, 0, 0)
    w = recon.whitener(psi)
    ks = H.fft2c(np.tensordot(w, imgs, 1)) * lines[:, None]             # orthonormal transforms there; whitened
    cg = H.sense_cg(ks, lines, np.tensordot(w, maps, 1), n_iter=400)
    mine = recon.sense(recon.fft2c(imgs) * lines[:, None], recon.sense_plan(maps, psi, 3))
    check("recon.sense against the other code base's conjugate-gradient SENSE of the same (whitened) data", np.abs(cg - mine).max() / np.abs(obj).max(), 1e-5)
    plan = recon.sense_plan(maps, psi, 3)
    sd = H.pseudo_replica(lambda k: recon.sense(k, plan).real, recon.fft2c(imgs) * lines[:, None], psi, n_rep=4000, seed=1, mask=lines[None, :, None])
    # their noise: E|n|^2 = psi_ii per sample, i.e. one component has sd^2 = psi_ii / 2
    sd_z = math.sqrt(0.5) * math.sqrt(lines.sum() * nx) / (ny * nx)
    check("noise of recon.sense by the other code base's pseudo_replica (4000) against R sd_z unfold_noise, mean ratio", abs((sd / (3 * sd_z * plan["unfold_noise"])).mean() - 1), 5e-3)
    su = H.snr_units_image(H.ifft2c(H.fft2c(imgs)), psi, sens=maps)
    mine = recon.roemer(imgs, maps, psi, snr_units=True) * math.sqrt(2)  # their psi = E[n n^H]: one component has variance psi_ii / 2
    check("roemer(snr_units) against the other code base's snr_units_image (its psi is E[n n^H], twice the variance of a component)", np.abs(su - mine).max() / np.abs(mine).max(), 1e-10)


if __name__ == "__main__":
    sense_dense()
    combination()
    chi()
    biot_savart()
    if "--helpers" in sys.argv:
        helpers(sys.argv[sys.argv.index("--helpers") + 1])
    print(f"{sum(results)} of {len(results)} checks pass")
    sys.exit(0 if all(results) else 1)
