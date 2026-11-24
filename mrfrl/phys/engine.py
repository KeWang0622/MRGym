"""World x realised protocol -> images. The scanner serves and the verifier replays exactly this.

Factorisation (memo 13): the object is a sum of tissue classes with soft partial-volume weights w_c(r); the signal of
class c depends on position only through the smooth fields. So

    image(r) = rx(r) * exp(2 pi i dB0(r) TE) * sum_c  w_c(r) * PD_c * exp(-TE / T2*_c) * s_c(B1(r))

and s_c is computed per class on a handful of B1 nodes (never per voxel) and interpolated:
    spgr, afi   tier 2: EPG steady state with RF spoiling, finite T2 and diffusion (``epg``)   [``tier=1``: closed forms]
    ir          tier 1 closed form (complete spoiling between repetitions is assumed; TR >= 1 s)
The sum is evaluated on a grid ``sim.FINE`` times finer than the requested matrix and truncated in k-space, so partial
volume and Gibbs ringing are in the data. Receiver noise is white complex Gaussian in the image, with the SD of
``cost.noise_factor``. Not modelled: slice profile, the approach to steady state (dummy time is charged, data are
served at the periodic state), T2* blurring and off-resonance displacement along the readout, flow, motion.
"""
import numpy as np

from .. import sim
from . import closed, cost, epg, world as W

GAMMA_RAD = 2 * np.pi * cost.GAMMA_HZ_PER_T


def cheb_nodes(lo, hi, m):
    """Chebyshev points of the first kind on [lo, hi] and their barycentric weights."""
    t = np.pi * (np.arange(m) + 0.5) / m
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(t), (-1.0) ** np.arange(m) * np.sin(t)


def bary(nodes, weights, values, x):
    """Barycentric polynomial interpolation of ``values`` (given on ``nodes``) at ``x``. Deterministic (scipy's
    BarycentricInterpolator permutes its nodes at random, which changes the last bit between runs and breaks replay)."""
    d = x[:, None] - nodes[None, :]
    hit = d == 0
    d[hit] = 1.0
    q = weights / d
    out = (q @ values) / q.sum(axis=1)
    rows, cols = np.nonzero(hit)
    out[rows] = values[cols]
    return out


def diffusion_bd(adc_um2_ms, spoiler_mt_ms_per_m, tr_ms):
    """Dimensionless D * dk^2 * TR for the EPG: D in um^2/ms (= 1e-9 m^2/s), spoiler area in mT*ms/m, TR in ms."""
    return adc_um2_ms * 1e-9 * (GAMMA_RAD * spoiler_mt_ms_per_m * 1e-6) ** 2 * tr_ms * 1e-3


def class_signal(cls, rp, a, tier=2, inv_eff=1.0):
    """Signal(s) of one tissue class at actual flip angles ``a`` (rad, array): array [n_contrast, len(a)], complex."""
    a = np.asarray(a, float)
    if rp["sequence"] == "ir":
        return np.asarray(closed.ir(a, rp["ti_ms"], rp["tr_ms"], cls["t1_ms"], inv_eff), complex)[None]
    if tier == 1:
        if rp["sequence"] == "spgr":
            return np.asarray(closed.spgr(a, rp["tr_ms"], cls["t1_ms"]), complex)[None]
        return np.asarray(closed.afi(a, rp["tr_ms"], rp["tr2_ms"], cls["t1_ms"]), complex)
    bd = diffusion_bd(cls["adc_um2_ms"], rp["spoiler_mt_ms_per_m"], rp["tr_ms"])
    if rp["sequence"] == "spgr":
        return epg.spgr(a, rp["tr_ms"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"], bd)[None]
    return np.array(epg.afi(a, rp["tr_ms"], rp["tr_ratio"], cls["t1_ms"], cls["t2_ms"], rp["rf_phase_increment_deg"], bd))


def physics_key(rp):
    """The part of a realised protocol that changes the noise-free image (not averages, not bandwidth)."""
    keys = ("sequence", "matrix", "te_ms", "tr_ms", "flip_deg", "spoiler_mt_ms_per_m", "rf_phase_increment_deg",
            "tr_ratio", "ti_ms", "n_echoes", "echo_spacing_ms")
    return tuple((k, rp[k]) for k in keys if rp.get(k) is not None)


class Engine:
    """Holds one world and caches what does not depend on the noise draw."""

    def __init__(self, world, tier=2, nodes=None):
        s = world["system"]
        if s.get("grad_delay_us") or s.get("rf_leak") or s.get("coil_faults"):
            raise NotImplementedError("gradient delay, RF leak and coil faults are hooks only")
        self.world, self.tier, self.nodes = world, tier, nodes
        self._grid, self._clean = {}, {}
        y, x = np.meshgrid(sim.grid(384), sim.grid(384), indexing="ij")
        w, b = W.class_weights(world, y, x), W.b1(world, y, x)
        self.b1_range = [(float(b[wc > 1e-4].min()), float(b[wc > 1e-4].max())) for wc in w]

    def _maps(self, n):
        if n not in self._grid:
            f = sim.FINE * n
            y, x = np.meshgrid(sim.grid(f), sim.grid(f), indexing="ij")
            wts = W.class_weights(self.world, y, x)
            pd = np.array([c["pd"] for c in self.world["classes"]])
            self._grid[n] = dict(w=wts * pd[:, None, None], sup=[wc > 1e-4 for wc in wts], b1=W.b1(self.world, y, x),
                                 rx=W.rx(self.world, y, x), b0=W.b0_hz(self.world, y, x))
        return self._grid[n]

    def _nodes(self, c):
        lo, hi = self.b1_range[c]
        pad = 0.01 * (hi - lo) + 1e-4
        lo, hi = lo - pad, hi + pad
        m = self.nodes or int(np.clip(np.ceil(30 * (hi - lo) / (hi + lo)) + 4, 5, 10))
        return cheb_nodes(lo, hi, m)

    def clean(self, rp):
        """Noise-free complex images [n_images, matrix, matrix] (band-limited), cached per physics key."""
        key = physics_key(rp)
        if key not in self._clean:
            n, m = rp["matrix"], self._maps(rp["matrix"])
            tes = rp.get("echo_times_ms") or [rp["te_ms"]]
            n_con = 2 if rp["sequence"] == "afi" else 1
            flip = np.deg2rad(rp.get("flip_deg", rp.get("readout_flip_deg")))
            fine = np.zeros((n_con, len(tes)) + m["b1"].shape, complex)
            for c, cls in enumerate(self.world["classes"]):
                sup = m["sup"][c]
                nodes, wts = self._nodes(c)
                sig = class_signal(cls, rp, flip * nodes, self.tier, self.world["system"]["inv_eff"])
                t2s = closed.t2star_ms(cls["t2_ms"], cls["t2p_ms"])
                for k in range(n_con):
                    val = bary(nodes, wts, sig[k], m["b1"][sup]) * m["w"][c][sup]
                    for e, te in enumerate(tes):
                        fine[k, e][sup] += val * np.exp(-te / t2s)
            for e, te in enumerate(tes):
                fine[:, e] *= m["rx"] * np.exp(2j * np.pi * m["b0"] * te * 1e-3)
            fine = fine.reshape((-1,) + m["b1"].shape)                     # afi: (S1, S2); multi-echo: echoes
            k = sim.crop_centre(sim.crop_centre(sim.cfft(fine, (1, 2)), n, 1), n, 2)
            if len(self._clean) >= 256:
                self._clean.pop(next(iter(self._clean)))
            self._clean[key] = sim.cifft(k, (1, 2))
        return self._clean[key]

    def acquire(self, rp, rng):
        """What one scan returns: complex images with receiver noise."""
        img = self.clean(rp)
        sd = self.world["system"]["noise_sd"] * cost.noise_factor(rp)
        return img + sd * (rng.standard_normal(img.shape) + 1j * rng.standard_normal(img.shape))
