"""Procedural phantoms of the hidden world: a brain-like slice, an abdomen-like slice and a quantitative QA phantom.

Nothing is loaded from disk and nothing is licensed: geometry is a set of implicit functions (layers of a warped
egg-shaped head, sulci as slits in polar coordinates, ellipses for ventricles / nuclei / organs) evaluated through a
smooth random coordinate warp, so every seed gives another subject. The QA phantom is a disc of gel with vials of
drawn T1 / T2 (and, on request, vials of oil), the layout of a relaxometry test object. Output is a stack of soft
partial-volume weights on a master grid much finer than any acquired pixel.

Conventions: positions in mm, arrays [y, x], index i at (i - n // 2) * extent / n; +y is posterior (anterior at the top
of a displayed image), +x is the patient's left (radiological display).

Turned grids. Every shape and every field is a function of the patient coordinates (y, x), and no random draw depends
on the grid. ``axes(n, extent, rot_deg)`` gives the patient coordinates of a grid turned by an angle about the
isocentre (rows from +y towards +x), and each phantom evaluated on it is the same object, sampled elsewhere: this
is how the scanner turns its field of view without interpolating anything. Two places needed care: the depth below
a surface is an implicit function divided by its gradient, taken on the grid for the upright grid and with the same
central differences along the patient axes otherwise; and the seed points of brain lesions are grid points of the
upright grid, handed to a turned build as coordinates (``picks``).
"""
import numpy as np

from . import tissue

EDGE_MM = 0.5            # half-width of every soft boundary


def sig(d, e=None):
    """Soft indicator of d > 0 (half-width ``e``, by default the module's ``EDGE_MM``)."""
    return 0.5 * (1.0 + np.tanh(d / (EDGE_MM if e is None else e)))


def soft(edge_mm=None):
    """``sig`` with the default half-width ``edge_mm`` (None: the module's ``EDGE_MM``): the soft indicator of ONE
    phantom build. Every builder below takes ``edge_mm`` and draws all of its boundaries with this local function.
    The half-width of a build is an argument, never a module attribute that is changed for the duration of a call:
    two threads that build worlds with different half-widths would read each other's value."""
    edge = float(EDGE_MM if edge_mm is None else edge_mm)
    return lambda d, e=None: 0.5 * (1.0 + np.tanh(d / (edge if e is None else e)))


def axes(n, extent_mm, rot_deg=0.0, warp=None):
    """Patient coordinates (y, x) in mm of the points of an n x n grid of the given extent, turned by ``rot_deg``
    about the isocentre: grid rows run from +y towards +x. ``warp(y, x) -> (y, x)`` then maps them once more (the
    hook of a gradient non-linearity). With the defaults: the upright grid, index i at (i - n // 2) * extent / n."""
    c = (np.arange(n) - n // 2) * (extent_mm / n)
    v, u = np.meshgrid(c, c, indexing="ij")
    if rot_deg:
        co, si = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
        v, u = v * co - u * si, v * si + u * co
    return warp(v, u) if warp is not None else [v, u]


def smooth_field(rng, n_modes, wl_lo, wl_hi, amp):
    """Random smooth function of position: a sum of cosines, evaluable anywhere. RMS amplitude ``amp``."""
    wl = np.exp(rng.uniform(np.log(wl_lo), np.log(wl_hi), n_modes))
    ang, ph = rng.uniform(0, 2 * np.pi, n_modes), rng.uniform(0, 2 * np.pi, n_modes)
    a = rng.normal(0, 1, n_modes)
    a *= amp * np.sqrt(2.0) / np.sqrt(np.sum(a * a))
    ky, kx = 2 * np.pi * np.sin(ang) / wl, 2 * np.pi * np.cos(ang) / wl

    def f(y, x):
        out = np.zeros(np.broadcast(y, x).shape)
        for i in range(n_modes):
            out += a[i] * np.cos(ky[i] * y + kx[i] * x + ph[i])
        return out
    return f


def periodic_field(rng, n_harm, amp):
    """Random smooth 2 pi-periodic function of an angle."""
    c, ph = rng.normal(0, 1, n_harm) / np.arange(1, n_harm + 1) ** 0.5, rng.uniform(0, 2 * np.pi, n_harm)
    c *= amp * np.sqrt(2.0) / np.sqrt(np.sum(c * c))
    return lambda th: sum(c[m] * np.cos((m + 1) * th + ph[m]) for m in range(n_harm))


def ell(y, x, cy, cx, ay, ax, rot_deg=0.0, p=2.0):
    """Approximate signed distance (mm, negative inside) to a rotated (super)ellipse with semi-axes (ay, ax)."""
    c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
    q = (np.abs(u / ax) ** p + np.abs(v / ay) ** p) ** (1.0 / p)
    return (q - 1.0) * min(ax, ay)


def sdf(q, d_mm):
    """First-order signed distance from an implicit function q (level set q = 0): q / |grad q|."""
    gy, gx = np.gradient(q, d_mm)
    return q / np.maximum(np.hypot(gy, gx), 1e-6)


def sdf_at(q, q_of, y, x, d_mm):
    """The same on points that are not an upright grid: central differences of ``q_of`` over ``d_mm`` along the
    patient axes (what ``np.gradient`` computes on the upright grid of spacing ``d_mm``)."""
    gy = (q_of(y + d_mm, x) - q_of(y - d_mm, x)) / (2 * d_mm)
    gx = (q_of(y, x + d_mm) - q_of(y, x - d_mm)) / (2 * d_mm)
    return q / np.maximum(np.hypot(gy, gx), 1e-6)


def _jitter(tis, names, rng, rel=0.05):
    out = {}
    for k in names:
        t = dict(tis[k])
        for key in ("t1_ms", "t2_ms", "t2s_ms"):
            t[key] *= float(np.exp(rng.normal(0, rel)))
        t["t2s_ms"] = min(t["t2s_ms"], 0.98 * t["t2_ms"])
        t["t2p_ms"] = 1.0 / (1.0 / t["t2s_ms"] - 1.0 / t["t2_ms"])
        t["pd"] = float(np.clip(t["pd"] * np.exp(rng.normal(0, 0.02)), 0.02, 1.0))
        out[k] = t
    return out


def _layers(t, bounds, sig=sig):
    """Weights of consecutive layers of depth ``t`` between boundaries bounds[i] .. bounds[i+1] (arrays or scalars)."""
    s = [sig(t - b) for b in bounds]
    return [s[i] - s[i + 1] for i in range(len(bounds) - 1)], s[-1]


# ------------------------------------------------------------------------------------------------ brain
def brain(rng, b0_t=3.0, n=640, extent_mm=256.0, lesion=None, atrophy=None, tissue_rng=None, rot_deg=0.0, warp=None, picks=None, edge_mm=None):
    """A brain-like axial slice at the level of the basal ganglia. lesion: None, 'ms' or 'tumour'.

    Returns dict(kind, n, extent_mm, names, weights [class, n, n] float32, tissues {name: params}, air [(y, x, z, r)],
    masks {'brain', 'lesion'}, picks). ``rot_deg``, ``warp``: the grid (``axes``); ``picks``: the lesion seed points
    of the upright build, for a turned one.
    """
    sig = soft(edge_mm)                                                  # every boundary of this build
    d = extent_mm / n
    upright = not rot_deg and warp is None
    y0, x0 = axes(n, extent_mm, rot_deg, warp)
    picked = []

    def pick(candidates):
        """Seed point of a lesion: a random grid point of the upright grid among ``candidates()``; its (warped)
        coordinates and the state of the generator after the draw are kept, so a turned build can take both over."""
        if picks is not None:
            py, px, state = picks[len(picked)]
            rng.bit_generator.state = state
        else:
            cand = np.argwhere(candidates())
            cy, cx = cand[rng.integers(len(cand))]
            py, px, state = float(y[cy, cx]), float(x[cy, cx]), rng.bit_generator.state
        picked.append((py, px, state))
        return py, px

    wy, wx = smooth_field(rng, 8, 80, 260, 1.1), smooth_field(rng, 8, 80, 260, 1.1)
    y, x = y0 + wy(y0, x0), x0 + wx(y0, x0)

    sc = rng.uniform(0.95, 1.05)
    ax_, ay_ = sc * rng.uniform(72, 78), sc * rng.uniform(90, 97)
    egg = rng.uniform(0.07, 0.13)                                     # wider posteriorly
    def q_of(ya, xa):
        yw, xw = ya + wy(ya, xa), xa + wx(ya, xa)
        return np.sqrt((xw / (ax_ * (1 + egg * yw / ay_))) ** 2 + (yw / ay_) ** 2) - 1.0

    q = np.sqrt((x / (ax_ * (1 + egg * y / ay_))) ** 2 + (y / ay_) ** 2) - 1.0
    t = -(sdf(q, d) if upright else sdf_at(q, q_of, y0, x0, d))       # depth below the skin surface, mm
    atrophy = rng.uniform(0.0, 1.0) if atrophy is None else float(atrophy)

    thick = smooth_field(rng, 6, 60, 250, 1.0)(y0, x0)
    b_skin = 1.6
    b_fat = b_skin + rng.uniform(3.0, 6.0) * (1 + 0.25 * thick)
    b_out = b_fat + 1.7
    b_dip = b_out + rng.uniform(1.8, 3.2) * (1 + 0.3 * smooth_field(rng, 6, 40, 200, 1.0)(y0, x0))
    b_in = b_dip + 1.5
    b_csf = b_in + rng.uniform(1.6, 2.6) + 1.2 * atrophy
    (skin, fat, outer, diploe, inner, sub), brain_w = _layers(t, [0.0, b_skin, b_fat, b_out, b_dip, b_in, b_csf], sig)
    s = t - b_csf                                                     # depth below the brain surface

    # ---- sulci: slits in (arc length, depth) coordinates; a finer warp, a periodic phase warp, tilt and curvature
    # make them irregular; a second, shallower family adds branching
    fy, fx = smooth_field(rng, 14, 9, 40, 1.7), smooth_field(rng, 14, 9, 40, 1.7)
    yb, xb = y + fy(y0, x0), x + fx(y0, x0)
    th = np.arctan2(yb, xb)
    r_loc = np.hypot(xb, yb)
    lateral = (np.exp(-(np.angle(np.exp(1j * (th + 0.12))) / 0.16) ** 2)
               + np.exp(-(np.angle(np.exp(1j * (th - np.pi - 0.12))) / 0.16) ** 2))

    def sulci(n_s, d_lo, d_hi, extra, width):
        phase = n_s * th + periodic_field(rng, 8, 1.6)(th)
        idx = np.floor((phase + np.pi) / (2 * np.pi)).astype(int) % n_s
        depth = (rng.uniform(d_lo, d_hi, n_s) * rng.choice([1.0, 1.0, 0.6], n_s))[idx] + extra
        tilt, curv = rng.uniform(-0.6, 0.6, n_s)[idx], rng.uniform(-0.5, 0.5, n_s)[idx]
        u = np.angle(np.exp(1j * (phase + tilt * s / 6.0 + curv * np.sin(s / 3.5))))
        arc = np.abs(u) / n_s * r_loc
        half = width * (0.55 + 0.9 * atrophy) * (1 + 0.8 * np.exp(-np.clip(s, 0, None) / 2.5)) + 0.9 * lateral * (extra > 0)
        da, ds = arc - half, s - depth
        return np.hypot(np.clip(da, 0, None), np.clip(ds, 0, None)) + np.minimum(np.maximum(da, ds), 0)

    f_sulc = np.minimum(sulci(int(rng.integers(15, 21)), 8.0, 17.0, rng.uniform(9, 13) * lateral, 1.0),
                        sulci(int(rng.integers(22, 30)), 3.0, 8.0, 0.0, 0.8))
    # ---- interhemispheric fissure, interrupted by the callosal region
    genu, splen = -rng.uniform(0.36, 0.44) * ay_, rng.uniform(0.38, 0.46) * ay_
    mid = 1.0 * np.sin(y / 23.0)
    da = np.abs(x - mid) - (0.7 + 0.6 * atrophy)
    ds = np.minimum(y - genu, splen - y)                              # > 0 between genu and splenium: no fissure
    f_fiss = np.hypot(np.clip(da, 0, None), np.clip(ds, 0, None)) + np.minimum(np.maximum(da, ds), 0)
    dn = np.minimum(np.minimum(s, f_sulc), f_fiss)                    # distance into tissue from the pial surface
    t_gm = rng.uniform(2.6, 3.4) * (1 - 0.2 * atrophy)
    pial_csf, gm, wm = 1 - sig(dn), sig(dn) - sig(dn - t_gm), sig(dn - t_gm)

    # ---- ventricles and deep grey matter (mirrored pairs with a small asymmetry)
    v = rng.uniform(0.8, 1.15) * (1 + 0.6 * atrophy)
    d_vent, d_deep = np.full(x.shape, 1e3), np.full(x.shape, 1e3)
    for sgn in (-1.0, 1.0):
        j = rng.uniform(0.92, 1.08, 6)
        d_vent = np.minimum(d_vent, ell(y, x, -26 * sc, sgn * (4.0 * v + 1.3) * j[0], 12.5 * j[1] * sc, 4.0 * v, sgn * 18))
        d_vent = np.minimum(d_vent, ell(y, x, 30 * sc, sgn * 22 * sc * j[2], 13 * j[3] * sc, 4.6 * v, -sgn * 28))
        d_deep = np.minimum(d_deep, ell(y, x, -24 * sc, sgn * (8.0 * v + 7.5), 9.5 * j[4], 4.8, sgn * 14))      # caudate head
        d_deep = np.minimum(d_deep, ell(y, x, -6 * sc, sgn * 27 * sc, 16.5 * j[5], 7.0, -sgn * 14, p=2.6))      # lentiform
        d_deep = np.minimum(d_deep, ell(y, x, 9 * sc, sgn * 11.5 * sc, 13.0, 8.5, -sgn * 12))                   # thalamus
    d_vent = np.minimum(d_vent, ell(y, x, 7 * sc, 0.0, 11.0, 1.0 + 0.9 * v))                                    # third ventricle
    deep = sig(-d_deep) * sig(dn - t_gm)
    vent = sig(-d_vent) * sig(dn - 1.0)

    comp = dict(wm=wm, gm=gm, csf=pial_csf)
    for k in comp:
        comp[k] = comp[k] * (1 - deep)
    comp["deep_gm"] = deep
    for k in comp:
        comp[k] = comp[k] * (1 - vent)
    comp["csf"] = comp["csf"] + vent

    # ---- lesions
    lesion_mask = np.zeros(x.shape)
    if lesion == "ms":
        les = np.full(x.shape, 1e3)
        for _ in range(int(rng.integers(4, 9))):
            py, px = pick(lambda: (comp["wm"] > 0.95) & (np.abs(d_vent) < 14) | (comp["wm"] > 0.98))
            les = np.minimum(les, ell(y, x, py, px, rng.uniform(1.8, 5.0), rng.uniform(1.5, 3.5), rng.uniform(0, 180)))
        lesion_mask = sig(-les) * (comp["wm"] + comp["gm"] + comp["deep_gm"])
        for k in comp:
            comp[k] = comp[k] * (1 - sig(-les)) if k != "csf" else comp[k]
        comp["lesion_ms"] = lesion_mask
    elif lesion == "tumour":
        py, px = pick(lambda: (comp["wm"] > 0.98) & (np.abs(x) > 18))
        core = ell(y, x, py, px, rng.uniform(7, 12), rng.uniform(7, 12), rng.uniform(0, 180))
        core = core + smooth_field(rng, 8, 8, 30, 1.5)(y0, x0)
        tissue_w = comp["wm"] + comp["gm"] + comp["deep_gm"]
        tum = sig(-core) * tissue_w
        oed = (sig(-(core - rng.uniform(7, 13))) - sig(-core)) * comp["wm"]                 # oedema follows white matter
        for k in ("wm", "gm", "deep_gm"):
            comp[k] = comp[k] * (1 - sig(-core))
        comp["wm"] = comp["wm"] - oed
        comp["tumour"], comp["oedema"] = tum, oed
        lesion_mask = tum + oed

    names = ["skin", "scalp_fat", "skull", "diploe", "csf"] + [k for k in comp if k != "csf"]
    w = dict(skin=skin, scalp_fat=fat, skull=outer + inner, diploe=diploe, csf=sub + brain_w * comp["csf"])
    for k in comp:
        if k != "csf":
            w[k] = brain_w * comp[k]
    weights = np.stack([w[k] for k in names]).astype(np.float32)
    # air cavities below the slice (frontal sinus, two mastoid / ear regions): (y, x, z, radius) in mm
    air = [(-(ay_ - rng.uniform(14, 20)), rng.uniform(-6, 6), -rng.uniform(20, 28), rng.uniform(11, 15)),
           (rng.uniform(8, 20), -(ax_ - 12), -rng.uniform(24, 32), rng.uniform(10, 14)),
           (rng.uniform(8, 20), (ax_ - 12), -rng.uniform(24, 32), rng.uniform(10, 14))]
    tis = _jitter(tissue.table(b0_t), names, tissue_rng or rng)
    return dict(kind="brain", b0_t=b0_t, n=n, extent_mm=extent_mm, names=names, weights=weights, tissues=tis, air=air,
                masks=dict(body=(sig(t) > 0.5), brain=(brain_w > 0.5), lesion=lesion_mask.astype(np.float32)),
                size_mm=(2 * ay_, 2 * ax_), picks=picked)


# ------------------------------------------------------------------------------------------------ abdomen
def abdomen(rng, b0_t=3.0, n=640, extent_mm=420.0, lesion=None, tissue_rng=None, rot_deg=0.0, warp=None, edge_mm=None):
    """An abdomen-like axial slice through liver, spleen and kidneys. lesion: None or 'liver'."""
    sig = soft(edge_mm)                                                  # every boundary of this build
    d = extent_mm / n
    upright = not rot_deg and warp is None
    y0, x0 = axes(n, extent_mm, rot_deg, warp)
    wy, wx = smooth_field(rng, 6, 160, 500, 3.0), smooth_field(rng, 6, 160, 500, 3.0)
    y, x = y0 + wy(y0, x0), x0 + wx(y0, x0)
    ax_, ay_ = rng.uniform(150, 172), rng.uniform(100, 118)
    def q_of(ya, xa):
        yw, xw = ya + wy(ya, xa), xa + wx(ya, xa)
        return (np.abs(xw / ax_) ** 2.6 + np.abs(yw / ay_) ** 2.6) ** (1 / 2.6) - 1.0

    q = (np.abs(x / ax_) ** 2.6 + np.abs(y / ay_) ** 2.6) ** (1 / 2.6) - 1.0
    t = -(sdf(q, d) if upright else sdf_at(q, q_of, y0, x0, d))
    oy, ox = smooth_field(rng, 12, 30, 140, 3.0), smooth_field(rng, 12, 30, 140, 3.0)
    y, x = y + oy(y0, x0), x + ox(y0, x0)                            # organs: stronger, finer warp than the outline
    b_skin = 2.0
    b_fat = b_skin + rng.uniform(9, 24) * (1 + 0.3 * smooth_field(rng, 5, 150, 500, 1.0)(y0, x0))
    b_mus = b_fat + rng.uniform(8, 13)
    (skin, fat, wall), cav = _layers(t, [0.0, b_skin, b_fat, b_mus], sig)
    cx_, cy_ = ax_ - 30.0, ay_ - 30.0                                 # nominal cavity half-axes

    def E(fy, fx, sy, sx, rot=0.0, p=2.0):
        return ell(y, x, fy * cy_, fx * cx_, sy, sx, rot, p)

    j = rng.uniform(0.92, 1.08, 12)
    parts = []                                                        # (class, soft mask), later entries win
    liver = np.minimum(E(-0.15, -0.48, 62 * j[0], 68 * j[1], 20, 2.4), E(-0.55, 0.02, 26, 52 * j[2], -10))
    parts.append(("liver", sig(-liver)))
    parts.append(("spleen", sig(-E(0.38, 0.72, 44 * j[3], 21 * j[4], -28))))
    parts.append(("bowel_fluid", sig(-E(-0.42, 0.52, 27 * j[5], 36 * j[6], 15))))                  # stomach
    gas = E(-0.62, 0.50, 9 * j[7], 20 * j[7], 10)
    parts.append(("air", sig(-gas)))
    air = [(-0.62 * cy_, 0.50 * cx_, 0.0, 12.0 * j[7])]
    for _ in range(int(rng.integers(3, 6))):                           # bowel loops, some with gas
        fy, fx, r = rng.uniform(-0.65, -0.1), rng.uniform(-0.15, 0.35), rng.uniform(9, 15)
        gasy = rng.uniform() < 0.5
        parts.append(("air" if gasy else "bowel_fluid", sig(-E(fy, fx, r, r * rng.uniform(1.0, 1.5), rng.uniform(0, 180)))))
        if gasy:
            air.append((fy * cy_, fx * cx_, 0.0, r))
    for sgn in (-1.0, 1.0):
        ky, kx = 0.50 + rng.uniform(-0.04, 0.04), sgn * 0.50
        kd = E(ky, kx, 30 * j[8], 20 * j[9], sgn * 20)
        parts.append(("kidney_cortex", sig(-kd)))
        parts.append(("kidney_medulla", sig(-(kd + 7.0))))
        parts.append(("visceral_fat", sig(-E(ky - 0.02, kx - sgn * 0.05, 9, 6, sgn * 20))))       # renal sinus
        parts.append(("abd_muscle", sig(-E(0.60, sgn * 0.24, 16, 13))))                           # psoas
        parts.append(("abd_muscle", sig(-E(1.02, sgn * 0.27, 20, 30, -sgn * 10))))                # paraspinal
    parts.append(("marrow", sig(-E(0.62, 0.0, 17, 22, 0, 2.6))))                                  # vertebral body
    parts.append(("csf", sig(-E(0.93, 0.0, 8.5, 9.5))))
    parts.append(("cord", sig(-E(0.93, 0.0, 4.5, 5.5))))
    parts.append(("blood", sig(-E(0.33, 0.07, 10.5 * j[10], 10.5 * j[10]))))                      # aorta
    parts.append(("blood", sig(-E(0.30, -0.14, 8, 12 * j[11], 15))))                              # vena cava
    for _ in range(5):                                                                            # hepatic vessels
        parts.append(("blood", sig(-E(rng.uniform(-0.4, 0.1), rng.uniform(-0.75, -0.25), rng.uniform(2, 4), rng.uniform(3, 9), rng.uniform(0, 180))) * sig(-liver - 4)))
    lesion_mask = np.zeros(x.shape)
    if lesion == "liver":
        lesion_mask = sig(-E(rng.uniform(-0.3, 0.05), rng.uniform(-0.7, -0.3), *([rng.uniform(7, 14)] * 2))) * sig(-liver - 3)
        parts.append(("liver_lesion", lesion_mask))

    w = {"visceral_fat": np.ones(x.shape)}
    for cls, m in parts:
        for k in w:
            w[k] = w[k] * (1 - m)
        w[cls] = w.get(cls, 0) + m
    # the posterior muscles and the spine reach into the wall: organs are confined to the cavity except those
    post = sig(-np.minimum(np.minimum(E(1.02, -0.27, 20, 30, 10), E(1.02, 0.27, 20, 30, -10)), E(0.93, 0.0, 12, 13)))
    region = np.clip(cav + wall * post, 0, 1)
    w = {k: v * region for k, v in w.items()}
    w["abd_muscle"] = w.get("abd_muscle", 0) + wall * (1 - post)
    w["skin"], w["subcut_fat"] = skin, fat
    w.pop("air", None)                                                # gas has no signal
    names = list(w)
    weights = np.stack([w[k] for k in names]).astype(np.float32)
    tis = _jitter(tissue.table(b0_t), names, tissue_rng or rng)
    return dict(kind="abdomen", b0_t=b0_t, n=n, extent_mm=extent_mm, names=names, weights=weights, tissues=tis, air=air,
                masks=dict(body=(sig(t) > 0.5), brain=(cav > 0.5), lesion=lesion_mask.astype(np.float32)),
                size_mm=(2 * ay_, 2 * ax_))


# ------------------------------------------------------------------------------------------------ QA phantom
QA_SPEC = dict(
    n_ring=(4, 6), p_centre=0.5, n_oil=0, body_rotation_deg=None,
    body_radius_mm=(89.0, 98.0), ring_radius_mm=(53.0, 59.0), vial_radius_mm=(17.8, 20.2), roi_radius_mm=11.5,
    offset_mm=3.6,
    vial_t1_ms=(300.0, 2500.0), vial_t2_ms=(40.0, 200.0), vial_t2p_ms=(40.0, 300.0), vial_pd=(0.55, 1.0),
    vial_adc_um2_ms=(0.6, 2.2), t2_over_t1_max=0.6, oil_t1_jitter=0.2,
)


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def qa(rng, b0_t=3.0, n=640, extent_mm=256.0, spec=None, tissue_rng=None, layout_only=False, rot_deg=0.0, warp=None, edge_mm=None):
    """Quantitative QA phantom: a disc of gel ("body") with a ring of vials and, in half of the draws, a centre vial.

    Every vial is one homogeneous material with its own PD, T1, T2, T2' and diffusion coefficient, drawn log-uniformly
    from the ranges in ``spec`` (they are test-object values, not tissue values). ``n_oil`` of the vials hold oil (the
    fat row of the tissue table, with its chemical shift); the T1 of each oil vial is scaled by a factor drawn
    uniformly within +- ``oil_t1_jitter``, so that no constant inversion time nulls the oil of every world.
    ``body_aspect`` (optional, a (low, high) range): the body is elongated, its second semi-axis is the first divided
    by a ratio drawn from the range.
    ``regions`` lists a scored circle inside every vial; the region ids are a random permutation, so an id says
    nothing about the draw order.
    ``inserts`` (optional): discs placed by the caller, a list of dicts ``y_mm, x_mm, radius_mm`` (mm from the
    isocentre, inside or beside the body), ``tissue`` ("oil": the fat row of the tissue table; or a dict with pd,
    t1_ms, t2_ms, t2p_ms, adc_um2_ms and optionally fat) and optionally ``weight`` (share of the disc this material
    fills, default 1: two inserts on one disc are a mixture, such as water and fat in one voxel). They form the
    classes ``insert0``, ``insert1`` ... after the vials and are not listed in ``regions``. Nothing is drawn for
    them: a spec without the key gives the world it gave before.
    """
    sig = soft(edge_mm)                                                  # every boundary of this build
    sp = dict(QA_SPEC, **(spec or {}))
    t = tissue_rng or rng
    by, bx = (float(v) for v in rng.uniform(-sp["offset_mm"], sp["offset_mm"], 2))
    ba, bb, bth = float(rng.uniform(*sp["body_radius_mm"])), float(rng.uniform(*sp["body_radius_mm"])), float(rng.uniform(0, 180))
    if sp.get("body_aspect") is not None:                               # an elongated body: second semi-axis = first / a drawn ratio
        bb = ba / float(rng.uniform(*sp["body_aspect"]))                # (a key of the spec only; without it no draw is added)
    if sp["body_rotation_deg"] is not None:                             # only these orientations (the draw above is kept)
        bth = float(sp["body_rotation_deg"][int(bth // (180.0 / len(sp["body_rotation_deg"])))])
    n_ring = int(rng.integers(sp["n_ring"][0], sp["n_ring"][1] + 1))
    ring, start = float(rng.uniform(*sp["ring_radius_mm"])), float(rng.uniform(0, 2 * np.pi))
    ang = start + 2 * np.pi * np.arange(n_ring) / n_ring + rng.uniform(-0.06, 0.06, n_ring)
    centres = [(by + ring * np.sin(a), bx + ring * np.cos(a)) for a in ang]
    if rng.uniform() < sp["p_centre"]:
        centres.append((by + float(rng.uniform(-2.4, 2.4)), bx + float(rng.uniform(-2.4, 2.4))))
    vials = []
    for cy, cx in centres:
        r, e = rng.uniform(*sp["vial_radius_mm"]), rng.uniform(-0.05, 0.05)
        vials.append(dict(y=float(cy), x=float(cx), a=float(r * (1 + e)), b=float(r * (1 - e)), rot=float(rng.uniform(0, 180))))
    order = rng.permutation(len(vials))
    oil = set(int(i) for i in rng.permutation(len(vials))[:int(sp["n_oil"])])

    fat = tissue.table(b0_t)["subcut_fat"]
    tis = {"body": dict(pd=float(t.uniform(0.7, 0.9)), t1_ms=float(t.uniform(700, 1300)), t2_ms=float(t.uniform(50, 110)),
                        t2p_ms=float(t.uniform(60, 200)), adc_um2_ms=float(t.uniform(0.8, 2.0)), fat=False)}
    for i in range(len(vials)):
        t1 = _logu(t, *sp["vial_t1_ms"])
        cls = dict(pd=float(t.uniform(*sp["vial_pd"])), t1_ms=t1,
                   t2_ms=min(_logu(t, *sp["vial_t2_ms"]), sp["t2_over_t1_max"] * t1),
                   t2p_ms=_logu(t, *sp["vial_t2p_ms"]), adc_um2_ms=float(t.uniform(*sp["vial_adc_um2_ms"])), fat=False)
        if i in oil:
            cls = dict(fat, fat=True)
        tis[f"vial{i}"] = cls
    for i in sorted(oil):                                               # drawn after every vial: the other draws do not move
        tis[f"vial{i}"]["t1_ms"] *= float(1.0 + sp["oil_t1_jitter"] * t.uniform(-1.0, 1.0))
    inserts = list(sp.get("inserts") or ())                             # placed, not drawn (see the docstring)
    for k, s in enumerate(inserts):
        tis[f"insert{k}"] = dict(fat, fat=True) if s["tissue"] == "oil" else dict(dict(adc_um2_ms=1.5, fat=False), **s["tissue"])
    for cls in tis.values():
        cls["t2s_ms"] = 1.0 / (1.0 / cls["t2_ms"] + 1.0 / cls["t2p_ms"])

    regions = [dict(id=f"R{k + 1}", cls=f"vial{int(i)}", y_mm=round(vials[int(i)]["y"], 2), x_mm=round(vials[int(i)]["x"], 2),
                    radius_mm=float(sp["roi_radius_mm"])) for k, i in enumerate(order)]
    if layout_only:
        return dict(kind="qa", b0_t=b0_t, n=n, extent_mm=extent_mm,
                    names=["body"] + [f"vial{i}" for i in range(len(vials))] + [f"insert{k}" for k in range(len(inserts))],
                    tissues=tis, regions=regions, size_mm=(2 * ba, 2 * bb), body=dict(y=by, x=bx, a=ba, b=bb, rot=bth))
    y, x = axes(n, extent_mm, rot_deg, warp)
    w_body = sig(-ell(y, x, by, bx, ba, bb, bth))
    w_vials = [sig(-ell(y, x, v["y"], v["x"], v["a"], v["b"], v["rot"])) for v in vials]
    names = ["body"] + [f"vial{i}" for i in range(len(vials))]
    for k, s in enumerate(inserts):
        w_vials.append(float(s.get("weight", 1.0)) * sig(-ell(y, x, s["y_mm"], s["x_mm"], s["radius_mm"], s["radius_mm"])))
        names.append(f"insert{k}")
    inside = np.clip(np.sum(w_vials, axis=0), 0, 1) if w_vials else 0.0
    weights = np.stack([w_body * (1 - inside)] + w_vials).astype(np.float32)
    body = w_body > 0.5
    return dict(kind="qa", b0_t=b0_t, n=n, extent_mm=extent_mm, names=names, weights=weights, tissues=tis, air=[],
                masks=dict(body=body, brain=body, lesion=np.zeros(y.shape, np.float32)), size_mm=(2 * ba, 2 * bb),
                regions=regions)


def param_map(ph, key):
    """Partial-volume weighted map of a tissue parameter (for display and for the verifier's truth)."""
    w = ph["weights"]
    vals = np.array([ph["tissues"][k][key] for k in ph["names"]], np.float32)
    tot = w.sum(0)
    return np.where(tot > 0.05, np.tensordot(vals, w, 1) / np.maximum(tot, 1e-6), 0.0)


def label_map(ph):
    w = ph["weights"]
    return np.where(w.sum(0) > 0.3, w.argmax(0) + 1, 0)


# ------------------------------------------------------------------------------------------------ ACR large phantom
# The large phantom of the ACR MRI accreditation programme, as far as one 2D slice can show it. A source tag per number:
#   [G]  "Large and Medium Phantom Test Guidance for the ACR MRI Accreditation Program" (American College of Radiology,
#        the version dated 10.19.2022), read in full for this module; the section or table is named
#   [D]  declared by this package: the guidance does not give the number (layout inside a section, the relaxation
#        times of the solution, how soft an edge is)
# Records: docs/physics/references.md ("Reference objects"); what is drawn and what is not: docs/physics/phantoms.md.
ACR = dict(
    diameter_mm=190.0,                        # [G] Table 1: internal (signal-producing) diameter
    length_mm=148.0,                          # [G] Table 1: internal length
    slices=11, slice_mm=5.0, gap_mm=5.0,      # [G] section 0.3: 11 slices of 5 mm with a 5 mm gap, 100 mm from slice 1 to slice 11
    hole_mm=(1.1, 1.0, 0.9),                  # [G] Table 1, section 2.2: hole diameters of the resolution insert, left to right
    hole_pitch=2.0,                           # [G] section 2.2: centre-to-centre separation in a row and between rows = 2 x diameter
    holes=4,                                  # [G] section 2.2: 4 rows of 4 holes (upper left), 4 columns of 4 holes (lower right), one shared
    lcd_contrast={8: 0.014, 9: 0.025, 10: 0.036, 11: 0.051},   # [G] section 7.2: contrast of the disks of slices 8 to 11
    lcd_spokes=10, lcd_disks=3,               # [G] section 7.2: ten spokes of three disks
    lcd_disk_mm=(7.0, 1.5),                   # [G] section 7.2: disk diameter of the first and of the tenth spoke
    # ---- declared
    hole_stagger=0.25,                        # [D] each row (column) is displaced by this share of the hole diameter ("slightly", section 2.2)
    hole_edge_mm=0.15,                        # [D] half-width of the soft edge of a hole
    insert_y_mm=(8.0, 42.0), insert_x_mm=(-48.0, 48.0),    # [D] the plastic block of the resolution insert (no signal), below the centre
    pair_x_mm=(-28.0, 0.0, 28.0), pair_y_mm=25.0,          # [D] centres of the three pairs of hole arrays
    grid_half_mm=60.0, grid_pitch_mm=15.0, grid_bar_mm=1.2,    # [D] the grid of slice 5: bars of plastic inside a square
    lcd_radius_mm=(16.0, 28.0, 40.0),         # [D] distance of the three disks of a spoke from the centre
    lcd_edge_mm=0.2,                          # [D] half-width of the soft edge of a disk
    lcd_first_deg=9.0,                        # [D] the first spoke, clockwise from 12 o'clock ("slightly to the right", section 7.2)
    lcd_sheet_mm=56.0,                        # [D] radius of the plastic sheet the disks are drilled in
    fill=dict(pd=1.0, t1_ms=150.0, t2_ms=130.0, t2p_ms=1000.0, adc_um2_ms=2.0, fat=False),   # [D] 10 mM NiCl2 + 75 mM NaCl ([G] 0.2): its relaxation times are not in the guidance
)
ACR_SLICES = (1, 5, 7, 8, 9, 10, 11, "sagittal")     # the sections this module draws
ACR_SPEC = dict(slice=7, offset_mm=1.0, rotation_deg=0.3)   # positioning error of the phantom: uniform within +- these


def acr_lcd_disk_mm():
    """Disk diameter of the ten spokes: the two ends are the guidance's, the eight between are a geometric
    progression (declared; the guidance says "decreases progressively")."""
    a, b = ACR["lcd_disk_mm"]
    return [float(a * (b / a) ** (k / (ACR["lcd_spokes"] - 1))) for k in range(ACR["lcd_spokes"])]


def acr_layout(spec=None, rng=None):
    """Where things are in a section of the ACR phantom, in mm from the isocentre: the position of the phantom
    (drawn from ``rng``; centred and upright without one), the holes of the resolution insert (slice 1) and the disks
    of the low-contrast section (slices 8 to 11). The same numbers place the shapes (``acr``) and the measurements
    (``mrgym.qa``); an analysis that is not told the positioning error uses ``acr_layout()``."""
    sp = dict(ACR_SPEC, **(spec or {}))
    if sp["slice"] not in ACR_SLICES:
        raise ValueError(f"the ACR phantom has the sections {ACR_SLICES}")
    oy, ox, rot = 0.0, 0.0, 0.0
    if rng is not None:                                                 # the same three draws for every section
        oy, ox = (float(v) for v in rng.uniform(-1.0, 1.0, 2) * float(sp["offset_mm"]))
        rot = float(rng.uniform(-1.0, 1.0) * float(sp["rotation_deg"]))
    co, si = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))

    def place(v, u):                                                    # phantom frame -> patient coordinates
        return float(oy + v * co + u * si), float(ox - v * si + u * co)

    out = dict(slice=sp["slice"], centre_mm=(oy, ox), rotation_deg=rot, holes=[], disks=[])
    if sp["slice"] == 1:
        for d, xc in zip(ACR["hole_mm"], ACR["pair_x_mm"]):
            p, s, n = ACR["hole_pitch"] * d, ACR["hole_stagger"] * d, ACR["holes"]
            y0 = ACR["pair_y_mm"] - (n - 1) * p - 0.5 * (n - 1) * s     # the shared hole sits at the centre of the pair
            x0 = xc - (n - 1) * p - (n - 1) * s
            yc, xc_ = y0 + (n - 1) * p, x0 + (n - 1) * p + (n - 1) * s  # the shared hole: last of the last row, first of the first column
            for r in range(n):                                          # upper left: rows, each displaced to the right
                for c in range(n):
                    y, x = place(y0 + r * p, x0 + c * p + r * s)
                    out["holes"].append(dict(d_mm=d, array="UL", line=r, k=c, y_mm=y, x_mm=x))
            for c in range(n):                                          # lower right: columns, each displaced downward
                for r in range(n):
                    if (c, r) == (0, 0):
                        continue                                        # the shared hole is already there
                    y, x = place(yc + r * p + c * s, xc_ + c * p)
                    out["holes"].append(dict(d_mm=d, array="LR", line=c, k=r, y_mm=y, x_mm=x))
            y, x = place(yc, xc_)
            out["holes"].append(dict(d_mm=d, array="LR", line=0, k=0, y_mm=y, x_mm=x, shared=True))
    if sp["slice"] in ACR["lcd_contrast"]:
        for k, d in enumerate(acr_lcd_disk_mm()):
            th = np.deg2rad(ACR["lcd_first_deg"] + k * 360.0 / ACR["lcd_spokes"])    # clockwise from 12 o'clock on the display
            for j, r in enumerate(ACR["lcd_radius_mm"]):
                y, x = place(-r * np.cos(th), r * np.sin(th))
                out["disks"].append(dict(spoke=k + 1, disk=j, d_mm=d, y_mm=y, x_mm=x))
    return out


def acr(rng, b0_t=1.5, n=640, extent_mm=256.0, spec=None, tissue_rng=None, rot_deg=0.0, warp=None, edge_mm=None):
    """One section of the ACR large phantom (``ACR``): a disc of solution, 190 mm across, and what the section holds.

        slice 1          the resolution insert: three pairs of hole arrays (1.1, 1.0, 0.9 mm) in a block of plastic
        slice 5          the grid (older large phantoms)
        slice 7          solution only: the uniform section
        slices 8 to 11   the low-contrast section: ten spokes of three disks, contrast 1.4, 2.5, 3.6, 5.1 %
        "sagittal"       the localiser plane: a 148 x 190 mm rectangle (length along y)

    One material ("fill"). A plastic part displaces solution, so it lowers the weight of the fill: to zero where the
    plastic fills the slice (insert, grid), by the contrast where a thin sheet lies in it (low-contrast section; the
    disks are holes in the sheet, [G] 7.2). Not drawn, because they need structure through the slice that a 2D world
    does not have: the slice-thickness ramps and the 45 degree wedges of slices 1 and 11, and everything inside the
    sagittal plane. The relaxation times of the fill are declared constants, the same for every seed."""
    sig = soft(edge_mm)                                                  # every boundary of this build
    sp = dict(ACR_SPEC, **(spec or {}))
    lay = acr_layout(sp, rng)
    y, x = axes(n, extent_mm, rot_deg, warp)
    (oy, ox), rot = lay["centre_mm"], lay["rotation_deg"]
    co, si = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    v, u = (y - oy) * co - (x - ox) * si, (y - oy) * si + (x - ox) * co      # phantom frame
    r = ACR["diameter_mm"] / 2
    if sp["slice"] == "sagittal":
        w = sig(ACR["length_mm"] / 2 - np.abs(v)) * sig(r - np.abs(u))
        size = (ACR["length_mm"], ACR["diameter_mm"])
    else:
        w = sig(r - np.hypot(v, u))
        size = (ACR["diameter_mm"], ACR["diameter_mm"])
    if sp["slice"] == 1:
        (y0, y1), (x0, x1) = ACR["insert_y_mm"], ACR["insert_x_mm"]
        block = sig(v - y0) * sig(y1 - v) * sig(u - x0) * sig(x1 - u)
        holes = np.zeros(y.shape)
        near = np.nonzero(block > 1e-3)                                  # the holes lie in the block: work on its points only
        yn, xn, hn = y[near], x[near], np.zeros(len(near[0]))
        for h in lay["holes"]:
            hn = np.maximum(hn, sig(h["d_mm"] / 2 - np.hypot(yn - h["y_mm"], xn - h["x_mm"]), ACR["hole_edge_mm"]))
        holes[near] = hn
        w = w * (1 - block * (1 - holes))
    elif sp["slice"] == 5:
        half, pitch, bar = ACR["grid_half_mm"], ACR["grid_pitch_mm"], ACR["grid_bar_mm"]
        inside = sig(half - np.abs(v)) * sig(half - np.abs(u))
        dv = np.abs(v % pitch - pitch / 2)                               # distance to the nearest bar along each axis:
        du = np.abs(u % pitch - pitch / 2)                               # bars at +-7.5, +-22.5, ... mm, the centre lines are open
        bars = np.maximum(sig(bar / 2 - dv, 0.2), sig(bar / 2 - du, 0.2))
        w = w * (1 - inside * bars)
    elif sp["slice"] in ACR["lcd_contrast"]:
        sheet = sig(ACR["lcd_sheet_mm"] - np.hypot(v, u))
        disks = np.zeros(y.shape)
        near = np.nonzero(sheet > 1e-3)                                  # the disks are holes in the sheet
        yn, xn, dn = y[near], x[near], np.zeros(len(near[0]))
        for dk in lay["disks"]:
            dn = np.maximum(dn, sig(dk["d_mm"] / 2 - np.hypot(yn - dk["y_mm"], xn - dk["x_mm"]), ACR["lcd_edge_mm"]))
        disks[near] = dn
        w = w * (1 - ACR["lcd_contrast"][sp["slice"]] * sheet * (1 - disks))
    fill = dict(ACR["fill"])
    fill["t2s_ms"] = 1.0 / (1.0 / fill["t2_ms"] + 1.0 / fill["t2p_ms"])
    body = w > 0.5
    return dict(kind="acr", b0_t=b0_t, n=n, extent_mm=extent_mm, names=["fill"], weights=w[None].astype(np.float32),
                tissues=dict(fill=fill), air=[], masks=dict(body=body, brain=body, lesion=np.zeros(y.shape, np.float32)),
                size_mm=size, regions=[], layout=lay)


# ------------------------------------------------------------------------------------------------ NIST T1 plate
# The NiCl2 (T1) array of the ISMRM / NIST system phantom. Source tags:
#   [S]  Stupic et al. 2021, Magn Reson Med 86:1194 (PMC8252537), article body, read for this module
#   [R]  reference T1 of the 14 spheres at 20 degrees C as listed in the analysis code of the ISMRM reproducibility
#        challenge (rrsg2020/analysis, src/nist.py, the list for serial numbers below 0043); the twelfth value equals
#        the T1 that [S] gives for the Ni-12 solution at 3 T and 20.0 degrees C (44.53 ms)
#   [D]  declared: not in what was opened. Tables S1 to S3 of [S] (positions, 1.5 T values, T2 of every sphere) were
#        NOT opened, so this world exists at 3 T only, the order of the spheres on the plate is a choice, and every
#        T2 except that of sphere 12 follows a declared rule
NIST_T1 = dict(
    shell_mm=200.0,                           # [S] 2.1: inner diameter of the spherical shell
    sphere_mm=15.0, sphere_outer_mm=20.0,     # [S] 2.3: inner and outer diameter of an array sphere (polypropylene)
    ring_mm=50.0, ring_n=10, grid_mm=40.0,    # [S] 2.3: 10 spheres on a 50 mm radius, 4 inside on a 40 mm grid
    t1_ms=(1989.0, 1454.0, 984.1, 706.0, 496.0, 351.5, 247.13, 175.3, 125.9, 89.0, 62.7, 44.53, 30.84, 21.719),   # [R]
    field_t=3.0,                              # [S] the field of the Ni-12 value that anchors [R]
    t2_ms_sphere_12=31.86,                    # [S] 2.3: Ni-12 at 3 T, 20.0 degrees C
    fill=dict(pd=1.0, t1_ms=2800.0, t2_ms=2000.0, t2p_ms=1000.0, adc_um2_ms=2.0, fat=False),   # [D] "long-T1 deionized water" ([S] 2.2); no value given
    roi_mm=5.0,                               # [S] text at Figure 5: "recommended 10.0 mm circular ROIs centered on the NiCl2 array spheres"
)


def nist_t1(rng, b0_t=3.0, n=640, extent_mm=256.0, spec=None, tissue_rng=None, rot_deg=0.0, warp=None, edge_mm=None):
    """The T1 plate of the ISMRM / NIST system phantom as a 2D world: 14 spheres (15 mm) of NiCl2 solution in a 200 mm
    disc of water, T1 from ``NIST_T1["t1_ms"]`` with no jitter. Regions ``T1-1`` .. ``T1-14`` (radius 5 mm) follow the
    order of the list: 1 to 10 round the ring, 11 to 14 on the inner grid (a declared order). 3 T only. T2 of a
    sphere = T1 x (31.86 / 44.53), the ratio of sphere 12: a declared rule, not the published T2. The object is the
    same for every seed; its position in the scanner is drawn like that of the ACR phantom (``ACR_SPEC``), and the
    regions lie where the spheres are."""
    sig = soft(edge_mm)                                                  # every boundary of this build
    if b0_t != NIST_T1["field_t"]:
        raise ValueError("the reference values that were opened for the NIST T1 plate are 3 T values; this world exists at 3 T only")
    sp = dict(ACR_SPEC, **(spec or {}))
    oy, ox = (float(v) for v in rng.uniform(-1.0, 1.0, 2) * float(sp["offset_mm"]))
    rot = float(rng.uniform(-1.0, 1.0) * float(sp["rotation_deg"]))
    co, si = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    y, x = axes(n, extent_mm, rot_deg, warp)
    ang = 2 * np.pi * np.arange(NIST_T1["ring_n"]) / NIST_T1["ring_n"]
    g = NIST_T1["grid_mm"] / 2
    plate = [(-NIST_T1["ring_mm"] * np.cos(a), NIST_T1["ring_mm"] * np.sin(a)) for a in ang] + [(-g, -g), (-g, g), (g, g), (g, -g)]
    centres = [(oy + v * co + u * si, ox - v * si + u * co) for v, u in plate]                 # plate frame -> patient coordinates
    ratio = NIST_T1["t2_ms_sphere_12"] / NIST_T1["t1_ms"][11]
    body = sig(NIST_T1["shell_mm"] / 2 - np.hypot(y - oy, x - ox))
    spheres = [sig(NIST_T1["sphere_mm"] / 2 - np.hypot(y - cy, x - cx)) for cy, cx in centres]
    walls = np.clip(np.sum([sig(NIST_T1["sphere_outer_mm"] / 2 - np.hypot(y - cy, x - cx)) for cy, cx in centres], axis=0), 0, 1)
    names = ["fill"] + [f"sphere{i + 1}" for i in range(len(centres))]
    tis = {"fill": dict(NIST_T1["fill"])}
    for i, t1 in enumerate(NIST_T1["t1_ms"]):
        tis[f"sphere{i + 1}"] = dict(pd=1.0, t1_ms=float(t1), t2_ms=float(t1 * ratio), t2p_ms=1000.0, adc_um2_ms=2.0, fat=False)
    for cls in tis.values():
        cls["t2s_ms"] = 1.0 / (1.0 / cls["t2_ms"] + 1.0 / cls["t2p_ms"])
    weights = np.stack([body * (1 - walls)] + spheres).astype(np.float32)          # the wall of a sphere gives no signal
    regions = [dict(id=f"T1-{i + 1}", cls=f"sphere{i + 1}", y_mm=round(float(cy), 2), x_mm=round(float(cx), 2), radius_mm=NIST_T1["roi_mm"])
               for i, (cy, cx) in enumerate(centres)]
    mask = body > 0.5
    return dict(kind="nist_t1", b0_t=b0_t, n=n, extent_mm=extent_mm, names=names, weights=weights, tissues=tis, air=[],
                masks=dict(body=mask, brain=mask, lesion=np.zeros(y.shape, np.float32)), size_mm=(NIST_T1["shell_mm"],) * 2,
                regions=regions, layout=dict(centre_mm=(oy, ox), rotation_deg=rot))
