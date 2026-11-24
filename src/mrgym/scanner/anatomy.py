"""Procedural phantoms of the hidden world: a brain-like slice, an abdomen-like slice and a quantitative QA phantom.

Nothing is loaded from disk and nothing is licensed: geometry is a set of implicit functions (layers of a warped
egg-shaped head, sulci as slits in polar coordinates, ellipses for ventricles / nuclei / organs) evaluated through a
smooth random coordinate warp, so every seed gives another subject. The QA phantom is a disc of gel with vials of
drawn T1 / T2 (and, on request, vials of oil), the layout of a relaxometry test object. Output is a stack of soft
partial-volume weights on a master grid much finer than any acquired pixel.

Conventions: positions in mm, arrays [y, x], index i at (i - n // 2) * extent / n; +y is posterior (anterior at the top
of a displayed image), +x is the patient's left (radiological display).
"""
import numpy as np

from . import tissue

EDGE_MM = 0.5            # half-width of every soft boundary


def sig(d, e=EDGE_MM):
    """Soft indicator of d > 0."""
    return 0.5 * (1.0 + np.tanh(d / e))


def axes(n, extent_mm):
    c = (np.arange(n) - n // 2) * (extent_mm / n)
    return np.meshgrid(c, c, indexing="ij")


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


def _layers(t, bounds):
    """Weights of consecutive layers of depth ``t`` between boundaries bounds[i] .. bounds[i+1] (arrays or scalars)."""
    s = [sig(t - b) for b in bounds]
    return [s[i] - s[i + 1] for i in range(len(bounds) - 1)], s[-1]


# ------------------------------------------------------------------------------------------------ brain
def brain(rng, b0_t=3.0, n=640, extent_mm=256.0, lesion=None, atrophy=None, tissue_rng=None):
    """A brain-like axial slice at the level of the basal ganglia. lesion: None, 'ms' or 'tumour'.

    Returns dict(kind, n, extent_mm, names, weights [class, n, n] float32, tissues {name: params}, air [(y, x, z, r)],
    masks {'brain', 'lesion'}).
    """
    d = extent_mm / n
    y0, x0 = axes(n, extent_mm)
    wy, wx = smooth_field(rng, 8, 80, 260, 1.1), smooth_field(rng, 8, 80, 260, 1.1)
    y, x = y0 + wy(y0, x0), x0 + wx(y0, x0)

    sc = rng.uniform(0.95, 1.05)
    ax_, ay_ = sc * rng.uniform(72, 78), sc * rng.uniform(90, 97)
    egg = rng.uniform(0.07, 0.13)                                     # wider posteriorly
    q = np.sqrt((x / (ax_ * (1 + egg * y / ay_))) ** 2 + (y / ay_) ** 2) - 1.0
    t = -sdf(q, d)                                                    # depth below the skin surface, mm
    atrophy = rng.uniform(0.0, 1.0) if atrophy is None else float(atrophy)

    thick = smooth_field(rng, 6, 60, 250, 1.0)(y0, x0)
    b_skin = 1.6
    b_fat = b_skin + rng.uniform(3.0, 6.0) * (1 + 0.25 * thick)
    b_out = b_fat + 1.7
    b_dip = b_out + rng.uniform(1.8, 3.2) * (1 + 0.3 * smooth_field(rng, 6, 40, 200, 1.0)(y0, x0))
    b_in = b_dip + 1.5
    b_csf = b_in + rng.uniform(1.6, 2.6) + 1.2 * atrophy
    (skin, fat, outer, diploe, inner, sub), brain_w = _layers(t, [0.0, b_skin, b_fat, b_out, b_dip, b_in, b_csf])
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
        cand = np.argwhere((comp["wm"] > 0.95) & (np.abs(d_vent) < 14) | (comp["wm"] > 0.98))
        les = np.full(x.shape, 1e3)
        for _ in range(int(rng.integers(4, 9))):
            cy, cx = cand[rng.integers(len(cand))]
            les = np.minimum(les, ell(y, x, y[cy, cx], x[cy, cx], rng.uniform(1.8, 5.0), rng.uniform(1.5, 3.5), rng.uniform(0, 180)))
        lesion_mask = sig(-les) * (comp["wm"] + comp["gm"] + comp["deep_gm"])
        for k in comp:
            comp[k] = comp[k] * (1 - sig(-les)) if k != "csf" else comp[k]
        comp["lesion_ms"] = lesion_mask
    elif lesion == "tumour":
        cand = np.argwhere((comp["wm"] > 0.98) & (np.abs(x) > 18))
        cy, cx = cand[rng.integers(len(cand))]
        core = ell(y, x, y[cy, cx], x[cy, cx], rng.uniform(7, 12), rng.uniform(7, 12), rng.uniform(0, 180))
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
                size_mm=(2 * ay_, 2 * ax_))


# ------------------------------------------------------------------------------------------------ abdomen
def abdomen(rng, b0_t=3.0, n=640, extent_mm=420.0, lesion=None, tissue_rng=None):
    """An abdomen-like axial slice through liver, spleen and kidneys. lesion: None or 'liver'."""
    d = extent_mm / n
    y0, x0 = axes(n, extent_mm)
    wy, wx = smooth_field(rng, 6, 160, 500, 3.0), smooth_field(rng, 6, 160, 500, 3.0)
    y, x = y0 + wy(y0, x0), x0 + wx(y0, x0)
    ax_, ay_ = rng.uniform(150, 172), rng.uniform(100, 118)
    q = (np.abs(x / ax_) ** 2.6 + np.abs(y / ay_) ** 2.6) ** (1 / 2.6) - 1.0
    t = -sdf(q, d)
    oy, ox = smooth_field(rng, 12, 30, 140, 3.0), smooth_field(rng, 12, 30, 140, 3.0)
    y, x = y + oy(y0, x0), x + ox(y0, x0)                            # organs: stronger, finer warp than the outline
    b_skin = 2.0
    b_fat = b_skin + rng.uniform(9, 24) * (1 + 0.3 * smooth_field(rng, 5, 150, 500, 1.0)(y0, x0))
    b_mus = b_fat + rng.uniform(8, 13)
    (skin, fat, wall), cav = _layers(t, [0.0, b_skin, b_fat, b_mus])
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
    vial_adc_um2_ms=(0.6, 2.2), t2_over_t1_max=0.6,
)


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def qa(rng, b0_t=3.0, n=640, extent_mm=256.0, spec=None, tissue_rng=None, layout_only=False):
    """Quantitative QA phantom: a disc of gel ("body") with a ring of vials and, in half of the draws, a centre vial.

    Every vial is one homogeneous material with its own PD, T1, T2, T2' and diffusion coefficient, drawn log-uniformly
    from the ranges in ``spec`` (they are test-object values, not tissue values). ``n_oil`` of the vials hold oil (the
    fat row of the tissue table, with its chemical shift). ``regions`` lists a scored circle inside every vial; the
    region ids are a random permutation, so an id says nothing about the draw order.
    """
    sp = dict(QA_SPEC, **(spec or {}))
    t = tissue_rng or rng
    by, bx = (float(v) for v in rng.uniform(-sp["offset_mm"], sp["offset_mm"], 2))
    ba, bb, bth = float(rng.uniform(*sp["body_radius_mm"])), float(rng.uniform(*sp["body_radius_mm"])), float(rng.uniform(0, 180))
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
    for cls in tis.values():
        cls["t2s_ms"] = 1.0 / (1.0 / cls["t2_ms"] + 1.0 / cls["t2p_ms"])

    regions = [dict(id=f"R{k + 1}", cls=f"vial{int(i)}", y_mm=round(vials[int(i)]["y"], 2), x_mm=round(vials[int(i)]["x"], 2),
                    radius_mm=float(sp["roi_radius_mm"])) for k, i in enumerate(order)]
    if layout_only:
        return dict(kind="qa", b0_t=b0_t, n=n, extent_mm=extent_mm, names=["body"] + [f"vial{i}" for i in range(len(vials))],
                    tissues=tis, regions=regions, size_mm=(2 * ba, 2 * bb), body=dict(y=by, x=bx, a=ba, b=bb, rot=bth))
    y, x = axes(n, extent_mm)
    w_body = sig(-ell(y, x, by, bx, ba, bb, bth))
    w_vials = [sig(-ell(y, x, v["y"], v["x"], v["a"], v["b"], v["rot"])) for v in vials]
    inside = np.clip(np.sum(w_vials, axis=0), 0, 1) if w_vials else 0.0
    names = ["body"] + [f"vial{i}" for i in range(len(vials))]
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
