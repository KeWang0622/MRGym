"""Signal synthesis for the recon-repair tasks: objects, coils, Cartesian signal synthesis.

Conventions used everywhere in this package
- Positions are in units of the prescribed field of view (FOV = 1). An array index ``i`` along an axis of
  ``n`` points sits at position ``(i - n // 2) / n``.
- Spatial frequency ``k`` is in cycles per FOV. On an ``n``-point axis index ``i`` holds ``k = i - n // 2``.
- The standard signal equation is ``S(k) = sum_r rho(r) exp(-2 pi i k.r)``.
- Signals are synthesised from a grid ``FINE`` times finer than the reconstructed one, so a reconstruction that
  uses the same discrete model as the simulator does not get a free ride (no inverse crime).
"""
import numpy as np

FINE = 3
EDGE = 0.006      # soft boundary of every shape, in FOV units; fixed so truth and data describe one object


def cfft(a, axes, norm="forward"):
    """Centred FFT. norm='forward' scales the forward transform by 1/n, so k-space values do not depend on grid size."""
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a, axes=axes), axes=axes, norm=norm), axes=axes)


def cifft(a, axes, norm="forward"):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(a, axes=axes), axes=axes, norm=norm), axes=axes)


def crop_centre(a, n, axis):
    """Keep the ``n`` samples of ``a`` along ``axis`` that hold k (or position index) -(n // 2) .. n - 1 - n // 2."""
    lo = a.shape[axis] // 2 - n // 2
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(lo, lo + n)
    return a[tuple(sl)]


# ----------------------------------------------------------------------------------------------- objects

def random_object(rng):
    """An asymmetric phantom: a body ellipse with inner structures, all inside radius 0.36 of the FOV centre."""
    body = dict(y=rng.uniform(-0.03, 0.03), x=rng.uniform(-0.03, 0.03), a=rng.uniform(0.24, 0.30),
                b=rng.uniform(0.17, 0.23), th=rng.uniform(0, np.pi), amp=rng.uniform(0.5, 0.8))
    shapes = [body]
    for _ in range(int(rng.integers(5, 9))):
        r, ang = rng.uniform(0.0, 0.14), rng.uniform(0, 2 * np.pi)
        shapes.append(dict(y=body["y"] + r * np.sin(ang), x=body["x"] + r * np.cos(ang),
                           a=rng.uniform(0.015, 0.07), b=rng.uniform(0.012, 0.05), th=rng.uniform(0, np.pi),
                           amp=rng.choice([-1, 1]) * rng.uniform(0.15, 0.4)))
    # a bright off-centre marker makes every mirror, transpose and shift error visible
    ang = rng.uniform(0, 2 * np.pi)
    shapes.append(dict(y=body["y"] + 0.13 * np.sin(ang), x=body["x"] + 0.13 * np.cos(ang),
                       a=0.02, b=0.012, th=rng.uniform(0, np.pi), amp=0.6))
    return shapes


def eval_object(shapes, y, x, edge=EDGE):
    """Object value at positions (y, x)."""
    out = np.zeros(np.broadcast(y, x).shape)
    for s in shapes:
        c, sn = np.cos(s["th"]), np.sin(s["th"])
        u = (x - s["x"]) * c + (y - s["y"]) * sn
        v = -(x - s["x"]) * sn + (y - s["y"]) * c
        q = np.sqrt((u / s["a"]) ** 2 + (v / s["b"]) ** 2)
        out += s["amp"] * 0.5 * (1 - np.tanh((q - 1) * min(s["a"], s["b"]) / edge))
    return np.clip(out, 0, None)


# ----------------------------------------------------------------------------------------------- coils

def random_coils(rng, nc):
    """Coil centres on a ring outside the object. Sensitivities are smooth complex functions of position."""
    start = rng.uniform(0, 2 * np.pi)
    coils = []
    for c in range(nc):
        ang = start + 2 * np.pi * c / nc + rng.uniform(-0.15, 0.15)
        rad = rng.uniform(0.55, 0.7)
        coils.append(dict(y=rad * np.sin(ang), x=rad * np.cos(ang), w=rng.uniform(0.38, 0.5),
                          ph=rng.uniform(0, 2 * np.pi), gy=rng.uniform(-2, 2), gx=rng.uniform(-2, 2)))
    return coils


def eval_coils(coils, y, x):
    out = []
    for c in coils:
        d2 = (y - c["y"]) ** 2 + (x - c["x"]) ** 2
        out.append(np.exp(-d2 / (2 * c["w"] ** 2)) * np.exp(1j * (c["ph"] + c["gy"] * y + c["gx"] * x)))
    return np.array(out)


def grid(n, extent=1.0):
    return (np.arange(n) - n // 2) / n * extent


# ----------------------------------------------------------------------------------------------- signals

def truth_image(shapes, ny, nx):
    """Band-limited magnitude of the object on the prescribed ny x nx grid, in the prescription frame [y, x]."""
    fy, fx = FINE * ny, FINE * nx
    y, x = np.meshgrid(grid(fy), grid(fx), indexing="ij")
    fine = eval_object(shapes, y, x)
    k = crop_centre(crop_centre(cfft(fine, (0, 1)), ny, 0), nx, 1)
    return np.abs(cifft(k, (0, 1)))


def coil_kspace(shapes, coils, centre, n_pe, n_s, ro_os, phase_axis):
    """Fine-lattice k-space of every coil image in the acquisition frame [pe, ro].

    The scanner acquires about the isocentre; the prescribed slice centre sits at physical ``centre`` = (y, x), so the
    object appears displaced by ``centre``. Readout samples are spaced 1 / ro_os in k, i.e. the readout FOV is ro_os.
    Returns an array [coil, FINE * n_pe, FINE * n_s]; index (FINE*n//2 + m) holds k = m (pe) or k = m / ro_os (ro).
    """
    f_pe, f_ro = FINE * n_pe, FINE * n_s
    pe, ro = np.meshgrid(grid(f_pe), grid(f_ro, ro_os), indexing="ij")
    y, x = (pe, ro) if phase_axis == "y" else (ro, pe)
    obj = eval_object(shapes, y - centre[0], x - centre[1])
    return cfft(eval_coils(coils, y, x) * obj, (1, 2))


def nrmse(img, truth):
    """Relative error of a magnitude image after the best positive global scale."""
    img = np.asarray(img, dtype=float)
    den = float(np.sum(img * img))
    scale = float(np.sum(img * truth)) / den if den > 0 else 0.0
    return float(np.linalg.norm(max(scale, 0.0) * img - truth) / np.linalg.norm(truth))
