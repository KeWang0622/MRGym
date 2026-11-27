"""From the fetched originals to the small extracts the repository keeps (``docs/physics/measured/*.json``).

One function per data set. Each reads the originals from the cache folder (``sources.fetch``), reduces them to the
numbers a rung compares, and returns a dict that starts with the provenance fields (source, url, licence, citation,
retrieved, sha256 of the original). ``run_all.py --fetch`` writes them; without ``--fetch`` the stored files are
read. Files of other people are parsed as data: nothing in them is imported or executed.
"""
import csv
import os
import re

import cfl
import dcm_min
import fits
import numpy as np
import scipy.io
import scipy.ndimage
import scipy.optimize
import scipy.signal
import sources as SRC

from mrgym import qa
from mrgym.scanner import anatomy


def _r(v, digits=5):
    """Numbers of an extract, rounded to ``digits`` significant digits (None for a missing value)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, digits) for x in v]
    if v is None or not np.isfinite(v):
        return None
    return float(f"{float(v):.{digits}g}")


# ------------------------------------------------------------------------------------------------ M0: the published tables
def nist_tables(cache=None) -> dict:
    """Tables S1, S2 and S4 as typed in ``mrgym.scanner.anatomy``, with the checksum of the PDF they were read from.
    With a cache the PDF is fetched and its checksum compared; its tables cannot be re-read by a program."""
    if cache:
        SRC.fetch("stupic", cache)
    return dict(SRC.provenance("stupic"), unit="ms, at 20 degrees C; columns: T1 at 1.5 T, T2 at 1.5 T, T1 at 3 T, T2 at 3 T",
                table_s1_nicl2=[list(r) for r in anatomy.NIST_S1], table_s2_mncl2=[list(r) for r in anatomy.NIST_S2],
                xz_mm=[list(p) for p in anatomy.NIST_XZ_MM], plate_y_mm=dict(anatomy.NIST_PLATE_Y_MM), position_accuracy_mm=0.4,
                table_s4_water_rate_per_s={str(k): list(v) for k, v in anatomy.NIST_WATER_RATE_PER_S.items()},
                t1_3t_standard_deviation_ms_nicl2=[1.0, 2.5, 0.33, 1.5, 0.41, 0.91, 0.086, 0.11, 0.33, 0.17, 0.13, 0.090, 0.016, 0.0054],
                t2_3t_standard_deviation_ms_mncl2=[0.39, 0.55, 0.28, 0.011, 0.073, 0.049, 0.034, 0.014, 0.083, 0.012, 0.0061, 0.0057, 0.0037, 0.0055],
                temperature_coefficient_mncl2_3t_percent_per_k=dict(t1=2.7, t2=1.6),      # article body, section 2.3
                note="The two standard-deviation columns are those of the 3 T values a rung uses as tolerance. The 4-ms entry of Ni-4 reads 1.5 in the table.")


# ------------------------------------------------------------------------------------------------ M1: the T1 challenge
def _nist_reference(path):
    """The two reference lists (20 degrees C) and the table against temperature of the challenge's ``nist.py``, read
    with regular expressions. Returns (V1, V2, temperatures, V2 against temperature [sphere, temperature])."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lists = re.findall(r"np\.array\(\[\s*([\d.,\s]+?)\s*\]\)", text)
    v1, v2 = (np.array([float(x) for x in s.replace("\n", " ").split(",") if x.strip()]) for s in lists[:2])
    rows = {int(k): {int(t): float(v) for t, v in re.findall(r"'(\d+)':\s*'([\d.]+)'", body)} for k, body in re.findall(r"'(\d+)':\s*\{([^}]*)\}", text)}
    temps = sorted(rows[1])
    return v1, v2, np.array(temps, float), np.array([[rows[k][t] for t in temps] for k in range(1, 15)])


def rrsg(cache) -> dict:
    """ISMRM 2020 T1 challenge, NIST phantom at 3 T: per sphere, the error of every submitted map against the
    challenge's temperature-corrected reference of that phantom version, and the scatter inside the sphere; reduced
    to medians and quartiles over the submissions (all, complex data, magnitude data)."""
    from scipy import interpolate
    d = SRC.fetch("rrsg", cache)
    v1, v2, temps, v2_t = _nist_reference(os.path.join(d, "nist.py"))
    if not np.allclose(v1, anatomy.NIST_T1["t1_ms"]):
        raise RuntimeError("the first reference list of the challenge is not the list in mrgym.scanner.anatomy.NIST_T1")

    def reference(temp_c, serial):                    # the challenge's own correction: quadratic in log T1 against log temperature
        out = np.array([10 ** float(interpolate.interp1d(np.log10(temps), np.log10(row), kind="quadratic")(np.log10(temp_c))) for row in v2_t])
        return out if serial >= 43 else out * v1 / v2

    csv.field_size_limit(10 ** 9)
    with open(os.path.join(d, "3T_NIST_T1maps_database.csv"), newline="", encoding="utf-8") as f:
        subs = list(csv.DictReader(f))
    rows = []
    for r in subs:
        if not r["phantom serial number"] or not r["phantom temperature"] or float(r["id"]) == 13.001:
            continue                                  # no serial number or temperature: no reference; 13.001 as in the analysis of the comparison
        ref = reference(float(r["phantom temperature"]), int(r["phantom serial number"]))
        mean, sd = [], []
        for i in range(1, 15):
            v = np.array([float(t) for t in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", r[f"T1 - NIST sphere {i}"])])
            mean.append(v.mean() if v.size else np.nan)
            sd.append(v.std(ddof=1) if v.size > 1 else np.nan)
        mean, sd = np.array(mean), np.array(sd)
        rows.append(dict(kind=r["Data type"], serial=int(r["phantom serial number"]), err=100 * (mean / ref - 1), cov=100 * sd / mean))

    def stats(sel):
        e, c = np.array([r["err"] for r in sel]), np.array([r["cov"] for r in sel])
        q = lambda a: [_r(np.nanpercentile(a, p, axis=0), 12) for p in (25, 50, 75)]   # noqa: E731  12 digits: the fitted noise level of M1 moves the collapse of spheres 11 to 14 in the last digit shown
        return dict(n=len(sel), error_percent_q25_median_q75=q(e), in_roi_cov_percent_q25_median_q75=q(c))

    old, new = [r for r in rows if r["serial"] <= 42], [r for r in rows if r["serial"] >= 43]
    bias = lambda sel: _r(float(np.nanmedian(np.array([r["err"] for r in sel])[:, :7])), 3)   # noqa: E731
    return dict(SRC.provenance("rrsg"), protocol=dict(sequence="inversion-recovery spin echo", tr_ms=2550, ti_ms=[50, 400, 1100, 2500], te_ms=14, slice_mm=2, pixel_mm=1,
                                                      estimator="Barral 2010: a + b exp(-TI / T1), complex, or magnitude with polarity restoration"),
                submissions_in_the_file=len(subs), used=len(rows), left_out="submissions without serial number or temperature, and id 13.001",
                reference_t1_ms=[float(v) for v in v1], all=stats(rows), complex=stats([r for r in rows if r["kind"] == "Complex"]),
                magnitude=stats([r for r in rows if r["kind"] == "Magnitude"]),
                median_error_percent_spheres_1_to_7=dict(serial_up_to_42=bias(old), n_up_to_42=len(old), serial_from_43=bias(new), n_from_43=len(new)))


# ------------------------------------------------------------------------------------------------ M6: ACR series
def _acr_series(folder):
    hs = [dcm_min.read(os.path.join(folder, f)) for f in sorted(os.listdir(folder))]
    hs.sort(key=lambda h: float(h.get("SliceLocation") or h["Position"].split("\\")[2]))
    imgs = [h["pixels"][0] for h in hs]
    fov = tuple(float(v) * n for v, n in zip(hs[0]["PixelSpacing"].split("\\"), imgs[0].shape))

    def flatness(im):                                 # slice 7 is the uniform one: of the two candidates the flatter inside
        c = qa.find_centre(im, fov, 0.5 * qa.water_level(im))
        y, x = qa.positions(im.shape, fov)
        m = np.hypot(y[:, None] - c[0], x[None, :] - c[1]) < 70
        return im[m].std() / im[m].mean()
    if flatness(imgs[4]) < flatness(imgs[6]):
        imgs = imgs[::-1]
    return hs[0], imgs, fov


def isocentre_ghosting(img, fov) -> float:
    """The ghosting ratio with the four background regions placed from the isocentre, as ``mrgym.qa`` placed them
    before the comparison with real images (kept to show what the error was)."""
    y, x = qa.positions(img.shape, fov)
    m = {k: float(img[np.ix_((y >= r[0]) & (y <= r[1]), (x >= r[2]) & (x <= r[3]))].mean()) for k, r in qa.background_rois(fov).items()}
    return abs((m["top"] + m["bottom"]) - (m["left"] + m["right"])) / (2.0 * qa.ghosting(img, fov)["large_mean"])


def acr_measures(img7, fov, img1=None, img5=None) -> dict:
    """What the QA functions of the package measure on the sections of one series (real or engine)."""
    sd = qa.noise_sd(img7, fov)
    u, g = qa.uniformity(img7, fov), qa.ghosting(img7, fov)
    c = qa.find_centre(img7, fov, 0.5 * qa.water_level(img7))
    y, x = qa.positions(img7.shape, fov)
    r = np.hypot(y[:, None] - c[0], x[None, :] - c[1])
    out = dict(piu=u["piu_percent"], ghost=g["ratio"], snr=g["large_mean"] / sd if sd > 0 else float("nan"), offset_mm=list(c),
               centre_over_rim=float(img7[r < 20].mean() / img7[(r > 70) & (r < 80)].mean()), line_z=qa.artifact_line(img7, fov)["z"])
    if img1 is not None:
        out["d1"] = qa.diameters(img1, fov, angles=(90.0, 0.0))["lengths_mm"]
    if img5 is not None:
        out["d5"] = qa.diameters(img5, fov, angles=(90.0, 0.0, 45.0, 135.0))["lengths_mm"]
    return out


OFF_CENTRE = "GE_Signa_3T_T1"                          # the series whose phantom lies 21 mm from the centre of the image


def acr(cache) -> tuple:
    """The 14 public ACR series: (extract, pixels of slice 7 of the off-centre series as uint16, its field of view)."""
    d = SRC.fetch("hazen", cache)
    series, files, keep = {}, [], None
    for name in SRC.ACR_SERIES:
        folder = os.path.join(d, name)
        files += [f"{name}/{f}" for f in sorted(os.listdir(folder))]
        h, imgs, fov = _acr_series(folder)
        m = acr_measures(imgs[6], fov, imgs[0], imgs[4])
        m["ghost_regions_from_isocentre"] = isocentre_ghosting(imgs[6], fov)
        series[name] = dict(field_t=float(h["FieldStrength"]), tr_ms=float(h["TR"]), te_ms=float(h["TE"]), bandwidth_hz_per_px=float(h["PixelBandwidth"]),
                            vendor=h["Manufacturer"].split()[0], model=h["Model"], coil=h.get("ReceiveCoil"), slices=len(imgs),
                            **{k: _r(v, 6) for k, v in m.items()})
        if name == OFF_CENTRE:
            keep = (np.round(imgs[6]).astype(np.uint16), fov)
            if not np.array_equal(keep[0], imgs[6]):
                raise RuntimeError("the pixels of the off-centre slice are not whole numbers of 16 bits")
    scanners = [k for k, s in series.items() if s["tr_ms"] < 1000]            # one T1 series per scanner
    out = dict(SRC.provenance("hazen", sha256=SRC.manifest_sha256(d, files)), files=len(files), scanners=len(scanners),
               measured_with="mrgym.qa: diameters (slices 1 and 5), uniformity, ghosting, noise_sd, artifact_line (slice 7); centre / rim = mean within 20 mm "
                             "of the centre over the mean between 70 and 80 mm",
               off_centre_slice=dict(series=OFF_CENTRE, file="acr_off_centre_slice7.npz", fov_mm=list(keep[1]), sha256=None,
                                     change="pixel array of slice 7 only (uint16); the DICOM header is not kept"),
               series=series)
    return out, keep[0], keep[1]


# ------------------------------------------------------------------------------------------------ M7: gel sphere
def gel(cache) -> dict:
    """The gel sphere: mean signal of a central disc (0.6 of the radius) at 13 inversion times and at 10 echo times."""
    root = os.path.join(SRC.fetch("gel", cache), "T1_T2_fitting_fBIRN_phantom")

    def series(folder):
        rows = []
        for sub in sorted(os.listdir(folder)):
            m = re.search(r"T[IE](\d+(?:_\d+)?)$", sub)
            if not m or not os.path.isdir(os.path.join(folder, sub)):
                continue
            f = sorted(os.listdir(os.path.join(folder, sub)))[0]
            rows.append((float(m.group(1).replace("_", ".")), dcm_min.read(os.path.join(folder, sub, f))))
        rows.sort(key=lambda r: r[0])
        return np.array([r[0] for r in rows]), [r[1] for r in rows]

    def region(h0, ref):
        m = ref > 0.5 * np.percentile(ref, 99)
        yy, xx = np.nonzero(m)
        cy, cx, r = yy.mean(), xx.mean(), np.sqrt(m.sum() / np.pi)
        y, x = np.mgrid[:ref.shape[0], :ref.shape[1]]
        k = max(4, ref.shape[0] // 12)
        bg = np.concatenate([ref[:k, :k].ravel(), ref[:k, -k:].ravel(), ref[-k:, :k].ravel(), ref[-k:, -k:].ravel()])
        return np.hypot(y - cy, x - cx) < 0.6 * r, float(bg.mean() / 1.2533), r * float(h0["PixelSpacing"].split("\\")[0])

    out = {}
    for key, sub, ref_last in (("ir", "T1", True), ("se", "T2", False)):
        t, hs = series(os.path.join(root, sub))
        imgs = [h["pixels"][0] for h in hs]
        disc, sd, radius = region(hs[0], imgs[-1] if ref_last else imgs[0])
        h = hs[0]
        rec = dict(times_ms=[float(v) for v in t], mean=_r([im[disc].mean() for im in imgs], 6), pixel_sd=_r([im[disc].std(ddof=1) for im in imgs], 4),
                   pixels=int(disc.sum()), corner_noise_sd=_r(sd, 4), phantom_radius_mm=_r(radius, 4),
                   scanner=f"{h['Manufacturer']} {h['Model']}", field_t=float(h["FieldStrength"]), series=h.get("SeriesDescription"), tr_ms=float(h["TR"]),
                   te_ms=float(h.get("TE", h.get("EffectiveTE"))), echo_train=int(h["EchoTrainLength"]), bandwidth_hz_per_px=float(h["PixelBandwidth"]),
                   matrix=int(h["Rows"]), pixel_mm=float(h["PixelSpacing"].split("\\")[0]), slice_mm=float(h["SliceThickness"]), coil=h.get("ReceiveCoil"))
        if key == "ir":
            px = np.stack([im[disc] for im in imgs], -1)
            t1 = fits.ir_rdnls(px, t, magnitude=True)
            rec["t1_pixelwise_ms_mean_sd"] = [_r(t1.mean(), 5), _r(t1.std(), 3)]
        out[key] = rec
    return dict(SRC.provenance("gel"), published=dict(t1_ms=[806.9, 2.4], t2_ms=[81.7, 0.5], where="the article of the data set, fitted from these images; not certified"), **out)


# ------------------------------------------------------------------------------------------------ M8: fat-water phantom
PDFF = (0.0, 2.6, 5.3, 7.9, 10.5, 15.7, 20.9, 31.2, 41.3, 51.4, 100.0)          # the 11 vials, percent (README of the data set)


def pdff(cache) -> dict:
    """Six echo magnitudes per vial of every acquisition. The 11 vials are segmented on the echo sum, ranked by the
    fat-fraction map that ships with the data (used ONLY to tell which vial is which and to start a fit) and given
    the 11 known fat fractions in that order. An acquisition where 11 vials are not found is left out and named."""
    d = SRC.fetch("pdff", cache)
    acq, skipped = [], []
    for f in sorted(x for x in os.listdir(d) if x.endswith(".mat")):
        mat = scipy.io.loadmat(os.path.join(d, f), squeeze_me=True, struct_as_record=False)
        data = mat["imDataAll"]
        im = np.asarray(data.images)
        if im.ndim == 5:
            im = im[:, :, :, 0, :]
        mag = np.abs(im)
        ref = mag.sum(-1).mean(-1)
        lab, n = scipy.ndimage.label(scipy.ndimage.binary_opening(ref > 0.25 * np.percentile(ref, 99.5), iterations=2))
        sizes = scipy.ndimage.sum(np.ones_like(ref), lab, range(1, n + 1))
        keep = np.argsort(sizes)[::-1][:11] + 1
        if len(keep) < 11:
            skipped.append(f"{f}: {len(keep)} vials found")
            continue
        ff_map, r2_map = np.asarray(mat["fwmc_ff"]), np.asarray(mat["fwmc_r2star"])
        rows = []
        for k in keep:
            m = lab == k
            cy, cx = scipy.ndimage.center_of_mass(m)
            yy, xx = np.mgrid[:ref.shape[0], :ref.shape[1]]
            roi = np.hypot(yy - cy, xx - cx) < 0.6 * np.sqrt(m.sum() / np.pi)
            rows.append(dict(sig=mag[roi].mean((0, 1)), ff=float(np.median(ff_map[roi])), r2=float(np.median(r2_map[roi])), n=int(roi.sum())))
        rows.sort(key=lambda v: v["ff"])
        scale = 100.0 if max(v["ff"] for v in rows) > 1.5 else 1.0
        acq.append(dict(file=f[:-4], field_t=_r(float(data.FieldStrength), 4), te_ms=_r(np.asarray(data.TE, float) * 1e3, 5),
                        signal=[_r(v["sig"], 5) for v in rows], shipped_fat_fraction=[_r(v["ff"] / scale, 4) for v in rows],
                        shipped_r2star_per_s=[_r(v["r2"], 4) for v in rows], pixels=[v["n"] for v in rows]))
    return dict(SRC.provenance("pdff", sha256=SRC.sha256_file(os.path.join(d, "PDFF_Phantom_Data.zip"))), pdff_percent=list(PDFF),
                published=dict(bias_percent=0.22, bias_ci_percent=[0.07, 0.38], slope=[0.96, 1.02], intercept_percent=[-0.56, 1.13],
                               where="abstract of the article of the data set (PMC4835219)"),
                acquisitions_in_the_archive=len(acq) + len(skipped), left_out=skipped, acquisitions=acq)


# ------------------------------------------------------------------------------------------------ M9: NIST T2 plate, raw data
XZ = np.array(anatomy.NIST_XZ_MM)
PLATE = np.stack([-XZ[:, 1], XZ[:, 0]], 1)            # (v, u) = (-Z, X): image rows, columns, mm
TURNED = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 12, 13, 10, 11]            # the labels after a turn of the plate by 180 degrees
MIRRORED = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 10, 13, 12]          # ... after a reflection left-right
ROI_MM = 5.0


def _coil_images(k):
    """[readout, phase, coil] k-space with twofold readout oversampling -> coil images, the central half of the readout."""
    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    n = im.shape[0]
    return im[n // 4:n // 4 + n // 2]


def _combined(base, ref):
    """The series of ``base`` as one complex image per contrast: coil images weighted with the conjugate of the coil
    images of contrast ``ref`` (unit norm per pixel). Linear in the data, so noise adds no floor to a decaying signal."""
    k = cfl.read(base)
    r = _coil_images(np.asarray(k[:, :, 0, :, 0, ref]))
    w = np.conj(r) / np.sqrt((np.abs(r) ** 2).sum(-1, keepdims=True))
    return np.stack([(w * _coil_images(np.asarray(k[:, :, 0, :, 0, i]))).sum(-1) for i in range(k.shape[5])], -1)


def _ring(r0, r1, px):
    n = int(np.ceil(r1 / px)) + 1
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1] * px
    k = ((np.hypot(yy, xx) >= r0) & (np.hypot(yy, xx) <= r1)).astype(float)
    return k / k.sum()


def _wall_map(mag, px):
    """How dark a sphere wall (7.5 to 10 mm) round each pixel is against the water just outside it."""
    out = scipy.signal.fftconvolve(mag, _ring(10.6, 12.6, px), mode="same")
    wall = scipy.signal.fftconvolve(mag, _ring(7.9, 9.6, px), mode="same")
    return (out - wall) / np.maximum(out, 1e-3 * mag.max())


def _places(p, mirror=False):
    """Pixel coordinates of the 14 sphere centres for the plate at (row, column, angle, mm per pixel)."""
    cy, cx, th, px = p
    co, si = np.cos(th), np.sin(th)
    v, u = PLATE[:, 0], PLATE[:, 1] * (-1 if mirror else 1)
    return cy + (v * co + u * si) / px, cx + (-v * si + u * co) / px


def locate_plate(mag, px_guess, px_range=0.07):
    """Where the plate lies in a magnitude image: (row, column, angle in rad, mm per pixel), from the 14 sphere walls.
    The centre starts where ``mrgym.qa.find_centre`` finds the phantom; angle, centre and pixel size are searched on
    a grid and refined. The plate looks the same after a half turn and after a reflection, so the LABELS of the
    spheres are not decided here."""
    fov = (px_guess * mag.shape[0], px_guess * mag.shape[1])
    c = qa.find_centre(mag, fov, 0.5 * qa.water_level(mag))
    c0 = (c[0] / px_guess + mag.shape[0] // 2, c[1] / px_guess + mag.shape[1] // 2)
    best = None
    for px in np.arange(px_guess - px_range, px_guess + px_range + 1e-9, 0.01):
        wm = _wall_map(mag, px)
        for th in np.deg2rad(np.arange(0.0, 180.0, 1.0)):
            for dy in range(-6, 7, 2):
                for dx in range(-6, 7, 2):
                    p = (c0[0] + dy, c0[1] + dx, th, px)
                    yy, xx = _places(p)
                    s = float(scipy.ndimage.map_coordinates(wm, [yy, xx], order=1).sum())
                    if best is None or s > best[0]:
                        best = (s, p)
    p = np.array(best[1])
    for _ in range(3):
        wm = _wall_map(mag, p[3])
        p = scipy.optimize.minimize(lambda q, wm=wm: -float(scipy.ndimage.map_coordinates(wm, list(_places(q)), order=1).sum()), p, method="Nelder-Mead",
                                    options=dict(xatol=1e-4, fatol=1e-8, maxiter=4000)).x
    return p


def _b1_map(base):
    """Relative B1 from the preconditioning-pulse scan (Chung 2010; the authors' script gives the nominal angle of
    the preconditioning pulse as 80 degrees): arccos(image with the pulse / image without) / 80 degrees. The readout
    is an asymmetric echo: zero-filled to twice the phase-encoding count. Returns (map, magnitude image without the
    pulse), both square."""
    k = np.asarray(cfl.read(base))
    k = k.reshape(k.shape[0], k.shape[1], k.shape[3], k.shape[-1])            # [readout, phase, coil, (without, with)]
    full = 2 * k.shape[1]
    peak = int(np.abs(k[..., 0]).sum((1, 2)).argmax())
    pad = full // 2 - peak                                                   # the echo centre goes to the middle
    kk = np.zeros((full,) + k.shape[1:], complex)
    n = min(k.shape[0], full - pad)
    kk[pad:pad + n] = k[:n]
    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kk, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))[full // 4:full // 4 + full // 2]
    pd, pre = im[..., 0], im[..., 1]
    ratio = ((np.conj(pd) * pre).sum(-1) / np.maximum((np.abs(pd) ** 2).sum(-1), 1e-30)).real
    return np.arccos(np.clip(ratio, -1.0, 1.0)) / np.deg2rad(80.0), np.sqrt((np.abs(pd) ** 2).sum(-1))


def _roi(shape, cy, cx, px, radius=ROI_MM):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return np.hypot(y - cy, x - cx) * px <= radius


def scholand(cache) -> dict:
    """The T2 plate of a NIST system phantom at 3 T from raw k-space: per sphere the signal at 12 echo times
    (single-echo spin echo) and at 11 inversion times (inversion-recovery spin echo), and B1."""
    d = SRC.fetch("scholand", cache)
    se, ir = _combined(os.path.join(d, "data_GSM_t2"), 0), _combined(os.path.join(d, "data_GSM_t1"), 10)
    te, ti = 15.0 + 40.0 * np.arange(se.shape[-1]), 30.0 + 250.0 * np.arange(ir.shape[-1])
    mag = np.abs(se[..., 0])
    stated_px = 200.0 / mag.shape[0]
    p = locate_plate(mag, stated_px + 0.07)                                   # the stated 200 mm does not fit the plate: searched from 0.78 to 0.92 mm
    fov = p[3] * mag.shape[0]
    outline = qa.diameters(mag, (fov, fov), angles=(0.0, 45.0, 135.0))["lengths_mm"]
    yy, xx = _places(p)
    rois = [_roi(mag.shape, yy[i], xx[i], p[3]) for i in range(14)]
    sig_se = np.array([[np.abs(se[..., k][m].mean()) for k in range(se.shape[-1])] for m in rois])
    sig_ir = np.array([[ir[..., k][m].mean().real for k in range(ir.shape[-1])] for m in rois])
    # the labels: of the four ways the plate can lie (as fitted, turned, mirrored, both), the one whose T1 values follow the published order
    t1 = np.array([fits.ir_three(ti, s)[2] for s in sig_ir])
    cert = np.array(anatomy.NIST_T2["t1_ms"][3.0])
    ways = {"as fitted": list(range(14)), "turned by 180 degrees": TURNED, "mirrored": MIRRORED, "mirrored and turned": [TURNED[i] for i in MIRRORED]}
    misfit = {name: float(np.sqrt(np.mean(np.log(t1[q] / cert) ** 2))) for name, q in ways.items()}
    way = min(misfit, key=misfit.get)
    order = ways[way]
    # B1: the plate is located in the image of the B1 scan as well, and the same four ways are tried against the spin-echo image
    b1, pd = _b1_map(os.path.join(d, "data_05b_b1map"))
    pb = locate_plate(pd, 0.5 * (200.0 + fov) / pd.shape[0], px_range=0.10)   # the field of view of this scan is searched as well
    best = None
    for mirror in (False, True):
        for turn in (0.0, np.pi):
            q = (pb[0], pb[1], pb[2] + turn, pb[3])
            # the spin-echo image resampled onto plate coordinates, and the B1-scan image likewise: the inserts decide
            g = np.mgrid[-80:81:2, -80:81:2].astype(float)
            a = _sample_plate(mag, p, g, False)
            b = _sample_plate(pd, q, g, mirror)
            cc = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            if best is None or cc > best[0]:
                best = (cc, q, mirror)
    cc, q, mirror = best
    by, bx = _places(q, mirror)
    b1_roi = [b1[_roi(b1.shape, by[i], bx[i], q[3])] for i in range(14)]
    # second-order fit of B1 in the coordinates of the plate with its labels: the fitted frame differs from it by a sign of v, of u, or of both
    frame = np.linalg.lstsq(PLATE[order], PLATE, rcond=None)[0].T.round()   # true (v, u) = frame @ fitted (v, u)
    g = np.mgrid[-90:91:3, -90:91:3].astype(float)
    inside = (np.hypot(g[0], g[1]) <= 90.0) & (np.min([np.hypot(g[0] - a, g[1] - b) for a, b in PLATE], axis=0) > 12.0)   # water only: the method
    fitted = np.einsum("ij,jkl->ikl", np.linalg.inv(frame), g)                                                           # misreads the doped spheres
    vals = _sample_plate(b1, q, fitted, mirror)[inside]
    v, u = g[0][inside] / 100.0, g[1][inside] / 100.0
    basis = np.stack([np.ones_like(v), v, u, v * v, v * u, u * u], 1)
    coef = np.linalg.lstsq(basis, vals, rcond=None)[0]
    keep = np.abs(basis @ coef - vals) <= 2.5 * np.sqrt(np.mean((basis @ coef - vals) ** 2))       # once more without the inserts and the fiducials
    basis, vals = basis[keep], vals[keep]
    coef = np.linalg.lstsq(basis, vals, rcond=None)[0]
    bg = np.concatenate([se[2:20, 2:20].real.ravel(), se[2:20, -20:-2].real.ravel(), se[-20:-2, 2:20].real.ravel(), se[-20:-2, -20:-2].real.ravel()])
    spheres = []
    for n, i in enumerate(order):
        m = rois[i]
        spheres.append(dict(sphere=n + 1, pixels=int(m.sum()), centre_px=[_r(yy[i], 5), _r(xx[i], 5)],
                            se_mean=_r(sig_se[i] / sig_se.max(), 5), se_pixel_sd=_r([se[..., k][m].real.std(ddof=1) / sig_se.max() for k in range(se.shape[-1])], 3),
                            ir_mean=_r(sig_ir[i] / sig_se.max(), 5), ir_pixel_sd=_r([ir[..., k][m].real.std(ddof=1) / sig_se.max() for k in range(ir.shape[-1])], 3),
                            b1_in_the_sphere=_r(b1_roi[i].mean(), 4), b1_pixel_sd=_r(b1_roi[i].std(ddof=1), 3),
                            b1=_r(float(np.array([1.0, PLATE[n][0] / 100, PLATE[n][1] / 100, (PLATE[n][0] / 100) ** 2, PLATE[n][0] * PLATE[n][1] / 1e4,
                                                  (PLATE[n][1] / 100) ** 2]) @ coef), 4)))
    return dict(SRC.provenance("scholand"),
                stated=dict(scanner="Siemens Skyra 3 T, 20-channel head coil", tr_ms=8000.0, te_ms=[float(v) for v in te], ir_te_ms=15.0, ti_ms=[float(v) for v in ti],
                            fov_mm=200.0, slice_mm=5.0, temperature_c=21.0,
                            where="Zenodo record (TR, TE, TI, field of view); Table 1 and Methods of the article's preprint, arXiv:2209.08027 (slice, coil, "
                                  "'constant room temperature (about 21 degrees C)')"),
                reconstruction="inverse FFT of the 512 x 256 x 20 k-space, central 256 of the oversampled readout, coils combined with the conjugate coil images "
                               "of the first TE (spin echo) or the last TI (inversion recovery); signal = magnitude (spin echo) or real part (inversion "
                               "recovery) of the complex mean over a 5 mm circle; signals in units of the largest spin-echo mean",
                plate=dict(centre_px=[_r(p[0], 6), _r(p[1], 6)], angle_deg=_r(np.rad2deg(p[2]), 5), pixel_mm=_r(p[3], 5), fov_mm=_r(fov, 5),
                           outline_mm=_r(outline, 5), labels=way, label_misfit={k: _r(v, 3) for k, v in misfit.items()},
                           how="14 sphere walls fitted with the published places of Table S2 (centre, angle, pixel size free); the start is the centre found by "
                               "mrgym.qa.find_centre. The plate looks alike in four positions; the labels are the position whose 14 fitted T1 follow "
                               "the published order (label_misfit: rms of log(T1 fitted / T1 published) for each)"),
                b1=dict(method="preconditioning pulse with TurboFLASH readout (Chung 2010), nominal 80 degrees as in the authors' script; asymmetric echo zero-filled",
                        image_correlation=_r(cc, 3), pixel_mm=_r(q[3], 5),
                        note="b1 of a sphere is the fit of the water round the spheres, evaluated at its place; b1_in_the_sphere is the map inside it, which the "
                             "method misreads where T1 is short (the magnetisation recovers between the preconditioning pulse and the readout)",
                        polynomial=dict(terms="1, v, u, v^2, v u, u^2 with v = -Z / 100 mm, u = X / 100 mm (plate coordinates)", coefficients=_r(coef, 5),
                                        rms_residual=_r(float(np.sqrt(np.mean((basis @ coef - vals) ** 2))), 3), radius_mm=90.0)),
                background_sd=_r(bg.std() / sig_se.max(), 3), spheres=spheres)


def _sample_plate(img, p, grid, mirror):
    """``img`` at the plate coordinates ``grid`` = (v, u) in mm, for the plate at ``p``."""
    cy, cx, th, px = p
    co, si = np.cos(th), np.sin(th)
    v, u = grid[0], grid[1] * (-1 if mirror else 1)
    return scipy.ndimage.map_coordinates(img, [cy + (v * co + u * si) / px, cx + (-v * si + u * co) / px], order=1, mode="constant")


# ------------------------------------------------------------------------------------------------ M5: tissue compilation (never stored)
def itis(cache) -> dict:
    """{tissue: {("t1" | "t2", field): dict(mean, sd, n, lo, hi) or None}} of the relaxation-time table of the IT'IS
    database. The licence of the database is unclear, so this is read from the fetched files only and never stored."""
    import json
    d = SRC.fetch("itis", cache)
    sheets = []
    for rel in SRC.SOURCES["itis"]["files"]:
        with open(os.path.join(d, rel + ".json"), encoding="utf-8") as f:
            sheets.append(json.load(f))
    a, b = sheets

    def cell(sheet, cols, r):
        v = [str(sheet.get(c + str(r), "")) for c in cols]
        if not v[0]:
            return None
        return dict(mean=float(v[0]), sd=float(v[1] or 0), n=int(float(v[2] or 0)), lo=float(v[3] or v[0]), hi=float(v[4] or v[0]))

    db = {}
    for r in range(4, 116):
        name = a.get(f"B{r}", "")
        if name:
            db[name] = {("t1", 1.5): cell(a, ("BZ", "CA", "CB", "CC", "CD"), r), ("t1", 3.0): cell(a, ("CJ", "CK", "CL", "CM", "CN"), r),
                        ("t2", 1.5): cell(b, ("CE", "CF", "CG", "CH", "CI"), r), ("t2", 3.0): cell(b, ("CO", "CP", "CQ", "CR", "CS"), r)}
    return db
