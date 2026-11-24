"""Scenario: 2D multi-coil Cartesian scan, raw records in a scanner-specific format, SENSE reconstruction.

One *dialect* is one scanner's raw-data convention (array layout, index conventions, units, field names). The same
dialect drives four things, so they cannot drift apart: the simulator that writes raw files, FORMAT.md, the reference
reconstruction, and the faulty reconstruction handed to the agent. A *variant* is one way the reconstruction can
disagree with the format; variants are grouped into mechanisms, and splits are made by mechanism.
"""
import json

import numpy as np

from .. import sim

# variant -> (mechanism group, scan features that must be present for the fault to show, dialect requirements)
VARIANTS = {
    "os_crop_start": ("oversampling", {"os2"}, {}),
    "os_crop_off": ("oversampling", {"os2", "ro_odd"}, {}),
    "rev_ignore": ("readout_direction", {"rev"}, {}),
    "rev_plain_flip": ("readout_direction", {"rev", "ns_even"}, {}),
    "order_ignore": ("line_order", {"order"}, {}),
    "ft_sign": ("ft_convention", set(), {}),
    "kind_ignore": ("record_kind", {"extra"}, {}),
    "kind_partial": ("record_kind", {"extra"}, {}),
    "avg_sum": ("averaging", {"reps"}, {}),
    "gain_ignore": ("receiver_gain", {"gain"}, {}),
    "gain_inverse": ("receiver_gain", {"gain"}, {}),
    "gain_db_power": ("receiver_gain", {"gain"}, {"gain_conv": ["db"]}),
    "chan_ignore": ("channel_order", {"perm"}, {}),
    "chan_inverse": ("channel_order", {"perm"}, {}),
    "shift_ignore": ("fov_shift", {"shift"}, {}),
    "shift_sign": ("fov_shift", {"shift"}, {}),
    "shift_swap": ("fov_shift", {"shift"}, {}),
    "shift_units": ("fov_shift", {"shift"}, {"shift_conv": ["mm", "pix"]}),
    "pa_out": ("phase_axis", {"pa_x"}, {}),
    "pa_sens": ("phase_axis", {"pa_x"}, {}),
    "sens_noconj": ("sense_model", set(), {}),
    "mask_full": ("sense_model", {"r2"}, {}),
    "cplx_real_only": ("complex_storage", set(), {"complex": ["split"]}),
    "cplx_imag_sign": ("complex_storage", set(), {"complex": ["split"]}),
    "fft_centre_odd": ("fft_centring", {"odd_both"}, {}),
}
GROUPS = sorted({g for g, _, _ in VARIANTS.values()})
HELD_OUT_GROUPS = ("line_order", "phase_axis", "receiver_gain")      # never injected in the train split

DIALECT_CHOICES = {
    "layout": ["RCS", "CRS", "RSC"],            # axis order of the raw array: Record, Coil, Sample
    "complex": ["c64", "split"],                # complex64, or float32 with a trailing (real, imag) axis
    "ky_conv": ["signed", "line0"],             # signed ky, or line number 0 .. n_pe - 1
    "kind_conv": ["str", "bits"],
    "dir_conv": ["pm1", "bool"],
    "gain_conv": ["db", "lin"],
    "shift_conv": ["mm", "frac", "pix"],
    "shift_order": ["yx", "xy"],
    "sens_layout": ["CYX", "YXC"],
    "chan_conv": ["data_to_sens", "sens_to_data"],
    "ft_sign": [1, -1],
}
NAME_CHOICES = {
    "data": ["data", "kdata", "raw"], "sens": ["sens", "coil_maps", "prescan"], "header": ["header", "hdr", "meta"],
    "ky": ["ky", "pe_index", "line"], "kind": ["kind", "rec_type", "flags"], "dir": ["ro_dir", "polarity", "reverse"],
    "gain": ["rx_gain", "receiver_gain", "adc_gain"], "n_pe": ["n_pe", "matrix_pe", "pe_lines"],
    "n_ro": ["n_ro", "matrix_ro", "ro_points"], "ro_os": ["ro_os", "readout_oversampling", "os_factor"],
    "phase_axis": ["phase_axis", "pe_axis", "phase_dir"], "coil_map": ["coil_map", "channel_index", "rx_order"],
    "centre": ["centre", "offcentre", "slice_centre"], "fov": ["fov_mm", "fov", "field_of_view_mm"],
}
KIND_BITS = {"noise": 1, "nav": 4, "dummy": 16}


def sample_dialect(rng, require=None):
    d = {k: v[int(rng.integers(len(v)))] for k, v in DIALECT_CHOICES.items()}
    for k, allowed in (require or {}).items():
        if d[k] not in allowed:
            d[k] = allowed[int(rng.integers(len(allowed)))]
    d["names"] = {k: v[int(rng.integers(len(v)))] for k, v in NAME_CHOICES.items()}
    if d["dir_conv"] == "bool":
        d["names"]["dir"] = "reverse"
    elif d["names"]["dir"] == "reverse":
        d["names"]["dir"] = "ro_dir"
    if d["kind_conv"] == "bits":
        d["names"]["kind"] = "flags"
    elif d["names"]["kind"] == "flags":
        d["names"]["kind"] = "kind"
    d["names"]["gain"] += "_db" if d["gain_conv"] == "db" else ""
    d["names"]["centre"] += {"mm": "_mm", "frac": "_frac", "pix": "_pix"}[d["shift_conv"]]
    return d


# ------------------------------------------------------------------------------------------------ scans

def sample_scan(rng, need=frozenset()):
    """Draw one acquisition. Every feature named in ``need`` is present; the others are present at random."""
    def on(token, p=0.5):
        return token in need or bool(rng.random() < p)

    odd = "odd_both" in need
    n_pe = int(rng.integers(28, 49)) * 2 + (1 if odd or rng.random() < 0.3 else 0)
    n_ro = int(rng.integers(28, 49)) * 2 + (1 if odd or "ro_odd" in need or
                                             ("ns_even" not in need and rng.random() < 0.3) else 0)
    ro_os = 2 if on("os2") or ("ns_even" in need and n_ro % 2) else 1
    if "ns_even" in need and (n_ro * ro_os) % 2:
        n_ro += 1
    n_coil = int(rng.choice([4, 6, 8]))
    accel = int(rng.choice([2, 3] if n_coil == 8 else [2])) if on("r2", 0.6) else 1
    lo, hi = -(n_pe // 2), n_pe - 1 - n_pe // 2
    off = int(rng.integers(accel))                # every accel-th line plus a fully sampled centre
    lines = [k for k in range(lo, hi + 1) if (k - lo) % accel == off or abs(k) < 6]
    if on("order"):
        style = str(rng.choice(["descending", "centric", "interleaved", "shuffled"]))
        if style == "descending":
            lines = lines[::-1]
        elif style == "centric":
            lines = sorted(lines, key=lambda k: (abs(k), k))
        elif style == "interleaved":
            lines = lines[0::2] + lines[1::2]
        else:
            lines = [lines[i] for i in rng.permutation(len(lines))]
    recs = [dict(kind="img", ky=k) for k in lines]
    if on("reps"):                               # signal averaging: central lines repeated 1 to 4 times, unevenly
        for k in [k for k in lines if abs(k) < n_pe // 6]:
            for _ in range(int(rng.integers(0, 4))):
                recs.insert(int(rng.integers(len(recs) + 1)), dict(kind="img", ky=k))
    if on("extra"):
        head = [dict(kind="noise", ky=int(rng.integers(lo, hi + 1))) for _ in range(int(rng.integers(3, 7)))]
        head += [dict(kind="dummy", ky=recs[i]["ky"]) for i in range(int(rng.integers(3, 6)))]
        step = int(rng.integers(5, 10))
        body = []
        for i, r in enumerate(recs):
            if i % step == 0:
                body.append(dict(kind="nav", ky=0))
            body.append(r)
        recs = head + body
    rev_style = str(rng.choice(["alternate", "blocks"])) if on("rev") else "none"
    edges, gains = [], [1.0]
    if on("gain"):                               # receiver gain stepped between bands of ky; one step at the centre
        edges = sorted({int(rng.integers(-1, 2))} | {int(e) for e in rng.integers(lo + 4, hi - 3, size=3)})
        gains = [float(np.exp(rng.uniform(np.log(0.2), np.log(3.0))))]
        while len(gains) <= len(edges):
            g = float(np.exp(rng.uniform(np.log(0.2), np.log(3.0))))
            if max(g / gains[-1], gains[-1] / g) >= 1.8:
                gains.append(g)
    for i, r in enumerate(recs):
        r["rev"] = rev_style == "alternate" and i % 2 == 1 or rev_style == "blocks" and (i // 7) % 2 == 1
        r["gain"] = gains[int(np.searchsorted(edges, r["ky"], side="right"))]
    perm = list(range(n_coil))
    if on("perm"):
        while perm == list(range(n_coil)) or [perm[p] for p in perm] == list(range(n_coil)):
            perm = [int(p) for p in rng.permutation(n_coil)]
    centre = (0.0, 0.0)
    if on("shift"):
        mag = rng.uniform(0.05, 0.1, size=2) * rng.choice([-1, 1], size=2)
        if abs(abs(mag[0]) - abs(mag[1])) < 0.02:
            mag[1] = np.sign(mag[1]) * (abs(mag[0]) - 0.03 if abs(mag[0]) > 0.075 else abs(mag[0]) + 0.03)
        centre = (float(mag[0]), float(mag[1]))
    return dict(n_pe=n_pe, n_ro=n_ro, ro_os=ro_os, n_coil=n_coil, records=recs, perm=perm, centre=centre,
                phase_axis="x" if on("pa_x") else "y", fov_mm=float(rng.integers(180, 321)),
                snr=float(rng.uniform(200, 500)), nav_factor=complex(rng.uniform(1.2, 2.0) * np.exp(1j * rng.uniform(0.5, 1.5) * np.pi)),
                obj_seed=int(rng.integers(2 ** 31)), noise_seed=int(rng.integers(2 ** 31)))


def out_shape(scan):
    return (scan["n_pe"], scan["n_ro"]) if scan["phase_axis"] == "y" else (scan["n_ro"], scan["n_pe"])


def simulate(scan, dialect):
    """Returns (npz dict in the dialect, truth magnitude image [ny, nx])."""
    rng = np.random.default_rng(scan["obj_seed"])
    shapes, coils = sim.random_object(rng), sim.random_coils(rng, scan["n_coil"])
    n_pe, n_ro, ro_os, n_coil = scan["n_pe"], scan["n_ro"], scan["ro_os"], scan["n_coil"]
    n_s = n_ro * ro_os
    ny, nx = out_shape(scan)
    truth = sim.truth_image(shapes, ny, nx)
    K = sim.coil_kspace(shapes, coils, scan["centre"], n_pe, n_s, ro_os, scan["phase_axis"])
    c_pe, c_ro = K.shape[1] // 2, K.shape[2] // 2
    sig = dialect["ft_sign"]
    nrng = np.random.default_rng(scan["noise_seed"])
    sd = 0.6 / (scan["snr"] * np.sqrt(n_pe * n_s))
    s_off = np.arange(n_s) - n_s // 2
    data = np.zeros((len(scan["records"]), n_coil, n_s), complex)
    for i, r in enumerate(scan["records"]):
        if r["kind"] != "noise":
            ky = 0 if r["kind"] == "nav" else r["ky"]
            line = K[:, c_pe + sig * ky, c_ro + sig * (-1 if r["rev"] else 1) * s_off]
            line = line * {"img": 1.0, "nav": scan["nav_factor"], "dummy": 1.6}[r["kind"]]
            data[i] = line
        data[i] += sd * (nrng.standard_normal((n_coil, n_s)) + 1j * nrng.standard_normal((n_coil, n_s)))
        data[i] *= r["gain"]
    data = data[:, scan["perm"], :]                      # data channel c holds sensitivity coil perm[c]

    m = int(np.random.default_rng(scan["obj_seed"] + 1).integers(24, 41))
    y, x = np.meshgrid(sim.grid(m), sim.grid(m), indexing="ij")
    sens = sim.eval_coils(coils, y, x)                   # [coil, y, x], physical frame about the isocentre

    N = dialect["names"]
    out = {}
    raw = data.astype(np.complex64)
    raw = {"RCS": raw, "CRS": raw.transpose(1, 0, 2), "RSC": raw.transpose(0, 2, 1)}[dialect["layout"]]
    out[N["data"]] = raw if dialect["complex"] == "c64" else np.stack([raw.real, raw.imag], -1).astype(np.float32)
    sens = sens.astype(np.complex64)
    out[N["sens"]] = sens if dialect["sens_layout"] == "CYX" else sens.transpose(1, 2, 0)
    ky = np.array([r["ky"] for r in scan["records"]])
    out[N["ky"]] = (ky if dialect["ky_conv"] == "signed" else ky + n_pe // 2).astype(np.int32)
    kinds = [r["kind"] for r in scan["records"]]
    out[N["kind"]] = (np.array(kinds) if dialect["kind_conv"] == "str"
                      else np.array([KIND_BITS.get(k, 0) for k in kinds], np.uint8))
    rev = np.array([r["rev"] for r in scan["records"]])
    out[N["dir"]] = rev if dialect["dir_conv"] == "bool" else np.where(rev, -1, 1).astype(np.int8)
    g = np.array([r["gain"] for r in scan["records"]])
    out[N["gain"]] = (20 * np.log10(g) if dialect["gain_conv"] == "db" else g).astype(np.float32)
    cy, cx = scan["centre"]
    unit = {"mm": (scan["fov_mm"], scan["fov_mm"]), "frac": (1, 1), "pix": (ny, nx)}[dialect["shift_conv"]]
    cy, cx = cy * unit[0], cx * unit[1]
    cmap = scan["perm"] if dialect["chan_conv"] == "data_to_sens" else [int(i) for i in np.argsort(scan["perm"])]
    hdr = {"sequence": "gre2d", "field_T": 3.0, "te_ms": 4.6, "tr_ms": 12.0, "flip_deg": 15,
           N["n_pe"]: n_pe, N["n_ro"]: n_ro, N["ro_os"]: ro_os, "n_coils": n_coil,
           N["phase_axis"]: scan["phase_axis"], N["coil_map"]: cmap, N["fov"]: scan["fov_mm"],
           N["centre"]: [cy, cx] if dialect["shift_order"] == "yx" else [cx, cy]}
    out[N["header"]] = np.array(json.dumps(hdr))
    return out, truth.astype(np.float32)


def scout(truth, factor=4):
    """Coarse localizer of the same slice: the truth band-limited to 1/factor of the matrix."""
    ny, nx = truth.shape
    k = sim.cfft(truth.astype(float), (0, 1))
    k = sim.crop_centre(sim.crop_centre(k, max(ny // factor, 8), 0), max(nx // factor, 8), 1)
    return np.abs(sim.cifft(k, (0, 1))).astype(np.float32)


# ------------------------------------------------------------------------------------------------ FORMAT.md

def format_doc(d):
    N = d["names"]
    axes = {"R": "record", "C": "coil", "S": "sample"}
    layout = ", ".join(axes[c] for c in d["layout"])
    cplx = ("complex64" if d["complex"] == "c64" else
            "float32 with one extra trailing axis of length 2 holding (real, imaginary)")
    sign = "-" if d["ft_sign"] == 1 else "+"
    ky = (f"`{N['ky']}[r]` is the phase-encode index ky of record r, a signed integer; ky = 0 is the centre of k-space."
          if d["ky_conv"] == "signed" else
          f"`{N['ky']}[r]` is the phase-encode line number of record r, counted from 0; "
          f"ky = `{N['ky']}[r]` - `{N['n_pe']}` // 2, so line `{N['n_pe']}` // 2 is the centre of k-space.")
    if d["kind_conv"] == "str":
        kind = (f"`{N['kind']}[r]` is a string: `img` (imaging data), `noise` (receiver noise, no excitation), "
                f"`nav` (navigator echo, acquired without phase encoding at a different echo time), "
                f"`dummy` (acquired before the steady state is reached).")
    else:
        kind = (f"`{N['kind']}[r]` is a bit mask (uint8): bit value {KIND_BITS['noise']} = receiver noise (no "
                f"excitation), {KIND_BITS['nav']} = navigator echo (acquired without phase encoding at a different "
                f"echo time), {KIND_BITS['dummy']} = acquired before the steady state is reached. "
                f"A record with no bit set is imaging data. No other bits are used.")
    rdir = (f"`{N['dir']}[r]` is +1 for a record sampled in the forward readout direction and -1 for a reversed one."
            if d["dir_conv"] == "pm1" else
            f"`{N['dir']}[r]` is true for a record sampled in the reversed readout direction, false for forward.")
    gain = (f"`{N['gain']}[r]` is the receiver gain of record r in dB (amplitude): the stored samples are the "
            f"received signal, noise included, multiplied by 10^(gain / 20)." if d["gain_conv"] == "db" else
            f"`{N['gain']}[r]` is the receiver gain of record r as a linear amplitude factor: the stored samples are "
            f"the received signal, noise included, multiplied by it.")
    first, second = ("y", "x") if d["shift_order"] == "yx" else ("x", "y")
    unit = {"mm": f"millimetres (the field of view is `{N['fov']}` mm in both directions)",
            "frac": "fractions of the field of view",
            "pix": "pixels of the reconstructed image (the pixel size differs between y and x when the matrix does)"}
    cmap = (f"data channel c was received by the coil whose sensitivity is map number `{N['coil_map']}[c]`"
            if d["chan_conv"] == "data_to_sens" else
            f"the coil whose sensitivity is map number j was recorded in data channel `{N['coil_map']}[j]`")
    sens_layout = "[coil, y, x]" if d["sens_layout"] == "CYX" else "[y, x, coil]"
    return f"""# Raw data format of this scanner (2D Cartesian gradient echo, multi-coil)

A scan is one `.npz` file. Everything below is a property of the scanner and holds for every scan it writes.

## Geometry and the image to reconstruct

Positions are measured in units of the field of view (FOV = 1, the same in y and x). On an axis of n points, index i
is at position (i - n // 2) / n, for images and for k-space alike (k-space index i holds integer frequency i - n // 2
in cycles per FOV).

The phase-encode (pe) direction is physical y and the readout (ro) direction is physical x when
`{N['phase_axis']}` is "y"; when it is "x", pe is physical x and ro is physical y.

The reconstructed image is a 2D magnitude array indexed [y, x]. Its shape is (`{N['n_pe']}`, `{N['n_ro']}`) when the
phase axis is "y" and (`{N['n_ro']}`, `{N['n_pe']}`) when it is "x". It is centred on the prescribed slice centre, and
it shows the object itself, with the coil sensitivities divided out. Its overall scale is arbitrary.

## Arrays

- `{N['data']}`: the raw samples, {cplx}. Axis order: {layout}. One record is one readout.
- {ky}
- {kind} Only imaging records belong in the image.
- {rdir}
- {gain}
- `{N['sens']}`: complex64 coil sensitivities from a prescan, layout {sens_layout}, on a coarse grid that covers the
  same FOV. They are in physical orientation (y, x), centred on the isocentre, whatever the phase axis of the scan.
- `{N['header']}`: a JSON string with the fields below.

## Header fields

- `{N['n_pe']}`, `{N['n_ro']}`: prescribed matrix along pe and ro.
- `{N['ro_os']}`: readout oversampling factor, an integer (1 or 2). A record has `{N['n_ro']}` x `{N['ro_os']}` samples spaced
  1 / `{N['ro_os']}` cycles per FOV, so they cover `{N['ro_os']}` times the prescribed FOV along ro, centred on the
  isocentre.
- `{N['phase_axis']}`: "y" or "x", see above.
- `{N['coil_map']}`: {cmap}.
- `{N['centre']}`: position of the prescribed slice centre relative to the isocentre, as [{first}, {second}], in
  {unit[d['shift_conv']]}. The scanner does not re-centre anything during acquisition: the raw data and the prescan
  are referenced to the isocentre, so the object appears displaced by this vector until the reconstruction moves the
  prescribed centre to the image centre. The offset is at most 0.1 FOV in each direction, and the object always lies
  inside the FOV covered by the prescan, so sensitivities are never needed outside the prescan grid.
- `{N['fov']}`: field of view in mm. `n_coils`, `sequence`, `field_T`, `te_ms`, `tr_ms`, `flip_deg`: informational.

## Signal model

With n_s the number of samples in a record, sample s of a forward record is the signal at
kx = (s - n_s // 2) / `{N['ro_os']}`; sample s of a reversed record is the signal at kx = -(s - n_s // 2) / `{N['ro_os']}`.
A record with phase-encode index ky measures

    S(kx, ky) = sum over the object of  rho(pe, ro) * c(pe, ro) * exp({sign} i 2 pi (ky * pe + kx * ro))

for each coil, where c is that coil's sensitivity and pe, ro are positions relative to the isocentre in FOV units.
ky runs from -(`{N['n_pe']}` // 2) to `{N['n_pe']}` - 1 - `{N['n_pe']}` // 2. When n_s is even, sample 0 of a reversed
record lies at kx = +(n_s // 2) / `{N['ro_os']}`, half the sampling period beyond the forward range; on the sampled grid
it is the same frequency as kx = -(n_s // 2) / `{N['ro_os']}` and may be used as such.

## Acquisition

- Records are stored in the order they were acquired. That order is not necessarily ascending in ky.
- A phase-encode line may be acquired more than once (signal averaging); repeated imaging records of one line measure
  the same signal with independent noise.
- Lines that were not acquired (parallel imaging) are simply absent.
- Receiver noise is independent between coils and has the same level in every coil.
"""


# ------------------------------------------------------------------------------------------------ recon.py

def recon_source(d, bugs=()):
    """Source text of the reconstruction for dialect ``d``. With ``bugs`` empty this is the reference."""
    b = set(bugs)
    unknown = b - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {unknown}")
    N = d["names"]
    L = []
    w = L.append
    w('"""Reconstruct one 2D multi-coil Cartesian scan.  Usage: python recon.py scan.npz image.npy"""')
    w("import json\nimport sys\n\nimport numpy as np\nfrom scipy.ndimage import map_coordinates\n\n")
    pre, post = ("fftshift", "ifftshift") if "fft_centre_odd" in b else ("ifftshift", "fftshift")
    for name, fn in (("cfft", "fftn"), ("cifft", "ifftn")):
        w(f"def {name}(a, axes):")
        w(f"    a = np.fft.{pre}(a, axes=axes)")
        w(f'    return np.fft.{post}(np.fft.{fn}(a, axes=axes, norm="ortho"), axes=axes)\n\n')

    # ---- load
    w("def load(path):")
    w("    f = np.load(path)")
    w(f'    hdr = json.loads(str(f["{N["header"]}"]))')
    w(f'    raw = f["{N["data"]}"]')
    if d["complex"] == "split":
        if "cplx_real_only" in b:
            w("    raw = raw[..., 0].astype(complex)")
        elif "cplx_imag_sign" in b:
            w("    raw = raw[..., 0] - 1j * raw[..., 1]")
        else:
            w("    raw = raw[..., 0] + 1j * raw[..., 1]")
    if d["layout"] == "CRS":
        w("    raw = raw.transpose(1, 0, 2)")
    elif d["layout"] == "RSC":
        w("    raw = raw.transpose(0, 2, 1)")
    w("    return f, hdr, raw            # raw is [record, coil, sample]\n\n")

    # ---- collect records into k-space lines
    sig = -d["ft_sign"] if "ft_sign" in b else d["ft_sign"]
    w("def collect(f, hdr, raw):")
    w(f'    n_pe = hdr["{N["n_pe"]}"]')
    w("    n_rec, n_coil, n_s = raw.shape")
    w(f'    ky = f["{N["ky"]}"].astype(int)' + ("" if d["ky_conv"] == "signed" else " - n_pe // 2"))
    if "kind_ignore" in b:
        w("    imaging = np.ones(n_rec, bool)")
    elif d["kind_conv"] == "str":
        w(f'    imaging = f["{N["kind"]}"] ' + ('!= "noise"' if "kind_partial" in b else '== "img"'))
    else:
        w(f'    imaging = ' + (f'(f["{N["kind"]}"] & {KIND_BITS["noise"]}) == 0' if "kind_partial" in b
                               else f'f["{N["kind"]}"] == 0'))
    if "rev_ignore" in b:
        w("    rev = np.zeros(n_rec, bool)")
    elif d["dir_conv"] == "pm1":
        w(f'    rev = f["{N["dir"]}"] < 0')
    else:
        w(f'    rev = f["{N["dir"]}"].astype(bool)')
    if "gain_ignore" in b:
        w("    gain = np.ones(n_rec)")
    elif d["gain_conv"] == "db":
        w(f'    gain = 10.0 ** (f["{N["gain"]}"] / {"10.0" if "gain_db_power" in b else "20.0"})')
    else:
        w(f'    gain = f["{N["gain"]}"].astype(float)')
    if "rev_plain_flip" in b:
        w("    back = np.arange(n_s)[::-1]")
    else:
        w("    back = (2 * (n_s // 2) - np.arange(n_s)) % n_s")
    w("    acc = np.zeros((n_coil, n_pe, n_s), complex)")
    w("    count = np.zeros(n_pe, int)")
    if "order_ignore" in b:
        w("    row = 0")
    w("    for r in np.flatnonzero(imaging):")
    w("        line = raw[r] " + ("*" if "gain_inverse" in b else "/") + " gain[r]")
    w("        if " + ("rev[r]" if sig == 1 else "not rev[r]") + ":")
    w("            line = line[:, back]")
    if "order_ignore" not in b:
        w("        row = (" + ("ky[r]" if sig == 1 else "-ky[r]") + " + n_pe // 2) % n_pe")
    w("        acc[:, row] += line")
    w("        count[row] += 1")
    if "order_ignore" in b:
        w("        row = (row + 1) % n_pe")
    w("    have = count > 0")
    if "avg_sum" not in b:
        w("    acc[:, have] /= count[have][None, :, None]")
    w("    return acc, have\n\n")

    # ---- readout oversampling
    w("def to_prescribed_fov(acc, hdr):")
    w(f'    n_ro = hdr["{N["n_ro"]}"]')
    w("    n_s = acc.shape[-1]")
    w("    hybrid = cifft(acc, (-1,))")
    w("    lo = " + ("0" if "os_crop_start" in b else "(n_s - n_ro) // 2" if "os_crop_off" in b
                     else "n_s // 2 - n_ro // 2"))
    w("    return cfft(hybrid[..., lo:lo + n_ro], (-1,))\n\n")

    # ---- coil maps
    w("def coil_maps(f, hdr, n_pe, n_ro):")
    w(f'    s = f["{N["sens"]}"]')
    if d["sens_layout"] == "YXC":
        w("    s = s.transpose(2, 0, 1)")
    if "chan_ignore" not in b:
        direct = (d["chan_conv"] == "data_to_sens") != ("chan_inverse" in b)
        w(f'    order = np.asarray(hdr["{N["coil_map"]}"])')
        w("    s = s[order]" if direct else "    s = s[np.argsort(order)]")
    if "pa_sens" not in b:
        w(f'    if hdr["{N["phase_axis"]}"] == "x":')
        w("        s = s.transpose(0, 2, 1)")
    w("    m_pe, m_ro = s.shape[1:]")
    w("    pe = (np.arange(n_pe) - n_pe // 2) / n_pe * m_pe + m_pe // 2")
    w("    ro = (np.arange(n_ro) - n_ro // 2) / n_ro * m_ro + m_ro // 2")
    w('    pts = np.meshgrid(pe, ro, indexing="ij")')
    w("    return np.array([map_coordinates(c.real, pts, order=3, mode=\"nearest\")")
    w("                     + 1j * map_coordinates(c.imag, pts, order=3, mode=\"nearest\") for c in s])\n\n")

    # ---- SENSE by conjugate gradients
    sc = "s" if "sens_noconj" in b else "np.conj(s)"
    w("def sense(k, have, s, lam=1e-3, n_iter=60):")
    w("    w = " + ("np.ones((1, k.shape[1], 1))" if "mask_full" in b else "have[None, :, None].astype(float)"))
    w("")
    w("    def normal(x):")
    w(f"        return np.sum({sc} * cifft(w * cfft(s * x, (1, 2)), (1, 2)), axis=0) + lam * x")
    w("")
    w(f"    b = np.sum({sc} * cifft(w * k, (1, 2)), axis=0)")
    w("    x = np.zeros_like(b)")
    w("    r = b.copy()")
    w("    p = r.copy()")
    w("    rs = np.vdot(r, r).real")
    w("    for _ in range(n_iter):")
    w("        q = normal(p)")
    w("        alpha = rs / np.vdot(p, q).real")
    w("        x += alpha * p")
    w("        r -= alpha * q")
    w("        rs_new = np.vdot(r, r).real")
    w("        if rs_new < 1e-14 * np.vdot(b, b).real:")
    w("            break")
    w("        p = r + (rs_new / rs) * p")
    w("        rs = rs_new")
    w("    return x\n\n")

    # ---- move the prescribed centre to the image centre, orient
    w("def finish(x, hdr):")
    w("    n_pe, n_ro = x.shape")
    w(f'    along_x = hdr["{N["phase_axis"]}"] == "x"')
    if "shift_ignore" not in b:
        a, c = ("cy", "cx") if d["shift_order"] == "yx" else ("cx", "cy")
        w(f'    {a}, {c} = hdr["{N["centre"]}"]')
        if d["shift_conv"] == "mm":
            if "shift_units" in b:
                w("    ny, nx = (n_ro, n_pe) if along_x else (n_pe, n_ro)")
                w("    cy, cx = cy / ny, cx / nx")
            else:
                w(f'    cy, cx = cy / hdr["{N["fov"]}"], cx / hdr["{N["fov"]}"]')
        elif d["shift_conv"] == "pix" and "shift_units" not in b:
            w("    ny, nx = (n_ro, n_pe) if along_x else (n_pe, n_ro)")
            w("    cy, cx = cy / ny, cx / nx")
        if "shift_swap" in b:
            w("    d_pe, d_ro = (cy, cx) if along_x else (cx, cy)")
        else:
            w("    d_pe, d_ro = (cx, cy) if along_x else (cy, cx)")
        w("    k_pe = np.arange(n_pe) - n_pe // 2")
        w("    k_ro = np.arange(n_ro) - n_ro // 2")
        w("    ramp = np.exp(" + ("-" if "shift_sign" in b else "") +
          "2j * np.pi * (k_pe[:, None] * d_pe + k_ro[None, :] * d_ro))")
        w("    x = cifft(cfft(x, (0, 1)) * ramp, (0, 1))")
    w("    img = np.abs(x)")
    if "pa_out" not in b:
        w("    if along_x:")
        w("        img = img.T")
    w("    return img\n\n")

    w("def main(src, dst):")
    w("    f, hdr, raw = load(src)")
    w("    acc, have = collect(f, hdr, raw)")
    w("    k = to_prescribed_fov(acc, hdr)")
    w("    s = coil_maps(f, hdr, k.shape[1], k.shape[2])")
    w("    x = sense(k, have, s)")
    w("    np.save(dst, finish(x, hdr).astype(np.float32))\n\n")
    w('if __name__ == "__main__":')
    w("    main(sys.argv[1], sys.argv[2])")
    return "\n".join(L) + "\n"
