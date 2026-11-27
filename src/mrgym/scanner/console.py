"""The console: protocol in; realised protocol, timing, scan time, limits and the header line out. Never any signal.

``plan(protocol)`` validates a request and returns the realised protocol (defaults filled, times on the raster,
minimum TE / TR implied by matrix, bandwidth and gradient limits), the scan time (per-scan overhead + dummy cycles +
acquisition), RF power and gradient use as percent of limit, and accept / reject with typed reasons. Nothing it
returns depends on the object: no signal, no noise level. The relative-SNR figure in the header is voxel volume times
the square root of the sampling time, a function of the protocol alone.

Partial Fourier. The lines that are not acquired are synthesised by homodyne detection (``kspace.Engine.recon``): the
resolution stays, the noise of a pixel rises. ``noise_lines`` is the number of lines a fully sampled scan would need
for the same noise; the relative SNR is computed with it, so it falls when lines are left out.

Timing model
  readout       duration 1 / bandwidth_hz_per_px; amplitude G = bandwidth * matrix / (gamma * FOV)
  lobes         prephaser (half the readout area) and spoiler: time-optimal trapezoids at ``derate`` of the gradient
                and slew limits. Phase encoding and slice selection are given no time of their own (they are taken
                to overlap the prephaser), and the prephaser area ignores the readout ramp.
  gradient echo TE >= rf / 2 + prephaser + ramp + readout / 2;  TR >= rf + last TE + readout / 2 + ramp + spoiler
  spin echo     TE >= 2 max(rf90 / 2 + prephaser + crusher + rf180 / 2, rf180 / 2 + crusher + ramp + readout / 2)
  bSSFP         TR >= rf + 2 (prephaser + ramp) + readout, TE = TR / 2
  EPI           echo spacing >= readout + 2 ramps; TE >= rf / 2 + prephaser + (lines before k = 0) * echo spacing
  scan time     overhead + dummy cycles * cycle + cycle * phase-encode lines * averages   (one line per cycle);
                EPI: overhead + TR * (averages + 2)
  RF power      B1rms^2 = sum over pulses of B1^2 * duration / cycle, as percent of (B1rms limit)^2. Computed from
                the NOMINAL flip angle, as a scanner does before it knows the transmit error, times ``tx_scale``
                squared (the amplitude the console itself was told to play).

Slice selection (``excitation: selective``, the default). The excitation is a windowed sinc of ``rf_pulse``: fast
(1 ms, time-bandwidth product 2), normal (2 ms, 4) or sharp (4 ms, 8); ``se`` and ``ir`` play the 90 degree pulse
with that duration and the 180 degree pulse 1.5 times as long, on a slice 1.5 times as thick.
  slice gradient  G_ss = k * tbw / (duration * gamma * slice thickness), k = ``rf.fwhm_factor`` (1.197 / 0.994 / 1.000):
                  calibrated per pulse so that the full width at half maximum of the small-flip-angle profile is
                  the prescribed thickness. Refused (SLICE_TOO_THIN) above ``derate`` of the gradient limit. Its
                  rephasing lobe, of area G_ss * (duration + ramp) / 2, overlaps the prephaser: the longer of the two
                  counts in the minimum TE; the minimum TR carries the pulse and its ramp
  peak B1         peak factor of the shape * alpha / (2 pi gamma duration); the pulse is lengthened (to the raster)
                  until this is at most ``b1_peak_ut``
  RF power        B1^2 * duration of the hard pulse of the same duration and angle, times the energy factor of the
                  shape (1.80 / 3.11 / 7.16 for time-bandwidth 2 / 4 / 8)
Fat saturation: an 8 ms pulse that ends ``fatsat_gap_ms`` before the excitation pulse starts; the saturation acts at
its centre, ``fatsat_delay_ms`` = fatsat_ms / 2 + fatsat_gap_ms + rf / 2 before the centre of the excitation.

Adjustments (every sequence; they cost no time). ``cf_shift_hz`` raises the centre frequency of the scanner,
``shim_y_hz_per_mm`` / ``shim_x_hz_per_mm`` add linear field terms, ``tx_scale`` multiplies the transmit amplitude of
the excitation and refocusing pulses (their peak B1 and RF power follow; the adiabatic inversion and the
fat-saturation pulse are not scaled). Their defaults change nothing.

``excitation: nonselective`` is a 3D acquisition with hard pulses: every voxel sees one flip angle, the acquisition
time is multiplied by ``partitions = ceil(slab / slice_mm)`` (``slab_mm``, the depth of the object, a constant) and
the noise falls with the square root of it. The returned image is the central partition; slab wrap is not modelled.
"""
from __future__ import annotations

import math

from . import rf as RF

GAMMA_HZ_PER_T = 42.577478518e6
FAT_PPM = 3.4
SYSTEM = dict(
    g_max_mt_m=40.0, slew_t_m_s=150.0, derate=0.8, raster_ms=0.01, dead_ms=0.05,
    rf_ms=1.0, rf_se90_ms=2.0, rf_180_ms=3.0, rf_180_scale=1.5, b1_peak_ut=20.0, slab_mm=160.0, inv_ms=8.0, inv_b1_ut=12.0, crusher_ms=1.0, se_spoil_ms=1.0,
    fatsat_ms=8.0, fatsat_b1_ut=1.2, fatsat_gap_ms=2.0,
    overhead_s=3.0, dummy_ms=4000.0, prepared_dummy_cycles=1, epi_dummy_shots=2, ro_os=2,
    spgr_phase_increment_deg=117.0, afi_phase_increment_deg=129.3,
    snr_ref=dict(voxel_mm3=5.0, lines=256, bandwidth_hz_per_px=150.0),
)

# name: (low, high, kind, default)
#   kind f = float, i = integer, c = one of the values in ``low``, b = boolean
#   default None = required, "min" = console minimum, "seq" = sequence default, "exam" = default of the exam
_GEOM = dict(
    fov_mm=(80.0, 500.0, "f", "exam"), phase_fov=(0.25, 1.0, "f", 1.0), matrix=(32, 512, "i", 128),
    slice_mm=(1.0, 10.0, "f", 5.0), bandwidth_hz_per_px=(50.0, 3000.0, "f", 300.0), pe_dir=(("y", "x"), None, "c", "y"),
    ro_polarity=((1, -1), None, "c", 1), partial_fourier=(0.55, 1.0, "f", 1.0), phase_oversampling=(0.0, 1.0, "f", 0.0),
    averages=(1, 64, "i", 1), centre_y_mm=(-150.0, 150.0, "f", 0.0), centre_x_mm=(-150.0, 150.0, "f", 0.0),
    rotation_deg=(-90.0, 90.0, "f", 0.0), rf_pulse=(tuple(RF.PULSES), None, "c", "normal"),
    excitation=(("selective", "nonselective"), None, "c", "selective"),
    cf_shift_hz=(-500.0, 500.0, "f", 0.0), tx_scale=(0.5, 2.0, "f", 1.0),
    shim_y_hz_per_mm=(-2.0, 2.0, "f", 0.0), shim_x_hz_per_mm=(-2.0, 2.0, "f", 0.0),
)
ADJUSTMENTS = dict(cf_shift_hz=0.0, tx_scale=1.0, shim_y_hz_per_mm=0.0, shim_x_hz_per_mm=0.0)      # name: the value that changes nothing
_TE = dict(te_ms=(0.5, 300.0, "f", "min"))
_FS = dict(fat_sat=(None, None, "b", False), fatsat_offset_hz=(-300.0, 300.0, "f", 0.0))
_GRE = dict(flip_deg=(1.0, 90.0, "f", None), spoiler_mt_ms_per_m=(10.0, 600.0, "f", 100.0),
            rf_phase_increment_deg=(0.0, 180.0, "f", "seq"))
SEQUENCES = dict(
    spgr={**_GEOM, **_TE, **_FS, **_GRE, "tr_ms": (3.0, 5000.0, "f", None), "n_echoes": (1, 8, "i", 1),
          "echo_spacing_ms": (0.5, 50.0, "f", "min")},
    afi={**_GEOM, **_TE, **_GRE, "tr_ms": (5.0, 200.0, "f", None), "tr_ratio": (2, 10, "i", 5),
         "spoiler_mt_ms_per_m": (10.0, 600.0, "f", 300.0)},
    se={**_GEOM, **_TE, **_FS, "tr_ms": (50.0, 20000.0, "f", None)},
    ir={**_GEOM, **_TE, **_FS, "tr_ms": (200.0, 20000.0, "f", None), "ti_ms": (5.0, 19000.0, "f", None)},
    bssfp={**_GEOM, "tr_ms": (2.0, 50.0, "f", None), "flip_deg": (1.0, 90.0, "f", None),
           "phase_cycle_deg": (0.0, 360.0, "f", 180.0)},
    epi={**_GEOM, **_TE, "matrix": (32, 192, "i", 96), "bandwidth_hz_per_px": (500.0, 3000.0, "f", 1500.0),
         "tr_ms": (50.0, 20000.0, "f", None),
         "flip_deg": (1.0, 90.0, "f", 90.0), "echo_spacing_ms": (0.2, 5.0, "f", "min"),
         "pe_polarity": ((1, -1), None, "c", 1), "fat_sat": (None, None, "b", True),
         "fatsat_offset_hz": _FS["fatsat_offset_hz"]},
)
PARAMETER_UNITS = dict(
    fov_mm="mm, along the readout direction", phase_fov="fraction of fov_mm along the phase-encode direction",
    matrix="pixels along the readout (even); phase-encode lines follow from phase_fov", slice_mm="mm",
    bandwidth_hz_per_px="Hz per pixel", pe_dir="phase-encode axis: y (rows) or x (columns)",
    ro_polarity="direction of the readout: with 1 a spin off resonance by df appears -df / bandwidth pixels along the readout",
    partial_fourier="fraction of phase-encode lines acquired; the others are synthesised (homodyne detection)",
    phase_oversampling="fraction of extra phase field of view, acquired and discarded", averages="count",
    centre_y_mm="mm, position of the centre of the field of view along y (off-centre shift; 0 = isocentre)",
    centre_x_mm="mm, position of the centre of the field of view along x (off-centre shift; 0 = isocentre)",
    rotation_deg="degrees, in-plane rotation of the field of view about its centre: image rows turned from +y towards +x",
    rf_pulse="excitation pulse: fast (1 ms, time-bandwidth 2), normal (2 ms, 4), sharp (4 ms, 8); the sharper, the "
             "closer the slice profile to a rectangle, the longer the pulse and the higher its RF energy",
    excitation="selective: 2D slice with the profile of rf_pulse; nonselective: hard pulses and 3D encoding, the "
               "acquisition time multiplied by the number of partitions",
    fatsat_offset_hz="Hz, shift of the centre frequency of the fat-saturation pulse from the nominal fat line",
    fat_sat="spectral fat saturation before every excitation", te_ms="ms", flip_deg="degrees (nominal)",
    spoiler_mt_ms_per_m="mT*ms/m per repetition", rf_phase_increment_deg="degrees", tr_ms="ms", n_echoes="count",
    echo_spacing_ms="ms", tr_ratio="integer TR2 / TR1", ti_ms="ms, centre of the inversion to centre of the excitation",
    phase_cycle_deg="degrees of RF phase advance per TR", pe_polarity="sign of the phase-encode blips",
    cf_shift_hz="Hz added to the scanner's centre frequency for this scan: a spin measured df off resonance is df - cf_shift_hz off",
    tx_scale="factor on the transmit amplitude of this scan: every excitation and refocusing angle is tx_scale x nominal x local transmit factor",
    shim_y_hz_per_mm="Hz per mm along y, a linear field added for this scan: the off-resonance at (y, x) changes by shim_y_hz_per_mm x y",
    shim_x_hz_per_mm="Hz per mm along x, a linear field added for this scan: the off-resonance at (y, x) changes by shim_x_hz_per_mm x x",
)
DESCRIPTIONS = dict(
    spgr="RF-spoiled gradient echo, 1 to 8 echoes: steady state with quadratic RF phase cycling and a gradient "
         "spoiler of the requested moment in every repetition",
    afi="actual flip-angle imaging: dual-TR RF-spoiled gradient echo, TR2 = tr_ratio * tr_ms, spoiler moments and RF "
        "phase increments in the ratio TR1 : TR2; image 0 is acquired in TR1, image 1 in TR2",
    se="spin echo: 90 degree excitation, 180 degree refocusing (both nominal; both scale with the transmit field), "
       "one phase-encode line per tr_ms",
    ir="inversion recovery: adiabatic inversion, delay ti_ms, spin-echo readout; signed signal (a negative signal is "
       "pi away in phase from a positive one of the same region; a phase value alone gives no sign)",
    bssfp="balanced steady-state free precession, echo at tr_ms / 2, RF phase advanced by phase_cycle_deg per TR",
    epi="single-shot gradient-echo echo-planar imaging: all phase-encode lines after one excitation, "
        "echo_spacing_ms apart; spectral fat saturation on by default",
)

# ------------------------------------------------------------------ tables of the agent-facing schema
# What a short word means, for the schema descriptions (the unit texts above are for the datasheet tables).
PARAMETER_MEANING = dict(
    tr_ms="repetition time", te_ms="echo time", flip_deg="nominal flip angle", ti_ms="inversion time",
    fov_mm="readout field of view", phase_fov="phase field of view as a fraction of fov_mm (0.75 = 75 %)",
    matrix="readout pixels, even; pixel = fov_mm / matrix", slice_mm="slice thickness",
    bandwidth_hz_per_px="receiver bandwidth per pixel", pe_dir="phase-encode axis: y = image rows, x = columns",
    averages="number of averages", phase_oversampling="extra phase field of view as a fraction (suppresses wrap-around)",
    partial_fourier="fraction of the phase-encode lines acquired", fat_sat="spectral fat saturation", n_echoes="number of echoes",
    echo_spacing_ms="time between echoes", spoiler_mt_ms_per_m="gradient spoiler moment per TR",
    rf_phase_increment_deg="RF spoiling phase increment", tr_ratio="TR2 / TR1", phase_cycle_deg="RF phase advance per TR",
    ro_polarity="direction of the readout", pe_polarity="sign of the phase-encode blips",
    centre_y_mm="centre of the field of view along y, from the isocentre", centre_x_mm="centre of the field of view along x, from the isocentre",
    rotation_deg="in-plane rotation of the field of view (image rows turned from +y towards +x)",
    rf_pulse="excitation pulse: fast (1 ms), normal (2 ms), sharp (4 ms): squarer slice profile, longer pulse, more RF energy",
    excitation="selective: one 2D slice; nonselective: hard pulses, 3D encoding, scan time x partitions",
    fatsat_offset_hz="frequency shift of the fat-saturation pulse",
    cf_shift_hz="centre-frequency adjustment: off-resonance everywhere falls by it",
    tx_scale="transmit adjustment: factor on every excitation and refocusing angle",
    shim_y_hz_per_mm="linear shim along y: adds shim_y_hz_per_mm x y to the off-resonance",
    shim_x_hz_per_mm="linear shim along x: adds shim_x_hz_per_mm x x to the off-resonance",
)
# One name per sequence for the schema (the full descriptions are in DESCRIPTIONS and behind scanner_info).
SEQUENCE_NAMES = dict(spgr="RF-spoiled gradient echo", afi="actual flip-angle imaging (dual TR)", se="spin echo",
                      ir="inversion recovery", bssfp="balanced SSFP", epi="single-shot gradient-echo EPI")
# Unit written after a number in a description ("" = none).
PARAMETER_UNIT = dict(
    tr_ms="ms", te_ms="ms", flip_deg="degrees", ti_ms="ms", fov_mm="mm", slice_mm="mm", bandwidth_hz_per_px="Hz per pixel",
    echo_spacing_ms="ms", spoiler_mt_ms_per_m="mT*ms/m", rf_phase_increment_deg="degrees", phase_cycle_deg="degrees",
    centre_y_mm="mm", centre_x_mm="mm", rotation_deg="degrees", fatsat_offset_hz="Hz", cf_shift_hz="Hz",
    shim_y_hz_per_mm="Hz per mm", shim_x_hz_per_mm="Hz per mm")
# Key order of the schema: what a protocol is usually written in. Parameters not listed here follow in sorted order.
PARAMETER_ORDER = ("tr_ms", "te_ms", "flip_deg", "ti_ms", "fov_mm", "phase_fov", "matrix", "slice_mm", "bandwidth_hz_per_px",
                   "pe_dir", "averages", "phase_oversampling", "partial_fourier", "fat_sat", "fatsat_offset_hz", "n_echoes",
                   "echo_spacing_ms", "spoiler_mt_ms_per_m", "rf_phase_increment_deg", "tr_ratio", "phase_cycle_deg", "ro_polarity",
                   "pe_polarity", "centre_y_mm", "centre_x_mm", "rotation_deg", "rf_pulse", "excitation",
                   "cf_shift_hz", "tx_scale", "shim_y_hz_per_mm", "shim_x_hz_per_mm")
# Times whose lower bound depends on the rest of the protocol: the schema states no minimum for them (a model copies
# a schema minimum into the call), ``plan`` prints the one that applies.
PROTOCOL_DEPENDENT = ("tr_ms", "te_ms", "ti_ms", "echo_spacing_ms")
MIN = "min"                                 # the literal for "the shortest the console can realise"
# What the adjustments before a scan depend on (centre frequency, transmit gain, shim): the sequence family and where
# the field of view lies. A console that states ``repeat_overhead_s`` (``exam["system"]``) charges its full
# ``overhead_s`` only when one of these differs from the previous scan; timing, matrix and averages are not among them,
# and neither are the adjustment values a protocol sets by hand (``ADJUSTMENTS``: they cost nothing).
ADJUSTMENT_FIELDS = ("sequence", "fov_mm", "phase_fov", "slice_mm", "pe_dir", "centre_y_mm", "centre_x_mm", "rotation_deg", "excitation")


def adjustment_key(realised):
    return tuple(realised.get(k) for k in ADJUSTMENT_FIELDS)
# Other spellings of a parameter that are accepted (logged, and named in a note): the long forms with the unit.
ALIASES = dict(flip_angle_deg="flip_deg", slice_thickness_mm="slice_mm", phase_fov_frac="phase_fov",
               phase_oversampling_frac="phase_oversampling", partial_fourier_frac="partial_fourier",
               fov_rotation_deg="rotation_deg", fat_sat_offset_hz="fatsat_offset_hz")
# Console words of the vendors for the same parameter, in the same unit (a rename, never a conversion): NEX (GE), NSA
# (Philips), the bare TR / TE / TI of every console (milliseconds there too), the DICOM name of the bandwidth per pixel.
ALIASES.update(nex="averages", nsa="averages", num_averages="averages", number_of_averages="averages",
               tr="tr_ms", te="te_ms", ti="ti_ms", repetition_time_ms="tr_ms", echo_time_ms="te_ms", inversion_time_ms="ti_ms",
               flip_angle="flip_deg", slice_thickness="slice_mm", pixel_bandwidth="bandwidth_hz_per_px",
               pixel_bandwidth_hz="bandwidth_hz_per_px", phase_encoding_direction="pe_dir", phase_enc_dir="pe_dir",
               fold_over_direction="pe_dir", fat_suppression="fat_sat", fat_saturation="fat_sat")
# Console words that mean a parameter of this scanner in ANOTHER unit or coding: never applied, answered with the
# conversion. name: (parameter here, what the word is, the conversion in words, callable(value, protocol) -> value here
# or None when the protocol does not carry what the conversion needs).
VENDOR_TERMS = dict(
    rbw_khz=("bandwidth_hz_per_px", "the receiver bandwidth as +- kHz over the whole readout",
             "bandwidth_hz_per_px = 2000 x rbw_khz / matrix",
             lambda v, p: round(2000.0 * v / p["matrix"], 3) if isinstance(p.get("matrix"), int) and not isinstance(p.get("matrix"), bool) else None),
    fov_phase_percent=("phase_fov", "the phase field of view in percent", "phase_fov = fov_phase_percent / 100", lambda v, p: round(v / 100.0, 6)),
    rfov_percent=("phase_fov", "the rectangular field of view in percent", "phase_fov = rfov_percent / 100", lambda v, p: round(v / 100.0, 6)),
    no_phase_wrap=("phase_oversampling", "phase oversampling as a switch", "phase_oversampling = 1.0 (twice the phase field of view is acquired)",
                   lambda v, p: 1.0 if v is True else None),
    fold_over_suppression=("phase_oversampling", "phase oversampling as a switch", "phase_oversampling = 1.0 (twice the phase field of view is acquired)",
                           lambda v, p: 1.0 if v is True else None),
    wfs_px=("bandwidth_hz_per_px", "the bandwidth as the water-fat shift in pixels",
            "bandwidth_hz_per_px = 3.4 ppm x 42.58 MHz/T x B0 / wfs_px (217 Hz / wfs_px at 1.5 T, 434 Hz / wfs_px at 3 T)", lambda v, p: None),
)
# Parameters of real consoles that this scanner does not model. They are refused with the code NOT_SIMULATED (and the
# reason), not with UNKNOWN_PARAMETER: the word is right, the instrument lacks it. name: (what it is, why it is not here);
# the reasons are the sections of docs/physics/limits.md.
_NO_FSE = "there is no fast (turbo) spin echo: every echo has its own excitation (limits.md, section 8)"
_ONE_SLICE = "a scan is one 2D slice at the isocentre (limits.md, section 4)"
_ONE_CHANNEL = "there is one receive channel (limits.md, section 5)"
_NO_FLOW = "nothing flows and nothing is gated: motion is a periodic modulation the console cannot synchronise with (limits.md, section 7)"
NOT_SIMULATED = dict(
    etl=("echo train length", _NO_FSE), echo_train_length=("echo train length", _NO_FSE), turbo_factor=("turbo factor", _NO_FSE),
    effective_te_ms=("effective echo time", _NO_FSE), te_eff_ms=("effective echo time", _NO_FSE),
    slices=("number of slices", _ONE_SLICE), number_of_slices=("number of slices", _ONE_SLICE), n_slices=("number of slices", _ONE_SLICE),
    slice_gap_mm=("slice gap", _ONE_SLICE), slice_gap=("slice gap", _ONE_SLICE), slice_order=("slice order", _ONE_SLICE),
    slice_position_mm=("slice position", _ONE_SLICE), concatenations=("concatenations", _ONE_SLICE),
    sense=("parallel imaging", _ONE_CHANNEL), grappa=("parallel imaging", _ONE_CHANNEL), ipat=("parallel imaging", _ONE_CHANNEL),
    asset=("parallel imaging", _ONE_CHANNEL), acceleration=("parallel imaging factor", _ONE_CHANNEL), coil=("coil selection", _ONE_CHANNEL),
    b_value=("diffusion weighting", "there is no diffusion-weighted sequence (limits.md, section 3)"),
    venc=("velocity encoding", _NO_FLOW), flow_comp=("gradient moment nulling", _NO_FLOW), gradient_moment_nulling=("gradient moment nulling", _NO_FLOW),
    gating=("physiologic gating", _NO_FLOW), trigger=("physiologic triggering", _NO_FLOW),
    sat_band=("spatial saturation band", "there is no spatial saturation pulse (limits.md, section 4)"),
    saturation_band=("spatial saturation band", "there is no spatial saturation pulse (limits.md, section 4)"),
    shim=("shim", "there is no automatic shim and no term above first order: the linear terms are the protocol parameters "
                  "shim_y_hz_per_mm and shim_x_hz_per_mm, the centre frequency is cf_shift_hz (limits.md, section 10)"),
    filter=("image filter", "the reconstruction is an unfiltered inverse FFT of one channel (limits.md, section 5)"),
    intensity_correction=("intensity (uniformity) correction", "the reconstruction is an unfiltered inverse FFT of one channel (limits.md, section 5)"),
    distortion_correction=("gradient distortion correction", "the reconstruction applies no correction (limits.md, section 10)"),
)
# Values the console fixes for a sequence. Sending the fixed value is accepted and dropped; any other value is refused.
FIXED = dict(se=dict(flip_deg=90.0), ir=dict(flip_deg=90.0))


def not_simulated_message(name: str) -> str:
    what, why = NOT_SIMULATED[name]
    return f"{name} ({what}) is a parameter of real consoles that this scanner does not simulate: {why}"


def vendor_term(name: str, value, protocol: dict):
    """(message, parameter here, converted value or None) for a word of ``VENDOR_TERMS``. The conversion is stated,
    never applied."""
    target, what, rule, convert = VENDOR_TERMS[name]
    try:
        new = convert(value, protocol) if isinstance(value, (bool, int, float)) else None
    except (TypeError, ValueError, ZeroDivisionError):
        new = None
    text = f"{name} is {what}; this console takes {target}: {rule}" + (f" = {new:g}" if new is not None and not isinstance(new, bool) else "")
    return text + " (a unit is never converted for you)", target, new


def parameter_names():
    """Every protocol parameter of any sequence, in schema order."""
    names = {k for spec in SEQUENCES.values() for k in spec}
    return [k for k in PARAMETER_ORDER if k in names] + sorted(names - set(PARAMETER_ORDER))


def accepts_min(name):
    """Whether ``"min"`` may be written for this parameter (its console default is the console's minimum)."""
    return any(spec[name][3] == "min" for spec in SEQUENCES.values() if name in spec)


def default_text(sequence, name, exam=None):
    """The default of a parameter of a sequence as a value a call could carry, or None when it is required."""
    d = SEQUENCES[sequence][name][3]
    if d is None:
        return None
    if d == "min":
        return f'"{MIN}"'
    if d == "seq":
        return f"{SYSTEM[f'{sequence}_phase_increment_deg']:g}"
    if d == "exam":
        return f"{(exam or {}).get(name, 240.0):g}"
    return str(d).lower() if isinstance(d, bool) else f"{d:g}" if isinstance(d, float) else str(d)


def _lobe_ms(area_t_s_per_m, s):
    """Duration of a time-optimal trapezoid of the given area (T*s/m) at the derated limits."""
    g, sl = s["g_max_mt_m"] * 1e-3 * s["derate"], s["slew_t_m_s"] * s["derate"]
    if area_t_s_per_m <= g * g / sl:
        return 2e3 * math.sqrt(area_t_s_per_m / sl)
    return 1e3 * (area_t_s_per_m / g + g / sl)


def _up(x, raster):
    return round(math.ceil(x / raster - 1e-9) * raster, 6)


def _b1_ut(flip_deg, dur_ms):
    """Amplitude of a hard pulse: alpha / (2 pi gamma tau), in microtesla."""
    return math.radians(flip_deg) / (2 * math.pi * GAMMA_HZ_PER_T * dur_ms * 1e-3) * 1e6


def noise_lines(lines_full, lines):
    """Number of fully sampled lines that give the pixel noise of this acquisition: the sum of the squared weights of
    the reconstruction (``kspace.Engine.line_weights``). Fully sampled: the number of lines. Partial Fourier with
    n = lines - lines_full / 2: 2 n + 1 lines of weight 1 and lines_full / 2 - 1 - n lines of weight 2."""
    if lines >= lines_full:
        return lines_full
    n = lines - lines_full // 2
    return (2 * n + 1) + 4 * (lines_full // 2 - 1 - n)


def _violation(code, parameter, message, **extra):
    return dict(code=code, parameter=parameter, message=message, **extra)


def validate(protocol, exam=None):
    """Return (sequence, parameter dict with defaults, violations). Unknown keys are errors, never ignored.

    A violation is ``{"code", "parameter", "message"}`` plus what a repair needs: ``requested``, ``low`` / ``high``,
    ``nearest_valid`` (for an odd matrix the two even neighbours), ``allowed`` (the parameters of the sequence) and,
    for a parameter that belongs to another sequence, ``applies_to``. ``"min"`` may be written for a parameter whose
    default is the console minimum (``te_ms``, ``echo_spacing_ms``); it means the same as leaving it out.
    """
    if not isinstance(protocol, dict):
        return None, None, [_violation("INVALID_REQUEST", None, "protocol must be a JSON object")]
    seq = protocol.get("sequence")
    if not isinstance(seq, str) or seq not in SEQUENCES:
        return None, None, [_violation("UNKNOWN_SEQUENCE", "sequence", f"sequence must be one of {sorted(SEQUENCES)}",
                                       allowed=sorted(SEQUENCES))]
    spec, p, bad = SEQUENCES[seq], {}, []
    for name in sorted((k for k in protocol if k not in spec and k != "sequence"), key=str):
        used = sorted(q for q, other in SEQUENCES.items() if name in other)
        if used:
            bad.append(_violation("UNKNOWN_PARAMETER", name, f"{seq} has no parameter {name} (it belongs to {', '.join(used)}); "
                                  f"allowed: {sorted(spec)}", allowed=sorted(spec), applies_to=used))
        elif isinstance(name, str) and name in NOT_SIMULATED:
            bad.append(_violation("NOT_SIMULATED", name, not_simulated_message(name), allowed=sorted(spec)))
        else:
            bad.append(_violation("UNKNOWN_PARAMETER", name if isinstance(name, str) else None,
                                  f"{seq} has no parameter {str(name)[:40]}; allowed: {sorted(spec)}", allowed=sorted(spec)))
    for name, (lo, hi, kind, default) in spec.items():
        v = protocol.get(name, default)
        if v is None:
            bad.append(_violation("MISSING_PARAMETER", name, f"{seq} needs {name} [{PARAMETER_UNITS[name]}]"))
            continue
        if default == MIN and isinstance(v, str) and v == MIN:
            p[name] = MIN
            continue
        if name not in protocol and v in ("min", "seq", "exam"):
            p[name] = (exam or {}).get(name, 240.0) if v == "exam" else v
            continue
        if kind == "b":
            if not isinstance(v, bool):
                bad.append(_violation("OUT_OF_RANGE", name, f"{name} must be true or false", allowed=[True, False]))
                continue
            p[name] = v
            continue
        if kind == "c":
            if isinstance(v, (bool, list, dict)) or v not in lo:
                bad.append(_violation("OUT_OF_RANGE", name, f"{name} must be one of {list(lo)}", allowed=list(lo)))
                continue
            p[name] = v
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            bad.append(_violation("OUT_OF_RANGE", name, f"{name} must be a finite number"
                                  + (f' or "{MIN}"' if default == MIN else "")))
            continue
        if kind == "i" and int(v) != v:
            bad.append(_violation("OUT_OF_RANGE", name, f"{name} must be an integer", requested=v,
                                  nearest_valid=[math.floor(v), math.ceil(v)]))
            continue
        v = int(v) if kind == "i" else float(v)
        if not lo <= v <= hi:
            bad.append(_violation("OUT_OF_RANGE", name, f"{name}={v} outside [{lo}, {hi}] {PARAMETER_UNITS[name]}",
                                  requested=v, low=lo, high=hi, nearest_valid=lo if v < lo else hi))
            continue
        p[name] = v
    if "matrix" in p and p["matrix"] % 2:
        lo, hi = spec["matrix"][:2]
        bad.append(_violation("OUT_OF_RANGE", "matrix", "matrix must be even", requested=p["matrix"],
                              nearest_valid=[m for m in (p["matrix"] - 1, p["matrix"] + 1) if lo <= m <= hi]))
    return seq, p, bad


def plan(protocol, exam=None, system=None):
    """Dry run. ``exam``: the session's console card (``fov_mm`` default, ``min_pixel_mm``, ``b1rms_limit_ut``,
    ``b0_t``). Returns a JSON-serialisable dict; ``accept`` False comes with ``violations`` (code, parameter, message).
    """
    exam = dict(dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0), **(exam or {}))
    s = dict(SYSTEM, **(exam.get("system") or {}), **(system or {}))    # exam["system"]: this console's own constants (world spec ``console_system``)
    seq, p, bad = validate(protocol, exam)
    out = dict(accept=False, sequence=seq, violations=bad)
    if bad:
        return out
    ras, b0_t = s["raster_ms"], exam["b0_t"]
    n, fov, bw = p["matrix"], p["fov_mm"], p["bandwidth_hz_per_px"]
    n_pe = max(2, int(round(n * p["phase_fov"] / 2) * 2))
    fov_pe = fov * n_pe / n                                              # phase pixel = readout pixel
    lines_full = int(round(n_pe * (1 + p["phase_oversampling"]) / 2) * 2)
    lines = int(math.ceil(lines_full * p["partial_fourier"]))
    px = fov / n
    if px < exam["min_pixel_mm"] - 1e-9:
        bad.append(_violation("RESOLUTION_LIMIT", "matrix", f"pixel size {px:.3f} mm is below the scanner's minimum "
                              f"{exam['min_pixel_mm']} mm; lower the matrix or enlarge the field of view",
                              minimum_mm=exam["min_pixel_mm"]))
    fov_m = fov * 1e-3
    t_ro = 1e3 / bw
    g_ro = bw * n / (GAMMA_HZ_PER_T * fov_m)                             # T/m
    ramp = 1e3 * g_ro / (s["slew_t_m_s"] * s["derate"])
    pre = _lobe_ms(n / (2 * GAMMA_HZ_PER_T * fov_m), s)
    fat_sat = bool(p.get("fat_sat", False))
    fs_ms = s["fatsat_ms"] + s["fatsat_gap_ms"] if fat_sat else 0.0
    # ---- excitation: duration, slice gradient and its rephasing lobe, energy relative to a hard pulse
    selective = p["excitation"] == "selective"
    spin_echo = seq in ("se", "ir")
    tx = p["tx_scale"]                                                   # the amplitude the console plays: nominal x tx_scale
    flip = (90.0 if spin_echo else p["flip_deg"]) * tx
    g_lim, slew = s["g_max_mt_m"] * 1e-3 * s["derate"], s["slew_t_m_s"] * s["derate"]
    if selective:
        rf_ms, tbw = RF.PULSES[p["rf_pulse"]]
        rf_ms = max(rf_ms, _up(RF.peak_factor(tbw) * _b1_ut(flip, 1.0) / s["b1_peak_ut"], ras))
        rf180_ms = max(s["rf_180_scale"] * rf_ms, _up(RF.peak_factor(tbw) * _b1_ut(180.0 * tx, 1.0) / s["b1_peak_ut"], ras))
        energy = RF.energy_factor(tbw)
        cal = RF.fwhm_factor(tbw)                                        # the gradient that makes the profile's FWHM the prescribed thickness
        g_ss = cal * tbw / (rf_ms * 1e-3 * GAMMA_HZ_PER_T * p["slice_mm"] * 1e-3)        # T/m
        if g_ss > g_lim * (1 + 1e-9):
            thin = cal * tbw / (rf_ms * 1e-3 * GAMMA_HZ_PER_T * g_lim) * 1e3
            bad.append(_violation("SLICE_TOO_THIN", "slice_mm", f"the slice gradient of this pulse would be {g_ss * 1e3:.1f} mT/m; "
                                  f"the thinnest slice is {thin:.2f} mm (or use excitation=nonselective)", minimum_mm=round(thin, 3)))
        ramp_ss = 1e3 * g_ss / slew
        pre = max(pre, _lobe_ms(g_ss * (rf_ms + ramp_ss) / 2 * 1e-3, s))                 # slice rephaser overlaps the prephaser
        partitions = 1
    else:
        rf_ms, rf180_ms, tbw, energy, g_ss, ramp_ss = (s["rf_se90_ms"] if spin_echo else s["rf_ms"]), s["rf_180_ms"], None, 1.0, 0.0, 0.0
        partitions = int(math.ceil(s["slab_mm"] / p["slice_mm"] - 1e-9))
    rf_tr = rf_ms + ramp_ss                                               # what the pulse adds to a repetition
    r = dict(sequence=seq, fov_mm=fov, phase_fov=p["phase_fov"], fov_pe_mm=round(fov_pe, 6), matrix=n, matrix_pe=n_pe, lines_full=lines_full,
             lines=lines, slice_mm=p["slice_mm"], bandwidth_hz_per_px=bw, pe_dir=p["pe_dir"], ro_polarity=p["ro_polarity"],
             partial_fourier=p["partial_fourier"], phase_oversampling=p["phase_oversampling"], averages=p["averages"],
             fat_sat=fat_sat, fatsat_offset_hz=p.get("fatsat_offset_hz", 0.0), centre_y_mm=p["centre_y_mm"], centre_x_mm=p["centre_x_mm"],
             rotation_deg=p["rotation_deg"], rf_pulse=p["rf_pulse"], excitation=p["excitation"], rf_ms=round(rf_ms, 6), rf_tbw=tbw,
             partitions=partitions, cf_shift_hz=p["cf_shift_hz"], tx_scale=tx, shim_y_hz_per_mm=p["shim_y_hz_per_mm"],
             shim_x_hz_per_mm=p["shim_x_hz_per_mm"])
    if fat_sat:                                                          # centre of the saturation pulse to centre of the excitation
        r["fatsat_delay_ms"] = round(s["fatsat_ms"] / 2 + s["fatsat_gap_ms"] + rf_ms / 2, 6)
    timing = dict(readout_ms=round(t_ro, 4), rf_ms=round(rf_ms, 6), slice_gradient_mt_m=round(g_ss * 1e3, 3))
    esp = None
    n_echo = 1

    if seq in ("spgr", "afi"):
        te_min = _up(rf_ms / 2 + pre + ramp + t_ro / 2 + s["dead_ms"], ras)
        te = te_min if p["te_ms"] == "min" else _up(p["te_ms"], ras)
        area = p["spoiler_mt_ms_per_m"] * 1e-6
        cyc_px = area * GAMMA_HZ_PER_T * px * 1e-3
        if cyc_px < 1 - 1e-9:
            need = 1 / (GAMMA_HZ_PER_T * px * 1e-3) * 1e6
            bad.append(_violation("SPOILER_BELOW_MINIMUM", "spoiler_mt_ms_per_m", "the spoiler must dephase at least one "
                                  f"cycle across a pixel: >= {need:.1f} mT*ms/m at this pixel size", minimum=round(need, 1)))
        spoil = _lobe_ms(area, s)
        esp_min = _up(t_ro + 2 * ramp + _lobe_ms(n / (GAMMA_HZ_PER_T * fov_m), s), ras)
        n_echo = p.get("n_echoes", 1)
        esp = esp_min if p.get("echo_spacing_ms", "min") == "min" else _up(p["echo_spacing_ms"], ras)
        if n_echo > 1 and esp < esp_min - 1e-9:
            bad.append(_violation("ECHO_SPACING_BELOW_MINIMUM", "echo_spacing_ms", f"minimum echo spacing is {esp_min} ms",
                                  requested=esp, minimum=esp_min))
        te_last = te + (n_echo - 1) * esp
        tr_min = _up(rf_tr + te_last + t_ro / 2 + ramp + spoil + s["dead_ms"] + fs_ms, ras)
        tr = _up(p["tr_ms"], ras)
        phi = s[f"{seq}_phase_increment_deg"] if p["rf_phase_increment_deg"] == "seq" else p["rf_phase_increment_deg"]
        r.update(tr_ms=tr, te_ms=te, flip_deg=p["flip_deg"], spoiler_mt_ms_per_m=p["spoiler_mt_ms_per_m"],
                 rf_phase_increment_deg=phi)
        timing.update(spoiler_ms=round(spoil, 3), spoiler_cycles_per_px=round(cyc_px, 2))
        pulses = [(_b1_ut(flip, rf_ms), rf_ms * energy)]
        if seq == "spgr":
            r.update(n_echoes=n_echo, echo_spacing_ms=esp if n_echo > 1 else None)
            timing.update(echo_spacing_min_ms=esp_min)
            cycle = tr
        else:
            r.update(tr_ratio=p["tr_ratio"], tr2_ms=round(tr * p["tr_ratio"], 6))
            cycle, pulses = tr * (1 + p["tr_ratio"]), pulses * 2
        dummies = math.ceil(s["dummy_ms"] / cycle)
    elif seq in ("se", "ir"):
        te_min = _up(2 * max(rf_ms / 2 + pre + s["crusher_ms"] + rf180_ms / 2,
                             rf180_ms / 2 + s["crusher_ms"] + ramp + t_ro / 2), ras)
        te = te_min if p["te_ms"] == "min" else _up(p["te_ms"], ras)
        tail = t_ro / 2 + ramp + s["se_spoil_ms"]
        tr = _up(p["tr_ms"], ras)
        pulses = [(_b1_ut(flip, rf_ms), rf_ms * energy), (_b1_ut(180 * tx, rf180_ms), rf180_ms * energy)]
        timing.update(rf_180_ms=round(rf180_ms, 6))
        r.update(tr_ms=tr, te_ms=te, flip_deg=90.0)
        if seq == "ir":
            ti = _up(p["ti_ms"], ras)
            ti_min = _up(s["inv_ms"] / 2 + s["crusher_ms"] + fs_ms + rf_tr / 2, ras)
            tr_min = _up(s["inv_ms"] / 2 + ti + te + tail, ras)
            if ti < ti_min - 1e-9:
                bad.append(_violation("TI_NOT_REALISABLE", "ti_ms", f"ti_ms={ti} is below the minimum {ti_min} ms",
                                      requested=ti, minimum=ti_min))
            r.update(ti_ms=ti)
            timing.update(ti_min_ms=ti_min)
            pulses.append((s["inv_b1_ut"], s["inv_ms"]))
        else:
            tr_min = _up(rf_tr / 2 + te + tail + fs_ms, ras)
        cycle, dummies = tr, s["prepared_dummy_cycles"]
    elif seq == "bssfp":
        tr_min = _up(rf_tr + 2 * (pre + ramp) + t_ro + s["dead_ms"], ras)
        tr = _up(p["tr_ms"], ras)
        te_min = te = round(tr / 2, 6)
        r.update(tr_ms=tr, te_ms=te, flip_deg=p["flip_deg"], phase_cycle_deg=p["phase_cycle_deg"])
        pulses = [(_b1_ut(flip, rf_ms), rf_ms * energy)]
        cycle, dummies = tr, math.ceil(s["dummy_ms"] / tr)
    else:                                                                # epi
        esp_min = _up(t_ro + 2 * ramp, ras)
        esp = esp_min if p["echo_spacing_ms"] == "min" else _up(p["echo_spacing_ms"], ras)
        if esp < esp_min - 1e-9:
            bad.append(_violation("ECHO_SPACING_BELOW_MINIMUM", "echo_spacing_ms", f"minimum echo spacing is {esp_min} ms",
                                  requested=esp, minimum=esp_min))
        before = lines - lines_full // 2                                 # lines acquired before k = 0
        te_min = _up(rf_ms / 2 + pre + before * esp + s["dead_ms"], ras)
        te = te_min if p["te_ms"] == "min" else _up(p["te_ms"], ras)
        tr_min = _up(rf_tr / 2 + te + (lines_full // 2) * esp + 1.0 + fs_ms, ras)
        tr = _up(p["tr_ms"], ras)
        r.update(tr_ms=tr, te_ms=te, flip_deg=p["flip_deg"], echo_spacing_ms=esp, pe_polarity=p["pe_polarity"])
        pulses = [(_b1_ut(flip, rf_ms), rf_ms * energy)]
        cycle, dummies = tr, s["epi_dummy_shots"]

    if te < te_min - 1e-9:
        bad.append(_violation("TE_BELOW_MINIMUM", "te_ms", f"te_ms={te} is below the minimum {te_min} ms for this "
                              "matrix, field of view and bandwidth", requested=te, minimum=te_min))
    if tr < tr_min - 1e-9:
        bad.append(_violation("TR_BELOW_MINIMUM", "tr_ms", f"tr_ms={tr} is below the minimum {tr_min} ms",
                              requested=tr, minimum=tr_min))
    if fat_sat:
        pulses = pulses + [(s["fatsat_b1_ut"], s["fatsat_ms"])]
    r["echo_times_ms"] = [round(te + i * (esp or 0.0), 6) for i in range(n_echo)]
    b1rms2 = sum(b * b * d for b, d in pulses) / cycle
    sar = 100.0 * b1rms2 / exam["b1rms_limit_ut"] ** 2
    grad = 100.0 * g_ro * 1e3 / s["g_max_mt_m"]
    if sar > 100.0:
        bad.append(_violation("RF_POWER_LIMIT", "tr_ms", f"time-averaged RF power is {sar:.0f} % of the limit; lower "
                              "the flip angle or lengthen the repetition time", percent_of_limit=round(sar, 1)))
    if grad > 100.0:
        bad.append(_violation("GRADIENT_LIMIT", "bandwidth_hz_per_px", f"readout gradient is {grad:.0f} % of the limit; "
                              "lower the bandwidth or the matrix, or enlarge the field of view", percent_of_limit=round(grad, 1)))
    if seq == "epi":
        acq, dummy_s = cycle * p["averages"] * partitions / 1e3, dummies * cycle / 1e3
    else:
        acq, dummy_s = cycle * lines * p["averages"] * partitions / 1e3, dummies * cycle / 1e3
    f_fat = FAT_PPM * 1e-6 * GAMMA_HZ_PER_T * b0_t
    wfs = f_fat * esp * 1e-3 * lines_full if seq == "epi" else f_fat / bw
    ref = s["snr_ref"]
    vox = (px, fov_pe / n_pe, p["slice_mm"])
    rel_snr = ((vox[0] * vox[1] * vox[2] / ref["voxel_mm3"]) * lines_full / math.sqrt(noise_lines(lines_full, lines) * ref["lines"])
               * math.sqrt(p["averages"] * partitions * ref["bandwidth_hz_per_px"] / bw) * b0_t / 3.0)
    out.update(
        accept=not bad, realised=r,
        timing=dict(timing, te_min_ms=te_min, tr_min_ms=tr_min, cycle_ms=round(cycle, 6), dummy_cycles=dummies,
                    voxel_mm=[round(v, 4) for v in vox]),
        scan_s=round(s["overhead_s"] + dummy_s + acq, 6),
        scan_s_parts=dict(overhead_s=s["overhead_s"], dummy_s=round(dummy_s, 6), acquisition_s=round(acq, 6)),
        limits=dict(rf_power_percent=round(sar, 2), gradient_percent=round(grad, 2)),
        derived=dict(water_fat_shift_px=round(wfs, 3), water_fat_shift_axis="phase" if seq == "epi" else "readout",
                     relative_snr=round(rel_snr, 4), field_t=b0_t))
    out["header"] = header(out)
    return out


def header(pl):
    """The line a console prints above an image."""
    r, d, ta = pl["realised"], pl["derived"], pl["scan_s"]
    pe = {"y": "A>>P", "x": "R>>L"}[r["pe_dir"]]
    extra = ""
    if r["sequence"] == "ir":
        extra = f" TI {r['ti_ms']:.0f}"
    elif r["sequence"] == "epi":
        extra = f" ESP {r['echo_spacing_ms']:.2f}"
    elif r["sequence"] == "afi":
        extra = f" TR2 {r['tr2_ms']:.1f}"
    elif r.get("n_echoes", 1) > 1:
        extra = f" x{r['n_echoes']} dTE {r['echo_spacing_ms']:.2f}"
    off = f" @ y {r['centre_y_mm']:+.0f} x {r['centre_x_mm']:+.0f}" if r.get("centre_y_mm") or r.get("centre_x_mm") else ""
    off += f" rot {r['rotation_deg']:+.0f}" if r.get("rotation_deg") else ""
    sl = f"SL {r['slice_mm']:.1f}" + (f" 3Dx{r['partitions']}" if r.get("partitions", 1) > 1 else "")
    adj = []
    if r.get("cf_shift_hz"):
        adj.append(f"cf {r['cf_shift_hz']:+g} Hz")
    if r.get("tx_scale", 1.0) != 1.0:
        adj.append(f"tx x{r['tx_scale']:g}")
    if r.get("shim_y_hz_per_mm") or r.get("shim_x_hz_per_mm"):
        adj.append(f"shim y {r.get('shim_y_hz_per_mm', 0.0):+.2f} x {r.get('shim_x_hz_per_mm', 0.0):+.2f} Hz/mm")
    adj = f" | ADJ {' '.join(adj)}" if adj else ""
    return (f"{r['sequence'].upper()} {d['field_t']:.1f}T | TR {r['tr_ms']:.1f} TE {r['te_ms']:.1f}{extra} FA {r['flip_deg']:.0f} | "
            f"FOV {r['fov_mm']:.0f}x{r['fov_pe_mm']:.0f} mm{off}  {r['matrix']}x{r['matrix_pe']}  {sl} | "
            f"BW {r['bandwidth_hz_per_px']:.0f} Hz/px  PE {pe}{'  FS' if r['fat_sat'] else ''} | NEX {r['averages']} | "
            f"TA {int(ta // 60)}:{ta % 60:04.1f} | SAR {pl['limits']['rf_power_percent']:.0f} % | "
            f"WFS {d['water_fat_shift_px']:.1f} px ({d['water_fat_shift_axis']}) | rel. SNR {d['relative_snr']:.2f}{adj}")


def request_of(realised):
    """The protocol request that reproduces a realised protocol (used by replay)."""
    spec = SEQUENCES[realised["sequence"]]
    return dict({k: v for k, v in realised.items() if k in spec and v is not None}, sequence=realised["sequence"])


def protocol_schema(exam=None, brief=False):
    """JSON Schema of the protocol parameters, generated from ``SEQUENCES`` so it cannot drift from what ``plan``
    accepts. One flat object for all sequences: these are the top-level arguments of the ``plan`` and ``scan`` tools.

    Rules. One type per parameter; the only unions are the parameters that accept ``"min"`` (number or ``"min"``).
    ``minimum`` / ``maximum`` appear only where the bound is a fixed range of the instrument (the union over the
    sequences; ``plan`` gives the sequence-specific verdict); times whose minimum depends on the rest of the protocol
    carry none. Every description names the unit, the sequences and the default of each sequence as a value a call
    can carry. ``exam``: the session's console card, for the exam's default field of view. ``brief``: name, type, the
    values of an enum, the unit and "see plan" (the ``scan`` tool takes the same parameters; descriptions and ranges
    are written once, in ``plan``, and the console checks the ranges of a scan as it does those of a plan).
    """
    seqs = sorted(SEQUENCES)
    props = {"sequence": {"type": "string", "enum": seqs, "description": "see plan" if brief else "; ".join(
        f"{k}: {SEQUENCE_NAMES.get(k) or DESCRIPTIONS[k].split(':')[0].split(';')[0].split(',')[0]}" for k in seqs)}}
    for name in parameter_names():
        used = [q for q in seqs if name in SEQUENCES[q]]
        others = [q for q in seqs if q not in used]
        kind = SEQUENCES[used[0]][name][2]
        unit = PARAMETER_UNIT.get(name, "")
        fixed = [q for q in seqs if name in FIXED.get(q, {})]
        where = "" if not others or set(others) == set(fixed) else "; not " + ", ".join(others) if len(others) < len(used) else "; only " + ", ".join(used)
        # the unit is in the name (tr_ms, fov_mm, flip_deg, ..._hz): it is written out only where the name does not say it
        named = name.endswith(("_ms", "_mm", "_deg", "_hz", "_hz_per_px", "_mt_ms_per_m"))
        doc = PARAMETER_MEANING.get(name, PARAMETER_UNITS[name]) + (f", {unit}" if unit and not named else "")
        if brief:
            doc = "see plan"
        else:
            by_default = {}
            for q in used:
                by_default.setdefault(default_text(q, name, exam), []).append(q)
            required = by_default.pop(None, [])
            parts = []
            if required:
                parts.append("required" + ("" if len(required) == len(used) else f" for {', '.join(required)}"))
            if by_default and not accepts_min(name):                    # the default of most sequences, the others in brackets
                (main, most), *rest = sorted(by_default.items(), key=lambda kv: -len(kv[1]))
                parts.append(f"default {main}" + (f" for {', '.join(most)}" if required and not rest else "")
                             + (f" ({'; '.join(f'{chr(44).join(qs)}: {d}' for d, qs in rest)})" if rest else ""))
            if accepts_min(name):
                parts.append(f'a number or "{MIN}" (default): the shortest the console can realise'
                             + ("; a smaller number is raised to it and reported" if name == "te_ms" else ""))
            elif name in PROTOCOL_DEPENDENT:
                parts.append("plan prints its minimum")
            doc += f"{where}; " + "; ".join(parts)
            if fixed:
                doc += f"; fixed at {FIXED[fixed[0]][name]:g} for {', '.join(fixed)}"
            if kind in "fi":
                ranges = {(SEQUENCES[q][name][0], SEQUENCES[q][name][1]) for q in used}
                if len(ranges) > 1 and name not in PROTOCOL_DEPENDENT:
                    doc += "; " + ", ".join(f"{q}: {SEQUENCES[q][name][0]:g} to {SEQUENCES[q][name][1]:g}" for q in used
                                            if (SEQUENCES[q][name][0], SEQUENCES[q][name][1]) != max(ranges, key=lambda r: r[1] - r[0]))
        if kind == "b":
            props[name] = {"type": "boolean", "description": doc}
        elif kind == "c":
            choices = list(SEQUENCES[used[0]][name][0])
            props[name] = {"type": "string" if isinstance(choices[0], str) else "integer", "enum": choices, "description": doc}
        else:
            number = {"type": "integer" if kind == "i" else "number"}
            if name not in PROTOCOL_DEPENDENT and not brief:
                number.update(minimum=min(SEQUENCES[q][name][0] for q in used), maximum=max(SEQUENCES[q][name][1] for q in used))
            if accepts_min(name):
                props[name] = {"anyOf": [number, {"type": "string", "enum": [MIN]}], "description": doc}
            else:
                props[name] = dict(number, description=doc)
    required = ["sequence"] + [n for n in parameter_names() if all(n in spec and spec[n][3] is None for spec in SEQUENCES.values())]
    return {"type": "object", "additionalProperties": False, "required": required, "properties": props}
