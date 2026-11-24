"""The console: protocol in; realised protocol, timing, scan time, limits and the header line out. Never any signal.

``plan(protocol)`` validates a request and returns the realised protocol (defaults filled, times on the raster,
minimum TE / TR implied by matrix, bandwidth and gradient limits), the scan time (per-scan overhead + dummy cycles +
acquisition), RF power and gradient use as percent of limit, and accept / reject with typed reasons. Nothing it
returns depends on the object: no signal, no noise level. The relative-SNR figure in the header is voxel volume times
the square root of the sampling time, a function of the protocol alone.

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
                the NOMINAL flip angle, as a scanner does before it knows the transmit error.
"""
from __future__ import annotations

import math

GAMMA_HZ_PER_T = 42.577478518e6
FAT_PPM = 3.4
SYSTEM = dict(
    g_max_mt_m=40.0, slew_t_m_s=150.0, derate=0.8, raster_ms=0.01, dead_ms=0.05,
    rf_ms=1.0, rf_se90_ms=2.0, rf_180_ms=3.0, inv_ms=8.0, inv_b1_ut=12.0, crusher_ms=1.0, se_spoil_ms=1.0,
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
    averages=(1, 64, "i", 1),
)
_TE = dict(te_ms=(0.5, 300.0, "f", "min"))
_FS = dict(fat_sat=(None, None, "b", False))
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
         "pe_polarity": ((1, -1), None, "c", 1), "fat_sat": (None, None, "b", True)},
)
PARAMETER_UNITS = dict(
    fov_mm="mm, along the readout direction", phase_fov="fraction of fov_mm along the phase-encode direction",
    matrix="pixels along the readout (even); phase-encode lines follow from phase_fov", slice_mm="mm",
    bandwidth_hz_per_px="Hz per pixel", pe_dir="phase-encode axis: y (rows) or x (columns)",
    ro_polarity="direction of the readout: with 1 a spin off resonance by df appears -df / bandwidth pixels along the readout", partial_fourier="fraction of phase-encode lines acquired",
    phase_oversampling="fraction of extra phase field of view, acquired and discarded", averages="count",
    fat_sat="spectral fat saturation before every excitation", te_ms="ms", flip_deg="degrees (nominal)",
    spoiler_mt_ms_per_m="mT*ms/m per repetition", rf_phase_increment_deg="degrees", tr_ms="ms", n_echoes="count",
    echo_spacing_ms="ms", tr_ratio="integer TR2 / TR1", ti_ms="ms, centre of the inversion to centre of the excitation",
    phase_cycle_deg="degrees of RF phase advance per TR", pe_polarity="sign of the phase-encode blips",
)
DESCRIPTIONS = dict(
    spgr="RF-spoiled gradient echo, 1 to 8 echoes: steady state with quadratic RF phase cycling and a gradient "
         "spoiler of the requested moment in every repetition",
    afi="actual flip-angle imaging: dual-TR RF-spoiled gradient echo, TR2 = tr_ratio * tr_ms, spoiler moments and RF "
        "phase increments in the ratio TR1 : TR2; image 0 is acquired in TR1, image 1 in TR2",
    se="spin echo: 90 degree excitation, 180 degree refocusing (both nominal; both scale with the transmit field), "
       "one phase-encode line per tr_ms",
    ir="inversion recovery: adiabatic inversion, delay ti_ms, spin-echo readout; signed signal (the sign is in the "
       "phase image)",
    bssfp="balanced steady-state free precession, echo at tr_ms / 2, RF phase advanced by phase_cycle_deg per TR",
    epi="single-shot gradient-echo echo-planar imaging: all phase-encode lines after one excitation, "
        "echo_spacing_ms apart; spectral fat saturation on by default",
)


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


def _violation(code, parameter, message, **extra):
    return dict(code=code, parameter=parameter, message=message, **extra)


def validate(protocol, exam=None):
    """Return (sequence, parameter dict with defaults, violations). Unknown keys are errors, never ignored."""
    if not isinstance(protocol, dict):
        return None, None, [_violation("INVALID_REQUEST", None, "protocol must be a JSON object")]
    seq = protocol.get("sequence")
    if seq not in SEQUENCES:
        return None, None, [_violation("UNKNOWN_SEQUENCE", "sequence", f"sequence must be one of {sorted(SEQUENCES)}")]
    spec, p, bad = SEQUENCES[seq], {}, []
    for name in sorted(set(protocol) - set(spec) - {"sequence"}):
        bad.append(_violation("UNKNOWN_PARAMETER", name, f"{seq} has no parameter {name}; allowed: {sorted(spec)}"))
    for name, (lo, hi, kind, default) in spec.items():
        v = protocol.get(name, default)
        if v is None:
            bad.append(_violation("MISSING_PARAMETER", name, f"{seq} needs {name} [{PARAMETER_UNITS[name]}]"))
            continue
        if name not in protocol and v in ("min", "seq", "exam"):
            p[name] = (exam or {}).get(name, 240.0) if v == "exam" else v
            continue
        if kind == "b":
            if not isinstance(v, bool):
                bad.append(_violation("OUT_OF_RANGE", name, f"{name} must be true or false"))
                continue
            p[name] = v
            continue
        if kind == "c":
            if isinstance(v, bool) or v not in lo:
                bad.append(_violation("OUT_OF_RANGE", name, f"{name} must be one of {list(lo)}"))
                continue
            p[name] = v
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            bad.append(_violation("OUT_OF_RANGE", name, f"{name} must be a finite number"))
            continue
        if kind == "i" and int(v) != v:
            bad.append(_violation("OUT_OF_RANGE", name, f"{name} must be an integer"))
            continue
        v = int(v) if kind == "i" else float(v)
        if not lo <= v <= hi:
            bad.append(_violation("OUT_OF_RANGE", name, f"{name}={v} outside [{lo}, {hi}] {PARAMETER_UNITS[name]}",
                                  requested=v, low=lo, high=hi))
            continue
        p[name] = v
    if "matrix" in p and p["matrix"] % 2:
        bad.append(_violation("OUT_OF_RANGE", "matrix", "matrix must be even"))
    return seq, p, bad


def plan(protocol, exam=None, system=None):
    """Dry run. ``exam``: the session's console card (``fov_mm`` default, ``min_pixel_mm``, ``b1rms_limit_ut``,
    ``b0_t``). Returns a JSON-serialisable dict; ``accept`` False comes with ``violations`` (code, parameter, message).
    """
    s = dict(SYSTEM, **(system or {}))
    exam = dict(dict(fov_mm=240.0, min_pixel_mm=0.8, b1rms_limit_ut=3.0, b0_t=3.0), **(exam or {}))
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
    r = dict(sequence=seq, fov_mm=fov, phase_fov=p["phase_fov"], fov_pe_mm=round(fov_pe, 6), matrix=n, matrix_pe=n_pe, lines_full=lines_full,
             lines=lines, slice_mm=p["slice_mm"], bandwidth_hz_per_px=bw, pe_dir=p["pe_dir"], ro_polarity=p["ro_polarity"],
             partial_fourier=p["partial_fourier"], phase_oversampling=p["phase_oversampling"], averages=p["averages"],
             fat_sat=fat_sat)
    timing = dict(readout_ms=round(t_ro, 4))
    esp = None
    n_echo = 1

    if seq in ("spgr", "afi"):
        te_min = _up(s["rf_ms"] / 2 + pre + ramp + t_ro / 2 + s["dead_ms"], ras)
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
        tr_min = _up(s["rf_ms"] + te_last + t_ro / 2 + ramp + spoil + s["dead_ms"] + fs_ms, ras)
        tr = _up(p["tr_ms"], ras)
        phi = s[f"{seq}_phase_increment_deg"] if p["rf_phase_increment_deg"] == "seq" else p["rf_phase_increment_deg"]
        r.update(tr_ms=tr, te_ms=te, flip_deg=p["flip_deg"], spoiler_mt_ms_per_m=p["spoiler_mt_ms_per_m"],
                 rf_phase_increment_deg=phi)
        timing.update(spoiler_ms=round(spoil, 3), spoiler_cycles_per_px=round(cyc_px, 2))
        pulses = [(_b1_ut(p["flip_deg"], s["rf_ms"]), s["rf_ms"])]
        if seq == "spgr":
            r.update(n_echoes=n_echo, echo_spacing_ms=esp if n_echo > 1 else None)
            timing.update(echo_spacing_min_ms=esp_min)
            cycle = tr
        else:
            r.update(tr_ratio=p["tr_ratio"], tr2_ms=round(tr * p["tr_ratio"], 6))
            cycle, pulses = tr * (1 + p["tr_ratio"]), pulses * 2
        dummies = math.ceil(s["dummy_ms"] / cycle)
    elif seq in ("se", "ir"):
        te_min = _up(2 * max(s["rf_se90_ms"] / 2 + pre + s["crusher_ms"] + s["rf_180_ms"] / 2,
                             s["rf_180_ms"] / 2 + s["crusher_ms"] + ramp + t_ro / 2), ras)
        te = te_min if p["te_ms"] == "min" else _up(p["te_ms"], ras)
        tail = t_ro / 2 + ramp + s["se_spoil_ms"]
        tr = _up(p["tr_ms"], ras)
        pulses = [(_b1_ut(90, s["rf_se90_ms"]), s["rf_se90_ms"]), (_b1_ut(180, s["rf_180_ms"]), s["rf_180_ms"])]
        r.update(tr_ms=tr, te_ms=te, flip_deg=90.0)
        if seq == "ir":
            ti = _up(p["ti_ms"], ras)
            ti_min = _up(s["inv_ms"] / 2 + s["crusher_ms"] + fs_ms + s["rf_se90_ms"] / 2, ras)
            tr_min = _up(s["inv_ms"] / 2 + ti + te + tail, ras)
            if ti < ti_min - 1e-9:
                bad.append(_violation("TI_NOT_REALISABLE", "ti_ms", f"ti_ms={ti} is below the minimum {ti_min} ms",
                                      requested=ti, minimum=ti_min))
            r.update(ti_ms=ti)
            timing.update(ti_min_ms=ti_min)
            pulses.append((s["inv_b1_ut"], s["inv_ms"]))
        else:
            tr_min = _up(s["rf_se90_ms"] / 2 + te + tail + fs_ms, ras)
        cycle, dummies = tr, s["prepared_dummy_cycles"]
    elif seq == "bssfp":
        tr_min = _up(s["rf_ms"] + 2 * (pre + ramp) + t_ro + s["dead_ms"], ras)
        tr = _up(p["tr_ms"], ras)
        te_min = te = round(tr / 2, 6)
        r.update(tr_ms=tr, te_ms=te, flip_deg=p["flip_deg"], phase_cycle_deg=p["phase_cycle_deg"])
        pulses = [(_b1_ut(p["flip_deg"], s["rf_ms"]), s["rf_ms"])]
        cycle, dummies = tr, math.ceil(s["dummy_ms"] / tr)
    else:                                                                # epi
        esp_min = _up(t_ro + 2 * ramp, ras)
        esp = esp_min if p["echo_spacing_ms"] == "min" else _up(p["echo_spacing_ms"], ras)
        if esp < esp_min - 1e-9:
            bad.append(_violation("ECHO_SPACING_BELOW_MINIMUM", "echo_spacing_ms", f"minimum echo spacing is {esp_min} ms",
                                  requested=esp, minimum=esp_min))
        before = lines - lines_full // 2                                 # lines acquired before k = 0
        te_min = _up(s["rf_ms"] / 2 + pre + before * esp + s["dead_ms"], ras)
        te = te_min if p["te_ms"] == "min" else _up(p["te_ms"], ras)
        tr_min = _up(s["rf_ms"] / 2 + te + (lines_full // 2) * esp + 1.0 + fs_ms, ras)
        tr = _up(p["tr_ms"], ras)
        r.update(tr_ms=tr, te_ms=te, flip_deg=p["flip_deg"], echo_spacing_ms=esp, pe_polarity=p["pe_polarity"])
        pulses = [(_b1_ut(p["flip_deg"], s["rf_ms"]), s["rf_ms"])]
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
        acq, dummy_s = cycle * p["averages"] / 1e3, dummies * cycle / 1e3
    else:
        acq, dummy_s = cycle * lines * p["averages"] / 1e3, dummies * cycle / 1e3
    f_fat = FAT_PPM * 1e-6 * GAMMA_HZ_PER_T * b0_t
    wfs = f_fat * esp * 1e-3 * lines_full if seq == "epi" else f_fat / bw
    ref = s["snr_ref"]
    vox = (px, fov_pe / n_pe, p["slice_mm"])
    rel_snr = ((vox[0] * vox[1] * vox[2] / ref["voxel_mm3"]) * lines_full / math.sqrt(lines * ref["lines"])
               * math.sqrt(p["averages"] * ref["bandwidth_hz_per_px"] / bw) * b0_t / 3.0)
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
    return (f"{r['sequence'].upper()} {d['field_t']:.1f}T | TR {r['tr_ms']:.1f} TE {r['te_ms']:.1f}{extra} FA {r['flip_deg']:.0f} | "
            f"FOV {r['fov_mm']:.0f}x{r['fov_pe_mm']:.0f} mm  {r['matrix']}x{r['matrix_pe']}  SL {r['slice_mm']:.1f} | "
            f"BW {r['bandwidth_hz_per_px']:.0f} Hz/px  PE {pe}{'  FS' if r['fat_sat'] else ''} | NEX {r['averages']} | "
            f"TA {int(ta // 60)}:{ta % 60:04.1f} | SAR {pl['limits']['rf_power_percent']:.0f} % | "
            f"WFS {d['water_fat_shift_px']:.1f} px ({d['water_fat_shift_axis']}) | rel. SNR {d['relative_snr']:.2f}")


def request_of(realised):
    """The protocol request that reproduces a realised protocol (used by replay)."""
    spec = SEQUENCES[realised["sequence"]]
    return dict({k: v for k, v in realised.items() if k in spec and v is not None}, sequence=realised["sequence"])


def protocol_schema():
    """JSON Schema of the ``protocol`` object, generated from ``SEQUENCES`` so it cannot drift from what ``plan``
    accepts. One flat object for all sequences; ranges are the union, ``plan`` gives the sequence-specific verdict."""
    seqs = sorted(SEQUENCES)
    props = {"sequence": {"type": "string", "enum": seqs,
                          "description": "; ".join(f"{k}: {DESCRIPTIONS[k].split(':')[0].split(';')[0].split(',')[0]}" for k in seqs)}}
    for name in sorted({k for spec in SEQUENCES.values() for k in spec}):
        used = [q for q in seqs if name in SEQUENCES[q]]
        kind = SEQUENCES[used[0]][name][2]
        dflt = sorted({("console minimum" if v == "min" else "sequence default" if v == "seq" else "exam default"
                        if v == "exam" else str(v).lower() if isinstance(v, bool) else str(v))
                       for v in (SEQUENCES[q][name][3] for q in used) if v is not None})
        req = [q for q in used if SEQUENCES[q][name][3] is None]
        where = "all sequences" if len(used) == len(seqs) else "/".join(used)
        doc = f"{PARAMETER_UNITS[name]}; {where}; " + ("required" if len(req) == len(used) else "default " + ", ".join(dflt)
                                                    + (f" (required for {'/'.join(req)})" if req else ""))
        if kind == "b":
            props[name] = {"type": "boolean", "description": doc}
        elif kind == "c":
            choices = list(SEQUENCES[used[0]][name][0])
            props[name] = {"type": "string" if isinstance(choices[0], str) else "integer", "enum": choices, "description": doc}
        else:
            props[name] = {"type": "integer" if kind == "i" else "number",
                           "minimum": min(SEQUENCES[q][name][0] for q in used),
                           "maximum": max(SEQUENCES[q][name][1] for q in used), "description": doc}
    return {"type": "object", "additionalProperties": False, "required": ["sequence"], "properties": props}
