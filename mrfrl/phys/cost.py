"""The priced, realisable protocol: what the console computes before anything is acquired.

``plan(protocol)`` validates a request and returns the realised protocol (defaults filled, times on the raster,
minimum TE / TR implied by matrix, bandwidth and spoiler), the scan time (per-scan overhead + dummy repetitions +
acquisition), RF power and gradient use as percent of limit, and accept / reject with reasons. It never returns
anything that depends on the object: no signal, no noise level, no SNR.

Model (documented for the agent in docs/scanner.md):
  readout       duration 1 / bandwidth_hz_per_px; amplitude G = bandwidth * matrix / (gamma * FOV)
  lobes         prephaser (half the readout area), phase encode, spoiler (requested area): time-optimal trapezoids at
                LOBE_DERATE of the gradient and slew limits
  TE minimum    rf / 2 + prephaser + readout ramp + readout / 2
  TR minimum    rf / 2 + last TE + readout / 2 + ramp + spoiler + rf / 2
  scan time     overhead_s + dummy time + cycle * matrix * averages  (one phase-encode line per cycle)
  RF power      B1rms^2 = sum over pulses of B1^2 * duration / cycle, as percent of the system's (B1rms limit)^2
  noise         per-pixel SD = sigma0 * (matrix / 96)^1.5 * sqrt(bandwidth / 300) / sqrt(averages): voxel area falls
                as 1 / matrix^2, the number of lines grows as matrix, the receiver bandwidth as bandwidth * matrix
"""
import math

GAMMA_HZ_PER_T = 42.577478518e6
SYSTEM = dict(
    fov_mm=240.0, slice_mm=5.0, g_max_mt_m=30.0, slew_t_m_s=150.0, lobe_derate=0.8,
    rf_ms=1.0, inv_ms=8.0, inv_b1_ut=12.0, inv_crusher_ms=3.0, dead_ms=0.05, raster_ms=0.01,
    overhead_s=3.0, dummy_ms=4000.0, ir_dummy_cycles=1,
    spgr_phase_increment_deg=117.0, afi_phase_increment_deg=129.3,
    ref_matrix=96, ref_bandwidth_hz_per_px=300.0,
)

# name: (low, high, kind, default)   kind f = float, i = integer; default None = required, "min" = console minimum
_COMMON = dict(matrix=(48, 192, "i", 96), bandwidth_hz_per_px=(100.0, 2000.0, "f", 300.0), averages=(1, 64, "i", 1),
               te_ms=(0.5, 100.0, "f", "min"))
_GRE = dict(_COMMON, flip_deg=(1.0, 90.0, "f", None), spoiler_mt_ms_per_m=(10.0, 600.0, "f", 100.0),
            rf_phase_increment_deg=(0.0, 180.0, "f", "seq"))
SEQUENCES = dict(
    spgr=dict(_GRE, tr_ms=(3.0, 500.0, "f", None), n_echoes=(1, 8, "i", 1), echo_spacing_ms=(0.5, 50.0, "f", "min")),
    afi=dict(_GRE, tr_ms=(5.0, 200.0, "f", None), tr_ratio=(2, 10, "i", 5), spoiler_mt_ms_per_m=(10.0, 600.0, "f", 300.0)),
    ir=dict(_COMMON, tr_ms=(1000.0, 20000.0, "f", None), ti_ms=(30.0, 19000.0, "f", None)),
)
PARAMETER_UNITS = dict(matrix="pixels (square, even)", bandwidth_hz_per_px="Hz per pixel", averages="count",
                       te_ms="ms", flip_deg="degrees (nominal)", spoiler_mt_ms_per_m="mT*ms/m per raster interval",
                       rf_phase_increment_deg="degrees", tr_ms="ms", n_echoes="count", echo_spacing_ms="ms",
                       tr_ratio="integer TR2 / TR1", ti_ms="ms")


def _lobe_ms(area_t_s_per_m, sysm):
    """Duration of a time-optimal trapezoid of the given area (T*s/m) at the derated limits."""
    g, s = sysm["g_max_mt_m"] * 1e-3 * sysm["lobe_derate"], sysm["slew_t_m_s"] * sysm["lobe_derate"]
    if area_t_s_per_m <= g * g / s:
        return 2e3 * math.sqrt(area_t_s_per_m / s)
    return 1e3 * (area_t_s_per_m / g + g / s)


def _up(x, raster):
    return round(math.ceil(x / raster - 1e-9) * raster, 6)


def _violation(code, parameter, message, **extra):
    return dict(code=code, parameter=parameter, message=message, **extra)


def validate(protocol):
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
        if v in ("min", "seq") and name not in protocol:
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


def noise_factor(realised, sysm=SYSTEM):
    """Per-pixel noise SD relative to the reference protocol (matrix 96, 300 Hz/px, 1 average). See module docstring."""
    return ((realised["matrix"] / sysm["ref_matrix"]) ** 1.5
            * math.sqrt(realised["bandwidth_hz_per_px"] / sysm["ref_bandwidth_hz_per_px"]) / math.sqrt(realised["averages"]))


def plan(protocol, system=None, b1rms_limit_ut=2.0):
    """Dry run. Returns a JSON-serialisable dict; ``accept`` False comes with ``violations`` (code, parameter, message)."""
    sysm = dict(SYSTEM, **(system or {}))
    seq, p, bad = validate(protocol)
    out = dict(accept=False, sequence=seq, violations=bad)
    if bad:
        return out
    ras, fov = sysm["raster_ms"], sysm["fov_mm"] * 1e-3
    n, bw = p["matrix"], p["bandwidth_hz_per_px"]
    t_ro = 1e3 / bw
    g_ro = bw * n / (GAMMA_HZ_PER_T * fov)                               # T/m
    ramp = 1e3 * g_ro / (sysm["slew_t_m_s"] * sysm["lobe_derate"])
    pre = _lobe_ms(n / (2 * GAMMA_HZ_PER_T * fov), sysm)
    px_mm = sysm["fov_mm"] / n
    te_min = _up(sysm["rf_ms"] / 2 + pre + ramp + t_ro / 2 + sysm["dead_ms"], ras)
    te = te_min if p["te_ms"] == "min" else _up(p["te_ms"], ras)
    if te < te_min - 1e-9:
        bad.append(_violation("TE_BELOW_MINIMUM", "te_ms", f"te_ms={te} is below the minimum {te_min} ms for matrix "
                              f"{n} and {bw} Hz/px", requested=te, minimum=te_min))
    r = dict(sequence=seq, matrix=n, bandwidth_hz_per_px=bw, averages=p["averages"], te_ms=te)
    timing = dict(readout_ms=round(t_ro, 4), te_min_ms=te_min)
    pulses = []                                                          # (B1 in uT, duration in ms) per cycle
    b1_of = lambda flip_deg: math.radians(flip_deg) / (2 * math.pi * GAMMA_HZ_PER_T * sysm["rf_ms"] * 1e-3) * 1e6  # noqa: E731

    if seq in ("spgr", "afi"):
        area = p["spoiler_mt_ms_per_m"] * 1e-6
        cyc_px = area * GAMMA_HZ_PER_T * px_mm * 1e-3
        if cyc_px < 1 - 1e-9:
            need = 1 / (GAMMA_HZ_PER_T * px_mm * 1e-3) * 1e6
            bad.append(_violation("SPOILER_BELOW_MINIMUM", "spoiler_mt_ms_per_m", "the spoiler must dephase at least one "
                                  f"cycle across a pixel: >= {need:.1f} mT*ms/m at matrix {n}", minimum=round(need, 1)))
        spoil = _lobe_ms(area, sysm)
        esp_min = _up(t_ro + 2 * ramp + _lobe_ms(n / (GAMMA_HZ_PER_T * fov), sysm), ras)
        n_echo = p.get("n_echoes", 1)
        esp = esp_min if p.get("echo_spacing_ms", "min") == "min" else _up(p["echo_spacing_ms"], ras)
        if n_echo > 1 and esp < esp_min - 1e-9:
            bad.append(_violation("ECHO_SPACING_BELOW_MINIMUM", "echo_spacing_ms", f"minimum echo spacing is {esp_min} ms",
                                  requested=esp, minimum=esp_min))
        te_last = te + (n_echo - 1) * esp
        tr_min = _up(sysm["rf_ms"] + te_last + t_ro / 2 + ramp + spoil + sysm["dead_ms"], ras)
        tr = _up(p["tr_ms"], ras)
        if tr < tr_min - 1e-9:
            bad.append(_violation("TR_BELOW_MINIMUM", "tr_ms", f"tr_ms={tr} is below the minimum {tr_min} ms (last echo "
                                  f"{te_last:.2f} ms, readout {t_ro:.2f} ms, spoiler {spoil:.2f} ms)", requested=tr,
                                  minimum=tr_min))
        default_phi = sysm["spgr_phase_increment_deg" if seq == "spgr" else "afi_phase_increment_deg"]
        r.update(tr_ms=tr, flip_deg=p["flip_deg"], spoiler_mt_ms_per_m=p["spoiler_mt_ms_per_m"],
                 rf_phase_increment_deg=default_phi if p["rf_phase_increment_deg"] == "seq" else p["rf_phase_increment_deg"])
        timing.update(tr_min_ms=tr_min, spoiler_ms=round(spoil, 3), spoiler_cycles_per_px=round(cyc_px, 2))
        if seq == "spgr":
            r.update(n_echoes=n_echo, echo_spacing_ms=esp if n_echo > 1 else None,
                     echo_times_ms=[round(te + i * esp, 6) for i in range(n_echo)])
            timing.update(echo_spacing_min_ms=esp_min)
            cycle, pulses = tr, [(b1_of(p["flip_deg"]), sysm["rf_ms"])]
        else:
            r.update(tr_ratio=p["tr_ratio"], tr2_ms=round(tr * p["tr_ratio"], 6))
            cycle, pulses = tr * (1 + p["tr_ratio"]), [(b1_of(p["flip_deg"]), sysm["rf_ms"])] * 2
        dummies = math.ceil(sysm["dummy_ms"] / cycle)
    else:
        tr, ti = _up(p["tr_ms"], ras), _up(p["ti_ms"], ras)
        ti_min = _up(sysm["inv_ms"] / 2 + sysm["inv_crusher_ms"] + sysm["rf_ms"] / 2, ras)
        rest_min = _up(sysm["rf_ms"] / 2 + te + t_ro / 2 + ramp + sysm["inv_crusher_ms"] + sysm["inv_ms"] / 2, ras)
        if ti < ti_min - 1e-9 or tr - ti < rest_min - 1e-9:
            bad.append(_violation("TI_NOT_REALISABLE", "ti_ms", f"need {ti_min} <= ti_ms <= tr_ms - {rest_min}",
                                  requested=ti, minimum=ti_min, maximum=round(tr - rest_min, 6)))
        r.update(tr_ms=tr, ti_ms=ti, readout_flip_deg=90.0)
        timing.update(ti_min_ms=ti_min)
        cycle, pulses = tr, [(sysm["inv_b1_ut"], sysm["inv_ms"]), (b1_of(90.0), sysm["rf_ms"])]
        dummies = sysm["ir_dummy_cycles"]

    b1rms2 = sum(b * b * d for b, d in pulses) / cycle
    sar = 100.0 * b1rms2 / b1rms_limit_ut ** 2
    grad = 100.0 * g_ro * 1e3 / sysm["g_max_mt_m"]
    if sar > 100.0:
        bad.append(_violation("RF_POWER_LIMIT", "flip_deg" if seq != "ir" else "tr_ms", f"time-averaged RF power is "
                              f"{sar:.0f} % of the limit; lower the flip angle or lengthen the repetition time",
                              percent_of_limit=round(sar, 1)))
    if grad > 100.0:
        bad.append(_violation("GRADIENT_LIMIT", "bandwidth_hz_per_px", f"readout gradient is {grad:.0f} % of the limit; "
                              "lower the bandwidth or the matrix", percent_of_limit=round(grad, 1)))
    acq = cycle * n * p["averages"] / 1e3
    dummy_s = dummies * cycle / 1e3
    out.update(
        accept=not bad, realised=r,
        timing=dict(timing, cycle_ms=round(cycle, 6), dummy_cycles=dummies, voxel_mm=[round(px_mm, 4), round(px_mm, 4),
                                                                                    sysm["slice_mm"]]),
        scan_s=round(sysm["overhead_s"] + dummy_s + acq, 6),
        scan_s_parts=dict(overhead_s=sysm["overhead_s"], dummy_s=round(dummy_s, 6), acquisition_s=round(acq, 6)),
        limits=dict(rf_power_percent=round(sar, 2), gradient_percent=round(grad, 2),
                    slew_percent=round(100 * sysm["lobe_derate"], 1)))
    return out
