# Virtual scanner — instrument datasheet

This is what you may know about the instrument. Everything about the object and about the state of the system has to
be measured.

## 1. What it is

A single-slice 2D MR scanner behind a small service. You send a protocol, the scanner charges scan time and returns
reconstructed images plus read-outs of circular regions you choose. One episode is one session: there is no reset.

| | |
|---|---|
| Field of view | 240 mm x 240 mm, fixed, centred on the isocentre |
| Slice | one slice, 5 mm |
| Matrix | square, even, 48 to 192 (default 96) |
| Gradients | 30 mT/m, 150 T/m/s per axis; lobes are played at 80 % of both |
| Excitation | hard pulse, 1 ms; inversion: adiabatic pulse, 8 ms |
| Time raster | 0.01 ms; requested times are rounded up to it |

## 2. Calls

| Call | Cost | Returns |
|---|---|---|
| `info` | free | this card in machine-readable form, the task card, the budget |
| `plan` | free, logged, counted as a request | realised protocol, timing, scan seconds, limits as percent, accept / reject with reasons. Never any signal |
| `scan` | `scan_s` seconds of budget | realised protocol, cost, ROI read-outs, images |
| `budget` | free | total / used / remaining seconds, scans and requests so far |

```
python3 mrscan.py info
python3 mrscan.py plan --protocol '{"sequence": "spgr", "tr_ms": 20, "flip_deg": 10}'
python3 mrscan.py scan --protocol '{"sequence": "spgr", "tr_ms": 20, "flip_deg": 10, "averages": 4}' --rois regions.json --out scans/fa10
python3 mrscan.py budget
```

A refused request costs nothing. Unknown parameters are errors. The scanner snaps times to its raster and fills
defaults; use the **realised** values it returns, not the ones you asked for.

## 3. Protocol parameters (names carry their units)

Common to all sequences:

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| `sequence` | `spgr`, `afi`, `ir` | required | |
| `matrix` | 48–192, even | 96 | pixels along both axes; pixel size = 240 mm / matrix |
| `bandwidth_hz_per_px` | 100–2000 | 300 | receiver bandwidth per pixel; readout duration = 1 / bandwidth |
| `averages` | 1–64 | 1 | complex averages |
| `te_ms` | ≥ console minimum | minimum | echo time (centre of the pulse to centre of the readout) |

`spgr`, `afi`:

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| `tr_ms` | spgr 3–500; afi 5–200 | required | repetition time (afi: the short one, TR1) |
| `flip_deg` | 1–90 | required | **nominal** flip angle |
| `spoiler_mt_ms_per_m` | 10–600 | spgr 100, afi 300 | gradient spoiler moment per raster interval; at least one cycle per pixel |
| `rf_phase_increment_deg` | 0–180 | spgr 117, afi 129.3 | quadratic RF phase cycling (see §5) |
| `n_echoes`, `echo_spacing_ms` | 1–8; ≥ minimum | 1; minimum | spgr only: monopolar multi-echo readout |
| `tr_ratio` | 2–10, integer | 5 | afi only: TR2 = tr_ratio × TR1 |

`ir`: `tr_ms` 1000–20000 (inversion to inversion), `ti_ms` 30–19000 (inversion to readout pulse). Readout pulse 90° nominal.

## 4. What a scan costs and what limits it

- **Scan time** `scan_s = overhead + dummy cycles × cycle + cycle × matrix × averages`. One phase-encode line per cycle.
  Cycle = `tr_ms` (spgr, ir) or `tr_ms × (1 + tr_ratio)` (afi). Overhead 3 s per scan. Gradient-echo scans play 4 s of
  dummy cycles before sampling; inversion recovery plays one dummy cycle. Images are served at the periodic state.
- **Minimum TE** = half the pulse + prephaser + readout ramp + half the readout. Narrow bandwidth and large matrix
  lengthen it.
- **Minimum TR** = pulse + last echo + half the readout + ramp + spoiler. A large spoiler moment costs repetition time
  (about 1 ms per 24 mT·ms/m).
- **RF power** is shown as percent of the limit (time-averaged B1², "SAR-shaped"); above 100 % the scan is refused.
- **Readout gradient** is shown as percent of the limit; bandwidth × matrix is bounded by it.
- **Noise.** Receiver noise is white and Gaussian in the complex image. Its standard deviation per pixel scales as
  `matrix^1.5 × sqrt(bandwidth_hz_per_px) / sqrt(averages)` (smaller pixels collect less signal, more lines average
  more, a wider receiver passes more noise). The level itself is not given; it can be measured.

## 5. Signal mechanism of each sequence

Notation: a = (local transmit factor) × nominal flip angle; E(t) = exp(−t / T1).

**All sequences.** At echo time TE the signal carries exp(−TE / T2*) with 1/T2* = 1/T2 + 1/T2′, and the phase
+2π·Δf·TE, where Δf (Hz) is the local off-resonance relative to the receiver reference (positive Δf, positive phase).
Images also carry the receive coil's smooth magnitude and phase profile. The image scale is arbitrary but constant
within a session.

**`spgr` — RF-spoiled gradient echo.** Identical pulses every TR; the phase of pulse j is Φ·j(j+1)/2 with Φ =
`rf_phase_increment_deg`; a spoiler of the requested moment in every TR. With *ideal* spoiling (no transverse
magnetisation survives a TR) the steady state is the Ernst formula

    S = M0 · sin a · (1 − E(TR)) / (1 − E(TR) · cos a).

The scanner does not assume ideal spoiling. Transverse magnetisation that survives a TR (T2 is finite) is dephased by
the spoiler, partly refocused by later pulses, and attenuated by diffusion in proportion to the square of the
accumulated spoiler moment. The steady state therefore depends on T2, on the diffusion coefficient, on Φ and on the
spoiler moment, and deviates from the Ernst formula — little at low flip angles, long TR and strong spoilers; by tens
of percent at high flip angles, short TR and weak spoilers. The sizes are properties of the object and are not given.

**`afi` — actual flip-angle imaging.** The same pulse with two alternating delays, TR1 = `tr_ms` and TR2 = `tr_ratio`
× TR1; image 1 is acquired in TR1, image 2 in TR2. Spoiler moments and RF phase increments are in the ratio TR1 : TR2
(the sequence runs on a raster of period TR1; the pulse at raster position j has phase Φ·j(j+1)/2, and positions
without a pulse still count). With ideal spoiling

    S1 = M0 sin a (1 − E2 + (1 − E1) E2 cos a) / (1 − E1 E2 cos² a),   S2: E1 ↔ E2,   E1 = E(TR1), E2 = E(TR2)

and for TR1, TR2 ≪ T1 the ratio r = S2 / S1 ≈ (1 + n cos a) / (n + cos a), n = `tr_ratio`. The same spoiling
mechanism as for `spgr` applies, and this sequence is the more sensitive of the two.

**`ir` — inversion recovery.** Adiabatic inversion (insensitive to the transmit factor, efficiency slightly below 1),
delay TI, 90° nominal readout pulse, gradient-echo readout, one line per TR; crushers after the inversion and after
the readout. The signal is signed (the sign is in the phase image):

    S = M0 · sin a · (1 − (1 + κ) E(TI) + κ E(TR)) / (1 + κ cos a · E(TR)),   a = transmit factor × 90°, κ = inversion efficiency.

## 6. What you get back

- `<out>_magnitude.npy`, `<out>_phase.npy`: float32 arrays `[n_images, matrix, matrix]`, row = y, column = x; pixel `i`
  is centred at `(i − matrix // 2) / matrix × 240 mm`. `afi` returns two images, `spgr` one per echo.
- ROI read-outs for every circle `{id, y_mm, x_mm, radius_mm}` you pass (at most 16 per scan): `n_pixels`,
  `mean_magnitude`, `sd_magnitude`, `complex_mean_magnitude`, `complex_mean_phase_rad`, one value per image. The
  magnitude mean has a noise floor; the complex mean does not.
- Images are band-limited reconstructions of the object: edges ring and pixels at a boundary mix materials.

## 7. What is hidden

The object (geometry, proton density, T1, T2, T2′, diffusion), the transmit field (a global gain error and a smooth
spatial pattern), the off-resonance field and centre-frequency offset, the receive profile, the inversion efficiency,
the noise level, the RF power limit in absolute units. The mechanisms above are complete; their magnitudes are not
given. Not simulated: slice profile, the approach to steady state, blurring and displacement along the readout,
motion, flow.
