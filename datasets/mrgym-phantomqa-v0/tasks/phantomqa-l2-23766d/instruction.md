# Phantom QA: measure, judge and diagnose
The scanner is a 1.5 T system. The uniform section (slice 7) of the ACR large
phantom lies at the isocentre: a disc of solution, 190 mm inner diameter. The scanner may have a fault, two, or none.

1. Run the site's ACR T1 scan and measure on its image:
    sequence se, tr_ms 500, te_ms 20, fov_mm 250, matrix 256, slice_mm 5, averages 1,
    bandwidth_hz_per_px 130, pe_dir x (phase encoding along x, readout along y).
  diameter_y_mm, diameter_x_mm   edge to edge through the centre of the phantom along y, edges at half the mean signal of the 200 cm^2 circle (radius 79.8 mm) centred on the isocentre; the same along x
  piu_percent                    100 (1 - (high - low) / (high + low)): highest and lowest mean of a 1 cm^2 circle (radius 5.64 mm) lying entirely inside the 200 cm^2 circle (radius 79.8 mm) centred on the phantom
  ghosting_ratio                 |((top + bottom) - (left + right)) / (2 large)|: large = mean of the 200 cm^2 circle; the others = mean of 16 x 63 mm rectangles in the background, centred 110 mm from the isocentre, the 63 mm side tangential (top, bottom: along x; left, right: along y)
  (profile gives the numbers along a line, and with samples = 1 the mean of a rectangle; read_regions the mean of circles.)
2. Judge the tests of the ACR guidance:
  geometric_accuracy     fails when a diameter is outside 190 +- 3 mm
  intensity_uniformity   fails when piu_percent is below 85
  ghosting               fails when ghosting_ratio is above 0.03
3. Name every fault of the system. The closed list (a fault moves the test named in brackets):
  gradient_x         [geometric_accuracy] the X gradient (left-right, image columns) is miscalibrated: lengths along x are wrong by the same percentage in every protocol
  gradient_y         [geometric_accuracy] the Y gradient (anterior-posterior, image rows) is miscalibrated: lengths along y are wrong by the same percentage in every protocol
  shim               [geometric_accuracy] the shim left a linear B0 gradient across the phantom: the image is stretched or compressed along the readout axis, in proportion to 1 / bandwidth
  b1_transmit        [intensity_uniformity] the transmit field is too high or too low near the centre of the phantom (the flip angle is off there)
  receive_coil       [intensity_uniformity] a receive coil element is down: the receive sensitivity falls off across the phantom
  signal_instability [ghosting] the signal amplitude varies from pulse to pulse (transmitter, receiver or gradient amplifier)
  phantom_motion     [ghosting] the phantom is not fixed: it rocks along the phase-encode axis during the scan
  A failed test is not a diagnosis: more than one fault moves each test. A failed test has exactly one fault. The
  scan-time budget leaves room after the QA scan for a short scan of your own design that tells them apart. A fault
  in the list is either present (its test fails) or absent; small imperfections that fail no test are not faults.

Deliver with `submit`: the four measured values, the failed tests, the faults. Passing: each value within
0.6 mm, 3 PIU points, and 0.006 or 15 % of the ratio (whichever is larger) of the scanner's own
noise-free measurement on the scan above; the failed tests exactly those that fail; the faults exactly those present.
The scanner's log must show that you measured. An answer in the wrong unit is wrong.

Limits. Scan time: 156.5 s (only `scan` spends it). Tool calls: 64 (every call counts once unless rejected). Rejected calls: 8 (a call refused for how it is written: unknown name, wrong type; costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 156.50 of 156.5 s · calls 0/64 · rejected 0/8]

## Working in this container

The scanner is a service at `http://scanner:8000` (`$MRSCAN_URL`); your token is in the file `$MRSCAN_TOKEN_FILE`. Its calls are `info`, `task`, `budget`, `plan`, `scan` and `submit`:

```sh
python -m mrgym.scanner.client info                 # the instrument: sequences, parameter ranges and units, cost rule
python -m mrgym.scanner.client task                 # this statement and the JSON Schema of the deliverable
python -m mrgym.scanner.client budget
python -m mrgym.scanner.client plan --protocol '<json>'              # free dry run
python -m mrgym.scanner.client scan --protocol '<json>' --rois regions.json --out scans/a     # costs scan time
python -m mrgym.compat.harbor submit '<json>'       # delivers; also writes /app/answer.json
```

`scan` prints the read-outs of the circles you pass and writes the images as `<out>_magnitude.npy` and `<out>_phase.npy` (float32, [image, y, x]). Other tools the statement may name (looking at an image, reading other circles, a profile, a calculator, a status line) are not calls of this service: compute them from those arrays; `budget` gives what is left. The limit on calls counts `plan` and `scan`.

Deliver the arguments of `submit` as one JSON object with `python -m mrgym.compat.harbor submit`. It sends the deliverable to the scanner, which ends the episode, and writes it to `/app/answer.json`, which is what gets scored. A deliverable the console cannot evaluate is refused with the reason and the episode continues. Over plain HTTP: `POST /v1/submit` with `Authorization: Bearer <token>`, then write the same object to `/app/answer.json`.
