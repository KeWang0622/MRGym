# Meet a signal-to-noise requirement in the shortest scan time
A test object with vials of unknown materials lies in the field of
view. The examination needs 2 spin-echo images, A and B. Each has a prescribed geometry and a required signal-to-noise
ratio (SNR) in two marked regions:
    image A: sequence se, fov_mm 240, matrix 112 (pixel 2.143 mm), slice_mm 5; SNR at least 59.5 in R1 and R2
    image B: sequence se, fov_mm 240, matrix 144 (pixel 1.667 mm), slice_mm 5; SNR at least 42.8 in R3 and R4
For each image you choose tr_ms, averages and bandwidth_hz_per_px. Everything else is fixed: the fields above, te_ms at
the console minimum for the bandwidth you choose, full k-space, every other parameter at its default.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
  R1: y 40.38, x 29.93, radius 11.5
  R2: y -46.86, x -31.11, radius 11.5
  R3: y -32.71, x 43.73, radius 11.5
  R4: y 25.36, x -45.5, radius 11.5
SNR of a region = magnitude of the mean complex signal of the region's pixels / standard deviation of the noise in one pixel, one channel (real or imaginary part).
Deliver the three numbers for each image with `submit`. The images are not run in this episode: the scanner evaluates
the protocols you deliver. Your own scans may use any sequence and any parameters the console accepts; only the deliverable is evaluated, on the scan the task names. That scan is not charged to the scan-time budget and not limited by it.
Nothing about the materials, the noise or the state of the system is known in advance.

Passing: for each image, (1) every one of its regions reaches the required SNR, evaluated on a noise-free scan of your
protocol with the noise level of this scanner, and (2) its scan time is at most 1.15 times the shortest scan time
with which any choice of tr_ms, averages and bandwidth_hz_per_px meets (1). The scanner's log must show that you
measured. An answer in the wrong unit is wrong.

Limits. Scan time: 330 s (only `scan` spends it). Tool calls: 48 (every call counts once unless rejected). Rejected calls: 8 (a call refused for how it is written: unknown name, wrong type; costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 330.00 of 330 s · calls 0/48 · rejected 0/8]

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
