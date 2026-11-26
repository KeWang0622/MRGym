# Adapt the standard protocol until the examination meets its specification
A test object with vials of unknown materials lies in the field of
view. The examination has 2 inversion-recovery spin-echo images, A and B, with the same geometry and repetition time:
    sequence ir, tr_ms 400, fov_mm 240, matrix 128, slice_mm 5, bandwidth_hz_per_px 200, full k-space.
Both run with the standard protocol ti_ms 80, te_ms 12, averages 1. Each image has its own specification:
    image A
      T1 contrast   R1 at least 4 times as bright as R4 (the inversion can null R4)
      T2 contrast   R2 (the longer T2) at least 1.5 times as bright as R1
      SNR           at least 20 in R5
    image B
      T1 contrast   R2 at least 4 times as bright as R4 (the inversion can null R4)
      T2 contrast   R2 (the longer T2) at least 1.5 times as bright as R1
      SNR           at least 20 in R5
and the examination has one more: the scan times of the images together are at most 552 s.
"As bright as" compares the signals of two regions in the image. SNR of a region = magnitude of the mean complex signal of the region's pixels / standard deviation of the noise in one pixel, one channel (real or imaginary part).
On this object the standard protocol does not meet the specification. For each image you choose ti_ms, te_ms and
averages; everything else stays as above.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
  R1: y -54.91, x -14.16, radius 11.5
  R2: y 9.5, x 57.08, radius 11.5
  R3: y -42.28, x 40.84, radius 11.5
  R4: y -13.26, x -51.86, radius 11.5
  R5: y 51.33, x 19, radius 11.5
  R6: y -2.71, x 0.3, radius 11.5
  R7: y 40.99, x -32.91, radius 11.5
Deliver the three numbers for each image with `submit`. The images are not run in this episode: the scanner evaluates
the protocols you deliver. Your own scans may use any sequence and any parameters the console accepts; only the deliverable is evaluated, on the scan the task names. That scan is not charged to the scan-time budget and not limited by it.
This console adjusts itself before a scan (centre frequency, transmit gain, shim): 30 s. A scan with the same sequence
and the same field of view, slice and orientation as the scan before it reuses those adjustments and starts after
3 s; changing only timing, matrix or averages does not repeat them. `plan` shows the cost of the next scan. Nothing
about the materials, the noise or the state of the system is known in advance.

Passing: every requirement holds, evaluated on noise-free scans of your protocols with the noise level of this scanner;
the scan time of each image is what the console plans for it as a scan of its own, adjustments included. The scanner's
log must show that you measured. An answer in the wrong unit is wrong.

Limits. Scan time: 63 s (only `scan` spends it). Tool calls: 48 (every call counts once unless rejected). Rejected calls: 8 (a call refused for how it is written: unknown name, wrong type; costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 63.00 of 63 s · calls 0/48 · rejected 0/8]

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
