# Measure T1 in marked regions with an uncalibrated transmit field
A test object of unknown composition lies in the field of view.
The transmit field is not calibrated: the flip angle a spin experiences at the centre of the slice is the nominal
flip angle times an unknown, position-dependent factor. A slice-selective pulse turns the spins by less away from
the slice centre (the slice profile; `scanner_info` with topic slice_profile describes it).

Deliver the longitudinal relaxation time T1 of each marked region, in milliseconds, with `submit`.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
  R1: y -45.17, x 31.2, radius 11.5
  R2: y 10.4, x -52.35, radius 11.5
  R3: y -45.16, x -32.45, radius 11.5
  R4: y 50.8, x -3.62, radius 11.5
  R5: y 13.49, x 50.25, radius 11.5
T1 values lie between 300 and 2500 ms. Nothing else about the object or the system is known.

Passing: for every region the delivered T1 lies within a factor 1.1 of the true T1 (between T1 / 1.1 and 1.1 x T1).
The scanner's log must show that you measured. An answer in the wrong unit is wrong.

Limits. Scan time: 88 s (only `scan` spends it). Tool calls: 64 (every executed call counts once). Rejected calls: 8 (a call that is refused for how it is written reaches no tool and costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 88.00 of 88 s · calls 0/64 · rejected 0/8]

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
