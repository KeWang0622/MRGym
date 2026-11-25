# Null each marked material with inversion recovery
A test object with vials of unknown materials lies in the field of
view. For each marked region the examination needs an inversion-recovery image in which the material of that region
gives no signal. The scan is the same for all of them except for its inversion time:
    sequence ir, tr_ms 2890, te_ms at the console minimum, matrix 128, bandwidth_hz_per_px 300,
    field of view at the console default, 1 average, slice 5 mm.
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
  R1: y 11.56, x 52.79, radius 11.5
  R2: y -58.59, x 8.46, radius 11.5
  R3: y -17.26, x -57.52, radius 11.5
Deliver one inversion time per region (R1, R2, R3) with `submit`. The target scans are not run in this episode: the scanner
evaluates the inversion times you deliver. Your own scans may use any sequence and any parameters the console accepts; only the deliverable is evaluated, on the scan the task names. That scan is not charged to the scan-time budget and not limited by it.
Nothing about the materials or the state of the system is known in advance.

Passing: for every region, in the target scan with the inversion time you delivered for it, its signal is at most
5 % of its signal in the same scan without the inversion pulse. The scanner's log must show that you
measured. An answer in the wrong unit is wrong.

Limits. Scan time: 200 s (only `scan` spends it). Tool calls: 48 (every executed call counts once). Rejected calls: 8 (a call that is refused for how it is written reaches no tool and costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 200.00 of 200 s · calls 0/48 · rejected 0/8]

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
