# Suppress the fat in each marked region and keep the water signal
A test object with vials of unknown materials lies in the field of
view. Each marked region lies inside a vial that contains fat, a different fat in every vial. For each of them the
examination needs a spin-echo image in which the fat of that region is suppressed:
    tr_ms 1700, te_ms at the console minimum, matrix 128, bandwidth_hz_per_px 300, field of view at the
    console default, 1 average, slice 5 mm,
    either  sequence se with fat_sat true                (method fat_sat: spectral fat saturation)
    or      sequence ir with an inversion time ti_ms     (method stir: inversion recovery at the null of the fat).
Regions (circles; centre and radius in mm from the isocentre; each lies inside one homogeneous material):
  R1: y 4.9, x -4.19, radius 9
  R2: y -15.46, x -56.07, radius 9
  R3: y 48.56, x -37.19, radius 9
  R4: y 48.12, x 33.74, radius 9
  R5: y -17.36, x 51.37, radius 9
Deliver one method per region (R1, R2, R3, R4, R5) with `submit`, and the inversion time where the method is stir. The target scans
are not run in this episode: the scanner evaluates what you deliver. Your own scans may use any sequence and any parameters the console accepts; only the deliverable is evaluated, on the scan the task names. That scan is not charged to the scan-time budget and not limited by it.
Nothing about the materials, the field or the state of the system is known in advance.

Passing, for every region:
1. Suppressed. In the target scan with the method you delivered for it, its signal is at most 5 % of its signal in
   the same spin echo without any suppression.
2. No water signal given away. Fat saturation leaves the water signal alone; inversion recovery removes a large part of it.
   stir therefore passes for a region only where fat saturation is not enough: where fat saturation leaves more than
   3 % of the region's signal. (Between 3 % and 5 % both methods pass.)
The scanner's log must show that you measured. An answer in the wrong unit is wrong.

Limits. Scan time: 120 s (only `scan` spends it). Tool calls: 64 (every executed call counts once). Rejected calls: 8 (a call that is refused for how it is written reaches no tool and costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 120.00 of 120 s · calls 0/64 · rejected 0/8]

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
