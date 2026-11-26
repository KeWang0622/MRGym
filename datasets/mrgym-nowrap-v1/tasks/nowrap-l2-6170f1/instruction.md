# Prescribe the fastest scan that shows the whole object without wrap-around
An object of unknown size, position and orientation lies in the
scanner. The examination needs one spin-echo image of the whole object:
    sequence se, tr_ms 500, te_ms at the console minimum, bandwidth_hz_per_px 300, 1 average, slice 5 mm,
    in-plane pixel size at most 2 mm along both axes.
Deliver the geometry of that scan with `submit`: the phase-encode axis (`pe_dir`), the readout field of view (`fov_mm`),
`phase_fov` (the phase field of view as a fraction of `fov_mm`), `matrix`, the centre of the field of view
(`centre_y_mm`, `centre_x_mm`, in mm from the isocentre), its in-plane rotation (`rotation_deg`: image rows turned
from +y towards +x) and, optionally, `phase_oversampling`. The target scan is not run in this episode: the scanner
evaluates the geometry you deliver. Your own scans may use any sequence and any parameters the console accepts; only the deliverable is evaluated, on the scan the task names. That scan is not charged to the scan-time budget and not limited by it.
Nothing about the object is known in advance.

Passing: the console accepts the geometry; the pixel is at most 2 mm; the whole object lies inside the image; the
image shows no wrap-around; and the scan takes at most 10 % longer than the fastest geometry that meets these
requirements (the field of view may be moved and turned to reach it). The scanner's log must show that you measured.
A length in the wrong unit is wrong.

Limits. Scan time: 30 s (only `scan` spends it). Tool calls: 40 (every call counts once unless rejected). Rejected calls: 8 (a call refused for how it is written: unknown name, wrong type; costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 30.00 of 30 s · calls 0/40 · rejected 0/8]

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
