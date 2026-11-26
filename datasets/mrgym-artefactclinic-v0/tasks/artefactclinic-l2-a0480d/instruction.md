# Artefact clinic: say what each marked finding is, and correct the protocol
The scanner is a 3 T system. A test object (a body of gel with vials
of unknown materials) lies in the scanner. The examination below was run on it, and a reader marked 4
findings on its image. Each finding is one of two things: produced by the acquisition (it would not be there with
another protocol), or the object itself (it is there whatever the protocol).

Current protocol (107.5 s):
    sequence spgr, fov_mm 240.0, matrix 104, phase_fov 0.61, phase_oversampling 0.0, partial_fourier 1.0, slice_mm 5.0, pe_dir x, ro_polarity -1, tr_ms 196.0, flip_deg 72.0, averages 8, fat_sat false, centre_y_mm 0.0, centre_x_mm 0.0, bandwidth_hz_per_px 2390.0, te_ms 3.46
    SPGR 3.0T | TR 196.0 TE 3.5 FA 72 | FOV 240x148 mm  104x64  SL 5.0 | BW 2390 Hz/px  PE R>>L | NEX 8 | TA 1:47.5 | SAR 2 % | WFS 0.2 px (readout) | rel. SNR 1.89
Findings (circles; centre and radius in mm from the isocentre):
  F1: y 0.9, x 62.4, radius 3
  F2: y -24, x -23.6, radius 3
  F3: y -24.4, x -47, radius 3
  F4: y 41.1, x -32.6, radius 3
Reference region for the signal-to-noise ratio (homogeneous gel):
  REF: y 0, x 0, radius 8

Deliver with `submit`:
  causes     for each finding (F1, F2, F3, F4) the list of its causes, from
             chemical_shift_displacement, opposed_phase_cancellation, truncation_ringing, wrap_around
             (at most 2 per finding; an empty list means the finding is the object itself).
  protocol   one protocol for the examination, on which every finding is free of acquisition artefacts. A field
             left out takes the console default, not the current protocol's value.
Requirements for the delivered protocol:
  scan time at most 117.5 s; pixel size (fov_mm / matrix) at most 2.3077 mm;
  signal-to-noise ratio in the reference region at least 1143.3 (mean magnitude in the region divided
  by the noise SD of one pixel; in an empty region the noise SD is the SD of the magnitude / 0.655);
  te_ms between 1.04 and 5.88; unchanged from the current protocol: sequence, tr_ms, flip_deg, slice_mm, fov_mm, pe_dir, ro_polarity, fat_sat, centre_y_mm, centre_x_mm.
The delivered protocol is not run in this episode: the scanner evaluates it. Your own scans may use any sequence and any parameters the console accepts; only the deliverable is evaluated, on the scan the task names. That scan is not charged to the scan-time budget and not limited by it.
The budget is smaller than one run of the current protocol: experiments have to be cheaper (shorter TR, one average). Nothing about the materials is known in advance.

Passing: every finding is given exactly its causes; on a noise-free scan of the delivered protocol every finding
agrees with the same scan without the artefact mechanisms (mean absolute difference at most 5 % of the
signal; at most 3 % against the object itself averaged over each pixel); every requirement is met.
The scanner's log must show that you measured.

Limits. Scan time: 67 s (only `scan` spends it). Tool calls: 160 (every call counts once unless rejected). Rejected calls: 8 (a call refused for how it is written: unknown name, wrong type; costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 67.00 of 67 s · calls 0/160 · rejected 0/8]

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
