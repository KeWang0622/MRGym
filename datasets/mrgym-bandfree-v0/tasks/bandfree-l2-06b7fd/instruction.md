# Keep the dark bands of two balanced SSFP images off their marked regions
A test object with vials of unknown materials lies in the field of
view. The examination needs two balanced SSFP images of it, image A and image B, each with a protocol of its own:
    sequence bssfp, flip_deg 60, matrix 120, field of view at the console default, slice 5 mm with the console's
    default slice-selective pulse, 1 average, bandwidth_hz_per_px of your choice between 60 and 600, tr_ms at the console
    minimum for that bandwidth (a plan shows it), phase_cycle_deg of your choice
No dark band may cross the marked regions of an image (circles; centre and radius in mm from the isocentre; each lies
inside one homogeneous material):
  image A
  R1: y 52.74, x -9.93, radius 6
  R2: y -43.31, x -38.56, radius 6
  R3: y -58.96, x 18.05, radius 6
  image B
  R4: y 32.82, x 47.73, radius 6
  R5: y 11.08, x -52.79, radius 6
Deliver with `submit`, for each image, the bandwidth and one phase cycle. If no single acquisition can keep every
marked region of an image out of the bands, deliver for that image the bandwidth and two phase cycles: two
acquisitions that differ only in the phase cycle. Every region is then read from one of the two, the one that shows
it better, and rule 1 must hold for it in that one.

Passing, for each image, all of:
1. Band-free, with a margin. Every marked region of the image keeps at least 80 % of the largest signal it can have
   at this bandwidth and repetition time (its signal at its best phase cycle). This must hold at the present centre
   frequency and at every drift of the centre frequency of up to 20 Hz either way (the same drift everywhere in the
   object).
2. No more acquisitions than needed. Two acquisitions pass only when no single acquisition at a bandwidth of 400 or
   less is band-free for that image.
3. Low bandwidth (the noise of the image grows with the square root of the bandwidth). The delivered bandwidth is at
   most 1.15 times the lowest bandwidth at which a band-free protocol with the same number of acquisitions exists
   for that image.
For rules 2 and 3 the scanner compares with protocols that stay band-free up to a drift of 26 Hz, so a protocol
that only just exists does not count against you.

The target images are not acquired in this episode: the scanner evaluates the protocols you deliver, noise-free.
Your own scans may use any sequence and any parameters the console accepts; only the deliverable is evaluated, on the scan the task names. That scan is not charged to the scan-time budget and not limited by it.
Nothing about the materials, the field or the frequency adjustment is known in advance. The scanner's log must show that you measured. An answer in the wrong unit is wrong.

Limits. Scan time: 43 s (only `scan` spends it). Tool calls: 64 (every call counts once unless rejected). Rejected calls: 8 (a call refused for how it is written: unknown name, wrong type; costs nothing else), of them at most 3 refused `submit` calls. The episode ends at `submit` or at a limit; without `submit` the score is 0.
[scan time left 43.00 of 43 s · calls 0/64 · rejected 0/8]

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
