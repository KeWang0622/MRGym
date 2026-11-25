# Scanner/PhantomQA-v0 — why thinking-off Qwen3.8-27B fails (run c04) — read 2026-11-25

Paths below are relative to `…/rebuild/runs/gpu/c04/records/cal/episodes/qwen3.8-27b/Scanner-PhantomQA-v0/`.
Scripts, compact transcripts and tables: `analyst-scratch/pq/`.

## 0. What was read, what was not
- Code: `envs/phantom_qa/task.py` (`statement`, `measure`, `within`, `score`, `outcome`), `controls.py` (`measure`, `_geometry_fault`,
  `_uniformity_fault`, `_ghost_fault`), `qa.py` (`chord`, `find_centre`, `uniformity`, `ghosting`), `scanner/session.py::profile`,
  `envs/scanner_env.py` (`scanner_tools`, `_tool_profile`), `scanner/kspace.py::_motion`.
- Transcripts read: 33 failing thinking-off episodes (11 per level, all 4 seeds each), all 7 passing thinking-off, 10 passing
  full-reasoning (c01). The other 56 failing c04 episodes were NOT read; they enter only through `record.json` and tool-call statistics.
  The read set leans to shorter transcripts. 42 failing c01 episodes were not read.
- One level = 4 hidden instances x 8 episodes (seeds 9000-9003), so rates per level rest on 4 objects. Hidden truth was recomputed
  with `task.make_instance` (task ids match the records); the scripted `reference` policy passes all 12 instances
  (worst error 0.43 tolerances, 27-33 tool calls) — the task is solvable through the tools on exactly these objects.

## 1. Counts
All 96 episodes (95 delivered an answer). "off" columns count failing episodes, "ok" columns count all 32 of the level:

| | pass | diameter off | PIU off | ghost off | all 4 values ok | verdict ok | faults ok |
|---|---|---|---|---|---|---|---|
| L1 (30 delivering fails) | 2 | 22 | 7 | 11 | 7 | 20 | 8 |
| L2 (28) | 4 | 18 | 10 | 9 | 9 | 24 | 10 |
| L3 (30 + 1 no submit) | 1 | 21 | 22 | 14 | 5 | 17 | 5 |
| thinking off, total | 7 | 61 | 39 | 34 | 21 | 61 | 23 |
| full reasoning (c01), 88 delivered | 54 | 12 | 14 | 2 | 63 | 86 | 75 |

Most-failed quantity: a diameter (61 of 88 delivering failures). Error against the 0.6 mm tolerance: 21 within 2x, 32 at 2-10x,
8 beyond 10x. Ghosting (34): 8 / 19 / 7. PIU (39): 24 / 14 / 1 — 30 too high (extremes not found), 9 too low (all 9 placed circles past r = 76 mm; the allowed centre radius is 74.2).
`outcome` puts `measurement_off` first and so hides the second wall: the fault list is right in 23 of 96 episodes; only 16 of the 89
failures have verdict AND faults right. 14 episodes have all four values inside tolerance and still fail (5 verdict, 9 diagnosis).

First decisive error, 33 read failures (10 of the 13 class-b episodes also carry a class-c diameter error):

| class | L1 | L2 | L3 | episodes |
|---|---|---|---|---|
| (a) interface wall | 0 | 1 | 0 | L2/s9001/e1 |
| (b) wrong method | 5 | 5 | 3 | L1/s9000/e3 e4, L1/s9003/e0 e2 e7; L2/s9001/e3 e5 e7, L2/s9002/e2, L2/s9003/e6; L3/s9001/e1 e5, L3/s9002/e3 |
| (c) right method, sloppy | 6 | 5 | 7 | L1/s9000/e0, L1/s9001/e4 e7, L1/s9002/e0 e5 e7; L2/s9000/e0 e4, L2/s9002/e0 e7, L2/s9003/e2; L3/s9000/e1 e4, L3/s9001/e6 e7, L3/s9002/e0, L3/s9003/e0 e4 |
| (d) limit | 0 | 0 | 1 | L3/s9002/e4 (8 rejected `calc` calls: `interp` misuse, 200000-step loop limit; no submit) |
| (e) other | 0 | 0 | 0 | — |

Class (b), all on the ghosting regions: circles of radius 21-31.5 mm at 110 mm in place of the 16 x 63 mm rectangles (7: L1/s9000/e3,
L1/s9003/e0, L2/s9001/e3 e5 e7, L3/s9001/e1, L3/s9002/e3); rectangles laid radially or with 63 mm as `width_mm` (4: L1/s9000/e4,
L1/s9003/e2, L2/s9002/e2, L2/s9003/e6); 1 cm2 circles sitting on the ghost (2: L1/s9003/e7, L3/s9001/e5). The first two reach in to
r = 78.5 mm (the phantom edge is at 95 mm); recomputed on the verifier's noise-free image: circles r 31.5 give 0.064 on L1/s9000 (truth 0.0007),
0.077 on L3/s9002 (0.0028), 0.025 on L1/s9003 (0.0907), 0.0074 on L3/s9001 (0.0973) — the delivered clusters at +10.5, +12, -5.6, -6 tolerances.
Over all 95 episodes: 4 correct rectangles read in 13 (ghost right in 13 of 13); a region overlapping the phantom read in 43
(26 of the 34 wrong ghost values).

Class (c): 8 never interpolate or read the edge by eye off 2-4 px bins (L1/s9001/e4 submits 190.2 / 190.0; L3/s9002/e0
`diam_y = 80.6-(-95.2)`); 3 interpolate on a 1 px profile between the wrong pair of samples (L1/s9000/e0, L1/s9001/e7, L2/s9002/e7);
4 use a wide `reduce: max` line or the 1 px step on 2 px bins (L2/s9000/e0 `…*0.9766` on 1.9531 mm samples; L2/s9002/e0; L3/s9001/e6 e7);
2 search PIU at 8 and 17 positions (L2/s9003/e2, L3/s9000/e1); 1 judges 189.59 mm as "fails 190 +- 3" (L1/s9002/e0).
Wrong numbers copied also occur: L3/s9003/e0 computes PIU 73.95 (inside tolerance) and submits the earlier 78.1; L3/s9001/e7 submits
a ghost ratio 0.043 from prose arithmetic on numbers that are not the read-out; L2/s9003/e6 drops its own lowest circles from the min.
Over all episodes: of 61 with a diameter off, 33 never read an edge at 1 px, 28 read one and still mis-computed.

Class (a), one case: L2/s9001/e1 asks `profile … axis x, position_mm 110, width_mm 63`, gets back
"profile of scan_index 0 … along x at y = 101.562 mm, 47 px = 45.898 mm wide, max per sample", writes "I'm hitting a constraint —
the width parameter is limited to one pixel", and falls back to r 31.5 circles. The request was already wrong (63 mm is the long
side); the tool clipped the width at the image edge and re-centred the line without the `note:` it prints for `start_mm` / `stop_mm`.

## 2. How the passing episodes differ
- Thinking-off passes (L1/s9000/e6, L1/s9002/e1, L2/s9000/e1 e7, L2/s9002/e3 e4, L3/s9003/e7): all 7 read each edge on a 1 px profile and
  place the crossing of half the 200 cm2 mean between the right pair (by formula or by eye); 6 of 7 are on objects whose out-of-limit diameter lies along the
  phase-encode axis (only a gradient can cause it — no second scan needed); none has a motion ghost. Several still land at
  0.4-0.55 mm error. They are the lucky tail of the same method, not a different one.
- Full reasoning (c01; L1/s9000/e5, L1/s9002/e4, L1/s9003/e7, L2/s9001/e3, L2/s9002/e4, L2/s9003/e4, L3/s9000/e3, L3/s9001/e7,
  L3/s9002/e7, L3/s9003/e2): the reasoning block works out the rectangle geometry before any call ("top: the edge is horizontal so
  the long side (63 mm) is along x…") and brackets each crossing by inequality ("1.0039 < 1.6753 < 3.4501"), then confirms in `calc`.
  4 correct rectangles: 89 of 96 episodes (thinking off 13 of 96). 1 px edge profile: 96 % of passes. PIU: 25-circle grid plus
  1-3 refinement reads. Diagnosis by axis logic or a <= 10 s second scan (PE swap; two flip angles). 57-71k output tokens per episode
  against 6.5k; thinking-off uses `calc` as the scratchpad instead (4.0 calls against 1.7) and the errors are in that code.

## 3. Statement against `score`
The statement gives every quantity the verifier computes; no mismatch found. Checked on the tool path: `profile` with `samples 1`,
`width_mm 16`, +-31.5 mm selects the same 16 x 65 pixels as `qa.rect_mean`; a chord through the isocentre equals the chord through
the phantom centre to 0.01 mm; the motion phase is a hash of the protocol (`kspace._motion`), so the verifier's noise-free scan
shows the same ghost. Five near misses redone by hand with the numbers the model had on screen:

| episode | delivered | correct use of the model's own numbers | verifier |
|---|---|---|---|
| L1/s9001/e7 dy | 190.52 | 189.573 | 189.569 |
| L1/s9000/e0 dx | 189.7 | 188.716 | 188.718 |
| L2/s9002/e7 dy | 190.0 | 188.962 | 188.958 |
| L1/s9003/e0 dy (motion-blurred edge) | 186.8 | 190.052 | 189.986 |
| L3/s9000/e1 PIU (17 circles) | 90.1 | 90.09 — extremes at (-24, 69) and (51, -24) mm never sampled | 84.41 |

So `measurement_off` is execution, not definition, in every case checked. Wording that a careful reader gets right and this
model does not: (i) "long side along the edge" — which edge is not said; the radial reading puts the region inside the phantom and
past the field of view, so only the tangential one is consistent; (ii) "anywhere inside the 200 cm2 circle" — the verifier keeps
the whole 1 cm2 circle inside (centre <= 74.2 mm); (iii) the level: half the image maximum instead of half the 200 cm2 mean moves a
diameter by 1.2 mm on the receive-coil object and 3-4.5 mm on the motion objects (verifier-side check).
One real gap: "A fault in the list is either present (its test fails) or absent" does not say that a failed test has exactly one
fault. That is true by construction (`LEVELS`, `PAIRS`) and stated nowhere. 11 episodes name two faults for one failed test,
3 of them with all values and the verdict right (L1/s9000/e5, L2/s9000/e2, L2/s9002/e5).

## 4. The single change
There is none that converts most failures: the errors are stacked (diameter 61, PIU 39, ghost 34, fault list wrong in 73 of 96).
Mechanical ceilings: only-the-diameters-wrong with verdict and faults right = 8 episodes (L1/s9001/e4 e5 e7, L1/s9002/e5,
L1/s9003/e3, L2/s9002/e0 e1 e7); all measurements fixed = 16; so perfect measurement tooling alone tops out near 23 of 96.
Largest single lever: let `profile` take a `level` and return the outermost sub-pixel crossings (an edge cursor). It removes the
bracket-and-interpolate arithmetic that fails in 61 episodes, states no answer and touches no tolerance; expect 7 -> about 15 passes.
Cost: the diameter stops being a computation the agent does. Weaker variant that keeps it: one sample per pixel by default
(`MAX_PROFILE_SAMPLES` 128 -> 256) — helps the 33 that never zoomed, not the 28 that mis-bracket.
Second lever, cheap: a rectangle read-out (`rects` in `read_regions`), or the four rectangles as coordinates in the statement.

## 5. Practice level / hint tier
What the hint has to state (each item is something the full-reasoning model derives and the thinking-off model does not):
1. Edge: profile each edge over <= 12 mm so samples are 1 px; the edge lies between the last sample below and the first above
   half the `mean|S|` of the r 79.8 circle; interpolate linearly between those two.
2. Ghost: the four calls spelled out (top: `axis x, position_mm -110, width_mm 16, start_mm -31.5, stop_mm 31.5, samples 1`, …).
3. PIU: circles r 5.64 with centres <= 74 mm, a lattice of <= 15 mm pitch, then 8 more round the lowest and the highest.
4. Diagnosis: one fault per failed test. Out-of-limit diameter on the phase-encode axis -> that gradient; on the readout axis -> a
   7 s scan with `pe_dir` swapped: back near 190 -> shim, else the gradient. Uniformity: dip at the centre -> b1_transmit, fall-off
   across -> receive_coil (or a flip-angle map). Ghost: filled flat band -> signal_instability, arcs with gaps -> phantom_motion.
Items 1-3 alone bound the pass rate by the scripted `no_discrimination` control (43 % L1, 67 % L2 in `gate_dev.json`), so with this
model's execution slips expect roughly 20-35 %; items 1-4 are needed for the top of the 20-60 % band. Not tested — no model was run.
What RL then has to add to drop the hint: (1) pick the bracketing pair and interpolate without scratch reasoning; (2) turn a region
defined in words into tool arguments (orientation, allowed zone); (3) lattice-then-refine search under 16 circles per call;
(4) compare against a limit before naming a failure; (5) choose the one short scan the failed test calls for, and one fault per test.

## 6. Defects
- Verifier (`task.score`, `measure`, `within`): none found. Hand recomputation agrees to 0.07 mm; reference passes 12 of 12.
- Task text: the one-fault-per-failed-test rule is missing (section 3). "long side along the edge" could name the edge.
- Design, PIU on phantom_motion objects: the value is set by ghost structure inside the phantom. A 5 x 5 grid errs +2.2 to +3.4
  points (tolerance 3.0); the reference's own 12 mm, 139-circle lattice errs +3.38 on L3/s9001 before its refinement rounds; an 8 mm
  lattice (313 circles = 20 calls) errs +0.8. Even with full reasoning these objects are the weakest (all values right in 3, 3 and
  2 of 8 on L1/s9003, L3/s9000, L3/s9001). This is why PIU is off in 22 of 30 L3 failures.
- Interface: (i) `profile` clips `width_mm` at the image edge and re-centres silently (4 episodes read: L2/s9001/e1, L2/s9002/e2,
  L2/s9003/e6, L3/s9000/e4; in L1/s9003/e2 the noted `stop_mm` clamp draws "Profile clamping is confusing"); (ii) the tool text says `samples` "default one per pixel"
  but a 256 px line returns 128 samples of 2 px, and `samples` > 128 is refused rather than clipped (12 episodes); (iii) `calc`
  errors count toward the 8 rejected calls (ended L3/s9002/e4); `interp` is mis-called in 6 episodes (5 arguments, a scalar for a
  list) and, as in numpy, clamps silently outside its range (L1/s9002/e5: right edge returned as the sample position 95.2).
- Reporting: `outcome` = worst-first hides that diagnosis fails as often as measurement (section 1).
