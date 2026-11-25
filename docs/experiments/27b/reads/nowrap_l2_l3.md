# Scanner/NoWrap-v1, thinking off (c04): why level 2 and level 3 are 0 of 32

Analyst memo, 2026-11-25. Read-only. Episode ids are relative to
`runs/c04/records/cal/episodes/qwen3.8-27b/Scanner-NoWrap-v1/`
(full reasoning: same path under `c01`). Code: `MRGym/src/mrgym/envs/nowrap/{task,controls,oracle}.py`.
Scripts and dumps: `work/failread/nowrap/` (`truth.py`, `mech.py`, `cdump.py`, `udump.py`, `verify.py`, `out/`).

## What was read
- Level 2: all 32 conversations. Level 3: 14 of 32 (s9000 e0 e1 e6 e7; s9001 e1 e3 e6 e7; s9002 e1 e2 e4; s9003 e3 e5 e7).
- Level 1 passes: 8 (s9000 e0 e5; s9001 e0 e6; s9002 e0 e4; s9003 e0 e4). Full-reasoning level-2 passes: 8 (s9000 e2 e5; s9001 e3 e4; s9002 e2 e3; s9003 e2 e7).
- Mechanical tags from `scanner_log.jsonl` + `record.json` for all 96 c04 episodes of levels 1-3 and the 32 c01 level-2 episodes.
- Hidden truth per instance (object box, optimum) recomputed with `task.outline` / `task.optimum`; each level is 4 instances x 8 episodes.
- NOT read: 18 level-3 conversations (mechanical tags only), level 0, the failing c01 episodes, c01 level 3, `console.plan` internals.
  Pass-rate predictions below are estimates; nothing was re-run with a model.

## 1. First decisive error per episode
| class | L2 (32 read) | L3 (14 read) |
|---|---|---|
| (a) interface | 1 | 3 |
| (b) wrong method | 28 | 8 |
| (c) right method, sloppy | 3 | 2 |
| (d) budget / cap | 0 | 1 |

Level 2, class (b), by kind:
- b1, the look-scan cannot show the object (14): the model looks with the target sequence (`se`, TR 500, 0.5 s per line), so it can
  afford about 50 lines; it cuts the field of view or `phase_fov` below the object size and reads the cut or wrapped image as the
  object. s9000 e2 e3; s9001 e0 e1 e3 e5 e6; s9002 e1 e2 e3 e5 e7; s9003 e5 e7.
- b2, extents about right, wrong axis decision (7): phase field of view put on the long axis, `phase_fov` > 1 requested, or a square.
  s9000 e0 e7; s9001 e2; s9003 e0 e1 e2 e6.
- b3, calls the object a centred disc against its own two unequal extents, delivers a square of the short side (4): s9000 e1 e6; s9001 e4; s9002 e0.
- b4, invents a rotation from chord edges of an upright ellipse (2): s9002 e6; s9003 e4.
- b5, 15.6 mm pixels, extents wrong by 40 mm (1): s9003 e3.
Level 2, class (c) (3): s9000 e4 (zero margin, 1.6 mm cut off, 50 lines where 52 are the minimum); s9001 e7 (chose 88 mm for its own
measured 92 mm); s9002 e4 (left `centre_x` at 0 against its own measured centre -9, and read the low y edge 10 mm short).
Level 2, class (a) (1): s9000 e5 (reads the digit map as a 55 x 70 mm blob, then rejects correct profile numbers; see section 4).

Level 3 (14 read): (a) s9000 e1 (digit map overrides correct projections), s9002 e4 and s9003 e7 (calc refusals, ended at the
rejected-call limit without a submission); (b) s9000 e0 e6 e7, s9001 e1 e3 e6, s9002 e1, s9003 e5 (no narrowest-direction search:
upright box, square, arbitrary 45 degrees, or long side on the phase axis); (c) s9001 e7 (turned -45 degrees with phase across the
width, the right idea, width under-read on 7.5 mm pixels), s9003 e3 (edge read 9 mm short); (d) s9002 e2 (40 calls used on a hand
rotation search in `calc`). The 18 unread: 15 look with `se` only.

Mechanical counts, all episodes:
| | L1 | L2 | L3 | c01 L2 |
|---|---|---|---|---|
| first scan is spgr or epi | 32 | 6 | 10 | 26 |
| no stored scan holds the whole object | 0 | 14 | 5 | not computed |
| used a projection profile (max, width >= 100 mm) | 32 | 4 | 7 | 19 |
| delivered `phase_fov` >= 0.95 (square) | 3 | 18 | 16 | 2 |
| delivered centre (0, 0) | 2 | 13 | 5 | 0 |
| pass | 22 | 0 | 0 | 23 |
The first `plan` or `scan` call names `se` in 29 of 32 (L2) and 28 of 32 (L3) episodes.

Three quotes for the dominant failure (b1):
- L2/s9001/e5: "The 30 s budget is tight. With se at TR 500 ... I can only afford about 52–56 lines total. I'll explore with a coarse
  56-line spin-echo". The scan is 129 mm tall for a 177 mm object; conclusion: "the object is a circular ring, radius ~30 mm, centered at the isocenter".
- L2/s9002/e3: scans `se`, fov 80, matrix 52 (29.5 s), then: "The object clearly fills the entire 80×80 mm FOV ... I have 0.5 s of scan time left".
- L2/s9000/e3: `se`, `phase_fov` 0.5 (120 mm for a 174 mm object); top and bottom rows of the view are all 9 (overlap), read as
  "a centered circular object ... diameter ~100 mm".
Second-stage error (b2), L2/s9003/e0: "the optimal design keeps the phase FOV equal to the object's longest (y) extent ... phase_fov = 1.7".

## 2. What changes from level 1 to level 2
`task.LEVELS` changes two things at once:
- `hints=("rule",)` is dropped. That paragraph (`task.hint_text`) gives the localiser protocol, the projection measurement, the
  2 mm margin, the phase-axis rule, the line and matrix formulas and the half-pixel centre shift.
- `offset_mm` goes from 3 to 25.
The conversations show the hint is what mattered. All 32 level-1 episodes run the hinted scan verbatim (`spgr`, tr 11, fov 288,
matrix 160, averages 2; 1.8 mm pixels, 10.5 s), take full-width max projections and evaluate the stated formulas in `calc`.
At level 2 nobody reproduces that scan. The offset itself was handled where the measurement worked: 19 of 32 deliveries are off-centre;
only L2/s9002/e4 failed mainly on centring. No level has the hint with an off-centre object, or a centred object without the hint, so
the ladder cannot separate the two.

## 3. What the full-reasoning passes do that thinking-off skips
Same four instances, 23 of 32 pass. In the 8 passes read:
1. Before any scan they derive the cost model: time = 3.5 + 0.5 x lines, readout pixels are free, so phase goes on the short side.
2. They settle that the 30 s are for looking and the target is not charged (c01 L2/s9000/e2: "my own budget doesn't matter for the target").
3. They pick a fast wide look-scan on purpose: `spgr` TR 12-20 over 360-500 mm at 2-4 mm pixels (8-11 s) or `epi` (3 s). 26 of 32 start
   that way; P(pass) is 19/26 after a fast first scan and 4/6 after `se`.
4. They convert digit-map rows to mm by explicit arithmetic, then measure with wide max projections and notice their own chords
   (c01 L2/s9002/e2: "far from the object's x-center (~-10), so it's a short chord").
5. They take the edge between the last bright and first dark sample, test tilt by comparing chord centres at two heights, conclude upright.
6. They check each side of the image against the object by arithmetic and add 2-4 lines of margin, checked against the 10 % window.
They do not apply the half-pixel centre shift either; the margin absorbs it (overhang 0.2-0.3 mm in two of the eight).
Cost: 25,000-128,000 reasoning tokens per passing episode. Level 3 with full reasoning is 4 of 32, so rotation is unsolved there too.

## 4. Interface: findings and the one change
There is no single interface change that converts most failures. The fast look-scan is the first error in 26 of 32 level-2 episodes,
but the 6 level-2 and 10 level-3 episodes that did start with spgr or epi also all failed, on later steps.

Ranked changes that state no method and lower no tolerance:
1. Task text (most first errors removed): say that scans taken to look may use any sequence and parameters of the datasheet, and that
   the delivered scan is not charged to the budget. Today the text only says "Use the scan-time budget for whatever you need to see
   first", the `scan` tool already carries an spgr example, and the budget refusal offers only ways to shrink the same scan:
   "error budget · this scan needs 33.50 s and 30.00 s are left: fewer lines or averages, or a shorter tr_ms" (84 times, 37 of 64 episodes).
   My estimate of the gain is small: under 20 % at level 2, about nothing at level 3.
2. `plan` output (`scanner_env.py`, the line built near line 657): "102x66 px over 170x110 mm, 66 lines" is readout x phase with no axis
   names, the reverse of "36x48 (y, x), FOV 225 x 300 mm" printed by `scan` and `view` when `pe_dir` = y. L2/s9000/e0 plans a
   "170×110 mm box" for an object 168 mm along y and delivers 110 mm along y; L3/s9000/e7 the same. Print it as (y, x) with the phase axis named.
3. Same line: "does NOT fit the budget (8.50 s left)" is printed for a geometry that will never be scanned. Explicit belief that the
   delivered scan must fit 30 s, read in L2 s9000 e3 e5; s9001 e5; s9002 e3 e5; s9003 e0 e1 and L3/s9002/e4 (L2/s9002/e5: "A larger square
   (112) would need 56 lines (31.5 s) = over budget").
4. `view`: the digit map has no coordinates beyond "top = -150 mm"; sizes are read 2-3 times too small (L2/s9003/e2: "~34×34 mm" for
   167 x 84 mm; L2/s9000/e7: "I miscounted rows"). Decisive in L2/s9000/e5 and L3/s9000/e1. A y value per row and an x ruler would fix it.
5. `calc` defect (`calc.py`, `_Run.assign`, line 616): `[0.0]*3` returns `[0]` and `[1,2]+[3]` returns `[4, 5]` without a message, and the
   later `sol[i] = ...` is refused with "only an element of a list or a dictionary can be assigned", which is what the model wrote.
   Reproduced with `mrgym.calc.calc`. 7 such refusals in L3/s9002/e4 and L3/s9003/e7; both ended at the rejected-call limit.

## 5. Practice level or hint tier
- Practice level "off-centre with the procedure": level-2 world with `hints=("rule",)`. The level-1 text is already correct for an
  off-centre upright object. Expect about the level-1 rate (22 of 32), which is above the 20-60 % band; its use is to isolate the offset.
- A tier aimed at 20-60 % would state three facts and withhold the arithmetic:
  (i) a look-scan example and the projection sentence (step 1 of the rule);
  (ii) only phase-encode lines cost time, the readout side is free, and `phase_fov` <= 1 makes the phase side the shorter side;
  (iii) nothing on margin, line count or centre.
  Failures that would remain: thin margin (L2/s9000/e4, s9001/e7) and centre not applied (L2/s9002/e4). Untested.
  `task.hint_text` knows only "rule" and "first_measurement", so this needs new hint keys.
- To drop (i), RL has to teach one early decision: do not look with the target sequence. To drop (ii): phase along the narrower
  projection. Then margin sizing inside the window: allowed lines are 52-57, 46-51, 48-53, 42-46 on the four level-2 instances, which
  with even line counts is 3.8-5.4 mm of margin per side, so pixels above about 4 mm cannot pass without sub-pixel edges.
- Level 3 needs its own tier (direction of the narrowest projection, rotation sign). The untrained model does not get there with reasoning.

## 6. Verifier
No verifier bug found. Checked by hand against `record.json` and by re-scoring with `task.score` (`verify.py`):
| episode | realised lines | image along the failing axis | object | overhang by hand | record |
|---|---|---|---|---|---|
| L2/s9000/e4 | 50 (asked 49.28) | x [-67.0, 33.0] | x [-66.6, 34.6] | 1.6 mm | 1.6 |
| L2/s9001/e7 | 44 (asked 45) | x [-39.0, 49.0] | x [-41.4, 47.8] | 2.4 mm | 2.4 |
| L3/s9003/e3 | 44 (asked 45) | y [-72.0, 76.0] | y [-70.2, 84.2] | 8.2 mm | 8.2 |
Excess time and wrap fraction also reproduce. The object of the pass rule (5 % of the maximum) lies 0.4-0.8 mm outside the half-height
edge on three instances, about the 0.5 mm slack, so an edge read at half height is not penalised by the rule.
Two things to know, neither changed an outcome here:
- An odd line count is rounded to the nearest multiple of 4 (45 -> 44, 43 -> 44, 47 -> 48), so a delivered `phase_fov` can lose 2 mm
  without notice; `submit` returns only "deliverable received". The scripted reference always emits even counts and never meets this.
- The image reaches half a pixel less far on the high side. Level 1 states it in the hint; at level 2 it is only in `scanner_info outputs`.
`outcome()` labels an episode `object_cut_off` even when the time check also fails (15 of the 26 at level 2).
